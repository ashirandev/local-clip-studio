"""Hearing what is said, against a fake hearing worker: python -m unittest discover -s tests"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, hearing, recording, thai, tools  # noqa: E402
from test_agent_loop import ScriptedMind, call  # noqa: E402
from test_map import make_moment  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class Worker:
    """A fake hearing worker: answers /hear with the phrases it is given, after a delay, and keeps what it got. The
    phrases are in seconds of the look: a look hears tools.SPILL s of sound before it too, so they come back that
    much later in the sound it was sent."""

    def __init__(self, phrases=(), delay=0.0, status=200):
        self.phrases, self.delay, self.status, self.got = list(phrases), delay, status, []
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, data):
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send(200, {"ok": True, "model": "omniASR_CTC_1B_v2", "dtype": "float32", "threads": 8})

            def do_POST(self):
                worker.got.append(self.rfile.read(int(self.headers["Content-Length"])))
                time.sleep(worker.delay)
                if worker.status != 200:
                    return self.send(worker.status, {"error": "the model is still loading"})
                self.send(200, {"phrases": [dict(p, **{"from": p["from"] + tools.SPILL, "to": p["to"] + tools.SPILL})
                                            for p in worker.phrases], "seconds": 0.1})

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.http.server_port

    def close(self):
        self.http.shutdown()
        self.http.server_close()


def fixes(*answers):
    """An ask(messages, schema) that answers with each item in turn (a dict is sent as JSON) and keeps the questions."""
    asked, left = [], list(answers)

    def ask(messages, schema):
        asked.append((messages, schema))
        item = left.pop(0)
        return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)

    return ask, asked


class Thai(unittest.TestCase):
    def test_the_marks_the_model_cannot_write_are_put_back_and_nothing_else_changes(self):
        self.assertEqual(thai.repair("ส⁇นค้า ค⁇ณ เจ็ดส⁇บห้า"), "สินค้า คุณ เจ็ดสิบห้า")
        self.assertEqual(thai.repair("ทํางาน"), "ทำงาน")
        plain = "ราคา 75 บาท, ok!"
        self.assertEqual(thai.repair(plain), plain)

    def test_a_number_word_the_model_began_with_a_digit_is_written_as_said(self):
        for heard, said in (("5้าสิบ", "ห้าสิบ"), ("ถึงเจ็ดสิบ5้าตารางเมตร", "ถึงเจ็ดสิบห้าตารางเมตร"),
                            ("มันเต็ม1ึ่ง.ิด5", "มันเต็มหนึ่งจุดห้า"), ("เ750", "750"),
                            ("pm 2.ุ5กันมา", "pm สองจุดห้ากันมา")):
            self.assertEqual(thai.repair(heard), said)
        for kept in ("ราคา 75 บาท", "ลด 1,990 บาท", "PM2.5 ok", "ค่า 4.5 ลิตร"):
            self.assertEqual(thai.repair(kept), kept)
        self.assertEqual(thai.lock("มันเต็มหนึ่งจุดห้าลิตร")[0], "มันเต็ม<N1>ลิตร")

    def test_number_words_are_read_as_numbers(self):
        for said, number in (("สามสิบห้า", 35), ("เจ็ดสิบห้า", 75), ("ยี่สิบเอ็ด", 21), ("หนึ่งจุดห้า", 1.5),
                             ("สองครึ่ง", 2.5), ("หนึ่งพันเก้าร้อยเก้าสิบ", 1990), ("สองล้านห้าแสน", 2500000),
                             ("ร้อย", 100), ("1,990", 1990), ("4.5", 4.5)):
            self.assertEqual(thai.value(said), number, said)
        self.assertEqual(thai.values("อายุเพิ่มจาก 32 เป็นสามสิบห้า"), {32, 35})

    def test_numbers_are_locked_and_only_come_back_exactly(self):
        locked, numbers = thai.lock("ราคาเจ็ดสิบห้าบาท ลด 20% เหลือ 1,990")
        self.assertEqual(locked, "ราคา<N1>บาท ลด <N2>% เหลือ <N3>")
        self.assertEqual(numbers, {"N1": "เจ็ดสิบห้า", "N2": "20", "N3": "1,990"})
        self.assertEqual(thai.unlock("ราคา <N1> บาทครับ ลด <N2>% เหลือ <N3>", numbers),
                         "ราคา เจ็ดสิบห้า บาทครับ ลด 20% เหลือ 1,990")
        for wrong in ("ราคา 75 บาท <N2> <N3>", "<N1> <N1> <N2> <N3>", "<N1> <N2> <N3> <N4>", "<N1> <N2>"):
            self.assertIsNone(thai.unlock(wrong, numbers), wrong)


class Fix(unittest.TestCase):
    def test_a_number_is_carried_back_exactly_or_the_line_keeps_what_was_heard(self):
        heard = ["ราคาเจ็ดสิบห้าบาท", "ใส่ทุกอยางเขาไป", "ซื้อสองแถมหนึ่ง"]
        ask, asked = fixes({"lines": ["ราคา 75 บาท", "ใส่ทุกอย่างเข้าไป", "ซื้อ <N2> แถม <N3>"]},
                           {"lines": ["ราคา <N1> บาท"]})
        fixed, kept = hearing.fix(ask, heard, "Product review")
        self.assertEqual(fixed, ["ราคา เจ็ดสิบห้า บาท", "ใส่ทุกอย่างเข้าไป", "ซื้อ สอง แถม หนึ่ง"])
        self.assertEqual(kept, [None, None, None])
        first = asked[0][0][1]["content"]
        self.assertIn("1. ราคา<N1>บาท\n2. ใส่ทุกอยางเขาไป\n3. ซื้อ<N2>แถม<N3>", first, "no two lines share a name")
        self.assertNotIn("เจ็ดสิบห้า", first)
        self.assertEqual(asked[1][0][1]["content"], "Lines to fix:\n1. ราคา<N1>บาท", "only the failed line again")
        self.assertEqual(asked[0][1]["properties"]["lines"]["minItems"], 3)

    def test_a_line_rewritten_too_much_or_an_answer_that_cannot_be_read_keeps_what_was_heard(self):
        ask, asked = fixes({"lines": ["อะไรก็ไม่รู้ที่ไม่ได้พูดเลยสักคำเดียวนะ", "ดีครับ"]}, "not json",)
        fixed, kept = hearing.fix(ask, ["ดีมากครับ", "ดีครบ"], "Game stream")
        self.assertEqual(fixed, ["ดีมากครับ", "ดีครับ"])
        self.assertTrue(kept[0].startswith("Muse's answer could not be read"), kept)
        self.assertIsNone(kept[1])
        self.assertEqual(len(asked), 2)

    def test_a_name_written_on_the_screen_may_replace_a_misheard_word(self):
        ask, asked = fixes({"lines": ["Eaglet", "ไปที่ ashiran.com"]})
        fixed, kept = hearing.fix(ask, ["egret", "ไปที่ achilan dotcom"], "Tutorial",
                                  names=["Eaglet", "ashiran.com", "Settings"])
        self.assertEqual((fixed, kept), (["Eaglet", "ไปที่ ashiran.com"], [None, None]))
        self.assertIn("Written on the screen at the time (when the speaker says one of these, spell it as written "
                      "here): Eaglet, ashiran.com, Settings\n\n", asked[0][0][1]["content"])
        ask, asked = fixes({"lines": ["Eaglet"]}, {"lines": ["Eaglet"]})
        fixed, kept = hearing.fix(ask, ["egret"], "Tutorial")
        self.assertEqual((fixed, kept), (["egret"], ["more than 40% of it changed"]), "without the screen, too much")
        self.assertNotIn("Written on the screen", asked[0][0][1]["content"])

    def test_named_lists_the_names_a_fix_took_from_the_screen(self):
        self.assertEqual(hearing.named("ดู egret", "ดู Eaglet", ["Eaglet", "Local"]), ["Eaglet"])
        self.assertEqual(hearing.named("ดู Eaglet", "ดู Eaglet", ["Eaglet"]), [], "it was heard already")

    def test_many_lines_go_a_few_at_a_time_with_what_came_before_as_context(self):
        heard = ["คำว่า" + letter for letter in "กขคฆงจฉชซฌญฎฏฐฑ"]
        ask, asked = fixes({"lines": heard[:12]}, {"lines": heard[12:]})
        fixed, kept = hearing.fix(ask, heard, "Game stream", before=["ก่อนหน้า"])
        self.assertEqual((fixed, kept), (heard, [None] * 15))
        self.assertEqual([len(q[1]["content"].split("Lines to fix:\n")[1].splitlines()) for q, _ in asked], [12, 3])
        self.assertIn("Heard just before (context only, do not answer these):\nก่อนหน้า", asked[0][0][1]["content"])


@unittest.skipUnless(HAS_NUMPY, "needs numpy")
class Ears(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("ears", os.path.join(ROOT, "hearing", "ears.py"))
        cls.ears = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ears)

    def test_long_sound_is_cut_at_its_quietest_tenth_between_20_and_30_s(self):
        import numpy as np
        rate = self.ears.RATE
        audio = np.full(75 * rate, 0.3, dtype=np.float32)
        audio[int(26.0 * rate):int(26.1 * rate)] = 0
        audio[int(48.5 * rate):int(48.7 * rate)] = 0
        points = self.ears.cut_points(audio)
        self.assertEqual([round(p / rate, 2) for p in points], [0, 26.05, 48.6, 75])

    def test_a_phrase_is_timed_by_its_letters_and_ends_at_a_space(self):
        letters = [(10, "ร", 0.9), (11, "", 0.99), (12, "า", 0.8), (15, "ค", 0.7), (16, "า", 0.6), (18, "", 0.99),
                   (20, " ", 0.99), (40, "บ", 0.5), (41, "", 0.99), (42, " ⁇ ", 0.2), (45, "ท", 0.9)]
        self.assertEqual(self.ears.phrases(letters, 30.0, 0.02),
                         [{"from": 30.2, "to": 30.34, "text": "ราคา", "sure": 0.75},
                          {"from": 30.8, "to": 30.92, "text": "บ⁇ท", "sure": 0.533}])


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Look(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.made = tempfile.mkdtemp()
        cls.source = os.path.join(cls.made, "moment.mp4")
        make_moment(cls.source)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.made, ignore_errors=True)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.videos = os.path.join(self.tmp, "videos")
        os.makedirs(self.videos)
        self.clip = os.path.join(self.videos, "review.mp4")
        shutil.copy(self.source, self.clip)
        os.environ["LCS_VIDEOS"] = self.videos
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        self.smallest, recording.SMALLEST = recording.SMALLEST, 0
        self.url = hearing.URL
        self.worker = Worker([{"from": 1.0, "to": 2.5, "text": "ส⁇นค้าราคาเจ็ดสิบห้าบาท"},
                              {"from": 4.0, "to": 5.0, "text": "ใส่ทุกอยางเขาไป"}])
        hearing.URL = self.worker.url
        self.chat = agent.Chat()

    def tearDown(self):
        hearing.URL = self.url
        recording.SMALLEST = self.smallest
        self.worker.close()
        agent.CHATS.clear()
        agent._shown.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_job(self, *script):
        brain = ScriptedMind(*script)
        self.chat.jobs += 1
        job = self.chat.job = agent.Job(self.chat, brain, "make a clip")
        return brain, job

    def steps(self):
        return [e for e in self.chat.events if e["kind"] == "step"]

    def test_a_look_says_what_is_said_by_time_as_heard_and_hears_a_stretch_once(self):
        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("look_at_piece", start="0:30", end="0:50", why="look")],
                                  [call("look_at_piece", start="0:30", end="0:50", frames=2, why="again, closer")],
                                  "Done.")
        job.run()
        self.assertEqual(len(self.worker.got), 1, "the second look of the same stretch was not heard again")
        self.assertEqual(len(self.worker.got[0]), (20 + 2 * tools.SPILL) * 16000 * 2,
                         "20 s of 16 kHz mono 16-bit sound, and SPILL s more at each side")
        for step in self.steps()[1:]:
            self.assertIn("What is said here, by time, as heard", step["result"])
            self.assertIn("0:31 สินค้าราคาเจ็ดสิบห้าบาท\n0:34 ใส่ทุกอยางเขาไป", step["result"])
        self.assertEqual([p["from"] for p in self.steps()[1]["look"]["said"]], [31.0, 34.0])
        self.assertEqual(len(brain.sent), 4, "a look asks Muse nothing aside")

    def test_the_proposed_moments_get_their_words_fixed_from_context_with_numbers_as_heard(self):
        fixed = json.dumps({"lines": ["สินค้าราคา <N1> บาท", "ใส่ทุกอย่างเข้าไป"]}, ensure_ascii=False)
        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("look_at_piece", start="0:30", end="0:50", why="look")],
                                  [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the price"}])],
                                  fixed, "Done.")
        job.run()
        aside = brain.sent[3]
        self.assertEqual(aside[0]["role"], "system")
        self.assertIn("1. สินค้าราคา<N1>บาท\n2. ใส่ทุกอยางเขาไป", aside[1]["content"])
        step = self.steps()[-1]
        said = step["moments"][0]["said"]
        self.assertEqual([(p["from"], p["text"], p["heard"], p["kept"]) for p in said],
                         [(31.0, "สินค้าราคา เจ็ดสิบห้า บาท", "สินค้าราคาเจ็ดสิบห้าบาท", None),
                          (34.0, "ใส่ทุกอย่างเข้าไป", "ใส่ทุกอยางเขาไป", None)])
        self.assertIn("   0:31 สินค้าราคา เจ็ดสิบห้า บาท\n   0:34 ใส่ทุกอย่างเข้าไป", step["result"])
        self.assertIsNotNone(step["fixing"])

    def test_stop_while_hearing_ends_the_look_at_once(self):
        self.worker.delay = 10
        seen = {}

        def press_stop():
            for _ in range(200):
                live = self.chat.live
                if live and str(live.get("work", "")).startswith("Writing down what is said"):
                    seen["at"] = time.time()
                    self.chat.stop()
                    return
                time.sleep(0.05)

        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("look_at_piece", start="0:30", end="0:50", why="look")], "Done.")
        presser = threading.Thread(target=press_stop)
        presser.start()
        job.run()
        presser.join()
        self.assertLess(time.time() - seen["at"], 1.5)
        self.assertEqual(self.steps()[-1]["result"], "Not looked at: the person pressed Stop.")
        self.assertEqual(self.chat.looks, [])
        self.assertIsNone(hearing.heard(recording.probe(self.clip), 30, 50))

    def test_stop_while_fixing_the_words_ends_the_proposal_and_keeps_nothing(self):
        from clipstudio import mind

        def press_stop(asking):
            self.chat.stop()
            raise mind.Cancelled()

        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("look_at_piece", start="0:30", end="0:50", why="look")],
                                  [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "x"}])],
                                  press_stop)
        job.run()
        self.assertEqual(self.steps()[-1]["result"], "Not proposed: the person pressed Stop.")
        self.assertIsNone(self.chat.moments)
        self.assertEqual(self.chat.events[-1]["why"], "stopped")

    def test_when_hearing_is_down_the_look_still_shows_and_says_why(self):
        self.worker.close()
        hearing.URL = "http://127.0.0.1:9"
        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("look_at_piece", start="0:30", end="0:50", why="look")], "Done.")
        job.run()
        result = self.steps()[-1]["result"]
        self.assertIn("What is said here could not be heard: The hearing worker did not answer", result)
        self.assertEqual(len(self.chat.looks), 1)
        self.worker = Worker()

    def test_muse_down_for_the_fix_keeps_the_lines_as_heard(self):
        def broken(asking):
            from clipstudio import mind
            raise mind.MindError("Muse is not answering")

        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("look_at_piece", start="0:30", end="0:50", why="look")],
                                  [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the price"}])],
                                  broken, "Done.")
        job.run()
        step = self.steps()[-1]
        self.assertFalse(step["refused"], step["result"])
        self.assertIn("   0:31 สินค้าราคาเจ็ดสิบห้าบาท (as heard)", step["result"])
        self.assertEqual(step["moments"][0]["said"][0]["kept"], "Muse could not be asked (Muse is not answering)")

    def test_a_recording_with_no_sound_is_not_sent_to_be_heard(self):
        quiet = os.path.join(self.videos, "quiet.mp4")
        make_moment(quiet, seconds=30, busy=(20, 25), sound=False)
        brain, job = self.run_job([call("open_recording", path=quiet, why="open it")],
                                  [call("look_at_piece", start="0:10", end="0:20", why="look")], "Done.")
        job.run()
        self.assertIn("The recording has no sound, so nothing is said.", self.steps()[-1]["result"])
        self.assertEqual(self.worker.got, [])


class Status(unittest.TestCase):
    def setUp(self):
        self.url, self.why = hearing.URL, hearing._state["why"]

    def tearDown(self):
        hearing.URL, hearing._state["why"] = self.url, self.why

    def test_the_status_says_what_is_ready_or_why_not(self):
        worker = Worker()
        hearing.URL = worker.url
        try:
            got = hearing.status()
        finally:
            worker.close()
        self.assertEqual((got["ok"], got["name"]), (True, "omniASR_CTC_1B_v2, float32, 8 threads"))
        hearing.URL = "http://127.0.0.1:9"
        with mock.patch("shutil.which", return_value=None):
            hearing._start()
        self.assertIn("Docker is not installed", hearing.status()["why"])

    def test_an_image_not_built_yet_says_how_to_build_it(self):
        hearing.URL = "http://127.0.0.1:9"
        done = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch("shutil.which", return_value="docker"), \
                mock.patch("subprocess.run", return_value=done) as run:
            hearing._start()
        self.assertEqual(run.call_args_list[0][0][0], ["docker", "images", "-q", hearing.IMAGE])
        self.assertIn("docker build -t local-clip-studio-hearing hearing", hearing.status()["why"])


if __name__ == "__main__":
    unittest.main()
