"""Looking at pieces and proposing moments, on small videos made by ffmpeg: python -m unittest discover -s tests"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, hearing, previews, recording, server, tools, words  # noqa: E402
from test_agent_loop import ScriptedMind, call  # noqa: E402
from test_hearing import Worker  # noqa: E402
from test_map import make_moment  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,height",
                          "-of", "json", path], capture_output=True, check=True).stdout
    return json.loads(out)


class Times(unittest.TestCase):
    def test_times_are_read_as_the_map_writes_them(self):
        read = recording.seconds_of
        self.assertEqual([read("1:51:44"), read("35:43"), read("6704"), read(" 6704.5 "), read("1:51:44.5"),
                          read("0:07")], [6704, 2143, 6704, 6704.5, 6704.5, 7])
        for wrong in ("1:60", "abc", "", "-5", "1::2", "inf", "1:2:3:4", "nan"):
            self.assertIsNone(read(wrong), wrong)

    def test_markdown_is_taken_out_of_answers(self):
        self.assertEqual(words.plain("**Done.** Here:\n* one\n  + two\n## Next\n- three, 2 * 3 = 6"),
                         "Done. Here:\n- one\n  - two\nNext\n- three, 2 * 3 = 6")


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Moments(unittest.TestCase):
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
        self.clip = os.path.join(self.videos, "stream.mp4")
        shutil.copy(self.source, self.clip)
        os.environ["LCS_VIDEOS"] = self.videos
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        self.smallest, recording.SMALLEST = recording.SMALLEST, 0
        self.worker, self.url = Worker(), hearing.URL
        hearing.URL = self.worker.url
        self.chat = agent.Chat()

    def tearDown(self):
        hearing.URL = self.url
        self.worker.close()
        recording.SMALLEST = self.smallest
        agent.CHATS.clear()
        agent._shown.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_job(self, *script):
        brain = ScriptedMind(*script)
        self.chat.jobs += 1
        job = self.chat.job = agent.Job(self.chat, brain, "make a clip")
        job.run()
        return brain, [e for e in self.chat.events if e.get("job") == job.n and e["kind"] == "step"]

    def previews_ready(self, n, within=60):
        until = time.time() + within
        while time.time() < until:
            ready = [e for e in self.chat.events if e["kind"] == "preview"]
            if len(ready) >= n:
                return ready
            time.sleep(0.1)
        self.fail("only %d of %d previews were cut" % (len(ready), n))

    def opened(self):
        return [call("open_recording", path=self.clip, why="open it")]

    def test_each_picture_of_a_look_comes_right_after_its_time(self):
        brain, steps = self.run_job(self.opened(), [call("map_recording", why="where to look", stretch_s=20)],
                                    [call("look_at_piece", start="0:30", end="0:50", why="the top stretch")], "Done.")
        look = steps[-1]["look"]
        self.assertEqual(steps[-1]["label"], "Looked at a piece")
        self.assertEqual([p["t"] for p in look["pictures"]], [32.5, 37.5, 42.5, 47.5])
        sizes = [os.path.getsize(os.path.join(recording.WORK, "frames", p["frame"])) for p in look["pictures"]]
        self.assertGreater(sizes[2], 3 * sizes[0], "the picture at 0:42 shows the moving picture, 0:32 is black")
        result = steps[-1]["result"]
        self.assertTrue(result.startswith("Look 1: 0:30-0:50 (20 s). Its 4 pictures come next"), result)
        self.assertRegex(result, r"The sound rises most at 0:4[4-6], \+\d")
        self.assertIn("look earlier if the start of the action is missing", result)
        sent = brain.sent[-1][-1]
        self.assertEqual(sent["role"], "user")
        kinds = [part["type"] for part in sent["content"]]
        self.assertEqual(kinds, ["text", "image_url"] * 4)
        captions = [part["text"] for part in sent["content"] if part["type"] == "text"]
        self.assertEqual(captions, ["[studio] Look 1, picture 1 of 4, at 0:32:", "Look 1, picture 2 of 4, at 0:38:",
                                    "Look 1, picture 3 of 4, at 0:42:", "Look 1, picture 4 of 4, at 0:48:"])

    def test_a_look_past_the_end_is_cut_to_the_end_and_a_wrong_look_says_what_to_send(self):
        brain, steps = self.run_job(self.opened(),
                                    [call("look_at_piece", start="0:50", end="1:30", frames=2, why="the end")],
                                    [call("look_at_piece", start="2:00", end="2:30", why="later")],
                                    [call("look_at_piece", start="0:40", end="0:30", why="backwards")],
                                    [call("look_at_piece", start="1:75", end="1:90", why="badly written")],
                                    [call("look_at_piece", start="0:50", end="1:30", frames=2, why="again")], "Done.")
        self.assertIn("The end was after the end of the recording, so this look ends there, at 1:00.",
                      steps[1]["result"])
        self.assertEqual((steps[1]["look"]["start"], steps[1]["look"]["end"]), (50, 60))
        said = [s["result"] for s in steps[2:]]
        self.assertEqual([s["refused"] for s in steps[2:]], [True] * 4)
        self.assertIn("start 2:00 is after the end of the recording, which is 1:00 long", said[0])
        self.assertIn("end 0:30 must come after start 0:40", said[1])
        self.assertIn('start must be a time as the map writes it, like 1:51:44 or 35:43', said[2])
        self.assertIn("Look 1 already showed 0:50-1:00 in 2 pictures", said[3])
        studio = [m for m in brain.sent[-1] if m["role"] == "user" and isinstance(m["content"], list)]
        self.assertEqual(len(studio), 2, "the opened picture and look 1; the repeated look sent no pictures")

    def test_moments_must_come_from_what_was_looked_at(self):
        moment = {"look": 1, "reason": "the tone"}
        brain, steps = self.run_job(
            self.opened(),
            [call("propose_moments", moments=[moment], why="too early")],
            [call("look_at_piece", start="0:30", end="0:50", why="look")],
            [call("propose_moments", moments=[dict(moment, look=2)], why="a look not made")],
            [call("propose_moments", moments=[dict(moment, start="0:20")], why="outside")],
            [call("propose_moments", moments=[dict(moment, start="0:30", end="0:36")], why="too short")],
            [call("propose_moments", moments=[dict(moment, end="0:45"), dict(moment, start="0:40")], why="overlap")],
            [call("propose_moments", moments=[dict(moment, scene=3)], why="unknown field")],
            [call("propose_moments", moments=[dict(moment, picture=5)], why="no such picture")], "Done.")
        said = [s["result"] for s in steps if s["tool"] == "propose_moments"]
        self.assertTrue(all(s["refused"] for s in steps if s["tool"] == "propose_moments"))
        self.assertIn("You have not looked at any stretch yet", said[0])
        self.assertIn("Moment 1 names look 2, but your looks are: 1. 0:30-0:50", said[1])
        self.assertIn("Moment 1 (0:20-0:50) is not inside look 1 (0:30-0:50)", said[2])
        self.assertIn("Moment 1 is 6 s long; a clip is 10 to 90 s", said[3])
        self.assertIn("Moments 1 and 2 overlap (0:30-0:45 and 0:40-0:50)", said[4])
        self.assertIn("moment 1 has no field named scene", said[5])
        self.assertIn("Look 1 has 4 pictures, so moment 1 cannot show picture 5", said[6])
        self.assertIsNone(self.chat.moments)
        self.assertEqual([e for e in self.chat.events if e["kind"] == "preview"], [])

    def test_moments_become_cards_and_each_gets_a_preview_that_plays(self):
        brain, steps = self.run_job(
            self.opened(), [call("map_recording", why="where to look", stretch_s=20)],
            [call("look_at_piece", start="0:30", end="0:50", frames=5, why="the top stretch")],
            [call("look_at_piece", start="0:05", end="0:25", frames=2, why="something calm")],
            [call("propose_moments", why="the best first", moments=[
                {"look": "1", "reason": "the picture moves and the tone rings"},
                json.dumps({"look": 2, "reason": "a calm start", "end": "0:20", "picture": 1})])], "Done.")
        step = steps[-1]
        self.assertFalse(step["refused"], step["result"])
        first, second = step["moments"]
        self.assertEqual((first["start"], first["end"], first["look"]), (30, 50, 1))
        self.assertEqual((second["start"], second["end"], second["t"]), (5, 20, 10))
        self.assertEqual(first["t"], 44, "with no picture named, the card shows the one nearest the loudest second")
        self.assertIn("Otherwise go on with moment 1 by yourself and make the clip from it: do not ask the person to "
                      "choose", step["result"])
        self.assertIn("moments proposed: 1. 0:30-0:50, 2. 0:05-0:20 · no clip made yet", brain.sent[-1][-1]["content"])
        self.assertIn("never tell the person a clip is ready when no tool made it", step["result"])
        ready = self.previews_ready(2)
        self.assertEqual([(e["name"], e["ok"]) for e in ready], [(first["preview"], True), (second["preview"], True)])
        made = ffprobe(os.path.join(previews.folder(), first["preview"]))
        self.assertAlmostEqual(float(made["format"]["duration"]), 20, delta=0.3)
        self.assertEqual(sorted(s["codec_type"] for s in made["streams"]), ["audio", "video"])
        self.assertEqual([s["height"] for s in made["streams"] if s["codec_type"] == "video"], [480])
        self.assertEqual([n for n in os.listdir(previews.folder()) if ".part" in n], [])

    def test_a_preview_that_cannot_be_told_about_does_not_stop_the_next(self):
        class Gone:
            def emit(self, *args, **fields):
                raise OSError("the chat's folder is gone")

        rec = recording.probe(self.clip)
        previews.cut(Gone(), 1, rec, 30, 40)
        name = previews.cut(self.chat, 1, rec, 40, 50)
        self.assertEqual([(e["name"], e["ok"]) for e in self.previews_ready(1)], [(name, True)])

    def test_an_earlier_chat_keeps_its_looks_moments_and_previews(self):
        self.run_job(self.opened(), [call("look_at_piece", start="0:30", end="0:50", why="look")],
                     [call("propose_moments", moments=[{"look": 1, "reason": "the moment"}], why="propose")], "Done.")
        name = self.previews_ready(1)[0]["name"]
        with open(self.chat.log_path, encoding="utf-8") as f:
            rows = [r for r in f if '"kind": "preview"' not in r]
        with open(self.chat.log_path, "w", encoding="utf-8") as f:
            f.writelines(rows)
        again = agent.Chat.load(self.chat.id)
        self.assertEqual([look["n"] for look in again.looks], [1])
        self.assertEqual([m["start"] for m in again.moments], [30])
        told = [e for e in again.events if e["kind"] == "preview"]
        self.assertEqual([(e["name"], e["ok"]) for e in told], [(name, True)])
        os.remove(os.path.join(previews.folder(), name))
        with open(again.log_path, "w", encoding="utf-8") as f:
            f.writelines(rows)
        lost = [e for e in agent.Chat.load(self.chat.id).events if e["kind"] == "preview"]
        self.assertEqual((lost[0]["ok"], lost[0]["error"]), (False, "The studio was closed before this preview was cut."))

    def test_opening_another_recording_forgets_the_looks(self):
        other = os.path.join(self.videos, "other.mp4")
        shutil.copy(self.source, other)
        self.run_job(self.opened(), [call("look_at_piece", start="0:30", end="0:50", why="look")],
                     [call("open_recording", path=other, why="the other one")],
                     [call("propose_moments", moments=[{"look": 1, "reason": "x"}], why="propose")], "Done.")
        self.assertEqual(self.chat.looks, [])
        last = [e for e in self.chat.events if e["kind"] == "step"][-1]
        self.assertIn("You have not looked at any stretch yet", last["result"])

    def test_a_call_with_no_plan_step_is_told_the_plan_did_not_move(self):
        brain, steps = self.run_job([call("set_plan", steps=["open it", "look"], why="plan")],
                                    [call("open_recording", path=self.clip, why="open it")],
                                    [call("look_at_piece", start="0:30", end="0:50", why="look", plan_step=2)],
                                    "Done.")
        self.assertIn("named no plan_step, so the plan on the page did not move", steps[1]["result"])
        self.assertNotIn("named no plan_step", steps[0]["result"] + steps[2]["result"])

    def test_markdown_in_an_answer_is_not_shown_or_read_back(self):
        self.run_job("**Done.**\n* it is open")
        answer = [e for e in self.chat.events if e["kind"] == "answer"][-1]
        self.assertEqual(answer["text"], "Done.\n- it is open")
        self.assertEqual(self.chat.messages[-1]["content"], "Done.\n- it is open")


class Serve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        os.makedirs(previews.folder())
        self.name = "0123456789abcdef.mp4"
        self.body = bytes(range(256)) * 40
        with open(os.path.join(previews.folder(), self.name), "wb") as f:
            f.write(self.body)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def get(self, name, byte_range=None):
        req = urllib.request.Request("http://127.0.0.1:%d/api/preview/%s" % (self.http.server_port, name))
        if byte_range:
            req.add_header("Range", byte_range)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def test_a_preview_is_sent_in_the_range_the_browser_asks_for(self):
        status, headers, body = self.get(self.name)
        self.assertEqual((status, headers["Content-Type"], headers["Accept-Ranges"], body),
                         (200, "video/mp4", "bytes", self.body))
        status, headers, body = self.get(self.name, "bytes=100-199")
        self.assertEqual((status, headers["Content-Range"], body), (206, "bytes 100-199/10240", self.body[100:200]))
        status, headers, body = self.get(self.name, "bytes=10000-")
        self.assertEqual((status, body), (206, self.body[10000:]))
        status, headers, body = self.get(self.name, "bytes=-40")
        self.assertEqual((status, headers["Content-Range"], body), (206, "bytes 10200-10239/10240", self.body[-40:]))
        self.assertEqual(self.get(self.name, "bytes=20000-")[0], 416)
        self.assertEqual(self.get("fedcba9876543210.mp4")[0], 404)
        self.assertEqual(self.get("..%5C..%5Cx.mp4")[0], 404)


if __name__ == "__main__":
    unittest.main()


class Snap(unittest.TestCase):
    LOOK = {"start": 30.0, "end": 80.0, "said": [{"from": 31.0, "to": 34.5, "text": "a"},
                                                 {"from": 46.0, "to": 49.0, "text": "b"},
                                                 {"from": 60.0, "to": 70.0, "text": "c"}]}

    def test_edges_inside_a_line_move_out_of_it_with_a_little_air(self):
        self.assertEqual(tools._snapped(self.LOOK, 32.0, 48.0)[:2], (30.75, 49.25))
        self.assertIn("starts 1.2 s earlier", tools._snapped(self.LOOK, 32.0, 48.0)[2])
        self.assertIn("ends 1.2 s later", tools._snapped(self.LOOK, 32.0, 48.0)[2])

    def test_edges_in_a_pause_or_far_inside_a_long_line_stay(self):
        self.assertEqual(tools._snapped(self.LOOK, 40.0, 55.0), (40.0, 55.0, None))
        self.assertEqual(tools._snapped(self.LOOK, 65.0, 75.0), (65.0, 75.0, None), "5 s into a line is too far")

    def test_an_end_inside_a_line_finishes_it_or_ends_before_it_never_inside(self):
        """Tarm 10-30 on EP 0's talk, its end 501.3 inside a line said 499.37 to 505.7: "ตัดห้วนเกินไป มันยังไม่จบ"."""
        ep0 = {"start": 450.0, "end": 510.0, "said": [{"from": 497.28, "to": 499.27, "text": "a"},
                                                      {"from": 499.37, "to": 505.7, "text": "b"}]}
        a, b, said = tools._snapped(ep0, 456.3, 501.3)
        self.assertEqual((a, b), (456.3, 505.95))
        self.assertIn("s later, so it closes after a line rather than inside it", said)
        long = dict(self.LOOK, said=self.LOOK["said"][:2] + [{"from": 60.0, "to": 85.0, "text": "c"}])
        a, b, said = tools._snapped(long, 40.0, 66.0)
        self.assertEqual((a, b), (40.0, 59.75), "19 s of the line left: it ends before the line, not inside it")
        self.assertIn("ends 6.2 s earlier, so it closes before a line it would cut through", said)
        self.assertEqual(tools._snapped(long, 52.0, 66.0)[1], 66.0, "ending before the line would leave under 10 s")
        self.assertEqual(tools._snapped(dict(ep0, start=400.0), 415.0, 501.3)[1], 499.12,
                         "finishing the line would run past 90 s, so it ends before it")

    def test_a_sellers_end_takes_in_the_price_said_right_after_it(self):
        """Tarm's green-screen seller, 10-30: the controller's moment ended at 0:39, its price was said at 0:41."""
        look = {"start": 9.0, "end": 39.0, "heard": [3.0, 45.0], "said": [
            {"from": 33.9, "to": 38.6, "text": "แล้วก็กดยิงได้เลยนะครับจอยนี้สนนราคาอยู่ที่"},
            {"from": 40.9, "to": 41.9, "text": "สามพันบาทนะครับ"},
            {"from": 42.3, "to": 44.3, "text": "ใครอยากได้กด CF ไปได้เลย"}]}
        a, b, said = tools._snapped(look, 9.0, 39.0, price=True)
        self.assertEqual(b, 42.15)
        self.assertIn("ends 3.1 s later, so the price said right after it is in the clip", said)
        self.assertEqual(tools._snapped(look, 9.0, 39.0)[1], 39.0, "not a seller: the end stays")
        self.assertEqual(tools._snapped(dict(look, heard=[3.0, 41.0]), 9.0, 39.0, price=True)[1], 39.0,
                         "a price not heard whole is not reached for")
        self.assertEqual(tools._snapped(look, 9.0, 32.0, price=True)[1], 32.0, "8.9 s on is past PRICE_REACH")
        self.assertEqual(tools._snapped(look, 9.0, 34.0, price=True)[1], 42.15, "inside the line before the price, "
                         "the end finishes the line, and the price follows it")

    def test_edges_never_leave_the_look(self):
        look = dict(self.LOOK, start=30.9)
        self.assertEqual(tools._snapped(look, 31.5, 48.0)[0], 30.9)

    def test_an_end_finishes_its_line_in_what_was_heard_past_the_look(self):
        look = dict(self.LOOK, end=67.0, heard=[24.0, 73.0])
        self.assertEqual(tools._snapped(look, 50.0, 67.0)[1], 70.25, "the line runs 3 s past the look: heard whole, "
                         "so the clip ends after it")
        self.assertEqual(tools._snapped(dict(self.LOOK, end=67.0), 50.0, 67.0)[1], 67.0, "a look heard only to its end "
                         "keeps it")
        self.assertEqual(tools._snapped(look, 50.0, 65.0)[1], 70.25, "5 s from the line's end: it finishes it")

    SELLER = {"start": 80.0, "end": 110.0, "heard": [74.32, 115.96], "said": [
        {"from": 75.16, "to": 75.30, "text": "ลิต"},
        {"from": 76.02, "to": 78.62, "text": "ซึ่งการเก็บความชื้นได้สูงสุด"},
        {"from": 79.10, "to": 79.82, "text": "เจ็ด้ยห้าสิ0"},
        {"from": 80.12, "to": 83.86, "text": "มิลิลิต์ต่อวันนี่ยเท่ากับว่าถ้าบ้านไหนชื้นมากก็คือเก็บสองวันก็เต็มละ"},
        {"from": 83.96, "to": 87.93, "text": "แต่ว่าถ้าบ้านไหนไม่ได้ชื้นนะครับก็อาจจะหลายวันกว่าจะเต็ม"},
        {"from": 88.69, "to": 91.35, "text": "ซึ่งวิธีการใช้ง่ายมากครับเราก็แค่ใส่ทุกอย่างเข้าไป"}]}

    def test_a_start_on_a_unit_moves_back_to_the_number_and_what_it_measures(self):
        a, b, said = tools._snapped(self.SELLER, 80.0, 110.0)
        self.assertEqual((a, b), (75.77, 110.0), "the seller's review opened on มิลลิลิตร, the 750 said before it")
        self.assertIn("starts 4.2 s earlier", said)
        self.assertIn('"มิลิลิต์ต่อวัน', said)

    def test_a_line_after_a_number_and_a_unit_after_a_misheard_one_go_on_from_it(self):
        model = dict(self.SELLER, said=[{"from": 79.10, "to": 79.82, "text": "ใช้โมเดลยี่สิบเจ็ด"},
                                        {"from": 80.12, "to": 83.86, "text": "บีรันบนการ์ดนี้ได้"}])
        self.assertEqual(tools._snapped(model, 80.0, 110.0)[0], 78.85, "27 B")
        unit = dict(self.SELLER, said=[{"from": 79.10, "to": 79.82, "text": "เก็บได้สูงสุดเจ็ด้ยห้าสิ"},
                                       {"from": 80.12, "to": 83.86, "text": "มิลิลิตต่อวัน"}])
        self.assertEqual(tools._snapped(unit, 80.0, 110.0)[0], 78.85, "the hearing dropped the บ of ห้าสิบ")

    def test_a_joining_word_a_breath_or_what_was_not_heard_keeps_the_start(self):
        self.assertEqual(tools._snapped(self.SELLER, 83.9, 110.0)[0], 83.9, "แต่ว่า after no number")
        self.assertEqual(tools._snapped(self.SELLER, 88.5, 110.0)[0], 88.5, "ซึ่ง after a breath")
        self.assertEqual(tools._snapped(dict(self.SELLER, heard=[79.2, 115.96]), 80.0, 110.0)[0], 80.0)
        self.assertEqual(tools._snapped(self.SELLER, 80.0, 166.0)[0], 78.85, "no further than a clip may be long")
        done = dict(self.SELLER, said=[{"from": 79.10, "to": 79.82, "text": "ประกอบเสร็จเรียบร้อย"},
                                       {"from": 80.12, "to": 83.86, "text": "ต่อไปดูวิธีใช้กันครับ"}])
        self.assertEqual(tools._snapped(done, 80.0, 110.0)[0], 80.0, "เรียบร้อย ends in ร้อย and is no number")


class Faces(unittest.TestCase):
    def test_faces_found_over_the_moment_serve_its_pieces(self):
        found = {"every": 0.5, "samples": [{"t": 30.0 + k * 0.5, "faces": [[100, 50, 40, 40, 0.9]]} for k in range(40)]}
        had = tools.faces.kept
        tools.faces.kept = lambda rec, a, b, every=0.5: found if (a, b) == (30.0, 50.0) else None
        try:
            state = {"clip": {"pieces": [[31.0, 35.0], [40.0, 49.0]], "span": [30.0, 50.0], "product": None,
                              "layout": tools.layouts.make("subject", (320, 180))}, "length": 13.0}
            times, _ = tools._face_times(state, {"width": 320, "height": 180})
        finally:
            tools.faces.kept = had
        self.assertEqual(len(times), 9 + 19, "the pictures inside the pieces, none from the stretch cut out")
        self.assertEqual([t for t, _ in times][:2] + [t for t, _ in times][-1:], [0.0, 0.5, 13.0])


class InClip(unittest.TestCase):
    def test_a_phrase_is_placed_in_clip_seconds_and_ends_with_its_piece(self):
        pieces = [[30.0, 35.0], [40.0, 50.0]]
        self.assertEqual(tools._in_clip(pieces, {"from": 41.0, "to": 43.0}, 15.0), (6.0, 8.0))
        self.assertEqual(tools._in_clip(pieces, {"from": 33.0, "to": 37.0}, 15.0), (3.0, 5.0),
                         "its end was cut out, so it ends where its piece does, not at the clip's end")
        self.assertIsNone(tools._in_clip(pieces, {"from": 36.0, "to": 38.0}, 15.0))


def said_line(a, b, text):
    return {"from": a, "to": b, "text": text}


class Pieces(unittest.TestCase):
    SAID = [said_line(31.0, 33.0, "one two three"), said_line(36.0, 38.0, "four five"),
            said_line(40.0, 43.0, "this is the long line we say"),
            said_line(46.0, 49.5, "this is the long line we say again")]

    def test_a_game_is_kept_whole(self):
        self.assertEqual(tools._pieces(self.SAID, 30.0, 50.0, None), ([[30.0, 50.0]], None))

    def test_still_silences_the_first_take_and_the_edges_are_cut(self):
        pieces, said = tools._pieces(self.SAID, 30.0, 50.0, lambda x, y: True)
        self.assertEqual(pieces, [[30.7, 33.3], [35.7, 38.3], [45.75, 49.8]])
        self.assertIn('the first take of a line said again at 0:40 ("this is the long line we say"; the take from 0:46 '
                      'is kept)', said)
        self.assertIn("1 silence longer than 1 s while nothing moved on the picture", said)
        self.assertIn("opens and closes 0.3 s from a word", said)

    def test_a_silence_while_the_picture_moves_is_kept(self):
        pieces, _ = tools._pieces(self.SAID, 30.0, 50.0, lambda x, y: False)
        self.assertEqual(pieces, [[30.0, 40.0], [45.75, 50.0]], "only the first take goes")
        pieces, _ = tools._pieces(self.SAID, 30.0, 50.0, lambda x, y: 33.0 <= x and y <= 36.0)
        self.assertEqual(pieces, [[30.0, 33.3], [35.7, 40.0], [45.75, 50.0]])

    def test_nothing_heard_keeps_the_moment(self):
        self.assertEqual(tools._pieces([], 30.0, 50.0, lambda x, y: True), ([[30.0, 50.0]], None))


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Still(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")
        path = os.path.join(cls.tmp, "busy.mp4")
        make_moment(path, seconds=30, busy=(20, 25))
        cls.rec = recording.probe(path)

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_still_is_true_only_where_the_picture_does_not_move(self):
        still = tools._still(self.rec, "explain", 10.0, 30.0, None)
        self.assertTrue(still(12.0, 16.0))
        self.assertFalse(still(21.0, 23.0))
        self.assertFalse(still(18.5, 20.5), "a silence that runs into the movement is kept")
        self.assertTrue(still(26.0, 29.0))

    def test_movement_inside_the_face_camera_does_not_count(self):
        still = tools._still(self.rec, "explain", 10.0, 30.0, [0, 0, 320, 180])
        self.assertTrue(still(21.0, 23.0))

    def test_a_game_cuts_nothing_and_a_talk_cuts_every_silence(self):
        self.assertIsNone(tools._still(self.rec, "game", 10.0, 30.0, None))
        self.assertTrue(tools._still(self.rec, "talk", 10.0, 30.0, None)(21.0, 23.0))

    def test_a_picture_that_cannot_be_looked_at_counts_as_moving(self):
        gone = dict(self.rec, path=os.path.join(self.tmp, "gone.mp4"))
        self.assertFalse(tools._still(gone, "explain", 10.0, 30.0, None)(12.0, 16.0))
