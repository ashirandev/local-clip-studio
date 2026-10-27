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

    def test_edges_never_leave_the_look(self):
        look = dict(self.LOOK, start=30.9)
        self.assertEqual(tools._snapped(look, 31.5, 48.0)[0], 30.9)
