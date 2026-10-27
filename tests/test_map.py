"""The map of a recording, on small videos made by ffmpeg: python -m unittest discover -s tests"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, maps, recording  # noqa: E402
from test_agent_loop import ScriptedMind, call  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def make_moment(path, seconds=60, busy=(40, 50), loud=(44, 46), sound=True):
    """A dark, quiet video with one moment: a moving picture from busy[0] to busy[1], a loud tone inside it.
    A keyframe every second, as a phone writes them."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    inputs = ["-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=%d" % seconds,
              "-f", "lavfi", "-i", "testsrc=s=320x180:r=30:d=%d" % seconds]
    graph = "[0:v][1:v]overlay=enable='between(t,%d,%d)'[v]" % busy
    maps_ = ["-map", "[v]"]
    if sound:
        inputs += ["-f", "lavfi", "-i", "sine=f=440:d=%d" % seconds]
        graph += ";[2:a]volume='if(between(t,%d,%d),1,0.02)':eval=frame[a]" % loud
        maps_ += ["-map", "[a]", "-c:a", "aac"]
    subprocess.run([FFMPEG, "-v", "error", "-y"] + inputs + ["-filter_complex", graph] + maps_ +
                   ["-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-pix_fmt", "yuv420p", path], check=True)


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Map(unittest.TestCase):
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
        self.chat = agent.Chat()

    def tearDown(self):
        recording.SMALLEST = self.smallest
        maps.EXTRA_INPUT = []
        agent.CHATS.clear()
        agent._shown.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_job(self, *script):
        brain = ScriptedMind(*script)
        self.chat.jobs += 1
        job = self.chat.job = agent.Job(self.chat, brain, "make a clip")
        return brain, job

    def results(self):
        return [e["result"] for e in self.chat.events if e["kind"] == "step"]

    def test_the_moment_tops_the_map_with_what_led_up_to_it(self):
        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("map_recording", why="find where to look", stretch_s=20)], "Done.")
        job.run()
        mapped = self.results()[-1]
        self.assertTrue(mapped.startswith("Mapped in"), mapped)
        first = [line for line in mapped.splitlines() if line.startswith("1. ")][0]
        start, end, peak = (int(x) for x in __import__("re").match(
            r"1\. 0:(\d\d)-0:(\d\d), peak 0:(\d\d)", first).groups())
        self.assertTrue(40 <= peak <= 50, first)
        self.assertEqual((end - start, peak - start), (20, 14))
        self.assertIn("about every 1", mapped)
        self.assertIn("Nothing in these stretches has been looked at yet", mapped)

    def test_a_second_map_is_instant_and_a_changed_file_is_read_again(self):
        rec = recording.probe(self.clip)
        maps.measure(rec)
        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("map_recording", why="where to look")], "Done.")
        with mock.patch.object(maps, "measure", side_effect=AssertionError("the file was read again")):
            started = time.time()
            job.run()
        self.assertLess(time.time() - started, 2)
        self.assertTrue(self.results()[-1].startswith("Mapped earlier, so this was instant."), self.results()[-1])
        later = time.time() + 10
        os.utime(self.clip, (later, later))
        self.assertIsNone(maps.kept(rec))

    def test_stop_ends_the_map_at_once_and_keeps_nothing(self):
        maps.EXTRA_INPUT = ["-re"]
        seen = {}

        def press_stop_once_it_maps():
            for _ in range(200):
                live = self.chat.live
                if live and live.get("work"):
                    seen["work"] = live["work"]
                    seen["at"] = time.time()
                    self.chat.stop()
                    return
                time.sleep(0.05)

        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("map_recording", why="where to look")], "Done.")
        presser = threading.Thread(target=press_stop_once_it_maps)
        presser.start()
        job.run()
        ended = time.time()
        presser.join()
        self.assertTrue(seen["work"].startswith("Mapping the video:"), seen)
        self.assertLess(ended - seen["at"], 1.5)
        self.assertEqual(self.results()[-1], "Not mapped: the person pressed Stop. Nothing was kept.")
        self.assertEqual(self.chat.events[-1]["why"], "stopped")
        self.assertIsNone(maps.kept(recording.probe(self.clip)))
        self.assertEqual([n for n in os.listdir(maps.folder()) if n.startswith("making-")], [])

    def test_the_recipe_says_what_ranks(self):
        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("load_recipe", type="seller", why="a product")],
                                  [call("map_recording", why="where to look")], "Done.")
        job.run()
        self.assertIn("Ranked by sound 0, screen 1 (the seller recipe). Stretches of 30 s, each starting 21 s",
                      self.results()[-1])

    def test_a_recording_shorter_than_a_stretch_is_one_stretch(self):
        brain, job = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                  [call("map_recording", why="where to look", stretch_s=90)], "Done.")
        job.run()
        mapped = self.results()[-1]
        self.assertRegex(mapped.splitlines()[0], r"^Mapped in \d+\.\d s and kept")
        self.assertIn("no longer than one stretch of 90 s, so all of it is one stretch:\n1. 0:00-1:00, peak", mapped)

    def test_nothing_open_is_refused(self):
        brain, job = self.run_job([call("map_recording", why="where to look")], "Done.")
        job.run()
        step = [e for e in self.chat.events if e["kind"] == "step"][0]
        self.assertTrue(step["refused"])
        self.assertIn("No recording is open", step["result"])

    def test_a_video_with_no_sound_is_ranked_by_its_picture(self):
        quiet = os.path.join(self.videos, "quiet.mp4")
        make_moment(quiet, seconds=30, busy=(20, 25), sound=False)
        rec = recording.probe(quiet)
        m = maps.measure(rec)
        stretches, about = maps.rank(m, {"sound": 1, "screen": 1}, 10, 3)
        self.assertFalse(about["heard"])
        self.assertTrue(19 <= stretches[0]["peak"] <= 26, stretches[0])
        self.assertIsNone(stretches[0]["louder"])


class Rank(unittest.TestCase):
    def test_flat_sound_is_called_flat_and_stretches_never_overlap(self):
        n = 600
        m = {"seconds": n, "sound": [-30.0 + (t % 7) * 0.1 for t in range(n)],
             "keyframes": [[float(t), 0.9 if t % 97 == 0 else 0.05] for t in range(0, n, 2)]}
        stretches, about = maps.rank(m, {"sound": 1, "screen": 1}, 30, 8)
        self.assertTrue(about["flat"])
        self.assertEqual(len(stretches), 8)
        spans = sorted((s["start"], s["end"]) for s in stretches)
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), spans)
        for s in stretches:
            self.assertEqual(s["end"] - s["start"], 30)
            if 21 <= s["peak"] <= n - 9:
                self.assertEqual(s["peak"] - s["start"], 21)

    def test_silence_before_a_stream_starts_does_not_make_a_peak(self):
        n = 1200
        sound = [-80.0] * 300 + [-25.0 + (t % 5) * 0.2 for t in range(n - 300)]
        sound[100:104] = [-25.0] * 4
        sound[700] = sound[701] = -12.0
        m = {"seconds": n, "sound": sound, "keyframes": [[float(t), 0.05] for t in range(0, n, 2)]}
        stretches, _ = maps.rank(m, {"sound": 1, "screen": 0}, 30, 1)
        self.assertTrue(698 <= stretches[0]["peak"] <= 703, stretches[0])

    def test_the_recipe_weights_decide_what_ranks(self):
        n = 600
        sound = [-25.0] * n
        sound[149:151] = [-10.0] * 2
        sound[445:455] = [-40.0] * 10
        keyframes = [[float(t), 0.9 if t == 450 else 0.15 if 146 <= t <= 154 else 0.05] for t in range(0, n, 2)]
        m = {"seconds": n, "sound": sound, "keyframes": keyframes}
        by_screen, _ = maps.rank(m, {"sound": 0, "screen": 1}, 30, 1)
        by_sound, _ = maps.rank(m, {"sound": 1, "screen": 0}, 30, 1)
        both, _ = maps.rank(m, {"sound": 1, "screen": 1}, 30, 1)
        self.assertTrue(447 <= by_screen[0]["peak"] <= 452, by_screen[0])
        self.assertTrue(147 <= by_sound[0]["peak"] <= 153, by_sound[0])
        self.assertTrue(147 <= both[0]["peak"] <= 153, both[0])

    def test_a_short_recording_is_one_stretch(self):
        m = {"seconds": 12, "sound": [-20.0] * 12, "keyframes": [[0.0, 0.0], [6.0, 0.4]]}
        stretches, _ = maps.rank(m, {"sound": 1, "screen": 1}, 30, 6)
        self.assertEqual([(s["start"], s["end"]) for s in stretches], [(0, 12)])


if __name__ == "__main__":
    unittest.main()
