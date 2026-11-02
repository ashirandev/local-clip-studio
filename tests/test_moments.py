"""Looking at pieces and proposing moments, on small videos made by ffmpeg: python -m unittest discover -s tests"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

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


class Quiet(unittest.TestCase):
    """Tarm 10-30: "ตัวตัดต่อ อยากให้ตัดเฉพาะตอนเสียงเงียบ แบบ wave เสียงมันเบาแล้ว ไม่เอาตัดกลางคำ"."""
    rec = {"path": "x.mp4", "seconds": 200.0, "sound_tracks": 1}
    level = (0.0, [-20.0] * 300 + [-60.0] * 100)

    def land(self, pieces, runs, said=None, far=None):
        """_on_quiet over pieces, runs the quiet stretch takes.quiet_near finds for each time, far the pause it finds
        further on, which is looked for when far is given."""
        from unittest import mock
        self.asked = []

        def near(rec, t, reach, under, bed=None, side=0, pause_only=False):
            self.asked.append((t, reach, under, bed, side))
            return (far if pause_only else runs).get(t)

        with mock.patch.object(tools.takes, "quiet_near", near):
            return tools._on_quiet(self.rec, (pieces, said), self.level, far is not None)

    def test_each_cut_moves_into_the_nearest_quiet_on_the_side_it_keeps(self):
        pieces, said = self.land([[10.0, 20.0]], {10.0: [9.4, 9.8], 20.0: [20.2, 21.0]})
        self.assertEqual(pieces, [[9.5, 20.5]], "an opening stays near the sound after it, a closing the sound before")
        self.assertEqual(said, "The program moved 2 cuts, at most 0.5 s, into the nearest quiet sound, so none falls "
                               "inside a word.")
        self.assertEqual(self.asked, [(10.0, tools.EDGE_REACH, -30.0, tools.EDGE_BED, -1),
                                      (20.0, tools.EDGE_REACH, -30.0, tools.EDGE_BED, 1)],
                         "quiet is 10 dB under the talk, over the stretch's loud quarter; a bed is tried next; a start "
                         "looks only earlier and an end only later, so no cut moves into what the clip keeps")

    def test_a_short_stretch_is_landed_in_whole(self):
        self.assertEqual(self.land([[10.0, 20.0]], {10.0: [9.9, 9.96], 20.0: [20.1, 20.16]})[0], [[9.9, 20.16]])

    def test_a_cut_already_in_the_quiet_stays_and_nothing_is_said(self):
        self.assertEqual(self.land([[10.0, 20.0]], {10.0: [9.0, 10.5], 20.0: [19.0, 20.0]}), ([[10.0, 20.0]], None))

    def test_with_nothing_quiet_near_the_cut_stays_and_it_is_said(self):
        pieces, said = self.land([[10.0, 20.0]], {10.0: [9.4, 9.8]}, "The program cut 2 s.")
        self.assertEqual(pieces, [[9.5, 20.0]])
        self.assertEqual(said, "The program cut 2 s. The program moved 1 cut, at most 0.5 s, into the nearest quiet "
                               "sound, so none falls inside a word. The cut at 0:20.0 lands on loud sound: nothing is "
                               "quiet within 1 s.")

    def test_a_last_end_on_loud_sound_looks_further_for_a_real_pause(self):
        """EP 22: a line ended at 26:19.30 with its last word in its text, the sound at talk level to 26:22.36."""
        pieces, said = self.land([[10.0, 20.0]], {10.0: [9.4, 9.8]}, far={20.0: [22.8, 23.3]})
        self.assertEqual(pieces, [[9.5, 23.1]])
        self.assertEqual(self.asked[-1], (20.0, tools.LAST_REACH, -30.0, None, 1), "a pause outward, no dip of a bed")
        self.assertEqual(said, "The program moved 2 cuts, at most 3.1 s, into the nearest quiet sound, so none falls "
                               "inside a word.")

    def test_only_the_last_end_looks_further_and_only_when_nothing_is_near(self):
        pieces, said = self.land([[10.0, 20.0], [30.0, 40.0]], {10.0: [9.9, 10.1], 30.0: [29.9, 30.1], 40.0: [40.2, 40.5]},
                                 far={20.0: [22.0, 23.0], 40.0: [42.0, 43.0]})
        self.assertEqual(pieces, [[10.0, 20.0], [30.0, 40.5]], "the piece before the last stays")
        self.assertIn("The cut at 0:20.0 lands on loud sound", said)
        self.assertEqual(self.land([[10.0, 20.0]], {10.0: [9.9, 10.1], 20.0: [20.2, 20.5]}, far={20.0: [22.0, 23.0]})[0],
                         [[10.0, 20.5]])

    def test_a_last_end_with_no_pause_stays_and_is_said(self):
        pieces, said = self.land([[10.0, 20.0]], {10.0: [9.9, 10.1]}, far={})
        self.assertEqual(pieces, [[10.0, 20.0]])
        self.assertIn("The cut at 0:20.0 lands on loud sound", said)

    def test_two_cuts_at_one_spot_are_said_once(self):
        said = self.land([[10.0, 20.01], [20.03, 30.0]], {10.0: [9.9, 10.1], 30.0: [29.9, 30.1]}, far={})[1]
        self.assertEqual(said.count("The cut at 0:20.0 lands on loud sound"), 1)

    def test_a_pause_that_would_take_a_clip_past_its_longest_is_left(self):
        end = 10.0 + tools.LONGEST - 1.0
        pieces, said = self.land([[10.0, end]], {10.0: [9.4, 9.8]}, far={end: [end + 2.0, end + 3.0]})
        self.assertEqual(pieces, [[9.5, end]], "the clip keeps its other cut")
        self.assertIn("lands on loud sound", said)

    def test_the_picture_or_the_person_may_end_a_clip_and_then_it_stays(self):
        self.assertTrue(tools._reaches_on("vlog", {}))
        self.assertTrue(tools._reaches_on(None, {}))
        self.assertFalse(tools._reaches_on("game", {}), "a game's end is its death")
        self.assertFalse(tools._reaches_on("sport", {}), "a goal's end is its celebration, the replay comes next")
        self.assertFalse(tools._reaches_on("vlog", {"ends": True}), "the person put it there")

    def test_the_first_and_last_moments_of_the_recording_are_not_cuts(self):
        self.assertEqual(self.land([[0.0, 200.0]], {}), ([[0.0, 200.0]], None))
        self.assertEqual(self.asked, [])

    def test_pieces_that_meet_in_the_quiet_become_one(self):
        pieces, _ = self.land([[10.0, 20.0], [20.3, 30.0]], {20.0: [20.1, 20.5], 20.3: [20.1, 20.5]})
        self.assertEqual(pieces, [[10.0, 30.0]])

    def test_a_clip_at_its_longest_keeps_its_cuts_rather_than_grow_past_it(self):
        pieces = [[10.0, 10.0 + tools.LONGEST]]
        self.assertEqual(self.land(pieces, {10.0: [9.0, 9.5], pieces[0][1]: [100.5, 101.0]}), (pieces, None))

    def test_no_level_leaves_the_pieces_as_they_are(self):
        made = [[10.0, 20.0]], "said"
        self.assertEqual(tools._on_quiet(self.rec, made, None), made)
        self.assertEqual(tools._on_quiet(self.rec, made, (0.0, [])), made)
        self.assertEqual(tools._on_quiet(self.rec, ([], None), self.level), ([], None))

    def test_the_check_names_the_cuts_left_on_loud_sound(self):
        from unittest import mock
        state = {"clip": {"pieces": [[10.0, 20.0]]}}

        def near(rec, t, reach, under, bed=None):
            return [t - 0.1, t + 0.1] if t == 10.0 else [t + 0.3, t + 0.5]

        with mock.patch.object(tools.takes, "heard_level", return_value=self.level):
            with mock.patch.object(tools.takes, "quiet_near", near):
                self.assertEqual(tools._cuts_check(self.rec, state, "talk"),
                                 (None, "on sound as loud as the talk at 0:20.0 of the recording, with nothing quiet "
                                        "near: tell the person", "Muse"))
            with mock.patch.object(tools.takes, "quiet_near", lambda rec, t, reach, under, bed=None: [t - 1, t + 1]):
                self.assertEqual(tools._cuts_check(self.rec, state, "talk"),
                                 (True, "every cut is in quiet sound", "program"))
        with mock.patch.object(tools.takes, "heard_level", return_value=None):
            self.assertEqual(tools._cuts_check(self.rec, state, "talk")[:2], (None, "the sound could not be read"))
        self.assertEqual(tools._cuts_check(self.rec, state, tools.MONTAGE)[:2],
                         (None, "a montage cuts on the picture, under its music"))


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
        self.addCleanup(setattr, tools, "EDGE_RECIPES", tools.EDGE_RECIPES)
        tools.EDGE_RECIPES = ()
        self.addCleanup(setattr, tools, "SHORT_RECIPES", tools.SHORT_RECIPES)
        tools.SHORT_RECIPES = ()
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

    def test_a_moment_on_a_long_line_is_refused_once_with_its_pieces_and_then_goes_through(self):
        self.worker.phrases = [{"from": 1.0, "to": 12.0, "text": "a long line of sound"}]
        moment = {"look": 1, "reason": "the tone", "start": "0:31", "end": "0:50"}
        with mock.patch.object(tools.takes, "dips", lambda rec, a, b: [a + 5.0]):
            brain, steps = self.run_job(
                self.opened(), [call("load_recipe", type="explain", why="its kind")],
                [call("look_at_piece", start="0:30", end="0:50", why="look")],
                [call("propose_moments", moments=[moment], why="first")],
                [call("propose_moments", moments=[dict(moment, start="0:36")], why="from the pieces")],
                '{"lines": ["a long line of sound"]}', "Done.")
        said = [s for s in steps if s["tool"] == "propose_moments"]
        self.assertEqual([s["refused"] for s in said], [True, False])
        self.assertIn("Nothing was proposed yet. A heard line is a stretch of sound, not a sentence", said[0]["result"])
        self.assertIn("The moment's start at 0:31.00:\n  0:31.00-0:36.00 a long line of sound\n  0:36.00-0:42.00 a long line",
                      said[0]["result"])
        self.assertIn("Propose again: start each moment at the time the piece that opens a new thought begins, and end "
                      "it at the time the piece that finishes a thought ends", said[0]["result"])
        self.assertIn("A piece that still finishes or agrees with what came before is the close of that thought, not "
                      "a start.", said[0]["result"])
        self.assertEqual(self.chat.moments[0]["start"], 36.0)
        self.assertEqual(self.chat.moments[0]["n"], 1)

    def test_an_overlap_the_program_made_is_settled_by_the_program(self):
        """Spark 10-30 on the seller: the program moved the edges of two moments Muse kept apart past each other, Muse
        could not see why it was refused, and it sent the same moments 5 times or dropped one."""
        self.worker.phrases = [{"from": 18.0, "to": 21.0, "text": "a line"}]
        moment = {"look": 1, "reason": "the tone"}
        fixed = '{"lines": ["a line"]}'
        _, steps = self.run_job(self.opened(), [call("look_at_piece", start="0:20", end="1:00", why="look")],
                                [call("propose_moments", moments=[dict(moment, start="0:20", end="0:40"),
                                                                  dict(moment, start="0:40", end="0:55")],
                                      why="side by side")], fixed,
                                [call("propose_moments", moments=[dict(moment, start="0:40", end="0:55"),
                                                                  dict(moment, start="0:20", end="0:40")],
                                      why="the other way round")], fixed, "Done.")
        said = [s for s in steps if s["tool"] == "propose_moments"]
        self.assertEqual([s["refused"] for s in said], [False, False], said[0]["result"][:200])
        self.assertEqual([(m["start"], m["end"]) for m in self.chat.moments], [(37.75, 55.0), (20.0, 37.75)])

    def test_a_moment_the_settling_leaves_too_short_is_refused(self):
        self.worker.phrases = [{"from": 18.0, "to": 21.0, "text": "a line"}]
        moment = {"look": 1, "reason": "the tone"}
        _, steps = self.run_job(self.opened(), [call("look_at_piece", start="0:20", end="1:00", why="look")],
                                [call("propose_moments", moments=[dict(moment, start="0:20", end="0:40"),
                                                                  dict(moment, start="0:40", end="0:50")],
                                      why="side by side")], "Done.")
        self.assertIn("Moment 2 is 9 s long", steps[-1]["result"])

    def test_a_moment_may_use_what_the_look_heard_beyond_its_pictures(self):
        moment = {"look": 1, "reason": "the tone"}
        _, steps = self.run_job(self.opened(), [call("look_at_piece", start="0:30", end="0:50", why="look")],
                                [call("propose_moments", moments=[dict(moment, start="0:25", end="0:52")],
                                      why="a line from just outside")], "Done.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertEqual((self.chat.moments[0]["start"], self.chat.moments[0]["end"]), (25.0, 52.0))

    def test_a_look_with_a_stretch_not_heard_says_so_and_is_heard_again_next_time(self):
        """Lane 4 10-31: a look's hearing lost, Muse moved on, and EP 22's joke went with it 3 of 3. A hole read as
        silence would be the same mistake, and kept it would stay."""
        line = {"from": 31.0, "to": 34.0, "text": "what was heard", "sure": 1.0}
        parts = [hearing.PartHeard("Muse Voice Transcribe did not answer.", [line], [(40.0, 48.0)]), [line]]
        rec = {"name": "x.mp4", "seconds": 60.0, "sound_tracks": 1}
        job = types.SimpleNamespace(languages=lambda: ())

        def hear(rec, a, b):
            got = parts.pop(0)
            if isinstance(got, Exception):
                raise got
            return got

        job.hear = hear
        look = {"start": 30.0, "end": 50.0}
        kept = []
        with mock.patch.object(hearing, "heard", lambda *a: kept[-1] if kept else None), \
                mock.patch.object(hearing, "keep", lambda rec, a, b, said, languages: kept.append(said)):
            said = tools._what_is_said(job, rec, look)
            self.assertEqual(said[1], "0:31-0:34 what was heard")
            self.assertIn("The hearing failed from 0:40 to 0:48, so what is said there is missing here", said[2])
            self.assertEqual(look["said"], [line], "what was heard is used")
            self.assertEqual(kept, [], "a hole is not kept")
            self.assertEqual(len(tools._what_is_said(job, rec, look)), 2, "the next look hears it whole")
            self.assertEqual(kept, [[line]])

    def test_a_look_with_nothing_but_a_hole_does_not_say_no_words(self):
        rec = {"name": "x.mp4", "seconds": 60.0, "sound_tracks": 1}

        def hear(rec, a, b):
            raise hearing.PartHeard("no", [{"from": 2.0, "to": 3.0, "text": "elsewhere", "sure": 1.0}], [(30.0, 50.0)])

        job = types.SimpleNamespace(languages=lambda: (), hear=hear)
        with mock.patch.object(hearing, "heard", lambda *a: None):
            said = tools._what_is_said(job, rec, {"start": 30.0, "end": 50.0})
        self.assertEqual(len(said), 1)
        self.assertIn("The hearing failed from 0:30 to 0:50", said[0])

    def test_a_hole_outside_the_look_is_not_said_and_still_not_kept(self):
        line = {"from": 31.0, "to": 34.0, "text": "what was heard", "sure": 1.0}

        def hear(rec, a, b):
            raise hearing.PartHeard("no", [line], [(24.0, 29.0)])

        job = types.SimpleNamespace(languages=lambda: (), hear=hear)
        kept = []
        with mock.patch.object(hearing, "heard", lambda *a: None), \
                mock.patch.object(hearing, "keep", lambda *a: kept.append(a)):
            said = tools._what_is_said(job, {"name": "x.mp4", "seconds": 60.0, "sound_tracks": 1},
                                       {"start": 30.0, "end": 50.0})
        self.assertEqual(said[1:], ["0:31-0:34 what was heard"])
        self.assertEqual(kept, [])

    def test_a_clip_is_made_with_its_last_end_free_to_reach_on(self):
        self.worker.phrases = [{"from": 1.0, "to": 3.0, "text": "a short line"}]
        seen, real = [], tools._on_quiet

        def spy(rec, made, level, last_end=False):
            seen.append(last_end)
            return real(rec, made, level, last_end)

        for recipe in ("explain", "game"):
            with mock.patch.object(tools, "_on_quiet", spy):
                self.run_job(self.opened(), [call("load_recipe", type=recipe, why="its kind")],
                             [call("look_at_piece", start="0:30", end="0:50", why="look")],
                             [call("propose_moments", moments=[{"look": 1, "reason": "the tone"}], why="first")],
                             '{"lines": ["a short line"]}',
                             [call("make_clip", moment=1, layout="subject", why="make it")], "Done.")
            self.chat = agent.Chat()
        self.assertEqual(seen, [True, False])

    def test_stop_while_a_line_is_heard_again_proposes_nothing(self):
        self.worker.phrases = [{"from": 1.0, "to": 12.0, "text": "a long line of sound"}]
        moment = {"look": 1, "reason": "the tone", "start": "0:31", "end": "0:50"}

        def stop(rec, a, b, *others, **more):
            raise hearing.Stopped()

        real = hearing.hear
        with mock.patch.object(tools.takes, "dips", lambda rec, a, b: [a + 5.0]):
            def hears(rec, a, b, **more):
                return (stop if b - a < 8 else real)(rec, a, b, **more)

            with mock.patch.object(hearing, "hear", hears):
                _, steps = self.run_job(
                    self.opened(), [call("load_recipe", type="explain", why="its kind")],
                    [call("look_at_piece", start="0:30", end="0:50", why="look")],
                    [call("propose_moments", moments=[moment], why="first")], "Done.")
        self.assertEqual(steps[-1]["result"], "Not proposed: the person pressed Stop.")
        self.assertIsNone(self.chat.moments)

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


class Finer(unittest.TestCase):
    """Lane 4 10-30: "a heard line is a sound piece, not a sentence", and Muse can only choose where a line begins or
    ends. Gold EP1 3:25 was one line from 205.60, opening on นะครับ, and the new thought began 1.4 s into it."""
    rec = {"path": "x.mp4", "seconds": 300.0, "sound_tracks": 1}
    HEARD = {205.6: "นะครับ", 206.05: "การเทรดก็เช่นกัน", 206.96: "ข้อแนะนำนะครับ"}

    def setUp(self):
        self.kept, self.asked, self.cut = {}, [], [206.05, 206.96]

        def heard(rec, a, b, languages=()):
            return self.kept.get((a, b))

        def keep(rec, a, b, said, languages=()):
            self.kept[(a, b)] = said

        def hear(rec, a, b):
            self.asked.append((a, b))
            return [{"from": a, "to": b, "text": self.HEARD[a], "sure": 0.9}] if a in self.HEARD else []

        self.job = types.SimpleNamespace(chat=types.SimpleNamespace(clip={"recipe": "explain"}), hear=hear,
                                         languages=lambda: ())
        patches = [mock.patch.object(tools.hearing, "heard", heard), mock.patch.object(tools.hearing, "keep", keep),
                   mock.patch.object(tools.takes, "dips", lambda rec, a, b: list(self.cut))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def look(self, *lines):
        return {"start": 200.0, "end": 260.0, "said": [dict(line) for line in lines]}

    LINE = {"from": 205.6, "to": 209.98, "text": "นะครับ การเทรดก็เช่นกัน ข้อแนะนำนะครับ", "sure": 1.0}
    LONG = {"from": 205.6, "to": 212.0, "text": "ข้อแนะนำ เกี่ยวกับเงิน", "sure": 1.0}

    def test_a_line_opening_on_a_closing_word_is_heard_again_and_muse_sees_its_pieces(self):
        look = self.look(self.LINE, {"from": 212.0, "to": 214.0, "text": "b"})
        said = tools._finer_edges(self.job, self.rec, look, 205.6, 230.0)
        self.assertEqual(self.asked, [(205.6, 206.05), (206.05, 206.96), (206.96, 209.98)])
        self.assertEqual(said, ["The moment's start at 3:25.60:\n  3:25.60-3:26.05 นะครับ\n"
                                "  3:26.05-3:26.96 การเทรดก็เช่นกัน\n  3:26.96-3:29.98 ข้อแนะนำนะครับ"])
        self.assertEqual([(p["from"], p["to"], p["text"], p["sure"]) for p in look["said"][:3]],
                         [(205.6, 206.05, "นะครับ", 0.9), (206.05, 206.96, "การเทรดก็เช่นกัน", 0.9),
                          (206.96, 209.98, "ข้อแนะนำนะครับ", 0.9)])
        self.assertEqual(look["said"][3]["text"], "b", "the lines around it stay")

    def test_a_long_line_is_heard_again_at_either_edge_and_once_when_it_holds_both(self):
        look = self.look({"from": 201.0, "to": 203.0, "text": "a"}, self.LONG)
        self.assertEqual(tools._finer_edges(self.job, self.rec, look, 190.0, 208.0)[0][:26], "The moment's end at 3:28.0")
        self.assertEqual(len(look["said"]), 4)
        look = self.look(self.LONG)
        self.assertEqual(len(tools._finer_edges(self.job, self.rec, look, 205.6, 208.0)), 1, "one line, both edges")

    def test_a_long_call_to_buy_after_the_price_is_heard_again_on_a_sellers_clip(self):
        self.HEARD = {204.5: "กดสี่เอฟไว้ได้เลย", 207.0: "สินค้าชิ้นที่สอง"}
        self.cut = [207.0]
        price = {"from": 201.0, "to": 204.0, "text": "ราคาเก้าสิบบาท", "sure": 1.0}
        ask = {"from": 204.5, "to": 212.0, "text": "กดสี่เอฟไว้ได้เลย สินค้าชิ้นที่สอง", "sure": 1.0}
        self.job.chat.clip = {"recipe": "seller"}
        look = self.look(price, ask)
        said = tools._finer_edges(self.job, self.rec, look, 201.0, 203.9)
        self.assertEqual(said, ["The moment's end at 3:23.90:\n  3:24.50-3:27.00 กดสี่เอฟไว้ได้เลย\n"
                                "  3:27.00-3:32.00 สินค้าชิ้นที่สอง"])
        self.assertEqual(tools._snapped(look, 201.0, 203.9, price=True)[1], 207.25, "the call and not what follows")
        self.job.chat.clip = {"recipe": "explain"}
        self.assertEqual(tools._finer_edges(self.job, self.rec, self.look(price, ask), 201.0, 203.9), [], "a seller's")

    def test_a_call_to_buy_with_room_for_more_than_the_ask_is_heard_again(self):
        """Lane 5's 035133-c553: 4.2 s in the look of the controller's clip, under FINER_OVER, with the balm's opening."""
        self.HEARD = {204.5: "กดสี่เอฟไว้ได้เลย", 207.0: "สินค้าชิ้นที่สอง"}
        self.cut = [207.0]
        price = {"from": 201.0, "to": 204.0, "text": "ราคาเก้าสิบบาท", "sure": 1.0}
        ask = {"from": 204.5, "to": 208.7, "text": "กดสี่เอฟไว้ได้เลย สินค้าชิ้นที่สอง", "sure": 1.0}
        self.job.chat.clip = {"recipe": "seller"}
        look = self.look(price, ask)
        self.assertEqual(len(tools._finer_edges(self.job, self.rec, look, 201.0, 203.9)), 1)
        self.assertEqual(tools._snapped(look, 201.0, 203.9, price=True)[1], 207.25, "the call and not what follows")
        self.assertEqual(tools._finer_edges(self.job, self.rec, self.look(price, dict(ask, to=207.5)), 201.0, 203.9), [],
                         "3 s is the ask alone")

    def test_a_closing_word_counts_at_the_start_only(self):
        look = self.look({"from": 201.0, "to": 203.0, "text": "a"}, self.LINE)
        self.assertEqual(tools._finer_edges(self.job, self.rec, look, 201.0, 208.0), [])
        self.assertEqual(self.asked, [])

    def test_a_line_is_tried_once_so_asking_cannot_go_on(self):
        look = self.look(self.LINE)
        self.assertEqual(len(tools._finer_edges(self.job, self.rec, look, 205.6, 230.0)), 1)
        self.asked.clear()
        self.assertEqual(tools._finer_edges(self.job, self.rec, look, 205.6, 230.0), [])
        self.assertEqual(self.asked, [])
        self.cut = []
        look = self.look(self.LINE)
        self.assertEqual(tools._finer_edges(self.job, self.rec, look, 205.6, 230.0), [], "no dips in it")
        self.assertEqual(tools._finer_edges(self.job, self.rec, look, 205.6, 230.0), [])
        self.assertTrue(look["said"][0]["finer"], "it is not looked at again")

    def past(self, line, heard=(190.0, 212.0), seconds=300.0, cut=(208.0,)):
        self.HEARD = {205.6: "ข้อแนะนำ", 208.0: "จบประโยคนี้"}
        self.cut = list(cut)
        look = dict(self.look({"from": 201.0, "to": 203.0, "text": "a"}, line), end=206.0, heard=list(heard))
        return look, tools._finer_edges(self.job, dict(self.rec, seconds=seconds), look, 201.0, 211.0)

    def test_a_looks_last_line_stopped_by_the_edge_of_the_hearing_is_heard_on_to_finish_its_thought(self):
        """Lane 5's showcase: the look's last piece ended at 1:25.62 on a thought that ended at 1:28."""
        look, said = self.past(self.LONG)
        self.assertEqual(self.asked, [(205.6, 208.0), (208.0, 216.0)])
        self.assertEqual(said, ["The moment's end at 3:31.00:\n  3:25.60-3:28.00 ข้อแนะนำ\n  3:28.00-3:36.00 จบประโยคนี้"])
        self.assertEqual(look["heard"], [190.0, 216.0], "a moment may end out there and is not refused")
        self.assertEqual(tools._snapped(look, 201.0, 216.0)[:2], (201.0, 216.0), "an end out there stands")
        self.assertEqual(look["said"][-1]["to"], 216.0)

    def test_the_hearing_stops_at_the_end_of_the_recording(self):
        look, _ = self.past(self.LONG, seconds=213.5)
        self.assertEqual(self.asked[-1], (208.0, 213.5))
        self.assertEqual(look["heard"][1], 213.5)

    def test_a_line_that_ended_before_the_edge_or_is_not_the_last_is_not_heard_past_its_end(self):
        look, _ = self.past(self.LONG, heard=(190.0, 214.0))
        self.assertEqual(self.asked[-1], (208.0, 212.0), "the speaker stopped: nothing was cut off")
        self.assertEqual(look["heard"], [190.0, 214.0])
        self.asked.clear()
        self.kept.clear()
        look = dict(self.look(self.LONG, {"from": 212.0, "to": 213.0, "text": "b"}), end=206.0, heard=[190.0, 212.0])
        tools._finer_edges(self.job, self.rec, look, 201.0, 211.0)
        self.assertEqual(self.asked[-1], (208.0, 212.0), "another line follows it")
        self.assertEqual(look["heard"], [190.0, 212.0])

    def test_a_line_that_stays_one_piece_does_not_widen_what_was_heard(self):
        look, said = self.past(self.LONG, cut=())
        self.assertEqual(said, [])
        self.assertEqual(look["heard"], [190.0, 212.0])

    def test_a_line_that_stays_one_piece_stays_as_it_was_heard(self):
        self.HEARD = {205.6: "นะครับ"}
        look = self.look(self.LINE)
        self.assertEqual(tools._finer_edges(self.job, self.rec, look, 205.6, 230.0), [])
        self.assertEqual(look["said"][0]["text"], self.LINE["text"])

    def test_clips_of_other_kinds_and_short_plain_lines_stay(self):
        self.job.chat.clip = {"recipe": "game"}
        self.assertEqual(tools._finer_edges(self.job, self.rec, self.look(self.LONG), 205.6, 230.0), [])
        self.job.chat.clip = None
        self.assertEqual(tools._finer_edges(self.job, self.rec, self.look(self.LONG), 205.6, 230.0), [])
        self.job.chat.clip = {"recipe": "explain"}
        plain = {"from": 205.6, "to": 209.0, "text": "ข้อแนะนำนะครับ", "sure": 1.0}
        self.assertEqual(tools._finer_edges(self.job, self.rec, self.look(plain), 205.6, 230.0), [])
        self.assertEqual(self.asked, [])

    def test_a_piece_heard_before_is_not_paid_for_again(self):
        self.kept[(205.6, 206.05)] = [{"from": 205.6, "to": 206.05, "text": "นะครับ"}]
        tools._finer_edges(self.job, self.rec, self.look(self.LINE), 205.6, 230.0)
        self.assertEqual(self.asked, [(206.05, 206.96), (206.96, 209.98)])
        self.assertEqual(sorted(self.kept), [(205.6, 206.05), (206.05, 206.96), (206.96, 209.98)])

    def test_a_hearing_that_fails_leaves_the_line_and_stop_stops(self):
        def fails(rec, a, b):
            raise tools.hearing.HearingError("no answer")

        self.job.hear = fails
        look = self.look(self.LINE)
        self.assertEqual(tools._finer_edges(self.job, self.rec, look, 205.6, 230.0), [])
        self.assertEqual(look["said"][0]["text"], self.LINE["text"])

        def stopped(rec, a, b):
            raise tools.hearing.Stopped()

        self.job.hear = stopped
        with self.assertRaises(tools.hearing.Stopped):
            tools._finer_edges(self.job, self.rec, self.look(self.LINE), 205.6, 230.0)


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
        self.assertEqual(b, 44.55)
        self.assertIn("ends 5.5 s later, so the call to buy said right after the price is in the clip", said)
        alone = dict(look, said=look["said"][:2])
        self.assertEqual(tools._snapped(alone, 9.0, 39.0, price=True)[1], 42.15)
        self.assertIn("ends 3.1 s later, so the price said right after it is in the clip",
                      tools._snapped(alone, 9.0, 39.0, price=True)[2])
        self.assertEqual(tools._snapped(look, 9.0, 39.0)[1], 39.0, "not a seller: the end stays")
        self.assertEqual(tools._snapped(dict(look, heard=[3.0, 41.0]), 9.0, 39.0, price=True)[1], 39.0,
                         "a price not heard whole is not reached for")
        self.assertEqual(tools._snapped(look, 9.0, 32.0, price=True)[1], 32.0, "8.9 s on is past PRICE_REACH")
        self.assertEqual(tools._snapped(look, 9.0, 34.0, price=True)[1], 44.55, "inside the line before the price, "
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

    CLOSING = {"start": 200.0, "end": 260.0, "said": [{"from": 205.76, "to": 205.94, "text": "นะครับ"},
                                                      {"from": 206.12, "to": 206.83, "text": "การเทสก็เช่นกัน"},
                                                      {"from": 207.01, "to": 210.44, "text": "ข้อแนะนำนะครับ"}]}

    def test_a_clip_does_not_open_on_a_word_that_only_closes_what_came_before(self):
        a, b, said = tools._snapped(self.CLOSING, 205.5, 240.0)
        self.assertEqual((a, b), (205.94, 240.0), "after the word, and it does not reach back into it")
        self.assertIn('starts 0.4 s later, so it does not open on "นะครับ", which closes what was said before it', said)
        self.assertEqual(tools._snapped(self.CLOSING, 205.8, 240.0)[0], 205.94, "from inside the word too")
        wide = dict(self.CLOSING, said=[self.CLOSING["said"][0], dict(self.CLOSING["said"][1], **{"from": 208.0, "to": 209.0})])
        self.assertEqual(tools._snapped(wide, 205.5, 240.0)[0], 207.75, "air before the next line, as any start has")

    def test_a_closing_word_stays_when_nothing_follows_or_the_clip_would_be_too_short(self):
        alone = dict(self.CLOSING, said=self.CLOSING["said"][:1])
        self.assertEqual(tools._snapped(alone, 205.5, 240.0)[:2], (205.5, 240.0))
        self.assertEqual(tools._snapped(self.CLOSING, 205.5, 215.0)[0], 205.5, "10 s is the least a clip can be")
        more = dict(self.CLOSING, said=self.CLOSING["said"] + [{"from": 211.0, "to": 212.0, "text": "b"}])
        self.assertEqual(tools._snapped(more, 207.0, 240.0)[0], 207.0, "a line that only ends on it is its own")

    ASK = {"start": 0.0, "end": 60.0, "said": [{"from": 12.0, "to": 39.5, "text": "จอยตัวนี้ดีมาก"},
                                               {"from": 39.71, "to": 41.71, "text": "สนนราคาอยู่ที่สามพันบาท"},
                                               {"from": 42.15, "to": 44.03, "text": "ใครอยากได้กดสี่เอฟไว้ได้เลย"},
                                               {"from": 44.65, "to": 46.34, "text": "สินค้าชิ้นที่สองครับ"}]}

    def test_a_sellers_call_to_buy_right_after_the_price_is_in_that_clip(self):
        """The balm clip opened on the PS5's "กดสี่เอฟไว้ได้เลย": the PS5's clip stopped at the price, and the line
        after it was left for the next clip to open on."""
        a, b, said = tools._snapped(self.ASK, 12.0, 41.96, price=True)
        self.assertEqual((a, b), (12.0, 44.28), "from a clip that ends right after the price")
        self.assertEqual(said, "ends 2.3 s later, so the call to buy said right after the price is in the clip")
        self.assertEqual(tools._snapped(self.ASK, 12.0, 40.0, price=True)[1], 44.28, "from a clip that ends in the price")
        self.assertEqual(tools._snapped(self.ASK, 12.0, 38.0, price=True)[1], 44.28, "from one the price follows")

    def test_a_call_to_buy_counts_only_for_a_seller_right_after_a_price(self):
        self.assertEqual(tools._snapped(self.ASK, 12.0, 41.96)[1], 41.96, "not a seller")
        plain = dict(self.ASK, said=[dict(self.ASK["said"][1], text="จอยตัวนี้ดี")] + self.ASK["said"][2:])
        self.assertEqual(tools._snapped(plain, 20.0, 41.96, price=True)[1], 41.96, "no price before it")
        far = dict(self.ASK, said=self.ASK["said"][:2] + [dict(self.ASK["said"][2], **{"from": 43.5, "to": 44.5})])
        self.assertEqual(tools._snapped(far, 12.0, 41.96, price=True)[1], 41.96, "too long after the price")
        other = dict(self.ASK, said=self.ASK["said"][:2] + [dict(self.ASK["said"][2], text="ต่อไปครับ")])
        self.assertEqual(tools._snapped(other, 12.0, 41.96, price=True)[1], 41.96, "not a call to buy")
        cut = dict(self.ASK, heard=[0.0, 43.0])
        self.assertEqual(tools._snapped(cut, 12.0, 41.96, price=True)[1], 41.96, "past what was heard")

    def test_the_letters_of_a_call_to_buy_inside_another_word_are_not_one(self):
        """The balm's clip ran on through "โทรศัพท์ Redmi", DM in it, and took away the clip of the phone."""
        for line, taken in (("สินค้าชิ้นที่สามคือโทรศัพท์ Redmi นะครับ", False), ("กด CF ไว้ได้เลย", True),
                            ("ทักDMมาได้เลย", True), ("dm มาเลย", True)):
            ask = dict(self.ASK, said=self.ASK["said"][:2] + [dict(self.ASK["said"][2], text=line)])
            self.assertEqual(tools._snapped(ask, 12.0, 41.96, price=True)[1] != 41.96, taken, line)

    def test_a_clip_with_no_room_keeps_to_the_longest_a_clip_can_be(self):
        with mock.patch.object(tools, "LONGEST", 30):
            self.assertEqual(tools._snapped(self.ASK, 12.0, 41.96, price=True)[1], 41.96)

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

    def test_a_long_line_over_the_clips_edge_keeps_the_words_said_inside_it(self):
        """EP22 11-01: a 24 s line began 14.7 s before the clip, and the clip had no captions for its first 10 s."""
        line = {"from": 10.0, "to": 20.0, "text": "one two three four five six seven eight nine zero", "sure": 0.9}
        pieces = [[15.0, 40.0]]
        got = tools._heard_in_piece(pieces, line)
        self.assertEqual(got["text"], "six seven eight nine zero")
        self.assertGreaterEqual(got["from"], 15.0)
        self.assertEqual((got["to"], got["sure"]), (20.0, 0.9))
        self.assertEqual(line["text"].split()[0], "one", "what was heard stays whole for the rest of the studio")
        self.assertEqual(tools._in_clip(pieces, got, 25.0), (round(got["from"] - 15, 2), 5.0))
        got = tools._heard_in_piece([[0.0, 15.0]], line)
        self.assertEqual(got["text"], "one two three four five")
        self.assertLessEqual(got["to"], 15.0)
        got = tools._heard_in_piece([[12.0, 14.0], [30.0, 40.0]], line)
        self.assertEqual(got["text"], "three four")
        got = tools._heard_in_piece([[11.4, 13.0]], line)
        self.assertEqual(got["text"], "three")
        self.assertGreater(got["from"], 11.4, "it starts with its own first word, not with the piece")

    def test_a_subtitle_loses_the_number_of_its_place_in_the_list(self):
        """EP0 11-02, Spark Open: "1. With such huge age gaps" on the clip."""
        self.assertEqual(tools._without_own_number("1. With such huge age gaps", 1), "With such huge age gaps")
        self.assertEqual(tools._without_own_number("2) So how", 2), "So how")
        self.assertEqual(tools._without_own_number("3. Three things", 1), "3. Three things", "a number that is its own stays")
        self.assertEqual(tools._without_own_number("1,000 people", 1), "1,000 people")

    def test_a_line_only_just_over_the_edge_keeps_its_words(self):
        line = {"from": 14.5, "to": 18.0, "text": "said at the very start"}
        self.assertEqual(tools._heard_in_piece([[15.0, 40.0]], line), dict(line, **{"from": 15.0}))
        line = {"from": 10.0, "to": 15.5, "text": "said at the very end"}
        self.assertIs(tools._heard_in_piece([[8.0, 15.0]], line), line, "and its fix reaches what was heard")

    def test_a_line_outside_every_piece_is_left_out(self):
        self.assertIsNone(tools._heard_in_piece([[30.0, 35.0]], {"from": 36.0, "to": 38.0, "text": "cut out"}))
        self.assertIsNone(tools._heard_in_piece([[30.0, 30.2]], {"from": 20.0, "to": 40.0, "text": "one long word"}),
                          "a word's middle is not inside it")


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

    def test_a_laugh_as_loud_as_the_talk_is_kept_and_a_quiet_silence_and_a_filler_are_cut(self):
        """Lane 5, 10-30: Muse Voice Transcribe writes no words for laughter, and a talk clip of a story lost the 11 s
        of laughing after its last line as "the silence at its edges"."""
        said = [said_line(31.0, 33.0, "one two three"), said_line(33.5, 34.5, "เอ่อ"),
                said_line(36.5, 38.0, "four five")]
        db = [-60.0] * 200
        for a, b, v in ((31.0, 33.0, -25.0), (33.5, 34.5, -26.0), (36.5, 38.0, -27.0), (38.4, 46.0, -28.0),
                        (41.0, 41.1, -45.0)):
            db[round((a - 30) * 10):round((b - 30) * 10)] = [v] * round((b - a) * 10)
        quiet = tools._pieces(said, 30.0, 50.0, lambda x, y: True)[0]
        self.assertEqual(quiet, [[30.7, 33.3], [36.2, 38.3]], "without the sound, the laugh is a silence")
        pieces, cut = tools._pieces(said, 30.0, 50.0, lambda x, y: True, level=(30.0, db))
        self.assertEqual(pieces, [[30.7, 33.3], [36.2, 46.3]], "the laugh stays with 0.3 s after it, a dip in it too; "
                                                               "the filler and the quiet after the laugh go")
        self.assertIn("1 silence longer than 1 s", cut)
        self.assertIn("It kept 8 s where nobody speaks but the sound is as loud as the talk, a laugh", cut)
        db[round(8.4 * 10):160] = [-38.0] * 76
        self.assertEqual(tools._pieces(said, 30.0, 50.0, lambda x, y: True, level=(30.0, db))[0], quiet,
                         "13 dB under the talk is quiet, as a room is")
        self.assertEqual(tools._pieces(said, 30.0, 50.0, lambda x, y: False, level=(30.0, db))[0], [[30.0, 50.0]])


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
