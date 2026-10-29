"""Many short clips made into one, on small videos made by ffmpeg: python -m unittest discover -s tests"""
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, hearing, montage, recording, takes, tools  # noqa: E402
from test_agent_loop import ScriptedMind, call  # noqa: E402
from test_hearing import Worker  # noqa: E402
from test_takes import ASIDE, GAVE_UP  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def make_clip(path, size=(360, 640), seconds=2.0, sound=True, shot=None, frozen_from=None):
    """A moving test picture, a tone when sound, the camera's clock when shot, and the picture held from
    frozen_from to its end."""
    inputs = ["-f", "lavfi", "-i", "testsrc2=s=%dx%d:r=30:d=%g" % (size + (seconds,))]
    if sound:
        inputs += ["-f", "lavfi", "-i", "sine=f=330:d=%g" % seconds]
    hold = ["-vf", "trim=end=%g,tpad=stop_mode=clone:stop_duration=%g" % (frozen_from, seconds - frozen_from)] \
        if frozen_from is not None else []
    tags = ["-movflags", "use_metadata_tags", "-metadata", "com.apple.quicktime.creationdate=" + shot] if shot else []
    subprocess.run([FFMPEG, "-v", "error", "-y"] + inputs + hold + tags +
                   ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"] +
                   (["-c:a", "aac", "-shortest"] if sound else []) + [path], check=True)


def tags_of(**times):
    return lambda path: times[os.path.basename(path)]


def rec(name, w=1080, h=1920, seconds=3.0, sound=1):
    return {"path": "/clips/" + name, "name": name, "size": 1, "mtime": 0, "seconds": seconds, "width": w,
            "height": h, "fps": 30, "sound_tracks": sound, "codec": "h264"}


class Shot(unittest.TestCase):
    def test_the_cameras_clock_wins_over_the_files(self):
        t, how = montage.shot({"creation_time": "2026-10-27T14:07:02.000000Z",
                               montage.CAMERA_TAG: "2026-03-10T14:38:34+0900"})
        self.assertEqual((t.isoformat(), how), ("2026-03-10T14:38:34+09:00", "camera"))
        t, how = montage.shot({"creation_time": "2023-10-28T10:34:51.000000Z"})
        self.assertEqual((t.hour, how), (10, "file"))
        self.assertEqual(montage.shot({"creation_time": "2023-10-28 10:34:51"}), (None, None),
                         "a time with no zone cannot be put in order with the others")
        self.assertEqual(montage.shot({}), (None, None))

    def test_clips_are_numbered_in_the_order_they_were_shot(self):
        found = montage.gather([rec("b.mov"), rec("a.mov"), rec("c.mov")], tags_of(**{
            "a.mov": {montage.CAMERA_TAG: "2026-03-10T15:00:00+0900"},
            "b.mov": {montage.CAMERA_TAG: "2026-03-10T14:00:00+0900"},
            "c.mov": {"creation_time": "2026-03-10T05:30:00Z"}}))
        self.assertEqual([(c["n"], c["rec"]["name"], c["how"]) for c in found],
                         [(1, "b.mov", "camera"), (2, "c.mov", "file"), (3, "a.mov", "camera")])
        self.assertEqual(montage.when(found[0]), "shot 2026-03-10 14:00:00")

    def test_files_saved_together_carry_no_time(self):
        found = montage.gather([rec("x.mov"), rec("y.mov"), rec("z.mov")], tags_of(**{
            "x.mov": {"creation_time": "2026-10-27T14:07:02Z"}, "y.mov": {"creation_time": "2026-10-27T14:07:01Z"},
            "z.mov": {montage.CAMERA_TAG: "2026-03-10T11:56:53+09:00"}}))
        self.assertEqual([(c["rec"]["name"], c["how"], c["shot"]) for c in found],
                         [("z.mov", "camera", found[0]["shot"]), ("x.mov", "saved", None), ("y.mov", "saved", None)])
        self.assertIn("saved", montage.when(found[1]))


class Shape(unittest.TestCase):
    def test_a_picture_fills_the_frame_unless_cropping_loses_too_much(self):
        self.assertTrue(montage.fills(1080, 1920))
        self.assertTrue(montage.fills(1440, 1920), "3:4 keeps three quarters of its width")
        self.assertFalse(montage.fills(1920, 1080))
        self.assertFalse(montage.fills(1080, 1080))

    def test_a_lying_picture_is_cropped_upright_around_its_centre_and_never_past_its_edge(self):
        lying = rec("l.mov", 1920, 1080)
        self.assertEqual(montage.upright(lying, 0.5), "scale=3414:1920,crop=1080:1920:1167:0")
        self.assertEqual(montage.upright(lying, 0.0), "scale=3414:1920,crop=1080:1920:0:0")
        self.assertEqual(montage.upright(lying, 1.0), "scale=3414:1920,crop=1080:1920:2334:0")

    def test_the_crop_follows_the_focus_given_then_the_face_found(self):
        lying = rec("l.mov", 1920, 1080)
        self.assertAlmostEqual(tools._upright_centre(lying, 0, 2, [600, 0, 800, 1000]), 0.7)
        face = {"size": [1920, 1080], "every": 1, "samples": [{"t": t, "faces": [[100, 200, 200, 200, 0.9]]}
                                                              for t in (0, 1)]}
        with mock.patch.object(tools.faces, "sample", return_value=face):
            self.assertAlmostEqual(tools._upright_centre(lying, 0, 2, None), 200 / 1920.0)
        with mock.patch.object(tools.faces, "sample", return_value=dict(face, samples=[])):
            self.assertIsNone(tools._upright_centre(lying, 0, 2, None), "no face: the caller says so")
        with mock.patch.object(tools.faces, "sample", side_effect=tools.faces.NoFaces("no model")):
            self.assertIsNone(tools._upright_centre(lying, 0, 2, None))

    def test_pieces_start_where_the_one_before_ends(self):
        self.assertEqual(montage.starts([{"from": 1, "to": 3.5}, {"from": 0, "to": 2}, {"from": 4, "to": 5}]),
                         [0.0, 2.5, 4.5])


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Join(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pieces_join_upright_with_sound_all_through(self):
        up, lying = os.path.join(self.tmp, "up.mp4"), os.path.join(self.tmp, "lying.mp4")
        make_clip(up, (360, 640), 3.0)
        make_clip(lying, (640, 360), 3.0, sound=False)
        pieces = [{"rec": recording.probe(up), "from": 0.5, "to": 2.0},
                  {"rec": recording.probe(lying), "from": 1.0, "to": 3.0}]
        out = montage.join(pieces, montage.target(pieces, "trip"))
        got = recording.probe(out)
        self.assertEqual((got["width"], got["height"], got["sound_tracks"]), (1080, 1920, 1))
        self.assertAlmostEqual(got["seconds"], 3.5, delta=0.1)
        self.assertFalse(os.path.exists(out + ".part"))
        self.assertIn("gblur", montage.graph(pieces[1:])[1], "the lying clip is shown whole over a blurred copy")
        self.assertNotIn("gblur", montage.graph(pieces[:1])[1], "the upright clip fills the frame")

    def test_a_lying_piece_with_a_centre_fills_the_frame_with_the_part_around_it(self):
        lying = os.path.join(self.tmp, "halves.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x360:d=2", "-f", "lavfi",
                        "-i", "color=c=blue:s=320x360:d=2", "-filter_complex", "[0:v][1:v]hstack", "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", lying], check=True)
        seen = {}
        for centre in (0.1, 0.9, None):
            pieces = [{"rec": recording.probe(lying), "from": 0.0, "to": 1.5, "centre": centre}]
            out = montage.join(pieces, montage.target(pieces, "halves"))
            got = recording.probe(out)
            self.assertEqual((got["width"], got["height"]), (1080, 1920))
            raw = subprocess.run([FFMPEG, "-v", "error", "-ss", "0.5", "-i", out, "-frames:v", "1", "-vf",
                                  "crop=2:2:539:200", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                 capture_output=True, check=True).stdout
            seen[centre] = (raw[0], raw[2], out)
        self.assertGreater(seen[0.1][0], 150, "around 0.1 the frame is the red half")
        self.assertGreater(seen[0.9][1], 150, "around 0.9 the frame is the blue half")
        self.assertLess(max(seen[None][:2]), 120, "with no centre the top of the frame is the dim blurred copy")
        self.assertEqual(len({v[2] for v in seen.values()}), 3, "each crop is joined and kept apart")

    def test_a_held_picture_is_found(self):
        held, moving = os.path.join(self.tmp, "held.mp4"), os.path.join(self.tmp, "moving.mp4")
        make_clip(held, seconds=4.0, frozen_from=1.5)
        make_clip(moving, seconds=4.0)
        found = montage.stills(recording.probe(held))
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0][0], 1.5, delta=0.15)
        self.assertAlmostEqual(found[0][1], 4.0, delta=0.15)
        self.assertEqual(montage.stills(recording.probe(moving)), [])


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Arrange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.made = tempfile.mkdtemp()
        make_clip(os.path.join(cls.made, "a_late.mov"), seconds=3.0, shot="2026-03-10T15:00:00+0900")
        make_clip(os.path.join(cls.made, "b_early.mov"), seconds=4.0, shot="2026-03-10T14:00:00+0900")
        make_clip(os.path.join(cls.made, "c_held.mov"), seconds=4.0, shot="2026-03-10T16:00:00+0900",
                  frozen_from=2.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.made, ignore_errors=True)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.videos = os.path.join(self.tmp, "videos")
        self.folder = os.path.join(self.videos, "trip")
        shutil.copytree(self.made, self.folder)
        os.environ["LCS_VIDEOS"] = self.videos
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        self.worker, self.url = Worker(), hearing.URL
        hearing.URL = self.worker.url
        self.shortest, tools.SHORTEST = tools.SHORTEST, 3
        self.chat = agent.Chat()

    def tearDown(self):
        tools.SHORTEST = self.shortest
        hearing.URL = self.url
        self.worker.close()
        agent.CHATS.clear()
        agent._shown.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_job(self, *script):
        self.chat.jobs += 1
        job = self.chat.job = agent.Job(self.chat, ScriptedMind(*script), "make one clip of the trip")
        job.run()
        return [e for e in self.chat.events if e.get("job") == job.n and e["kind"] == "step"]

    def opened(self):
        return [[call("open_clips", folder=self.folder, why="the trip's clips")],
                [call("load_recipe", type="montage", why="clips of a trip")]]

    def test_the_clips_are_shown_in_the_order_they_were_shot_with_what_holds_still(self):
        steps = self.run_job(*self.opened(), "Done.")
        said = steps[0]["result"]
        self.assertIn("1. b_early.mov · shot 2026-03-10 14:00:00 · 4 s · upright 360x640", said)
        self.assertIn("2. a_late.mov", said)
        self.assertIn("3. c_held.mov", said)
        self.assertRegex(said, r"does not move at (1\.9\d|2(\.\d+)?)-4 s: a frozen screen")
        self.assertEqual(len(self.chat.clips["clips"]), 3)
        self.assertEqual([c["n"] for c in agent.Chat.load(self.chat.id).clips["clips"]], [1, 2, 3])

    def test_with_clips_open_any_recipe_asked_is_the_montage_until_another_recording_opens(self):
        steps = self.run_job([call("open_clips", folder=self.folder, why="the review's takes")],
                             [call("load_recipe", type="seller", length_s=60, why="a review")],
                             [call("open_recording", path=os.path.join(self.folder, "a_late.mov"), why="one take")],
                             [call("load_recipe", type="seller", length_s=60, why="a review of one take")], "Done.")
        self.assertIn("so this is the montage recipe, not the seller one you asked for", steps[1]["result"])
        self.assertIn("# Trip or event clips", steps[1]["result"])
        self.assertNotIn("not the seller one", steps[3]["result"])
        self.assertIn("Recipe loaded: Trip or event clips (montage).", steps[1]["result"])
        self.assertIn("about 60 s", steps[1]["result"], "Muse's guess is kept")
        self.assertEqual((self.chat.clip["recipe"], self.chat.clip["length_s"]), ("seller", 60))

    def test_the_montage_recipe_needs_clips_open(self):
        steps = self.run_job([call("load_recipe", type="montage", why="a trip")], "Done.")
        self.assertIn("for several clips opened together", steps[0]["result"])
        self.assertIsNone(self.chat.clip)

    def test_every_clip_is_used_or_left_out_and_no_piece_holds_still(self):
        piece = {"clip": 1, "words": "At the station", "why": "where we are"}
        steps = self.run_job(*self.opened(), [call("arrange_clips", story="the trip", clips=[piece], why="order")],
                             [call("arrange_clips", story="the trip", clips=[piece, {"clip": 1, "why": "again"}],
                                   left_out=[{"clip": 2, "why": "x"}, {"clip": 3, "why": "x"}], why="order")],
                             [call("arrange_clips", story="the trip", clips=[piece, {"clip": 3, "why": "the end"}],
                                   left_out=[{"clip": 2, "why": "the same as clip 1"},
                                             {"clip": 2, "why": "the same as clip 1"}], why="order")],
                             "Done.")
        self.assertIn("Clips 2, 3 are neither used nor left out", steps[2]["result"])
        self.assertIn("Clip 1 named twice", steps[3]["result"])
        self.assertTrue(steps[2]["refused"] and steps[3]["refused"], "nothing is joined while anything is wrong")
        self.assertFalse(steps[4]["refused"], steps[4]["result"])
        self.assertRegex(steps[4]["result"], r"The program fitted the pieces: clip 3: 0-[12](\.\d+)? s of the 0-4 s you "
                                             r"gave \(its picture holds still at ")
        self.assertRegex(steps[4]["result"], r"4 s: clip 3, 0-[12](\.\d+)? s of it", "the frozen end is left out")

    def test_pieces_are_short_the_clip_near_its_length_and_the_words_tell_it(self):
        had = tools.LONGEST_PIECE, tools.OVER_LENGTH, tools.LINE_EVERY
        tools.LONGEST_PIECE = 2.5
        try:
            ok = [{"clip": 2, "start": 0.5, "end": 2.5, "words": "We arrive", "why": "where we are"},
                  {"clip": 1, "start": 1.0, "end": 2.0, "why": "what we did"},
                  {"clip": 3, "start": 0.0, "end": 1.9, "words": "The end", "why": "the last thing"}]
            long = [dict(ok[0], start=0.0, end=2.9)] + ok[1:]
            quiet = [dict(ok[0], words=None)] + ok[1:]
            steps = self.run_job(*self.opened(),
                                 [call("arrange_clips", story="s", clips=long, why="order")],
                                 [call("arrange_clips", story="s", clips=[{k: v for k, v in c.items() if v is not None}
                                                                          for c in quiet], why="order")],
                                 lambda asking: (setattr(tools, "OVER_LENGTH", 0.1), [call(
                                     "arrange_clips", story="s", clips=ok, why="order")])[1],
                                 lambda asking: (setattr(tools, "LINE_EVERY", 1.0), [call(
                                     "arrange_clips", story="s", clips=ok, why="order")])[1],
                                 "Done.")
        finally:
            tools.LONGEST_PIECE, tools.OVER_LENGTH, tools.LINE_EVERY = had
        self.assertIn("The program fitted the pieces: clip 2: 0-2.5 s of the 0-2.9 s you gave (a piece is at most "
                      "2.5 s)", steps[2]["result"])
        self.assertIn("The first piece (clip 2) opens the clip: give it words", steps[3]["result"])
        self.assertIn("The pieces add up to 4.9 s; this clip is 3 to 4 s (about 40 s was asked). Use fewer clips.",
                      steps[4]["result"], "no piece is over 2 s, so none is cut")
        self.assertNotIn("lines of words", steps[4]["result"])
        self.assertIn("2 lines of words for 4.9 s: give at least 5", steps[5]["result"])

    def test_the_pieces_are_joined_in_order_and_each_line_stays_until_the_next(self):
        steps = self.run_job(*self.opened(), [call(
            "arrange_clips", story="from the station to the end of the day", why="tells the day in order",
            clips=[{"clip": 2, "start": 0.5, "end": 2.5, "words": "We arrive", "why": "it shows where we are"},
                   {"clip": 1, "start": 1.0, "end": 2.0, "why": "what we did first"},
                   {"clip": 3, "start": 0.0, "end": 1.9, "words": "The end", "why": "the last thing seen"}],
            left_out=[])], "Done.")
        said = steps[-1]["result"]
        self.assertIn("Joined 3 clips into one recording, 4.9 s long", said)
        self.assertIn("0 s: clip 2, 0.5-2.5 s of it: it shows where we are", said)
        self.assertIn("2 s: clip 1, 1-2 s of it", said)
        self.assertIn("3 s: clip 3, 0-1.9 s of it", said)
        self.assertIn("Made the clip from moment 1", said)
        state = self.chat.project.state()
        self.assertEqual([(t["text"], t["start"], t["end"]) for t in state["titles"]],
                         [("We arrive", 0.0, 3.0), ("The end", 3.0, 4.9)])
        self.assertAlmostEqual(state["length"], 4.9, delta=0.05)
        self.assertEqual(state["clip"]["pieces"], [[0.0, self.chat.recording["seconds"]]],
                         "nothing is cut out of a montage: Muse chose every second")
        self.assertIsNone(tools._still(self.chat.recording, tools.MONTAGE, 0.0, 4.9, None),
                          "no silence is cut from a montage, however still its picture")
        self.assertTrue(self.chat.recording["path"].startswith(os.path.join(recording.WORK, "montage")))
        again = agent.Chat.load(self.chat.id)
        self.assertEqual(len(again.moments), 1)
        self.assertEqual(again.looks[again.moments[0]["look"] - 1]["end"], again.moments[0]["end"])

    def test_words_asked_for_in_english_go_on_in_english_unless_the_person_typed_them(self):
        self.chat.emit("person", job=None, text="ทำคลิปทริป ซับภาษาอังกฤษ เปิดด้วยคำว่า สถานีแรก")
        thai = [{"clip": 2, "start": 0.5, "end": 2.5, "words": "ถึงสถานีแล้ว", "why": "where we are"},
                {"clip": 1, "start": 1.0, "end": 2.0, "words": "Lunch", "why": "what we did first"},
                {"clip": 3, "start": 0.0, "end": 1.9, "words": "จบวัน", "why": "the last thing seen"}]
        typed = [dict(thai[0], words="สถานีแรก"), thai[1], dict(thai[2], words="The end")]
        steps = self.run_job([call("open_clips", folder=self.folder, why="the trip's clips")],
                             [call("load_recipe", type="montage", subtitles="English", why="clips of a trip")],
                             [call("arrange_clips", story="the day", clips=thai, left_out=[], why="order")],
                             [call("arrange_clips", story="the day", clips=typed, left_out=[], why="order")],
                             "Done.")
        self.assertTrue(steps[2]["refused"])
        self.assertIn('in English, so a viewer who reads no Thai can follow: write these in English: "ถึงสถานีแล้ว"; '
                      '"จบวัน".', steps[2]["result"])
        self.assertFalse(steps[3]["refused"], steps[3]["result"])
        self.assertEqual([t["text"] for t in self.chat.project.state()["titles"]], ["สถานีแรก", "Lunch", "The end"])


    def test_at_an_event_every_face_is_kept_and_one_is_blurred_by_its_number(self):
        def people(rec, pieces, face, span=None):
            return [{"n": 1, "seen": 4, "of": 10, "from": 0.5, "to": 2.0, "box": [10, 10, 40, 40], "crop": None,
                     "points": [[0.5 * k, 10 + 2 * k, 10, 40, 40] for k in range(1, 5)]},
                    {"n": 2, "seen": 2, "of": 10, "from": 3.0, "to": 3.5, "box": [100, 20, 30, 30], "crop": None,
                     "points": [[3.0, 100, 20, 30, 30], [3.5, 100, 20, 30, 30]]}]

        had = tools._people
        tools._people = people
        try:
            steps = self.run_job(*self.opened(), [call(
                "arrange_clips", story="the day", why="in order", left_out=[],
                clips=[{"clip": 2, "start": 0.5, "end": 2.5, "words": "We arrive", "why": "where we are"},
                       {"clip": 1, "start": 1.0, "end": 2.0, "why": "what we did first"},
                       {"clip": 3, "start": 0.0, "end": 1.9, "words": "The end", "why": "the last thing seen"}])],
                [call("show_people", at=9, why="too late")],
                [call("show_people", at=1.0, why="the man in white at 1 s")],
                [call("decide_people", blur=[1], why="the person asked")], "Done.")
        finally:
            tools._people = had
        arranged, late, shown, decided = steps[-4:]
        self.assertIn("2 people's faces are in the clip, all kept as they are", arranged["result"])
        self.assertNotIn("Decide about each", arranged["result"])
        self.assertTrue(late["refused"])
        self.assertFalse(shown["refused"], shown["result"])
        self.assertIn("1 face at 1 s, numbered in the picture that comes next:\n- person 1: 0:00-0:02 of the clip, kept",
                      shown["result"])
        self.assertNotIn("person 2", shown["result"])
        self.assertFalse(decided["refused"], decided["result"])
        self.assertEqual([q["blur"] for q in self.chat.project.state()["people"]], [True, False])
        faces = self.chat.faces(1.0)
        self.assertEqual([(f["n"], f["blur"]) for f in faces], [(1, True)], "the page sees who shows at 1 s")
        self.assertTrue(all(0 <= v <= 1920 for v in faces[0]["box"]), "in frame pixels")
        self.assertEqual(self.chat.faces(9.0), [])
        self.assertTrue(self.chat.set_person(1, False), "the person taps the blurred face")
        self.assertFalse(self.chat.set_person(9, True), "there is no person 9")
        self.assertEqual([q["blur"] for q in self.chat.project.state()["people"]], [False, False])
        self.assertIn("the person kept person 1 by tapping their face", self.chat.told)
        self.assertEqual(self.chat.clip_view()["people"], 2)

    def test_a_line_that_would_cover_a_face_goes_up_where_it_covers_none(self):
        had = tools._face_times
        tools._face_times = lambda state, rec: ([(t / 4, [[0, 0, 1080, 1920]]) for t in range(1, 4)], 0.5)
        try:
            steps = self.run_job(*self.opened(), [call(
                "arrange_clips", story="the day", why="in order", left_out=[],
                clips=[{"clip": 2, "start": 0.5, "end": 2.5, "words": "We arrive", "why": "where we are"},
                       {"clip": 1, "start": 1.0, "end": 2.0, "why": "what we did first"},
                       {"clip": 3, "start": 0.0, "end": 1.9, "words": "The end", "why": "the last thing seen"}])],
                "Done.")
        finally:
            tools._face_times = had
        said = steps[-1]["result"]
        self.assertIn('"We arrive" at the top from 1 to 3 s, the first part of its piece where it covers no face', said)
        self.assertEqual([(t["text"], t["start"], t["end"]) for t in self.chat.project.state()["titles"]],
                         [("We arrive", 1.0, 3.0), ("The end", 3.0, 4.9)])


class Fit(unittest.TestCase):
    def test_a_long_piece_keeps_the_moving_seconds_from_its_start(self):
        self.assertEqual(tools._moving(0, 25, [], 8, 1), (0, 8))
        self.assertEqual(tools._moving(0, 25, [[0, 3.5]], 8, 1), (3.5, 11.5))
        self.assertEqual(tools._moving(0, 10, [[2, 4], [4.5, 10]], 8, 1), (0, 2))
        self.assertEqual(tools._moving(0, 10, [[0.5, 4], [4.5, 10]], 8, 1), None)

    def test_pieces_over_the_length_are_cut_by_the_same_share(self):
        pieces = [{"from": 0, "to": 8}, {"from": 2, "to": 4}, {"from": 0, "to": 12}]
        tools._shortened(pieces, 14)
        self.assertEqual([round(p["to"] - p["from"], 2) for p in pieces], [5.0, 2.0, 7.0])
        tools._shortened(pieces, 30)
        self.assertEqual([round(p["to"] - p["from"], 2) for p in pieces], [5.0, 2.0, 7.0], "never longer")


class Talking(unittest.TestCase):
    """A review told over several takes (Tarm's yeti folder, 10-28): Muse joined the first 8 s of each, his clap and
    the take he gave up before it included. Here his own lines, 100 s earlier, in a 26 s clip that talks through."""

    @classmethod
    def setUpClass(cls):
        cls.made = tempfile.mkdtemp()
        make_clip(os.path.join(cls.made, "review.mov"), seconds=26.0, shot="2026-03-10T14:00:00+0900")
        make_clip(os.path.join(cls.made, "desk.mov"), seconds=3.0, shot="2026-03-10T15:00:00+0900")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.made, ignore_errors=True)

    def setUp(self):
        Arrange.setUp(self)
        rec = recording.probe(os.path.join(self.folder, "review.mov"))
        hearing.keep(rec, 0.0, rec["seconds"], [dict(p, **{"from": round(p["from"] - 100, 2),
                                                           "to": round(p["to"] - 100, 2)}) for p in GAVE_UP])
        self.heard_claps, self.fix = takes.heard_claps, hearing.fix
        takes.heard_claps = lambda r, a=0.0, b=None: [[16.02, 16.37, 0]] if r["name"] == "review.mov" else []
        hearing.fix = lambda ask, lines, kind, before=(), names=(): (list(lines), [None] * len(lines))

    def tearDown(self):
        takes.heard_claps, hearing.fix = self.heard_claps, self.fix
        Arrange.tearDown(self)

    def test_a_clip_someone_talks_through_is_cut_by_its_lines_without_the_take_given_up(self):
        opened = [[call("open_clips", folder=self.folder, why="the review's takes")],
                  [call("load_recipe", type="montage", why="clips of a review")]]
        desk = {"clip": 2, "why": "the desk"}
        steps = Arrange.run_job(self, *opened, [call(
            "arrange_clips", story="the review", why="order",
            clips=[{"clip": 1, "words": "This mic", "why": "the review"}, desk])], [call(
                "arrange_clips", story="the review", why="order",
                clips=[{"clip": 1, "start": 2.5, "end": 12, "words": "This mic", "why": "the start"},
                       {"clip": 1, "start": 10, "end": 23.5, "why": "the presets"}, desk])], [call(
                "arrange_clips", story="the review", why="order",
                clips=[{"clip": 1, "start": 2.5, "end": 23.5, "words": "This mic", "why": "the review"}, desk])],
            "Done.")
        listed, refused, overlap, joined = (s["result"] for s in steps[0:1] + steps[2:5])
        self.assertIn("Someone talks through it", listed)
        self.assertIn("6.5-17.1 s cut: a take given up at a clap, said again after it", listed)
        self.assertNotIn("ขอโทษอันนี้จะเป็น", listed)
        self.assertIn("Clip 1 is someone talking for 26 s: give its piece a start and an end", refused)
        self.assertIn("Clip 1: 10-23.5 s overlaps a piece of it you already use", overlap)
        self.assertNotIn("named twice", overlap, "a clip someone talks through may give more than one piece")
        self.assertFalse(steps[4]["refused"], joined + steps[4].get("fault", ""))
        self.assertIn("a take given up at the clap at 0:16", joined)
        said = " ".join(p["text"] for p in self.chat.looks[-1]["said"])
        self.assertIn("ซึ่งตรงนี้นะครับก็จะเป็น", said)
        self.assertIn("แต่สำหรับใครนะครับที่แบบอยากจะลองใช้", said)
        self.assertNotIn("ขอโทษอันนี้จะเป็น", said, "the take he gave up is not in the clip")
        self.assertAlmostEqual(self.chat.recording["seconds"], (6.5 - 2.5) + (23.5 - 17.14) + 3.0, delta=0.3)

    def test_a_reviews_takes_are_cut_by_what_each_line_does_and_the_hook_stays_on_its_piece(self):
        """Tarm's yeti review (10-28): Glimmer was refused eleven times arranging the takes itself. Muse says what each
        line does; the program gives the cut, which goes through as it is."""
        jobs = json.dumps({"lines": [{"n": 1, "job": "what"}, {"n": 2, "job": "what"}, {"n": 3, "job": "other"},
                                     {"n": 4, "job": "point"}, {"n": 5, "job": "point"}]})
        opened = [[call("open_clips", folder=self.folder, why="the review's takes")],
                  [call("load_recipe", type="seller", why="a mic review")], jobs]
        steps = Arrange.run_job(self, *opened, [call("arrange_clips", story="the review", why="the cut", clips=[])],
                                [call("arrange_clips", story="the review", why="the program's cut", clips=[],
                                      hook="This mic", left_out=[{"clip": 2, "why": "the desk"}])], "Done.")
        loaded, no_hook, joined = steps[1]["result"], steps[2], steps[3]
        self.assertIn("The program asked what each of the 5 lines said in the takes does in a review", loaded)
        self.assertIn("1. clip 1, 2.82-6.08 s, what: ", loaded)
        self.assertIn("2. clip 1, 17.75-23.23 s, point: ", loaded)
        self.assertIn("Left out: clip 2, no one talks through it", loaded)
        self.assertIn("call arrange_clips with clips left empty ([]) and hook", loaded)
        asked = self.chat.job.brain.sent[2][-1]["content"]
        self.assertIn("3. take 1, 17.4 s: เอาละ", asked, "a line heard less surely is asked about too")
        self.assertTrue(no_hook["refused"])
        self.assertIn("Give hook: the words over the first piece of the program's cut", no_hook["result"])
        self.assertFalse(joined["refused"], joined["result"] + joined.get("fault", ""))
        self.assertIn("0 s: clip 1, 2.82-", joined["result"])
        self.assertIn("s: clip 1, 17.75-", joined["result"])
        arranged = [e for e in self.chat.events if e.get("arranged")][-1]["arranged"]
        self.assertEqual(arranged["left_out"], [{"clip": 2, "why": "the desk"}], "Muse's own reason is kept")
        self.assertEqual([p["why"] for p in arranged["pieces"]][:1], ["says what it is"])
        lines = self.chat.moments[0]["lines"]
        self.assertEqual([ln["text"] for ln in lines], ["This mic"])
        self.assertLess(lines[0]["end"], self.chat.recording["seconds"] - 4,
                        "over a talked review the hook stays over its own piece, not to the next words or the end")
        self.assertAlmostEqual(lines[0]["end"], 6.08 - 2.82, delta=0.4)
        self.assertIn("Captions put on by the program, as a talked review has them", joined["result"])
        captions = self.chat.project.state()["captions"]
        self.assertIs(captions.get("one"), True, "a talked review's captions go one line at a time")
        self.assertTrue(all(tools.text._letters(ln["text"].strip()) <= tools.ONE_LETTERS for ln in captions["lines"]))

    def test_the_cut_is_made_when_the_person_said_review_whichever_recipe_muse_loads(self):
        """Which recipe Muse loads for the same folder changes from run to run; the person's words do not."""
        opened = [[call("open_clips", folder=self.folder, why="the takes")],
                  [call("load_recipe", type="montage", why="clips")]]
        trip = Arrange.run_job(self, *opened, "Done.")[1]["result"]
        self.assertNotIn("The program asked", trip, "a trip someone talks through costs no question")
        self.chat.emit("person", job=self.chat.jobs + 1, text="ช่วยตัดคลิปรีวิวไมค์ให้หน่อย")
        jobs = json.dumps({"lines": [{"n": 1, "job": "what"}, {"n": 4, "job": "verdict"}], "hook": 1})
        said = Arrange.run_job(self, *opened, jobs, "Done.")[1]["result"]
        self.assertIn("made the cut in the review order", said)
        self.assertIn("1. clip 1, 2.82-4.84 s, proof: ", said, "the hook Muse named opens the cut")
        talk = json.dumps({"lines": [{"n": 1, "job": "look"}, {"n": 4, "job": "other"}], "hook": 1})
        other = Arrange.run_job(self, *opened, talk, "Done.")[1]["result"]
        self.assertIn("No line said in the takes reads as part of a review", other,
                      "lines that never say what the product is make no review, whatever the person called it")
        self.assertNotIn("made the cut in the review order", other)


class Listing(unittest.TestCase):
    def test_lines_the_hearing_was_not_sure_of_are_counted_not_listed(self):
        c = {"n": 1, "rec": {"name": "a.mov", "seconds": 9.0, "width": 1080, "height": 1920}, "when": "shot 10:00",
             "stills": [], "unheard": None,
             "said": [{"from": 1.0, "to": 2.0, "text": "here we are", "sure": 0.97},
                      {"from": 3.0, "to": 4.0, "text": "cabufro sinema", "sure": 0.4},
                      {"from": 5.0, "to": 6.0, "text": "losa", "sure": 0.5}]}
        self.assertEqual(tools._clip_lines(c)[1:], [
            "   1 s here we are",
            "   2 more lines heard but not sure, such as voices around the camera: left out."])

    def test_a_clip_someone_talks_through_lists_what_is_cut_instead_of_its_lines(self):
        said = [dict(p, **{"from": round(p["from"] - 75, 2), "to": round(p["to"] - 75, 2)}) for p in ASIDE] + [
            {"from": 16.5, "to": 18.5, "text": "แต่สำหรับใครนะครับที่แบบอยากจะลองใช้", "sure": 0.95}]
        c = {"n": 6, "rec": {"name": "IMG_3375.MOV", "seconds": 19.0, "width": 1920, "height": 1080},
             "when": "shot 22:45", "stills": [], "unheard": None, "said": said, "claps": [[14.43, 14.71, 0]]}
        listed = tools._clip_lines(c)[1:]
        self.assertTrue(listed[0].startswith("   Someone talks through it: give each piece of it a start and an end"))
        self.assertEqual(listed[1:5], ["   1.3 s ตั้งเสียงประมาณนี้", "   2.7-15.1 s cut: talk off the script before a clap",
                                       "   15.4 s แต่สำหรับใครที่แบบ", "   16.5 s แต่สำหรับใครนะครับที่แบบอยากจะลองใช้"])

    def test_a_clip_someone_talks_through_lists_lines_heard_less_surely_marked(self):
        """The verdict and the price of Tarm's yeti review were heard 0.83 to 0.86 and never shown to Muse (10-28)."""
        said = [{"from": 1.0 + 3 * k, "to": 3.5 + 3 * k, "text": "line %d" % k, "sure": 0.95} for k in range(3)] + [
            {"from": 10.0, "to": 12.0, "text": "ราคาไม่ถึงสองพันเลย", "sure": 0.84},
            {"from": 12.5, "to": 13.0, "text": "losa", "sure": 0.6}]
        c = {"n": 7, "rec": {"name": "IMG_3376.MOV", "seconds": 14.0, "width": 1920, "height": 1080},
             "when": "shot 22:50", "stills": [], "unheard": None, "said": said, "claps": []}
        listed = tools._clip_lines(c)[1:]
        self.assertIn("   A line marked ~ was heard less surely: some of its words may be misheard.", listed)
        self.assertIn("   10 s ~ ราคาไม่ถึงสองพันเลย", listed)
        self.assertEqual(listed[-1], "   1 more line heard but not sure, such as voices around the camera: left out.")
        c["said"] = said[:1] + said[3:]
        self.assertNotIn("ราคาไม่ถึงสองพันเลย", "\n".join(tools._clip_lines(c)),
                         "in a clip nobody talks through, a less sure line is a voice around the camera")


class Crowd(unittest.TestCase):
    def test_a_crowd_is_summed_up_and_decided_together(self):
        class Job:
            def __init__(self):
                self.pictures = []

            def attach(self, picture, caption):
                self.pictures.append(caption)

        people = [{"n": n, "seen": 3 if n <= 14 else 1, "of": 99, "from": 1.0, "to": 2.0, "box": [10, 10, 20, 20]}
                  for n in range(1, 100)]
        job = Job()
        lines = tools._people_lines(job, people, rec("crowd.mov"))
        self.assertEqual(len(lines), 2 + tools.PEOPLE_LISTED)
        self.assertIn("the other 87 (2 seen in two pictures or more, 85 seen once) are decided together with others",
                      lines[1])
        self.assertEqual(tools._people_named(list(range(1, 15))),
                         "14 people (others: blur or keep decides everyone not named)")
        self.assertEqual(tools._people_named([2, 5]), "person 2 and 5")

    def test_a_step_shows_the_clips_by_their_names(self):
        opened = {"clips": [{"n": 3, "rec": {"name": "IMG_0003.MOV"}}, {"n": 4, "rec": {"name": "IMG_0004.MOV"}}]}
        said = {"story": "s", "clips": [{"clip": 3, "why": "a"}, {"clip": 4, "why": "b"}], "left_out": [{"clip": 9}]}
        self.assertEqual(agent._named(said, opened), {"story": "s", "left_out": [{"clip": 9}], "clips": [
            {"clip": 3, "why": "a", "name": "IMG_0003.MOV"}, {"clip": 4, "why": "b", "name": "IMG_0004.MOV"}]},
            "Tarm 10-28: the step read [object Object]")
        self.assertEqual(agent._named(said, None), said)


class Overview(unittest.TestCase):
    """A long recording opens with four moments across it in one picture, not one picture from near its start."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work = mock.patch.object(recording, "WORK", self.tmp)
        self.work.start()
        os.makedirs(os.path.join(self.tmp, "frames"))
        self.asked = []

        def frame(rec, at=None):
            self.asked.append(at)
            name = "%d.jpg" % len(self.asked)
            cv2.imwrite(os.path.join(self.tmp, "frames", name), np.full((304, 540, 3), 60 * len(self.asked), np.uint8))
            return name
        self.frame = mock.patch.object(recording, "frame", frame)
        self.frame.start()

    def tearDown(self):
        self.frame.stop()
        self.work.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rec(self, seconds):
        return {"path": "ep81.mp4", "size": 1, "mtime": 1.0, "seconds": seconds}

    def test_an_hour_long_episode_is_seen_at_four_moments_two_by_two(self):
        path, times = montage.overview(self.rec(3604.34))
        self.assertEqual(times, [360.4, 1261.5, 2162.6, 3063.7])
        self.assertEqual(self.asked, times)
        h, w = cv2.imread(path).shape[:2]
        self.assertEqual((w, h), (400 * 2 + montage.GAP, 225 * 2 + montage.GAP))

    def test_a_phone_clip_keeps_its_one_picture(self):
        self.assertIsNone(montage.overview(self.rec(45.0)))
        self.assertEqual(self.asked, [])


if __name__ == "__main__":
    unittest.main()
