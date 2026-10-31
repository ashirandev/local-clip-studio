"""Telling two voices apart by how high they are, and matching them to faces: python -m unittest discover -s tests"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import recording, talk, voices  # noqa: E402
from test_talk import LEFT, RIGHT, persons, two_people  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def voice(hz, seconds=1.0, rate=voices.RATE):
    """A voiced sound at hz with its first harmonics, as a voice has."""
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * hz * t) + 0.15 * np.sin(4 * np.pi * hz * t) +
            0.08 * np.sin(6 * np.pi * hz * t)).astype(np.float32)


def line(a, b):
    return {"from": a, "to": b, "text": "x"}


class NamedTurns(unittest.TestCase):
    def test_one_speakers_neighbours_join_and_a_short_turn_is_left_out(self):
        got = voices.turns_of([(0.0, 3.0, "A"), (3.0, 6.0, "A"), (6.0, 6.5, "B"), (6.5, 9.0, "C")], 100.0)
        self.assertEqual(got, [{"from": 100.0, "to": 106.0}, {"from": 106.5, "to": 109.0}])

    def test_the_longest_turns_are_named_and_stay_in_order(self):
        found = [(k * 2.0, k * 2.0 + 1.0 + k * 0.05, "AB"[k % 2]) for k in range(voices.MOST_TURNS + 3)]
        got = voices.turns_of(found, 0.0)
        self.assertEqual(len(got), voices.MOST_TURNS)
        self.assertEqual(got, sorted(got, key=lambda t: t["from"]))
        self.assertNotIn(0.0, [t["from"] for t in got], "the shortest goes")

    def test_a_person_is_named_only_when_wonder_is_sure_and_they_are_followed(self):
        people = [{"n": 1}, {"n": 2}]
        self.assertEqual(voices.who_answer('{"who": 2, "sure": 0.8, "why": "x"}', people), 2)
        self.assertIsNone(voices.who_answer('{"who": 2, "sure": 0.3, "why": "x"}', people), "not sure")
        self.assertIsNone(voices.who_answer('{"who": 0, "sure": 0.9, "why": "x"}', people), "someone with no number")
        self.assertIsNone(voices.who_answer('{"who": 4, "sure": 0.9, "why": "x"}', people), "nobody the layout follows")
        self.assertIsNone(voices.who_answer("not json", people))

    def test_a_look_follows_the_speaker_only_when_nearly_all_of_the_talk_is_named(self):
        turn = lambda a, b, who: {"from": a, "to": b, "who": who}
        self.assertTrue(talk.told_apart([turn(0, 95, 1), turn(95, 100, None)]))
        self.assertFalse(talk.told_apart([turn(0, 50, 1), turn(50, 100, None)]))
        self.assertFalse(talk.told_apart([turn(0, 5, None)]))
        self.assertFalse(talk.told_apart(None))

    @unittest.skipUnless(FFMPEG, "needs ffmpeg")
    def test_someone_the_faces_missed_joins_the_people_from_the_heads_muse_counted(self):
        """EP0 10-31: the man turned away had no number, and Wonder gave his turn to person 1."""
        people = [{"n": 1, "box": [280, 400, 160, 260], "seen": 40, "of": 50},
                  {"n": 2, "box": [820, 330, 120, 150], "seen": 40, "of": 50},
                  {"n": 3, "box": [1170, 316, 116, 145], "seen": 40, "of": 50}]
        heads = [[60, 336, 440, 344], [810, 330, 210, 215], [1140, 330, 230, 226], [1244, 440, 300, 270]]
        out = talk.with_heads(people, heads)
        self.assertEqual([p["n"] for p in out], [1, 2, 3, 4])
        self.assertEqual([p["box"] for p in out[:3]], [p["box"] for p in people], "a face found is the better box for a shot")
        self.assertEqual([p["head"] for p in out], heads, "every number goes on a head")
        self.assertEqual(out[3]["box"], [1264.0, 445.0, 260, 260], "the shots stay as close")
        fewer = talk.with_heads(people, heads[:2])
        self.assertEqual(len(fewer), 3, "a face found stays when the count misses it")
        self.assertIsNone(fewer[2].get("head"))
        self.assertEqual(len(talk.with_heads(people, heads + [[1700, 300, 100, 100]])), talk.MOST_PEOPLE)
        left = talk.with_heads(people, [[10, 300, 100, 100]] + heads[:3])
        self.assertEqual([p["n"] for p in left], [1, 2, 3, 4], "numbered from the left again")
        self.assertEqual(left[0]["head"], [10, 300, 100, 100])
        self.assertEqual(left[0]["seen"], 0)

    def test_heads_come_back_in_pixels_from_the_left_and_a_bad_answer_is_none(self):
        rec = {"width": 1920, "height": 1080}
        said = {"people": [{"head": [500, 100, 600, 300], "looks": "b"}, {"head": [0, 0, 100, 200], "looks": "a"},
                           {"head": [700, 300, 650, 100], "looks": "no size"}, {"head": [900, 0, 1100, 200], "looks": "off"}]}
        self.assertEqual(voices.heads_answer(json.dumps(said), rec), [[0.0, 0.0, 192.0, 216.0], [960.0, 108.0, 192.0, 216.0]])
        self.assertIsNone(voices.heads_answer("no", rec))
        self.assertIsNone(voices.heads_answer('{"people": [{"head": [1, 2]}]}', rec))
        self.assertIn("Count a person turned away", voices.HEADS_ASK)

    def test_a_number_sits_on_the_head_muse_saw(self):
        """Tag 1 floated in mid-air: 0.45 of a face box above a face looking sideways, off the middle of his head."""
        folder = tempfile.mkdtemp()
        try:
            path = os.path.join(folder, "tags.png")
            voices.numbers({"width": 960, "height": 540}, [{"n": 1, "box": [150, 240, 60, 80], "head": [100, 200, 120, 120]}],
                           (960, 540), path)
            ys, xs = np.nonzero(cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, 3])
            self.assertLess(abs(xs.mean() - 160), 3)
            self.assertLess(abs(ys.mean() - 190), 3)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_the_video_wonder_sees_has_numbers_and_sound_and_is_made_once(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        had, recording.WORK = recording.WORK, folder
        self.addCleanup(setattr, recording, "WORK", had)
        path = os.path.join(folder, "talk.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:s=320x180:r=10:d=3", "-f", "lavfi",
                        "-i", "sine=frequency=200:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        path], check=True)
        rec = {"path": path, "width": 320, "height": 180}
        made = voices.numbered(rec, 0.5, 2.5, [{"n": 1, "box": [60, 80, 40, 40]}], "turn")
        self.assertTrue(os.path.getsize(made) > 0)
        streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width", "-of", "csv=p=0",
                                  made], capture_output=True, text=True).stdout.split()
        self.assertEqual(sorted(s.split(",")[0] for s in streams), ["audio", "video"])
        self.assertIn("video,960", streams)
        os.remove(path)
        self.assertEqual(voices.numbered(rec, 0.5, 2.5, [{"n": 1, "box": [60, 80, 40, 40]}], "turn"), made, "kept")
        tags = cv2.imread(os.path.join(folder, "voices", "turn.png"), cv2.IMREAD_UNCHANGED)
        self.assertGreater(int(tags[:, :, 3].max()), 200, "the number is drawn on a see-through picture")
        self.assertEqual(int(tags[-1, -1, 3]), 0)

    def test_wonder_is_asked_with_the_video_and_how_many_people_have_numbers(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        path = os.path.join(folder, "turn.mp4")
        with open(path, "wb") as f:
            f.write(b"a video")
        parts = voices.who_question(path, [{"n": 1}, {"n": 2}, {"n": 3}])[0]["content"]
        self.assertTrue(parts[0]["text"].startswith("3 people in this video have a yellow number"))
        self.assertTrue(parts[1]["video_url"]["url"].startswith("data:video/mp4;base64,"))


class Pitch(unittest.TestCase):
    def test_a_voice_is_found_at_its_pitch_and_silence_has_none(self):
        self.assertAlmostEqual(voices.pitch(voice(118)), 118, delta=4)
        self.assertAlmostEqual(voices.pitch(voice(226)), 226, delta=6)
        self.assertIsNone(voices.pitch(np.zeros(voices.RATE, np.float32)))
        self.assertIsNone(voices.pitch(voice(200, seconds=0.1)), "too short to be a line")

    def test_two_voices_part_where_their_pitches_part_most(self):
        heard = [(line(0, 1), 113), (line(1, 2), 137), (line(2, 3), 205), (line(3, 4), 314), (line(4, 5), None)]
        cut = voices.split(heard)
        self.assertTrue(137 < cut < 205)
        self.assertEqual([v for _, v in voices.lines_of(heard, cut)], ["low", "low", "high", "high", None])
        self.assertIsNone(voices.split([(line(0, 1), 115), (line(1, 2), 128), (line(2, 3), 140)]),
                          "two men alike are not split")
        self.assertIsNone(voices.split([(line(0, 1), 115)]))

    def test_the_windows_follow_the_voices(self):
        wins = [{"from": 0.0, "to": 0.5, "who": 1}, {"from": 0.5, "to": 1.0, "who": 1},
                {"from": 1.0, "to": 1.5, "who": 1}, {"from": 1.5, "to": 2.0, "who": 2}]
        said = [{"from": 0.1, "to": 1.2, "who": 2}, {"from": 1.6, "to": 1.9, "who": None}]
        self.assertEqual([w["who"] for w in talk.by_voice(wins, said)], [2, 2, 2, 2],
                         "a voice decides; a line whose voice is not known leaves the mouths' choice")
        self.assertEqual([w["who"] for w in talk.by_voice([{"from": 3.0, "to": 3.5, "who": 1}], said)], [None],
                         "nobody speaks where no line is")


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Match(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.video = os.path.join(cls.tmp, "talk.mp4")
        two_people(cls.video)
        cls.rec = {"path": cls.video, "width": 1280, "height": 720}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")
        self.voiced = [(line(0.2, 1.8), "low"), (line(2.1, 3.9), "high"), (line(4.2, 5.8), "low"),
                       (line(4.5, 4.9), "high")]

    def tearDown(self):
        recording.WORK = self.work

    def ask(self, *speaking):
        asked = []

        def answer(content, schema):
            asked.append(content)
            return json.dumps({"speaking": list(speaking)})
        return answer, asked

    def test_each_voice_goes_to_the_person_named_in_most_of_its_pictures(self):
        answer, asked = self.ask(1, 1, 2, 2)
        self.assertEqual(voices.match(answer, self.rec, self.voiced, persons()), {"low": 1, "high": 2})
        pictures = [c for c in asked[0] if c["type"] == "image_url"]
        self.assertEqual(len(pictures), 4, "the longest lines of each voice, up to PICTURES each")
        self.assertIn("number drawn over their head", asked[0][0]["text"])

    def test_one_settled_voice_gives_the_other_voice_the_other_person(self):
        for speaking in ((1, 2, 2, 2), (0, 0, 2, 2), (1, 1, 0, 0)):
            answer, _ = self.ask(*speaking)
            self.assertEqual(voices.match(answer, self.rec, self.voiced, persons()), {"low": 1, "high": 2}, speaking)

    def test_answers_that_do_not_settle_it_are_refused(self):
        for speaking, why in (((1, 2, 1, 2), "do not agree"), ((1, 1, 1, 1), "same person"),
                              ((0, 0, 0, 0), "could not tell")):
            answer, _ = self.ask(*speaking)
            with self.assertRaises(voices.VoiceError) as e:
                voices.match(answer, self.rec, self.voiced, persons())
            self.assertIn(why, str(e.exception))

    def test_a_picture_marks_each_person_by_number(self):
        path = voices.picture(self.rec, 1.0, persons(), "marks")
        self.assertTrue(os.path.isfile(path))

    def test_the_shots_follow_the_voices_where_the_mouths_say_otherwise(self):
        def still_mouths(rec, a, b, people, fps=10):
            n = round((b - a) * fps)
            return [[9.0] * n, [1.0] * n]

        said = [line(0.2, 2.9), line(3.1, 5.8)]
        spoken = [dict(said[0], who=2), dict(said[1], who=1)]
        shots, _ = talk.plan(None, [[0, 6]], said, persons(), read=still_mouths, voices=spoken)
        self.assertEqual([s["show"] for s in shots], [2, 1])
        shots, _ = talk.plan(None, [[0, 6]], said, persons(), read=still_mouths)
        self.assertEqual([s["show"] for s in shots], [1], "without the voices the mouth that moves wins")


if __name__ == "__main__":
    unittest.main()
