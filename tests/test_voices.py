"""Telling two voices apart by how high they are, and matching them to faces: python -m unittest discover -s tests"""
import json
import os
import shutil
import sys
import tempfile
import unittest

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

    def test_answers_that_do_not_settle_it_are_refused(self):
        for speaking, why in (((1, 2, 2, 2), "do not agree"), ((1, 1, 1, 1), "same person"),
                              ((0, 0, 2, 2), "could not tell")):
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
