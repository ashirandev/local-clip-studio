"""The upright frame following a game's play: the mark read from pictures, the path, and the clip that uses it."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from clipstudio import focus, pitch  # noqa: E402

GRASS, CYAN = (40, 140, 40), (255, 220, 0)
REC = {"path": "x.mp4", "width": 1920, "height": 1080}
WIDTH = 608


def picture(x=None, grass=True, y=500, side=12):
    p = np.zeros((pitch.H, pitch.W, 3), np.uint8)
    if grass:
        p[:] = GRASS
    if x is not None:
        p[y:y + side, x - side // 2:x + side // 2] = CYAN
    return p


def filmed(pictures):
    return lambda rec, a, b: iter(pictures)


class Read(unittest.TestCase):
    def test_the_mark_on_the_pitch_is_found(self):
        self.assertAlmostEqual(pitch._read(picture(700))[0], 699.5, delta=1)

    def test_a_picture_with_little_pitch_has_no_mark(self):
        """A player card's edge in a close-up is the same colour."""
        self.assertIsNone(pitch._read(picture(700, grass=False))[0])

    def test_the_score_board_and_the_names_are_not_read(self):
        self.assertIsNone(pitch._read(picture(700, y=100))[0], "the board")
        self.assertIsNone(pitch._read(picture(700, y=940))[0], "under the pitch")
        self.assertIsNone(pitch._read(picture(950, y=860))[0], "the names")
        self.assertIsNotNone(pitch._read(picture(300, y=860))[0], "the same height away from the names is pitch")

    def test_a_speck_or_a_blot_is_no_mark(self):
        self.assertIsNone(pitch._read(picture(700, side=4))[0], "too small")
        self.assertIsNone(pitch._read(picture(700, side=40))[0], "too big")


class Path(unittest.TestCase):
    def test_small_moves_do_not_move_the_frame_and_a_big_one_does_not_jump(self):
        still = pitch._aim(np.array([960.0 + 40 * np.sin(k / 15.0) for k in range(180)]), WIDTH)
        self.assertLess(still.max() - still.min(), 30, "it does not breathe with each pass")
        moved = pitch._aim(np.array([500.0] * 60 + [1400.0] * 60), WIDTH)
        self.assertLessEqual(np.abs(np.diff(moved)).max(), pitch.MOST_PAN / pitch.TRACK_FPS + 0.01)
        self.assertLess(moved[0], moved[-1])

    def test_a_stretch_ending_at_one_end_goes_to_that_goal(self):
        left = pitch._aim(np.array([700.0] * 90), WIDTH)
        self.assertGreater(left[30], 200, "with the play through the stretch")
        self.assertLess(left[-1], 150, "towards the left edge, where the goal is, the still zone short of it")
        right = pitch._aim(np.array([1220.0] * 90), WIDTH)
        self.assertLess(right[30], pitch.W - WIDTH - 200)
        self.assertGreater(right[-1], pitch.W - WIDTH - 150)

    def test_a_stretch_with_no_mark_holds_the_middle(self):
        self.assertTrue((pitch._aim(np.full(30, np.nan), WIDTH) == (pitch.W - WIDTH) / 2).all())

    def test_a_clip_follows_the_mark_and_a_close_up_goes_to_the_middle_at_the_cut(self):
        match = [picture(600 + k) for k in range(75)]
        close = [np.full((pitch.H, pitch.W, 3), 200, np.uint8)] * 45
        with mock.patch.object(pitch, "_frames", filmed(match + close)):
            plan = pitch.follow(REC, [[0.0, 8.0]], WIDTH)
        self.assertEqual(plan["seen"], 1.0)
        shots = plan["shots"]
        self.assertTrue(all(s["line"] and s["y"] == 0 for s in shots))
        self.assertLess(focus.at(shots, 2.0)[0], (pitch.W - WIDTH) / 2 - 100, "on the side the play is on")
        self.assertEqual(focus.at(shots, 7.5)[0], 656, "the middle")
        self.assertLess(next(b["from"] - a["from"] for a, b in zip(shots, shots[1:]) if abs(b["x"] - a["x"]) > 100), 0.1,
                        "a cut is a jump, not a pan across the close-up")

    def test_the_pieces_of_a_clip_run_on_in_clip_seconds(self):
        with mock.patch.object(pitch, "_frames", lambda rec, a, b: iter([picture(600)] * 30)):
            shots = pitch.follow(REC, [[10.0, 12.0], [20.0, 22.0]], WIDTH)["shots"]
        times = [s["from"] for s in shots]
        self.assertEqual(times, sorted(times))
        self.assertEqual((times[0], times[-1]), (0.0, round(2 + 29 / 15.0, 3)))

    def test_a_recording_with_no_mark_is_not_followed(self):
        """A real match, or another game: the card stays the way to see it."""
        with mock.patch.object(pitch, "_frames", filmed([picture()] * 60)):
            self.assertIsNone(pitch.follow(REC, [[0.0, 4.0]], WIDTH))

    def test_a_mark_seen_too_rarely_is_not_followed(self):
        few = [picture(600)] * 10 + [picture()] * 80
        with mock.patch.object(pitch, "_frames", filmed(few)):
            self.assertIsNone(pitch.follow(REC, [[0.0, 6.0]], WIDTH))

    def test_a_standing_recording_is_not_followed(self):
        with mock.patch.object(pitch, "_frames", lambda rec, a, b: iter([picture(600)] * 60)):
            self.assertIsNone(pitch.follow(dict(REC, width=1080, height=1920), [[0.0, 4.0]], WIDTH))

    def test_the_frame_is_in_the_recordings_own_pixels(self):
        small = dict(REC, width=1280, height=720)
        with mock.patch.object(pitch, "_frames", lambda rec, a, b: iter([picture(1500)] * 90)):
            plan = pitch.follow(small, [[0.0, 6.0]], 406)
        self.assertEqual(plan["size"], [406, 720])
        self.assertEqual(plan["shots"][-1]["x"], 874, "1280 - 406")


class Line(unittest.TestCase):
    def evaluate(self, text, t):
        return eval(text.replace("if(", "(lambda c, a, b: a if c else b)(").replace("lt(", "(lambda a, b: a < b)("),
                    {"t": t})

    def shots(self, n=161):
        return [{"from": k * 0.5, "x": 100 + (k * 37) % 900, "y": 7, "line": True} for k in range(n)]

    def test_a_straight_line_goes_on_without_stopping_at_its_points(self):
        shots = [{"from": 0.0, "x": 0, "y": 0, "line": True}, {"from": 1.0, "x": 100, "y": 0, "line": True},
                 {"from": 2.0, "x": 200, "y": 0, "line": True}]
        self.assertEqual([focus.at(shots, t)[0] for t in (0.5, 1.0, 1.5)], [50.0, 100.0, 150.0])

    def test_ffmpegs_expression_is_the_same_path(self):
        shots = self.shots()
        text = focus.expression(shots, "x")
        for t in (0.0, 0.25, 11.1, 40.0, 79.9):
            self.assertAlmostEqual(self.evaluate(text, t), focus.at(shots, t)[0], places=3)
        self.assertEqual(focus.expression(shots, "y"), "7")

    def test_a_long_path_stays_shallow_for_ffmpegs_parser(self):
        deepest = depth = 0
        for c in focus.expression(self.shots(), "x"):
            depth += (c == "(") - (c == ")")
            deepest = max(deepest, depth)
        self.assertLess(deepest, 30, "160 points one inside the other do not parse")

    def test_a_product_follow_is_as_it_was(self):
        shots = [{"from": 0.0, "x": 0, "y": 0}, {"from": 2.0, "x": 100, "y": 0}]
        self.assertAlmostEqual(focus.at(shots, 2.3)[0], 50.0, msg="eased, half way through its pan")
        self.assertIn("(3-2*", focus.expression(shots, "x"))


if __name__ == "__main__":
    unittest.main()
