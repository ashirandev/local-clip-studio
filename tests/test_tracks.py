"""Face tracks from the faces found every half second: one track per person, never swapped, bridged over one missed
picture, smooth, inside the picture, and ready for effects.blur_track: python -m unittest discover -s tests"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import effects, recording, tracks  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def found(pictures, size=(1920, 1080), every=0.5, start=10.0):
    """faces.sample()'s shape from a list of pictures, each a list of face boxes."""
    return {"size": list(size), "every": every,
            "samples": [{"t": round(start + k * every, 2), "faces": [list(b) + [0.9] for b in boxes]}
                        for k, boxes in enumerate(pictures)]}


def gaps(points):
    return [b[0] - a[0] for a, b in zip(points, points[1:])]


class Tracks(unittest.TestCase):
    def test_two_people_who_pass_each_other_stay_two_tracks(self):
        walker = [[100 + 40 * k, 300, 80, 80] for k in range(20)]
        sitter = [500, 300, 80, 80]
        for order in ("same", "swapped"):
            pictures = [[w, sitter] if order == "same" or k == 0 else [sitter, w] for k, w in enumerate(walker)]
            got = tracks.tracks(found(pictures))
            self.assertEqual(len(got), 2, order)
            moving = [tr for tr in got if tr["points"][-1][1] > 700][0]
            still = [tr for tr in got if tr is not moving][0]
            self.assertTrue(all(abs(p[1] - 500) < 1 for p in still["points"]), "the sitter's track never moves")
            xs = [p[1] for p in moving["points"]]
            self.assertEqual(xs, sorted(xs), "the walker's track only moves right (%s order)" % order)

    def test_a_face_missed_in_one_picture_is_bridged_and_in_two_the_track_ends(self):
        face = [600, 300, 90, 110]
        once = [[face] if k != 4 else [] for k in range(10)]
        (tr,) = tracks.tracks(found(once))
        self.assertLessEqual(max(gaps(tr["points"])), 0.5 + 1e-9)
        self.assertIn(12.0, [p[0] for p in tr["points"]], "a point is put where the face was missed")
        twice = [[face] if k not in (4, 5) else [] for k in range(10)]
        got = tracks.tracks(found(twice))
        self.assertEqual(len(got), 2, "missed in two pictures, the track ends")
        for tr in got:
            self.assertLessEqual(max(gaps(tr["points"])), 0.5 + 1e-9)
            effects.blur_track([[p[0] - 9.75] + p[1:] for p in tr["points"]])

    def test_a_face_that_jumps_across_the_picture_is_another_person(self):
        got = tracks.tracks(found([[[100, 300, 80, 80]], [[1500, 300, 80, 80]], [[1500, 300, 80, 80]]]))
        self.assertEqual(len(got), 2)
        got = tracks.tracks(found([[[100, 300, 80, 80]], [[110, 300, 200, 200]]]))
        self.assertEqual(len(got), 2, "a face more than half again bigger is another face")

    def test_the_points_are_smoothed(self):
        shaky = [[[600 + (8 if k % 2 else -8), 300, 90, 110]] for k in range(12)]
        (tr,) = tracks.tracks(found(shaky))
        inner = tr["points"][2:-2]
        self.assertLessEqual(max(abs(p[1] - 600) for p in inner), 4.01, [p[1] for p in inner])

    def test_the_track_is_held_half_a_picture_at_both_ends_and_kept_inside_the_picture(self):
        (tr,) = tracks.tracks(found([[[-20, 1000, 90, 110]] for _ in range(4)], size=(1920, 1080)))
        self.assertEqual((tr["from"], tr["to"]), (9.75, 11.75))
        self.assertEqual(tr["points"][0][1:], tr["points"][1][1:])
        for t, x, y, w, h in tr["points"]:
            self.assertTrue(x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080, (x, y, w, h))

    def test_the_face_that_sits_still_is_marked_steady_and_comes_first(self):
        streamer = [1690, 430, 92, 114]
        pictures = [[streamer] + ([[300 + 30 * k, 500, 60, 70]] if 3 <= k < 9 else []) for k in range(20)]
        got = tracks.tracks(found(pictures))
        self.assertEqual([tr["n"] for tr in got], [1, 2])
        self.assertTrue(got[0]["steady"])
        self.assertEqual(got[0]["seen"], 20)
        self.assertFalse(got[1]["steady"])
        self.assertEqual((got[1]["from"], got[1]["seen"]), (11.25, 6))
        self.assertIsNone(got[0]["crop"], "no crop without the recording")


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Crops(unittest.TestCase):
    def test_each_person_gets_a_small_picture_of_their_face(self):
        tmp = tempfile.mkdtemp()
        work, recording.WORK = recording.WORK, os.path.join(tmp, "work")
        try:
            path = os.path.join(tmp, "two.mp4")
            subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                            "color=gray:s=640x360:r=30:d=3,drawbox=x=100:y=100:w=60:h=60:color=red:t=fill,"
                            "drawbox=x=450:y=200:w=60:h=60:color=blue:t=fill", "-c:v", "libx264", "-preset",
                            "ultrafast", "-pix_fmt", "yuv420p", path], check=True)
            rec = recording.probe(path)
            got = tracks.tracks(found([[[100, 100, 60, 60], [450, 200, 60, 60]]] * 4, size=(640, 360), start=0.0),
                                rec)
            for tr, colour in zip(sorted(got, key=lambda tr: tr["box"][0]), ("red", "blue")):
                raw = subprocess.run([FFMPEG, "-v", "error", "-i", tr["crop"], "-f", "rawvideo", "-pix_fmt", "rgb24",
                                      "-"], capture_output=True, check=True).stdout
                side = int((len(raw) // 3) ** 0.5)
                self.assertEqual(side, tracks.CROP_HEIGHT)
                mid = (side // 2 * side + side // 2) * 3
                r, g, b = raw[mid:mid + 3]
                self.assertTrue((r > 180 and b < 80) if colour == "red" else (b > 180 and r < 80), (colour, r, g, b))
        finally:
            recording.WORK = work
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
