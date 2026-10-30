"""A person in front of a green cloth, cut off it by its colour, on recordings FFmpeg draws: python -m unittest
discover -s tests"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import green, picture, recording  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
W, H = 640, 360
# the cloth with a darker crease across it; the 'person' (a skin box touching the bottom), the white product they
# hold and their grey head, all one shape; a white lamp in the top right corner the cloth does not cover
SCENE = ("color=c=%s:s=640x360:r=30:d=3,drawbox=x=0:y=100:w=640:h=30:color=0x2E7A38:t=fill,"
         "drawbox=x=220:y=120:w=200:h=240:color=0xC89678:t=fill,drawbox=x=420:y=200:w=80:h=60:color=0xF0F0F0:t=fill,"
         "drawbox=x=250:y=60:w=140:h=60:color=0x808080:t=fill,drawbox=x=590:y=0:w=50:h=40:color=0xFFFFFF:t=fill")


def record(folder, name, cloth="0x46AA50"):
    path = os.path.join(folder, name)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", SCENE % cloth, "-c:v", "libx264", "-crf", "12",
                    "-pix_fmt", "yuv420p", path], check=True)
    return {"path": path, "name": name, "width": W, "height": H}


def frame(path, n):
    import numpy as np
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    every = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
    return len(every), every[n]


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Cloth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.rec = record(cls.tmp, "green.mp4")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        patch = mock.patch.object(recording, "WORK", os.path.join(self.tmp, "work"))
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_cloth_is_measured_along_the_frame_edge_its_darkest_crease_included(self):
        found = green.find(self.rec["path"], 0, 3, (W, H))
        self.assertTrue(50 < found["clear"] < 75, "the crease scores about 66, the cloth about 90: %s" % found)
        self.assertEqual(found["solid"], round(found["clear"] * green.SOLID, 1))
        self.assertGreaterEqual(found["edge"], green.LEAST_EDGE)

    def test_the_person_and_what_they_hold_are_kept_and_the_cloth_and_a_lamp_beside_them_are_not(self):
        made = green.mask(self.rec, 0.5, 2.5)
        self.assertEqual({k: made[k] for k in ("person", "fps", "size", "from", "to")},
                         {"person": None, "fps": 30, "size": [W, H], "from": 0.5, "to": 2.5})
        self.assertIn("clear", made["green"])
        count, grey = frame(made["path"], 30)
        self.assertEqual(count, 60, "30 frames a second over the 2 s asked")
        for (x, y), want, what in (((320, 300), 255, "skin"), ((460, 230), 255, "the white product"),
                                   ((320, 90), 255, "the grey head"), ((100, 300), 0, "the cloth"),
                                   ((100, 115), 0, "the crease"), ((615, 20), 0, "the lamp the cloth misses")):
            self.assertEqual(int(grey[y, x]), want, what)
        again = os.path.getmtime(made["path"])
        self.assertEqual(green.mask(self.rec, 0.5, 2.5), made)
        self.assertEqual(os.path.getmtime(made["path"]), again, "made once")

    def test_no_cloth_is_no_cloth_and_a_grey_one_is_said(self):
        grey = record(self.tmp, "grey.mp4", "0x808080")
        self.assertIsNone(green.find(grey["path"], 0, 3, (W, H)))
        self.assertIsNone(green.mask(grey, 0, 3))
        dull = record(self.tmp, "dull.mp4", "0x203620")
        with self.assertRaises(green.GreenError) as e:
            green.find(dull["path"], 0, 3, (W, H))
        self.assertIn("too dark or too grey", str(e.exception))


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Backdrop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        patch = mock.patch.object(recording, "WORK", os.path.join(self.tmp, "work"))
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_background_is_drawn_from_its_words_at_the_frame_size_and_fills_the_frame(self):
        drawn = os.path.join(self.tmp, "drawn.jpg")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=1152x2048", "-frames:v", "1",
                        drawn], check=True)
        asked = []

        def make(about, ask, size):
            asked.append((about, ask, size))
            return {"path": drawn, "usd": 0.01, "seconds": 9.0, "reused": False}

        with mock.patch.object(picture, "make", make):
            got = green.backdrop("a tidy gaming desk")
        self.assertEqual(asked, [("a tidy gaming desk", green.ASK, "1080x1920")])
        self.assertIn("No people, hands, writing", green.ASK)
        size = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                               got["path"]], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(size, "1080,1920")
        self.assertEqual(got["usd"], 0.01)


if __name__ == "__main__":
    unittest.main()
