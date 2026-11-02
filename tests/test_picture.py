"""The picture Muse Image draws for what is said, against a fake images API, with no key and no network:
python -m unittest discover -s tests"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import cover, picture, recording  # noqa: E402
from test_cover import KEY, Images, size  # noqa: E402
from test_cover import picture as drawn  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def alpha(path, x, y):
    got = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vf", "crop=1:1:%d:%d,format=rgba" % (x, y), "-f",
                          "rawvideo", "-pix_fmt", "rgba", "-"], capture_output=True, check=True)
    return got.stdout[3]


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Drawn(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.images = Images(drawn(os.path.join(self.tmp, "drawn.jpg"), "color=c=red:s=1920x1280"))
        for patch in (mock.patch.object(recording, "WORK", os.path.join(self.tmp, "work")),
                      mock.patch.dict(os.environ, {cover.KEY_ENV: KEY, "LCS_IMAGE_URL": self.images.url})):
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.images.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_only_the_words_go_to_muse_image_and_the_same_words_are_drawn_once(self):
        got = picture.make("  a pale ghost   peeking from a dark doorway. ")
        self.assertEqual((got["usd"], got["reused"]), (picture.USD, False))
        self.assertEqual(size(got["path"]), (1920, 1280))
        (asked,) = self.images.got
        self.assertEqual(asked["path"], "/v1/images/generations")
        self.assertEqual(asked["auth"], "Bearer " + KEY)
        self.assertNotIn("images", asked["body"], "no frame of the footage goes to Meta")
        self.assertEqual(asked["body"]["size"], "1536x1024")
        self.assertIn("showing: a pale ghost peeking from a dark doorway. ", asked["body"]["prompt"])
        self.assertIn("no real, famous or recognisable person", asked["body"]["prompt"])
        self.assertIn("No writing", asked["body"]["prompt"])
        again = picture.make("a pale ghost peeking from a dark doorway")
        self.assertEqual((again["path"], again["usd"], again["reused"]), (got["path"], 0.0, True))
        self.assertEqual(len(self.images.got), 1, "the same words are paid for once")
        picture.make("a crowd of people looking at their phones")
        self.assertEqual(len(self.images.got), 2)

    def test_it_is_cut_to_its_band_and_fades_into_the_footage_at_its_lower_edge(self):
        path = picture.make("a ghost")["path"]
        fit = picture.fitted(path, 1080, 600)
        self.assertEqual(size(fit), (1080, 600))
        self.assertEqual([alpha(fit, 540, y) for y in (5, 380)], [255, 255], "opaque down to the fade")
        middle = alpha(fit, 540, 495)
        self.assertTrue(80 < middle < 175, middle)
        self.assertLess(alpha(fit, 540, 599), 6, "see-through at its lower edge")
        self.assertEqual(picture.fitted(path, 1080, 600), fit, "made once")

    def test_what_goes_wrong_is_one_sentence_to_act_on(self):
        with self.assertRaisesRegex(picture.PictureError, "^A picture needs words saying what it shows"):
            picture.make("   ")
        with mock.patch.dict(os.environ, {cover.KEY_ENV: ""}):
            with self.assertRaisesRegex(picture.PictureError, "paste your Meta API key in Options"):
                picture.make("a ghost")
        self.images.status = 400
        with self.assertRaisesRegex(picture.PictureError, r"refused to draw it \(HTTP 400: made-up refusal 400\)\. "
                                                          r"Say what the picture shows in other words\.$"):
            picture.make("a ghost")
        self.images.status = 429
        with self.assertRaisesRegex(picture.PictureError, "Wait a minute, then draw the picture again"):
            picture.make("a ghost")
        self.images.status, self.images.answer = 200, {"data": []}
        with self.assertRaises(picture.PictureError) as e:
            picture.make("a ghost")
        self.assertEqual(e.exception.usd, picture.USD, "a picture asked for and not given is still paid for")


if __name__ == "__main__":
    unittest.main()
