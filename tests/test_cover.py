"""The cover from Muse Image, against a fake images API, with no key and no network:
python -m unittest discover -s tests"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import cover, recording  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
KEY = "made-up-key-for-the-test"
TITLE = "รีวิวไมค์ Yeti Orb"


class Images:
    """A fake images API: answers /v1/images/edits with the picture it is given (or with status and a refusal),
    and keeps each request it got."""

    def __init__(self, picture=b"", status=200, answer=None):
        self.picture, self.status, self.answer, self.got = picture, status, answer, []
        images = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, data):
                body = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                images.got.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                   "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
                if images.status != 200:
                    return self.send(images.status, {"error": {"message": "made-up refusal %d" % images.status,
                                                               "type": "invalid_request_error"}})
                self.send(200, images.answer if images.answer is not None else
                          {"created": 1, "data": [{"b64_json": base64.b64encode(images.picture).decode("ascii")}]})

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d/v1" % self.http.server_port

    def close(self):
        self.http.shutdown()
        self.http.server_close()


def picture(path, source):
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", source, "-frames:v", "1", "-q:v", "2", path],
                   check=True)
    with open(path, "rb") as f:
        return f.read()


def size(path):
    got = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return tuple(int(v) for v in got.stdout.strip().split(","))


def pixel(path, x, y):
    got = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vf", "crop=2:2:%d:%d" % (x, y), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    return tuple(got.stdout[:3])


@unittest.skipUnless(FFMPEG, "FFmpeg is not installed")
class Cover(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pics = tempfile.mkdtemp()
        cls.frame = os.path.join(cls.pics, "frame.jpg")
        picture(cls.frame, "testsrc2=s=1080x1920")
        cls.other_frame = os.path.join(cls.pics, "other.jpg")
        picture(cls.other_frame, "testsrc2=s=1080x1920,hflip")
        # what Muse Image drew on 10-29 was 2:3, title nearly edge to edge: mid blue, with red at both sides
        cls.drawn = picture(os.path.join(cls.pics, "drawn.jpg"),
                            "color=c=0x4070D0:s=1280x1920,drawbox=x=0:y=0:w=40:h=ih:c=0xD03030:t=fill,"
                            "drawbox=x=1240:y=0:w=40:h=ih:c=0xD03030:t=fill")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.pics, ignore_errors=True)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")
        self.api = Images(self.drawn)
        self.env = mock.patch.dict(os.environ, {"META_API_KEY": KEY, "LCS_IMAGE_URL": self.api.url})
        self.env.start()
        os.environ.pop("LCS_IMAGE_MODEL", None)

    def tearDown(self):
        self.env.stop()
        self.api.close()
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def covers(self):
        folder = os.path.join(recording.WORK, "covers")
        return sorted(os.listdir(folder)) if os.path.isdir(folder) else []

    def test_the_frame_and_the_title_go_once_with_search_off_and_the_cover_is_the_whole_picture_at_9_16(self):
        got = cover.make(self.frame, "  %s \n" % TITLE, "Thai")
        self.assertEqual(len(self.api.got), 1)
        sent = self.api.got[0]
        self.assertEqual(sent["path"], "/v1/images/edits")
        self.assertEqual(sent["auth"], "Bearer " + KEY)
        body = sent["body"]
        with open(self.frame, "rb") as f:
            self.assertEqual(body["images"], [{"image_url": "data:image/jpeg;base64," +
                                               base64.b64encode(f.read()).decode("ascii")}], "the frame as it is")
        self.assertEqual({k: body[k] for k in ("model", "n", "size", "output_format", "response_format",
                                               "reasoning_strength")},
                         {"model": "muse-image-1.0", "n": 1, "size": "1080x1920", "output_format": "jpeg",
                          "response_format": "b64_json", "reasoning_strength": "low"})
        self.assertEqual(body["tool_enablement"], {"enable_image_search": False, "enable_web_search": False,
                                                   "enable_shell": False}, "the frame goes to no search")
        self.assertTrue(body["prompt"].endswith(": %s The title is in Thai: write it letter for letter as given, "
                                                "with every mark on its letters." % TITLE), body["prompt"])
        for kept in ("same faces", "Do not change any face", "Do not add any object"):
            self.assertIn(kept, body["prompt"])
        self.assertEqual((got["usd"], got["reused"]), (cover.USD, False))
        self.assertEqual(os.path.dirname(got["path"]), os.path.join(recording.WORK, "covers"))
        self.assertEqual(self.covers(), [os.path.basename(got["path"])], "the cover and nothing left beside it")
        self.assertEqual(size(got["path"]), (1080, 1920))
        for x in (10, 1069):
            r, g, b = pixel(got["path"], x, 960)
            self.assertTrue(r > 160 and b < 100, "the picture's side at x %d is in the cover, not cut: %s"
                            % (x, (r, g, b)))
        middle = pixel(got["path"], 540, 960)
        self.assertTrue(middle[2] > 170 and middle[0] < 100, middle)
        r, g, b = pixel(got["path"], 540, 40)
        self.assertTrue(60 < b < middle[2] - 15 and b > r + 30, "above the picture is its own copy, blurred and "
                        "darker, not black: %s against %s" % ((r, g, b), middle))

    def test_the_same_frame_and_words_give_the_same_cover_free_and_again_makes_a_new_one_that_stays(self):
        first = cover.make(self.frame, TITLE, "Thai")
        same = cover.make(self.frame, " ".join(TITLE.split()), "Thai")
        self.assertEqual(len(self.api.got), 1, "nothing is asked or paid for twice")
        self.assertEqual((same["path"], same["usd"], same["reused"]), (first["path"], 0.0, True))
        new = cover.make(self.frame, TITLE, "Thai", again=True)
        self.assertEqual(len(self.api.got), 2)
        self.assertNotEqual(new["path"], first["path"])
        self.assertEqual((new["usd"], new["reused"]), (cover.USD, False))
        self.assertTrue(os.path.isfile(first["path"]), "the cover before is kept")
        self.assertEqual(cover.make(self.frame, TITLE, "Thai")["path"], new["path"], "the newest is the cover")
        os.remove(first["path"])
        newer = cover.make(self.frame, TITLE, "Thai", again=True)
        self.assertNotIn(newer["path"], (first["path"], new["path"]), "a new cover never writes over one")
        self.assertEqual(cover.make(self.frame, TITLE, "Thai")["path"], newer["path"])
        cover.make(self.frame, "Yeti Orb in one minute", "English")
        cover.make(self.other_frame, TITLE, "Thai")
        self.assertEqual(len(self.api.got), 5, "new words or a new frame make a new cover")
        with mock.patch.dict(os.environ, {"LCS_IMAGE_MODEL": "muse-image-2.0"}):
            cover.make(self.frame, TITLE, "Thai")
        self.assertEqual((len(self.api.got), self.api.got[-1]["body"]["model"]), (6, "muse-image-2.0"))

    def test_with_no_key_nothing_is_sent_or_written_and_the_person_is_told_where_the_key_goes(self):
        os.environ["META_API_KEY"] = " "
        with self.assertRaises(cover.CoverError) as e:
            cover.make(self.frame, TITLE)
        self.assertIn("META_API_KEY", str(e.exception))
        self.assertIn("LCS_KEYS", str(e.exception))
        self.assertEqual((self.api.got, self.covers(), e.exception.usd), ([], [], 0.0))

    def test_each_refusal_is_one_sentence_with_what_the_api_said_and_no_cover_is_written(self):
        for status, says in ((400, "refused the frame and the title"), (401, "refused your key"),
                             (403, "refused your key"), (429, "Wait a minute"), (500, "could not make the cover")):
            self.api.status = status
            with self.assertRaises(cover.CoverError) as e:
                cover.make(self.frame, TITLE)
            sentence = str(e.exception)
            self.assertIn("HTTP %d: made-up refusal %d" % (status, status), sentence)
            self.assertIn(says, sentence)
            self.assertNotIn(KEY, sentence)
            self.assertEqual((e.exception.usd, self.covers()), (0.0, []))
            if status in (401, 403):
                self.assertIn("META_API_KEY", sentence)

    def test_an_answer_with_no_picture_or_one_that_is_not_a_picture_leaves_no_cover(self):
        for answer, says, usd in (({"data": []}, "answered without a picture", 0.0),
                                  ({"data": [{"url": "https://example.com/x.jpg"}]}, "answered without a picture",
                                   0.0),
                                  ({"data": [{"b64_json": base64.b64encode(b"not a picture").decode("ascii")}]},
                                   "picture could not be read", cover.USD)):
            self.api.answer = answer
            with self.assertRaises(cover.CoverError) as e:
                cover.make(self.frame, TITLE)
            self.assertIn(says, str(e.exception))
            self.assertEqual(e.exception.usd, usd, "a picture Meta drew is paid for even when it cannot be used")
            self.assertEqual(self.covers(), [], "no cover and no part file is left")

    def test_an_api_that_does_not_answer_a_frame_that_cannot_be_read_and_no_title(self):
        self.api.close()
        with self.assertRaises(cover.CoverError) as e:
            cover.make(self.frame, TITLE)
        self.assertIn("did not answer at %s" % self.api.url, str(e.exception))
        with self.assertRaises(cover.CoverError) as e:
            cover.make(os.path.join(self.tmp, "no-such-frame.jpg"), TITLE)
        self.assertIn("could not be read", str(e.exception))
        for title in ("", " \n ", None):
            with self.assertRaises(cover.CoverError) as e:
                cover.make(self.frame, title)
            self.assertIn("needs the clip's title", str(e.exception))
        self.assertEqual(self.covers(), [])
        self.api = Images(self.drawn)


if __name__ == "__main__":
    unittest.main()
