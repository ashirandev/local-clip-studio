"""Reading the screen: which lines are private, how long each is hidden, and RapidOCR reading a real picture.

    python -m unittest discover -s tests
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import recording, screen  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
try:
    import rapidocr_onnxruntime  # noqa: F401
    OCR = True
except ImportError:
    OCR = False


def thai_id(first12):
    check = (11 - sum(int(d) * (13 - k) for k, d in enumerate(first12)) % 11) % 10
    return first12 + str(check)


class Private(unittest.TestCase):
    def test_each_kind_of_private_line_is_named(self):
        tid = thai_id("110170020345")
        for line, kind in [
            ("ashiranscore.com | Someone.dev@gmail.com's Account | Cloudflare", "an email address"),
            ("Call 081-234-5678 today", "a phone number"),
            ("+66 81 234 5678", "a phone number"),
            ("02-123-4567", "a phone number"),
            ("Tel:0912345678", "a phone number"),
            ("+66912345678", "a phone number"),
            ("4111 1111 1111 1111", "a card number"),
            ("4111111111111111", "a card number"),
            ("%s-%s-%s-%s-%s" % (tid[0], tid[1:5], tid[5:10], tid[10:12], tid[12]), "an ID number"),
            (tid, "an ID number"),
            ("dash.cloudflare.com/b4b0a8b12a7ffaa7f2a43fdaf370ca9/ashiranscore.com", "a key or an account number"),
            ("export OPENAI=sk-proj-abcdefghijklmnop1234", "a key or an account number"),
            ("hf_AbCdEfGhIjKlMnOpQrStUvWx", "a key or an account number"),
            ("password: hunter22", "a password"),
            ("API key = 9f8e7d6c", "a password"),
        ]:
            self.assertEqual(screen.private(line), kind, line)

    def test_ordinary_lines_of_a_screen_are_left_alone(self):
        for line in ["ashiran.com", "Ashiran Local AI - 127.0.0.1:8791", "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
                     "Muse-Glimmer-30B-UD-Q3_K_XL", "Huihui-Qwen3.8-27B-abliterated-Q3_K", "2026-10-12 14-17-28.mkv",
                     "-0.08%", "2,065+", "Forex 29 n.s. 2569", "luuatiuwhnForex100%Aaasls?luuauuu",
                     "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "127.0.0.1:8850/#chat/01c097110521", "Head to head, on one PC", "4111 1111 1111 1112",
                     "Password", "1234567890123"]:
            self.assertIsNone(screen.private(line), line)


class Hidden(unittest.TestCase):
    """hidden() on readings given by hand: what the reader saw at each second."""

    def setUp(self):
        self.real = screen.readings
        self.given = {}
        screen.readings = lambda rec, a, b, on_start=None: self.given[(a, b)]
        self.rec = {"width": 1920, "height": 1080}

    def tearDown(self):
        screen.readings = self.real

    def line(self, text, box):
        return {"text": text, "box": box, "sure": 0.99}

    def test_a_private_line_is_hidden_from_the_reading_before_it_until_the_first_without_it(self):
        mail = "write to me@example.com"
        self.given[(100.0, 120.0)] = [
            {"at": 100.0, "lines": [self.line("Welcome", [10, 10, 200, 20])]},
            {"at": 103.0, "lines": [self.line(mail, [100, 200, 300, 20])]},
            {"at": 106.0, "lines": [self.line(mail, [100, 210, 300, 20])]},
            {"at": 109.0, "lines": []},
            {"at": 112.0, "lines": [self.line(mail, [100, 200, 300, 20])]},
        ]
        found = screen.hidden(self.rec, [[100.0, 120.0]])
        self.assertEqual([(f["kind"], f["start"], f["end"]) for f in found],
                         [("an email address", 0.0, 9.0), ("an email address", 9.0, 20.0)])
        self.assertEqual(found[0]["box"], [93, 193, 314, 44], "padded, and grown to cover both places")

    def test_times_are_clip_seconds_across_pieces_and_a_line_at_the_start_is_hidden_from_0(self):
        key = [self.line("token ghp_abcdefghijklmnopqrstuvwxyz0123", [0, 0, 400, 18])]
        self.given[(10.0, 15.0)] = [{"at": 10.0, "lines": key}, {"at": 13.0, "lines": []}]
        self.given[(50.0, 60.0)] = [{"at": 50.0, "lines": []}, {"at": 52.0, "lines": key}]
        found = screen.hidden(self.rec, [[10.0, 15.0], [50.0, 60.0]])
        self.assertEqual([(f["start"], f["end"]) for f in found], [(0.0, 3.0), (5.0, 15.0)])
        self.assertEqual(found[0]["box"][:2], [0, 0], "kept inside the picture")

    def test_a_line_is_covered_only_by_one_blur_over_its_whole_box_and_time(self):
        item = {"kind": "an email address", "box": [100, 200, 300, 40], "start": 2.0, "end": 8.0}
        self.assertTrue(screen.covered(item, [{"box": [90, 190, 320, 60], "start": 0.0, "end": 8.0}]))
        self.assertFalse(screen.covered(item, [{"box": [90, 190, 320, 60], "start": 3.0, "end": 8.0}]))
        self.assertFalse(screen.covered(item, [{"box": [150, 190, 320, 60], "start": 0.0, "end": 9.0}]))
        self.assertFalse(screen.covered(item, [{"track": [[0, 100, 200, 300, 40]], "start": 0.0, "end": 9.0}]))


class Ahead(unittest.TestCase):
    """A stretch read in the background is waited for by the clip that needs it, never read twice."""

    def setUp(self):
        import threading
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, self.tmp
        self.real = screen._read_stretch
        self.calls, self.go = [], threading.Event()

        def slow(rec, a, b, on_start):
            self.calls.append((a, b))
            self.go.wait(5)
            out = [{"at": a, "lines": []}]
            screen._keep(rec, a, b, out)
            return out

        screen._read_stretch = slow
        self.rec = {"path": "x.mp4", "size": 1, "mtime": 1, "width": 1920, "height": 1080}

    def tearDown(self):
        import time
        self.go.set()
        for _ in range(250):
            if not screen._now:
                break
            time.sleep(0.02)
        screen._read_stretch = self.real
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def wait_for_the_first_call(self):
        import time
        for _ in range(100):
            if self.calls:
                return
            time.sleep(0.02)

    def test_the_clip_waits_for_the_reading_under_way(self):
        import threading
        screen.ahead(self.rec, 10.0, 20.0)
        self.wait_for_the_first_call()
        threading.Timer(0.3, self.go.set).start()
        self.assertEqual(screen.readings(self.rec, 10.0, 20.0), [{"at": 10.0, "lines": []}])
        self.assertEqual(self.calls, [(10.0, 20.0)], "read once")

    def test_stop_ends_the_wait(self):
        screen.ahead(self.rec, 30.0, 40.0)
        self.wait_for_the_first_call()
        with self.assertRaises(screen.Stopped):
            screen.readings(self.rec, 30.0, 40.0, on_start=lambda stop: stop())


@unittest.skipUnless(FFMPEG and OCR, "needs ffmpeg and RapidOCR")
class Reads(unittest.TestCase):
    """RapidOCR reads a made-up screen: a white page with an email address written on it."""

    def setUp(self):
        import cv2
        import numpy as np
        self.tmp = tempfile.mkdtemp()
        self.work = recording.WORK
        recording.WORK = os.path.join(self.tmp, "work")
        page = np.full((720, 1280, 3), 255, np.uint8)
        cv2.putText(page, "Settings", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2)
        cv2.putText(page, "Contact: someone@example.com", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        still = os.path.join(self.tmp, "page.png")
        cv2.imwrite(still, page)
        self.path = os.path.join(self.tmp, "screen.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-loop", "1", "-i", still, "-t", "2", "-r", "30", "-c:v",
                        "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", self.path], check=True)
        st = os.stat(self.path)
        self.rec = {"path": self.path, "size": st.st_size, "mtime": st.st_mtime, "width": 1280, "height": 720}

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_email_on_the_page_is_found_once_and_kept(self):
        found = screen.hidden(self.rec, [[0.0, 2.0]])
        self.assertEqual([(f["kind"], f["start"], f["end"]) for f in found], [("an email address", 0.0, 2.0)])
        x, y, w, h = found[0]["box"]
        self.assertTrue(x < 70 and y < 280 and x + w > 500 and y + h > 300, found[0]["box"])
        read = screen.kept(self.rec, 0.0, 2.0)
        self.assertEqual(len(read), 1, "a still page is read once, not every picture")
        self.assertTrue(any("Settings" in line["text"] for line in read[0]["lines"]), read)
