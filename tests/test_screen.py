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
            ("dash.cloudflare.com/a1b2c3d4e5f60718293a4b5c6d7e8f9/example.com", "a key or an account number"),
            ("export OPENAI=sk-proj-abcdefghijklmnop1234", "a key or an account number"),
            ("hf_AbCdEfGhIjKlMnOpQrStUvWx", "a key or an account number"),
            ("password: hunter22", "a password"),
            ("API key = 9f8e7d6c", "a password"),
            ("@dojin-25761 2 weeks ago", "a person's handle"),
            ("Thanks @somchai_99!", "a person's handle"),
        ]:
            self.assertEqual(screen.private(line), kind, line)

    def test_ordinary_lines_of_a_screen_are_left_alone(self):
        for line in ["ashiran.com", "Ashiran Local AI - 127.0.0.1:8791", "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
                     "Muse-Glimmer-30B-UD-Q3_K_XL", "Huihui-Qwen3.8-27B-abliterated-Q3_K", "2026-10-12 14-17-28.mkv",
                     "-0.08%", "2,065+", "Forex 29 n.s. 2569", "luuatiuwhnForex100%Aaasls?luuauuu",
                     "https://www.facebook.com/reel/1234567890", "127.0.0.1:8850/#chat/01c097110521", "Head to head, on one PC", "4111 1111 1111 1112",
                     "Password", "1234567890123", "price @ 20 baht", "@ab", "พี่ตูมตามอยู่จังหวัดไหนครับ"]:
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

    def test_a_blur_on_the_box_muse_was_told_covers_the_line_though_the_grid_rounded_it(self):
        """10-29, SIFU on Glimmer: told the names were at [781, 45, 879, 147] on its grid, it blurred exactly that, and
        the check failed three renders running: the grid rounds to the nearest of 1000 steps, so a blur on the box
        it was told can fall under a pixel short of the line."""
        item = {"kind": "a person's handle", "box": [1500.4, 48.4, 187.2, 110.3], "start": 0.0, "end": 50.0}
        told = [781 * 1.92, 45 * 1.08, (879 - 781) * 1.92, (147 - 45) * 1.08]
        self.assertTrue(screen.covered(item, [{"box": told, "start": 0.0, "end": 50.0}]))
        short = [told[0] + 5, told[1], told[2] - 5, told[3]]
        self.assertFalse(screen.covered(item, [{"box": short, "start": 0.0, "end": 50.0}]), "5 px short shows letters")


class Words(unittest.TestCase):
    """The words a stretch shows, for names: sure lines only, never a private one, most seen first."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, self.tmp
        self.real = screen._pictures
        self.rec = {"path": "x.mp4", "size": 1, "mtime": 1, "width": 1920, "height": 1080}

    def tearDown(self):
        screen._pictures = self.real
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def lines(self, *pairs):
        return [{"text": t, "box": [0, 0, 10, 10], "sure": s} for t, s in pairs]

    def test_words_come_from_the_readings_kept(self):
        screen._keep(self.rec, 10.0, 20.0, [
            {"at": 10.0, "lines": self.lines(("Get Elmoro", 0.99), ("ashiran.com", 0.99), ("luuatiuwhn", 0.6))},
            {"at": 12.0, "lines": self.lines(("Elmoro Local AI", 0.95), ("write to me@example.com Secret", 0.99))}])
        self.assertEqual(screen.words(self.rec, 10.0, 20.0), ["Elmoro", "Get", "Local", "ashiran.com"])
        os.remove(os.path.join(self.tmp, "screen", screen._key(self.rec, 10.0, 20.0) + ".json"))
        self.assertEqual(screen.words(self.rec, 10.0, 20.0), ["Elmoro", "Get", "Local", "ashiran.com"], "kept")

    def test_a_web_address_read_at_one_second_spells_a_name_said_at_another(self):
        from clipstudio import tools
        screen._keep(self.rec, 10.0, 20.0, [{"at": 10.0, "lines": self.lines(
            ("Get Elmoro at ashiran.com", 0.99), ("Ashiran-CV.pdf llama.cpp th.jobsdb.com", 0.99))}])
        self.assertEqual(screen.sites(self.rec), [])
        screen.words(self.rec, 10.0, 20.0)
        self.assertEqual(screen.sites(self.rec), ["ashiran.com", "ashiran", "th.jobsdb.com", "jobsdb"],
                         "addresses and their names; not a file name")
        screen.words(self.rec, 10.0, 20.0)
        self.assertEqual(len(screen.sites(self.rec)), 4, "kept once")
        shown = ["Elmoro"] + screen.sites(self.rec)
        self.assertEqual(tools._misspelled("ดูที่ ashilan.do.com กับ Achilan แล้วหางานบน JobDB", shown),
                         [("ashilan.do.com", "ashiran.com"), ("Achilan", "Ashiran"), ("JobDB", "Jobsdb")])

    def test_a_stretch_not_read_yet_gives_its_words_from_a_few_pictures(self):
        asked = []

        def pictures(rec, times):
            asked.extend(times)
            return [{"at": t, "lines": self.lines(("Elmoro", 0.99))} for t in times]

        screen._pictures = pictures
        self.assertEqual(screen.words(self.rec, 30.0, 60.0), ["Elmoro"])
        self.assertEqual(asked, [35.0, 45.0, 55.0])


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


@unittest.skipUnless(FFMPEG and OCR, "needs ffmpeg and RapidOCR")
class OwnWords(unittest.TestCase):
    """The video's own words found whatever their language: a 12 s recording with a Thai subtitle low in the picture
    from 1 to 3 s, a channel's name held still at the bottom left the whole time, and a Thai line at the top from 5 to
    7 s."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work = recording.WORK
        recording.WORK = os.path.join(self.tmp, "work")
        fonts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
        for name, words in (("sub.txt", "พี่ตูมตามมาแล้ว"), ("logo.txt", "OUR CHANNEL NAME"), ("top.txt", "ตอนที่หนึ่งเริ่ม")):
            with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as f:
                f.write(words)
        shutil.copy(os.path.join(fonts, "Kanit-Bold.ttf"), os.path.join(self.tmp, "k.ttf"))
        draw = "drawtext=fontfile=k.ttf:textfile=%s:fontsize=%d:fontcolor=white:borderw=4:bordercolor=black:x=%s:y=%d%s"
        self.path = os.path.join(self.tmp, "talk.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x506070:s=1280x720:r=30:d=12", "-vf",
                        ",".join([draw % ("sub.txt", 56, "(w-tw)/2", 560, ":enable='between(t,1,3)'"),
                                  draw % ("logo.txt", 40, "30", 650, ""),
                                  draw % ("top.txt", 56, "(w-tw)/2", 80, ":enable='between(t,5,7)'")]),
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "talk.mp4"],
                       check=True, cwd=self.tmp)
        st = os.stat(self.path)
        self.rec = {"path": self.path, "size": st.st_size, "mtime": st.st_mtime, "width": 1280, "height": 720}

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_thai_subtitle_low_in_the_picture_is_found_for_as_long_as_it_shows(self):
        found = screen.own_words(self.rec, [[0.0, 12.0]])
        self.assertEqual(len(found), 1, found)
        self.assertEqual(found[0]["start"], 1.0)
        self.assertTrue(3.0 <= found[0]["end"] <= 3.5, found)
        x, y, w, h = found[0]["box"]
        self.assertTrue(y > 520 and y + h < 660 and 300 < x + w / 2 < 980, found[0]["box"])
        self.assertEqual(screen.own_words(self.rec, [[0.0, 12.0]]), found, "kept, so asking again is instant")

    def test_clip_seconds_count_from_the_clips_own_start(self):
        found = screen.own_words(self.rec, [[0.0, 0.5], [2.0, 4.0]])
        self.assertEqual([(f["start"], f["end"]) for f in found], [(0.5, 1.5)], "shown from 2 to 3 s of the recording")


class Misspelled(unittest.TestCase):
    def test_a_name_written_another_way_than_on_the_screen_is_found(self):
        from clipstudio import tools
        shown = ["Elmoro", "ashiran.com", "Ashiran", "Models", "Settings", "com", "Your"]
        self.assertEqual(tools._misspelled("ดูข้อมูล Elmero บน Achilan.com #Achilan", shown),
                         [("Elmero", "Elmoro"), ("Achilan.com", "ashiran.com"), ("Achilan", "Ashiran")])
        for fine in ("Choose the right Model", "Elmoro on ashiran.com", "Your machine", "Gemma"):
            self.assertEqual(tools._misspelled(fine, shown), [], fine)

    def test_a_made_up_tag_half_like_a_screen_word_is_left_as_written(self):
        from clipstudio import tools
        self.assertEqual(tools._misspelled("#Elmoro #AutoApply", ["Elmoro", "Automation"]), [],
                         "round 8: #AutoApply became #Automation")


class Span(unittest.TestCase):
    def test_pieces_cut_from_a_read_stretch_are_not_read_again(self):
        key = {"text": "me@example.com", "box": [100, 100, 300, 20], "sure": 0.99}
        read = [{"at": 10.0, "lines": []}, {"at": 14.0, "lines": [key]}, {"at": 22.0, "lines": []}]
        asked = []

        def readings(rec, a, b, on_start=None):
            asked.append((a, b))
            return read

        had, screen.readings = screen.readings, readings
        try:
            found = screen.hidden({"width": 1920, "height": 1080}, [[10.0, 12.0], [15.0, 25.0]], span=[10.0, 30.0])
        finally:
            screen.readings = had
        self.assertEqual(asked, [(10.0, 30.0)], "read once, over the moment")
        self.assertEqual([(f["start"], f["end"]) for f in found], [(2.0, 9.0)], "shown from the second piece's start "
                         "(the reading at 14 s holds at 15 s) to the reading at 22 s, clip seconds")
