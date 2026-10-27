"""The clip's drawing stages, checked on raw 1080x1920 frames FFmpeg drew from the project state (no encoding, so
unchanged means byte-equal): python -m unittest discover -s tests"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import layouts, project, recording, render, text  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
FACE = [530, 50, 60, 60]
RECORDING = ("nullsrc=s=640x360:r=30:d=%s,geq=lum='mod(X*37+Y*91,251)':cb='mod(X*13+Y*7,200)+28':"
             "cr='mod(X*5+Y*17,180)+40',drawbox=x=520:y=40:w=80:h=80:color=red:t=6,"
             "drawbox=x=100:y=200:w=120:h=80:color=blue:t=fill,format=yuv420p")


def state(kind="subject", **entries):
    """The project state of a 4 s clip of the busy recording in layout kind, with the entries given by type."""
    p = project.Project([{"do": "clip", "moment": 1, "pieces": [[0, 4]],
                          "layout": layouts.make(kind, (640, 360), FACE)}])
    for do in ("title", "captions", "blur", "hit"):
        for e in entries.get(do, []):
            p.add(dict(e, do=do))
    return p.state()


def drawn(st, t):
    """The 1080x1920 frame at clip second t as an (h, w, 3) array, from render.after_source."""
    import numpy as np
    graph, files = render.after_source(st, "src", "whole", 1)
    k = round(t * 30)
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", RECORDING % (k / 30 + 1)]
    for f in files:
        cmd += ["-loop", "1", "-framerate", "30", "-i", f]
    cmd += ["-filter_complex", "[0:v]trim=start_frame=%d[src];%s" % (k, graph), "-map", "[whole]",
            "-fps_mode", "passthrough", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    run = subprocess.run(cmd, capture_output=True, timeout=120)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode("utf-8", "replace")[-2000:])
    return np.frombuffer(run.stdout, np.uint8).reshape(1920, 1080, 3)


def changed(a, b):
    """(x0, y0, x1, y1) of the pixels that differ, or None."""
    import numpy as np
    ys, xs = np.nonzero((a != b).any(axis=2))
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def detail(picture, box):
    x, y, w, h = [int(round(v)) for v in box]
    a = picture[y:y + h, x:x + w].astype(int)
    return int(abs(a[:, 1:] - a[:, :-1]).sum() + abs(a[1:, :] - a[:-1, :]).sum())


def inside(found, box, slack=0):
    x, y, w, h = box
    return (found[0] >= x - slack and found[1] >= y - slack and found[2] <= x + w + slack
            and found[3] <= y + h + slack)


TITLE = {"text": "Boss down", "start": 0.5, "end": 3.5, "place": "top", "box": layouts.TOP, "key": None}


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg, and numpy for the pixel checks")
class Stages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a_blur_is_drawn_where_the_layout_shows_its_box_and_only_while_it_runs(self):
        box = [300, 100, 60, 60]
        plain = state()
        blurred = state(blur=[{"box": box, "start": 1.0, "end": 2.0}])
        (shown,) = layouts.to_frame(plain["clip"]["layout"], box)
        a, b = drawn(plain, 1.5), drawn(blurred, 1.5)
        inner = [shown[0] + 12, shown[1] + 12, shown[2] - 24, shown[3] - 24]
        self.assertLess(detail(b, inner), detail(a, inner) / 4, "nothing in the box can be read")
        bleed = 5 * shown[2] / box[2] + 4
        self.assertTrue(inside(changed(a, b), shown, slack=bleed), "only the box changed, give or take how far the "
                        "scaling reaches (lanczos 3 recording pixels, the colour planes 2 more) and the sharpening "
                        "(2 px): %s vs %s" % (changed(a, b), shown))
        for t in (0.9, 2.1):
            self.assertIsNone(changed(drawn(plain, t), drawn(blurred, t)), "no blur at %s s" % t)

    def test_a_blur_on_the_face_camera_is_blurred_in_the_circle(self):
        plain, blurred = state("card"), state("card", blur=[{"box": FACE, "start": 0.0, "end": 4.0}])
        lay = plain["clip"]["layout"]
        circle = [p["dst"] for p in lay["parts"] if p["name"] == "circle"][0]
        a, b = drawn(plain, 1.0), drawn(blurred, 1.0)
        ring = [circle[0] + 100, circle[1] + 100, 200, 200]
        (face,) = [f for f in layouts.to_frame(lay, FACE) if inside(f, circle, 1)]
        self.assertGreater(face[2], 100, "the face camera fills the circle")
        self.assertLess(detail(b, ring), detail(a, ring), "the face is blurred where the circle shows it")

    def test_captions_are_drawn_in_their_band_while_their_words_are_said(self):
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "watch this now"}])
        band = [layouts.CAPTIONS[1], layouts.CAPTIONS[3]]
        plain, captioned = state(), state(captions=[{"lines": lines, "band": band}])
        found = changed(drawn(plain, 1.5), drawn(captioned, 1.5))
        self.assertIsNotNone(found, "the captions are drawn")
        self.assertTrue(found[1] >= band[0] - 2 and found[3] <= band[0] + band[1] + 2, "inside the band: %s" % (found,))
        self.assertIsNone(changed(drawn(plain, 0.5), drawn(captioned, 0.5)), "nothing before the words are said")

    def test_a_word_hit_pops_on_its_centre_for_its_length(self):
        plain, hit = state(), state(hit=[{"kind": "word", "at": 1.0, "word": "POW", "centre": [540, 960]}])
        found = changed(drawn(plain, 1.4), drawn(hit, 1.4))
        self.assertIsNotNone(found, "the word is drawn")
        self.assertLess(abs((found[0] + found[2]) / 2 - 540), 12, found)
        self.assertLess(abs((found[1] + found[3]) / 2 - 960), 12, found)
        self.assertIsNone(changed(drawn(plain, 1.0 + render.WORD_SECONDS + 0.05),
                                  drawn(hit, 1.0 + render.WORD_SECONDS + 0.05)), "gone after its length")

    def test_a_flash_is_drawn_last_so_it_lights_the_words_too(self):
        plain, titled = state("card"), state("card", title=[TITLE])
        flashed = state("card", title=[TITLE], hit=[{"kind": "flash", "at": 1.0, "word": None, "centre": [540, 960]}])
        a, b, c = drawn(plain, 1.0), drawn(titled, 1.0), drawn(flashed, 1.0)
        words = (a != b).any(axis=2)
        edge = words & (b.max(axis=2) < 40)
        self.assertGreater(int(edge.sum()), 500, "the title has a dark outline")
        self.assertGreater(int(c[edge].min(axis=1).mean()), 100, "the flash lights the outline")
        self.assertIsNone(changed(drawn(titled, 1.3), drawn(flashed, 1.3)), "the flash is over")

    def test_camera_hits_move_the_picture_and_the_words_hold_still(self):
        titled = state("card", title=[TITLE])
        moved = state("card", title=[TITLE], hit=[{"kind": "punch", "at": 1.0, "word": None, "centre": [540, 960]},
                                                  {"kind": "shake", "at": 2.0, "word": None, "centre": [540, 960]}])
        x, y, w, h = layouts.TOP
        letters = drawn(titled, 0.9)[y:y + h, x:x + w].min(axis=2) > 240
        ys, xs = letters.nonzero()
        x, y, w, h = x + xs.min() - 6, y + ys.min() - 6, xs.max() - xs.min() + 13, ys.max() - ys.min() + 13
        for t in (1.1, 2.05):
            a, b = drawn(titled, t), drawn(moved, t)
            card = (a[1000:1300] != b[1000:1300]).any(axis=2).mean()
            self.assertGreater(card, 0.3, "the picture moved at %s s" % t)
            white_a = (a[y:y + h, x:x + w].min(axis=2) > 240)
            white_b = (b[y:y + h, x:x + w].min(axis=2) > 240)
            self.assertGreater(int(white_a.sum()), 2000, "the title's letters are there")
            self.assertLess(int((white_a != white_b).sum()), white_a.sum() * 0.02, "the title held still at %s s" % t)

    def test_a_drawing_that_cannot_be_made_is_refused_with_its_sentence(self):
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "watch this now"}])
        st = state(captions=[{"lines": lines, "band": [1090, 10]}])
        with self.assertRaises(ValueError) as e:
            render.after_source(st, "src", "whole", 1)
        self.assertIn("band", str(e.exception))
        st = state(hit=[{"kind": "wobble", "at": 1.0, "word": None, "centre": [540, 960]}])
        with self.assertRaises(ValueError) as e:
            render.after_source(st, "src", "whole", 1)
        self.assertEqual(str(e.exception), "hit kind 'wobble' is not a hit: use one of flash, shake, punch, word.")


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Preview(unittest.TestCase):
    """The preview frame Muse sees is drawn by the same stages."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")
        cls.path = os.path.join(cls.tmp, "stream.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", RECORDING % 5, "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", cls.path], check=True)
        cls.rec = recording.probe(cls.path)

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def grey(self, st, t):
        raw = subprocess.run([FFMPEG, "-v", "error", "-i", render.frame(self.rec, st, t, width=1080), "-f", "rawvideo",
                              "-pix_fmt", "gray", "pipe:1"], capture_output=True, check=True).stdout
        return raw

    def test_the_preview_frame_shows_the_captions_the_blur_and_the_word(self):
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "watch this now"}])
        plain = state()
        full = state(captions=[{"lines": lines, "band": [1090, 160]}],
                     blur=[{"box": [300, 100, 60, 60], "start": 1.0, "end": 2.0}],
                     hit=[{"kind": "word", "at": 1.2, "word": "POW", "centre": [540, 1450]}])
        a, b = self.grey(plain, 1.5), self.grey(full, 1.5)
        (shown,) = layouts.to_frame(plain["clip"]["layout"], [300, 100, 60, 60])

        def most(rows, cols):
            return max(abs(a[r * 1080 + c] - b[r * 1080 + c]) for r in rows for c in cols)

        self.assertGreater(most(range(1100, 1240, 4), range(100, 980, 4)), 100, "captions")
        self.assertGreater(most(range(1410, 1490, 4), range(460, 620, 4)), 100, "the big word")
        self.assertGreater(most(range(int(shown[1]) + 10, int(shown[1] + shown[3]) - 10, 4),
                                range(int(shown[0]) + 10, int(shown[0] + shown[2]) - 10, 4)), 60, "the blur")
        self.assertLessEqual(most(range(1700, 1900, 4), range(100, 980, 4)), 12, "the rest is the same picture")

    def test_a_frame_that_cannot_be_drawn_says_why(self):
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "watch this now"}])
        with self.assertRaises(render.NotDrawn) as e:
            render.frame(self.rec, state(captions=[{"lines": lines, "band": [1090, 10]}]), 1.5)
        self.assertIn("band [1090, 10]", str(e.exception).replace("(", "[").replace(")", "]"))


if __name__ == "__main__":
    unittest.main()
