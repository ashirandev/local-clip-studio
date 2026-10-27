"""Effects, checked on raw frames FFmpeg drew (no encoding, so unchanged means byte-equal):
python -m unittest discover -s tests"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import effects  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
W, H = 640, 360
BUSY = ("nullsrc=s=%dx%d:r=30:d=%s,geq=lum='mod(X*37+Y*91,251)':cb='mod(X*13+Y*7,200)+28':"
        "cr='mod(X*5+Y*17,180)+40',format=yuv420p")


def frames(fragments, seconds=3, first=0, size=(W, H), source=BUSY, inputs=()):
    """The yuv420p frames, from frame first on, of a busy picture through the fragments: [(Y, U, V)] as arrays."""
    import numpy as np
    w, h = size
    graph, files = effects.join(fragments, "src", "outv", 1 + len(inputs)) if fragments else ("[src]null[outv]", [])
    full = "[0:v]trim=start_frame=%d,format=yuv420p[src];%s" % (first, graph)
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", source % (w, h, seconds)]
    for extra in list(inputs) + files:
        cmd += ["-loop", "1", "-framerate", "30", "-i", extra]
    cmd += ["-filter_complex", full, "-map", "[outv]", "-fps_mode", "passthrough", "-frames:v",
            str(int(seconds * 30) - first), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    run = subprocess.run(cmd, capture_output=True, timeout=120)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode("utf-8", "replace")[-2000:])
    size_y, size_c = w * h, (w // 2) * (h // 2)
    raw = np.frombuffer(run.stdout, np.uint8).reshape(-1, size_y + 2 * size_c)
    return [(f[:size_y].reshape(h, w), f[size_y:size_y + size_c].reshape(h // 2, w // 2),
             f[size_y + size_c:].reshape(h // 2, w // 2)) for f in raw]


def detail(y):
    a = y.astype(int)
    return int(abs(a[:, 1:] - a[:, :-1]).sum() + abs(a[1:, :] - a[:-1, :]).sum())


def changed(a, b):
    """(x0, y0, x1, y1) of the luma pixels that differ, or None."""
    import numpy as np
    ys, xs = np.nonzero(a[0] != b[0])
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def same_outside(test, a, b, box):
    """Every pixel of every plane outside box (an even box) is byte-equal."""
    x, y, w, h = box
    for plane, (pa, pb) in enumerate(zip(a, b)):
        s = 1 if plane == 0 else 2
        mask = pa != pb
        mask[y // s:(y + h) // s, x // s:(x + w) // s] = False
        test.assertEqual(int(mask.sum()), 0, "plane %d differs outside %r" % (plane, box))


class Join(unittest.TestCase):
    def test_labels_chain_and_files_are_numbered(self):
        a = effects.Fragment("[in]split[_a][_b];[_b]negate[_c];[_a][_c]overlay[out]", [])
        b = effects.Fragment("[in][f0]overlay[_x];[_x][f1]overlay[out]", ["one.png", "two.png"])
        c = effects.Fragment("[in][f0]overlay[out]", ["three.png"])
        graph, files = effects.join([a, b, c], "v0", "v9", 2)
        self.assertEqual(files, ["one.png", "two.png", "three.png"])
        self.assertEqual(graph, "[v0]split[v0_0_a][v0_0_b];[v0_0_b]negate[v0_0_c];[v0_0_a][v0_0_c]overlay[v0_1];"
                                "[v0_1][2:v]overlay[v0_1_x];[v0_1_x][3:v]overlay[v0_2];[v0_2][4:v]overlay[v9]")
        self.assertEqual(effects.join([], "a", "b", 1), ("[a]null[b]", []))

    def test_join_refusals(self):
        with self.assertRaisesRegex(ValueError, r"fragment 0 has \[in\] 2 times"):
            effects.join([effects.Fragment("[in]null[_a];[in]null[out]", [])], "a", "b", 1)
        with self.assertRaisesRegex(ValueError, r"fragment 0 uses \[f1\] but brings 1 file"):
            effects.join([effects.Fragment("[in][f1]overlay[out]", ["x.png"])], "a", "b", 1)
        with self.assertRaisesRegex(ValueError, "first_label 'a b' is not a label"):
            effects.join([], "a b", "c", 1)


class Refusals(unittest.TestCase):
    def test_blur_refusals_say_what_to_change(self):
        with self.assertRaisesRegex(ValueError, r"box \(10, 10, 0, 5\) has no size: w and h must be above 0"):
            effects.blur_box((10, 10, 0, 5), 0, 1)
        with self.assertRaisesRegex(ValueError, "end 1 is not after start 2"):
            effects.blur_box((0, 0, 10, 10), 2, 1)
        with self.assertRaisesRegex(ValueError, r"points 1 and 2 are 0.60 s apart \(0.2 to 0.8 s\), more than 0.5 s: "
                                                "split the track"):
            effects.blur_track([(0, 0, 0, 10, 10), (0.2, 0, 0, 10, 10), (0.8, 0, 0, 10, 10)])
        with self.assertRaisesRegex(ValueError, "point 1 is at 0.1 s, not after point 0 at 0.2 s"):
            effects.blur_track([(0.2, 0, 0, 10, 10), (0.1, 0, 0, 10, 10)])


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Blur(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain = frames([])

    def test_a_box_is_unreadable_inside_and_untouched_outside(self):
        box = (101, 61, 199, 119)
        got = frames([effects.blur_box(box, 1.0, 2.0)])
        even = (100, 60, 200, 120)
        for k in (0, 29, 60, 75):
            self.assertEqual(got[k][0].tobytes() + got[k][1].tobytes() + got[k][2].tobytes(),
                             self.plain[k][0].tobytes() + self.plain[k][1].tobytes() + self.plain[k][2].tobytes(), k)
        for k in (30, 45, 59):
            same_outside(self, got[k], self.plain[k], even)
            inside = (slice(even[1], even[1] + even[3]), slice(even[0], even[0] + even[2]))
            self.assertLess(detail(got[k][0][inside]), 0.1 * detail(self.plain[k][0][inside]), k)
            x0, y0, x1, y1 = changed(got[k], self.plain[k])
            self.assertTrue(x0 <= 101 and y0 <= 61 and x1 >= 300 and y1 >= 180, (x0, y0, x1, y1))

    def test_a_big_box_is_as_unreadable_as_a_small_one(self):
        for box in ((20, 20, 40, 40), (300, 20, 320, 320)):
            period = box[2] / 8
            source = ("nullsrc=s=%%dx%%d:r=30:d=%%s,geq=lum='128+100*sin(2*PI*X/%s)*sin(2*PI*Y/%s)':cb=128:cr=128,"
                      "format=yuv420p" % (period, period))
            plain = frames([], seconds=0.2, source=source)
            got = frames([effects.blur_box(box, 0, 1)], seconds=0.2, source=source)
            inside = (slice(box[1], box[1] + box[3]), slice(box[0], box[0] + box[2]))
            self.assertLess(detail(got[3][0][inside]), 0.1 * detail(plain[3][0][inside]), box)

    def test_a_track_moves_in_straight_lines_between_its_points(self):
        points = [(0.5 + 0.1 * k, 40 + 30 * k, 40 + (12 * k if k < 8 else 96 - 10 * (k - 8)), 80, 70)
                  for k in range(16)]
        got = frames([effects.blur_track(points)])
        for k in (14, 62):
            self.assertIsNone(changed(got[k], self.plain[k]), k)
        self.assertIsNotNone(changed(got[15], self.plain[15]))
        self.assertIsNotNone(changed(got[60], self.plain[60]))
        for k in (22, 37, 41):
            t = k / 30
            i = int((t - 0.5) / 0.1 + 1e-9)
            a, b = points[i], points[i + 1]
            share = (t - a[0]) / (b[0] - a[0])
            want = (a[1] + 40 + (b[1] - a[1]) * share, a[2] + 35 + (b[2] - a[2]) * share)
            x0, y0, x1, y1 = changed(got[k], self.plain[k])
            self.assertEqual((x1 - x0, y1 - y0), (92, 82), k)
            self.assertLessEqual(abs((x0 + x1) / 2 - want[0]), 2, (k, want, (x0, y0, x1, y1)))
            self.assertLessEqual(abs((y0 + y1) / 2 - want[1]), 2, (k, want, (x0, y0, x1, y1)))
            same_outside(self, got[k], self.plain[k], (x0 - x0 % 2, y0 - y0 % 2, 92 + x0 % 2 * 2, 82 + y0 % 2 * 2))
            inside = (slice(y0, y1), slice(x0, x1))
            self.assertLess(detail(got[k][0][inside]), 0.1 * detail(self.plain[k][0][inside]), k)

    def test_a_track_stays_inside_the_picture(self):
        got = frames([effects.blur_track([(0, 590, 300, 80, 80), (0.3, 600, 330, 80, 80)])], seconds=0.4)
        x0, y0, x1, y1 = changed(got[6], self.plain[6])
        self.assertEqual((x1, y1, x1 - x0, y1 - y0), (W, H, 92, 92))

    def test_a_frame_is_the_same_from_any_start(self):
        points = [(0.2 * k, 30 * k, 20 * k, 90, 90) for k in range(11)]
        whole = frames([effects.blur_track(points), effects.blur_box((300, 200, 100, 80), 0.5, 2.5)])
        alone = frames([effects.blur_track(points), effects.blur_box((300, 200, 100, 80), 0.5, 2.5)], first=40)
        self.assertEqual(whole[47][0].tobytes(), alone[7][0].tobytes())
        self.assertEqual(whole[47][1].tobytes(), alone[7][1].tobytes())


SQUARE = ("color=c=black:s=%dx%d:r=30:d=%s,drawbox=x=440:y=860:w=200:h=200:color=white:t=fill,"
          "drawbox=x=100:y=200:w=60:h=60:color=0x808080:t=fill,format=yuv420p")


def bright(frame, at=200):
    """(x0, y0, x1, y1) of the luma above `at`."""
    import numpy as np
    ys, xs = np.nonzero(frame[0] > at)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def same(a, b):
    return a[0].tobytes() + a[1].tobytes() + a[2].tobytes() == b[0].tobytes() + b[1].tobytes() + b[2].tobytes()


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Hits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.size = (effects.FRAME_W, effects.FRAME_H)
        cls.plain = frames([], seconds=1.6, first=24, size=cls.size, source=SQUARE)

    def got(self, fragment):
        return frames([fragment], seconds=1.6, first=24, size=self.size, source=SQUARE)

    def test_a_flash_brightens_inside_its_window_only(self):
        got = self.got(effects.flash(1.0, 0.15, 0.8))
        for k in (29, 35, 40):
            self.assertTrue(same(got[k - 24], self.plain[k - 24]), k)
        for k in (30, 32, 34):
            self.assertGreater(got[k - 24][0].mean(), self.plain[k - 24][0].mean() + 10, k)
        self.assertGreater(got[30 - 24][0].mean(), got[34 - 24][0].mean())

    def test_a_shake_moves_the_frame_and_settles(self):
        got = self.got(effects.shake(1.0, 0.4, 18))
        for k in (29, 42, 45):
            self.assertTrue(same(got[k - 24], self.plain[k - 24]), k)
        moved = []
        for k in range(30, 42):
            x0, y0, x1, y1 = bright(got[k - 24])
            moved.append(abs((x0 + x1) / 2 - 540) + abs((y0 + y1) / 2 - 960))
        self.assertGreater(max(moved[:5]), 6)
        self.assertLess(moved[-1], 2)

    def test_a_punch_in_is_bigger_around_its_centre(self):
        got = self.got(effects.punch_in(1.0, 0.5, 1.15, (540, 960)))
        for k in (29, 45):
            self.assertTrue(same(got[k - 24], self.plain[k - 24]), k)
        for k in (30, 37, 44):
            x0, y0, x1, y1 = bright(got[k - 24])
            self.assertLessEqual(abs((x1 - x0) - 230), 2, k)
            self.assertLessEqual(abs((x0 + x1) / 2 - 540), 1, k)
            self.assertLessEqual(abs((y0 + y1) / 2 - 960), 1, k)

    def test_hit_refusals(self):
        with self.assertRaisesRegex(ValueError, "length 9 is not a hit's length: use seconds above 0, up to 5"):
            effects.flash(1, 9)
        with self.assertRaisesRegex(ValueError, "zoom 0.9 is not a punch-in: use more than 1"):
            effects.punch_in(1, zoom=0.9)
        with self.assertRaisesRegex(ValueError, r"centre \(2000, 10\) is off the 1080x1920 frame"):
            effects.punch_in(1, centre=(2000, 10))
        with self.assertRaisesRegex(ValueError, "at -1 is below 0"):
            effects.shake(-1)


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Speed(unittest.TestCase):
    def test_a_frame_with_a_title_captions_and_a_hit_draws_in_under_a_second(self):
        import time
        from clipstudio import text
        folder = tempfile.mkdtemp(prefix="lcs speed ")
        self.addCleanup(shutil.rmtree, folder, True)
        st = text.STYLES["pop"]
        lines = text.caption_lines([{"from": 0.5, "to": 4.0, "text": "วันนี้ลดราคาเหลือเจ็ดสิบห้าบาทนะครับ"}])
        events = text.title("ลง AI ในคอม ฟรี", 0, 5, (90, 200, 900, 400), st, key="ฟรี") + text.captions(
            lines, st, (1350, 400))
        path = os.path.join(folder, "words.ass")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.document([st], events))
        graph, files = effects.join([text.burn(path), effects.punch_in(2.0, 0.5)], "src", "outv", 1)
        started = time.time()
        run = subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=30:d=3",
                              "-filter_complex", "[0:v]trim=start_frame=63:end_frame=64,format=yuv420p[src];" + graph,
                              "-map", "[outv]", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                              "-"], capture_output=True, timeout=60)
        took = time.time() - started
        self.assertEqual(run.returncode, 0, run.stderr[-500:])
        self.assertEqual(len(run.stdout), 1080 * 1920 * 3 // 2)
        self.assertLess(took, 1.0)
