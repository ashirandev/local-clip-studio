"""Effects, checked on raw frames FFmpeg drew (no encoding, so unchanged means byte-equal):
python -m unittest discover -s tests"""
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import effects, layouts, recording  # noqa: E402

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


def marks(centre, other):
    """A black 1080x1920 picture with a white 40 px square on centre and a grey one on other."""
    return ("color=c=black:s=%%dx%%d:r=30:d=%%s,drawbox=x=%d:y=%d:w=40:h=40:color=white:t=fill,"
            "drawbox=x=%d:y=%d:w=40:h=40:color=0x808080:t=fill,format=yuv420p"
            % (centre[0] - 20, centre[1] - 20, other[0] - 20, other[1] - 20))


def found(frame, near, level, reach=80):
    """(centre x, centre y, side) of the square of luma level within reach px of near, each pixel weighed by how
    much of it the square covers."""
    import numpy as np
    x0, y0 = max(0, int(near[0]) - reach), max(0, int(near[1]) - reach)
    y = frame[0][y0:int(near[1]) + reach, x0:int(near[0]) + reach].astype(float)
    weight = np.clip((y - 16) / (level - 16), 0, 1)
    ys, xs = np.nonzero(weight > 0)
    w = weight[ys, xs]
    return float(((xs + x0) * w).sum() / w.sum()), float(((ys + y0) * w).sum() / w.sum()), float(w.sum()) ** 0.5


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Push(unittest.TestCase):
    START, END, ZOOM = 1.0, 2.0, 1.2

    def zoom_at(self, k):
        u = min(max((k / 30 - self.START) / (self.END - self.START), 0), 1)
        z = 1 + (self.ZOOM - 1) * (1 - math.cos(math.pi * u)) / 2
        return 2 * round(540 * z) / 1080, 2 * round(960 * z) / 1920

    def pushed(self, centre, other):
        size = (effects.FRAME_W, effects.FRAME_H)
        got = frames([effects.slow_push(self.START, self.END, self.ZOOM, centre)], seconds=2.2, first=27, size=size,
                     source=marks(centre, other))
        plain = frames([], seconds=2.2, first=27, size=size, source=marks(centre, other))
        return got, plain

    def test_it_grows_around_its_centre_and_only_inside_its_time(self):
        for centre, other in (((540, 960), (840, 1460)), ((130, 230), (540, 960)), ((1050, 1890), (700, 1300))):
            got, plain = self.pushed(centre, other)
            for k in (29, 60, 63):
                self.assertTrue(same(got[k - 27], plain[k - 27]), (centre, k))
            for k in (33, 40, 45, 52, 58):
                zx, zy = self.zoom_at(k)
                cx, cy, _side = found(got[k - 27], centre, 235)
                self.assertLessEqual(abs(cx - (centre[0] - 0.5)), 0.75, (centre, k, cx))
                self.assertLessEqual(abs(cy - (centre[1] - 0.5)), 0.75, (centre, k, cy))
                want = (centre[0] - 0.5 + (other[0] - centre[0]) * zx, centre[1] - 0.5 + (other[1] - centre[1]) * zy)
                gx, gy, side = found(got[k - 27], want, 126)
                self.assertLessEqual(abs(gx - want[0]), 1.5, (centre, k, gx, want))
                self.assertLessEqual(abs(gy - want[1]), 1.5, (centre, k, gy, want))
                self.assertLessEqual(abs(side - 40 * zx), 2, (centre, k, side))

    def test_it_eases_in_and_out_and_reaches_its_zoom(self):
        got, _plain = self.pushed((540, 960), (840, 1460))
        apart = [found(got[k - 27], (539.5 + 300 * self.zoom_at(k)[0], 1459.5), 126)[0] - 539.5
                 for k in range(30, 60)]
        steps = [b - a for a, b in zip(apart, apart[1:])]
        self.assertTrue(all(s >= -0.5 for s in steps), steps)
        self.assertLess(sum(steps[:4]), sum(steps[13:17]) / 3, "slow at the start")
        self.assertLess(sum(steps[-4:]), sum(steps[13:17]) / 3, "slow at the end")
        self.assertLessEqual(abs(apart[-1] - 300 * self.ZOOM), 2, apart[-1])

    def test_a_frame_is_the_same_from_any_start(self):
        push = effects.slow_push(0.5, 2.5, 1.1, (300, 700))
        size = (effects.FRAME_W, effects.FRAME_H)
        whole = frames([push], seconds=2, size=size, source=marks((300, 700), (800, 1500)))
        alone = frames([push], seconds=2, first=40, size=size, source=marks((300, 700), (800, 1500)))
        self.assertTrue(same(whole[47], alone[7]))

    def test_push_refusals(self):
        with self.assertRaisesRegex(ValueError, "start 1 to end 1.2 is 0.2 s: a slow push runs 0.5 to 30 seconds"):
            effects.slow_push(1, 1.2)
        with self.assertRaisesRegex(ValueError, "zoom 1.6 is not a slow push: use more than 1, up to 1.5"):
            effects.slow_push(1, 3, zoom=1.6)
        with self.assertRaisesRegex(ValueError, r"centre \(540, 2000\) is off the 1080x1920 frame"):
            effects.slow_push(1, 3, centre=(540, 2000))
        with self.assertRaisesRegex(ValueError, "end 1 is not after start 3"):
            effects.slow_push(3, 1)


RAMP = "color=black:s=256x16:d=1,format=rgb24,geq=r='X':g='X':b='X'"
PATCHES = ("color=0x7080a0:s=64x16:d=1[a];color=0xe02020:s=64x16:d=1[b];color=0xc68863:s=64x16:d=1[c];"
           "color=0x3a6ea5:s=64x16:d=1[d];[a][b][c][d]hstack=4,format=rgb24")


def graded(source, name):
    """The picture (rgb, as an (h, w, 3) int array) made yuv420p like a recording, through look name (or none)."""
    import numpy as np
    look = effects.look(name).graph[len("[in]"):-len("[out]")] if name else "null"
    run = subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", source, "-vf", "format=yuv420p,%s" % look,
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=60)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode("utf-8", "replace")[-2000:])
    return np.frombuffer(run.stdout, np.uint8).reshape(16, 256, 3).astype(int)


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Looks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ramp = {n: graded(RAMP, n)[8] for n in [None] + list(effects.LOOKS)}
        cls.patches = {n: graded(PATCHES, n)[8, [32, 96, 160, 224]] for n in [None] + list(effects.LOOKS)}

    def test_no_look_turns_a_darker_grey_lighter_than_a_brighter_one(self):
        for name in effects.LOOKS:
            luma = self.ramp[name] @ [0.2126, 0.7152, 0.0722]
            self.assertGreaterEqual(min(b - a for a, b in zip(luma[::4], luma[4::4])), -0.5, name)

    def test_vivid_deepens_the_darks_lifts_the_lights_and_keeps_grey_grey(self):
        plain, vivid = self.ramp[None], self.ramp["vivid"]
        self.assertLess(vivid[64].mean(), plain[64].mean() - 6)
        self.assertGreater(vivid[192].mean(), plain[192].mean() + 6)
        self.assertLessEqual(int(abs(vivid - vivid.mean(axis=1, keepdims=True)).max()), 4, "no colour cast")

    def test_vivid_gives_blues_and_greens_more_colour_and_leaves_skin_and_red_as_they_were(self):
        vivid, plain = self.patches["vivid"], self.patches[None]
        for k, what in ((0, "a dull blue"), (3, "a sky blue")):
            more = (vivid[k].max() - vivid[k].min()) / (plain[k].max() - plain[k].min())
            self.assertGreater(more, 1.15, "%s gets more colour: %.2f times" % (what, more))
        for k, what in ((1, "red"), (2, "skin")):
            shift = (vivid[k] - vivid[k].mean()) - (plain[k] - plain[k].mean())
            self.assertLessEqual(float(abs(shift).max()), 4, "%s keeps its colour (only its brightness follows the "
                                 "contrast): %s vs %s" % (what, vivid[k], plain[k]))

    def test_warm_warms_the_middle_and_keeps_black_and_white(self):
        plain, warm = self.ramp[None], self.ramp["warm"]
        self.assertGreaterEqual(warm[128][0] - warm[128][2], 12)
        for x in (0, 255):
            self.assertLessEqual(int(abs(warm[x] - plain[x]).max()), 3, "the %s end stays" % x)

    def test_film_lifts_black_cools_it_and_softens_white_and_colour(self):
        plain, film = self.ramp[None], self.ramp["film"]
        self.assertGreaterEqual(int(film[0].min()), 10)
        self.assertGreaterEqual(film[0][2] - film[0][0], 4, "the shadows lean blue")
        self.assertLessEqual(int(film[255].max()), 246)
        strong, strong_p = self.patches["film"][1], self.patches[None][1]
        self.assertLess(int(strong.max() - strong.min()), int(strong_p.max() - strong_p.min()) - 10)

    def test_a_look_that_does_not_exist_says_which_do(self):
        with self.assertRaisesRegex(ValueError, "look 'sepia' is not a look: use one of vivid, warm, film."):
            effects.look("sepia")


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Speed(unittest.TestCase):
    def test_a_frame_with_a_push_a_hit_a_title_and_captions_draws_in_under_a_second(self):
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
        graph, files = effects.join([effects.slow_push(1.0, 3.0), effects.punch_in(2.0, 0.5), text.burn(path)],
                                    "src", "outv", 1)
        started = time.time()
        run = subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=30:d=3",
                              "-filter_complex", "[0:v]trim=start_frame=63:end_frame=64,format=yuv420p[src];" + graph,
                              "-map", "[outv]", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                              "-"], capture_output=True, timeout=60)
        took = time.time() - started
        self.assertEqual(run.returncode, 0, run.stderr[-500:])
        self.assertEqual(len(run.stdout), 1080 * 1920 * 3 // 2)
        self.assertLess(took, 1.0)


def sticker_frames(fragment, numbers, seconds=2):
    """{n: frame n (1080x1920, grey bytes) of a grey frame of seconds with the fragment drawn on it}."""
    import numpy as np
    graph, files = effects.join([fragment], "src", "outv", 1)
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=0x808080:s=1080x1920:r=30:d=%s" % seconds]
    for f in files:
        cmd += ["-loop", "1", "-framerate", "30", "-i", f]
    pick = "+".join("eq(n\\,%d)" % n for n in numbers)
    cmd += ["-filter_complex", "[0:v]format=yuv420p[src];%s;[outv]select='%s'[o]" % (graph, pick), "-map", "[o]",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    run = subprocess.run(cmd, capture_output=True, timeout=120)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode("utf-8", "replace")[-2000:])
    raw = np.frombuffer(run.stdout, np.uint8).reshape(-1, 1920, 1080)
    return dict(zip(sorted(numbers), raw))


def drawn_box(frame):
    """(x0, y0, x1, y1) of the pixels that are not the grey ground, or None."""
    import numpy as np
    ys, xs = np.nonzero(np.abs(frame.astype(int) - 128) > 12)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Stickers(unittest.TestCase):
    """An orange 200x100 picture with a see-through round hole in its middle, in a 400x400 box at (340, 760)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.png = os.path.join(cls.tmp, "it's a sticker.png")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=black@0:s=200x100", "-vf",
                        "format=rgba,geq=r=255:g=140:b=0:a='255*gt(hypot(X-100,Y-50),30)'", "-frames:v", "1",
                        cls.png], check=True)
        cls.box = (340, 760, 400, 400)
        cls.frames = sticker_frames(effects.sticker(cls.png, 0.5, 1.0, cls.box),
                                    [14, 15, 18, 20, 23, 30, 40, 43, 44, 45])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def size(self, n):
        x0, y0, x1, y1 = drawn_box(self.frames[n])
        self.assertLess(abs((x0 + x1) / 2 - 540), 3, "centred across the box")
        self.assertLess(abs((y0 + y1) / 2 - 960), 3, "centred down the box")
        return x1 - x0, y1 - y0

    def test_it_fits_the_box_keeping_its_shape_and_pops_in_then_settles(self):
        self.assertEqual(self.size(30), (400, 200), "settled: 200x100 fitted into 400x400")
        w, h = self.size(15)
        self.assertLess(abs(w - 0.6 * 400), 4, "it starts at 0.6 times its size: %d" % w)
        biggest = max(self.size(n)[0] for n in (18, 20, 23))
        self.assertGreater(biggest, 1.05 * 400, "it grows past its size: %d" % biggest)
        self.assertEqual(self.size(23), (400, 200), "settled within 0.25 s")

    def test_its_see_through_hole_shows_the_frame_and_nothing_shows_outside_its_time(self):
        self.assertEqual(int(self.frames[30][960, 540]), 128, "the hole shows the grey frame")
        self.assertIsNone(drawn_box(self.frames[14]), "nothing before at")
        self.assertIsNone(drawn_box(self.frames[45]), "nothing after its length")

    def test_it_fades_out_over_its_last_moment(self):
        edge = (int(self.frames[n][880, 360]) for n in (40, 43, 44))
        held, fading, last = [abs(v - 128) for v in edge]
        self.assertGreater(held, 20, "held at full strength")
        self.assertTrue(held > fading > last > 0, (held, fading, last))

    def test_wrong_input_says_what_to_send(self):
        with self.assertRaisesRegex(ValueError, "is not a PNG picture: give the path of a .png file that exists"):
            effects.sticker(os.path.join(self.tmp, "none.png"), 1, 1, self.box)
        with self.assertRaisesRegex(ValueError, r"box \(900, 760, 400, 400\) runs off the 1080x1920 frame"):
            effects.sticker(self.png, 1, 1, (900, 760, 400, 400))
        with self.assertRaisesRegex(ValueError, "length 0.2 is not a sticker's length: use 0.4 to 60 seconds"):
            effects.sticker(self.png, 1, 0.2, self.box)


def heard(fragments, seconds=3, base="0.1*sin(2*PI*220*t)"):
    """The 48 kHz stereo samples of a steady tone (base, an expression in t) through the sound fragments, as an
    (n, 2) array of floats."""
    import numpy as np
    graph, _ = effects.join(fragments, "src", "outa", 0) if fragments else ("[src]anull[outa]", [])
    full = "aevalsrc=exprs='%s':s=48000:c=stereo:d=%s[src];%s" % (base, seconds, graph)
    run = subprocess.run([FFMPEG, "-v", "error", "-filter_complex", full, "-map", "[outa]", "-f", "f32le", "-"],
                         capture_output=True, timeout=60)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode("utf-8", "replace")[-2000:])
    return np.frombuffer(run.stdout, "<f4").reshape(-1, 2)


def centroid(x):
    """The middle of a stretch of sound's spectrum, in Hz, weighted by its energy."""
    import numpy as np
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return float(np.fft.rfftfreq(len(x), 1 / 48000) @ power / power.sum())


def loudest(x):
    """The loudest 400 ms of a sound, in LUFS, as FFmpeg's ebur128 meter reads it."""
    import re
    run = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-f", "f32le", "-ar", "48000", "-ac", "2", "-i", "-",
                          "-af", "apad=pad_dur=1,ebur128", "-f", "null", "-"], input=x.astype("<f4").tobytes(),
                         capture_output=True, timeout=60)
    return max(float(m) for m in re.findall(r"M:\s*(-?[\d.]+)", run.stderr.decode("utf-8", "replace")))


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Glass(unittest.TestCase):
    """A glass box at (100, 400, 880, 600) from 0.5 to 2.5 s on the busy picture, its shine sweeping from 0.75 to
    1.45 s; frames picked at 0.17, 0.5, 0.6, 0.7, 1.0, 1.2, 1.5, 2.0, 2.47 and 2.67 s."""
    BOX = (100, 400, 880, 600)
    PICKS = [5, 15, 18, 21, 30, 36, 45, 60, 74, 80]

    @classmethod
    def setUpClass(cls):
        size = (effects.FRAME_W, effects.FRAME_H)
        cls.got = {}
        for name, fragments in (("plain", []), ("light", [layouts.glass(cls.BOX, 0.5, 2.5)]),
                                ("calm", [layouts.glass(cls.BOX, 0.5, 2.5, shine=False)]),
                                ("dark", [layouts.glass(cls.BOX, 0.5, 2.5, "dark", shine=False)])):
            every = frames(fragments, 3, size=size)
            cls.got[name] = {k: tuple(plane.copy() for plane in every[k]) for k in cls.PICKS}
            del every

    def mid(self):
        x, y, w, h = self.BOX
        return slice(y + 150, y + h - 100), slice(x + 100, x + w - 100)

    def test_outside_its_box_and_its_time_every_pixel_is_the_frames_own(self):
        x, y, w, h = self.BOX
        sigma, drop, _ = layouts.SHADOW["glass"]
        pad = 3 * sigma
        around = (x - pad, y - pad + drop, w + 2 * pad, h + 2 * pad)
        for k in self.PICKS:
            for name in ("light", "dark"):
                same_outside(self, self.got["plain"][k], self.got[name][k], (0, 0, 0, 0) if k in (5, 80) else around)
        self.assertIsNotNone(changed(self.got["plain"][60], self.got["light"][60]))

    def test_inside_it_the_picture_is_frosted_and_mixed_with_white_or_black(self):
        plain = self.got["plain"][60][0][self.mid()].astype(int)
        for name, lighter in (("calm", True), ("dark", False)):
            got = self.got[name][60][0][self.mid()].astype(int)
            self.assertLess(detail(got), detail(plain) * 0.02, "%s: nothing behind it can be read" % name)
            moved = got.mean() - plain.mean()
            self.assertGreater(moved if lighter else -moved, 15, "%s: %.1f" % (name, moved))

    def test_its_corners_are_round_and_its_edge_catches_the_light(self):
        x, y, w, h = self.BOX
        plain, got = self.got["plain"][60][0].astype(int), self.got["calm"][60][0].astype(int)
        corner = (slice(y, y + 6), slice(x, x + 6))
        self.assertGreater(detail(got[corner]), detail(plain[corner]) * 0.4, "the corner is outside its round shape")
        top = slice(x + w // 2 - 50, x + w // 2 + 50)
        self.assertGreater(got[y:y + 2, top].mean(), got[y + 6:y + 8, top].mean() + 10, "a light edge")

    def test_it_fades_in(self):
        full = abs(self.got["calm"][60][0][self.mid()].astype(int) - self.got["plain"][60][0][self.mid()]).mean()
        half = abs(self.got["calm"][18][0][self.mid()].astype(int) - self.got["plain"][18][0][self.mid()]).mean()
        self.assertTrue(0.3 < half / full < 0.7, "half way into its 0.2 s fade: %.2f" % (half / full))

    def test_the_shine_sweeps_across_once(self):
        x, y, w, h = self.BOX
        peaks = []
        for k in (30, 36):
            row = self.got["light"][k][0][y + h // 2, x:x + w].astype(int) - self.got["calm"][k][0][y + h // 2, x:x + w]
            self.assertGreater(row.max(), 20, "the band of light at frame %d" % k)
            peaks.append(int(row.argmax()))
        self.assertGreater(peaks[1], peaks[0] + 100, "it moves right: %s" % peaks)
        for k in (18, 45, 60):
            self.assertTrue((self.got["light"][k][0] == self.got["calm"][k][0]).all(), "no shine at frame %d" % k)

    def test_glass_refusals(self):
        with self.assertRaisesRegex(ValueError, "tint 'blue' is not a glass tint: use one of light, dark"):
            layouts.glass(self.BOX, 0, 1, "blue")
        with self.assertRaisesRegex(ValueError, "shine 'yes' is not True or False"):
            layouts.glass(self.BOX, 0, 1, shine="yes")
        with self.assertRaisesRegex(ValueError, "start 1 to end 1.2 is too short for a glass box: give it at least "
                                                "0.35 s"):
            layouts.glass(self.BOX, 1.0, 1.2)
        with self.assertRaisesRegex(ValueError, r"box \(100, 100, 40, 300\) is too small for a glass box: give it at "
                                                r"least 60 x 60 px"):
            layouts.glass((100, 100, 40, 300), 0, 1)
        with self.assertRaisesRegex(ValueError, r"box \(900, 100, 400, 300\) runs off the 1080x1920 frame"):
            layouts.glass((900, 100, 400, 300), 0, 1)


def ground_stats(frame, ground):
    """(box of the pixels that are not the ground or None, white pixels, black pixels, centre of the white ones or
    None, the brightest pixel) of a grey frame."""
    import numpy as np
    a = frame.astype(int)
    ys, xs = np.nonzero(a != ground)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else None
    wy, wx = np.nonzero(a > ground + 40)
    centre = (float(wx.mean()), float(wy.mean())) if len(wx) > 200 else None
    return box, int((a > 200).sum()), int((a < 60).sum()), centre, int(a.max())


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Flurries(unittest.TestCase):
    """A white disc and a black square, ten copies by each move in a 640 x 1000 box at (200, 400) from 0.5 s, and the
    white disc alone by each move in an 800 x 1300 box at (140, 300), on a grey frame."""
    BOX = (200, 400, 640, 1000)
    ONE = (140, 300, 800, 1300)

    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")
        cls.disc, cls.bar = os.path.join(cls.tmp, "disc.png"), os.path.join(cls.tmp, "square.png")
        for path, size, paint in ((cls.disc, "120x120", "r=255:g=255:b=255:a='255*lt(hypot(X-59.5,Y-59.5),58)'"),
                                  (cls.bar, "120x120", "r=0:g=0:b=0:a=255")):
            subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=black@0:s=%s" % size, "-vf",
                            "format=rgba,geq=%s" % paint, "-frames:v", "1", path], check=True)
        cls.many, cls.alone = {}, {}
        for move, (_, _, length) in effects.MOVES.items():
            end = round((0.5 + length) * 30)
            seconds = (end + 12) / 30
            got = sticker_frames(effects.flurry([cls.disc, cls.bar], 0.5, cls.BOX, move, 10),
                                 list(range(0, end + 12, 3)), seconds)
            ground = int(got[0][0, 0])
            cls.many[move] = {n: ground_stats(f, ground) for n, f in got.items()}
            got = sticker_frames(effects.flurry(cls.disc, 0.5, cls.ONE, move, 1), list(range(15, end + 1)), seconds)
            cls.alone[move] = {n: ground_stats(f, ground) for n, f in got.items()}
            del got
        cls.ground = ground
        got = sticker_frames(effects.flurry(cls.disc, 0.5, cls.ONE, "popper", 2), list(range(15, 30)), 1.2)
        x, y, w, h = cls.ONE
        cls.pair = []
        for f in got.values():
            seen = (f[y + int(h * 0.8):y + h] != ground).any(axis=0).astype(int)
            edges = list(np.nonzero(np.diff(np.concatenate(([0], seen, [0]))))[0])
            cls.pair += list(zip(edges[::2], edges[1::2]))

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_copy_stays_inside_its_box_and_shows_only_in_its_time(self):
        x, y, w, h = self.BOX
        for move, got in self.many.items():
            end = 0.5 + effects.MOVES[move][2]
            both = 0
            for n, (box, whites, blacks, _, _) in sorted(got.items()):
                if n / 30 < 0.5 - 1e-6 or n / 30 > end + 1e-6:
                    self.assertIsNone(box, "%s: nothing at %.2f s" % (move, n / 30))
                elif box:
                    self.assertTrue(x <= box[0] and y <= box[1] and box[2] <= x + w and box[3] <= y + h,
                                    "%s at %.2f s draws %s, outside %s" % (move, n / 30, box, self.BOX))
                    both += whites > 300 and blacks > 300
            self.assertGreater(both, 3, "%s shows both pictures" % move)

    def test_a_spinning_picture_covers_only_itself_and_leaves_no_trail(self):
        import numpy as np
        thin = os.path.join(self.tmp, "thin.png")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=12x120", "-vf", "format=rgba",
                        "-frames:v", "1", thin], check=True)
        most = 12 * 120 * effects.GROW[1] ** 2 + 2 * (12 + 120) * effects.GROW[1] * 2
        for move in ("burst", "popper"):
            got = sticker_frames(effects.flurry(thin, 0.2, self.ONE, move, 1, size=120), list(range(12, 40, 2)), 1.4)
            covered = [int((f.astype(int) < 60).sum()) for f in got.values()]
            self.assertGreater(max(covered), 600, "%s: the bar is drawn" % move)
            self.assertLess(max(covered), most, "%s: a 12 x 120 bar covers at most %d px in a frame, not the circle "
                                                "it sweeps as it spins: %s" % (move, most, covered))
        got = sticker_frames(effects.flurry(effects.CONFETTI, 0.2, self.ONE, "popper", 6), list(range(20, 40, 4)), 1.4)
        for n, f in got.items():
            self.assertLess(int((np.abs(f.astype(int) - self.ground) > 12).sum()), 6 * 20 * 50 * effects.GROW[1] ** 2,
                            "paper pieces stay paper-sized at frame %d" % n)

    def ways(self, move):
        """[(x, y)] of the lone disc's centre while it shows, and the box its centre keeps to."""
        shown = [self.alone[move][n][3] for n in sorted(self.alone[move]) if self.alone[move][n][3]]
        self.assertGreater(len(shown), 15, move)
        return shown, self.ONE

    def test_burst_flies_out_from_the_middle(self):
        shown, (x, y, w, h) = self.ways("burst")
        far = [math.hypot(cx - (x + w / 2), cy - (y + h / 2)) for cx, cy in shown]
        self.assertLess(far[0], 40, "it starts in the middle: %.0f" % far[0])
        self.assertGreater(far[len(far) // 2], 150, "half way it is far out: %.0f" % far[len(far) // 2])
        half = far[:len(far) // 2]
        self.assertTrue(all(b >= a - 2 for a, b in zip(half, half[1:])), "always outwards: %s" % half)

    def test_rain_falls_and_rise_floats_up_all_the_way(self):
        for move, down in (("rain", True), ("rise", False)):
            shown, (x, y, w, h) = self.ways(move)
            ys = [cy if down else -cy for _, cy in shown]
            self.assertTrue(all(b >= a - 1 for a, b in zip(ys, ys[1:])), "%s keeps going one way" % move)
            self.assertGreater(ys[-1] - ys[0], 0.5 * h, "%s goes most of the box: %.0f" % (move, ys[-1] - ys[0]))

    def test_fountain_shoots_up_from_the_bottom_and_falls_back(self):
        shown, (x, y, w, h) = self.ways("fountain")
        ys = [cy for _, cy in shown]
        top = ys.index(min(ys))
        self.assertTrue(len(ys) / 4 < top < len(ys) * 3 / 4, "highest in its middle: %d of %d" % (top, len(ys)))
        self.assertGreater(ys[0], y + h * 0.75, "it starts at the bottom: %.0f" % ys[0])
        self.assertGreater(ys[-1], min(ys) + 0.5 * (ys[0] - min(ys)), "it comes back down: %.0f" % ys[-1])
        self.assertLess(min(ys), ys[0] - 0.4 * h, "it goes up high: %.0f" % min(ys))

    def test_bounce_drops_bounces_lower_and_lies_on_the_bottom(self):
        shown, (x, y, w, h) = self.ways("bounce")
        ys = [cy for _, cy in shown]
        floor = max(ys)
        self.assertGreater(floor, y + h * 0.8, "it lands at the bottom of its box: %.0f" % floor)
        landed = next(k for k, v in enumerate(ys) if v >= floor - 0.05 * h)
        again = min(ys[landed:])
        self.assertLess(again, floor - 0.1 * h, "it bounces up again: %.0f" % again)
        self.assertGreater(again, ys[0] + 0.3 * (floor - ys[0]), "lower than it fell from: %.0f" % again)
        self.assertTrue(all(abs(v - floor) < 2 for v in ys[-10:]), "it lies there: %s" % ys[-10:])

    def test_scatter_pops_up_and_stays_put(self):
        shown, _ = self.ways("scatter")
        self.assertLess(max(abs(cx - shown[0][0]) + abs(cy - shown[0][1]) for cx, cy in shown), 3)
        counts = [self.many["scatter"][n][1] for n in sorted(self.many["scatter"])]
        start, length = 15, effects.MOVES["scatter"][2] * 30
        early, late = (self.many["scatter"][3 * round((start + length * s) / 3)][1] for s in (0.12, 0.6))
        self.assertGreater(late, early * 2, "one by one: %d then %d white pixels (%s)" % (early, late, counts))

    def test_popper_shoots_up_and_in_from_a_bottom_corner_and_drifts_down(self):
        shown, (x, y, w, h) = self.ways("popper")
        xs, ys = [cx for cx, _ in shown], [cy for _, cy in shown]
        self.assertLess(xs[0], x + w * 0.35, "from the left corner: %.0f" % xs[0])
        self.assertGreater(ys[0], y + h * 0.7, "from the bottom: %.0f" % ys[0])
        self.assertGreater(xs[len(xs) // 2], xs[0] + 50, "inwards: %.0f to %.0f" % (xs[0], xs[len(xs) // 2]))
        top = min(ys)
        self.assertLess(ys.index(top), len(ys) * 0.6, "up early")
        self.assertGreater(ys[-1], top + 0.3 * (ys[0] - top), "then down: %.0f after %.0f" % (ys[-1], top))
        new = [(a + b) / 2 for a, b in self.pair if b - a < 60]
        self.assertTrue(any(c < x + w * 0.3 for c in new) and any(c > x + w * 0.7 for c in new),
                        "a copy just shot shows near each bottom corner: %s" % new)

    def test_copies_pop_or_fade_in_and_fade_out(self):
        rain, burst = self.alone["rain"], self.alone["burst"]
        self.assertLess(rain[16][4], rain[20][4] - 40, "rain fades in: %d then %d" % (rain[16][4], rain[20][4]))
        self.assertLess(burst[15][1], burst[21][1] * 0.3, "burst pops in: %d then %d" % (burst[15][1], burst[21][1]))
        held, fading = burst[45][4], burst[53][4]
        self.assertTrue(held - 20 > fading > self.ground + 10, "it fades out: %d then %d" % (held, fading))

    def test_a_frame_is_the_same_from_any_start(self):
        for move in ("popper", "bounce", "scatter"):
            fragment = effects.flurry([self.disc, self.bar], 0.3, (60, 40, 520, 300), move, 12, 1.2, size=60)
            full = frames([fragment], 2)
            for k in (17, 26):
                late = frames([fragment], 2, first=k)[0]
                for plane, (a, b) in enumerate(zip(full[k], late)):
                    self.assertTrue((a == b).all(), "%s frame %d plane %d" % (move, k, plane))

    def test_paper_pieces_come_in_six_colours_and_flutter(self):
        fragment = effects.flurry(effects.CONFETTI, 0.5, self.ONE, "rise", 12)
        self.assertEqual(fragment.files, [effects.confetti_png(c) for c in effects.CONFETTI_COLOURS])
        for colour, png in zip(effects.CONFETTI_COLOURS, fragment.files):
            rgba = subprocess.run([FFMPEG, "-v", "error", "-i", png, "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                                  capture_output=True, check=True).stdout
            self.assertEqual(rgba[:4], bytes.fromhex(colour[1:]) + b"\xff", colour)
        import numpy as np
        picks = list(range(24, 60))
        got = sticker_frames(effects.flurry(effects.CONFETTI, 0.5, self.ONE, "rise", 1), picks, 2.2)
        area = [int((np.abs(got[n].astype(int) - int(got[n][0, 0])) > 8).sum()) for n in picks]
        self.assertLess(min(area), max(area) * 0.5, "it turns over as it goes: %s" % area)

    def test_flurry_refusals(self):
        with self.assertRaisesRegex(ValueError, "move 'spin' is not a flurry's move: use one of burst, rain, fountain, "
                                                "bounce, rise, scatter, popper"):
            effects.flurry(self.disc, 0, self.ONE, "spin")
        for count in (0, 41, 2.5, True):
            with self.assertRaisesRegex(ValueError, "count .* is not a number of copies: use a whole number from 1 to "
                                                    "40"):
                effects.flurry(self.disc, 0, self.ONE, count=count)
        with self.assertRaisesRegex(ValueError, "length 7 is not a flurry's length: use 0.8 to 6 seconds, or leave it "
                                                "out for the rain move's 2.4 s"):
            effects.flurry(self.disc, 0, self.ONE, "rain", length=7)
        with self.assertRaisesRegex(ValueError, "size 500 is not a picture size: use 40 to 400 px"):
            effects.flurry(self.disc, 0, self.ONE, size=500)
        for pngs in ([], [self.disc] * 7, 5):
            with self.assertRaisesRegex(ValueError, "is not one to 6 pictures: give a PNG's path or a list of them, "
                                                    "confetti for paper pieces"):
                effects.flurry(pngs, 0, self.ONE)
        with self.assertRaisesRegex(ValueError, "is not a PNG picture"):
            effects.flurry([self.disc, os.path.join(self.tmp, "none.png")], 0, self.ONE)
        with self.assertRaisesRegex(ValueError, r"box \(100, 100, 200, 900\) is too small for these pictures to move "
                                                r"in: give it at least 262 x 262 px, or smaller pictures"):
            effects.flurry(self.disc, 0, (100, 100, 200, 900))
        with self.assertRaisesRegex(ValueError, r"box \(900, 100, 400, 900\) runs off the 1080x1920 frame"):
            effects.flurry(self.disc, 0, (900, 100, 400, 900))


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Moves(unittest.TestCase):
    def test_every_move_is_listed_named_in_thai_and_english_with_a_moving_sample_made_once(self):
        tmp = tempfile.mkdtemp()
        work, recording.WORK = recording.WORK, os.path.join(tmp, "work")
        try:
            listed = effects.moves()
            self.assertEqual([m["name"] for m in listed], list(effects.MOVES))
            for m in listed:
                self.assertTrue(re.search("[฀-๿]", m["th"]) and m["en"].strip(), m["name"])
                self.assertEqual(m["length"], effects.MOVES[m["name"]][2])
                with open(m["clip"], "rb") as f:
                    self.assertEqual(f.read(8)[4:], b"ftyp", "%s is an MP4" % m["name"])
                grey = subprocess.run([FFMPEG, "-v", "error", "-i", m["clip"], "-f", "rawvideo", "-pix_fmt", "gray",
                                       "-"], capture_output=True, check=True).stdout
                count = len(grey) // (270 * 480)
                self.assertEqual(len(grey), count * 270 * 480, "%s is 270 x 480" % m["name"])
                self.assertEqual(count, round((0.5 + m["length"]) * 30), "%s plays at 30 fps" % m["name"])
                first, middle = grey[:270 * 480], grey[count // 2 * 270 * 480:(count // 2 + 1) * 270 * 480]
                self.assertNotEqual(first, middle, "%s moves" % m["name"])
            made = {m["clip"]: os.path.getmtime(m["clip"]) for m in listed}
            self.assertEqual({m["clip"]: os.path.getmtime(m["clip"]) for m in effects.moves()}, made, "made once")
        finally:
            recording.WORK = work
            shutil.rmtree(tmp, ignore_errors=True)


class StudioStickers(unittest.TestCase):
    def test_every_sticker_in_the_folder_is_listed_see_through_and_named_in_thai_and_english(self):
        listed = effects.stickers()
        names = sorted(f[:-4] for f in os.listdir(effects.STICKERS) if f.endswith(".png"))
        self.assertEqual([s["name"] for s in listed], names, "every picture in stickers/ is listed, and only those")
        self.assertGreaterEqual(len(listed), 60)
        for s in listed:
            with open(s["png"], "rb") as f:
                head = f.read(26)
            self.assertEqual(head[:8], effects.PNG_SIGNATURE, s["name"])
            self.assertEqual(head[25], 6, "%s keeps its see-through parts (RGBA)" % s["name"])
            self.assertTrue(re.search("[\u0e00-\u0e7f]", s["th"]) and s["en"].strip(), s["name"])
        with open(os.path.join(effects.STICKERS, "LICENSE-Fluent-Emoji.txt"), encoding="utf-8") as f:
            self.assertIn("MIT License", f.read())
        self.assertEqual(effects.sticker_png("fire"), os.path.join(effects.STICKERS, "fire.png"))
        with self.assertRaisesRegex(ValueError, r"sticker 'logo' is not one of the studio's stickers: use a name "
                                                r"stickers\(\) lists, such as fire, money_bag, star\."):
            effects.sticker_png("logo")


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Sounds(unittest.TestCase):
    """Each sound added at 1 s to a quiet tone of 3 s, and alone on silence."""

    @classmethod
    def setUpClass(cls):
        cls.plain = heard([])
        cls.added = {kind: heard([effects.sound(kind, 1.0)]) for kind in effects.SOUNDS}
        cls.alone = {kind: heard([effects.sound(kind, 1.0)], base="0")[48000:] for kind in effects.SOUNDS}

    def test_a_sound_starts_at_its_time_and_every_other_sample_is_left_as_it_was(self):
        import numpy as np
        for kind, x in self.added.items():
            seconds = effects.SOUNDS[kind][0]
            self.assertEqual(x.shape, self.plain.shape, "%s: the clip's sound keeps its length" % kind)
            gone = x - self.plain
            self.assertEqual(np.abs(gone[:48000]).max(), 0, "%s: nothing before 1 s" % kind)
            first = int(np.nonzero(gone[:, 0])[0][0])
            self.assertTrue(48000 <= first <= 48002, "%s: it starts at 1 s, not at sample %d" % (kind, first))
            self.assertEqual(np.abs(gone[48000 + round(seconds * 48000) + 1:]).max(), 0,
                             "%s: nothing after its %s s" % (kind, seconds))
            self.assertEqual(np.abs(gone[:, 0] - gone[:, 1]).max(), 0, "%s: the same in both ears" % kind)

    def test_each_sound_starts_and_ends_at_silence_so_it_never_clicks(self):
        import numpy as np
        for kind, x in self.alone.items():
            n = round(effects.SOUNDS[kind][0] * 48000)
            peak = np.abs(x[:n, 0]).max()
            self.assertLess(abs(x[0, 0]), 1e-6, kind)
            self.assertLess(np.abs(x[:12, 0]).max(), peak / 3, "%s: it rises from silence, never jumps" % kind)
            self.assertLess(np.abs(x[n - 12:n, 0]).max(), peak / 100, "%s: it has fallen to silence" % kind)

    def test_each_sound_is_heard_and_sits_under_the_voice(self):
        for kind, x in self.alone.items():
            loud = loudest(x)
            self.assertTrue(-23 < loud < -15, "%s: its loudest 400 ms are %.1f LUFS, the voice's are -14"
                            % (kind, loud))

    def test_each_sound_is_its_own_kind(self):
        x = {kind: v[:, 0] for kind, v in self.alone.items()}
        self.assertLess(centroid(x["thump"][:9600]), 150, "a thump is deep")
        self.assertTrue(500 < centroid(x["pop"][:4800]) < 1200, "a pop is a small high bubble")
        self.assertGreater(centroid(x["ding"][:24000]), 1200, "a ding is a bell")
        self.assertGreater(abs(x["ding"][38400:40800]).max(), 0.01, "and it still rings at 0.8 s")
        start, middle, end = (centroid(x["whoosh"][k:k + 4800]) for k in (0, 9600, 19200))
        self.assertTrue(middle > 1.5 * start and middle > 1.5 * end,
                        "a whoosh brightens to its middle, then fades: %d %d %d Hz" % (start, middle, end))

    def test_level_makes_it_that_many_db_louder_or_quieter(self):
        import numpy as np
        quieter = heard([effects.sound("pop", 1.0, -6)], base="0")[48000:]
        self.assertLess(abs(np.abs(quieter).max() / np.abs(self.alone["pop"]).max() - 10 ** (-6 / 20)), 0.005)

    def test_a_sound_past_the_clips_end_is_cut_there(self):
        self.assertEqual(heard([effects.sound("ding", 2.5)]).shape, self.plain.shape)

    def test_wrong_input_says_what_to_send(self):
        with self.assertRaisesRegex(ValueError, "sound 'bang' is not a sound: use one of thump, whoosh, pop, ding"):
            effects.sound("bang", 1)
        with self.assertRaisesRegex(ValueError, "level 9 is not a sound's level: use -20 to 6 dB"):
            effects.sound("pop", 1, 9)
        with self.assertRaisesRegex(ValueError, "level 'loud' is not a number: give it in dB"):
            effects.sound("pop", 1, "loud")
        with self.assertRaisesRegex(ValueError, "at -1 is below 0"):
            effects.sound("pop", -1)


def levels(x):
    """The level of a 1 kHz tone in each millisecond (one whole wave), in dB."""
    import numpy as np
    rms = np.sqrt((x[:len(x) // 48 * 48, 0].astype(float) ** 2).reshape(-1, 48).mean(axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-9))


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Duck(unittest.TestCase):
    """A steady 1 kHz tone of 6 s ducked by 12 dB under three spans: two 0.4 s apart and one 2.1 s later."""

    TONE = "0.5*sin(2*PI*1000*t)"

    @classmethod
    def setUpClass(cls):
        cls.full = levels(heard([], 6, cls.TONE))
        cls.ducked = levels(heard([effects.duck([(4.5, 5.0), (1.0, 1.6), (2.0, 2.4)], -12)], 6, cls.TONE))

    def under(self, a, b):
        return self.ducked[round(a * 1000):round(b * 1000)] - self.full[round(a * 1000):round(b * 1000)]

    def test_it_is_down_by_db_while_someone_speaks_and_as_it_was_between(self):
        import numpy as np
        for a, b in ((1.0, 2.4), (4.5, 5.0)):
            self.assertLess(np.abs(self.under(a, b) + 12).max(), 0.05, "down 12 dB from %s to %s s" % (a, b))
        for a, b in ((0, 0.75), (2.9, 4.25), (5.5, 6)):
            self.assertLess(np.abs(self.under(a, b)).max(), 0.01, "as it was from %s to %s s" % (a, b))

    def test_it_goes_down_before_the_first_word_and_comes_back_after_the_last(self):
        self.assertLess(self.under(0.74, 0.75).max(), 0.01, "still up at 0.75 s")
        self.assertLess(abs(self.under(1.0, 1.001)[0] + 12), 0.05, "fully down when the span starts")
        self.assertLess(self.under(2.6, 2.601)[0], -3, "on its way back up at 2.6 s")

    def test_spans_less_than_a_breath_apart_stay_down_between_them(self):
        self.assertLess(abs(self.under(1.6, 2.0) + 12).max(), 0.05)

    def test_it_never_steps_so_it_never_clicks(self):
        import numpy as np
        self.assertLess(np.abs(np.diff(self.ducked)).max(), 0.2, "no step bigger than 0.2 dB from one ms to the next")

    def test_no_spans_leave_the_sound_as_it_was(self):
        import numpy as np
        self.assertEqual(np.abs(heard([effects.duck([])], 1, self.TONE) - heard([], 1, self.TONE)).max(), 0)

    def test_wrong_input_says_what_to_send(self):
        with self.assertRaisesRegex(ValueError, "db -50 is not a duck: use -40 to -1 dB"):
            effects.duck([(1, 2)], -50)
        with self.assertRaisesRegex(ValueError, "end 1 is not after start 2"):
            effects.duck([(2, 1)])
        with self.assertRaisesRegex(ValueError, r"span 0 \(1,\) is not \(start, end\)"):
            effects.duck([(1,)])
        with self.assertRaisesRegex(ValueError, "spans 3 is not a list"):
            effects.duck(3)
