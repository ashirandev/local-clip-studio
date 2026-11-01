"""The person cut out of the footage, checked on raw frames FFmpeg drew: python -m unittest discover -s tests"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import cutout, effects, layouts, recording, text  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
W, H = 1080, 1920
# the 'person' of the made-up recording: an ellipse 140 x 240 px moving across a 640 x 360 picture, at recording
# second T
SHAPE = "lt(hypot((X-(320+150*sin(1.3*T)))/70,(Y-200)/120),1)"


def make_mask(folder, name, seconds, start, expr=SHAPE, size=(640, 360)):
    """An FFV1 file of grey frames, white where expr (of X, Y and T, the recording second) is 1: frame k at
    recording second start + k / 30."""
    path = os.path.join(folder, name)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=%dx%d:r=30:d=%s,format=gray,"
                    "geq=lum='255*%s'" % (size + (seconds, expr.replace("T", "(T+%s)" % start))),
                    "-c:v", "ffv1", "-pix_fmt", "gray", path], check=True)
    return path


def grey_frames(path, picks):
    """{n: frame n of an FFV1 grey 1080x1920 file, as an array}."""
    import numpy as np
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    every = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
    return len(every), {n: every[n].copy() for n in picks if n < len(every)}


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class FrameMask(unittest.TestCase):
    """A mask of recording seconds 0.3 to 3.6 laid on a clip of two pieces, 0.5-2.5 and 3.0-4.0 s."""
    PIECES = [[0.5, 2.5], [3.0, 4.0]]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")
        self.path = make_mask(self.tmp, "mask.mkv", 3.3, 0.3)
        self.mask = {"path": self.path, "person": "main", "fps": 30, "size": [640, 360], "from": 0.3, "to": 3.6}

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_person_is_where_the_clip_shows_them_through_the_pieces_and_the_layout(self):
        import numpy as np
        import math
        layout = layouts.make("subject", (640, 360))
        sx, sy, sw, sh = layout["parts"][0]["src"]
        made = cutout.frame_mask(self.mask, self.PIECES, layout, 3.0)
        count, got = grey_frames(made, [0, 30, 59, 60, 75, 89])
        self.assertEqual(count, 90, "one frame per clip frame")
        k = W / sw
        for n, second in ((0, 0.5), (30, 1.5), (59, 2.4667), (60, 3.0), (75, 3.5)):
            ys, xs = np.nonzero(got[n] > 128)
            cx = 320 + 150 * math.sin(1.3 * second)
            seen_left, seen_right = (max(cx - 70, sx) - sx) * k, (min(cx + 70, sx + sw) - sx) * k
            if seen_right - seen_left < 20:
                continue
            self.assertLess(abs(xs.min() - seen_left), 6, "frame %d: left edge %d, the clip shows %.0f" % (
                n, xs.min(), seen_left))
            self.assertLess(abs(xs.max() + 1 - seen_right), 6, "frame %d: right edge %d, the clip shows %.0f" % (
                n, xs.max() + 1, seen_right))
            self.assertLess(abs((ys.min() + ys.max() + 1) / 2 - (200 - sy) * k), 6, "frame %d: its height" % n)
        self.assertEqual(int(got[89].max()), 0, "after recording second 3.6 the mask has no frames: no person")

    def test_the_card_and_the_circle_cut_the_mask_as_they_cut_the_picture(self):
        full = make_mask(self.tmp, "full.mkv", 2.0, 0, expr="1")
        mask = dict(self.mask, path=full, **{"from": 0.0, "to": 2.0})
        layout = layouts.make("card", (640, 360), [560, 280, 60, 60])
        parts = {p["name"]: p["dst"] for p in layout["parts"]}
        _, got = grey_frames(cutout.frame_mask(mask, [[0, 1]], layout, 1.0), [10])
        frame = got[10]
        cx, cy, cw, ch = parts["card"]
        dx, dy, d, _ = parts["circle"]
        self.assertEqual(int(frame[cy + ch // 2, cx + cw // 2]), 255, "the card shows the person")
        self.assertEqual(int(frame[cy + 2, cx + 2]), 0, "but not in its rounded corner")
        self.assertEqual(int(frame[dy + d // 2, dx + d // 2]), 255, "the circle shows the person")
        self.assertEqual(int(frame[dy + 4, dx + 4]), 0, "but not outside its round")
        self.assertEqual(int(frame[10, 10]), 0, "the frame's design shows no person")

    def test_a_talk_clip_shows_the_person_in_their_shots_and_on_the_card_of_everyone(self):
        from clipstudio import talk
        persons = [{"n": 1, "box": [120, 120, 80, 80]}, {"n": 2, "box": [440, 120, 80, 80]}]
        shots = [{"from": 0, "to": 1.0, "show": 1}, {"from": 1.0, "to": 2.0, "show": 0},
                 {"from": 2.0, "to": 3.0, "show": 2}]
        layout = talk.make((640, 360), persons, shots)
        # person 1, and a patch top left where the card of everyone has its corner
        one = make_mask(self.tmp, "one.mkv", 3.0, 0, expr="(lt(hypot(X-160,Y-200),60)+lt(X,100)*lt(Y,100))")
        mask = dict(self.mask, path=one, **{"from": 0.0, "to": 3.0})
        made = cutout.frame_mask(mask, [[0, 3]], layout, 3.0)
        _, got = grey_frames(made, [15, 45, 75])
        parts = {p["name"]: p for p in layout["parts"]}

        def at(part, x, y):
            (sx, sy, sw, sh), (dx, dy, dw, dh) = part["src"], part["dst"]
            return int(dy + (y - sy) * dh / sh), int(dx + (x - sx) * dw / sw)
        self.assertEqual(int(got[15][at(parts["person 1"], 160, 200)]), 255, "person 1's shot shows them")
        self.assertEqual(int(got[15][at(parts["person 1"], 160, 270)]), 0, "and only them")
        cx, cy, cw, ch = parts["everyone"]["dst"]
        self.assertEqual(int(got[45][at(parts["everyone"], 160, 200)]), 255, "the card of everyone shows them")
        self.assertEqual(int(got[45][cy + 12, cx + 12]), 255, "the patch reaches the card's corner")
        self.assertEqual(int(got[45][cy + 2, cx + 2]), 0, "but the corner is rounded off")
        outside = got[45].copy()
        outside[cy:cy + ch, cx:cx + cw] = 0
        self.assertEqual(int(outside.max()), 0, "around the card the frame shows no person")
        self.assertEqual(int(got[75].max()), 0, "person 2's shot does not show person 1")
        moved = talk.with_shot(shots, 2.0, 3.0, 1)
        self.assertNotEqual(cutout.frame_mask(mask, [[0, 3]], dict(layout, shots=moved), 3.0), made,
                            "new shots, a new frame mask")

    def test_a_talk_shot_given_a_place_moves_its_mask_with_the_picture(self):
        from clipstudio import talk
        persons = [{"n": 1, "box": [120, 120, 80, 80]}, {"n": 2, "box": [440, 120, 80, 80]}]
        plain = talk.make((640, 360), persons, [{"from": 0, "to": 3.0, "show": 1}])
        x, y = plain["parts"][0]["src"][:2]
        layout = dict(plain, shots=[{"from": 0, "to": 1.5, "show": 1}, {"from": 1.5, "to": 3.0, "show": 1, "crop": [x + 50, y]}])
        one = make_mask(self.tmp, "placed.mkv", 3.0, 0, expr="lt(hypot(X-160,Y-200),60)")
        mask = dict(self.mask, path=one, **{"from": 0.0, "to": 3.0})
        _, got = grey_frames(cutout.frame_mask(mask, [[0, 3]], layout, 3.0), [15, 60])
        row = int((200 - y) * 1920 / plain["parts"][0]["src"][3])
        self.assertEqual(int(got[15][row, 30]), 0, "left of the person, at the shot's own place")
        self.assertEqual(int(got[60][row, 30]), 255, "the frame moved right, so the person is further left in it")

    def test_in_a_talk_the_mask_is_whoever_speaks(self):
        """Tarm 10-30, four people talking with one kept in colour all through: "ถ้าผมพูดก็ไฮไลท์ผม"."""
        from clipstudio import talk
        persons = [{"n": 1, "box": [120, 120, 80, 80]}, {"n": 2, "box": [440, 120, 80, 80]}]
        shots = [{"from": 0, "to": 1.0, "show": 1}, {"from": 1.0, "to": 2.0, "show": 0},
                 {"from": 2.0, "to": 3.0, "show": 2}]
        layout = talk.make((640, 360), persons, shots)
        one = make_mask(self.tmp, "one.mkv", 3.0, 0, expr="lt(hypot(X-160,Y-200),60)")
        everywhere = make_mask(self.tmp, "two.mkv", 3.0, 0, expr="1")
        mask = dict(self.mask, path=one, each=[one, everywhere], **{"from": 0.0, "to": 3.0})
        made = cutout.speaking_mask(mask, [[0, 3]], layout, 3.0)
        _, got = grey_frames(made, [15, 45, 75])
        parts = {p["name"]: p for p in layout["parts"]}
        (sx, sy, sw, sh), (dx, dy, dw, dh) = parts["person 1"]["src"], parts["person 1"]["dst"]
        at = int(dy + (270 - sy) * dh / sh), int(dx + (160 - sx) * dw / sw)
        self.assertEqual(int(got[15][int(dy + (200 - sy) * dh / sh), at[1]]), 255, "person 1 speaks in their shot")
        self.assertEqual(int(got[15][at]), 0, "and person 2, whose mask fills the picture, is not lit then")
        cx, cy, cw, ch = parts["everyone"]["dst"]
        self.assertEqual(int(got[45][cy + ch // 2, cx + cw // 2]), 255, "on the card of everyone, everyone")
        self.assertEqual(int(got[75][H // 2, W // 2]), 255, "person 2 speaks in theirs")
        self.assertEqual(cutout.speaking_mask(dict(mask, each=None), [[0, 3]], layout, 3.0),
                         cutout.frame_mask(mask, [[0, 3]], layout, 3.0), "without each one's own, the one person")
        self.assertEqual(cutout.speaking_mask(dict(mask, each=[None, None]), [[0, 3]], layout, 3.0),
                         cutout.frame_mask(mask, [[0, 3]], layout, 3.0))

    def test_voices_that_are_not_told_apart_leave_everyone_in_colour(self):
        """Tarm 10-31, EP0: the colour stayed on the man in glasses while another spoke."""
        from clipstudio import talk
        persons = [{"n": 1, "box": [120, 120, 80, 80]}, {"n": 2, "box": [440, 120, 80, 80]}]
        shots = [{"from": 0, "to": 1.5, "show": 1}, {"from": 1.5, "to": 3.0, "show": 2}]
        layout = talk.make((640, 360), persons, shots)
        one = make_mask(self.tmp, "one.mkv", 3.0, 0, expr="lt(hypot(X-160,Y-200),60)")
        everyone = make_mask(self.tmp, "all.mkv", 3.0, 0, expr="1")
        mask = dict(self.mask, path=one, each=[one, one], everyone=everyone, **{"from": 0.0, "to": 3.0})
        made = cutout.speaking_mask(mask, [[0, 3]], layout, 3.0)
        self.assertEqual(made, cutout.frame_mask(dict(mask, path=everyone), [[0, 3]], layout, 3.0))
        told = cutout.speaking_mask(mask, [[0, 3]], layout, 3.0, told_apart=True)
        self.assertNotEqual(told, made, "voices told apart still follow who speaks")
        self.assertEqual(cutout.speaking_mask(dict(mask, everyone=None), [[0, 3]], layout, 3.0), told,
                         "a mask made before keeps its look")

    def test_what_stands_behind_people_stands_behind_everyone_the_talk_shows(self):
        """Lane 5 11-01, EP0: the title stood behind the one SAM kept and across the other three in their shots."""
        from clipstudio import talk, tools
        persons = [{"n": 1, "box": [120, 120, 80, 80]}, {"n": 2, "box": [440, 120, 80, 80]}]
        shots = [{"from": 0, "to": 1.5, "show": 1}, {"from": 1.5, "to": 3.0, "show": 2}]
        layout = talk.make((640, 360), persons, shots)
        one = make_mask(self.tmp, "one.mkv", 3.0, 0, expr="lt(hypot(X-160,Y-200),60)")
        both = make_mask(self.tmp, "both.mkv", 3.0, 0,
                         expr="(lt(hypot(X-160,Y-200),60)+lt(hypot(X-480,Y-200),60))")
        mask = dict(self.mask, path=one, each=[one, one], everyone=both, **{"from": 0.0, "to": 3.0})
        _, lone = grey_frames(cutout.frame_mask(mask, [[0, 3]], layout, 3.0), [15, 75])
        self.assertEqual(int(lone[75].max()), 0, "the one person is not in the other's shot")
        made = cutout.people_mask(mask, [[0, 3]], layout, 3.0)
        _, got = grey_frames(made, [15, 75])
        self.assertEqual(int(got[15].max()), 255, "the first person in their shot")
        self.assertEqual(int(got[75].max()), 255, "and the second in theirs")
        self.assertEqual(cutout.people_mask(dict(mask, everyone=None), [[0, 3]], layout, 3.0),
                         cutout.frame_mask(mask, [[0, 3]], layout, 3.0), "a mask with one person keeps them")
        state = {"mask": mask, "clip": {"pieces": [[0, 3]], "layout": layout}, "length": 3.0, "pushes": []}
        seen = tools._person_frames(state, 0, 3.0)
        self.assertTrue(seen[tools.PERSON_STEP * 2:].any(), "the title is measured against the mask it is drawn with")

    def test_the_render_goes_by_the_clips_voices(self):
        from unittest import mock
        from clipstudio import render
        state = {"cutouts": [{"look": "grey", "start": 0.0, "end": 1.0}], "titles": [], "hits": [], "mask": {"path": "p"},
                 "clip": {"pieces": [[0, 1]], "layout": {}, "voices": None}, "length": 1.0, "pushes": [],
                 "captions": None, "channel": None}
        with mock.patch.object(render.cutout, "frame_mask", return_value="m"), \
                mock.patch.object(render.cutout, "speaking_mask", return_value="s") as speaking:
            render._cut_out(state, "clean")
            self.assertIs(speaking.call_args.kwargs["told_apart"], False)
            state["clip"]["voices"] = [{"from": 0, "to": 1, "who": 1}]
            render._cut_out(state, "clean")
            self.assertIs(speaking.call_args.kwargs["told_apart"], True)
            state["cutouts"][0]["everyone"] = True
            render._cut_out(state, "clean")
            self.assertIs(speaking.call_args.kwargs["told_apart"], False, "the person's words win over the voices")

    def test_a_slow_push_pushes_the_mask_with_the_picture(self):
        import numpy as np
        layout = layouts.make("subject", (640, 360))
        still = cutout.frame_mask(self.mask, [[1.0, 2.5]], layout, 1.5)
        pushed = cutout.frame_mask(self.mask, [[1.0, 2.5]], layout, 1.5,
                                   [{"start": 0.0, "end": 1.5, "zoom": 1.3, "centre": [540, 960]}])
        tall = []
        for path in (still, pushed):
            _, got = grey_frames(path, [42])
            ys, _xs = np.nonzero(got[42] > 128)
            tall.append(ys.max() + 1 - ys.min())
        self.assertGreater(tall[1], tall[0] * 1.15, "the person grows with the push: %s" % tall)

    def test_a_frame_that_follows_the_product_moves_the_mask_with_the_picture(self):
        import numpy as np
        still = make_mask(self.tmp, "left.mkv", 2.0, 0, expr="lt(abs(X-60),40)")
        lay = layouts.followed(layouts.make("subject", (640, 360)),
                               {"size": [202, 360], "shots": [{"from": 0, "x": 0, "y": 0},
                                                              {"from": 1.0, "x": 438, "y": 0}]}, "whole")
        made = cutout.frame_mask({"path": still, "person": None, "fps": 30, "size": [640, 360], "from": 0, "to": 2},
                                 [[0, 2]], lay, 2.0)
        _, got = grey_frames(made, [15, 54])
        xs = np.nonzero(got[15].max(axis=0) > 128)[0]
        self.assertLess(abs(xs.min() - 20 * W / 202.0), 8, "while the frame is at the left, the person shows there")
        self.assertEqual(int(got[54].max()), 0, "the frame has moved right, away from them")

    def test_it_is_made_once_and_again_when_the_mask_changes(self):
        layout = layouts.make("subject", (640, 360))
        made = cutout.frame_mask(self.mask, self.PIECES, layout, 3.0)
        when = os.path.getmtime(made)
        self.assertEqual(cutout.frame_mask(self.mask, self.PIECES, layout, 3.0), made)
        self.assertEqual(os.path.getmtime(made), when, "made once")
        time.sleep(0.05)
        make_mask(self.tmp, "mask.mkv", 3.3, 0.3, expr="lt(hypot(X-320,Y-180),50)")
        self.assertNotEqual(cutout.frame_mask(self.mask, self.PIECES, layout, 3.0), made, "a new mask, a new file")

    def test_a_mask_that_does_not_fit_says_what_to_send(self):
        layout = layouts.make("subject", (640, 360))
        for wrong, said in (({"size": [1280, 720]}, r"mask size \[1280, 720\] is not the recording's size "
                                                    r"\[640, 360\]"),
                            ({"path": os.path.join(self.tmp, "none.mkv")}, "is not a file: run cut_out again"),
                            ({"fps": 0}, "mask fps 0 is not a frame rate"),
                            ({"from": 2.0, "to": 1.0}, "mask from 2.0 to 1.0 is not a span")):
            with self.assertRaisesRegex(ValueError, said):
                cutout.frame_mask(dict(self.mask, **wrong), self.PIECES, layout, 3.0)


def drawn(fragments, picks, seconds=2.0, first=0):
    """{n: (Y, U, V)} of frame n of a busy colour 1080x1920 picture through the fragments, the cut-out masks opened
    as videos from clip second 0; the picture starts at frame first."""
    import numpy as np
    graph, files = effects.join(fragments, "src", "o", 1)
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i",
           "testsrc2=s=%dx%d:r=30:d=%s,format=yuv420p" % (W, H, seconds)]
    for f in files:
        cmd += ["-i", f] if f.endswith(".mkv") else ["-loop", "1", "-framerate", "30", "-i", f]
    pick = "+".join("eq(n\\,%d)" % (n - first) for n in picks)
    cmd += ["-filter_complex", "[0:v]trim=start_frame=%d[src];%s;[o]select='%s'[p]" % (first, graph, pick),
            "-map", "[p]", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    run = subprocess.run(cmd, capture_output=True, timeout=120)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode("utf-8", "replace")[-2000:])
    y, c = W * H, (W // 2) * (H // 2)
    raw = np.frombuffer(run.stdout, np.uint8).reshape(-1, y + 2 * c)
    return {n: (f[:y].reshape(H, W).astype(int), f[y:y + c].reshape(H // 2, W // 2).astype(int),
                f[y + c:].reshape(H // 2, W // 2).astype(int)) for n, f in zip(sorted(picks), raw)}


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Looks(unittest.TestCase):
    """A frame mask with a 'person' standing still at x 440-640, y 500-1400 of the frame, on a colourful picture."""
    BOX = (440, 500, 200, 900)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        x, y, w, h = cls.BOX
        cls.mask = os.path.join(cls.tmp, "frame_mask.mkv")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=%dx%d:r=30:d=2" % (W, H),
                        "-vf", "format=gray,drawbox=x=%d:y=%d:w=%d:h=%d:color=white:t=fill" % (x, y, w, h),
                        "-c:v", "ffv1", "-pix_fmt", "gray", cls.mask], check=True)
        cls.plain = drawn([], [12, 18, 30, 42, 57])
        cls.grey = drawn([cutout.grey(cls.mask, 0.5, 1.5)], [12, 18, 30, 42, 57])
        cls.edge = drawn([cutout.outline(cls.mask, 0.5, 1.5)], [12, 30, 57])
        st = text.STYLES["sticker"]
        cls.ass = os.path.join(cls.tmp, "behind.ass")
        with open(cls.ass, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.document([st], text.title("SALE SALE", 0.5, 1.5, (60, 800, 960, 300), st)))
        words = text._ass_filter(cls.ass, text.fonts_folder(st))
        cls.words = drawn([effects.Fragment("[in]%s[out]" % words, [])], [30])
        cls.behind = drawn([cutout.behind(words, cls.mask, [(0.5, 1.5)])], [12, 30, 57])
        cls.ground = os.path.join(cls.tmp, "ground.png")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x3050A0:s=%dx%d" % (W, H),
                        "-frames:v", "1", cls.ground], check=True)
        cls.backdrop = drawn([cutout.backdrop(cls.mask, cls.ground)], [3, 30, 57])
        cls.despilled = drawn([cutout.backdrop(cls.mask, cls.ground, despill=True)], [30])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def inside(self, plane, s=1):
        x, y, w, h = self.BOX
        return plane[(y + 20) // s:(y + h - 20) // s, (x + 20) // s:(x + w - 20) // s]

    def outside(self, plane, s=1):
        x, y, w, h = self.BOX
        return plane[(y + h + 40) // s:, :]

    def test_grey_drains_the_colour_around_the_person_and_keeps_theirs(self):
        _, u, v = self.grey[30]
        _, pu, pv = self.plain[30]
        self.assertLessEqual(abs(self.outside(u, 2) - 128).max(), 2, "around the person: no colour")
        self.assertLessEqual(abs(self.outside(v, 2) - 128).max(), 2)
        self.assertTrue((self.inside(u, 2) == self.inside(pu, 2)).all(), "the person keeps every colour")
        self.assertTrue((self.inside(v, 2) == self.inside(pv, 2)).all())
        for n in (12, 57):
            self.assertTrue(all((a == b).all() for a, b in zip(self.grey[n], self.plain[n])), "frame %d is its own" % n)

    def test_a_background_is_behind_the_person_through_the_whole_clip_and_they_stay_as_they_are(self):
        for n in (3, 30, 57):
            y, u, v = self.backdrop[n]
            py, pu, pv = self.plain[n] if n in self.plain else drawn([], [n])[n]
            self.assertTrue((self.inside(y) == self.inside(py)).all(), "frame %d: the person as they were" % n)
            self.assertTrue((self.inside(u, 2) == self.inside(pu, 2)).all())
            for plane, s in ((y, 1), (u, 2), (v, 2)):
                around = self.outside(plane, s)
                self.assertLessEqual(int(around.max() - around.min()), 2, "frame %d: the background's one colour "
                                                                          "around them, not the busy picture" % n)
        y, _, _ = self.despilled[30]
        self.assertFalse((self.inside(y) == self.inside(self.plain[30][0])).all(),
                         "off a green cloth the green is taken off the person's colours")
        self.assertTrue((self.outside(y) == self.outside(self.backdrop[30][0])).all())

    def test_grey_eases_in_and_out(self):
        full = abs(self.outside(self.plain[30][1], 2) - 128).mean()
        for n in (18, 42):
            half = abs(self.outside(self.grey[n][1], 2) - 128).mean()
            self.assertTrue(0.25 < half / full < 0.75, "frame %d, 0.1 s into a 0.2 s ease: %.2f of the colour"
                            % (n, half / full))

    def test_outline_is_a_white_edge_around_the_person_only(self):
        import numpy as np
        x, y, w, h = self.BOX
        lum = self.edge[30][0]
        white = lum > 220
        plain_white = self.plain[30][0] > 220
        added = white & ~plain_white
        ys, xs = np.nonzero(added)
        self.assertGreater(len(xs), 2000, "an edge is drawn")
        near = (xs >= x - 14) & (xs < x + w + 14) & (ys >= y - 14) & (ys < y + h + 14)
        self.assertEqual(int((~near).sum()), 0, "nothing far from the person")
        inner = (xs >= x + 2) & (xs < x + w - 2) & (ys >= y + 2) & (ys < y + h - 2)
        self.assertEqual(int(inner.sum()), 0, "nothing over the person")
        for n in (12, 57):
            self.assertTrue(all((a == b).all() for a, b in zip(self.edge[n], self.plain[n])), "frame %d is its own" % n)

    def test_words_behind_the_person_show_around_them_and_not_over_them(self):
        x, y, w, h = self.BOX
        lum, words, plain = self.behind[30][0], self.words[30][0], self.plain[30][0]
        drawn_words = words != plain
        self.assertGreater(int(drawn_words[:, :x].sum() + drawn_words[:, x + w:].sum()), 3000, "the words show")
        self.assertGreater(int(drawn_words[y:y + h, x:x + w].sum()), 3000, "and some of them would be on the person")
        self.assertTrue((lum[y + 2:y + h - 2, x + 2:x + w - 2] == plain[y + 2:y + h - 2, x + 2:x + w - 2]).all(),
                        "the person stands in front of them")
        self.assertTrue((lum[:, :x - 2] == words[:, :x - 2]).all(), "around the person the words are as drawn")
        for n in (12, 57):
            self.assertTrue(all((a == b).all() for a, b in zip(self.behind[n], self.plain[n])), "frame %d" % n)

    def test_a_frame_is_the_same_from_any_start(self):
        for fragment in (cutout.grey(self.mask, 0.5, 1.5), cutout.outline(self.mask, 0.5, 1.5)):
            full = drawn([fragment], [27])[27]
            late = drawn([fragment], [27], first=20)[27]
            self.assertTrue(all((a == b).all() for a, b in zip(full, late)))

    def test_cut_out_refusals(self):
        with self.assertRaisesRegex(ValueError, "start 1 to end 1.3 is too short for a cut-out: give it at least "
                                                "0.4 s"):
            cutout.grey(self.mask, 1.0, 1.3)
        with self.assertRaisesRegex(ValueError, "width 40 is not an outline's width: use 2 to 30 px, such as 8"):
            cutout.outline(self.mask, 0.5, 1.5, 40)
        with self.assertRaisesRegex(ValueError, r"windows \[\] has no times"):
            cutout.behind("null", self.mask, [])
