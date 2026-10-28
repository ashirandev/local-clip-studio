"""The layouts as drawn: a box of the recording shows where to_frame() says, whatever is drawn around the parts, and
no place for words is on the face. Checked on raw frames FFmpeg drew (no encoding):
python -m unittest discover -s tests"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import layouts, recording  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
SIZE = (1920, 1080)
FACE = [1693.9, 428.7, 92.3, 113.8]


def draw(lay, marker):
    """The 1080x1920 frame of layout lay drawn from a grey recording with a magenta box at marker, as an (h, w, 3)
    array."""
    import numpy as np
    x, y, w, h = [int(round(v)) for v in marker]
    source = ("color=gray:s=%dx%d:r=30:d=1,drawbox=x=%d:y=%d:w=%d:h=%d:color=magenta:t=fill,format=yuv420p"
              % (SIZE + (x, y, w, h)))
    graph, files = layouts.graph(lay, "src", "out", 1)
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", source]
    for f in files:
        cmd += ["-loop", "1", "-framerate", "30", "-i", f]
    cmd += ["-filter_complex", "[0:v]null[src];" + graph, "-map", "[out]", "-frames:v", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-"]
    run = subprocess.run(cmd, capture_output=True, timeout=120)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode("utf-8", "replace")[-2000:])
    return np.frombuffer(run.stdout, np.uint8).reshape(1920, 1080, 3)


def magenta(frame, region):
    """(x0, y0, x1, y1) of the magenta pixels inside region (x, y, w, h) of the frame, or None."""
    import numpy as np
    x, y, w, h = [int(round(v)) for v in region]
    part = frame[y:y + h, x:x + w].astype(int)
    ys, xs = np.nonzero((part[:, :, 0] > 170) & (part[:, :, 1] < 90) & (part[:, :, 2] > 170))
    if not len(xs):
        return None
    return x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1


def overlap(a, b):
    return min(a[0] + a[2], b[0] + b[2]) > max(a[0], b[0]) and min(a[1] + a[3], b[1] + b[3]) > max(a[1], b[1])


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg, and numpy for the pixel checks")
class Drawn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a_box_of_the_recording_shows_where_to_frame_says_in_every_part(self):
        for kind in layouts.KINDS:
            lay = layouts.make(kind, SIZE, FACE)
            for part in lay["parts"]:
                sx, sy, sw, sh = part["src"]
                marker = [round(v) for v in (sx + sw * 0.4, sy + sh * 0.4, sw * 0.2, sh * 0.2)]
                (want,) = [b for b in layouts.to_frame(lay, marker) if overlap(b, part["dst"])]
                r = layouts.CARD_RADIUS
                dx, dy, dw, dh = part["dst"]
                found = magenta(draw(lay, marker), [dx + r, dy + r, dw - 2 * r, dh - 2 * r])
                self.assertIsNotNone(found, (kind, part["name"]))
                got = list(found)
                want = [want[0], want[1], want[0] + want[2], want[1] + want[3]]
                reach = 1.2 * max(1, dw / sw) + 2
                for k in range(4):
                    self.assertLess(abs(got[k] - want[k]), reach, "%s %s: drawn at %s, to_frame says %s; the colour "
                                    "planes blur one pixel of the recording and two of the frame, %.1f px here"
                                    % (kind, part["name"], got, [round(v, 1) for v in want], reach))

    def test_the_card_has_rounded_corners_a_shadow_and_a_ring(self):
        lay = layouts.make("card", SIZE, FACE)
        frame = draw(lay, [10, 10, 4, 4]).astype(int)
        x, y, w, h = [p for p in lay["parts"] if p["name"] == "card"][0]["dst"]
        grey = frame[y + h // 2, x + w // 2]
        self.assertLess(abs(grey - 128).max(), 6, "the card's middle is the recording's grey: %s" % grey)
        corner = frame[y + 2, x + 2]
        self.assertGreater(abs(corner - grey).max(), 20, "the corner is cut round, so the background shows there")
        below, beside = frame[y + h + 12, x + w // 2], frame[y + h // 2, x - 20]
        lower = frame[y + h + 60, x + w // 2]
        self.assertLess(below.mean(), lower.mean() - 10, "just under the card is darker than 60 px lower, though "
                        "the background darkens toward the bottom: the card's shadow (%s vs %s)" % (below, lower))
        self.assertTrue(beside.mean() < grey.mean(), "the background beside the card is darker than the card")
        self.assertLess(frame[100, 540].mean(), frame[450, 540].mean() - 20, "the background darkens toward the top, "
                        "where the phone's own bar and the title sit")
        cx, cy, d, _ = [p for p in lay["parts"] if p["name"] == "circle"][0]["dst"]
        self.assertGreater(frame[cy - layouts.RING // 2, cx + d // 2].min(), 235, "the ring above the circle is white")

    def test_the_stack_has_a_dark_seam_between_its_panels(self):
        lay = layouts.make("stack", SIZE, FACE)
        frame = draw(lay, [10, 10, 4, 4]).astype(int)
        top, bottom = [p["dst"] for p in lay["parts"]]
        seam = frame[top[1] + top[3]:bottom[1], 100:980]
        self.assertGreater(seam.shape[0], 0, "there is a seam")
        self.assertLess(seam.max(), 40, "the seam is dark")
        self.assertLess(frame[bottom[1] + 4, 540].mean(), frame[bottom[1] + 120, 540].mean() - 10,
                        "the face panel casts a shadow onto the game")


def moving(lay, t, pushes=(), source="testsrc2=s=%dx%d:r=30:d=3" % SIZE):
    """The 1080x1920 frame at second t of layout lay drawn from a moving test picture, with the slow pushes."""
    import numpy as np
    graph, files = layouts.graph(lay, "src", "out", 1, pushes)
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", source]
    for f in files:
        cmd += ["-loop", "1", "-framerate", "30", "-i", f]
    cmd += ["-filter_complex", "[0:v]trim=start_frame=%d[src];%s" % (round(t * 30), graph), "-map", "[out]",
            "-fps_mode", "passthrough", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    run = subprocess.run(cmd, capture_output=True, timeout=120)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode("utf-8", "replace")[-2000:])
    return np.frombuffer(run.stdout, np.uint8).reshape(1920, 1080, 3)


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg, and numpy for the pixel checks")
class Pushed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a_slow_push_moves_only_the_main_parts_picture_and_only_while_it_runs(self):
        import numpy as np
        for kind, main, centre in (("card", "card", [540, 1100]), ("card", "card", [540, 300]),
                                   ("stack", "bottom", [700, 1300])):
            lay = layouts.make(kind, SIZE, FACE)
            x, y, w, h = [p["dst"] for p in lay["parts"] if p["name"] == main][0]
            pushes = [{"start": 0.5, "end": 1.5, "zoom": 1.15, "centre": centre}]
            for t in (0.4, 1.5):
                self.assertTrue((moving(lay, t) == moving(lay, t, pushes)).all(), (kind, centre, t))
            ys, xs = np.nonzero((moving(lay, 1.0) != moving(lay, 1.0, pushes)).any(axis=2))
            self.assertGreater(len(xs), w * h * 0.1, "the main part's picture pushed in (the test picture's flat "
                               "bars do not change): %s %s" % (kind, centre))
            self.assertTrue(x <= xs.min() and xs.max() < x + w and y <= ys.min() and ys.max() < y + h,
                            "%s %s: only the %s %s changed, not %s" % (kind, centre, main, [x, y, w, h],
                                                                      [xs.min(), ys.min(), xs.max(), ys.max()]))

    def test_the_push_holds_still_the_point_it_was_given_on_the_frame(self):
        import numpy as np

        def weighed(frame, box):
            """(centre x, centre y, area) of the magenta in box, each pixel weighed by how magenta it is."""
            x, y, w, h = box
            rgb = frame[y:y + h, x:x + w].astype(float)
            weight = np.clip((rgb[:, :, 0] - rgb[:, :, 1]) / 200, 0, 1)
            ys, xs = np.nonzero(weight)
            wt = weight[ys, xs]
            return (xs * wt).sum() / wt.sum() + x, (ys * wt).sum() / wt.sum() + y, wt.sum()

        for kind, main in (("card", "card"), ("stack", "bottom")):
            lay = layouts.make(kind, SIZE, FACE)
            part = [p for p in lay["parts"] if p["name"] == main][0]
            sx, sy, sw, sh = part["src"]
            marker = [round(v) for v in (sx + sw * 0.3, sy + sh * 0.6, sw * 0.1, sh * 0.1)]
            source = ("color=gray:s=%dx%d:r=30:d=3,drawbox=x=%d:y=%d:w=%d:h=%d:color=magenta:t=fill,format=yuv420p"
                      % (SIZE + tuple(marker)))
            (shown,) = [b for b in layouts.to_frame(lay, marker) if overlap(b, part["dst"])]
            centre = [shown[0] + shown[2] / 2, shown[1] + shown[3] / 2]
            pushes = [{"start": 0.5, "end": 1.5, "zoom": 1.15, "centre": centre}]
            plain = weighed(moving(lay, 1.0, source=source), part["dst"])
            pushed = weighed(moving(lay, 1.0, pushes, source), part["dst"])
            w, h = part["dst"][2:]
            grown = (w + 2 * round(w / 2 * 0.075)) / w * (h + 2 * round(h / 2 * 0.075)) / h
            for k in (0, 1):
                self.assertLess(abs(pushed[k] - plain[k]), 1.5, "%s: the marker under the centre stays put, give or "
                                "take the push's half pixel and the half-size colour planes it is found by: %s vs %s"
                                % (kind, pushed, plain))
            self.assertLess(abs(pushed[2] / plain[2] - grown), 0.02, "%s: halfway it is %.3f times the area, not "
                            "%.3f" % (kind, pushed[2] / plain[2], grown))

class Places(unittest.TestCase):
    def test_no_place_for_words_is_on_the_face(self):
        for kind in layouts.KINDS:
            lay = layouts.make(kind, SIZE, FACE)
            on_face = layouts.to_frame(lay, FACE)
            self.assertEqual(bool(on_face), kind != "subject", "%s shows the face: %s" % (kind, on_face))
            for name, box in list(lay["places"].items()) + [("captions", lay["captions"])]:
                for face in on_face:
                    self.assertFalse(overlap(box, face), "%s: the %s place %s is on the face %s"
                                     % (kind, name, box, [round(v) for v in face]))

    def test_the_face_layout_keeps_words_off_the_whole_head_and_above_the_apps_own_words(self):
        for size, face, what in ((SIZE, FACE, "a face camera in a wide recording"),
                                 ((1080, 1920), [474.7, 292.3, 345.1, 454.4], "a vertical recording, the face high"),
                                 ((1080, 1920), [400, 450, 280, 360], "a vertical recording, the hair just under "
                                  "the title's place"),
                                 ((1080, 1920), [400, 458, 280, 360], "a vertical recording, the chin just above "
                                  "the title's room"),
                                 ((1080, 1920), [380, 900, 300, 380], "a vertical recording, the face low"),
                                 ((1080, 1920), [380, 1300, 300, 380], "a vertical recording, the face at the bottom")):
            lay = layouts.make("face", size, face)
            (f,) = layouts.to_frame(lay, face)
            head = [f[0] - 0.15 * f[2], f[1] - 0.4 * f[3], f[2] * 1.3, f[3] * 1.4]
            self.assertEqual(len(lay["places"]), 1, what)
            boxes = list(lay["places"].items()) + [("captions", lay["captions"])]
            for name, box in boxes:
                self.assertFalse(overlap(box, head), "%s: the %s place %s is on the head %s"
                                 % (what, name, box, [round(v) for v in head]))
                self.assertLessEqual(box[1] + box[3], layouts.NAME_TOP, "%s: %s is on the channel's name"
                                     % (what, name))
                self.assertGreaterEqual(box[1], 200, "%s: %s is below the phone's own bar" % (what, name))
            (_, title), (_, captions) = boxes
            self.assertFalse(overlap(title, captions), what)
            self.assertGreaterEqual(title[3], 192, "%s: the title place holds two lines of the tallest style" % what)


# What Instagram and Facebook Reels draw over a 1080x1920 clip, measured on Tarm's iPhone screenshots (10-27,
# 1179x2556): both apps fill the screen to their own bottom bar and cut the sides; boxes in clip pixels.
COVERED = {
    "the status bar": [151, 56, 756, 34],
    "Instagram's back and + icons": [130, 186, 868, 48],
    "Facebook's Reels header": [117, 187, 254, 49],
    "Instagram's buttons": [923, 1098, 64, 752],
    "Facebook's buttons": [917, 1103, 84, 742],
    "Facebook's friend avatar": [94, 1552, 111, 95],
    "Facebook's page name and Follow": [84, 1683, 597, 108],
    "Instagram's name": [88, 1668, 270, 89],
    "the caption": [85, 1784, 735, 71],
}
SEEN_X = (52, 1028)
RECORDINGS = ((SIZE, FACE, "a face camera in a wide recording"),
              ((1080, 1920), [474.7, 292.3, 345.1, 454.4], "a vertical recording, the face high"),
              ((1080, 1920), [400, 450, 280, 360], "a vertical recording, the title under the chin"),
              ((1080, 1920), [380, 900, 300, 380], "a vertical recording, the face low"))


def shows(test, box, what, below=236, above=1552):
    """box is drawn whole on the phone: inside the sides it keeps, between the header and the avatar."""
    test.assertTrue(SEEN_X[0] <= box[0] and box[0] + box[2] <= SEEN_X[1], "%s %s: the phone cuts the sides at x %s"
                    % (what, [round(v) for v in box], SEEN_X))
    test.assertTrue(below <= box[1] and box[1] + box[3] <= above, "%s %s: covered at the top or the bottom"
                    % (what, [round(v) for v in box]))


class Phones(unittest.TestCase):
    def test_no_words_go_under_what_instagram_or_facebook_draw(self):
        for size, face, what in RECORDINGS:
            for kind in layouts.KINDS:
                lay = layouts.make(kind, size, face)
                for name, box in list(lay["places"].items()) + [("captions", lay["captions"])]:
                    shows(self, box, "%s, %s, the %s place" % (what, lay["kind"], name))
                    for ui, cover in COVERED.items():
                        self.assertFalse(overlap(box, cover), "%s, %s: the %s place %s is under %s %s"
                                         % (what, lay["kind"], name, box, ui, cover))
                    self.assertEqual(layouts.unsafe(box), [], (what, lay["kind"], name))
                self.assertIn("250", lay["safe"]["said"], "Muse is told where the phone covers the clip")

    def test_the_card_and_the_stacks_game_show_whole_on_the_phone(self):
        for crop in (None, [600, 0, 700, 1080], [0, 300, 1920, 500]):
            lay = layouts.make("card", SIZE, FACE, crop)
            (card,) = [p["dst"] for p in lay["parts"] if p["name"] == "card"]
            shows(self, card, "the card for crop %s, its round corners included" % (crop,))
        lay = layouts.make("stack", SIZE, FACE)
        (game,) = [p["dst"] for p in lay["parts"] if p["name"] == "bottom"]
        self.assertLessEqual(game[1] + game[3], 1552, "the game ends above the apps' name and caption: %s" % game)

    def test_the_cards_captions_cover_neither_the_picture_nor_the_face_and_suit_an_ad(self):
        for size, face, what in list(RECORDINGS) + [((1920, 1080), None, "no face"), ((2560, 1080), FACE, "ultrawide"),
                                                    ((1440, 1080), FACE, "4:3")]:
            for crop in (None, [600, 0, 700, 1080], [0, 300, 1920, 500]):
                lay = layouts.make("card", size, face, crop)
                if lay["kind"] != "card":
                    continue
                parts = {p["name"]: p["dst"] for p in lay["parts"]}
                band = lay["captions"]
                for name, box in parts.items():
                    self.assertFalse(overlap(band, box), "%s, crop %s: the captions %s cover the %s %s"
                                     % (what, crop, band, name, box))
                self.assertFalse(overlap(band, lay["places"]["top"]), what)
                self.assertLessEqual(band[1] + band[3], 1250, "%s: an ad keeps the bottom 670 px for its button" % what)
                if "circle" in parts:
                    self.assertGreaterEqual(parts["circle"][2], layouts.CIRCLE_SMALLEST, what)
                    self.assertLess(parts["circle"][1], band[1], "%s: the captions read under the face" % what)
                self.assertLess(band[1], parts["card"][1], what)

    def test_nothing_the_studio_places_reaches_the_row_of_the_channels_name(self):
        for size, face, what in list(RECORDINGS) + [((1920, 1080), None, "no face"), ((2560, 1080), FACE, "ultrawide"),
                                                    ((1440, 1080), FACE, "4:3"),
                                                    ((1080, 1920), [380, 1300, 300, 380], "the face at the bottom")]:
            for kind in layouts.KINDS:
                for crop in (None, [600, 0, 700, 1080], [0, 300, 1920, 500]):
                    try:
                        lay = layouts.make(kind, size, face, crop)
                    except ValueError:
                        continue
                    boxes = list(lay["places"].items()) + [("captions", lay["captions"])]
                    if lay["kind"] == "card":
                        boxes += [(p["name"], p["dst"]) for p in lay["parts"]]
                    for name, box in boxes:
                        self.assertLessEqual(box[1] + box[3], layouts.NAME_TOP, "%s, %s, crop %s: the %s %s"
                                             % (what, lay["kind"], crop, name, box))
        lay = layouts.make("card", SIZE, FACE)
        parts = {p["name"]: p["dst"] for p in lay["parts"]}
        self.assertEqual(parts["card"][3], 514, "a 16:9 screen with a face camera: the card gave 14 px to the name")
        self.assertEqual(parts["circle"][2], layouts.CIRCLE_SMALLEST, "and the face circle none")

    def test_a_subject_shows_whole_between_the_header_and_the_name(self):
        for crop in ([800, 300, 400, 500], [300, 400, 300, 300], [1400, 100, 350, 400]):
            lay = layouts.make("subject", SIZE, None, crop)
            (shown,) = layouts.to_frame(lay, crop)
            shows(self, shown, "the subject %s" % (crop,))
            self.assertEqual(lay["notes"], [], crop)
        for crop in ([100, 100, 1700, 300], [900, 0, 120, 1080]):
            lay = layouts.make("subject", SIZE, None, crop)
            self.assertIn("too big to show whole", " ".join(lay["notes"]), "a subject no 9:16 part of the recording "
                          "can hold says so: %s" % (crop,))

    def test_unsafe_says_what_covers_a_box(self):
        said = " ".join(layouts.unsafe([65, 1450, 950, 200]))
        self.assertIn("caption", said)
        self.assertIn("buttons", said)
        self.assertIn("sides", " ".join(layouts.unsafe([20, 600, 300, 100])))
        self.assertIn("top", " ".join(layouts.unsafe([65, 150, 300, 100])))
        self.assertEqual(layouts.unsafe(layouts.TOP), [])


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg, and numpy for the pixel checks")
class Band(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_under_the_stacks_game_the_same_moment_shows_blurred_and_darker(self):
        lay = layouts.make("stack", SIZE, FACE)
        (game,) = [p["dst"] for p in lay["parts"] if p["name"] == "bottom"]
        frame = moving(lay, 1.0).astype(int)
        band = frame[game[1] + game[3] + 40:1900, 40:1040]
        above = frame[game[1] + 100:game[1] + game[3] - 40, 40:1040]
        self.assertGreater(band.std(), 8, "the band is a picture, not a flat colour")
        self.assertLess(band.mean(), above.mean(), "the band is darker than the game, so the apps' words read on it")
        sharp = lambda a: abs(a[:, 1:] - a[:, :-1]).mean()
        self.assertLess(sharp(band), sharp(above) / 3, "and blurred, so it does not pull the eye")


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg, and numpy for the pixel checks")
class Moved(unittest.TestCase):
    """The safe zone is where the program puts things by itself; a part goes wherever the person asks."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def face_shows_in(self, lay, where):
        circle = [p["dst"] for p in lay["parts"] if p["name"] == "circle"][0]
        marker = [round(v) for v in (FACE[0] + 20, FACE[1] + 30, 50, 50)]
        found = magenta(draw(lay, marker), circle)
        self.assertIsNotNone(found, "%s: the face shows in the circle at %s" % (where, circle))
        frame = draw(lay, [10, 10, 4, 4])
        dx, dy, d, _ = circle
        self.assertGreater(frame[dy - layouts.RING // 2, dx + d // 2].min(), 235, "%s: with its white ring" % where)
        return circle

    def test_the_face_circle_goes_to_the_bottom_right_corner_when_asked(self):
        lay = layouts.make("card", SIZE, FACE)
        corner = [700, 1500, 340, 340]
        there = layouts.moved(lay, "circle", corner)
        circle = self.face_shows_in(there, "card")
        self.assertEqual(circle[2], circle[3], "it stays round")
        self.assertEqual(circle, [700, 1500, 340, 340])
        said = there["notes"][-1]
        self.assertIn("As asked", said)
        self.assertIn("caption", said)
        self.assertIn("buttons", said)
        self.assertNotEqual([p["dst"] for p in lay["parts"] if p["name"] == "circle"][0], circle,
                            "the layout it came from is left as it was")
        (shown,) = [b for b in layouts.to_frame(there, FACE) if overlap(b, circle)]
        self.assertTrue(circle[0] <= shown[0] and shown[0] + shown[2] <= circle[0] + circle[2])

    def test_a_circle_in_a_long_box_is_round_in_its_middle(self):
        there = layouts.moved(layouts.make("card", SIZE, FACE), "circle", [100, 300, 600, 300])
        self.assertEqual([p["dst"] for p in there["parts"] if p["name"] == "circle"][0], [250, 300, 300, 300])

    def test_the_game_full_screen_takes_the_face_in_a_corner(self):
        lay = layouts.make("subject", SIZE, FACE, crop=[0, 0, 1500, 1080])
        there = layouts.moved(lay, "circle", [80, 1150, 300, 300])
        self.face_shows_in(there, "subject")
        self.assertEqual(there["notes"][-1], "As asked, the circle is at [80, 1150, 300, 300].",
                         "inside the safe zone the note says only where")

    def test_a_moved_card_shows_the_same_picture_unstretched(self):
        lay = layouts.make("card", SIZE, FACE)
        there = layouts.moved(lay, "card", [60, 400, 960, 540])
        card = [p for p in there["parts"] if p["name"] == "card"][0]
        before = [p for p in lay["parts"] if p["name"] == "card"][0]
        self.assertEqual(card["dst"], [60, 400, 960, 540])
        self.assertLess(abs(card["src"][2] / card["src"][3] - 960 / 540), 0.01, "not stretched: %s" % card["src"])
        for k in (0, 1):
            self.assertLess(abs((card["src"][k] + card["src"][k + 2] / 2) - (before["src"][k] + before["src"][k + 2] / 2)),
                            1, "cut around the same middle")
        self.assertIn("Words in the top place would sit on it.", there["notes"][-1], "the places it covers are named")
        self.assertIn("The captions would cover it", there["notes"][-1], "and the captions it would sit under")

    def test_only_a_box_the_frame_cannot_hold_is_refused(self):
        lay = layouts.make("card", SIZE, FACE)
        with self.assertRaisesRegex(ValueError, r"box \(900, 1800, 300, 300\) runs off the 1080x1920 frame"):
            layouts.moved(lay, "circle", (900, 1800, 300, 300))
        with self.assertRaisesRegex(ValueError, r"box \(10, 10, 60, 60\) is too small to show a part"):
            layouts.moved(lay, "circle", (10, 10, 60, 60))
        with self.assertRaisesRegex(ValueError, "part 'logo' is not in the card layout: use one of card, circle"):
            layouts.moved(lay, "logo", (10, 10, 300, 300))
        with self.assertRaisesRegex(ValueError, "The subject layout cannot take a face circle without a face"):
            layouts.moved(layouts.make("subject", SIZE), "circle", (10, 300, 300, 300))
        self.assertTrue(layouts.moved(lay, "circle", (0, 1580, 340, 340)), "a corner under the caption is fine")



@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Shapes(unittest.TestCase):
    def test_a_shape_drawn_another_way_is_a_new_picture_never_a_stale_one(self):
        work, recording.WORK = recording.WORK, tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, recording.WORK, True)
        self.addCleanup(setattr, recording, "WORK", work)
        dim = layouts._shape("same name", 4, 4, "format=gray,geq=lum=10[o]")
        bright = layouts._shape("same name", 4, 4, "format=gray,geq=lum=200[o]")
        self.assertNotEqual(dim, bright)
        self.assertEqual(layouts._shape("same name", 4, 4, "format=gray,geq=lum=10[o]"), dim, "made once")
        grey = [subprocess.run([FFMPEG, "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                               capture_output=True, check=True).stdout[0] for p in (dim, bright)]
        self.assertLess(grey[0], 30)
        self.assertGreater(grey[1], 180)


if __name__ == "__main__":
    unittest.main()
