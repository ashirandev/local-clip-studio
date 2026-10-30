"""The clip's drawing stages, checked on raw 1080x1920 frames FFmpeg drew from the project state (no encoding, so
unchanged means byte-equal): python -m unittest discover -s tests"""
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import effects, layouts, project, recording, render, text  # noqa: E402

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
    return dict(p.state(), stickers=entries.get("sticker", []), pushes=entries.get("push", []),
                look=entries.get("look"), glass=entries.get("glass", []), flurries=entries.get("flurry", []))


def drawn(st, t, words=True):
    """The 1080x1920 frame at clip second t as an (h, w, 3) array, from render.after_source."""
    import numpy as np
    graph, files = render.after_source(st, "src", "whole", 1, words=words)
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
ON_CARD = dict(TITLE, box=layouts.CARD_TITLE)


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

    def test_the_channel_name_is_in_the_words_the_whole_clip_and_not_in_the_video_under_them(self):
        from clipstudio import layer
        named = dict(state(), channel={"do": "channel", "name": "@SineySociety", "spot": "bottom right"})
        x, y, w, h = text.channel_box("@SineySociety", render._styles(named, "clean")["caption"], "bottom right")
        with open(render.ass(named), encoding="utf-8") as f:
            said = [line for line in f.read().splitlines() if line.startswith("Dialogue: ") and "_channel_" in line]
        self.assertEqual(len(said), 1)
        self.assertIn(",0:00:00.00,%s,clean_channel_title," % text._cs_clock(text._edge(4.0)), said[0])
        self.assertIsNone(render.ass(named, behind=True), "the name never goes behind the person")
        for t in (0.0, 3.9):
            plain, with_name = drawn(state(), t), drawn(named, t)
            found = changed(plain, with_name)
            self.assertIsNotNone(found, "the name at %.1f s" % t)
            self.assertTrue(inside(found, (x, y, w, h), slack=2), "only the name's box changed at %.1f s: %s"
                            % (t, found))
        self.assertIsNone(changed(drawn(state(), 1.0, words=False), drawn(named, 1.0, words=False)),
                          "the video under the words has no name")
        steps = layer.words(named)
        self.assertEqual((steps[0]["from"], steps[-1]["to"]), (0.0, text._edge(4.0) / 100.0),
                         "the words layer holds the name from the first frame to the last")

    def test_the_words_are_drawn_with_the_chosen_fonts_own_file(self):
        found = [f for f in text.fonts("กขค") if f["where"] == "computer" and f["free"]]
        if not found:
            self.skipTest("this computer has no free Thai font besides the studio's")
        st = dict(state("card", title=[TITLE]), style={"do": "style", "name": "clean", "font": found[0]["family"],
                                                       "colour": None})
        graph, _ = render.after_source(st, "src", "whole", 1)
        folder = text.fonts_folder(render.style_of(st))
        self.assertNotEqual(folder, text.FONTS)
        self.assertIn("fontsdir=%s:" % text._filter_value(folder), graph)
        own = dict(st, style={"do": "style", "name": "clean", "font": None, "colour": None,
                              "fonts": {"caption": found[0]["family"]}})
        graph, _ = render.after_source(own, "src", "whole", 1)
        folder = graph.split("fontsdir=")[1].split(":shaping")[0].replace("\\", "")
        given = os.listdir(folder)
        self.assertIn(os.path.basename(render.style_of(own, kind="caption").file[0]), given, "the captions' font")
        self.assertIn(os.path.basename(render.style_of(own, kind="title").file[0]), given, "and the title's")

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

    def test_a_blur_along_a_track_follows_the_face(self):
        plain = state()
        track = [[1.0, 280, 100, 40, 40], [1.5, 300, 100, 40, 40], [2.0, 320, 100, 40, 40]]
        moving = state(blur=[{"track": track, "start": 1.0, "end": 2.0}])
        lay = plain["clip"]["layout"]
        for t, x in ((1.0, 280), (1.5, 300), (1.75, 310)):
            found = changed(drawn(plain, t), drawn(moving, t))
            (want,) = layouts.to_frame(lay, [x, 100, 40, 40])
            self.assertIsNotNone(found, t)
            self.assertLess(abs((found[0] + found[2]) / 2 - (want[0] + want[2] / 2)), 12, "at %s s: %s vs %s"
                            % (t, found, want))
        for t in (0.9, 2.1):
            self.assertIsNone(changed(drawn(plain, t), drawn(moving, t)), "no blur at %s s" % t)

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

    def test_each_part_of_the_words_can_have_a_font_of_its_own(self):
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "ลดราคาเหลือ"}])
        band = [layouts.CAPTIONS[1], layouts.CAPTIONS[3]]
        base = state("card", title=[TITLE], captions=[{"lines": lines, "band": band}])
        house = dict(base, style={"do": "style", "name": "clean", "font": None, "colour": None})
        own = dict(house, style=dict(house["style"], fonts={"caption": "Kanit"}))
        self.assertEqual(render.style_of(own, kind="title").fontname, "Google Sans SemiBold")
        self.assertEqual(render.style_of(own, kind="caption").fontname, "Kanit SemiBold")
        found = changed(drawn(house, 1.5), drawn(own, 1.5))
        self.assertIsNotNone(found, "the captions are drawn in Kanit")
        self.assertGreaterEqual(found[1], band[0] - 2, "and only they: the title is drawn as it was, %s" % (found,))
        with self.assertRaisesRegex(ValueError, "kind 'logo' is not a part of the words: use one of title"):
            render.style_of(own, kind="logo")

    def test_the_style_entry_gives_every_part_or_each_part_its_size(self):
        entry = {"do": "style", "name": "pop", "font": None, "colour": None, "size": 2, "sizes": {"title": 5}}
        sized = dict(state(), style=entry)
        self.assertEqual([render.style_of(sized, kind=k).size for k in text.KINDS], [5, 2, 2])
        house = dict(state(), style=dict(entry, size=None, sizes=None))
        self.assertEqual([render.style_of(house, kind=k).size for k in text.KINDS], [3, 3, 3])
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "ลดราคาเหลือ"}])
        band = [1200, 400]
        tall = []
        for size in (1, 5):
            st = dict(state(captions=[{"lines": lines, "band": band}]), style=dict(entry, size=size, sizes=None))
            found = changed(drawn(state(), 1.5), drawn(st, 1.5))
            tall.append(found[3] - found[1])
        self.assertGreater(tall[1], tall[0] * 1.5, "size 5 is drawn bigger than size 1: %s" % tall)

    def test_the_style_entry_gives_every_part_or_each_part_its_colours(self):
        entry = {"do": "style", "name": "pop", "font": None, "colour": None, "colours": {"text": "#00FF66"},
                 "part_colours": {"title": {"text": "#FF2D95"}}}
        painted = dict(state(), style=entry)
        self.assertEqual([render.style_of(painted, kind=k).colour for k in text.KINDS],
                         ["#FF2D95", "#00FF66", "#00FF66"])
        house = dict(state(), style=dict(entry, colours=None, part_colours=None))
        self.assertEqual([render.style_of(house, kind=k).colour for k in text.KINDS], [text.STYLES["pop"].colour] * 3)
        st = dict(state(title=[{"text": "SALE", "start": 0.5, "end": 3.0, "box": [140, 300, 800, 300]}]),
                  style=entry)
        frame = drawn(st, 1.5).astype(int)
        pink = (abs(frame[..., 0] - 255) < 30) & (frame[..., 1] < 90) & (abs(frame[..., 2] - 149) < 40)
        self.assertGreater(int(pink[300:600, 140:940].sum()), 500, "the title is drawn in the colour its part was given")

    def test_captions_light_their_words_only_when_the_entry_says_lit(self):
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "watch this now"}])
        band = [layouts.CAPTIONS[1], layouts.CAPTIONS[3]]
        for entry, karaoke in (({}, False), ({"lit": False}, False), ({"lit": True}, True)):
            with open(render.ass(state(captions=[dict(entry, lines=lines, band=band)])), encoding="utf-8") as f:
                self.assertEqual("\\kf" in f.read(), karaoke, entry)

    def test_captions_pop_their_words_in_only_when_the_entry_says_pop(self):
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "watch this now"}])
        band = [layouts.CAPTIONS[1], layouts.CAPTIONS[3]]
        for entry, popped in (({}, False), ({"pop": False}, False), ({"pop": True}, True)):
            with open(render.ass(state(captions=[dict(entry, lines=lines, band=band)])), encoding="utf-8") as f:
                self.assertEqual(f.read().count("\\fscx60"), 3 if popped else 0, entry)

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

    def test_without_words_the_picture_is_drawn_as_if_there_were_none_and_keeps_its_flash(self):
        lines = text.caption_lines([{"from": 0.5, "to": 2.0, "text": "watch this now"}])
        band = [layouts.CAPTIONS[1], layouts.CAPTIONS[3]]
        flash = {"kind": "flash", "at": 1.0, "word": None, "centre": [540, 960]}
        word = {"kind": "word", "at": 1.0, "word": "POW", "centre": [540, 960]}
        bare = state("card", hit=[flash])
        full = state("card", title=[TITLE], captions=[{"lines": lines, "band": band}], hit=[flash, word])
        for t in (1.0, 1.4):
            self.assertIsNone(changed(drawn(bare, t), drawn(full, t, words=False)), "at %s s" % t)
        self.assertIsNotNone(changed(drawn(bare, 1.4), drawn(full, 1.4)), "with words they are drawn")
        self.assertIsNotNone(changed(drawn(state("card"), 1.0), drawn(full, 1.0, words=False)), "the flash stays")

    def test_the_camera_moves_the_picture_and_the_words_hold_still(self):
        titled = state("card", title=[ON_CARD])
        moved = state("card", title=[ON_CARD], hit=[{"kind": "punch", "at": 1.0, "word": None, "centre": [540, 960]},
                                                  {"kind": "shake", "at": 2.0, "word": None, "centre": [540, 960]}],
                      push=[{"start": 2.5, "end": 3.5, "zoom": 1.1, "centre": [540, 1150]}])
        x, y, w, h = layouts.CARD_TITLE
        letters = drawn(titled, 0.9)[y:y + h, x:x + w].min(axis=2) > 240
        ys, xs = letters.nonzero()
        x, y, w, h = x + xs.min() - 6, y + ys.min() - 6, xs.max() - xs.min() + 13, ys.max() - ys.min() + 13
        for t in (1.1, 2.05, 3.0):
            a, b = drawn(titled, t), drawn(moved, t)
            card = (a[1000:1300] != b[1000:1300]).any(axis=2).mean()
            self.assertGreater(card, 0.3, "the picture moved at %s s" % t)
            white_a = (a[y:y + h, x:x + w].min(axis=2) > 240)
            white_b = (b[y:y + h, x:x + w].min(axis=2) > 240)
            self.assertGreater(int(white_a.sum()), 2000, "the title's letters are there")
            self.assertLess(int((white_a != white_b).sum()), white_a.sum() * 0.02, "the title held still at %s s" % t)

    def test_a_glass_box_frosts_the_footage_under_the_stickers_and_the_words(self):
        png = os.path.join(self.tmp, "blue.png")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=0x2040ff:s=300x300", "-frames:v",
                        "1", png], check=True)
        glass = {"box": [100, 950, 880, 420], "start": 0.5, "end": 3.5, "tint": "dark", "shine": False}
        stuck = {"png": png, "at": 0.5, "length": 3.0, "box": [700, 1100, 240, 240]}
        words = dict(TITLE, box=[120, 960, 560, 200])
        plain = drawn(state("card"), 1.5)
        frosted = drawn(state("card", glass=[glass]), 1.5)
        inner = [160, 1220, 500, 120]
        self.assertLess(detail(frosted, inner), detail(plain, inner) * 0.1, "the card's footage is frosted")
        found = changed(plain, frosted)
        self.assertTrue(inside(found, [100 - 48, 950 - 36, 880 + 96, 420 + 96]), "only around the box: %s" % (found,))
        on_top = drawn(state("card", title=[words], sticker=[stuck], glass=[glass]), 1.5)
        alone = drawn(state("card", sticker=[stuck]), 1.5)
        middle = (slice(1180, 1260), slice(780, 860))
        self.assertLessEqual(int(abs(on_top[middle].astype(int) - alone[middle]).max()), 3, "the sticker is not frosted")
        x, y, w, h = words["box"]
        self.assertGreater(int((on_top[y:y + h, x:x + w].min(axis=2) > 240).sum()), 2000, "the title stands on it")

    def test_a_look_grades_the_footage_and_never_the_words_or_the_frames_design(self):
        titled = state("card", title=[ON_CARD])
        lay = titled["clip"]["layout"]
        x, y, w, h = [p["dst"] for p in lay["parts"] if p["name"] == "card"][0]
        dx, dy, d, _ = [p["dst"] for p in lay["parts"] if p["name"] == "circle"][0]
        tx, ty, tw, th = layouts.CARD_TITLE
        a = drawn(titled, 1.0)
        for name in effects.LOOKS:
            b = drawn(state("card", title=[ON_CARD], look=name), 1.0)
            self.assertGreater((a[y + 40:y + h - 40, x + 40:x + w - 40] != b[y + 40:y + h - 40, x + 40:x + w - 40]
                                ).any(axis=2).mean(), 0.3, "%s grades the card's footage" % name)
            white_a = a[ty:ty + th, tx:tx + tw].min(axis=2) > 240
            white_b = b[ty:ty + th, tx:tx + tw].min(axis=2) > 240
            self.assertGreater(int(white_a.sum()), 2000, "the title's letters are there")
            self.assertLess(int((white_a != white_b).sum()), white_a.sum() * 0.01, "%s leaves the title white" % name)
            ring = (slice(dy - layouts.RING + 2, dy - 2), slice(dx + d // 2 - 20, dx + d // 2 + 20))
            self.assertTrue((a[ring] == b[ring]).all(), "%s leaves the ring as it is" % name)

    def test_a_sticker_is_drawn_in_its_box_under_the_words(self):
        png = os.path.join(self.tmp, "sticker.png")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=0x2040ff:s=300x300", "-frames:v",
                        "1", png], check=True)
        box = [65, 285, 950, 235]
        stuck = {"png": png, "at": 0.5, "length": 3.0, "box": box}
        plain, with_sticker = state("card"), state("card", sticker=[stuck])
        found = changed(drawn(plain, 1.5), drawn(with_sticker, 1.5))
        self.assertTrue(inside(found, box, slack=2), "the sticker is inside its box: %s" % (found,))
        titled = drawn(state("card", title=[TITLE], sticker=[stuck]), 1.5)
        x, y, w, h = box
        letters = (titled[y:y + h, x:x + w].min(axis=2) > 240).sum()
        self.assertGreater(int(letters), 2000, "the title's white letters are on top of the blue sticker")

    def test_a_picture_drawn_for_what_is_said_covers_the_top_under_the_words(self):
        png = os.path.join(self.tmp, "picture.png")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=0x2040ff:s=1080x600", "-frames:v",
                        "1", png], check=True)
        shown = {"png": png, "at": 0.5, "length": 3.0, "box": [0, 0, 1080, 600], "about": "a ghost"}
        plain = state("card")
        with_picture = dict(state("card"), pictures=[shown])
        found = changed(drawn(plain, 1.5), drawn(with_picture, 1.5))
        self.assertTrue(inside(found, [0, 0, 1080, 600], slack=2), "the picture is over the top: %s" % (found,))
        self.assertIsNone(changed(drawn(plain, 0.2), drawn(with_picture, 0.2)), "nothing before its time")
        titled = drawn(dict(state("card", title=[TITLE]), pictures=[shown]), 1.5)
        x, y, w, h = layouts.TOP
        letters = (titled[y:y + h, x:x + w].min(axis=2) > 240).sum()
        self.assertGreater(int(letters), 2000, "the title's white letters are on top of the picture")

    def test_a_background_goes_first_behind_the_person_found_over_the_whole_clip(self):
        from test_cutout import make_mask
        png = os.path.join(self.tmp, "ground.png")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=0x3050A0:s=1080x1920", "-frames:v",
                        "1", png], check=True)
        ground = {"do": "backdrop", "png": png, "about": "a tidy desk"}
        mask = {"path": make_mask(self.tmp, "person.mkv", 4, 0), "person": None, "fps": 30, "size": [640, 360],
                "from": 0, "to": 4, "green": {"clear": 60, "solid": 24, "edge": 0.8}}
        graph, files = render.after_source(dict(state(), backdrop=ground, mask=mask), "src", "whole", 1)
        self.assertEqual(files[1], png, "the background is the first thing on the frame, after the person's mask")
        self.assertIn("despill=type=green", graph, "off a green cloth the green is taken off their edges")
        for cut in (None, dict(mask, to=3)):
            with self.assertRaises(ValueError) as e:
                render.after_source(dict(state(), backdrop=ground, mask=cut), "src", "whole", 1)
            self.assertIn("put the background on again", str(e.exception))
        self.assertNotIn(png, render.after_source(dict(state(), mask=mask), "src", "whole", 1)[1])

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


class OneLine(unittest.TestCase):
    def test_captions_one_line_at_a_time_are_drawn_big_unless_the_person_chose_a_size(self):
        st = state()
        self.assertEqual(render._caption_style(st, "clean").size, render.ONE_SIZE)
        self.assertEqual(render._caption_style(st, "clean").name[:3], "pop",
                         "with no look chosen, the heavy letters and thick outline of the reviews Tarm follows")
        plain = text.look("pop", sizes={"caption": render.ONE_SIZE})["caption"].sizes[1]
        lines = text.caption_lines([{"from": 0.0, "to": 3.0, "text": "ราคาไม่ถึงสองพันเลย"}], 12)
        grown = render._caption_style(st, "clean", lines, layouts.CAPTIONS[3]).sizes[1]
        self.assertGreater(grown, plain * 1.2, "a short line is drawn well past size 5, about half the frame wide")
        wide = text.caption_lines([{"from": 0.0, "to": 3.0, "text": "WWWWWWWWWWWW"}], 12)
        self.assertLessEqual(text.measure(wide[0]["text"], render._caption_style(st, "clean", wide, 400), "caption",
                                          4000).w, layouts.CAPTIONS[2], "a line too wide to grow is not grown past "
                                                                       "the band")
        st["style"] = {"name": "pop"}
        self.assertEqual((render._caption_style(st, "clean").size, render._caption_style(st, "clean").name[:3]),
                         (render.ONE_SIZE, "pop"))
        st["style"] = {"name": "pop", "sizes": {"caption": 2}}
        self.assertEqual(render._caption_style(st, "clean").size, 2)
        st["style"] = {"name": "pop", "size": 4}
        self.assertEqual(render._caption_style(st, "clean").size, 4)

    def test_people_talking_get_news_subtitles_and_their_title_keeps_its_own_look(self):
        st = state()
        self.assertEqual(render._caption_style(st, "pop").name[:3], "pop", "a talked review keeps the heavy look")
        st["clip"]["layout"]["kind"] = "talk"
        news = render._caption_style(st, "pop")
        self.assertEqual((news.name[:4], news.size), ("news", render.ONE_SIZE))
        title = render._title_style(st, "pop", news, {"text": "แกล้งผีปิดไฟ", "box": [65, 250, 950, 300]})
        self.assertEqual((title.name[:3], title.box), ("pop", False), "the title is not drawn in the captions' box")
        st["style"] = {"name": "glow"}
        self.assertEqual(render._caption_style(st, "pop").name[:4], "glow", "the person's own look wins")

    def test_people_talking_see_each_line_whole_on_its_box(self):
        """EP0 10-30: popped in on the news box, each word came on a box of its own, darker where it met the line's
        box, and the line's box grew a word at a time. Tarm 10-30: no box over another."""
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "watch this now"}])
        band = [layouts.CAPTIONS[1], layouts.CAPTIONS[3]]
        st = state(captions=[{"lines": lines, "band": band, "one": True, "pop": True}])
        self.assertFalse(render.whole_lines(st))
        with open(render.ass(st), encoding="utf-8") as f:
            self.assertGreater(f.read().count("\\fscx60"), 0, "a talked review still pops its words in")
        st["clip"]["layout"]["kind"] = "talk"
        self.assertTrue(render.whole_lines(st))
        with open(render.ass(st), encoding="utf-8") as f:
            body = f.read()
        self.assertIn("news", body)
        self.assertEqual((body.count("\\fscx60"), body.count("\\alpha&HFF&")), (0, 0),
                         "the line shows whole on one box, no word on a box of its own, none hidden")
        st["captions"]["one"] = False
        st["style"] = {"name": "news"}
        self.assertTrue(render.whole_lines(st), "the news look chosen for captions two lines a page")
        st["style"] = {"name": "clean"}
        self.assertFalse(render.whole_lines(st))

    @unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg, and numpy for the pixel checks")
    def test_one_line_captions_are_drawn_bigger_in_their_band(self):
        """Seen 10-28: the big style was not in the file's styles, and the yeti review would not render."""
        lines = text.caption_lines([{"from": 0.0, "to": 3.0, "text": "ราคาไม่ถึงสองพันเลย"}], 14)
        band = [layouts.CAPTIONS[1], layouts.CAPTIONS[3]]
        found = {}
        for chosen in (None, {"name": "clean", "sizes": {"caption": text.TUNED_SIZE}}):
            st = state(captions=[{"lines": lines, "band": band, "one": True, "lit": False, "pop": False}])
            st["style"] = chosen
            found[bool(chosen)] = changed(drawn(st, 1.0), drawn(st, 1.0, words=False))
        self.assertTrue(inside(found[False], [0, band[0], 1080, layouts.SAFE_BOTTOM - band[0]], slack=4),
                        "%s: never above the band, where the face is, nor in the bottom a phone covers" % (found[False],))
        self.assertGreater(found[False][2] - found[False][0], (found[True][2] - found[True][0]) * 1.2,
                           "one line at a time is drawn bigger than the usual size the person can still choose")

    @unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg, and numpy for the pixel checks")
    def test_one_line_captions_are_drawn_in_the_face_they_were_measured_in(self):
        """The yeti review 10-29: a clean clip's captions, in the pop look, were drawn in another face, half as wide
        again as measured and out of their band."""
        lines = text.caption_lines([{"from": 0.0, "to": 3.0, "text": "พอสมควรนะครับ"}], 14)
        band = [layouts.CAPTIONS[1], layouts.CAPTIONS[3]]
        st = state(captions=[{"lines": lines, "band": band, "one": True, "lit": False, "pop": False}])
        found = changed(drawn(st, 1.0), drawn(st, 1.0, words=False))
        wide = text.measure(lines[0]["text"], render._caption_style(st, "clean", lines, band[1]), "caption", 4000).w
        self.assertLess(abs(found[2] - found[0] - wide), wide * 0.1, "drawn %s, measured %d px wide" % (found, wide))

    @unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg, and numpy for the pixel checks")
    def test_a_title_over_one_line_captions_stands_as_tall_as_they_do_inside_its_box(self):
        """The yeti review 10-29: the hook at the title's own size stood half as tall as the captions under it."""
        hook = dict(TITLE, text="นั่งไกล เสียงยังชัด", start=0.0, box=[65, 285, 950, 235])
        lines = text.caption_lines([{"from": 0.0, "to": 3.0, "text": "พอสมควรนะครับ"}], 14)
        talked = [{"lines": lines, "band": [layouts.CAPTIONS[1], layouts.CAPTIONS[3]], "one": True, "lit": False,
                   "pop": False}]
        found = {}
        for name, chosen in (("own", None), ("talked", None), ("sized", {"name": "clean", "sizes": {"title": 3}})):
            st, under = (state(title=[hook]), state()) if name == "own" else \
                (state(title=[hook], captions=talked), state(captions=talked))
            st["style"] = under["style"] = chosen
            found[name] = changed(drawn(st, 1.0), drawn(under, 1.0))
        tall = {name: box[3] - box[1] for name, box in found.items()}
        self.assertTrue(inside(found["talked"], hook["box"], slack=4), "%s inside the title's box" % (found["talked"],))
        self.assertGreater(tall["talked"], tall["own"] * 1.5, "the hook stands as tall as the captions: %s" % tall)
        self.assertLess(tall["sized"], tall["own"] * 1.2, "a size the person chose for titles wins: %s" % tall)
        long = dict(hook, text="ไมค์ตัวนี้นั่งไกลแค่ไหนเสียงก็ยังชัดอยู่ดีนะ")
        found = changed(drawn(state(title=[long], captions=talked), 1.0), drawn(state(captions=talked), 1.0))
        self.assertTrue(inside(found, hook["box"], slack=4), "a longer title shrinks into its box: %s" % (found,))

    def test_a_title_kept_with_a_grow_is_drawn_that_much_taller_inside_its_box(self):
        """Matrix 10-30: every title stood 61 px tall, a third of the frame wide, and a montage's labels as small."""
        hook = dict(TITLE, text="ขยับนิดเดียวเล็งยิงได้เลย", start=0.0, box=[307, 286, 707, 232])
        found = {}
        for name, grow, chosen in (("own", None, None), ("grown", render.TITLE_GROW, None),
                                   ("sized", render.TITLE_GROW, {"name": "clean", "sizes": {"title": 3}})):
            st = state(title=[dict(hook, grow=grow) if grow else hook])
            st["style"] = chosen
            under = state()
            under["style"] = chosen
            found[name] = changed(drawn(st, 1.0), drawn(under, 1.0))
        tall = {name: box[3] - box[1] for name, box in found.items()}
        self.assertGreater(tall["grown"], tall["own"] * 1.4, "the hook grows: %s" % tall)
        self.assertTrue(inside(found["grown"], hook["box"], slack=4), "%s inside its box" % (found["grown"],))
        self.assertLess(tall["sized"], tall["own"] * 1.2, "a size the person chose for titles wins: %s" % tall)
        long = dict(hook, text="ขยับนิดเดียวก็เล็งยิงได้เลยไม่ต้องใช้ปุ่มอนาล็อกอีกต่อไป", grow=render.TITLE_GROW)
        found = changed(drawn(state(title=[long]), 1.0), drawn(state(), 1.0))
        self.assertTrue(inside(found, hook["box"], slack=4), "words that outgrow the box shrink into it: %s" % (found,))
        lines = text.caption_lines([{"from": 0.0, "to": 3.0, "text": "ซื้อถูกขายแพงได้กำไรครับ"}], 14)
        talked = [{"lines": lines, "band": [layouts.CAPTIONS[1], layouts.CAPTIONS[3]], "one": True, "lit": False,
                   "pop": False}]
        over = {}
        for name, grow in (("as_captions", None), ("grown", render.TITLE_GROW)):
            over[name] = changed(drawn(state(title=[dict(hook, grow=grow) if grow else hook], captions=talked), 1.0),
                                 drawn(state(captions=talked), 1.0))
        tall = {name: box[3] - box[1] for name, box in over.items()}
        self.assertGreater(tall["grown"], tall["as_captions"] * 1.08,
                           "the explainer replay 10-30: its grow was lost once one-line captions went on: %s" % tall)
        self.assertTrue(inside(over["grown"], hook["box"], slack=4), "%s inside its box" % (over["grown"],))

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

    def test_a_preview_at_any_time_shows_the_layouts_shapes_and_the_stickers(self):
        import numpy as np
        box = [380, 1400, 320, 320]
        plain = state("card")
        st = state("card", sticker=[{"png": effects.sticker_png("fire"), "at": 0.2, "length": 3.6, "box": box}])
        dx, dy, d, _ = next(p for p in plain["clip"]["layout"]["parts"] if p["name"] == "circle")["dst"]
        cx = int(dx + d / 2)
        for t in (0.7, 1.1, 1.3, 31 / 30, 1.55, 2.45, 2.6, 3.3):
            a = np.frombuffer(self.grey(plain, t), np.uint8).reshape(1920, 1080).astype(int)
            b = np.frombuffer(self.grey(st, t), np.uint8).reshape(1920, 1080).astype(int)
            ring = b[int(dy) - layouts.RING + 2:int(dy) - 1, cx - 20:cx + 20]
            self.assertGreater(ring.mean(), 200, "the face circle's white ring at %.3f s" % t)
            x, y, w, h = box
            self.assertGreater(np.abs(b - a)[y + 60:y + h - 60, x + 60:x + w - 60].mean(), 40,
                               "the sticker at %.3f s" % t)

    def test_a_preview_at_any_time_shows_the_flurry_where_its_move_has_it(self):
        import numpy as np
        box = [140, 300, 800, 1300]
        plain = state("card")
        for move, early, late in (("rain", 0, 1), ("rise", 1, 0)):
            st = state("card", flurry=[{"pngs": [effects.sticker_png("coin")], "at": 0.5, "box": box, "move": move,
                                        "count": 1, "length": 2.0, "size": 150}])
            heights = []
            for t in (0.62, 0.9 + 1 / 60, 1.05, 1.2):
                a = np.frombuffer(self.grey(plain, t), np.uint8).reshape(1920, 1080).astype(int)
                b = np.frombuffer(self.grey(st, t), np.uint8).reshape(1920, 1080).astype(int)
                ys, xs = np.nonzero(np.abs(b - a) > 40)
                self.assertTrue(len(xs) > 2000, "%s: the coin at %.3f s" % (move, t))
                self.assertTrue(box[0] <= xs.min() and xs.max() < box[0] + box[2] and box[1] <= ys.min()
                                and ys.max() < box[1] + box[3], "%s: inside its box at %.3f s" % (move, t))
                heights.append(ys.mean())
            self.assertEqual(sorted(heights, reverse=bool(early)), heights, "%s: %s" % (move, heights))

    def test_a_preview_cuts_the_person_out_where_the_clip_shows_them(self):
        import numpy as np
        mask_path = os.path.join(self.tmp, "person.mkv")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=640x360:r=30:d=4", "-vf",
                        "format=gray,drawbox=x=300:y=60:w=40:h=240:color=white:t=fill:enable='gte(t,0.9)'", "-c:v", "ffv1", "-pix_fmt",
                        "gray", mask_path], check=True)
        mask = {"path": mask_path, "person": "main", "fps": 30, "size": [640, 360], "from": 0.0, "to": 4.0}
        plain = state()
        (shown,) = layouts.to_frame(plain["clip"]["layout"], [300, 60, 40, 240])
        x, y, w, h = (int(v) for v in shown)
        person = (slice(y + 30, y + h - 30), slice(x + 30, x + w - 30))
        around = (slice(y + h + 60, y + h + 260), slice(60, 1020))

        def rgb(st, t):
            raw = subprocess.run([FFMPEG, "-v", "error", "-i", render.frame(self.rec, st, t, width=1080), "-f",
                                  "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True).stdout
            return np.frombuffer(raw, np.uint8).reshape(1920, 1080, 3).astype(int)

        def colour(f):
            return (f.max(axis=2) - f.min(axis=2)).mean()

        grey = dict(plain, mask=mask, cutouts=[{"look": "grey", "start": 0.5, "end": 3.0}])
        title = {"text": "SALE SALE SALE", "start": 0.5, "end": 3.0, "box": [100, y + h // 2 - 150, 880, 300]}
        behind = dict(state(title=[dict(title, behind=True)]), mask=mask)
        front = state(title=[title])
        for t in (1.0 + 1 / 60, 2.2):
            a = rgb(plain, t)
            b = rgb(grey, t)
            self.assertLess(colour(b[around]), 8, "black and white around the person at %.3f s" % t)
            self.assertGreater(colour(a[around]), 40, "where the footage has colour")
            self.assertLess(np.abs(b[person] - a[person]).mean(), 6, "the person keeps theirs at %.3f s" % t)
            c, d = rgb(behind, t), rgb(front, t)
            words = np.abs(d - a).max(axis=2) > 40
            self.assertGreater(int(words[person].sum()), 500, "the words cross the person")
            self.assertLess(np.abs(c[person] - a[person]).mean(), 6, "the person stands in front of the words")
            beside = words.copy()
            beside[:, x - 6:x + w + 6] = False
            self.assertGreater(int(beside.sum()), 300, "the words reach past the person")
            self.assertLess(np.abs(c - d).max(axis=2)[beside].mean(), 12, "and show there as drawn")
        with self.assertRaisesRegex(render.NotDrawn, "need the person's mask, and this clip has none: run cut_out"):
            render.frame(self.rec, dict(plain, cutouts=[{"look": "grey", "start": 0.5, "end": 3.0}]), 1.0)
        with self.assertRaisesRegex(render.NotDrawn, "cut-out look 'blur' is not a look: use one of grey, outline"):
            render.frame(self.rec, dict(grey, cutouts=[{"look": "blur", "start": 0.5, "end": 3.0}]), 1.0)

    def test_the_cut_out_looks_follow_whoever_speaks_and_words_go_behind_the_person(self):
        """Tarm 10-30: in a talk the speaker keeps their colour; the words behind stay behind the person found."""
        from unittest import mock
        st = dict(state(title=[dict(TITLE, behind=True)]), mask={"path": "found.mkv"},
                  cutouts=[{"look": "grey", "start": 0.5, "end": 3.0}])
        with mock.patch.object(render.cutout, "frame_mask", return_value="person.mkv"), \
                mock.patch.object(render.cutout, "speaking_mask", return_value="speaking.mkv"):
            grey, behind = render._cut_out(st, "clean")
        self.assertEqual((grey.files, behind.files), (["speaking.mkv"], ["person.mkv"]))

    def test_a_crowd_of_blurs_too_long_for_a_command_line_is_drawn(self):
        tracks = [{"track": [[0.2 + 0.03 * k + 0.01 * i, 8 * k + i, 100 + 10 * (k % 9), 20 + k % 40, 20 + k % 40]
                             for i in range(3)], "start": 0.2 + 0.03 * k, "end": 0.22 + 0.03 * k} for k in range(90)]
        st = state(blur=tracks)
        graph, _ = render.after_source(st, "src", "whole", 1)
        self.assertGreater(len(graph), 32767, "longer than Windows lets a command line be")
        self.assertLessEqual(graph.count("overlay="), effects.MOST_LANES + 4, "faces of many sizes, few at once")
        self.assertTrue(os.path.isfile(render.frame(self.rec, st, 1.0)))

    def test_two_requests_for_the_same_frame_at_once_both_get_it(self):
        import threading
        st = state("card", title=[TITLE])
        for t in (0.6, 1.0, 1.4, 1.8, 2.2, 2.6):
            start, got = threading.Barrier(2), []

            def ask():
                start.wait()
                try:
                    got.append(render.frame(self.rec, st, t))
                except Exception as e:
                    got.append(e)
            both = [threading.Thread(target=ask) for _ in range(2)]
            for b in both:
                b.start()
            for b in both:
                b.join()
            self.assertTrue(all(isinstance(g, str) for g in got), "at %s s: %s" % (t, got))
            self.assertEqual(got[0], got[1])
            self.assertGreater(os.path.getsize(got[0]), 1000)
        for name in ("clipframes", "shapes", "ass"):
            folder = os.path.join(recording.WORK, name)
            self.assertEqual([f for f in os.listdir(folder) if ".part" in f], [], "nothing half written is left")

    @unittest.skipUnless(os.name == "nt", "only Windows refuses to move a file onto one that is open")
    def test_a_file_made_again_while_the_first_is_open_keeps_the_first(self):
        folder = tempfile.mkdtemp(dir=self.tmp)
        target = os.path.join(folder, "frame.jpg")
        with open(target, "wb") as f:
            f.write(b"the same frame")
        part = render._part(target)
        with open(part, "wb") as f:
            f.write(b"the same frame")
        with open(target, "rb") as shown:
            render.settle(part, target)
            self.assertEqual(shown.read(), b"the same frame", "the page still reads the frame it opened")
        self.assertEqual(os.listdir(folder), ["frame.jpg"], "the second one is dropped")

    def test_a_frame_that_cannot_be_drawn_says_why(self):
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "watch this now"}])
        with self.assertRaises(render.NotDrawn) as e:
            render.frame(self.rec, state(captions=[{"lines": lines, "band": [1090, 10]}]), 1.5)
        self.assertIn("band [1090, 10]", str(e.exception).replace("(", "[").replace(")", "]"))
        broken = os.path.join(self.tmp, "not a video.mp4")
        with open(broken, "wb") as f:
            f.write(b"not a video" * 100)
        with self.assertRaises(render.NotDrawn) as e:
            render.frame(dict(self.rec, path=broken), state(), 1.5)
        self.assertIn("FFmpeg could not draw the frame", str(e.exception))
        self.assertNotIn("no reason", str(e.exception), "FFmpeg's own reason is passed on")
        self.assertEqual([f for f in os.listdir(os.path.join(recording.WORK, "clipframes")) if ".part" in f], [],
                         "a frame FFmpeg could not draw leaves nothing behind")


MOVING = ("nullsrc=s=640x360:r=30:d=6,geq=lum='120+50*sin(2*PI*(X/80-T*2))+70*lt(abs(Y-20-50*T),15)':cb=128:cr=128,"
          "format=yuv420p")


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,"
                          "r_frame_rate,nb_frames,duration,sample_rate", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def grey_frames(path, numbers):
    """{n: the grey 1080x1920 frame n of an mp4 as bytes}."""
    pick = "+".join("eq(n\,%d)" % n for n in numbers)
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vf", "select='%s'" % pick, "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    size = 1080 * 1920
    return {n: raw[k * size:(k + 1) * size] for k, n in enumerate(sorted(numbers))}


def apart(a, b):
    """The mean difference of two grey frames, sampled."""
    picks = range(0, len(a), 997)
    return sum(abs(a[i] - b[i]) for i in picks) / len(picks)


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe") and HAS_NUMPY, "needs ffmpeg, ffprobe and numpy")
class Base(unittest.TestCase):
    """The page's video of a 3 s card clip with a title and captions: made once, without the words."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")
        path = os.path.join(cls.tmp, "stream.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", MOVING, "-f", "lavfi", "-i",
                        "sine=frequency=440:sample_rate=48000:duration=6", "-af", "volume=-20dB", "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path], check=True)
        cls.rec = recording.probe(path)
        lines = text.caption_lines([{"from": 0.2, "to": 2.8, "text": "watch this now"}])
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[0.5, 2.0], [3.0, 4.5]],
                              "layout": layouts.make("card", (640, 360), [560, 280, 60, 60])},
                             {"do": "title", "text": "Boss down", "start": 0.2, "end": 2.8, "place": "top",
                              "box": layouts.TOP, "key": None},
                             {"do": "captions", "lines": lines, "band": [layouts.CAPTIONS[1], layouts.CAPTIONS[3]]}])
        cls.state = p.state()
        cls.made = render.base(cls.rec, cls.state)

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def frame(self, path, n):
        import numpy as np
        w, h = render.BASE_SIZE
        raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vf", "select='eq(n\\,%d)'" % n, "-fps_mode",
                              "passthrough", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(int)

    def test_the_page_video_is_small_at_30_fps_with_sound_and_without_the_words(self):
        streams = probe(self.made)
        video = streams["video"]
        self.assertEqual((video["width"], video["height"], video["r_frame_rate"], int(video["nb_frames"])),
                         (540, 960, "30/1", 90))
        self.assertIn("audio", streams, "it plays with its sound")
        worded = render.export(self.rec, self.state, os.path.join(self.tmp, "worded.mp4"), size=render.BASE_SIZE,
                               preset=render.BASE_PRESET, crf=render.BASE_CRF)
        a, b = self.frame(self.made, 30), self.frame(worded, 30)
        x, y, w, h = [v // 2 for v in layouts.TOP]
        self.assertGreater(abs(a - b)[y:y + h, x:x + w].max(), 100, "the title is not in it")
        cy, ch = layouts.CAPTIONS[1] // 2, layouts.CAPTIONS[3] // 2
        self.assertGreater(abs(a - b)[cy:cy + ch].max(), 100, "nor the captions")
        rest = abs(a - b).max(axis=2)
        rest[y:y + h, x:x + w] = 0
        rest[cy:cy + ch] = 0
        self.assertLess(rest.mean(), 3.0, "and the rest of the picture is the same (both are lossy: measured 1.2)")

    def test_it_is_made_once_and_again_only_when_the_picture_changes(self):
        with mock.patch.object(render.subprocess, "Popen", wraps=subprocess.Popen) as popen, \
                mock.patch.object(render.subprocess, "run", wraps=subprocess.run) as run:
            self.assertEqual(render.base(self.rec, self.state), self.made)
            fixed = dict(self.state, titles=[dict(self.state["titles"][0], text="Boss beaten")],
                         captions=dict(self.state["captions"], pop=True))
            self.assertEqual(render.base(self.rec, fixed), self.made, "a change to the words: the same video")
        self.assertEqual(popen.call_count + run.call_count, 0, "nothing made")
        flashed = dict(self.state, hits=[{"kind": "flash", "at": 1.0, "word": None, "centre": [540, 960]}])
        self.assertNotEqual(render.base(self.rec, flashed), self.made, "a change to the picture: made again")

    def test_a_mute_makes_the_page_video_again(self):
        with mock.patch.object(render, "export", side_effect=lambda rec, state, path, *a, **k: path) as made:
            muted = render.base(self.rec, dict(self.state, mutes=[[0.5, 1.5]]))
        self.assertNotEqual(muted, self.made, "the page plays the clip as it sounds now")
        self.assertEqual(made.call_args[0][1]["mutes"], [[0.5, 1.5]])


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Export(unittest.TestCase):
    """A 3 s card clip of two pieces from a recording of moving stripes, a bar that moves down without ever coming
    back, and a quiet tone, with a title, and a slow push and a sticker across the join, exported once."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")
        cls.path = os.path.join(cls.tmp, "stream.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", MOVING, "-f", "lavfi", "-i",
                        "sine=frequency=440:sample_rate=48000:duration=6", "-af", "volume=-20dB", "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", cls.path],
                       check=True)
        cls.rec = recording.probe(cls.path)
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[0.5, 2.0], [3.0, 4.5]],
                              "layout": layouts.make("card", (640, 360), [560, 280, 60, 60])},
                             {"do": "title", "text": "Boss down", "start": 0.5, "end": 2.5, "place": "top",
                              "box": layouts.TOP, "key": None}])
        png = os.path.join(cls.tmp, "sticker.png")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=0x2040ff:s=300x200", "-frames:v",
                        "1", png], check=True)
        cls.state = dict(p.state(), stickers=[{"png": png, "at": 1.2, "length": 1.0, "box": [240, 1300, 600, 400]}],
                         pushes=[{"start": 1.0, "end": 2.6, "zoom": 1.1, "centre": [540, 900]}])
        cls.out = os.path.join(cls.tmp, "clip.mp4")
        cls.progress = []
        cls.made = render.export(cls.rec, cls.state, cls.out, on_progress=cls.progress.append)

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_file_is_the_clip_at_30_fps_1080x1920_with_its_sound_at_minus_14_lufs(self):
        self.assertEqual(self.made, self.out)
        self.assertFalse(os.path.exists(self.out + ".part"), "the part file is gone")
        streams = probe(self.out)
        video, sound = streams["video"], streams["audio"]
        self.assertEqual((video["codec_name"], video["width"], video["height"], video["r_frame_rate"]),
                         ("h264", 1080, 1920, "30/1"))
        self.assertEqual(int(video["nb_frames"]), 90, "3 s at 30 fps")
        self.assertLess(abs(float(video["duration"]) - self.state["length"]), 1 / 30)
        self.assertEqual((sound["codec_name"], sound["sample_rate"]), ("aac", "48000"))
        self.assertLess(abs(float(sound["duration"]) - self.state["length"]), 1 / 30)
        meter = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", self.out, "-map", "0:a", "-af", "ebur128",
                                "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8",
                               errors="replace").stderr
        loud = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", meter)[-1])
        self.assertLess(abs(loud - render.LUFS), 1, "the sound is at %s LUFS" % loud)

    def test_the_index_is_at_the_front_so_it_plays_while_it_downloads(self):
        with open(self.out, "rb") as f:
            head = f.read()
        self.assertLess(head.find(b"moov"), head.find(b"mdat"))

    def test_progress_runs_up_to_the_clip_length(self):
        self.assertGreater(len(self.progress), 1)
        self.assertEqual(self.progress, sorted(self.progress))
        self.assertEqual(self.progress[-1], self.state["length"])

    def test_every_frame_is_the_preview_frame_at_its_time_across_the_join(self):
        times = (0.5, 1.4, 1.6, 2.4)
        made = grey_frames(self.out, [round(t * 30) for t in times] + [round(times[2] * 30) + 1])
        for t in times:
            preview = subprocess.run([FFMPEG, "-v", "error", "-i", render.frame(self.rec, self.state, t, width=1080),
                                      "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True,
                                     check=True).stdout
            self.assertLess(apart(made[round(t * 30)], preview), 3, "the frame at %s s" % t)
            if t == times[2]:
                self.assertGreater(apart(made[round(t * 30) + 1], preview), 6,
                                   "the next frame differs, so a frame's shift would show")

    def test_stop_ends_ffmpeg_at_once_and_leaves_no_file(self):
        out = os.path.join(self.tmp, "stopped.mp4")
        whole = dict(self.state, clip=dict(self.state["clip"], pieces=[[0, 6]]), length=6.0)
        stops, seen = [], []

        def progress(done):
            seen.append(done)
            if done > 0.3 and stops:
                stops.pop()()

        with self.assertRaises(render.Stopped):
            render.export(self.rec, whole, out, on_progress=progress, on_start=stops.append)
        self.assertFalse(stops, "Stop was pressed part way")
        self.assertLess(max(seen), 4, "FFmpeg ended when Stop was pressed: %s" % seen)
        self.assertFalse(os.path.exists(out))
        self.assertFalse(os.path.exists(out + ".part"), "the part written before Stop is gone")

    def test_a_clip_that_cannot_be_made_says_why(self):
        with self.assertRaises(render.NotDrawn) as e:
            render.export(self.rec, dict(self.state, hits=[{"kind": "wobble", "at": 1.0, "word": None,
                                                            "centre": [540, 960]}]), os.path.join(self.tmp, "x.mp4"))
        self.assertIn("hit kind 'wobble' is not a hit", str(e.exception))
        with self.assertRaises(render.NotDrawn) as e:
            render.export(self.rec, dict(self.state, clip=None), os.path.join(self.tmp, "x.mp4"))
        self.assertIn("no clip yet", str(e.exception))


VOICE = "0.3*(between(t,1,2.5)+between(t,4,5))*(sin(2*PI*220*t)+0.5*sin(2*PI*440*t))"
UNEVEN = "(0.6*between(t,1,2.5)+0.06*between(t,4,5))*(sin(2*PI*220*t)+0.5*sin(2*PI*440*t))"


def meter(path):
    """(integrated loudness in LUFS, true peak in dBTP) of a file's sound, as FFmpeg's ebur128 meter reads it."""
    run = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", path, "-map", "0:a", "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (float(re.findall(r"I:\s+(-?[\d.]+) LUFS", run.stderr)[-1]),
            float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", run.stderr)[-1]))


def samples(path):
    """A file's sound as one channel of 48 kHz floats."""
    import numpy as np
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-map", "0:a", "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<f4")


def tone(x, a, b, f):
    """The level in dB of the f Hz part of the sound from second a to b."""
    import numpy as np
    seg = x[round(a * 48000):round(b * 48000)]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) * 4 / len(seg)
    k = int(np.argmin(np.abs(np.fft.rfftfreq(len(seg), 1 / 48000) - f)))
    return 20 * math.log10(spectrum[k - 2:k + 3].max() + 1e-12)


def band(x, a, b, low, high):
    """The power in dB of the sound between low and high Hz from second a to b."""
    import numpy as np
    seg = x[round(a * 48000):round(b * 48000)]
    power = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / 48000)
    return 10 * math.log10(power[(freqs >= low) & (freqs <= high)].sum() / len(seg) + 1e-12)


def power(x, a, b):
    import numpy as np
    return 10 * math.log10(float(np.mean(x[round(a * 48000):round(b * 48000)].astype(float) ** 2)))


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe") and HAS_NUMPY, "needs ffmpeg, ffprobe and numpy")
class ExportSound(unittest.TestCase):
    """A 7 s clip of a recording where someone speaks from 0.5 to 2 s and from 3.5 to 4.5 s of the clip, with music
    of 2 s (a chord of 600 and 750 Hz, so it plays on from its start again), a thump at 1.6 s and a pop at 4.2 s, both
    while someone speaks, exported once; and a recording with no sound at all, given the music and a pop."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")
        talk = os.path.join(cls.tmp, "talk.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=8", "-f",
                        "lavfi", "-i", "aevalsrc=exprs='%s':s=48000:c=stereo:d=8" % VOICE, "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", talk], check=True)
        cls.music = os.path.join(cls.tmp, "music.wav")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                        "aevalsrc=exprs='0.2*sin(2*PI*600*t)+0.2*sin(2*PI*750*t)':s=48000:c=stereo:d=2", cls.music],
                       check=True)
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[0.5, 7.5]],
                              "layout": layouts.make("subject", (640, 360))}])
        cls.state = dict(p.state(), sounds=[{"kind": "thump", "at": 1.6}, {"kind": "pop", "at": 4.2}],
                         music={"path": cls.music, "from": 0.5, "level": -6, "duck": -12},
                         speech=[[0.5, 2.0], [3.5, 4.5]])
        cls.rec = recording.probe(talk)
        cls.out = render.export(cls.rec, cls.state, os.path.join(cls.tmp, "clip.mp4"))
        cls.sound = samples(cls.out)
        still = os.path.join(cls.tmp, "still.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=4", "-c:v",
                        "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", still], check=True)
        cls.still = recording.probe(still)
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[0, 4]],
                              "layout": layouts.make("subject", (640, 360))}])
        cls.quiet = dict(p.state(), sounds=[{"kind": "pop", "at": 1.0}], music={"path": cls.music}, speech=[])
        cls.still_out = render.export(cls.still, cls.quiet, os.path.join(cls.tmp, "still_clip.mp4"))

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_with_its_music_and_sounds_the_clip_is_still_at_minus_14_lufs_and_never_over_its_peak(self):
        loud, peak = meter(self.out)
        self.assertLess(abs(loud - render.LUFS), 0.5, "the sound is at %s LUFS" % loud)
        self.assertLess(peak, render.PEAK + 0.5, "its true peak is %s dBTP" % peak)
        sound = probe(self.out)["audio"]
        self.assertLess(abs(float(sound["duration"]) - self.state["length"]), 1 / 30)

    def test_the_music_goes_down_by_its_duck_while_someone_speaks(self):
        up, down = tone(self.sound, 2.6, 3.0, 600), tone(self.sound, 0.6, 1.0, 600)
        self.assertLess(abs(up - down - 12), 1, "the music is %.1f dB lower under the voice" % (up - down))
        self.assertLess(abs(tone(self.sound, 3.6, 4.1, 600) - down), 1, "and under the second span too")

    def test_the_music_plays_its_level_under_the_voice_and_on_from_its_start_again(self):
        gap = power(self.sound, 2.6, 3.0) - power(self.sound, 0.6, 1.0)
        self.assertTrue(-8.5 < gap < -5, "between the words the music is %.1f dB under the voice" % gap)
        self.assertGreater(tone(self.sound, 2.6, 3.0, 750), -40, "music still plays after its 2 s ran out")

    def test_the_music_fades_in_and_out(self):
        self.assertGreater(tone(self.sound, 0.2, 0.25, 600) - tone(self.sound, 0, 0.05, 600), 10, "it fades in")
        self.assertGreater(tone(self.sound, 5.1, 5.4, 600) - tone(self.sound, 6.8, 7.0, 600), 10,
                           "it fades out over the clip's last 1.5 s")

    def test_the_sounds_are_heard_at_their_times(self):
        self.assertGreater(band(self.sound, 1.6, 1.8, 40, 120) - band(self.sound, 1.1, 1.4, 40, 120), 15,
                           "the thump at 1.6 s")
        self.assertGreater(band(self.sound, 4.2, 4.3, 800, 1000) - band(self.sound, 3.8, 4.1, 800, 1000), 15,
                           "the pop at 4.2 s")

    def test_a_recording_with_no_sound_gets_its_music_and_sounds_at_minus_14_lufs(self):
        sound = probe(self.still_out).get("audio")
        self.assertIsNotNone(sound, "the clip has sound")
        self.assertLess(abs(float(sound["duration"]) - 4), 1 / 30)
        loud, _ = meter(self.still_out)
        self.assertLess(abs(loud - render.LUFS), 0.5, "the music alone is at %s LUFS" % loud)

    def test_music_that_cannot_be_played_says_why(self):
        out = os.path.join(self.tmp, "x.mp4")
        for music, said in (({"path": os.path.join(self.tmp, "none.mp3")}, "is not a file: give the path"),
                            ({"path": self.music, "level": 3}, "music level 3 is not a music level: use -30 to 0"),
                            ({"path": self.music, "duck": 0}, "db 0 is not a duck")):
            with self.assertRaises(render.NotDrawn) as e:
                render.export(self.rec, dict(self.state, music=music), out)
            self.assertIn(said, str(e.exception))
        with self.assertRaises(render.NotDrawn) as e:
            render.export(self.rec, dict(self.state, speech=None), out)
        self.assertIn("The music needs the times someone speaks", str(e.exception))
        silent = os.path.join(self.tmp, "silent.wav")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "2",
                        silent], check=True)
        with self.assertRaises(render.NotDrawn) as e:
            render.export(self.rec, dict(self.state, music={"path": silent}), out)
        self.assertIn("The music silent.wav is silent from its second 0", str(e.exception))
        self.assertFalse(os.path.exists(out))


def peak(x, a, b):
    """The loudest sample from second a to b, 0 to 1."""
    import numpy as np
    return float(np.abs(x[round(a * 48000):round(b * 48000)]).max())


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe") and HAS_NUMPY, "needs ffmpeg, ffprobe and numpy")
class ExportMute(unittest.TestCase):
    """The ExportSound recording, where someone speaks from 1 to 2.5 s and from 4 to 5 s, as a 7 s clip from 0.5 s
    with 1.0-1.5 s (inside the first words) and 3.4-4.6 s (all the second) muted; and as a 2 s clip from 1.2 s, in
    the middle of the first words, with its first 0.8 s muted."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")
        talk = os.path.join(cls.tmp, "talk.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=8", "-f",
                        "lavfi", "-i", "aevalsrc=exprs='%s':s=48000:c=stereo:d=8" % VOICE, "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", talk], check=True)
        cls.rec = recording.probe(talk)
        lay = layouts.make("subject", (640, 360))
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[0.5, 7.5]], "layout": lay},
                             {"do": "mute", "spans": [[1.5, 2.0], [3.9, 5.1]], "start": 1.0, "end": 4.6,
                              "back": False}])
        cls.state = p.state()
        cls.sound = samples(render.export(cls.rec, cls.state, os.path.join(cls.tmp, "clip.mp4")))
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[1.2, 3.2]], "layout": lay},
                             {"do": "mute", "spans": [[1.2, 2.0]], "start": 0, "end": 0.8, "back": False}])
        cls.opening = samples(render.export(cls.rec, p.state(), os.path.join(cls.tmp, "opening.mp4")))
        uneven = os.path.join(cls.tmp, "uneven.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=8", "-f",
                        "lavfi", "-i", "aevalsrc=exprs='%s':s=48000:c=stereo:d=8" % UNEVEN, "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", uneven], check=True)
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[0.5, 7.5]], "layout": lay},
                             {"do": "mute", "spans": [[0.9, 2.6]], "start": 0.4, "end": 2.1, "back": False}])
        cls.quiet = render.export(recording.probe(uneven), p.state(), os.path.join(cls.tmp, "quiet.mp4"))

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_nothing_of_the_recording_is_heard_where_it_is_muted_and_the_words_around_it_are(self):
        self.assertEqual(self.state["mutes"], [[1.0, 1.5], [3.4, 4.6]])
        self.assertLess(peak(self.sound, 1.06, 1.44), 1e-3, "muted inside the first words")
        self.assertLess(peak(self.sound, 3.45, 4.55), 1e-3, "and over all the second")
        self.assertGreater(power(self.sound, 0.6, 0.95), -30, "the first words before the mute are heard")
        self.assertGreater(power(self.sound, 1.55, 1.95), -30, "and after it")

    def test_the_mute_eases_in_over_its_first_30_ms_without_a_click(self):
        full = peak(self.sound, 0.9, 0.99)
        steps = [round(peak(self.sound, t, t + 0.005) / full, 2) for t in (1.01, 1.02, 1.035)]
        self.assertTrue(0.3 < steps[0] < 0.85 and steps[1] < 0.5 and steps[2] < 0.1,
                        "10, 20 and 35 ms in, at %s of the words' level" % steps)

    def test_a_clip_muted_from_its_first_second_has_no_sound_before_the_mute_ends(self):
        self.assertLess(peak(self.opening, 0, 0.75), 1e-3, "no ease at the clip's start, where nothing came before")
        self.assertGreater(power(self.opening, 0.85, 1.25), -30, "the words after the mute are heard")

    def test_the_clip_is_still_levelled_to_minus_14_lufs_on_what_is_heard(self):
        loud, top = meter(os.path.join(self.tmp, "clip.mp4"))
        self.assertLess(abs(loud - render.LUFS), 0.5, "the sound is at %s LUFS" % loud)
        self.assertLess(top, render.PEAK + 0.5)
        loud, top = meter(self.quiet)
        self.assertLess(abs(loud - render.LUFS), 1, "with a loud count-in muted, the quiet words left are levelled "
                                                    "up, not left 20 dB down: %s LUFS" % loud)



class SettledRead(unittest.TestCase):
    def test_a_file_being_moved_into_place_is_read_once_it_opens(self):
        import builtins
        from clipstudio import agent
        path = os.path.join(tempfile.mkdtemp(), "frame.jpg")
        with open(path, "wb") as f:
            f.write(b"jpg")
        real, refused = builtins.open, []

        def busy(name, *a, **k):
            if name == path and len(refused) < 2:
                refused.append(1)
                raise PermissionError(13, "in use", name)
            return real(name, *a, **k)

        builtins.open = busy
        try:
            self.assertEqual(render.read_settled(path), b"jpg")
            refused.clear()
            self.assertEqual(agent._picture(path)["image_url"]["url"], "data:image/jpeg;base64,anBn",
                             "lane 5, 10-30: a job ended on the frame Muse was to see")
            refused.clear()
            with self.assertRaises(PermissionError):
                render.read_settled(path, tries=2)
        finally:
            builtins.open = real

    def test_a_move_refused_while_another_is_under_way_is_tried_again(self):
        folder = tempfile.mkdtemp()
        target = os.path.join(folder, "words.ass")
        part = render._part(target)
        with open(part, "w") as f:
            f.write("ass")
        real, refused = os.replace, []

        def busy(a, b):
            if len(refused) < 2:
                refused.append(1)
                raise PermissionError(13, "delete pending", b)
            return real(a, b)

        with mock.patch.object(render.os, "replace", busy):
            render.settle(part, target)
        self.assertEqual(os.listdir(folder), ["words.ass"], "lane 5, 10-30: Errno 13 on a .ass file")


if __name__ == "__main__":
    unittest.main()
