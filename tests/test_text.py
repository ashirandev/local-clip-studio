"""Words on screen, measured on frames libass really drew: python -m unittest discover -s tests"""
import importlib.util
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import effects, recording, text  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
GREY = 128
SEEN = 12


def draw(ass_text, picks, folder, first=None, folder_fonts=text.FONTS):
    """{frame number: the 1920x1080x3 frame} of the ASS text burned on a grey 30 fps clip, with libass given the
    fonts in folder_fonts. The clip starts at frame first (the smallest pick when None), with its own timestamps, as
    the preview draws a frame."""
    import numpy as np
    path = os.path.join(folder, "words.ass")
    with open(path, "w", encoding="utf-8") as f:
        f.write(ass_text)
    graph, files = effects.join([text.burn(path, folder_fonts)], "src", "outv", 1)
    lo = min(picks) if first is None else first
    hi = max(picks) + 1
    full = "[0:v]trim=start_frame=%d:end_frame=%d,format=rgb24[src];%s" % (lo, hi, graph)
    run = subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                          "color=c=0x808080:s=1080x1920:r=30:d=%.3f" % (hi / 30 + 1), "-filter_complex", full,
                          "-map", "[outv]", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, timeout=120)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode("utf-8", "replace"))
    got = np.frombuffer(run.stdout, np.uint8).reshape(-1, 1920, 1080, 3)
    return {lo + k: got[k] for k in range(len(got)) if lo + k in picks}


def drawn(frame):
    """(x0, y0, x1, y1) of the pixels that are not the grey ground, or None."""
    import numpy as np
    mask = np.abs(frame.astype(int) - GREY).max(axis=2) > SEEN
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def doc(*events, styles=None):
    return text.document(styles or list(text.STYLES.values()), [e for group in events for e in group])


class Styles(unittest.TestCase):
    def test_every_house_style_names_a_face_in_fonts(self):
        self.assertTrue(os.path.isfile(os.path.join(text.FONTS, "OFL.txt")))
        for name, st in text.STYLES.items():
            path, index = st.file
            self.assertEqual(os.path.dirname(os.path.abspath(path)), os.path.abspath(text.FONTS), name)
            names = text._faces(path)[index][1]
            self.assertIn(st.fontname.lower(), names, name)

    def test_house_colours_stand_out(self):
        for name, st in text.STYLES.items():
            self.assertGreaterEqual(text.ratio(st.colour, st.edge), 4.5, name)
            self.assertGreaterEqual(text.ratio(st.key, st.edge), 4.5, name)
            self.assertEqual(st.note, "")

    def test_a_dim_colour_gets_an_outline_that_passes(self):
        st = text.style("clean", colour="#202020")
        self.assertEqual(st.colour, "#202020")
        self.assertEqual(st.edge, "#FFFFFF")
        self.assertIn("outline changed", st.note)
        self.assertGreaterEqual(text.ratio(st.colour, st.edge), 4.5)
        box = text.style("sticker", colour="#F0F0F0")
        self.assertEqual(box.edge, "#000000")
        self.assertIn("box changed", box.note)

    def test_refusals_name_the_field_and_what_is_allowed(self):
        with self.assertRaisesRegex(ValueError, "style 'neon' is not a house style: use one of clean, pop, sticker"):
            text.style("neon")
        with self.assertRaisesRegex(ValueError, "colour 'red' is not a colour: give it as #RRGGBB"):
            text.style(colour="red")
        with self.assertRaisesRegex(ValueError, "font 'No Such Face 9' is neither in the studio's fonts \\(Google Sans"):
            text.style(font="No Such Face 9")

    def test_a_font_without_thai_letters_is_refused_for_thai(self):
        found = text._find_font("Arial")
        if not found or text._has(found[0], found[1], "ก"):
            self.skipTest("this computer has no Arial without Thai letters")
        st = text.style(font="Arial")
        self.assertEqual(text.measure("Hello", st, "title", 900).lines, ["Hello"])
        with self.assertRaisesRegex(ValueError, "font 'Arial' has no Thai letters"):
            text.measure("สวัสดี", st, "title", 900)

    def test_a_shipped_face_is_found_by_any_of_its_names(self):
        for name in ("Kanit Bold", "Kanit-Bold", "kanit bold"):
            self.assertEqual(text.style(font=name).fontname, "Kanit Bold")


def drawn_title(st, folder, work):
    """A hash of the 1080x1920 grey frame with a Thai title in style st, burned with libass given folder."""
    import hashlib
    path = os.path.join(work, "%s.ass" % st.name.replace("~", "_"))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.document([st], ["Dialogue: 0,0:00:00.00,0:00:02.00,%s_title,,0,0,0,,{\\an8\\pos(540,300)}"
                                     "ลดราคา 75 บาท" % st.name]))
    graph, _ = effects.join([text.burn(path, folder)], "src", "o", 1)
    run = subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=0x808080:s=1080x1920:r=30:d=1",
                          "-filter_complex", "[0:v]format=yuv420p[src];" + graph, "-map", "[o]", "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, timeout=60)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode("utf-8", "replace")[-1000:])
    return hashlib.sha1(run.stdout).hexdigest()


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Fonts(unittest.TestCase):
    """Any font on the computer can be chosen, it is drawn as itself, and its licence is read from its own file."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def computer(self, free=True):
        """The families on this computer with Thai letters, free or not, else skip."""
        found = [f for f in text.fonts("กขค") if f["where"] == "computer" and f["free"] == free]
        if not found:
            self.skipTest("this computer has no %s Thai font besides the studio's" % ("free" if free else "unfree"))
        return found

    def test_a_family_gives_its_face_nearest_the_styles_own_weight(self):
        self.assertEqual(text.style("clean", "Kanit").fontname, "Kanit SemiBold", "clean is Google Sans SemiBold")
        self.assertEqual(text.style("pop", "Kanit").fontname, "Kanit Bold", "pop is Google Sans Bold")
        self.assertEqual(text.style("clean", "Google Sans").fontname, "Google Sans SemiBold")
        self.assertEqual(text.style("pop", "Kanit Regular").fontname, "Kanit Regular", "a face's own name wins")

    def test_each_part_can_have_a_font_of_its_own(self):
        parts = text.look("pop", None, None, {"caption": "Kanit", "big": "Kanit Regular"})
        self.assertEqual({kind: st.fontname for kind, st in parts.items()},
                         {"title": "Google Sans Bold", "caption": "Kanit Bold", "big": "Kanit Regular"})
        self.assertEqual(text.look("pop", "Kanit")["title"].fontname, "Kanit Bold", "font is every part's font")
        self.assertEqual(text.look("pop", "Kanit", fonts={"title": "Google Sans"})["title"].fontname,
                         "Google Sans Bold", "a part's own font wins")
        with self.assertRaisesRegex(ValueError, r"fonts \{'logo': 'Kanit'\} is not a font for each part: give "
                                                r"\{part: font\} with the parts title, caption, big"):
            text.look(fonts={"logo": "Kanit"})

    def test_a_size_means_how_tall_the_letters_stand_in_any_font(self):
        def stands(st, kind=2):
            path, index = st.file
            return text._tall(path, index, st.fontname, text.TALL_THAI) * st.sizes[kind] / 200
        for name, (face, tuned) in {"clean": ("Kanit SemiBold", 180), "pop": ("Kanit Black", 200),
                                    "sticker": ("Kanit Bold", 170)}.items():
            kanit = text.style(name, face)
            was = text._tall(kanit.file[0], kanit.file[1], kanit.fontname, text.TALL_THAI) * tuned / 200
            self.assertLess(abs(stands(text.STYLES[name]) / was - 1), 0.03,
                            "%s in Google Sans stands as tall as the %s it was tuned in" % (name, face))
        self.assertEqual(text.style("clean", "Kanit").sizes, (83, 63, 177),
                         "Kanit is drawn smaller: its Thai is 1.49 times taller at the same size")
        for name in text.HOUSE_NAMES:
            st = text.style(name, "Kanit")
            for kind in range(3):
                self.assertLess(abs(stands(st, kind) / stands(text.STYLES[name], kind) - 1), 0.03, (name, kind))

    def test_the_studios_fonts_come_first_and_are_free_by_their_own_file(self):
        google, kanit = text.fonts()[:2]
        for family, faces in ((google, ["Google Sans Regular", "Google Sans SemiBold", "Google Sans Bold"]),
                              (kanit, ["Kanit Regular", "Kanit SemiBold", "Kanit Bold", "Kanit Black"])):
            self.assertEqual((family["where"], family["free"], family["licence"]),
                             ("studio", True, "SIL Open Font License"), family["family"])
            self.assertEqual(family["faces"], faces, "lightest first")
        self.assertTrue(all(f["family"] not in ("Kanit", "Google Sans") for f in text.fonts()[2:]),
                        "listed once, as the studio's")
        self.assertTrue(os.path.isfile(os.path.join(text.FONTS, "GoogleSans-OFL.txt")), "its licence ships with it")

    def test_only_the_faces_with_the_letters_are_listed(self):
        for family in text.fonts("กขค"):
            for full in family["faces"]:
                path, index, _ = text._find_font(full)
                self.assertTrue(text._has(path, index, "กขค"), full)
        with self.assertRaisesRegex(ValueError, "letters '' is not text"):
            text.fonts("")

    def test_a_licence_is_what_the_fonts_file_says(self):
        self.assertEqual(text._licence({13: "This Font Software is licensed under the SIL Open Font License, "
                                            "Version 1.1.", 14: "https://openfontlicense.org"}),
                         ("SIL Open Font License", True))
        self.assertEqual(text._licence({14: "http://www.apache.org/licenses/LICENSE-2.0"}), ("Apache License", True))
        self.assertEqual(text._licence({}), ("its file names no licence", False))
        said, free = text._licence({13: "Microsoft supplied font. You may use this font to create, display, and print "
                                        "content as permitted by the license terms or terms of use, of the Microsoft "
                                        "product, service, or content in which this font was included."})
        self.assertFalse(free)
        self.assertTrue(said.startswith('its file says "Microsoft supplied font.') and said.endswith('..."'), said)
        self.assertLess(len(said), 140)

    def test_a_font_that_is_not_free_is_drawn_and_the_note_says_so(self):
        family = self.computer(free=False)[0]
        st = text.style("clean", family["family"])
        self.assertIn("is not known to be free: %s" % family["licence"], st.note)
        self.assertEqual(text.style("clean", "Kanit").note, "")

    def test_a_font_from_the_computer_is_given_to_libass_alone(self):
        st = text.style("clean", self.computer()[0]["family"])
        folder = text.fonts_folder(st)
        self.assertEqual(os.listdir(folder), [os.path.basename(st.file[0])])
        with open(st.file[0], "rb") as a, open(os.path.join(folder, os.listdir(folder)[0]), "rb") as b:
            self.assertEqual(a.read(), b.read(), "a copy of the very file")
        self.assertEqual(text.fonts_folder(st), folder, "made once")

    def test_libass_is_given_font_files_only_for_the_studios_own_faces_too(self):
        styles = [text.STYLES[name] for name in text.HOUSE_NAMES]
        folder = text.fonts_folder(*styles)
        self.assertEqual(sorted(os.listdir(folder)), sorted({os.path.basename(st.file[0]) for st in styles}))
        self.assertTrue(any(f.endswith(".txt") for f in os.listdir(text.FONTS)), "the licences stay in fonts/")
        path = os.path.join(self.tmp, "words.ass")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.document(styles, [e for st in styles
                                           for e in text.title(text.SAMPLE, 0, 1, (40, 40, 1000, 320), st)]))
        run = subprocess.run([FFMPEG, "-v", "info", "-f", "lavfi", "-i", "color=c=gray:s=1080x1920:d=0.1", "-vf",
                              text._ass_filter(path, folder), "-frames:v", "1", "-f", "null", "-"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(run.returncode, 0, run.stderr[-800:])
        self.assertIn("Loading font file", run.stderr, "libass reports what it loads at this log level")
        self.assertNotIn("Error opening memory font", run.stderr)

    def test_every_free_thai_font_on_this_computer_is_drawn_as_itself(self):
        fake = text.STYLES["clean"]._replace(name="fake", fontname="No Such Font Anywhere")
        stand_in = drawn_title(fake, text.FONTS, self.tmp)
        for family in self.computer():
            seen = set()
            for weight in (400, 700):
                path, index, full = text._find_font(family["family"], weight)
                self.assertFalse(text._face_at(path, index).italic and not all(
                    text._face_at(*text._find_font(f)[:2]).italic for f in family["faces"]), "%s is upright" % full)
                st = text.style("clean", full)
                picture = drawn_title(st, text.fonts_folder(st), self.tmp)
                self.assertNotEqual(picture, stand_in, "%s is drawn as itself, not as libass's stand-in" % full)
                seen.add((full, picture))
            self.assertEqual(len({p for _f, p in seen}), len({f for f, _p in seen}),
                             "%s: each weight is drawn as its own face" % family["family"])

    @unittest.skipUnless(HAS_NUMPY, "needs numpy for the pixel checks")
    def test_a_font_from_the_computer_is_measured_as_it_is_drawn(self):
        st = text.style("clean", self.computer()[0]["family"])
        said = "ลดราคา 75 บาท"
        events = text.title(said, 0, 2, (90, 300, 900, 400), st)
        frame = draw(text.document([st], events), [15], self.tmp, folder_fonts=text.fonts_folder(st))[15]
        box = drawn(frame)
        self.assertIsNotNone(box, "the title is drawn")
        self.assertLess(abs((box[2] - box[0]) - text.measure(said, st, "title", 900).w), 3,
                        "%s: drawn %d px wide, measured %d" % (st.fontname, box[2] - box[0],
                                                               text.measure(said, st, "title", 900).w))

    def test_the_name_written_is_one_libass_draws_the_face_by(self):
        real = text._drawn_from
        text._drawn_name_of.cache_clear()
        self.addCleanup(text._drawn_name_of.cache_clear)
        try:
            text._drawn_from = lambda name, folder: "kanit-bold" if name == "Kanit-Bold" else "arialmt"
            self.assertEqual(text.style("clean", "Kanit Bold").fontname, "Kanit-Bold",
                             "the PostScript name, when only that one draws the face")
            text._drawn_name_of.cache_clear()
            text._drawn_from = lambda name, folder: "arialmt"
            with self.assertRaisesRegex(ValueError, "font 'Kanit Bold' cannot be drawn: libass draws another font "
                                                    "for every one of Kanit Bold's names"):
                text.style("clean", "Kanit Bold")
        finally:
            text._drawn_from = real


class Wrapping(unittest.TestCase):
    def test_numbers_and_particles_stay_whole(self):
        said = "ราคาเจ็ดสิบห้าบาท ลดเหลือ 1,990 บาทนะครับ ส่งฟรีเลย"
        for width in (220, 300, 420):
            lines = text.measure(said, text.STYLES["pop"], "title", width).lines
            self.assertEqual("".join(lines).replace(" ", ""), said.replace(" ", ""))
            joined = "|".join(lines)
            for number in ("เจ็ดสิบห้า", "1,990"):
                self.assertIn(number, joined.replace(" ", ""))
                self.assertTrue(any(number in line for line in lines), (number, lines))
            for line in lines[1:]:
                self.assertFalse(line.startswith(text.PARTICLES), lines)

    def test_english_breaks_at_spaces(self):
        st = text.STYLES["clean"]
        lines = text.measure("Make a short clip from a long stream", st, "title", 500).lines
        self.assertGreater(len(lines), 1)
        self.assertEqual(" ".join(lines), "Make a short clip from a long stream")
        for line in lines:
            self.assertLessEqual(text.measure(line, st, "title", 1080).w, 500, line)
        whole = text.measure("Make a short clip", st, "title", 1080)
        self.assertEqual(len(whole.lines), 1)
        tight = text.measure("Make a short clip", st, "title", whole.w - 8)
        self.assertEqual(len(tight.lines), 2)
        self.assertLessEqual(tight.w, whole.w - 8)

    def test_a_measure_is_quick(self):
        st = text.STYLES["clean"]
        text.measure("อุ่นเครื่องก่อน", st, "title", 900)
        started = time.time()
        text.measure("วันนี้ลดราคา %d บาท ส่งฟรีทั่วประเทศ" % int(time.time()), st, "title", 700)
        self.assertLess(time.time() - started, 0.3)

    def test_width_refusal(self):
        with self.assertRaisesRegex(ValueError, "width 0 is not a width"):
            text.measure("abc", text.STYLES["clean"], "title", 0)
        with self.assertRaisesRegex(ValueError, "kind 'subtitle' is not a kind of words: use one of title"):
            text.measure("abc", text.STYLES["clean"], "subtitle", 500)


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Drawn(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="lcs text ")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_measure_is_the_box_libass_draws(self):
        cases = [("clean", "ลงโปรแกรม AI ในคอมฟรี", 1000), ("sticker", "Cut a clip by itself", 1000),
                 ("pop", "ไลฟ์ขายของสองชั่วโมง ได้คลิปสั้นทุกสินค้า", 620)]
        for name, words, width in cases:
            st = text.STYLES[name]
            m = text.measure(words, st, "title", width)
            events = text.title(words, 0, 1, (0, 0, 1080, 1920), st)
            box = drawn(draw(doc(events), [0], self.folder)[0])
            self.assertIsNotNone(box, name)
            self.assertLessEqual(abs((box[2] - box[0]) - m.w), 2, (name, box, m))
            self.assertLessEqual(abs((box[3] - box[1]) - m.h), 2, (name, box, m))
            self.assertLessEqual(abs((box[0] + box[2]) / 2 - 540), 2, (name, box))
            self.assertLessEqual(abs((box[1] + box[3]) / 2 - 960), 2, (name, box))
        self.assertEqual(len(m.lines), 2)

    def test_a_title_shows_from_its_first_frame_to_its_last(self):
        events = text.title("เริ่มตรงนี้", 0.51, 1.5, (90, 300, 900, 400), text.STYLES["pop"])
        got = draw(doc(events), [15, 16, 44, 45], self.folder, first=0)
        self.assertIsNone(drawn(got[15]))
        self.assertIsNotNone(drawn(got[16]))
        self.assertIsNotNone(drawn(got[44]))
        self.assertIsNone(drawn(got[45]))
        on_grid = text.title("หนึ่งวิ", 1.0, 2.0, (90, 300, 900, 400), text.STYLES["clean"])
        got = draw(doc(on_grid), [29, 30, 59, 60], self.folder, first=0)
        self.assertEqual([drawn(got[k]) is not None for k in (29, 30, 59, 60)], [False, True, True, False])

    def test_a_frame_is_the_same_from_any_start(self):
        events = text.title("ภาพเดียวกัน", 0.2, 2.0, (90, 300, 900, 400), text.STYLES["sticker"])
        whole = draw(doc(events), [0, 40], self.folder, first=0)
        alone = draw(doc(events), [40], self.folder, first=40)
        self.assertEqual(whole[40].tobytes(), alone[40].tobytes())

    def test_a_title_is_drawn_inside_its_box_with_its_key(self):
        st = text.STYLES["pop"]
        place = (140, 1300, 800, 420)
        events = text.title("ลง AI ในคอม ฟรี ทีละขั้น", 0, 1, place, st, key="ฟรี")
        frame = draw(doc(events), [0], self.folder)[0]
        x0, y0, x1, y1 = drawn(frame)
        self.assertTrue(place[0] <= x0 and x1 <= place[0] + place[2] and place[1] <= y0 and y1 <= place[1] + place[3],
                        (x0, y0, x1, y1))
        key = [int(st.key[k:k + 2], 16) for k in (1, 3, 5)]
        import numpy as np
        near = (np.abs(frame.astype(int) - key).max(axis=2) < 40).sum()
        self.assertGreater(near, 300)

    def test_a_title_that_does_not_fit_says_the_size_it_needs(self):
        with self.assertRaisesRegex(ValueError, r"the title needs \d+ x \d+ px but box \(0, 0, 300, 60\) is 300 x 60"):
            text.title("คำยาวมากจนกล่องเล็กใส่ไม่ได้", 0, 1, (0, 0, 300, 60), text.STYLES["clean"])

    def test_title_refusals(self):
        st = text.STYLES["clean"]
        with self.assertRaisesRegex(ValueError, "end 1 is not after start 2: use an end later than the start"):
            text.title("x", 2, 1, (0, 0, 500, 500), st)
        with self.assertRaisesRegex(ValueError, r"box \(900, 0, 400, 300\) runs off the 1080x1920 frame"):
            text.title("x", 0, 1, (900, 0, 400, 300), st)
        with self.assertRaisesRegex(ValueError, "key 'แพง' is not one of the title's words: use one of 'ลดราคา', "
                                                "'วันนี้'"):
            text.title("ลดราคา วันนี้", 0, 1, (0, 0, 900, 500), st, key="แพง")
        with self.assertRaisesRegex(ValueError, "libass reads as a command"):
            text.title("a {b}", 0, 1, (0, 0, 900, 500), st)

    def test_burn_takes_a_folder_with_spaces_and_a_drive(self):
        path = os.path.join(self.folder, "words.ass")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc([]))
        graph = text.burn(path).graph
        self.assertTrue(graph.startswith("[in]ass=") and graph.endswith("[out]"))
        odd = os.path.join(self.folder, "it's, a [clip]; 2")
        os.makedirs(odd)
        events = text.title("ชื่อแปลก", 0, 1, (90, 300, 900, 400), text.STYLES["clean"])
        self.assertIsNotNone(drawn(draw(doc(events), [0], odd)[0]))


POPPED = [{"from": 0.5, "to": 3.5, "text": "วันนี้ลดราคาทั้งร้าน ซื้อหนึ่งแถมหนึ่ง"}]
SAID = [{"from": 0.5, "to": 4.0, "text": "วันนี้ลดราคาเหลือเจ็ดสิบห้าบาทนะครับ ส่งฟรีทั่วประเทศเลยค่ะ"},
        {"from": 4.3, "to": 6.0, "text": "ONE TWO three 1,990 baht"},
        {"from": 6.5, "to": 9.0, "text": "ราคา 1,250 บาทจ้า ซื้อสองชิ้นลดอีกห้าสิบครับ"}]


class CaptionLines(unittest.TestCase):
    def test_lines_are_short_whole_and_give_the_phrase_back(self):
        for most in (8, 12, 16):
            lines = text.caption_lines(SAID, most=most)
            for k, phrase in enumerate(SAID):
                own = [line for line in lines if line["phrase"] == k]
                self.assertEqual("".join(line["text"] for line in own), phrase["text"])
                for line in own:
                    self.assertLessEqual(text._letters(line["text"]), most, line["text"])
                    self.assertEqual("".join(w["text"] for w in line["words"]), line["text"])
                for m in text.thai.NUMBER.finditer(phrase["text"]):
                    if m.group(0) in ("สอง",):
                        continue
                    self.assertTrue(any(m.group(0) in line["text"] for line in own), (m.group(0), most))
                for line in own[1:]:
                    self.assertFalse(line["text"].lstrip().startswith(text.PARTICLES), line["text"])

    def test_words_take_the_phrase_time_in_order(self):
        lines = text.caption_lines(SAID)
        for k, phrase in enumerate(SAID):
            words = [w for line in lines if line["phrase"] == k for w in line["words"]]
            self.assertAlmostEqual(words[0]["from"], phrase["from"], places=3)
            self.assertAlmostEqual(words[-1]["to"], phrase["to"], places=3)
            for a, b in zip(words, words[1:]):
                self.assertAlmostEqual(a["to"], b["from"], places=3)
                self.assertLess(a["from"], a["to"])

    def test_marks_do_not_count_as_letters(self):
        self.assertEqual(text._letters("ที่นี่"), 2)
        self.assertEqual(text._letters("ราคา"), 4)
        self.assertEqual(text._letters(" ok "), 2)

    def test_a_long_word_is_cut_but_a_long_number_is_refused(self):
        lines = text.caption_lines([{"from": 0, "to": 2, "text": "supercalifragilisticexpialidocious"}], most=10)
        self.assertTrue(all(text._letters(line["text"]) <= 10 for line in lines))
        self.assertEqual("".join(line["text"] for line in lines), "supercalifragilisticexpialidocious")
        with self.assertRaisesRegex(ValueError, r"most 6 is fewer letters than the number '1,000,000' \(9 letters\)"):
            text.caption_lines([{"from": 0, "to": 2, "text": "ราคา 1,000,000 บาท"}], most=6)

    def test_phrase_refusals(self):
        with self.assertRaisesRegex(ValueError, "phrase 0's to 1 is not after its from 2"):
            text.caption_lines([{"from": 2, "to": 1, "text": "x"}])
        with self.assertRaisesRegex(ValueError, "most 2 is not a line length: use a whole number of letters from 4"):
            text.caption_lines(SAID, most=2)


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Looks(unittest.TestCase):
    BOX = (90, 700, 900, 400)

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="lcs looks ")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def titled(self, st, words="ลดราคา"):
        """The frame with words titled in st on the grey ground, as an int array."""
        events = text.title(words, 0, 2, self.BOX, st)
        return draw(doc(events, styles=[st]), [30], self.folder, folder_fonts=text.fonts_folder(st))[30].astype(int)

    def near(self, frame, hexcolour, within=24):
        """How many pixels are within `within` of the colour in every channel."""
        import numpy as np
        rgb = [int(hexcolour[i:i + 2], 16) for i in (1, 3, 5)]
        return int((np.abs(frame - rgb) <= within).all(axis=2).sum())

    def test_the_glow_is_a_layer_of_its_own_under_letters_that_keep_a_dark_outline(self):
        st = text.STYLES["glow"]
        events = text.title("ลดราคา", 0, 2, self.BOX, st)
        self.assertEqual(len(events), 2, "the glow, then the letters")
        self.assertIn(",glow_title_glow,", events[0])
        self.assertIn("\\1a&HFF&", events[0], "the glow layer hides its fill")
        self.assertIn("\\blur", events[0])
        self.assertNotIn("\\blur", events[1], "the letters stay sharp")
        self.assertGreaterEqual(text.ratio(st.colour, st.edge), 4.5, "and keep an outline that reads")
        orange, cyan = self.titled(st), self.titled(text.style("glow", colours={"glow": "#22D3EE"}))
        warm = lambda f: int(((f[..., 0] - f[..., 2]) > 60).sum())
        cold = lambda f: int(((f[..., 2] - f[..., 0]) > 60).sum())
        self.assertGreater(warm(orange), 500, "an orange glow round the letters")
        self.assertLess(warm(cyan), 50)
        self.assertGreater(cold(cyan), 500, "the glow is the colour chosen")

    def test_3d_letters_cast_a_hard_deep_copy_in_the_shade_colour(self):
        st = text.STYLES["3d"]
        fields = text._ass_style(st, "title").split(",")
        self.assertEqual(fields[6], text._colour(st.shade), "a solid shadow in the shade colour")
        self.assertGreater(float(fields[17]), 4)
        house = self.near(self.titled(st), st.shade)
        self.assertGreater(house, 300)
        blue = self.titled(text.style("3d", colours={"shade": "#0B1B4D"}))
        self.assertGreater(self.near(blue, "#0B1B4D"), 300)
        self.assertLess(self.near(blue, st.shade), house / 4, "only where the orange rim meets the blue")

    def test_every_look_has_a_card_in_english_and_thai_with_a_sample(self):
        import numpy as np
        work, recording.WORK = recording.WORK, self.folder
        self.addCleanup(setattr, recording, "WORK", work)
        cards = text.cards()
        self.assertEqual([c["name"] for c in cards], list(text.HOUSE_NAMES))
        for c in cards:
            self.assertTrue(c["en"].strip() and re.search("[\u0e00-\u0e7f]", c["th"]), c["name"])
            self.assertEqual(set(c["colours"]), set(text.PAINTS))
            with open(c["png"], "rb") as f:
                head = f.read(24)
            self.assertEqual(struct.unpack(">II", head[16:24]), (540, 200), c["name"])
            grey = np.frombuffer(subprocess.run([FFMPEG, "-v", "error", "-i", c["png"], "-f", "rawvideo", "-pix_fmt",
                                                 "gray", "-"], capture_output=True, check=True).stdout, np.uint8)
            self.assertGreater(int((grey > 200).sum()), 400, "%s's sample shows its words" % c["name"])
            ys, xs = np.nonzero(grey.reshape(200, 540) > 110)
            fill = (xs.max() + 1 - xs.min()) / 540
            self.assertTrue(0.55 < fill < 0.9, "%s's words fill %.2f of the card's width" % (c["name"], fill))
            self.assertTrue(ys.min() > 0 and ys.max() < 199, "%s's words are not cut" % c["name"])
        self.assertEqual([c["png"] for c in text.cards()], [c["png"] for c in cards], "made once")
        with mock.patch.object(text, "CARD_GROUND", "geq=lum=200:cb=128:cr=128"):
            light = text._card_png(text.STYLES["clean"])
        self.assertNotEqual(light, cards[0]["png"], "a card drawn another way is a new picture")


class Colours(unittest.TestCase):
    def test_every_colour_of_a_look_can_be_chosen_for_every_part_or_one_part(self):
        st = text.style("3d", colours={"key": "#22D3EE", "edge": "#1E3A8A", "shade": "#0B1B4D"})
        self.assertEqual((st.colour, st.key, st.edge, st.shade), ("#FFFFFF", "#22D3EE", "#1E3A8A", "#0B1B4D"))
        fields = text._ass_style(st, "title").split(",")
        self.assertEqual((fields[5], fields[6]), (text._colour("#1E3A8A"), text._colour("#0B1B4D")))
        self.assertEqual(text.style("pop", colours={"text": "#FFE8D0"}).colour, "#FFE8D0")
        names = {text.style("3d").name, st.name, text.style("3d", colours={"key": "#FF00FF"}).name}
        self.assertEqual(len(names), 3, "every set of colours is an ASS style of its own")
        parts = text.look("pop", colours={"key": "#00FF00"}, part_colours={"caption": {"key": "#FF00FF"}})
        self.assertEqual([parts[k].key for k in text.KINDS], ["#00FF00", "#FF00FF", "#00FF00"])

    def test_a_glow_can_be_given_to_any_look(self):
        st = text.style("clean", colours={"glow": "#FF00FF"})
        self.assertEqual(st.glow, "#FF00FF")
        events = text.title("ลดราคา", 0, 2, (90, 700, 900, 400), st)
        self.assertEqual(len(events), 2)
        self.assertIn("Style: %s_title_glow," % st.name, text.document([st], events))

    def test_a_colour_the_person_chose_is_kept_even_when_hard_to_read_and_the_note_says_so(self):
        st = text.style("pop", colours={"edge": "#FFFFFF"})
        self.assertEqual(st.edge, "#FFFFFF")
        self.assertIn("text #FFFFFF on the outline #FFFFFF stands out only 1.0:1, below 4.5:1, so it may be hard to "
                      "read: kept as asked.", st.note)
        key = text.style("pop", colours={"key": "#F0F0F0", "edge": "#FFFFFF", "text": "#000000"})
        self.assertEqual(key.key, "#F0F0F0")
        self.assertIn("key colour #F0F0F0 on the outline #FFFFFF stands out only", key.note)
        self.assertIn("the clean look has no shadow, so shade #000000 is not drawn",
                      text.style("clean", colours={"shade": "#000000"}).note)

    def test_colour_refusals(self):
        with self.assertRaisesRegex(ValueError, r"colours \{'logo': '#FFFFFF'\} is not a colour for each part of a "
                                                r"look: give \{name: #RRGGBB\} with the names text, key, edge, glow, "
                                                r"shade\."):
            text.style(colours={"logo": "#FFFFFF"})
        with self.assertRaisesRegex(ValueError, "key 'red' is not a colour: give it as #RRGGBB"):
            text.style(colours={"key": "red"})
        for bad in ({"logo": {}}, {"title": "#FFFFFF"}):
            with self.assertRaisesRegex(ValueError, r"part_colours %s is not the colours of each part: give "
                                                    r"\{part: \{name: #RRGGBB\}\} with the parts title, caption, big"
                                        % re.escape(repr(bad))):
                text.look(part_colours=bad)
        with self.assertRaisesRegex(ValueError, "colours '#FFFFFF' is not a colour for each part of a look"):
            text.look(colours="#FFFFFF")


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Sizes(unittest.TestCase):
    TWO = [{"from": 0.5, "to": 2.0, "text": "ลดราคาเหลือ"}, {"from": 2.2, "to": 3.0, "text": "บาทนะครับ"}]

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="lcs sizes ")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_each_step_stands_the_letters_a_set_ratio_taller_in_every_font(self):
        for name in text.HOUSE_NAMES:
            self.assertEqual(text.style(name, size=text.TUNED_SIZE), text.style(name), "3 is the tuned sizes")
            tuned = text.style(name).sizes
            for n in text.SIZES:
                st = text.style(name, size=n)
                self.assertEqual(st.size, n)
                for got, base in zip(st.sizes, tuned):
                    self.assertLessEqual(abs(got - base * text.SIZE_STEP ** (n - 3)), 1, (name, n, st.sizes))
        tall = {}
        for font in (None, "Kanit"):
            for n in text.SIZES:
                st = text.style("pop", font, size=n)
                tall[font, n] = text.measure(text.TALL_THAI, st, "caption", 1080).h
                lh = text._line_height(st, "caption") / text._line_height(text.style("pop", font), "caption")
                outline = float(text._ass_style(st, "caption").split(",")[16])
                self.assertLess(abs(lh / text.SIZE_STEP ** (n - 3) - 1), 0.04, "lines grow with it: %s %d" % (font, n))
                self.assertLess(abs(outline - 68 * 0.10 * text.SIZE_STEP ** (n - 3)), 0.3, "and the outline")
        for n in text.SIZES:
            want = text.SIZE_STEP ** (n - 3)
            self.assertLess(abs(tall[None, n] / tall[None, 3] / want - 1), 0.06, tall)
            self.assertLess(abs(tall[None, n] / tall["Kanit", n] - 1), 0.06, "as tall in Kanit: %s" % tall)

    def test_each_part_can_have_a_size_of_its_own(self):
        parts = text.look("pop", sizes={"caption": 5})
        self.assertEqual(parts["title"], text.style("pop"))
        self.assertEqual((parts["caption"].size, parts["big"].size), (5, 3))
        self.assertNotEqual(parts["caption"].name, parts["title"].name, "two sizes are two ASS styles")
        every = text.look("pop", size=2, sizes={"title": 4, "big": None})
        self.assertEqual([every[k].size for k in text.KINDS], [4, 2, 2])
        self.assertEqual(len({st.name for st in text.look("pop", sizes={"title": 4, "caption": 5, "big": 1}).values()}),
                         3, "every size has a name of its own")
        lines = text.caption_lines(self.TWO[:1])
        small, big = text.style("pop"), text.style("pop", size=5)
        heights = []
        for styles in ([small, big], [big, small]):
            frame = draw(text.document(styles, text.captions(lines, big, (1200, 400))), [30], self.folder)[30]
            x0, y0, x1, y1 = drawn(frame)
            heights.append(y1 - y0)
        want = text.measure(lines[0]["text"], big, "caption", 1080).h
        self.assertTrue(all(abs(h - want) <= 3 for h in heights), "drawn at its own size, whatever the order: %s "
                        "against %d" % (heights, want))

    def test_two_lines_too_tall_for_the_band_show_one_at_a_time(self):
        two = text.caption_lines(self.TWO)
        band = (1300, 130)
        pages = lambda events: sorted({tuple(e.split(",")[1:3]) for e in events})
        self.assertEqual(len(pages(text.captions(two, text.style("pop"), band))), 1, "two lines fit at 3")
        big = text.captions(two, text.style("pop", size=5), band)
        self.assertEqual(len(big), 2)
        (a, b), (c, d) = pages(big)
        self.assertLessEqual(b, c, "the second line follows the first")
        self.assertIn("ลดราคาเหลือ", big[0])
        self.assertIn("บาทนะครับ", big[1])
        for frame in draw(doc(big, styles=[text.style("pop", size=5)]), [30, 75], self.folder, first=0).values():
            x0, y0, x1, y1 = drawn(frame)
            self.assertTrue(band[0] <= y0 and y1 <= sum(band), (y0, y1))

    def test_refusals_say_which_sizes_there_are_and_offer_a_smaller_one(self):
        for bad in (0, 6, 2.5, True, "3", None):
            with self.assertRaisesRegex(ValueError, r"size %s is not a size: use a whole number from 1 \(smallest\) "
                                        r"to 5 \(biggest\), 3 for the size tuned on phones" % re.escape(repr(bad))):
                text.style("clean", size=bad)
        for bad in ({"logo": 2}, [3]):
            with self.assertRaisesRegex(ValueError, r"sizes %s is not a size for each part: give \{part: size\} "
                                        r"with the parts title, caption, big and sizes from 1 to 5"
                                        % re.escape(repr(bad))):
                text.look(sizes=bad)
        with self.assertRaisesRegex(ValueError, "size 9 is not a size"):
            text.look(sizes={"caption": 9})
        with self.assertRaisesRegex(ValueError, r"box \(0, 0, 300, 60\) is 300 x 60: give a box at least that size, "
                                    r"or fewer words, or size 3\.$"):
            text.title("ลดราคาเหลือ 1,990", 0, 1, (0, 0, 300, 60), text.style("clean", size=4))
        with self.assertRaisesRegex(ValueError, r"or fewer words\.$"):
            text.title("ลดราคาเหลือ 1,990", 0, 1, (0, 0, 300, 60), text.style("clean", size=1))
        with self.assertRaisesRegex(ValueError, r"give a band at least \d+ high, or size 4\.$"):
            text.captions(text.caption_lines(self.TWO), text.style("pop", size=5), (1300, 50))
        with self.assertRaisesRegex(ValueError, r"use a shorter word, or size 4\.$"):
            text.big_word("WWWWWWWW", 1, 1, (540, 960), text.style("pop", size=5))


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Captions(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="lcs captions ")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_two_lines_at_most_and_pages_never_overlap(self):
        events = text.captions(text.caption_lines(SAID, most=10), text.STYLES["pop"], (1350, 400))
        pages = {}
        for e in events:
            self.assertNotIn("\\N", e, "each line is an event of its own")
            pages.setdefault(tuple(e.split(",")[1:3]), []).append(e)
        self.assertTrue(all(len(lines) <= 2 for lines in pages.values()), "two lines at most")
        spans = sorted(pages)
        for (a, b), (c, d) in zip(spans, spans[1:]):
            self.assertLessEqual(b, c)

    def test_captions_are_drawn_inside_their_band(self):
        band = (1400, 320)
        st = text.STYLES["sticker"]
        events = text.captions(text.caption_lines(SAID), st, band)
        got = draw(doc(events), [45, 150, 240], self.folder, first=0)
        for k, frame in got.items():
            x0, y0, x1, y1 = drawn(frame)
            self.assertTrue(band[0] <= y0 and y1 <= band[0] + band[1], (k, y0, y1))

    def lit_pixels(self, st, lit, frames):
        """The ONE TWO caption (said from 1 to 2 s) drawn at frames: [(pixels in the key colour over ONE, over TWO),
        (pixels in the text colour over the whole line)] per frame."""
        import numpy as np
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "ONE TWO"}])
        self.assertEqual([w["text"] for w in lines[0]["words"]], ["ONE ", "TWO"])
        got = draw(doc(text.captions(lines, st, (1500, 300), lit=lit)), frames, self.folder, first=0)
        x0, y0, x1, y1 = drawn(got[frames[0]])
        one, two = (text.measure(w, st, "caption", 1080).w for w in ("ONE", "TWO"))
        split = x0 + one + ((x1 - x0) - one - two) // 2

        def near(frame, colour, a, b):
            want = [int(colour[k:k + 2], 16) for k in (1, 3, 5)]
            return int((np.abs(frame[y0:y1, a:b].astype(int) - want).max(axis=2) < 40).sum())
        return [(near(got[k], st.key, x0, split), near(got[k], st.key, split, x1), near(got[k], st.colour, x0, x1))
                for k in frames]

    def test_lines_and_outlines_stand_as_they_do_in_kanit_whatever_the_font(self):
        import numpy as np
        two = text.caption_lines([{"from": 0.5, "to": 2.0, "text": "ลดราคาเหลือ"},
                                  {"from": 2.2, "to": 3.0, "text": "บาทนะครับ"}])
        tops = {}
        for font in (None, "Kanit"):
            st = text.style("pop", font)
            frame = draw(doc(text.captions(two, st, (1300, 400)), styles=[st]), [75], self.folder,
                         folder_fonts=text.fonts_folder(st))[75]
            rows = np.nonzero((np.abs(frame.astype(int) - GREY).max(axis=2) > SEEN).any(axis=1))[0]
            breaks = [k for k in range(1, len(rows)) if rows[k] - rows[k - 1] > 1]
            self.assertEqual(len(breaks), 1, "%s: two lines with a gap between them" % font)
            tops[font] = rows[breaks[0]] - rows[0]
            outline = float(text._ass_style(st, "caption").split(",")[16])
            self.assertLess(abs(outline - 68 * 0.10), 0.3, "%s: the outline pop was tuned with in Kanit" % font)
        self.assertLess(abs(tops[None] - tops["Kanit"]), 4, "lines stand as far apart: %s" % tops)

    def test_a_lit_second_line_waits_unlit_until_its_first_word(self):
        two = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "ONE TWO"}, {"from": 2.4, "to": 3.0, "text": "THREE"}])
        first, second = text.captions(two, text.STYLES["clean"], (1300, 400), lit=True)
        self.assertEqual(first.split(",")[1:3], second.split(",")[1:3], "one page, one time")
        waits = sum(int(n) for n in re.findall(r"\\kf?(\d+)", second.split("THREE")[0])[:-1])
        h, m, sec, cs = re.search(r"Dialogue: \d+,(\d+):(\d+):(\d+)\.(\d+)", second).groups()
        begun = ((int(h) * 60 + int(m)) * 60 + int(sec)) * 100 + int(cs)
        self.assertEqual(begun + waits, round(two[1]["words"][0]["from"] * 100), "THREE lights when it is said")

    def test_with_lit_the_word_being_said_lights_up(self):
        (a, b, _), (c, d, _), (_, e, _) = self.lit_pixels(text.STYLES["clean"], True, [30, 38, 53])
        self.assertEqual(a + b, 0)
        self.assertGreater(c, 50)
        self.assertEqual(d, 0)
        self.assertGreater(e, 50)

    def test_plain_captions_by_default_never_light_a_word(self):
        for name in ("clean", "pop"):
            st = text.STYLES[name]
            self.assertNotEqual(st.key.lower(), st.colour.lower())
            for one, two, plain in self.lit_pixels(st, False, [30, 38, 53]):
                self.assertEqual(one + two, 0, name)
                self.assertGreater(plain, 200, name)
        lines = text.caption_lines(SAID)
        self.assertEqual(text.captions(lines, text.STYLES["pop"], (1350, 400)),
                         text.captions(lines, text.STYLES["pop"], (1350, 400), lit=False))
        self.assertNotIn("\\k", "".join(text.captions(lines, text.STYLES["pop"], (1350, 400))))

    def test_once_all_its_words_are_said_a_popped_page_is_the_plain_page(self):
        import numpy as np
        lines = text.caption_lines(POPPED)
        for name in ("pop", "sticker", "glow", "3d"):
            st = text.STYLES[name]
            plain, popped = (draw(doc(text.captions(lines, st, (1300, 400), pop=p)), [103], self.folder, first=0)[103]
                             for p in (False, True))
            self.assertEqual(int(np.abs(plain.astype(int) - popped.astype(int)).max()), 0, name)

    def test_each_word_pops_in_at_its_place_as_it_is_said_and_stays(self):
        import numpy as np
        lines = text.caption_lines(POPPED)
        one, two, three = lines[0]["words"][:3]
        before, alone, growing, said = (text._frame(one["from"]) - 2, text._frame(two["from"]) - 1,
                                        text._frame(two["from"] + 0.1), text._frame(three["from"]) - 1)
        for name in ("pop", "3d"):
            st = text.STYLES[name]
            got = draw(doc(text.captions(lines, st, (1300, 400), pop=True)), [before, alone, growing, said],
                       self.folder, first=0)
            plain = draw(doc(text.captions(lines, st, (1300, 400))), [103], self.folder, first=0)[103]
            whole = drawn(plain)
            self.assertIsNone(drawn(got[before]), "%s: nothing before the first word is said" % name)
            x0, y0, x1, y1 = drawn(got[said])
            self.assertEqual(x0, whole[0], "%s: the first word starts where the line does" % name)
            self.assertLess(x1, whole[2] - 100, "%s: the words not yet said are not drawn" % name)
            self.assertLess(y1, (whole[1] + whole[3]) // 2, "%s: nor the second line" % name)
            near = np.abs(got[said][y0:y1, x0:x1 - 30].astype(int) - plain[y0:y1, x0:x1 - 30].astype(int)).max()
            self.assertEqual(int(near), 0, "%s: the words said so far are the plain page's, one outline and one "
                                           "shadow" % name)
            r1 = drawn(got[alone])[2]
            settled, grown = (drawn(got[k][:, r1 + 10:]) for k in (said, growing))
            self.assertIsNotNone(grown, "%s: the second word is drawn while it pops in" % name)
            self.assertLess(abs((grown[2] - settled[2]) - 0.25 * (settled[2] + 10) / 2), 8,
                            "%s: the second word grows around its own centre: %s %s" % (name, settled, grown))
            self.assertLess(abs((grown[1] + grown[3]) - (settled[1] + settled[3])) / 2, 3,
                            "%s: at its own height: %s %s" % (name, settled, grown))

    def test_popped_lines_are_measured_in_a_few_runs_however_many_there_are(self):
        said = [{"from": 1.0 + 2 * k, "to": 2.6 + 2 * k, "text": "ทดสอบบรรทัดที่ %d ของคำบรรยาย" % k} for k in range(12)]
        st = text.style("pop", size=2)
        with mock.patch.dict(text._boxes, clear=True), \
                mock.patch.object(text, "_render_and_read", wraps=text._render_and_read) as runs:
            text.captions(text.caption_lines(said), st, (1300, 400), pop=True)
        self.assertLessEqual(runs.call_count, 6, "12 lines: %d FFmpeg runs, not two a line" % runs.call_count)

    def test_caption_refusals(self):
        lines = text.caption_lines(SAID)
        with self.assertRaisesRegex(ValueError, "pop 1 is not True or False"):
            text.captions(lines, text.STYLES["clean"], (1500, 300), pop=1)
        with self.assertRaisesRegex(ValueError, "lit and pop are both True: .* so set one of them"):
            text.captions(lines, text.STYLES["clean"], (1500, 300), lit=True, pop=True)
        with self.assertRaisesRegex(ValueError, r"band \(1800, 300\) is off the frame"):
            text.captions(lines, text.STYLES["clean"], (1800, 300))
        with self.assertRaisesRegex(ValueError, r"the captions need \d+ x \d+ px but band \(1500, 40\) is 40 high"):
            text.captions(lines, text.STYLES["clean"], (1500, 40))
        with self.assertRaisesRegex(ValueError, "lit 'yes' is not True or False"):
            text.captions(lines, text.STYLES["clean"], (1500, 300), lit="yes")


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class BigWord(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="lcs big word ")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_it_pops_then_settles_and_is_gone_after_its_length(self):
        for at in (1.0, 0.51):
            events = text.big_word("ว้าว", at, 1.0, (540, 700), text.STYLES["pop"])
            k = {d: text._frame(at + d) for d in (0.1, 0.4, 1.0)}
            got = draw(doc(events), list(k.values()) + [k[0.1] - 4], self.folder, first=0)
            pop, settled = drawn(got[k[0.1]]), drawn(got[k[0.4]])
            area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
            self.assertGreater(area(pop), area(settled) * 1.1, at)
            self.assertLessEqual(abs((settled[0] + settled[2]) / 2 - 540), 2)
            self.assertLessEqual(abs((settled[1] + settled[3]) / 2 - 700), 2)
            self.assertIsNone(drawn(got[k[1.0]]))
            self.assertIsNone(drawn(got[k[0.1] - 4]))

    def test_big_word_refusals(self):
        st = text.STYLES["pop"]
        with self.assertRaisesRegex(ValueError, "length 0.2 is too short for a big word: use 0.35 s or more"):
            text.big_word("POW", 1, 0.2, (540, 700), st)
        with self.assertRaisesRegex(ValueError, r"centre \(10, 700\) puts word 'POW' off the frame: use x from \d+"):
            text.big_word("POW", 1, 1, (10, 700), st)
