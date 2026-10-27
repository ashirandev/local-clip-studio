"""Words on screen, measured on frames libass really drew: python -m unittest discover -s tests"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import effects, text  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
GREY = 128
SEEN = 12


def draw(ass_text, picks, folder, first=None):
    """{frame number: the 1920x1080x3 frame} of the ASS text burned on a grey 30 fps clip. The clip starts at frame
    first (the smallest pick when None), with its own timestamps, as the preview draws a frame."""
    import numpy as np
    path = os.path.join(folder, "words.ass")
    with open(path, "w", encoding="utf-8") as f:
        f.write(ass_text)
    graph, files = effects.join([text.burn(path)], "src", "outv", 1)
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
        with self.assertRaisesRegex(ValueError, "font 'No Such Face 9' is neither in the studio's fonts \\(Kanit"):
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
class Captions(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="lcs captions ")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_two_lines_at_most_and_pages_never_overlap(self):
        events = text.captions(text.caption_lines(SAID, most=10), text.STYLES["pop"], (1350, 400))
        spans = []
        for e in events:
            self.assertLessEqual(e.count("\\N"), 1, e)
            start, end = e.split(",")[1:3]
            spans.append((start, end))
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

    def test_the_word_being_said_lights_up(self):
        st = text.STYLES["clean"]
        lines = text.caption_lines([{"from": 1.0, "to": 2.0, "text": "ONE TWO"}])
        self.assertEqual([w["text"] for w in lines[0]["words"]], ["ONE ", "TWO"])
        events = text.captions(lines, st, (1500, 300))
        got = draw(doc(events), [30, 38, 53], self.folder, first=0)
        import numpy as np
        x0, y0, x1, y1 = drawn(got[30])
        one, two = (text.measure(w, st, "caption", 1080).w for w in ("ONE", "TWO"))
        split = x0 + one + ((x1 - x0) - one - two) // 2
        key = [int(st.key[k:k + 2], 16) for k in (1, 3, 5)]

        def lit(frame, a, b):
            return int((np.abs(frame[y0:y1, a:b].astype(int) - key).max(axis=2) < 40).sum())
        self.assertEqual(lit(got[30], x0, x1), 0)
        self.assertGreater(lit(got[38], x0, split), 50)
        self.assertEqual(lit(got[38], split, x1), 0)
        self.assertGreater(lit(got[53], split, x1), 50)

    def test_caption_refusals(self):
        lines = text.caption_lines(SAID)
        with self.assertRaisesRegex(ValueError, r"band \(1800, 300\) is off the frame"):
            text.captions(lines, text.STYLES["clean"], (1800, 300))
        with self.assertRaisesRegex(ValueError, r"the captions need \d+ x \d+ px but band \(1500, 40\) is 40 high"):
            text.captions(lines, text.STYLES["clean"], (1500, 40))


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
