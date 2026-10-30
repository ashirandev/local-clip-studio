"""The editor core: layouts, faces, the project list, the drawn frame and Muse's editing tools.

    python -m unittest discover -s tests
"""
import base64
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, channels, effects, faces, focus, hearing, layouts, mind, project, recording, render, screen, sends, talk, text, tools, words  # noqa: E402,E501
from test_agent_loop import ScriptedMind, call  # noqa: E402
from test_hearing import Worker  # noqa: E402
from test_map import make_moment  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
SIFU_FACE = [1693.9, 428.7, 92.3, 113.8]


def found(samples, size=(1920, 1080), every=0.5):
    return {"size": list(size), "every": every,
            "samples": [{"t": t, "faces": [list(b) + [0.9] for b in boxes]} for t, boxes in samples]}


def inside(box, size):
    x, y, w, h = box
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= size[0] + 1e-6 and y + h <= size[1] + 1e-6


def overlap(a, b):
    return min(a[0] + a[2], b[0] + b[2]) > max(a[0], b[0]) and min(a[1] + a[3], b[1] + b[3]) > max(a[1], b[1])


class Layouts(unittest.TestCase):
    def test_the_card_puts_the_face_camera_in_the_circle_and_leaves_it_out_of_the_card(self):
        lay = layouts.make("card", (1920, 1080), SIFU_FACE)
        parts = {p["name"]: p for p in lay["parts"]}
        circle, card = parts["circle"], parts["card"]
        self.assertTrue(inside(SIFU_FACE, [circle["src"][0] + circle["src"][2], circle["src"][1] + circle["src"][3]])
                        and circle["src"][0] <= SIFU_FACE[0] and circle["src"][1] <= SIFU_FACE[1])
        self.assertEqual(circle["src"][2], circle["src"][3], "the circle is cut from a square")
        self.assertFalse(overlap(card["src"], layouts._grown(SIFU_FACE, layouts.LEAVE_OUT, 1920, 1080)))
        self.assertIn("The card leaves out the face camera, so the face does not show twice.", lay["notes"])
        self.assertTrue(layouts.CARD_LOWEST <= card["dst"][3] <= layouts.CARD_TALLEST)
        shown = layouts.to_frame(lay, SIFU_FACE)
        self.assertEqual(len(shown), 1, "the face shows once, in the circle")
        cx, cy, d = circle["dst"][0], circle["dst"][1], circle["dst"][2]
        self.assertTrue(cx <= shown[0][0] and shown[0][0] + shown[0][2] <= cx + d + 1e-6 and
                        cy <= shown[0][1] and shown[0][1] + shown[0][3] <= cy + d + 1e-6)

    def test_words_places_stay_clear_of_the_circle_and_the_captions(self):
        lay = layouts.make("card", (1920, 1080), SIFU_FACE)
        circle = [p for p in lay["parts"] if p["name"] == "circle"][0]["dst"]
        for name, box in lay["places"].items():
            self.assertFalse(overlap(box, lay["captions"]), name)
            self.assertGreaterEqual(min(abs(box[1] + box[3] - circle[1]), abs(circle[1] + circle[3] - box[1])), 20,
                                    name)
        self.assertTrue(all(65 <= b[0] and b[0] + b[2] <= 1015 and layouts.SAFE[1] <= b[1] and b[1] + b[3] <= 1250
                            for b in list(lay["places"].values()) + [lay["captions"]]), "inside the phone's safe zone")

    def test_words_muse_puts_in_the_row_kept_for_the_channel_name_are_refused_with_no_name_on_the_clip_yet(self):
        lay, state = layouts.make("card", (1920, 1080), SIFU_FACE), {"captions": None, "channel": None}
        with self.assertRaises(tools.Refused) as row:
            tools._phone([60, 1325, 820, 144], lay, state, asked=False)
        self.assertIn("On a phone, the row kept for the channel's name runs from y 758 to 792 on your grid. Pick one of "
                      "the layout's places, or a box inside x 52 to 948, y 131 to 758 on your grid;", str(row.exception))
        self.assertEqual(tools._phone([60, 1312, 820, 144], lay, state, asked=False), [], "ending at 1456 is clear")
        self.assertEqual(tools.SAFE_ROWS, (250, 1456), "a title behind the person is placed above the row too")

    def test_a_vertical_recording_fills_the_frame_instead_of_a_card(self):
        lay = layouts.make("card", (1080, 1920), [400, 300, 300, 400])
        self.assertEqual(lay["kind"], "subject")
        self.assertIn("already vertical", lay["notes"][0])

    def test_the_face_layouts_need_a_face_and_say_how_to_give_one(self):
        for kind in ("stack", "face"):
            with self.assertRaises(ValueError) as e:
                layouts.make(kind, (1920, 1080), None)
            self.assertIn("Give face as a box on your 0-1000 grid, or choose the subject layout", str(e.exception))
        self.assertIn("no circle", layouts.make("card", (1920, 1080), None)["notes"][0])

    def test_every_part_is_inside_the_picture_whatever_the_boxes(self):
        rng = random.Random(7)
        for _ in range(400):
            w, h = rng.choice([(1920, 1080), (1280, 720), (1080, 1920), (640, 360), (2560, 1440)])
            fw, fh = rng.uniform(10, w / 2), rng.uniform(10, h / 2)
            face = [rng.uniform(-20, w - fw + 20), rng.uniform(-20, h - fh + 20), fw, fh]
            cw, ch = rng.uniform(4, w), rng.uniform(4, h)
            crop = [rng.uniform(0, w - cw), rng.uniform(0, h - ch), cw, ch] if rng.random() < 0.6 else None
            lay = layouts.make(rng.choice(layouts.KINDS), (w, h), face, crop)
            for part in lay["parts"]:
                self.assertTrue(inside(part["src"], (w, h)), (lay["kind"], part, w, h, face, crop))
                self.assertTrue(all(int(v) == v for v in part["src"]), part)
                self.assertTrue(inside(part["dst"], (layouts.W, layouts.H)), part)


class Faces(unittest.TestCase):
    def test_the_face_camera_is_the_face_that_stays_in_one_place(self):
        cam, moving = [1700, 430, 90, 110], [[300 + 80 * i, 500, 120, 150] for i in range(10)]
        f = found([(i * 0.5, ([cam] if i not in (3, 7) else []) + [moving[i]]) for i in range(10)])
        best = faces.steady(f)
        self.assertEqual((best["box"], best["seen"], best["of"]), (cam, 8, 10))
        self.assertEqual(faces.snap([1650, 400, 200, 200], f)["box"], cam)
        self.assertIsNone(faces.snap([100, 100, 50, 50], f))

    def test_a_face_that_moves_is_not_steady_and_two_faces_in_one_picture_are_two_places(self):
        f = found([(i * 0.5, [[200 * i, 300, 100, 120]]) for i in range(8)])
        self.assertIsNone(faces.steady(f))
        two = found([(0, [[100, 100, 90, 110], [120, 105, 90, 110]]), (0.5, [[100, 100, 90, 110]])])
        self.assertEqual([g["seen"] for g in faces.places(two)], [2, 1])

    def test_pictures_are_read_from_the_recording_in_one_pass(self):
        if not FFMPEG:
            self.skipTest("needs ffmpeg")
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "v.mp4")
            make_moment(path, seconds=8, busy=(2, 4), loud=(3, 4))
            work, recording.WORK = recording.WORK, os.path.join(tmp, "work")
            seen, real = [], faces._detector
            faces._detector = lambda w, h: (lambda data, w, h: seen.append((len(data), w, h)) or [(10, 20, 30, 40, .9)])
            try:
                rec = recording.probe(path)
                got = faces.sample(rec, 2.0, 5.0)
                self.assertEqual([s["t"] for s in got["samples"]], [2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
                self.assertEqual(seen[0], (320 * 180 * 3, 320, 180))
                self.assertEqual(got["samples"][0]["faces"], [[10.0, 20.0, 30.0, 40.0, 0.9]])
                self.assertEqual(faces.kept(rec, 2.0, 5.0), got, "kept, so asking again is instant")
            finally:
                faces._detector, recording.WORK = real, work
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Projects(unittest.TestCase):
    def test_undo_takes_the_last_change_back(self):
        p = project.Project()
        p.add({"do": "clip", "moment": 1, "pieces": [[30, 50]], "layout": {"kind": "card"}})
        p.add({"do": "title", "text": "a", "start": 5, "end": 8, "place": "top", "box": [0, 0, 1, 1]})
        p.add({"do": "title", "text": "b", "start": 6, "end": 9, "place": "top", "box": [0, 0, 1, 1]})
        self.assertEqual([t["text"] for t in p.state()["titles"]], ["b"], "same place, overlapping: replaced")
        self.assertEqual(p.undo()["text"], "b")
        self.assertEqual([t["text"] for t in p.state()["titles"]], ["a"])
        p.add({"do": "clip", "moment": 2, "pieces": [[60, 62]], "layout": {"kind": "card"}})
        self.assertEqual(p.state()["titles"], [], "a title that starts after the new clip ends is not drawn")
        self.assertEqual(p.state()["length"], 2)
        with self.assertRaises(ValueError):
            p.add({"do": "nothing"})

    def test_a_mute_stays_on_the_seconds_of_the_recording_when_the_clip_is_cut_again(self):
        pieces = [[30, 40], [50, 60]]
        spans = project.to_recording_spans(pieces, 8, 12)
        self.assertEqual(spans, [[38, 40], [50, 52]], "clip 8-12 s crosses the cut between the pieces")
        p = project.Project([{"do": "clip", "moment": 1, "pieces": pieces, "layout": {"kind": "card"},
                              "speech": [[0, 20]]},
                             {"do": "mute", "spans": spans, "start": 8, "end": 12, "back": False}])
        self.assertEqual(p.state()["mutes"], [[8, 12]], "one stretch where the pieces meet")
        self.assertEqual(p.state()["speech"], [[0, 8], [12, 20]], "nobody is heard there, so the music stays up")
        p.add({"do": "clip", "moment": 1, "pieces": [[35, 40], [50, 60]], "layout": {"kind": "card"}})
        self.assertEqual(p.state()["mutes"], [[3, 7]], "cut 5 s later, the mute moves with the words")
        p.add({"do": "mute", "spans": [[39, 40]], "start": 4, "end": 5, "back": True})
        self.assertEqual(p.state()["mutes"], [[3, 4], [5, 7]], "the sound back over 39-40 s of the recording")
        p.add({"do": "clip", "moment": 2, "pieces": [[0, 10]], "layout": {"kind": "card"}})
        self.assertEqual(p.state()["mutes"], [], "a clip that does not play those seconds has nothing muted")
        p.add({"do": "clip", "moment": 3, "pieces": [[38, 40], [38, 40]], "layout": {"kind": "card"}})
        self.assertEqual(p.state()["mutes"], [[0, 1], [2, 3]], "a stretch the clip plays twice is muted twice")

    def test_the_captions_leave_out_the_words_said_where_the_sound_is_muted(self):
        lines = text.caption_lines([{"from": 0, "to": 2, "text": "one two three"},
                                    {"from": 3, "to": 5, "text": "hello there"}])
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[10, 20]], "layout": {"kind": "card"}},
                             {"do": "captions", "lines": lines, "band": [1500, 200]},
                             {"do": "mute", "spans": [[10, 12.1], [13, 13.9]], "start": 0, "end": 3.9,
                              "back": False}])
        kept = p.state()["captions"]["lines"]
        self.assertEqual(["".join(w["text"] for w in ln["words"]) for ln in kept], [ln["text"] for ln in kept])
        said = "".join(ln["text"] for ln in kept)
        self.assertNotIn("one", said, "the count-in is off the captions")
        self.assertNotIn("hello", said, "and the word whose middle is muted")
        self.assertIn("there", said, "the word after the mute stays")
        self.assertEqual(kept[0]["from"], [w for w in kept[0]["words"]][0]["from"], "the line starts at its first word")
        p.undo()
        self.assertIn("one", "".join(ln["text"] for ln in p.state()["captions"]["lines"]), "undo brings them back")

    def test_a_clip_with_its_ends_moved_keeps_the_footage_things_on_the_footage_and_the_titles_in_place(self):
        lines = text.caption_lines([{"from": 0.5, "to": 2, "text": "one two"}, {"from": 4, "to": 6, "text": "hello there"}])
        before = [{"do": "clip", "moment": 1, "pieces": [[10, 20]], "layout": {"kind": "card"}},
                  {"do": "captions", "lines": lines, "band": [1500, 200]},
                  {"do": "title", "text": "Boss", "start": 0, "end": 3, "place": "top", "box": [0, 0, 1, 1]},
                  {"do": "hit", "kind": "flash", "at": 1.0}, {"do": "hit", "kind": "flash", "at": 5.0},
                  {"do": "sound", "kind": "whoosh", "at": 4.5, "level": 0},
                  {"do": "push", "start": 4, "end": 8, "zoom": 1.2, "centre": [540, 960]},
                  {"do": "blur", "box": [0, 0, 10, 10], "start": 1, "end": 9},
                  {"do": "blur", "track": [[1, 0, 0, 5, 5], [5, 1, 1, 5, 5]], "start": 1, "end": 5}]
        p = project.Project(before)
        was = p.state()
        p.add({"do": "clip", "moment": 2, "pieces": [[13, 21]], "layout": {"kind": "card"}, "moved": True})
        now = p.state()
        words = [w for ln in now["captions"]["lines"] for w in ln["words"]]
        self.assertEqual("".join(w["text"] for w in words).split(), ["hello", "there"], "the cut-off words go")
        first = [w for ln in was["captions"]["lines"] for w in ln["words"] if w["text"].strip() == "hello"][0]
        self.assertAlmostEqual(words[0]["from"], first["from"] - 3, places=3, msg="a word stays on its second")
        self.assertEqual([(t["text"], t["start"], t["end"]) for t in now["titles"]], [("Boss", 0, 3)],
                         "the title still opens the clip")
        self.assertEqual([h["at"] for h in now["hits"]], [2.0], "the hit at 1 s was cut off; 5 s is now 2 s")
        self.assertEqual([s["at"] for s in now["sounds"]], [1.5])
        self.assertEqual([(x["start"], x["end"]) for x in now["pushes"]], [(1.0, 5.0)])
        boxes = [b for b in now["blurs"] if "box" in b]
        self.assertEqual([(b["start"], b["end"]) for b in boxes], [(0.0, 6.0)], "a blur keeps its seconds")
        tracked = [b for b in now["blurs"] if "track" in b]
        self.assertEqual([p_[0] for p_ in tracked[0]["track"]], [2.0], "a face blur's points move with the face")
        p.undo()
        self.assertEqual(p.state(), was, "undo puts the clip back exactly as it was")
        p.add({"do": "clip", "moment": 2, "pieces": [[13, 21]], "layout": {"kind": "card"}})
        self.assertEqual([h["at"] for h in p.state()["hits"]], [1.0, 5.0], "a clip made from another moment does not "
                                                                         "move anything")

    def test_caption_drafts_are_not_lines_and_move_and_go_quiet_with_the_clip(self):
        lines = text.caption_lines([{"from": 0.5, "to": 2, "text": "one two"}])
        drafts = text.caption_lines([{"from": 4, "to": 6, "text": "hello there"}])
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[10, 20]], "layout": {"kind": "card"}},
                             {"do": "captions", "lines": lines, "drafts": drafts, "band": [1500, 200]}])
        self.assertEqual("".join(ln["text"] for ln in p.state()["captions"]["lines"]), "one two",
                         "a draft is never a line, so it is never drawn")
        p.add({"do": "clip", "moment": 1, "pieces": [[13, 21]], "layout": {"kind": "card"}, "moved": True})
        moved = p.state()["captions"]
        self.assertEqual(moved["lines"], [], "the cut-off words go")
        self.assertAlmostEqual(moved["drafts"][0]["words"][0]["from"], drafts[0]["words"][0]["from"] - 3, places=3,
                               msg="a draft stays on its second")
        p.add({"do": "mute", "spans": [[14, 16.5]], "start": 1, "end": 3.5, "back": False})
        self.assertEqual(p.state()["captions"]["drafts"], [], "a draft said where the sound is muted goes")

    def test_a_private_blur_belongs_to_its_clip_and_a_plain_one_is_kept(self):
        p = project.Project()
        p.add({"do": "clip", "moment": 1, "pieces": [[30, 50]], "layout": {"kind": "card"}})
        p.add({"do": "blur", "box": [0, 0, 10, 10], "start": 0, "end": 5, "private": "an email address"})
        p.add({"do": "blur", "box": [0, 0, 10, 10], "start": 0, "end": 5})
        self.assertEqual(len(p.state()["blurs"]), 2)
        p.add({"do": "clip", "moment": 2, "pieces": [[60, 70]], "layout": {"kind": "card"}})
        self.assertEqual([b.get("private") for b in p.state()["blurs"]], [None])

    def test_stickers_are_drawn_in_time_order_and_only_inside_the_clip(self):
        p = project.Project()
        p.add({"do": "clip", "moment": 1, "pieces": [[30, 40]], "layout": {"kind": "card"}})
        for at in (6, 2, 12):
            p.add({"do": "sticker", "png": "s%d.png" % at, "at": at, "length": 1, "box": [0, 0, 100, 100]})
        self.assertEqual([k["png"] for k in p.state()["stickers"]], ["s2.png", "s6.png"])
        p.undo()
        p.undo()
        self.assertEqual([k["png"] for k in p.state()["stickers"]], ["s6.png"])
        self.assertEqual(tools.describe(p.entries[-1]), "the sticker at 6 s")
        self.assertEqual(project.Project().state()["stickers"], [], "no clip, nothing drawn")

    def test_a_sticker_or_glass_box_on_the_same_spot_at_an_overlapping_time_replaces_the_earlier_one(self):
        p = project.Project()
        p.add({"do": "clip", "moment": 1, "pieces": [[30, 50]], "layout": {"kind": "card"}})
        p.add({"do": "sticker", "png": "a.png", "at": 2, "length": 2, "box": [100, 100, 200, 200], "name": "fire"})
        p.add({"do": "sticker", "png": "b.png", "at": 3, "length": 2, "box": [250, 250, 200, 200], "name": "star"})
        p.add({"do": "sticker", "png": "c.png", "at": 4, "length": 2, "box": [600, 600, 100, 100], "name": "bell"})
        p.add({"do": "sticker", "png": "d.png", "at": 5, "length": 2, "box": [100, 100, 200, 200], "name": "gem_stone"})
        self.assertEqual([k["name"] for k in p.state()["stickers"]], ["star", "bell", "gem_stone"],
                         "fire 2-4 s went under star 3-5 s on its spot; gem_stone at 5 s missed fire's time")
        for start, end, x in ((1, 5, 100), (4, 8, 300), (9, 12, 300), (2, 3, 900)):
            p.add({"do": "glass", "box": [x, 900, 400, 200], "start": start, "end": end, "tint": "light", "shine": True})
        self.assertEqual([(g["start"], g["box"][0]) for g in p.state()["glass"]], [(2, 900), (4, 300), (9, 300)])
        self.assertEqual(tools.describe(p.entries[-1]), "the light glass box at 2-3 s")
        self.assertEqual(tools.describe(p.entries[4]), "the sticker gem_stone at 5 s")
        p.add({"do": "glass", "box": [0, 0, 100, 100], "start": 25, "end": 28, "tint": "dark", "shine": False})
        self.assertEqual(len(p.state()["glass"]), 3, "a glass box after the end of the clip is not drawn")

    def test_the_last_look_wins_and_a_push_over_an_earlier_one_replaces_it(self):
        p = project.Project()
        self.assertEqual((p.state()["look"], p.state()["pushes"]), (None, []))
        p.add({"do": "clip", "moment": 1, "pieces": [[30, 50]], "layout": {"kind": "card"}})
        p.add({"do": "look", "name": "vivid"})
        p.add({"do": "look", "name": "warm"})
        self.assertEqual(p.state()["look"], "warm")
        p.add({"do": "push", "start": 2, "end": 6, "zoom": 1.08, "centre": [540, 960]})
        p.add({"do": "push", "start": 12, "end": 16, "zoom": 1.08, "centre": [540, 960]})
        p.add({"do": "push", "start": 5, "end": 9, "zoom": 1.1, "centre": [540, 960]})
        p.add({"do": "push", "start": 25, "end": 28, "zoom": 1.1, "centre": [540, 960]})
        self.assertEqual([(q["start"], q["end"]) for q in p.state()["pushes"]], [(5, 9), (12, 16)],
                         "2-6 overlapped 5-9 and went; 25-28 starts after the 20 s clip")
        p.undo()
        p.undo()
        self.assertEqual([(q["start"], q["end"]) for q in p.state()["pushes"]], [(2, 6), (12, 16)])
        p.add({"do": "look", "name": None})
        self.assertIsNone(p.state()["look"])

    def test_the_last_channel_name_is_drawn_and_an_empty_one_takes_it_off(self):
        p = project.Project()
        p.add({"do": "channel", "name": "@early", "spot": None})
        self.assertIsNone(p.state()["channel"], "no clip yet")
        p.add({"do": "clip", "moment": 1, "pieces": [[30, 50]], "layout": {"kind": "card"}})
        self.assertEqual(p.state()["channel"]["name"], "@early")
        p.add({"do": "channel", "name": "@SineySociety", "spot": None})
        self.assertEqual(p.state()["channel"], {"do": "channel", "name": "@SineySociety", "spot": None})
        p.add({"do": "channel", "name": "", "spot": None})
        self.assertIsNone(p.state()["channel"])
        p.undo()
        self.assertEqual(p.state()["channel"]["name"], "@SineySociety")
        self.assertEqual(tools.describe({"do": "channel", "name": "@SineySociety", "spot": None}),
                         "the channel name @SineySociety on the clip")

    def test_clip_seconds_and_recording_seconds(self):
        pieces = [[100, 110], [200, 205]]
        self.assertEqual(project.to_clip(pieces, 0), (0, 100))
        self.assertEqual(project.to_clip(pieces, 12), (1, 202))
        self.assertEqual(project.to_clip(pieces, 15), (1, 205))
        self.assertIsNone(project.to_clip(pieces, 15.1))
        self.assertEqual(project.from_recording(pieces, 203), 13)
        self.assertIsNone(project.from_recording(pieces, 150))


def pixels(jpg):
    """(width, height, rgb bytes) of a picture."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", jpg],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", jpg, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         capture_output=True, check=True).stdout
    return w, h, raw


def at(picture, x, y):
    w, _, raw = picture
    i = (int(y) * w + int(x)) * 3
    return tuple(raw[i:i + 3])


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Draw(unittest.TestCase):
    """Frames drawn from a 640x360 recording: grey, a red square where the face camera is, a blue one in the game."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.work, recording.WORK = recording.WORK, os.path.join(cls.tmp, "work")
        cls.path = os.path.join(cls.tmp, "stream.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=30:d=6",
                        "-vf", "drawbox=x=520:y=40:w=80:h=80:color=red:t=fill,"
                               "drawbox=x=100:y=200:w=120:h=80:color=blue:t=fill",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", cls.path], check=True)
        cls.rec = recording.probe(cls.path)

    @classmethod
    def tearDownClass(cls):
        recording.WORK = cls.work
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def state(self, kind, titles=(), face=(530, 50, 60, 60)):
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[1, 5]],
                              "layout": layouts.make(kind, (640, 360), list(face) if face else None)}])
        for t in titles:
            p.add(dict({"do": "title", "place": "top", "box": layouts.TOP, "key": None}, **t))
        return p.state()

    def test_the_card_draws_the_face_camera_in_a_ringed_circle_and_the_game_below(self):
        shot = pixels(render.frame(self.rec, self.state("card"), 2.0))
        self.assertEqual(shot[:2], (540, 960))
        half = 0.5
        parts = {p["name"]: p["dst"] for p in self.state("card")["clip"]["layout"]["parts"]}
        circle_y, d, card_y = parts["circle"][1], parts["circle"][2], parts["card"][1]
        cx, cy = 540 * half, (circle_y + d / 2) * half
        r, g, b = at(shot, cx, cy)
        self.assertTrue(r > 180 and g < 80 and b < 80, "red face camera in the circle: %s" % ((r, g, b),))
        r, g, b = at(shot, cx, (circle_y - layouts.RING / 2) * half)
        self.assertTrue(min(r, g, b) > 200, "white ring: %s" % ((r, g, b),))
        row = [at(shot, x, (card_y + 300) * half) for x in range(0, 540, 4)]
        self.assertFalse(any(p[0] > 180 and p[1] < 80 for p in row), "the face camera is not in the card again")
        self.assertTrue(any(p[2] > 150 and p[0] < 80 for p in
                            [at(shot, x, y) for x in range(0, 540, 6) for y in range(int(card_y * half),
                                                                                     int(card_y * half) + 300, 6)]),
                        "the blue square of the game is in the card")

    def test_every_layout_draws_a_whole_vertical_frame(self):
        for kind in layouts.KINDS:
            shot = pixels(render.frame(self.rec, self.state(kind), 1.0))
            self.assertEqual(shot[:2], (540, 960), kind)

    def test_a_title_shows_only_from_its_start_to_its_end(self):
        plain = self.state("card")
        titled = self.state("card", [{"text": "Boss down", "start": 1.0, "end": 2.0}])
        band = range(int(layouts.TOP[1] / 2), int((layouts.TOP[1] + layouts.TOP[3]) / 2), 3)

        def diff(t):
            a, b = pixels(render.frame(self.rec, plain, t)), pixels(render.frame(self.rec, titled, t))
            return max(abs(x - y) for col in range(0, 540, 3) for row in band
                       for x, y in zip(at(a, col, row), at(b, col, row)))

        self.assertGreater(diff(1.5), 100, "the words are drawn at 1.5 s")
        self.assertLessEqual(diff(0.9), 3, "not before the start")
        self.assertLessEqual(diff(2.1), 3, "not after the end")

    def test_a_frame_outside_the_clip_says_so(self):
        with self.assertRaises(render.NotDrawn) as e:
            render.frame(self.rec, self.state("card"), 9.0)
        self.assertIn("outside the clip, which is 4.00 s long", str(e.exception))


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Tools(unittest.TestCase):
    """Muse's editing tools in a scripted job, on a 320x180 recording with faces given by a fake face finder: one
    that stays low in the picture, and from 35 s to 40 s one that rises to the top."""

    @classmethod
    def setUpClass(cls):
        cls.made = tempfile.mkdtemp()
        cls.source = os.path.join(cls.made, "moment.mp4")
        make_moment(cls.source)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.made, ignore_errors=True)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.videos = os.path.join(self.tmp, "videos")
        os.makedirs(self.videos)
        self.clip = os.path.join(self.videos, "stream.mp4")
        shutil.copy(self.source, self.clip)
        os.environ["LCS_VIDEOS"] = self.videos
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        self.smallest, recording.SMALLEST = recording.SMALLEST, 0
        self.worker, self.url = Worker(), hearing.URL
        hearing.URL = self.worker.url
        self.real = faces.sample, faces.kept
        self.faces = found([(30 + i * 0.5, [[140, 10, 40, 40] if 35 <= 30 + i * 0.5 < 40 else [140, 120, 40, 40]])
                            for i in range(40)], size=(320, 180))
        self.looked = []

        def sample(rec, a, b, every=0.5, on_start=None):
            self.looked.append((a, b))
            return self.faces

        faces.sample, faces.kept = sample, lambda rec, a, b, every=0.5: self.faces if self.looked else None
        self.chat = agent.Chat()

    def tearDown(self):
        faces.sample, faces.kept = self.real
        hearing.URL = self.url
        self.worker.close()
        recording.SMALLEST = self.smallest
        agent.CHATS.clear()
        agent._shown.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_job(self, *script):
        brain = ScriptedMind(*script)
        self.chat.jobs += 1
        job = self.chat.job = agent.Job(self.chat, brain, "make a clip")
        job.run()
        return brain, [e for e in self.chat.events if e.get("job") == job.n and e["kind"] == "step"]

    def to_moment(self):
        return [[call("open_recording", path=self.clip, why="open it")],
                [call("look_at_piece", start="0:30", end="0:50", why="the top stretch")],
                [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the tone rings"}])]]

    def test_a_person_who_asked_to_see_the_moments_first_gets_the_cards_and_no_clip_until_a_card_is_tapped(self):
        script = self.to_moment()
        wait = [call("propose_moments", why="they asked to see the highlights first", then="wait", plan_step=1,
                     moments=[{"look": 1, "reason": "the tone rings"}])]
        _, steps = self.run_job([call("set_plan", steps=["find the moments", "make the clip", "caption it"], why="plan")],
                                *script[:2], wait, "The cards are ready.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertIn("no clip is made in this job: answer now", steps[-1]["result"])
        self.assertNotIn("go on with moment 1 by yourself", steps[-1]["result"])
        answers = [e for e in self.chat.events if e["kind"] == "answer" and e.get("job") == self.chat.job.n]
        self.assertEqual([(a["text"], a["final"]) for a in answers], [("The cards are ready.", True)],
                         "the plan's open step does not send Muse on to make the clip")
        self.assertIsNone(self.chat.project.state()["clip"])
        _, steps = self.run_job(wait, [call("make_clip", moment=1, layout="subject", why="go on with the best")],
                                "The cards are ready.")
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("the person asked to see the moments first", steps[-1]["result"])
        self.assertIsNone(self.chat.project.state()["clip"])
        _, steps = self.run_job([call("make_clip", moment=1, layout="subject", why="the person tapped the card")],
                                "Done.")
        self.assertFalse(steps[-1]["refused"], "a tap on a card is a new job, which makes the clip")

    def test_the_clip_is_made_from_a_moment_and_muse_sees_its_frame(self):
        brain, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                                    "Done.")
        step = steps[-1]
        self.assertFalse(step["refused"], step["result"])
        self.assertEqual(self.looked, [(30.0, 50.0)], "the faces are looked for in the moment")
        self.assertIn("Made the clip from moment 1 (0:30-0:50, 20 s) in the subject layout.", step["result"])
        self.assertIn("A face sits at the same place in 30 of 40 pictures, at [438, 667, 562, 889] on your grid; the "
                      "layout uses it.", step["result"])
        self.assertIn("- top (y 149 to 270): clear of faces at 0-4.7 s, 9.8-20 s", step["result"])
        self.assertIn("- middle (y 472 to 559): clear of faces for the whole clip", step["result"])
        self.assertIn("comes next as a picture", step["result"])
        self.assertIn("Other people's faces in the clip: 1 (the faces the layout shows are not among them).", step["result"])
        self.assertIn("- person 1: seen in 10 of 40 pictures, 0:05-0:10 of the clip, at [438, 56, 562, 278] on your grid; "
                      "decide before you render; its face comes next as a picture", step["result"])
        sent = brain.sent[-1][-1]
        self.assertEqual([p["type"] for p in sent["content"]], ["text", "image_url", "text", "image_url"])
        self.assertTrue(sent["content"][0]["text"].startswith("[studio] The clip at "))
        self.assertEqual(sent["content"][2]["text"], "Person 1:")
        self.assertIn("clip: moment 1 (0:30-0:50, 20 s), subject layout, clean style, no words on screen, no captions, other "
                      "people: 0 blurred, 0 kept, 1 still to decide before rendering · no finished file yet: the person sees "
                      "only a preview on the page, so never say the clip is ready", brain.sent[-1][-2]["content"])
        edits = [e for e in self.chat.events if e["kind"] == "edit"]
        self.assertEqual(edits[-1]["clip"]["layout"], "subject")

    def test_a_title_over_a_face_stands_in_the_clear_part_of_its_place_or_is_refused_with_the_clear_times(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_title", words="Boss down", start=0, end=15, place="top", why="the hook")],
            [call("add_title", words="The boss goes down after the longest fight of the whole night", start=0, end=15,
                  place="top", why="the long hook")],
            [call("add_title", words="Boss down", start=0, end=4.5, place="top", why="before the face rises")],
            "Done.")
        beside, refused, drawn = steps[-3:]
        self.assertFalse(beside["refused"], beside["result"])
        self.assertIn("Centred, it would cover a face at 4.8-9.8 s, so it stands in the part of the top that is clear: "
                      "x 61 to 294 on your grid.", beside["result"],
                      "the seller beat 10-29: no place was clear across its width and the clip went out untitled")
        placed = [e["entry"] for e in self.chat.events if e["kind"] == "edit" and e["entry"].get("do") == "title"]
        self.assertEqual(placed[0]["box"], [65, 285, 253, 235])
        self.assertTrue(refused["refused"], "too long for the clear part")
        self.assertIn("At the top, the title would cover a face at 4.8-9.8 s. Clear: top: 0-4.7 s, 9.8-20 s; middle: "
                      "0-20 s. Pick a time or a place from these.", refused["result"])
        self.assertFalse(drawn["refused"], drawn["result"])
        self.assertIn('The title "Boss down" is on screen at the top from 0 to 4.5 s: 1 line', drawn["result"])
        self.assertIn('words on screen: "Boss down" 0-4.5 s at the top', brain.sent[-1][-2]["content"])

    def test_with_no_place_clear_across_its_width_the_title_is_offered_the_clear_part_of_the_opening(self):
        """The seller beat 10-29: the face filled the top place's middle all clip long and the product the middle
        place, so no place was clear and the clip went out with no title; the top's left was clear at the start."""
        from unittest import mock

        def times(state, rec):
            return [(k * 0.5, [[480, 300, 300, 400] if k * 0.5 < 8 else [65, 300, 950, 400]]) for k in range(41)], 0.5

        with mock.patch.object(tools, "_face_times", times):
            brain, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_title", words="Boss down", start=0, end=15, place="top", why="the hook")],
                [call("add_title", words="Boss down", start=0, end=7.5, place="top", why="the clear part")], "Done.")
        made, refused, drawn = steps[-3:]
        self.assertIn("- top (y 149 to 270): never clear across its width, but from 0 to 7.5 s its part from x 61 to "
                      "425 is: a title from the first frame given this place goes there when it fits", made["result"])
        self.assertTrue(refused["refused"])
        self.assertIn("Clear: top: its part at x 61 to 425, y 149 to 270 on your grid at 0-7.5 s;", refused["result"])
        self.assertFalse(drawn["refused"], drawn["result"])
        self.assertIn("so it stands in the part of the top that is clear: x 61 to 425 on your grid.", drawn["result"])
        self.assertIsNone(tools._clear_part(layouts.make("subject", (1080, 1920))["places"]["middle"], "Boss down",
                                            render.style_of({"style": None}, "clean"), None, [], 0.5, 0, 4,
                                            [[562, 538, 497, 1152]]),
                          "never beside the product: it moves in its shot (the dehumidifier crossed the clear left)")

    def test_the_clear_part_a_refusal_names_sent_back_as_a_box_is_where_the_title_is_drawn(self):
        """The seller beat 10-29: the refusals named the clear part in pixels of the frame ("x 755 to 1015"), Muse sent
        them back as a box on its 0-1000 grid, and every title landed off the clear part until its 30 steps ran out."""
        import re
        from unittest import mock

        def times(state, rec):
            return [(k * 0.5, [[480, 300, 300, 400] if k * 0.5 < 8 else [65, 300, 950, 400]]) for k in range(41)], 0.5

        def from_the_refusal(asking):
            said = [e for e in self.chat.events if e["kind"] == "step"][-1]["result"]
            x1, x2, y1, y2, end = re.search(r"its part at x (\d+) to (\d+), y (\d+) to (\d+) on your grid at 0-([\d.]+) "
                                            r"s", said).groups()
            return [call("add_title", words="Boss down", start=0, end=float(end),
                         box=[int(x1), int(y1), int(x2), int(y2)], why="the clear part, as told")]

        with mock.patch.object(tools, "_face_times", times):
            brain, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_title", words="Boss down", start=0, end=15, place="top", why="the hook")],
                from_the_refusal, "Done.")
        refused, drawn = steps[-2:]
        self.assertTrue(refused["refused"])
        self.assertFalse(drawn["refused"], drawn["result"])
        x, y, w, h = [e["entry"] for e in self.chat.events if e["kind"] == "edit"
                      and e["entry"].get("do") == "title"][-1]["box"]
        self.assertTrue(65 <= x and x + w <= 460 and 285 <= y and y + h <= 520,
                        "inside the clear part, x 65 to 460 and y 285 to 520 of the frame: %s" % [x, y, w, h])

    def test_a_vlogs_card_shows_its_faces_bigger(self):
        moment = self.to_moment()
        moment[0] = [call("open_recording", path=self.clip, why="open it"),
                     call("load_recipe", type="vlog", why="an episode")]
        brain, steps = self.run_job(*moment, [call("make_clip", moment=1, layout="card", why="the scene")], "Done.")
        step = steps[-1]
        self.assertFalse(step["refused"], step["result"])
        self.assertIn("the card shows [162, 0, 838, 1000] on your grid", step["result"])
        lay = [e["entry"] for e in self.chat.events if e["kind"] == "edit" and e["entry"].get("do") == "clip"][-1]["layout"]
        (card,) = [p for p in lay["parts"] if p["name"] == "card"]
        self.assertEqual(card["src"], [52, 0, 216, 180])
        self.assertEqual(card["dst"][3], 800, "taller than a whole 16:9 picture's 540 px card")

    def test_wrong_title_input_says_what_to_send(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="card", why="the game")],
            [call("add_title", words="Hi", start=25, end=30, place="top", why="late")],
            [call("add_title", words="Hi", start=1, end=3, place="corner", why="corner")],
            [call("add_title", words="Hi there", start=1, end=3, place="top", key="you", why="key")],
            [call("make_clip", moment=2, layout="card", why="no such")],
            "Done.")
        results = [s["result"] for s in steps[-4:]]
        self.assertIn("start 25 s is outside the clip, which is 20 s long: use a start from 0 to 19", results[0])
        self.assertIn("The card layout has no place called corner; its places are: top.", results[1])
        self.assertIn("key 'you' is not one of the title's words", results[2])
        self.assertIn("There are 1 moments, so moment 2 is not one of them: 1. 0:30-0:50.", results[3])
        self.assertTrue(all(s["refused"] for s in steps[-4:]))

    def test_undo_by_muse_and_by_the_person_and_the_chat_opens_again_with_the_clip(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_title", words="Boss down", start=0, end=4, place="top", why="the hook")],
            [call("undo", why="too early")], "Done.")
        self.assertIn('Took back the title "Boss down" at 0-4 s.', steps[-1]["result"])
        self.assertIn("no words on screen", brain.sent[-1][-1]["content"])
        self.assertEqual(self.chat.undo()["do"], "clip")
        self.assertIsNone(self.chat.project.state()["clip"], "the person took the clip itself back")
        self.assertEqual(self.chat.told, ["the person took back the clip from moment 1 in the subject layout"])
        self.assertIsNone(self.chat.undo(), "nothing left to take back")
        again = agent.Chat.load(self.chat.id)
        self.assertEqual(again.project.entries, self.chat.project.entries)
        self.chat.edit({"do": "style", "name": "pop", "font": None, "colour": None})
        self.assertEqual(agent.Chat.load(self.chat.id).project.entries, self.chat.project.entries)

    def test_captions_are_proofread_and_the_line_said_is_spelled_as_the_caption(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "ตังค่าเครื่องง่ายมาก"},
                               {"from": 5.0, "to": 8.0, "text": "ราคาถูกมาก"}]
        script = self.to_moment()
        _, steps = self.run_job(
            *script[:2], [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the setup"}])],
            '{"lines": ["ตังค่าเครื่องง่ายมาก", "ราคาถูกมาก"]}',
            [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_captions", why="say what he says")],
            '{"texts": ["ตั้งค่าเครื่องง่ายมาก", "ราคาแพงมาก"]}', "Done.")
        caps = steps[-1]
        self.assertFalse(caps["refused"], caps["result"])
        self.assertIn('The program fixed the Thai spelling: "ตังค่าเครื่องง่ายมาก" is now "ตั้งค่าเครื่องง่ายมาก".',
                      caps["result"])
        shown = "".join(line["text"] for line in self.chat.project.state()["captions"]["lines"])
        self.assertIn("ตั้งค่า", shown)
        self.assertIn("ราคาถูกมาก", shown, "a change of words, not of spelling, is refused")
        self.assertEqual([p["text"] for p in self.chat.moments[0]["said"]],
                         ["ตั้งค่าเครื่องง่ายมาก", "ราคาถูกมาก"], "the check reads the line as the caption shows it")

    def test_a_recording_talked_through_gets_its_captions_one_big_line_at_a_time(self):
        said = ["เครื่องนี้ดูดความชื้นได้ดีมากจริงๆครับ", "บ้านไหนชื้นมากก็เก็บน้ำได้เต็มถังเลย"]
        self.worker.phrases = [{"from": 1.0, "to": 9.0, "text": said[0]}, {"from": 9.2, "to": 18.5, "text": said[1]}]
        _, steps = self.run_job(
            *self.to_moment()[:2],
            [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the review"}])],
            json.dumps({"lines": said}, ensure_ascii=False),
            [call("make_clip", moment=1, layout="subject", why="the review")],
            [call("add_captions", why="say what he says")],
            json.dumps({"texts": said}, ensure_ascii=False), "Done.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertTrue(self.chat.project.state()["captions"].get("one"), "a seller's review talked through 30 s had "
                        "two small lines, as only clips joined by arrange_clips were found talked")
        talk = [{"from": 0.0, "to": 16.0}]
        self.assertTrue(tools._talked({}, talk, 20.0))
        self.assertFalse(tools._talked({"recipe": "game"}, talk, 20.0), "a game keeps them small, under the action")
        self.assertFalse(tools._talked({}, talk, 20.0, translated=True), "ONE_LETTERS was measured on Thai")
        self.assertFalse(tools._talked({"talked": False}, talk, 20.0), "arrange_clips found it not talked")
        self.assertFalse(tools._talked({}, [{"from": 0.0, "to": 11.0}], 20.0))

    def test_each_caption_line_has_the_word_a_viewer_must_not_miss_in_the_key_colour_when_it_is_a_whole_word(self):
        said = ["เครื่องนี้ดูดความชื้นได้ดีมากจริงๆครับ", "บ้านไหนชื้นมากก็เก็บน้ำได้เต็มถังเลย"]
        self.worker.phrases = [{"from": 1.0, "to": 9.0, "text": said[0]}, {"from": 9.2, "to": 18.5, "text": said[1]}]
        keys = ["เครื้อง", "ความชื้น", "", "บ้านไหน", "น้ำได้", "ถังเ"]
        brain, steps = self.run_job(
            *self.to_moment()[:2],
            [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the review"}])],
            json.dumps({"lines": said}, ensure_ascii=False),
            [call("make_clip", moment=1, layout="subject", why="the review")],
            [call("add_captions", why="say what he says")], json.dumps({"texts": said}, ensure_ascii=False),
            json.dumps({"keys": keys}, ensure_ascii=False), "Done.")
        caps = steps[-1]
        self.assertFalse(caps["refused"], caps["result"])
        got = self.chat.project.state()["captions"]
        self.assertEqual([ln["text"].strip() for ln in got["lines"]],
                         ["เครื่องนี้ดูด", "ความชื้นได้ดีมาก", "จริงๆครับ", "บ้านไหนชื้นมากก็", "เก็บน้ำได้เต็ม", "ถังเลย"])
        self.assertEqual([ln.get("key") for ln in got["lines"]],
                         ["เครื่อง", "ความชื้น", None, "บ้านไหน", "น้ำได้", None],
                         "Glimmer's slip put back; half a word (ถังเ) leaves its line plain")
        self.assertTrue(got["pop"], "a clip talked through pops its words in as they are said")
        self.assertIn("In 4 of the 6 lines the word a viewer must not miss is in the key colour.", caps["result"])
        self.assertIn(tools.KEYS_SCHEMA, brain.schemas)
        self.assertIsNone(self.chat.set_captions("plain"))
        tapped = self.chat.project.state()["captions"]
        self.assertFalse(tapped["pop"], "Plain on the page is plain")
        self.assertEqual([ln.get("key") for ln in tapped["lines"]], [ln.get("key") for ln in got["lines"]],
                         "a tap never asks Muse and keeps the keys")

    def test_english_captions_are_whole_sentences_translated_and_checked_against_what_was_said(self):
        self.worker.phrases = [{"from": 2.0, "to": 2.6, "text": "ตั้งค่าเครื่อง"}, {"from": 2.8, "to": 4.0, "text": "ง่ายมาก"},
                               {"from": 5.0, "to": 8.0, "text": "ราคา 990 บาท"}]

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            brain, steps = self.run_job(
                *self.to_moment()[:2],
                [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the setup"}])],
                '{"lines": ["ตั้งค่าเครื่อง", "ง่ายมาก", "ราคา <N1> บาท"]}',
                [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_captions", language="english", why="for viewers abroad")],
                '{"lines": ["Setting it up is so easy.", "It costs 990 baht."]}',
                [call("decide_people", keep=[1], why="it is him")],
                [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                [call("add_captions", language="English", lit=True, why="lit")], "Done.")
            lit = steps[-1]
            caps = self.chat.project.state()["captions"]
            self.chat.edit(dict(caps, lines=[dict(ln, text=ln["text"].replace("990", "99"), words=[
                dict(w, text=w["text"].replace("990", "99")) for w in ln["words"]]) for ln in caps["lines"]]))
            _, again = self.run_job([call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                                    "Done.")
            _, fixed = self.run_job([call("add_captions", language="English", why="the check")],
                                    '{"lines": ["Setting it up is so easy.", "It costs 990 baht."]}',
                                    [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                                    "Done.")
        finally:
            render.export = had
        caps, _, _, checked = steps[-5:-1]
        wrong = again[-1]
        self.assertFalse(lit["refused"], lit["result"])
        self.assertEqual(len([m for m in brain.sent if len(m) == 1 and "English subtitle" in m[0]["content"]]), 1,
                         "captions put on again lit keep the translation the clip holds")
        self.assertFalse(caps["refused"], caps["result"])
        self.assertRegex(caps["result"], r"^Captions on: \d+ lines in English, translated from the 2 sentences said in "
                                         r"Thai, numbers kept")
        self.assertIn("write the title, the words on screen and the post in English too", caps["result"])
        asked = [m for m in brain.sent if len(m) == 1 and "English subtitle" in m[0]["content"]]
        self.assertIn("in Thai", asked[0][0]["content"])
        self.assertIn("\n\n1. (2.0 s) ตั้งค่าเครื่อง ง่ายมาก\n2. (3.0 s) ราคา 990 บาท", asked[0][0]["content"],
                      "a phrase 0.2 s after a short one is the same sentence: one subtitle for both")
        self.assertFalse(checked["refused"], checked["result"])
        self.assertIn("- passed: Captions are words that were said (", checked["result"])
        self.assertIn("- passed: Translated captions keep the numbers (2 of 2 sentences keep every number said)",
                      checked["result"])
        self.assertIn('- FAILED: Translated captions keep the numbers (1 of 2 sentences keep every number said; not '
                      'kept: "It costs 99 baht." for "ราคา 990 บาท"; put the captions on again and those lines are '
                      'translated again)', wrong["result"])
        self.assertIn("- passed: Translated captions keep the numbers (2 of 2 sentences keep every number said)",
                      fixed[-1]["result"], "Eaglet 10-28: the same translation was kept and the clip was never handed "
                                           "over")
        captions = self.chat.project.state()["captions"]
        self.assertEqual(captions["language"], "English")
        self.assertEqual("".join(line["text"] for line in captions["lines"]),
                         "Setting it up is so easy.It costs 990 baht.")
        self.assertEqual({line["said"] for line in captions["lines"]}, {"ตั้งค่าเครื่อง ง่ายมาก", "ราคา 990 บาท"})

    def test_sam_finds_the_person_once_and_a_title_goes_behind_them_only_where_they_cross_it(self):
        asked, broken = [], []

        def cut_out(rec, start, end, face=None, on_progress=None, on_start=None, people=None):
            asked.append((start, end, face))
            self.assertIsNone(people, "one person in a subject layout")
            on_progress(5, 10)
            if broken:
                raise tools.person.PersonError("Muse Glimmer could not step off the graphics card for SAM (refused).")
            path = os.path.join(self.tmp, "person.mkv")
            subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=10:d=%g"
                            % (end - start), "-vf", "drawbox=x=130:y=110:w=60:h=70:color=white:t=fill,drawbox=x=140:"
                            "y=38:w=40:h=142:color=white:t=fill:enable='between(t,5,10)',format=gray", "-c:v", "ffv1",
                            path], check=True)
            return {"path": path, "person": "main", "fps": 10, "size": [320, 180], "from": start, "to": end}

        had = tools.person.cut_out
        tools.person.cut_out = cut_out
        try:
            _, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_title", words="Boss down", start=5, end=9, place="top", behind=True, why="cover")],
                [call("cut_out", look="grey", why="make him stand out")],
                [call("cut_out", look="outline", start=2, end=4, why="an edge")],
                [call("cut_out", look="grey", start=-5, why="the last 5 seconds")],
                [call("add_title", words="Boss down", start=12, end=16, place="top", behind=True, why="cover")],
                [call("add_title", words="Still here", start=12, end=16, box=[60, 546, 940, 668], behind=True,
                      why="on his head")],
                [call("add_title", words="Boss down", start=5, end=9, place="middle", behind=True, why="cover")],
                [call("add_title", words="Boss down", start=5, end=9, place="top", behind=True, why="cover")],
                [call("cut_out", look="glow", why="a glow")], "Done.")
            early, grey, outline, last, missed, moved, hidden, title, glow = steps[-9:]
            pieces = self.chat.project.state()["clip"]["pieces"]
            face = self.chat.project.state()["clip"]["layout"].get("face")
            state = self.chat.project.state()
            broken.append(True)
            self.chat.edit({"do": "mask", **{k: v for k, v in state["mask"].items() if k != "do"}, "from": 0, "to": 1})
            _, failed = self.run_job([call("cut_out", why="again")], "Done.")
        finally:
            tools.person.cut_out = had
        self.assertTrue(early["refused"])
        self.assertIn("call cut_out first", early["result"])
        self.assertFalse(grey["refused"], grey["result"])
        self.assertIn("SAM 3.1 found the person through the clip, 0:30 to 0:50 of the recording, 10 pictures a second",
                      grey["result"])
        self.assertIn("The grey look is on from 0 to 20 s of the 20 s clip:", grey["result"])
        self.assertIn("On the frame the person is at [215, 612, 807, 1000] on your grid. A title goes behind them "
                      "only where they cross it, and it must still be read, so "
                      "the top of their head should cut only the bottom of the letters: the places for words that "
                      "cross them: top at 5-10.2 s; middle at 5-10.2 s; for one line of words on top of their head, a "
                      "box such as [60, 546, 824, 668].", grey["result"],
                      "the mask stands up from 5 to 10 s, low the rest")
        self.assertIn("The person was already found for this clip, so SAM did not run again.", outline["result"])
        self.assertIn("The outline look is on from 2 to 4 s of the 20 s clip, so the clip's last 16 s are without it",
                      outline["result"])
        self.assertIn("The grey look is on from 15 to 20 s of the 20 s clip:", last["result"],
                      "lane 5 10-28: 'the last 5 seconds' of a 16.79 s clip came out as 10-15")
        self.assertTrue(missed["refused"])
        self.assertIn("Behind the person, the title must cross them for at least 1 s, and at the top they cross its "
                      "words (y 196 to 223 on your grid) for 0 s of 12 to 16 s; they are at [215, 612, 807, 1000] on your grid",
                      missed["result"], "lane 5 10-28: a behind title above the face circle went behind nobody")
        self.assertIn("Give a box such as [60, 546, 940, 668], on top of their head", missed["result"])
        self.assertFalse(moved["refused"], moved["result"])
        self.assertIn("Behind the person: they cross the words from 12 to 16 s and hide about 0% of their width.",
                      moved["result"], "the box the refusal names is one that crosses them and can be read")
        self.assertTrue(hidden["refused"])
        self.assertIn('Behind the person, the words must still be read, and at the middle the person hides more than '
                      'a third of their width for 4 s of 5 to 9 s (about 100% most of the time), so a viewer cannot '
                      'read "Boss down". Put it higher, so the top of their head cuts only the bottom of the letters: '
                      'a box such as [60, 146, 940, 268]', hidden["result"],
                      "lane 5 10-28: the head hid half the title for the whole clip")
        self.assertEqual(asked[0], (min(p[0] for p in pieces), max(p[1] for p in pieces), face))
        self.assertEqual(state["mask"]["path"], os.path.join(self.tmp, "person.mkv"))
        self.assertEqual([(c["look"], c["start"], c["end"]) for c in state["cutouts"]],
                         [("outline", 2, 4), ("grey", 15, 20)], "a look at an overlapping time replaces the one before")
        self.assertFalse(title["refused"], title["result"])
        self.assertIn("Behind the person: they cross the words from 5 to 9 s and hide about 0% of their width.",
                      title["result"])
        self.assertIn("It crosses a face at", title["result"])
        self.assertIn("the person stands in front of it", title["result"])
        self.assertTrue([t for t in state["titles"] if t["text"] == "Boss down"][0]["behind"])
        self.assertTrue(glow["refused"])
        self.assertIn("look must be one of: grey, outline.", glow["result"])
        self.assertIn("The person could not be cut out: Muse Glimmer could not step off the graphics card for SAM "
                      "(refused). Tell the person.", failed[-1]["result"])
        self.assertEqual(len(asked), 2, "SAM runs again only when the clip's stretch is not covered")
        self.assertEqual(self.chat.project.state()["mask"]["to"], 1, "a failed run leaves the clip as it was")

    def test_in_a_talk_sam_finds_each_person_and_the_look_follows_whoever_speaks(self):
        """Tarm 10-30 on four people talking: one person kept in colour through the whole clip "มันดูตลก"."""
        self.faces = found([(30 + i * 0.5, [[60, 40, 40, 40], [220, 40, 40, 40]]) for i in range(40)], size=(320, 180))
        self.worker.phrases = [{"from": 1, "to": 9, "text": "hello there", "sure": 0.99},
                               {"from": 10.2, "to": 19, "text": "and you", "sure": 0.99}]

        def moving(rec, a, b, persons, fps=10):
            first = [a + (i + 1) / fps < 40 for i in range(round((b - a) * fps))]
            return [[9.0 if f else 1.0 for f in first], [1.0 if f else 9.0 for f in first]]

        asked = []

        def cut_out(rec, start, end, face=None, on_progress=None, on_start=None, people=None):
            asked.append(people)
            path = os.path.join(self.tmp, "person.mkv")
            subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=10:d=%g"
                            % (end - start), "-vf", "drawbox=x=50:y=30:w=60:h=150:color=white:t=fill,format=gray",
                            "-c:v", "ffv1", path], check=True)
            made = {"path": path, "person": None, "fps": 10, "size": [320, 180], "from": start, "to": end}
            return dict(made, each=[path, None]) if people else made

        real, talk.motion = talk.motion, moving
        had, tools.person.cut_out = tools.person.cut_out, cut_out
        try:
            _, steps = self.run_job(
                *self.to_moment(), '{"lines": ["hello there", "and you"]}',
                [call("make_clip", moment=1, layout="talk", why="two people talking")],
                [call("cut_out", look="grey", why="the one speaking")],
                [call("cut_out", look="outline", why="an edge")], "Done.")
            state = self.chat.project.state()
            self.chat.edit({"do": "mask", **{k: v for k, v in state["mask"].items() if k not in ("do", "each")}})
            _, again = self.run_job([call("cut_out", look="grey", why="a mask from before")], "Done.")
        finally:
            talk.motion, tools.person.cut_out = real, had
        grey, outline = steps[-2:]
        boxes = [p["box"] for p in state["clip"]["layout"]["people"]]
        self.assertEqual(asked, [boxes, boxes], "each person's face, and again for a mask made before")
        self.assertIn("SAM 3.1 found 1 of the 2 people through the clip", grey["result"])
        self.assertIn("The grey look is on from 0 to", grey["result"])
        self.assertIn("the footage is black and white around whoever speaks, who keeps their colour while they speak",
                      grey["result"])
        self.assertIn("The person was already found for this clip, so SAM did not run again.", outline["result"])
        self.assertIn("a white edge goes around whoever speaks, while they speak", outline["result"])
        self.assertIn("SAM 3.1 found 1 of the 2 people", again[-1]["result"])

    def test_a_line_that_leaves_out_a_number_is_asked_once_with_the_number_then_stands_and_is_named(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "นะครับสามก่อนเข้าไปดู"},
                               {"from": 5.0, "to": 8.0, "text": "ราคา 990 บาท"}]

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            brain, steps = self.run_job(
                *self.to_moment()[:2],
                [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the site"}])],
                '{"lines": ["นะครับ<N1>ก่อนเข้าไปดู", "ราคา <N2> บาท"]}',
                [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_captions", language="English", why="for viewers abroad")],
                '{"lines": ["You can go see it.", "It costs 99 baht."]}',
                '{"lines": ["You can go see it.", "It costs 990 baht."]}',
                [call("decide_people", keep=[1], why="it is him")],
                [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                [call("add_captions", language="English", lit=True, why="lit")], "Done.")
        finally:
            render.export = had
        caps, checked, lit = steps[-5], steps[-2], steps[-1]
        asked = [m for m in brain.sent if len(m) == 3 and "do not carry the numbers" in m[2]["content"]]
        self.assertEqual(len(asked), 1)
        self.assertIn('1. "นะครับสามก่อนเข้าไปดู" says 3; your line: "You can go see it."\n2. "ราคา 990 บาท" says 990; '
                      'your line: "It costs 99 baht."', asked[0][2]["content"])
        self.assertEqual(json.loads(asked[0][1]["content"])["lines"], ["You can go see it.", "It costs 99 baht."])
        self.assertIn("numbers kept but in 1 where you were shown the number and left it out", caps["result"])
        self.assertIn('- passed: Translated captions keep the numbers (1 of 2 sentences keep every number said; Muse was '
                      'shown the number and wrote it as no number, which the person can check: "You can go see it." '
                      'for "นะครับสามก่อนเข้าไปดู")', checked["result"], "Eaglet 10-28: four renders, then a red check")
        self.assertFalse(lit["refused"], lit["result"])
        self.assertEqual(len([m for m in brain.sent if "English subtitle" in m[0]["content"]]), 2,
                         "a tap keeps the lines Muse already looked at twice")
        self.assertEqual("".join(line["text"] for line in self.chat.project.state()["captions"]["lines"]),
                         "You can go see it.It costs 990 baht.")

    def test_english_captions_that_do_not_line_up_are_asked_once_more_then_left_as_said(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "ตั้งค่าเครื่องง่ายมาก"},
                               {"from": 5.0, "to": 8.0, "text": "ราคาถูกมาก"}]
        script = self.to_moment()
        brain, steps = self.run_job(
            *script[:2], [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the setup"}])],
            '{"lines": ["ตั้งค่าเครื่องง่ายมาก", "ราคาถูกมาก"]}',
            [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_captions", language="English", why="abroad")], '{"lines": ["Easy to set up."]}', "not json",
            [call("add_captions", language="English", why="again")], '{"lines": ["Easy."]}',
            '{"lines": ["Easy to set up.", "Really cheap."]}',
            [call("add_captions", language="Japanese", why="for Japan")],
            "Done.")
        left, again, japanese = steps[-3:]
        self.assertFalse(left["refused"], left["result"])
        self.assertRegex(left["result"], r"^Captions on: \d+ lines of what was said")
        self.assertIn("did not line up one for one with the sentences said, twice, so the captions are in Thai as said",
                      left["result"])
        self.assertRegex(again["result"], r"^Captions on: 2 lines in English, translated from the 2 sentences said in "
                                          r"Thai")
        self.assertEqual([line["text"] for line in self.chat.project.state()["captions"]["lines"]],
                         ["Easy to set up.", "Really cheap."], "a gap of 1 s ends a sentence")
        self.assertTrue(japanese["refused"])
        self.assertIn("The studio writes captions in English or Thai, the languages its fonts draw", japanese["result"])

    def test_a_line_heard_unsure_that_the_fix_made_read_right_is_a_caption(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "หรือว่ามันต้องเป็นไพ่", "sure": 0.85},
                               {"from": 4.5, "to": 6.0, "text": "แ้ไพ่มันเข้าทาง", "sure": 0.85},
                               {"from": 6.5, "to": 7.0, "text": "เลย", "sure": 0.44},
                               {"from": 7.5, "to": 9.0, "text": "ใช่คือสิ่งที่ฉันแจกเอง", "sure": 0.95}]
        script = self.to_moment()
        _, steps = self.run_job(
            *script[:2], [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the talk"}])],
            '{"lines": ["หรือว่ามันต้องเป็นไพ่", "แ้ไพ่มันเข้าทาง", "เลย", "ใช่คือสิ่งที่ฉันแจกเอง"]}',
            [call("make_clip", moment=1, layout="subject", why="the talk")],
            [call("add_captions", why="say what they say")], "Done.")
        caps = steps[-1]
        self.assertFalse(caps["refused"], caps["result"])
        shown = "".join(line["text"] for line in self.chat.project.state()["captions"]["lines"])
        self.assertIn("หรือว่ามันต้องเป็นไพ่", shown, "talk clip 10-28: two people in a room are heard at 0.83-0.87")
        self.assertIn("ใช่คือสิ่งที่ฉันแจกเอง", shown)
        self.assertNotIn("แ้ไพ่", shown, "odd bits left after the fix: still not sure")
        self.assertIn('2 lines the hearing was not sure of are under the clip as drafts, not on the video',
                      caps["result"], "below READ_SURE a line is a draft even when it reads as a word: at 0.44 it is a "
                                      "guess")

    def test_a_translated_line_keeps_the_numbers_said_as_a_thai_speaker_says_them(self):
        differ = tools._numbers_differ
        self.assertFalse(differ("กลับมาเป็นแสนหนึ่งและนะครับ", "It's back to one hundred thousand now."),
                         "Eaglet 10-28: แสนหนึ่ง is said for 100,000")
        self.assertFalse(differ("นะครับ สามารถเข้าไปดูได้ใน", "Okay, you can go check it in"), "สามารถ holds no three")
        self.assertFalse(differ("ราคา 990 บาท", "It costs 990 baht."))
        self.assertFalse(differ("ได้สามสิบห้าคน หนึ่งวัน", "35 people in a day."))
        self.assertTrue(differ("ราคา 990 บาท", "It costs 99 baht."), "a number lost")
        self.assertTrue(differ("สามคน", "Two people."), "a number changed")
        self.assertTrue(differ("ได้ห้าสิบ", "We got fifty, plus 7."), "a number made up")

    def test_a_subtitle_sentence_never_joins_two_phrases_with_a_left_out_one_between(self):
        phrases = [{"from": 0.0, "to": 1.0, "text": "a", "n": 0}, {"from": 1.2, "to": 2.0, "text": "b", "n": 2},
                   {"from": 2.1, "to": 3.0, "text": "c", "n": 3}, {"from": 3.1, "to": 9.0, "text": "d", "n": 4}]
        self.assertEqual(tools._sentences(phrases), [{"from": 0.0, "to": 1.0, "text": "a", "first": 0},
                                                     {"from": 1.2, "to": 3.0, "text": "b c", "first": 1},
                                                     {"from": 3.1, "to": 9.0, "text": "d", "first": 3}],
                         "the check finds a sentence in what was said only when its phrases were said one after "
                         "another; and two long phrases are two sentences")

    def test_a_recipe_told_english_subtitles_has_the_captions_translated_but_never_thai_into_thai(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "ตั้งค่าเครื่องง่ายมาก"},
                               {"from": 5.0, "to": 8.0, "text": "ราคาถูกมาก"}]
        script = self.to_moment()
        brain, steps = self.run_job(
            script[0], [call("load_recipe", type="talk", subtitles="อังกฤษ", why="the kind")], script[1],
            [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the setup"}])],
            '{"lines": ["ตั้งค่าเครื่องง่ายมาก", "ราคาถูกมาก"]}',
            [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_captions", why="captions")], '{"lines": ["Easy to set up.", "Really cheap."]}',
            [call("add_title", words="ตั้งง่าย", start=0, end=3, place="top", why="the hook")],
            [call("add_title", words="Easy setup", start=0, end=3, place="top", why="the hook")],
            [call("add_captions", language="ไทย", why="back to Thai")],
            [call("add_title", words="ราคาถูก", start=3, end=6, place="middle", why="Thai again")],
            [call("load_recipe", type="talk", subtitles="Klingon", why="the kind")],
            "Done.")
        recipe, english, thai_, klingon = steps[1], steps[-6], steps[-3], steps[-1]
        self.assertFalse(steps[-2]["refused"], steps[-2]["result"])
        self.assertIn("captions in English", recipe["result"])
        self.assertTrue(steps[-5]["refused"])
        self.assertIn('the words on the clip in English, so a viewer who reads no Thai can follow: write "ตั้งง่าย" in '
                      'English.', steps[-5]["result"])
        self.assertFalse(steps[-4]["refused"], steps[-4]["result"])
        self.assertRegex(english["result"], r"^Captions on: 2 lines in English")
        self.assertRegex(thai_["result"], r"^Captions on: \d+ lines of what was said")
        self.assertNotIn("line up", thai_["result"])
        self.assertEqual(self.chat.clip["subtitles"], "Thai", "the batch of clips made next keeps the person's choice")
        self.assertTrue(klingon["refused"])
        self.assertEqual(len([m for m in brain.sent if len(m) == 1 and "subtitle" in m[0]["content"]]), 1,
                         "captions said in Thai and asked for in Thai are not translated")

    def test_words_on_the_clip_name_no_other_companys_app(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_title", words="คลิปเด็ดลงติ๊กต็อก", start=0, end=3, place="top", why="the hook")],
            [call("add_title", words="คลิปเด็ดลง Reels", start=0, end=3, place="top", why="the hook")], "Done.")
        self.assertTrue(steps[-2]["refused"])
        self.assertIn("This names \"ติ๊กต็อก\", which belongs to another company's app.", steps[-2]["result"])
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])

    def test_captions_blurs_and_hits_go_on_the_clip_and_muse_is_told_what_they_cover(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "this is the moment"},
                               {"from": 5.0, "to": 8.0, "text": "watch him fall down now"}]
        fixed = '{"lines": ["this is the moment", "watch him fall down now"]}'
        script = self.to_moment()
        brain, steps = self.run_job(
            *script[:2], [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "he falls"}])],
            fixed, [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_captions", why="say what he says")],
            [call("add_captions", lit=True, why="karaoke")],
            [call("blur_box", box=[0, 0, 200, 1000], why="the left edge")],
            [call("blur_box", box=[400, 600, 600, 900], why="the face")],
            [call("blur_box", box=[400, 0, 600, 300], why="the names")],
            [call("add_hit", at=5.2, kind="word", word="POW", why="the fall")],
            [call("add_hit", at=30, kind="flash", why="too late")],
            [call("add_hit", at=6, kind="word", why="no word")],
            "Done.")
        caps, lit, outside, onface, blur, word, late, wordless = steps[-8:]
        self.assertTrue(onface["refused"])
        self.assertIn("That box covers the face at [438, 667, 562, 889] on your grid, the face the clip shows",
                      onface["result"])
        self.assertFalse(caps["refused"], caps["result"])
        self.assertRegex(caps["result"], r"^Captions on: 4 lines of what was said, numbers exactly as heard, in the band "
                                         r"from y 567 to 652 on your grid")
        self.assertIn("0:00 this is the\n0:02 moment\n", caps["result"], "the clip opens 0.3 s before the first word")
        self.assertIn("plain words, none lit, so the eye stays on the picture", caps["result"])
        self.assertNotIn("on camera", caps["result"])
        self.assertFalse(lit["refused"], "lit is Muse's choice, or the person's")
        self.assertIn("each word lights up as it is said", lit["result"])
        self.assertIn("A person is on camera (a face in the same place in 30 of 40 pictures). Lit words pull the eye "
                      "away", lit["result"])
        self.assertTrue(self.chat.project.state()["captions"]["lit"])
        self.assertIn("lit captions on", self.chat.clip_line())
        self.assertTrue(outside["refused"])
        self.assertIn("is not in the clip's picture (the subject layout leaves that part out)", outside["result"])
        self.assertFalse(blur["refused"], blur["result"])
        self.assertIn("Blurred [400, 0, 600, 300] on your grid from 0 to 18.3 s, so nothing in it can be read. In the "
                      "clip it shows in the frame at x ", blur["result"])
        self.assertEqual(word["result"], 'Added a big word that pops in, such as POW at 5.2 s ("POW").')
        self.assertIn("at 30 s is outside the clip, which is 18.3 s long", late["result"],
                      "the 1.7 s of still silence before the first word is cut")
        self.assertIn("A word hit needs its word", wordless["result"])
        self.assertIn("captions on, 1 blur, other people: 0 blurred, 0 kept, 1 still to decide before rendering, hits: "
                      "word at 5.2 s · no finished file yet", brain.sent[-1][-1]["content"])
        state = self.chat.project.state()
        self.assertEqual([round(x["from"], 2) for x in state["captions"]["lines"]][:1], [0.3],
                         "caption times are clip seconds: heard at 32 s of the recording, the clip opening at 31.7 s")
        self.assertEqual(state["blurs"][0]["box"], [128, 0, 64, 54])

    def test_the_clips_ends_move_where_the_person_says_and_the_captions_go_with_the_words(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "this is the moment"},
                               {"from": 5.0, "to": 8.0, "text": "watch him fall down now"}]
        script = self.to_moment()
        _, steps = self.run_job(
            *script[:2], [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "he falls"}])],
            '{"lines": ["this is the moment", "watch him fall down now"]}',
            [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_captions", why="say what he says")],
            [call("add_hit", at=5.2, kind="word", word="POW", why="the fall")],
            [call("change_ends", start="0:34.8", why="the person: cut the first line off")], "Done.")
        trimmed = steps[-1]
        self.assertFalse(trimmed["refused"], trimmed["result"])
        self.assertTrue(trimmed["result"].startswith("Moved the clip's ends: it now runs from 0:34.8 to 0:50.0 of the "
                                                     "recording (it ran from 0:31.7 to 0:50.0)."), trimmed["result"])
        state = self.chat.project.state()
        self.assertEqual(state["clip"]["pieces"], [[34.8, 50.0]])
        self.assertEqual(state["clip"]["layout"]["kind"], "subject", "made again in the same layout")
        said = "".join(ln["text"] for ln in state["captions"]["lines"])
        self.assertNotIn("moment", said, "the line said before the new start is gone")
        self.assertIn("watch him", said)
        self.assertEqual([round(h["at"], 2) for h in state["hits"]], [2.1], "the hit stays on the fall, 3.1 s sooner")
        self.assertEqual([m["n"] for m in self.chat.moments], [1, 2], "the moment Muse proposed is still there")
        _, steps = self.run_job([call("change_ends", start="0:30", why="the person: start earlier")], "Done.")
        earlier = steps[-1]
        self.assertFalse(earlier["refused"], earlier["result"])
        state = self.chat.project.state()
        self.assertEqual(state["clip"]["pieces"][0][0], 30.0, "the person's start stays where it was put, though the "
                                                              "picture is still and nobody speaks there")
        self.assertIn("this is the moment", "".join(ln["text"] for ln in state["captions"]["lines"]),
                      "the line the clip took in is captioned again")
        self.assertEqual(state["clip"]["moment"], 3)
        self.chat.undo()
        state = self.chat.project.state()
        self.assertEqual(state["clip"]["pieces"], [[34.8, 50.0]], "one undo puts the ends back, captions and all")
        self.assertNotIn("moment", "".join(ln["text"] for ln in state["captions"]["lines"]))
        again = agent.Chat.load(self.chat.id)
        self.assertEqual([m["n"] for m in again.moments], [1, 2, 3], "the moved moments come back with the chat")
        self.assertEqual(again.project.acts, self.chat.project.acts, "and which entries one action made")
        again.undo()
        self.assertEqual(again.project.state()["clip"]["pieces"], [[31.7, 50.0]], "so Undo still takes back the "
                                                                                  "whole move after a restart")

    def test_a_start_before_what_was_heard_is_heard_first_and_wrong_ends_are_refused(self):
        _, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("change_ends", start="0:45", why="too short")],
            [call("change_ends", start="0:31", end="0:31.5", why="backwards")],
            [call("change_ends", start="0:20", why="the person: start earlier")], "Done.")
        short, wrong, earlier = steps[-3:]
        self.assertTrue(short["refused"])
        self.assertIn("From 0:45.0 to 0:50.0 is 5 s, and a clip is 10 to 90 s. The clip runs from 0:30.0 to 0:50.0 "
                      "now.", short["result"])
        self.assertTrue(wrong["refused"])
        self.assertFalse(earlier["refused"], earlier["result"])
        self.assertEqual(self.chat.project.state()["clip"]["pieces"], [[20.0, 50.0]])
        self.assertEqual(len(self.chat.looks), 2, "a start before what the look heard is heard as a wider look")
        self.assertEqual((self.chat.looks[1]["start"], self.chat.looks[1]["heard"]), (20.0, [14.0, 56.0]))
        self.assertEqual(self.chat.moments[-1]["look"], 2)

    def test_words_stay_off_the_product_muse_names_and_a_key_inside_a_word_is_fixed(self):
        brain, steps = self.run_job(
            *self.to_moment(),
            [call("make_clip", moment=1, layout="subject", product=[400, 450, 600, 900], why="the product")],
            [call("add_title", words="Covers 50-75 m2", start=0, end=4, place="middle", why="the spec")],
            [call("add_title", words="Covers 50-75 m2", start=0, end=4, place="top", key="75", why="the spec")],
            "Done.")
        made, middle, top = steps[-3:]
        self.assertIn("kept clear of the faces the program found and the product you named", made["result"])
        self.assertIn("- middle (y 472 to 559): never clear for 2 s or more; do not use it", made["result"])
        self.assertTrue(middle["refused"])
        self.assertIn("At the middle, the title would cover a face or the product at 0-4 s.", middle["result"])
        self.assertFalse(top["refused"], top["result"])
        self.assertIn('The key "75" is part of the word "50-75", so that word is the key.', top["result"])
        self.assertEqual(self.chat.project.state()["titles"][0]["key"], "50-75")

    def test_render_waits_for_the_file_and_hand_over_needs_one_made_from_the_clip_as_it_is(self):
        made = []

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            on_start(lambda: None)
            on_progress(state["length"] / 2)
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            made.append((state["length"], default))
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            brain, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("hand_over", title="t", caption="c", hashtags=["#a"], why="too soon")],
                [call("render_clip", why="nobody decided")],
                [call("decide_people", keep=[1], why="it is him, higher up")],
                [call("render_clip", why="make the file")],
                [call("render_clip", why="again")],
                [call("add_title", words="Boss down", start=0, end=4, place="top", why="the hook")],
                [call("hand_over", title="t", caption="c", hashtags=["a"], why="stale")],
                [call("undo", why="back to what was rendered")],
                [call("check_clip", why="before it goes out")],
                [call("hand_over", title="Boss down", caption="He falls.\n#gaming #boss", why="many",
                      hashtags=["#sifu", "#kungfu", "#streamer", "#clip", "#game"])],
                [call("hand_over", title="Boss down", caption="He falls.\nTwice.\n#gaming #boss", why="post",
                      hashtags=["gaming", "#gaming", "sifu clip", "#"])],
                "Done.")
        finally:
            render.export = had
        early, undecided, kept, first, again, titled, stale, undone, checked, many, ready = steps[-11:]
        self.assertFalse(checked["refused"], checked["result"])
        self.assertTrue(many["refused"])
        self.assertIn("7 hashtags (#sifu #kungfu #streamer #clip #game #gaming #boss): give at most 5", many["result"],
                      "the yeti review 10-29 was handed over with eight; the house standard says five")
        self.assertTrue(early["refused"])
        self.assertIn("Render the clip first, then hand it over.", early["result"])
        self.assertTrue(undecided["refused"])
        self.assertEqual(undecided["result"], "Decide about person 1 first, with decide_people: blur or keep. Nothing "
                                              "undecided is rendered.")
        self.assertFalse(kept["refused"], kept["result"])
        self.assertRegex(first["result"], r"^Rendered the finished clip in 0:0\d: 20 s, 1080x1920, [\d.]+ MB\.")
        self.assertIn("already made from the clip as it is now", again["result"])
        self.assertEqual(made, [(20, "clean")],
                         "rendered once, in the style the preview uses: the second call had nothing new to render")
        self.assertTrue(stale["refused"], "a title came after the render, so the file is not the clip any more")
        self.assertFalse(ready["refused"], ready["result"])
        self.assertIn("The clip is 20 s, from 0:30 to 0:50 of the recording: when you say its length or its times, "
                      "say these, not the ones asked for.", ready["result"],
                      "lane 5 10-28: Muse told the person 15 s from 1:40 for a 16.79 s clip from 1:38")
        shown = [e for e in self.chat.events if e["kind"] == "ready"][-1]
        self.assertEqual((shown["title"], shown["hashtags"]), ("Boss down", ["#gaming", "#sifuclip", "#boss"]))
        self.assertEqual(shown["caption"], "He falls.\nTwice.", "a line of hashtags goes to the hashtags, shown once")
        self.assertTrue(os.path.isfile(os.path.join(render.folder("clips"), shown["file"])))
        self.assertIn("finished file %s, made from the clip as it is now" % shown["file"],
                      brain.sent[-1][-1]["content"])
        self.chat.edit({"do": "style", "name": "pop", "font": None, "colour": None})
        self.assertIn("the last finished file is older than the last change", self.chat.clip_line())
        self.assertEqual(agent.Chat.load(self.chat.id).rendered, self.chat.rendered)

    def test_the_cover_is_the_peak_frame_without_words_paid_once_asked_on_this_computer_and_handed_over(self):
        from unittest import mock
        from test_cover import Images
        import test_cover

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        drawn = os.path.join(self.tmp, "drawn.jpg")
        api = Images(test_cover.picture(drawn, "color=c=0x4070D0:s=1280x1920"))
        had = getattr(render, "export", None)
        render.export = export
        try:
            with mock.patch.dict(os.environ, {"META_API_KEY": "made-up", "LCS_IMAGE_URL": api.url}):
                brain, steps = self.run_job(
                    *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                    [call("decide_people", keep=[1], why="it is him")],
                    [call("add_title", words="Boss down", start=0, end=20, place="middle", why="the hook")],
                    [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                    [call("make_cover", title="บอสล้มสองรอบ", why="the cover")],
                    [call("make_cover", title="บอสล้มสองรอบ", asked=True, why="the person asked")],
                    [call("make_cover", title="บอสล้มสองรอบ", asked=True, why="again, same words")],
                    lambda asking: sends.allow(True) or [call("make_cover", title="บอสล้มสองรอบ", why="the switch")],
                    [call("hand_over", title="Boss down", caption="He falls.", hashtags=["#gaming"], why="post")],
                    "Done.")
        finally:
            render.export = had
            api.close()
        titled, rendered, checked, unasked, made, kept, switched, ready = steps[-8:]
        self.assertFalse(titled["refused"], titled["result"])
        self.assertIn("Hand it over next.", checked["result"], "Muse on this computer is not sent to make a cover")
        self.assertTrue(unasked["refused"])
        self.assertIn("no picture of the person's leaves it unasked", unasked["result"])
        self.assertFalse(made["refused"], made["result"])
        self.assertIn("Muse Image drew the cover from the clip at", made["result"])
        self.assertIn("for $0.01", made["result"])
        self.assertFalse(kept["refused"], kept["result"])
        self.assertIn("made before, at no cost", kept["result"])
        self.assertFalse(switched["refused"], "with Meta sees pieces on in Options, the cover needs no asking")
        self.assertEqual(len(api.got), 1, "one picture asked for and paid")
        state = self.chat.project.state()
        at = tools._cover_second(types.SimpleNamespace(chat=self.chat), state)
        moment = self.chat.moments[0]
        self.assertAlmostEqual(at, project.from_recording(state["clip"]["pieces"], moment["t"]), places=2)
        plain = render.frame(self.chat.recording, state, at, "clean", width=1080, words=False)
        worded = render.frame(self.chat.recording, state, at, "clean", width=1080)
        with open(plain, "rb") as f:
            sent = api.got[0]["body"]["images"][0]["image_url"]
            self.assertEqual(sent, "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii"),
                             "the clip's frame at its peak, without its title: Muse Image writes the words")
        self.assertNotEqual(plain, worded)
        self.assertIn("บอสล้มสองรอบ The title is in Thai", api.got[0]["body"]["prompt"])
        cover = [e for e in self.chat.events if e["kind"] == "cover"][-1]
        shown = [e for e in self.chat.events if e["kind"] == "ready"][-1]
        self.assertEqual(shown["cover"], cover["cover"])
        self.assertEqual(cover["file"], shown["file"])
        self.assertIn("its cover %s is made" % cover["cover"], self.chat.clip_line())
        self.assertIn("finished clip and its cover", ready["result"])
        self.assertTrue(os.path.isfile(os.path.join(render.folder("covers"), cover["cover"])))
        with open(os.path.join(agent.WORK, "spent.jsonl"), encoding="utf-8") as f:
            paid = [json.loads(line) for line in f if '"image"' in line]
        self.assertEqual([(r["mind"], r["label"], r["usd"]) for r in paid], [("image", "Muse Image", 0.01)])

    def test_muse_decides_who_is_blurred_and_a_blurred_face_is_followed_in_clip_seconds(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("decide_people", why="nobody named")],
            [call("decide_people", blur=[1], keep=[1], why="both")],
            [call("decide_people", blur=[2], why="no such person")],
            [call("decide_people", blur="1", why="someone passing behind")],
            "Done.")
        empty, both, unknown, blurred = steps[-4:]
        self.assertTrue(empty["refused"] and both["refused"] and unknown["refused"])
        self.assertIn("Name at least one person to blur or to keep.", empty["result"])
        self.assertIn("Person 1 is in both lists", both["result"])
        self.assertIn("There is no person 2; the clip's people are 1.", unknown["result"])
        self.assertFalse(blurred["refused"], blurred["result"])
        self.assertIn("Person 1 is blurred where their face shows, 0:05-0:10 of the clip, and the blur moves with the face.",
                      blurred["result"])
        self.assertIn("Every person seen in two pictures or more is decided.", blurred["result"])
        self.assertIn("comes next as a picture", blurred["result"], "the frame is drawn with the moving blur")
        state = self.chat.project.state()
        track = [b for b in state["blurs"] if b.get("person") == 1][0]["track"]
        times = [p[0] for p in track]
        self.assertEqual((times[0], times[-1]), (4.75, 9.75), "clip seconds, held half a picture at each end")
        self.assertTrue(all(0 < b - a <= 0.5 for a, b in zip(times, times[1:])), times)
        self.assertEqual(track[len(track) // 2][1:3], [140, 10], "the face's place in the recording")
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), "the decision on people (blur person 1)")
        self.assertIn("other people: 1 blurred, 0 kept", self.chat.clip_line())
        self.chat.project.add({"do": "people", "blur": [], "keep": [1]})
        self.assertEqual([b for b in self.chat.project.state()["blurs"] if b.get("person")], [], "the later one wins")
        self.chat.project.undo()
        self.chat.project.undo()
        self.assertEqual(tools._undecided(self.chat.project.state()), [1])
        _, crowd = self.run_job([call("decide_people", others="blur", why="a crowd behind")], "Done.")
        self.assertIn("Person 1 is blurred where their face shows", crowd[-1]["result"],
                      "others decides everyone not named")
        self.chat.project.undo()
        clip = dict(self.chat.project.state()["clip"])
        self.chat.project.add({"do": "people", "blur": [1], "keep": []})
        self.chat.project.add(clip)
        self.assertEqual(tools._undecided(self.chat.project.state()), [1], "a new clip starts with nobody decided")

    def test_muse_grades_the_footage_and_pushes_in_on_the_build_up(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("set_look", look="vivid", why="a game")],
            [call("set_look", look="vivid", why="again")],
            [call("slow_push", start=1, end=1.2, why="too short")],
            [call("slow_push", start=4, end=9, why="into the fall")],
            [call("slow_push", start=8, end=12, zoom=1.1, why="later")],
            "Done.")
        vivid, again, short, first, later = steps[-5:]
        self.assertFalse(vivid["refused"], vivid["result"])
        self.assertIn("The footage now has the vivid look", vivid["result"])
        self.assertIn("comes next as a picture", vivid["result"])
        self.assertTrue(again["refused"])
        self.assertIn("The clip already has the vivid look.", again["result"])
        self.assertTrue(short["refused"])
        self.assertIn("a slow push runs 0.5 to 30 seconds", short["result"])
        self.assertFalse(first["refused"], first["result"])
        self.assertIn("A slow push in on the main picture from 4 to 9 s, to 1.08 times its size", first["result"])
        self.assertIn("It replaces the push at 4-9 s.", later["result"])
        state = self.chat.project.state()
        self.assertEqual(state["look"], "vivid")
        self.assertEqual([(q["start"], q["end"], q["zoom"]) for q in state["pushes"]], [(8, 12, 1.1)])
        self.assertEqual(state["pushes"][0]["centre"], tools._main_part(state["clip"]["layout"]))
        self.assertIn("vivid look, slow push 8-12 s", self.chat.clip_line())
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), "the slow push at 8-12 s")

    def test_muse_is_offered_the_looks_the_renderer_draws(self):
        self.assertEqual(set(tools.LOOKS) - {"none"}, set(effects.LOOKS))

    def test_muse_picks_only_a_free_font_unless_the_person_named_one(self):
        self.chat.emit("person", job=None, text="ใช้ฟอนต์ Kanit Black นะ")
        kanit = ["Kanit Regular", "Kanit SemiBold", "Kanit Bold", "Kanit Black"]
        listed = {"free": True}
        asked = []

        def fonts(letters=None):
            asked.append(letters)
            return [{"family": "Kanit", "faces": kanit, "free": listed["free"],
                     "licence": "SIL Open Font License" if listed["free"] else "its file names no licence",
                     "where": "studio"},
                    {"family": "Thin Latin", "faces": ["Thin Latin"], "free": True, "licence": "Apache License",
                     "where": "computer"}][:1 if letters and any("ก" <= c <= "๛" for c in letters) else 2]

        real, tools.text.fonts = tools.text.fonts, fonts
        try:
            brain, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_title", words="บอสล้ม", start=0, end=4, place="middle", why="the fall")],
                [call("list_fonts", why="the person wants another font")],
                [call("set_style", style="clean", font="Comic Sans", why="no such font")],
                [call("set_style", style="clean", font="Thin Latin", why="it cannot write Thai")],
                [call("set_style", style="pop", font="kanit black", why="the person asked")],
                [call("set_style", style="pop", part="caption", why="a part needs its font")],
                [call("set_style", style="pop", font="Kanit Regular", part="caption", why="lighter captions")],
                [call("set_style", style="sticker", why="another style")],
                [call("set_style", style="sticker", size=4, part="caption", why="bigger captions")],
                [call("set_style", style="sticker", size=9, why="too big")],
                "Done.")
            parts = self.chat.project.state()["style"]
            brain, whole = self.run_job([call("set_style", style="sticker", size=2, why="smaller everywhere")], "Done.")
            smaller = self.chat.project.state()["style"]
            self.chat.project.entries.pop()
            listed["free"] = False
            brain, later = self.run_job(
                [call("set_style", style="clean", font="Kanit", why="not free now")],
                [call("set_style", style="clean", font="Kanit", person_said="ฟอนต์ Kanit Black", why="they named it")],
                "Done.")
        finally:
            tools.text.fonts = real
        title, listing, none, latin, black, bare, caption, sticker, bigger, nine = steps[-10:]
        self.assertFalse(title["refused"], title["result"])
        self.assertIn("บ", asked[0], "the fonts asked for must write the title's letters")
        self.assertIn("1 free to use in a video, 0 not known to be free.", listing["result"])
        self.assertIn("- Kanit (studio, SIL Open Font License): Kanit Regular, Kanit SemiBold, Kanit Bold, Kanit Black",
                      listing["result"])
        self.assertTrue(none["refused"])
        self.assertIn("There is no font Comic Sans in the studio or on this computer. Free fonts that can: Kanit.",
                      none["result"])
        self.assertTrue(latin["refused"])
        self.assertIn("Thin Latin cannot write every letter of the clip's words.", latin["result"])
        self.assertFalse(black["refused"], black["result"])
        self.assertIn("The words on screen now use the pop style with the font Kanit.", black["result"])
        self.assertTrue(bare["refused"])
        self.assertIn("A part takes a font, a size or colours of its own: give them for the captions", bare["result"])
        self.assertIn("with the font Kanit for the captions only.", caption["result"])
        self.assertIn("Kept from before: the font Kanit Regular for the captions.", sticker["result"])
        self.assertIn("The words on screen now use the sticker style at size 4 for the captions only.", bigger["result"])
        self.assertTrue(nine["refused"])
        self.assertIn("size must be from 1 to 5; 9 is outside that.", nine["result"])
        self.assertEqual((parts["name"], parts["font"], parts["fonts"], parts.get("size"), parts["sizes"]),
                         ("sticker", "kanit black", {"caption": "Kanit Regular"}, None, {"caption": 4}))
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), "the clean style with the font Kanit, the captions at size 4")
        not_free, named = later
        self.assertTrue(not_free["refused"])
        self.assertIn("Kanit is not known to be free (its file names no licence)", not_free["result"])
        self.assertFalse(named["refused"], named["result"])
        self.assertEqual(self.chat.project.state()["style"]["font"], "Kanit")
        self.assertNotIn("fonts", self.chat.project.state()["style"], "a font for every part starts the parts over")
        drawn = render.style_of(parts and {"style": dict(parts, do="style")}, "clean", kind="caption")
        self.assertEqual((drawn.fontname, drawn.size), ("Kanit Regular", 4), "the export draws the part as asked")
        self.assertEqual(render.style_of({"style": dict(parts, do="style")}, "clean", kind="title").size, 3)
        self.assertIn("at size 2.", whole[-1]["result"])
        self.assertEqual((smaller["size"], smaller.get("sizes")), (2, None), "a size for every part starts them over")

    def test_muse_is_offered_the_sounds_the_renderer_makes(self):
        self.assertEqual(set(tools.SOUNDS), set(effects.SOUNDS))

    def test_muse_puts_the_studios_stickers_and_glass_boxes_on_the_clip_clear_of_faces(self):
        self.chat.emit("person", job=None, text="ใส่ไฟบนหน้าผมเลย")
        face = [430, 660, 570, 760]
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_sticker", sticker="fire", at=2, place="middle", why="the peak")],
            [call("add_sticker", sticker="star", at=2.5, place="middle", why="a better one")],
            [call("add_sticker", sticker="fire", at=19.8, place="middle", why="at the very end")],
            [call("add_sticker", sticker="fire", at=1, box=face, why="on the face")],
            [call("add_sticker", sticker="fire", at=1, box=face, person_said="ใส่ไฟบนหน้าผม", why="they asked")],
            [call("add_sticker", sticker="fire", at=1, why="nowhere")],
            [call("add_sticker", sticker="fire", at=1, box=[100, 900, 300, 990], why="low")],
            [call("add_glass", box=[100, 440, 800, 555], start=1, end=5, why="for the title")],
            [call("add_glass", box=[100, 440, 800, 555], start=10, end=14, tint="light", why="light")],
            [call("add_glass", box=[100, 440, 800, 555], start=6, end=6.2, why="too short")],
            [call("add_glass", box=[100, 460, 130, 470], start=6, end=9, why="too small")],
            [call("add_glass", box=face[:3] + [755], start=1, end=4, tint="dark", why="on the face")],
            [call("add_glass", box=[100, 940, 900, 1000], start=6, end=9, why="at the bottom")],
            [call("set_style", style="sticker", why="labels")],
            [call("add_glass", box=[100, 440, 800, 555], start=15, end=18, why="under a label")],
            "Done.")
        fire, star, late, on_face, asked, nowhere, under, glass, light, short, small, glass_face, low = steps[-15:-2]
        self.assertIn("The sticker style draws its own box behind the words, so they read on any glass.",
                      steps[-1]["result"])
        self.assertFalse(fire["refused"], fire["result"])
        self.assertIn("The sticker fire pops in at 2 s at the middle and stays 2 s: 170 x 170 px, clear of every face "
                      "the program found.", fire["result"])
        self.assertIn("comes next as a picture", fire["result"])
        self.assertIn("It replaces the sticker fire at 2 s.", star["result"])
        self.assertTrue(late["refused"])
        self.assertIn("a sticker needs 0.4 s to pop in and fade out. Put it earlier.", late["result"])
        self.assertTrue(on_face["refused"])
        self.assertIn("The sticker would cover a face at", on_face["result"])
        self.assertIn("- middle (y 472 to 559): clear of faces for the whole clip", on_face["result"])
        self.assertFalse(asked["refused"], asked["result"])
        self.assertIn("It covers a face at", asked["result"])
        self.assertIn("where the person asked for it; tell the person.", asked["result"])
        self.assertTrue(nowhere["refused"])
        self.assertIn("Give a box, or one of the layout's places for words: top, middle.", nowhere["result"])
        self.assertTrue(under["refused"])
        self.assertIn("On a phone, the name and the caption cover everything below y 791 on your grid", under["result"])
        self.assertFalse(glass["refused"], glass["result"])
        self.assertIn("A dark glass box at [100, 440, 800, 555] on your grid from 1 to 5 s", glass["result"])
        self.assertIn("Dark glass, because the clean style's words are light.", glass["result"])
        self.assertFalse(light["refused"], light["result"])
        self.assertIn("The clean style's words are light, so they stand on their outline here; dark glass would read "
                      "better under them.", light["result"])
        self.assertIn("give add_title the same box", glass["result"])
        self.assertIn("comes next as a picture", glass["result"])
        self.assertTrue(short["refused"])
        self.assertIn("too short for a glass box", short["result"])
        self.assertTrue(small["refused"])
        self.assertIn("too small for a glass box", small["result"])
        self.assertTrue(glass_face["refused"])
        self.assertIn("The glass box would cover a face at", glass_face["result"])
        self.assertTrue(low["refused"])
        self.assertIn("On a phone, the name and the caption cover everything below y 791 on your grid", low["result"])
        state = self.chat.project.state()
        self.assertEqual([(k["name"], k["at"]) for k in state["stickers"]], [("fire", 1), ("star", 2.5)])
        self.assertEqual(state["stickers"][1]["png"], effects.sticker_png("star"))
        self.assertEqual([(g["box"], g["start"], g["end"], g["tint"], g["shine"]) for g in state["glass"]],
                         [([108, 845, 756, 221], 1, 5, "dark", True), ([108, 845, 756, 221], 10, 14, "light", True),
                          ([108, 845, 756, 221], 15, 18, "light", True)])
        self.assertIn("glass boxes: dark 1-5 s, light 10-14 s, light 15-18 s, stickers: fire at 1 s, star at 2.5 s", self.chat.clip_line())

    def test_muse_sends_stickers_flying_in_a_box_clear_of_faces_and_the_clip_draws_them(self):
        face = [430, 660, 570, 755]
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_flurry", stickers=["coin", "dollar_banknote"], at=2, move="rain", place="middle",
                  why="the payout")],
            [call("add_flurry", stickers="confetti", at=3, count=12, place="middle", why="a better one")],
            [call("add_flurry", stickers=["red_heart"], at=18.8, move="rise", place="top", why="to the end")],
            [call("add_flurry", stickers=["coin"], at=19.5, place="top", why="too late")],
            [call("add_flurry", stickers=["coin"], at=1, box=face, why="on the face")],
            [call("add_flurry", stickers=["coin"], at=6, box=[100, 460, 130, 470], why="too small")],
            [call("add_flurry", stickers=["coin"] * 7, at=6, place="middle", why="too many")],
            [call("add_flurry", stickers=["coin"], at=6, box=[100, 900, 300, 990], why="low")],
            [call("add_flurry", stickers=["coin"], at=25, place="top", why="after the end")],
            "Done.")
        rain, confetti, short, late, on_face, small, many, low, after = steps[-9:]
        self.assertFalse(rain["refused"], rain["result"])
        self.assertIn("A rain flurry of 8 copies of coin and dollar_banknote from 2 to 4.4 s in the middle, clear of "
                      "every face the program found: they fall from the top of the box to its bottom one after "
                      "another, swaying.", rain["result"])
        self.assertRegex(rain["result"], r"The pictures are \d+ px, not 150, so they move inside the box")
        self.assertIn("comes next as a picture", rain["result"])
        self.assertFalse(confetti["refused"], confetti["result"])
        self.assertIn("It replaces the rain flurry at 2 s.", confetti["result"])
        self.assertNotIn("The pictures are", confetti["result"], "paper pieces have their own size")
        self.assertFalse(short["refused"], short["result"])
        self.assertIn("It ends with the clip, so the move takes 1.2 s instead of 2.6 s.", short["result"])
        self.assertTrue(late["refused"])
        self.assertIn("At 19.5 s the clip ends 0.5 s later; a flurry needs 0.8 s to move. Put it earlier.",
                      late["result"])
        self.assertTrue(on_face["refused"])
        self.assertIn("The flurry would cover a face at", on_face["result"])
        self.assertTrue(small["refused"])
        self.assertIn("is too small for these pictures to move in", small["result"])
        self.assertTrue(many["refused"])
        self.assertIn("stickers must be a list of 1 to 6 names.", many["result"])
        self.assertTrue(low["refused"])
        self.assertIn("On a phone, the name and the caption cover everything below y 791 on your grid", low["result"])
        self.assertTrue(after["refused"])
        self.assertIn("at 25 s is outside the clip, which is 20 s long", after["result"])
        state = self.chat.project.state()
        middle = state["clip"]["layout"]["places"]["middle"]
        self.assertEqual([(f["names"], f["at"], f["move"], f["count"], f["length"], f["box"]) for f in state["flurries"]],
                         [(["confetti"], 3, "burst", 12, 1.4, middle), (["red_heart"], 18.8, "rise", 8, 1.2,
                                                                       state["clip"]["layout"]["places"]["top"])])
        self.assertEqual(state["flurries"][1]["pngs"], [effects.sticker_png("red_heart")])
        self.assertLess(state["flurries"][1]["size"], tools.FLURRY_SIZE, "shrunk to move inside the top place")
        self.assertIn("flurries: confetti burst at 3 s, red_heart rise at 18.8 s", self.chat.clip_line())
        plain = dict(state, flurries=[])
        for t in (3.7, 19.4):
            self.assertNotEqual(pixels(render.frame(self.chat.recording, plain, t))[2],
                                pixels(render.frame(self.chat.recording, state, t))[2], "the flurry shows at %g s" % t)
        self.assertEqual(pixels(render.frame(self.chat.recording, plain, 10))[2],
                         pixels(render.frame(self.chat.recording, state, 10))[2], "and only while it moves")

    def test_the_person_chooses_the_colours_of_a_look_for_every_part_or_for_one(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("set_style", style="pop", colours={"key": "#E11D48", "text": "#FFFFFF"}, why="their brand")],
            [call("set_style", style="pop", colours={"text": "#FFE600"}, part="title", why="a yellow title")],
            [call("set_style", style="glow", colours={"edge": "#FFFFFF"}, why="a white outline they asked for")],
            [call("set_style", style="glow", colours={"logo": "#FFFFFF"}, why="no such colour")],
            [call("set_style", style="glow", colours={"key": "red"}, why="not a colour")],
            [call("set_style", style="3d", colours={"key": "#2563EB"}, why="blue everywhere")],
            "Done.")
        red, yellow, white, logo, word, blue = steps[-6:]
        self.assertFalse(red["refused"], red["result"])
        self.assertIn("The words on screen now use the pop style with the letters in #FFFFFF, the key word in #E11D48.", red["result"])
        self.assertIn("now use the pop style with the letters in #FFE600 for the title only.", yellow["result"])
        self.assertIn("Kept from before: the letters in #FFFFFF, the key word in #E11D48.", yellow["result"])
        self.assertIn("Kept from before: the letters in #FFFFFF, the key word in #E11D48, the letters in #FFE600 for the "
                      "title.", white["result"])
        self.assertIn("stands out only", white["result"])
        self.assertIn("kept as asked", white["result"])
        self.assertTrue(logo["refused"])
        self.assertIn("has no field named logo", logo["result"])
        self.assertTrue(word["refused"])
        self.assertIn("is not a colour: give it as #RRGGBB", word["result"])
        style = self.chat.project.state()["style"]
        self.assertEqual((style["name"], style["colours"], style["part_colours"]),
                         ("3d", {"key": "#2563EB", "text": "#FFFFFF", "edge": "#FFFFFF"}, {"title": {"text": "#FFE600"}}),
                         "a colour for every part replaces that colour only")
        self.assertEqual(tools.describe(self.chat.project.entries[-1]),
                         "the 3d style, the letters in #FFFFFF, the key word in #2563EB, the outline in #FFFFFF, the "
                         "letters in #FFE600 for the title")
        self.assertIn("Kept from before: the letters in #FFFFFF, the outline in #FFFFFF, the letters in #FFE600 for the "
                      "title.", blue["result"])
        brain, steps = self.run_job([call("set_style", style="3d", colours={"text": "#FFFFFF"}, why="white again")],
                                    "Done.")
        self.assertNotIn("part_colours", self.chat.project.state()["style"],
                         "the letters' colour for every part takes the title's own colour back")
        self.chat.edit({"do": "style", "name": "pop", "font": None, "colour": "#FFE600"})
        brain, steps = self.run_job([call("set_style", style="pop", colours={"key": "#E11D48"}, why="red key")],
                                    "Done.")
        self.assertEqual(self.chat.project.state()["style"]["colours"], {"text": "#FFE600", "key": "#E11D48"},
                         "a colour from before colours were named is kept as the letters' colour")

    def test_every_look_is_a_picture_the_person_can_tap(self):
        looks = [c for c in tools.chips({"recipe": "game", "length_s": 30, "style": "pop"}) if c.get("look")]
        self.assertEqual([c["look"] for c in looks], list(text.HOUSE_NAMES))
        self.assertEqual([c["look"] for c in looks if c["now"]], ["pop"])
        self.assertEqual(set(tools.STYLES), set(text.HOUSE_NAMES))
        self.assertIn("(tapped) Use the glow style.", [c["say"] for c in looks])

    def test_a_talk_clip_follows_whoever_speaks_and_its_shots_can_be_changed(self):
        self.faces = found([(30 + i * 0.5, [[60, 40, 40, 40], [220, 40, 40, 40]]) for i in range(40)], size=(320, 180))
        self.worker.phrases = [{"from": 1, "to": 9, "text": "hello there", "sure": 0.99},
                               {"from": 10.2, "to": 19, "text": "and you", "sure": 0.99}]
        def moving(rec, a, b, persons, fps=10):
            first = [a + (i + 1) / fps < 40 for i in range(round((b - a) * fps))]
            return [[9.0 if f else 1.0 for f in first], [1.0 if f else 9.0 for f in first]]

        real, talk.motion = talk.motion, moving
        listing, asked = tools._people, []

        def people(*args, **kw):
            asked.append(kw.get("shown"))
            return listing(*args, **kw)

        tools._people = people
        try:
            brain, steps = self.run_job(
                *self.to_moment(), '{"lines": ["hello there", "and you"]}',
                [call("make_clip", moment=1, layout="talk", why="two people talking")],
                [call("change_shot", start=3, end=5, show=2, why="her reaction")],
                [call("change_shot", start=15, end=17, show=0, why="they laugh")],
                [call("change_shot", start=1, end=2, show=3, why="nobody")],
                [call("slow_push", start=1, end=4, why="build-up")],
                [call("move_part", part="circle", box=[600, 700, 900, 900], why="move it")],
                "Done.")
        finally:
            talk.motion, tools._people = real, listing
        made, reaction, together, nobody, push, move = steps[-6:]
        self.assertFalse(made["refused"], made["result"])
        self.assertEqual(asked[0], [p["box"] for p in self.chat.project.state()["clip"]["layout"]["people"]],
                         "the people the clip follows are not listed to blur (10-29: Glimmer blurred a podcast's hosts)")
        self.assertIn("in the talk layout. It follows 2 people, numbered from the left", made["result"])
        self.assertIn("The program cut 2 s out of the moment", made["result"], "the stop between the two is cut")
        self.assertIn("Shots: 0-8.6 s person 1; 8.6-18 s person 2.", made["result"], "the cut is where the pieces join")
        self.assertIn("comes next as a picture", made["result"])
        self.assertIn("Shots now: 0-3 s person 1; 3-5 s person 2; 5-8.6 s person 1; 8.6-18 s person 2.",
                      reaction["result"])
        self.assertIn("Everyone is shown from 15 to 17 s.", together["result"])
        self.assertIn("comes next as a picture", together["result"])
        self.assertTrue(nobody["refused"])
        self.assertIn("show 3 is not one of the people: use 1 to 2", nobody["result"])
        self.assertTrue(push["refused"])
        self.assertIn("no slow push", push["result"])
        self.assertTrue(move["refused"])
        self.assertIn("change_shot", move["result"])
        lay = self.chat.project.state()["clip"]["layout"]
        self.assertEqual([p["name"] for p in lay["parts"]], ["person 1", "person 2", "everyone"])
        times, _ = tools._face_times(self.chat.project.state(), self.chat.recording)
        on = dict(times)
        near = lambda s: on[min(on, key=lambda k: abs(k - s))]
        self.assertEqual((len(near(1.0)), len(near(16.0))), (1, 2),
                         "words keep clear of the face on screen: one person at 1 s, both on the card at 16 s")
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), "changing the shots")
        self.faces = found([(30 + i * 0.5, [[140, 40, 40, 40]]) for i in range(40)], size=(320, 180))
        brain, steps = self.run_job([call("make_clip", moment=1, layout="talk", why="one person")], "Done.")
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("found 1 sitting still", steps[-1]["result"])

    def test_a_split_talk_clip_shows_both_people_and_hides_a_viewers_handle(self):
        self.faces = found([(30 + i * 0.5, [[60, 40, 40, 40], [220, 40, 40, 40]]) for i in range(40)], size=(320, 180))
        hidden = [{"kind": "a person's handle", "box": [200, 10, 100, 20], "start": 1.0, "end": 5.0}]
        had = screen.hidden, screen.ahead, screen.words
        screen.hidden = lambda rec, pieces, on_start=None, span=None: hidden
        screen.ahead, screen.words = (lambda rec, a, b: None), (lambda rec, a, b: [])
        try:
            brain, steps = self.run_job([call("load_recipe", type="talk", why="two people talking")],
                                        *self.to_moment(),
                                        [call("make_clip", moment=1, layout="split", why="both faces, all the time")],
                                        [call("make_clip", moment=1, layout="split",
                                              why="the person said: keep both on screen")], "Done.")
        finally:
            screen.hidden, screen.ahead, screen.words = had
        made = steps[-1]
        self.assertFalse(made["refused"], made["result"])
        self.assertIn("in the split layout. It follows 2 people", made["result"])
        self.assertIn("Person 1 is in the top half and person 2 in the lower half for the whole clip", made["result"])
        self.assertIn("hid 1 private line", made["result"])
        self.assertNotIn("Shots:", made["result"], "nothing in the split layout depends on who speaks")
        state = self.chat.project.state()
        self.assertEqual([(b["private"], b["start"], b["end"]) for b in state["blurs"] if b.get("private")],
                         [("a person's handle", 1.0, 5.0)])
        self.assertEqual(state["clip"]["layout"]["kind"], "split")

    def test_in_a_split_clip_each_line_is_captioned_under_the_face_that_says_it(self):
        self.faces = found([(30 + i * 0.5, [[60, 40, 40, 40], [220, 40, 40, 40]]) for i in range(40)], size=(320, 180))
        self.worker.phrases = [{"from": 1, "to": 9, "text": "hello there", "sure": 0.99},
                               {"from": 10.2, "to": 19, "text": "and you", "sure": 0.99}]
        had = tools._voices
        tools._voices = lambda job, rec, m, persons: (
            [{"from": p["from"], "to": p["to"], "who": 2 - i} for i, p in enumerate(m["said"])],
            "Two voices were heard, one at about 120 Hz and one at about 220 Hz.")
        try:
            brain, steps = self.run_job(*self.to_moment(), '{"lines": ["hello there", "and you"]}',
                                        [call("make_clip", moment=1, layout="split", why="both faces")],
                                        [call("add_captions", why="the words are the clip")], "Done.")
        finally:
            tools._voices = had
        made, captioned = steps[-2:]
        self.assertIn("Two voices were heard", made["result"])
        self.assertIn("each under the face of the person who says it", captioned["result"])
        caps = self.chat.project.state()["captions"]
        lay = self.chat.project.state()["clip"]["layout"]
        self.assertEqual(sorted(caps["bands"]), sorted(lay["bands"]))
        self.assertEqual([line["who"] for line in caps["lines"]], [2] * len([l for l in caps["lines"]
                                                                             if l["phrase"] == 0]) +
                         [1] * len([l for l in caps["lines"] if l["phrase"] == 1]))

    def test_muse_is_offered_every_studio_sticker(self):
        offered = tools.TOOLS["add_sticker"].params["sticker"]["enum"]
        self.assertEqual(offered, [s["name"] for s in effects.stickers()])
        self.assertEqual(len(offered), 60)

    def test_the_clip_knows_when_someone_speaks_in_clip_seconds(self):
        said = {"said": [{"from": 29, "to": 31}, {"from": 39, "to": 46}, {"from": 60, "to": 61}]}
        self.assertEqual(tools._speech({"sound_tracks": 1}, said, [[30, 40], [45, 50]]), [[0, 1], [9, 10], [10, 11]])
        self.assertEqual(tools._speech({"sound_tracks": 0}, {}, [[30, 40]]), [], "no sound: nobody speaks")
        self.assertIsNone(tools._speech({"sound_tracks": 1}, {}, [[30, 40]]), "never heard: nobody knows")

    def test_a_recording_shorter_than_a_clip_is_said_at_once_and_never_proposed_again(self):
        short = os.path.join(self.videos, "short.mp4")
        make_moment(short, seconds=7, busy=(0, 7), loud=(1, 6))
        brain, steps = self.run_job(
            [call("open_recording", path=short, why="the person's clip")],
            [call("look_at_piece", start="0:00", end="0:07", why="all of it")],
            [call("propose_moments", why="the whole", moments=[{"look": 1, "reason": "all of it"}])],
            [call("open_recording", path=self.clip, why="a longer one")],
            [call("look_at_piece", start="0:30", end="0:36", why="a bit")],
            [call("propose_moments", why="that bit", moments=[{"look": 1, "reason": "the tone"}])],
            "Done.")
        opened, _, proposed, _, _, short_look = steps[-6:]
        self.assertIn("It is only 7 s long, shorter than the 10 s a clip needs, so no part of it can be a clip on its "
                      "own. Tell the person, and offer to join it with the other recordings of its folder",
                      opened["result"])
        self.assertTrue(proposed["refused"])
        self.assertIn("short.mp4 is only 7 s long", proposed["result"])
        self.assertIn("Do not propose it again.", proposed["result"])
        self.assertNotIn("Give it a start and an end inside", proposed["result"], "no advice that cannot be followed")
        self.assertTrue(short_look["refused"])
        self.assertIn("Look 1 (0:30-0:36) is shorter than the 10 s a clip needs, so no moment fits inside it. Look at "
                      "a stretch of 10 s or more first.", short_look["result"])

    def test_muse_mutes_a_part_of_the_clip_brings_it_back_and_the_captions_follow(self):
        self.worker.phrases = [{"from": 2, "to": 4.5, "text": "hello there", "sure": 0.99}]
        brain, steps = self.run_job(
            *self.to_moment(), '{"lines": ["hello there"]}',
            [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_captions", why="say what he says")],
            [call("mute_sound", start=0.2, end=1.6, why="a count-in")],
            [call("mute_sound", start=2, end=1, why="backwards")],
            [call("mute_sound", start=5, end=5.05, why="too short")],
            [call("mute_sound", start=5, end=6, back=True, why="nothing muted there")],
            [call("mute_sound", start=1, end=1.6, back=True, why="the person wants that back")],
            [call("mute_sound", start=0, end=0.5, why="the very start")],
            "Done.")
        state = self.chat.project.state()
        line = self.chat.clip_line()
        muted, backwards, short, nothing, back, opening = steps[-6:]
        start = state["clip"]["pieces"][0][0]
        self.assertFalse(muted["refused"], muted["result"])
        self.assertIn("The clip's own sound is muted from 0.2 to 1.6 s (%s to %s of the recording)" % (
            recording.clock(start + 0.2), recording.clock(start + 1.6)), muted["result"])
        self.assertIn("Muted now: 0.2-1.6 s.", muted["result"])
        self.assertRegex(muted["result"], r"\d+ caption words? said there (is|are) off the captions")
        self.assertTrue(backwards["refused"])
        self.assertIn("do not make a time inside the clip", backwards["result"])
        self.assertTrue(short["refused"])
        self.assertIn("is shorter than 0.1 s", short["result"])
        self.assertTrue(nothing["refused"])
        self.assertIn("The sound is not muted anywhere from 5 to 6 s. Muted now: 0.2-1.6 s.", nothing["result"])
        self.assertIn("The clip's own sound is back from 1 to 1.6 s. Muted now: 0.2-1 s.", back["result"])
        self.assertIn("The clip now opens without sound for 0.5 s", opening["result"])
        self.assertEqual(state["mutes"], [[0, 1]])
        self.assertEqual(state["speech"], [[1, 2.8]], "the words before 1 s are not heard")
        self.assertIn("sound muted 0-1 s", line)
        brain, (undone,) = self.run_job([call("undo", why="the person liked the start")], "Done.")
        self.assertIn("Took back the mute at 0-0.5 s.", undone["result"])
        self.assertEqual(self.chat.project.state()["mutes"], [[0.2, 1]])

    def test_muse_adds_sounds_and_the_persons_music_goes_down_while_someone_speaks(self):
        music = os.path.join(self.tmp, "music")
        os.makedirs(music)
        for name, source in (("summer beat.wav", "sine=frequency=440:duration=3"),
                             ("silence.wav", "anullsrc=r=48000:cl=stereo:d=3")):
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", source, os.path.join(music, name)],
                           check=True)
        outside = os.path.join(self.tmp, "other.wav")
        shutil.copy(os.path.join(music, "summer beat.wav"), outside)
        os.environ["LCS_MUSIC"] = music
        self.worker.phrases = [{"from": 2, "to": 4.5, "text": "hello there", "sure": 0.99}]
        try:
            brain, steps = self.run_job(
                *self.to_moment(), '{"lines": ["hello there"]}',
                [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_sound", kind="pop", at=3, why="the word")],
                [call("add_sound", kind="pop", at=3.3, why="a bit later")],
                [call("add_sound", kind="boom", at=1, why="no such sound")],
                [call("add_sound", kind="ding", at=25, why="after the end")],
                [call("set_music", music="summer", why="the person's song")],
                [call("set_music", music=outside, why="not in the folders")],
                [call("set_music", music="silence", why="a silent file")],
                [call("set_music", music="polka", why="no such file")],
                "Done.")
            state = self.chat.project.state()
            line = self.chat.clip_line()
            brain, off = self.run_job([call("set_music", why="the person wants it off")], "Done.")
        finally:
            del os.environ["LCS_MUSIC"]
        pop, later, boom, past, song, outside_, silent, none = steps[-8:]
        self.assertEqual(state["speech"], [[0.3, 2.8]], "the clip opens 0.3 s before the word at 2 s")
        self.assertFalse(pop["refused"], pop["result"])
        self.assertIn("A pop at 3 s: a bubble", pop["result"])
        self.assertIn("It replaces the pop at 3 s.", later["result"])
        self.assertTrue(boom["refused"])
        self.assertIn("kind must be one of: thump, whoosh, pop, ding.", boom["result"])
        self.assertTrue(past["refused"])
        self.assertIn("outside the clip", past["result"])
        self.assertEqual([(s["kind"], s["at"]) for s in state["sounds"]], [("pop", 3.3)])
        self.assertFalse(song["refused"], song["result"])
        self.assertIn("Music: summer beat.wav from its second 0, 6 dB under the voice, and 12 dB lower while someone "
                      "speaks (1 times in this clip).", song["result"])
        self.assertIn("It is 3 s long from there, shorter than the clip, so it plays again from its start.",
                      song["result"])
        self.assertTrue(outside_["refused"])
        self.assertIn("is not inside the folders music may come from", outside_["result"])
        self.assertTrue(silent["refused"])
        self.assertIn("silence.wav is silent", silent["result"])
        self.assertTrue(none["refused"])
        self.assertIn('No music file fits "polka"', none["result"])
        self.assertIn("summer beat.wav", none["result"])
        self.assertEqual(os.path.basename(state["music"]["path"]), "summer beat.wav")
        self.assertEqual((state["music"]["level"], state["music"]["duck"]), (-6, -12))
        self.assertEqual(render._music(state)["path"], state["music"]["path"], "the export takes the state as it is")
        self.assertIn("sounds: pop at 3.3 s, music summer beat.wav", line)
        self.assertFalse(off[-1]["refused"], off[-1]["result"])
        self.assertIsNone(self.chat.project.state()["music"])
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), "taking the music off")

    def test_a_tutorial_hides_what_is_private_on_the_screen_and_the_check_reads_it_again(self):
        mail = [{"text": "Account: someone@example.com", "box": [100, 60, 120, 10], "sure": 0.99}]
        asked = []

        def readings(rec, a, b, on_start=None):
            asked.append((a, b))
            return [{"at": a, "lines": []}, {"at": a + 2, "lines": mail}, {"at": a + 6, "lines": []}]

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            shutil.copy(self.clip, out)
            return out

        had = screen.readings, getattr(render, "export", None)
        screen.readings, render.export = readings, export
        try:
            brain, steps = self.run_job(
                *self.to_moment(), [call("load_recipe", type="explain", why="a tutorial")],
                [call("make_clip", moment=1, layout="card", why="the screen")],
                [call("decide_people", keep=[1], why="it is him")],
                [call("render_clip", why="make the file")], [call("check_clip", why="check it")], "Done.")
            made, hidden, checked = steps[4], self.chat.project.state()["blurs"], steps[-1]
            p = self.chat.project
            p.entries, p.acts = p.entries[:-2], p.acts[:-2]
            _, again = self.run_job([call("decide_people", keep=[1], why="it is him")],
                                    [call("render_clip", why="again")], [call("check_clip", why="check it")], "Done.")
        finally:
            screen.readings, render.export = had
        self.assertIn("The program read the screen and hid 1 private line, each for as long as it shows: an email "
                      "address at 0-6 s. The words themselves are not repeated here", made["result"])
        self.assertEqual([(b["private"], b["start"], b["end"]) for b in hidden], [("an email address", 0, 6)])
        self.assertNotIn("someone@example.com", str(brain.sent), "the private words never reach Muse")
        self.assertIn("- passed: Nothing private can be read on the screen (1 private line, each hidden for as long "
                      "as it shows)", checked["result"])
        self.assertIn("- FAILED: Nothing private can be read on the screen (not hidden: an email address at 0-6 s)",
                      again[-1]["result"])
        self.assertEqual(asked, [(30.0, 50.0)] * 5, "asked over the moment for the card's zoom, for its ground and "
                         "for the private lines when the clip is made (the second and third are the reading kept "
                         "from the first), and at every check")

    def test_the_moments_of_a_tutorial_are_read_ahead_when_they_are_proposed(self):
        ahead = []
        had, screen.ahead = screen.ahead, lambda rec, a, b: ahead.append((a, b))
        try:
            for kind in ("explain", "game", "seller"):
                self.run_job([call("open_recording", path=self.clip, why="open it")],
                             [call("load_recipe", type=kind, why="the kind")],
                             [call("look_at_piece", start="0:30", end="0:50", why="the top stretch")],
                             [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the tone"}])],
                             "Done.")
                ahead.append(kind)
        finally:
            screen.ahead = had
        self.assertEqual(ahead, [(30.0, 50.0), "explain", (30.0, 50.0), "game", "seller"],
                         "a game's moments are read too, for the game's menu")

    def test_the_words_on_the_screen_fix_names_and_a_name_written_another_way_is_pointed_out(self):
        names = []
        real_fix = hearing.fix

        def fix(ask, lines, kind, before=(), names_=()):
            names.append(list(names_))
            return real_fix(ask, lines, kind, before, names_)

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "ดู egret", "sure": 0.7}]
        had = hearing.fix, screen.words, screen.readings, getattr(render, "export", None)
        hearing.fix, render.export = fix, export
        screen.words = lambda rec, a, b: ["Eaglet", "Settings"]
        screen.readings = lambda rec, a, b, on_start=None: [{"at": a, "lines": []}]
        try:
            _, steps = self.run_job(
                [call("open_recording", path=self.clip, why="open it")],
                [call("load_recipe", type="explain", why="a tutorial")],
                [call("look_at_piece", start="0:30", end="0:50", why="the top stretch")],
                [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "สาธิต Egret Settings"}])],
                '{"lines": ["ดู Eaglet"]}',
                [call("make_clip", moment=1, layout="card", why="the screen")],
                [call("decide_people", keep=[1], why="it is him")],
                [call("add_captions", why="the words")],
                [call("add_title", words="Egret on your PC", start=0, end=5, place="top", why="the name")],
                [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                [call("hand_over", title="Egret", caption="c", hashtags=["#Egret"], why="post")],
                [call("hand_over", title="Egret", caption="c", hashtags=["#Egret"], why="as said")], "Done.")
        finally:
            hearing.fix, screen.words, screen.readings, render.export = had
        titled = next(s for s in steps if s["tool"] == "add_title")
        proposed = next(s for s in steps if s["tool"] == "propose_moments")
        refused, ready = steps[-2], steps[-1]
        self.assertIn("(20 s, from look 1): สาธิต Eaglet Settings\n", proposed["result"])
        self.assertIn('The program wrote the names in your reasons as the screen writes them: "Egret" as "Eaglet".',
                      proposed["result"])
        self.assertEqual(self.chat.moments[0]["reason"], "สาธิต Eaglet Settings", "2de1 10-28: the card said Egret")
        self.assertTrue(names and all(n == ["Eaglet", "Settings"] for n in names), names)
        self.assertEqual([line["text"] for line in self.chat.project.state()["captions"]["lines"]], ["ดู Eaglet"],
                         "heard unsure, but the screen wrote the name, so it is captioned")
        self.assertIn('Spelled as the screen writes it? "Egret" is not on the screen, which shows "Eaglet". If the '
                      'speaker meant the screen\'s words, put the title on again.', titled["result"])
        self.assertTrue(refused["refused"])
        self.assertIn('"Egret" is not on the screen, which shows "Eaglet"', refused["result"])
        self.assertIn("hand over again unchanged", refused["result"])
        self.assertFalse(ready["refused"], ready["result"])

    def test_a_tutorial_card_follows_the_action_and_muse_can_turn_it_off(self):
        planned = []

        def plan(rec, pieces, region, aspect, avoid=None, text=None, shown=None):
            planned.append((pieces, region, round(aspect, 2), avoid))
            return {"zoom": 0.5, "size": [160, 104], "shots": [{"from": 0.0, "to": 8.0, "x": 10, "y": 6},
                                                                  {"from": 8.0, "to": 20.0, "x": 150, "y": 70}]}

        had, focus.plan = focus.plan, plan
        screen_had, screen.readings = screen.readings, lambda rec, a, b, on_start=None: [{"at": a, "lines": []}]
        try:
            _, steps = self.run_job(*self.to_moment(), [call("load_recipe", type="explain", why="a tutorial")],
                                    [call("make_clip", moment=1, layout="card", why="the screen")],
                                    [call("make_clip", moment=1, layout="card", follow=False, why="whole")],
                                    [call("make_clip", moment=1, layout="card", crop=[0, 0, 500, 500], why="mine")],
                                    "Done.")
            followed = self.chat.project.entries[-3]["layout"]
        finally:
            focus.plan, screen.readings = had, screen_had
        made, whole, cropped = steps[-3:]
        self.assertFalse(made["refused"], made["result"])
        self.assertIn("The card follows the action on the screen: it shows 160 x 104 of the 320 x 180 picture",
                      made["result"])
        self.assertIn("moves to where the screen changes, 2 places:", made["result"])
        self.assertNotIn("could not be drawn", made["result"], "FFmpeg draws the moving card")
        card = next(p for p in followed["parts"] if p["name"] == "card")
        self.assertEqual((card["src"], card["follow"][1]["x"]), ([10, 6, 160, 104], 150))
        self.assertEqual(len(planned), 1, "only the first make_clip follows")
        self.assertEqual((planned[0][0], planned[0][1]), ([[30.0, 50.0]], [0, 0, 320, 180]))
        self.assertIsNotNone(planned[0][3], "the face camera is avoided")
        self.assertNotIn("follows the action", whole["result"])
        self.assertNotIn("follows the action", cropped["result"])
        self.assertFalse(any(p.get("follow") for p in self.chat.project.state()["clip"]["layout"]["parts"]))

    def grounded(self, lines, *calls):
        """The steps of an explain job making the given calls on a screen holding lines at each of its two looks."""
        looks = [{"text": "line %d" % n, "box": [10, 20 * n, 200, 16], "sure": 0.95} for n in range(lines)]
        had, focus.plan = focus.plan, lambda *a, **k: None
        screen_had, screen.readings = screen.readings, lambda rec, a, b, on_start=None: [
            {"at": a, "lines": looks}, {"at": a + 1, "lines": looks}]
        try:
            return self.run_job(*self.to_moment(), [call("load_recipe", type="explain", why="a lesson")], *calls,
                                "Done.")[1]
        finally:
            focus.plan, screen.readings = had, screen_had

    def test_a_card_on_a_screen_full_of_words_sits_on_a_plain_ground(self):
        steps = self.grounded(8, [call("make_clip", moment=1, layout="card", why="the slide")],
                              [call("make_clip", moment=1, layout="card", crop=[0, 0, 500, 500], why="mine")])
        for step in steps[-2:]:
            self.assertFalse(step["refused"], step["result"])
            self.assertIn("The screen holds about 8 lines of words at a time, so the card sits on a plain ground",
                          step["result"])
        self.assertEqual(self.chat.project.state()["clip"]["layout"].get("ground"), "plain",
                         "a crop of the screen sits on a plain ground too: the ground is drawn from the whole screen")

    def test_a_card_on_a_screen_with_few_words_keeps_its_blurred_copy(self):
        steps = self.grounded(2, [call("make_clip", moment=1, layout="card", why="the camera")])
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertNotIn("plain ground", steps[-1]["result"])
        self.assertIsNone(self.chat.project.state()["clip"]["layout"].get("ground"))

    def test_a_screen_that_cannot_be_read_keeps_the_blurred_copy(self):
        def fails(rec, a, b, on_start=None):
            raise screen.ScreenError("no reader")
        job = type("Job", (), {"working": lambda self, what: None, "done_working": lambda self: None})()
        lay = layouts.make("card", (320, 180))
        had, screen.readings = screen.readings, fails
        try:
            self.assertEqual(tools._plain(job, {"width": 320, "height": 180}, lay, 0.0, 10.0), (lay, ""))
        finally:
            screen.readings = had

    def several(self, *clips_calls, black=False, recipe="explain", quiet=0, shown=("Eaglet",)):
        """A tutorial with two moments proposed, then the given calls; screen and render are faked (the screen shows
        the words in shown), the render black and silent at its start when black, and moment 2's first render silent
        for its first quiet seconds."""
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "one heard", "sure": 0.95}]
        quieted = []

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            hushed = quiet if state["clip"]["moment"] == 2 and not quieted else 0
            if hushed:
                quieted.append(out)
            make_moment(out, seconds=10, busy=(5, 10) if black else (0, 10),
                        loud=(5, 10) if black else (hushed, 10))
            return out

        had = screen.words, screen.readings, focus.plan, getattr(render, "export", None)
        screen.words = lambda rec, a, b: list(shown)
        screen.readings = lambda rec, a, b, on_start=None: [{"at": a, "lines": []}]
        focus.plan, render.export = (lambda *a, **k: None), export
        try:
            return self.run_job(
                [call("open_recording", path=self.clip, why="open it")],
                [call("load_recipe", type=recipe, why="its kind")],
                [call("look_at_piece", start="0:30", end="0:50", why="the top stretch")],
                [call("propose_moments", why="two", moments=[
                    {"look": 1, "start": "0:30", "end": "0:40", "reason": "one"},
                    {"look": 1, "start": "0:40", "end": "0:50", "reason": "two"}])],
                '{"lines": ["one heard"]}', *clips_calls)
        finally:
            screen.words, screen.readings, focus.plan, render.export = had

    def test_several_clips_are_made_in_one_go_and_each_is_reported_on_its_own(self):
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        brain, steps = self.several([call("make_clips", clips=clips, why="the person asked for both")], "Done.")
        made = steps[-1]
        self.assertFalse(made["refused"], made["result"])
        self.assertIn("Made 2 clips, 2 handed over; each is reported on its own:", made["result"])
        self.assertIn("Clip 1 (moment 1, 0:32-0:34, 2.6 s, card layout): handed over, ", made["result"],
                      "the still silence around the one line, from 2 to 4 s, is cut")
        self.assertIn("Clip 2 (moment 2, 0:40-0:50, 10 s, card layout): handed over, ", made["result"])
        self.assertIn("no captions: Nothing was heard clearly", made["result"], "clip 2 has no words")
        ready = [e for e in self.chat.events if e["kind"] == "ready"]
        self.assertEqual([e["title"] for e in ready], ["Eaglet one", "Eaglet two"])
        self.assertEqual(len({e["file"] for e in ready}), 2, "two files")
        self.assertEqual(self.chat.project.state()["clip"]["moment"], 1, "the page shows the first clip")
        self.assertEqual(self.chat.finished()["name"], ready[0]["file"])
        sent = str(brain.sent[-1][-1])
        self.assertIn("Clip 2: The finished clip, picture 4 of 4:", sent, "each clip is looked at as a viewer sees it")
        self.assertEqual(sent.count("The finished clip, picture"), 8)
        self.assertIn("Each clip's finished file comes next in 4 pictures", made["result"])
        self.assertNotIn("The clip at", str(brain.sent[-1]), "no picture of each clip is sent to Muse")

    def test_a_game_streams_clips_each_hide_the_chat_box_muse_gives_once(self):
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("make_clips", clips=clips, why="both")],
                                [call("make_clips", clips=clips, blur=[[600, 0, 1000, 300]], why="both")], "Done.",
                                recipe="game")
        bare, made = steps[-2], steps[-1]
        self.assertTrue(bare["refused"])
        self.assertIn("Nothing was made. A game stream's clips show its chat box", bare["result"])
        self.assertFalse(made["refused"], made["result"])
        self.assertIn("Made 2 clips, 2 handed over", made["result"])
        ready = [e for e in self.chat.events if e["kind"] == "ready"]
        self.assertEqual(len(ready), 2)
        for e in ready:
            self.assertIn(("No viewer's name can be read", True), [(c["name"], c["ok"]) for c in e["checks"]])

    def test_a_title_that_would_cover_a_face_goes_up_where_no_face_is(self):
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        had = tools._face_times
        tools._face_times = lambda state, rec: ([(t / 4, [[0, 0, 1080, 1920]]) for t in range(1, 6)], 0.5)
        try:
            _, steps = self.several([call("make_clips", clips=clips, why="both")], "Done.")
        finally:
            tools._face_times = had
        made = steps[-1]["result"]
        self.assertIn("Clip 2 (moment 2, 0:40-0:50, 10 s, card layout): handed over", made)
        self.assertIn("the title is on the clip only at the top from 1.5 to 10 s, where it covers no face", made)
        self.assertIn("no title on the clip", made, "clip 1, 2.6 s long, has no clear 2 s after the face")

    def test_each_clips_words_are_proofread_together_before_its_title_goes_on(self):
        clips = [{"moment": 1, "title": "ต้อสู้หนึ่ง", "caption": "ต่อสู้ในโกดัง", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        brain, _ = self.several([call("make_clips", clips=clips, why="both")],
                                '{"texts": ["ต่อสู้หนึ่ง", "ต่อสู้ในโกดัง"]}', "Done.")
        ready = [e for e in self.chat.events if e["kind"] == "ready"]
        self.assertEqual(ready[0]["title"], "ต่อสู้หนึ่ง")
        self.assertIn("ต่อสู้หนึ่ง", [t["text"] for t in self.chat.project.state()["titles"]], "the title on the clip too")
        spelling = [m for m in brain.sent if len(m) == 1 and "Fix only the spelling" in m[0]["content"]]
        self.assertEqual(len(spelling), 1, "asked once, before the title went on")
        self.assertIn("ต่อสู้ในโกดัง", spelling[0][0]["content"], "with the post text beside the title")

    def test_a_clip_that_fails_a_check_is_not_handed_over(self):
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("make_clips", clips=clips, why="both")], "Done.", black=True)
        made = steps[-1]
        self.assertIn("Made 2 clips, 0 handed over", made["result"])
        self.assertIn("Clip 1 (moment 1, 0:32-0:34, 2.6 s, card layout): rendered but not handed over, these checks "
                      "failed: It does not start black or silent", made["result"])
        self.assertEqual([e for e in self.chat.events if e["kind"] == "ready"], [])

    def test_a_clip_that_opens_on_silence_is_started_at_its_first_sound(self):
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("make_clips", clips=clips, why="both")], "Done.", quiet=2)
        made = steps[-1]["result"]
        self.assertIn("Made 2 clips, 2 handed over", made)
        self.assertRegex(made, r"Clip 2 \(moment 2, 0:4\d-0:50, 8\.\d+ s, card layout\): handed over, \d of \d checks "
                               r"passed; it opened on silence, so the program started it 1\.8 s later, 0\.2 s before "
                               r"its first sound")
        self.assertAlmostEqual(self.chat.moments[1]["start"], 41.8, delta=0.1)

    def test_a_silence_after_the_first_second_is_not_a_silent_start(self):
        later, first = os.path.join(self.tmp, "later.mp4"), os.path.join(self.tmp, "first.mp4")
        for path, quiet in ((later, "between(t,0.95,1.6)"), (first, "lt(t,0.6)")):
            subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=3", "-af",
                            "volume='if(%s,0,1)':eval=frame" % quiet, path], check=True)
        self.assertEqual(tools._start_of(later, 1), (False, False))
        self.assertEqual(tools._start_of(first, 1), (False, True))

    def test_a_click_in_the_silence_is_not_the_first_sound(self):
        path = os.path.join(self.tmp, "click.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=6", "-af",
                        "volume='if(between(t,1,1.1)+between(t,3,6),1,0)':eval=frame", path], check=True)
        self.assertAlmostEqual(tools._first_sound(path), 3.0, delta=0.05)

    def test_a_name_spelled_another_way_is_written_as_the_screen_writes_it(self):
        clips = [{"moment": 1, "title": "Egret one", "caption": "Egret's c1", "hashtags": ["#Egret", "#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("make_clips", clips=clips, why="both")],
                                [call("make_clips", clips=[clips[1], dict(clips[1], title="again")], why="twice")],
                                "Done.")
        made, twice = steps[-2], steps[-1]
        self.assertFalse(made["refused"], made["result"])
        self.assertIn('The program wrote names as the screen shows them: clip of moment 1: "Egret" as "Eaglet".',
                      made["result"])
        ready = [e for e in self.chat.events if e["kind"] == "ready"]
        self.assertEqual(ready[0]["title"], "Eaglet one")
        self.assertIn("Eaglet's c1", ready[0]["caption"])
        self.assertIn("#Eaglet", ready[0]["hashtags"])
        self.assertTrue(twice["refused"])
        self.assertIn("Each moment makes one clip; moment 2 is given twice.", twice["result"])

    def test_clips_made_in_a_channel_look_get_its_style_captions_name_and_hashtag(self):
        channels.use(channels.save({"name": "Siney Society", "handle": "@sineysociety", "look": "warm",
                                    "style": {"name": "pop", "font": None}, "captions": "lit"})["name"])
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("make_clips", clips=clips, why="both")],
                                [call("save_channel_look", channel="Travel", why="the person asked")],
                                [call("use_channel_look", channel="", why="the person asked for none")], "Done.")
        made, saved, none = steps[-3:]
        self.assertFalse(made["refused"], made["result"])
        self.assertIn('Every clip is in the "Siney Society" channel look the person saved: the pop style, the warm '
                      'colour look, lit captions (add_captions with lit true), the name @sineysociety drawn on the '
                      'clip.', made["result"])
        state = self.chat.project.state()
        self.assertEqual((state["style"]["name"], state["look"], state["captions"]["lit"]), ("pop", "warm", True))
        self.assertIn({"do": "channel", "name": "@sineysociety", "spot": None}, self.chat.project.entries)
        ready = [e for e in self.chat.events if e["kind"] == "ready"][0]
        self.assertEqual(ready["hashtags"], ["#a", "#sineysociety"], "the channel's name joins the post's hashtags")
        self.assertFalse(saved["refused"], saved["result"])
        self.assertIn('Saved the "Travel" channel look: the pop style, the warm colour look, lit captions',
                      saved["result"])
        self.assertEqual(channels.get("Travel")["look"], "warm")
        self.assertEqual(none["result"], "No channel look is in use now: new clips start in the recipe's guesses.")
        self.assertIsNone(channels.in_use())
        self.assertIsNone(self.chat.project.state()["channel"], "the drawn name came off this clip")

    def test_a_name_in_capitals_the_screen_does_not_show_stops_the_clips_once_whatever_the_recipe(self):
        clips = [{"moment": 1, "title": "ASTI pack", "caption": "ASTI c", "hashtags": ["#ASTI", "#SineySociety"]},
                 {"moment": 2, "title": "PDRN pack", "caption": "c", "hashtags": ["#b"]}]
        fixed = dict(clips[0], title="Jabs pack", caption="Jabs c", hashtags=["#Jabs"])
        _, steps = self.several([call("make_clips", clips=clips, why="both")],
                                [call("make_clips", clips=[fixed, clips[1]], why="as the pack says")], "Done.",
                                recipe="seller", shown=["Jabs", "PDRN", "ASTAXANTHIN"])
        asked, made = steps[-2], steps[-1]
        self.assertTrue(asked["refused"])
        self.assertEqual(asked["result"], 'Nothing was made. Clip of moment 1: "ASTI", "SineySociety" are not on the '
                                          'screen there, not said and not written by the person; the screen there '
                                          'shows Jabs, PDRN, ASTAXANTHIN. Write each name as the screen shows it, or '
                                          'leave it out; if the person gave you the name, make the clips again '
                                          'unchanged.', "EP81 10-28: the pack said Jabs, three clips said ASTI")
        self.assertFalse(made["refused"], made["result"])
        self.assertEqual([e["title"] for e in self.chat.events if e["kind"] == "ready"], ["Jabs pack", "PDRN pack"])

    def test_a_tutorial_is_a_card_unless_muse_says_again_the_person_asked_for_their_face(self):
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"], "layout": "face"},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("make_clips", clips=clips, why="a clip of me talking")],
                                [call("make_clips", clips=clips, why="the person said: only my face")], "Done.")
        asked, made = steps[-2:]
        self.assertTrue(asked["refused"], "lane 5 10-28: 'a clip of me talking' on the Eaglet tutorial lost the page")
        self.assertEqual(asked["result"], "Nothing was made. A tutorial's clip shows the screen being explained: the "
                                          "card layout keeps the screen and puts the face in a circle above it. Use "
                                          "face only when the person asked for their face to fill the frame, in words "
                                          "such as 'only my face' or 'just me, big', or the recording has no screen to "
                                          "show; then make it again unchanged.")
        self.assertFalse(made["refused"], made["result"])
        self.assertEqual([e["title"] for e in self.chat.events if e["kind"] == "ready"], ["Eaglet one", "Eaglet two"])

    def test_a_talk_is_faces_unless_muse_says_again_the_person_asked_for_the_screen(self):
        _, steps = self.run_job([call("load_recipe", type="talk", why="two people talking")], *self.to_moment(),
                                [call("make_clip", moment=1, layout="card", why="the screen")],
                                [call("make_clip", moment=1, layout="card", why="the person said: show the screen")],
                                [call("make_clip", moment=1, layout="face", why="one face")], "Done.")
        card, again, face = steps[-3:]
        self.assertTrue(card["refused"])
        self.assertTrue(card["result"].startswith("Not made. A talk's clip cuts between its faces: talk, or face for "
                                                  "one person"), card["result"])
        self.assertIn("Use card only when the person asked for the screen", card["result"])
        self.assertFalse(again["refused"], again["result"])
        self.assertFalse(face["refused"], face["result"])

    def test_a_vlog_keeps_its_whole_height_with_no_face_circle_and_no_face_crop_unasked(self):
        _, steps = self.run_job([call("load_recipe", type="vlog", why="a travel episode")], *self.to_moment(),
                                [call("make_clip", moment=1, layout="face", why="her face")],
                                [call("make_clip", moment=1, layout="card", why="the whole scene")], "Done.")
        face, card = steps[-2:]
        self.assertTrue(face["refused"], "10-29: EP81 cut down to faces showed an eye and hair in the circles")
        self.assertTrue(face["result"].startswith("Not made. A vlog's clip keeps the whole picture"), face["result"])
        self.assertFalse(card["refused"], card["result"])
        self.assertIn("A vlog keeps its whole height", card["result"])
        self.assertIn("The card has no face circle, as nobody looks for one.", card["result"])
        parts = self.chat.project.state()["clip"]["layout"]["parts"]
        self.assertEqual([p["name"] for p in parts], ["card"], "a face the finder sees stays in the picture")

    def test_a_talk_cuts_between_faces_and_circles_are_for_when_the_person_asks(self):
        _, steps = self.run_job([call("load_recipe", type="talk", why="three friends talking")], *self.to_moment(),
                                [call("make_clip", moment=1, layout="speaker", why="everyone at once")], "Done.")
        circles = steps[-1]
        self.assertTrue(circles["refused"], "Tarm 10-29: cut between faces, not faces in circles")
        self.assertIn("Use speaker only when the person asked for the screen or everyone on the screen at once",
                      circles["result"])

    def test_a_clip_whose_recipe_does_not_read_the_screen_is_not_read(self):
        asked = []
        had, screen.readings = screen.readings, lambda rec, a, b, on_start=None: asked.append((a, b)) or []
        try:
            _, steps = self.run_job(*self.to_moment(), [call("load_recipe", type="seller", why="a product")],
                                    [call("make_clip", moment=1, layout="card", why="the product")], "Done.")
        finally:
            screen.readings = had
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertNotIn("read the screen", steps[-1]["result"])
        self.assertEqual(asked, [])

    def test_a_clip_whose_start_was_muted_is_not_failed_for_opening_on_silence(self):
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(2, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            brain, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("decide_people", keep=[1], why="it is him")],
                [call("mute_sound", start=0, end=2, why="a count-in")],
                [call("render_clip", why="make the file")],
                [call("check_clip", why="check it")], "Done.")
        finally:
            render.export = had
        self.assertIn("- not needed: It does not start black or silent (silent for the first 0.5 s, where the sound "
                      "is muted (0-2 s) as asked: music under it would fill it)", steps[-1]["result"])

    def test_the_finished_file_is_checked_before_it_is_handed_over_and_a_failure_is_fixed(self):
        renders = []

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            if renders:
                make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            else:
                shutil.copy(self.clip, out)
            renders.append(out)
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            brain, steps = self.run_job(
                *self.to_moment(), [call("load_recipe", type="game", why="a stream")],
                [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("decide_people", keep=[1], why="it is him")],
                [call("render_clip", why="make the file")],
                [call("hand_over", title="t", caption="c", hashtags=["#a"], why="unchecked")],
                [call("check_clip", why="check it")],
                [call("hand_over", title="t", caption="c", hashtags=["#a"], why="failed")],
                [call("blur_box", box=[400, 0, 600, 300], why="the chat")],
                [call("render_clip", why="again")],
                [call("check_clip", why="again")],
                [call("hand_over", title="t", caption="clips from H:/Phone/yeti", hashtags=["#a"], why="where")],
                [call("hand_over", title="t", caption="c", hashtags=["#reels", "fyp", "TikTokThailand"], why="tags")],
                [call("hand_over", title="t", caption="c", hashtags=["#reels", "#instagram", "#facebook"], why="post")],
                "Done.")
        finally:
            render.export = had
        unchecked, first, failed, _, _, second, where, other, ready = steps[-9:]
        self.assertTrue(where["refused"], "Glimmer's post named the folder of the clips (10-28)")
        self.assertIn('The post names "H:/Phone/yeti", a place on this computer', where["result"])
        self.assertTrue(other["refused"], "Tarm 10-28: no #tiktok, the studio makes Reels for Instagram and Facebook")
        self.assertIn("This names \"#fyp\", \"TikTok\", which belongs to another company's app.", other["result"])
        self.assertFalse(ready["refused"], ready["result"])
        self.assertTrue(unchecked["refused"])
        self.assertIn("Check the finished file first, with check_clip", unchecked["result"])
        self.assertFalse(first["refused"], first["result"])
        self.assertTrue(first["result"].startswith("Checked 1 of 3 on the finished file "), first["result"])
        self.assertIn("- FAILED: No viewer's name can be read (no blur covers the whole clip", first["result"])
        self.assertIn("- FAILED: It does not start black or silent (black and silent for the first 0.5 s)",
                      first["result"])
        self.assertIn("- passed: The length fits a Reel (60.0 s; a Reel is at most 90 s)", first["result"])
        self.assertIn("- not needed: Captions are words that were said (no captions)", first["result"])
        self.assertIn("Fix what failed, render again and check again", first["result"])
        self.assertIn("The finished clip, picture 4 of 4:", str(brain.sent[len(steps) - 4]))
        self.assertIn("4 pictures of the finished clip come next, as a viewer sees it.", first["result"])
        self.assertIn("a viewer's name, a phone number or a face in the chat that can be read", first["result"])
        self.assertTrue(failed["refused"])
        self.assertIn("These checks failed on the finished file: No viewer's name can be read; It does not start black "
                      "or silent.", failed["result"])
        self.assertTrue(second["result"].startswith("Checked 3 of 3 on the finished file "), second["result"])
        self.assertFalse(ready["refused"], ready["result"])
        shown = [e for e in self.chat.events if e["kind"] == "ready"][-1]
        self.assertEqual([c["ok"] for c in shown["checks"]], [None, None, True, None, True, True])
        self.assertEqual(agent.Chat.load(self.chat.id).checks, self.chat.checks)

    def test_a_game_clip_passes_the_names_check_only_when_the_names_read_on_the_screen_are_under_a_blur(self):
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        chat = {"text": "@viewer_one ผมว่าดีครับ", "box": [130, 10, 60, 12], "sure": 0.95}
        had = screen.readings, getattr(render, "export", None)
        screen.readings = lambda rec, a, b, on_start=None: [{"at": a + 1, "lines": [chat]}, {"at": b - 1, "lines": [chat]}]
        render.export = export
        try:
            _, steps = self.run_job(
                *self.to_moment(), [call("load_recipe", type="game", why="a stream")],
                [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("decide_people", keep=[1], why="it is him")],
                [call("blur_box", box=[400, 500, 600, 1000], why="the chat")],
                [call("blur_box", box=[400, 300, 600, 450], why="the chat, lower")],
                [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                [call("blur_box", box=[380, 0, 620, 200], why="the names")],
                [call("render_clip", why="again")], [call("check_clip", why="again")], "Done.")
        finally:
            screen.readings, render.export = had
        on_face, low = [s for s in steps if s["tool"] == "blur_box"][:2]
        first, second = [s for s in steps if s["tool"] == "check_clip"]
        self.assertTrue(on_face["refused"])
        self.assertIn("The viewers' names were read on the screen at [394, 33, 606, 144] on your grid.",
                      on_face["result"], "game afc1 10-28: Muse moved the box down four times, away from the chat")
        self.assertFalse(low["refused"], low["result"])
        self.assertIn("- FAILED: No viewer's name can be read (names can be read at [394, 33, 606, 144] on your grid "
                      "from 0 to ", first["result"], "a blur over the whole clip in the wrong place passed before")
        self.assertIn("- passed: No viewer's name can be read (1 line with a name, under a blur for as long as it "
                      "shows)", second["result"])

    def test_only_the_first_moment_is_fixed_before_the_cards_and_another_when_its_clip_is_made(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "first heard", "sure": 0.95},
                               {"from": 12.0, "to": 14.0, "text": "second heard", "sure": 0.95}]
        script = self.to_moment()
        brain, steps = self.run_job(
            *script[:2], [call("propose_moments", why="two", moments=[
                {"look": 1, "start": "0:30", "end": "0:40", "reason": "one"},
                {"look": 1, "start": "0:40", "end": "0:50", "reason": "two"}])],
            '{"lines": ["first fixed"]}',
            [call("make_clip", moment=2, layout="subject", why="the second")],
            '{"lines": ["second fixed"]}', "Done.")
        proposed, made = steps[-2], steps[-1]
        self.assertIn("Misheard words are fixed in moment 1's lines; the other moments show theirs as heard",
                      proposed["result"])
        self.assertIn("0:42 second heard (as heard)", proposed["result"])
        self.assertFalse(made["refused"], made["result"])
        first, second = self.chat.moments
        self.assertEqual([(s["text"], s["kept"]) for s in first["said"]], [("first fixed", None)])
        self.assertEqual([(s["text"], s["heard"], s["kept"]) for s in second["said"]],
                         [("second fixed", "second heard", None)], "fixed when its clip was made")

    def test_a_moment_does_not_open_in_the_middle_of_a_line(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "first heard", "sure": 0.95}]
        script = self.to_moment()
        _, steps = self.run_job(
            *script[:2], [call("propose_moments", why="mid", moments=[
                {"look": 1, "start": "0:33", "end": "0:45", "reason": "one"}])],
            '{"lines": ["first heard"]}', "Done.")
        self.assertIn("The program moved its edges: it starts 1.2 s earlier, so it opens before a line rather than "
                      "inside it.", steps[-1]["result"])
        self.assertEqual((self.chat.moments[0]["start"], [s["text"] for s in self.chat.moments[0]["said"]]),
                         (31.75, ["first heard"]), "the line it opened inside is now in it")

    def test_captions_can_pop_in_word_by_word_but_not_lit_as_well(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "this is the moment", "sure": 0.95}]
        script = self.to_moment()
        _, steps = self.run_job(
            *script[:2], [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "he falls"}])],
            '{"lines": ["this is the moment"]}', [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_captions", pop=True, why="the person wants words that pop")],
            [call("add_captions", lit=True, pop=True, why="both")], "Done.")
        popped, both = steps[-2], steps[-1]
        self.assertFalse(popped["refused"], popped["result"])
        self.assertIn("each word pops in at its place as it is said and stays", popped["result"])
        captions = self.chat.project.state()["captions"]
        self.assertEqual((captions["pop"], captions["lit"]), (True, False))
        self.assertTrue(both["refused"])
        self.assertIn("Captions are lit or popped, not both", both["result"])
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), "the popped captions")
        self.assertIn("popped captions on", self.chat.clip_line())
        _, steps = self.run_job([call("set_style", style="news", why="subtitles")],
                                [call("add_captions", pop=True, why="pop again")], "Done.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertIn("each line shows whole on its see-through box", steps[-1]["result"],
                      "Muse is not told the words pop in when the look's box shows lines whole")
        self.assertNotIn("pops in", steps[-1]["result"])

    def test_a_line_the_hearing_was_not_sure_of_is_marked_for_muse_and_left_out_of_the_captions(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "this is the moment", "sure": 0.95},
                               {"from": 5.0, "to": 8.0, "text": "bxo noqa mumble", "sure": 0.61},
                               {"from": 9.0, "to": 11.0, "text": "watch him fall", "sure": 0.9}]
        fixed = '{"lines": ["this is the moment", "bxo noqa mumble", "watch him fall"]}'
        script = self.to_moment()
        brain, steps = self.run_job(
            *script[:2], [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "he falls"}])],
            fixed, [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_captions", why="say what he says")],
            "Done.")
        looked = [s for s in steps if s.get("look")][-1]
        self.assertIn("0:35 bxo noqa mumble (not sure)", looked["result"])
        self.assertIn("0:32 this is the moment\n", looked["result"])
        caps = steps[-1]
        self.assertFalse(caps["refused"], caps["result"])
        self.assertIn('1 line the hearing was not sure of is under the clip as a draft, not on the video, because a '
                      'wrong caption is worse than none: 0:03 "bxo noqa mumble". The person keeps or fixes each draft '
                      'there before posting: say so in your answer.', caps["result"])
        shown = " ".join(line["text"] for line in self.chat.project.state()["captions"]["lines"])
        self.assertNotIn("mumble", shown)
        self.assertIn("fall", shown)
        self.assertEqual("".join(line["text"] for line in self.chat.project.state()["captions"]["drafts"]),
                         "bxo noqa mumble")

    def test_a_spot_the_person_asked_for_is_used_and_what_covers_it_is_said(self):
        self.chat.emit("person", job=None, text="Put the title at the bottom. Up top over my face is fine too, and "
                                                "blur my face please")
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_title", words="Boss down", start=0, end=4, box=[50, 900, 950, 1000], why="low")],
            [call("add_title", words="Boss down", start=0, end=4, box=[50, 900, 950, 1000],
                  person_said="I want it at the very bottom", why="claims a request")],
            [call("add_title", words="Boss down", start=0, end=4, box=[50, 900, 950, 1000],
                  person_said="put the title at the bottom", why="the person wants it at the bottom")],
            [call("add_title", words="Up top", start=5, end=9, box=[60, 150, 940, 270], why="over the face")],
            [call("add_title", words="Up top", start=5, end=9, box=[60, 150, 940, 270],
                  person_said="Up top over my face is fine too", why="the person asked")],
            [call("add_title", words="Up top", start=5, end=9, why="no place")],
            [call("blur_box", box=[400, 600, 600, 900], person_said="blur my face please",
                  why="the person wants their face hidden")],
            "Done.")
        low, claimed, low_asked, face, face_asked, nowhere, blurred = steps[-7:]
        self.assertTrue(claimed["refused"])
        self.assertIn('The person did not say "I want it at the very bottom" in this chat.', claimed["result"])
        self.assertTrue(low["refused"])
        self.assertIn("On a phone, the name and the caption cover everything below y 791 on your grid", low["result"])
        self.assertIn("give person_said only when the person asked for this spot", low["result"])
        self.assertFalse(low_asked["refused"], low_asked["result"])
        self.assertIn("Placed where the person asked. On a phone, the name and the caption cover everything below y "
                      "791 on your grid", low_asked["result"])
        self.assertFalse(face["refused"], face["result"])
        self.assertIn("Centred, it would cover a face at", face["result"], "unasked, it steps aside to the clear part")
        self.assertFalse(face_asked["refused"], face_asked["result"])
        self.assertIn("It covers a face at", face_asked["result"])
        self.assertNotIn("clear of every face", face_asked["result"])
        self.assertTrue(nowhere["refused"])
        self.assertIn("Give one of the layout's places for words (top, middle), or a box.", nowhere["result"])
        self.assertFalse(blurred["refused"], blurred["result"])
        titles = self.chat.project.state()["titles"]
        self.assertEqual([(x["text"], x["box"][1]) for x in titles], [("Boss down", 1728), ("Up top", 288)])

    def test_the_face_circle_goes_where_the_person_asks_and_undo_brings_it_back(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("decide_people", keep=[1], why="it is him")],
            [call("move_part", part="circle", box=[650, 800, 950, 980], why="the person wants the face bottom right")],
            [call("move_part", part="circle", box=[0, 0, 50, 20], why="too small")],
            "Done.")
        moved, small = steps[-2:]
        self.assertFalse(moved["refused"], moved["result"])
        self.assertIn("As asked, the circle is at", moved["result"])
        self.assertIn("On a phone", moved["result"])
        self.assertIn("comes next as a picture", moved["result"])
        self.assertTrue(small["refused"])
        state = self.chat.project.state()
        self.assertIn("circle", [part["name"] for part in state["clip"]["layout"]["parts"]])
        self.assertEqual(tools._undecided(state), [], "moving a part keeps the decisions about people")
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), "moving the circle")
        self.chat.project.undo()
        self.assertNotIn("circle", [part["name"] for part in self.chat.project.state()["clip"]["layout"]["parts"]])

    def test_a_title_kept_from_the_clip_before_is_named_and_can_be_taken_off(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_title", words="Boss down", start=0, end=4, place="top", why="the hook")],
            [call("make_clip", moment=1, layout="subject", product=[400, 0, 600, 300], why="again with the product")],
            [call("remove_title", words="Boss up", why="wrong words")],
            [call("remove_title", words="Boss down", why="it covers the product now")],
            "Done.")
        again, wrong, removed = steps[-3:]
        self.assertIn('Kept from before: the title "Boss down" at the top, 0-4 s; it now covers a face or the product at',
                      again["result"])
        self.assertTrue(wrong["refused"])
        self.assertIn('The clip has no title "Boss up"; its titles are: "Boss down".', wrong["result"])
        self.assertFalse(removed["refused"], removed["result"])
        self.assertEqual(self.chat.project.state()["titles"], [])
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), 'taking the title "Boss down" off')

    def test_thai_words_no_dictionary_has_are_found(self):
        self.assertEqual(tools._odd_thai("ดูรายละเอียดเพื่มเติมได้ในลิงก์"), ["เพื่มเติม"])
        self.assertEqual(tools._odd_thai("เครื่องลดความชื้น ใช้งานง่าย แค่เสียบปลั๊กแล้วกดเปิดเลย"), [])
        self.assertEqual(tools._odd_thai("He falls. Twice."), [])
        self.assertEqual(tools._odd_thai("ไตเติลสั้นๆ กับแฮชแท็ก ช่วงที่สวย ๆ"), [])
        self.assertEqual(tools._odd_thai("ต้วเลขสวยๆ"), ["ต้วเลข"])

    def test_glimmers_slips_are_taken_out_and_right_thai_is_left_alone(self):
        mend = tools.words.mend
        self.assertEqual(mend("ชั้่นสสส่วนสรุปประสบการณ์คาสิโนและคำตอบสสุดท้าย"),
                         "ชั้นส่วนสรุปประสบการณ์คาสิโนและคำตอบสุดท้าย")
        self.assertEqual(mend("ดูชั้่นส่วนที่ดีที่สุดเพื่ือเช็กว่าเป็่นช่วงคุย"), "ดูชั้นส่วนที่ดีที่สุดเพื่อเช็กว่าเป็นช่วงคุย")
        self.assertEqual(mend("มมุมกว้างพรมแดงและรถ Mercedes"), "มุมกว้างพรมแดงและรถ Mercedes")
        self.assertEqual(mend("ให้ผู้ใช้เลื่ือก"), "ให้ผู้ใช้เลือก")
        for right in ("คนนั้นบอกว่าจนนานแล้วกันนะ", "เล็กกว่าแบบบนจอออก", "ไม่เคยมีเทสสักตัว บอทที่เปิดแอปปิด",
                      "ถูกกดด้วยมือ ไม่มีสเปรด", "ลงช่องมิลลิวินาที", "ออปชันรองมีเดดไลน์ อ่านบรรทัดในบรีฟ"):
            self.assertEqual(mend(right), right)
        self.assertEqual(mend("H:\\Phone\\ชั้่น"), "H:\\Phone\\ชั้่น", "a path is never changed")
        self.assertEqual(tools.words.tidy("สร้างคลิปแนวคคุยจากชั้่นสสสส่วน"), "สร้างคลิปแนวคุยจากชั้นส่วน",
                         "what Muse writes is mended before it is read back, so the slip is not copied")
        self.assertEqual(tools.words.tidy("สู่พื้�นที่งาน"), "สู่พื้นที่งาน", "half a letter is taken out")

    def test_a_mark_glimmer_wrote_one_letter_late_is_moved_back_before_muse_reads_it_again(self):
        mend = tools.words.mend
        self.assertEqual(mend("ทุกคลิปผ่านการตรวจสอบ 4/4 ขอ้"), "ทุกคลิปผ่านการตรวจสอบ 4/4 ข้อ",
                         "2de1 10-28: the next answer copied the slip, 5 in 100 letters; a / between numbers is no path")
        self.assertEqual(mend("ตอ้งตัง้ค่าใหม่ให้ปรบัจากที่ใช้ไดเ้ลย"), "ต้องตั้งค่าใหม่ให้ปรับจากที่ใช้ได้เลย")
        self.assertEqual(mend("เรนเดอร์คลิปส้ันแนวต้ังท้ังหมด"), "เรนเดอร์คลิปสั้นแนวตั้งทั้งหมด", "a vowel after its tone mark")
        self.assertEqual(mend("เปิดคลิปในโฟลเดอรน์ั้น"), "เปิดคลิปในโฟลเดอร์นั้น")
        for right in ("อัปสกิลใหม่ ทดสอบพิน บิลลิ่ง อิมพลีเมนต์ เรซูเม่", "ใช้งานง่ายและผ่านการตรวจ"):
            self.assertEqual(mend(right), right, "a loanword the dictionary lacks stays as written")
        self.assertEqual(mend("work/chats ขอ้"), "work/chats ขอ้", "a path is never changed")

    def test_a_spelling_fix_is_kept_only_when_it_is_only_a_spelling_fix(self):
        def answer(*texts):
            asked = []

            def ask(messages, schema):
                asked.append(messages)
                return json.dumps({"texts": list(texts)}, ensure_ascii=False)
            return ask, asked

        ask, asked = answer("ตัวเลข 75 ถูก")
        self.assertEqual(words.spelled(ask, ["ok", "ต้วเลข 75 ถูก"]),
                         (["ok", "ตัวเลข 75 ถูก"], [("ต้วเลข 75 ถูก", "ตัวเลข 75 ถูก")]))
        self.assertEqual(len(asked[0]), 1, "a short conversation of its own")
        self.assertIn('["ต้วเลข 75 ถูก"]', asked[0][0]["content"], "only the flagged text is asked about")
        ask, asked = answer("ok")
        self.assertEqual(words.spelled(ask, ["ok", "fine"]), (["ok", "fine"], []))
        self.assertEqual(asked, [], "nothing flagged, nothing asked")
        for wrong in ("ตัวเลข 76 ถูก", "ตัวเลข เจ็ดสิบหก ถูก", "เลขนี้ผิดทั้งหมดแล้วเขียนใหม่ 75", "ต้วเลข 75 ถูก ไมไ่ด้",
                      "ต้วเลข 75 ผิด"):
            self.assertEqual(words.spelled(answer(wrong)[0], ["ต้วเลข 75 ถูก"]), (["ต้วเลข 75 ถูก"], []), wrong)
        self.assertEqual(words.spelled(lambda m, s: "not json", ["ต้วเลข 75 ถูก"]), (["ต้วเลข 75 ถูก"], []))
        kept = "ต้วเลข 75 ถูกต้องครบถ้วนทุกอย่างแล้ว"
        self.assertEqual(words.spelled(answer(kept + "นะ")[0], [kept]), ([kept], []), "the misspelling is still there")

    def test_what_the_viewer_reads_is_proofread_even_where_every_bit_is_a_word(self):
        asked = []

        def ask(messages, schema):
            texts = json.loads(messages[-1]["content"].rsplit("\n\n", 1)[1])
            asked.append(texts)
            fixes = {"ต้อสู้หลายคนแล้วตาย": "ต่อสู้หลายคนแล้วตาย", "อายุเพิ่มที่ไม่คาดคิด": "อายุเพิ่มที่ไม่ได้คาดคิด",
                     "เลือกโมเดลและคาความเร็ว": "เลือกโมเดลและความเร็ว"}
            return json.dumps({"texts": [fixes.get(x, x) for x in texts]}, ensure_ascii=False)

        texts = ["ต้อสู้หลายคนแล้วตาย", "อายุเพิ่มที่ไม่คาดคิด", "เลือกโมเดลและคาความเร็ว", "ok"]
        self.assertEqual(words.spelled(ask, texts), (texts, []), "the dictionary flags none of these")
        self.assertEqual(asked, [])
        kept = {}
        self.assertEqual(words.spelled(ask, texts, every=True, kept=kept),
                         (["ต่อสู้หลายคนแล้วตาย"] + texts[1:], [("ต้อสู้หลายคนแล้วตาย", "ต่อสู้หลายคนแล้วตาย")]),
                         "a tone mark fixed is kept; a reworded line and a dropped word are not")
        self.assertEqual(asked, [texts[:3]], "every Thai text is asked about")
        self.assertEqual(words.spelled(ask, texts[:1] + texts[3:], every=True, kept=kept)[0],
                         ["ต่อสู้หลายคนแล้วตาย", "ok"])
        self.assertEqual(len(asked), 1, "a text fixed once is not asked about again")
        words.spelled(ask, texts[1:3], every=True, kept=kept)
        self.assertEqual(asked[-1], texts[1:3], "a change the rules refused is asked about again")

    def test_a_letter_muse_dropped_from_a_real_word_is_put_back(self):
        def ask(messages, schema):
            fixes = {"ห้อบอลรูมและเวทีแสงสี": "ห้องบอลรูมและเวทีแสงสี", "เลือกโมเดลและคาความเร็ว": "เลือกโมเดลและความเร็ว",
                     "อายุเพิ่มที่ไม่คาดคิด": "อายุเพิ่มที่ไม่ได้คาดคิด"}
            texts = json.loads(messages[-1]["content"].rsplit("\n\n", 1)[1])
            return json.dumps({"texts": [fixes.get(x, x) for x in texts]}, ensure_ascii=False)

        texts = ["ห้อบอลรูมและเวทีแสงสี", "เลือกโมเดลและคาความเร็ว", "อายุเพิ่มที่ไม่คาดคิด"]
        self.assertEqual(words.spelled(ask, texts, every=True),
                         (["ห้องบอลรูมและเวทีแสงสี"] + texts[1:], [(texts[0], "ห้องบอลรูมและเวทีแสงสี")]),
                         "XM 10-28: ห้อ is a word, so only the count says ห้อง is meant; a word dropped or added is not "
                         "a spelling fix")

    def test_a_tone_mark_that_makes_a_rarer_word_is_not_a_fix(self):
        right = "ไว้ใจให้ทำงานแทนตั้งแต่เลือกเงินเดือน"

        def ask(messages, schema):
            fixes = {right: "ไว้ใจให้ทำงานแทนตั้งแต้เลือกเงินเดือน", "สาธิตการตังค่าอัตโนมัติ": "สาธิตการตั้งค่าอัตโนมัติ"}
            texts = json.loads(messages[-1]["content"].rsplit("\n\n", 1)[1])
            return json.dumps({"texts": [fixes.get(x, x) for x in texts]}, ensure_ascii=False)

        self.assertEqual(words.spelled(ask, [right, "สาธิตการตังค่าอัตโนมัติ"], every=True),
                         ([right, "สาธิตการตั้งค่าอัตโนมัติ"], [("สาธิตการตังค่าอัตโนมัติ", "สาธิตการตั้งค่าอัตโนมัติ")]),
                         "round 8: ตั้งแต่ became ตั้งแต้")

    def test_thai_spelling_is_fixed_before_it_is_burned_in_or_handed_over(self):
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            brain, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_title", words="ไมไ่ด้ล้ม", start=0, end=4, place="middle", why="the hook")],
                '{"texts": ["ไม่ได้ล้ม"]}',
                [call("decide_people", keep=[1], why="it is him")],
                [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                [call("hand_over", title="สู้จบ", caption="ดูรายละเอียดเพื่มเติมได้ในลิงก์", hashtags=["#ต้อสู้"],
                      why="post")],
                '{"texts": ["สู้จบ", "ดูรายละเอียดเพิ่มเติมได้ในลิงก์", "#ต่อสู้"]}',
                "Done.")
        finally:
            render.export = had
        title, handed = steps[-5], steps[-1]
        self.assertFalse(title["refused"], title["result"])
        self.assertIn('The program fixed the Thai spelling: "ไมไ่ด้ล้ม" is now "ไม่ได้ล้ม".', title["result"])
        self.assertEqual([t["text"] for t in self.chat.project.state()["titles"]], ["ไม่ได้ล้ม"])
        self.assertFalse(handed["refused"], handed["result"])
        self.assertIn('"ดูรายละเอียดเพื่มเติมได้ในลิงก์" is now "ดูรายละเอียดเพิ่มเติมได้ในลิงก์"', handed["result"])
        ready = [e for e in self.chat.events if e["kind"] == "ready"][-1]
        self.assertEqual((ready["title"], ready["caption"], ready["hashtags"]),
                         ("สู้จบ", "ดูรายละเอียดเพิ่มเติมได้ในลิงก์", ["#ต่อสู้"]), "the hashtags are proofread too")
        aside = [m for m in brain.sent if len(m) == 1]
        self.assertEqual(len(aside), 2, "each fix is asked apart from the job's conversation")

    def test_after_the_job_the_reasons_and_answers_shown_are_spelled_and_the_next_message_cancels_that(self):
        brain, steps = self.run_job(
            [call("open_recording", path=self.clip, why="เปิดไฟล์ที่ระบุ")],
            [call("look_at_piece", start="0:30", end="0:50", why="ดูช่วงที่ไมไ่ด้ดู")],
            [call("propose_moments", why="เสนอ", moments=[{"look": 1, "reason": "ช่วงที่ต้วเลขชัด"}])],
            "เสร็จแล้ว ตวเลขครบ",
            '{"texts": ["ดูช่วงที่ไม่ได้ดู", "ช่วงที่ตัวเลขชัด", "เสร็จแล้ว ตัวเลขครบ"]}')
        self.chat.job.speller.join(10)
        spelled = [e for e in self.chat.events if e["kind"] == "spelled"]
        self.assertEqual(spelled[-1]["fixes"], [
            {"kind": "step", "step": 2, "field": "why", "text": "ดูช่วงที่ไม่ได้ดู"},
            {"kind": "moment", "step": 3, "index": 0, "text": "ช่วงที่ตัวเลขชัด"},
            {"kind": "answer", "index": 0, "text": "เสร็จแล้ว ตัวเลขครบ"}])
        self.assertEqual(len(brain.sent[-1]), 1, "asked apart from the job's conversation")
        whys = [json.loads(c["function"]["arguments"]).get("why") for m in self.chat.messages
                for c in m.get("tool_calls") or []]
        answers = [m["content"] for m in self.chat.messages if m["role"] == "assistant" and not m.get("tool_calls")]
        self.assertIn("ดูช่วงที่ไมไ่ด้ดู", whys, "what Muse reads back is not changed")
        self.assertIn("เสร็จแล้ว ตวเลขครบ", answers)

        waiting = threading.Event()

        def typed_meanwhile(asking):
            waiting.set()
            for _ in range(200):
                if asking.cancelled:
                    raise mind.Cancelled()
                time.sleep(0.05)
            return '{"texts": ["too late"]}'

        brain, steps = self.run_job([call("look_at_piece", start="0:10", end="0:20", why="ดูตวอย่าง")], "ok",
                                    typed_meanwhile)
        self.assertTrue(waiting.wait(10))
        speller = self.chat.job.speller
        real, mind.get = mind.get, lambda key: ScriptedMind("ok")
        try:
            self.chat.say("next", "local")
        finally:
            mind.get = real
        speller.join(3)
        self.assertFalse(speller.is_alive(), "the next message cancels the pass at once")
        self.assertEqual(len([e for e in self.chat.events if e["kind"] == "spelled"]), 1, "the cancelled pass shows nothing")

    def test_a_number_on_screen_must_be_one_that_was_said(self):
        said = ["สินค้าราคา เจ็ดสิบห้า บาท", "ใช้ได้ 50 ตารางเมตร"]
        self.assertEqual(tools._not_said("ลด 75 บาท ห้อง 50-75 m²", said), [])
        self.assertEqual(tools._not_said("ราคา 1,990 บาท ลด 75", said), ["1,990"])
        self.assertEqual(tools._not_said("ลด 20%", []), ["20"])
        self.assertEqual(tools._not_said("75 sqm for 1,990", ["covers seventy-five square metres",
                                                            "one thousand nine hundred and ninety baht"]), [])
        self.assertEqual(tools._not_said("๗๕ ตร.ม.", ["เจ็ดสิบห้า"]), [])
        self.assertEqual(tools._not_said("75 m²", ["七十五平方米"]), ["75"])

    def test_a_seller_clip_in_a_language_whose_numbers_are_not_read_is_checked_by_muse(self):
        self.worker.phrases = [{"from": 2, "to": 4.5, "text": "七十五平方米", "sure": 0.99}]

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            brain, steps = self.run_job(
                [call("open_recording", path=self.clip, why="open it")],
                [call("load_recipe", type="seller", language="Chinese", why="a seller")],
                [call("look_at_piece", start="0:30", end="0:50", why="the top stretch")],
                [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the size"}])],
                '{"lines": ["七十五平方米"]}',
                [call("make_clip", moment=1, layout="subject", why="the product")],
                [call("add_title", words="75 m²", start=0, end=4, place="middle", why="the size")],
                [call("decide_people", keep=[1], why="it is the seller")],
                [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                [call("hand_over", title="75 m²", caption="七十五平方米", hashtags=["#home"], why="post")],
                "Done.")
        finally:
            render.export = had
        made, checked, handed = steps[-6], steps[-2], steps[-1]
        self.assertIn("This is a seller clip and no product box was named, so nothing keeps words off the product.",
                      made["result"])
        self.assertFalse(checked["refused"], checked["result"])
        self.assertIn("- for you to check: Numbers on screen were said (the clip is said in Chinese, and the studio "
                      "reads number words only in Thai and English", checked["result"])
        self.assertFalse(handed["refused"], handed["result"])

    def test_numbers_said_in_words_the_studio_cannot_read_are_left_to_muse_not_failed(self):
        self.chat.clip = {"recipe": "seller", "language": "Chinese"}
        self.assertIn("the clip is said in Chinese, and the studio reads number words only in Thai and English",
                      tools._numbers_unread(self.chat.job or agent.Job(self.chat, ScriptedMind(), "x"), ["七十五"]))
        for language, said in (("Thai", ["เจ็ดสิบห้า"]), ("", ["seventy five"]), ("", ["เจ็ดสิบห้า"])):
            self.chat.clip = {"recipe": "seller", "language": language}
            self.assertIsNone(tools._numbers_unread(agent.Job(self.chat, ScriptedMind(), "x"), said), said)
        self.chat.clip = {"recipe": "seller", "language": ""}
        self.assertIn("a language it does not read", tools._numbers_unread(agent.Job(self.chat, ScriptedMind(), "x"),
                                                                            ["七十五平方米"]))

    def test_a_thai_key_word_is_one_word_of_the_title_not_the_whole_title(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_title", words="การต่อสู้ใกล้ตัว", start=0, end=4, place="top", key="ต่อสู้", why="the hook")],
            [call("add_title", words="การต่อสู้ใกล้ตัว", start=10, end=14, place="top", key="ใกล้", why="again")],
            "Done.")
        part, word = steps[-2:]
        self.assertFalse(part["refused"], part["result"])
        self.assertIn('The key "ต่อสู้" is part of the word "การต่อสู้", so that word is the key.', part["result"])
        self.assertNotIn("so that word is the key", word["result"])
        self.assertEqual([t["key"] for t in self.chat.project.state()["titles"]], ["การต่อสู้", "ใกล้"])

    def test_the_product_is_kept_clear_when_no_face_was_found(self):
        self.faces = found([], size=(320, 180))
        brain, steps = self.run_job(
            *self.to_moment(),
            [call("make_clip", moment=1, layout="subject", product=[400, 450, 600, 900], why="the product")],
            "Done.")
        self.assertIn("No face was found in the moment's 0 pictures.", steps[-1]["result"])
        self.assertIn("- middle (y 472 to 559): never clear for 2 s or more; do not use it", steps[-1]["result"])
        self.assertIn("- top (y 149 to 270): clear of faces for the whole clip", steps[-1]["result"])

    def test_captions_over_the_product_say_the_product_not_a_face(self):
        self.faces = found([], size=(320, 180))
        said = ["เครื่องนี้ดูดความชื้นได้ดีมากจริงๆครับ", "บ้านไหนชื้นมากก็เก็บน้ำได้เต็มถังเลย"]
        self.worker.phrases = [{"from": 1.0, "to": 9.0, "text": said[0]}, {"from": 9.2, "to": 18.5, "text": said[1]}]
        _, steps = self.run_job(
            *self.to_moment()[:2],
            [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the review"}])],
            json.dumps({"lines": said}, ensure_ascii=False),
            [call("make_clip", moment=1, layout="subject", product=[400, 450, 600, 900], why="the product")],
            [call("add_captions", why="say what he says")], json.dumps({"texts": said}, ensure_ascii=False), "Done.")
        caps = steps[-1]["result"]
        self.assertIn("The captions cross the product's box", caps)
        self.assertNotIn("cover a face", caps, "seen 10-29 on the seller's review: no face was under them")

    def test_the_next_message_starts_with_what_the_person_took_back(self):
        self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")], "Done.")
        self.chat.undo()
        brain = ScriptedMind("Fine.")
        import clipstudio.mind as mind
        real, mind.get = mind.get, lambda key: brain
        try:
            job = self.chat.say("make it again", None)
            job.thread.join(10)
        finally:
            mind.get = real
        first = brain.sent[0][-1]["content"]
        self.assertTrue(first.startswith("[studio] Since your last answer: the person took back the clip from moment 1 "
                                         "in the subject layout. Now: no clip made yet"), first)
        self.assertTrue(first.endswith("make it again"))
        self.assertEqual(self.chat.told, [])


class FaceToFollow(unittest.TestCase):
    """A trip seen through glasses, 10-29: riders' faces come and go, the layout follows none, and the answer said "No
    face was found" a few lines above "11 people's faces are in the clip"."""

    def face(self, seen, of=20):
        found = {"samples": [{"t": float(t), "faces": [[100, 100, 40, 40, 0.9]] if t < seen else []}
                             for t in range(of)]}
        job = type("Job", (), {"find_faces": lambda self, rec, a, b: found})()
        return tools._the_face(job, {"width": 1000, "height": 1000}, 0.0, 20.0, None)

    def test_faces_that_come_and_go_are_not_called_none(self):
        box, said = self.face(3)
        self.assertIsNone(box)
        self.assertIn("faces come and go, the one seen most in 3 of the moment's 20 pictures", said)
        self.assertNotIn("No face was found", said)
        self.assertEqual(self.face(0)[1], "No face was found in the moment's 20 pictures.")
        self.assertIsNotNone(self.face(6)[0], "seen in a quarter of them: the layout follows it")


class VlogTrim(unittest.TestCase):
    """EP 2.2 of บ้านไร้นาม, 10-30: four people a quarter of the frame's height on a vlog's whole-picture card."""

    def trim(self, boxes):
        found = {"samples": [{"t": 0.0, "faces": boxes}, {"t": 0.5, "faces": [[0, 0, 900, 900, 0.3]]}]}
        job = type("Job", (), {"find_faces": lambda self, rec, a, b: found})()
        return tools._vlog_trim(job, {"width": 1920, "height": 1080}, 0.0, 20.0)

    def test_a_vlogs_card_is_cut_to_its_faces_and_muse_is_told_how(self):
        part, said = self.trim([[600, 300, 150, 150, 0.9], [1150, 300, 150, 150, 0.95]])
        self.assertEqual(part, [320, 0, 1280, 1080], "a face the finder is not sure of does not hold the card wide")
        self.assertIn("the card shows [167, 0, 833, 1000] on your grid", said)
        self.assertIn("holds 100% of the faces", said)
        self.assertIn("1.5 times as big", said)
        self.assertIn("make the clip again with crop [0, 0, 1000, 1000]", said)

    def test_a_vlog_with_no_face_keeps_its_whole_picture(self):
        self.assertEqual(self.trim([]), (None, tools.VLOG_WHOLE))

    def test_a_vlog_whose_faces_cannot_be_looked_for_is_not_cut(self):
        def fails(self, rec, a, b):
            raise tools.faces.NoFaces("no pictures")
        job = type("Job", (), {"find_faces": fails})()
        part, said = tools._vlog_trim(job, {"width": 1920, "height": 1080}, 0.0, 20.0)
        self.assertIsNone(part)
        self.assertIn("so its sides are not cut", said)


if __name__ == "__main__":
    unittest.main()
