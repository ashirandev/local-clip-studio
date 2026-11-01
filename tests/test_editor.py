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
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, channels, effects, faces, focus, hearing, layouts, mind, project, recording, render, screen, sends, takes, talk, text, tools, words  # noqa: E402,E501
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
                with mock.patch.object(faces, "STILL", 0):
                    got = faces.sample(rec, 2.0, 5.0)
                self.assertEqual([s["t"] for s in got["samples"]], [2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
                self.assertEqual(seen[0], (320 * 180 * 3, 320, 180))
                self.assertEqual(got["samples"][0]["faces"], [[10.0, 20.0, 30.0, 40.0, 0.9]])
                self.assertEqual(faces.kept(rec, 2.0, 5.0), got, "kept, so asking again is instant")
            finally:
                faces._detector, recording.WORK = real, work
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_face_that_never_changes_is_a_drawing_and_not_a_face_camera(self):
        """A drawn woman on Tarm's gold lesson sat in 50 of 50 pictures, and the card cut the slide round her (10-31)."""
        if not FFMPEG:
            self.skipTest("needs ffmpeg")
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "v.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=white:s=160x180:r=10:d=4", "-f", "lavfi",
                            "-i", "testsrc=s=160x180:r=10:d=4,noise=alls=60:allf=t", "-filter_complex", "hstack", "-pix_fmt",
                            "yuv420p", path], check=True)
            work, recording.WORK = recording.WORK, os.path.join(tmp, "work")
            real = faces._detector
            faces._detector = lambda w, h: (lambda data, w, h: [(20, 20, 40, 40, .9), (200, 20, 40, 40, .9)])
            try:
                rec = recording.probe(path)
                got = faces.sample(rec, 0.0, 4.0)
                self.assertGreater(len(got["samples"]), 5)
                self.assertTrue(all(s["faces"] == [[200.0, 20.0, 40.0, 40.0, 0.9]] for s in got["samples"]), got["samples"])
                self.assertEqual(faces.steady(got)["box"], [200.0, 20.0, 40.0, 40.0])
                one = faces.sample(rec, 1.0, 1.4)
                self.assertEqual(len(one["samples"]), 1)
                self.assertEqual(len(one["samples"][0]["faces"]), 2, "one picture cannot tell")
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

    def test_a_picture_at_an_overlapping_time_replaces_the_earlier_one_and_stays_on_its_second(self):
        p = project.Project([{"do": "clip", "moment": 1, "pieces": [[10, 30]], "layout": {"kind": "card"}}])
        for about, at, length in (("a ghost", 2, 3), ("a crowd", 4, 3), ("money", 8, 2), ("a cat", 25, 2)):
            p.add({"do": "picture", "png": "%s.png" % about, "at": at, "length": length, "box": [0, 0, 1080, 900],
                   "about": about})
        self.assertEqual([(k["about"], k["at"]) for k in p.state()["pictures"]], [("a crowd", 4), ("money", 8)],
                         "a crowd at 4-7 s went over a ghost's 2-5 s; a cat at 25 s is after the 20 s clip")
        p.add({"do": "clip", "moment": 1, "pieces": [[13, 30]], "layout": {"kind": "card"}, "moved": True})
        self.assertEqual([(k["about"], k["at"]) for k in p.state()["pictures"]], [("a crowd", 1), ("money", 5)],
                         "each stays on the second of the recording it was put on")

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
    that stays low in the picture, and from 35 s to 40 s one that rises to the top. Its tone plays the same where the
    fake hearing puts words and where it puts none, so the sound's level says nothing here and is left unknown
    (takes.heard_level), as before it was read; test_a_sound_as_loud_as_the_talk_after_the_last_line_stays reads it."""

    @classmethod
    def setUpClass(cls):
        cls.made = tempfile.mkdtemp()
        cls.source = os.path.join(cls.made, "moment.mp4")
        make_moment(cls.source)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.made, ignore_errors=True)

    def setUp(self):
        self.edge_recipes = tools.EDGE_RECIPES
        self.addCleanup(setattr, tools, "EDGE_RECIPES", self.edge_recipes)
        tools.EDGE_RECIPES = ()
        self.short_recipes = tools.SHORT_RECIPES
        self.addCleanup(setattr, tools, "SHORT_RECIPES", self.short_recipes)
        tools.SHORT_RECIPES = ()
        self.product_recipes = tools.PRODUCT_RECIPES
        self.addCleanup(setattr, tools, "PRODUCT_RECIPES", self.product_recipes)
        tools.PRODUCT_RECIPES = ()
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
        self.level, takes.heard_level = takes.heard_level, lambda rec, a=0.0, b=None: None
        self.chat = agent.Chat()

    def tearDown(self):
        faces.sample, faces.kept = self.real
        takes.heard_level = self.level
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

    def test_a_title_with_room_grows_and_keeps_its_size_where_growing_would_cover_a_face(self):
        """Matrix 10-30: every title stood 61 px tall, a third of the frame wide, on 5 of 8 clips."""
        import re
        from unittest import mock

        def faces_at(boxes):
            return lambda state, rec: ([(k * 0.5, boxes) for k in range(41)], 0.5)

        with mock.patch.object(tools, "_face_times", faces_at([])):
            _, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_title", words="Boss down", start=0, end=5, place="top", why="the hook")], "Done.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        state = self.chat.project.state()
        title = state["titles"][-1]
        self.assertEqual(title.get("grow"), render.TITLE_GROW)
        own = text.measure("Boss down", render.style_of(state, "clean"), "title", title["box"][2])
        tall = int(re.search(r"1 line, \d+ x (\d+) px", steps[-1]["result"]).group(1))
        self.assertGreater(tall, own.h * 1.4, "the result tells Muse the size it is drawn at")
        under = title["box"][1] + (title["box"][3] + own.h) / 2 + tools.FACE_MARGIN + 1
        with mock.patch.object(tools, "_face_times", faces_at([[title["box"][0], under, title["box"][2], 200]])):
            _, steps = self.run_job(
                [call("add_title", words="Boss down", start=6, end=11, place="top", why="the hook")], "Done.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertIn("clear of every face", steps[-1]["result"], "at its own size it covers none")
        self.assertNotIn("grow", self.chat.project.state()["titles"][-1], "grown, it would cover the face under it")
        said = text.caption_lines([{"from": 12.0, "to": 15.0, "text": "ซื้อถูกขายแพงได้กำไรครับ"}], 14)
        self.chat.project.add({"do": "captions", "lines": said, "band": [layouts.CAPTIONS[1], layouts.CAPTIONS[3]],
                               "one": True, "lit": False, "pop": False})
        with mock.patch.object(tools, "_face_times", faces_at([])):
            _, steps = self.run_job(
                [call("add_title", words="Boss down", start=12, end=17, place="top", why="later")], "Done.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertEqual(self.chat.project.state()["titles"][-1].get("grow"), render.TITLE_GROW,
                         "over one-line captions too: which step came first does not decide how big it is")

    def test_a_refusal_offers_a_clear_part_that_holds_the_title_longer_than_the_place_is_clear_across(self):
        """The green seller 10-30: the frame followed the product and the head came into the top place at times, the
        refusal named only the stretches clear across its width, the first 0-2.7 s, and the title held 2.7 s of 31
        though a part of the top held it from 0 to 17.5 s."""
        from unittest import mock

        def times(state, rec):
            return [(k * 0.5, [] if k * 0.5 < 3 else [[480, 300, 300, 400]] if k * 0.5 < 12 else
                     [[65, 300, 950, 400]]) for k in range(41)], 0.5

        with mock.patch.object(tools, "_face_times", times):
            brain, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_title", words="Boss down", start=0, end=15, place="top", why="the hook")], "Done.")
        refused = steps[-1]
        self.assertTrue(refused["refused"])
        self.assertIn("Clear: top: 0-2.7 s, or its part at x 61 to 425, y 149 to 270 on your grid at 0-11.5 s; middle: "
                      "0-20 s.", refused["result"])

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

    def test_a_games_card_stands_tall_and_another_kind_of_videos_card_does_not(self):
        _, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="card", why="the screen")],
                                [call("load_recipe", type="game", why="a stream")],
                                [call("make_clip", moment=1, layout="card", why="the game")], "Done.")
        plain, _, game = steps[-3:]
        self.assertFalse(game["refused"], game["result"])
        self.assertIn("The card was cut at the sides to stand tall, so the game shows bigger.", game["result"])
        lay = self.chat.project.state()["clip"]["layout"]
        self.assertTrue(lay.get("tall"))
        self.assertEqual([p["dst"] for p in lay["parts"] if p["name"] == "card"], [[60, 774, 960, 682]])
        self.assertNotIn("stand tall", plain["result"])

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

    def test_a_word_no_dictionary_has_sends_a_title_back_once_and_the_person_may_type_what_they_like(self):
        said = "ตายแล้วรีสไตน์ภายในวินาที"
        _, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="card", why="the game")],
            [call("add_title", words=said, start=1, end=3, place="top", why="the hook")],
            '{"texts": ["%s"]}' % said,
            [call("add_title", words=said, start=1, end=3, place="top", why="the game's own word")],
            "Done.")
        first, second = steps[-2:]
        self.assertTrue(first["refused"])
        self.assertIn('no dictionary has "รีสไต"', first["result"])
        self.assertFalse(second["refused"], second["result"])
        self.assertEqual([t["text"] for t in self.chat.project.state()["titles"]], [said])
        self.assertIsNone(self.chat.set_words("title", 0, said, "รีสไตน์อีกรอบ"), "what the person types is theirs")
        self.assertEqual([t["text"] for t in self.chat.project.state()["titles"]], ["รีสไตน์อีกรอบ"])

    def test_a_title_the_proofread_made_into_two_short_words_still_goes_back_once(self):
        """015633-a2c5 on 11-01: Glimmer's อายุขึ้อน for อายุขึ้น, made อายุขี้อน, stayed on screen for 59 s."""
        wrote, proofread, meant = "ตายซ้ำๆ อายุขึ้อนโดนยิง", "ตายซ้ำๆ อายุขี้อนโดนยิง", "ตายซ้ำๆ อายุขึ้นโดนยิง"
        _, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="card", why="the game")],
            [call("add_title", words=wrote, start=1, end=3, place="top", why="the hook")],
            '{"texts": ["%s"]}' % proofread,
            [call("add_title", words=meant, start=1, end=3, place="top", why="the word I meant")],
            "Done.")
        first, second = steps[-2:]
        self.assertTrue(first["refused"], first["result"])
        self.assertIn('no dictionary has "', first["result"])
        self.assertFalse(second["refused"], second["result"])
        self.assertEqual([t["text"] for t in self.chat.project.state()["titles"]], [meant])

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

    def test_a_lone_foreign_word_on_a_laugh_is_no_caption_and_capitals_stay(self):
        """Voice wrote "Yeah," for a laugh among EP0's Thai lines (10-31)."""
        said = [{"from": 2.0, "to": 4.0, "text": "ราคาถูกมาก"}, {"from": 5.0, "to": 6.5, "text": "Yeah,"},
                {"from": 7.0, "to": 8.0, "text": "OK"}, {"from": 9.0, "to": 11.0, "text": "ของดีมาก"}]
        self.assertEqual([tools._stray(p, said) for p in said], [False, True, False, False])
        long_ = dict(said[1], to=7.5)
        self.assertFalse(tools._stray(long_, said), "two seconds of it is someone saying it")
        english = [{"from": 1.0, "to": 2.0, "text": "Yeah,"}, {"from": 3.0, "to": 5.0, "text": "that is right"}]
        self.assertFalse(tools._stray(english[0], english), "English among English lines is what was said")
        self.worker.phrases = said
        script = self.to_moment()
        self.run_job(*script[:2], [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the setup"}])],
                     '{"lines": ["ราคาถูกมาก", "Yeah,", "OK", "ของดีมาก"]}',
                     [call("make_clip", moment=1, layout="subject", why="the action")],
                     [call("add_captions", why="say what he says")],
                     '{"texts": ["ราคาถูกมาก", "OK", "ของดีมาก"]}', "Done.")
        shown = "".join(line["text"] for line in self.chat.project.state()["captions"]["lines"])
        self.assertNotIn("Yeah", shown)
        self.assertIn("OK", shown)

    def test_a_talk_clip_leaves_out_the_lines_said_while_the_video_shows_words_of_its_own_in_the_frame(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "this is the moment"},
                               {"from": 5.0, "to": 8.0, "text": "watch him fall down now"}]
        _, steps = self.run_job(
            *self.to_moment()[:2], [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "talk"}])],
            '{"lines": ["this is the moment", "watch him fall down now"]}',
            [call("make_clip", moment=1, layout="subject", why="the talk")], "Done.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        from unittest import mock
        x, _, w, _ = next(p for p in self.chat.project.state()["clip"]["layout"]["parts"] if p["name"] == "whole")["src"]
        asked = []

        def own(rec, pieces, on_start=None):
            asked.append(pieces)
            return [{"box": [x + 10, 140, w - 20, 20], "start": 0.0, "end": 2.5},
                    {"box": [x + w + 5, 140, 40, 20], "start": 3.0, "end": 7.0}]
        with mock.patch.object(tools.screen, "own_words", own):
            self.chat.clip = {"recipe": "seller", "title": "Review", "length_s": 30, "style": "clean", "language": "",
                              "brief": "", "subtitles": None}
            _, steps = self.run_job([call("add_captions", why="the words")],
                                    '{"texts": ["this is the moment", "watch him fall down now"]}', "Done.")
            self.assertEqual(asked, [], "a seller's pack has words of its own that are not subtitles")
            self.chat.clip["recipe"] = "talk"
            _, steps = self.run_job([call("add_captions", why="the words")],
                                    '{"texts": ["this is the moment", "watch him fall down now"]}', "Done.")
        caps = steps[-1]
        self.assertFalse(caps["refused"], caps["result"])
        self.assertEqual(len(asked), 1)
        shown = "".join(line["text"] for line in self.chat.project.state()["captions"]["lines"])
        self.assertNotIn("moment", shown, "said while the video's own words show in the frame")
        self.assertIn("fall down", shown, "the second words are outside the frame, so they are not seen")
        self.assertIn("The video shows words of its own in the clip's frame (an editor's subtitles or names) at 0-2.5 s, "
                      "so the 2 caption lines said then are left out: two sets of words fight. The other 2 are drawn.",
                      caps["result"])
        self.assertIn("Captions on: %d lines" % len(self.chat.project.state()["captions"]["lines"]), caps["result"])

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

    def test_a_clip_talked_through_is_talked_though_the_hearing_was_unsure_of_some_lines(self):
        """EP0 talk 10-30: the sure lines filled 26.7 s of 45.3, the unsure ones (drafts) 16.6 s more, and the clip got
        two small lines, 0.5 s short of TALKED."""
        said = ["เครื่องนี้ดูดความชื้นได้ดีมากจริงๆครับ", "บ้านไหนชื้นมากก็เก็บน้ำได้เต็มถังเลย"]
        self.worker.phrases = [{"from": 1.0, "to": 9.0, "text": said[0]},
                               {"from": 9.2, "to": 18.5, "text": said[1], "sure": 0.3}]
        _, steps = self.run_job(
            *self.to_moment()[:2],
            [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the review"}])],
            json.dumps({"lines": said}, ensure_ascii=False),
            [call("make_clip", moment=1, layout="subject", why="the review")],
            [call("add_captions", why="say what he says")],
            json.dumps({"texts": said[:1]}, ensure_ascii=False), "Done.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        got = self.chat.project.state()["captions"]
        self.assertTrue(got["drafts"], "the unsure line is a draft, not on the video")
        self.assertTrue(got.get("one"), "said through, it is talked: its captions come one big line at a time")
        self.assertFalse(tools._talked({}, [{"from": 1.0, "to": 9.0}], 20.0), "the sure line alone fills 40%")

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
        self.assertIn('1 line, 353 x 81 px', moved["result"], "a title behind the person grows as far as its box holds it "
                      "(PS5 seller 10-30: 477 x 62 px under the hands)")
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
        self.assertEqual({t["text"]: t.get("grow") for t in state["titles"]},
                         {"Boss down": tools.render.TITLE_GROW, "Still here": tools.render.TITLE_GROW},
                         "behind the person, a title grows over the face it crosses: they stand in front of it")
        self.assertTrue(glow["refused"])
        self.assertIn("look must be one of: grey, outline.", glow["result"])
        self.assertIn("The person could not be cut out: Muse Glimmer could not step off the graphics card for SAM "
                      "(refused). Tell the person.", failed[-1]["result"])
        self.assertEqual(len(asked), 2, "SAM runs again only when the clip's stretch is not covered")
        self.assertEqual(self.chat.project.state()["mask"]["to"], 1, "a failed run leaves the clip as it was")

    def test_a_title_behind_the_person_grows_only_while_they_leave_it_readable(self):
        def cut_out(rec, start, end, face=None, on_progress=None, on_start=None, people=None):
            path = os.path.join(self.tmp, "person.mkv")
            subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=10:d=%g"
                            % (end - start), "-vf", "drawbox=x=0:y=0:w=160:h=180:color=white:t=fill,format=gray",
                            "-c:v", "ffv1", path], check=True)
            return {"path": path, "person": "main", "fps": 10, "size": [320, 180], "from": start, "to": end}

        had = tools.person.cut_out
        tools.person.cut_out = cut_out
        try:
            _, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("cut_out", look="grey", why="make him stand out")],
                [call("add_title", words="Go now", start=2, end=6, box=[344, 400, 744, 522], behind=True, why="a")],
                [call("add_title", words="Go now", start=8, end=12, box=[364, 400, 764, 522], behind=True, why="b")],
                "Done.")
        finally:
            tools.person.cut_out = had
        near, clear = steps[-2:]
        self.assertFalse(near["refused"], near["result"])
        self.assertIn("1 line, 200 x 51 px", near["result"], "grown, the person would hide 39% of it")
        self.assertIn("hide about 32% of their width", near["result"])
        self.assertFalse(clear["refused"], clear["result"])
        self.assertIn("1 line, 319 x 81 px", clear["result"])
        self.assertIn("hide about 32% of their width", clear["result"], "23% of the box at its own size")
        self.assertEqual([t.get("grow") for t in self.chat.project.state()["titles"]], [None, tools.render.TITLE_GROW])

    def test_with_the_switch_on_wonder_names_who_speaks_and_the_shots_follow_it(self):
        """Tarm 10-31 by ear on EP0's four men: "EP0 ถูกต้อง!", Wonder right on 6 turns of 6, where mouths moved it."""
        from unittest import mock
        self.faces = found([(30 + i * 0.5, [[60, 40, 40, 40], [220, 40, 40, 40]]) for i in range(40)], size=(320, 180))
        self.worker.phrases = [{"from": 1, "to": 9, "text": "hello there", "sure": 0.99},
                               {"from": 10.2, "to": 19, "text": "and you", "sure": 0.99}]

        def moving(rec, a, b, persons, fps=10):
            first = [a + (i + 1) / fps < 40 for i in range(round((b - a) * fps))]
            return [[9.0 if f else 1.0 for f in first], [1.0 if f else 9.0 for f in first]]

        class Namer:
            key, label, model_id = "wonder", "Muse Wonder", "muse-wonder-1.0"
            answers = ['{"who": 2, "sure": 0.9, "why": "his mouth"}', '{"who": 1, "sure": 0.9, "why": "hers"}']
            sent = []

            def secret(self):
                return "a key"

            def ask(self, messages, tools=None, temperature=None, max_tokens=None, on_delta=None, asking=None,
                    schema=None):
                self.sent.append(messages)
                return {"text": self.answers[len(self.sent) - 1], "thinking": "", "tool_calls": [],
                        "message": {"role": "assistant"}, "engine": {"usd": 0.05}}

        class NoCount:
            key, label, model_id = "spark", "Muse Spark", "muse-spark-1.3"

            def ask(self, *args, **kwargs):
                raise tools.mind.MindError("it did not answer")

        turns = mock.Mock(return_value=([(0.0, 10.0, "A"), (10.0, 20.0, "B")], 0.001))
        real, talk.motion = talk.motion, moving
        wonder, tools.mind.MINDS["wonder"] = tools.mind.MINDS.get("wonder"), Namer()
        try:
            with mock.patch.object(tools, "_wonder_names", lambda: True), mock.patch.object(tools.asr, "turns", turns), \
                    mock.patch.dict(tools.mind.MINDS, spark=NoCount()):
                _, steps = self.run_job(*self.to_moment(), '{"lines": ["hello there", "and you"]}',
                                        [call("make_clip", moment=1, layout="talk", why="two people talking")],
                                        [call("make_clip", moment=1, layout="talk", why="again")], "Done.")
        finally:
            talk.motion = real
            tools.mind.MINDS["wonder"] = wonder
        first, again = steps[-2:]
        self.assertIn("Muse Spark could not count the people (it did not answer), so only the 2 faces found have numbers.",
                      first["result"])
        self.assertIn("Muse Wonder named who speaks in 2 of 2 turns from their voices and faces ($0.10).", first["result"])
        self.assertIn("Who speaks was named by Muse Wonder earlier", again["result"])
        self.assertEqual(len(Namer.sent), 2, "the clip made again asks nobody again")
        self.assertEqual(turns.call_count, 1)
        clip = self.chat.project.state()["clip"]
        self.assertEqual([v["who"] for v in clip["voices"]], [2, 1])
        self.assertEqual(clip["layout"]["shots"][0]["show"], 2, "the mouths said person 1")
        with open(os.path.join(tools.recording.WORK, "spent.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(sum(1 for r in rows if r["mind"] == "wonder"), 2)
        self.assertEqual(sum(1 for r in rows if r["mind"] == "voice" and r["usd"] == 0.001), 1)

    def test_someone_the_faces_missed_is_counted_named_and_shown(self):
        """EP0 10-31: Wonder had no number for the man turned away, and gave his turn to person 1."""
        from unittest import mock
        self.faces = found([(30 + i * 0.5, [[60, 40, 40, 40], [160, 40, 40, 40]]) for i in range(40)], size=(320, 180))
        self.worker.phrases = [{"from": 1, "to": 9, "text": "hello there", "sure": 0.99},
                               {"from": 10.2, "to": 19, "text": "and you", "sure": 0.99}]

        class Asked:
            def __init__(self, key, label, answers):
                self.key, self.label, self.model_id, self.answers, self.sent = key, label, key, answers, []

            def secret(self):
                return "a key"

            def ask(self, messages, tools=None, temperature=None, max_tokens=None, on_delta=None, asking=None,
                    schema=None):
                self.sent.append(messages)
                return {"text": self.answers[len(self.sent) - 1], "thinking": "", "tool_calls": [],
                        "message": {"role": "assistant"}, "engine": {"usd": 0.01}}

        heads = {"people": [{"head": [170, 200, 320, 500], "looks": "left"}, {"head": [480, 200, 620, 500], "looks": "middle"},
                            {"head": [780, 200, 920, 500], "looks": "turned away on the right"}]}
        spark = Asked("spark", "Muse Spark", [json.dumps(heads)])
        wonder = Asked("wonder", "Muse Wonder", ['{"who": %d, "sure": 0.9, "why": "seen"}' % who for who in (3, 1, 2, 2, 1, 3, 2)])
        turns = mock.Mock(return_value=([(0.0, 14.0, "A"), (14.0, 17.0, "B"), (17.0, 20.0, "C")], 0.001))
        with mock.patch.object(tools, "_wonder_names", lambda: True), mock.patch.object(tools.asr, "turns", turns), \
                mock.patch.dict(tools.mind.MINDS, spark=spark, wonder=wonder), \
                mock.patch.object(talk, "motion", lambda rec, a, b, persons, fps=10: [[1.0] * 30 for _ in persons]):
            _, steps = self.run_job(*self.to_moment(), '{"lines": ["hello there", "and you"]}',
                                    [call("make_clip", moment=1, layout="talk", why="three people talking")],
                                    [call("make_clip", moment=1, layout="talk", why="again")], "Done.")
        first, again = steps[-2:]
        self.assertIn("It follows 3 people", first["result"])
        self.assertIn("Muse Spark counted 3 people in one picture, and the faces found were 2", first["result"])
        self.assertEqual(len(spark.sent), 1, "the clip made again asks nobody again")
        self.assertIn("image_url", json.dumps(spark.sent[0]))
        self.assertEqual(len(wonder.sent), 7, "a short turn is asked twice, and a third time when the two differ")
        self.assertIn("near their head: 1, left; 2, middle; 3, turned away on the right.", json.dumps(wonder.sent[0]))
        clip = self.chat.project.state()["clip"]
        self.assertEqual([v["who"] for v in clip["voices"]], [3, 1, 2, None], "three answers that differ name nobody")
        self.assertEqual(clip["voices"][0]["to"], 39.0, "one long turn, named in pieces cut where a heard line ends")
        self.assertEqual(clip["layout"]["shots"][0]["show"], 3, "the zoom goes to him")
        self.assertIn("person 3", [p["name"] for p in clip["layout"]["parts"]])

    def test_wonder_names_only_when_the_meta_hearing_is_on(self):
        from unittest import mock

        class Keyed:
            def secret(self):
                return "a key"

        with mock.patch.dict(tools.mind.MINDS, {"wonder": Keyed()}):
            with mock.patch.object(tools.hearing, "engine", lambda: "local"):
                self.assertFalse(tools._wonder_names(), "with the switch off nothing goes to Meta")
            with mock.patch.object(tools.hearing, "engine", lambda: "meta"):
                self.assertTrue(tools._wonder_names())
        with mock.patch.dict(tools.mind.MINDS, {"wonder": type("NoKey", (), {"secret": lambda self: ""})()}), \
                mock.patch.object(tools.hearing, "engine", lambda: "meta"):
            self.assertFalse(tools._wonder_names(), "no key, no Wonder")

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
        whole = {k: v for k, v in state["mask"].items() if k != "do"}
        self.chat.edit(dict(whole, do="mask", everyone=whole["path"]))
        _, told = self.run_job([call("cut_out", look="grey", why="everyone in the talk")], "Done.")
        self.assertIn("only the room is black and white and everyone keeps their colour", told[-1]["result"])
        _, asked_for = self.run_job([call("cut_out", look="grey", everyone=True, why="they said everyone keeps colour")],
                                    "Done.")
        self.assertIn("everyone keeps their colour, as the person asked", asked_for[-1]["result"])
        self.assertTrue(self.chat.project.state()["cutouts"][-1]["everyone"])
        self.chat.emit("person", text="ตัดตัวคนออกจากฉากด้วย SAM ให้ทุกคนคงสีไว้ ฉากรอบข้างเป็นขาวดำ")
        _, set_for = self.run_job([call("cut_out", look="grey", why="the grey look")], "Done.")
        self.assertIn("everyone keeps their colour, as the person asked", set_for[-1]["result"], "072526-70cc on 11-01")
        self.assertTrue(self.chat.project.state()["cutouts"][-1]["everyone"])
        self.run_job([call("cut_out", look="outline", why="an edge")], "Done.")
        self.assertFalse(self.chat.project.state()["cutouts"][-1]["everyone"], "colour words say nothing about an edge")
        self.chat.emit("person", text="เอาคนพูดคนเดียวที่มีสี")
        self.run_job([call("cut_out", look="grey", why="the speaker alone")], "Done.")
        self.assertFalse(self.chat.project.state()["cutouts"][-1]["everyone"], "the newest words about colour win")

    def test_everyone_in_colour_is_read_from_the_persons_words(self):
        for words in ("ให้ทุกคนคงสีไว้", "ทุกคนในคลิปมีสี ฉากขาวดำ", "คงสีทุกคน", "keep everyone in colour",
                      "everyone keeps their color", "colour on everyone, grey room"):
            self.assertTrue(tools.ALL_IN_COLOUR.search(words), words)
        for words in ("ทุกคนเป็นสีขาวดำ", "ไม่ให้ทุกคนมีสี", "อย่าให้ทุกคนมีสี", "คนพูดคงสีคนเดียว", "not everyone keeps colour",
                      "only the speaker keeps colour, not everyone", "everyone in black and white, no colour"):
            self.assertIsNone(tools.ALL_IN_COLOUR.search(words), words)

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
            '{"lines": ["設定は簡単です。", "とても安いです。"]}', "Done.")
        left, again, japanese = steps[-3:]
        self.assertFalse(left["refused"], left["result"])
        self.assertRegex(left["result"], r"^Captions on: \d+ lines of what was said")
        self.assertIn("did not line up one for one with the sentences said, twice, so the captions are in Thai as said",
                      left["result"])
        self.assertRegex(again["result"], r"^Captions on: 2 lines in English, translated from the 2 sentences said in "
                                          r"Thai")
        self.assertEqual([line["text"] for line in self.chat.project.state()["captions"]["lines"]],
                         ["設定は簡単です。", "とても安いです。"], "a gap of 1 s ends a sentence")
        self.assertFalse(japanese["refused"], japanese["result"])
        self.assertRegex(japanese["result"], r"^Captions on: \d+ lines in Japanese, translated from the 2 sentences said")

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
            [call("load_recipe", type="talk", subtitles="en-US 2", why="not the name of a language")],
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
            [call("add_title", words="ซื้oมาขายไป", start=0, end=3, place="top", why="the hook")],
            [call("add_title", words="คลิปเด็ดลง Reels", start=0, end=3, place="top", why="the hook")], "Done.")
        self.assertTrue(steps[-2]["refused"], "a Latin o inside a Thai word (Glimmer 10-30)")
        self.assertIn('"ซื้oมาขายไป" has a Latin letter inside a Thai word', steps[-2]["result"])
        steps = steps[:-2] + steps[-1:]
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

    def test_a_recipe_asked_for_before_the_recording_is_asked_for_again_before_a_moment(self):
        """10-31 045116-b599, 045946-c52f: refused before the recording was open, never called again, and a clip made with
        no recipe."""
        early = [call("load_recipe", type="talk", why="the ask says so")]
        again = [call("load_recipe", type="talk", why="what the pictures show")]
        _, steps = self.run_job(early, *self.to_moment(), again, self.to_moment()[2], "Done.")
        self.assertTrue(steps[3]["refused"], steps[3]["result"])
        self.assertTrue(steps[3]["result"].startswith("Not proposed yet: no recipe is loaded."), steps[3]["result"])
        self.assertFalse(steps[4]["refused"], steps[4]["result"])
        self.assertTrue(steps[5]["result"].startswith("Proposed 1 moment"), steps[5]["result"])

    def test_a_recipe_loaded_after_the_refusal_is_enough(self):
        early = [call("load_recipe", type="talk", why="the ask says so")]
        opened, look, moment = self.to_moment()
        _, steps = self.run_job(early, opened, [call("load_recipe", type="talk", why="what the pictures show")], look, moment,
                                "Done.")
        self.assertTrue(steps[4]["result"].startswith("Proposed 1 moment"), steps[4]["result"])

    def test_a_moment_is_refused_for_no_recipe_once(self):
        early = [call("load_recipe", type="talk", why="the ask says so")]
        _, steps = self.run_job(early, *self.to_moment(), self.to_moment()[2], "Done.")
        self.assertTrue(steps[3]["refused"], steps[3]["result"])
        self.assertTrue(steps[4]["result"].startswith("Proposed 1 moment"), steps[4]["result"])

    def test_the_persons_whole_ask_for_the_clip_is_no_word_on_one_thing(self):
        """031834-b64b on 11-01: the opening ask as person_said put a title on a forehead for 32 s."""
        ask = "ทำคลิปรีวิวจาก C:\\Users\\trip.mov ให้พร้อมโพสต์ ให้ Muse Wonder ดูทั้งภาพและเสียงแล้วใส่เอฟเฟกต์ตรงจังหวะที่มันเจอ และทำปกด้วย"
        long_ask = "ทำคลิปรีวิวเครื่องดูดความชื้นให้พร้อมโพสต์ ใส่เอฟเฟกต์ตรงจังหวะที่มันเจอ และเอาชื่อไว้บนหน้าผม ทำปกด้วยนะ"
        from unittest import mock
        job = mock.Mock(asked=False)
        job.chat.events = [{"kind": "person", "text": ask}, {"kind": "person", "text": long_ask},
                           {"kind": "person", "text": "เอาไว้บนหน้าผม"}]
        for quote in (ask, ask[:len(ask) // 2 + 5], "C:\\Users\\trip.mov", long_ask, long_ask[:len(long_ask) // 2 + 5]):
            with self.assertRaises(tools.Refused) as said:
                tools._asked(job, quote)
            self.assertEqual(str(said.exception), tools.WHOLE_ASK, quote)
        self.assertTrue(tools._asked(job, "เอาชื่อไว้บนหน้าผม"), "a few words of a long ask")
        self.assertTrue(tools._asked(job, "เอาไว้บนหน้าผม"), "a short message is all about this")

    def test_a_talk_keeps_its_people_unless_the_person_asked(self):
        """10-31 050337-bff5: Glimmer blurred all four people on the sofa; 11-01 021005-389c: sent again unasked, it stood."""
        self.chat.emit("person", job=None, text="please blur them all")
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="talk", why="a talk")],
                                *self.to_moment()[1:], [call("make_clip", moment=1, layout="subject", why="the talk")],
                                [call("decide_people", others="blur", why="people on the sofa")],
                                [call("decide_people", others="blur", why="again")],
                                [call("decide_people", others="blur", person_said="nobody said this", why="made up")],
                                [call("decide_people", others="blur", person_said="blur them all", why="the person asked")],
                                "Done.")
        refused, again, made_up, blurred = steps[-4:]
        self.assertIn("did not say", made_up["result"])
        self.assertTrue(refused["refused"], refused["result"])
        self.assertTrue(refused["result"].startswith("Not blurred. A talk is its people"), refused["result"])
        self.assertTrue(again["refused"], again["result"])
        self.assertFalse(blurred["refused"], blurred["result"])
        self.assertIn("Person 1 is blurred", blurred["result"])

    def test_muse_moves_an_effect_on_the_picture_only_when_the_person_asked(self):
        script = self.to_moment() + [[call("make_clip", moment=1, layout="subject", why="the cooking")],
                                     [call("add_sticker", sticker="fire", at=2, place="top", why="the peak")]]
        moved = dict(what="sticker", called="fire", at=2, box=[100, 500, 300, 700], why="a better spot")
        self.chat.emit("person", job=None, text="move the fire down")
        _, steps = self.run_job(*script, [call("change_effect", **moved)],
                                [call("change_effect", person_said="nobody said this", **moved)],
                                [call("change_effect", person_said="move the fire down", **moved)], "Done.")
        unasked, made_up, asked = steps[-3:]
        self.assertTrue(unasked["refused"], unasked["result"])
        self.assertIn("person_said", unasked["result"])
        self.assertFalse(asked["refused"], asked["result"])
        self.assertTrue(made_up["refused"], made_up["result"])
        self.assertIn("did not say", made_up["result"])

    def test_everyone_in_a_clip_blurred_goes_back_once(self):
        """11-01 005953-ecc0: Glimmer blurred all 9 people of a cooking episode."""
        from unittest import mock
        self.chat.emit("person", job=None, text="blur everyone please")
        with mock.patch.object(tools, "BLUR_MANY", 1):
            _, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the cooking")],
                                    [call("decide_people", others="blur", why="privacy")],
                                    [call("decide_people", others="blur", why="again")],
                                    [call("decide_people", others="blur", person_said="blur everyone", why="they asked")], "Done.")
        first, again, asked = steps[-3:]
        self.assertTrue(first["result"].startswith("Not blurred. That is everyone in the clip"), first["result"])
        self.assertTrue(again["refused"], "sent again unasked it stays refused (11-01, 021005-389c)")
        self.assertFalse(asked["refused"], asked["result"])

    def test_a_crowd_behind_a_seller_may_be_blurred(self):
        from unittest import mock
        with mock.patch.object(tools, "BLUR_MANY", 1):
            script = self.to_moment()
            _, steps = self.run_job(script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                                    [call("make_clip", moment=1, layout="subject", product=[100, 300, 250, 600], why="it")],
                                    [call("decide_people", others="blur", why="the crowd behind")], "Done.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])

    def test_muse_reads_when_to_pick_each_recipe(self):
        said = tools.TOOLS["load_recipe"].summary
        for recipe in tools.RECIPES.values():
            self.assertIn(recipe["description"], said)

    def test_a_clip_with_a_recipe_and_no_title_is_told_after_its_render(self):
        """021005-389c on 11-01: Glimmer rendered EP 0's talk with no title on it."""
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            script = self.to_moment()
            _, steps = self.run_job(script[0], [call("load_recipe", type="game", why="a game")], *script[1:],
                                    [call("make_clip", moment=1, layout="subject", why="the action")],
                                    [call("no_chat_box", why="none shows")], [call("decide_people", keep=[1], why="him")],
                                    [call("render_clip", why="make the file")],
                                    [call("add_title", words="Boss down", start=0, end=4, place="top", why="the hook")],
                                    [call("render_clip", why="again")], "Done.")
        finally:
            render.export = had
        bare, titled = steps[-3], steps[-1]
        self.assertIn(tools.NO_TITLE, bare["result"])
        self.assertNotIn(tools.NO_TITLE, titled["result"])

    def test_render_says_which_goals_have_no_burst(self):
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            script = self.to_moment()
            _, steps = self.run_job(script[0], [call("load_recipe", type="sport", why="a match")], *script[1:],
                                    [call("make_clip", moment=1, layout="card", why="the goal")],
                                    [call("no_chat_box", why="none shows")], [call("decide_people", keep=[1], why="him")],
                                    [call("add_title", words="Goal", start=0, end=10, place="top", why="the hook")],
                                    [call("render_clip", why="make the file")],
                                    [call("add_flurry", stickers=["coin"], at=5, box=[120, 235, 880, 400], why="the goal")],
                                    [call("render_clip", why="again")], "Done.")
        finally:
            render.export = had
        self.assertIn(tools.NO_BURST % '"Goal"', steps[-3]["result"])
        self.assertNotIn("No burst yet", steps[-1]["result"])

    def test_a_match_post_is_held_to_its_goals_scores(self):
        """033344-78ec on 11-01: every score right, and the post said 4 ประตูรวด on a game of 1-3."""
        titles = [{"text": t, "start": s, "end": s + 5} for s, t in ((20, "ยิงซ้ำ 1-2"), (0, "โหม่ง 0-1"), (10, "หนี 0-2"), (30, "ปิดเกม 1-3"))]
        state = {"titles": titles}
        self.assertEqual(tools._score_slips(state, "Hull 4 ประตูรวด"), (['"4 ประตูรวด"'], ["0-1", "0-2", "1-2", "1-3"]))
        self.assertEqual(tools._score_slips(state, "4 goals in a row")[0], ['"4 goals in a row"'])
        self.assertEqual(tools._score_slips(state, "ยิง 2 ประตูรวด จบ ARS 1-3 HUL, Hull ชนะ 3-1 นาที 3:10-3:33")[0], [])
        self.assertEqual(tools._score_slips(state, "ชนะ 4-1")[0], ['"4-1"'])
        self.assertEqual(tools._score_slips({"titles": titles[:1]}, "4 ประตูรวด"), ([], []), "one goal has no line to hold it to")
        self.assertEqual(tools._score_slips({"titles": [{"text": "Hull go ahead", "start": 0, "end": 5}]}, "ชนะ 4-1"), ([], []))
        run = [{"text": "%d-0" % n, "start": n, "end": n + 1} for n in (1, 2, 3)]
        self.assertEqual(tools._score_slips({"titles": run}, "3 ประตูรวด")[0], [], "a run from the first goal")
        self.assertEqual(tools._score_slips({"titles": run}, "4 ประตูรวด")[0], ['"4 ประตูรวด"'])

    def handed_twice(self, kind):
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            script = self.to_moment()
            _, steps = self.run_job(script[0], [call("load_recipe", type=kind, why="the kind")], *script[1:],
                                    [call("make_clip", moment=1, layout="card", why="the goals")],
                                    [call("no_chat_box", why="none shows")], [call("decide_people", keep=[1], why="him")],
                                    [call("add_title", words="Goal 0-1", start=0, end=4, place="top", why="the first")],
                                    [call("add_title", words="Goal 1-1", start=5, end=9, place="top", why="the second")],
                                    [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                                    [call("hand_over", title="5 goals in a row", caption="c", hashtags=["#a"], why="post")],
                                    [call("hand_over", title="5 goals in a row", caption="c", hashtags=["#a"], why="true")],
                                    "Done.")
        finally:
            render.export = had
        return steps[-2:]

    def test_a_match_post_that_breaks_the_score_is_told_once(self):
        first, second = self.handed_twice("sport")
        self.assertTrue(first["refused"], first["result"])
        self.assertEqual(first["result"], tools.SCORE_SLIP % ('"5 goals in a row"', "0-1, 1-1"))
        self.assertFalse(second["refused"], second["result"])

    def test_only_a_match_post_is_held_to_scores(self):
        first, _ = self.handed_twice("game")
        self.assertFalse(first["refused"], first["result"])

    def test_a_clip_with_no_title_is_sent_back_once(self):
        """021005-389c and 050402-c12f on 11-01: Glimmer read past the sentence in render_clip's answer and handed over."""
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        try:
            script = self.to_moment()
            _, steps = self.run_job(script[0], [call("load_recipe", type="game", why="a game")], *script[1:],
                                    [call("make_clip", moment=1, layout="subject", why="the action")],
                                    [call("no_chat_box", why="none shows")], [call("decide_people", keep=[1], why="him")],
                                    [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                                    [call("hand_over", title="Boss down", caption="c", hashtags=["#a"], why="post")],
                                    [call("hand_over", title="Boss down", caption="c", hashtags=["#a"], why="none asked")],
                                    "Done.")
        finally:
            render.export = had
        first, second = steps[-2:]
        self.assertTrue(first["refused"], first["result"])
        self.assertIn(tools.NO_TITLE, first["result"])
        self.assertFalse(second["refused"], second["result"])

    def test_no_title_is_not_asked_of_a_clip_with_no_recipe_a_batch_or_a_job_out_of_steps(self):
        from unittest import mock

        def handed(recipe="game", why="post", finishing=False, titles=()):
            job = types.SimpleNamespace(finishing=finishing, n=1, chat=types.SimpleNamespace(
                clip={"recipe": recipe} if recipe else None, finished=lambda: {"name": "a.mp4"}, checks={"a.mp4": []},
                events=[]))
            with mock.patch.object(tools, "_open", return_value={}), \
                    mock.patch.object(tools, "_clip_state", return_value={"titles": list(titles)}), \
                    mock.patch.object(tools, "_told_once", return_value=False), \
                    mock.patch.object(tools, "_other_apps", side_effect=tools.Refused("past it")), \
                    self.assertRaises(tools.Refused) as said:
                tools.hand_over(job, why, "t", "c", ["#a"])
            return str(said.exception)

        self.assertIn(tools.NO_TITLE, handed())
        for kept in ({"recipe": None}, {"why": tools.BATCH_WHY}, {"finishing": True}, {"titles": [{"text": "x"}]}):
            self.assertEqual(handed(**kept), "past it", kept)

    def test_slips_in_the_post_text_are_sent_back_once(self):
        """054318-527c on 11-01: the title was mended and the caption went out with หน้ฝน and ใช้งานง่ย."""
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        title, caption = "ตายแล้วฟื้นภายในวินาที", "ใช้งานง่ย"
        had = getattr(render, "export", None)
        render.export = export
        try:
            _, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                                    [call("decide_people", keep=[1], why="him")], [call("render_clip", why="make the file")],
                                    [call("check_clip", why="check it")],
                                    [call("hand_over", title=title, caption=caption, hashtags=["#a"], why="post")],
                                    '{"texts": ["%s", "%s", "#a"]}' % (title, caption),
                                    [call("hand_over", title=title, caption=caption, hashtags=["#a"], why="a word people use")],
                                    '{"texts": ["%s", "%s", "#a"]}' % (title, caption), "Done.")
        finally:
            render.export = had
        first, second = steps[-2:]
        self.assertTrue(first["refused"], first["result"])
        self.assertEqual(first["result"], tools.ODD_POST % '"ใช้งานง่ย"')
        self.assertFalse(second["refused"], second["result"])

    def test_a_title_handed_over_is_checked_too(self):
        """A title that never went on the clip reaches the post and the cover here, unread by add_title."""
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        said = "ตายแล้วรีสไตน์ภายในวินาที"
        had = getattr(render, "export", None)
        render.export = export
        try:
            _, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                                    [call("decide_people", keep=[1], why="him")], [call("render_clip", why="make the file")],
                                    [call("check_clip", why="check it")],
                                    [call("hand_over", title=said, caption="c", hashtags=["#a"], why="post")],
                                    '{"texts": ["%s", "c", "#a"]}' % said,
                                    [call("hand_over", title="ตายแล้วฟื้นภายในวินาที", caption="c", hashtags=["#a"], why="post")],
                                    "Done.")
        finally:
            render.export = had
        first, second = steps[-2:]
        self.assertTrue(first["refused"], first["result"])
        self.assertIn("Not handed over: no dictionary has", first["result"])
        self.assertFalse(second["refused"], second["result"])

    def test_a_clip_made_again_on_the_page_keeps_what_was_decided_about_each_person(self):
        """Lane 5 10-31: a tap on a layout wiped the decisions, so the render was refused until each face was tapped."""
        self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                     [call("decide_people", blur="1", why="someone passing behind")], "Done.")
        self.assertIsNone(self.chat.set_layout("card"))
        until = time.time() + 30
        while self.chat.making and time.time() < until:
            time.sleep(0.05)
        state = self.chat.project.state()
        self.assertEqual(state["clip"]["layout"]["kind"], "card")
        self.assertEqual([p["blur"] for p in state["people"]], [True])
        self.assertEqual(tools._undecided(state), [], "nothing stands in the render's way")
        self.assertEqual(len([b for b in state["blurs"] if b.get("person") == 1]), 1, "and the blur is still drawn")
        self.assertTrue(self.chat.set_person(1, False))
        self.assertEqual([p["blur"] for p in self.chat.project.state()["people"]], [False], "a later tap counts")
        self.chat.undo()
        self.chat.undo()
        self.assertEqual(self.chat.project.state()["clip"]["layout"]["kind"], "subject", "Undo goes back through it")

    def test_a_person_is_carried_by_their_face_and_a_new_clip_decides_again(self):
        old = {"people": [{"n": 1, "box": [1, 2, 3, 4]}, {"n": 2, "box": [5, 6, 7, 8]}]}
        new = {"people": [{"n": 1, "box": [5, 6, 7, 8]}, {"n": 2, "box": [1, 2, 3, 4]}, {"n": 3, "box": [9, 9, 9, 9]}]}
        self.assertEqual(project._carried({1: True}, old, new), {2: True}, "by the face, not its number")
        self.assertEqual(project._carried({}, None, new), {})
        p = project.Project()
        new["people"] = [dict(x, points=[[0.0, 1, 1, 1, 1]], **{"from": 0.0, "to": 1.0}) for x in new["people"]]
        clip = dict(new, do="clip", pieces=[[0.0, 5.0]], layout={"kind": "subject"})
        p.entries = [dict(old, do="clip", pieces=[[0.0, 5.0]], layout={"kind": "subject"}), {"do": "people", "blur": [1], "keep": []}, clip]
        self.assertEqual([x["blur"] for x in p.state()["people"]], [None, None, None], "a clip not made again starts over")
        p.entries[-1] = dict(clip, moved=True)
        self.assertEqual([x["blur"] for x in p.state()["people"]], [None, True, None], "made again, the face keeps it")

    PLAY = {"size": [100, 180], "seen": 0.73, "shots": [{"from": 0.0, "x": 10, "y": 0, "line": True},
                                                        {"from": 5.0, "x": 150, "y": 0, "line": True}]}

    def a_match(self, recipe="sport", **made):
        return self.run_job(self.to_moment()[0], [call("load_recipe", type=recipe, why="a match")], self.to_moment()[1],
                            self.to_moment()[2], [call("make_clip", moment=1, layout="subject", why="close up", **made)], "Done.")[1]

    def whole(self):
        return next(p for p in self.chat.project.state()["clip"]["layout"]["parts"] if p["name"] == "whole")

    def test_a_match_in_the_upright_frame_follows_the_play(self):
        """Tarm 10-31: "คลิปบอล ผมว่า fix กลางจอ ไม่ดี ควรตามลูกบอล"."""
        asked = []
        with mock.patch.object(tools.pitch, "follow", lambda *a: asked.append(a) or self.PLAY):
            steps = self.a_match()
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertEqual(self.whole()["follow"], self.PLAY["shots"])
        self.assertEqual(asked[0][2:], tuple(self.whole()["src"][2:3]) + (0,), "the frame's width and its top")
        self.assertIn("The frame follows the play: the game's mark over the player in control is in 73%", steps[-1]["result"])
        self.assertTrue(self.chat.framing(self.chat.project.state())["follows"], "and the Frame tab says so")

    def test_a_match_with_no_mark_holds_the_middle_and_says_so(self):
        with mock.patch.object(tools.pitch, "follow", lambda *a: None):
            steps = self.a_match()
        self.assertNotIn("follow", self.whole())
        self.assertIn("The frame holds the middle of the picture", steps[-1]["result"])

    def test_only_a_match_follows_the_play_and_a_crop_or_no_follow_stay_as_asked(self):
        def never(*a):
            raise AssertionError("not asked")

        with mock.patch.object(tools.pitch, "follow", never):
            self.assertFalse(self.a_match("game")[-1]["refused"])
            self.assertFalse(self.a_match(follow=False)[-1]["refused"])
            self.assertFalse(self.a_match(crop=[100, 0, 400, 1000])[-1]["refused"])

    def test_new_ends_follow_the_play_again_with_nothing_to_say(self):
        old = {"layout": {"kind": "subject", "parts": [{"name": "whole", "src": [0, 0, 9, 16], "follow": self.PLAY["shots"]}]}}
        new = {"layout": {"kind": "subject", "parts": [{"name": "whole", "src": [0, 0, 9, 16], "follow": self.PLAY["shots"]}]}}
        self.assertIsNone(tools._frame_kept(None, None, old, new))
        new["layout"]["parts"][0].pop("follow")
        self.assertIn("no longer follows", tools._frame_kept(None, None, old, new))

    def test_a_match_keeps_its_faces_as_any_event_does(self):
        """Lane 5 10-31: 318 faces in a match, and Muse's steps went on them."""
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="sport", why="a match")], self.to_moment()[1],
                                self.to_moment()[2], [call("make_clip", moment=1, layout="card", why="the goal")], "Done.")
        self.assertIn("all kept as they are: at a trip, an event or a match", steps[-1]["result"])
        self.assertEqual(tools._undecided(self.chat.project.state()), [])

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
        self.assertIn(tools.BLUR_SLIPS, steps[-1]["result"])
        self.assertIn("Person 1 is blurred where their face was found, 0:05-0:10 of the clip, and the blur follows those finds.",
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
        self.assertIn("Person 1 is blurred where their face was found", crowd[-1]["result"],
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

    def test_muse_image_draws_a_picture_of_what_is_said_over_the_top_above_the_people(self):
        """Tarm 10-30, on a talk that shows a new picture on top as each thing is named: "ตอนที่พูดถึงเรื่องผีก็มีภาพผี"."""
        from unittest import mock
        drawn = os.path.join(self.tmp, "drawn.jpg")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=1920x1280", "-frames:v", "1",
                        drawn], check=True)
        asked = []

        def make(about):
            asked.append(about)
            if about == "a refused thing":
                raise tools.picture.PictureError("Muse Image refused to draw it (HTTP 400: no).", 0.0)
            again = len(asked) > 2
            return {"path": drawn, "usd": 0.0 if again else 0.01, "seconds": 0.0 if again else 9.5, "reused": again}

        low, high = (([(t / 2.0, [[300, y, 300, 300]]) for t in range(41)], 0.5) for y in (1100, 200))
        with mock.patch.object(tools.picture, "make", make), mock.patch.object(tools, "_face_times",
                                                                             return_value=low):
            brain, steps = self.run_job(
                *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                [call("add_picture", about="a pale ghost in a dark doorway", at=2, why="the ghost story")],
                [call("add_picture", about="a crowd looking at their phones", at=3, length=4, why="the viewers")],
                [call("add_picture", about="a crowd looking at their phones", at=12, why="again")],
                [call("add_picture", about="a ghost", at=19.8, why="at the very end")],
                [call("add_picture", about="a refused thing", at=8, why="refused")],
                "Done.")
            with mock.patch.object(tools, "_face_times", return_value=high):
                _, more = self.run_job([call("add_picture", about="a ghost", at=8, why="faces high")], "Done.")
        ghost, crowd, again, late, refused = steps[-5:]
        self.assertFalse(ghost["refused"], ghost["result"])
        self.assertIn("Muse Image drew the picture of a pale ghost in a dark doorway in 9.5 s, for $0.01.",
                      ghost["result"])
        self.assertIn("It covers the top of the frame from 2 to 5 s, down to just above the people (y 0 to 500 on "
                      "your grid), its lower edge fading into the footage, under the titles and captions.",
                      ghost["result"])
        self.assertIn("comes next as a picture", ghost["result"])
        self.assertIn("It replaces the picture of a pale ghost in a dark doorway at 2 s.", crowd["result"])
        self.assertIn("Kept the picture of a crowd looking at their phones (made before, at no cost).",
                      again["result"])
        self.assertTrue(late["refused"])
        self.assertIn("a picture needs 0.6 s to come in and go out. Put it earlier.", late["result"])
        self.assertFalse(refused["refused"])
        self.assertIn("No picture: Muse Image refused to draw it (HTTP 400: no). Tell the person, or go on without "
                      "it.", refused["result"])
        self.assertTrue(more[-1]["refused"])
        self.assertIn("From 8 to 11 s a face or the product comes up to y 104 on your grid, so a picture above them "
                      "would be too short to see.", more[-1]["result"])
        state = self.chat.project.state()
        self.assertEqual([(k["about"], k["at"], k["length"], k["box"]) for k in state["pictures"]],
                         [("a crowd looking at their phones", 3, 4, [0, 0, 1080, 960]),
                          ("a crowd looking at their phones", 12, 3.0, [0, 0, 1080, 960])])
        self.assertTrue(state["pictures"][0]["png"].endswith(".png"))
        self.assertIn("pictures drawn: a crowd looking at their phones at 3 s, a crowd looking at their phones at 12 s",
                      self.chat.clip_line())
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), "the picture of a crowd looking at their "
                                                                        "phones at 12 s")
        with open(os.path.join(agent.WORK, "spent.jsonl"), encoding="utf-8") as f:
            paid = [json.loads(line) for line in f if '"image"' in line]
        self.assertEqual([(r["label"], r["usd"]) for r in paid], [("Muse Image", 0.01)] * 2,
                         "the two pictures drawn are in the ledger; the one kept cost nothing")

    def test_a_picture_on_a_card_stops_at_the_card_so_the_screen_it_explains_stays_whole(self):
        """Tarm 10-30 liked an ad's picture over the top, above what it explains; on a card with no face the picture
        ran down to the middle of the frame, over the top of the slide."""
        from unittest import mock
        drawn = os.path.join(self.tmp, "drawn.jpg")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=1920x1280", "-frames:v", "1",
                        drawn], check=True)

        def make(about):
            return {"path": drawn, "usd": 0.01, "seconds": 1.0, "reused": False}

        with mock.patch.object(tools.picture, "make", make), mock.patch.object(tools, "_face_times",
                                                                             return_value=([], 0.5)), \
                mock.patch.object(tools, "_the_face", return_value=(None, "No face was found.")), \
                mock.patch.object(screen, "readings", lambda rec, a, b, on_start=None: [{"at": a, "lines": []}]):
            brain, steps = self.run_job(*self.to_moment(), [call("load_recipe", type="explain", why="a tutorial")],
                                        [call("make_clip", moment=1, layout="card", why="the slide")],
                                        [call("add_picture", about="gold bars on a table", at=2, why="the gold")],
                                        "Done.")
        made, placed = steps[-2:]
        self.assertIn("With no face circle, the frame above the card (y 0 to", made["result"],
                      "10-30, 154248-dd25: Spark had the recipe's Pictures line and left that third empty")
        self.assertIn("add_picture at the second it is said", made["result"])
        self.assertFalse(placed["refused"], placed["result"])
        state = self.chat.project.state()
        card = next(p for p in state["clip"]["layout"]["parts"] if p["name"] == "card")
        self.assertLess(card["dst"][1], tools.PICTURE_MOST, "a card higher than where a picture stops by itself")
        self.assertEqual(state["pictures"][-1]["box"], [0, 0, 1080, layouts._even(card["dst"][1])])
        self.assertIn("down to just above the card (y 0 to", placed["result"])

    def green_cloth(self):
        """Patches for a recording in front of a green cloth: green.find says it is there, green.mask makes a person's
        mask over the seconds asked (counted in self.cut), green.backdrop draws a blue picture once per words."""
        from unittest import mock
        from test_cutout import make_mask
        png = os.path.join(self.tmp, "ground.png")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x3050A0:s=1080x1920", "-frames:v",
                        "1", png], check=True)
        self.cut, drawn = [], set()
        cloth = {"clear": 60.0, "solid": 24.0, "edge": 0.8}

        def mask(rec, a, b):
            self.cut.append((a, b))
            return {"path": make_mask(self.tmp, "green_%d.mkv" % len(self.cut), b - a, a, size=(320, 180)),
                    "person": None, "fps": 30, "size": [320, 180], "from": a, "to": b, "green": cloth}

        def backdrop(about):
            again = about in drawn
            drawn.add(about)
            return {"path": png, "usd": 0.0 if again else 0.01, "seconds": 0.0 if again else 9.0, "reused": again}

        return [mock.patch.object(tools.green, "find", return_value=dict(cloth, scores=[])),
                mock.patch.object(tools.green, "mask", mask), mock.patch.object(tools.green, "backdrop", backdrop)]

    def test_a_person_on_a_green_cloth_gets_a_background_muse_image_draws_and_the_layout_stays(self):
        """Tarm 10-30, on a seller recorded in front of a green cloth: "ตัดผ้าเขียวออก แล้วใส่พื้นหลังให้เหมาะสมด้วยโมเดล
        ของเมต้า"; and the product stays the focus: "โปรแกรมต้องฉลาดพอจะรู้ว่าต้องโฟกัสสินค้ามากกว่าคน"."""
        patches = self.green_cloth()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the product")],
            [call("put_background", about="A tidy gaming desk with soft blue light.", why="the example")],
            [call("put_background", about="a tidy gaming desk", why="the cloth")],
            [call("put_background", about="a bright kitchen counter", why="another place")],
            [call("put_background", about="a tidy gaming desk", why="back to the first")],
            "Done.")
        made, copied, desk, kitchen, again = steps[-5:]
        self.assertTrue(copied["refused"], "Glimmer 10-30: the description's example behind an iPad, twice")
        self.assertIn("is the example in this tool's description, written for no clip in particular", copied["result"])
        self.assertFalse(kitchen["refused"], "once a job: the person may have asked for those very words")
        self.assertIn(tools.GREEN_SAYS, made["result"], "make_clip tells Muse the cloth is there")
        self.assertNotIn(tools.FOLLOW_SAYS, made["result"], "not a seller clip")
        self.assertFalse(desk["refused"], desk["result"])
        self.assertIn("The person was cut off the green cloth by its colour, on this computer with no GPU, with what "
                      "they hold; the cloth covers 80% of the frame's edge", desk["result"])
        self.assertIn("Muse Image drew the background, a tidy gaming desk, in 9 s, for $0.01.", desk["result"])
        self.assertIn("It is behind the person wherever the clip does not show them, through the whole clip",
                      desk["result"])
        self.assertIn("comes next as a picture", desk["result"])
        self.assertIn("The person was already cut off the green cloth for this clip, so it was not done again.",
                      kitchen["result"])
        self.assertIn("Kept the background of a tidy gaming desk (made before, at no cost).", again["result"])
        self.assertEqual(len(self.cut), 1, "the cloth is cut once for the clip")
        state = self.chat.project.state()
        self.assertEqual(state["backdrop"]["about"], "a tidy gaming desk")
        self.assertTrue(state["mask"]["green"])
        self.assertEqual(state["clip"]["layout"]["kind"], "subject", "the layout Muse chose stays")
        self.assertIn("background drawn: a tidy gaming desk", self.chat.clip_line())
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), "the background of a tidy gaming desk")
        with open(os.path.join(agent.WORK, "spent.jsonl"), encoding="utf-8") as f:
            paid = [json.loads(line) for line in f if '"image"' in line]
        self.assertEqual([(r["label"], r["usd"]) for r in paid], [("Muse Image", 0.01)] * 2,
                         "two backgrounds drawn are in the ledger; the one kept cost nothing")

    def test_after_sam_found_the_person_the_cloth_is_still_cut_so_what_they_hold_stays_in_front(self):
        """Live 10-30: Muse ran cut_out (SAM, the person alone) for a title behind them, then put the background on;
        SAM's person leaves out the controller in their hands, the cloth's cut keeps it."""
        from unittest import mock
        from test_cutout import make_mask
        patches = self.green_cloth()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        def sam(rec, a, b, **kw):
            return {"path": make_mask(self.tmp, "sam.mkv", b - a, a, size=(320, 180)), "person": None, "fps": 10,
                    "size": [320, 180], "from": a, "to": b}

        with mock.patch.object(tools.person, "cut_out", sam):
            with mock.patch.object(tools.green, "mask", return_value=None):
                _, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="go")],
                                        [call("cut_out", why="a title behind them")],
                                        [call("put_background", about="a desk", why="no cloth")], "Done.")
            self.assertIn("The person SAM 3.1 found for this clip is kept in front", steps[-1]["result"],
                          "with no cloth, SAM's person is used")
            self.assertFalse(self.chat.project.state()["mask"].get("green"))
            _, steps = self.run_job([call("put_background", about="a desk", why="the cloth")], "Done.")
        self.assertIn("cut off the green cloth by its colour", steps[-1]["result"])
        self.assertTrue(self.chat.project.state()["mask"]["green"], "the cloth's cut replaces SAM's person")

    def test_without_a_green_cloth_the_background_waits_for_sam_and_a_bad_cloth_or_drawing_is_said(self):
        from unittest import mock
        patches = self.green_cloth()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        def refused_drawing(about):
            raise tools.picture.PictureError("Muse Image refused to draw it (HTTP 400: no).", 0.0)

        with mock.patch.object(tools.green, "mask", return_value=None):
            _, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="go")],
                                    [call("put_background", about="a desk", why="no cloth")], "Done.")
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("No green cloth is behind the person, so the person must be found with SAM 3.1 first: call "
                      "cut_out, then put_background again.", steps[-1]["result"])
        with mock.patch.object(tools.green, "mask", side_effect=tools.green.GreenError("The cloth is too grey.")):
            _, steps = self.run_job([call("put_background", about="a desk", why="a grey cloth")], "Done.")
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("The cloth is too grey. Or find the person with cut_out (SAM 3.1), then put the background on "
                      "again.", steps[-1]["result"])
        with mock.patch.object(tools.green, "backdrop", refused_drawing):
            _, steps = self.run_job([call("put_background", about="a desk", why="refused")], "Done.")
        self.assertFalse(steps[-1]["refused"])
        self.assertIn("No background: Muse Image refused to draw it (HTTP 400: no). Tell the person, or go on without "
                      "it.", steps[-1]["result"])
        self.assertIsNone(self.chat.project.state()["backdrop"])

    def test_sam_follows_the_product_and_the_vertical_frame_moves_with_it(self):
        """Tarm 10-30: "โปรแกรมต้องฉลาดพอจะรู้ว่าต้องโฟกัสสินค้ามากกว่าคน ... ไม่ได้ขายหน้าตา"."""
        from unittest import mock
        from test_cutout import make_mask
        asked = []

        def cut_out(rec, a, b, on_start=None, on_progress=None, what=None):
            asked.append(what)
            if what == "a broken thing":
                raise tools.person.PersonError("SAM is not set up.")
            where = "-1000" if what in ("a missing thing", "thing") else "280"
            return {"path": make_mask(self.tmp, "thing_%d.mkv" % len(asked), b - a, a, "lt(abs(X-%s),20)" % where,
                                      size=(320, 180)), "person": None, "fps": 10, "size": [320, 180], "from": a,
                    "to": b}

        script = self.to_moment()
        with mock.patch.object(tools.person, "cut_out", cut_out):
            _, steps = self.run_job(
                script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                [call("make_clip", moment=1, layout="subject", why="the product")],
                [call("follow_product", what="white game controller", why="the product leads")],
                [call("follow_product", what="a missing thing", why="not there")],
                [call("follow_product", what="a broken thing", why="SAM fails")],
                "Done.")
            made, followed, missing, broken = steps[-4:]
            self.assertIn(tools.FOLLOW_SAYS, made["result"], "a seller in a landscape recording is told to follow it")
            self.assertFalse(followed["refused"], followed["result"])
            self.assertIn("SAM 3.1 followed the white game controller through the clip: it is in 100% of the pictures.",
                          followed["result"])
            self.assertIn("The frame holds one place", followed["result"])
            self.assertIn("comes next as a picture", followed["result"])
            self.assertIn("SAM 3.1 did not see the thing anywhere in the clip, so the frame stays where it is.",
                          missing["result"])
            self.assertIn("The a broken thing could not be followed: SAM is not set up.", broken["result"])
            whole = self.chat.project.state()["clip"]["layout"]["parts"][0]
            self.assertEqual(whole["src"][0], 218, "the window at the product, on the right")
            self.assertTrue(whole["follow"])
            self.assertEqual(tools.describe([e for e in self.chat.project.entries if e["do"] == "layout"][-1]),
                             "following the product")
            _, steps = self.run_job([call("make_clip", moment=1, layout="card", why="a card")],
                                    [call("make_clip", moment=1, layout="card", why="the person said: a card")],
                                    [call("follow_product", what="white game controller", why="on a card")], "Done.")
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("Only the subject layout's frame can follow the product", steps[-1]["result"])
        self.assertEqual(asked, ["white game controller", "a missing thing", "thing", "a broken thing"])

    def test_a_line_never_breaks_between_two_short_thai_pieces(self):
        """PS5 pad 10-31: "จอย PS5 มีไจ / โร เล็งปืนได้"."""
        from clipstudio import text as writing
        pieces = [w[a:b].strip() for w in ["จอย PS5 มีไจโร เล็งปืนได้"] for a, b in writing._atoms(w)]
        self.assertNotIn("โร", pieces)
        self.assertTrue(any("ไจโร" in p for p in pieces), pieces)
        joined = "".join("เครื่องกรองเฮปาใหม่"[a:b] for a, b in writing._atoms("เครื่องกรองเฮปาใหม่"))
        self.assertEqual(joined, "เครื่องกรองเฮปาใหม่", "the pieces joined are the words exactly")

    def test_a_sellers_title_stays_in_front(self):
        """PS5 pad 10-31: behind him it stood behind his hands and the pad for 15 s."""
        script = self.to_moment()
        _, steps = self.run_job(
            script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
            [call("make_clip", moment=1, layout="subject", why="the product")],
            [call("add_title", words="Big win", start=0, end=5, place="top", behind=True, why="behind")], "Done.")
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("A seller's title stays in front", steps[-1]["result"])

    def test_a_title_keeps_out_from_under_an_effect_put_on_first(self):
        """Talk 035009-dfe9 on 11-01: the sticker went on first, and sat on the title's first letter for 2 s."""
        self.chat.emit("person", job=None, text="เอาชื่อไว้ใต้ดาวเลย")
        _, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_sticker", sticker="star", at=2, place="top", why="the peak")],
            [call("add_flurry", stickers=["coin"], at=10, place="top", why="the second")],
            [call("add_title", words="Big win", start=0, end=4, place="top", why="under the star")],
            [call("add_title", words="Big win", start=9, end=12, place="top", why="under the coins")],
            [call("add_title", words="Big win", start=0, end=1.9, place="top", why="before the star")],
            [call("add_title", words="Big win", start=1, end=3, place="top", person_said="เอาชื่อไว้ใต้ดาว", why="asked")],
            "Done.")
        star, coins, before, asked = steps[-4:]
        self.assertTrue(star["refused"])
        self.assertEqual(star["result"], tools.UNDER_EFFECT % "the sticker star at 2 s")
        self.assertTrue(coins["refused"])
        self.assertEqual(coins["result"], tools.UNDER_EFFECT % "the burst flurry at 10 s")
        self.assertFalse(before["refused"], before["result"])
        self.assertFalse(asked["refused"], asked["result"])
        self.assertIn("It stands under the sticker star at 2 s, where the person asked for it", asked["result"])

    def test_a_blur_for_a_stretch_says_it_stays_still_and_shows_its_last_frame(self):
        """Montage 035936-7931 on 11-01: boxes on riders were shown at their middle, and stood on the trees by their end."""
        _, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="go")],
                                [call("blur_box", box=[400, 0, 600, 300], start=2, end=5, why="a rider")],
                                [call("blur_box", box=[400, 0, 600, 300], why="a logo")], "Done.")
        stretch, whole = steps[-2:]
        self.assertIn(tools.STILL_BOX, stretch["result"])
        self.assertIn("Its frame at 4.9 s", stretch["result"])
        self.assertNotIn(tools.STILL_BOX, whole["result"])

    def test_a_flurry_keeps_off_the_title(self):
        """Vlog 10-31, 201717-f1d9: the laugh's flurry went in the top place and stood over the title held there."""
        _, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_title", words="Big win", start=0, end=15, place="top", why="the hook")],
            [call("add_flurry", stickers=["coin"], at=5, place="top", why="over it")],
            [call("add_flurry", stickers=["coin"], at=5, place="middle", why="clear of it")],
            [call("add_flurry", stickers=["coin"], at=16, place="top", why="after it")], "Done.")
        over, clear, after = steps[-3:]
        self.assertTrue(over["refused"])
        self.assertIn('The flurry would cover the title "Big win" (x ', over["result"])
        self.assertIn("Clear of it at 5 s: middle", over["result"])
        self.assertFalse(clear["refused"], clear["result"])
        self.assertFalse(after["refused"], after["result"])

    def test_a_refusal_with_no_place_clear_names_a_box_that_is(self):
        """Sport 032940-bc6c on 11-01: a card's one place held the title, and no goal got its burst in 7 tries."""
        _, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="card", why="the play")],
            [call("add_title", words="Big win", start=0, end=15, place="top", why="the hook")],
            [call("add_flurry", stickers=["coin"], at=5, place="top", why="over it")],
            [call("add_flurry", stickers=["coin"], at=5, box=[120, 235, 880, 400], why="the box it named")], "Done.")
        over, named = steps[-2:]
        self.assertTrue(over["refused"])
        self.assertIn("No place is clear of it at 5 s: give the box [120, 235, 880, 400], or a time the title is not shown.",
                      over["result"])
        self.assertFalse(named["refused"], named["result"])

    def test_the_box_a_refusal_names_keeps_off_titles_faces_and_what_a_phone_covers(self):
        from unittest import mock
        high = [450, 500, 180, 100]
        state = {"clip": {"layout": layouts.make("card", (1920, 1080))}, "captions": [], "length": 30}

        def named(titles=(), found=(), captions=()):
            with mock.patch.object(tools, "_face_times", return_value=([(5, list(found))], 1)), \
                    mock.patch.object(tools.render, "title_boxes", return_value=list(titles)):
                return tools._clear_box(dict(state, captions=list(captions)), None, "clean", 5, 2, lambda spot: spot)

        strip, under = tools.EFFECT_BOXES[0], (300, 420, 700, 600)
        self.assertEqual(named(), strip)
        self.assertEqual(named(titles=[({"start": 0, "end": 30, "text": "x"}, high)]), under)
        self.assertEqual(named(titles=[({"start": 8, "end": 30, "text": "x"}, high)]), strip, "shown later")
        self.assertEqual(named(found=[high]), under)
        self.assertEqual(named(captions=["a line"]), under)
        self.assertIsNone(named(found=[[0, 0, 1080, 1920]]))

    def test_a_refusal_for_a_phone_or_a_face_names_a_clear_box_too(self):
        """Sport 033344-78ec on 11-01: 7 tries on one goal, from the title to the buttons to a face, and it got no burst."""
        from unittest import mock
        script = self.to_moment()
        with mock.patch.object(tools, "_face_times", return_value=([(5, [[400, 900, 300, 300]])], 1)):
            _, steps = self.run_job(
                *script, [call("make_clip", moment=1, layout="card", why="the play")],
                [call("add_flurry", stickers=["coin"], at=5, box=[100, 300, 900, 700], why="the buttons")],
                [call("add_flurry", stickers=["coin"], at=5, box=[300, 420, 700, 600], why="a face")],
                [call("add_sticker", sticker="star", at=5, box=[850, 620, 940, 700], why="the buttons")],
                [call("add_sticker", sticker="star", at=5, box=[300, 420, 700, 600], why="a face")], "Done.")
        said = "Clear of all that at 5 s: the box [120, 235, 880, 400]."
        for step, why in zip(steps[-4:], ("On a phone", "would cover a face", "On a phone", "would cover a face")):
            self.assertTrue(step["refused"], step["result"])
            self.assertIn(why, step["result"])
            self.assertIn(said, step["result"])

    def test_a_match_is_told_which_goals_have_no_burst(self):
        """033344-78ec and 032940-bc6c on 11-01: 1 of 4 goals marked, and none."""
        job = types.SimpleNamespace(chat=types.SimpleNamespace(clip={"recipe": "sport"}))
        first, second = {"text": "0-1", "start": 0, "end": 20}, {"text": "0-2", "start": 20, "end": 40}
        state = {"titles": [first, second], "flurries": [{"at": 7, "length": 1.4}]}
        self.assertEqual(tools._no_burst(job, state), tools.NO_BURST % '"0-2"')
        self.assertEqual(tools._no_burst(job, dict(state, flurries=[])), tools.NO_BURST % '"0-1", "0-2"')
        self.assertEqual(tools._no_burst(job, dict(state, flurries=[{"at": 7, "length": 1.4}, {"at": 25, "length": 1.4}])), "")
        self.assertEqual(tools._no_burst(types.SimpleNamespace(chat=types.SimpleNamespace(clip={"recipe": "game"})), state), "")

    def test_captions_are_translated_when_the_speaker_speaks_another_language_than_the_person(self):
        """Tarm 10-31: a person asking in Thai over an English recording gets Thai captions without asking for them."""
        job = types.SimpleNamespace(chat=types.SimpleNamespace(events=[
            {"kind": "person", "text": "ทำคลิปจาก F:\\OBS\\talk.mkv ให้หน่อย"}]))
        self.assertEqual(tools._asked_in(job), "Thai")
        job.chat.events.append({"kind": "person", "text": "Make the cover again."})
        self.assertEqual(tools._asked_in(job), "English", "the person's last words")
        job.chat.events.append({"kind": "person", "text": "F:\\OBS\\talk.mkv"})
        self.assertIsNone(tools._asked_in(job), "a path says nothing of the language")
        job.chat.events[:] = []
        self.assertIsNone(tools._asked_in(job))
        job.chat.events.append({"kind": "person", "text": "Haz un vídeo corto de esta grabación"})
        self.assertIsNone(tools._asked_in(job), "Spanish is not English: its captions stay as said")
        job.chat.events.append({"kind": "person", "text": "短いクリップを作って"})
        self.assertEqual(tools._asked_in(job), "Japanese")
        job.chat.events.append({"kind": "person", "text": "做一个短视频"})
        self.assertEqual(tools._asked_in(job), "Chinese")
        job.chat.events.append({"kind": "person", "text": "اصنع مقطعا قصيرا"})
        self.assertEqual(tools._asked_in(job), "Arabic")
        self.assertEqual(tools._subtitles("spanish"), "Spanish", "any language Muse can write")
        self.assertEqual(tools._subtitles("ภาษาอังกฤษ"), "English")
        with self.assertRaisesRegex(tools.Refused, "is not the name of a language"):
            tools._subtitles("en-US 2")

    def test_a_thai_clip_for_a_person_who_asked_in_english_gets_english_captions_unasked(self):
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": "ตั้งค่าเครื่องง่ายมาก"},
                               {"from": 5.0, "to": 8.0, "text": "ราคา 990 บาท"}]
        self.chat.emit("person", job=None, text="Make a clip from this")
        _, steps = self.run_job(
            *self.to_moment()[:2],
            [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the setup"}])],
            '{"lines": ["ตั้งค่าเครื่องง่ายมาก", "ราคา <N1> บาท"]}',
            [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_captions", why="the words")], '{"lines": ["Setting it up is so easy.", "It costs 990 baht."]}',
            "Done.")
        self.assertRegex(steps[-1]["result"], r"^Captions on: \d+ lines in English, translated from the 2 sentences "
                                              r"said in Thai")

    def test_a_piece_goes_to_metas_sam_only_with_a_key_and_the_switch_or_muse_on_metas_servers(self):
        from unittest import mock
        job = types.SimpleNamespace(chat=self.chat, n=1)
        with mock.patch.object(tools.asr, "secret", lambda: ""):
            self.assertEqual(tools._sam_goes(job), {}, "no key, nothing goes")
        with mock.patch.object(tools.asr, "secret", lambda: "a key"), mock.patch.object(tools, "_here", lambda j: True):
            tools.sends.allow(False)
            self.assertEqual(tools._sam_goes(job), {}, "Muse here and the switch off, nothing goes")
            tools.sends.allow(True)
            self.assertTrue(tools._sam_goes(job)["send"])
            tools.sends.allow(False)
        with mock.patch.object(tools.asr, "secret", lambda: "a key"), mock.patch.object(tools, "_here", lambda j: False):
            goes = tools._sam_goes(job)
        self.assertTrue(goes["send"], "Muse on Meta's servers already sees the pictures")
        goes["on_paid"](0.06)
        with open(os.path.join(tools.recording.WORK, "spent.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual([(r["label"], r["usd"]) for r in rows if r["mind"] == "sam"], [("Meta SAM 3.1", 0.06)])

    def test_both_callers_of_sam_say_what_may_go_to_meta(self):
        from unittest import mock
        from test_cutout import make_mask
        asked = []

        def cut_out(rec, a, b, **k):
            asked.append(k.get("send"))
            return {"path": make_mask(self.tmp, "sent.mkv", b - a, a, "lt(abs(X-280),20)", size=(320, 180)),
                    "person": None, "fps": 10, "size": [320, 180], "from": a, "to": b}

        script = self.to_moment()
        with mock.patch.object(tools.person, "cut_out", cut_out), \
                mock.patch.object(tools, "_sam_goes", lambda job: {"send": True}):
            self.run_job(script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                         [call("make_clip", moment=1, layout="subject", why="the product")],
                         [call("follow_product", what="white game controller", why="the product leads")],
                         [call("cut_out", why="the person")], "Done.")
        self.assertEqual(asked, [True, True])

    def test_with_no_sam_the_frame_holds_the_product_box(self):
        """Spark 10-31, Tarm's green-cloth seller with no SAM: 1 of 4 clips handed over, the frame on his face."""
        from unittest import mock
        script = self.to_moment()
        with mock.patch.object(tools.person, "PYTHON", ""):
            _, steps = self.run_job(
                script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                [call("make_clip", moment=1, layout="subject", why="no box")],
                [call("follow_product", what="white game controller", why="the product leads")],
                [call("make_clip", moment=1, layout="subject", product=[700, 200, 850, 600], why="the product")],
                [call("follow_product", what="white game controller", why="the product leads")], "Done.")
        boxless, held = steps[-3], steps[-1]
        self.assertIn("Make the clip again with product", boxless["result"])
        self.assertFalse(held["refused"], held["result"])
        self.assertIn("With no SAM on this computer the frame does not follow the white game controller: it holds one "
                      "place", held["result"])
        self.assertIn("comes next as a picture", held["result"])
        self.assertNotIn("LCS_SAM_PYTHON", held["result"])
        whole = self.chat.project.state()["clip"]["layout"]["parts"][0]
        self.assertLessEqual(whole["src"][0], 224, "the product's left side is in the window")
        self.assertGreaterEqual(whole["src"][0] + whole["src"][2], 272, "and its right side")

    def test_a_product_at_the_edge_of_the_picture_is_not_in_the_frame(self):
        """Glimmer's balm 10-31: his face on all 6 sampled frames, and the check passed."""
        lay = {"parts": [{"name": "whole", "src": [600, 0, 608, 1080], "dst": [0, 0, 1080, 1920]}]}
        self.assertEqual(layouts.shown_share(lay, [700, 400, 100, 100]), 1.0)
        self.assertAlmostEqual(layouts.shown_share(lay, [1188, 400, 200, 200]), 0.1)
        self.assertEqual(layouts.shown_share(lay, [1400, 400, 100, 100]), 0.0)
        self.assertTrue(layouts.to_frame(lay, [1188, 400, 200, 200]), "the sliver that used to pass")
        followed = {"parts": [dict(lay["parts"][0], follow=[{"x": 600, "y": 0}, {"x": 1000, "y": 0}])]}
        self.assertEqual(layouts.shown_share(followed, [1188, 400, 200, 200]), 1.0, "the frame goes to it")

    def test_a_title_takes_a_stretch_it_can_be_read_in(self):
        """Tarm's seller 10-31: the earliest stretch was under 2 s, with 8 s free later."""
        got = tools._title_order([(3.7, 5.6, "middle"), (9.0, 17.0, "top"), (6.0, 10.0, "middle"), (0.5, 2.0, "top")])
        self.assertEqual([s[0] for s in got], [6.0, 9.0, 3.7, 0.5], "the earliest long one, then the longer of the rest")

    def test_on_a_frame_that_follows_the_product_it_stands_in_the_frames_middle(self):
        """Its box at every place the frame went filled both places for words (Tarm's seller, 10-31)."""
        from unittest import mock
        from test_cutout import make_mask

        def cut_out(rec, a, b, on_start=None, on_progress=None, what=None):
            return {"path": make_mask(self.tmp, "followed.mkv", b - a, a, "lt(abs(X-280),20)", size=(320, 180)),
                    "person": None, "fps": 10, "size": [320, 180], "from": a, "to": b}

        script = self.to_moment()
        with mock.patch.object(tools.person, "cut_out", cut_out):
            self.run_job(script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                         [call("make_clip", moment=1, layout="subject", product=[100, 300, 250, 600], why="the product")],
                         [call("follow_product", what="white game controller", why="the product leads")], "Done.")
        state = self.chat.project.state()
        with mock.patch.object(tools.focus, "thing_boxes", return_value=[]):
            times, _ = tools._face_times(state, self.chat.recording)
        boxes = [b for _, found in times for b in found]
        self.assertTrue(boxes)
        for x, y, w, h in boxes:
            self.assertAlmostEqual(x + w / 2.0, 540, delta=2, msg="in the middle of the frame, though Muse's box was far left")

    def test_sam_tries_the_things_own_noun_when_muses_words_find_nothing(self):
        """024201-3215 on 11-01: "black smartphone" for a blue Redmi; 2 of 5 clips lost their frame and their title."""
        from unittest import mock
        from test_cutout import make_mask
        asked = []

        def cut_out(rec, a, b, on_start=None, on_progress=None, what=None):
            asked.append(what)
            if len(asked) == 1:
                raise tools.person.PersonError("SAM found no %s in the stretch." % what)
            return {"path": make_mask(self.tmp, "noun.mkv", b - a, a, "lt(abs(X-280),20)", size=(320, 180)),
                    "person": None, "fps": 10, "size": [320, 180], "from": a, "to": b}

        script = self.to_moment()
        with mock.patch.object(tools.person, "cut_out", cut_out):
            _, steps = self.run_job(script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                                    [call("make_clip", moment=1, layout="subject", product=[100, 300, 250, 600], why="it")],
                                    [call("follow_product", what="black smartphone", why="the product leads")], "Done.")
        self.assertEqual(asked, ["black smartphone", "smartphone"])
        self.assertIn("SAM 3.1 followed the smartphone through the clip", steps[-1]["result"])
        self.assertEqual(self.chat.project.state()["clip"]["layout"]["thing"]["what"], "smartphone")

    def test_a_sam_that_broke_is_not_asked_again_and_a_one_word_thing_has_no_second_try(self):
        from unittest import mock
        asked = []

        def cut_out(rec, a, b, on_start=None, on_progress=None, what=None):
            asked.append(what)
            raise tools.person.PersonError("SAM stopped." if what == "black phone" else "SAM found no %s in the stretch." % what)

        script = self.to_moment()
        with mock.patch.object(tools.person, "cut_out", cut_out):
            _, steps = self.run_job(script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                                    [call("make_clip", moment=1, layout="subject", product=[100, 300, 250, 600], why="it")],
                                    [call("follow_product", what="black phone", why="the product leads")],
                                    [call("follow_product", what="phone", why="again")], "Done.")
        self.assertEqual(asked, ["black phone", "phone"])
        self.assertIn("could not be followed: SAM stopped.", steps[-2]["result"])
        self.assertIn("could not be followed: SAM found no phone", steps[-1]["result"])

    def test_the_things_own_noun_comes_before_what_it_is_with(self):
        for what, noun in (("black smartphone", "smartphone"), ("tablet with dog case", "tablet"), ("phone", "phone"),
                           ("white game controller", "controller"), ("jar of yellow balm", "jar")):
            self.assertEqual(tools._thing_noun(what), noun)

    def test_a_title_keeps_off_the_product_where_sam_saw_it_at_that_second(self):
        """005549-f1e8 on 11-01: 2 of 5 seller clips had no title, as one box stood for a phone that moved."""
        from unittest import mock
        from test_cutout import make_mask

        def cut_out(rec, a, b, on_start=None, on_progress=None, what=None):
            return {"path": make_mask(self.tmp, "followed.mkv", b - a, a, "lt(abs(X-280),20)", size=(320, 180)),
                    "person": None, "fps": 10, "size": [320, 180], "from": a, "to": b}

        script = self.to_moment()
        with mock.patch.object(tools.person, "cut_out", cut_out):
            self.run_job(script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                         [call("make_clip", moment=1, layout="subject", product=[100, 300, 250, 600], why="the product")],
                         [call("follow_product", what="white game controller", why="the product leads")], "Done.")
        state = self.chat.project.state()
        whole = next(p for p in state["clip"]["layout"]["parts"] if p["name"] == "whole")
        x = whole["src"][0]
        small, big = [x + 10, 20, 30, 40], [x, 0, whole["src"][2], 180]
        with mock.patch.object(tools.focus, "thing_boxes", return_value=[(1.0, small), (3.0, big), (5.0, small)]):
            times, _ = tools._face_times(state, self.chat.recording)

        def at(t):
            return [b for when, found in times if when == t for b in found]

        self.assertEqual(len({tuple(b) for b in at(1.0)}), 1)
        self.assertAlmostEqual(at(1.0)[0][1], 20 * 1920 / 180.0, delta=2, msg="where SAM saw it, not where Muse's box is")
        self.assertEqual(at(3.0), [], "a product filling the frame cannot be kept off")
        self.assertEqual(at(10.0), [], "SAM did not see it then")

    def test_moving_the_ends_keeps_the_frame_on_the_product_it_followed(self):
        """Glimmer's green seller 10-30: after change_ends the frame went back to the face and no title fit."""
        from unittest import mock
        from test_cutout import make_mask
        asked = []

        def cut_out(rec, a, b, on_start=None, on_progress=None, what=None):
            asked.append((a, b))
            return {"path": make_mask(self.tmp, "followed.mkv", b - a, a, "lt(abs(X-280),20)", size=(320, 180)),
                    "person": None, "fps": 10, "size": [320, 180], "from": a, "to": b}

        script = self.to_moment()
        with mock.patch.object(tools.person, "cut_out", cut_out):
            _, steps = self.run_job(
                script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                [call("make_clip", moment=1, layout="subject", why="the product")],
                [call("follow_product", what="white game controller", why="the product leads")],
                [call("change_ends", start="0:32", why="start on the product")], "Done.")
        moved = steps[-1]
        self.assertFalse(moved["refused"], moved["result"])
        self.assertIn("The frame follows the white game controller again over the new ends, from SAM's first pass.",
                      moved["result"])
        lay = self.chat.project.state()["clip"]["layout"]
        self.assertEqual((lay["parts"][0]["src"][0], bool(lay["parts"][0].get("follow")), "thing" in lay),
                         (218, True, True))
        self.assertEqual(len(asked), 1, "SAM is not run again")
        self.chat.undo()
        self.assertEqual(self.chat.project.state()["clip"]["pieces"][0][0], 30.0, "one undo takes back the move whole")

    def test_the_check_fails_where_the_frame_cuts_the_product_sam_follows(self):
        """Tarm 10-30, on a seller's ad that split the screen as she raised the tumbler, so it was gone: "เราต้องดู
        ก่อนว่าเราต้องการจะโฟกัสอะไรในแต่ละคลิปในแต่ละช่วง". A tray wider than the frame is cut all through."""
        from unittest import mock
        from test_cutout import make_mask

        def cut_out(rec, a, b, on_start=None, on_progress=None, what=None):
            where = "lt(abs(X-160),80)" if what == "wide tray" else "lt(abs(X-280),20)"
            return {"path": make_mask(self.tmp, "%s.mkv" % what.replace(" ", "_"), b - a, a, where, size=(320, 180)),
                    "person": None, "fps": 10, "size": [320, 180], "from": a, "to": b}

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        script = self.to_moment()
        had = getattr(render, "export", None)
        render.export = export
        try:
            with mock.patch.object(tools.person, "cut_out", cut_out):
                _, steps = self.run_job(
                    script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                    [call("make_clip", moment=1, layout="subject", why="the product")],
                    [call("follow_product", what="wide tray", why="the product leads")],
                    [call("decide_people", keep=[1], why="the seller")],
                    [call("render_clip", why="the file")], [call("check_clip", why="check it")],
                    [call("follow_product", what="white game controller", why="a smaller one")],
                    [call("render_clip", why="again")], [call("check_clip", why="again")], "Done.")
        finally:
            render.export = had
        wide, checked, small, again = steps[-7], steps[-4], steps[-3], steps[-1]
        self.assertIn("The frame cuts the wide tray or has left it at 0-", wide["result"])
        self.assertIn("SAM sees it there, but most of it is outside the frame.", wide["result"])
        self.assertIn("- FAILED: The product is in the frame (the frame cuts the wide tray or has left it at 0-",
                      checked["result"])
        self.assertNotIn("The frame cuts", small["result"])
        self.assertIn("- passed: The product is in the frame (SAM followed the white game controller, and the frame "
                      "holds it whenever it is seen)", again["result"])
        with mock.patch.object(tools, "_product_cut", return_value=[(2.0, 2.6)]):
            _, brief = self.run_job([call("check_clip", why="a brief cut")], "Done.")
        self.assertIn("- for you to check: The product is in the frame (the frame cuts the white game controller or has "
                      "left it at 2-2.6 s, while SAM sees it; each under 1 s, the frame catching up with the hands: look "
                      "at those seconds)", brief[-1]["result"],
                      "Glimmer 10-30: a 0.5 s cut failed the check and Muse remade the clip without its follow")

    def test_a_sticker_never_covers_a_title_unless_the_person_asked_for_it_there(self):
        self.chat.emit("person", job=None, text="เอาดาวไว้บนชื่อคลิปเลย")
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_title", words="Big win", start=0, end=4, place="top", why="the hook")],
            [call("add_sticker", sticker="star", at=2, place="top", why="the peak")],
            [call("add_sticker", sticker="star", at=10, place="top", why="after the title")],
            [call("add_sticker", sticker="star", at=2, place="top", person_said="เอาดาวไว้บนชื่อคลิป", why="asked")],
            "Done.")
        over, after, asked = steps[-3:]
        self.assertTrue(over["refused"], "game 10-30: a star at the top over the title at 7 s")
        self.assertIn('The sticker would cover the title "Big win" (x ', over["result"])
        self.assertIn("from 0 to 4 s): words on the clip are never covered. Clear of it at 2 s: middle, or a time "
                      "the title is not shown.", over["result"])
        self.assertFalse(after["refused"], after["result"])
        self.assertFalse(asked["refused"], asked["result"])
        self.assertIn('It covers the title "Big win", where the person asked for it; tell the person.', asked["result"])
        self.assertEqual([k["at"] for k in self.chat.project.state()["stickers"]], [2, 10])

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
                [call("change_shot", start=6, end=7, show=1, centre=[120, 300], why="his hands")],
                [call("change_shot", start=1, end=2, show=0, centre=[120, 300], why="the card")],
                [call("slow_push", start=1, end=4, why="build-up")],
                [call("move_part", part="circle", box=[600, 700, 900, 900], why="move it")],
                "Done.")
        finally:
            talk.motion, tools._people = real, listing
        made, reaction, together, nobody, framed, card, push, move = steps[-8:]
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
        self.assertFalse(framed["refused"], framed["result"])
        self.assertTrue(card["refused"])
        self.assertIn("no frame to move", card["result"])
        self.assertTrue(push["refused"])
        self.assertIn("no slow push", push["result"])
        self.assertTrue(move["refused"])
        self.assertIn("change_shot", move["result"])
        lay = self.chat.project.state()["clip"]["layout"]
        self.assertEqual([p["name"] for p in lay["parts"]], ["person 1", "person 2", "everyone"])
        (placed,) = [s for s in lay["shots"] if s.get("crop")]
        self.assertEqual((placed["from"], placed["to"], placed["show"]), (6, 7, 1))
        self.assertEqual(placed["crop"], talk.crop_at(lay, 1, [120, 300]))
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
        self.assertIn("load the vlog recipe", steps[-1]["result"], "Glimmer took EP 22's cooking for a talk (10-31)")

    def test_a_split_talk_clip_shows_both_people_and_hides_a_viewers_handle(self):
        self.faces = found([(30 + i * 0.5, [[60, 40, 40, 40], [220, 40, 40, 40]]) for i in range(40)], size=(320, 180))
        hidden = [{"kind": "a person's handle", "box": [200, 10, 100, 20], "start": 1.0, "end": 5.0}]
        had = screen.hidden, screen.ahead, screen.words
        screen.hidden = lambda rec, pieces, on_start=None, span=None: hidden
        screen.ahead, screen.words = (lambda rec, a, b: None), (lambda rec, a, b: [])
        try:
            brain, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="talk", why="two people talking")], *self.to_moment()[1:],
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
        self.assertEqual(len(offered), 61)

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

    def test_each_clip_of_a_seller_batch_follows_its_product_with_its_background_behind_the_person(self):
        """Lane 5, 10-30, Spark's 205011-1a6e: four clips of Tarm's green-cloth seller came back "handed over, 4 of 4
        checks" with a face filling the frame, the product out of the picture, the raw cloth and no title."""
        from unittest import mock
        from test_cutout import make_mask
        asked = []

        def cut_out(rec, a, b, on_start=None, on_progress=None, what=None):
            asked.append(what)
            return {"path": make_mask(self.tmp, "thing_%d.mkv" % len(asked), b - a, a, "lt(abs(X-280),20)",
                                      size=(320, 180)), "person": None, "fps": 10, "size": [320, 180], "from": a,
                    "to": b}

        for p in self.green_cloth():
            p.start()
            self.addCleanup(p.stop)
        bare = [{"moment": 1, "title": "Eaglet pad", "caption": "a pad", "hashtags": ["#pad"]},
                {"moment": 2, "title": "Eaglet balm", "caption": "a balm", "hashtags": ["#balm"]}]
        box = {"product": [600, 200, 900, 800]}
        named = [dict(bare[0], what="white game controller", **box), dict(bare[1], what="yellow balm jar", **box)]
        dressed = [dict(c, background="a calm shop shelf") for c in named]
        with mock.patch.object(tools.person, "cut_out", cut_out):
            _, steps = self.several([call("make_clips", clips=bare, why="no product")],
                                    [call("make_clips", clips=[dict(bare[0], **box), named[1]], why="no words")],
                                    [call("make_clips", clips=[named[0], dict(bare[1], what="a jar")], why="no box")],
                                    [call("make_clips", clips=named, why="no background")],
                                    [call("make_clips", clips=dressed, why="all of it")], '{"lines": ["one heard"]}', "Done.",
                                    recipe="seller")
        nothing, half, unboxed, plain, made = steps[-5:]
        self.assertTrue(nothing["refused"])
        self.assertIn("Nothing was made. A seller's clip shows its product: give the clip of moment 1 and 2 its "
                      "product, the box around it in the picture, and what, the product in a few English words",
                      nothing["result"])
        self.assertTrue(half["refused"])
        self.assertIn("give the clip of moment 1 its product", half["result"])
        self.assertTrue(unboxed["refused"])
        self.assertIn("give the clip of moment 2 its product", unboxed["result"])
        self.assertTrue(plain["refused"])
        self.assertIn("A green cloth is behind the person in moment 1 and 2: give each clip a background",
                      plain["result"])
        self.assertFalse(made["refused"], made["result"])
        self.assertEqual(asked, ["white game controller", "yellow balm jar"], "one pass for each clip's own product")
        self.assertEqual(len(self.cut), 2, "the cloth is cut once for each clip")
        self.assertIn("handed over, 5 of 5 checks passed; SAM 3.1 followed the white game controller through the clip: "
                      "it is in 100% of the pictures; the background is a calm shop shelf", made["result"])
        self.assertIn("SAM 3.1 followed the yellow balm jar through the clip", made["result"])
        self.assertIn("Made 2 clips, 2 handed over", made["result"])
        with open(os.path.join(agent.WORK, "spent.jsonl"), encoding="utf-8") as f:
            paid = [json.loads(line) for line in f if '"image"' in line]
        self.assertEqual(len(paid), 1, "the same words are drawn once for all the clips")
        state = self.chat.project.state()
        self.assertEqual(state["clip"]["layout"]["thing"]["what"], "white game controller", "the first clip is shown")
        self.assertEqual(state["backdrop"]["about"], "a calm shop shelf")

    def test_one_clip_that_breaks_leaves_the_others_of_the_batch(self):
        """025525-3e8e on 11-01: one line of one clip raised, and the whole batch of 5 was lost, twice."""
        from unittest import mock
        real, calls = tools._one_clip, []

        def one_clip(job, rec, k, c, layout, blur, opened=None):
            calls.append(k)
            if k == 1:
                with mock.patch.object(tools, "make_clip", side_effect=ValueError("a line that cannot be cut")):
                    return real(job, rec, k, c, layout, blur, opened)
            return real(job, rec, k, c, layout, blur, opened)

        with mock.patch.object(tools, "_one_clip", one_clip):
            _, steps = self.batch()
        made = steps[-1]["result"]
        self.assertFalse(steps[-1]["refused"], made)
        self.assertEqual(calls, [1, 2])
        self.assertIn("Clip 1 (moment 1): broke (a line that cannot be cut)", made)
        self.assertIn("Made 2 clips, 1 handed over", made)

    def test_a_sellers_title_helps_fix_a_name_the_hearing_got_wrong(self):
        """020751-cdcd on 11-01: "Let me" where he says Redmi, under the title Redmi."""
        from unittest import mock
        clips = [{"moment": n, "title": "Redmi 200 บาท", "caption": "c1", "hashtags": ["#a"], "product": [600, 200, 900, 800],
                  "what": "a phone"} for n in (1, 2)]
        with mock.patch.object(tools, "_on_the_product", return_value=True), mock.patch.object(tools.green, "find", return_value=None):
            brain, steps = self.several([call("make_clips", clips=clips, why="the phone")], '{"lines": ["Redmi heard"]}',
                                        "Done.", recipe="seller", shown=("Redmi",))
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        asked = next(m for m in brain.sent if "Lines to fix" in str(m[-1]) and "Redmi" in str(m[-1]))
        self.assertIn("Written on the screen at the time", asked[-1]["content"])
        shown = "".join(line["text"] for line in self.chat.project.state()["captions"]["lines"])
        self.assertIn("Redmi", shown)

    def test_an_example_background_is_refused_before_a_batch_does_any_work(self):
        """153416-7e05: 12 minutes of SAM and rendering, then "not handed over" for the tool's own example."""
        from unittest import mock
        asked = []

        def cut_out(rec, a, b, **k):
            asked.append(a)
            raise tools.person.PersonError("not this far")

        for p in self.green_cloth():
            p.start()
            self.addCleanup(p.stop)
        copied = [{"moment": 1, "title": "Eaglet pad", "caption": "a pad", "hashtags": ["#pad"], "product": [600, 200, 900, 800],
                   "what": "white game controller", "background": "A tidy gaming desk with soft blue light."}]
        with mock.patch.object(tools.person, "cut_out", cut_out):
            _, steps = self.several([call("make_clips", clips=copied + [dict(copied[0], moment=2)], why="two")], "Done.", recipe="seller")
        self.assertTrue(steps[-1]["refused"], steps[-1]["result"])
        self.assertTrue(steps[-1]["result"].startswith("Nothing was made. \"A tidy gaming desk with soft blue light.\" is the example"),
                        steps[-1]["result"])
        self.assertEqual(asked, [], "no work was done first")

    def test_a_seller_batch_is_shown_what_no_look_saw_once_and_the_second_answer_stands(self):
        """153416-7e05: 3 looks of 30 s of 2:20 missed the balm."""
        tools.PRODUCT_RECIPES = self.product_recipes
        two = [call("propose_moments", why="two", moments=[{"look": 1, "start": "0:05", "end": "0:25", "reason": "one"},
                                                           {"look": 2, "start": "0:30", "end": "0:50", "reason": "two"}])]
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="seller", why="a seller")],
                                [call("look_at_piece", start="0:05", end="0:25", why="the first")],
                                [call("look_at_piece", start="0:30", end="0:50", why="the second")], two, two, "Done.")
        told, again = steps[-2:]
        self.assertFalse(told["refused"], told["result"])
        self.assertTrue(told["result"].startswith("Not proposed yet: a seller shows one product after another, and your looks had "
                                                  "not seen 0:50-1:00 of this recording"), told["result"])
        self.assertIn("Look 3: 0:50-", told["result"], "the program looked there")
        self.assertTrue(again["result"].startswith("Proposed 2 moments"), again["result"])

    def test_one_moment_or_a_look_at_everything_is_not_sent_to_see_more(self):
        tools.PRODUCT_RECIPES = self.product_recipes
        job = mock.Mock()
        job.chat.clip, job.chat.events, job.chat.looks = {"recipe": "seller"}, [], [{"start": 0.0, "end": 20.0}]
        self.assertIsNone(tools._all_seen(job, "why", {"seconds": 120.0}, [{}]), "one clip asked for")
        self.assertIsNone(tools._all_seen(job, "why", {"seconds": 900.0}, [{}, {}]), "a long live")
        job.chat.clip = {"recipe": "talk"}
        self.assertIsNone(tools._all_seen(job, "why", {"seconds": 120.0}, [{}, {}]), "not a seller")

    def test_what_no_look_saw_is_told_once_a_job_even_when_a_stretch_stays_unseen(self):
        tools.PRODUCT_RECIPES = self.product_recipes
        job = mock.Mock()
        job.chat.clip, job.chat.looks, job.chat.events = {"recipe": "seller"}, [{"start": 0.0, "end": 20.0}], []
        job.emit = lambda kind: job.chat.events.append({"kind": kind, "job": job.n})
        with mock.patch.object(tools, "look_at_piece", lambda *a: "looked"):
            self.assertIn("looked", tools._all_seen(job, "why", {"seconds": 120.0}, [{}, {}]))
            self.assertIsNone(tools._all_seen(job, "why", {"seconds": 120.0}, [{}, {}]), "the second answer stands")

    def test_a_proposed_moment_with_another_products_opening_is_sent_back_once(self):
        """153416-7e05: the Redmi and the Samsung in one clip, graded 6."""
        tools.PRODUCT_RECIPES = self.product_recipes
        self.worker.phrases = [{"from": 8.0, "to": 10.0, "text": "ต่อไปเป็นโทรศัพท์ซัมซุง", "sure": 0.95}]
        one = [call("propose_moments", why="the phones", moments=[{"look": 1, "start": "0:30", "end": "0:50", "reason": "phones"}])]
        with mock.patch.object(tools, "_fixed_said", lambda job, rec, look, a, b, said=None: [dict(
                p, heard=p["text"], kept="", named=[]) for p in look.get("said") or []]):
            _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="seller", why="a seller")],
                                    self.to_moment()[1], one, one, "Done.")
        told, again = steps[-2:]
        self.assertTrue(told["refused"], told["result"])
        self.assertTrue(told["result"].startswith("Not proposed yet: a clip holds one product, and moment 1 (0:30-0:50) has another "
                                                  "product's opening at 0:38"), told["result"])
        self.assertTrue(again["result"].startswith("Proposed 1 moment"), again["result"])

    def test_the_stretches_no_look_saw_are_found_whatever_order_the_looks_came_in(self):
        looks = [{"start": 106.0, "end": 136.0}, {"start": 9.0, "end": 39.0}, {"start": 66.0, "end": 96.0}, {"start": 20.0, "end": 30.0}]
        self.assertEqual(tools._not_looked(looks, 140.0), [(0.0, 9.0), (39.0, 66.0), (96.0, 106.0)], "4 s at the end is no product")
        self.assertEqual(tools._not_looked(looks, 150.0)[-1], (136.0, 150.0))

    def test_a_moment_with_another_products_opening_in_it_is_told_once(self):
        tools.PRODUCT_RECIPES = self.product_recipes
        job = mock.Mock()
        job.chat.clip = {"recipe": "seller"}
        said = [{"from": 64.0, "to": 67.0, "text": "สินค้าชิ้นที่สามคือโทรศัพท์"}, {"from": 79.0, "to": 80.0, "text": "ต่อไปเป็นโทรศัพท์ซัมซุง"}]
        looks = [{"said": said}]
        two = tools._two_products(job, looks, [{"n": 2, "look": 1, "start": 64.0, "end": 96.0}])
        self.assertEqual(two, ['moment 2 (1:04-1:36) has another product\'s opening at 1:19: "ต่อไปเป็นโทรศัพท์ซัมซุง"'])
        self.assertEqual(tools._two_products(job, looks, [{"n": 1, "look": 1, "start": 64.0, "end": 78.0}]), [], "ends before it")
        job.chat.clip = {"recipe": "talk"}
        self.assertEqual(tools._two_products(job, looks, [{"n": 2, "look": 1, "start": 64.0, "end": 96.0}]), [])

    def test_a_seller_batch_says_what_it_could_not_do_and_stops_when_the_person_stops_it(self):
        from unittest import mock
        from test_cutout import make_mask

        def cut_out(rec, a, b, on_start=None, on_progress=None, what=None):
            if what == "a stopped thing":
                raise tools.person.Stopped()
            if what == "a broken thing":
                raise tools.person.PersonError("SAM is not set up.")
            return {"path": make_mask(self.tmp, "kept.mkv", b - a, a, size=(320, 180)), "person": None, "fps": 10,
                    "size": [320, 180], "from": a, "to": b}

        def clip(n, what):
            return {"moment": n, "title": "Eaglet pad", "caption": "a pad", "hashtags": ["#pad"], "what": what,
                    "product": [200, 200, 800, 800], "background": "a calm shop shelf"}

        broken = [clip(1, "a broken thing"), clip(2, "a stopped thing")]
        with mock.patch.object(tools.person, "cut_out", cut_out), mock.patch.object(tools.green, "find",
                                                                                     return_value=None):
            _, steps = self.several([call("make_clips", clips=broken, why="both")], '{"lines": ["one heard"]}', "Done.",
                                    recipe="seller")
        made = steps[-1]["result"]
        self.assertIn("The a broken thing could not be followed: SAM is not set up. Tell the person, or leave the "
                      "frame where it is", made)
        self.assertIn("no green cloth is behind the person, so the background was left out", made)
        self.assertIn(": handed over, 5 of 5 checks passed; The a broken thing", made,
                      "the frame could not follow, so the box it was given shows the product is in the picture")
        self.assertIn("Clip 2 (moment 2, 0:40-0:50, 10 s, subject layout): not made, the person pressed Stop.", made)

    def batch(self, *more):
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        return self.several([call("make_clips", clips=clips, why="both")], *more, "Done.")

    def test_the_clips_made_in_one_go_sit_in_an_album_each_with_its_own_edits(self):
        """Tarm 10-30: "ช่วยดูเรื่อง preview ให้มันออกมาเป็น album จะได้เลือกแก้ไขต่อได้ง่าย"."""
        _, steps = self.batch()
        self.assertIn("The first clip is open: your tools change the clip that is open", steps[-1]["result"])
        album = self.chat.album_now()
        self.assertEqual(album["at"], 0)
        self.assertEqual([(c["title"], c["state"]) for c in album["clips"]],
                         [("Eaglet one", "ready"), ("Eaglet two", "ready")])
        self.assertTrue(all(c["file"].endswith(".mp4") for c in album["clips"]))
        self.assertEqual(self.chat.project.state()["clip"]["moment"], 1)
        self.assertIsNone(self.chat.pick_clip(1))
        self.assertEqual(self.chat.project.state()["clip"]["moment"], 2, "clips after the first can be edited too")
        picked = [e for e in self.chat.events if e["kind"] == "picked"][-1]
        self.assertEqual((picked["at"], picked["ready"]["title"], picked["file"]["name"], picked["title"]),
                         (1, "Eaglet two", album["clips"][1]["file"], "Eaglet two"))
        self.assertEqual(picked["clip"]["moment"], 2)
        self.assertTrue(self.chat.told[-1].startswith("the person opened clip 2 of the album"))
        self.assertIn("clip 2 of 2 in the album", self.chat.clip_line())
        self.run_job([call("add_title", words="Only two", start=0, end=1, place="top", why="on this clip")], "Done.")
        self.assertEqual(self.chat.album_now()["clips"][1]["state"], "changed", "the file is older than the change")
        self.assertIsNone(self.chat.pick_clip(0))
        self.assertNotIn("Only two", [t["text"] for t in self.chat.project.state()["titles"]])
        self.assertIsNone(self.chat.pick_clip(1))
        self.assertIn("Only two", [t["text"] for t in self.chat.project.state()["titles"]], "its edit waited for it")
        self.chat.undo()
        self.assertNotIn("Only two", [t["text"] for t in self.chat.project.state()["titles"]], "its own undo")

    TAGLINE = [call("set_tagline", words="Daily notes", spot="bottom right", why="mine")]
    MATCH = [call("match_album", parts=["tagline", "colour"], why="the whole album")]

    def test_one_clips_look_goes_on_the_others_and_each_keeps_its_own_headline(self):
        """Tarm 10-30: "Tagline ทั้งอัลบั้ม ... ส่วนพวกตัวหนังสือที่เป็นหัวข้อก็เปลี่ยนเอาแล้วแต่ คลิปวิดีโอ"."""
        _, steps = self.batch(self.TAGLINE, [call("set_look", look="warm", why="warm")], self.MATCH)
        self.assertIn("The look is on 1 clip of the album.", steps[-1]["result"])
        other = self.chat.album[1]["project"].state()
        self.assertEqual((other["channel"]["name"], other["channel"]["spot"], other["look"]),
                         ("Daily notes", "bottom right", "warm"))
        self.assertEqual(self.chat.album_at, 0, "the clip open stays open")
        self.assertEqual(self.chat.album_now()["clips"][1]["state"], "changed", "its video is older than the look")
        self.assertTrue(self.chat.album_now()["matched"])
        self.assertEqual(self.chat.album_now()["changed"], 2)
        self.assertEqual(len([e for e in self.chat.events if e["kind"] == "picked"]), 0, "no clip was opened to do it")

    def test_one_undo_on_a_clip_takes_the_whole_look_back_from_that_clip(self):
        self.batch(self.TAGLINE, [call("set_look", look="warm", why="warm")], self.MATCH)
        self.assertIsNone(self.chat.pick_clip(1))
        self.chat.undo()
        other = self.chat.project.state()
        self.assertEqual((other["channel"], other["look"]), (None, None))

    def test_the_album_look_is_taken_back_and_stays_back_when_the_chat_is_read_back(self):
        _, steps = self.batch(self.TAGLINE, self.MATCH, [call("undo_album", why="not that")])
        self.assertEqual(steps[-1]["result"], "The look was taken back from 1 clip.")
        self.assertIsNone(self.chat.album[1]["project"].state()["channel"])
        self.assertFalse(self.chat.album_now()["matched"])
        back = agent.Chat.load(self.chat.id)
        self.assertEqual([p["project"].entries for p in back.album], [p["project"].entries for p in self.chat.album])
        self.assertFalse(back.album_now()["matched"])
        self.assertEqual(self.chat.unmatch_album(), "There is no album-wide look to take back.")

    def test_a_look_on_the_album_comes_back_when_the_chat_is_read_back(self):
        self.batch(self.TAGLINE, self.MATCH)
        back = agent.Chat.load(self.chat.id)
        self.assertEqual(back.album[1]["project"].state()["channel"]["name"], "Daily notes")
        self.assertEqual(back.album[1]["project"].acts, self.chat.album[1]["project"].acts)
        self.assertTrue(back.album_now()["matched"])

    def test_a_clip_changed_since_keeps_its_change_when_the_look_is_taken_back(self):
        self.batch(self.TAGLINE, self.MATCH)
        self.assertIsNone(self.chat.pick_clip(1))
        self.chat.edit({"do": "look", "name": "film"}, act="after")
        said = self.chat.unmatch_album()
        self.assertEqual(said, "The look was taken back from 0 clips. Clip 2 changed since, so take it back there "
                               "with Undo.")
        self.assertEqual(self.chat.project.state()["look"], "film")

    def test_a_tagline_a_clip_cannot_draw_stays_off_it_and_the_other_parts_still_go(self):
        from unittest import mock
        self.batch(self.TAGLINE, [call("set_look", look="warm", why="warm")])

        def unfit(state, default):
            return "The letters do not fit." if state["channel"] and state["clip"]["moment"] == 2 else None

        with mock.patch.object(agent, "channel_unfit", unfit):
            said = self.chat.match_album(["tagline", "colour"])
        self.assertIn("Clip 2 kept its own: The letters do not fit.", said)
        other = self.chat.album[1]["project"].state()
        self.assertEqual((other["channel"], other["look"]), (None, "warm"))

    def test_a_clip_with_no_circle_says_so_and_nothing_moves(self):
        from unittest import mock
        self.batch()
        with mock.patch.object(agent, "_circle_box", return_value=None), \
                mock.patch.object(tools, "_circle_box", return_value=None):
            said = self.chat.match_album(["circle"])
        self.assertIn("This clip has no face circle, so no circle moved.", said)

    def test_nothing_to_do_is_said_and_nothing_is_logged(self):
        self.batch()
        self.assertEqual(self.chat.match_album(["colour"]), "Every other clip already has this look.")
        self.assertFalse(any(e["kind"] == "matched" for e in self.chat.events))

    def test_there_is_no_album_with_one_clip(self):
        self.several([call("make_clip", moment=1, layout="subject", why="one")],
                     [call("match_album", parts=["colour"], why="all")], "Done.")
        steps = [e for e in self.chat.events if e["kind"] == "step" and e["tool"] == "match_album"]
        self.assertIn("There is no other clip to change", steps[-1]["result"])

    def test_update_all_makes_the_changed_clips_only_and_opens_the_clip_it_left(self):
        _, steps = self.batch([call("open_clip", clip=2, why="that one")],
                              [call("set_look", look="warm", why="warm")],
                              [call("open_clip", clip=1, why="back")],
                              [call("render_album", why="update")],
                              [call("render_album", why="again")])
        self.assertIn("Clip 2: Rendered the finished clip", steps[-2]["result"])
        self.assertNotIn("Clip 1", steps[-2]["result"])
        self.assertEqual(self.chat.album_at, 0, "the clip it left")
        self.assertEqual([c["state"] for c in self.chat.album_now()["clips"]], ["ready", "ready"])
        self.assertIn("No clip of the album has changed", steps[-1]["result"])

    def test_a_clip_changed_by_chat_shows_as_changed_though_no_panel_looked_in_between(self):
        self.batch([call("open_clip", clip=2, why="that one")], [call("set_look", look="warm", why="warm")])
        self.assertEqual(self.chat.album_now()["clips"][1]["state"], "changed")
        self.assertTrue(self.chat.album_now()["clips"][1]["file"].endswith(".mp4"), "its last video stays its picture")

    def test_update_all_stops_on_stop_and_goes_past_a_clip_it_cannot_make(self):
        from unittest import mock
        self.batch(self.TAGLINE, self.MATCH)
        job = types.SimpleNamespace(chat=self.chat)
        with mock.patch.object(tools, "render_clip", return_value="Not rendered: the person pressed Stop.") as made:
            said = tools.render_album(job, "why")
        self.assertEqual((made.call_count, said), (1, "Clip 1: Not rendered: the person pressed Stop."))
        self.assertEqual(self.chat.album_at, 0, "it still opens the clip it left")
        with mock.patch.object(tools, "render_clip", side_effect=tools.Refused("Decide about the man first.")) as made:
            said = tools.render_album(job, "why")
        self.assertEqual(made.call_count, 2)
        self.assertEqual(said, "Clip 1: Decide about the man first.\nClip 2: Decide about the man first.")

    def test_the_album_tools_are_offered_only_with_an_album_so_a_first_draft_is_asked_as_before(self):
        names = lambda schemas: [s["function"]["name"] for s in schemas]
        job = lambda album: types.SimpleNamespace(finishing=0, chat=types.SimpleNamespace(events=[], album=album, clip=None))
        none = agent.Job.offered(job([]))
        later = list(tools.ALBUM_TOOLS) + ["no_chat_box"]
        self.assertEqual(names(none), [n for n in names(tools.UNASKED) if n not in later])
        self.assertEqual(sorted(set(names(tools.UNASKED)) - set(names(none))), sorted(later))
        some = agent.Job.offered(job([{}]))
        self.assertEqual(names(some), [n for n in names(tools.UNASKED) if n != "no_chat_box"])
        self.assertIn("open_clip", names(none), "as it was")

    def test_the_clip_open_is_not_asked_about_itself(self):
        from unittest import mock
        self.batch()
        with mock.patch.object(agent, "album_entries", return_value=([], ["no"])) as asked:
            said = self.chat.match_album(["colour"])
        self.assertEqual(asked.call_count, 1)
        self.assertNotIn("Clip 1", said)

    def test_a_new_album_cannot_take_back_the_look_of_the_old_one(self):
        self.batch(self.TAGLINE, self.MATCH)
        self.chat.fill_album([], 0)
        self.assertIsNone(self.chat._last_match())

    def test_spaces_in_a_tagline_are_tidied(self):
        self.batch([call("set_tagline", words="Daily   notes", why="mine")])
        self.assertEqual(self.chat.project.state()["channel"]["name"], "Daily notes")

    def test_empty_words_take_the_tagline_off(self):
        _, steps = self.batch(self.TAGLINE, [call("set_tagline", words="  ", why="off")])
        self.assertEqual(steps[-1]["result"], "The tagline is off the clip.")
        self.assertIsNone(self.chat.project.state()["channel"])

    def test_an_album_comes_back_with_its_clips_apart_when_the_chat_is_read_back(self):
        """make_clips kept the first clip's entries in memory, and a chat read back piled every clip's on one."""
        self.batch()
        self.chat.pick_clip(1)
        self.run_job([call("add_title", words="Only two", start=0, end=1, place="top", why="on this clip")], "Done.")
        kept = [(p["moment"], list(p["project"].entries), list(p["project"].acts)) for p in self.chat.album]
        back = agent.Chat.load(self.chat.id)
        self.assertEqual([(p["moment"], p["project"].entries, p["project"].acts) for p in back.album], kept)
        self.assertEqual(back.album_at, 1)
        self.assertEqual(back.project.state()["clip"]["moment"], 2)
        self.assertEqual(back.project.entries, self.chat.project.entries)
        self.assertEqual(back.album_now()["clips"][0]["state"], "ready")

    def test_muse_opens_a_clip_of_the_album_and_make_clip_goes_to_that_clips_place(self):
        _, steps = self.batch([call("open_clip", clip=3, why="not there")], [call("open_clip", clip=2, why="that one")],
                              [call("make_clip", moment=1, layout="card", why="fix the first")])
        wrong, opened, remade = steps[-3:]
        self.assertTrue(wrong["refused"])
        self.assertIn("The album has 2 clips, so clip 3 is not one of them.", wrong["result"])
        self.assertIn("Clip 2 of 2 is open. clip 2 of 2 in the album", opened["result"])
        self.assertFalse(remade["refused"], remade["result"])
        self.assertEqual((self.chat.album_at, len(self.chat.album)), (0, 2), "clip 2 is not made over")
        self.assertEqual(self.chat.album[1]["project"].state()["clip"]["moment"], 2)

    def test_a_clip_from_a_moment_not_in_the_album_gets_a_new_place_and_comes_back_with_it(self):
        self.batch()
        self.chat.album_to(9)
        self.assertEqual((self.chat.album_at, len(self.chat.album)), (2, 3))
        self.assertEqual(self.chat.album[0]["project"].state()["clip"]["moment"], 1, "the clips before stay")
        self.assertIsNone(self.chat.project.state()["clip"])
        back = agent.Chat.load(self.chat.id)
        self.assertEqual((back.album_at, len(back.album), back.album[2]["moment"]), (2, 3, 9))

    def test_a_second_batch_takes_its_moments_places_and_leaves_the_others_clean(self):
        clips = [{"moment": 1, "title": "Eaglet again", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet too", "caption": "c2", "hashtags": ["#b"]}]
        self.batch()
        self.chat.album_to(9)
        since = self.chat.events[-1]["id"]
        self.run_job([call("make_clips", clips=clips, why="both again")], "Done.")
        self.assertEqual([e for e in self.chat.events if e["kind"] == "picked" and e["id"] > since], [],
                         "the page does not jump from clip to clip while a batch is made")
        self.assertEqual([p["moment"] for p in self.chat.album], [1, 2, 9], "the same moments keep their places")
        self.assertEqual(self.chat.album_at, 0)
        self.assertEqual(self.chat.album[2]["project"].entries, [], "no edit of the batch went to another place")
        self.assertEqual([c["title"] for c in self.chat.album_now()["clips"][:2]], ["Eaglet again", "Eaglet too"])
        back = agent.Chat.load(self.chat.id)
        self.assertEqual([p["project"].entries for p in back.album], [p["project"].entries for p in self.chat.album])

    def test_a_clip_of_the_album_with_no_hand_over_says_so(self):
        self.batch()
        self.chat.album_to(9)
        now = self.chat.album_now()["clips"][2]
        self.assertEqual((now["state"], now["file"], now["title"]), ("unfinished", None, ""))

    def test_the_clip_open_before_a_batch_keeps_its_place_and_no_clip_made_leaves_it_open(self):
        self.several([call("make_clip", moment=1, layout="card", why="the first")], "Done.")
        before = self.chat.project
        self.chat.hold_open_clip()
        self.assertEqual(self.chat.project.entries, [], "the next clip starts clean")
        self.chat.fill_album([], 0)
        self.assertIs(self.chat.project, before, "nothing was made, so nothing changed")
        self.assertIsNone(self.chat.album_now(), "one clip is no album yet")
        made = [{"moment": 2, "title": "Two", "project": tools.project.Project(before.entries)}]
        self.chat.hold_open_clip()
        self.chat.fill_album(made, 0)
        self.assertEqual([p["moment"] for p in self.chat.album], [1, 2])
        self.assertIs(self.chat.album[0]["project"], before)
        self.assertEqual(self.chat.album_at, 1, "the first of the clips just made opens")

    def test_opening_a_new_recording_empties_the_album(self):
        self.batch()
        self.chat.open(dict(self.chat.recording, path=self.clip + ".other", name="another.mp4"))
        self.assertEqual(self.chat.album, [])
        self.assertEqual(agent.Chat.load(self.chat.id).album, [], "and so does reading the chat back")

    def test_the_album_waits_while_the_video_is_made_or_muse_works(self):
        self.batch()
        self.assertEqual(self.chat.pick_clip(5), "That clip is not in the album.")
        self.chat.making = object()
        self.assertIn("The video is being made now", self.chat.pick_clip(1))
        self.chat.making = None
        self.assertEqual(self.chat.album_at, 0)

    def test_a_clips_cuts_and_the_persons_own_land_in_the_quiet_and_say_where(self):
        """Tarm 10-30: "ไม่เอาตัดกลางคำ". The sound is loud from 32 to 46 s; lane 4: change_ends kept a drag where it was
        put, so it could land inside a word."""
        takes.heard_level = self.level
        make_moment(self.clip, loud=(32, 46))
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("make_clips", clips=clips, why="both")],
                                [call("change_ends", start="0:32.5", end="0:45", why="into the first word")],
                                [call("change_ends", start="0:35", why="in the middle of the sound")], "Done.")
        lands, stays = steps[-2:]
        self.assertFalse(lands["refused"], lands["result"])
        self.assertIn("it now runs from 0:31.7 to 0:45.0", lands["result"], "0.3 s before the sound, not 0:32.5")
        self.assertIn("The program moved 1 cut, at most 0.78 s, into the nearest quiet sound", lands["result"])
        self.assertIn("it now runs from 0:35.0 to", stays["result"])
        self.assertIn("Nothing is quiet within 1 s of the cut at 0:35.0: it is on sound as loud as the talk.",
                      stays["result"])
        self.assertEqual(self.chat.project.state()["clip"]["pieces"][0][0], 35.0)
        checked = [c for checks in self.chat.checks.values() for c in checks if c["name"] == "The cuts are in quiet sound"]
        self.assertEqual(len(checked), 2, "each clip's check asks it")
        self.assertIn("on sound as loud as the talk at 0:40.0 of the recording", checked[0]["found"])

    def test_a_talk_clips_cuts_land_in_the_quiet_too(self):
        from unittest import mock
        with mock.patch.object(tools, "_make_talk", return_value="Made.") as talk, \
                mock.patch.object(tools, "_on_quiet", lambda rec, made, level, last_end: ([[1.0, 2.0]], "landed")):
            self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="talk", why="the talk")], "Done.")
        self.assertEqual(talk.call_args[0][4:6], ([[1.0, 2.0]], "landed"))

    def test_a_game_clips_cuts_are_moved_too_and_a_montages_are_not(self):
        from unittest import mock
        asked = []

        def land(rec, made, level, last_end=False):
            asked.append(level)
            return made

        with mock.patch.object(tools, "_on_quiet", land), mock.patch.object(tools.takes, "heard_level",
                                                                             return_value=(0.0, [-20.0])):
            self.several([call("make_clip", moment=1, layout="card", why="the game")], "Done.", recipe="game")
            self.assertEqual(asked, [(0.0, [-20.0])], "the level reaches the cuts of a game, whose moment stays whole")
            asked.clear()
            self.chat.clip = dict(self.chat.clip, recipe=tools.MONTAGE)
            self.run_job([call("make_clip", moment=1, layout="card", why="the trip")], "Done.")
        self.assertEqual(asked, [None], "a montage cuts on the picture, under its music")

    def test_a_sound_as_loud_as_the_talk_after_the_last_line_stays(self):
        """Lane 5, 10-30: the hearing writes no words for a laugh, and a talk clip lost the 11 s of laughing after its
        last line. Here the line is said at 32-34 s and the sound goes on as loud to 40 s, then the room is quiet."""
        takes.heard_level = self.level
        make_moment(self.clip, loud=(32, 40))
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("make_clips", clips=clips, why="both")], "Done.")
        self.assertIn("Clip 1 (moment 1, 0:32-0:40, 8.62 s, card layout): handed over, ", steps[-1]["result"],
                      "from 0.3 s before the line to 0.3 s after the sound, which runs to the moment's end")

    def test_a_saved_look_keeps_the_captions_off_and_its_layout_and_length_for_its_kind_of_clip(self):
        """Tarm 10-30: "ทำให้ user save template ได้ไปเลย เช่น Enable / disable caption"."""
        channels.save({"name": "Mine", "captions": "off", "layouts": {"explain": "subject", "talk": "split"},
                       "lengths": {"explain": 25}})
        channels.use("Mine")
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("make_clips", clips=clips, why="both")],
                                [call("add_captions", why="the recipe has them")],
                                [call("add_captions", why="the person asked")], "Done.")
        loaded, made, refused, asked = steps[1], steps[-3], steps[-2], steps[-1]
        self.assertIn("buttons to change them: about 25 s", loaded["result"])
        self.assertIn("The person's saved look keeps this kind of clip to the subject layout, about 25 s, no captions.",
                      loaded["result"])
        self.assertIn("Clip 1 (moment 1, 0:32-0:34, 2.6 s, subject layout): handed over, ", made["result"],
                      "the look's layout, with no layout given")
        self.assertIn("no captions, as the person's saved look keeps them off", made["result"])
        self.assertTrue(refused["refused"])
        self.assertIn('The person\'s saved look, "Mine", keeps the captions off', refused["result"])
        self.assertFalse(asked["refused"], "the person asked: their words win over the look")
        self.assertEqual(tools.look_said(channels.get("Mine"), muse=False), "no captions")
        job = types.SimpleNamespace(n=1, chat=types.SimpleNamespace(events=[]), emit=lambda kind: None)
        self.assertIsNotNone(tools._layout_unasked(job, "seller", ["card"]), "the recipe's rule")
        channels.save({"name": "Seller", "layouts": {"seller": "card"}})
        channels.use("Seller")
        self.assertIsNone(tools._layout_unasked(job, "seller", ["card"]), "the layout the person saved is their choice")

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
        from unittest import mock
        pack = {"product": [200, 200, 800, 800], "what": "white pack"}
        clips = [dict(pack, moment=1, title="ASTI pack", caption="ASTI c", hashtags=["#ASTI", "#SineySociety"]),
                 dict(pack, moment=2, title="PDRN pack", caption="c", hashtags=["#b"])]
        fixed = dict(clips[0], title="Jabs pack", caption="Jabs c", hashtags=["#Jabs"])
        with mock.patch.object(tools.person, "cut_out", side_effect=tools.person.PersonError("SAM is not set up.")):
            _, steps = self.several([call("make_clips", clips=clips, why="both")],
                                    [call("make_clips", clips=[fixed, clips[1]], why="as the pack says")],
                                    '{"lines": ["one heard"]}', "Done.", recipe="seller", shown=["Jabs", "PDRN", "ASTAXANTHIN"])
        asked, made = steps[-2], steps[-1]
        self.assertTrue(asked["refused"])
        self.assertEqual(asked["result"], 'Nothing was made. Clip of moment 1: "ASTI", "SineySociety" are not on the '
                                          'screen there, not said and not written by the person; the screen there '
                                          'shows Jabs, PDRN, ASTAXANTHIN. Write each name as the screen shows it, or '
                                          'leave it out; if the person gave you the name, make the clips again '
                                          'unchanged.', "EP81 10-28: the pack said Jabs, three clips said ASTI")
        self.assertFalse(made["refused"], made["result"])
        self.assertEqual([e["title"] for e in self.chat.events if e["kind"] == "ready"], ["Jabs pack", "PDRN pack"])

    def test_a_name_heard_without_its_vowels_counts_as_said_and_one_muse_made_up_still_does_not(self):
        from unittest import mock
        job = type("Job", (), {"n": 1, "chat": type("Chat", (), {"events": []})()})()
        said = ["วันนี้มารีวิว iแpd ตัวใหม่ครับ", "astaxanthin ช่วยเรื่องผิว"]
        with mock.patch.object(screen, "words", lambda rec, a, b: ["Jabs"]),                 mock.patch.object(screen, "sites", lambda rec: []):
            names, _ = tools._unseen_names(job, {"name": "review.mp4"}, 0, 10, "iPad ASTI PDRN", said)
        self.assertEqual(names, ["ASTI", "PDRN"], "lane 5 10-30: iPad was heard as ipd and its title refused")

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
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="talk", why="two people talking")], *self.to_moment()[1:],
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
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="vlog", why="a travel episode")], *self.to_moment()[1:],
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

    def test_a_talk_far_under_its_length_is_told_once_a_job_and_the_second_answer_stands(self):
        """Lane 4 10-31: 054011-5f01 sent 18:21 back unchanged, and 043921-4388 kept 25:58 with its look open to 26:29."""
        tools.EDGE_RECIPES, tools.SHORT_RECIPES = self.edge_recipes, self.short_recipes
        short = [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the end", "start": "0:34",
                                                                   "end": "0:46"}])]
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="talk", why="a talk")],
                                self.to_moment()[1], short, short, "Done.")
        told, again = steps[3], steps[4]
        self.assertTrue(told["result"].startswith("Not proposed yet: moment 1 runs 12 s of the 60 s this clip can hold."),
                        told["result"])
        self.assertTrue(again["result"].startswith("Proposed 1 moment"), again["result"])

    def test_a_moment_far_under_its_length_is_told_of_a_rise_in_the_sound_past_its_end(self):
        """Lane 4 10-31: 082708-6b99 and 092124-25d8 stopped before the laugh at 26:20 and moved their starts."""
        tools.EDGE_RECIPES, tools.SHORT_RECIPES = self.edge_recipes, self.short_recipes
        job = mock.Mock()

        def told(loudest):
            job.chat.clip, job.chat.events = {"recipe": "vlog", "length_s": 60}, []
            return tools._seen_past(job, "why", {"seconds": 60.0}, 1, {}, {"n": 1, "start": 0.0, "end": 60.0,
                                                                          "loudest": loudest}, 10.0, 30.0)

        self.assertIn("The sound rises most at 0:34, after this end: if that is a laugh", told(34.0))
        self.assertTrue(told(34.0).startswith("Not proposed yet: moment 1 runs 20 s of the 60 s"))
        for loudest in (33.0, 20.0, None):
            self.assertNotIn("rises most", told(loudest), "3 s on is the end's own, and inside it is the moment's")

    def test_a_vlog_ending_before_the_rise_in_its_sound_is_told_once_with_its_start_even_with_its_look_done(self):
        """Lane 4 10-31: 101550-5bd0 ran 25:14 to 26:02, 48 s of 40, past far-short, on a look the edge step had opened."""
        tools.EDGE_RECIPES, tools.SHORT_RECIPES = self.edge_recipes, self.short_recipes
        job = mock.Mock()
        job.chat.clip, job.chat.events = {"recipe": "vlog", "length_s": 30}, []
        job.emit = lambda kind: job.chat.events.append({"kind": kind, "job": job.n})
        look = {"n": 1, "start": 0.0, "end": 60.0, "seen_past": True, "loudest": 34.0}
        said = tools._seen_past(job, "why", {"seconds": 60.0}, 1, {}, look, 10.0, 30.0)
        self.assertTrue(said.startswith("Not proposed yet: moment 1 ends at 0:30, and the sound rises most at 0:34, after "
                                        "this end."), said)
        self.assertIn("move the start on so the clip stays near its 30 s", said)
        self.assertIsNone(tools._seen_past(job, "why", {"seconds": 60.0}, 1, {}, look, 10.0, 30.0), "the second stands")

    def test_a_job_hears_of_the_rise_once_and_a_talk_does_not_on_its_own(self):
        tools.EDGE_RECIPES, tools.SHORT_RECIPES = self.edge_recipes, self.short_recipes
        job = mock.Mock()
        job.emit = lambda kind: job.chat.events.append({"kind": kind, "job": job.n})
        look = {"n": 1, "start": 0.0, "end": 60.0, "seen_past": True, "loudest": 34.0}

        def told(recipe, length, a=10.0):
            job.chat.clip = {"recipe": recipe, "length_s": length}
            return tools._seen_past(job, "why", {"seconds": 60.0}, 1, {}, look, a, 30.0)

        job.chat.events = []
        self.assertIn("rises most at 0:34", told("vlog", 60), "far-short says it")
        self.assertIsNone(told("vlog", 30, 5.0), "the job has heard it")
        job.chat.events = []
        self.assertIsNone(told("talk", 30))
        job.chat.events = []
        look["loudest"] = 33.0
        self.assertIsNone(told("vlog", 30, 5.0), "3 s on is the end's own")

    def test_only_a_talk_or_a_vlog_far_under_its_length_is_told_and_once_a_look_is_done_it_still_is(self):
        tools.EDGE_RECIPES, tools.SHORT_RECIPES = self.edge_recipes, self.short_recipes
        job = mock.Mock()
        job.chat.events = []
        look = {"n": 1, "start": 0.0, "end": 60.0, "seen_past": True}

        def told(recipe, seconds, length=60):
            job.chat.clip = {"recipe": recipe, "length_s": length}
            job.chat.events = []
            return tools._seen_past(job, "why", {"seconds": 60.0}, 1, {}, look, 10.0, 10.0 + seconds)

        self.assertIn("runs 35 s of the 60 s", told("talk", 35))
        self.assertIn("runs 19 s of the 40 s", told("vlog", 19, 40))
        self.assertIsNone(told("talk", 37), "at 60% of its length it is left alone")
        self.assertIsNone(told("explain", 12))

    def test_a_moment_at_its_looks_end_is_seen_past_it_once_and_the_second_answer_stands(self):
        """Lane 4 10-30: 94 of 127 moments ended within 3 s of the look's end."""
        tools.EDGE_RECIPES = self.edge_recipes
        first = [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the site", "start": "0:34",
                                                                   "end": "0:49"}])]
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="vlog", why="an episode")],
                                self.to_moment()[1], first, first, "Done.")
        looked, again = steps[3], steps[4]
        self.assertFalse(looked["refused"], looked["result"])
        self.assertTrue(looked["result"].startswith("Not proposed yet: moment 1 sits at the edge of look 1 (0:50)"),
                        looked["result"])
        self.assertIn("Look 2: 0:30-1:00", looked["result"], "on to the recording's end, 60 s here")
        self.assertEqual([(s["start"], s["end"]) for s in self.chat.looks], [(30.0, 50.0), (30.0, 60.0)])
        self.assertFalse(again["refused"], again["result"])
        self.assertTrue(again["result"].startswith("Proposed 1 moment"), again["result"])

    def test_a_moment_at_its_looks_end_is_seen_30_s_on(self):
        """Lane 5 10-31: EP 22's joke on 30 s started at 25:52 and ran under 40 s, 3 of 3."""
        tools.EDGE_RECIPES = self.edge_recipes
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="vlog", why="an episode")],
                                [call("look_at_piece", start="0:05", end="0:25", why="the top stretch")],
                                [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the site",
                                                                                   "start": "0:09", "end": "0:24"}])],
                                "Done.")
        self.assertIn("Look 2: 0:05-0:55", steps[3]["result"])

    def test_a_moment_at_its_looks_start_is_seen_before_it(self):
        tools.EDGE_RECIPES = self.edge_recipes
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="talk", why="two people")],
                                self.to_moment()[1],
                                [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the site",
                                                                                   "start": "0:31", "end": "0:44"}])],
                                [call("propose_moments", why="from the start", moments=[
                                    {"look": 2, "reason": "the site", "start": "0:06", "end": "0:40"}])], "Done.")
        self.assertTrue(steps[3]["result"].startswith("Not proposed yet: moment 1 sits at the edge of look 1 (0:30)"),
                        steps[3]["result"])
        self.assertIn("Look 2: 0:05-0:50", steps[3]["result"], "25 s before it, the end stays")
        self.assertTrue(steps[4]["result"].startswith("Proposed 1 moment"), "the wider look is not looked past again")

    def test_a_moment_inside_its_look_is_read_from_a_strangers_seat_once_and_the_second_answer_stands(self):
        """10-30: Glimmer opened EP 0's story at 18:21 3 of 3, the story under way since 17:48."""
        tools.EDGE_RECIPES = self.edge_recipes
        self.worker.phrases = [{"from": 1.0, "to": 3.0, "text": "ตอนเด็กพี่ชอบแกล้งเรื่องผี"},
                               {"from": 3.1, "to": 3.8, "text": "แล้วคืนนั้นล่ะ"},
                               {"from": 4.0, "to": 19.0, "text": "เขาก็เดินไปปิดไฟ"}]
        moment = [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the prank", "start": "0:34",
                                                                    "end": "0:46"}])]
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="talk", why="two people")],
                                self.to_moment()[1], moment, moment,
                                json.dumps({"lines": ["เขาก็เดินไปปิดไฟ"]}, ensure_ascii=False), "Done.")
        read, again = steps[3], steps[4]
        self.assertFalse(read["refused"], read["result"])
        self.assertTrue(read["result"].startswith("Not proposed yet: a stranger who never saw this recording starts "
                                                  "moment 1 at 0:34, on "), read["result"])
        self.assertIn("เขาก็เดินไปปิดไฟ", read["result"].split("has not")[0], "the line the moment opens on")
        lines = read["result"].split("has not")[1]
        for said in ("ตอนเด็กพี่ชอบแกล้งเรื่องผี", "แล้วคืนนั้นล่ะ"):
            self.assertIn(said, lines, "what is said before the start")
        self.assertNotIn("เขาก็เดินไปปิดไฟ", lines, "only what the stranger missed")
        self.assertEqual(len(self.chat.looks), 1, "these words were seen: no new look")
        self.assertTrue(again["result"].startswith("Proposed 1 moment"), again["result"])

    def test_a_vlog_moment_is_not_read_from_a_strangers_seat(self):
        """10-31: it ran EP 22's joke back to 25:19 and 25:26, 2 of 2, past where it starts at 25:43 (lane 4)."""
        tools.EDGE_RECIPES = self.edge_recipes
        self.worker.phrases = [{"from": 1.0, "to": 3.0, "text": "ตอนเด็กพี่ชอบแกล้งเรื่องผี"},
                               {"from": 4.0, "to": 19.0, "text": "เขาก็เดินไปปิดไฟ"}]
        moment = [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the joke", "start": "0:34",
                                                                    "end": "0:46"}])]
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="vlog", why="an episode")],
                                self.to_moment()[1], moment,
                                json.dumps({"lines": ["เขาก็เดินไปปิดไฟ"]}, ensure_ascii=False), "Done.")
        self.assertTrue(steps[3]["result"].startswith("Proposed 1 moment"), steps[3]["result"])

    def test_a_stranger_is_shown_a_long_line_that_began_before_the_reach_and_nothing_older(self):
        """EP 0: the one line before 18:21 ran 27 s from 17:54, on a pause-cut hearing, and was missed at first."""
        tools.EDGE_RECIPES = self.edge_recipes
        look = {"said": [{"from": 10.0, "to": 12.0, "text": "too old"}, {"from": 20.0, "to": 49.0, "text": "the long one"},
                         {"from": 49.2, "to": 52.0, "text": "the first"}]}
        cold = tools._read_cold(look, 1, {"start": "0:49"}, 49.0)
        self.assertIn("the long one", cold)
        self.assertNotIn("too old", cold)
        self.assertTrue(look["seen_past"])
        for bare in ({}, {"said": [{"from": 49.2, "to": 52.0, "text": "the first"}]}):
            self.assertIsNone(tools._read_cold(bare, 1, {"start": "0:49"}, 49.0), "nothing was said before it")
        self.assertIsNone(tools._read_cold(dict(look), 1, {}, 49.0), "Muse gave no start: the whole look is the moment")

    def test_a_moment_clear_of_the_edges_a_game_and_the_recordings_own_end_are_proposed_at_once(self):
        tools.EDGE_RECIPES = self.edge_recipes
        clear = ("vlog", "0:30", "0:50", {"start": "0:34", "end": "0:45"})
        game = ("game", "0:10", "0:50", {"start": "0:12", "end": "0:49"})
        last = ("vlog", "0:40", "1:00", {"start": "0:44", "end": "0:59"})
        whole = ("vlog", "0:30", "0:50", {})
        for kind, a, b, ends in (clear, game, last, whole):
            self.chat.recording, self.chat.looks[:] = None, []
            _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type=kind, why="the kind")],
                                    [call("look_at_piece", start=a, end=b, why="the stretch")],
                                    [call("propose_moments", why="the best",
                                          moments=[dict(ends, look=1, reason="the site")])], "Done.")
            self.assertTrue(steps[-1]["result"].startswith("Proposed 1 moment"), (kind, steps[-1]["result"]))
        self.chat.recording, self.chat.looks[:] = None, []
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="vlog", why="the kind")],
                                [call("look_at_piece", start="0:10", end="0:50", why="the stretch")],
                                [call("propose_moments", why="the two best", moments=[
                                    {"look": 1, "reason": "first", "start": "0:15", "end": "0:30"},
                                    {"look": 1, "reason": "second", "start": "0:32", "end": "0:49"}])], "Done.")
        self.assertTrue(steps[-1]["result"].startswith("Proposed 2 moments"), "only the one made next is looked past")

    def test_a_game_moment_under_its_length_is_seen_back_from_once_and_its_end_stays(self):
        """Lane 4 10-31: Sifu's 35:52-36:11 shipped as 13 s of 30, its look opening a second after the lead-up."""
        tools.EDGE_RECIPES = self.edge_recipes
        short = [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the death", "start": "0:38",
                                                                   "end": "0:49"}])]
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="game", why="a stream")],
                                self.to_moment()[1], short, short, "Done.")
        looked, again = steps[3], steps[4]
        self.assertFalse(looked["refused"], looked["result"])
        self.assertTrue(looked["result"].startswith("Not proposed yet: moment 1 runs 11 s of the 30 s this clip can hold"),
                        looked["result"])
        self.assertIn("Look 2: 0:05-0:50", looked["result"], "25 s before it, the end stays on the peak")
        self.assertTrue(again["result"].startswith("Proposed 1 moment"), again["result"])

    def test_a_game_is_seen_back_from_once_a_job(self):
        """10-31 024736-221a: sent back twice, the clip left the deaths for a stretch with none."""
        tools.EDGE_RECIPES = self.edge_recipes
        short = [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the death", "start": "0:38",
                                                                   "end": "0:49"}])]
        newer = [call("propose_moments", why="the play before", moments=[{"look": 3, "reason": "the play", "start": "0:06",
                                                                          "end": "0:18"}])]
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="game", why="a stream")],
                                self.to_moment()[1], short, [call("look_at_piece", start="0:05", end="0:20", why="before")],
                                newer, "Done.")
        self.assertTrue(steps[3]["result"].startswith("Not proposed yet: moment 1 runs 11 s"), steps[3]["result"])
        self.assertTrue(steps[5]["result"].startswith("Proposed 1 moment"), steps[5]["result"])

    def test_a_game_moment_that_is_its_whole_look_is_seen_back_from_once(self):
        """Lane 4 10-31: 35:43-36:13, the look itself, in 8 of 13 Glimmer runs, past the death at 35:34 unseen."""
        tools.EDGE_RECIPES = self.edge_recipes
        whole = [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the fight", "start": "0:30",
                                                                   "end": "0:50"}])]
        with mock.patch.object(tools, "GAME_SHORT", 0.1):
            _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="game", why="a stream")],
                                    self.to_moment()[1], whole, "Done.")
        told = steps[3]["result"]
        self.assertTrue(told.startswith("Not proposed yet: moment 1 runs 20 s of the 30 s"), told)
        self.assertIn("A look's end is where the look stopped, not the peak: end just after the reaction.", told)
        self.assertIn("Look 2: 0:05-0:50", told, "25 s back")

    def test_a_game_moment_inside_its_look_is_not_seen_back_from_for_being_whole(self):
        tools.EDGE_RECIPES = self.edge_recipes
        job = mock.Mock()
        job.chat.clip, job.chat.events = {"recipe": "game", "length_s": 30}, []
        look = {"n": 1, "start": 0.0, "end": 60.0}

        def told(a, b):
            job.chat.events = []
            return tools._seen_past(job, "why", {"seconds": 60.0}, 1, {"start": "x", "end": "y"}, look, a, b)

        self.assertIsNone(told(10.0, 37.0))
        self.assertIsNone(told(2.0, 60.0), "from the look's start only")
        self.assertIsNone(told(0.0, 57.0), "to the look's end only")

    def test_a_game_moment_near_its_length_is_not_seen_back_from(self):
        tools.EDGE_RECIPES = self.edge_recipes
        job = mock.Mock()
        job.chat.clip, job.chat.events = {"recipe": "game", "length_s": 30}, []
        look = {"n": 1, "start": 0.0, "end": 60.0}
        self.assertIsNone(tools._seen_past(job, "why", {"seconds": 60.0}, 1, {"start": "0:10", "end": "0:37"}, look, 10.0, 37.0))

    def test_a_game_moment_of_its_length_is_not_read_again(self):
        tools.EDGE_RECIPES = self.edge_recipes
        job = mock.Mock()
        job.chat.clip = {"recipe": "game", "length_s": 30}
        look = {"n": 1, "start": 0.0, "end": 60.0, "said": [{"from": 1.0, "to": 3.0, "text": "before"},
                                                            {"from": 4.0, "to": 9.0, "text": "its first line"}]}
        moment = {"start": "0:04", "end": "0:50"}
        self.assertIsNone(tools._seen_past(job, "why", {"seconds": 60.0}, 1, moment, look, 4.0, 50.0))
        self.assertNotIn("seen_past", look)

    def test_a_seller_clip_is_not_cut_in_two_unless_the_person_asked(self):
        _, steps = self.run_job(*self.to_moment(), [call("load_recipe", type="seller", why="a product")],
                                [call("make_clip", moment=1, layout="split", why="the face and the product")],
                                [call("make_clip", moment=1, layout="card", why="the person said: a card")], "Done.")
        split, card = steps[-2:]
        self.assertTrue(split["refused"], "Tarm 10-30: the ad split the screen as she raised the tumbler")
        self.assertTrue(split["result"].startswith("Not made. A seller's clip keeps the product whole in the frame: "
                                                   "the subject layout fills the frame around it"), split["result"])
        self.assertIn("Use split only when the person asked for their face alone, a screen or the frame in two",
                      split["result"])
        self.assertFalse(card["refused"], card["result"])

    def test_a_talk_cuts_between_faces_and_circles_are_for_when_the_person_asks(self):
        _, steps = self.run_job(self.to_moment()[0], [call("load_recipe", type="talk", why="three friends talking")], *self.to_moment()[1:],
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
                                    [call("make_clip", moment=1, layout="subject", why="the product")], "Done.")
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
                [call("add_title", words="Boss down", start=0, end=4, place="top", why="the hook")],
                [call("render_clip", why="make the file")],
                [call("hand_over", title="t", caption="c", hashtags=["#a"], why="unchecked")],
                [call("check_clip", why="check it")],
                [call("hand_over", title="t", caption="c", hashtags=["#a"], why="failed")],
                [call("blur_box", box=[400, 0, 600, 300], why="the chat")],
                [call("render_clip", why="again")],
                [call("check_clip", why="again")],
                [call("hand_over", title="t", caption="clips from H:/Phone/yeti", hashtags=["#a"], why="where")],
                [call("hand_over", title="t", caption="c", hashtags=["#reels", "fyp", "TikTokThailand"], why="tags")],
                [call("hand_over", title="ของดี ซื้oมาขายไป", caption="c", hashtags=["#reels"], why="slip")],
                [call("hand_over", title="t", caption="c", hashtags=["#reels", "#instagram", "#facebook"], why="post")],
                "Done.")
        finally:
            render.export = had
        unchecked, first, failed, _, _, second, where, other, slipped, ready = steps[-10:]
        self.assertTrue(slipped["refused"], "Glimmer 10-30: a Latin o in a Thai word of the post's title")
        self.assertIn('"ซื้oมาขายไป" has a Latin letter inside a Thai word', slipped["result"])
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
        self.assertEqual([c["ok"] for c in shown["checks"]], [None, None, True, None, True, None, True])
        self.assertEqual(agent.Chat.load(self.chat.id).checks, self.chat.checks)

    GAME_MOMENT = [[call("load_recipe", type="game", why="a stream")],
                   [call("make_clip", moment=1, layout="subject", why="the action")],
                   [call("decide_people", keep=[1], why="it is him")]]

    def test_a_blur_over_most_of_the_picture_for_the_whole_clip_is_refused(self):
        """Lane 5's 022940-2e20: a console match, so no chat box, and the check said one must be blurred."""
        self.chat.emit("person", job=None, text="ทำให้เบลอทั้งจอเลย")
        with mock.patch.object(tools, "_share", lambda *a: 0):
            _, steps = self.run_job(*self.to_moment(), *self.GAME_MOMENT,
                                    [call("blur_box", box=[0, 0, 1000, 1000], why="the check")],
                                    [call("blur_box", box=[0, 0, 1000, 1000], start=1, end=3, why="a page for a moment")],
                                    [call("blur_box", box=[0, 0, 1000, 1000], why="asked", person_said="เบลอทั้งจอ")],
                                    "Done.")
        whole, moment, asked = [s for s in steps if s["tool"] == "blur_box"]
        self.assertTrue(whole["refused"])
        self.assertIn("That blur covers most of the picture for the whole clip", whole["result"])
        self.assertIn("A recording with no chat box needs no blur: say so with no_chat_box.", whole["result"])
        self.assertFalse(moment["refused"], "the whole picture for a moment is a blur that leaves the clip to watch")
        self.assertFalse(asked["refused"], "a person may ask for it in their own words: " + asked["result"])

    def test_half_the_picture_may_be_blurred(self):
        self.assertFalse(tools._blurs_the_picture([0, 0, 500, 1000], {"width": 1000, "height": 1000}))
        self.assertTrue(tools._blurs_the_picture([0, 0, 501, 1000], {"width": 1000, "height": 1000}))

    def test_a_blur_over_the_whole_picture_makes_no_clips(self):
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("make_clips", clips=clips, blur=[[0, 0, 1000, 1000]], why="both")], "Done.",
                                recipe="game")
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("Nothing was made. That blur covers most of the picture", steps[-1]["result"])

    def test_a_game_with_no_chat_box_says_so_and_its_clips_are_made_and_pass(self):
        clips = [{"moment": 1, "title": "Eaglet one", "caption": "c1", "hashtags": ["#a"]},
                 {"moment": 2, "title": "Eaglet two", "caption": "c2", "hashtags": ["#b"]}]
        _, steps = self.several([call("no_chat_box", why="a console game, no stream")],
                                [call("make_clips", clips=clips, why="both")], "Done.", recipe="game")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        ready = [e for e in self.chat.events if e["kind"] == "ready"]
        self.assertEqual(len(ready), 2)
        for e in ready:
            self.assertIn(("No viewer's name can be read", True, "Muse"), [(c["name"], c["ok"], c.get("who"))
                                                                            for c in e["checks"]])
        back = agent.Chat.load(self.chat.id)
        self.assertTrue(tools._no_chat(types.SimpleNamespace(chat=back)), "a chat read back still knows")

    def test_saying_there_is_no_chat_box_does_not_hide_a_name_on_the_screen(self):
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        chat = {"text": "@viewer_one ผมว่าดีครับ", "box": [130, 10, 60, 12], "sure": 0.95}
        had = screen.readings, getattr(render, "export", None)
        screen.readings = lambda rec, a, b, on_start=None: [{"at": a + 1, "lines": [chat]}, {"at": b - 1, "lines": [chat]}]
        render.export = export
        try:
            _, steps = self.run_job(*self.to_moment(), *self.GAME_MOMENT, [call("no_chat_box", why="none seen")],
                                    [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                                    "Done.")
        finally:
            screen.readings, render.export = had
        self.assertIn("- FAILED: No viewer's name can be read (names can be read at", steps[-1]["result"])

    def test_no_chat_box_is_for_a_game_after_its_moments_and_is_forgotten_with_the_recording(self):
        offered = lambda recipe, moments: "no_chat_box" in [s["function"]["name"] for s in agent.Job.offered(
            types.SimpleNamespace(finishing=0, chat=types.SimpleNamespace(events=[], album=[], moments=moments,
                                                                          clip={"recipe": recipe})))]
        self.assertTrue(offered("game", [{}]))
        self.assertFalse(offered("game", None), "a tool in every request moved Glimmer's moment")
        self.assertFalse(offered("vlog", [{}]))
        job = types.SimpleNamespace(chat=types.SimpleNamespace(events=[{"kind": "no_chat"}, {"kind": "opened"}]))
        self.assertFalse(tools._no_chat(job))
        job.chat.events.append({"kind": "no_chat"})
        self.assertTrue(tools._no_chat(job))
        self.assertFalse(tools._no_chat(types.SimpleNamespace(chat=types.SimpleNamespace(events=[]))))

    def test_only_a_game_is_checked_for_a_chat_box(self):
        _, steps = self.run_job(*self.to_moment(), [call("no_chat_box", why="none")], "Done.")
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("Only a game stream is checked for a chat box.", steps[-1]["result"])

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
                [call("blur_box", box=[380, 0, 620, 80], why="the names")],
                [call("render_clip", why="again")], [call("check_clip", why="again")], "Done.")
        finally:
            screen.readings, render.export = had
        on_face, low, names = [s for s in steps if s["tool"] == "blur_box"]
        first, second = [s for s in steps if s["tool"] == "check_clip"]
        self.assertTrue(on_face["refused"])
        self.assertIn("The viewers' names were read on the screen at [394, 33, 606, 144] on your grid.",
                      on_face["result"], "game afc1 10-28: Muse moved the box down four times, away from the chat")
        self.assertFalse(low["refused"], low["result"])
        self.assertIn("- FAILED: No viewer's name can be read (names can be read at [394, 33, 606, 144] on your grid "
                      "from 0 to ", first["result"], "a blur over the whole clip in the wrong place passed before")
        self.assertIn("- passed: No viewer's name can be read (1 line with a name, under a blur for as long as it "
                      "shows)", second["result"])
        self.assertIn("The box reached into 1 line with a viewer's name, so it was grown to cover it whole.",
                      names["result"], "game 10-30: three blurs, three renders, for the bottom of the names")
        self.assertNotIn("grown to cover", low["result"])

    def test_a_clip_never_rendered_is_rendered_and_handed_over_out_of_steps(self):
        """Lane 5's 051633-f8a0: effects up to step 30 on a clip that was never rendered, and no file for the person."""
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        made = [*self.to_moment(), *self.GAME_MOMENT]
        try:
            with mock.patch.object(agent, "MAX_STEPS", len(made)):
                _, steps = self.run_job(*made, [call("render_clip", why="make the file")],
                                        [call("check_clip", why="check it")],
                                        [call("hand_over", title="a good one", caption="short", hashtags=["#a"],
                                              why="give it")], "Done.")
        finally:
            render.export = had
        self.assertEqual([s["tool"] for s in steps[-3:]], ["render_clip", "check_clip", "hand_over"])
        self.assertEqual([s["refused"] for s in steps[-3:]], [False, False, False], steps[-1]["result"])
        self.assertEqual(len([e for e in self.chat.events if e["kind"] == "ready"]), 1)

    def test_a_clip_out_of_steps_goes_over_with_the_check_that_failed_said_on_the_page(self):
        """Lane 5's 022940-2e20 reached step 30 on a failed check, with a rendered file and nothing handed over."""
        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        chat = {"text": "@viewer_one ผมว่าดีครับ", "box": [130, 10, 60, 12], "sure": 0.95}
        had = screen.readings, getattr(render, "export", None)
        screen.readings = lambda rec, a, b, on_start=None: [{"at": a + 1, "lines": [chat]}, {"at": b - 1, "lines": [chat]}]
        render.export = export
        made = [*self.to_moment(), *self.GAME_MOMENT, [call("render_clip", why="make the file")]]
        give = call("hand_over", title="a good one", caption="short", hashtags=["#a"], why="give it")
        try:
            with mock.patch.object(agent, "MAX_STEPS", len(made)):
                _, steps = self.run_job(*made, [call("check_clip", why="check it")], [give], "Done.")
        finally:
            screen.readings, render.export = had
        self.assertIn("- FAILED: No viewer's name can be read", steps[-2]["result"])
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertIn("these checks still fail on it: say so: No viewer's name can be read.", steps[-1]["result"])
        ready = [e for e in self.chat.events if e["kind"] == "ready"]
        self.assertEqual(len(ready), 1)
        self.assertIn("Muse ran out of steps, so this clip was handed over with a check that failed: No viewer's name "
                      "can be read.", [e["text"] for e in self.chat.events if e["kind"] == "note"])

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
        self.assertIn("0:35-0:38 bxo noqa mumble (not sure)", looked["result"])
        self.assertIn("0:32-0:34 this is the moment\n", looked["result"])
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

    def test_a_talk_moment_the_hearing_was_sure_of_no_line_in_is_refused_once_with_where_the_clear_lines_are(self):
        self.worker.phrases = [{"from": 1.0, "to": 4.0, "text": "bxo noqa mumble", "sure": 0.5},
                               {"from": 4.5, "to": 8.0, "text": "haha qrx all at once", "sure": 0.55},
                               {"from": 14.0, "to": 17.0, "text": "this is the moment", "sure": 0.95}]
        self.chat.clip = {"recipe": "talk", "title": "Podcast", "length_s": 30, "style": "clean", "language": "",
                          "brief": "", "subtitles": None}
        loud = [call("propose_moments", why="the laugh", moments=[{"look": 1, "reason": "they laugh", "start": "0:30",
                                                                    "end": "0:40"}])]
        fixed = '{"lines": ["bxo noqa mumble", "haha qrx all at once"]}'
        _, steps = self.run_job(*self.to_moment()[:2], loud, fixed, loud, fixed, "Done.")
        tried = [s for s in steps if s["tool"] == "propose_moments"]
        self.assertTrue(tried[0]["refused"], tried[0]["result"])
        self.assertIn("The hearing was sure of none of the 2 lines said in moment 1 (0:30-0:40)", tried[0]["result"])
        self.assertIn("lines were heard clearly in look 1 at 0:44.", tried[0]["result"])
        self.assertFalse(tried[1]["refused"], "the second time it goes on: " + tried[1]["result"])
        fresh = types.SimpleNamespace(chat=self.chat, warned=set())
        mumbled = {"said": [{"from": 31.0, "to": 34.0, "text": "x", "sure": 0.1}], "start": 30.0, "end": 40.0}
        self.assertIsNotNone(tools._unclear(fresh, mumbled, self.chat.looks), "a new job is refused once again")
        fresh.warned.clear()
        clear = dict(mumbled, said=mumbled["said"] + [{"from": 35.0, "to": 37.0, "text": "y", "sure": 0.95}])
        self.assertIsNone(tools._unclear(fresh, clear, self.chat.looks), "one line heard clearly is enough")
        self.chat.clip["recipe"] = "game"
        self.assertIsNone(tools._unclear(fresh, mumbled, self.chat.looks), "in a game the play is the clip, not words")

    def test_a_long_line_unsure_only_for_a_word_no_dictionary_has_keeps_its_other_lines_on_the_video(self):
        self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")], "Done.")
        start = self.chat.project.state()["clip"]["pieces"][0][0]
        line = "ไจโรสโคปเราสามารถที่จะทำมือแบบนี้เพื่อที่จะเล็งปืนได้ครับ"
        self.chat.moments[0]["said"] = [
            {"from": start + 1, "to": start + 9, "text": line, "sure": 0.862, "heard": line},
            {"from": start + 10, "to": start + 11, "text": "Resident", "sure": 0.597, "heard": "Resident"},
            {"from": start + 12, "to": start + 14, "text": "เดินได้นะครับ", "sure": 0.64, "heard": "เดินได้นะครับ"}]
        _, steps = self.run_job([call("add_captions", why="say what he says")], "Done.")
        self.assertFalse(steps[0]["refused"], steps[0]["result"])
        caps = self.chat.project.state()["captions"]
        shown = "".join(x["text"] for x in caps["lines"])
        drafted = "".join(x["text"] for x in caps["drafts"])
        self.assertIn("สามารถที่จะทำมือ", shown, "10-30 seller: 13 s of a line read right went to the drafts")
        self.assertIn("เล็งปืน", shown)
        self.assertIn("ไจโรสโคป", drafted)
        self.assertNotIn("ไจโรสโคป", shown)
        self.assertIn("Resident", drafted, "a line heard unsure is still a draft whole")
        self.assertNotIn("Resident", shown)
        self.assertIn("เดินได้นะครับ", drafted, "under READ_SURE the words are guesses, dictionary words or not")
        self.assertNotIn("เดินได้", shown)

    def test_a_phrase_heard_from_the_very_end_of_a_piece_is_left_out_and_the_others_are_captioned(self):
        self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")], "Done.")
        (start, end), length = self.chat.project.state()["clip"]["pieces"][0], self.chat.project.state()["length"]
        self.assertLessEqual(start + 3, end)
        self.assertGreater(length, 3)
        self.chat.moments[0]["said"] = [{"from": start + 1, "to": start + 3, "text": "watch him fall", "sure": 0.9},
                                        {"from": end, "to": end + 0.5, "text": "and then", "sure": 0.9},
                                        {"from": end - 0.02, "to": end + 0.5, "text": "so close", "sure": 0.9}]
        _, steps = self.run_job([call("add_captions", why="say what he says")], "Done.")
        self.assertFalse(steps[0]["refused"], "10-30 seller: 'add_captions broke (phrase 3's to 28.69 is not after "
                                              "its from 28.69 ...)', and the clip went out with no captions")
        shown = " ".join(line["text"] for line in self.chat.project.state()["captions"]["lines"])
        self.assertIn("fall", shown)
        self.assertNotIn("then", shown)
        self.assertNotIn("close", shown, "0.02 s of it plays: less than a frame")
        self.assertEqual(tools._in_clip([[6.27, 34.96], [35.68, 37.0]], {"from": 34.9, "to": 35.3}, 30.01),
                         (28.63, 28.69), "two frames of it play, so it stays")

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
        self.assertEqual(tools._not_said("จอย PlayStation 5 ราคา 3,000 บาท", ["ราคา 3,000 บาท"]), [],
                         "a number in a product's name is no price")
        for name in ("Galaxy S24", "PS5", "A55", "iPhone 16", "iPad 11"):
            self.assertEqual(tools._not_said(name, []), [], name)
        for claim, numbers in (("Samsung 3,500 บาท", ["3,500"]), ("Redmi 200", ["200"]), ("Redmi 90 บาท", ["90"]),
                               ("iPad 90 บาท", ["90"]), ("iPad 90 Baht", ["90"]), ("iPhone 50%", ["50"]),
                               ("iPad 5000", ["5000"]), ("iPad 5,000", ["5,000"]),
                               ("Flash Sale 50%", ["50"]), ("Only 5 left", ["5"]), ("Just 99", ["99"]),
                               ("Buy 1 Get 1", ["1", "1"]), ("Top 3 picks", ["3"]), ("2 for 90", ["2", "90"]),
                               ("Pixel 9", ["9"]), ("Note 20", ["20"]), ("buy2", ["2"]), ("SALE50", ["50"])):
            self.assertEqual(tools._not_said(claim, []), numbers, claim)
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
                [call("make_clip", moment=1, layout="subject", product=[300, 760, 700, 960], why="the product")],
                [call("add_title", words="75 m²", start=0, end=4, place="middle", why="the size")],
                [call("decide_people", keep=[1], why="it is the seller")],
                [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                [call("hand_over", title="75 m²", caption="七十五平方米", hashtags=["#home"], why="post")],
                "Done.")
        finally:
            render.export = had
        checked, handed = steps[-2], steps[-1]
        self.assertFalse(checked["refused"], checked["result"])
        self.assertIn("- for you to check: Numbers on screen were said (the clip is said in Chinese, and the studio "
                      "reads number words only in Thai and English", checked["result"])
        self.assertFalse(handed["refused"], handed["result"])

    def seller_finished(self, *made):
        """A seller's recording to a moment, then the given calls with the render faked to a loud file."""
        self.worker.phrases = [{"from": 2, "to": 4.5, "text": "a fine pad", "sure": 0.99}]

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            return out

        had = getattr(render, "export", None)
        render.export = export
        script = self.to_moment()
        try:
            return self.run_job(script[0], [call("load_recipe", type="seller", why="a seller")], *script[1:],
                                '{"lines": ["a fine pad"]}', *made)
        finally:
            render.export = had

    def test_a_seller_clip_with_no_product_named_fails_the_check_until_one_is(self):
        """Lane 5, 10-30: "handed over" must mean the product is seen. The check used to skip a clip with no box."""
        made = ([call("decide_people", keep=[1], why="it is the seller")], [call("render_clip", why="make the file")],
                [call("check_clip", why="check it")])
        _, steps = self.seller_finished(
            [call("make_clip", moment=1, layout="subject", why="the product")], *made,
            [call("hand_over", title="A pad", caption="a fine pad", hashtags=["#pad"], why="post")],
            [call("make_clip", moment=1, layout="subject", product=[200, 200, 800, 800], why="the product")], *made,
            "Done.")
        first, checked, refused, again, passed = steps[-9], steps[-6], steps[-5], steps[-4], steps[-1]
        self.assertIn("This is a seller clip and no product box was named, so nothing keeps words off the product.",
                      first["result"])
        self.assertIn("- FAILED: The product is in the frame (no product box was named and the frame follows nothing",
                      checked["result"])
        self.assertTrue(refused["refused"])
        self.assertIn("These checks failed on the finished file: The product is in the frame.", refused["result"])
        self.assertFalse(again["refused"], again["result"])
        self.assertIn("- passed: The product is in the frame (the product box is inside the clip's picture)",
                      passed["result"])

    def test_a_seller_clip_showing_under_half_of_its_product_fails_the_check(self):
        """Glimmer's balm 10-31 passed on a box that only touched the frame."""
        with mock.patch.object(layouts, "shown_share", return_value=0.4):
            _, steps = self.seller_finished(
                [call("make_clip", moment=1, layout="subject", product=[200, 200, 800, 800], why="the product")],
                [call("decide_people", keep=[1], why="it is the seller")], [call("render_clip", why="make the file")],
                [call("check_clip", why="check it")], "Done.")
        self.assertIn("- FAILED: The product is in the frame (under half of the product box is inside the clip's "
                      "picture", steps[-1]["result"])

    def test_a_seller_clip_in_a_layout_that_cuts_between_faces_is_not_failed_for_its_product(self):
        """The person asked for it, and a layout that cuts between faces takes no product box."""
        from unittest import mock
        self.seller_finished([call("make_clip", moment=1, layout="subject", why="the product")],
                             [call("decide_people", keep=[1], why="it is the seller")],
                             [call("render_clip", why="make the file")], "Done.")
        real = tools._clip_state

        def split(job):
            state = real(job)
            return dict(state, clip=dict(state["clip"], layout=dict(state["clip"]["layout"], kind="split")))

        with mock.patch.object(tools, "_clip_state", split):
            said = tools.check_clip(self.chat.job, "the person asked for it")
        self.assertIn("- not needed: The product is in the frame (the person asked for a layout that cuts between "
                      "faces)", said)

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

    def test_a_face_box_with_no_face_in_it_draws_no_circle(self):
        """183706-b632: a box on a slide, no face in 146 pictures, and the clip went out with an empty white circle."""
        found = {"samples": [{"t": float(t), "faces": [[700, 700, 40, 40, 0.9]]} for t in range(20)]}
        job = type("Job", (), {"find_faces": lambda self, rec, a, b: found})()
        rec = {"width": 1000, "height": 1000}
        box, said = tools._the_face(job, rec, 0.0, 20.0, [50, 50, 100, 100])
        self.assertIsNone(box)
        self.assertEqual(said, "No face was found inside your face box in 20 pictures, so the layout goes without it: a "
                               "circle needs a face in it.")
        self.assertEqual(layouts.make("card", (1920, 1080), box)["notes"][0], "No face was given or found, so the card has no "
                                                                              "circle.")
        self.assertIsNotNone(tools._the_face(job, rec, 0.0, 20.0, [690, 690, 750, 750])[0], "a face in the box is used")


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
