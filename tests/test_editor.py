"""The editor core: layouts, faces, the project list, the drawn frame and Muse's editing tools.

    python -m unittest discover -s tests
"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, effects, faces, focus, hearing, layouts, mind, project, recording, render, screen, talk, text, tools, words  # noqa: E402,E501
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
        self.assertTrue(all(65 <= b[0] and b[0] + b[2] <= 1015 and 270 <= b[1] and b[1] + b[3] <= 1250
                            for b in list(lay["places"].values()) + [lay["captions"]]), "inside the phone's safe zone")

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
        cx, cy = 540 * half, (layouts.CIRCLE_Y + layouts.CIRCLE / 2) * half
        r, g, b = at(shot, cx, cy)
        self.assertTrue(r > 180 and g < 80 and b < 80, "red face camera in the circle: %s" % ((r, g, b),))
        r, g, b = at(shot, cx, (layouts.CIRCLE_Y - layouts.RING / 2) * half)
        self.assertTrue(min(r, g, b) > 200, "white ring: %s" % ((r, g, b),))
        card_y = (layouts.CARD_Y + 300) * half
        row = [at(shot, x, card_y) for x in range(0, 540, 4)]
        self.assertFalse(any(p[0] > 180 and p[1] < 80 for p in row), "the face camera is not in the card again")
        self.assertTrue(any(p[2] > 150 and p[0] < 80 for p in
                            [at(shot, x, y) for x in range(0, 540, 6) for y in range(int(layouts.CARD_Y * half),
                                                                                     int(layouts.CARD_Y * half) + 300,
                                                                                     6)]),
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

    def test_the_clip_is_made_from_a_moment_and_muse_sees_its_frame(self):
        brain, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                                    "Done.")
        step = steps[-1]
        self.assertFalse(step["refused"], step["result"])
        self.assertEqual(self.looked, [(30.0, 50.0)], "the faces are looked for in the moment")
        self.assertIn("Made the clip from moment 1 (0:30-0:50, 20 s) in the subject layout.", step["result"])
        self.assertIn("A face sits at the same place in 30 of 40 pictures, at [438, 667, 562, 889] on your grid; the "
                      "layout uses it.", step["result"])
        self.assertIn("- top (y 285 to 520): clear of faces at 0-4.7 s, 9.8-20 s", step["result"])
        self.assertIn("- middle (y 905 to 1075): clear of faces for the whole clip", step["result"])
        self.assertIn("comes next as a picture", step["result"])
        self.assertIn("Other people's faces in the clip: 1 (the face the layout shows is not one of them).", step["result"])
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

    def test_a_title_over_a_face_is_refused_with_the_clear_times_and_a_clear_one_is_drawn(self):
        brain, steps = self.run_job(
            *self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
            [call("add_title", words="Boss down", start=0, end=15, place="top", why="the hook")],
            [call("add_title", words="Boss down", start=0, end=4.5, place="top", why="before the face rises")],
            "Done.")
        refused, drawn = steps[-2], steps[-1]
        self.assertTrue(refused["refused"])
        self.assertIn("At the top, the title would cover a face at 4.8-9.8 s. Clear: top: 0-4.7 s, 9.8-20 s; middle: "
                      "0-20 s. Pick a time or a place from these.", refused["result"])
        self.assertFalse(drawn["refused"], drawn["result"])
        self.assertIn('The title "Boss down" is on screen at the top from 0 to 4.5 s: 1 line', drawn["result"])
        self.assertIn('words on screen: "Boss down" 0-4.5 s at the top', brain.sent[-1][-2]["content"])

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
        self.assertIn("The card layout has no place called corner; its places are: top, middle.", results[1])
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
                                         r"from y 1090 to 1250")
        self.assertIn("0:02 this is the\n0:03 moment\n", caps["result"])
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
        self.assertIn("Blurred [400, 0, 600, 300] on your grid from 0 to 20 s, so nothing in it can be read. In the "
                      "clip it shows in the frame at x ", blur["result"])
        self.assertEqual(word["result"], 'Added a big word that pops in, such as POW at 5.2 s ("POW").')
        self.assertIn("at 30 s is outside the clip, which is 20 s long", late["result"])
        self.assertIn("A word hit needs its word", wordless["result"])
        self.assertIn("captions on, 1 blur, other people: 0 blurred, 0 kept, 1 still to decide before rendering, hits: "
                      "word at 5.2 s · no finished file yet", brain.sent[-1][-1]["content"])
        state = self.chat.project.state()
        self.assertEqual([round(x["from"], 2) for x in state["captions"]["lines"]][:1], [2.0],
                         "caption times are clip seconds: heard at 32 s of the recording, shown at 2 s of the clip")
        self.assertEqual(state["blurs"][0]["box"], [128, 0, 64, 54])

    def test_words_stay_off_the_product_muse_names_and_a_key_inside_a_word_is_fixed(self):
        brain, steps = self.run_job(
            *self.to_moment(),
            [call("make_clip", moment=1, layout="subject", product=[400, 450, 600, 900], why="the product")],
            [call("add_title", words="Covers 50-75 m2", start=0, end=4, place="middle", why="the spec")],
            [call("add_title", words="Covers 50-75 m2", start=0, end=4, place="top", key="75", why="the spec")],
            "Done.")
        made, middle, top = steps[-3:]
        self.assertIn("kept clear of the faces the program found and the product you named", made["result"])
        self.assertIn("- middle (y 905 to 1075): never clear for 2 s or more; do not use it", made["result"])
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
                [call("hand_over", title="Boss down", caption="He falls.\nTwice.\n#gaming #boss", why="post",
                      hashtags=["gaming", "#gaming", "sifu clip", "#"])],
                "Done.")
        finally:
            render.export = had
        early, undecided, kept, first, again, titled, stale, undone, checked, ready = steps[-10:]
        self.assertFalse(checked["refused"], checked["result"])
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
        shown = [e for e in self.chat.events if e["kind"] == "ready"][-1]
        self.assertEqual((shown["title"], shown["hashtags"]), ("Boss down", ["#gaming", "#sifuclip", "#boss"]))
        self.assertEqual(shown["caption"], "He falls.\nTwice.", "a line of hashtags goes to the hashtags, shown once")
        self.assertTrue(os.path.isfile(os.path.join(render.folder("clips"), shown["file"])))
        self.assertIn("finished file %s, made from the clip as it is now" % shown["file"],
                      brain.sent[-1][-1]["content"])
        self.chat.edit({"do": "style", "name": "pop", "font": None, "colour": None})
        self.assertIn("the last finished file is older than the last change", self.chat.clip_line())
        self.assertEqual(agent.Chat.load(self.chat.id).rendered, self.chat.rendered)

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
            [call("add_glass", box=face, start=1, end=4, tint="dark", why="on the face")],
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
        self.assertIn("- middle (y 905 to 1075): clear of faces for the whole clip", on_face["result"])
        self.assertFalse(asked["refused"], asked["result"])
        self.assertIn("It covers a face at", asked["result"])
        self.assertIn("where the person asked for it; tell the person.", asked["result"])
        self.assertTrue(nowhere["refused"])
        self.assertIn("Give a box, or one of the layout's places for words: top, middle.", nowhere["result"])
        self.assertTrue(under["refused"])
        self.assertIn("On a phone, the name and the caption cover everything below y 1520", under["result"])
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
        self.assertIn("On a phone, the name and the caption cover everything below y 1520", low["result"])
        state = self.chat.project.state()
        self.assertEqual([(k["name"], k["at"]) for k in state["stickers"]], [("fire", 1), ("star", 2.5)])
        self.assertEqual(state["stickers"][1]["png"], effects.sticker_png("star"))
        self.assertEqual([(g["box"], g["start"], g["end"], g["tint"], g["shine"]) for g in state["glass"]],
                         [([108, 845, 756, 221], 1, 5, "dark", True), ([108, 845, 756, 221], 10, 14, "light", True),
                          ([108, 845, 756, 221], 15, 18, "light", True)])
        self.assertIn("glass boxes: dark 1-5 s, light 10-14 s, light 15-18 s, stickers: fire at 1 s, star at 2.5 s", self.chat.clip_line())

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
        moving = lambda rec, a, b, persons, fps=10: [[9.0] * 100 + [1.0] * 100, [1.0] * 100 + [9.0] * 100]
        real, talk.motion = talk.motion, moving
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
            talk.motion = real
        made, reaction, together, nobody, push, move = steps[-6:]
        self.assertFalse(made["refused"], made["result"])
        self.assertIn("in the talk layout. It follows 2 people, numbered from the left", made["result"])
        self.assertIn("Shots: 0-10.2 s person 1; 10.2-20 s person 2.", made["result"])
        self.assertIn("comes next as a picture", made["result"])
        self.assertIn("Shots now: 0-3 s person 1; 3-5 s person 2; 5-10.2 s person 1; 10.2-20 s person 2.",
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
        self.assertEqual((len(on[1.0]), len(on[16.0])), (1, 2),
                         "words keep clear of the face on screen: one person at 1 s, both on the card at 16 s")
        self.assertEqual(tools.describe(self.chat.project.entries[-1]), "changing the shots")
        self.faces = found([(30 + i * 0.5, [[140, 40, 40, 40]]) for i in range(40)], size=(320, 180))
        brain, steps = self.run_job([call("make_clip", moment=1, layout="talk", why="one person")], "Done.")
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("found 1 sitting still", steps[-1]["result"])

    def test_muse_is_offered_every_studio_sticker(self):
        offered = tools.TOOLS["add_sticker"].params["sticker"]["enum"]
        self.assertEqual(offered, [s["name"] for s in effects.stickers()])
        self.assertEqual(len(offered), 60)

    def test_the_clip_knows_when_someone_speaks_in_clip_seconds(self):
        said = {"said": [{"from": 29, "to": 31}, {"from": 39, "to": 46}, {"from": 60, "to": 61}]}
        self.assertEqual(tools._speech({"sound_tracks": 1}, said, [[30, 40], [45, 50]]), [[0, 1], [9, 10], [10, 11]])
        self.assertEqual(tools._speech({"sound_tracks": 0}, {}, [[30, 40]]), [], "no sound: nobody speaks")
        self.assertIsNone(tools._speech({"sound_tracks": 1}, {}, [[30, 40]]), "never heard: nobody knows")

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
        self.assertEqual(state["speech"], [[2, 4.5]])
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
            self.chat.undo(), self.chat.undo()
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
        self.assertEqual(asked, [(30.0, 50.0)] * 3, "read when the clip is made and at every check")

    def test_the_moments_of_a_tutorial_are_read_ahead_when_they_are_proposed(self):
        ahead = []
        had, screen.ahead = screen.ahead, lambda rec, a, b: ahead.append((a, b))
        try:
            for kind in ("explain", "game"):
                self.run_job([call("open_recording", path=self.clip, why="open it")],
                             [call("load_recipe", type=kind, why="the kind")],
                             [call("look_at_piece", start="0:30", end="0:50", why="the top stretch")],
                             [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the tone"}])],
                             "Done.")
                ahead.append(kind)
        finally:
            screen.ahead = had
        self.assertEqual(ahead, [(30.0, 50.0), "explain", "game"])

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
                [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the tone"}])],
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
        refused, ready = steps[-2], steps[-1]
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

        def plan(rec, pieces, region, aspect, avoid=None):
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

    def test_a_clip_whose_recipe_does_not_read_the_screen_is_not_read(self):
        asked = []
        had, screen.readings = screen.readings, lambda rec, a, b, on_start=None: asked.append((a, b)) or []
        try:
            _, steps = self.run_job(*self.to_moment(), [call("load_recipe", type="game", why="a stream")],
                                    [call("make_clip", moment=1, layout="card", why="the game")], "Done.")
        finally:
            screen.readings = had
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertNotIn("read the screen", steps[-1]["result"])
        self.assertEqual(asked, [])

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
                [call("hand_over", title="t", caption="c", hashtags=["#a"], why="post")],
                "Done.")
        finally:
            render.export = had
        unchecked, first, failed, _, _, second, ready = steps[-7:]
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
        self.assertIn("The finished clip, picture 1 of 3:", str(brain.sent[len(steps) - 4]))
        self.assertTrue(failed["refused"])
        self.assertIn("These checks failed on the finished file: No viewer's name can be read; It does not start black "
                      "or silent.", failed["result"])
        self.assertTrue(second["result"].startswith("Checked 3 of 3 on the finished file "), second["result"])
        self.assertFalse(ready["refused"], ready["result"])
        shown = [e for e in self.chat.events if e["kind"] == "ready"][-1]
        self.assertEqual([c["ok"] for c in shown["checks"]], [None, None, True, None, True, True])
        self.assertEqual(agent.Chat.load(self.chat.id).checks, self.chat.checks)

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
        self.assertIn('Left out 1 line the hearing was not sure of, because a wrong caption is worse than none: 0:05 '
                      '"bxo noqa mumble".', caps["result"])
        shown = " ".join(line["text"] for line in self.chat.project.state()["captions"]["lines"])
        self.assertNotIn("mumble", shown)
        self.assertIn("fall", shown)

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
        self.assertIn("On a phone, the name and the caption cover everything below y 1520", low["result"])
        self.assertIn("give person_said only when the person asked for this spot", low["result"])
        self.assertFalse(low_asked["refused"], low_asked["result"])
        self.assertIn("Placed where the person asked. On a phone, the name and the caption cover everything below y "
                      "1520", low_asked["result"])
        self.assertTrue(face["refused"])
        self.assertIn("the title would cover a face at", face["result"])
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
                [call("hand_over", title="สู้จบ", caption="ดูรายละเอียดเพื่มเติมได้ในลิงก์", hashtags=["#เกม"],
                      why="post")],
                '{"texts": ["ดูรายละเอียดเพิ่มเติมได้ในลิงก์"]}',
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
        self.assertEqual((ready["title"], ready["caption"]), ("สู้จบ", "ดูรายละเอียดเพิ่มเติมได้ในลิงก์"))
        aside = [m for m in brain.sent if len(m) == 1]
        self.assertEqual(len(aside), 2, "each fix is asked apart from the job's conversation")

    def test_after_the_job_the_reasons_and_answers_shown_are_spelled_and_the_next_message_cancels_that(self):
        brain, steps = self.run_job(
            [call("open_recording", path=self.clip, why="เปิดไฟล์ที่ระบุ")],
            [call("look_at_piece", start="0:30", end="0:50", why="ดูช่วงที่ไมไ่ด้ดู")],
            [call("propose_moments", why="เสนอ", moments=[{"look": 1, "reason": "ช่วงที่ต้วเลขชัด"}])],
            "เสร็จแล้ว ต้วเลขครบ",
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
        self.assertIn("เสร็จแล้ว ต้วเลขครบ", answers)

        waiting = threading.Event()

        def typed_meanwhile(asking):
            waiting.set()
            for _ in range(200):
                if asking.cancelled:
                    raise mind.Cancelled()
                time.sleep(0.05)
            return '{"texts": ["too late"]}'

        brain, steps = self.run_job([call("look_at_piece", start="0:10", end="0:20", why="ดูต้วอย่าง")], "ok",
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
        self.assertIn("- middle (y 905 to 1075): never clear for 2 s or more; do not use it", steps[-1]["result"])
        self.assertIn("- top (y 285 to 520): clear of faces for the whole clip", steps[-1]["result"])

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


if __name__ == "__main__":
    unittest.main()
