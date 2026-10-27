"""The editor core: layouts, faces, the project list, the drawn frame and Muse's editing tools.

    python -m unittest discover -s tests
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, faces, hearing, layouts, project, recording, render, tools  # noqa: E402
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
            shutil.copy(self.clip, out)
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
                [call("hand_over", title="Boss down", caption="He falls.\nTwice.\n#gaming #boss", why="post",
                      hashtags=["gaming", "#gaming", "sifu clip", "#"])],
                "Done.")
        finally:
            render.export = had
        early, undecided, kept, first, again, titled, stale, undone, ready = steps[-9:]
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
