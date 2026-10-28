"""The words layer the page lays over the video, checked against the words FFmpeg burns in:
python -m unittest discover -s tests"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import layer, layouts, recording, render, text  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
W, H = layer.SIZE
GROUND = "testsrc2=s=%dx%d:r=30:d=%s,format=rgb24"
SAID = [{"from": 0.3, "to": 2.0, "text": "วันนี้ลดราคาทั้งร้าน ซื้อหนึ่งแถมหนึ่ง"},
        {"from": 2.4, "to": 3.6, "text": "ถึงเที่ยงคืนนี้เท่านั้น"}]


def state(look="pop", mode="plain", title=True, word=True, length=4.0, said=SAID, behind=False):
    return {"titles": [{"text": "ลดครึ่งราคา", "start": 0.2, "end": 3.0, "box": layouts.TOP, "key": None,
                        "behind": behind}] if title else [],
            "captions": {"lines": text.caption_lines(said), "band": [layouts.CAPTIONS[1], layouts.CAPTIONS[3]],
                         "lit": mode == "lit", "pop": mode == "pop"} if said else None,
            "hits": [{"kind": "word", "at": 1.0, "word": "ว้าว", "centre": [540, 700]}] if word else [],
            "style": {"name": look}, "length": length}


def burned(st, frames):
    """{n: frame n of a moving test picture with the clip's words burned in by the export's own filter}, 540x960."""
    import numpy as np
    path = render.ass(st)
    fonts = text.fonts_folder(*render._styles(st, "clean").values())
    pick = layer._any(["eq(n\\,%d)" % n for n in frames])
    raw = subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", GROUND % (W, H, st["length"] + 1), "-vf",
                          "select='%s',%s" % (pick, text._ass_filter(path, fonts)), "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    got = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    return dict(zip(sorted(frames), got))


def grounds(frames, seconds):
    import numpy as np
    pick = layer._any(["eq(n\\,%d)" % n for n in frames])
    raw = subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", GROUND % (W, H, seconds), "-vf",
                          "select='%s'" % pick, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-"], capture_output=True, check=True).stdout
    return dict(zip(sorted(frames), np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)))


def picture(path):
    import numpy as np
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 4)


def laid(png, ground):
    """The page's drawing: the layer's picture over the video frame with plain alpha."""
    import numpy as np
    a = png[..., 3:4].astype(float) / 255
    return np.round(png[..., :3] * a + ground.astype(float) * (1 - a))


def entry_at(index, n):
    return next((e for e in index if e["from"] <= n / 30 < e["to"]), None)


class Changes(unittest.TestCase):
    def test_an_event_changes_at_its_edges_its_transforms_fades_and_karaoke_words(self):
        line = "Dialogue: 1,0:00:01.00,0:00:03.00,s,,0,0,0,,{\\t(20,120,\\fscx125)\\fad(0,150)}{\\k10}{\\kf20}a{\\kf30}b"
        self.assertEqual(sorted(layer.changes(1.0, 3.0, line)), [1.0, 1.02, 1.1, 1.12, 1.3, 1.6, 2.85, 3.0])

    def test_steps_hold_a_frame_and_are_drawn_at_their_last(self):
        ass_text = ("Dialogue: 1,0:00:00.50,0:00:01.00,s,,0,0,0,,a\n"
                    "Dialogue: 1,0:00:00.98,0:00:00.99,s,,0,0,0,,b\n")
        self.assertEqual(layer.steps(ass_text, 2.0), [(0.5, 0.98, 29)],
                         "a step between two frames shows on no frame and is not drawn; no step without an event")


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Words(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_each_step_laid_over_the_video_is_the_video_with_the_words_burned_in(self):
        for look, mode in (("pop", "pop"), ("glow", "lit"), ("sticker", "plain"), ("3d", "pop")):
            st = state(look, mode)
            index = layer.words(st)
            with open(render.ass(st), encoding="utf-8") as f:
                drawn = [n for _, _, n in layer.steps(f.read(), st["length"])]
            self.assertGreater(len(drawn), 5, look)
            words = burned(st, drawn)
            under = grounds(drawn, st["length"] + 1)
            for n in drawn:
                e = entry_at(index, n)
                self.assertIsNotNone(e, "%s %s: frame %d has a step" % (look, mode, n))
                diff = abs(laid(picture(e["file"]), under[n]) - words[n]).max(axis=2)
                # libass rounds each layer it draws (glow, edge, shadow, letters), so over another ground a blurred
                # glow's edge lands a few levels apart on a few pixels; a wrong alpha is off on thousands
                self.assertLessEqual(int(diff.max()), 6, "%s %s frame %d: %d levels off" % (look, mode, n, diff.max()))
                self.assertLess(int((diff > 3).sum()), 100, "%s %s frame %d: %d pixels over 3 levels off"
                                % (look, mode, n, (diff > 3).sum()))

    def test_every_frame_of_still_words_is_the_burned_frame(self):
        st = state("glow", "plain", word=False)
        index = layer.words(st)
        frames = list(range(0, int(st["length"] * 30)))
        words, under = burned(st, frames), grounds(frames, st["length"] + 1)
        for n in frames:
            e = entry_at(index, n)
            if e is None:
                self.assertTrue((words[n] == under[n]).all(), "frame %d shows words but has no step" % n)
                continue
            diff = abs(laid(picture(e["file"]), under[n]) - words[n]).max(axis=2)
            self.assertLess(int((diff > 3).sum()), 100, "frame %d: %d pixels off (the step's edges are off by a "
                                                        "frame?)" % (n, (diff > 3).sum()))

    def test_the_index_runs_in_order_and_steps_that_look_alike_are_one(self):
        index = layer.words(state("pop", "lit"))
        for a, b in zip(index, index[1:]):
            self.assertLessEqual(a["to"], b["from"] + 1e-9)
            self.assertFalse(a["file"] == b["file"] and abs(a["to"] - b["from"]) < 1e-9, "joined: %s %s" % (a, b))
        self.assertTrue(all(0 <= e["from"] < e["to"] <= 4.0 for e in index))

    def test_the_same_still_words_at_two_times_are_one_picture_and_back_to_back_one_entry(self):
        st = state("pop", said=None, word=False)
        title = st["titles"][0]
        st["titles"] = [dict(title, start=0.2, end=1.0), dict(title, start=1.0, end=2.0),
                        dict(title, start=2.5, end=3.5)]
        index = layer.words(st)
        edge = lambda t: text._edge(t) / 100  # noqa: E731
        self.assertEqual([(e["from"], e["to"]) for e in index], [(edge(0.2), edge(2.0)), (edge(2.5), edge(3.5))])
        self.assertEqual(index[0]["file"], index[1]["file"], "the same title later is the same picture")

    def test_drawn_once_and_a_fixed_caption_draws_only_its_own_time(self):
        st = state("pop", "pop")
        first = layer.words(st)
        with mock.patch.object(layer.subprocess, "run", wraps=subprocess.run) as run:
            self.assertEqual(layer.words(st), first)
        self.assertEqual(run.call_count, 0, "nothing changed, nothing drawn")
        fixed = [dict(SAID[0]), dict(SAID[1], text="ถึงเที่ยงคืนพรุ่งนี้")]
        later = state("pop", "pop", said=fixed)
        again = layer.words(later)
        old = {e["file"] for e in first}
        new = [e for e in again if e["file"] not in old]
        self.assertTrue(new, "the fixed caption is drawn")
        start = text.caption_lines(fixed)[-1]["from"]
        self.assertTrue(all(e["from"] >= start - 0.05 for e in new),
                        "only the steps from the fixed line on are new: %s" % [(e["from"], e["to"]) for e in new])
        self.assertTrue(any(e["file"] in old for e in again), "the steps before it are kept")

    def test_no_words_no_layer_and_words_behind_the_person_stay_in_the_picture(self):
        self.assertEqual(layer.words(state(title=False, word=False, said=None)), [])
        self.assertEqual(layer.words(state(word=False, said=None, behind=True)), [])


if __name__ == "__main__":
    unittest.main()
