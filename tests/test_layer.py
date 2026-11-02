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

from clipstudio import effects, layer, layouts, recording, render, text  # noqa: E402

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
    fonts = render.fonts(st, "clean")
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

    def test_captions_one_line_at_a_time_on_the_page_are_the_ones_burned_in(self):
        """The yeti review 10-29: a talked review's captions, in their own look, were drawn in another face."""
        st = dict(state(word=False, said=[dict(SAID[1], **{"from": 0.3})]), style=None)
        st["captions"]["one"] = True
        index = layer.words(st)
        n = 30
        diff = abs(laid(picture(entry_at(index, n)["file"]), grounds([n], 2)[n]) - burned(st, [n])[n]).max(axis=2)
        self.assertLess(int((diff > 3).sum()), 100, "%d pixels over 3 levels off" % (diff > 3).sum())

    def test_no_words_no_layer_and_words_behind_the_person_stay_in_the_picture(self):
        self.assertEqual(layer.words(state(title=False, word=False, said=None)), [])
        self.assertEqual(layer.words(state(word=False, said=None, behind=True)), [])


@unittest.skipUnless(FFMPEG and HAS_NUMPY, "needs ffmpeg and numpy")
class Apart(unittest.TestCase):
    """Tarm 11-02: "อยากให้สามารถ Export ออกมาเป็น Track ของใครของมัน"."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work's ลอง")

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def alpha(self, path, at):
        import numpy as np
        raw = subprocess.run([FFMPEG, "-v", "error", "-ss", "%.3f" % at, "-i", path, "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "rgba", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(1920, 1080, 4)[:, :, 3]

    def test_each_part_holds_its_one_thing_alone(self):
        st = dict(state(), channel={"name": "@ours", "spot": None})

        def said(part):
            with open(render.ass(st, part=part), encoding="utf-8") as f:
                return f.read().split("[Events]")[1]

        for part, kept in ((("title", 0), "ลดครึ่งราคา"), (("word", 0), "ว้าว"), (render.ONLY_CAPTIONS, "ซื้อหนึ่ง"), (render.ONLY_CHANNEL, "@ours")):
            words = said(part)
            for other in ("ลดครึ่งราคา", "ว้าว", "ซื้อหนึ่ง", "@ours"):
                self.assertEqual(other in words, other == kept, (part, other))

    def test_a_clip_with_no_captions_has_no_captions_part(self):
        self.assertIsNone(render.ass(state(said=None), part=render.ONLY_CAPTIONS))
        self.assertIsNone(layer.captions_video(state(said=None)))

    def test_the_captions_video_shows_them_at_their_times_over_nothing_for_the_whole_clip(self):
        path = layer.captions_video(state())
        for at, seen in ((0.1, False), (1.0, True), (2.2, False), (3.0, True), (3.9, False)):
            self.assertEqual(int(self.alpha(path, at).max()) > 200, seen, at)
        self.assertEqual(int(self.alpha(path, 1.0)[:layouts.CAPTIONS[1] - 60].max()), 0, "nothing but the captions: the title is on the clip")
        self.assertEqual(int(self.alpha(path, 1.0).min()), 0, "the rest shows the clip through")
        got = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                              "stream=codec_name,profile,pix_fmt,width,height,nb_read_frames", "-of", "csv=p=0", path],
                             capture_output=True, text=True, check=True).stdout
        self.assertEqual(got.strip().split(",")[0], "prores")
        self.assertIn("4444", got)
        self.assertIn("yuva", got)
        self.assertIn("1080,1920", got)
        self.assertEqual(got.strip().split(",")[-1], "120", "as long as the clip, or Premiere reads past its end")

    def test_the_captions_video_is_as_long_as_the_clip_after_many_steps(self):
        lines = [{"from": k * 0.5, "to": k * 0.5 + 0.4, "text": "คำที่ %d" % k} for k in range(30)]
        path = layer.captions_video(state("pop", "pop", title=False, word=False, length=15.26, said=lines))
        got = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                              "stream=nb_read_frames", "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
        self.assertEqual(got.strip().strip(","), "458")

    def test_the_same_captions_are_made_once(self):
        first = layer.captions_video(state())
        with mock.patch.object(layer.subprocess, "run", side_effect=AssertionError("made again")):
            self.assertEqual(layer.captions_video(state()), first)
        self.assertEqual(sorted(n for n in os.listdir(render.folder("plates")) if n.endswith(".txt") or ".part" in n), [],
                         "nothing is left behind")

    def test_a_step_of_one_frame_lasts_one_frame(self):
        """EP22 11-02: FFmpeg's concat list stretched each to 1/25 s, and the captions ran 2.8 s late by the end of the clip."""
        import numpy as np
        lines = [{"from": k * 0.2, "to": k * 0.2 + 0.15, "text": "คำที่ %d" % k} for k in range(30)]
        st = state("pop", "pop", title=False, word=False, length=6.0, said=lines)
        path = layer.captions_video(st)
        index = layer.words(st, "clean", layer.FULL, render.ONLY_CAPTIONS, smooth=True)
        raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vf", "scale=108:192:flags=neighbor", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                             capture_output=True, check=True).stdout
        seen = np.frombuffer(raw, np.uint8).reshape(-1, 192, 108, 4)[:, :, :, 3].astype(int)
        self.assertEqual(len(seen), 180)
        step = index[-3]
        small = subprocess.run([FFMPEG, "-v", "error", "-i", step["file"], "-vf", "scale=108:192:flags=neighbor", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                               capture_output=True, check=True).stdout
        want = np.frombuffer(small, np.uint8).reshape(192, 108, 4)[:, :, 3].astype(int)
        at = layer._frame(step["from"])
        self.assertLess(float(np.abs(seen[at] - want).mean()), 0.5, "the last words at their own frame")

    def test_a_big_word_pops_on_its_own_track_as_the_export_draws_it(self):
        st = state(title=False, said=None)
        got = layer.clips(st)
        self.assertEqual([(c["kind"], c["name"], c["at"], round(c["length"], 3)) for c in got], [("titles", "Big word: ว้าว", 1.0, 0.8)])
        first, later = self.alpha(got[0]["path"], 1 / 30), self.alpha(got[0]["path"], 2 / 30)
        self.assertNotEqual(int((first > 100).sum()), int((later > 100).sum()), "it grows, frame by frame")

    def test_a_step_that_holds_no_frame_adds_none(self):
        """EP22 11-02: the empty stretch before such a step was counted again at the next one, and the captions came 84 frames late."""
        png = os.path.join(self.tmp, "word.png")
        shutil.copy(layer._clear(), png)
        path = layer._steps_video([{"from": 0.5, "to": 0.5, "file": png}, {"from": 1.0, "to": 1.2, "file": png}], 0, 36)
        got = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                              "stream=nb_read_frames", "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
        self.assertEqual(got.strip().strip(","), "36")

    def test_the_tracks_laid_back_together_are_the_finished_clip(self):
        import numpy as np
        source = os.path.join(self.tmp, "s.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=6", "-f", "lavfi", "-i", "sine=f=440:d=6",
                        "-f", "lavfi", "-i", "sine=f=880:d=6", "-filter_complex", "[1:a]volume='0.3+0.25*sin(t*3)':eval=frame[a];"
                        "[2:a]volume='0.3+0.25*cos(t*2)':eval=frame[b]", "-map", "0:v", "-map", "[a]", "-map", "[b]", "-c:v", "libx264", "-crf", "12",
                        "-c:a", "aac", "-b:a", "256k", "-shortest", source], check=True)
        song = os.path.join(self.tmp, "song.wav")
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=f=220:d=8", "-af", "volume='0.3+0.2*sin(t*2)':eval=frame", "-ac", "2", song], check=True)
        rec = recording.probe(source)
        ball, coin = effects.sticker_png("soccer_ball"), effects.sticker_png("coin")
        st = dict(state(), clip={"pieces": [[0.0, 4.0]], "layout": layouts.make("subject", (1280, 720))},
                  stickers=[{"png": ball, "at": 0.5, "length": 1.5, "box": [100, 300, 300, 300]}],
                  flurries=[{"pngs": [coin], "at": 2.0, "box": [100, 300, 880, 900], "move": "burst", "count": 6}],
                  pictures=[{"png": coin, "at": 1.0, "length": 1.0, "box": [600, 300, 300, 300]}],
                  sounds=[{"kind": "ding", "at": 1.0}], channel={"name": "@ours", "spot": None}, tracks=[0, 1],
                  music={"path": song, "from": 0, "level": -6, "duck": -6}, speech=[[0.5, 1.5]])
        st["hits"] = st["hits"] + [{"kind": "flash", "at": 3.0}]
        whole = os.path.join(self.tmp, "whole.mp4")
        render.export(rec, st, whole)
        plate, clips, stems = render.plate(rec, st), layer.clips(st), render.stems(rec, st)
        self.assertEqual([c["kind"] for c in clips], ["pictures", "stickers", "stickers", "titles", "titles", "titles", "captions", "flashes"])
        self.assertEqual([s["name"] for s in stems], ["Sound track 1", "Sound track 2", "Sound effects", "Music"])
        self.assertEqual(subprocess.run([shutil.which("ffprobe"), "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name",
                                         "-of", "csv=p=0", plate], capture_output=True, text=True, check=True).stdout.strip(), "", "the plate is silent")
        inputs, graph, last = ["-i", plate], [], "0:v"
        for k, c in enumerate(clips, 1):
            inputs += ["-itsoffset", "%.6f" % c["at"], "-i", c["path"]]
            graph.append("[%s][%d:v]overlay=eof_action=pass:format=rgb[o%d]" % (last, k, k))
            last = "o%d" % k

        def frames(args):
            raw = subprocess.run([FFMPEG, "-v", "error"] + args + ["-vf", "fps=30,scale=270:480:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
                                 if "-map" not in args else [FFMPEG, "-v", "error"] + args + ["-f", "rawvideo", "-pix_fmt", "gray", "-"],
                                 capture_output=True, check=True).stdout
            return np.frombuffer(raw, np.uint8).reshape(-1, 480, 270).astype(int)

        graph.append("[%s]fps=30,scale=270:480:flags=area[back]" % last)
        back, want = frames(inputs + ["-filter_complex", ";".join(graph), "-map", "[back]"]), frames(["-i", whole])
        self.assertEqual(len(back), len(want))
        off = np.abs(back - want).mean(axis=(1, 2))
        calm = np.delete(off, range(90, 96))
        self.assertLess(float(calm.max()), 4.0, "frame %d" % calm.argmax())
        self.assertLess(float(off[90:96].max()), 40.0, "a flash comes close")
        self.assertGreater(float(np.abs(frames(["-i", plate]) - want).mean(axis=(1, 2)).min()), 0.0)

        def sound(args):
            raw = subprocess.run([FFMPEG, "-v", "error"] + args + ["-f", "f32le", "-ac", "2", "-ar", "48000", "-"], capture_output=True, check=True).stdout
            return np.frombuffer(raw, np.float32)

        heard = sound(["-i", whole, "-map", "0:a:0"])
        files = [x for s in stems for x in ("-i", s["path"])]
        laid = sound(files + ["-filter_complex", "".join("[%d:a]" % k for k in range(len(stems))) + "amix=inputs=%d:normalize=0:duration=longest[o]" % len(stems), "-map", "[o]"])
        n = min(len(heard), len(laid))
        best = min(np.sqrt(np.mean((heard[s:n] - laid[:n - s]) ** 2)) for s in range(0, 8192, 2))
        self.assertLess(best / np.sqrt(np.mean(heard[:n] ** 2)), 0.05, "the tracks together sound as the clip does")

    def test_a_picture_under_a_glass_box_stays_on_the_plate_with_the_glass(self):
        ball, coin = effects.sticker_png("soccer_ball"), effects.sticker_png("coin")
        st = dict(state(title=False, word=False, said=None), clip={"pieces": [[0.0, 4.0]], "layout": layouts.make("subject", (1280, 720))},
                  pictures=[{"png": ball, "at": 1.0, "length": 1.0, "box": [600, 300, 300, 300]},
                            {"png": coin, "at": 3.0, "length": 0.8, "box": [600, 300, 300, 300]}],
                  glass=[{"box": [100, 100, 500, 400], "start": 0.5, "end": 2.5}])
        self.assertEqual([c["at"] for c in layer.clips(st)], [3.0])
        files = render.after_source(st, "src", "fin", 1, "clean", render.PLATE)[1]
        self.assertIn(ball, files, "under the glass, it stays with it")
        self.assertNotIn(coin, files, "on its own track, not twice")


if __name__ == "__main__":
    unittest.main()
