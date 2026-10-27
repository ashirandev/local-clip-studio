"""Following the action on a screen: where the card looks, when it moves, and the path FFmpeg draws.

    python -m unittest discover -s tests
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import focus, layouts, recording  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def screen(path, seconds=12, spots=(((100, 80), 0, 6), ((950, 560), 6, 12)), whole=None, area=(1280, 720)):
    """A still dark 1280x720 screen where a small moving pattern shows at each spot from its start to its end, or
    noise over area at the top-left from whole[0] to whole[1]."""
    inputs = ["-f", "lavfi", "-i", "color=c=0x202020:s=1280x720:r=30:d=%d" % seconds]
    graph, last = [], "0:v"
    for k, ((x, y), a, b) in enumerate(spots):
        inputs += ["-f", "lavfi", "-i", "testsrc=s=200x120:r=30:d=%d" % seconds]
        graph.append("[%s][%d:v]overlay=%d:%d:enable='between(t,%d,%d)'[v%d]" % (last, k + 1, x, y, a, b, k))
        last = "v%d" % k
    if whole:
        inputs += ["-f", "lavfi", "-i", "color=c=gray:s=%dx%d:r=30:d=%d,noise=alls=80:allf=t" % (area + (seconds,))]
        graph.append("[%s][%d:v]overlay=0:0:enable='between(t,%d,%d)'[vw]" % (last, len(spots) + 1, whole[0],
                                                                              whole[1]))
        last = "vw"
    subprocess.run([FFMPEG, "-v", "error", "-y"] + inputs + ["-filter_complex", ";".join(graph) or "[0:v]null[v]",
                                                             "-map", "[%s]" % (last if graph else "v"), "-c:v",
                                                             "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                                                             path], check=True)
    st = os.stat(path)
    return {"path": path, "name": os.path.basename(path), "size": st.st_size, "mtime": st.st_mtime, "width": 1280,
            "height": 720}


def inside(box, x, y, w, h):
    return box[0] <= x and box[1] <= y and box[0] + box[2] >= x + w and box[1] + box[3] >= y + h


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Plan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_card_zooms_where_the_screen_changes_and_moves_when_the_change_moves(self):
        rec = screen(os.path.join(self.tmp, "a.mp4"))
        plan = focus.plan(rec, [[0, 12]], [0, 0, 1280, 720], 16 / 9.0)
        self.assertEqual((plan["zoom"], plan["size"]), (0.34, [436, 246]), "the tightest zoom that holds the change")
        self.assertEqual(len(plan["shots"]), 2, plan)
        first, second = plan["shots"]
        self.assertEqual((first["from"], first["to"], second["to"]), (0.0, second["from"], 12))
        self.assertTrue(5 <= second["from"] <= 7, second)
        self.assertTrue(inside([first["x"], first["y"]] + plan["size"], 100, 80, 200, 120), plan)
        self.assertTrue(inside([second["x"], second["y"]] + plan["size"], 950, 560, 200, 120), plan)

    def test_the_card_never_shows_what_it_must_avoid(self):
        rec = screen(os.path.join(self.tmp, "b.mp4"), spots=(((400, 100), 0, 12),))
        ax, ay, aw, ah = avoid = [0, 0, 380, 220]
        plan = focus.plan(rec, [[0, 12]], [0, 0, 1280, 720], 16 / 9.0, avoid)
        w, h = plan["size"]
        for s in plan["shots"]:
            x, y = s["x"], s["y"]
            self.assertTrue(x + w <= ax or ax + aw <= x or y + h <= ay or ay + ah <= y, (s, plan["size"]))

    def test_a_change_everywhere_or_nowhere_shows_the_whole_screen(self):
        self.assertIsNone(focus.plan(screen(os.path.join(self.tmp, "c.mp4"), spots=(), whole=(0, 12)), [[0, 12]],
                                     [0, 0, 1280, 720], 16 / 9.0))
        self.assertIsNone(focus.plan(screen(os.path.join(self.tmp, "d.mp4"), spots=()), [[0, 12]],
                                     [0, 0, 1280, 720], 16 / 9.0))
        self.assertIsNone(focus.plan(screen(os.path.join(self.tmp, "g.mp4"), spots=(), whole=(0, 12), area=(1100, 620)),
                                     [[0, 12]], [0, 0, 1280, 720], 16 / 9.0),
                          "a change over most of the screen, such as a page that scrolls, is not followed")

    def test_a_short_flicker_does_not_move_the_card(self):
        rec = screen(os.path.join(self.tmp, "e.mp4"), spots=(((100, 80), 0, 12), ((950, 560), 5, 6)))
        plan = focus.plan(rec, [[0, 12]], [0, 0, 1280, 720], 16 / 9.0)
        self.assertEqual(len(plan["shots"]), 1, plan)

    def test_times_are_clip_seconds_across_pieces(self):
        rec = screen(os.path.join(self.tmp, "f.mp4"))
        plan = focus.plan(rec, [[0, 4], [8, 12]], [0, 0, 1280, 720], 16 / 9.0)
        self.assertEqual(len(plan["shots"]), 2, plan)
        self.assertTrue(3.5 <= plan["shots"][1]["from"] <= 5.5, plan)
        self.assertEqual(plan["shots"][-1]["to"], 8)


class Path(unittest.TestCase):
    SHOTS = [{"from": 0.0, "to": 4.0, "x": 100, "y": 50}, {"from": 4.0, "to": 9.0, "x": 700, "y": 300},
             {"from": 9.0, "to": 12.0, "x": 300, "y": 20}]

    def evaluate(self, text, t):
        code = re.sub(r"\bif\(", "_if(", text).replace("lt(", "_lt(")
        return eval(code, {"_if": lambda c, a, b: a if c else b, "_lt": lambda a, b: a < b, "t": t})

    def test_ffmpeg_draws_the_path_at_gives(self):
        for k in range(0, 121):
            t = k / 10.0
            x, y = focus.at(self.SHOTS, t)
            self.assertAlmostEqual(self.evaluate(focus.expression(self.SHOTS, "x"), t), x, places=6, msg=t)
            self.assertAlmostEqual(self.evaluate(focus.expression(self.SHOTS, "y"), t), y, places=6, msg=t)

    def test_the_card_holds_then_pans_smoothly(self):
        self.assertEqual(focus.at(self.SHOTS, 3.9), (100, 50))
        self.assertEqual(focus.at(self.SHOTS, 4.0), (100, 50))
        self.assertEqual([round(v, 6) for v in focus.at(self.SHOTS, 4.0 + focus.PAN / 2)], [400, 175])
        self.assertEqual(focus.at(self.SHOTS, 4.0 + focus.PAN), (700, 300))
        self.assertEqual(focus.at(self.SHOTS, 11.9), (300, 20))


class Followed(unittest.TestCase):
    def setUp(self):
        lay = layouts.make("card", (1920, 1080), [1605, 872, 136, 170])
        self.lay = layouts.followed(lay, {"zoom": 0.5, "size": [960, 630], "shots": Path.SHOTS})
        self.card = next(p for p in self.lay["parts"] if p["name"] == "card")

    def test_the_card_shows_the_window_and_its_crop_moves_with_t(self):
        self.assertEqual(self.card["src"], [100, 50, 960, 630])
        self.assertIn("crop=w=960:h=630:x='if(lt(t,4.000)", layouts._scaled(self.card))

    def test_a_box_shows_where_the_card_is_then(self):
        box = [350, 100, 40, 40]
        self.assertEqual(len(layouts.to_frame(self.lay, box, 2.0)), 1, "inside the first place")
        self.assertEqual(layouts.to_frame(self.lay, box, 6.0), [], "the card has moved away")
        self.assertEqual(len(layouts.to_frame(self.lay, box)), 2, "the first and the last place show it")
        self.assertEqual(layouts.parts_at(self.lay, 6.0)[0]["src"], [700, 300, 960, 630])
