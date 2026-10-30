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


def slide(path, rows=range(8), first=300, letters=35, top=150, camera=False):
    """A still 1280x720 slide, cream, with written lines: each a row of small dark letters apart from each other,
    from x first and y top, 50 px apart; the mouse's yellow ring at the left and a small logo in the bottom-right
    corner. camera: a moving test picture fills it instead."""
    marks = ["drawbox=x=%d:y=%d:w=12:h=24:color=0x202020:t=fill" % (first + k * 20, top + r * 50)
             for r in rows for k in range(letters)]
    marks += ["drawbox=x=120:y=300:w=16:h=16:color=yellow:t=fill", "drawbox=x=1236:y=660:w=24:h=40:color=0xC03030:t=fill"]
    source = {True: "testsrc=s=1280x720:r=30:d=3", "sky": "color=c=0xFEF7ED:s=1280x720:r=30:d=3,format=yuv420p,"
              "geq=lum='if(lt(Y,240),Y/2+40,240)':cb=128:cr=128"}.get(camera, "color=c=0xFEF7ED:s=1280x720:r=30:d=3")
    script = path + ".graph"
    with open(script, "w", encoding="utf-8") as f:
        f.write("[0:v]" + ("null" if camera is True else ",".join(marks)) + "[v]")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", source, "-/filter_complex", script, "-map",
                    "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path], check=True)
    st = os.stat(path)
    return {"path": path, "name": os.path.basename(path), "size": st.st_size, "mtime": st.st_mtime, "width": 1280,
            "height": 720}


def read(sure, of):
    return [{"box": [300, 150 + k, 600, 24], "sure": 0.95 if k < sure else 0.4, "text": "x"} for k in range(of)]


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Slide(unittest.TestCase):
    """The gold lesson, 10-30: Thai slides the reader cannot read; the card zoomed from its unsure lines and cut them."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_lines_read_surely_under_half_the_time_are_a_script_the_reader_does_not_know(self):
        self.assertTrue(focus.unread(read(4, 17)))
        self.assertFalse(focus.unread(read(15, 17)))
        self.assertFalse(focus.unread([]))
        self.assertFalse(focus.unread(None))

    def test_a_slides_written_part_leaves_out_its_border_the_mouse_and_a_corner_logo(self):
        box = focus.content(slide(os.path.join(self.tmp, "s.mp4")), [[0, 3]])
        x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        self.assertTrue(260 <= x0 <= 300 and 992 <= x1 <= 1030, "the lines run from x 300 to 992: %s" % box)
        self.assertTrue(110 <= y0 <= 150 and 524 <= y1 <= 564, "and from y 150 to 524: %s" % box)

    def test_a_camera_or_a_slide_written_to_its_edges_is_shown_whole(self):
        self.assertIsNone(focus.content(slide(os.path.join(self.tmp, "c.mp4"), camera=True), [[0, 3]]))
        edges = slide(os.path.join(self.tmp, "e.mp4"), rows=range(14), first=40, letters=60, top=30)
        self.assertIsNone(focus.content(edges, [[0, 3]]), "written from x 40 to 1232 and y 30 to 704")
        sky = slide(os.path.join(self.tmp, "k.mp4"), camera="sky")
        self.assertIsNone(focus.content(sky, [[0, 3]]), "a sky fading down its top third: its edge is not one colour")

    def test_the_part_is_grown_to_a_shape_the_card_shows_without_cutting_it(self):
        for box in ([500, 100, 200, 600], [100, 300, 1000, 100], [0, 0, 300, 720]):
            part = layouts.framed(box, (1280, 720))
            self.assertTrue(inside(part, box[0] + 2, box[1] + 2, box[2] - 4, box[3] - 4), (box, part))
            self.assertTrue(inside([0, 0, 1280, 720], *part), part)
            lay = layouts.make("card", (1280, 720), None, part)
            self.assertNotIn("The card was trimmed at the top and bottom to fit the frame.", lay["notes"], (box, part))
            self.assertNotIn("The card was trimmed at the sides to fit the frame.", lay["notes"], (box, part))

    def test_a_slide_the_reader_cannot_read_shows_its_written_part_whole_and_bigger(self):
        from clipstudio import screen, tools
        rec = slide(os.path.join(self.tmp, "s.mp4"))
        lay = layouts.make("card", (1280, 720))
        real = screen.readings
        self.addCleanup(setattr, screen, "readings", real)
        screen.readings = lambda rec, a, b: [{"at": 0.0, "lines": read(4, 17)}]
        got, said = tools._follow(rec, lay, [[0, 3]], [0, 3])
        (card,) = [p["src"] for p in got["parts"] if p["name"] == "card"]
        self.assertTrue(inside(card, 300, 150, 692, 374), "every line whole: %s" % card)
        self.assertLess(card[2], 1280 / 1.6, "the lines show over 1.6 times as big: %s" % card)
        self.assertIn("(4 of 17 lines read surely), so the card does not follow them", said)
        self.assertIn("whole and 1.", said)
        column = slide(os.path.join(self.tmp, "n.mp4"), letters=10)
        got, said = tools._follow(column, lay, [[0, 3]], [0, 3])
        (card,) = [p["src"] for p in got["parts"] if p["name"] == "card"]
        self.assertTrue(inside(card, 300, 150, 192, 374), "a tall column of lines is not cut at its top: %s" % card)
        camera = slide(os.path.join(self.tmp, "c.mp4"), camera=True)
        got, said = tools._follow(camera, lay, [[0, 3]], [0, 3])
        self.assertIs(got, lay)
        self.assertIn("or it is not a slide, so the card shows it whole", said)


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


def lines(x=300, y=200, n=8, height=18, width=420, sure=0.99):
    """Lines of text as the screen reader gives them: n lines starting at x, one under the other from y."""
    return [{"text": "line %d" % k, "box": [x, y + k * 2 * height, width, height], "sure": sure} for k in range(n)]


class Letters(unittest.TestCase):
    def test_the_usual_line_is_the_median_of_the_lines_read_surely_outside_the_face_camera(self):
        camera = [{"box": [1000, 560 + k, 200, 90], "sure": 1} for k in range(9)]
        text = lines(height=18) + lines(y=600, n=3, height=60, sure=0.5) + camera
        self.assertEqual(focus.letters(text, avoid=[950, 550, 330, 170]), 18.0, "the face camera's own words are not "
                         "the screen's")
        self.assertIsNone(focus.letters([]))

    def test_tarms_floor_of_16_px_on_a_phone(self):
        width = focus.readable(18, 960)
        self.assertAlmostEqual(focus.on_phone(18, width, 960), focus.PHONE_PX, places=6)
        self.assertTrue(355 < width < 365, "18 px lines on a 960-pixel card: about 360 px of the screen, %g" % width)


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Readable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_still_screen_holds_on_its_lines_from_their_start_at_a_readable_size(self):
        rec = screen(os.path.join(self.tmp, "s.mp4"), spots=())
        text = lines(x=700, y=300, n=10, width=200) + lines(x=40, y=40, n=1)
        plan = focus.plan(rec, [[0, 12]], [0, 0, 1280, 720], 16 / 9.0, text=text, shown=960)
        self.assertEqual(plan["on"], "lines")
        self.assertGreaterEqual(plan["letters"], focus.PHONE_PX - 0.5)
        (shot,) = plan["shots"]
        self.assertTrue(698 - 2 * 1280 / focus.SMALL_W <= shot["x"] <= 700, "the window starts where the lines do: %s"
                        % shot)
        self.assertTrue(shot["y"] <= 300 and shot["y"] + plan["size"][1] >= 300 + 18, "it holds the lines from the "
                        "first: %s" % plan)

    def test_the_card_leaves_a_place_whose_lines_are_gone(self):
        rec = screen(os.path.join(self.tmp, "s.mp4"), spots=())
        dialog = [dict(line, at=t) for t in (0.0, 1.0, 2.0, 3.0) for line in lines(x=700, y=300, n=10, width=300)]
        page = [dict(line, at=t) for t in (0.0, 2.0, 5.0) for line in lines(x=40, y=40, n=3)]
        later = [dict(line, at=5.0) for line in lines(x=40, y=400, n=8, width=300)]
        chip = [dict(line, at=5.0) for line in lines(x=720, y=330, n=1, width=60)]
        plan = focus.plan(rec, [[0, 12]], [0, 0, 1280, 720], 16 / 9.0, text=dialog + page + later + chip, shown=960)
        self.assertEqual(plan["on"], "lines")
        first, second = plan["shots"]
        self.assertTrue(inside([first["x"], first["y"]] + plan["size"], 700, 300, 300, 18), "on the dialog: %s" % plan)
        self.assertEqual((second["from"], first["to"]), (5.0, 5.0), "it stays while the dialog is there (round 8), "
                         "and a small chip left in its place does not hold it (round 9)")
        self.assertTrue(inside([second["x"], second["y"]] + plan["size"], 40, 400, 300, 18), "then where the lines "
                        "are: %s" % plan)
        held = focus.plan(rec, [[0, 12]], [0, 0, 1280, 720], 16 / 9.0, text=[{k: v for k, v in line.items() if k != "at"}
                                                                              for line in dialog + page + later], shown=960)
        self.assertEqual(len(held["shots"]), 1, "lines read without their second are held as before")

    def test_a_blink_that_is_not_one_part_of_the_screen_is_not_chased(self):
        rec = screen(os.path.join(self.tmp, "n.mp4"), spots=(), whole=(0, 12), area=(1100, 620))
        plan = focus.plan(rec, [[0, 12]], [0, 0, 1280, 720], 16 / 9.0, text=lines(x=700, y=300), shown=960)
        self.assertEqual(plan["on"], "lines", "a change everywhere is not followed; the card holds on the lines")

    def test_a_change_in_one_part_is_followed_at_the_readable_size(self):
        rec = screen(os.path.join(self.tmp, "a.mp4"))
        plan = focus.plan(rec, [[0, 12]], [0, 0, 1280, 720], 16 / 9.0, text=lines(height=14), shown=960)
        self.assertNotIn("on", plan)
        self.assertEqual(len(plan["shots"]), 2, plan)
        self.assertGreaterEqual(plan["letters"], focus.PHONE_PX - 0.5)
        self.assertLess(plan["size"][0], 0.34 * 1280, "tighter than the change alone needs, so the lines can be read")

    def test_without_lines_the_card_is_as_before(self):
        rec = screen(os.path.join(self.tmp, "d.mp4"), spots=())
        self.assertIsNone(focus.plan(rec, [[0, 12]], [0, 0, 1280, 720], 16 / 9.0, text=[], shown=960))


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

@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Thing(unittest.TestCase):
    """A product SAM followed in a 640x360 recording, 60 px wide at the middle of the height: where it is from
    recording second T (8 s from 0 = second 0 of the clip), and the window of the vertical frame, 202 px wide."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def plan(self, where, rest=320):
        from test_cutout import make_mask
        path = make_mask(self.tmp, "thing.mkv", 8, 0, "lt(abs(X-(%s)),30)*lt(abs(Y-180),40)" % where)
        return focus.thing({"path": path, "fps": 30, "from": 0, "to": 8}, [[0, 8]], (640, 360), 9 / 16.0, rest)

    def test_the_frame_holds_on_the_product_moves_when_it_leaves_and_rests_on_the_person_when_it_is_gone(self):
        plan = self.plan("if(lt(T,2),520,if(lt(T,4),610,if(lt(T,6),-1000,100)))")
        self.assertEqual(plan["size"], [202, 360])
        self.assertEqual([s["x"] for s in plan["shots"]], [418, 220, 0],
                         "centred on it; 20 px further to the edge is no move; put down, back to the person; "
                         "then to where it is held again")
        for got, want in zip([s["from"] for s in plan["shots"]], [0, 4.0, 6.0 - focus.THING_LEAD]):
            self.assertAlmostEqual(got, want, delta=0.1, msg="a move to the thing starts THING_LEAD early")
        self.assertEqual(plan["seen"], 0.75)
        self.assertEqual([s["x"] for s in self.plan("if(lt(T,2),520,if(lt(T,4),610,if(lt(T,6),-1000,100)))",
                                                    rest=None)["shots"]], [418, 0], "with no person, it holds")

    def test_a_product_not_there_at_first_starts_on_the_person_and_one_never_there_is_no_plan(self):
        plan = self.plan("if(lt(T,2),-1000,100)")
        self.assertEqual([s["x"] for s in plan["shots"]], [220, 0])
        self.assertAlmostEqual(plan["shots"][1]["from"], 2.0 - focus.THING_LEAD, delta=0.1)
        self.assertIsNone(self.plan("-1000"))
        self.assertIsNone(focus.thing({"path": "never read", "fps": 30, "from": 0, "to": 8}, [[0, 8]], (360, 640),
                                      9 / 16.0), "a vertical recording has nowhere to move")

    def test_a_product_as_wide_as_the_frame_is_kept_whole_though_its_middle_never_nears_the_side(self):
        """Tarm's controller, 10-30: as wide as the window, its middle inside, 15-25% of it off the window for 2.5 s.
        Here 180 px wide in the 202 px window, its middle moving from 200 to 240, still inside the middle 60%."""
        from test_cutout import make_mask
        path = make_mask(self.tmp, "wide.mkv", 8, 0, "lt(abs(X-if(lt(T,2),200,240)),90)*lt(abs(Y-180),40)")
        mask = {"path": path, "fps": 30, "from": 0, "to": 8}
        plan = focus.thing(mask, [[0, 8]], (640, 360), 9 / 16.0, 320)
        self.assertEqual([s["x"] for s in plan["shots"]], [98, 138])
        self.assertAlmostEqual(plan["shots"][1]["from"], 2.0 - focus.THING_LEAD, delta=0.1)
        self.assertEqual(focus.cut_off(mask, [[0, 8]], (640, 360), lambda t: [round(focus.at(plan["shots"], t)[0]),
                                                                             0, 202, 360]), [])

    def test_the_seconds_the_frame_cuts_the_product_are_found_and_a_frame_catching_up_is_not(self):
        """Tarm 10-30, on a seller's ad that split the screen as she raised the tumbler, so it was gone. The window
        holds at x 0-202; the clip is recording seconds 1-3 and 4-8."""
        from test_cutout import make_mask
        path = make_mask(self.tmp, "cut.mkv", 8, 0, "lt(abs(X-(%s)),30)*lt(abs(Y-180),40)" % (
            "if(lt(T,2),100,if(lt(T,2.3),300,if(lt(T,4),100,if(lt(T,6),300,if(lt(T,6.5),180,"
            "if(lt(T,7.5),190,-1000))))))"))
        cut = focus.cut_off({"path": path, "fps": 30, "from": 0, "to": 8}, [[1, 3], [4, 8]], (640, 360),
                            lambda t: [0, 0, 202, 360])
        self.assertEqual(len(cut), 2, "out for 0.3 s is the frame catching up; 87%% inside is in the frame: %s" % cut)
        for got, want in zip(cut, [(2.0, 4.0), (4.5, 5.5)]):
            self.assertAlmostEqual(got[0], want[0], delta=0.05)
            self.assertAlmostEqual(got[1], want[1], delta=0.05)
        self.assertEqual(focus.cut_off({"path": path, "fps": 30, "from": 0, "to": 8}, [[1, 3], [4, 8]], (640, 360),
                                       lambda t: [0, 0, 640, 360]), [], "the whole picture holds it all")
