"""The talk layout: who speaks, the turns and shots that follow them, and the frames they draw:
python -m unittest discover -s tests"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import layouts, project, recording, render, talk, tools  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
LEFT, RIGHT = [200, 200, 160, 160], [920, 200, 160, 160]


def two_people(path, seconds=6):
    """A 1280x720 talk: a red person on the left, a blue one on the right, each with a patch that moves at the mouth
    (left from 0 to 2 s and from 4 s, right from 2 to 4 s), and a quiet tone."""
    patch = "color=c=gray:size=64x48:rate=30,noise=alls=100:allf=t"
    graph = ("color=c=0xC03030:size=640x720:rate=30[l];color=c=0x3040C0:size=640x720:rate=30[r];[l][r]hstack[bg];"
             "%s[p1];%s[p2];[bg][p1]overlay=240:300:enable='lt(t,2)+gte(t,4)'[a];"
             "[a][p2]overlay=960:300:enable='between(t,2,4)'[v]" % (patch, patch))
    subprocess.run([FFMPEG, "-v", "error", "-y", "-filter_complex", graph, "-map", "[v]", "-t", str(seconds),
                    "-pix_fmt", "yuv420p", path], check=True)


def persons():
    return [{"n": 1, "box": list(LEFT), "seen": 10, "of": 10}, {"n": 2, "box": list(RIGHT), "seen": 10, "of": 10}]


class Turns(unittest.TestCase):
    def wins(self, rows):
        return [{"from": i * 0.5, "to": (i + 1) * 0.5, "who": who, "scores": scores}
                for i, (who, scores) in enumerate(rows)]

    def test_a_window_is_decided_by_who_moves_clearly_more_while_words_are_heard(self):
        quiet, left, both = [1.0] * 10, [9.0] * 5 + [1.0] * 5, [9.0] * 10
        got = talk.windows([left, quiet], 10, 0.0, [[0.0, 0.5]])
        self.assertEqual([w["who"] for w in got], [1, None], "the second half is silent: nobody is decided")
        self.assertEqual([w["who"] for w in talk.windows([[1.0] * 15 + both[:5], [1.2] * 15 + both[:5]], 10, 0.0,
                                                          [[0, 2]])][-1], 0,
                         "both moving almost alike, well over their stillness, are together")
        self.assertEqual([w["who"] for w in talk.windows([[1.5] * 5, [1.4] * 5], 10, 0.0, None)], [None],
                         "nobody clearly ahead")
        self.assertEqual([w["who"] for w in talk.windows([left[:5]], 10, 0.0, None)], [1], "one person always speaks")

    def test_each_person_is_counted_against_their_own_stillness(self):
        noisy, speaking = [8.0] * 20, [1.0] * 10 + [9.0] * 10
        self.assertEqual(talk.windows([noisy, speaking], 10, 0.0, None)[-1]["who"], 2,
                         "a mouth area that always flickers (light, a beard) does not win over the one speaking")

    def test_a_heard_line_goes_whole_to_whoever_wins_most_of_it(self):
        """Tarm 10-30, on EP0's ghost story told by Henri while he laughed: "อองรีพูด ทำไม highlight ผมล่ะ"."""
        laugh, talking, still = [1.0, 1.0, 8.0], [1.0, 3.0, 1.0], [1.0, 1.0, 1.0]
        rows = [(3, laugh), (3, laugh), (2, talking), (2, talking), (2, talking), (2, talking), (None, still),
                (0, [3.0, 3.0, 1.0]), (1, [3.0, 1.0, 1.0])]
        wins = self.wins(rows)
        got = talk.by_line(wins, [[0.0, 4.0]])
        self.assertEqual([w["who"] for w in got], [2, 2, 2, 2, 2, 2, 2, 0, 1],
                         "the line is the one telling it; a window of two together stays; after the line, its own")
        self.assertEqual([w["who"] for w in wins][:2], [3, 3], "the windows given are not changed")
        tie = [(3, laugh), (2, talking)]
        self.assertEqual([w["who"] for w in talk.by_line(self.wins(tie), [[0.0, 1.0]])], [3, 2],
                         "nobody clearly wins: as decided")
        self.assertEqual([w["who"] for w in talk.by_line(self.wins(rows), [])], [w for w, _ in rows])

    def test_a_laugh_in_the_middle_of_a_line_does_not_take_the_shot(self):
        def laugh(rec, a, b, people):
            n = round((b - a) * talk.MOTION_FPS)
            return [[1.0] * 10 + [9.0] * 15 + [1.0] * (n - 25), [2.0] * 10 + [1.5] * 15 + [6.0] * (n - 25)]

        said = [{"from": 0.0, "to": 6.0}]
        shots, _ = talk.plan(None, [[0, 6]], said, persons(), read=laugh)
        self.assertEqual([s["show"] for s in shots], [2], "person 1 laughs for 1.5 s of person 2's 6 s line")

    def test_a_laugh_holds_on_the_one_who_spoke_last_and_no_card_of_everyone_is_planned(self):
        """Tarm 10-31, shown a small card of everyone beside a cut to one person: "นั่งคุยเอาแบบซูมเข้าคนที่พูด"."""
        def laugh(rec, a, b, people):
            return [[6.0] * 40 + [1.0] * 50, [1.0] * 20 + [6.0] * 40 + [1.0] * 30]

        said = [{"from": 0.0, "to": 2.0}, {"from": 2.0, "to": 4.0}, {"from": 4.0, "to": 6.0}]
        wins = talk.plan(None, [[0, 9]], said, persons(), read=laugh)[1]
        self.assertIn(0, [w["who"] for w in wins], "the two move alike from 2 to 4 s")
        shots = talk.plan(None, [[0, 9]], said, persons(), read=laugh)[0]
        self.assertEqual([(s["from"], s["show"]) for s in shots], [(0.0, 1), (4.0, 2)],
                         "person 1 spoke last, so the laugh stays on them until person 2 speaks")
        self.assertNotIn("everyone", [p["name"] for p in talk.make((1280, 720), persons(), shots)["parts"]])

    def test_a_clip_that_opens_on_a_laugh_starts_on_the_first_to_speak(self):
        def laugh(rec, a, b, people):
            return [[6.0] * 20 + [1.0] * 70, [6.0] * 40 + [1.0] * 50]

        said = [{"from": 0.0, "to": 2.0}, {"from": 2.0, "to": 4.0}]
        shots = talk.plan(None, [[0, 9]], said, persons(), read=laugh)[0]
        self.assertEqual([s["show"] for s in shots], [2])

    def test_a_changed_shot_cuts_into_its_neighbours_and_joins_the_same_one(self):
        shots = [{"from": 0, "to": 10.2, "show": 1}, {"from": 10.2, "to": 20, "show": 2}]
        self.assertEqual(talk.with_shot(shots, 9, 10.2, 2), [{"from": 0, "to": 9, "show": 1},
                                                             {"from": 9, "to": 20, "show": 2}])
        self.assertEqual(talk.with_shot(shots, 9, 12, 0), [{"from": 0, "to": 9, "show": 1},
                                                           {"from": 9, "to": 12, "show": 0},
                                                           {"from": 12, "to": 20, "show": 2}])

    def test_short_turns_join_the_one_before_and_cuts_move_to_where_words_begin(self):
        wins = self.wins([(1, [5, 1])] * 6 + [(2, [1, 5])] + [(1, [5, 1])] * 2 + [(2, [1, 5])] * 6 + [(None, [1, 1])])
        got = talk.turns(wins, starts=[0.1, 4.3])
        self.assertEqual(got, [[0.0, 4.3, 1], [4.3, 8.0, 2]],
                         "person 2's half second at 3 s is part of person 1's turn; the cut at 4.5 s moves to 4.3")

    def test_a_long_turn_gets_a_reaction_from_the_one_listening(self):
        rows = [(1, [5.0, 1.0])] * 30
        rows[12] = (1, [5.0, 3.0])
        wins = self.wins(rows)
        got = talk.shots(wins, talk.turns(wins), 2)
        self.assertEqual(got, [{"from": 0.0, "to": 5.5, "show": 1}, {"from": 5.5, "to": 7.0, "show": 2},
                               {"from": 7.0, "to": 15.0, "show": 1}])
        rows[12] = (1, [5.0, 1.5])
        self.assertEqual(len(talk.shots(self.wins(rows), talk.turns(self.wins(rows)), 2)), 1,
                         "a listener who barely moves gets no reaction shot")

    def test_whoever_speaks_is_who_a_shot_shows_the_speaker_through_a_reaction_and_everyone_together(self):
        rows = [(1, [5.0, 1.0])] * 30
        rows[12] = (1, [5.0, 3.0])
        wins = self.wins(rows)
        shots = talk.shots(wins, talk.turns(wins), 2) + [{"from": 15.0, "to": 17.0, "show": 0},
                                                         {"from": 17.0, "to": 18.0, "show": 3},
                                                         {"from": 18.0, "to": 25.0, "show": 2}]
        self.assertEqual(talk.speaking(shots, 3), [
            {"from": 0.0, "to": 15.0, "who": [1]}, {"from": 15.0, "to": 17.0, "who": [1, 2, 3]},
            {"from": 17.0, "to": 18.0, "who": [3]}, {"from": 18.0, "to": 25.0, "who": [2]}],
            "person 1 talks on through person 2's reaction; a short shot between two others is its own")
        longer = [{"from": 0, "to": 5, "show": 1}, {"from": 5, "to": 8, "show": 2}, {"from": 8, "to": 12, "show": 1}]
        self.assertEqual([t["who"] for t in talk.speaking(longer, 2)], [[1], [2], [1]],
                         "a shot longer than a reaction is someone's turn")


class Followed(unittest.TestCase):
    """10-29, the brother's podcast in the speaker layout: the three hosts were listed as other people to decide
    about, and Glimmer blurred them all, which hid the clip. The people a talk layout follows are the clip."""

    def test_the_people_a_talk_layout_follows_are_not_listed_to_blur_and_a_passer_by_is(self):
        passer = [600, 500, 90, 90]

        def track(box, steady=False):
            return {"box": box, "steady": steady, "seen": 8, "crop": None,
                    "points": [[t, box[0], box[1], box[2], box[3]] for t in (1.0, 2.0, 3.0)]}

        found = {"samples": [None] * 10}
        with mock.patch.object(tools, "_found_faces", lambda rec, pieces, span: [(0.0, 4.0, found)]),                 mock.patch.object(tools.tracks, "tracks", lambda f, rec: [
                    track([LEFT[0] + 6, LEFT[1] - 4, 150, 166]), track(list(RIGHT)), track(passer)]):
            listed = tools._people({"width": 1280, "height": 720}, [[0.0, 4.0]], None, [0.0, 4.0],
                                   shown=[p["box"] for p in persons()])
            everyone = tools._people({"width": 1280, "height": 720}, [[0.0, 4.0]], None, [0.0, 4.0])
        self.assertEqual([p["box"] for p in listed], [passer])
        self.assertEqual(len(everyone), 3, "without the layout's people, all three are listed")


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Talk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.video = os.path.join(cls.tmp, "talk.mp4")
        two_people(cls.video)
        cls.rec = {"path": cls.video, "width": 1280, "height": 720, "size": os.path.getsize(cls.video),
                   "mtime": os.path.getmtime(cls.video)}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")

    def tearDown(self):
        recording.WORK = self.work

    def test_the_mouth_that_moves_is_measured_for_each_person(self):
        left, right = talk.motion(self.rec, 0, 4, persons())
        self.assertGreater(min(left[2:17]), 5 * max(right[2:17]), "0-2 s: the left mouth moves")
        self.assertGreater(min(right[22:37]), 5 * max(left[22:37]), "2-4 s: the right one")

    def test_the_shots_follow_whoever_speaks_and_cut_where_their_words_begin(self):
        said = [{"from": 0.2, "to": 1.9}, {"from": 2.05, "to": 3.9}, {"from": 4.1, "to": 5.8}]
        shots, wins = talk.plan(self.rec, [[0, 6]], said, persons())
        self.assertEqual([s["show"] for s in shots], [1, 2, 1])
        self.assertEqual((shots[0]["from"], shots[-1]["to"]), (0.0, 6.0))
        self.assertEqual([shots[1]["from"], shots[2]["from"]], [2.05, 4.1], "each cut is on a phrase's start")
        shots, _ = talk.plan(self.rec, [[2, 4], [0, 2]], said, persons())
        self.assertEqual([(s["from"], s["to"], s["show"]) for s in shots][:1], [(0.0, 2.0, 2)],
                         "clip seconds: the clip starts with the right person's piece")

    def test_a_piece_where_one_person_speaks_throughout_still_shows_them(self):
        def fidget(rec, a, b, people):
            n = round((b - a) * talk.MOTION_FPS)
            return [[9.0] * n, [3.0] * n] if a < 2 else [[3.0] * n, [9.0] * n]

        said = [{"from": 0.2, "to": 1.9}, {"from": 2.05, "to": 3.9}]
        shots, _ = talk.plan(None, [[0, 2], [2, 4]], said, persons(), read=fidget)
        self.assertEqual([s["show"] for s in shots], [1, 2], "resting movement is measured over the whole clip: per "
                         "piece, the speaker's own talking would be their rest and the listener who fidgets would "
                         "score close to them")

    def test_the_layout_crops_each_person_the_same_size_and_keeps_words_off_their_heads(self):
        shots = [{"from": 0, "to": 2.05, "show": 1}, {"from": 2.05, "to": 4.1, "show": 2},
                 {"from": 4.1, "to": 6, "show": 0}]
        lay = talk.make((1280, 720), persons(), shots)
        one, two, every = lay["parts"]
        self.assertEqual(one["src"][2:], two["src"][2:])
        self.assertLess(one["src"][0] + one["src"][2] / 2 - (LEFT[0] + LEFT[2] / 2), 2)
        self.assertEqual(every["name"], "everyone")
        self.assertEqual([p["name"] for p in talk.parts_at(lay, 3.0)], ["person 2"])
        self.assertEqual([p["name"] for p in talk.parts_at(lay, 5.0)], ["everyone"])
        head = layouts._head(LEFT, one["src"])
        for name, box in lay["places"].items():
            self.assertFalse(layouts._overlap(box, head), name)
        self.assertEqual(list(lay["places"]), ["top"], "the title stands above the heads, not on a chest")
        self.assertEqual(lay["places"]["top"][1], layouts.SAFE[1], "right under the phone's header")
        self.assertGreaterEqual(lay["places"]["top"][3], talk.HEADLINE_LEAST)
        with self.assertRaises(ValueError) as e:
            talk.make((1280, 720), persons()[:1], shots)
        self.assertIn("found 1 sitting still", str(e.exception))
        with self.assertRaises(ValueError):
            layouts.graph(lay, "src", "out", 1, pushes=[{"start": 0, "end": 2, "zoom": 1.1, "centre": [540, 960]}])

    def colour(self, t, lay, x=5, y=75):
        """The colour of a 20 px square at (x, y) of the frame drawn 270 px wide (a quarter of the clip's)."""
        state = project.Project([{"do": "clip", "moment": 1, "pieces": [[0, 6]], "layout": lay}]).state()
        path = render.frame(self.rec, state, t, width=270)
        raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vf", "crop=20:20:%d:%d" % (x, y), "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True).stdout
        r, b = sum(raw[0::3]) / (len(raw) / 3), sum(raw[2::3]) / (len(raw) / 3)
        return "red" if r > b + 40 else "blue" if b > r + 40 else "mixed"

    def test_a_frame_between_two_steps_shows_the_shot_of_its_time(self):
        shots = [{"from": 0, "to": 2.05, "show": 1}, {"from": 2.05, "to": 4.1, "show": 2},
                 {"from": 4.1, "to": 6, "show": 0}]
        lay = talk.make((1280, 720), persons(), shots)
        self.assertEqual(self.colour(2.04, lay), "red", "just before the cut")
        self.assertEqual(self.colour(2.06, lay), "blue", "just after it, between two 1/30 s steps")
        self.assertEqual(self.colour(1.0, lay, 220, 270), "red", "person 1 fills the frame")
        self.assertEqual(self.colour(5.0, lay, 220, 270), "blue", "everyone: the card's right side is person 2")
        self.assertEqual(self.colour(5.0, lay, 25, 270), "red", "and its left side person 1")

    def test_the_speaker_layout_keeps_the_others_in_circles_that_switch_with_the_shot(self):
        shots = [{"from": 0, "to": 2.05, "show": 1}, {"from": 2.05, "to": 4.1, "show": 2},
                 {"from": 4.1, "to": 6, "show": 0}]
        lay = talk.make((1280, 720), persons(), shots, seats=True)
        self.assertEqual(talk.said(lay["shots"]), "0-2 s person 1; 2-6 s person 2",
                         "everyone is on screen all the time, so a shot of everyone keeps the one before")
        self.assertNotIn("everyone", [p["name"] for p in lay["parts"]])
        self.assertEqual([p["name"] for p in talk.parts_at(lay, 1.0)], ["person 1", "circle 2"])
        self.assertEqual([p["name"] for p in talk.parts_at(lay, 3.0)], ["person 2", "circle 1"])
        one, two = (next(p for p in lay["parts"] if p["name"] == "circle %d" % n) for n in (1, 2))
        self.assertEqual(one["src"][2:], two["src"][2:])
        for n in ("1", "2"):
            (other, box), = lay["seats"][n]
            self.assertEqual(other, 3 - int(n))
            self.assertLess(box[1] + box[3], 600, "the circles sit at the top")
        for name, box in lay["places"].items():
            for seat in talk.SEATS:
                self.assertFalse(layouts._overlap(box, seat), "%s is clear of the circles" % name)
        x, y = [round((v + talk.CIRCLE / 2) / 4) - 10 for v in lay["seats"]["1"][0][1][:2]]
        self.assertEqual(self.colour(1.0, lay, x, y), "blue", "person 2 in the circle while person 1 speaks")
        self.assertEqual(self.colour(1.0, lay, 125, 270), "red", "person 1 fills the frame")
        x, y = [round((v + talk.CIRCLE / 2) / 4) - 10 for v in lay["seats"]["2"][0][1][:2]]
        self.assertEqual(self.colour(3.0, lay, x, y), "red", "person 1 in the circle while person 2 speaks")
        self.assertEqual(self.colour(3.0, lay, 125, 270), "blue")

    def test_the_split_layout_shows_two_people_one_above_the_other_for_the_whole_clip(self):
        lay = talk.split((1280, 720), persons())
        top, low = lay["parts"]
        self.assertEqual((top["name"], low["name"]), ("person 1", "person 2"))
        self.assertEqual(top["dst"][1] + top["dst"][3] + layouts.SEAM, low["dst"][1])
        self.assertEqual([p["name"] for p in layouts.parts_at(lay, 3.0)], ["person 1", "person 2"])
        heads = [layouts._head(LEFT, top["src"]), layouts._head(RIGHT, low["src"])]
        heads[1][1] += low["dst"][1]
        self.assertEqual(sorted(lay["bands"]), ["1", "2"], "a caption band for each of them")
        for n, head in zip(("1", "2"), heads):
            band = lay["bands"][n]
            self.assertTrue(head[1] + head[3] <= band[1] <= head[1] + head[3] + 2 * layouts.GAP,
                            "person %s's captions sit right under their face" % n)
            self.assertLessEqual(band[1] + band[3], layouts.NAME_TOP, "person %s's captions stay above the channel's name" % n)
            for other in heads:
                self.assertFalse(layouts._overlap(band, other), "a caption band covers no face")
        self.assertGreaterEqual(heads[0][1] + 0.4 * heads[0][3] / 1.4, layouts.SAFE[1], "no face under the header")
        self.assertGreaterEqual(heads[1][1], low["dst"][1] - 1, "person 2's hair is in their half, not cut at the seam")
        for part, other in ((top, RIGHT), (low, LEFT)):
            self.assertFalse(layouts._overlap(part["src"], other), "each half leaves the other face out")
        self.assertEqual(self.colour(1.0, lay, 125, 100), "red", "person 1 above")
        self.assertEqual(self.colour(1.0, lay, 125, 300), "blue", "person 2 below")
        with self.assertRaises(ValueError) as e:
            talk.split((1280, 720), persons() + [{"n": 3, "box": [600, 200, 160, 160], "seen": 10, "of": 10}])
        self.assertIn("found 3", str(e.exception))

    def test_the_split_layouts_captions_stay_above_the_channels_name_when_a_face_sits_low(self):
        for left, right, what in ((LEFT, [920, 400, 160, 160], "the lower face low in the recording"),
                                  ([200, 560, 160, 160], [920, 560, 160, 160], "both faces at its bottom")):
            people = [{"n": 1, "box": list(left), "seen": 10, "of": 10},
                      {"n": 2, "box": list(right), "seen": 10, "of": 10}]
            lay = talk.split((1280, 720), people)
            for name, band in list(lay["bands"].items()) + [("captions", lay["captions"])]:
                self.assertLessEqual(band[1] + band[3], layouts.NAME_TOP, "%s: band %s %s" % (what, name, band))


if __name__ == "__main__":
    unittest.main()
