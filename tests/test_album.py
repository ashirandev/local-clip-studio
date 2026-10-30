"""One clip's look across the album: what goes on the other clips, part by part, and what stays each clip's own."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import layouts, project, tools  # noqa: E402

FACE = [1500, 100, 200, 200]
EVERY = list(tools.ALBUM_PARTS)
LINES = [{"from": 0.0, "to": 1.0, "text": "hello"}]


def clip(kind="card", n=1):
    return {"do": "clip", "moment": n, "pieces": [[0.0, 20.0]], "layout": layouts.make(kind, (1920, 1080), face=FACE)}


def caps(**more):
    return dict({"do": "captions", "lines": LINES, "band": [1090, 160], "lit": False, "pop": False}, **more)


def state(*entries):
    return project.Project(entries).state()


def between(source, target, parts=EVERY):
    return tools.album_entries(state(*source), list(target), parts, "clean")


class AlbumEntries(unittest.TestCase):
    def test_a_clip_that_has_the_look_gets_nothing_so_its_video_stays_current(self):
        same = [clip(), caps(), {"do": "look", "name": "warm"}, {"do": "channel", "name": "mine", "spot": None}]
        self.assertEqual(between(same, same), ([], []))

    def test_only_the_parts_ticked_go(self):
        source = [clip(), {"do": "look", "name": "warm"}, {"do": "channel", "name": "mine", "spot": "bottom right"}]
        self.assertEqual(between(source, [clip()], ["colour"]), ([{"do": "look", "name": "warm"}], []))
        self.assertEqual(between(source, [clip()], ["tagline"]),
                         ([{"do": "channel", "name": "mine", "spot": "bottom right"}], []))
        self.assertEqual(between(source, [clip()], []), ([], []))

    def test_a_clip_with_no_tagline_or_colour_takes_them_off_the_others(self):
        target = [clip(), {"do": "look", "name": "warm"}, {"do": "channel", "name": "theirs", "spot": None}]
        self.assertEqual(between([clip()], target, ["tagline", "colour"])[0],
                         [{"do": "channel", "name": "", "spot": None}, {"do": "look", "name": None}])

    def test_the_tagline_moves_with_its_spot(self):
        source = [clip(), {"do": "channel", "name": "mine", "spot": "bottom right"}]
        target = [clip(), {"do": "channel", "name": "mine", "spot": "bottom left"}]
        self.assertEqual(between(source, target, ["tagline"])[0],
                         [{"do": "channel", "name": "mine", "spot": "bottom right"}])

    def test_the_style_of_the_words_goes_whole(self):
        style = {"do": "style", "name": "bold", "font": "Kanit", "fonts": {"title": "Sarabun"}, "size": 60}
        with mock.patch.object(tools.text, "fonts", return_value=[
                {"family": "Kanit", "faces": []}, {"family": "Sarabun", "faces": []}]):
            self.assertEqual(between([clip(), style], [clip()], ["words"]), ([style], []))

    def test_a_clip_with_no_style_entry_puts_the_house_style_back(self):
        theirs = [clip(), {"do": "style", "name": "bold", "font": None}]
        self.assertEqual(between([clip()], theirs, ["words"])[0], [{"do": "style", "name": "clean", "font": None}])

    def test_a_font_that_cannot_write_a_clips_letters_stays_off_it_and_is_said(self):
        style = {"do": "style", "name": "bold", "font": "Latin Only"}
        with mock.patch.object(tools.text, "fonts", return_value=[]) as fonts:
            self.assertEqual(between([clip(), style], [clip(), caps()], ["words"]),
                             ([], ["Latin Only cannot write every letter of its words"]))
        fonts.assert_called_once_with("ehlo")

    def test_captions_off_hides_the_others_and_keeps_their_lines(self):
        self.assertEqual(between([clip()], [clip(), caps()], ["captions"])[0], [{"do": "eye", "hidden": True}])
        hidden = [clip(), caps(), {"do": "eye", "hidden": True}]
        self.assertEqual(between(hidden, [clip(), caps()], ["captions"])[0], [{"do": "eye", "hidden": True}])
        self.assertEqual(between([clip()], hidden, ["captions"]), ([], []), "already off")

    def test_captions_on_shows_a_hidden_clips_own_lines_in_the_same_way(self):
        hidden = [clip(), caps(lit=True), {"do": "eye", "hidden": True}]
        self.assertEqual(between([clip(), caps(lit=True)], hidden, ["captions"])[0], [{"do": "eye", "hidden": False}])
        self.assertEqual(between([clip(), caps(pop=True)], hidden, ["captions"])[0],
                         [{"do": "eye", "hidden": False}, caps(pop=True)])

    def test_the_way_of_the_captions_goes_and_a_clips_own_lines_stay(self):
        theirs = [{"from": 2.0, "to": 3.0, "text": "their words"}]
        got = between([clip(), caps(lit=True)], [clip(), caps(lines=theirs)], ["captions"])[0]
        self.assertEqual(got, [caps(lines=theirs, lit=True)])
        self.assertEqual(between([clip(), caps()], [clip(), caps(pop=True)], ["captions"])[0], [caps()])

    def test_a_clip_with_no_captions_yet_is_said_and_none_are_made_up(self):
        self.assertEqual(between([clip(), caps()], [clip()], ["captions"]), ([], ["it has no captions yet"]))

    def test_the_face_circle_goes_to_the_same_place(self):
        source = project.Project([clip()])
        box = [56, 250, 300, 300]
        source.add({"do": "layout", "part": "circle", "box": box, "layout": layouts.moved(
            source.state()["clip"]["layout"], "circle", box)})
        got, left = tools.album_entries(source.state(), [clip()], ["circle"], "clean")
        self.assertEqual(left, [])
        self.assertEqual([(e["do"], e["part"], e["box"]) for e in got], [("layout", "circle", box)])
        self.assertEqual(tools._circle_box(state(clip(), got[0])), box)

    def test_a_clip_without_a_circle_moves_none_and_one_that_cannot_have_one_is_said(self):
        self.assertEqual(between([clip("subject")], [clip()], ["circle"]), ([], []))
        talk = dict(clip(), layout=dict(clip()["layout"], kind="talk", parts=[]))
        self.assertEqual(between([clip()], [talk], ["circle"]), ([], ["its layout has no face circle"]))

    def test_no_part_touches_a_clips_headline_or_its_effects(self):
        source = [clip(), caps(pop=True), {"do": "look", "name": "warm"}, {"do": "style", "name": "bold", "font": None},
                  {"do": "channel", "name": "mine", "spot": None},
                  {"do": "title", "text": "Their own", "start": 0.0, "end": 2.0, "place": "top"}]
        got, _ = between(source, [clip(), caps()])
        self.assertEqual(sorted(e["do"] for e in got), ["captions", "channel", "look", "style"])


class Corners(unittest.TestCase):
    def test_a_corner_is_where_the_circle_lands_so_the_panel_can_say_which_it_is_in(self):
        for kind in ("card", "subject"):
            lay = clip(kind)["layout"]
            for name in layouts.CORNERS:
                box = layouts.corner(lay, name)
                self.assertEqual(next(p["dst"] for p in layouts.moved(lay, "circle", box)["parts"]
                                      if p["name"] == "circle"), box, (kind, name))

    def test_the_bottom_corners_keep_clear_of_the_captions_and_the_phones_buttons(self):
        lay = clip("subject")["layout"]
        x, y, side, _ = layouts.corner(lay, "bottom right")
        self.assertEqual((y + side, x + side), (lay["captions"][1], layouts.BUTTONS[0]))
        self.assertEqual(layouts.corner(lay, "bottom left")[0], layouts.SAFE[0])
        self.assertEqual(layouts.corner(lay, "top right")[:2], [layouts.SAFE[0] + layouts.SAFE[2] - side, layouts.SAFE[1]])

    def test_a_circle_keeps_its_own_size(self):
        lay = layouts.moved(clip()["layout"], "circle", [100, 300, 320, 320])
        self.assertEqual(layouts.corner(lay, "top left")[2:], [320, 320])


if __name__ == "__main__":
    unittest.main()
