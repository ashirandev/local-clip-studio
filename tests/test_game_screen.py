"""A game clip and the words the game writes on its screen (screen.menu, screen.labels): the clip does not open on the
game's menu, and its title and post do not copy the screen's labels. The screen is a fake that shows the lines given.

    python -m unittest test_game_screen
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import render, screen, tools  # noqa: E402
from test_agent_loop import call  # noqa: E402
from test_map import make_moment  # noqa: E402
import test_editor  # noqa: E402

AGE = {"text": "Age32", "box": [111, 68, 113, 33], "sure": 1.0}
MENU = [{"text": "DEATH", "box": [97, 522, 113, 33], "sure": 0.98}, {"text": "GiveUp", "box": [275, 527, 76, 27],
                                                                     "sure": 0.99},
        {"text": "UNLOCK", "box": [1406, 795, 100, 27], "sure": 0.99}, AGE]
PLAY = [AGE, {"text": "SIFU", "box": [800, 40, 120, 40], "sure": 0.99},
        {"text": "Continue", "box": [900, 900, 100, 30], "sure": 0.99}]


def export(rec, state, out, on_progress=None, on_start=None, *, default):
    make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
    return out


@unittest.skipUnless(test_editor.FFMPEG, "needs ffmpeg")
class GameScreen(unittest.TestCase):
    setUpClass = classmethod(test_editor.Tools.setUpClass.__func__)
    tearDownClass = classmethod(test_editor.Tools.tearDownClass.__func__)
    run_job, to_moment = test_editor.Tools.run_job, test_editor.Tools.to_moment

    def setUp(self):
        test_editor.Tools.setUp(self)
        self.shown, self.read = PLAY, []

        def pictures(rec, times):
            source = os.path.basename(rec["path"]) == os.path.basename(self.clip)
            if source:
                self.read.append(list(times))
            return [{"at": round(t, 2), "lines": [dict(line) for line in self.shown] if source else []} for t in times]

        self.screen = mock.patch.object(screen, "_pictures", pictures)
        self.screen.start()
        self.looks, self.stretches, self.ahead = [(t, PLAY) for t in range(0, 20, 2)], [], []

        def readings(rec, a, b, on_start=None):
            if os.path.basename(rec["path"]) != os.path.basename(self.clip):
                return []
            self.stretches.append((a, b))
            return [{"at": round(a + t, 2), "lines": [dict(line) for line in lines]} for t, lines in self.looks]

        self.reader = mock.patch.object(screen, "readings", readings)
        self.reader.start()
        self.early = mock.patch.object(screen, "ahead", lambda rec, a, b: self.ahead.append((a, b)))
        self.early.start()

    def tearDown(self):
        self.early.stop()
        self.reader.stop()
        self.screen.stop()
        test_editor.Tools.tearDown(self)

    def pieces(self):
        return self.chat.project.state()["clip"]["pieces"]

    def game(self, *calls):
        script = self.to_moment()
        return self.run_job(script[0], [call("load_recipe", type="game", why="a game stream")], *script[1:], *calls,
                            "Done.")

    def test_a_game_clip_that_opens_on_the_games_menu_is_refused_once(self):
        self.shown = MENU
        self.looks = [(0, MENU), (2, MENU), (5, PLAY), (9, PLAY)]
        _, steps = self.game([call("make_clip", moment=1, layout="card", why="the death")],
                             [call("make_clip", moment=1, layout="card", why="the menu is the joke")])
        refused, made = steps[-2:]
        self.assertTrue(refused["refused"])
        self.assertIn("The clip would open on the game's menu: at 0:30 the screen shows DEATH, GiveUp, UNLOCK, and a "
                      "viewer scrolls past a menu in the first second. Start on the play that led to it: look at "
                      "0:00-0:30", refused["result"])
        self.assertEqual(self.read[0], [30.0, 31.0, 32.0], "the clip's first two seconds")
        self.assertFalse(made["refused"], made["result"])
        self.assertNotIn("menu out of the clip", made["result"], "the menu it opens on was asked for")

    def test_play_with_one_hint_on_screen_is_made(self):
        _, steps = self.game([call("make_clip", moment=1, layout="card", why="the fight")])
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertEqual(self.pieces(), [[30.0, 50.0]], "a hint left on the screen in play is not the menu")
        self.assertEqual(self.ahead, [(30.0, 50.0)], "a game's moments are read while Muse goes on")

    def test_the_games_menu_inside_the_clip_shows_for_its_first_seconds_and_play_goes_on(self):
        self.looks = [(0, PLAY), (2, PLAY), (4, MENU), (7, MENU), (10, MENU), (12, PLAY), (15, PLAY), (19, PLAY)]
        _, steps = self.game([call("make_clip", moment=1, layout="card", why="the death")])
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertEqual(self.pieces(), [[30.0, 35.5], [42.0, 50.0]])
        self.assertIn("The program cut 6.5 s of the game's menu out of the clip: the menu at 0:34 for 8 s. The clip "
                      "shows the first 1.5 s of each, the death landing, and goes on where play starts again.",
                      steps[-1]["result"])

    def test_a_menu_still_shown_at_the_moments_end_is_cut_to_the_end(self):
        self.looks = [(0, PLAY), (5, PLAY), (10, MENU), (16, MENU)]
        _, steps = self.game([call("make_clip", moment=1, layout="card", why="the death")])
        self.assertEqual(self.pieces(), [[30.0, 41.5]])
        self.assertIn("the menu at 0:40 for 10 s", steps[-1]["result"])

    def test_a_menu_shown_for_a_moment_is_left_whole(self):
        self.looks = [(0, PLAY), (4, MENU), (5.5, PLAY), (8, PLAY)]
        _, steps = self.game([call("make_clip", moment=1, layout="card", why="the fight")])
        self.assertEqual(self.pieces(), [[30.0, 50.0]])
        self.assertNotIn("menu out of the clip", steps[-1]["result"])

    def test_a_clip_that_is_not_a_game_does_not_read_its_start(self):
        self.shown = MENU
        self.looks = [(t, MENU) for t in range(0, 20, 2)]
        _, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                                "Done.")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertEqual(self.read, [])
        self.assertEqual((self.stretches, self.pieces()), ([], [[30.0, 50.0]]), "nor looks for the game's menu")

    def test_a_title_and_a_post_that_copy_the_screens_labels_are_refused_once_each(self):
        had = getattr(render, "export", None)
        render.export = export
        names = mock.patch.object(tools, "_names_check", lambda job, rec, state: (True, "no chat box in this test"))
        names.start()
        try:
            _, steps = self.game(
                [call("make_clip", moment=1, layout="card", why="the fight")],
                [call("add_title", words="ตาย AGE 35 แล้วรีสตาร์ท", start=0, end=4, place="top", why="the hook")],
                [call("add_title", words="SIFU โดนรุมทีเดียวหลายคน จะรอดไหม", start=0, end=4, place="top",
                      why="in a viewer's words")],
                [call("decide_people", keep=[1], why="it is him")],
                [call("render_clip", why="make the file")], [call("check_clip", why="check it")],
                [call("hand_over", title="SIFU โดนรุมทีเดียวหลายคน จะรอดไหม", caption="ตายตอน AGE 35 แล้วรีสตาร์ท",
                      hashtags=["#SIFU"], why="post")],
                [call("hand_over", title="SIFU โดนรุมทีเดียวหลายคน จะรอดไหม", caption="โดนรุมจนล้ม แล้วขำตัวเอง",
                      hashtags=["#SIFU"], why="post")])
        finally:
            names.stop()
            render.export = had
        copied, titled, decided, rendered, checked, post, handed = steps[-7:]
        self.assertTrue(copied["refused"])
        self.assertIn("The title copies what the game writes on its screen (AGE): a viewer reads it there already",
                      copied["result"])
        self.assertFalse(titled["refused"], "SIFU is written alone on the screen: a name, not a label")
        self.assertTrue(post["refused"])
        self.assertIn("The title or the post copies what the game writes on its screen (AGE)", post["result"])
        self.assertFalse(handed["refused"], handed["result"])

    def test_the_label_the_person_asked_for_goes_on(self):
        self.chat.emit("person", job=None, text="ใส่คำว่า AGE 35 ไว้บนหัวคลิปด้วย")
        _, steps = self.game([call("make_clip", moment=1, layout="card", why="the fight")],
                             [call("add_title", words="AGE 35", start=0, end=4, place="top", person_said="AGE 35",
                                   why="they asked")])
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])


if __name__ == "__main__":
    unittest.main()
