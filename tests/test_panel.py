"""The style panel, the words under the clip and making the video again, all without Muse, on a small video made
by ffmpeg: python -m unittest discover -s tests"""
import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, channels, faces, hearing, layouts, mind, project, recording, render, server, takes, tools  # noqa: E402,E501
from test_agent_loop import ScriptedMind, call  # noqa: E402
from test_editor import found  # noqa: E402
from test_hearing import Worker  # noqa: E402
from test_map import make_moment  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
SAID = [{"from": 2.0, "to": 4.0, "text": "this is the moment"},
        {"from": 5.0, "to": 8.0, "text": "watch him fall down now"}]
FIXED = '{"lines": ["this is the moment", "watch him fall down now"]}'


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Clip(unittest.TestCase):
    """A clip made by a scripted Muse from a 320x180 recording, with faces given by a fake face finder: one that
    stays low in the picture, and from 35 s to 40 s one that rises to the top. Its tone plays the same through words
    and silences, so the sound's level is left unknown (takes.heard_level), as before it was read."""

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
        videos = os.path.join(self.tmp, "videos")
        os.makedirs(videos)
        self.clip = os.path.join(videos, "stream.mp4")
        shutil.copy(self.source, self.clip)
        os.environ["LCS_VIDEOS"] = videos
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        self.smallest, recording.SMALLEST = recording.SMALLEST, 0
        self.worker, self.url = Worker(), hearing.URL
        hearing.URL = self.worker.url
        self.real = faces.sample, faces.kept, render.export, mind.get
        self.faces = found([(30 + i * 0.5, [[140, 10, 40, 40] if 35 <= 30 + i * 0.5 < 40 else [140, 120, 40, 40]])
                            for i in range(40)], size=(320, 180))
        self.looked, self.exported, self.based = [], [], []

        def sample(rec, a, b, every=0.5, on_start=None):
            self.looked.append((a, b))
            return self.faces

        def export(rec, state, out, on_progress=None, on_start=None, default="clean", **how):
            on_start(lambda: None)
            on_progress(state["length"] / 2)
            if how.get("words") is False:
                self.based.append((os.path.basename(out), dict(self.chat.live or {})))
            make_moment(out, seconds=20, busy=(0, 20), loud=(0, 20))
            if how.get("words") is not False:
                self.exported.append(default)
            return out

        faces.sample, faces.kept = sample, lambda rec, a, b, every=0.5: self.faces if self.looked else None
        render.export = export
        self.level, takes.heard_level = takes.heard_level, lambda rec, a=0.0, b=None: None
        self.chat = agent.Chat()
        agent.CHATS[self.chat.id] = self.chat

    def tearDown(self):
        faces.sample, faces.kept, render.export, mind.get = self.real
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

    def make(self, said=SAID, *calls):
        """The clip from the moment at 0:30-0:50, made by Muse, then calls; what was said is heard there."""
        self.worker.phrases = list(said)
        _, steps = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                [call("look_at_piece", start="0:30", end="0:50", why="the top stretch")],
                                [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "he falls"}])],
                                *([FIXED] if said else []),
                                [call("make_clip", moment=1, layout="subject", why="the action")],
                                *[[c] for c in calls], "Done.")
        self.assertFalse(any(s["refused"] for s in steps), [s["result"] for s in steps if s["refused"]])
        return steps

    def until(self, done, seconds=30):
        stop = time.time() + seconds
        while not done() and time.time() < stop:
            time.sleep(0.05)
        self.assertTrue(done(), "done within %g s" % seconds)

    def made_again(self):
        self.assertIsNone(self.chat.make_again())
        until = time.time() + 30
        while self.chat.making and time.time() < until:
            time.sleep(0.05)
        self.assertIsNone(self.chat.making, "the video is made in under 30 s")


class Panel(Clip):
    def test_the_panel_shows_each_choice_and_the_one_the_clip_has_now(self):
        self.assertEqual(self.chat.panel(), {"clip": False})
        self.make()
        panel = self.chat.panel()
        self.assertEqual([(c["name"], c["label"], c["now"], c["why"]) for c in panel["captions"]],
                         [("off", "Off", True, None), ("plain", "Plain", False, None), ("lit", "Lit", False, None),
                          ("pop", "Pop", False, None)])
        self.assertEqual([c["name"] for c in panel["styles"]], ["clean", "pop", "sticker", "glow", "3d", "classic",
                                                               "news"])
        self.assertEqual([c["label"] for c in panel["styles"]][4], "3D")
        self.assertEqual([c["name"] for c in panel["styles"] if c["now"]], ["clean"], "the house style till one is set")
        self.assertEqual([(c["label"], c["now"]) for c in panel["looks"]],
                         [("None", True), ("Vivid", False), ("Warm", False), ("Film", False)])
        self.assertTrue(all(c["why"] is None for c in panel["styles"] + panel["looks"]))
        self.assertEqual((panel["length"], panel["words"], panel["again"], panel["making"]), (18.3, [], False, False))

    def test_a_choice_the_tool_would_refuse_is_off_with_the_tools_own_reason(self):
        self.make(said=[])
        had = list(self.chat.project.entries)
        panel = self.chat.panel()
        self.assertEqual(self.chat.project.entries, had, "trying each choice changes nothing")
        why = {c["name"]: c["why"] for c in panel["captions"]}
        self.assertEqual(why, {"off": None, "plain": "Nothing was heard clearly in this moment, so there are no words "
                                                     "to caption.", "lit": why["plain"], "pop": why["plain"]})
        _, steps = self.run_job([call("add_captions", lit=True, why="karaoke")], "Done.")
        self.assertTrue(steps[0]["refused"])
        self.assertTrue(steps[0]["result"].startswith(why["lit"]), "the page and Muse never disagree")
        self.assertEqual(self.chat.set_captions("plain"), why["plain"])
        self.assertEqual(self.chat.project.entries, had)

    def test_a_tap_makes_the_tools_edit_at_once_and_undo_takes_it_back(self):
        self.make()
        for tap in (lambda: self.chat.set_captions("lit"), lambda: self.chat.set_style("pop"),
                    lambda: self.chat.set_look("warm")):
            self.assertIsNone(tap())
        state = self.chat.project.state()
        self.assertTrue(state["captions"]["lit"])
        self.assertEqual("".join(line["text"] for line in state["captions"]["lines"]),
                         "this is the momentwatch him fall down now")
        self.assertEqual((state["style"]["name"], state["look"]), ("pop", "warm"))
        edits = [e for e in self.chat.events if e["kind"] == "edit"][-3:]
        self.assertEqual([(e["job"], e["said"]) for e in edits],
                         [(None, "the lit captions"), (None, "the pop style"), (None, "the warm look")])
        self.assertEqual(self.chat.told, ["the person put on the lit captions on the page",
                                          "the person chose the pop style on the page",
                                          "the person chose the warm look on the page"])
        self.assertEqual([c["name"] for c in self.chat.panel()["captions"] if c["now"]], ["lit"])
        self.assertEqual(self.chat.undo()["do"], "look")
        self.assertIsNone(self.chat.project.state()["look"])
        self.assertEqual(self.chat.set_look("none"), "The clip already has no look.", "set_look's own refusal")
        self.assertIsNone(self.chat.set_captions("off"))
        self.assertIsNone(self.chat.project.state()["captions"])
        self.assertEqual(self.chat.told[-1], "the person turned the captions off on the page")
        self.assertEqual(agent.Chat.load(self.chat.id).project.entries, self.chat.project.entries)

    def test_a_choice_the_page_does_not_know_is_refused_in_words(self):
        self.make()
        had = list(self.chat.project.entries)
        self.assertEqual(self.chat.set_captions("loud"), "Choose Off, Plain, Lit or Pop for the captions.")
        self.assertEqual(self.chat.set_style("comic"), "Choose one of the text styles: Clean, Pop, Sticker, Glow, 3D, "
                                                       "Classic, News.")
        self.assertEqual(self.chat.set_look("sepia"), "Choose None, Vivid, Warm or Film for the colour look.")
        self.assertEqual(self.chat.project.entries, had)
        self.assertEqual(self.chat.told, [])

    def test_a_tool_that_breaks_is_a_sentence_and_the_panel_still_shows(self):
        self.make()

        def broken(job, why, look):
            raise KeyError("look")

        had = tools.TOOLS["set_look"].run
        tools.TOOLS["set_look"].run = broken
        try:
            self.assertEqual(self.chat.set_look("warm"), "The studio broke while trying that ('look').")
            looks = self.chat.panel()["looks"]
        finally:
            tools.TOOLS["set_look"].run = had
        self.assertEqual([c["why"] for c in looks][1:], ["The studio broke while trying that ('look')."] * 3)
        self.assertEqual(self.chat.told, [])

    def test_before_there_is_a_clip_a_tap_is_told_there_is_none(self):
        for tap in (lambda: self.chat.set_captions("lit"), lambda: self.chat.set_style("pop"),
                    lambda: self.chat.set_look("warm"), lambda: self.chat.make_again(),
                    lambda: self.chat.set_words("title", 0, "a", "b")):
            with self.assertRaises(agent.Missing) as e:
                tap()
            self.assertEqual(str(e.exception), "There is no clip yet. Ask Muse to make one first.")

    def test_muse_is_told_of_each_tap_before_its_next_job(self):
        self.make()
        self.chat.set_style("glow")
        brain = ScriptedMind("Fine.")
        mind.get = lambda key: brain
        self.chat.say("what changed?", None)
        self.chat.job.thread.join(10)
        said = brain.sent[0][-1]["content"]
        self.assertTrue(said.startswith("[studio] Since your last answer: the person chose the glow style on the page. "
                                        "Now: clip: moment 1"), said)
        self.assertIn("glow style", said)
        self.assertEqual(self.chat.told, [], "told once")


class Words(Clip):
    def test_every_title_and_caption_phrase_is_a_block_at_its_time(self):
        self.make(SAID, call("add_captions", why="say it"),
                  call("add_title", words="Boss down", start=0, end=3, place="top", why="the hook"))
        blocks = self.chat.panel()["words"]
        self.assertEqual([(b["kind"], b["n"], b["text"]) for b in blocks],
                         [("title", 0, "Boss down"), ("caption", 0, "this is the moment"),
                          ("caption", 1, "watch him fall down now")])
        self.assertEqual([(b["start"], b["end"]) for b in blocks], [(0, 3), (0.3, 2.3), (3.3, 6.3)])
        self.assertFalse(any(b.get("typed") for b in blocks))

    def test_the_person_changes_a_titles_words_where_it_was_and_undo_takes_it_back(self):
        self.make(SAID, call("add_title", words="Boss down", start=0, end=3, place="top", why="the hook"))
        self.assertIsNone(self.chat.set_words("title", 0, "Boss down", "  The boss\nis down "))
        self.assertEqual([(t["text"], t["start"], t["end"], t["place"]) for t in self.chat.project.state()["titles"]],
                         [("The boss is down", 0, 3, "top")], "it replaces the title it was")
        self.assertEqual(self.chat.told, ['the person typed their own words for the title at 0-3 s on the page: "Boss '
                                          'down" is now "The boss is down"'])
        self.assertIsNone(self.chat.set_words("title", 0, "The boss is down", "The boss is down"), "nothing to change")
        self.assertEqual(self.chat.undo()["text"], "The boss is down")
        self.assertEqual([t["text"] for t in self.chat.project.state()["titles"]], ["Boss down"])

    def test_the_person_drags_a_titles_ends_and_it_shows_then_in_its_place_and_undo_takes_it_back(self):
        self.make(SAID, call("add_title", words="Boss down", start=0, end=3, place="top", why="the hook"),
                  call("add_title", words="Again", start=9, end=11, place="top", why="later"))
        times = lambda: [(t["text"], t["start"], t["end"], t["place"]) for t in self.chat.project.state()["titles"]]
        self.assertIsNone(self.chat.set_title_time(0, "Boss down", 0.5, 2.5))
        self.assertEqual(times(), [("Again", 9, 11, "top"), ("Boss down", 0.5, 2.5, "top")], "the same words, moved")
        self.assertEqual(self.chat.told, ['the person dragged the ends of the title "Boss down" on the page: it shows '
                                          'from 0.5 to 2.5 s now, not 0-3 s'])
        had = list(self.chat.project.entries)
        self.assertIsNone(self.chat.set_title_time(1, "Boss down", 0.52, 2.5), "nothing to change")
        self.assertEqual(self.chat.set_title_time(1, "Boss down", 0.5, 9.5),
                         'It would run into "Again", which shows in the same place from 9 to 11 s: keep it before 9 s.')
        self.assertEqual(self.chat.set_title_time(0, "Again", 2, 11),
                         'It would run into "Boss down", which shows in the same place from 0.5 to 2.5 s: keep it '
                         'after 2.5 s.')
        self.assertEqual(self.chat.set_title_time(1, "Boss down", 0.5, 1.2),
                         "The title must stay on screen at least 1 s; 0.5-1.2 s is shorter.")
        self.assertEqual(self.chat.project.entries, had, "a refused drag changes nothing")
        wide = self.chat.project.state()["titles"][1]["box"]
        self.assertIsNone(self.chat.set_title_time(1, "Boss down", 0.5, 6))
        moved = self.chat.project.state()["titles"][1]
        self.assertEqual((moved["start"], moved["end"]), (0.5, 6))
        self.assertLess(moved["box"][2], wide[2], "the face that rises at 3.3 s keeps it in the clear part of the top")
        with self.assertRaises(agent.Missing):
            self.chat.set_title_time(0, "Boss down", 0, 3)
        self.assertEqual([self.chat.undo()["end"], self.chat.undo()["end"]], [6, 2.5])
        self.assertEqual(times(), [("Boss down", 0, 3, "top"), ("Again", 9, 11, "top")])

    def test_a_title_in_a_box_keeps_its_box_and_its_key_word_while_the_words_hold_it(self):
        self.make(SAID)
        self.chat.edit({"do": "title", "text": "Look", "start": 1.0, "end": 3.0, "place": "box at 100, 400",
                        "box": [100, 400, 400, 200], "key": "Look"})
        self.assertIsNone(self.chat.set_words("title", 0, "Look", "Look here"))
        titles = self.chat.project.state()["titles"]
        self.assertEqual([(t["text"], t["place"], t["box"], t["key"]) for t in titles],
                         [("Look here", "box at [93, 208, 463, 312] on your grid", [100, 400, 400, 200], "Look")])
        self.assertIsNone(self.chat.set_words("title", 0, "Look here", "See here"))
        self.assertEqual([(t["text"], t["key"]) for t in self.chat.project.state()["titles"]], [("See here", None)])

    def test_a_channel_look_saved_on_the_page_goes_on_every_new_clip_and_what_the_person_changes_stays(self):
        self.make(SAID, call("set_style", style="pop", why="bold"), call("set_look", look="warm", why="warm"),
                  call("add_captions", lit=True, why="lit"),
                  call("add_title", words="Boss down", start=0, end=3, place="top", why="the hook"))
        self.assertEqual(self.chat.save_channel("", ""), "Give the channel a name of 1 to 40 letters.")
        self.assertIsNone(self.chat.save_channel("Siney Society", "@sineysociety"))
        look = channels.in_use()
        self.assertEqual((look["name"], look["handle"], look["style"]["name"], look["look"], look["captions"],
                          look["title_place"]), ("Siney Society", "@sineysociety", "pop", "warm", "lit", "top"))
        self.assertEqual(self.chat.project.state()["channel"]["name"], "@sineysociety", "drawn on this clip at once")
        self.assertEqual(self.chat.set_channel_spot("top"), "Choose where the name goes: bottom left, bottom middle, "
                                                            "bottom right.")
        self.assertIsNone(self.chat.set_channel_spot("bottom right"))
        shown = self.chat.panel()["channels"]
        self.assertEqual((shown["in_use"], shown["handle"], [s["name"] for s in shown["spots"] if s["now"]]),
                         ("Siney Society", "@sineysociety", ["bottom right"]))
        self.assertEqual("the pop style, the warm colour look, lit captions, titles at the top, the name "
                         "@sineysociety drawn on the clip", shown["saved"][0]["says"], "no tool names for the person")
        self.assertEqual(channels.in_use()["spot"], "bottom right", "the spot is kept with the look")

        self.chat = agent.Chat()
        agent.CHATS[self.chat.id] = self.chat
        made = self.make(SAID)[-1]
        state = self.chat.project.state()
        self.assertEqual((state["style"]["name"], state["look"], state["channel"]),
                         ("pop", "warm", {"do": "channel", "name": "@sineysociety", "spot": "bottom right"}),
                         "a new clip in a new chat starts in the look by itself")
        self.assertIn('The clip is in the "Siney Society" channel look the person saved:', made["result"])
        self.assertIsNone(self.chat.set_style("clean"))
        _, again = self.run_job([call("make_clip", moment=1, layout="subject", why="again")], "Done.")
        self.assertFalse(again[-1]["refused"], again[-1]["result"])
        self.assertEqual(self.chat.project.state()["style"]["name"], "clean", "what the person chose on the clip stays")
        self.assertIsNone(self.chat.use_channel(""))
        self.assertIsNone(channels.in_use())
        self.assertIsNone(self.chat.project.state()["channel"], "None takes the drawn name off this clip")
        self.assertEqual(self.chat.use_channel("Nobody"), "There is no channel look called Nobody. Saved looks: "
                                                          "Siney Society.")
        self.assertIsNone(self.chat.use_channel("Siney Society"))
        self.assertEqual(self.chat.project.state()["style"]["name"], "pop", "picking a look puts all of it on")

    def test_a_channel_name_the_clip_cannot_draw_is_refused_before_anything_is_kept_and_the_clip_still_renders(self):
        wide = "@" + "W" * 29
        self.make(SAID, call("set_style", style="clean", why="plain"))
        self.assertIsNone(self.chat.save_channel("Wide", wide), "758 px fits the bottom left's 828 in clean")
        spots = {s["name"]: s["why"] for s in self.chat.panel()["channels"]["spots"]}
        self.assertIsNone(spots["bottom left"])
        self.assertIn("the bottom middle spot holds 688 px", spots["bottom middle"], "the chip says why it is off")
        self.assertIn("would not fit on the clip then", self.chat.set_channel_spot("bottom middle"))
        self.assertIsNone(self.chat.project.state()["channel"]["spot"], "the name stays where it fits")
        styles = {s["name"]: s["why"] for s in self.chat.panel()["styles"]}
        self.assertIn("853 px wide", styles["classic"], "the style chip that would push the name off says so")
        self.assertIn("853 px wide", self.chat.set_style("classic"))
        self.assertEqual(self.chat.project.state()["style"]["name"], "clean")
        self.assertTrue(render.ass(self.chat.project.state(), "clean"), "the clip with its name still draws")

        self.assertIsNone(self.chat.use_channel(None))
        self.assertIsNone(self.chat.set_style("classic"), "with no name on the clip, any style goes")
        self.assertIn("Nothing was saved: name '%s' is 853 px wide" % wide, self.chat.save_channel("Wider", wide) or "")
        self.assertIsNone(channels.get("Wider"))
        channels.use(channels.save(dict(channels.get("Wide"), style={"name": "classic", "font": None}))["name"])
        self.chat = agent.Chat()
        agent.CHATS[self.chat.id] = self.chat
        made = self.make(SAID)[-1]
        self.assertFalse(made["refused"], made["result"])
        self.assertIn("The name %s is not drawn on this clip: name '%s' is 853 px wide" % (wide, wide), made["result"])
        state = self.chat.project.state()
        self.assertEqual((state["style"]["name"], state["channel"]), ("classic", None),
                         "a look saved before the check still goes on; only the name it cannot draw stays off")
        render.ass(state, "clean")

    def test_words_a_title_would_refuse_are_refused_with_its_own_reason(self):
        self.make(SAID, call("add_title", words="Boss down", start=0, end=3, place="top", why="the hook"))
        had = list(self.chat.project.entries)
        self.assertEqual(self.chat.set_words("title", 0, "Boss down", "x" * 81),
                         "words is too long; keep it under 80 characters.")
        self.assertRegex(self.chat.set_words("title", 0, "Boss down", " ".join(["WWWWWWWW"] * 8)),
                         r"^These words take \d+ lines at the title size \(\d+ tall on your grid\), and the top place holds "
                         r"\d+: use fewer words\.$")
        self.assertEqual(self.chat.set_words("title", 0, "Boss down", " \n "), "Type the words first.")
        self.assertEqual(self.chat.project.entries, had)
        self.assertEqual(self.chat.told, [])

    def test_words_that_changed_since_the_page_showed_them_are_not_changed(self):
        self.make(SAID, call("add_title", words="Boss down", start=0, end=3, place="top", why="the hook"))
        for kind, n, was in (("title", 0, "Boss up"), ("title", 3, "Boss down"), ("caption", 0, "Boss down")):
            with self.assertRaises(agent.Missing) as e:
                self.chat.set_words(kind, n, was, "new words")
            self.assertEqual(str(e.exception), "Those words are not on the clip as the page showed them. Look at the "
                                               "clip again.")

    def test_a_caption_the_person_types_is_theirs_and_the_check_passes_it(self):
        self.make(SAID, call("add_captions", why="say it"), call("decide_people", keep=[1], why="him"))
        self.assertIsNone(self.chat.set_words("caption", 1, "watch him fall down now", "watch him fall, now"))
        lines = self.chat.project.state()["captions"]["lines"]
        self.assertEqual([(ln["phrase"], ln["text"], bool(ln.get("typed"))) for ln in lines],
                         [(0, "this is the ", False), (0, "moment", False), (1, "watch him fall, ", True),
                          (1, "now", True)])
        self.assertEqual((lines[2]["from"], lines[-1]["to"]), (3.3, 6.3), "the phrase keeps its time")
        self.assertTrue(self.chat.panel()["words"][1]["typed"])
        self.assertEqual(self.chat.told, ['the person typed their own words for the caption at 3.3-6.3 s on the page: '
                                          '"watch him fall down now" is now "watch him fall, now"'])
        _, steps = self.run_job([call("render_clip", why="make the file")], [call("check_clip", why="check")], "Done.")
        checked = [e for e in self.chat.events if e["kind"] == "checked"][-1]["checks"]
        said = [c for c in checked if c["name"] == "Captions are words that were said"][0]
        self.assertEqual((said["ok"], said["found"]), (True, "4 of 4 lines are in what was said or are the person's "
                                                             "own words, typed on the page"))

    def test_captions_put_on_again_keep_the_words_the_person_typed(self):
        self.make(SAID, call("add_captions", why="say it"))
        self.chat.set_words("caption", 0, "this is the moment", "this is it")
        self.assertIsNone(self.chat.set_captions("pop"))
        lines = self.chat.project.state()["captions"]["lines"]
        self.assertEqual([(ln["text"], bool(ln.get("typed"))) for ln in lines if ln["phrase"] == 0],
                         [("this is it", True)])
        self.assertTrue(self.chat.project.state()["captions"]["pop"])
        _, steps = self.run_job([call("add_captions", lit=True, why="karaoke")], "Done.")
        self.assertIn("The words the person typed on the page for 1 of the lines are kept as they typed them.",
                      steps[0]["result"])
        self.assertEqual([ln["text"] for ln in self.chat.project.state()["captions"]["lines"]][0], "this is it")

    def test_the_eye_hides_every_caption_and_shows_them_again_as_the_person_left_them(self):
        """Tarm 10-30, showing clips to Khun Eve: the hearing is not always right, so one tap hides every caption,
        and the lines the person fixed come back when they show again (Off emptied them)."""
        self.make(SAID, call("add_captions", why="say it"))
        self.chat.set_words("caption", 0, "this is the moment", "this is it")
        typed = self.chat.project.state()["captions"]
        self.assertEqual(self.chat.panel()["eye"], {"has": True, "hidden": False})
        self.assertIsNone(self.chat.set_eye(True))
        state = self.chat.project.state()
        self.assertIsNone(state["captions"], "off the video, the checks and the page")
        self.assertEqual(state["hidden"], typed)
        self.assertNotIn("this is it", render.ass(state) or "")
        panel = self.chat.panel()
        self.assertEqual(panel["eye"], {"has": True, "hidden": True})
        self.assertNotIn("caption", [b["kind"] for b in panel["words"]])
        self.assertEqual({c["name"]: c["why"] for c in panel["captions"] if not c["now"]},
                         dict.fromkeys(("plain", "lit", "pop"), "The captions are hidden with the eye on the clip: "
                                                                   "they show again from there."))
        self.assertEqual(self.chat.told[-1], "the person hid every caption (kept as they are, off the video until they "
                                             "show them again) with the eye on the page")
        _, steps = self.run_job([call("add_captions", lit=True, why="karaoke")], "Done.")
        self.assertTrue(steps[0]["refused"])
        self.assertIn("The person hid them, so leave them hidden", steps[0]["result"])
        self.assertIsNone(self.chat.set_eye(True), "hidden already: nothing changes")
        self.assertIsNone(self.chat.set_eye(False))
        self.assertEqual(self.chat.project.state()["captions"], typed, "as the person left them, typed lines too")
        self.assertEqual(self.chat.told[-1], "the person showed the captions again with the eye on the page")
        self.chat.set_eye(True)
        self.chat.undo()
        self.assertEqual(self.chat.project.state()["captions"], typed, "Undo takes the hiding back")
        self.chat.set_captions("off")
        self.assertEqual(self.chat.set_eye(True), "The clip has no captions to hide.")
        self.assertEqual(self.chat.panel()["eye"], {"has": False, "hidden": False})

    def test_a_look_saved_with_the_captions_hidden_keeps_them_off_and_keeps_the_clips_layout_and_length(self):
        self.make(SAID, call("add_captions", why="say it"))
        self.assertIsNone(self.chat.set_eye(True))
        self.chat.clip = {"recipe": "talk", "style": "clean", "length_s": 60}
        self.assertIsNone(self.chat.save_channel("Mine", ""))
        kind, look = "talk", channels.get("Mine")
        self.assertEqual(look["captions"], "off")
        self.assertEqual(look["layouts"], {kind: self.chat.project.state()["clip"]["layout"]["kind"]})
        self.assertEqual(look["lengths"], {kind: self.chat.clip["length_s"]})
        self.chat.clip = dict(self.chat.clip, recipe="another kind", length_s=33)
        self.assertIsNone(self.chat.save_channel("Mine", ""))
        self.assertEqual(sorted(channels.get("Mine")["lengths"].items()), sorted({kind: look["lengths"][kind],
                                                                                  "another kind": 33}.items()),
                         "saved again from another kind of clip, the first kind's stays")

    def test_the_person_can_ask_muse_to_hide_and_show_the_captions_as_the_eye_does(self):
        """Tarm 10-30: every tool on the page can be asked for in the chat. Off empties the captions; the eye keeps
        them."""
        self.make(SAID, call("add_captions", why="say it"))
        self.chat.set_words("caption", 0, "this is the moment", "this is it")
        typed = self.chat.project.state()["captions"]
        _, steps = self.run_job([call("add_captions", hidden=True, why="the person asked to hide them")],
                                [call("add_captions", hidden=True, why="again")], "Done.")
        self.assertEqual(steps[0]["result"], "Every caption is hidden and kept as it is: add_captions with hidden "
                                             "false shows them again.")
        self.assertEqual(steps[1]["result"], "The captions are already hidden.")
        self.assertEqual(self.chat.project.state()["hidden"], typed)
        self.assertEqual(self.chat.panel()["eye"], {"has": True, "hidden": True}, "the page's eye shows it")
        _, steps = self.run_job([call("add_captions", lit=True, why="karaoke")],
                                [call("add_captions", hidden=False, why="the person asked to see them")], "Done.")
        self.assertIn("add_captions with hidden false shows them as they were", steps[0]["result"])
        self.assertEqual(self.chat.project.state()["captions"], typed, "typed lines too")

    def test_a_line_the_hearing_was_not_sure_of_is_a_draft_until_the_person_keeps_or_fixes_it(self):
        self.make([dict(p, sure=0.6) for p in SAID])
        _, steps = self.run_job([call("add_captions", why="say it")], "Done.")
        self.assertFalse(steps[0]["refused"], steps[0]["result"])
        self.assertTrue(steps[0]["result"].startswith(
            'No line was heard clearly, so no captions are drawn on the video. 2 lines the hearing was not sure of are '
            'under the clip as drafts, not on the video, because a wrong caption is worse than none: '),
            steps[0]["result"])
        self.assertEqual(self.chat.project.state()["captions"]["lines"], [], "nothing is drawn")
        self.assertIn("no captions, 2 caption drafts the person has not kept yet", self.chat.clip_line())
        self.assertEqual([(b["kind"], b["n"], b["text"]) for b in self.chat.panel()["words"]],
                         [("draft", 0, "this is the moment"), ("draft", 1, "watch him fall down now")])
        self.assertIsNone(self.chat.set_words("draft", 1, "watch him fall down now", "watch him fall down now"),
                          "Save with the words as they are keeps the draft")
        caps = self.chat.project.state()["captions"]
        self.assertEqual(("".join(ln["text"] for ln in caps["lines"]), {ln.get("typed") for ln in caps["lines"]}),
                         ("watch him fall down now", {True}))
        self.assertEqual("".join(ln["text"] for ln in caps["drafts"]), "this is the moment")
        self.assertEqual(self.chat.told[-1], 'the person kept the caption draft at 3.3-6.3 s on the page, so it is on '
                                             'the video now: "watch him fall down now"')
        self.assertIsNone(self.chat.set_words("draft", 0, "this is the moment", "this is it"))
        caps = self.chat.project.state()["captions"]
        self.assertEqual([("".join(ln["text"] for ln in caps["lines"] if ln["phrase"] == k)) for k in (0, 1)],
                         ["this is it", "watch him fall down now"], "the phrases in the order they are said")
        self.assertEqual((caps["drafts"], [b["kind"] for b in self.chat.panel()["words"]]), ([], ["caption"] * 2))
        self.assertIsNone(self.chat.set_captions("pop"))
        caps = self.chat.project.state()["captions"]
        self.assertEqual(("".join(ln["text"] for ln in caps["lines"]), caps.get("drafts")),
                         ("this is itwatch him fall down now", None), "put on again, what the person kept stays")

    def test_translated_captions_change_at_a_tap_without_asking_muse_again(self):
        self.make(SAID)
        _, steps = self.run_job([call("add_captions", language="Thai", why="for Thai viewers")],
                                '{"lines": ["นี่คือช่วงนั้น", "ดูเขาล้มตอนนี้"]}', "Done.")
        self.assertFalse(steps[0]["refused"], steps[0]["result"])
        self.assertEqual([c["why"] for c in self.chat.panel()["captions"]], [None] * 4,
                         "the page cannot ask Muse, and needs not: the clip holds the translation")
        self.assertIsNone(self.chat.set_captions("lit"))
        caps = self.chat.project.state()["captions"]
        self.assertEqual((caps["lit"], caps["language"]), (True, "Thai"))
        self.assertEqual("".join(ln["text"] for ln in caps["lines"]), "นี่คือช่วงนั้นดูเขาล้มตอนนี้")
        self.assertIsNone(self.chat.set_words("caption", 1, "ดูเขาล้มตอนนี้", "ดูเขาล้มสิ"))
        self.assertIsNone(self.chat.set_captions("pop"))
        lines = self.chat.project.state()["captions"]["lines"]
        self.assertEqual("".join(ln["text"] for ln in lines), "นี่คือช่วงนั้นดูเขาล้มสิ")
        self.assertEqual({ln["said"] for ln in lines}, {"this is the moment", "watch him fall down now"})

    def test_captions_muse_could_not_translate_leave_the_taps_working(self):
        def refused(asking):
            raise mind.MindError("Muse Spark refused the question (HTTP 400)")

        self.make(SAID, call("add_captions", why="say it"))
        _, steps = self.run_job([call("load_recipe", type="talk", why="the kind")],
                                [call("add_captions", language="Thai", why="for Thai viewers")], refused, "Done.")
        self.assertFalse(steps[0]["refused"], steps[0]["result"])
        self.assertTrue(steps[1]["refused"], steps[1]["result"])
        self.assertIsNotNone(self.chat.clip)
        self.assertNotEqual(self.chat.clip.get("subtitles"), "Thai",
                            "10-29: the clip kept the language Muse never wrote, and every tap after asked for it")
        self.assertEqual([c["why"] for c in self.chat.panel()["captions"]], [None] * 4)
        self.assertIsNone(self.chat.set_captions("lit"))
        caps = self.chat.project.state()["captions"]
        self.assertEqual((caps["lit"], caps.get("language")), (True, None))

    def test_a_line_nobody_said_or_typed_still_fails_the_check(self):
        self.make(SAID, call("add_captions", why="say it"), call("decide_people", keep=[1], why="him"))
        caps = self.chat.project.state()["captions"]
        self.chat.edit(dict(caps, lines=[dict(caps["lines"][0], text="made up words")] + caps["lines"][1:]))
        self.run_job([call("render_clip", why="make the file")], [call("check_clip", why="check")], "Done.")
        checked = [e for e in self.chat.events if e["kind"] == "checked"][-1]["checks"]
        said = [c for c in checked if c["name"] == "Captions are words that were said"][0]
        self.assertEqual((said["ok"], said["found"]), (False, "3 of 4 lines are in what was said; not said: made up "
                                                              "words"))


class Again(Clip):
    def test_the_video_is_made_again_after_the_clip_changed_without_muse(self):
        self.make(SAID, call("decide_people", keep=[1], why="him"))
        self.assertFalse(self.chat.panel()["again"], "nothing finished yet: Muse makes the first file")
        self.run_job([call("render_clip", why="make the file")], "Done.")
        self.assertFalse(self.chat.panel()["again"], "the file is the clip as it is")
        self.chat.set_style("pop")
        self.assertTrue(self.chat.panel()["again"])
        self.made_again()
        self.assertEqual(self.exported, ["clean", "clean"], "the same render as render_clip's")
        made = [e for e in self.chat.events if e["kind"] == "rendered"]
        self.assertEqual((len(made), made[-1]["job"]), (2, None))
        self.assertEqual(self.chat.finished()["name"], made[-1]["file"]["name"])
        self.assertFalse(self.chat.panel()["again"])
        self.assertEqual(self.chat.told[-1], "the person made the video again on the page")
        self.assertTrue(self.chat.make_again().startswith("The finished file is already made from the clip as it is "
                                                          "now"))
        _, steps = self.run_job([call("check_clip", why="check")],
                                [call("hand_over", title="Down", caption="He falls.", hashtags=["#a"], why="post")],
                                "Done.")
        self.assertFalse(any(s["refused"] for s in steps), [s["result"] for s in steps])
        self.assertEqual(agent.Chat.load(self.chat.id).rendered, self.chat.rendered)

    def test_a_video_made_before_the_program_was_updated_is_made_again_from_the_clip_as_it_is(self):
        self.make(SAID, call("decide_people", keep=[1], why="him"))
        self.run_job([call("render_clip", why="make the file")], "Done.")
        old = self.chat.finished()["name"]
        with mock.patch.object(project, "program", lambda: "an update"):
            self.assertIsNone(self.chat.finished(), "10-30: the file the old code drew was given back as the clip")
            self.assertTrue(self.chat.panel()["again"])
            _, steps = self.run_job([call("render_clip", why="make the file")], "Done.")
            self.assertTrue(steps[0]["result"].startswith("Rendered the finished clip"), steps[0]["result"])
            self.assertNotEqual(self.chat.finished()["name"], old)
        self.assertEqual(self.exported, ["clean", "clean"])
        self.assertTrue(os.path.isfile(os.path.join(render.folder("clips"), old)), "the file handed over before stays")

    def test_the_programs_fingerprint_is_the_files_that_draw_the_video_and_not_pythons_cache(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)

        def put(path, data):
            os.makedirs(os.path.dirname(os.path.join(root, path)), exist_ok=True)
            with open(os.path.join(root, path), "wb") as f:
                f.write(data)

        put("clipstudio/render.py", b"draw")
        put("fonts/Kanit-Bold.ttf", b"font")
        put("web/app.js", b"page")
        first = project.fingerprint(root)
        put("clipstudio/__pycache__/render.cpython-311.pyc", b"run")
        put("web/app.js", b"another page")
        self.assertEqual(project.fingerprint(root), first, "what has run, and the page, do not draw the video")
        put("fonts/Kanit-Bold.ttf", b"another font")
        self.assertNotEqual(project.fingerprint(root), first)
        put("fonts/Kanit-Bold.ttf", b"font")
        put("cutting/cut.py", b"the worker")
        self.assertNotEqual(project.fingerprint(root), first)
        here = os.path.dirname(os.path.dirname(os.path.abspath(project.__file__)))
        self.assertEqual(project.program(), project.fingerprint(here))

    def test_the_video_is_not_made_again_while_someone_is_undecided_or_muse_works(self):
        self.make(SAID)
        self.assertEqual(self.chat.make_again(), "Decide about each person in the clip first: tap a face to blur it, "
                                                 "and tap it again to show it.")
        self.chat.set_person(1, False)
        self.chat.job = type("Working", (), {"done": False})()
        self.assertEqual(self.chat.make_again(), "Muse is working on the clip now. Wait until it is done, or press "
                                                 "Stop.")
        self.assertEqual(self.exported, [])

    def test_a_video_that_could_not_be_made_says_why(self):
        self.make(SAID, call("decide_people", keep=[1], why="him"))

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            raise render.NotDrawn("FFmpeg could not make the clip: no room on the disk")

        render.export = export
        self.made_again()
        self.assertEqual([e["error"] for e in self.chat.events if e["kind"] == "not_made"],
                         ["The clip could not be rendered: FFmpeg could not make the clip: no room on the disk"])
        self.assertIsNone(self.chat.finished())
        self.assertNotIn("the person made the video again on the page", self.chat.told)

    def test_stop_ends_the_video_being_made(self):
        self.make(SAID, call("decide_people", keep=[1], why="him"))
        started, stopped = threading.Event(), threading.Event()

        def export(rec, state, out, on_progress=None, on_start=None, *, default):
            on_start(stopped.set)
            started.set()
            stopped.wait(10)
            raise render.Stopped()

        render.export = export
        self.assertIsNone(self.chat.make_again())
        self.assertTrue(started.wait(10))
        self.assertEqual(self.chat.make_again(), "The video is being made now.")
        self.assertTrue(self.chat.stop())
        until = time.time() + 10
        while self.chat.making and time.time() < until:
            time.sleep(0.05)
        self.assertEqual([e["error"] for e in self.chat.events if e["kind"] == "not_made"],
                         ["Not rendered: the person pressed Stop."])


class Served(Clip):
    """A clip with the studio's server running on a free port."""

    def setUp(self):
        super().setUp()
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        super().tearDown()

    def ask(self, method, path, body=None, kind="application/json"):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=30)
        try:
            data = json.dumps(body) if body is not None else None
            conn.request(method, path, data, {"Content-Type": kind} if data else {})
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

    def fetch(self, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=30)
        try:
            conn.request("GET", path, None, headers or {})
            resp = conn.getresponse()
            return resp.status, resp.getheader("Content-Type"), resp.read()
        finally:
            conn.close()


class Frame(Clip):
    def whole(self):
        return next(p for p in self.chat.project.state()["clip"]["layout"]["parts"] if p["name"] == "whole")

    def test_the_person_moves_the_upright_frame_and_it_holds_there_until_undo(self):
        self.make()
        frame = self.chat.panel()["frame"]
        self.assertEqual((frame["why"], frame["size"], frame["width"], frame["follows"]), (None, [320, 180], 102, False))
        self.assertTrue(os.path.isfile(os.path.join(recording.WORK, "frames", frame["picture"])))
        self.assertIsNone(self.chat.set_frame(0), "Tarm 10-30: the frame further to the left")
        self.assertEqual((self.whole()["src"][0], self.chat.panel()["frame"]["x"]), (0, 0))
        self.assertIn("moved the upright frame on the page: it now shows 0% to 32% of the recording's width",
                      self.chat.told[-1])
        self.assertIsNone(self.chat.set_frame(5000))
        self.assertEqual(self.whole()["src"][0], 320 - 102, "the frame stays on the picture")
        self.chat.undo()
        self.chat.undo()
        self.assertEqual(self.whole()["src"][0], frame["x"], "two undos put it back where Muse made it")

    def test_a_frame_that_followed_the_product_holds_where_the_person_puts_it(self):
        self.make()
        lay = self.chat.project.state()["clip"]["layout"]
        plan = {"shots": [{"from": 0, "to": 10, "x": 0, "y": 0}, {"from": 10, "to": 20, "x": 200, "y": 0}],
                "size": [102, 180]}
        followed = dict(layouts.followed(lay, plan, "whole"), thing={"what": "white game controller"})
        self.chat.edit({"do": "layout", "part": "product", "box": None, "layout": followed})
        self.assertTrue(self.chat.panel()["frame"]["follows"])
        self.assertIsNone(self.chat.set_frame(60))
        self.assertEqual((self.whole()["src"][0], "follow" in self.whole()), (60, False))
        self.assertNotIn("thing", self.chat.project.state()["clip"]["layout"],
                         "lane 5: the check would time the product against the follow the person left")
        self.assertFalse(self.chat.panel()["frame"]["follows"])
        self.assertTrue(self.chat.told[-1].endswith("(it no longer follows the product)"))

    def test_a_frame_the_person_placed_stays_when_the_ends_move(self):
        self.make()
        self.assertIsNone(self.chat.set_frame(0))
        self.assertIsNone(self.chat.set_ends(31.0, 49.0))
        self.until(lambda: not self.chat.making)
        self.assertEqual(self.chat.project.state()["clip"]["pieces"][0][0], 31.0)
        self.assertEqual(self.whole()["src"][0], 0, "the Trim tab made the clip again where the person had framed it")

    def test_a_drag_says_on_the_page_where_it_landed_and_what_stayed_loud(self):
        """Tarm 10-30: a cut only where the sound is quiet. Lane 4: show the grip landing there."""
        from unittest import mock
        self.make()

        def land(rec, made, level, last_end=False):
            return [[33.0, made[0][-1][1]]], None

        with mock.patch.object(tools, "_on_quiet", land), mock.patch.object(tools, "_loud_cuts", lambda *a: [49.0]), \
                mock.patch.object(tools.takes, "heard_level", return_value=(0.0, [-20.0])):
            self.assertIsNone(self.chat.set_ends(34.8, 49.0))
            self.until(lambda: not self.chat.making)
        self.assertEqual(self.chat.panel()["ends"]["start"], 33.0, "the Trim tab's grip reads where it landed")
        moved = [e for e in self.chat.events if e["kind"] == "moved"][-1]
        self.assertEqual(moved["said"], "The program moved 1 cut, at most 1.8 s, into the nearest quiet sound, so none "
                                        "falls inside a word. Nothing is quiet within 1 s of the cut at 0:49.0: it is "
                                        "on sound as loud as the talk.")

    def test_a_clip_without_an_upright_frame_says_why_and_is_not_changed(self):
        self.make()
        for layout in ("card", "face"):
            _, steps = self.run_job([call("make_clip", moment=1, layout=layout, why="another layout")], "Done.")
            self.assertFalse(steps[0]["refused"], steps[0]["result"])
            why = self.chat.panel()["frame"]["why"]
            self.assertIn("subject layout", why, layout)
            entries = len(self.chat.project.entries)
            self.assertEqual(self.chat.set_frame(0), why, "the face layout's picture is cut around the face")
            self.assertEqual(len(self.chat.project.entries), entries)


class Endpoints(Served):
    def test_a_cover_is_sent_by_its_name_as_a_jpeg_and_nothing_else_reaches_the_disk(self):
        folder = render.folder("covers")
        for name in ("0123456789abcdef.jpg", "0123456789abcdef.2.jpg", "0123456789abcdef.x.jpg"):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(b"\xff\xd8\xff a cover")
        for name in ("0123456789abcdef.jpg", "0123456789abcdef.2.jpg"):
            self.assertEqual(self.fetch("/api/cover/" + name), (200, "image/jpeg", b"\xff\xd8\xff a cover"))
        for name in ("0123456789abcdeX.jpg", "0123456789abcdef.png", "..%5C0123456789abcdef.jpg",
                     "0123456789abcdef.x.jpg", "fedcba9876543210.jpg"):
            self.assertEqual(self.fetch("/api/cover/" + name)[0], 404, name)

    def test_the_panel_is_read_with_a_get(self):
        status, panel = self.ask("GET", "/api/panel?chat=" + self.chat.id)
        self.assertEqual((status, panel), (200, {"ok": True, "clip": False}))
        self.make()
        status, panel = self.ask("GET", "/api/panel?chat=" + self.chat.id)
        self.assertEqual((status, panel["ok"], len(panel["captions"]), len(panel["styles"]), len(panel["looks"])),
                         (200, True, 4, 7, 4))
        status, answer = self.ask("GET", "/api/panel?chat=nope")
        self.assertEqual((status, answer["error"]), (404, server.GONE))

    def test_each_colour_look_choice_is_a_picture_of_the_clip_in_that_look(self):
        self.make()
        base = "/api/clipframe?chat=%s&t=1&width=160" % self.chat.id
        got = {look: self.fetch(base + "&look=" + look) for look in agent.COLOUR_LOOKS}
        self.assertEqual({k: (s, kind) for k, (s, kind, _) in got.items()},
                         {k: (200, "image/jpeg") for k in agent.COLOUR_LOOKS})
        state, default = self.chat.project.state(), self.chat.clip["style"] if self.chat.clip else "clean"
        for look, (_, _, body) in got.items():
            with open(render.frame(self.chat.recording, dict(state, look=None if look == "none" else look), 1,
                                   default, width=160), "rb") as f:
                self.assertEqual(body, f.read(), "the picture is the clip drawn in the %s look" % look)
        self.assertEqual(got["none"][2], self.fetch(base)[2], "None is the clip with no look, as it stands now")
        self.assertIsNone(self.chat.project.state()["look"], "a picture of a look changes nothing")
        self.assertEqual(self.fetch(base + "&look=sepia")[0], 404)

    def test_each_tap_is_a_json_post_that_answers_ok_or_says_why(self):
        status, answer = self.ask("POST", "/api/style", {"chat": self.chat.id, "style": "pop"})
        self.assertEqual((status, answer["error"]), (404, "There is no clip yet. Ask Muse to make one first."))
        self.make(SAID, call("add_captions", why="say it"))
        for path, body in (("/api/captions", {"mode": "pop"}), ("/api/style", {"style": "glow"}),
                           ("/api/colour", {"look": "film"})):
            self.assertEqual(self.ask("POST", path, dict(body, chat=self.chat.id)), (200, {"ok": True}), path)
        state = self.chat.project.state()
        self.assertEqual((state["captions"]["pop"], state["style"]["name"], state["look"]), (True, "glow", "film"))
        for path, body, said in (("/api/captions", {"mode": "loud"}, "Choose Off, Plain, Lit or Pop for the captions."),
                                 ("/api/style", {}, "Choose one of the text styles: Clean, Pop, Sticker, Glow, 3D, "
                                                    "Classic, News."),
                                 ("/api/colour", {"look": "film"}, "The clip already has the film look.")):
            self.assertEqual(self.ask("POST", path, dict(body, chat=self.chat.id)),
                             (400, {"ok": False, "error": said}), path)
        self.assertEqual(self.ask("POST", "/api/captions", {"chat": self.chat.id, "hidden": True}), (200, {"ok": True}))
        self.assertTrue(self.chat.project.state()["hidden"])
        self.assertEqual(self.ask("POST", "/api/captions", {"chat": self.chat.id, "hidden": False}), (200, {"ok": True}))
        self.assertTrue(self.chat.project.state()["captions"]["pop"])
        self.assertEqual(self.ask("POST", "/api/captions", {"chat": "nope", "mode": "pop"})[0], 404)
        self.assertEqual(self.ask("POST", "/api/style", {"chat": self.chat.id, "style": "pop"}, "text/plain")[0], 415)

    def test_a_part_is_muted_with_a_post_of_its_seconds_and_muse_is_told(self):
        self.make(SAID, call("add_captions", why="say it"))
        panel = "/api/panel?chat=" + self.chat.id
        self.assertEqual(self.ask("GET", panel)[1]["sound"], {"mutes": [], "has": True})
        self.assertEqual(self.ask("POST", "/api/mute", {"chat": self.chat.id, "start": 0, "end": 1.5}),
                         (200, {"ok": True}))
        self.assertEqual(self.ask("GET", panel)[1]["sound"]["mutes"], [[0, 1.5]])
        self.assertIn("the person muted the sound from 0 to 1.5 s of the clip on the page", self.chat.told)
        numbers = "Send the start and the end of the part to mute, in seconds of the clip."
        for body, said in (({"start": "0", "end": 1}, numbers), ({"start": 0, "end": True}, numbers),
                           ({"start": 5, "end": 6, "back": True}, "The sound is not muted anywhere from 5 to 6 s.")):
            self.assertEqual(self.ask("POST", "/api/mute", dict(body, chat=self.chat.id)),
                             (400, {"ok": False, "error": said}), body)
        self.assertEqual(self.ask("POST", "/api/mute", {"chat": self.chat.id, "start": 0, "end": 1.5, "back": True}),
                         (200, {"ok": True}))
        self.assertEqual(self.chat.project.state()["mutes"], [])

    def test_a_clip_of_the_album_is_opened_with_a_post(self):
        self.make()
        self.assertEqual(self.ask("POST", "/api/album", {"chat": self.chat.id, "n": "1"}),
                         (400, {"ok": False, "error": "Send which clip of the album to open: its number."}))
        self.assertEqual(self.ask("POST", "/api/album", {"chat": self.chat.id, "n": 1}),
                         (400, {"ok": False, "error": "That clip is not in the album."}))
        self.assertIsNone(self.ask("GET", "/api/panel?chat=" + self.chat.id)[1]["album"])
        other = tools.project.Project(self.chat.project.entries)
        self.chat.fill_album([{"moment": 1, "title": "One", "project": self.chat.project},
                              {"moment": 2, "title": "Two", "project": other}], 0)
        self.assertEqual(self.ask("POST", "/api/album", {"chat": self.chat.id, "n": 1}), (200, {"ok": True}))
        album = self.ask("GET", "/api/panel?chat=" + self.chat.id)[1]["album"]
        self.assertEqual((album["at"], [c["title"] for c in album["clips"]]), (1, ["One", "Two"]))
        self.assertEqual([c["state"] for c in album["clips"]], ["unfinished", "unfinished"], "nothing handed over yet")

    def album(self):
        other = tools.project.Project(self.chat.project.entries)
        self.chat.fill_album([{"moment": 1, "title": "One", "project": self.chat.project},
                              {"moment": 2, "title": "Two", "project": other}], 0)
        return other

    def post(self, path, **body):
        return self.ask("POST", path, dict(body, chat=self.chat.id))

    def test_the_album_tab_uses_a_clips_look_takes_it_back_and_says_what_it_did(self):
        self.make()
        other = self.album()
        self.assertIsNone(self.ask("GET", "/api/panel?chat=" + self.chat.id)[1]["circle"], "this clip has none")
        self.assertEqual(self.post("/api/tagline", words="Daily notes"), (200, {"ok": True}))
        self.assertEqual(self.chat.project.state()["channel"]["name"], "Daily notes")
        self.assertIn('the person set the tagline of the clip to "Daily notes" on the page', self.chat.told)
        self.assertEqual(self.post("/api/album/look", parts="tagline"),
                         (400, {"ok": False, "error": "Send which parts to put on the other clips."}))
        self.assertEqual(self.post("/api/album/look", parts=["headline"])[0], 400)
        self.assertEqual(self.post("/api/album/look", parts=[])[0], 400)
        self.assertIsNone(other.state()["channel"])
        self.assertEqual(self.post("/api/album/look", parts=["tagline"]), (200, {"ok": True}))
        self.assertEqual(other.state()["channel"]["name"], "Daily notes")
        album = self.ask("GET", "/api/panel?chat=" + self.chat.id)[1]["album"]
        self.assertEqual((album["matched"], [p["name"] for p in album["parts"]]), (True, list(tools.ALBUM_PARTS)))
        self.assertTrue(self.chat.told[-1].startswith("the person used this clip's look on the whole album on the page"))
        self.assertEqual(self.post("/api/album/back"), (200, {"ok": True}))
        self.assertIsNone(other.state()["channel"])
        self.assertEqual(self.post("/api/album/back"),
                         (400, {"ok": False, "error": "There is no album-wide look to take back."}))

    def test_the_album_waits_for_a_video_being_made_and_for_muse(self):
        self.make()
        self.album()
        self.chat.making = object()
        for path, body in (("/api/album/look", {"parts": ["colour"]}), ("/api/album/back", {}), ("/api/album/make", {})):
            self.assertEqual(self.post(path, **body), (400, {"ok": False, "error": "The video is being made now. Wait until "
                                                                                   "it is done."}))
        self.chat.making = None
        with mock.patch.object(self.chat, "busy", return_value=True):
            self.assertEqual(self.post("/api/album/make")[1]["error"],
                             "Muse is working on the clip now. Wait until it is done, or press Stop.")

    def test_update_all_videos_runs_in_the_background_and_says_how_it_went(self):
        self.make(SAID, call("decide_people", keep=[1], why="him"))
        other = self.album()
        self.assertEqual(self.post("/api/album/make"),
                         (400, {"ok": False, "error": "No clip of the album has changed since its video was made."}))
        self.made_again()
        other.add({"do": "look", "name": "warm"})
        self.assertEqual(self.post("/api/album/make"), (200, {"ok": True}))
        self.until(lambda: any(e["kind"] == "album_made" for e in self.chat.events))
        made = [e for e in self.chat.events if e["kind"] == "album_made"][-1]
        self.assertEqual(made["said"], "Updated 1 video.", "for the person: Muse's next step is not theirs")
        self.assertEqual(agent._album_said("Clip 1: Rendered it.\nClip 2: Decide first.\nClip 3: Rendered it."),
                         "Updated 2 videos. Clip 2: Decide first.")
        self.assertEqual(agent._album_said("Clip 1: Not rendered: Stop."), "Clip 1: Not rendered: Stop.")
        self.assertIsNone(self.chat.making)
        self.assertEqual(self.chat.album_at, 0)

    def test_the_face_circle_goes_to_a_corner_with_a_post(self):
        self.make(SAID, call("move_part", part="circle", box=[60, 300, 400, 500], why="her face"))
        self.assertEqual(self.post("/api/circle", spot="middle")[0], 400)
        self.assertEqual(self.post("/api/circle", spot="bottom right"), (200, {"ok": True}))
        lay = self.chat.project.state()["clip"]["layout"]
        self.assertEqual(tools._circle_box(self.chat.project.state()), layouts.corner(lay, "bottom right"))
        self.assertEqual([c["now"] for c in self.ask("GET", "/api/panel?chat=" + self.chat.id)[1]["circle"]],
                         [False, False, False, True])
        self.chat.undo()
        self.assertNotEqual(tools._circle_box(self.chat.project.state()), layouts.corner(lay, "bottom right"))
        self.assertIn("the person moved the face circle to the bottom right on the page", self.chat.told)

    def test_a_clip_without_a_face_circle_says_so(self):
        self.make()
        self.assertEqual(self.post("/api/circle", spot="top left"),
                         (400, {"ok": False, "error": "This clip has no face circle."}))

    def test_the_clips_ends_are_moved_with_a_post_and_muse_is_told(self):
        self.make(SAID, call("add_captions", why="say it"))
        panel = "/api/panel?chat=" + self.chat.id
        ends = self.ask("GET", panel)[1]["ends"]
        self.assertEqual((ends["start"], ends["end"], ends["shortest"], ends["longest"], ends["why"]),
                         (31.7, 50.0, 10, 90, None))
        self.assertEqual(self.ask("POST", "/api/ends", {"chat": self.chat.id, "start": "34", "end": 50}),
                         (400, {"ok": False, "error": "Send the new start and end of the clip, in seconds of the "
                                                      "recording."}))
        self.assertEqual(self.ask("POST", "/api/ends", {"chat": self.chat.id, "start": 34.8, "end": 50}),
                         (200, {"ok": True}))
        self.until(lambda: self.chat.making is None and any(e["kind"] == "moved" for e in self.chat.events))
        self.assertEqual(self.ask("GET", panel)[1]["ends"]["start"], 34.8)
        self.assertIn("watch him", "".join(ln["text"] for ln in self.chat.project.state()["captions"]["lines"]))
        self.assertTrue(any("the person moved the clip's ends on the page: it now runs from 0:35 to 0:50" in t
                            for t in self.chat.told), self.chat.told)
        self.assertEqual(self.ask("POST", "/api/ends", {"chat": self.chat.id, "start": 45, "end": 50}),
                         (200, {"ok": True}))
        self.until(lambda: self.chat.making is None and any(e["kind"] == "not_moved" for e in self.chat.events))
        refused = [e for e in self.chat.events if e["kind"] == "not_moved"][-1]
        self.assertTrue(refused["error"].startswith("From 0:45.0 to 0:50.0 is 5 s"), refused)

    def test_the_layout_tab_makes_the_clip_again_in_the_layout_tapped_and_undo_takes_it_back(self):
        """Tarm 10-30: "ให้คนเลือก layout เองง่ายที่สุด": one tap, the clip made again in that layout, no Muse."""
        self.make(SAID, call("add_title", words="Boss down", start=0, end=4, place="top", why="the hook"))
        panel = "/api/panel?chat=" + self.chat.id
        shown = self.ask("GET", panel)[1]["layouts"]
        self.assertEqual([(c["name"], c["label"], c["now"], c["picture"]) for c in shown],
                         [("card", "Card", False, True), ("stack", "Stack", False, True),
                          ("face", "Face", False, True), ("subject", "Fill", True, True)])
        state = self.chat.project.state()
        standing = dict(state, clip=dict(state["clip"], layout=dict(state["clip"]["layout"], size=[180, 320])))
        self.assertEqual([c["name"] for c in self.chat.layouts_now(standing)], ["face", "subject"],
                         "a recording standing up: a card or a stack would show its face twice")
        recipe = self.chat.clip
        self.chat.clip = dict(recipe or {}, recipe="talk")
        self.assertEqual([(c["name"], c["picture"]) for c in self.chat.layouts_now(state)][4:],
                         [("talk", False), ("speaker", False), ("split", False)], "people talking take the talk layouts")
        self.chat.clip = recipe
        faceless = dict(state, clip=dict(state["clip"], layout=dict(state["clip"]["layout"], face=None)))
        self.assertEqual([(c["name"], c["picture"]) for c in self.chat.layouts_now(faceless)],
                         [("card", True), ("stack", False), ("face", False), ("subject", True)],
                         "no face found yet: the stack and the face cannot be drawn until a tap finds it")
        base = "/api/clipframe?chat=%s&t=1&width=160" % self.chat.id
        got = self.fetch(base + "&layout=card")
        self.assertEqual(got[:2], (200, "image/jpeg"))
        self.assertNotEqual(got[2], self.fetch(base)[2], "the picture is the clip drawn in the card")
        self.assertEqual(self.fetch(base + "&layout=talk")[0], 404)
        self.assertEqual(self.chat.project.state()["clip"]["layout"]["kind"], "subject", "a picture changes nothing")
        self.assertEqual(self.ask("POST", "/api/layout", {"chat": self.chat.id, "layout": "split"}),
                         (400, {"ok": False, "error": "Choose one of the layouts: Card, Stack, Face, Fill."}))
        self.assertEqual(self.ask("POST", "/api/layout", {"chat": self.chat.id, "layout": "subject"}),
                         (200, {"ok": True}), "the layout it has already")
        self.assertEqual(self.ask("POST", "/api/layout", {"chat": self.chat.id, "layout": "card"}), (200, {"ok": True}))
        self.until(lambda: self.chat.making is None and any(e["kind"] == "relaid" for e in self.chat.events))
        state = self.chat.project.state()
        self.assertEqual(state["clip"]["layout"]["kind"], "card")
        self.assertEqual([t["text"] for t in state["titles"]], ["Boss down"], "the words stay")
        self.assertTrue(any("the person chose the card layout on the page: the clip was made again in it from "
                            "moment 1" in t for t in self.chat.told), self.chat.told)
        self.assertEqual([c["name"] for c in self.ask("GET", panel)[1]["layouts"] if c["now"]], ["card"])
        self.chat.undo()
        self.assertEqual(self.chat.project.state()["clip"]["layout"]["kind"], "subject", "one Undo puts it back")

    def test_words_are_changed_with_a_post_naming_the_block_and_its_words(self):
        self.make(SAID, call("add_captions", why="say it"))
        body = {"chat": self.chat.id, "kind": "caption", "n": 0, "was": "this is the moment", "words": "this is it"}
        self.assertEqual(self.ask("POST", "/api/words", body), (200, {"ok": True}))
        status, answer = self.ask("POST", "/api/words", body)
        self.assertEqual((status, answer["error"]), (404, "Those words are not on the clip as the page showed them. "
                                                          "Look at the clip again."))
        for wrong in ({"kind": "sticker"}, {"n": "0"}, {"n": True}):
            status, answer = self.ask("POST", "/api/words", dict(body, **wrong))
            self.assertEqual((status, answer["error"]), (400, "Send which words to change: a title or a caption, "
                                                              "and its number."), wrong)
        status, answer = self.ask("POST", "/api/words", dict(body, was="this is it", words=""))
        self.assertEqual((status, answer["error"]), (400, "Type the words first."))

    def test_a_titles_time_is_changed_with_a_post_naming_it_and_its_words(self):
        self.make(SAID, call("add_title", words="Boss down", start=0, end=3, place="top", why="the hook"))
        body = {"chat": self.chat.id, "n": 0, "was": "Boss down", "start": 0.5, "end": 2.5}
        self.assertEqual(self.ask("POST", "/api/title/time", body), (200, {"ok": True}))
        self.assertEqual([(t["start"], t["end"]) for t in self.chat.project.state()["titles"]], [(0.5, 2.5)])
        status, answer = self.ask("POST", "/api/title/time", dict(body, was="Boss up"))
        self.assertEqual((status, answer["error"]), (404, agent.WORDS_GONE))
        for wrong, said in (({"start": "0"}, "Send the title's new start and end, in seconds of the clip."),
                            ({"end": True}, "Send the title's new start and end, in seconds of the clip."),
                            ({"n": "0"}, "Send which title to move: its number."),
                            ({"n": True}, "Send which title to move: its number."),
                            ({"end": 0.9}, "The title must stay on screen at least 1 s; 0.5-0.9 s is shorter.")):
            self.assertEqual(self.ask("POST", "/api/title/time", dict(body, **wrong)),
                             (400, {"ok": False, "error": said}), wrong)

    def test_the_frame_is_moved_with_a_post_of_its_left_edge(self):
        self.make()
        self.assertEqual(self.ask("POST", "/api/frame", {"chat": self.chat.id, "x": 10}), (200, {"ok": True}))
        whole = next(p for p in self.chat.project.state()["clip"]["layout"]["parts"] if p["name"] == "whole")
        self.assertEqual(whole["src"][0], 10)
        for wrong in ("10", True, None, 1e12):
            self.assertEqual(self.ask("POST", "/api/frame", {"chat": self.chat.id, "x": wrong}), (400, {
                "ok": False, "error": "Send where the frame's left edge goes, in pixels of the recording."}), wrong)

    def test_the_video_is_made_again_with_a_post(self):
        self.make(SAID)
        status, answer = self.ask("POST", "/api/remake", {"chat": self.chat.id})
        self.assertEqual((status, answer["error"]), (400, "Decide about each person in the clip first: tap a face to "
                                                          "blur it, and tap it again to show it."))
        self.chat.set_person(1, False)
        self.assertEqual(self.ask("POST", "/api/remake", {"chat": self.chat.id}), (200, {"ok": True}))
        until = time.time() + 30
        while self.chat.making and time.time() < until:
            time.sleep(0.05)
        self.assertIsNotNone(self.chat.finished())

    def test_a_path_that_is_not_listed_is_not_posted_to(self):
        self.assertEqual(self.ask("POST", "/api/panel", {"chat": self.chat.id})[0], 404)



class Preview(Served):
    def setUp(self):
        super().setUp()
        self.wait, agent.PREVIEW_WAIT = agent.PREVIEW_WAIT, 30

    def tearDown(self):
        agent.PREVIEW_WAIT = self.wait
        super().tearDown()

    def test_the_words_layer_is_read_with_a_get_and_its_pictures_only_from_work_layers(self):
        status, answer = self.ask("GET", "/api/layer?chat=nope")
        self.assertEqual((status, answer["error"]), (404, server.GONE))
        status, answer = self.ask("GET", "/api/layer?chat=" + self.chat.id)
        self.assertEqual((status, answer["error"]), (404, "There is no clip yet. Ask Muse to make one first."))
        self.make(SAID, call("add_captions", why="say it"),
                  call("add_title", words="Boss down", start=0, end=3, place="top", why="the hook"))
        status, answer = self.ask("GET", "/api/layer?chat=" + self.chat.id)
        self.assertEqual((status, answer["size"]), (200, [540, 960]))
        steps = answer["steps"]
        self.assertEqual((steps[0]["from"], steps[-1]["to"] <= 18.3), (0.0, True))
        self.assertTrue(all(re.fullmatch(r"/api/layer/[0-9a-f]{20}\.png", st["url"]) for st in steps), steps)
        self.assertTrue(all(a["to"] <= b["from"] for a, b in zip(steps, steps[1:])), "in time order, none overlap")
        status, kind, body = self.fetch(steps[0]["url"])
        self.assertEqual((status, kind, body[:8]), (200, "image/png", b"\x89PNG\r\n\x1a\n"))
        for name in ("..%2F..%2Fwork%2Flayers%2Fx.png", "0" * 20 + ".png", "abc.png", "%s.PNG" % ("a" * 20)):
            self.assertEqual(self.fetch("/api/layer/" + name)[0], 404, name)
        with open(os.path.join(self.tmp, "outside.png"), "wb") as f:
            f.write(b"\x89PNG")
        self.assertEqual(self.fetch("/api/layer/../../outside.png")[0], 404)

    def test_words_that_cannot_be_drawn_say_why(self):
        self.make(SAID)
        had = agent.layer.words

        def words(state, default="clean"):
            raise ValueError("key 'x' is not one of the title's words")

        agent.layer.words = words
        try:
            status, answer = self.ask("GET", "/api/layer?chat=" + self.chat.id)
        finally:
            agent.layer.words = had
        self.assertEqual((status, answer["error"]), (404, "The words could not be drawn: key 'x' is not one of the "
                                                          "title's words"))

    def test_a_words_edit_keeps_the_base_and_a_look_change_makes_a_new_one(self):
        status, answer = self.ask("POST", "/api/base", {"chat": self.chat.id})
        self.assertEqual((status, answer["error"]), (404, "There is no clip yet. Ask Muse to make one first."))
        self.assertEqual(self.ask("POST", "/api/base", {"chat": "nope"})[0], 404)
        self.make(SAID, call("add_title", words="Boss down", start=0, end=3, place="top", why="the hook"))
        status, first = self.ask("POST", "/api/base", {"chat": self.chat.id})
        self.assertEqual((status, first["from_cache"]), (200, False))
        self.assertRegex(first["url"], r"^/api/base/[0-9a-f]{16}\.mp4$")
        self.assertEqual(self.ask("POST", "/api/base", {"chat": self.chat.id})[1]["from_cache"], True)
        self.assertIsNone(self.chat.set_words("title", 0, "Boss down", "The boss is down"))
        self.assertIsNone(self.chat.set_captions("plain"))
        status, words = self.ask("POST", "/api/base", {"chat": self.chat.id})
        self.assertEqual((status, words["url"], words["from_cache"]), (200, first["url"], True),
                         "a change of words needs no new video")
        self.assertIsNone(self.chat.set_look("warm"))
        status, look = self.ask("POST", "/api/base", {"chat": self.chat.id})
        self.assertEqual((status, look["from_cache"]), (200, False))
        self.assertNotEqual(look["url"], first["url"], "a look changes the picture")
        self.assertEqual(len(self.based), 2)
        self.assertEqual(self.exported, [], "the finished clip is not made for the preview")
        status, kind, body = self.fetch(look["url"], {"Range": "bytes=0-99"})
        self.assertEqual((status, kind, len(body)), (206, "video/mp4", 100))
        self.assertEqual(self.fetch("/api/base/" + "0" * 16 + ".mp4")[0], 404)

    def test_a_base_made_in_the_background_shows_its_progress_and_says_when_it_is_made(self):
        agent.PREVIEW_WAIT = 0.01
        self.make(SAID)
        status, answer = self.ask("POST", "/api/base", {"chat": self.chat.id})
        self.assertEqual((status, answer), (200, {"ok": True, "url": None, "making": True}))
        self.until(lambda: any(e["kind"] == "base" for e in self.chat.events))
        made = [e for e in self.chat.events if e["kind"] == "base"]
        self.assertEqual((len(made), made[0]["job"]), (1, None))
        self.assertRegex(made[0]["url"], r"^/api/base/[0-9a-f]{16}\.mp4$")
        live = self.based[0][1]
        self.assertEqual((live["job"], live["page"]), (None, "preview"))
        self.assertRegex(live["work"], r"^Making the preview: 9 of 18 s \(50%\)$")
        agent.PREVIEW_WAIT = 30
        status, again = self.ask("POST", "/api/base", {"chat": self.chat.id})
        self.assertEqual((again["url"], again["from_cache"]), (made[0]["url"], True))
        self.assertEqual(len([e for e in self.chat.events if e["kind"] == "base"]), 1, "no event for one already made")

    def test_one_preview_is_made_at_a_time_and_stop_ends_it(self):
        agent.PREVIEW_WAIT = 0.01
        self.make(SAID)
        started, stopped = threading.Event(), threading.Event()

        def export(rec, state, out, on_progress=None, on_start=None, default="clean", **how):
            on_start(stopped.set)
            started.set()
            stopped.wait(10)
            raise render.Stopped()

        render.export = export
        self.assertEqual(self.chat.base(), {"url": None, "making": True})
        self.assertTrue(started.wait(10))
        self.assertEqual(self.chat.base(), {"url": None, "making": True})
        self.assertTrue(self.chat.stop())
        self.until(lambda: self.chat.basing is None)
        self.until(lambda: any(e["kind"] == "base" for e in self.chat.events))
        self.assertEqual([e.get("error") for e in self.chat.events if e["kind"] == "base"],
                         ["The preview was not made: Stop was pressed."])

    def test_a_preview_that_could_not_be_made_says_why(self):
        self.make(SAID)

        def export(rec, state, out, on_progress=None, on_start=None, default="clean", **how):
            raise render.NotDrawn("FFmpeg could not make the clip: no room on the disk")

        render.export = export
        status, answer = self.ask("POST", "/api/base", {"chat": self.chat.id})
        self.assertEqual((status, answer["error"]), (400, "The preview could not be made: FFmpeg could not make the "
                                                          "clip: no room on the disk"))
        self.assertIsNone(self.chat.basing)


class Worked(Clip):
    def test_ready_says_how_long_muse_worked_what_the_person_typed_and_tapped_and_the_steps(self):
        self.make(SAID)
        self.chat.emit("person", job=1, text="(opened) stream.mp4 · 1:00 long · 320x180")
        self.chat.emit("person", job=1, text="a funny clip")
        self.chat.set_style("pop")
        self.chat.set_person(1, False)
        brain = ScriptedMind([call("render_clip", why="make the file")], [call("check_clip", why="check")],
                             [call("hand_over", title="Down", caption="He falls.", hashtags=["#a"], why="post")],
                             "Done.")
        mind.get = lambda key: brain
        self.chat.say("post it", None)
        self.chat.job.thread.join(30)
        ready = [e for e in self.chat.events if e["kind"] == "ready"][-1]
        work = ready["work"]
        self.assertEqual((work["messages"], work["taps"], work["steps"]), (2, 3, 3))
        self.assertRegex(work["line"], r"^Muse worked 0:0\d · you typed 2 messages and tapped 3 times · 3 steps$")
        job = [e for e in self.chat.events if e["kind"] == "job"][-1]
        self.assertAlmostEqual(work["seconds"], ready["t"] - job["t"], delta=1)

    def test_a_person_who_typed_nothing_is_told_so(self):
        self.chat.jobs = 1
        self.chat.emit("job", job=1, mind="scripted")
        self.chat.emit("person", job=1, text="(opened) stream.mp4 · 1:00 long · 320x180")
        self.assertEqual(self.chat.worked(1, 1)["line"], "Muse worked 0:00 · you typed nothing and tapped once · 1 step")
        self.chat.emit("person", job=1, text="(opened) stream.mp4 · 1:00 long · 320x180\nfunny, 20 s")
        self.chat.emit("person", job=1, text="(tapped) Shorter.")
        self.assertEqual(self.chat.worked(1, 12)["line"], "Muse worked 0:00 · you typed 1 message and tapped 3 times · "
                                                          "12 steps")


class Checks(Clip):
    def checked(self):
        self.run_job([call("render_clip", why="make the file")], [call("check_clip", why="check")], "Done.")
        return [e for e in self.chat.events if e["kind"] == "checked"][-1]

    def test_a_check_that_failed_before_and_passes_now_is_marked_fixed(self):
        self.make(SAID, call("add_captions", why="say it"), call("decide_people", keep=[1], why="him"))
        caps = self.chat.project.state()["captions"]
        self.chat.edit(dict(caps, lines=[dict(caps["lines"][0], text="made up words")] + caps["lines"][1:]))
        first = self.checked()
        said = {c["name"]: c for c in first["checks"]}["Captions are words that were said"]
        self.assertEqual((said["ok"], said.get("fixed"), first["moment"]), (False, None, 1))
        self.chat.undo()
        second = self.checked()
        fixed = [c["name"] for c in second["checks"] if c.get("fixed")]
        self.assertEqual(fixed, ["Captions are words that were said"], "only the line that turned green")
        self.assertEqual([c["name"] for c in self.checked()["checks"] if c.get("fixed")], [], "fixed once")

    def test_a_check_of_another_recording_fixes_nothing(self):
        self.make(SAID, call("add_captions", why="say it"), call("decide_people", keep=[1], why="him"))
        caps = self.chat.project.state()["captions"]
        self.chat.edit(dict(caps, lines=[dict(caps["lines"][0], text="made up words")] + caps["lines"][1:]))
        self.checked()
        self.chat.undo()
        self.chat.emit("opened", recording=self.chat.recording, frame=None, line="another video")
        self.assertEqual([c["name"] for c in self.checked()["checks"] if c.get("fixed")], [])


class Languages(Clip):
    def test_as_said_is_put_back_at_a_tap_and_english_is_a_message_to_muse(self):
        self.make(SAID, call("add_captions", lit=True, why="karaoke"))
        self.assertEqual([(c["name"], c["now"], c["why"]) for c in self.chat.panel()["languages"]],
                         [("said", True, None), ("English", False, None)])
        _, steps = self.run_job([call("add_captions", language="Thai", lit=True, why="for Thai viewers")],
                                '{"lines": ["นี่คือช่วงนั้น", "ดูเขาล้มตอนนี้"]}', "Done.")
        self.assertFalse(steps[0]["refused"], steps[0]["result"])
        languages = self.chat.panel()["languages"]
        self.assertEqual([(c["name"], c["label"], c["now"], c["why"]) for c in languages],
                         [("said", "As said", False, None), ("English", "English", False, None),
                          ("Thai", "Thai", True, None)])
        self.assertEqual(languages[1]["say"], "(tapped) Put the captions in English.")
        self.assertEqual(self.chat.set_captions_language("English"), "Only As said changes here; Muse writes the "
                                                                     "captions in another language when you tap it.")
        self.assertIsNone(self.chat.set_captions_language("said"))
        caps = self.chat.project.state()["captions"]
        self.assertEqual((caps.get("language"), caps["lit"]), (None, True), "lit stays lit")
        self.assertEqual("".join(ln["text"] for ln in caps["lines"]), "this is the momentwatch him fall down now")
        self.assertEqual(self.chat.told[-1], "the person put the captions back in the words as said, on the page")
        self.assertEqual([c["name"] for c in self.chat.panel()["languages"] if c["now"]], ["said"])
        had = list(self.chat.project.entries)
        self.assertIsNone(self.chat.set_captions_language("said"), "already as said")
        self.assertEqual(self.chat.project.entries, had)

    def test_a_language_the_captions_cannot_be_in_is_off_with_the_tools_reason(self):
        self.make(said=[])
        whys = [c["why"] for c in self.chat.panel()["languages"]]
        self.assertEqual(whys, [None, "Nothing was heard clearly in this moment, so there are no words to caption."])


if __name__ == "__main__":
    unittest.main()
