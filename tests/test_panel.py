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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, channels, faces, hearing, mind, recording, render, server, tools  # noqa: E402
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
    stays low in the picture, and from 35 s to 40 s one that rises to the top."""

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
        self.chat = agent.Chat()
        agent.CHATS[self.chat.id] = self.chat

    def tearDown(self):
        faces.sample, faces.kept, render.export, mind.get = self.real
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
        self.assertEqual([c["name"] for c in panel["styles"]], ["clean", "pop", "sticker", "glow", "3d", "classic"])
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
                                                       "Classic.")
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

    def test_a_title_in_a_box_keeps_its_box_and_its_key_word_while_the_words_hold_it(self):
        self.make(SAID)
        self.chat.edit({"do": "title", "text": "Look", "start": 1.0, "end": 3.0, "place": "box at 100, 400",
                        "box": [100, 400, 400, 200], "key": "Look"})
        self.assertIsNone(self.chat.set_words("title", 0, "Look", "Look here"))
        titles = self.chat.project.state()["titles"]
        self.assertEqual([(t["text"], t["place"], t["box"], t["key"]) for t in titles],
                         [("Look here", "box at 100, 400", [100, 400, 400, 200], "Look")])
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
                         r"^These words take \d+ lines at the title size \(\d+ px tall\), and the top place holds "
                         r"\d+ px: use fewer words\.$")
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


class Endpoints(Served):
    def test_the_panel_is_read_with_a_get(self):
        status, panel = self.ask("GET", "/api/panel?chat=" + self.chat.id)
        self.assertEqual((status, panel), (200, {"ok": True, "clip": False}))
        self.make()
        status, panel = self.ask("GET", "/api/panel?chat=" + self.chat.id)
        self.assertEqual((status, panel["ok"], len(panel["captions"]), len(panel["styles"]), len(panel["looks"])),
                         (200, True, 4, 6, 4))
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
                                                    "Classic."),
                                 ("/api/colour", {"look": "film"}, "The clip already has the film look.")):
            self.assertEqual(self.ask("POST", path, dict(body, chat=self.chat.id)),
                             (400, {"ok": False, "error": said}), path)
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
