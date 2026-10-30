"""The agent loop, run against a scripted mind: python -m unittest discover -s tests"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, chatname, mind, needs, recipes, tools, words  # noqa: E402


def call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


KEYS_SCHEMA = tools.KEYS_SCHEMA


class ScriptedMind:
    """Answers with the next item of its script and keeps a copy of every conversation it was sent.

    An item is the words of an answer, a list of tool calls, or a function(asking) that acts while the question is
    in flight (the person types, or presses Stop) and then raises or returns one of the other two. A spelling question
    is answered with its texts unchanged, unless the next item is a scripted spelling answer or a function. The
    question whether a message asks for work (needs.py) is answered work, and the chat's name (chatname.py) "a chat",
    unless the next item is a scripted answer to it ('{"needs": ...', '{"name": ...') or a function; one answered so
    is not kept in sent, which holds the conversations a test scripted. The key word of each caption line is answered
    none ('{"keys": ...' names them), and kept out of sent the same way. offered and schemas keep the tools and the
    schema of every question.
    """
    label = "scripted"

    def __init__(self, *script):
        self.script, self.sent, self.offered, self.schemas = list(script), [], [], []

    def ask(self, messages, tools=None, temperature=None, max_tokens=None, on_delta=None, asking=None, schema=None):
        self.offered.append(tools)
        self.schemas.append(schema)
        for asked, start, default in ((needs.SCHEMA, '{"needs"', '{"needs": "work"}'),
                                      (chatname.SCHEMA, '{"name"', '{"name": "a chat"}')):
            if schema is asked and not (self.script and (callable(self.script[0]) or (
                    isinstance(self.script[0], str) and self.script[0].startswith(start)))):
                return {"text": default, "thinking": "", "tool_calls": [], "message": {"role": "assistant"},
                        "engine": {"read_new": 10, "reused": 0, "seconds": 0.1}}
        if schema is KEYS_SCHEMA and not (self.script and (callable(self.script[0]) or (
                isinstance(self.script[0], str) and self.script[0].startswith('{"keys"')))):
            none = json.dumps({"keys": [""] * len(json.loads(messages[-1]["content"].rsplit("\n\n", 1)[1]))})
            return {"text": none, "thinking": "", "tool_calls": [], "message": {"role": "assistant"},
                    "engine": {"read_new": 10, "reused": 0, "seconds": 0.1}}
        self.sent.append(json.loads(json.dumps(messages)))
        if schema is words.SPELL_SCHEMA and not (self.script and (callable(self.script[0]) or (
                isinstance(self.script[0], str) and self.script[0].startswith('{"texts"')))):
            asked = json.loads(messages[-1]["content"].rsplit("\n\n", 1)[1])
            text = json.dumps({"texts": asked}, ensure_ascii=False)
            return {"text": text, "thinking": "", "tool_calls": [], "message": {"role": "assistant", "content": text},
                    "engine": {"read_new": 10, "reused": 0, "seconds": 0.1}}
        item = self.script.pop(0)
        if callable(item):
            item = item(asking)
        if on_delta:
            on_delta("thinking", "thinking about it")
        calls = item if isinstance(item, list) else []
        text = "" if calls else item
        back = {"role": "assistant", "content": text}
        if calls:
            back["tool_calls"] = calls
        return {"text": text, "thinking": "", "tool_calls": calls, "message": back,
                "engine": {"read_new": 10, "reused": 0, "seconds": 0.1}}


class Loop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.videos = os.path.join(self.tmp, "videos")
        os.makedirs(os.path.join(self.videos, "sifu"))
        for name in ("sifu/sifu_run.mkv", "review.mp4"):
            with open(os.path.join(self.videos, name), "wb") as f:
                f.truncate(2_000_000)
        os.environ["LCS_VIDEOS"] = self.videos
        agent.WORK = os.path.join(self.tmp, "work")
        self.chat = agent.Chat()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        tools.TOOLS.pop("touch", None)

    def run_job(self, brain, words="make a clip"):
        self.chat.jobs += 1
        job = self.chat.job = agent.Job(self.chat, brain, words)
        job.run()
        return job, [e for e in self.chat.events if e.get("job") == job.n]

    def test_steps_are_numbered_logged_and_the_plan_ticks(self):
        brain = ScriptedMind([call("set_plan", steps=["find it", "check it"], why="to plan")],
                             [call("find_recordings", why="look", plan_step=1)],
                             [call("find_recordings", words="sifu", why="narrow", plan_step=2)],
                             "Found it.")
        job, events = self.run_job(brain)
        steps = [e for e in events if e["kind"] == "step"]
        self.assertEqual([s["step"] for s in steps], [1, 2, 3])
        self.assertEqual([s["why"] for s in steps], ["to plan", "look", "narrow"])
        self.assertIn("sifu_run.mkv", steps[2]["result"])
        self.assertNotIn("review.mp4", steps[2]["result"])
        plan = [e for e in events if e["kind"] == "plan"][-1]
        self.assertEqual([p["state"] for p in plan["items"]], ["done", "done"])
        self.assertEqual(events[-1]["kind"], "end")
        self.assertEqual((events[-1]["steps"], events[-1]["why"]), (3, "answered"))
        with open(self.chat.log_path, encoding="utf-8") as f:
            self.assertEqual([json.loads(line)["id"] for line in f], [e["id"] for e in self.chat.events])

    def test_a_job_that_hands_over_on_its_last_step_ends_with_muses_words_not_a_stop(self):
        handed = tools.Tool("hand_over", "Handed over", "", {}, [], lambda job, why: "handed over")
        with mock.patch.dict(tools.TOOLS, {"hand_over": handed}), mock.patch.object(agent, "MAX_STEPS", 2):
            brain = ScriptedMind([call("find_recordings", why="look")], [call("hand_over", why="give it")],
                                 "Here it is.")
            job, events = self.run_job(brain)
        self.assertEqual((events[-1]["why"], events[-1]["steps"]), ("answered", 2),
                         "SIFU on Spark, 10-29: the hand-over was step 30 and the page said the studio stopped it")
        self.assertEqual([e for e in events if e["kind"] == "note"], [])
        self.assertEqual([e["text"] for e in events if e["kind"] == "answer"], ["Here it is."])

    def test_a_job_at_the_most_steps_without_a_hand_over_is_stopped_and_says_so(self):
        with mock.patch.object(agent, "MAX_STEPS", 2):
            brain = ScriptedMind([call("find_recordings", why="look")],
                                 [call("find_recordings", words="sifu", why="narrow")], "never asked")
            job, events = self.run_job(brain)
        self.assertEqual(events[-1]["why"], "too many steps")
        self.assertIn("the most one message may take", [e["text"] for e in events if e["kind"] == "note"][-1])
        self.assertEqual(brain.script, ["never asked"])

    def test_a_recording_is_found_by_the_day_it_was_recorded_as_a_person_says_it(self):
        day = time.mktime((2026, 10, 12, 14, 17, 28, 0, 0, -1))
        os.utime(os.path.join(self.videos, "sifu", "sifu_run.mkv"), (day, day))
        os.utime(os.path.join(self.videos, "review.mp4"), (day + 86400, day + 86400))
        asked = ["12 ตุลาคม", "2026-10-12", "12 Oct", "october 12", "sifu monday"]
        brain = ScriptedMind(*[[call("find_recordings", words=w, why="the day")] for w in asked], "Found it.")
        _, events = self.run_job(brain)
        steps = [e for e in events if e["kind"] == "step"]
        self.assertEqual(len(steps), len(asked))
        for words, step in zip(asked, steps):
            self.assertIn("sifu_run.mkv", step["result"], "%s: tutorial 5561 10-28 took 11 searches" % words)
            self.assertNotIn("review.mp4", step["result"], words)

    def pictures(self, many):
        picture = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}
        self.chat.messages += [{"role": "user", "content": [{"type": "text", "text": "old"}] + [picture] * many},
                               {"role": "assistant", "content": "seen"},
                               {"role": "user", "content": [{"type": "text", "text": "new"}, picture]},
                               {"role": "assistant", "content": "seen too"}]
        self.addCleanup(agent.ROOMS.clear)
        return picture

    def test_a_full_memory_takes_out_the_oldest_pictures_and_asks_again(self):
        picture = self.pictures(60)

        def full(asking):
            raise mind.TooLong("full", 32800, 32768)

        brain = ScriptedMind(full, "Done.")
        job, events = self.run_job(brain)
        self.assertEqual(events[-1]["why"], "answered")
        sent = brain.sent[-1]
        old = next(m["content"] for m in sent if isinstance(m["content"], list) and m["content"][0]["text"] == "old")
        new = next(m["content"] for m in sent if isinstance(m["content"], list) and m["content"][0]["text"] == "new")
        self.assertEqual(sum(p.get("text") == agent.TAKEN_OUT for p in old), 47,
                         "32,800 + 2,048 for the answer, down to 70% of 32,768: 11,911 tokens, 256 a picture")
        self.assertEqual(sum(p["type"] == "image_url" for p in old), 13)
        self.assertEqual(new[1], picture, "the newest pictures stay")
        notes = [e["text"] for e in events if e["kind"] == "note"]
        self.assertIn("took out the 47 oldest pictures it had already looked at, and asked again", notes[0])
        self.assertEqual(agent.ROOMS, {"scripted": 32768})
        self.assertEqual(self.chat.held, 10, "what Muse read for its answer")

    def test_a_memory_known_to_be_nearly_full_makes_room_before_the_question_not_after_a_refusal(self):
        self.pictures(60)
        agent.ROOMS["scripted"] = 32768
        self.chat.held = 30000
        brain = ScriptedMind("Done.")
        job, events = self.run_job(brain)
        self.assertEqual(events[-1]["why"], "answered")
        old = next(m["content"] for m in brain.sent[0] if isinstance(m["content"], list)
                   and m["content"][0]["text"] == "old")
        self.assertEqual(sum(p.get("text") == agent.TAKEN_OUT for p in old), 36,
                         "30,000 + 2,048 over 85% of 32,768, down to 70%: 9,111 tokens, 256 a picture")
        self.assertIn("took out the 36 oldest pictures it had already looked at before asking",
                      [e["text"] for e in events if e["kind"] == "note"][0])
        self.chat.held = 24000
        brain = ScriptedMind("Done again.")
        self.run_job(brain, "and again")
        old = next(m["content"] for m in brain.sent[0] if isinstance(m["content"], list)
                   and m["content"][0]["text"] == "old")
        self.assertEqual(sum(p["type"] == "image_url" for p in old), 24,
                         "over 70% but under 85% nothing more comes out: one reading again, not one a question")

    def test_a_full_memory_with_no_picture_to_take_out_ends_the_job_saying_so(self):
        def full(asking):
            raise mind.TooLong("full", 32800, 32768)

        job, events = self.run_job(ScriptedMind(full))
        self.assertEqual(events[-1]["why"], "mind")
        self.assertIn("no picture is left to take out", [e["text"] for e in events if e["kind"] == "note"][-1])

    def test_the_conversation_only_grows_at_its_end(self):
        brain = ScriptedMind([call("set_plan", steps=["find it"], why="plan")],
                             [call("find_recordings", why="look", plan_step=1)], "Done.",
                             [call("find_recordings", words="review", why="again")], "Done again.")
        self.run_job(brain)
        self.run_job(brain, "and the review")
        for before, after in zip(brain.sent, brain.sent[1:]):
            self.assertEqual(after[:len(before)], before)
        tool_results = [m["content"] for m in brain.sent[-1] if m["role"] == "tool"]
        self.assertTrue(all(r.splitlines()[-1].startswith("Now: step") for r in tool_results))

    def test_a_retry_after_a_fix_runs(self):
        brain = ScriptedMind([call("find_recordings", words="sifu")],
                             [call("find_recordings", words="sifu")],
                             [call("find_recordings", words="sifu", why="the person named the game")],
                             "Here it is.")
        job, events = self.run_job(brain)
        steps = [e for e in events if e["kind"] == "step"]
        self.assertEqual([s["refused"] for s in steps], [True, True, False])
        self.assertIn("Say why", steps[0]["result"])
        self.assertIn("sifu_run.mkv", steps[2]["result"])
        self.assertEqual((job.refused, job.fixed), (2, 1))

    def test_the_same_call_and_result_a_third_time_is_refused_until_something_changes(self):
        tools.TOOLS["touch"] = tools.Tool("touch", "Touched", "", {}, [], lambda job, why: "touched", changes=True)
        ran = []
        real = tools.TOOLS["find_recordings"].run
        tools.TOOLS["find_recordings"].run = lambda job, why, **a: ran.append(1) or real(job, why, **a)
        try:
            same = call("find_recordings", newest=1, why="look")
            brain = ScriptedMind([same], [same], [same], [call("touch", why="change")], [same], "Done.")
            job, events = self.run_job(brain)
        finally:
            tools.TOOLS["find_recordings"].run = real
        steps = [e for e in events if e["kind"] == "step"]
        self.assertEqual([bool(s.get("guard")) for s in steps], [False, False, True, False, False])
        self.assertIn("already ran twice", steps[2]["result"])
        self.assertEqual(len(ran), 3)

    def test_wrong_input_is_refused_with_what_to_send(self):
        brain = ScriptedMind([call("find_recordings", newest=50, why="many")],
                             [call("cut_everything", why="x")],
                             [{"type": "function", "function": {"name": "find_recordings", "arguments": "{oops"}}],
                             [call("find_recordings", folder="F:", why="x")],
                             "Sorry.")
        _, events = self.run_job(brain)
        results = [e["result"] for e in events if e["kind"] == "step"]
        self.assertIn("newest must be from 1 to 20", results[0])
        self.assertIn("The tools are: set_plan, find_recordings", results[1])
        self.assertIn("not readable JSON", results[2])
        self.assertIn("no field named folder", results[3])
        sent = [c for m in brain.sent[-1] for c in m.get("tool_calls") or []]
        self.assertEqual([json.loads(c["function"]["arguments"]) for c in sent][2], {},
                         "unreadable arguments never go back to llama-server, which answers 500 to them")

    def test_arguments_cut_off_before_their_last_brackets_are_read(self):
        cut = '{"newest": 2, "why": "look'
        brain = ScriptedMind([{"type": "function", "function": {"name": "find_recordings", "arguments": cut}}],
                             "Done.")
        _, events = self.run_job(brain)
        step = [e for e in events if e["kind"] == "step"][0]
        self.assertFalse(step["refused"], step["result"])
        self.assertEqual(step["why"], "look")

    def test_plan_steps_left_open_get_one_reminder(self):
        brain = ScriptedMind([call("set_plan", steps=["a", "b", "c"], why="plan")],
                             [call("find_recordings", why="look", plan_step=1)],
                             "I stopped.", "b and c are not needed.")
        job, events = self.run_job(brain)
        answers = [e for e in events if e["kind"] == "answer"]
        self.assertEqual([a["final"] for a in answers], [False, True])
        self.assertIn("[studio] Your plan has steps no call has named yet: 2. b; 3. c.",
                      self.chat.messages[-2]["content"])
        self.assertEqual(events[-1]["why"], "answered")

    def test_an_answer_that_speaks_of_a_cover_none_made_is_asked_again_once(self):
        """Beat 1, 10-29: make_cover was refused, and the answer still said "และปกพร้อมโพสต์ให้แล้ว"."""
        brain = ScriptedMind("คลิปและปกพร้อมโพสต์ให้แล้ว", "คลิปพร้อมแล้ว ยังไม่มีหน้าปก", "")
        job, events = self.run_job(brain)
        answers = [e for e in events if e["kind"] == "answer"]
        self.assertEqual([(a["text"], a["final"]) for a in answers],
                         [("คลิปและปกพร้อมโพสต์ให้แล้ว", False), ("คลิปพร้อมแล้ว ยังไม่มีหน้าปก", True)],
                         "asked once; the second answer stands, whatever it says")
        self.assertEqual(self.chat.messages[-2]["content"], agent.NO_COVER)
        self.assertIn("no cover is made", [e for e in events if e["kind"] == "note"][-1]["text"])
        for said in ("ทุกอย่างปกติดี", "Done, the clip is ready.", "ปกครองง่าย"):
            self.assertFalse(job.false_cover(said), said)
        for said in ("The cover is ready.", "ภาพปกสวยมาก", "ทำหน้าปกให้แล้ว"):
            self.assertTrue(job.false_cover(said), said)
        self.chat.emit("cover", job=job.n, file="made.mp4", cover="0123456789abcdef.jpg", title="t", usd=0.01)
        with mock.patch.object(self.chat, "finished", lambda: {"name": "made.mp4"}):
            self.assertFalse(job.false_cover("The cover is ready."), "a cover made for the finished file is there")
        with mock.patch.object(self.chat, "finished", lambda: {"name": "newer.mp4"}):
            self.assertTrue(job.false_cover("The cover is ready."), "a cover of an older file is not this clip's")

    def test_a_doubled_thai_mark_is_not_read_back(self):
        brain = ScriptedMind([call("find_recordings", words="sifu", why="ลองค้้้้นหาอีกคร้้ง")], "เจอแล้ว")
        _, events = self.run_job(brain)
        self.assertEqual([e["why"] for e in events if e["kind"] == "step"], ["ลองค้นหาอีกคร้ง"])
        self.assertNotIn("้้", json.dumps(brain.sent[-1], ensure_ascii=False))

    def test_no_reminder_when_muse_stops_to_explain_a_refusal(self):
        brain = ScriptedMind([call("set_plan", steps=["pick the kind", "plan the clip", "make it"], why="plan")],
                             [call("load_recipe", type="travel", why="horses on a trail", plan_step=1)],
                             "There is no travel recipe yet.")
        _, events = self.run_job(brain)
        self.assertEqual([e["kind"] for e in events if e["kind"] in ("answer", "note")], ["answer"])
        plan = [e for e in events if e["kind"] == "plan"][-1]
        self.assertEqual([p["state"] for p in plan["items"]], ["todo"] * 3,
                         "a step whose only call was refused is not ticked done, nor the last after it")

    def test_the_answer_itself_does_the_last_plan_step(self):
        brain = ScriptedMind([call("set_plan", steps=["find both", "compare them"], why="plan")],
                             [call("find_recordings", why="look", plan_step=1)], "axi has more.")
        _, events = self.run_job(brain)
        self.assertEqual([e["kind"] for e in events if e["kind"] in ("answer", "note")], ["answer"])
        plan = [e for e in events if e["kind"] == "plan"][-1]
        self.assertEqual([p["state"] for p in plan["items"]], ["done", "done"])

    def test_the_recipe_reaches_muse_the_page_and_the_state(self):
        self.chat.recording = {"path": "review.mp4", "name": "review.mp4", "seconds": 60.0}
        brain = ScriptedMind([call("load_recipe", type="Seller", language="Thai", why="it is a product review")],
                             [call("find_recordings", why="look")], "Ready.")
        job, events = self.run_job(brain)
        kinds = [e["kind"] for e in events]
        self.assertEqual(kinds[kinds.index("step") + 1], "guesses")
        step = [e for e in events if e["kind"] == "step"][0]
        self.assertIn("Recipe loaded: Product review or selling live (seller).", step["result"])
        self.assertIn("Every price or number on screen is one the seller said", step["result"])
        guesses = [e for e in events if e["kind"] == "guesses"][0]
        self.assertEqual((guesses["clip"]["recipe"], guesses["clip"]["length_s"], guesses["clip"]["style"]),
                         ("seller", 30, "clean"))
        self.assertIn({"label": "It's a game stream", "say": "(tapped) It's a game stream."}, guesses["chips"])
        self.assertTrue(all(c["say"].startswith("(tapped) ") for c in guesses["chips"]))
        last_result = [m for m in brain.sent[-1] if m["role"] == "tool"][-1]["content"]
        self.assertIn("recipe seller: about 30 s, clean style, Thai", last_result.splitlines()[-1])
        self.assertEqual(agent.Chat.load(self.chat.id).clip, self.chat.clip)

    def test_a_kind_with_no_recipe_is_refused_with_the_kinds_there_are(self):
        brain = ScriptedMind([call("load_recipe", type="cooking", why="it is a cooking show")],
                             [call("load_recipe", type="seller", style="neon", why="closest")], "Not ready.")
        _, events = self.run_job(brain)
        results = [e["result"] for e in events if e["kind"] == "step"]
        self.assertIn("There is no recipe for cooking yet. The recipes are: explain (Tutorial or screen explainer), game "
                      "(Game stream), montage (Trip or event clips), seller (Product review or selling live), talk "
                      "(Podcast or interview), vlog (Vlog or edited episode).", results[0])
        self.assertIn("Do not use a recipe made for another kind of video", results[0])
        self.assertIn("style must be one of: clean, pop, sticker", results[1])
        self.assertIsNone(self.chat.clip)

    def test_a_recipe_is_picked_once_the_recording_is_open(self):
        """Glimmer 10-30: a seller's recipe loaded before EP 22, a cooking vlog, was opened, and the hosts were blurred."""
        brain = ScriptedMind([call("load_recipe", type="seller", why="the ask says a clip to post")], "Opened first.")
        _, events = self.run_job(brain)
        step = [e for e in events if e["kind"] == "step"][0]
        self.assertTrue(step["refused"])
        self.assertIn("Open the recording first (open_recording, or open_clips for a folder)", step["result"])
        self.assertIsNone(self.chat.clip)

    def test_another_recipe_replaces_the_first(self):
        self.chat.recording = {"path": "review.mp4", "name": "review.mp4", "seconds": 60.0}
        brain = ScriptedMind([call("load_recipe", type="seller", why="a review")],
                             [call("load_recipe", type="game", length_s=20, why="the person says it is a stream")],
                             "Switched.")
        _, events = self.run_job(brain)
        results = [e["result"] for e in events if e["kind"] == "step"]
        self.assertIn("This recipe replaces the seller recipe read earlier", results[1])
        self.assertEqual((self.chat.clip["recipe"], self.chat.clip["length_s"]), ("game", 20))

    def test_recipes_say_what_to_do_in_words_not_tool_names(self):
        for recipe in recipes.RECIPES.values():
            self.assertTrue(all(recipe[k] for k in recipes.FIELDS), recipe["name"])
            for name in tools.TOOLS:
                self.assertNotIn(name, recipe["body"], recipe["name"])

    def test_a_job_that_keeps_repeating_is_stopped(self):
        same = call("find_recordings", why="look")
        job, events = self.run_job(ScriptedMind(*[[same]] * 5))
        self.assertEqual(events[-1]["why"], "repeating")
        self.assertEqual(job.steps, 5)

    def test_words_typed_while_muse_thinks_go_in_next(self):
        def person_types(asking):
            self.chat.say("only the review, please", None)
            self.assertTrue(asking.cancelled)
            raise mind.Cancelled()

        brain = ScriptedMind([call("find_recordings", why="look")], person_types,
                             [call("find_recordings", words="review", why="the person asked for the review")],
                             "Here is the review.")
        job, events = self.run_job(brain)
        self.assertEqual(brain.sent[2][-1], {"role": "user", "content": "only the review, please"})
        cut = [e for e in events if e["kind"] == "cut"]
        self.assertEqual([(c["step"], c["by"]) for c in cut], [(2, "person")])
        said = [e for e in events if e["kind"] == "person"]
        self.assertEqual([(p["text"], p.get("cut_in")) for p in said], [("only the review, please", True)])
        self.assertEqual((job.steps, events[-1]["why"]), (2, "answered"))

    def test_words_typed_as_muse_answers_are_answered_too(self):
        def person_types_late(asking):
            self.chat.say("and the game one?", None)
            return "Here is the review."

        brain = ScriptedMind(person_types_late, "The game one is not here.")
        job, events = self.run_job(brain)
        answers = [e for e in events if e["kind"] == "answer"]
        self.assertEqual([(a["text"], a["final"]) for a in answers],
                         [("Here is the review.", False), ("The game one is not here.", True)])
        self.assertEqual(brain.sent[1][-1], {"role": "user", "content": "and the game one?"})

    def test_stop_ends_the_job_while_muse_thinks(self):
        def person_stops(asking):
            self.assertTrue(self.chat.stop())
            self.assertTrue(asking.cancelled)
            raise mind.Cancelled()

        brain = ScriptedMind([call("find_recordings", why="look")], person_stops, "never asked")
        job, events = self.run_job(brain)
        self.assertEqual(len(brain.sent), 2)
        self.assertEqual(events[-1]["why"], "stopped")
        self.assertFalse(self.chat.busy())
        self.assertFalse(self.chat.stop())

    def test_a_stop_after_the_answer_came_runs_no_tool_and_keeps_the_conversation_whole(self):
        ran = []
        real = tools.TOOLS["find_recordings"].run
        tools.TOOLS["find_recordings"].run = lambda job, why, **a: ran.append(1) or real(job, why, **a)

        def person_stops(asking):
            self.chat.stop()
            return [call("find_recordings", why="look"), call("find_recordings", words="x", why="look")]

        try:
            job, events = self.run_job(ScriptedMind(person_stops))
        finally:
            tools.TOOLS["find_recordings"].run = real
        self.assertEqual((ran, job.steps, events[-1]["why"]), ([], 0, "stopped"))
        asked = [c["id"] for c in self.chat.messages[-3]["tool_calls"]]
        answered = [m["tool_call_id"] for m in self.chat.messages if m["role"] == "tool"]
        self.assertEqual(answered, asked)

    def test_a_message_after_the_job_starts_a_new_one(self):
        self.run_job(ScriptedMind("Hello."))
        self.assertFalse(self.chat.busy())
        brain, real = ScriptedMind("Hello again."), mind.get
        mind.get = lambda key: brain
        try:
            job = self.chat.say("hi again", None)
            job.thread.join(5)
        finally:
            mind.get = real
        self.assertEqual((job.n, self.chat.events[-1]["why"]), (2, "answered"))

    def test_muse_is_told_the_language_the_person_typed_in_and_the_page_shows_only_their_words(self):
        brain, real = ScriptedMind("ได้ครับ", "Sure."), mind.get
        mind.get = lambda key: brain
        try:
            self.chat.say("มันดูไม่รู้เรื่องเลยนะซีนที่เลือกมาอะ", None).thread.join(5)
            self.chat.say("try another moment", None).thread.join(5)
        finally:
            mind.get = real
        thai, english = [s[-1]["content"] for s in brain.sent]
        self.assertTrue(thai.endswith("\n[studio] The person wrote this in Thai: answer in Thai."), thai)
        self.assertNotIn("[studio] The person wrote", english)
        self.assertEqual([e["text"] for e in self.chat.events if e["kind"] == "person"],
                         ["มันดูไม่รู้เรื่องเลยนะซีนที่เลือกมาอะ", "try another moment"])

    def test_an_earlier_chat_opens_again_as_it_was(self):
        self.run_job(ScriptedMind([call("set_plan", steps=["find it"], why="plan")],
                                  [call("find_recordings", why="look", plan_step=1)], "Found it."))
        again = agent.Chat.load(self.chat.id)
        self.assertEqual(again.messages, self.chat.messages)
        self.assertEqual([e["kind"] for e in again.events], [e["kind"] for e in self.chat.events])
        self.assertEqual((again.jobs, again.busy()), (1, False))

    def test_a_chat_closed_mid_job_opens_with_that_job_ended(self):
        self.chat.jobs = 1
        self.chat.emit("person", job=1, text="make a clip")
        self.chat.emit("job", job=1, mind="scripted")
        again = agent.Chat.load(self.chat.id)
        self.assertEqual((again.events[-1]["kind"], again.events[-1]["why"]), ("end", "closed"))

    def test_only_a_real_chat_name_is_looked_up_on_disk(self):
        for name in ("evil.messages.jsonl", "evil.events.jsonl"):
            with open(os.path.join(agent.WORK, name), "w", encoding="utf-8") as f:
                f.write("{}\n")
        self.assertIsNone(agent.open_chat("../evil"))
        self.assertIsNone(agent.open_chat("20990101-000000-abcd"))

    def test_words_typed_just_before_muse_is_asked_go_into_that_question(self):
        brain = ScriptedMind("Both done.")
        self.chat.jobs += 1
        job = self.chat.job = agent.Job(self.chat, brain, "find the review")
        take, typed = job.take_inbox, []

        def take_then_person_types():
            take()
            if not typed:
                typed.append(self.chat.say("and the game", None))

        job.take_inbox = take_then_person_types
        job.run()
        self.assertEqual([m["content"] for m in brain.sent[0][1:]], ["find the review", "and the game"])


class Talk(unittest.TestCase):
    """A typed message that only talks starts no work, however Muse first answers it (needs.py)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["LCS_VIDEOS"] = self.tmp
        agent.WORK = os.path.join(self.tmp, "work")
        self.chat, self.real = agent.Chat(), mind.get

    def tearDown(self):
        mind.get = self.real
        shutil.rmtree(self.tmp, ignore_errors=True)

    def say(self, brain, said):
        mind.get = lambda key: brain
        job = self.chat.say(said, None)
        job.thread.join(10)
        self.assertFalse(job.thread.is_alive())
        if getattr(job, "speller", None):
            job.speller.join(10)
        return [e for e in self.chat.events if e.get("job") == job.n]

    def test_small_talk_is_answered_in_words_and_no_tool_runs(self):
        brain = ScriptedMind([call("find_recordings", why="หาคลิป")], '{"needs": "talk"}', "ฝนตกเหรอ พักก่อนนะ")
        events = self.say(brain, "ฝนตกทั้งวันเลย")
        self.assertEqual([e["kind"] for e in events if e["kind"] == "step"], [])
        answer = [e for e in events if e["kind"] == "answer"][-1]
        self.assertEqual((answer["text"], answer["talk"], events[-1]["why"]), ("ฝนตกเหรอ พักก่อนนะ", True, "answered"))
        self.assertEqual(brain.schemas[1], needs.SCHEMA)
        self.assertIn("The person's message: ฝนตกทั้งวันเลย", brain.sent[1][-1]["content"])
        self.assertIsNotNone(brain.offered[0])
        self.assertIsNone(brain.offered[2], "asked again with no tools, so no call can come back")
        self.assertFalse(any(m.get("tool_calls") for m in self.chat.messages), "the call that was not run is not kept")
        self.assertEqual([m["role"] for m in self.chat.messages], ["system", "user", "assistant"])

    def test_the_countdown_is_offered_only_once_the_person_asked_for_one(self):
        def names(brain):
            return {t["function"]["name"] for t in brain.offered[0]}

        brain = ScriptedMind("ok")
        self.say(brain, "ทำคลิปตลกจากไลฟ์นี้ 1 คลิป ใส่เอฟเฟกต์ตรงจังหวะฮาด้วย")
        self.assertNotIn("make_countdown", names(brain), "it refuses unless asked, and offered it moved Glimmer's pick")
        self.assertEqual(len(names(brain)), len(tools.TOOLS) - 1, "every other tool is still offered")
        brain = ScriptedMind("ok")
        self.say(brain, "ทำท็อป 5 ช่วงมันส์จากไลฟ์นี้")
        self.assertIn("make_countdown", names(brain))

    def test_a_request_for_work_runs_its_tools(self):
        brain = ScriptedMind([call("find_recordings", why="หาไลฟ์")], '{"needs": "work"}', "เจอแล้ว")
        events = self.say(brain, "หาไลฟ์เมื่อวานให้หน่อย")
        self.assertEqual([e["tool"] for e in events if e["kind"] == "step"], ["find_recordings"])
        self.assertFalse([e for e in events if e["kind"] == "answer"][-1]["talk"])

    def test_an_answer_in_words_asks_no_question_apart(self):
        brain = ScriptedMind("สวัสดีครับ")
        self.say(brain, "สวัสดีครับ")
        self.assertEqual(brain.schemas, [None])

    def test_a_tap_and_an_opened_recording_are_never_asked_about(self):
        for said in (agent.TAP_SAID + "Put the captions in Thai.", agent.OPEN_SAID + "review.mp4"):
            brain = ScriptedMind([call("find_recordings", why="look")], "Done.")
            events = self.say(brain, said)
            self.assertNotIn(needs.SCHEMA, brain.schemas, said)
            self.assertEqual([e["tool"] for e in events if e["kind"] == "step"], ["find_recordings"], said)

    def test_the_question_sees_the_editors_last_answer(self):
        self.say(ScriptedMind("อยากให้ตัดคลิปขายของจากไลฟ์นี้ไหมครับ"), "มีไลฟ์ยาวอยู่")
        brain = ScriptedMind([call("find_recordings", why="look")], '{"needs": "work"}', "Done.")
        self.say(brain, "เอาเลย")
        self.assertIn("The editor's last answer: อยากให้ตัดคลิปขายของจากไลฟ์นี้ไหมครับ", brain.sent[1][-1]["content"])

    def test_a_question_that_fails_leaves_the_job_as_work(self):
        def broken(asking):
            raise mind.MindError("Muse is not there.")

        brain = ScriptedMind([call("find_recordings", why="look")], broken, "Done.")
        events = self.say(brain, "หาไลฟ์ให้หน่อย")
        self.assertEqual([e["tool"] for e in events if e["kind"] == "step"], ["find_recordings"])

    def named(self):
        return [e["title"] for e in self.chat.events if e["kind"] == "named"]

    def test_muse_names_the_chat_after_its_first_work_and_the_list_shows_that_name(self):
        """Tarm 10-28: "ให้ model คิดชื่อเลย"."""
        brain = ScriptedMind([call("find_recordings", why="หาไลฟ์")], '{"needs": "work"}', "เจอไลฟ์เมื่อวานแล้ว",
                             '{"name": "\\"ไลฟ์ขายของ เมื่อวาน\\""}')
        self.say(brain, "หาไลฟ์เมื่อวานให้หน่อย")
        self.assertEqual(self.named(), ["ไลฟ์ขายของ เมื่อวาน"], "quotes around the name are taken off")
        asked = brain.sent[-1][-1]["content"]
        self.assertIn("- หาไลฟ์เมื่อวานให้หน่อย", asked)
        self.assertIn("The editor answered: เจอไลฟ์เมื่อวานแล้ว", asked)
        self.assertEqual(agent.earlier()[0]["title"], "ไลฟ์ขายของ เมื่อวาน")

    def test_talk_does_not_name_the_chat_and_a_named_chat_is_not_named_again(self):
        self.say(ScriptedMind("สวัสดีครับ"), "สวัสดีครับ")
        self.assertEqual(self.named(), [])
        self.say(ScriptedMind([call("find_recordings", why="look")], "Found.", '{"name": "รีวิวไมค์"}'), "หาคลิปไมค์")
        self.assertEqual(self.named(), ["รีวิวไมค์"])
        brain = ScriptedMind([call("find_recordings", why="look")], "Found again.")
        self.say(brain, "หาอีกที")
        self.assertNotIn(chatname.SCHEMA, brain.schemas)

    def test_the_name_comes_from_the_messages_that_asked_for_work_not_from_small_talk(self):
        """10-28, live on Glimmer: "ฝนตกทั้งวันเลย เหนื่อยจัง" then a look at the yeti folder was named
        "รวมคลิปฝนตกทั้งวัน"."""
        self.say(ScriptedMind("พักก่อนนะ"), "ฝนตกทั้งวันเลย เหนื่อยจัง")
        brain = ScriptedMind([call("find_recordings", why="look")], "มี 8 ไฟล์", '{"name": "คลิปรีวิวไมค์"}')
        self.say(brain, "มีคลิปรีวิวไมค์กี่ไฟล์")
        asked = brain.sent[-1][-1]["content"]
        self.assertIn("- มีคลิปรีวิวไมค์กี่ไฟล์", asked)
        self.assertNotIn("ฝนตก", asked)

    def test_a_name_that_shows_a_file_or_folder_is_not_kept(self):
        for said in ('{"name": "yeti-c149d0193c.mp4"}', '{"name": "H:\\\\Phone\\\\yeti"}', '{"name": "H:/Phone/yeti"}',
                     '{"name": "clips in /Users/me"}', '{"name": "  "}', "not json"):
            self.assertIsNone(chatname.read(said), said)
        self.assertEqual(chatname.read('{"name": "เครื่องดูดความชื้น 750 มล/วัน"}'), "เครื่องดูดความชื้น 750 มล/วัน",
                         "a slash between words is not a folder (10-28: both Muses named the dehumidifier so)")
        self.assertEqual(chatname.read('{"name": "' + "ไมค์ " * 20 + '"}'), ("ไมค์ " * 9).strip())

    def test_only_the_first_answer_is_asked_about(self):
        brain = ScriptedMind([call("find_recordings", why="look")], '{"needs": "work"}',
                             [call("find_recordings", words="sifu", why="narrow")], "Done.")
        self.say(brain, "หาไลฟ์ให้หน่อย")
        self.assertEqual(brain.schemas.count(needs.SCHEMA), 1)


if __name__ == "__main__":
    unittest.main()
