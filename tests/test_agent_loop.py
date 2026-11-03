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

    def told_to_finish(self):
        return [m["content"] for m in self.chat.messages if m["role"] == "user" and "steps are left" in str(m["content"])]

    def test_muse_is_told_once_near_the_most_steps_to_finish_with_what_it_has(self):
        """Lane 4, 10-30: four jobs of lane 5's rounds reached step 30 with no clip, one with a clip rendered."""
        with mock.patch.object(agent, "MAX_STEPS", agent.STEPS_WARNED + 2):
            brain = ScriptedMind(*[[call("find_recordings", words="w%d" % k, why="look")] for k in range(9)], "never")
            _, events = self.run_job(brain)
        self.assertEqual(events[-1]["why"], "too many steps")
        self.assertEqual(self.told_to_finish(), [agent.LAST_STEPS % agent.STEPS_WARNED])
        self.assertIn("make the clip if there is none", self.told_to_finish()[0])
        at = [m["content"] for m in self.chat.messages if m["role"] == "user"].index(self.told_to_finish()[0])
        self.assertGreater(at, 0, "it comes after the person's message, once the steps are gone")

    def test_a_job_that_handed_over_or_is_waiting_for_a_card_is_not_told_to_finish(self):
        handed = tools.Tool("hand_over", "Handed over", "", {}, [],
                            lambda job, why: (job.emit("ready", file="a.mp4"), "handed over")[1])
        with mock.patch.dict(tools.TOOLS, {"hand_over": handed}), mock.patch.object(agent, "MAX_STEPS", agent.STEPS_WARNED + 2):
            brain = ScriptedMind([call("hand_over", why="give it")],
                                 *[[call("find_recordings", words="w%d" % k, why="look")] for k in range(9)], "Done.")
            self.run_job(brain)
        self.assertEqual(self.told_to_finish(), [])
        with mock.patch.object(agent, "MAX_STEPS", agent.STEPS_WARNED + 2), mock.patch.object(agent, "_waits",
                                                                                              lambda job: True):
            self.run_job(ScriptedMind(*[[call("find_recordings", words="x%d" % k, why="look")] for k in range(3)],
                                      "They are on the cards."))
        self.assertEqual(self.told_to_finish(), [])

    TWO_LOOKS = ([call("find_recordings", why="look")], [call("find_recordings", words="sifu", why="narrow")])

    def out_of_steps(self, *script, waits=False, made=True, clip=True, people=(), handed="a.mp4"):
        """A job of two steps at most on a chat with a finished file, or made False on a clip with no file as it is, or
        clip False too on no clip, with people undecided in it; hand_over gives the file handed. Returns (the mind, the
        job's events)."""
        files, people = [{"name": "a.mp4"}] if made else [], [{"n": n, "blur": None, "seen": 3} for n in people]
        check = tools.Tool("check_clip", "Checked", "", {}, [], lambda job, why: "checked")
        hand = tools.Tool("hand_over", "Handed over", "", {}, [],
                          lambda job, why: (job.emit("ready", file=handed), "handed over")[1])
        render = tools.Tool("render_clip", "Rendered", "", {}, [],
                            lambda job, why: (files.append({"name": "a.mp4"}), "rendered")[1])
        decide = tools.Tool("decide_people", "Decided", "", {}, [], lambda job, why: (people.clear(), "decided")[1])
        state = self.chat.project.state
        brain = ScriptedMind(*script)
        with mock.patch.dict(tools.TOOLS, {"check_clip": check, "hand_over": hand, "render_clip": render,
                                           "decide_people": decide}), mock.patch.object(
                agent, "MAX_STEPS", 2), mock.patch.object(self.chat, "finished", lambda: files[-1] if files else None), \
                mock.patch.object(agent, "_waits", lambda job: waits), mock.patch.object(
                    self.chat, "clip_line", lambda: "a clip"), mock.patch.object(
                    self.chat.project, "state", lambda: dict(state(), clip=clip and {"moment": 1}, people=people)):
            _, events = self.run_job(brain)
        return brain, events

    def offers(self, brain):
        """The tools of each question of the job itself, by name, or None."""
        return [o and sorted(t["function"]["name"] for t in o) for o, s in zip(brain.offered, brain.schemas) if s is None]

    def test_a_rendered_clip_at_the_most_steps_still_reaches_the_person(self):
        """Lane 5's 031052-26c7: step 30 passed the file's checks 4 of 4 and the job ended with nothing handed over."""
        brain, events = self.out_of_steps(*self.TWO_LOOKS, [call("check_clip", why="check it")],
                                          [call("hand_over", why="give it")], "Here it is.")
        self.assertEqual((events[-1]["why"], events[-1]["steps"]), ("answered", 4))
        self.assertEqual(len([e for e in events if e["kind"] == "ready"]), 1)
        self.assertEqual(self.offers(brain)[2:], [["check_clip", "hand_over"], ["check_clip", "hand_over"], None])
        said = agent.FINISH % (agent.FINISH_STEPS, "check_clip, hand_over")
        self.assertEqual([m["content"] for m in self.chat.messages].count(said), 1)
        self.assertIn("Muse has used its 2 steps, so the studio left it only the check and the hand-over",
                      [e["text"] for e in events if e["kind"] == "note"][0])

    def test_a_clip_with_no_file_as_it_is_is_rendered_and_reaches_the_person(self):
        """Lane 5's 051633-f8a0, 053131-51de, 060327-ece9: 3 of 21 Glimmer runs lost at step 30 with a clip made and no
        file as it was. One was never rendered, one lost a title at step 30, one was made again at step 27."""
        brain, events = self.out_of_steps(*self.TWO_LOOKS, [call("render_clip", why="make it")],
                                          [call("check_clip", why="check it")], [call("hand_over", why="give it")],
                                          "Here it is.", made=False)
        self.assertEqual((events[-1]["why"], events[-1]["steps"]), ("answered", 5))
        self.assertEqual(len([e for e in events if e["kind"] == "ready"]), 1)
        self.assertEqual(self.offers(brain)[2:], [["check_clip", "hand_over", "render_clip"], ["check_clip", "hand_over"],
                                                  ["check_clip", "hand_over"], None])
        said = agent.FINISH % (agent.FINISH_STEPS, "render_clip, check_clip, hand_over")
        self.assertEqual([m["content"] for m in self.chat.messages].count(said), 1)
        self.assertIn("left it only the render, the check and the hand-over",
                      [e["text"] for e in events if e["kind"] == "note"][0])

    def test_a_late_clip_with_people_undecided_can_still_be_decided_and_rendered(self):
        """Nothing undecided is rendered, and a clip made again late has people nobody decided about yet."""
        brain, events = self.out_of_steps(*self.TWO_LOOKS, [call("decide_people", why="decide")],
                                          [call("render_clip", why="make it")], [call("check_clip", why="check it")],
                                          [call("hand_over", why="give it")], "Here it is.", made=False, people=[1])
        self.assertEqual((events[-1]["why"], events[-1]["steps"]), ("answered", 6))
        self.assertEqual(len([e for e in events if e["kind"] == "ready"]), 1)
        offers = self.offers(brain)
        self.assertEqual(offers[2], ["check_clip", "decide_people", "hand_over", "render_clip"])
        self.assertEqual(offers[3], ["check_clip", "hand_over", "render_clip"], "the decision is made")

    def test_a_job_with_no_clip_gets_no_more_steps(self):
        brain, events = self.out_of_steps(*self.TWO_LOOKS, "never asked", made=False, clip=False)
        self.assertEqual((events[-1]["why"], events[-1]["steps"], brain.script), ("too many steps", 2, ["never asked"]))

    def test_a_later_message_out_of_steps_does_not_hand_the_same_clip_over_again(self):
        self.out_of_steps(*self.TWO_LOOKS, [call("hand_over", why="give it")], "Here it is.")
        brain, events = self.out_of_steps(*self.TWO_LOOKS, "never asked")
        self.assertEqual((events[-1]["why"], brain.script), ("too many steps", ["never asked"]))

    def test_a_job_out_of_steps_gets_four_more_and_then_stops_as_before(self):
        brain, events = self.out_of_steps(*self.TWO_LOOKS, *[[call("find_recordings", words=w, why="again")]
                                                             for w in "abcd"], "never asked")
        self.assertEqual((events[-1]["why"], events[-1]["steps"]), ("too many steps", 6))
        self.assertEqual(brain.script, ["never asked"])
        self.assertIn("the most one message may take", [e["text"] for e in events if e["kind"] == "note"][-1])

    def test_a_job_that_gave_a_clip_gets_no_more_steps_for_a_later_one(self):
        brain, events = self.out_of_steps([call("hand_over", why="one of them")], [call("find_recordings", why="look")],
                                          "One is ready, the other is not.", handed="the one before.mp4")
        self.assertEqual((events[-1]["why"], brain.script), ("too many steps", []))
        self.assertEqual([(e["text"], e["final"]) for e in events if e["kind"] == "answer"],
                         [("One is ready, the other is not.", True)], "024201-3215: the job just stopped, with no word")
        self.assertIn(agent.STOPPED_SAY, [m["content"] for m in self.chat.messages if m["role"] == "user"])
        self.assertIsNone(brain.offered[-1], "words, no tools")

    def test_an_album_with_a_clip_to_give_finishes_that_clip_though_it_gave_others(self):
        """024201-3215: 4 of 5 clips were handed over, and the iPad, made at step 40, was left out."""
        self.chat.album = [{"moment": k, "title": "", "project": self.chat.project} for k in (1, 2)]
        _, events = self.out_of_steps([call("hand_over", why="one of them")], [call("find_recordings", why="look")],
                                      [call("hand_over", why="the last")], "All here.", handed="the one before.mp4")
        self.assertEqual(events[-1]["why"], "answered")
        self.assertEqual(len([e for e in events if e["kind"] == "ready"]), 2)

    def test_a_job_waiting_on_a_card_gets_no_more_steps(self):
        brain, events = self.out_of_steps(*self.TWO_LOOKS, "never asked", waits=True)
        self.assertEqual((events[-1]["why"], brain.script), ("too many steps", ["never asked"]))

    def test_words_alone_do_not_end_a_job_with_a_clip_still_to_give(self):
        _, events = self.out_of_steps(*self.TWO_LOOKS, "It is ready.", [call("hand_over", why="give it")], "Here it is.")
        self.assertEqual(events[-1]["why"], "answered")
        self.assertEqual(len([e for e in events if e["kind"] == "ready"]), 1)
        self.assertEqual([(e["text"], e["final"]) for e in events if e["kind"] == "answer"],
                         [("It is ready.", False), ("Here it is.", True)])
        self.assertIn(agent.FINISH_CALL % "check_clip, hand_over",
                      [m["content"] for m in self.chat.messages if m["role"] == "user"])
        brain, events = self.out_of_steps(*self.TWO_LOOKS, "One.", "Two.", "Three.", "Four.", "never asked", made=False)
        self.assertEqual((events[-1]["why"], events[-1]["steps"], brain.script), ("too many steps", 6, ["never asked"]))
        self.assertIn(agent.FINISH_CALL % "render_clip, check_clip, hand_over",
                      [m["content"] for m in self.chat.messages if m["role"] == "user"])

    def test_no_other_tool_runs_once_only_the_hand_over_is_left(self):
        _, events = self.out_of_steps(*self.TWO_LOOKS, [call("find_recordings", words="more", why="again")],
                                      [call("hand_over", why="give it")], "Here it is.")
        steps = [e for e in events if e["kind"] == "step"]
        self.assertEqual((steps[2]["refused"], steps[2]["result"]), (True, agent.FINISH_ONLY % "check_clip, hand_over"))
        self.assertEqual(events[-1]["why"], "answered")

    def test_the_refusal_names_the_render_while_it_is_left(self):
        _, events = self.out_of_steps(*self.TWO_LOOKS, [call("find_recordings", words="more", why="again")],
                                      [call("render_clip", why="make it")], [call("check_clip", why="check it")],
                                      [call("hand_over", why="give it")], "Here it is.", made=False)
        steps = [e for e in events if e["kind"] == "step"]
        self.assertEqual(steps[2]["result"], agent.FINISH_ONLY % "render_clip, check_clip, hand_over")
        self.assertEqual((events[-1]["why"], events[-1]["steps"]), ("answered", 6))

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

    def long_results(self, *lengths):
        """The chat's messages: a recipe then tool results of lengths, each with a call before it, then short ones."""
        self.chat.messages += [{"role": "tool", "tool_call_id": "r", "content": agent.RECIPE_SAID + " x" * 1000}]
        self.chat.messages += [{"role": "tool", "tool_call_id": str(n), "content": "y" * n} for n in lengths]
        self.chat.messages += [{"role": "tool", "tool_call_id": "s%d" % k, "content": "short"} for k in range(agent.RESULT_KEPT)]

    def test_long_old_results_are_cut_oldest_first_and_the_newest_stay_whole(self):
        self.long_results(8000, 4000, 1000)
        self.chat.messages[-1]["content"] = "z" * 9000
        self.assertEqual(agent.cut_results(self.chat.messages, 1000), 1, "4,000 tokens out of the oldest are enough")
        tools_said = [m["content"] for m in self.chat.messages if m["role"] == "tool"]
        self.assertEqual(tools_said[1], "y" * agent.RESULT_HEAD + agent.CUT_OUT)
        self.assertEqual(tools_said[2], "y" * 4000)
        self.assertEqual(agent.cut_results(self.chat.messages, 100000), 1, "the next long one, and no more")
        tools_said = [m["content"] for m in self.chat.messages if m["role"] == "tool"]
        self.assertEqual(tools_said[3], "y" * 1000, "a short result stays")
        self.assertTrue(tools_said[0].startswith(agent.RECIPE_SAID), "the recipe the job follows stays")
        self.assertEqual(tools_said[-1], "z" * 9000, "the newest messages stay")

    def test_a_recipe_replaced_by_another_is_only_memory(self):
        self.long_results(10)
        self.chat.messages += [{"role": "tool", "tool_call_id": "new", "content": agent.RECIPE_SAID + " new" * 1000}]
        self.chat.messages += [{"role": "tool", "tool_call_id": "t%d" % k, "content": "short"} for k in range(agent.RESULT_KEPT)]
        self.assertEqual(agent.cut_results(self.chat.messages, 100000), 1)
        self.assertTrue(next(m for m in self.chat.messages if m.get("tool_call_id") == "r")["content"].endswith(agent.CUT_OUT))
        self.assertFalse(next(m for m in self.chat.messages if m.get("tool_call_id") == "new")["content"].endswith(agent.CUT_OUT))

    def test_a_full_memory_with_no_picture_left_cuts_old_results_and_the_job_goes_on(self):
        """Lane 5's 130454-90c1: its reel joined, step 25, and "no picture is left to take out": no clip for the person."""
        self.long_results(8000, 8000)

        def full(asking):
            raise mind.TooLong("full", 32800, 32768)

        brain = ScriptedMind(full, "Done.")
        _, events = self.run_job(brain)
        self.assertEqual(events[-1]["why"], "answered")
        notes = [e["text"] for e in events if e["kind"] == "note"]
        self.assertIn("took out the rest of 2 long old results, and asked again", notes[0])
        self.assertEqual(sum(str(m.get("content")).endswith(agent.CUT_OUT) for m in brain.sent[-1]), 2)

    def test_pictures_come_out_before_any_result_is_cut(self):
        self.long_results(8000)
        picture = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}
        self.chat.messages[1:1] = [{"role": "user", "content": [picture] * 20}, {"role": "user", "content": [picture]}]
        self.assertEqual(agent.free_room(self.chat.messages, 100), (1, 0))
        self.assertEqual(agent.free_room(self.chat.messages, 20 * agent.PICTURE_TOKENS), (19, 1), "what pictures leave")

    def test_a_memory_nearly_full_with_no_picture_left_cuts_results_before_the_question(self):
        self.long_results(8000, 8000)
        agent.ROOMS["scripted"] = 32768
        self.addCleanup(agent.ROOMS.clear)
        self.chat.held = 30000
        brain = ScriptedMind("Done.")
        _, events = self.run_job(brain)
        self.assertIn("took out the rest of 2 long old results before asking",
                      [e["text"] for e in events if e["kind"] == "note"][0])

    def test_a_full_memory_with_no_picture_to_take_out_ends_the_job_saying_so(self):
        def full(asking):
            raise mind.TooLong("full", 32800, 32768)

        job, events = self.run_job(ScriptedMind(full))
        self.assertEqual(events[-1]["why"], "mind")
        self.assertIn("nothing is left to take out", [e["text"] for e in events if e["kind"] == "note"][-1])

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

    def test_a_new_plan_keeps_what_is_done(self):
        """EP0 11-02: Muse rewrote its plan midway with "... - done", and the studio asked it to go on after the hand-over."""
        brain = ScriptedMind([call("set_plan", steps=["a", "b", "c"], why="plan")],
                             [call("find_recordings", why="look", plan_step=1)],
                             [call("find_recordings", words="sifu", why="on", plan_step=2)],
                             [call("set_plan", steps=["a", "b reworded - done", "c", "d"], why="plan again")],
                             [call("find_recordings", words="ep0", why="on", plan_step=3)],
                             [call("find_recordings", words="ep22", why="on", plan_step=4)],
                             "All done.")
        job, events = self.run_job(brain)
        items = [e for e in events if e["kind"] == "plan"][-1]["items"]
        self.assertEqual([(p["text"], p["state"]) for p in items], [(s, "done") for s in ("a", "b reworded", "c", "d")])
        self.assertEqual([e for e in events if e["kind"] == "note"], [])
        self.assertEqual([a["final"] for a in events if a["kind"] == "answer"], [True])

    def test_a_rewritten_plan_does_not_tick_a_step_nothing_worked_on(self):
        brain = ScriptedMind([call("set_plan", steps=["a", "b"], why="plan")],
                             [call("find_recordings", why="look", plan_step=1)],
                             [call("set_plan", steps=["a", "b", "c"], why="plan again")],
                             [call("find_recordings", words="sifu", why="on", plan_step=3)],
                             "Stopped.", "b was not needed.")
        job, events = self.run_job(brain)
        self.assertEqual([p["state"] for p in [e for e in events if e["kind"] == "plan"][-1]["items"]],
                         ["done", "todo", "done"])

    def test_a_step_passed_over_stays_open_and_gets_the_reminder(self):
        """EP0 10-31: step 5 was SAM, Muse named step 6, and the page ticked all 6 with no cut-out made."""
        brain = ScriptedMind([call("set_plan", steps=["a", "b", "c"], why="plan")],
                             [call("find_recordings", why="look", plan_step=1)],
                             [call("find_recordings", words="sifu", why="on to the last", plan_step=3)],
                             "All done.", "b was not needed.")
        job, events = self.run_job(brain)
        plans = [e for e in events if e["kind"] == "plan"]
        self.assertEqual([p["state"] for p in plans[-1]["items"]], ["done", "todo", "done"])
        self.assertIn("[studio] Your plan has steps no call has named yet: 2. b.", self.chat.messages[-2]["content"])

    def test_a_tool_call_written_as_words_is_never_shown_and_is_asked_again_once(self):
        """Walk 11-01, Spark Open: raw tags reached the person as the answer, and nothing ran."""
        raw = ("<|start|>assistant to=commentary<|message|>ย้อนให้แล้วค่ะ<|eom|><|start|>assistant to=mute_sound<|message|>"
               "<atem:function_calls>\n<atem:invoke name=\"mute_sound\">")
        job, events = self.run_job(ScriptedMind(raw, "เรียบร้อยค่ะ"))
        self.assertEqual([(a["text"], a["final"]) for a in events if a["kind"] == "answer"], [("เรียบร้อยค่ะ", True)])
        self.assertEqual(self.chat.messages[-2]["content"], agent.TOOL_AS_WORDS)
        self.assertFalse([m for m in self.chat.messages if "<|" in str(m.get("content"))], "nor does Muse read it back")
        self.assertIn("came out as code", [e for e in events if e["kind"] == "note"][-1]["text"])

    def test_a_second_answer_of_tags_shows_only_its_words(self):
        raw = "<|start|>assistant to=commentary<|message|>ย้อนให้แล้วค่ะ<|eom|>"
        job, events = self.run_job(ScriptedMind(raw, raw))
        self.assertEqual([(a["text"], a["final"]) for a in events if a["kind"] == "answer"], [("ย้อนให้แล้วค่ะ", True)])

    def test_an_answer_of_nothing_but_tags_says_so(self):
        raw = "<|start|>assistant to=mute_sound<|message|><atem:function_calls>"
        job, events = self.run_job(ScriptedMind(raw, raw))
        self.assertEqual([a["text"] for a in events if a["kind"] == "answer"],
                         ["Muse's answer came out as code. Please say it again."])

    def test_an_answer_of_plain_words_with_a_less_than_sign_stands(self):
        job, events = self.run_job(ScriptedMind("ใช้เวลา < 3 นาที"))
        self.assertEqual([a["text"] for a in events if a["kind"] == "answer"], ["ใช้เวลา < 3 นาที"])

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
                      "(Game stream, not a match with a score), montage (Trip or event clips), seller (Product review or selling live), "
                      "sport (Match with a score, real or in a video game), talk "
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
        self.assertEqual(self.chat.clip["recipe"], "game")

    def test_a_length_the_person_never_named_is_the_recipes_and_one_they_named_is_theirs(self):
        """Lane 5 10-30: Glimmer gave talk 45 s with the recipe at 60, its four looks were 45 s each, and it picked the
        clip Tarm had called cut short; over every chat, 49 of its 106 guesses were shorter than the recipe's."""
        self.chat.recording = {"path": "talk.mp4", "name": "talk.mp4", "seconds": 600.0}
        talk = tools.RECIPES["talk"]["length"]
        brain = ScriptedMind([call("load_recipe", type="talk", length_s=45, why="a podcast")], "Loaded.")
        _, events = self.run_job(brain)
        said = [e["result"] for e in events if e["kind"] == "step"][0]
        self.assertEqual(self.chat.clip["length_s"], talk)
        self.assertIn("The person named no length, so the clip is the recipe's %d s, not the 45 s you gave" % talk, said)
        self.assertIn("about %d s" % talk, said)
        self.chat.emit("person", job=None, text="(tapped) Make it shorter, about 45 s.")
        brain = ScriptedMind([call("load_recipe", type="talk", length_s=45, why="shorter, as asked")], "Loaded.")
        _, events = self.run_job(brain)
        self.assertEqual(self.chat.clip["length_s"], 45, "the Shorter button named it")
        self.assertNotIn("named no length", [e["result"] for e in events if e["kind"] == "step"][0])

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
        self.assertEqual([m["role"] for m in self.chat.messages], ["system", "user", "user", "assistant"])
        self.assertEqual(self.chat.messages[2]["content"], agent.TALK_ONLY, "or Muse says it did what no tool ran")

    def test_undo_a_saved_look_and_fonts_are_named_as_work(self):
        """Walk 11-01: each was read as talk, and Muse then said it had done them. Measured on real Muses, 36 of 36."""
        for words in ("undo or take back", "save or use a look", "which recordings or fonts", '("undo that, no new cover")'):
            self.assertIn(words, " ".join(needs.SYSTEM.split()))

    def test_the_countdown_is_offered_only_once_the_person_asked_for_one(self):
        def names(brain):
            return {t["function"]["name"] for t in brain.offered[0]}

        brain = ScriptedMind("ok")
        self.say(brain, "ทำคลิปตลกจากไลฟ์นี้ 1 คลิป ใส่เอฟเฟกต์ตรงจังหวะฮาด้วย")
        self.assertNotIn("make_countdown", names(brain), "it refuses unless asked, and offered it moved Glimmer's pick")
        self.assertEqual(len(names(brain)), len(tools.TOOLS) - 2 - len(tools.ALBUM_TOOLS),
                         "every other tool is still offered, but the album's and no_chat_box, which wait for their moment")
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

    def test_a_chat_that_went_on_to_another_recording_is_named_again(self):
        """Lane 5 11-01: dropped into the Sifu chat, a chicken pot kept the Sifu name in the list."""
        opened = lambda name: {"kind": "opened", "recording": {"name": name}}
        named = {"kind": "named", "title": "x"}
        self.assertEqual(agent._renamed_from([opened("a")]), 0)
        self.assertIsNone(agent._renamed_from([opened("a"), named]))
        self.assertIsNone(agent._renamed_from([opened("a"), named, opened("a")]), "the same one again")
        self.assertEqual(agent._renamed_from([opened("a"), named, opened("b")]), 2)
        self.assertIsNone(agent._renamed_from([opened("a"), named, opened("b"), named]))
        self.say(ScriptedMind([call("find_recordings", why="look")], "Found.", '{"name": "ไฮไลต์ Sifu"}'), "หาไลฟ์ Sifu")
        self.chat.emit("opened", recording={"name": "pots.mkv"})
        brain = ScriptedMind([call("find_recordings", why="look")], "Found.", '{"name": "ตุ๋นไก่หม้อแรก"}')
        self.say(brain, "ทำคลิปหม้อแรก")
        self.assertEqual(self.named(), ["ไฮไลต์ Sifu", "ตุ๋นไก่หม้อแรก"])
        asked = brain.sent[-1][-1]["content"]
        self.assertIn("- ทำคลิปหม้อแรก", asked)
        self.assertNotIn("Sifu", asked, "the name comes from what was said about this recording")
        self.assertEqual(agent.earlier()[0]["title"], "ตุ๋นไก่หม้อแรก")

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


class KeptAnswers(unittest.TestCase):
    """Tarm 11-03 on EP0: a stop at who speaks, turn 8 of 10, threw away $0.72 of Muse Wonder, and the follow-up asked
    every turn again from turn 1."""

    def setUp(self):
        from clipstudio import recording
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, self.tmp
        os.makedirs(os.path.join(self.tmp, "voices"))
        self.rec = {"path": "ep0.mp4", "name": "ep0.mp4", "size": 1, "mtime": 1, "width": 1920, "height": 1080, "seconds": 600}
        self.moment = {"start": 100.0, "end": 160.0, "said": []}
        self.persons = [{"n": 1, "box": [0, 0, 10, 10]}, {"n": 2, "box": [20, 0, 10, 10]}]
        self.asked, self.heard = [], []
        self.turns = [(0.0, 10.0, "A"), (10.0, 20.0, "B"), (20.0, 30.0, "A"), (30.0, 40.0, "B")]

    def tearDown(self):
        from clipstudio import recording
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_job(self, stop_after=None):
        from types import SimpleNamespace
        from clipstudio import asr, hearing, voices
        lines = []

        def ask_other(key, messages, schema, doing, room):
            if stop_after is not None and len(self.asked) >= stop_after:
                raise hearing.Stopped()
            self.asked.append(doing)
            lines.append(doing)
            return {"text": json.dumps({"who": 1 + len(self.asked) % 2, "sure": 0.9}), "engine": {"usd": 0.09}}
        job = SimpleNamespace(chat=SimpleNamespace(id="c"), n=1, ask_other=ask_other, working=lambda line: None,
                              done_working=lambda: None, languages=lambda: ())

        def turns(*a):
            self.heard.append(1)
            return list(self.turns), 0.01
        import numpy as np
        with mock.patch.object(voices, "sound", lambda *a: np.zeros(16000, np.float32)), \
                mock.patch.object(asr, "turns", turns), mock.patch.object(tools.spent, "add"), \
                mock.patch.object(voices, "numbered", lambda rec, a, b, persons, name: name + ".mp4"), \
                mock.patch.object(voices, "who_question", lambda path, persons: [path]):
            return tools._named_voices(job, self.rec, self.moment, self.persons, None), lines

    def test_a_stop_keeps_every_answer_paid_for_and_the_next_job_goes_on_from_there(self):
        from clipstudio import hearing
        with self.assertRaises(hearing.Stopped):
            self.run_job(stop_after=2)
        self.assertEqual(len(self.asked), 2)
        (named, said), lines = self.run_job()
        self.assertEqual(len(self.asked), 4, "the two paid answers were not asked again")
        self.assertEqual(lines, ["Muse Wonder is watching who speaks, turn 3 of 4 (50%)",
                                 "Muse Wonder is watching who speaks, turn 4 of 4 (75%)"])
        self.assertEqual(sum(self.heard), 1, "the turns were kept too, so Muse Voice Transcribe was paid once")
        self.assertEqual([t["who"] for t in named], [2, 1, 2, 1])
        self.assertIn("2 answers from an earlier run cost nothing now", said)
        self.assertEqual(len(self.run_job()[1]), 0, "the same moment again asks nothing at all")

    def test_a_watched_stretch_is_watched_once(self):
        from clipstudio import watch
        clip = os.path.join(self.tmp, "stretch.mp4")
        open(clip, "wb").close()
        calls = []

        def ask_other(key, messages, schema, doing, room):
            calls.append(key)
            return {"text": json.dumps({"moments": []}), "engine": {"usd": 0.12}}
        from types import SimpleNamespace
        job = SimpleNamespace(chat=SimpleNamespace(id="c", moments=[1], project=None), n=1, ask_other=ask_other,
                              working=lambda line: None, done_working=lambda: None, started_work=lambda stop: None,
                              stopped=False, show=lambda **k: None)
        wonder = mock.Mock()
        wonder.secret.return_value = "key"
        with mock.patch.dict(tools.mind.MINDS, {"wonder": wonder}), mock.patch.object(watch, "cut", lambda *a, **k: clip), \
                mock.patch.object(watch, "question", lambda *a: []), mock.patch.object(tools, "_here", lambda job: False),                 mock.patch.object(tools, "_open", lambda job, why: dict(self.rec)), mock.patch.object(tools, "_asked", lambda job, said: True):
            first = tools.watch_piece(job, "who reacts", 0, 30)
            again = tools.watch_piece(job, "who reacts", 0, 30)
        self.assertEqual(calls, ["wonder"])
        self.assertIn("$0.120", first)
        self.assertIn("watched in an earlier run, so it cost nothing now", again)


class WorkSince(unittest.TestCase):
    def test_the_work_of_one_step_keeps_when_it_began_and_the_next_step_starts_again(self):
        import threading
        from types import SimpleNamespace
        chat = SimpleNamespace(cond=threading.Condition(), live=None, live_rev=0)
        job = SimpleNamespace(chat=chat, n=1, steps=8)
        with mock.patch.object(agent.time, "time", return_value=1000.0):
            agent.Job.working(job, "turn 5")
        with mock.patch.object(agent.time, "time", return_value=1250.0):
            agent.Job.working(job, "turn 6")
            self.assertEqual(chat.live["since"], 1000.0)
            self.assertEqual(chat.live["work"], "turn 6")
            self.assertEqual(chat.live["at"], 1250.0, "a new line is news")
        with mock.patch.object(agent.time, "time", return_value=1290.0):
            agent.Job.working(job, "turn 6")
            self.assertEqual(chat.live["at"], 1250.0, "the same line again is not")
        with mock.patch.object(agent.time, "time", return_value=1250.0):
            job.steps = 9
            agent.Job.working(job, "the next tool")
            self.assertEqual(chat.live["since"], 1250.0)
            job.n = 2
            agent.Job.working(job, "the next tool")
            self.assertEqual(chat.live["since"], 1250.0)
            chat.live = {"job": 2, "step": 9, "thinking": "", "text": ""}
            with mock.patch.object(agent.time, "time", return_value=1300.0):
                agent.Job.working(job, "after thinking")
            self.assertEqual(chat.live["since"], 1300.0, "work after Muse thought again starts its own clock")


if __name__ == "__main__":
    unittest.main()
