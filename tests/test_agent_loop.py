"""The agent loop, run against a scripted mind: python -m unittest discover -s tests"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, mind, recipes, tools  # noqa: E402


def call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedMind:
    """Answers with the next item of its script and keeps a copy of every conversation it was sent.

    An item is the words of an answer, a list of tool calls, or a function(asking) that acts while the question is
    in flight (the person types, or presses Stop) and then raises or returns one of the other two.
    """
    label = "scripted"

    def __init__(self, *script):
        self.script, self.sent = list(script), []

    def ask(self, messages, tools=None, temperature=None, max_tokens=None, on_delta=None, asking=None):
        self.sent.append(json.loads(json.dumps(messages)))
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

    def test_the_answer_itself_does_the_last_plan_step(self):
        brain = ScriptedMind([call("set_plan", steps=["find both", "compare them"], why="plan")],
                             [call("find_recordings", why="look", plan_step=1)], "axi has more.")
        _, events = self.run_job(brain)
        self.assertEqual([e["kind"] for e in events if e["kind"] in ("answer", "note")], ["answer"])
        plan = [e for e in events if e["kind"] == "plan"][-1]
        self.assertEqual([p["state"] for p in plan["items"]], ["done", "done"])

    def test_the_recipe_reaches_muse_the_page_and_the_state(self):
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
        self.assertIn("There is no recipe for cooking yet. The recipes are: game (Game stream), seller", results[0])
        self.assertIn("Do not use a recipe made for another kind of video", results[0])
        self.assertIn("style must be one of: clean, pop, calm", results[1])
        self.assertIsNone(self.chat.clip)

    def test_another_recipe_replaces_the_first(self):
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


if __name__ == "__main__":
    unittest.main()
