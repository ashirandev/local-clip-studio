"""The agent loop, run against a scripted mind: python -m unittest discover -s tests"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatstudio import agent, tools  # noqa: E402


def call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedMind:
    """Answers with the next item of its script and keeps a copy of every conversation it was sent."""
    label = "scripted"

    def __init__(self, *script):
        self.script, self.sent = list(script), []

    def ask(self, messages, tools=None, temperature=None, max_tokens=None):
        self.sent.append(json.loads(json.dumps(messages)))
        item = self.script.pop(0)
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
        job = agent.Job(self.chat, brain, words)
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

    def test_a_job_that_keeps_repeating_is_stopped(self):
        same = call("find_recordings", why="look")
        job, events = self.run_job(ScriptedMind(*[[same]] * 5))
        self.assertEqual(events[-1]["why"], "repeating")
        self.assertEqual(job.steps, 5)


if __name__ == "__main__":
    unittest.main()
