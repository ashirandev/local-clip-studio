"""Muse Wonder watching a stretch with its sound (watch.py, the watch_piece tool), with a fake Wonder that keeps what it
was sent: nothing goes to Meta.

    python -m unittest test_watch
"""
import base64
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import mind, recording, watch  # noqa: E402
from test_agent_loop import call  # noqa: E402
import test_editor  # noqa: E402


class FakeWonder:
    key, label, model_id = "wonder", "Muse Wonder", "muse-wonder-1.0"

    def __init__(self, answer, key="a key"):
        self.answer, self.key_value, self.sent = answer, key, []

    def secret(self):
        return self.key_value

    def ask(self, messages, tools=None, temperature=None, max_tokens=None, on_delta=None, asking=None, schema=None):
        self.sent.append((messages, schema))
        return {"text": self.answer, "thinking": "", "tool_calls": [], "message": {"role": "assistant"},
                "engine": {"read_new": 2735, "reused": 0, "written": 2375, "seconds": 30.0, "usd": 0.0584}}


SAW = json.dumps({"moments": [
    {"at": 4.3, "kind": "scream", "what": "He yelps as the hit lands", "effect": "shake", "word": ""},
    {"at": 12.0, "kind": "silent joke", "what": "He stares as the number jumps", "effect": "word", "word": "44"},
    {"at": 99, "kind": "laugh", "what": "He laughs at the end", "effect": "pop", "word": "ignored"},
    {"at": "soon", "kind": "laugh", "what": "no second", "effect": "pop", "word": ""}]})


@unittest.skipUnless(test_editor.FFMPEG, "needs ffmpeg")
class Watch(unittest.TestCase):
    setUpClass = classmethod(test_editor.Tools.setUpClass.__func__)
    tearDownClass = classmethod(test_editor.Tools.tearDownClass.__func__)
    run_job, to_moment = test_editor.Tools.run_job, test_editor.Tools.to_moment

    def setUp(self):
        test_editor.Tools.setUp(self)
        self.real_wonder = mind.MINDS.get("wonder")
        self.wonder = mind.MINDS["wonder"] = FakeWonder(SAW)

    def tearDown(self):
        mind.MINDS["wonder"] = self.real_wonder
        test_editor.Tools.tearDown(self)

    def test_wonder_watches_the_stretch_and_its_moments_come_back_at_the_clips_seconds(self):
        _, steps = self.run_job(*self.to_moment(), [call("make_clip", moment=1, layout="subject", why="the action")],
                                [call("watch_piece", start="0:30", end="0:50", why="what the words miss")], "Done.")
        step = steps[-1]
        self.assertFalse(step["refused"], step["result"])
        messages, schema = self.wonder.sent[-1]
        self.assertIs(schema, watch.SCHEMA)
        parts = messages[0]["content"]
        self.assertIn("This is 20 seconds of stream.mp4, with its sound.", parts[0]["text"])
        video = parts[1]["video_url"]["url"]
        self.assertTrue(video.startswith("data:video/mp4;base64,"))
        self.assertEqual(base64.b64decode(video.split(",", 1)[1])[4:8], b"ftyp", "a real mp4 is sent")
        self.assertIn("Muse Wonder watched 0:30-0:50 with its sound (", step["result"])
        self.assertIn("$0.058", step["result"])
        self.assertIn("- 0:34.3 (4.3 s of the clip), scream: He yelps as the hit lands → add_hit kind shake at 4.3",
                      step["result"], "the clip opens at 30 s of the recording")
        self.assertIn('- 0:42.0 (12.0 s of the clip), silent joke: He stares as the number jumps → add_hit kind word '
                      'word "44" at 12.0', step["result"])
        self.assertNotIn("laughs at the end", step["result"], "a second past the 20 s it was sent is not guessed")
        self.assertNotIn("no second", step["result"], "nor a moment without a second")
        with open(os.path.join(recording.WORK, "spent.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual([(r["label"], r["usd"]) for r in rows if r["mind"] == "wonder"], [("Muse Wonder", 0.0584)],
                         "what Wonder cost is written under Wonder, whichever Muse runs the chat")
        cut = os.path.join(watch.folder(), watch.name(self.chat.recording, 30.0, 50.0))
        self.assertTrue(os.path.isfile(cut))
        when = os.path.getmtime(cut)
        self.run_job([call("watch_piece", start="0:30", end="0:50", why="again")], "Done.")
        self.assertEqual(os.path.getmtime(cut), when, "the same stretch is cut once")

    def test_without_a_key_nothing_is_sent_and_a_long_stretch_is_refused(self):
        self.wonder.key_value = ""
        _, steps = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                [call("watch_piece", start="0:30", end="0:50", why="no key")], "Done.")
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("none is set, so nothing was sent", steps[-1]["result"])
        self.assertEqual(self.wonder.sent, [])
        self.wonder.key_value = "a key"
        _, steps = self.run_job([call("watch_piece", start="0:00", end="0:10", why="short enough")],
                                [call("watch_piece", start="0:10", end="0:10.5", why="too short")], "Done.")
        self.assertFalse(steps[-2]["refused"], steps[-2]["result"])
        self.assertIn("outside the clip", steps[-2]["result"], "with no clip yet, every moment is outside it")
        self.assertTrue(steps[-1]["refused"])
        old, watch.LONGEST = watch.LONGEST, 15
        try:
            _, steps = self.run_job([call("watch_piece", start="0:30", end="0:50", why="too long")], "Done.")
        finally:
            watch.LONGEST = old
        self.assertIn("Muse Wonder watches at most 15 s at a time, and 0:30-0:50 is 20 s.", steps[-1]["result"])

    def test_an_answer_that_is_not_the_list_is_said_and_nothing_is_made_up(self):
        self.wonder.answer = "I saw a man playing a game."
        _, steps = self.run_job([call("open_recording", path=self.clip, why="open it")],
                                [call("watch_piece", start="0:30", end="0:50", why="look")], "Done.")
        self.assertEqual(steps[-1]["result"], "Muse Wonder did not watch 0:30-0:50: Muse Wonder's answer was not the "
                                              "list asked for, so nothing it saw can be used. Go on from your looks.")
        self.wonder.answer = '{"moments": []}'
        _, steps = self.run_job([call("watch_piece", start="0:30", end="0:50", why="look")], "Done.")
        self.assertTrue(steps[-1]["result"].endswith("and saw nothing a viewer would react to that the words do not "
                                                     "already carry."), steps[-1]["result"])
