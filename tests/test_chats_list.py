"""The list of earlier chats: python -m unittest discover -s tests"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent  # noqa: E402


class Earlier(unittest.TestCase):
    def setUp(self):
        self.tmp, self.work = tempfile.mkdtemp(), agent.WORK
        agent.WORK = self.tmp
        os.makedirs(os.path.join(self.tmp, "chats"))

    def tearDown(self):
        agent.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def chat(self, name, events):
        with open(os.path.join(self.tmp, "chats", name + ".events.jsonl"), "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))

    def test_a_chat_is_named_by_its_clip_and_says_which_muse_answered_local_or_api(self):
        """Tarm 10-28: the list read "yeti-c149d0193c.mp4" eight times over, and did not say which ran on the API."""
        self.chat("20261028-215618-dc93", [
            {"kind": "person", "text": "ลองดู H:\\Phone\\yeti ช่วยตัดหน่อย"},
            {"kind": "job", "mind": "Muse Spark"}, {"kind": "opened", "recording": {"name": "yeti-c149d0193c.mp4"}},
            {"kind": "ready", "title": "ร่างแรก"}, {"kind": "job", "mind": "Muse Glimmer"},
            {"kind": "ready", "title": "ลองฟังเสียง Yeti Orb ไมค์ชัดไม่ถึงสองพัน"}])
        self.chat("20261028-220043-e01b", [{"kind": "person", "text": "ทำคลิปจากไลฟ์"},
                                           {"kind": "job", "mind": "Muse Glimmer"}])
        got = {c["id"]: c for c in agent.earlier()}
        spark = got["20261028-215618-dc93"]
        self.assertEqual(spark["title"], "ลองฟังเสียง Yeti Orb ไมค์ชัดไม่ถึงสองพัน", "the newest finished clip names it")
        self.assertEqual(spark["minds"], [{"label": "Muse Spark", "local": False}, {"label": "Muse Glimmer", "local": True}])
        self.assertEqual((got["20261028-220043-e01b"]["title"], got["20261028-220043-e01b"]["first"]),
                         ("", "ทำคลิปจากไลฟ์"), "before a clip is finished, the page names it by what was typed")

    def test_each_chat_says_which_clips_it_made_and_what_it_cost(self):
        """Tarm 10-29: the list of clips says which chat made which clip, and what it cost."""
        self.chat("20261029-003136-20b0", [
            {"kind": "person", "text": "ทำคลิปไฮไลต์"}, {"kind": "job", "mind": "Muse Spark"},
            {"kind": "ready", "file": "01ee4ed55632271c.mp4", "title": "โดนรุมในโกดัง", "length": 30.0},
            {"kind": "ready", "file": "01ee4ed55632271c.mp4", "title": "โดนรุมในโกดัง", "length": 30.0},
            {"kind": "ready", "file": "4d90d3bc8a46a5f0.mp4", "title": "Outnumbered", "length": 30.0}])
        self.chat("20261029-010451-636e", [{"kind": "person", "text": "ทำคลิป"}, {"kind": "job", "mind": "Muse Spark"}])
        self.chat("20261029-010150-6c55", [{"kind": "person", "text": "ทำคลิป"}, {"kind": "job", "mind": "Muse Glimmer"}])
        with open(os.path.join(self.tmp, "spent.jsonl"), "w", encoding="utf-8") as f:
            for row in ({"t": 1, "chat": "20261029-003136-20b0", "job": 1, "mind": "spark", "label": "Muse Spark",
                         "usd": 0.5}, {"t": 2, "chat": "20261029-003136-20b0", "job": 2, "mind": "spark",
                                       "label": "Muse Spark", "usd": 0.25},
                        {"t": 3, "chat": "20261029-010150-6c55", "job": 1, "mind": "local", "label": "Muse Glimmer",
                         "usd": 0.0}):
                f.write(json.dumps(row) + "\n")
        got = {c["id"]: c for c in agent.earlier()}
        self.assertEqual(got["20261029-003136-20b0"]["clips"],
                         [{"file": "01ee4ed55632271c.mp4", "title": "โดนรุมในโกดัง", "length": 30.0},
                          {"file": "4d90d3bc8a46a5f0.mp4", "title": "Outnumbered", "length": 30.0}],
                         "a clip handed over twice is listed once")
        self.assertEqual(got["20261029-003136-20b0"]["usd"], 0.75)
        self.assertIsNone(got["20261029-010451-636e"]["usd"], "an API chat from before answers were priced: not $0")
        self.assertEqual((got["20261029-010150-6c55"]["usd"], got["20261029-010150-6c55"]["clips"]), (0.0, []))


if __name__ == "__main__":
    unittest.main()
