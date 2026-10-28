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


if __name__ == "__main__":
    unittest.main()
