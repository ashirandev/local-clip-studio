"""The list of earlier chats: python -m unittest discover -s tests"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent  # noqa: E402


class DeleteChat(unittest.TestCase):
    ID, OTHER = "20261103-090000-aaaa", "20261103-090001-bbbb"

    def setUp(self):
        self.tmp, self.work = tempfile.mkdtemp(), agent.WORK
        agent.WORK = self.tmp
        self.chats = os.path.join(self.tmp, "chats")
        os.makedirs(self.chats)
        self.had, self.were = dict(agent.CHATS), list(agent._shown)
        agent.CHATS.clear()
        agent._shown.clear()
        for cid in (self.ID, self.OTHER):
            for end, row in ((".events.jsonl", {"kind": "person", "text": "cut this"}), (".messages.jsonl", {"role": "user", "content": "cut this"})):
                with open(os.path.join(self.chats, cid + end), "w", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

    def tearDown(self):
        agent.WORK = self.work
        agent.CHATS.clear()
        agent.CHATS.update(self.had)
        agent._shown[:] = self.were
        shutil.rmtree(self.tmp, ignore_errors=True)

    def listed(self):
        return sorted(c["id"] for c in agent.earlier())

    def here(self, cid=None):
        return sorted(n for n in os.listdir(self.chats) if n.startswith(cid or self.ID))

    def test_a_deleted_chat_leaves_the_list_and_its_two_files_can_be_put_back_by_hand(self):
        """Tarm 11-03: there was no way to delete a chat. A move, so a wrong yes can be taken back."""
        self.assertIsNone(agent.delete_chat(self.ID))
        self.assertEqual(self.listed(), [self.OTHER])
        self.assertIsNone(agent.open_chat(self.ID))
        self.assertEqual(self.here(), [])
        kept = os.path.join(self.chats, "deleted")
        self.assertEqual(len(os.listdir(kept)), 2)
        for name in os.listdir(kept):
            os.replace(os.path.join(kept, name), os.path.join(self.chats, name))
        self.assertEqual(self.listed(), [self.ID, self.OTHER])

    def test_its_clips_and_what_it_cost_stay(self):
        paths = [os.path.join(self.tmp, "spent.jsonl"), os.path.join(self.tmp, "clip.mp4")]
        for path in paths:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.ID)
        agent.delete_chat(self.ID)
        for path in paths:
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), self.ID)

    def test_a_chat_at_work_stays(self):
        for working in ("job", "making", "basing", "layering"):
            chat = mock.Mock(making=None, basing=None, layering=None)
            chat.busy.return_value = working == "job"
            if working != "job":
                setattr(chat, working, object())
            agent.CHATS[self.ID] = chat
            self.assertIn("working", agent.delete_chat(self.ID), working)
            self.assertEqual(len(self.here()), 2, working)
            self.assertIs(agent.find(self.ID), chat)

    def test_the_chat_every_page_shows_gives_way_to_a_new_one_and_the_pages_follow(self):
        chat = agent.open_chat(self.ID)
        with mock.patch.object(chat, "stop") as woken:
            self.assertIsNone(agent.delete_chat(self.ID))
        woken.assert_called_once_with()
        self.assertNotEqual(agent.shown(), self.ID)
        self.assertIsNotNone(agent.find(agent.shown()))
        self.assertIsNone(agent.find(self.ID))

    def test_a_chat_no_page_shows_leaves_the_pages_where_they_are(self):
        shown = agent.open_chat(self.OTHER)
        agent.open_chat(self.ID)
        agent._show(shown)
        self.assertIsNone(agent.delete_chat(self.ID))
        self.assertEqual(agent.shown(), self.OTHER)
        self.assertIsNone(agent.find(self.ID))

    def test_a_name_that_is_no_chat_moves_nothing(self):
        for cid in ("", None, "../" + self.ID, "..\\" + self.ID, self.ID + ".events.jsonl", self.ID + "\n", "20261103-090002-cccc"):
            self.assertIn("There is no chat", agent.delete_chat(cid), cid)
        self.assertEqual(self.listed(), [self.ID, self.OTHER])
        self.assertFalse(os.path.exists(os.path.join(self.chats, "deleted")))

    def test_a_chat_with_only_its_job_log_can_go_too(self):
        os.remove(os.path.join(self.chats, self.ID + ".messages.jsonl"))
        self.assertIsNone(agent.delete_chat(self.ID))
        self.assertEqual(self.listed(), [self.OTHER])

    def test_a_file_that_cannot_move_leaves_the_chat_whole(self):
        replace, calls = os.replace, []

        def second_refused(a, b):
            calls.append(a)
            if len(calls) == 2:
                raise PermissionError("in use")
            replace(a, b)
        agent.open_chat(self.ID)
        with mock.patch.object(agent.os, "replace", second_refused):
            self.assertIn("could not be deleted", agent.delete_chat(self.ID))
        self.assertEqual(len(self.here()), 2)
        self.assertEqual(self.listed(), [self.ID, self.OTHER])
        self.assertEqual(agent.shown(), self.ID)
        self.assertIsNotNone(agent.find(self.ID))


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

    def test_a_clip_of_an_album_made_again_counts_once_with_its_newest_file(self):
        """Tarm 11-01: the home card said 7 and the album inside held 4, as each clip made again was counted."""
        self.chat("20261031-212136-cdc1", [
            {"kind": "person", "text": "ทำคลิปสินค้า"}, {"kind": "job", "mind": "Muse Spark"}, {"kind": "album"},
            {"kind": "ready", "file": "a.mp4", "moment": 1, "title": "จอย", "length": 20.0},
            {"kind": "ready", "file": "b.mp4", "moment": 2, "title": "ยาหม่อง", "length": 21.0},
            {"kind": "ready", "file": "c.mp4", "moment": 1, "title": "จอย", "length": 22.0}])
        got = {c["id"]: c for c in agent.earlier()}["20261031-212136-cdc1"]["clips"]
        self.assertEqual(got, [{"file": "c.mp4", "title": "จอย", "length": 22.0}, {"file": "b.mp4", "title": "ยาหม่อง", "length": 21.0}])

    def test_with_no_album_two_clips_of_a_moment_are_two_clips(self):
        self.chat("20261029-003136-20b0", [
            {"kind": "person", "text": "ทำคลิป"}, {"kind": "job", "mind": "Muse Spark"},
            {"kind": "ready", "file": "a.mp4", "moment": 1, "title": "แรก", "length": 20.0},
            {"kind": "ready", "file": "b.mp4", "moment": 1, "title": "สอง", "length": 21.0}])
        got = {c["id"]: c for c in agent.earlier()}["20261029-003136-20b0"]["clips"]
        self.assertEqual([c["file"] for c in got], ["a.mp4", "b.mp4"])

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
