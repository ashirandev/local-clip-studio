"""Glimmer's slips of the Thai marks (words.unslipped), put back on what the person reads: each step's why, what Muse
says with it, and its answer. The lines are Glimmer's own, from Tarm's beat 1 (10-29).

    python -m unittest test_slips
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, words  # noqa: E402
from test_agent_loop import ScriptedMind, call  # noqa: E402
import test_editor  # noqa: E402

SLIPPED = {
    "สร้างปกคลิปเพื่อดึงผู้ชมที้เลื่อนผ่าน": "สร้างปกคลิปเพื่อดึงผู้ชมที่เลื่อนผ่าน",
    "ใส้สติ๊กเกอรืหัวระเบิดเพื่อเน้นความตลกที่จังหวะตาย": "ใส่สติ๊กเกอร์หัวระเบิดเพื่อเน้นความตลกที่จังหวะตาย",
    "ดูภาพและคำพูดในช้วงที่เสียงดังสุด": "ดูภาพและคำพูดในช่วงที่เสียงดังสุด",
    "ยืนยันความยาวและเอฟเฟกตื": "ยืนยันความยาวและเอฟเฟกต์",
    "โดนลุมจนมองไม่เห็นศัตรู จะรอดไหม": "โดนรุมจนมองไม่เห็นศัตรู จะรอดไหม",
}
UNMARKED = {
    "Akihabara ถนนเกมและป๊อปคัลเจอร": "Akihabara ถนนเกมและป๊อปคัลเจอร์",
    "เดินชมถนนเกมและป๊อปคัลเจอรใน Akihabara": "เดินชมถนนเกมและป๊อปคัลเจอร์ใน Akihabara",
    "เปิดทุกคลิปในโฟลเดอรเพื่อดู": "เปิดทุกคลิปในโฟลเดอร์เพื่อดู",
    "ซ่อนชื่อของสตรีมเมอร": "ซ่อนชื่อของสตรีมเมอร์",
    "ตรวจสอบหลังเรนเดอรใหม่": "ตรวจสอบหลังเรนเดอร์ใหม่",
}
HEARD_SLIP = "เหมือนว่าเดิมว้ยนะ"


class Slips(unittest.TestCase):
    def test_each_slip_is_put_back(self):
        for slipped, meant in SLIPPED.items():
            self.assertEqual(words.unslipped(slipped), meant)

    def test_what_is_right_or_not_one_swap_away_stays_as_written(self):
        for text in ("ตัดคลิปเกมให้หน่อย", "Render it again.", "ไตเติลสั้นๆ กับแฮชแท็ก",
                     "ใส่สติกเกอรหัวระเบิด", "ตรวจสอคคลิป", "พร้อมทิ้ตเติลเพื่อโพสต์", "ลองค้นหาอีกคร้ง"):
            self.assertEqual(words.unslipped(text), text)

    def test_a_ror_stays_a_ror(self):
        for text in ("ข่าวหุ้น คริปโต น้ำมัน", "ตอนผลลัพธ์ปรากฎ ตรวจสอบผล", "ลองเล่นเกมนี้แล้วชอบ"):
            self.assertEqual(words.unslipped(text), text)

    def test_a_dropped_mark_is_put_back_where_the_word_ends(self):
        for slipped, meant in UNMARKED.items():
            self.assertEqual(words.marked(slipped), meant)
            self.assertEqual(words.unslipped(slipped), meant)
        for text in ("เราเจอรถคันนี้", "เธอรู้ไหม", "ผมเจอร้านนี้", "ชื่อเอมอร", "คอมพิวเตอร์", "เลเซอร�์ใน",
                     "ใส่สติกเกอรหัวระเบิด", None):
            self.assertEqual(words.marked(text), text)


class SaysWithItsCall(ScriptedMind):
    """Says words with its first call, as Glimmer does."""

    def __init__(self, said, *script):
        ScriptedMind.__init__(self, *script)
        self.said = said

    def ask(self, messages, **kw):
        got = ScriptedMind.ask(self, messages, **kw)
        if got["tool_calls"] and self.said:
            got["text"] = got["message"]["content"] = self.said
            self.said = None
        return got


@unittest.skipUnless(test_editor.FFMPEG, "needs ffmpeg")
class OnThePage(unittest.TestCase):
    setUpClass = classmethod(test_editor.Tools.setUpClass.__func__)
    tearDownClass = classmethod(test_editor.Tools.tearDownClass.__func__)
    setUp, tearDown = test_editor.Tools.setUp, test_editor.Tools.tearDown
    run_job, to_moment = test_editor.Tools.run_job, test_editor.Tools.to_moment

    def test_titles_are_put_back_around_muses_spelling_check_and_heard_words_stay_as_heard(self):
        self.assertNotEqual(words.unslipped(HEARD_SLIP), HEARD_SLIP, "a swap would change it")
        self.worker.phrases = [{"from": 2.0, "to": 4.0, "text": HEARD_SLIP}]
        dropped, slipped = "Akihabara ถนนเกมและป๊อปคัลเจอร", "เดินเล่นที้ตลาด"
        _, steps = self.run_job(
            *self.to_moment()[:2],
            [call("propose_moments", why="the best", moments=[{"look": 1, "reason": "the walk"}])],
            '{"lines": ["%s"]}' % HEARD_SLIP,
            [call("make_clip", moment=1, layout="subject", why="the walk")],
            [call("add_title", words=dropped, start=0, end=3, place="middle", why="the place")],
            '{"texts": ["%s"]}' % dropped,
            [call("add_title", words=slipped, start=3, end=6, place="middle", why="the walk")],
            '{"texts": ["%s"]}' % slipped,
            [call("add_captions", why="say what he says")],
            '{"texts": ["%s"]}' % HEARD_SLIP, "Done.")
        first, second, caps = steps[-3:]
        self.assertFalse(first["refused"], first["result"])
        self.assertFalse(second["refused"], second["result"])
        self.assertEqual([t["text"] for t in self.chat.project.state()["titles"]],
                         [UNMARKED[dropped], "เดินเล่นที่ตลาด"],
                         "Muse's check dropped the mark again and passed the slip, and the program put both back")
        self.assertIn('"%s" is now "%s"' % (dropped, UNMARKED[dropped]), first["result"])
        self.assertFalse(caps["refused"], caps["result"])
        shown = "".join(line["text"] for line in self.chat.project.state()["captions"]["lines"])
        self.assertEqual("".join(shown.split()), HEARD_SLIP, "words heard are not Glimmer's to put back")

    def test_the_why_what_is_said_and_the_answer_are_shown_put_back_and_muse_keeps_its_own(self):
        brain = SaysWithItsCall("เดี๋ยวเปิดไฟล์ที้ยาวก่อน",
                                [call("open_recording", path=self.clip, why="เปิดไฟล์ที้ยาวเพื่อเรนเดอรือีกครั้ง")],
                                "เรนเดอรือีกครั้งแล้ว")
        self.chat.jobs += 1
        job = self.chat.job = agent.Job(self.chat, brain, "open it")
        job.run()
        steps = [e for e in self.chat.events if e.get("job") == job.n and e["kind"] == "step"]
        self.assertEqual(steps[-1]["why"], "เปิดไฟล์ที่ยาวเพื่อเรนเดอร์อีกครั้ง")
        self.assertEqual(steps[-1]["said"], "เดี๋ยวเปิดไฟล์ที่ยาวก่อน")
        answer = [e for e in self.chat.events if e["kind"] == "answer"][-1]
        self.assertEqual(answer["text"], "เรนเดอร์อีกครั้งแล้ว")
        self.assertIn("ที้ยาว", "".join(str(m.get("content")) + str(m.get("tool_calls")) for m in self.chat.messages),
                      "the conversation Muse reads keeps what it wrote")


if __name__ == "__main__":
    unittest.main()
