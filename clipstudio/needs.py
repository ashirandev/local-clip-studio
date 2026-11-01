"""What a person's message needs: work (Muse may call tools) or talk (Muse answers in words and nothing runs).

The studio is an editor you can talk to, so a greeting, a thanks or a word about the weather must not start a job.
The rule was in Muse's instructions and was not enough: measured 10-28, in a new chat Glimmer called a tool for 4 of
12 messages that only talked and Spark for 3 ("ฝนตกทั้งวันเลย" went looking for recordings with both). This one short
question, asked apart, sorts 70 of 70 with both (36 talk, 34 work). Glimmer missed "มีไฟล์อะไรบ้าง" until "which
recordings they have" was named; naming "what is in one" too cost it two talk messages ("คลิปนี้ยาวกี่วิ"). It takes
3 to 5 seconds, so it is asked only when Muse's first answer calls a tool. The walk of 11-01 found "ย้อนกลับ ไม่ต้องทำปกใหม่",
"ฟอนต์ไหนเขียนไทยได้บ้าง" and a saved look read as talk, with Muse then saying it had done them: with undo, a look and
fonts named, 36 of 36 on Spark and on Spark Open, against 31 of 36.
"""
import json

ROOM = 2048
LAST_LETTERS = 600
SYSTEM = """A person is talking with their video editor in a chat. Read their newest message (and the editor's last \
answer, when there is one) and say what the message needs.

work: it asks the editor to do something with the person's recordings or clips now: find, open, look at, cut, \
change, add, remove, undo or take back, save or use a look, render or hand over, or it asks which recordings or fonts they have. A yes to something the \
editor offered to do is work. So is a request said politely as a question ("can you make it shorter?") or as a complaint about the clip ("the caption \
is too big"). A request stays work when it also says what not to do ("undo that, no new cover").

talk: anything else. Greetings, thanks, how they feel, small talk, an opinion, a no or a wait with nothing asked for, and questions the \
editor can answer from what it knows: a tip, what it can do, why it made a choice in the clip it already made."""
SCHEMA = {"type": "object", "properties": {"needs": {"type": "string", "enum": ["work", "talk"]}},
          "required": ["needs"]}


def question(last, said):
    """The messages and schema that ask what said needs; last is the editor's last answer in words, or None."""
    shown = ("The editor's last answer: %s\n\n" % last[:LAST_LETTERS] if last else "") + "The person's message: " + said
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": shown}], SCHEMA


def read(answer):
    """"talk" only when the answer says so plainly; anything else is work, so a garbled answer leaves the job as it
    was before this question existed."""
    try:
        return "talk" if json.loads(answer).get("needs") == "talk" else "work"
    except (ValueError, AttributeError):
        return "work"


def last_answer(messages):
    """The words of Muse's last answer that ended a job (no tool calls with it), or None."""
    for m in reversed(messages):
        if m.get("role") == "assistant" and not m.get("tool_calls") and isinstance(m.get("content"), str) \
                and m["content"].strip():
            return m["content"].strip()
    return None
