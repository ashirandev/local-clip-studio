"""A chat's name in the list of earlier chats, thought up by Muse in the person's own words.

Tarm 10-28: the list read "yeti-c149d0193c.mp4" eight times over; named by the newest clip's post title it read like
a post, not like a chat ("ให้ model คิดชื่อเลย"). Muse names the chat once, after the first job that did any work, in
one short question apart: a few words in the person's language, the way a person names a folder.
"""
import json
import re

ROOM = 2048
MOST_LETTERS = 48
SAID_LETTERS = 200
SAID_MOST = 3
SYSTEM = """Name a chat between a person and their video editor, for the person's list of earlier chats. Use a few \
words (six at most) in the language the person writes in, saying what the clip is about, the way a person names a \
folder: "รีวิวไมค์ Yeti Orb", "ไฮไลต์ SIFU ด่านแรก", "Akihabara trip vlog". No quotes, no emoji, no dates, no file or \
folder names."""
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
FILEISH = re.compile(r"\\|(?<![A-Za-z0-9])[A-Za-z]:/|(?:^|\s)/\S|\.(mp4|mov|mkv|avi|webm|m4v)\b", re.I)
QUOTES = "\"'“”‘’«»「」`"


def question(said, answer, title):
    """The messages and schema that ask for a name: what the person typed (said, oldest first), Muse's last answer
    and the finished clip's title ("" before one)."""
    lines = ["The person said:"] + ["- " + s[:SAID_LETTERS] for s in said[:SAID_MOST]]
    if answer:
        lines += ["", "The editor answered: " + answer[:SAID_LETTERS * 2]]
    if title:
        lines += ["", "The finished clip's title: " + title]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}], SCHEMA


def read(answer):
    """The name, tidied, or None when there is none fit for the list: empty, or naming a file or a folder (the list is
    on the page, and a viewer of the screen must not see where the files are kept)."""
    try:
        name = json.loads(answer).get("name")
    except (ValueError, AttributeError):
        return None
    if not isinstance(name, str):
        return None
    name = " ".join(name.strip().strip(QUOTES).split())
    if not name or FILEISH.search(name):
        return None
    if len(name) > MOST_LETTERS:
        name = name[:MOST_LETTERS].rsplit(" ", 1)[0] or name[:MOST_LETTERS]
    return name
