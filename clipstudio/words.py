"""Words Muse writes, tidied by the program before they are shown or read back.

Glimmer sometimes writes the same Thai mark twice ("ค้้น"). Read back in the next question, the slip is copied and
grows step by step (measured 10-27: 2, 3, 4, then 8 marks in a row, then 15,000 tokens of thinking without an
answer). Thai never stacks the same mark twice, so a run of one mark is cut to one.

Muse is asked for plain text, and still writes some markdown (seen 10-27: * bullets and ** bold). The page shows
text as it is, so bold marks and heading marks are dropped and * bullets become - bullets, before the answer is
shown or read back.

Glimmer misspells Thai late in a long run, and not early (measured 10-27, 16 answers each, through the studio's own
Mind: with only the question, no bit any dictionary lacks was misspelled; after a run's 12,500 tokens, 5 to 8 bits
in 100 Thai words, in half the answers; temperature 0.1 instead of 0.5 only halved it). So the words a person keeps
are fixed in a short conversation of their own (spelled): on the 16 long-run answers the pointer flagged, 59 bits
went to 1, every number was kept, and 97% of the text stayed as it was. One question for many texts is cheaper
than one each (8 texts in 40 s, against about 15 s each), because the thinking is shared.
"""
import difflib
import json
import re

from . import thai

THAI_MARK_RUN = re.compile("([ัิ-ฺ็-๎])\\1+")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
HEADING = re.compile(r"^#{1,6}[ \t]+", re.M)
BULLET = re.compile(r"^([ \t]*)[*+][ \t]+", re.M)


def tidy(text):
    return THAI_MARK_RUN.sub(r"\1", text) if isinstance(text, str) else text


def plain(text):
    """An answer without the markdown Muse was asked not to write."""
    if not isinstance(text, str):
        return text
    return BULLET.sub(r"\1- ", HEADING.sub("", BOLD.sub(r"\1", text)))


THAI = re.compile("[\u0e01-\u0e5b]")
CREATOR_WORDS = {"ไตเติล", "แคปชัน", "แคปชั่น", "แฮชแท็ก", "คอมโบ", "รีล", "รีลส์", "ช็อต", "ทรานซิชัน"}
_WORDS = []
SPELL_ASK = ("Fix only the spelling in each of these Thai texts: tone marks, vowels and letters that are wrong. Keep "
             "every word, every number, every hashtag and the meaning; do not reword or add anything. Give back the "
             "same number of texts, in order, each with its line breaks.\n\n%s")
SPELL_SCHEMA = {"type": "object", "properties": {"texts": {"type": "array", "items": {"type": "string"}}},
                "required": ["texts"], "additionalProperties": False}
MOST_CHANGED = 0.2


def odd_thai(words):
    """Bits of Thai in words that no dictionary word makes up, each with the words on either side: the place a tone
    mark or a vowel may be wrong (Glimmer writes ที้ for ที่, เพื่ม for เพิ่ม). PyThaiNLP's own corrections were
    wrong on these (10-27), so the program only points, and Muse fixes the spelling.

    The repeat mark ๆ is not a word (สวยๆ is สวย said twice), and CREATOR_WORDS are loanwords for making clips that the
    dictionary does not have (ไตเติล, แฮชแท็ก). Measured 10-27: in 32 answers Glimmer wrote with a short context, every
    bit pointed at was one of these."""
    if not THAI.search(words or ""):
        return []
    from pythainlp.corpus import thai_words
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import dict_trie
    if not _WORDS:
        _WORDS.append(thai_words() | CREATOR_WORDS)
        _WORDS.append(dict_trie(_WORDS[0]))
    toks = word_tokenize(words, engine="newmm", keep_whitespace=True, custom_dict=_WORDS[1])
    out = []
    for i, w in enumerate(toks):
        base = w.strip().rstrip("ๆ").strip()
        if THAI.search(base) and base not in _WORDS[0]:
            near = [x for x in (toks[i - 1] if i else "", w, toks[i + 1] if i + 1 < len(toks) else "") if x.strip()]
            out.append("".join(near))
    return out


def spelled(ask, texts):
    """(texts with their Thai spelling fixed, [(before, after)] for each one fixed). Only texts odd_thai flags are
    asked about, all in one question to Muse in a conversation of their own: ask(messages, schema) returns the
    answer's words. A fix is kept only when it leaves fewer flagged bits, keeps every number, and changes at most a
    fifth of the text; otherwise the text stays as it was written."""
    texts = list(texts)
    todo = [i for i, t in enumerate(texts) if isinstance(t, str) and odd_thai(t)]
    if not todo:
        return texts, []
    answer = ask([{"role": "user", "content": SPELL_ASK % json.dumps([texts[i] for i in todo], ensure_ascii=False)}],
                 SPELL_SCHEMA)
    try:
        back = json.loads(answer)["texts"]
    except (ValueError, KeyError, TypeError):
        return texts, []
    changed = []
    for i, after in zip(todo, back if isinstance(back, list) else []):
        before = texts[i]
        if (isinstance(after, str) and after != before and len(odd_thai(after)) < len(odd_thai(before))
                and thai.values(after) == thai.values(before)
                and difflib.SequenceMatcher(None, before, after).ratio() >= 1 - MOST_CHANGED):
            texts[i] = after
            changed.append((before, after))
    return texts, changed


def tidy_all(value):
    """Every string inside a JSON value, tidied."""
    if isinstance(value, str):
        return tidy(value)
    if isinstance(value, list):
        return [tidy_all(v) for v in value]
    if isinstance(value, dict):
        return {k: tidy_all(v) for k, v in value.items()}
    return value


def tidy_arguments(raw):
    """Tool call arguments (a JSON string) with every string tidied; unreadable JSON is left as it came."""
    if not isinstance(raw, str):
        return tidy_all(raw)
    try:
        value = json.loads(raw)
    except ValueError:
        return raw
    fixed = tidy_all(value)
    return raw if fixed == value else json.dumps(fixed, ensure_ascii=False)
