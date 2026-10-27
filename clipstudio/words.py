"""Words Muse writes, tidied by the program before they are shown or read back.

Glimmer sometimes writes the same Thai mark twice ("ค้้น"). Read back in the next question, the slip is copied and
grows step by step (measured 10-27: 2, 3, 4, then 8 marks in a row, then 15,000 tokens of thinking without an
answer). Thai never stacks the same mark twice, so a run of one mark is cut to one. Glimmer also writes half a
Thai letter now and then, which reaches the studio as U+FFFD, the sign for a broken letter (seen 10-27 after the
tone mark of พื้นที่ and of ชั้น); that sign never means anything, so it is taken out.

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

BROKEN = "\ufffd"
THAI_MARK_RUN = re.compile("([ัิ-ฺ็-๎])\\1+")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
HEADING = re.compile(r"^#{1,6}[ \t]+", re.M)
BULLET = re.compile(r"^([ \t]*)[*+][ \t]+", re.M)


def tidy(text):
    return mend(THAI_MARK_RUN.sub(r"\1", text.replace(BROKEN, ""))) if isinstance(text, str) else text


def plain(text):
    """An answer without the markdown Muse was asked not to write."""
    if not isinstance(text, str):
        return text
    return BULLET.sub(r"\1- ", HEADING.sub("", BOLD.sub(r"\1", text)))


THAI = re.compile("[\u0e01-\u0e5b]")
CREATOR_WORDS = {"ไตเติล", "แคปชัน", "แคปชั่น", "แฮชแท็ก", "คอมโบ", "รีล", "รีลส์", "ช็อต", "ทรานซิชัน", "แอป", "เทส",
                 "บอท", "บอต", "เกต", "คอปปี้", "คอมมิชชัน", "บูลีน", "สวีต", "มิลลิวินาที"}
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

    The repeat mark ๆ is not a word (สวยๆ is สวย said twice), and CREATOR_WORDS are loanwords for making clips and
    apps that the dictionary does not have (ไตเติล, แฮชแท็ก, แอป, เทส). Measured 10-27: in 32 answers Glimmer wrote with a short context, every
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


TONES = "่้๊๋"
PATH = re.compile(r"[\\/]")


def mend(text):
    """text with Glimmer's slips taken out where the dictionary says so (seen 10-27 in Muse's reasons on the talk
    clip, where each step copied the one before and added a letter: ชั้่นส่วน, ชั้่นสส่วน, ชั้่นสสส่วน, ชั้่นสสสส่วน).
    Only deletions, so no word is made up, and text with a path in it is left as it is. Only where Thai cannot be
    written so: a tone mark after another tone mark, after ็ or after ์ (ชั้่น, เป็่น), a vowel written again around
    a tone mark (เพื่ือ), and a letter written twice (มมุม), each kept only when fewer bits of the text are no word;
    and three or more of one letter (สสส่วน), cut to one when that is no worse and leaves the text in fewer words. As
    in every slip seen, the last of the letters carries a mark, and a letter written twice is one the dictionary
    leaves standing alone: one that ends a word and starts the next (แบบบน, จอออก, เทสสี, บอทที่) is left as it is.
    Measured 10-27 on 1,500 lines of the team's own Thai notes: looser rules (any tone mark; any letter written twice
    when it left fewer words) changed 21 lines that were right, such as เล็กกว่า to เล็กว่า."""
    if not isinstance(text, str) or not THAI.search(text) or PATH.search(text):
        return text
    odd, count = _odd(text)
    while True:
        for gone, run in _tries(text):
            better = "".join(ch for k, ch in enumerate(text) if k not in gone)
            n, c = _odd(better)
            if n < odd or (run and n <= odd and c < count):
                text, odd, count = better, n, c
                break
        else:
            break
    return text


MEND_REACH = 4
MARKS = "ัิีึืฺุู็" + TONES + "์ํ๎"
NO_TONE_AFTER = TONES + "\u0e47\u0e4c"


def _tries(text):
    """[(the places to delete, whether it is a run of three or more of one letter)]: each such run cut to one, each
    slip on its own, and each two slips within MEND_REACH letters."""
    from pythainlp.tokenize import word_tokenize

    def carried(i):
        return i + 1 < len(text) and text[i + 1] in MARKS

    lone, at = set(), 0
    for w in word_tokenize(text, engine="newmm", keep_whitespace=True, custom_dict=_WORDS[1]):
        if len(w) == 1:
            lone.add(at)
        at += len(w)
    slips = [i for i, ch in enumerate(text) if i and THAI.match(ch) and (
        (ch in TONES and text[i - 1] in NO_TONE_AFTER) or (ch == text[i - 1] and carried(i) and i - 1 in lone) or
        (i > 1 and ch in MARKS and text[i - 1] in TONES and ch == text[i - 2]) or
        (ch in TONES and text[i - 1] in MARKS and i + 1 < len(text) and text[i + 1] == text[i - 1]))]
    out, i = [], 0
    while i < len(text):
        j = i
        while j + 1 < len(text) and text[j + 1] == text[i]:
            j += 1
        if j >= i + 2 and THAI.match(text[i]) and carried(j):
            out.append((tuple(range(i + 1, j + 1)), True))
        i = j + 1
    out += [((i,), False) for i in slips]
    out += [((i, j), False) for i in slips for j in slips if i < j <= i + MEND_REACH]
    return out


def _odd(text):
    """(how many bits of text are not words, how many words it is cut into), compared in that order: the bits
    odd_thai points at and the Thai letters standing alone, then the words. A letter written twice can still cut into
    words (สสส่วน into สส and ส่วน, การมมี into กา, รม and มี), so of two spellings with as many bits that are not
    words, the one in fewer words is the one meant."""
    from pythainlp.tokenize import word_tokenize
    odd = len(odd_thai(text))
    toks = [w for w in word_tokenize(text, engine="newmm", keep_whitespace=True, custom_dict=_WORDS[1]) if w.strip()]
    lone = sum(1 for w in toks if len(w) == 1 and "\u0e01" <= w <= "\u0e2e")
    return odd + lone, len(toks)


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
