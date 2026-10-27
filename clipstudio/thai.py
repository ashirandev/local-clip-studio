"""Thai text from the hearing step: the vowel marks it cannot write put back, and the numbers locked.

Omnilingual's letters have no SARA I, no SARA U and no MAI TRI: where the speech has one, the model writes "⁇"
(checked 10-27 on its tokenizer). Which mark a "⁇" stands for is read from the letters around it, by how often each
mark sits between the same letters in Thai text: thai_marks.json, counted from the Thai transcripts of Google's
FLEURS *train* set (CC BY 4.0), never from the test sentences the error rate is measured on. The longest context
that was seen decides; with none, SARA I, the most common of the three.

The model also writes a digit where a number word begins and leaves the rest of the word behind (seen 10-27 on the
review clip: "5้าสิบ" for ห้าสิบ, "เจ็ดสิบ5้า" for เจ็ดสิบห้า, "1ึ่ง.ิด5" for หนึ่งจุดห้า, "เ750"). A Thai mark
cannot follow a digit or a point, so such a digit stands for its word and takes back the rest of it; a digit written
against a number word joins it as a word; a leading vowel left before digits is dropped. Digits the model wrote on
their own (75, 1,990, 2.5) stay digits: they are the number as heard.

Numbers are locked before Muse fixes misheard words: each run of digits or Thai number words becomes <N1>, <N2>...,
so a number said can never be changed, only carried back exactly as it was heard. Locking a little too much (สาม in
สามารถ) is safe: those letters just stay as heard.

    python -m clipstudio.thai <FLEURS th train.tsv>     counts the marks again and writes thai_marks.json
"""
import json
import os
import re
import sys

UNHEARD = "⁇"
MARKS = ("ิ", "ุ", "๊")
CONTEXTS = ((2, 2), (2, 1), (1, 2), (1, 1), (1, 0), (0, 1))
TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thai_marks.json")
NUMBER_WORDS = ("ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า", "สิบ", "ยี่", "เอ็ด", "ร้อย",
                "พัน", "หมื่น", "แสน", "ล้าน", "ครึ่ง", "จุด")
_WORDS = "|".join(w for w in NUMBER_WORDS if w != "จุด")
NUMBER = re.compile(r"[0-9]+(?:[.,][0-9]+)*|(?:%s)+(?:จุด(?:%s)+)?" % (_WORDS, _WORDS))
LOCK = re.compile(r"<N(\d+)>")
SAID = dict(zip("0123456789.", ("ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า", "จุด")))
THAI_MARK = re.compile("[ัิ-ฺ็-๎]")
HALF_WORD = re.compile("[0-9.](?=[ัิ-ฺ็-๎])")
LEFT_VOWEL = re.compile("[เแโใไ](?=[0-9])")
_table = []


def repair(text):
    """text with every "⁇" made SARA I, SARA U or MAI TRI from the letters around it, NIKHAHIT + SARA AA written
    as SARA AM, and number words the model began with a digit written as words."""
    text = text.replace("ํา", "ำ")
    if UNHEARD in text:
        counts = _counts()
        letters = list(text)
        for k, ch in enumerate(letters):
            if ch == UNHEARD:
                letters[k] = _pick(counts, text[:k], text[k + 1:])
        text = "".join(letters)
    return _numbers_as_said(text)


def _numbers_as_said(text):
    text = LEFT_VOWEL.sub("", text)
    while True:
        m = HALF_WORD.search(text)
        if not m:
            break
        word, rest = SAID[m.group(0)], text[m.end():]
        tail = next((word[k:] for k in range(1, len(word)) if rest.startswith(word[k:])), None)
        if tail is None:
            tail = re.match("[ัิ-ฺ็-๎]+", rest).group(0)
            if rest[len(tail):len(tail) + 1] == word[-1]:
                tail += word[-1]
        text = text[:m.start()] + word + rest[len(tail):]
    changed = True
    while changed:
        changed = False
        for m in re.finditer("[0-9]", text):
            before, after = text[:m.start()], text[m.end():]
            if before.endswith(NUMBER_WORDS) or after.startswith(NUMBER_WORDS):
                text, changed = before + SAID[m.group(0)] + after, True
                break
    return text


def _pick(counts, before, after):
    for left, right in CONTEXTS:
        seen = counts.get(_key(before[max(0, len(before) - left):] if left else "", after[:right]))
        if seen:
            return MARKS[max(range(len(MARKS)), key=lambda i: seen[i])]
    return MARKS[0]


def _key(left, right):
    return "%d,%d|%s|%s" % (len(left), len(right), left, right)


def _counts():
    if not _table:
        with open(TABLE, encoding="utf-8") as f:
            _table.append(json.load(f))
    return _table[0]


def lock(text, first=1):
    """(text with every number as <N1>, <N2>..., {"N1": the number as heard, ...}). first is the number the names
    start from, so lines asked together never share a name."""
    numbers = {}

    def hold(m):
        name = "N%d" % (first + len(numbers))
        numbers[name] = m.group(0)
        return "<%s>" % name

    return NUMBER.sub(hold, text), numbers


def unlock(text, numbers):
    """text with each <Nk> put back as heard, or None when a number is missing, repeated or made up."""
    found = LOCK.findall(text)
    if sorted(found, key=int) != sorted((name[1:] for name in numbers), key=int) or len(set(found)) != len(found):
        return None
    return LOCK.sub(lambda m: numbers["N" + m.group(1)], text)


def count(train_tsv):
    """{context: [times SARA I, SARA U, MAI TRI sit there]} from the transcripts of a FLEURS tsv file."""
    out = {}
    with open(train_tsv, encoding="utf-8") as f:
        for row in f:
            cols = row.rstrip("\n").split("\t")
            text = cols[2].replace(" ", "") if len(cols) > 2 else ""
            for k, ch in enumerate(text):
                if ch in MARKS:
                    for left, right in CONTEXTS:
                        key = _key(text[max(0, k - left):k] if left else "", text[k + 1:k + 1 + right])
                        out.setdefault(key, [0] * len(MARKS))[MARKS.index(ch)] += 1
    return out


if __name__ == "__main__":
    table = count(sys.argv[1])
    with open(TABLE + ".part", "w", encoding="utf-8", newline="\n") as f:
        json.dump(table, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    os.replace(TABLE + ".part", TABLE)
    print("%d contexts written to %s" % (len(table), TABLE))
