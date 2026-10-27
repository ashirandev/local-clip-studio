"""Numbers as they are said, so a number on screen can be matched to one that was said.

The hearing writes a number the way the speaker said it: as digits, or as words ("75", "seventy five",
"เจ็ดสิบห้า"). Digits are read in any script (๗๕, ٧٥ and ७५ are 75). Number words are read in Thai (thai.value) and
in English. In another language a number said only in words cannot be matched here, so a check built on this says
it could not check, never that the number was not said (measured 10-27: before this, "seventy five square metres"
and 七十五 made "75" on screen fail as not said).
"""
import re

from . import thai

DIGITS = re.compile(r"\d+(?:[.,]\d+)*")
ONES = {w: n for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                   "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * n for n, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
WORDS = set(ONES) | set(TENS) | set(SCALES) | {"point", "and", "a"}
READ = {"thai": ("thai", "th", "ไทย"), "english": ("english", "en", "eng")}


def digits(text):
    """Every number written in digits in text, in any script, as numbers (1,990 is 1990)."""
    out = set()
    for m in DIGITS.finditer(text or ""):
        try:
            out.add(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def english(text):
    """Every number said in English words in text (seventy-five, one hundred and twenty, one point five). Words
    that cannot follow each other in one number start the next one: "one two" is 1 and 2, "one and two" too."""
    words = re.findall(r"[a-z]+", (text or "").lower().replace("-", " "))
    out, run = set(), []
    for i, w in enumerate(words + [""]):
        follows = words[i + 1] if i + 1 < len(words) else ""
        if run and not _follows(run, w):
            _end(run, out)
            run = []
        if not run and (w in ONES or w in TENS or w == "a" and follows in SCALES):
            run = [w]
        elif run:
            run.append(w)
    return out


def _follows(run, w):
    """True when w can go on the number the run of words has so far."""
    prev = run[-1]
    if "point" in run:
        return w in ONES and ONES[w] < 10
    if w in ONES:
        return prev in TENS and 0 < ONES[w] < 10 or prev in SCALES or prev == "and"
    if w in TENS:
        return prev in SCALES or prev == "and"
    if w in SCALES:
        return prev in ONES or prev in TENS or prev == "a" or prev in SCALES and SCALES[prev] < SCALES[w]
    if w == "and":
        return prev in SCALES
    if w == "point":
        return prev in ONES or prev in TENS or prev in SCALES
    return False


def _end(run, out):
    while run and run[-1] in ("and", "point", "a"):
        run.pop()
    if run:
        out.add(_english_run(run))


def _english_run(run):
    total = current = 0
    whole, _, decimals = " ".join(run).partition(" point ")
    for w in whole.split():
        if w in ONES:
            current += ONES[w]
        elif w in TENS:
            current += TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w in SCALES:
            total += (current or 1) * SCALES[w]
            current = 0
        elif w == "a":
            current = 1
    places = [ONES[w] for w in decimals.split() if w in ONES and ONES[w] < 10]
    return float(total + current + sum(d / 10 ** (k + 1) for k, d in enumerate(places)))


def values(text):
    """Every number said in text: digits in any script, and Thai and English number words."""
    return digits(text) | thai.values(text or "") | english(text)


def reads(language, said):
    """True when the number words of the language the clip is said in are read here: the language the recipe was
    told, else the script of what was said (Thai letters, or Latin read as English)."""
    named = " ".join((language or "").lower().split())
    if named:
        return any(named.startswith(names) or named in names for names in READ.values())
    joined = " ".join(said)
    if re.search("[ก-๛]", joined):
        return True
    return not re.search(r"[^\W\da-zA-Z_À-ɏ]", joined)
