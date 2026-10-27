"""Lines said again: a speaker who stumbles stops, then says the line again, and the clip keeps the last take.

said_again() finds them in what the hearing heard (a look's "said", recording seconds). A new take starts after the
speaker stops for RESTART s or more; the take before it is the stretch just before that stop whose words are the new
take's opening words, letter for letter at ALIKE or more (the hearing spells a name two ways, so not all of them).
It is at most WITHIN s back, and has at least LEAST letters: a short word said twice is emphasis or a filler, not a
take. Of the stretches that alike, give or take NEAR, the shortest is the take: an earlier start only adds the
end of the line before it (measured on Tarm's retake: "นะ" from the line before scored 0.779, the take itself 0.773).
The cut runs from the earlier take's first word to just before the new take, so the stop goes with it.

Tarm, 10-27, on a clip that opened with his first try: "มันมีจุดที่ผมพูดซ้ำ มันก็เอาครั้งแรกมา ซึ่งผมพูดผิด แล้วเริ่มพูด
ประโยคเดิมครั้งที่สองเพื่ออัดต่อ". The speaker records the line again because the first one was wrong, so the last take is
the one meant.
"""
import difflib
import re

from . import pauses

RESTART = 1.0
WITHIN = 20.0
LEAST = 12
ALIKE = 0.6
NEAR = 0.05
AIR = 0.25
_NOT_LETTERS = re.compile(r"[\W_]+", re.UNICODE)


def _letters(text):
    return _NOT_LETTERS.sub("", (text or "").lower())


def said_again(said, a, b):
    """[{"cut": [from, to], "first": the earlier take's words, "again": the new take's opening words, "alike": share
    of letters the same}] for every line said again inside a to b (recording seconds), in time order."""
    lines = sorted((p for p in said or [] if p.get("to", 0) > p.get("from", 0) and not pauses.filler(p.get("text"))
                    and a <= p["from"] < b), key=lambda p: p["from"])
    found, taken = [], a
    for j in range(1, len(lines)):
        if lines[j]["from"] - lines[j - 1]["to"] < RESTART:
            continue
        again = "".join(_letters(p["text"]) for p in lines[j:])
        scored = []
        for i in range(j - 1, -1, -1):
            if lines[j]["from"] - lines[i]["from"] > WITHIN or lines[i]["from"] < taken:
                break
            first = "".join(_letters(p["text"]) for p in lines[i:j])
            if len(first) >= LEAST:
                scored.append((i, difflib.SequenceMatcher(None, first, again[:len(first)], autojunk=False).ratio()))
        top = max([s for _, s in scored] or [0])
        if top >= ALIKE:
            i, alike = next((i, s) for i, s in scored if s >= top - NEAR)
            found.append({"cut": [round(lines[i]["from"], 2), round(lines[j]["from"] - AIR, 2)],
                          "first": " ".join(p["text"] for p in lines[i:j]),
                          "again": " ".join(p["text"] for p in lines[j:j + (j - i)]), "alike": round(alike, 2)})
            taken = lines[j]["from"]
    return found


def without(pieces, cuts):
    """pieces [[a, b]] with the stretches in cuts [[a, b]] taken out, in the same order."""
    out = []
    for a, b in pieces:
        kept = [[a, b]]
        for x, y in sorted(cuts):
            kept = [part for p, q in kept for part in ([[p, min(q, x)]] if x > p else []) +
                    ([[max(p, y), q]] if y < q else [])
                    if part[1] - part[0] > 1e-6] if x < b and a < y else kept
        out += kept
    return [[round(x, 3), round(y, 3)] for x, y in out]
