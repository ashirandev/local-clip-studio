"""The pauses cut out of a clip: where nobody says a word for a while, the clip jumps to the next word.

tighten() takes the clip's pieces of the recording and what the hearing heard in them (a look's "said", recording
seconds) and gives the pieces back with every pause longer than `gap` taken out, `air` seconds kept at each side so
a cut never clips a word. A phrase that is only a filler (เอ่อ, อ่า, อืม, um) counts as a pause. The words, the
pieces' order and what is said are kept; only the time between words goes. A moment of the old clip is found in
the new one with project.to_clip and project.from_recording, so titles and hits can move with the words.
"""
import re

GAP = 0.7
AIR = 0.2
GAPS = (0.3, 5.0)
AIRS = (0.0, 1.0)
FILLERS = re.compile(r"^(?:\s|[.,!?…]|เอ่อ+|เออ+|อ่า+|อ้า+|อืม+|อือ+|อื|อึ+|ฮึ+|เฮ้อ|เอิ่ม+|อ่ะ|อะ+|um+|uh+|erm+|hmm+"
                     r"|mm+)+$", re.IGNORECASE)


def filler(text):
    """True when text says nothing but fillers (เอ่อ, อ่า, อืม, um, uh...), so the time it takes is a pause."""
    return bool(FILLERS.match(text or ""))


def _number(name, value, low, high, what):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value != value or not low <= value <= high:
        raise ValueError("%s %r is not %s: use %s to %s seconds." % (name, value, what, low, high))
    return float(value)


def tighten(pieces, said, gap=GAP, air=AIR):
    """(the pieces with their pauses cut, the cuts): pieces [[a, b]] of the recording in clip order, said the phrases
    heard in the recording [{"from", "to", "text"}] (seconds of the recording). A pause is time inside a piece that
    no phrase with a word in it covers: at the piece's start, between two phrases, or at its end. One longer than
    gap is cut down to air seconds at each side (at the start or end of a piece, air on the side of the words); a
    piece with no word in it at all is left out. The cuts are [[a, b]] of the recording, in the order of the clip.
    ValueError for numbers out of range, pieces that are not pieces, and when nothing was heard."""
    gap = _number("gap", gap, GAPS[0], GAPS[1], "a pause length")
    air = _number("air", air, AIRS[0], AIRS[1], "the air left around words")
    if 2 * air >= gap:
        raise ValueError("air %s is too much for gap %s: keep 2 x air under gap, such as air %s for gap %s."
                         % (air, gap, AIR, GAP))
    if (not isinstance(pieces, (list, tuple)) or not pieces
            or not all(isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float))
                                                                            and not isinstance(v, bool) for v in p)
                       and 0 <= p[0] < p[1] for p in pieces)):
        raise ValueError("pieces %r are not pieces: give [[a, b], ...] in seconds of the recording, each a before b."
                         % (pieces,))
    if said is None:
        raise ValueError("Nobody has heard this moment yet, so the pauses are not known: hear it first.")
    words = sorted((float(p["from"]), float(p["to"])) for p in said
                   if isinstance(p, dict) and not filler(p.get("text")) and p.get("to", 0) > p.get("from", 0))
    kept, cuts = [], []
    for a, b in pieces:
        spoken = [(max(a, x), min(b, y)) for x, y in words if x < b and a < y]
        if not spoken:
            cuts.append([a, b])
            continue
        merged = [list(spoken[0])]
        for x, y in spoken[1:]:
            if x - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], y)
            else:
                merged.append([x, y])
        start = a if merged[0][0] - a <= gap else merged[0][0] - air
        end = b if b - merged[-1][1] <= gap else merged[-1][1] + air
        if start > a:
            cuts.append([a, start])
        for (_, y), (x, _) in zip(merged, merged[1:]):
            cuts.append([y + air, x - air])
        bounds = [start] + [v for (_, y), (x, _) in zip(merged, merged[1:]) for v in (y + air, x - air)] + [end]
        kept += [[round(bounds[k], 3), round(bounds[k + 1], 3)] for k in range(0, len(bounds), 2)]
        if end < b:
            cuts.append([end, b])
    if not kept:
        raise ValueError("Nobody says a word in these pieces, so cutting the pauses would leave nothing: choose a "
                         "moment where someone speaks.")
    return kept, [[round(x, 3), round(y, 3)] for x, y in cuts]
