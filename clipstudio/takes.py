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

Claps: Tarm, 10-28, on a folder of his review takes: "มันไม่รู้ว่าผมพูดซ้ำ แถมผมปรบมือเพื่อให้เกิดจังหวะในการคัท slate". He
gives a take up, claps, and starts again. claps() finds the crack of a clap in the sound, and at_claps() cuts the
take before it, the clap, and up to AFTER_CLAP s after it. Only on evidence from the words, because a key pressed on
a keyboard cracks as loud and as sharp (his Eaglet screen recording: 39 in 14.5 min, peaks of -20 to -39 dB against
his claps' -15 to -29). Keys come in runs and a clap comes alone, so a clap counts as said again only when no other
crack is heard within ALONE s of it and what follows says the take again at CLAP_ALIKE or more (his three retakes at
a clap: 0.35, 0.63, 0.71; at 34 stops with no clap, 12% reach 0.35). Of his 10 claps 7 are alone, of the 39 keys
5. Or the take is nothing but lines the hearing was not sure of (him talking to himself, off the script). A clap
with nothing said in the WITHIN s before it and words within SLATE_WORDS s after it is a slate before a take, and
only the clap goes.

The same listen gives the sound's level, one value every LEVEL_STEP s (heard_level), so a pause can be told from a
laugh: the hearing writes no words for laughter, and a stretch with no words is not always a quiet one.
"""
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess

import numpy as np

from . import pauses, recording

RESTART = 1.0
WITHIN = 20.0
LEAST = 12
ALIKE = 0.6
NEAR = 0.05
AIR = 0.25
_NOT_LETTERS = re.compile(r"[\W_]+", re.UNICODE)

RATE = 16000
QUIET_STEP, QUIET_LEAST, PAUSE_LEAST = 0.02, 0.06, 0.16
DIP_LEAST, DIP_BED, PIECE_LEAST = 0.12, 6.0, 0.3
FRAME = 0.01
CLAP_FLOOR = -40.0
CLAP_RISE = 20.0
CLAP_DROP = 12.0
CLAP_FALL = 0.12
CLAP_BRIGHT = 0.25
CLAP_TOGETHER = 0.6
INTO_LINE = 0.3
CLAP_TAIL = 0.25
AFTER_CLAP = 1.0
CLAP_ALIKE = 0.35
SLATE_WORDS = 3.0
ALONE = 2.0
CRACK_FLOOR = -50.0
CRACK_RISE = 10.0
CRACK_DROP = 8.0
CRACK_BRIGHT = 0.15
CLAPS_VERSION = 4
LEVEL_STEP = 0.1


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


def claps(samples, rate=RATE):
    """[[first, last, others]] seconds of each clap in samples (mono, -1 to 1, at rate), and how many other cracks are
    heard within ALONE s of it. A clap is a sound over CLAP_FLOOR dB that jumps CLAP_RISE dB over the 0.3 s before
    it, falls CLAP_DROP dB within CLAP_FALL s, and is bright, its change from one sample to the next carrying
    CLAP_BRIGHT of its energy or more (a clap cracks, a voice hums: Tarm's claps 0.26 to 1.43, his voice under 0.2).
    Claps less than CLAP_TOGETHER s apart are one: a double clap is one mark, and each one past two counts among the
    others (keys pressed in a run). A crack is anything sharp: CRACK_RISE dB over the 30 ms before it, CRACK_DROP dB
    down within CLAP_FALL s, CRACK_BRIGHT bright, over CRACK_FLOOR dB."""
    x = np.asarray(samples, dtype=np.float32)
    hop = int(round(rate * FRAME))
    n = len(x) // hop
    if n < 45:
        return []
    energy = (x[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-10
    change = np.diff(x[:n * hop], prepend=x[:1]).reshape(n, hop)
    bright = ((change ** 2).mean(axis=1) + 1e-10) / energy
    db = 10 * np.log10(energy)
    fall = int(round(CLAP_FALL / FRAME))
    found, i = [], 30
    while i < n - fall - 1:
        if db[i] > CLAP_FLOOR and db[i] - np.median(db[i - 30:i - 2]) >= CLAP_RISE:
            peak = i + int(np.argmax(db[i:i + 3]))
            if db[peak] - db[peak + fall] >= CLAP_DROP and bright[peak] >= CLAP_BRIGHT:
                t = round(peak * FRAME, 2)
                if found and t - found[-1][-1] < CLAP_TOGETHER:
                    found[-1].append(t)
                else:
                    found.append([t])
            i = peak + 25
        else:
            i += 1
    lead = np.concatenate([np.full(3, db[0]), db])
    sharp = ((db - (lead[:-3] + lead[1:-2] + lead[2:-1]) / 3 >= CRACK_RISE) & (bright >= CRACK_BRIGHT)
             & (db > CRACK_FLOOR) & (db - np.concatenate([db[fall:], np.full(fall, db[-1])]) >= CRACK_DROP))
    cracks = np.flatnonzero(sharp) * FRAME
    return [[g[0], g[-1], int(((cracks >= g[0] - ALONE) & (cracks < g[0] - 0.05)).sum()
                              + ((cracks > g[-1] + 0.3) & (cracks <= g[-1] + ALONE)).sum()) + max(0, len(g) - 2)]
            for g in found]


def heard_claps(rec, a=0.0, b=None):
    """claps() in rec from a to b (recording seconds; b None for its end), kept in WORK/claps so a stretch is listened
    to once. [] when it has no sound or ffmpeg cannot read it: then no clap is cut, as before claps were known."""
    return _heard(rec, a, b)["claps"]


def heard_level(rec, a=0.0, b=None):
    """(a, [the sound's level in dB, one value every LEVEL_STEP s from a]) in rec from a to b, from the same listen as
    heard_claps; None when it has no sound or ffmpeg cannot read it."""
    level = _heard(rec, a, b)["level"]
    return (a, level) if level else None


def quiet_near(rec, t, reach, under, bed=None, side=0, pause_only=False):
    """[from, to] in recording seconds: the pause of at least PAUSE_LEAST s nearest t, within reach s of it, where the
    sound stays under the level under (dB), read every QUIET_STEP s; with side -1 one that starts at or before t, with
    side 1 one that ends at or after it. With none, and bed, the same for bed dB under the middle level around t: over
    music or a crowd nothing is quiet, and the gap between two words is still a dip. With neither, the same for a
    stretch of at least QUIET_LEAST s, unless pause_only. None when there is none or no sound can be read. A tenth of a second averages away
    the dip between two words, so this listens closer than heard_level. A pause comes first because a shorter stretch
    may be the gap inside a word: "สามพัน" at gold EP1 3:54 has gaps of 60 and 120 ms before its พัน, and a cut in one
    ended the clip on สาม, 3,000 become 3; the pauses between phrases there run from 160 ms. Fast talk over a crowd has
    few of them (of 140 cuts, 69 had no pause near), so a shorter stretch is still better than loud sound."""
    a = max(0.0, t - reach)
    samples = _decoded(rec, a, min(rec["seconds"], t + reach))
    if samples is None:
        return None
    db = np.asarray(_levels(samples, step=QUIET_STEP))
    limits = [under] + ([float(np.median(db)) - bed] if bed and len(db) else [])
    for limit, least in [(limit, PAUSE_LEAST) for limit in limits] + (
            [] if pause_only else [(limit, QUIET_LEAST) for limit in limits]):
        low = np.r_[False, db < limit, False]
        turns = np.flatnonzero(np.diff(low.astype(np.int8)))
        runs = [[round(a + i * QUIET_STEP, 2), round(a + j * QUIET_STEP, 2)] for i, j in zip(turns[::2], turns[1::2])
                if j - i >= round(least / QUIET_STEP)]
        runs = [run for run in runs if side == 0 or (run[0] <= t if side < 0 else run[1] >= t)]
        if runs:
            return min(runs, key=lambda run: max(run[0] - t, t - run[1], 0.0))
    return None


def _levels(samples, rate=RATE, step=LEVEL_STEP):
    """The level of samples (mono, -1 to 1, at rate) in dB, one value every step s: the mean of its energy."""
    x = np.asarray(samples, dtype=np.float32)
    hop = int(round(rate * step))
    n = len(x) // hop
    return [round(float(v), 1) for v in 10 * np.log10((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-10)]


def dips(rec, a, b):
    """The middles of the dips in the sound from a to b in recording seconds, where one heard line can be cut into what
    was said in it: stretches of DIP_LEAST s or more, DIP_BED dB under the stretch's median level, at least PIECE_LEAST
    s from the last cut and from both ends, so no piece is too short to hear. Lane 4 10-30 on gold EP1 3:25: the line
    from 205.60 to 210.46 was one piece of sound, and the new thought began at the dip at 206.96."""
    samples = _decoded(rec, a, b)
    if samples is None:
        return []
    db = np.asarray(_levels(samples, step=QUIET_STEP))
    if not len(db):
        return []
    low = np.r_[False, db < float(np.median(db)) - DIP_BED, False]
    turns = np.flatnonzero(np.diff(low.astype(np.int8)))
    cuts, last = [], a
    for i, j in zip(turns[::2], turns[1::2]):
        at = round(a + (i + j) / 2 * QUIET_STEP, 2)
        if j - i >= round(DIP_LEAST / QUIET_STEP) and at - last >= PIECE_LEAST and b - at >= PIECE_LEAST:
            cuts.append(at)
            last = at
    return cuts


def _decoded(rec, a, b):
    """The samples of rec's used sound tracks from a to b in recording seconds, mono at RATE from -1 to 1, or None."""
    tool = shutil.which("ffmpeg")
    if not rec.get("sound_tracks") or b <= a or not tool:
        return None
    try:
        got = subprocess.run([tool, "-v", "error", "-ss", "%.2f" % a, "-t", "%.2f" % (b - a), "-i", rec["path"],
                              ] + recording.sound_args(rec) + ["-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
                             stdin=subprocess.DEVNULL, capture_output=True)
    except OSError:
        return None
    if got.returncode != 0 or not got.stdout:
        return None
    return np.frombuffer(got.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _heard(rec, a, b):
    """{"claps": claps() from a to b in recording seconds, "level": _levels()}, kept in WORK/claps."""
    b = rec["seconds"] if b is None else b
    nothing = {"claps": [], "level": []}
    if not rec.get("sound_tracks") or b <= a or not shutil.which("ffmpeg"):
        return nothing
    try:
        st = os.stat(rec["path"])
    except OSError:
        return nothing
    key = hashlib.sha1(("%d|%s|%d|%s|%.2f|%.2f" % (CLAPS_VERSION, rec["path"], st.st_size, st.st_mtime, a, b))
                       .encode("utf-8")).hexdigest()[:16]
    target = os.path.join(recording.WORK, "claps", key + ".json")
    try:
        with open(target, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        pass
    samples = _decoded(rec, a, b)
    if samples is None:
        return nothing
    found = {"claps": [[round(a + x, 2), round(a + y, 2), n] for x, y, n in claps(samples)], "level": _levels(samples)}
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target + ".part", "w", encoding="utf-8") as f:
        json.dump(found, f)
    os.replace(target + ".part", target)
    return found


def at_claps(said, claps_found, unsure=lambda p: False, slates=False):
    """[{"cut": [from, to], "clap": seconds, "first": the words cut, "why": "again", "unsure" or "slate"}] for every clap
    [first, last, others] (claps()) that is not inside a word, in time order, on the evidence the module's note gives. The
    take is the words just before the clap, back to a stop of RESTART s, at most WITHIN s; said again, the cut starts
    where the words most alike start (at the clap Tarm marked at 116 s, 0.35 from his first wrong word and 0.31 one
    word later, which would leave "อันนี้จะไม่ใช่" hanging before the new take); not sure,
    it also takes the lines the hearing was not sure of just before it. unsure(phrase) says which are not sure. Slates
    only when said holds a whole take from its first second (a phone clip): in a screen recording a key pressed in a
    pause looked like one 3 times in 10 min."""
    lines = sorted((p for p in said or [] if p.get("to", 0) > p.get("from", 0) and not pauses.filler(p.get("text"))),
                   key=lambda p: p["from"])
    found, bound = [], float("-inf")
    for t0, t1, others in sorted(claps_found):
        if any(p["from"] + INTO_LINE < t0 < p["to"] for p in lines):
            continue
        before = [p for p in lines if bound < p["from"] < t0 - INTO_LINE and t0 - p["from"] <= WITHIN]
        after = [p for p in lines if p["from"] >= t0 - INTO_LINE]
        bound = t1
        end = t1 + CLAP_TAIL
        if after and after[0]["from"] - AIR > end:
            end = min(after[0]["from"] - AIR, t1 + AFTER_CLAP)
        if not before:
            if slates and after and after[0]["from"] - t1 <= SLATE_WORDS:
                found.append({"cut": [round(t0 - 0.05, 2), round(end, 2)], "clap": t0, "first": "", "why": "slate"})
            continue
        k = len(before) - 1
        while k > 0 and before[k]["from"] - before[k - 1]["to"] < RESTART:
            k -= 1
        run = before[k:]
        again = "".join(_letters(p["text"]) for p in after)
        scored = []
        for i in range(len(run) - 1, -1, -1):
            first = "".join(_letters(p["text"]) for p in run[i:])
            if len(first) >= LEAST:
                scored.append((i, difflib.SequenceMatcher(None, first, again[:len(first)], autojunk=False).ratio()))
        top = max([s for _, s in scored] or [0])
        if top >= CLAP_ALIKE and not others:
            i = next(i for i, s in scored if s == top)
            cut, why = run[i:], "again"
        elif all(unsure(p) for p in run):
            while k > 0 and unsure(before[k - 1]):
                k -= 1
            cut, why = before[k:], "unsure"
        else:
            continue
        found.append({"cut": [round(cut[0]["from"], 2), round(end, 2)], "clap": t0,
                      "first": " ".join(p["text"] for p in cut), "why": why})
    return found
