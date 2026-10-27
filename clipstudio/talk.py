"""Two or more people talking (a podcast, an interview): the clip follows whoever speaks.

Cropping one person of a conversation hides the other's reaction, and showing the whole picture makes everyone small.
So the talk layout cuts between them the way an editor would: to whoever speaks, at the start of what they say; to
the one listening for a short reaction during a long answer; and to all of them together, as a card across the
middle, when they talk over each other or laugh.

Who speaks is measured, never guessed. A person is a face that sits at the same place in the moment's pictures; their
mouth area is the lower part of that face box. One FFmpeg pass reads every person's mouth area, small and grey, ten
times a second, and how much it changes from one picture to the next is how much that person moves, counted against
their own stillness (the median of the stretch), so a person lit brighter or sitting closer does not win by it.
While the hearing heard words, the person who moves clearly more than the others is the one speaking; when two move
almost alike, they are together on screen.

The shots are clip seconds, [{"from", "to", "show": a person's number, or 0 for everyone}]. The drawing is one crop of
the same size for every person whose x and y switch by time, so a cut costs nothing, and the everyone shot is a card
over the same moment blurred, laid over the crop only while it is on.
"""
import math
import shutil
import statistics
import subprocess

from . import faces, layouts

W, H = layouts.W, layouts.H
SAID = ("two or more people talking, such as a podcast or an interview: the clip cuts to whoever speaks, to the one "
        "listening for a short reaction, and to everyone when they laugh or talk together")
MOTION_FPS = 10
TILE = (48, 32)
WINDOW = 0.5
AHEAD = 1.3
TOGETHER = 0.75
MOVING = 2.0
SHORTEST_TURN = 1.2
SNAP = 0.6
LONG_TURN = 8.0
REACTION = 1.5
EDGE = 2.0
PERSON_SHARE = 0.3
MOST_PEOPLE = 4
SHARE = 4.0
FACE_AT = 0.45
WIDE = 16 / 9


def people(found):
    """The people of a stretch, left to right: [{"n", "box", "seen", "of"}], each a face at the same place in at
    least PERSON_SHARE of the pictures, a smaller face that overlaps a bigger one left out."""
    kept = []
    for g in faces.places(found):
        if g["of"] and g["seen"] >= PERSON_SHARE * g["of"] and not any(layouts._overlap(g["box"], k["box"])
                                                                       for k in kept):
            kept.append(g)
    kept = sorted(kept[:MOST_PEOPLE], key=lambda g: g["box"][0])
    return [dict(g, n=i) for i, g in enumerate(kept, 1)]


def mouth(box, size):
    """The mouth area of a face box (x, y, w, h): its lower half and a little under the chin, inside the picture."""
    x, y, w, h = box
    return [round(v) for v in layouts._inside([x + 0.2 * w, y + 0.55 * h, 0.6 * w, 0.55 * h], *size)]


def motion(rec, a, b, persons, fps=MOTION_FPS):
    """How much each person's mouth area moves from a to b (recording seconds): [[value per picture] per person],
    picture i at a + (i + 1) / fps."""
    import numpy
    tw, th = TILE
    n = len(persons)
    split = "[0:v]fps=%g,format=gray,split=%d%s;" % (fps, n, "".join("[m_s%d]" % i for i in range(n))) if n > 1 \
        else "[0:v]fps=%g,format=gray[m_s0];" % fps
    tiles = ";".join("[m_s%d]crop=%d:%d:%d:%d,scale=%d:%d:flags=area[m_t%d]" % (
        (i,) + tuple(mouth(p["box"], (rec["width"], rec["height"]))[2:]) +
        tuple(mouth(p["box"], (rec["width"], rec["height"]))[:2]) + (tw, th, i)) for i, p in enumerate(persons))
    joined = ";%shstack=inputs=%d[m_o]" % ("".join("[m_t%d]" % i for i in range(n)), n) if n > 1 else \
        ";[m_t0]null[m_o]"
    out = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-ss", "%.3f" % a, "-t", "%.3f" % (b - a),
                          "-i", rec["path"], "-filter_complex", split + tiles + joined, "-map", "[m_o]", "-f",
                          "rawvideo", "-pix_fmt", "gray", "pipe:1"], capture_output=True, check=True).stdout
    frames = numpy.frombuffer(out, dtype=numpy.uint8)
    frames = frames[:len(frames) // (tw * th * n) * tw * th * n].reshape(-1, th, tw * n).astype(numpy.int16)
    if len(frames) < 2:
        return [[] for _ in persons]
    change = numpy.abs(frames[1:] - frames[:-1])
    return [change[:, :, i * tw:(i + 1) * tw].mean(axis=(1, 2)).round(3).tolist() for i in range(n)]


def stillness(values):
    """Each person's resting movement: the median of their values, plus one."""
    return [statistics.median(v) + 1.0 for v in values]


def windows(values, fps, start, speech, still=None):
    """Who speaks in each WINDOW seconds from start: [{"from", "to", "who", "scores"}], who a person's number, 0 when
    they move almost alike (together), None when nobody is heard or nobody is clearly ahead. speech is [[from, to]]
    in the same seconds, or None when nobody listened, so movement alone decides. still is each person's resting
    movement, from these values when not given."""
    if not values or not values[0]:
        return []
    still = still or stillness(values)
    per = max(1, round(WINDOW * fps))
    out = []
    for w in range(math.ceil(len(values[0]) / per)):
        a, b = start + w * per / fps, start + min(len(values[0]), (w + 1) * per) / fps
        scores = [round(sum(v[w * per:(w + 1) * per]) / len(v[w * per:(w + 1) * per]) / s, 3)
                  for v, s in zip(values, still)]
        heard = speech is None or any(x < b and a < y for x, y in speech)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        who = None
        if heard and len(scores) == 1:
            who = 1
        elif heard:
            first, second = scores[order[0]], scores[order[1]]
            if second >= TOGETHER * first and second >= MOVING:
                who = 0
            elif first >= AHEAD * second:
                who = order[0] + 1
        out.append({"from": round(a, 3), "to": round(b, 3), "who": who, "scores": scores})
    return out


def turns(wins, starts=(), joins=()):
    """The turns of the talk, [[from, to, who]]: each window not decided keeps the turn before it (the first ones
    take the first decided), a turn shorter than SHORTEST_TURN is part of the one before, and a change of turn moves
    to where two pieces of the clip join, or else to a phrase start, within SNAP seconds of it, so a cut lands where
    the picture already cuts or where words begin."""
    if not wins:
        return []
    known = next((w["who"] for w in wins if w["who"] is not None), 1)
    runs = []
    for w in wins:
        known = known if w["who"] is None else w["who"]
        if runs and runs[-1][2] == known:
            runs[-1][1] = w["to"]
        else:
            runs.append([w["from"], w["to"], known])
    merged = []
    for run in runs:
        if merged and (run[1] - run[0] < SHORTEST_TURN or merged[-1][2] == run[2]):
            merged[-1][1] = run[1]
        else:
            merged.append(list(run))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < SHORTEST_TURN:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    for prev, run in zip(merged, merged[1:]):
        for marks in (joins, starts):
            near = [s for s in marks if abs(s - run[0]) <= SNAP and
                    prev[0] + SHORTEST_TURN / 2 < s < run[1] - SHORTEST_TURN / 2]
            if near:
                prev[1] = run[0] = round(min(near, key=lambda s: abs(s - run[0])), 3)
                break
    return merged


def shots(wins, turns_, people_count):
    """The shots from the turns, [{"from", "to", "show"}]: a person's turn shows them; a turn longer than LONG_TURN
    gets a REACTION-second cut to the one who moves most while listening, where they move at least MOVING times their
    stillness and not within EDGE seconds of the turn's ends, at most one in each LONG_TURN."""
    out = []
    for a, b, who in turns_:
        cuts = []
        if who and people_count > 1 and b - a > LONG_TURN:
            last = a
            for w in wins:
                if not (a + EDGE <= w["from"] and w["to"] <= b - EDGE) or w["from"] - last < LONG_TURN / 2:
                    continue
                others = [(s, i + 1) for i, s in enumerate(w["scores"]) if i + 1 != who]
                best = max(others)
                if best[0] >= MOVING and (not cuts or w["from"] - cuts[-1][0] >= LONG_TURN):
                    mid = (w["from"] + w["to"]) / 2
                    cuts.append((round(mid - REACTION / 2, 3), round(mid + REACTION / 2, 3), best[1]))
                    last = w["from"]
        at = a
        for c0, c1, other in cuts:
            out.append({"from": at, "to": c0, "show": who})
            out.append({"from": c0, "to": c1, "show": other})
            at = c1
        out.append({"from": at, "to": b, "show": who})
    return [s for s in out if s["to"] - s["from"] > 0.01]


def plan(rec, pieces, said, persons, read=None):
    """The shots of a clip made of pieces ([[recording a, recording b]]) of rec, with what the hearing heard (said:
    [{"from", "to"}] in recording seconds, or None when nobody listened) and the people found: clip seconds from 0 to
    the clip's end, and the windows they were decided from. A person's resting movement is measured over the whole
    clip: measured over each piece, a piece where one person speaks throughout makes their speaking their rest, so a
    listener who fidgets at a third of it scores within AHEAD of them and nobody is shown speaking. Clips cut at their
    silences have such pieces."""
    read = read or motion
    wins, starts, joins, at = [], [], [], 0.0
    read_ = [read(rec, a, b, persons) for a, b in pieces]
    whole = [sum((values[i] for values in read_ if values and values[0]), []) for i in range(len(persons))]
    still = stillness(whole) if whole and whole[0] else None
    for (a, b), values in zip(pieces, read_):
        speech = None if said is None else [[p["from"], p["to"]] for p in said if p["from"] < b and a < p["to"]]
        for w in windows(values, MOTION_FPS, a, speech, still):
            wins.append(dict(w, **{"from": round(w["from"] - a + at, 3), "to": round(w["to"] - a + at, 3)}))
        starts += [round(p["from"] - a + at, 3) for p in said or [] if a <= p["from"] < b]
        at += b - a
        joins.append(round(at, 3))
    if not wins:
        return [{"from": 0.0, "to": round(at, 3), "show": 1}], wins
    found = shots(wins, turns(wins, starts, joins[:-1]), len(persons))
    found[0]["from"], found[-1]["to"] = 0.0, round(at, 3)
    return found, wins


def make(size, persons, shots_):
    """The talk layout for a recording of size (w, h): a part for each person (the same size, filling the frame)
    and, when a shot shows everyone, a card of all of them across the middle; places for words clear of every head;
    the shots. ValueError when there are fewer than two people."""
    if len(persons) < 2:
        raise ValueError("The talk layout follows two or more people talking, and the program found %d sitting still "
                         "in the moment's pictures. Choose the face or the subject layout." % len(persons))
    w, h = size
    room = [0, 0, w, h]
    tallest = max(max(p["box"][2], p["box"][3]) for p in persons)
    bh = min(h, SHARE * tallest)
    bw = bh * W / H
    if bw > w:
        bw, bh = w, w * H / W
    parts, heads, notes = [], [], []
    for p in persons:
        fx, fy, fw, fh = p["box"]
        src = layouts._place(fx + fw / 2, fy + fh * 0.42 - bh * (FACE_AT - 0.5), bw, bh, room)
        parts.append({"name": "person %d" % p["n"], "src": src, "dst": [0, 0, W, H]})
        heads.append(layouts._head(p["box"], src))
    sizes = {tuple(part["src"][2:]) for part in parts}
    if len(sizes) > 1:
        side = min(s[0] for s in sizes), min(s[1] for s in sizes)
        for part in parts:
            x, y = part["src"][:2]
            part["src"] = layouts._place(x + part["src"][2] / 2, y + part["src"][3] / 2, side[0], side[1], room)
    if any(s["show"] == 0 for s in shots_):
        x0 = min(p["box"][0] - 0.8 * p["box"][3] for p in persons)
        x1 = max(p["box"][0] + p["box"][2] + 0.8 * p["box"][3] for p in persons)
        y0 = min(p["box"][1] - 0.6 * p["box"][3] for p in persons)
        y1 = max(p["box"][1] + 1.8 * p["box"][3] for p in persons)
        wide = layouts._fit(layouts._inside([x0, y0, x1 - x0, y1 - y0], w, h), WIDE, room)
        card_h = layouts._even(layouts.CARD_W / WIDE)
        parts.append({"name": "everyone", "src": wide, "dst": [layouts.CARD_X, layouts.CARD_Y, layouts.CARD_W,
                                                               card_h]})
    top = min(hd[1] for hd in heads)
    bottom = max(hd[1] + hd[3] for hd in heads)
    places, captions = layouts._around_the_head([0, top, W, bottom - top], notes)
    return {"kind": "talk", "size": [w, h], "face": None, "parts": parts, "places": places, "captions": captions,
            "background": False, "notes": notes, "people": [{"n": p["n"], "box": p["box"]} for p in persons],
            "shots": [dict(s) for s in shots_],
            "safe": {"box": layouts.SAFE, "buttons": layouts.BUTTONS, "said": layouts.SAFE_SAID}}


def with_shot(shots_, start, end, show):
    """The shots with show from start to end (clip seconds), cutting into the ones it overlaps; neighbours that show
    the same are joined."""
    out = []
    for s in shots_:
        if s["to"] <= start or s["from"] >= end:
            out.append(dict(s))
            continue
        if s["from"] < start:
            out.append(dict(s, to=start))
        if s["to"] > end:
            out.append(dict(s, **{"from": end}))
    out.append({"from": start, "to": end, "show": show})
    out.sort(key=lambda s: s["from"])
    joined = []
    for s in out:
        if joined and joined[-1]["show"] == s["show"] and abs(joined[-1]["to"] - s["from"]) < 1e-6:
            joined[-1]["to"] = s["to"]
        else:
            joined.append(s)
    return joined


def said(shots_):
    """The shots as a short line: "0-3.2 s person 1; 3.2-4.7 s everyone"."""
    return "; ".join("%g-%g s %s" % (round(s["from"], 1), round(s["to"], 1),
                                     "everyone" if s["show"] == 0 else "person %d" % s["show"]) for s in shots_)


def parts_at(layout, t):
    """The parts on screen at clip second t."""
    shown = next((s["show"] for s in layout["shots"] if s["from"] <= t < s["to"]), layout["shots"][-1]["show"])
    name = "everyone" if shown == 0 else "person %d" % shown
    return [p for p in layout["parts"] if p["name"] == name]


def _switch(shots_, values, last):
    """An FFmpeg expression of t that is values[i] during shot i."""
    text = "%d" % last
    for s, v in reversed(list(zip(shots_, values))[:-1]):
        text = "if(lt(t,%s),%d,%s)" % (_f(s["to"]), v, text)
    return text


def _f(v):
    return ("%.3f" % v).rstrip("0").rstrip(".") or "0"


def graph(layout, src, out, first_input, pushes=()):
    """(filter graph text from [src] to [out] drawing the talk layout at 1080x1920, extra files), the same shape as
    layouts.graph. Every switch is a function of t alone, so a preview frame at any second matches the export. A slow
    push raises ValueError: it would fight the cuts."""
    if pushes:
        raise ValueError("A talk clip cuts between the people, so it has no slow push: take the push off.")
    files = []

    def shape(path):
        files.append(path)
        return "[%d:v]" % (first_input + len(files) - 1)

    parts = {p["name"]: p for p in layout["parts"]}
    shots_ = layout["shots"]
    crops = [parts["person %d" % s["show"]]["src"] if s["show"] else None for s in shots_]
    held, filled = crops[0] or parts["person 1"]["src"], []
    for c in crops:
        held = c or held
        filled.append(held)
    cw, ch = filled[0][2], filled[0][3]
    one = "crop=w=%d:h=%d:x='%s':y='%s',scale=%d:%d,setsar=1" % (
        cw, ch, _switch(shots_, [c[0] for c in filled], filled[-1][0]),
        _switch(shots_, [c[1] for c in filled], filled[-1][1]), W, H)
    tail = "setsar=1,format=yuv420p[%s]" % out
    together = [s for s in shots_ if s["show"] == 0]
    if not together:
        return "[%s]%s,%s" % (src, one, tail), files
    card = parts["everyone"]
    cx, cy, cwd, chd = card["dst"]
    sx, sy, sw, sh = card["src"]
    sigma, drop, _ = layouts.SHADOW["card"]
    pad = 3 * sigma
    when = "+".join("between(t,%s,%s)" % (_f(s["from"]), _f(s["to"] - 0.001)) for s in together)
    text = ("[%s]split=3[tk_a][tk_b][tk_c];[tk_a]%s[tk_one];"
            "[tk_b]scale=540:960:force_original_aspect_ratio=increase,crop=540:960,gblur=sigma=9,"
            "eq=saturation=1.3:contrast=1.05:brightness=-0.04,scale=%d:%d[tk_bg0];"
            "[tk_bg0]%soverlay=0:0:shortest=1[tk_bg];[tk_bg]%soverlay=%d:%d:shortest=1[tk_1];"
            "[tk_c]crop=%d:%d:%d:%d,scale=%d:%d,format=yuva420p[tk_c0];%sformat=gray[tk_cm];"
            "[tk_c0][tk_cm]alphamerge[tk_card];[tk_1][tk_card]overlay=%d:%d:shortest=1[tk_all];"
            "[tk_one][tk_all]overlay=0:0:shortest=1:enable='%s',%s"
            % (src, one, W, H, shape(layouts.shade()), shape(layouts.shadow("card", cwd, chd, layouts.CARD_RADIUS)),
               cx - pad, cy - pad + drop, sw, sh, sx, sy, cwd, chd,
               shape(layouts.rounded(cwd, chd, layouts.CARD_RADIUS)), cx, cy, when, tail))
    return text, files
