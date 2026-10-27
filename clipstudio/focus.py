"""Following the action on a screen: where a screen recording changes, so the card shows that part at a size a phone
can read, and moves with it. Shown whole on a 960-pixel card, a 1920-pixel screen's words are half their size.

The card's part of the picture is looked at FPS times a second on small grey pictures. A pixel changed when its grey
moved more than LEVEL; the changes are summed over each STEP seconds into a heat map. For each step, the window of the
card's shape that holds the most change is found (an integral image, so every place is tried). One zoom serves the
whole clip, so the words keep one size. ZOOMS are the widths of Tarm's 15 hand crops of five tutorials (one row of
a list, one pane, nearly the whole screen: 0.34, 0.55 and 0.76 of it, measured by lane 5 on 10-27; 0.5 keeps a
1920-pixel screen at its own size on the card). The smallest is used whose windows hold at least ENOUGH of the
change, and CONCENTRATED times the share of the picture they cover (a window over half the picture holds half of a
change that is everywhere). When none does, the change is everywhere (a page that scrolls, a game) and the card shows
the whole picture. Each window is centred on the change it holds, so the action is not at its edge.

The window starts where the first change is. A step with less than QUIET of the picture changing keeps the window.
A window moves only when another place holds MOVE_GAIN times its change for PERSIST busy steps in a row, the
picture there stays changed (STAYED of it differs from before), and it has held HOLD seconds, so the card neither
chases the mouse nor jumps to a flicker; it moves from the first of those steps, since the clip is planned whole.
Between two places the card pans for PAN seconds. The pictures are read once and kept in work/focus.

When the lines written on the screen are known (read by screen.py), the zoom comes from them instead: the window is
as wide as lets the screen's usual line reach PHONE_PX on a phone (Tarm, 10-27: "ขนาดตัวหนังสือ น้อยที่สุดต้องเท่า
ในมือถือ 16px"), no narrower than SMALLEST of the picture. A phone 393 px wide shows 978 of the clip's 1080 px (Tarm's
iPhone cuts 51 a side), and a line of text is about LINE times its font from its tallest letter to its lowest, so a
1920-pixel screen whose lines are 18 px tall shows about 360 px of its width on a 960-pixel card. And when nothing
changes, the card holds where the most lines are (the highest such place, within TOP_NEAR of the most), its left
edge at their start, so they read from the first word:
showing the whole screen because nothing moved put 10 px words on Tarm's phone with a blank middle between them.
The card still follows a change only where the change is one part of the screen, as above: followed at the readable
size without that test, it chased a blinking button in the browser's corner (10-27).
"""
import statistics
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np

from . import recording

VERSION = 2
FPS = 2
SMALL_W = 384
LEVEL = 20
STEP = 1.0
ZOOMS = (0.34, 0.5, 0.75)
ENOUGH = 0.6
QUIET = 0.002
HOLD = 2.5
MOVE_GAIN = 1.5
PERSIST = 2
STAYED = 0.01
CONCENTRATED = 1.5
PAN = 0.6
PHONE_PX = 16
PHONE_SHOWS = 978 / 393.0
LINE = 1.2
SMALLEST = 0.15
TOP_NEAR = 0.95
SURE = 0.9


class FocusError(Exception):
    """The screen could not be looked at. The sentence says why."""


def _key(rec, region, a, b):
    return hashlib.sha1(("%d|%s|%d|%s|%s|%.2f|%.2f" % (VERSION, rec["path"], rec["size"], rec["mtime"], region, a, b))
                        .encode("utf-8")).hexdigest()[:16]


def heat(rec, region, a, b):
    """(heat maps, one per STEP seconds of a to b, each sw x sh: how many times each small pixel of region changed;
    marks, the small picture at the start and at the end of each step; the small size (sw, sh)). Kept in
    work/focus."""
    x, y, w, h = [int(round(v)) for v in region]
    sw, sh = SMALL_W, max(2, int(round(SMALL_W * h / float(w))) // 2 * 2)
    target = os.path.join(recording.WORK, "focus", _key(rec, [x, y, w, h], a, b) + ".npz")
    if os.path.isfile(target):
        with np.load(target) as kept:
            return kept["heat"], kept["marks"], (sw, sh)
    tool = shutil.which("ffmpeg")
    if not tool:
        raise FocusError("FFmpeg is not installed, so the screen cannot be looked at.")
    proc = subprocess.Popen([tool, "-v", "error", "-ss", "%.2f" % a, "-t", "%.2f" % (b - a), "-i", rec["path"],
                             "-map", "0:v:0", "-vf", "fps=%d,crop=%d:%d:%d:%d,scale=%d:%d,format=gray"
                             % (FPS, w, h, x, y, sw, sh), "-f", "rawvideo", "pipe:1"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    steps = max(1, int(np.ceil((b - a) / STEP)))
    out = np.zeros((steps, sh, sw), np.float32)
    marks = np.zeros((steps + 1, sh, sw), np.uint8)
    last, k, seen = None, 0, set()
    try:
        while True:
            data = proc.stdout.read(sw * sh)
            if len(data) < sw * sh:
                break
            now = np.frombuffer(data, np.uint8).reshape(sh, sw)
            step = min(steps - 1, int(k / float(FPS) / STEP))
            if last is None:
                marks[0] = now
            else:
                out[step] += np.abs(now.astype(np.int16) - last.astype(np.int16)) > LEVEL
            marks[step + 1] = now
            seen.add(step + 1)
            last, k = now, k + 1
    finally:
        proc.stdout.close()
        proc.wait()
    if last is None:
        raise FocusError("No picture could be read from %s at %s." % (rec["name"], recording.clock(a)))
    for s in range(1, steps + 1):
        if s not in seen:
            marks[s] = marks[s - 1]
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target + ".part", "wb") as f:
        np.savez(f, heat=out, marks=marks)
    os.replace(target + ".part", target)
    return out, marks, (sw, sh)


def _best(step, ww, wh, banned=None, near=None):
    """(change held, x, y): the ww x wh window of a heat map holding the most change, never one that overlaps a
    banned pixel. (-1, 0, 0) when every window does. With near, the highest of the windows holding near times the
    most or more: lines are read from the top."""
    s = np.zeros((step.shape[0] + 1, step.shape[1] + 1), np.float64)
    s[1:, 1:] = step.cumsum(0).cumsum(1)
    held = s[wh:, ww:] - s[:-wh, ww:] - s[wh:, :-ww] + s[:-wh, :-ww]
    if banned is not None:
        n = np.zeros(s.shape, np.float64)
        n[1:, 1:] = banned.cumsum(0).cumsum(1)
        held[(n[wh:, ww:] - n[:-wh, ww:] - n[wh:, :-ww] + n[:-wh, :-ww]) > 0] = -1
    if near and held.max() > 0:
        ys, xs = np.nonzero(held >= near * held.max())
        k = int(np.lexsort((xs, ys))[0])
        return float(held[ys[k], xs[k]]), int(xs[k]), int(ys[k])
    y, x = np.unravel_index(int(held.argmax()), held.shape)
    return float(held[y, x]), int(x), int(y)


def _held(step, x, y, ww, wh):
    return float(step[y:y + wh, x:x + ww].sum())


def _centred(step, x, y, ww, wh, banned):
    """The window moved so the change it holds is at its middle, kept inside the map and off the banned pixels."""
    held = step[y:y + wh, x:x + ww]
    if held.sum() <= 0:
        return x, y
    cy, cx = [float((held.sum(axis=1 - k) * np.arange(held.shape[k])).sum() / held.sum()) for k in (0, 1)]
    nx = int(min(max(0, round(x + cx - ww / 2.0)), step.shape[1] - ww))
    ny = int(min(max(0, round(y + cy - wh / 2.0)), step.shape[0] - wh))
    if banned is not None and banned[ny:ny + wh, nx:nx + ww].sum() > 0:
        return x, y
    return nx, ny


def _stayed(before, after, x, y, ww, wh):
    """True when the window's picture is not what it was before the change began: a form filled in, a result shown.
    A flicker, such as a tooltip or a notification that came and went, leaves it as it was."""
    window = (slice(y, y + wh), slice(x, x + ww))
    changed = np.abs(after[window].astype(np.int16) - before[window].astype(np.int16)) > LEVEL
    return float(changed.mean()) >= STAYED


def _far(a, b, ww, wh):
    return abs(a[0] - b[0]) > ww / 4.0 or abs(a[1] - b[1]) > wh / 4.0


def letters(text, avoid=None):
    """The height in pixels of the recording of the screen's usual line: the median of the lines read surely (text:
    [{"box": [x, y, w, h], "sure"}]), the face camera's (avoid) left out. None when no line was read."""
    heights = [line["box"][3] for line in text or [] if line.get("sure", 0) >= SURE and line["box"][3] > 0
               and not (avoid and _covers(avoid, line["box"]))]
    return float(statistics.median(heights)) if heights else None


def on_phone(height, window, shown):
    """The font size in pixels on a phone of letters height pixels tall in a window that many pixels wide of the
    recording, drawn shown pixels wide on the clip."""
    return height * shown / float(window) / PHONE_SHOWS / LINE


def _covers(box, other):
    return (box[0] <= other[0] and box[1] <= other[1] and other[0] + other[2] <= box[0] + box[2]
            and other[1] + other[3] <= box[1] + box[3])


def _text_map(text, region, scale, sw, sh):
    """A map of the small picture: how many readings put a line on each pixel."""
    x0, y0 = region[:2]
    out = np.zeros((sh, sw), np.float64)
    for line in text:
        if line.get("sure", 0) < SURE:
            continue
        x, y, w, h = line["box"]
        out[max(0, int((y - y0) / scale)):int(np.ceil((y + h - y0) / scale)),
            max(0, int((x - x0) / scale)):int(np.ceil((x + w - x0) / scale))] += 1
    return out


def plan(rec, pieces, region, aspect, avoid=None, text=None, shown=None):
    """{"zoom", "size": [w, h] in pixels of the recording, "shots": [{"from", "to": clip seconds, "x", "y": the
    window's top-left in pixels of the recording}], "letters": the font size on a phone, or None}, for a card of
    aspect (width over height) showing a window of region over the clip's pieces, never over avoid (the face camera's
    box, grown). text is the lines read on the screen in the clip ([{"box", "sure"}], pixels of the recording) and
    shown the card's width on the clip: with them the zoom makes the letters readable on a phone, and the card holds on
    the lines when no one part of it changes (nothing does, or a little everywhere: a blinking icon, a page that
    scrolls). None when the card should not follow: no one part changes and the screen's lines are not known."""
    x0, y0, w, h = region
    maps, at_ = [], 0.0
    for a, b in pieces:
        found, marks, (sw, sh) = heat(rec, region, a, b)
        maps += [(at_ + k * STEP, m, marks[k], marks[k + 1]) for k, m in enumerate(found)]
        at_ += b - a
    length, scale = at_, w / float(sw)
    banned = None
    if avoid:
        banned = np.zeros((sh, sw), np.float64)
        ax, ay = int((avoid[0] - x0) / scale), int((avoid[1] - y0) / scale)
        bx, by = int(np.ceil((avoid[0] + avoid[2] - x0) / scale)), int(np.ceil((avoid[1] + avoid[3] - y0) / scale))
        banned[max(0, ay):max(0, by), max(0, ax):max(0, bx)] = 1
        maps = [(t, m * (1 - banned), before, after) for t, m, before, after in maps]
    busy = [(t, m) for t, m, _, _ in maps if m.sum() >= QUIET * sw * sh]
    height = letters(text, avoid) if shown else None
    zoom = None
    if busy:
        total = sum(float(m.sum()) for _, m in busy)
        for z in ZOOMS:
            ww = max(1, int(round(sw * z)))
            wh = max(1, int(round(ww / aspect)))
            if wh > sh:
                wh, ww = sh, max(1, int(round(sh * aspect)))
            held = sum(max(0.0, _best(m, ww, wh, banned)[0]) for _, m in busy)
            if held >= max(ENOUGH, CONCENTRATED * ww * wh / float(sw * sh)) * total:
                zoom = z
                break
    if zoom is None and not height:
        return None
    if height:
        ww = int(round(min(sw, max(SMALLEST * sw, readable(height, shown) / scale))))
        wh = max(1, int(round(ww / aspect)))
        if wh > sh:
            wh, ww = sh, max(1, int(round(sh * aspect)))
    follow = zoom is not None
    if height:
        zoom = round(ww / float(sw), 3)
    size = [int(round(ww * scale)) // 2 * 2, int(round(wh * scale)) // 2 * 2]
    if not follow:
        lines = _text_map(text, region, scale, sw, sh)
        if banned is not None:
            lines = lines * (1 - banned)
        held, x, y = _best(lines, ww, wh, banned, near=TOP_NEAR)
        if held <= 0:
            return None
        starts = [int((line["box"][0] - x0) / scale) for line in text if line.get("sure", 0) >= SURE
                  and x <= (line["box"][0] - x0) / scale < x + ww and y <= (line["box"][1] - y0) / scale < y + wh]
        nx = max(0, min(starts or [x]) - 2)
        if banned is None or banned[y:y + wh, nx:nx + ww].sum() == 0:
            x = min(nx, sw - ww)
        return {"zoom": zoom, "size": size, "letters": round(on_phone(height, size[0], shown), 1), "on": "lines",
                "shots": [{"from": 0.0, "to": round(length, 2),
                           "x": min(int(round(x0 + x * scale)), int(x0 + w - size[0])) // 2 * 2,
                           "y": min(int(round(y0 + y * scale)), int(y0 + h - size[1])) // 2 * 2}]}
    shots, place, since, streak = [], None, 0.0, None
    for t, m, before, after in maps:
        if m.sum() < QUIET * sw * sh:
            continue
        held, x, y = _best(m, ww, wh, banned)
        if held < 0:
            continue
        x, y = _centred(m, x, y, ww, wh, banned)
        if place is None:
            place, since = (x, y), 0.0
            shots.append({"from": 0.0, "x": x, "y": y})
            continue
        if not _far(place, (x, y), ww, wh) or held < MOVE_GAIN * _held(m, place[0], place[1], ww, wh):
            streak = None
            continue
        if streak is None or _far(streak["at"], (x, y), ww, wh):
            streak = {"from": t, "steps": 0, "before": before}
        streak["steps"], streak["at"] = streak["steps"] + 1, (x, y)
        if streak["steps"] >= PERSIST and _stayed(streak["before"], after, x, y, ww, wh):
            move = max(streak["from"], since + HOLD)
            place, since = (x, y), move
            shots.append({"from": round(move, 2), "x": x, "y": y})
            streak = None
    if not shots:
        return None
    for k, s in enumerate(shots):
        s["to"] = shots[k + 1]["from"] if k + 1 < len(shots) else round(length, 2)
        s["x"] = min(int(round(x0 + s["x"] * scale)), int(x0 + w - size[0])) // 2 * 2
        s["y"] = min(int(round(y0 + s["y"] * scale)), int(y0 + h - size[1])) // 2 * 2
    return {"zoom": zoom, "size": size, "letters": round(on_phone(height, size[0], shown), 1) if height else None,
            "shots": [{"from": s["from"], "to": s["to"], "x": s["x"], "y": s["y"]} for s in shots]}


def readable(height, shown):
    """The widest window, in pixels of the recording, in which letters height pixels tall reach PHONE_PX on a phone
    when the window is drawn shown pixels wide on the clip."""
    return shown * height / (PHONE_PX * LINE * PHONE_SHOWS)


def at(shots, t):
    """(x, y) of the window at clip second t: held through each shot, panning for PAN seconds into the next."""
    x, y = shots[0]["x"], shots[0]["y"]
    for before, s in zip(shots, shots[1:]):
        if t < s["from"]:
            break
        u = min(1.0, (t - s["from"]) / PAN)
        e = u * u * (3 - 2 * u)
        x, y = before["x"] + (s["x"] - before["x"]) * e, before["y"] + (s["y"] - before["y"]) * e
    return x, y


def expression(shots, axis):
    """The window's x or y as an FFmpeg expression of t, the same path at() gives."""
    text = "%d" % shots[-1][axis]
    for before, s in reversed(list(zip(shots, shots[1:]))):
        u = "((t-%.3f)/%.3f)" % (s["from"], PAN)
        text = "if(lt(t,%.3f),%d,if(lt(t,%.3f),%d+(%d)*%s*%s*(3-2*%s),%s))" % (
            s["from"], before[axis], s["from"] + PAN, before[axis], s[axis] - before[axis], u, u, u, text)
    return text
