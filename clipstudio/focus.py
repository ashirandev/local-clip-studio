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
When the lines carry the second they were read at, the card leaves a place once a reading finds in it under FEW of
what the busiest place holds then, for where the most lines are then (no sooner than HOLD after its last move): held
on the most lines of the whole clip, a model dialog, the card showed the blank page under it for 27 of 40 s when the
dialog closed (Eaglet, 10-28 round 8); and with "no line left" as the test, it held 7 s on two small chips left on an
empty page (round 9). With FEW, that clip's card moves three more times and is near blank 2 s; the three other
clips of that run, which never went blank, do not move more.
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
FEW = 0.25
# A screen whose lines are read surely under READ_SHARE of the time is written in a script the reader does not know
# (Thai on RapidOCR: 4 of 17 and 8 of 29 lines on the gold lesson's slides, 774 of 911 and 834 of 1154 on Eaglet's
# own screen, 10-30), so its lines say nothing of the letters' size: from them the card zoomed to 874 of 1920 px and
# cut the slide's lines in half, and Muse turned following off. Such a screen, when it is a slide, shows the part
# that is written on instead: what is not the colour of its border (the picture's edge, FLAT_EDGE of it one colour),
# the marks grown by SPREAD of the width so a line's letters join, a mark under SPECK of the picture left out (the
# mouse's ring, a logo in a corner), whole when that part reaches WHOLE of the picture both ways.
READ_SHARE = 0.5
FLAT_EDGE = 0.9
SPREAD = 0.02
SPECK = 0.01
WHOLE = 0.95


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


def unread(text):
    """True when the screen's lines (text: [{"sure"}]) were read surely under READ_SHARE of the time: a script the
    reader does not know. False when no line was found."""
    lines = list(text or [])
    return bool(lines) and sum(1 for line in lines if line.get("sure", 0) >= SURE) < READ_SHARE * len(lines)


def content(rec, pieces):
    """[x, y, w, h] in pixels of the recording: the part of a slide that is written on, over the clip's pieces (the
    pictures heat() keeps), or None when a picture's edge is not one colour (a screen, a camera) or the written part
    reaches WHOLE of the picture both ways."""
    import cv2
    w, h = rec["width"], rec["height"]
    found = []
    for a, b in pieces:
        _, marks, (sw, sh) = heat(rec, [0, 0, w, h], a, b)
        grow = np.ones((max(1, int(round(SPREAD * sw))),) * 2, np.uint8)
        for pic in marks:
            pic = pic.astype(np.int16)
            edge = np.concatenate([pic[0], pic[-1], pic[:, 0], pic[:, -1]])
            ground = np.median(edge)
            if np.mean(np.abs(edge - ground) <= LEVEL) < FLAT_EDGE:
                return None
            ink = cv2.dilate((np.abs(pic - ground) > LEVEL).astype(np.uint8), grow)
            _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
            found += [(x, y, x + bw, y + bh) for x, y, bw, bh, area in stats[1:] if area >= SPECK * sw * sh]
    if not found:
        return None
    k = w / float(sw)
    x0, y0 = min(f[0] for f in found) * k, min(f[1] for f in found) * k
    x1, y1 = max(f[2] for f in found) * k, max(f[3] for f in found) * k
    if x1 - x0 >= WHOLE * w and y1 - y0 >= WHOLE * h:
        return None
    return [round(x0), round(y0), round(x1 - x0), round(y1 - y0)]


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
        start = _on_lines(text, region, scale, (sw, sh), ww, wh, banned)
        if start is None:
            return None
        shots = _kept_on_lines(text, pieces, region, scale, (sw, sh), ww, wh, banned, start, length)
    else:
        shots = _followed(maps, sw, sh, ww, wh, banned)
    if not shots:
        return None
    for k, s in enumerate(shots):
        s["to"] = shots[k + 1]["from"] if k + 1 < len(shots) else round(length, 2)
        s["x"] = min(int(round(x0 + s["x"] * scale)), int(x0 + w - size[0])) // 2 * 2
        s["y"] = min(int(round(y0 + s["y"] * scale)), int(y0 + h - size[1])) // 2 * 2
    out = {"zoom": zoom, "size": size, "letters": round(on_phone(height, size[0], shown), 1) if height else None,
           "shots": [{"from": s["from"], "to": s["to"], "x": s["x"], "y": s["y"]} for s in shots]}
    if not follow:
        out["on"] = "lines"
    return out


def _on_lines(text, region, scale, small, ww, wh, banned):
    """(x, y) in small pixels: the window over the most lines of text (the highest within TOP_NEAR of the most), its
    left edge at their start; None when no window holds a line."""
    x0, y0 = region[:2]
    sw, sh = small
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
    return x, y


def _kept_on_lines(text, pieces, region, scale, small, ww, wh, banned, start, length):
    """The card's shots on a screen where no one part changes: from start, and to where the most lines are whenever a
    reading finds its place holding under FEW of what the busiest place holds."""
    shots, since = [{"from": 0.0, "x": start[0], "y": start[1]}], 0.0
    for t, read in _readings(text, pieces):
        now = _text_map(read, region, scale, *small)
        if banned is not None:
            now = now * (1 - banned)
        place = (shots[-1]["x"], shots[-1]["y"])
        if now.sum() <= 0 or _held(now, place[0], place[1], ww, wh) >= FEW * _best(now, ww, wh, banned)[0]:
            continue
        found = _on_lines(read, region, scale, small, ww, wh, banned)
        if found is None:
            continue
        if len(shots) == 1 and t < HOLD:
            shots[0].update(x=found[0], y=found[1])
            continue
        move = round(max(t, since + HOLD), 2)
        if move >= length:
            break
        shots.append({"from": move, "x": found[0], "y": found[1]})
        since = move
    return shots


def _readings(text, pieces):
    """[(clip second, the lines read then)] in order, for the lines that carry the second they were read at; a reading
    before a piece is the screen at that piece's start."""
    out = {}
    for line in text or []:
        if line.get("at") is None:
            continue
        at_ = 0.0
        for a, b in pieces:
            if line["at"] < b:
                out.setdefault(round(at_ + max(0.0, line["at"] - a), 2), []).append(line)
                break
            at_ += b - a
    return sorted(out.items())


def _followed(maps, sw, sh, ww, wh, banned):
    """The card's shots following the change in one part of the screen (the module's note)."""
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
    return shots


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


# ---- following a thing SAM followed ------------------------------------------------------------------------------
# A seller's product, held out to one side, then up by the face, then low: Tarm 10-30, on a frame that sat on his face
# while the controller he sold was at its edge, "โปรแกรมต้องฉลาดพอจะรู้ว่าต้องโฟกัสสินค้ามากกว่าคน ... ไม่ได้ขายหน้าตา".
THING_W = 192
EDGE_SHARE = 0.2
THING_PERSIST = 0.5
THING_HOLD = 1.0
THING_SMOOTH = 0.5
THING_LOST = 1.5
THING_STEP = 0.15


def thing(mask, pieces, size, aspect, rest=None):
    """The window of the vertical frame following a thing SAM followed (mask: person.cut_out's entry made with what,
    on the recording's pixel grid), as plan() gives one: {"size": [w, h], "shots": [{"from", "x", "y"}], "seen": the
    share of the clip's pictures the thing is in}, or None when it is in none or the recording is no wider than the
    window. The window is the recording's full height and as wide as aspect makes it. It starts centred on the thing
    (at rest when the thing is not in the first picture) and moves to centre it again once the thing's middle has
    been within EDGE_SHARE of the window's side, or past it, for THING_PERSIST s, no sooner than THING_HOLD s after
    its last move, so it follows a product held out to the side and does not shake with the hand that holds it; a
    move under THING_STEP of its width is not made. Where the thing is not seen the window holds, and once it has
    been gone THING_LOST s (put down, out of the picture) the window goes to rest, the x the person is at, so the
    clip shows them rather than an empty place. Measured 10-30 on Tarm's controller, 39 s: SAM saw it in 79% of the
    pictures, and the window moved 6 times."""
    w, h = size
    ww = min(w, int(round(h * aspect / 2.0)) * 2)
    if ww >= w:
        return None
    sw, sh = THING_W, max(2, int(round(THING_W * h / float(w) / 2)) * 2)
    raw = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-i", mask["path"], "-vf",
                          "scale=%d:%d" % (sw, sh), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True).stdout
    frames = np.frombuffer(raw[:len(raw) // (sw * sh) * sw * sh], np.uint8).reshape(-1, sh, sw)
    seen, offset = [], 0.0
    for a, b in pieces:
        for k, f in enumerate(frames):
            t = mask["from"] + k / float(mask["fps"])
            if a <= t < b:
                cols = (f >= 128).sum(axis=0)
                seen.append((offset + t - a, float((cols * np.arange(sw)).sum() / cols.sum()) * w / sw
                             if cols.sum() else None))
        offset += b - a
    found = [(t, c) for t, c in seen if c is not None]
    if not found:
        return None
    middles = [(t, statistics.median(c for u, c in found if abs(u - t) <= THING_SMOOTH / 2)) for t, _ in found]

    def place(c):
        return int(round(min(max(c - ww / 2.0, 0), w - ww) / 2.0)) * 2

    x, smooth = place(rest if seen[0][1] is None and rest is not None else middles[0][1]), dict(middles)
    shots, since, out, lost = [{"from": 0.0, "x": x, "y": 0}], 0.0, [], None
    for t, c in seen:
        if c is None:
            out, lost = [], t if lost is None else lost
            if rest is None or t - lost < THING_LOST:
                continue
            move, to = max(lost, since + THING_HOLD), place(rest)
        else:
            lost, c = None, smooth[t]
            if x + ww * EDGE_SHARE <= c <= x + ww * (1 - EDGE_SHARE):
                out = []
                continue
            out.append((t, c))
            if t - out[0][0] < THING_PERSIST:
                continue
            move, to = max(out[0][0], since + THING_HOLD), place(statistics.median(c for _, c in out))
            out = []
        if abs(to - x) >= THING_STEP * ww and move < offset:
            x, since = to, move
            shots.append({"from": round(move, 2), "x": x, "y": 0})
    return {"size": [ww, h], "shots": shots, "seen": round(len(found) / float(len(seen)), 2)}
