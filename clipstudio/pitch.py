"""Following the play of a football game in the upright frame.

On a game's wide view the ball is a few pixels: SAM saw it in 4% of a reel's pictures (10-31), and the busiest part of the
picture is often empty midfield, with a fixed middle holding 0 of 4 goals, as they fall at both ends. Tarm 10-31: "คลิปบอล
ผมว่า fix กลางจอ ไม่ดี ควรตามลูกบอล". The game itself marks the play: a bright cyan arrow over the player in control, who is
near the ball. So the frame follows that mark, leaning to the side the play is on, and in the last 1.5 s of a stretch of
match it goes to the goal the play ends at. Numbers are lane 5's mock of 10-31 on a 4-goal reel, which Tarm found smooth.

A picture with little pitch in it is a close-up or a replay, with no mark to follow: the frame goes to the middle, and a
hard camera cut is the only place it jumps. A mark is this game's, so another game or a real match finds none and follow()
says so: the card stays the way to see them. It is a colour, not a reading of words on the screen.
"""
import shutil
import subprocess

import numpy as np

W, H = 1920, 1080
TRACK_FPS = 15
MARKER = (np.array([84, 150, 200]), np.array([102, 255, 255]))
GRASS = (np.array([30, 40, 40]), np.array([85, 255, 255]))
BOARD, BOTTOM = 140, 930
NAMES = (840, 800, 1110)
MARK_AREA = (25, 700)
PITCH_LEAST = 0.45
CUT = 26.0
PLAY_AFTER = 3
LEAN = 0.5
STILL = 90.0
MOST_PAN = 420.0
SMOOTH = 0.8
LAST = 1.5
GOAL_SIDE = 150
STEP = 0.5
JUMP = 2 * MOST_PAN / TRACK_FPS
SEEN_LEAST = 0.25


def _frames(rec, a, b):
    """The pictures of rec from a to b seconds, TRACK_FPS a second, W x H, BGR."""
    p = subprocess.Popen([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-ss", "%.3f" % a, "-t", "%.3f" % (b - a), "-i",
                          rec["path"], "-vf", "fps=%d,scale=%d:%d" % (TRACK_FPS, W, H), "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = W * H * 3
    while True:
        raw = p.stdout.read(size)
        if len(raw) < size:
            break
        yield np.frombuffer(raw, np.uint8).reshape(H, W, 3)
    p.stdout.close()
    p.wait()


def _read(picture):
    """(the mark's x or None, the picture small and grey, to tell a camera cut by). The score board, the names under it
    and the bottom of the picture hold the same colour, and so do a player card's edges in a close-up, so only a picture
    that is mostly pitch is read."""
    import cv2
    small = cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    if cv2.inRange(hsv[BOARD:BOTTOM:4, ::4], *GRASS).mean() / 255 < PITCH_LEAST:
        return None, small
    mask = cv2.inRange(hsv, *MARKER)
    mask[:BOARD], mask[BOTTOM:] = 0, 0
    mask[NAMES[0]:, NAMES[1]:NAMES[2]] = 0
    n, _, stats, middles = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, area = None, 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if MARK_AREA[0] <= a <= MARK_AREA[1] and a > area:
            best, area = float(middles[i][0]), a
    return best, small


def _smooth(values, sigma):
    r = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(values, r, mode="edge"), kernel / kernel.sum(), mode="valid")


def _eased(values, most):
    """No more than most pixels a picture, both ways, so a pan starts before the play gets there."""
    out = values.copy()
    for i in range(1, len(out)):
        out[i] = np.clip(out[i], out[i - 1] - most, out[i - 1] + most)
    for i in range(len(out) - 2, -1, -1):
        out[i] = np.clip(out[i], out[i + 1] - most, out[i + 1] + most)
    return out


def _settled(values):
    """The aim stays put until the play leaves STILL pixels around it, so the frame does not breathe with each pass."""
    out = values.copy()
    for i in range(1, len(out)):
        d = values[i] - out[i - 1]
        out[i] = out[i - 1] if abs(d) < STILL else values[i] - np.sign(d) * STILL
    return out


def _aim(xs, width):
    """The frame's left edge, width wide, through one stretch of match, xs the mark's x in each picture or nan."""
    known = np.where(~np.isnan(xs))[0]
    if not len(known):
        return np.full(len(xs), (W - width) / 2.0)
    part = np.interp(np.arange(len(xs)), known, xs[known])
    part = part + LEAN * (part - W / 2)
    if abs(part[-1] - W / 2) > GOAL_SIDE:
        part[-int(LAST * TRACK_FPS):] = width / 2 if part[-1] < W / 2 else W - width / 2
    k = TRACK_FPS // 2 | 1
    part = np.array([np.median(part[max(0, n - k): n + k + 1]) for n in range(len(part))])
    return _eased(np.clip(_smooth(_settled(part), SMOOTH * TRACK_FPS) - width / 2, 0, W - width), MOST_PAN / TRACK_FPS)


def _left(rec, a, b, width):
    """(the frame's left edge in each picture from a to b, pictures of match, of them the ones with the mark)."""
    seen = [_read(f) for f in _frames(rec, a, b)]
    xs = np.array([np.nan if x is None else x for x, _ in seen], float)
    cut = [False] + [float(np.abs(seen[i][1] - seen[i - 1][1]).mean()) > CUT for i in range(1, len(seen))]
    match, on, run = np.zeros(len(xs), bool), True, 0
    for i in range(len(xs)):
        run = 0 if np.isnan(xs[i]) else run + 1
        on = False if cut[i] else on or run >= PLAY_AFTER
        match[i] = on
    left, i = np.full(len(xs), (W - width) / 2.0), 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and match[j + 1] == match[i]:
            j += 1
        if match[i]:
            left[i:j + 1] = _aim(xs[i:j + 1], width)
        i = j + 1
    return left, int(match.sum()), int((~np.isnan(xs[match])).sum())


def follow(rec, pieces, width, y=0):
    """The upright frame following the play over a clip's pieces, width pixels wide in rec, as focus.plan and focus.thing
    give one: {"size": [w, h], "shots": [{"from": clip seconds, "x", "y", "line": True}], "seen": the share of the
    match's pictures with the mark}, the path a straight line from each shot to the next (focus.at), a point every STEP s
    and both sides of a camera cut. None when rec is not wide, or it shows no game that draws the mark."""
    scale = rec["width"] / float(W)
    if rec["width"] / float(rec["height"]) < 1.5 or width >= rec["width"]:
        return None
    points, at, match, marked = [], 0.0, 0, 0
    for a, b in pieces:
        left, played, found = _left(rec, a, b, width / scale)
        match, marked = match + played, marked + found
        if len(left):
            jumps = np.where(np.abs(np.diff(left)) > JUMP)[0]
            keep = set(range(0, len(left), max(1, int(STEP * TRACK_FPS)))) | {len(left) - 1}
            points += [(at + k / float(TRACK_FPS), left[k]) for k in sorted(keep | set(jumps) | set(jumps + 1))]
        at += b - a
    if not match or marked < SEEN_LEAST * match:
        return None
    return {"size": [int(width), int(rec["height"])], "seen": round(marked / float(match), 2),
            "shots": [{"from": round(t, 3), "x": int(round(x * scale)) // 2 * 2, "y": y, "line": True} for t, x in points]}
