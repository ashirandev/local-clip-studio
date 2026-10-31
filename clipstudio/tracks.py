"""Face tracks: each person's face followed through the pictures faces.sample() took, so another person's face can be
blurred as it moves (effects.blur_track), and Muse can look at each person and decide who to blur.

A track never jumps to another face: in each picture, the faces are matched to the tracks nearest first, measured
from where each track's face is expected (moving on as it moved last), and a face further than REACH face sizes per
half second, or more than half again bigger or smaller, starts a track of its own.
A face missed for up to MISS half seconds (the finder misses a turned head) is bridged with a point for each picture
it was missed in, so the blur does not let it show for a moment; missed for longer, the track ends there, so no two
points of a track are more than 0.5 s apart, which is what blur_track asks. The points are smoothed (1-2-1), held a
quarter of a second before the first and after the last, and kept inside the picture.

Faces that will not stay in a track are looked for again every FINE s (finer): a straight line between pictures half a
second apart is not where a face went when the phone is on a horse. On the trip of 11-01, 7 clips in 28.5 s, the blur
from half a second covered 13% of the faces found on other frames whole and 38% more than half, and three montages had
passed on sheets too small to show it. There the points are not smoothed, as a shake is the face's real path, and each
box is WIDE times bigger.

Times are recording seconds and boxes are recording pixels, as faces.sample() gives them; lane 1 turns the times into
clip seconds with project.from_recording().
"""
import hashlib
import math
import os
import shutil
import subprocess
import tempfile

from . import faces, recording

REACH = 1.5
SIZE_CHANGE = 1.5
MISS = 1
STILL = 0.5
FINE = 0.05
SHORT = 6
FEW = 3
WIDE = 1.13
CROP_HEIGHT = 160
CROP_SHARE = 1.8


def tracks(found, rec=None):
    """One track per person in found (from faces.sample), most seen first:
    [{"n", "from", "to", "seen": pictures the face was found in, "steady": True for the face that sits still (a
    stream's face camera), "box": the median box, "points": [[t, x, y, w, h]], "crop": a small jpg of the face in its
    clearest picture when rec is given, else None}]."""
    every = found["every"]
    size = found["size"]
    per_half = every / faces.EVERY
    fine = every <= FINE
    open_tracks, done = [], []
    for sample in found["samples"]:
        t = sample["t"]
        boxes = [f[:4] for f in sample["faces"]]
        scores = [f[4] if len(f) > 4 else 1.0 for f in sample["faces"]]
        pairs = []
        for k, tr in enumerate(open_tracks):
            steps = max(1, round((t - tr["raw"][-1][0]) / every))
            ahead = _ahead(tr["raw"], steps)
            for j, box in enumerate(boxes):
                d = _apart(ahead, box, steps)
                if d is not None:
                    pairs.append((d, k, j))
        used_tracks, used_faces = set(), set()
        for d, k, j in sorted(pairs):
            if k in used_tracks or j in used_faces:
                continue
            used_tracks.add(k)
            used_faces.add(j)
            _extend(open_tracks[k], t, boxes[j], scores[j], every)
        still = []
        for k, tr in enumerate(open_tracks):
            if k in used_tracks or round(t - tr["raw"][-1][0], 2) <= (MISS + 1) * faces.EVERY - every:
                still.append(tr)
            else:
                done.append(tr)
        for j, box in enumerate(boxes):
            if j not in used_faces:
                still.append({"raw": [[t] + list(box)], "seen": 1, "best": (scores[j] * box[2] * box[3], t, box)})
        open_tracks = still
    done += open_tracks
    sit = faces.steady(found)
    out = []
    for tr in done:
        box = [sorted(p[i] for p in tr["raw"])[len(tr["raw"]) // 2] for i in range(1, 5)]
        points = _held(tr["raw"] if fine else _smoothed(tr["raw"]), faces.EVERY)
        points = [[round(t, 2)] + [round(v, 1) for v in _inside(_wider(p) if fine else p, size)] for t, *p in points]
        out.append({"from": points[0][0], "to": points[-1][0], "seen": max(1, round(tr["seen"] * per_half)),
                    "steady": _still(sit, box),
                    "box": [round(v, 1) for v in box], "points": points,
                    "crop": _crop(rec, tr["best"]) if rec else None})
    out.sort(key=lambda tr: (-tr["seen"], tr["from"]))
    for n, tr in enumerate(out, 1):
        tr["n"] = n
    return out


def finer(rec, a, b, found, on_start=None):
    """found, or the faces from a to b looked for every FINE s when found's faces will not stay in a track; found when
    that cannot be done. The clips of 11-01 tell them apart: 1.4 and 1.5 faces a track on the trip, 16 and more on a
    talk, a seller and a vlog."""
    made = tracks(found) if found["every"] > FINE else []
    if len(made) < FEW or sum(tr["seen"] for tr in made) >= SHORT * len(made):
        return found
    try:
        return faces.sample(rec, a, b, every=FINE, on_start=on_start)
    except (faces.NoFaces, faces.Stopped):
        return found


def _wider(box):
    """box WIDE times bigger around its middle."""
    x, y, w, h = box
    return [x - w * (WIDE - 1) / 2, y - h * (WIDE - 1) / 2, w * WIDE, h * WIDE]


def _still(sit, box):
    """True when box is the face that sits still (faces.steady), give or take STILL face sizes."""
    d = _apart(sit["box"], box, 1) if sit else None
    return d is not None and d < STILL


def _ahead(raw, steps):
    """Where the track's face is expected steps pictures after its last point: moving on as it moved last, so two
    faces that pass each other are not swapped."""
    last = raw[-1][1:]
    if len(raw) < 2:
        return last
    before = raw[-2][1:]
    return [v + (v - b) * steps if i < 2 else v for i, (v, b) in enumerate(zip(last, before))]


def _apart(a, b, steps):
    """How far face b is from face a, in face sizes per picture, or None when b cannot be the same face steps
    pictures later."""
    size = max(a[2], a[3], b[2], b[3])
    d = math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2) / size / steps
    ratio = b[3] / max(1e-6, a[3])
    if d > REACH or not 1 / SIZE_CHANGE <= ratio <= SIZE_CHANGE:
        return None
    return d


def _extend(tr, t, box, score, every):
    """Adds the face at t to the track, with a point between for each picture it was missed in."""
    last_t, last = tr["raw"][-1][0], tr["raw"][-1][1:]
    missed = round((t - last_t) / every) - 1
    for m in range(1, missed + 1):
        share = m / (missed + 1)
        tr["raw"].append([last_t + (t - last_t) * share] + [a + (b - a) * share for a, b in zip(last, box)])
    tr["raw"].append([t] + list(box))
    tr["seen"] += 1
    clear = score * box[2] * box[3]
    if clear > tr["best"][0]:
        tr["best"] = (clear, t, box)


def _smoothed(raw):
    """The points with each box the 1-2-1 mean of itself and its neighbours (the ends stay as they are)."""
    if len(raw) < 3:
        return [list(p) for p in raw]
    out = [list(raw[0])]
    for a, b, c in zip(raw, raw[1:], raw[2:]):
        out.append([b[0]] + [(x + 2 * y + z) / 4 for x, y, z in zip(a[1:], b[1:], c[1:])])
    out.append(list(raw[-1]))
    return out


def _held(points, every):
    """The track held half a picture before its first point and after its last, so the face is covered between
    the pictures at both ends."""
    first, last = points[0], points[-1]
    return [[max(0.0, first[0] - every / 2)] + first[1:]] + points + [[last[0] + every / 2] + last[1:]]


def _inside(box, size):
    x, y, w, h = box
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(size[0], x + w), min(size[1], y + h)
    return [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]


def _crop(rec, best):
    """A jpg CROP_HEIGHT px tall of the face at its clearest picture, with some room around it, made once."""
    _clear, t, box = best
    side = CROP_SHARE * max(box[2], box[3])
    x = min(max(0.0, box[0] + box[2] / 2 - side / 2), max(0.0, rec["width"] - side))
    y = min(max(0.0, box[1] + box[3] / 2 - side / 2), max(0.0, rec["height"] - side))
    side = min(side, rec["width"], rec["height"])
    x, y, side = int(x) // 2 * 2, int(y) // 2 * 2, max(2, int(side) // 2 * 2)
    key = hashlib.sha1(("%s|%d|%s|%.2f|%d|%d|%d" % (rec["path"], rec["size"], rec["mtime"], t, x, y, side)
                        ).encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(recording.WORK, "tracks")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, key + ".jpg")
    if not os.path.isfile(path):
        fd, part = tempfile.mkstemp(prefix=key + ".", suffix=".part.jpg", dir=folder)
        os.close(fd)
        done = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-ss", "%.2f" % t, "-i",
                               rec["path"], "-frames:v", "1", "-vf", "crop=%d:%d:%d:%d,scale=-2:%d"
                               % (side, side, x, y, CROP_HEIGHT), "-q:v", "3", "-f", "image2", part],
                              capture_output=True)
        if done.returncode != 0 or not os.path.getsize(part):
            os.remove(part)
            return None
        try:
            os.replace(part, path)
        except PermissionError:
            # made by another request at the same moment and open now (Windows): the same picture, so keep it
            os.remove(part)
            if not os.path.isfile(path):
                raise
    return path
