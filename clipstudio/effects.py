"""Effects: the pieces of FFmpeg filter graph that draw on the clip or add to its sound, each one exactly what its
numbers say.

Every piece returns a Fragment: filter graph text with exactly one [in] and one [out], its own inner labels written
[_a], [_b]..., and any extra file it needs (a sticker picture) as [f0], [f1]... in the order of its files. join()
chains the fragments of one stage, so the renderer never has to know what is inside them. The sound pieces (sound,
duck) take the clip's sound at 48 kHz stereo as [in] and need no file.

Times are seconds of the clip (0 is its first frame). An expression uses only t, the frame's time: never n, never
random(), and never a filter that remembers earlier frames, so a frame drawn on its own (the preview starts the graph
at that frame's time) is the same frame a full render draws.
"""
import collections
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile

Fragment = collections.namedtuple("Fragment", "graph files")

LABEL = re.compile(r"\[(in|out|_[a-z0-9]+|f[0-9]+)\]")
FRAME_W, FRAME_H, FPS = 1080, 1920, 30
MOST_GAP = 0.5
LANE_GAP = 0.1
SUM_MOST = 50
SAME_SIZE = 1.5
MOST_LANES = 16
STILL = 0.5
FLASH_LENGTH, FLASH_STRENGTH = 0.15, 0.8
POP = ((0.0, 0.6), (0.15, 1.1), (0.25, 1.0))
FADE = 0.15
PNG_SIGNATURE = bytes((137, 80, 78, 71, 13, 10, 26, 10))
BLUR = "boxblur=luma_radius='max(1,min(w,h)/6)':luma_power=3:chroma_radius='max(1,min(cw,ch)/6)':chroma_power=3"
STICKERS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stickers")


def join(fragments, first_label, last_label, first_input):
    """(graph text, files) for the fragments chained one after another, from [first_label] to [last_label].

    Inner labels are renamed after first_label, so two stages joined into one graph never share a label, and each
    fragment's [fN] becomes the input the renderer opens for that file: -loop 1 -framerate 30 -i <file>, numbered
    from first_input in the order of the returned files. No fragments at all is a plain pass-through."""
    for name, label in (("first_label", first_label), ("last_label", last_label)):
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_]+", label):
            raise ValueError("%s %r is not a label: use letters, digits and _ only, without brackets." % (name, label))
    if not isinstance(first_input, int) or isinstance(first_input, bool) or first_input < 0:
        raise ValueError("first_input %r is not an input number: use a whole number from 0." % (first_input,))
    fragments = list(fragments)
    if not fragments:
        return "[%s]null[%s]" % (first_label, last_label), []
    graphs, files = [], []
    for k, fragment in enumerate(fragments):
        graph, own = _check(k, fragment)
        before = first_label if k == 0 else "%s_%d" % (first_label, k)
        after = last_label if k == len(fragments) - 1 else "%s_%d" % (first_label, k + 1)

        def rename(m, k=k, before=before, after=after, base=len(files)):
            label = m.group(1)
            if label == "in":
                return "[%s]" % before
            if label == "out":
                return "[%s]" % after
            if label.startswith("f"):
                return "[%d:v]" % (first_input + base + int(label[1:]))
            return "[%s_%d%s]" % (first_label, k, label)

        graphs.append(LABEL.sub(rename, graph))
        files.extend(own)
    return ";".join(graphs), files


def _check(k, fragment):
    if not (isinstance(fragment, tuple) and len(fragment) == 2):
        raise ValueError("fragment %d is not a Fragment(graph, files)." % k)
    graph, files = fragment
    files = list(files)
    labels = LABEL.findall(graph)
    for label in ("in", "out"):
        if labels.count(label) != 1:
            raise ValueError("fragment %d has [%s] %d times: a fragment has exactly one [in] and one [out]."
                             % (k, label, labels.count(label)))
    for label in labels:
        if label.startswith("f") and int(label[1:]) >= len(files):
            raise ValueError("fragment %d uses [%s] but brings %d file(s)." % (k, label, len(files)))
    return graph, files


def _number(name, value, low=None):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value != value or abs(value) == math.inf:
        raise ValueError("%s %r is not a number: give it in %s." % (name, value, _unit(name)))
    if low is not None and value < low:
        raise ValueError("%s %s is below %s: use %s from %s." % (name, _f(value), _f(low), name, _f(low)))
    return float(value)


def _unit(name):
    if name in ("start", "end", "at", "length") or name.startswith("t of"):
        return "seconds of the clip"
    if name in ("level", "db"):
        return "dB"
    if name == "zoom":
        return "times the size, such as 1.1"
    if name in ("strength", "margin"):
        return "a share from 0 to 1"
    return "pixels"


def _window(start, end):
    start, end = _number("start", start, 0), _number("end", end, 0)
    if end <= start:
        raise ValueError("end %s is not after start %s: use an end later than the start." % (_f(end), _f(start)))
    return start, end


def _f(v):
    """A number written short for an expression."""
    return ("%.4f" % v).rstrip("0").rstrip(".") or "0"


def _even_box(box, field="box"):
    if (not isinstance(box, (tuple, list)) or len(box) != 4
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)):
        raise ValueError("%s %r is not a box: give (x, y, w, h) in pixels of the recording." % (field, box))
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError("%s %r has no size: w and h must be above 0." % (field, tuple(box)))
    if x < 0 or y < 0:
        raise ValueError("%s %r starts off the picture: x and y must be 0 or more." % (field, tuple(box)))
    x0, y0 = 2 * math.floor(x / 2), 2 * math.floor(y / 2)
    return x0, y0, 2 * math.ceil((x + w) / 2) - x0, 2 * math.ceil((y + h) / 2) - y0


def _blurred(w, h, x, y, cx, cy, when):
    """Crops w x h at (x, y), blurs it so nothing in it can be read, and puts it back at (cx, cy) while when is
    true. Outside the box, and while when is false, every pixel is the input's own."""
    return ("[in]split[_a][_b];[_b]crop=w='min(%s,iw)':h='min(%s,ih)':x='%s':y='%s',%s[_c];"
            "[_a][_c]overlay=x='%s':y='%s':eval=frame:format=auto:enable='%s'[out]" % (w, h, x, y, BLUR, cx, cy, when))


def blur_box(box, start, end):
    """Blurs a fixed box of the recording's picture (x, y, w, h in its pixels), such as a stream's chat box, from
    start to end, so nothing in it can be read. The blur grows with the box, so a big box is as unreadable as a small
    one. The box is widened to even pixels, so the picture's colour planes line up and everything outside it stays
    exactly as it was."""
    x, y, w, h = _even_box(box)
    start, end = _window(start, end)
    when = "gte(t,%s)*lt(t,%s)" % (_f(start), _f(end))
    return Fragment(_blurred(w, h, x, y, x, y, when), [])


def blur_track(points, margin=0.15):
    """Blurs a moving box, such as a face, along points [(t, x, y, w, h)] from lane 1's face tracker (t in seconds of
    the clip). Between two points the box moves in a straight line. Its size is the track's largest w and h plus
    margin, centred on the moving centre and kept inside the picture. Nothing is blurred before the first point or
    after the last one. Points more than 0.5 s apart are refused: split the track at its gaps."""
    return blur_tracks([points], margin)[0]


def blur_tracks(tracks, margin=0.15):
    """[Fragment]: the tracks blurred as blur_track blurs each, with tracks that never show at the same time sharing
    one fragment, so a crowd costs as many blurs as faces are on screen at once, not one for each face.

    Measured 10-28 on a 47 s montage of a trip where 69 faces were blurred one after another: FFmpeg drew 42 s in 37 s
    and then stood still at the end of the recording for over an hour, with every blur its own split and overlay;
    with 35 it stood still too, with 20 it finished in 22 s. The 69 faces were never more than 5 at once. A track
    joins a fragment whose last track ends LANE_GAP s before it starts and whose box is within SAME_SIZE of its own,
    as the fragment's box is its largest track's: a small face is not hidden under a big face's box. When that makes
    more than MOST_LANES fragments, sizes are not kept apart, so there are as many as faces show at once. More faces
    than about 30 at once would still stand FFmpeg still."""
    margin = _number("margin", margin, 0)
    if margin > 1:
        raise ValueError("margin %s is more than 1: use a share of the box from 0 to 1, such as 0.15." % margin)
    made = sorted((_track(points, margin) for points in tracks), key=lambda m: m[0][0][0])
    lanes = _lanes(made, SAME_SIZE)
    if len(lanes) > MOST_LANES:
        lanes = _lanes(made, None)
    return [_lane(lane["tracks"], lane["w"], lane["h"]) for lane in lanes]


def _lanes(made, same):
    """The tracks (track, w, h) in time order, put in lanes that never show two at once; with same, only tracks
    whose boxes are within same of each other share a lane."""
    lanes = []
    for track, w, h in made:
        free = [lane for lane in lanes if lane["end"] + LANE_GAP < track[0][0] and (
            same is None or (max(lane["w"], w) <= same * min(lane["w"], w) and
                             max(lane["h"], h) <= same * min(lane["h"], h)))]
        if free:
            lane = min(free, key=lambda lane: abs(lane["w"] - w) + abs(lane["h"] - h))
        else:
            lane = {"tracks": [], "w": w, "h": h}
            lanes.append(lane)
        lane["tracks"].append(track)
        lane["w"], lane["h"], lane["end"] = max(lane["w"], w), max(lane["h"], h), track[-1][0]
    return lanes


def _track(points, margin):
    """(track [(t, centre x, centre y, w, h)], box w, box h) for blur_track's points, or ValueError saying what to
    change."""
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError("points %r is not a track: give [(t, x, y, w, h)], one or more." % (points,))
    track = []
    for k, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) != 5:
            raise ValueError("point %d %r is not (t, x, y, w, h)." % (k, p))
        t = _number("t of point %d" % k, p[0], 0)
        x, y = _number("x of point %d" % k, p[1], 0), _number("y of point %d" % k, p[2], 0)
        w, h = _number("w of point %d" % k, p[3]), _number("h of point %d" % k, p[4])
        if w <= 0 or h <= 0:
            raise ValueError("point %d %r has no size: w and h must be above 0." % (k, tuple(p)))
        if track and t <= track[-1][0]:
            raise ValueError("point %d is at %s s, not after point %d at %s s: give the points in time order."
                             % (k, _f(t), k - 1, _f(track[-1][0])))
        if track and round(t - track[-1][0], 3) > MOST_GAP:
            raise ValueError("points %d and %d are %.2f s apart (%s to %s s), more than %s s: split the track at "
                             "the gap." % (k - 1, k, t - track[-1][0], _f(track[-1][0]), _f(t), MOST_GAP))
        track.append((t, x + w / 2, y + h / 2, w, h))
    w = 2 * math.ceil(max(p[3] for p in track) * (1 + margin) / 2)
    h = 2 * math.ceil(max(p[4] for p in track) * (1 + margin) / 2)
    return track, w, h


def _sum(terms):
    """terms added up, as FFmpeg can read them: it refuses a sum of more than about 100 terms without saying why, only
    "Conversion failed!" (measured 10-31: 83 are read, 105 are not), so a long sum is grouped in brackets. A face
    followed for most of a 58 s talk clip made 111, and every render of the clip failed until its blur was taken off."""
    terms = list(terms)
    while len(terms) > SUM_MOST:
        terms = ["(%s)" % "+".join(terms[k:k + SUM_MOST]) for k in range(0, len(terms), SUM_MOST)]
    return "+".join(terms)


def _lane(tracks, w, h):
    """One fragment blurring a w x h box along each track in turn; the tracks never show at the same time."""
    spans, xs, ys = [], [], []
    for track in tracks:
        path = _simplify([(t, cx, cy) for t, cx, cy, _w, _h in track])
        spans.append("gte(t,%s)*lte(t,%s)" % (_f(track[0][0]), _f(track[-1][0])))
        xs.append(_along(path, 1))
        ys.append(_along(path, 2))
    if len(tracks) == 1:
        cx, cy = xs[0], ys[0]
    else:
        cx = "(%s)" % _sum("%s*%s" % (s, x) for s, x in zip(spans, xs))
        cy = "(%s)" % _sum("%s*%s" % (s, y) for s, y in zip(spans, ys))
    left = "clip(2*floor((%s-%s)/2),0,%s)" % (cx, _f(w / 2), "%s")
    top = "clip(2*floor((%s-%s)/2),0,%s)" % (cy, _f(h / 2), "%s")
    return Fragment(_blurred(w, h, left % "iw-ow", top % "ih-oh", left % "W-w", top % "H-h", _sum(spans)), [])


def _along(path, k):
    """An expression in t for coordinate k of the path, in straight lines between its points."""
    if len(path) == 1:
        return _f(path[0][k])
    terms = []
    for a, b in zip(path, path[1:]):
        slope = (b[k] - a[k]) / (b[0] - a[0])
        line = _f(a[k]) if abs(slope) < 1e-9 else "(%s+(t-%s)*%s)" % (_f(a[k]), _f(a[0]), _f(slope))
        terms.append("gte(t,%s)*lt(t,%s)*%s" % (_f(a[0]), _f(b[0]), line))
    terms.append("gte(t,%s)*%s" % (_f(path[-1][0]), _f(path[-1][k])))
    return "(%s)" % _sum(terms)


def _simplify(path, within=STILL):
    """The path with the points dropped that the straight line between their neighbours already puts within
    `within` px, so a long, smooth track gives a short expression."""
    if len(path) < 3:
        return path
    kept = [path[0]]
    for k in range(1, len(path) - 1):
        a, b, c = kept[-1], path[k], path[k + 1]
        share = (b[0] - a[0]) / (c[0] - a[0])
        guess = [a[i] + (c[i] - a[i]) * share for i in (1, 2)]
        if max(abs(guess[0] - b[1]), abs(guess[1] - b[2])) > within or not _covers(kept[-1], path, k + 1, within):
            kept.append(b)
    kept.append(path[-1])
    return kept


def _covers(a, path, upto, within):
    """True when the straight line from a to path[upto] puts every point between them within `within` px."""
    c = path[upto]
    for b in path[path.index(a) + 1:upto]:
        share = (b[0] - a[0]) / (c[0] - a[0])
        if max(abs(a[1] + (c[1] - a[1]) * share - b[1]), abs(a[2] + (c[2] - a[2]) * share - b[2])) > within:
            return False
    return True


# ---- hits ------------------------------------------------------------------------------------------------------

def _hit(at, length, most=5.0):
    at, length = _number("at", at, 0), _number("length", length)
    if not 0 < length <= most:
        raise ValueError("length %s is not a hit's length: use seconds above 0, up to %s." % (_f(length), _f(most)))
    return at, length, "gte(t,%s)*lt(t,%s)" % (_f(at), _f(at + length))


def flash(at, length=FLASH_LENGTH, strength=FLASH_STRENGTH):
    """A white flash at `at`: the frame jumps brighter by strength (0 to 1) and fades back over length seconds.
    Outside the window every frame is the input's own."""
    at, length, when = _hit(at, length)
    strength = _number("strength", strength)
    if not 0 < strength <= 1:
        raise ValueError("strength %s is not a flash strength: use a share from above 0 to 1, such as 0.8."
                         % _f(strength))
    fade = "(%s*(1-(t-%s)/%s))" % (_f(strength), _f(at), _f(length))
    return Fragment("[in]eq=brightness='%s':saturation='1-%s':eval=frame:enable='%s'[out]" % (fade, fade, when), [])


def flash_alone(length=FLASH_LENGTH):
    """flash() from second 0 for a track of its own, over nothing: white fading to nothing. flash() adds light and an
    overlay mixes it, so this starts fully white to come as close (on EP22, at worst 22 of 255 off, against 41 from 0.8)."""
    return Fragment("color=c=white:s=%dx%d:r=30:d=%s,format=rgba,fade=t=out:st=0:d=%s:alpha=1[_w];"
                    "[in][_w]overlay=shortest=1:format=auto[out]" % (FRAME_W, FRAME_H, _f(length), _f(length)), [])


def shake(at, length=0.4, pixels=18):
    """The frame shakes from `at`, up to `pixels` px each way, and settles by itself within length seconds: a sine
    that dies away, from t only. The picture is enlarged just enough that no edge shows."""
    at, length, when = _hit(at, length)
    pixels = _number("pixels", pixels)
    if not 1 <= pixels <= 120:
        raise ValueError("pixels %s is not a shake size: use 1 to 120 pixels." % _f(pixels))
    p = int(math.ceil(pixels))
    wide = 2 * math.ceil((FRAME_W + 2 * p) / 2)
    tall = 2 * math.ceil(wide * FRAME_H / FRAME_W / 2)
    s = "(t-%s)" % _f(at)
    dying = "exp(-4*%s/%s)" % (s, _f(length))
    dx = "%s*%s*sin(2*PI*9*%s)" % (_f(pixels), dying, s)
    dy = "%s*%s*cos(2*PI*7*%s)*0.7" % (_f(pixels), dying, s)
    x = "%d+%s" % ((wide - FRAME_W) // 2, dx)
    y = "%d+%s" % ((tall - FRAME_H) // 2, dy)
    return Fragment("[in]split[_a][_b];[_b]scale=%d:%d,crop=%d:%d:x='%s':y='%s'[_c];"
                    "[_a][_c]overlay=0:0:format=auto:enable='%s'[out]" % (wide, tall, FRAME_W, FRAME_H, x, y, when), [])


def punch_in(at, length=0.5, zoom=1.15, centre=(540, 960)):
    """A punch-in: from `at`, for length seconds, the frame is shown zoom times bigger around centre (x, y on the
    1080x1920 frame), then cuts back. The point at centre stays where it is, unless that would show an edge."""
    at, length, when = _hit(at, length)
    zoom = _number("zoom", zoom)
    if not 1 < zoom <= 3:
        raise ValueError("zoom %s is not a punch-in: use more than 1, up to 3, such as 1.15." % _f(zoom))
    cx, cy = _centre(centre)
    wide = 2 * math.ceil(FRAME_W * zoom / 2)
    tall = 2 * math.ceil(FRAME_H * zoom / 2)
    x = min(max(0, round(cx * wide / FRAME_W - cx)), wide - FRAME_W)
    y = min(max(0, round(cy * tall / FRAME_H - cy)), tall - FRAME_H)
    return Fragment("[in]split[_a][_b];[_b]scale=%d:%d,crop=%d:%d:%d:%d[_c];"
                    "[_a][_c]overlay=0:0:format=auto:enable='%s'[out]" % (wide, tall, FRAME_W, FRAME_H, x, y, when), [])


def slow_push(start, end, zoom=1.08, centre=(540, 960)):
    """A slow push-in on the whole frame: from start to end it grows smoothly from its own size to zoom times it
    around centre (x, y on the 1080x1920 frame), easing in and out, then cuts back at end. Outside start to end every
    frame is the input's own. The renderer pushes the layout's main part instead (layouts.graph), so the frame's
    design holds still; this is the same push for a picture that is the whole frame."""
    return Fragment("[in]%s[out]" % push_filters(start, end, zoom, centre, (FRAME_W, FRAME_H)), [])


def push_filters(start, end, zoom, centre, size):
    """The filters (text for one chain) of a slow push on a picture of size (w, h), around centre (x, y) in its
    pixels: see slow_push. The point at centre stays where it is, give or take half a pixel, so no edge ever shows.
    The picture grows in steps of 2 pixels, each frame drawn sharp at its own size."""
    start, end = _window(start, end)
    if not 0.5 <= end - start <= 30:
        raise ValueError("start %s to end %s is %s s: a slow push runs 0.5 to 30 seconds." % (
            _f(start), _f(end), _f(end - start)))
    zoom = _number("zoom", zoom)
    if not 1 < zoom <= 1.5:
        raise ValueError("zoom %s is not a slow push: use more than 1, up to 1.5, such as 1.08." % _f(zoom))
    w, h = size
    cx, cy = _centre(centre, size)
    u = "clip((t-%s)/%s,0,1)" % (_f(start), _f(end - start))
    when = "gte(t,%s)*lt(t,%s)" % (_f(start), _f(end))
    z = "(%s*(1-cos(PI*%s))/2*%s)" % (_f(zoom - 1), u, when)
    wide = "(%d+2*round(%s*%s))" % (w, _f(w / 2), z)
    tall = "(%d+2*round(%s*%s))" % (h, _f(h / 2), z)
    # crop's offsets come from t too (crop keeps the size it first saw as iw and ih); exact=1 lets them be odd, so
    # the centre does not jump by a pixel each time the size steps.
    return ("scale=w='%s':h='%s':eval=frame:flags=lanczos,crop=w=%d:h=%d:x='round(%s*(%s/%d-1))':"
            "y='round(%s*(%s/%d-1))':exact=1" % (wide, tall, w, h, _f(cx), wide, w, _f(cy), tall, h))


def _centre(centre, size=(FRAME_W, FRAME_H)):
    w, h = size
    where = "the %dx%d frame" % size if size == (FRAME_W, FRAME_H) else "the %dx%d picture" % size
    if (not isinstance(centre, (tuple, list)) or len(centre) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in centre)):
        raise ValueError("centre %r is not a point: give (x, y) on %s." % (centre, where))
    cx, cy = centre
    if not (0 <= cx <= w and 0 <= cy <= h):
        raise ValueError("centre %r is off %s: use x from 0 to %d and y from 0 to %d." % (tuple(centre), where, w, h))
    return cx, cy


# ---- looks -------------------------------------------------------------------------------------------------------

# vivid: more contrast for games and anything flat, as an S-curve on the brightness alone (u - 0.25 sin(2 pi u) / 2 pi
# on the 16 to 235 range, which only rises), so no colour shifts, then more colour in the greens, cyans and blues
# only, so a game's world and a sky get richer while skin and reds stay exactly as they were.
# warm: the middle tones warmer for a face or food, black and white left where they are, so a white product stays
# white. film: a calm, soft clip: black lifted and a little blue, white softened, colour a little less.
LOOKS = {
    "vivid": "lutyuv=y='if(between(val,16,235),16+219*((val-16)/219-0.25*sin(2*PI*(val-16)/219)/(2*PI)),val)',"
             "huesaturation=saturation=0.3:colors=g+c+b:strength=10",
    "warm": "curves=r='0/0 0.5/0.525 1/1':g='0/0 0.5/0.505 1/1':b='0/0 0.5/0.465 1/1',vibrance=intensity=0.05",
    "film": "curves=r='0/0.05 0.5/0.51 1/0.95':g='0/0.05 0.5/0.5 1/0.95':b='0/0.08 0.5/0.49 1/0.92',"
            "vibrance=intensity=-0.15",
}


def look(name):
    """A colour look by name (see LOOKS) for the recording's picture: the same for every frame, so it never moves,
    and every curve only rises, so a darker colour never comes out lighter than a brighter one. The picture keeps
    its size."""
    if name not in LOOKS:
        raise ValueError("look %r is not a look: use one of %s." % (name, ", ".join(LOOKS)))
    return Fragment("[in]%s[out]" % LOOKS[name], [])


# ---- stickers ----------------------------------------------------------------------------------------------------

def sticker(png, at, length, box):
    """A picture (a PNG, its see-through parts left see-through) fitted inside box (x, y, w, h on the 1080x1920
    frame), keeping its shape, centred: from `at` it pops in at 0.6 times that size, grows to 1.1 and settles at 1.0
    within 0.25 s, holds, and fades out over its last 0.15 s. Outside its time every frame is the input's own. The
    picture is an extra file, so its stream's time is clip time: the renderer opens it from clip second 0 (from the
    preview frame's second for a preview)."""
    size = _png_size(png)
    at, length = _number("at", at, 0), _number("length", length)
    shortest = POP[-1][0] + FADE
    if not shortest <= length <= 60:
        raise ValueError("length %s is not a sticker's length: use %s to 60 seconds, so it can pop and fade."
                         % (_f(length), _f(shortest)))
    x, y, w, h = _frame_box(box)
    fit = min(w / size[0], h / size[1])
    base_w, base_h = size[0] * fit, size[1] * fit
    grow = _popping("(t-%s)" % _f(at))
    when = "gte(t,%s)*lt(t,%s)" % (_f(at), _f(at + length))
    return Fragment("[f0]format=rgba,scale=w='max(2,trunc(%s*%s))':h='max(2,trunc(%s*%s))':eval=frame,"
                    "fade=t=out:st=%s:d=%s:alpha=1[_s];"
                    "[in][_s]overlay=x='%s-w/2':y='%s-h/2':eval=frame:shortest=1:format=auto:enable='%s'[out]"
                    % (_f(base_w), grow, _f(base_h), grow, _f(at + length - FADE), _f(FADE),
                       _f(x + w / 2), _f(y + h / 2), when), [png])


PICTURE_FADE = 0.3


def picture(png, at, length, box):
    """A picture drawn for what is said (picture.fitted's PNG, already box's size) over box (x, y, w, h on the
    1080x1920 frame) from `at` for length seconds, coming in and going out over PICTURE_FADE s: it changes what the
    viewer looks at without a jump. Outside its time every frame is the input's own; its stream's time is clip time,
    as a sticker's is."""
    at, length = _number("at", at, 0), _number("length", length)
    if not 2 * PICTURE_FADE <= length <= 60:
        raise ValueError("length %s is not a picture's length: use %s to 60 seconds, so it can come in and go out."
                         % (_f(length), _f(2 * PICTURE_FADE)))
    x, y, w, h = _frame_box(box)
    when = "gte(t,%s)*lt(t,%s)" % (_f(at), _f(at + length))
    return Fragment("[f0]format=rgba,scale=%d:%d,fade=t=in:st=%s:d=%s:alpha=1,fade=t=out:st=%s:d=%s:alpha=1[_p];"
                    "[in][_p]overlay=x=%d:y=%d:shortest=1:format=auto:enable='%s'[out]"
                    % (w, h, _f(at), _f(PICTURE_FADE), _f(at + length - PICTURE_FADE), _f(PICTURE_FADE), x, y,
                       when), [png])


def _popping(u):
    """The size, in times its own, of a picture u seconds after it appears: POP's steps joined by straight lines."""
    (t0, s0), (t1, s1), (t2, s2) = POP
    return "if(lt(%s,%s),%s,if(lt(%s,%s),%s+%s*(%s-%s)/%s,if(lt(%s,%s),%s+%s*(%s-%s)/%s,%s)))" % (
        u, _f(t0), _f(s0), u, _f(t1), _f(s0), _f(s1 - s0), u, _f(t0), _f(t1 - t0),
        u, _f(t2), _f(s1), _f(s2 - s1), u, _f(t1), _f(t2 - t1), _f(s2))


def stickers():
    """The studio's own stickers, [{"name", "png", "en", "th"}] sorted by name, from stickers/stickers.json (Fluent
    Emoji 3D by Microsoft, MIT): what Muse and the person choose a shipped sticker from, by its name or its English
    or Thai words. A person's own PNG is a sticker too: sticker() takes any PNG."""
    with open(os.path.join(STICKERS, "stickers.json"), encoding="utf-8") as f:
        listed = json.load(f)["stickers"]
    return [{"name": name, "png": os.path.join(STICKERS, name + ".png"), "en": words["en"], "th": words["th"]}
            for name, words in sorted(listed.items())]


def sticker_png(name):
    """The path of the studio's sticker called name, one of the names stickers() lists."""
    shipped = stickers()
    for s in shipped:
        if s["name"] == name:
            return s["png"]
    raise ValueError("sticker %r is not one of the studio's stickers: use a name stickers() lists, such as %s."
                     % (name, ", ".join(s["name"] for s in shipped if s["name"] in ("fire", "money_bag", "star"))))


def _png_size(png):
    """(w, h) of a PNG file, read from its header."""
    try:
        with open(png, "rb") as f:
            head = f.read(24)
    except (OSError, TypeError, ValueError):
        head = b""
    if len(head) < 24 or head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        raise ValueError("png %r is not a PNG picture: give the path of a .png file%s."
                         % (png, "" if isinstance(png, str) and os.path.isfile(png) else " that exists"))
    w, h = struct.unpack(">II", head[16:24])
    if not w or not h:
        raise ValueError("png %r has no size: give a picture at least 1 x 1 pixel." % (png,))
    return w, h


def _frame_box(box):
    if (not isinstance(box, (tuple, list)) or len(box) != 4
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)):
        raise ValueError("box %r is not a box: give (x, y, w, h) on the %dx%d frame." % (box, FRAME_W, FRAME_H))
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError("box %r has no size: w and h must be above 0." % (tuple(box),))
    if x < 0 or y < 0 or x + w > FRAME_W or y + h > FRAME_H:
        raise ValueError("box %r runs off the %dx%d frame: keep x + w at most %d and y + h at most %d."
                         % (tuple(box), FRAME_W, FRAME_H, FRAME_W, FRAME_H))
    return x, y, w, h


# ---- flurries: many stickers moving together --------------------------------------------------------------------

# {move: (English, Thai, its length in seconds when none is given)}: what Muse and the person pick a flurry's move from
MOVES = {
    "burst": ("Fly out from the middle of the box every way, spinning, then drop a little and fade.",
              "พุ่งกระจายออกจากกลางกล่อง หมุนไปมา แล้วตกลงนิดหนึ่งก่อนจางหาย", 1.4),
    "rain": ("Fall from the top of the box to its bottom one after another, swaying.",
             "ร่วงจากด้านบนลงด้านล่างทีละชิ้น ส่ายไปมา", 2.4),
    "fountain": ("Shoot up from the bottom of the box and fall back, one after another.",
                 "พุ่งขึ้นจากด้านล่างแล้วตกกลับลงมาทีละชิ้น เหมือนน้ำพุ", 1.8),
    "bounce": ("Drop from the top of the box, bounce twice on its bottom and lie there.",
               "หล่นจากด้านบน เด้งบนพื้นสองครั้งแล้วนิ่งอยู่", 2.2),
    "rise": ("Float up from the bottom of the box to its top, swaying, like balloons.",
             "ลอยขึ้นจากด้านล่างถึงด้านบน ส่ายเบา ๆ เหมือนลูกโป่ง", 2.6),
    "scatter": ("Pop up one by one at spots all over the box, stay, then fade together.",
                "โผล่ขึ้นมาทีละชิ้นทั่วกล่อง ค้างไว้ แล้วจางหายพร้อมกัน", 2.0),
    "popper": ("Shoot up and in from both bottom corners of the box, like a party popper, then drift down.",
               "ยิงเฉียงขึ้นจากมุมล่างทั้งสองข้าง แบบพลุกระดาษ แล้วค่อย ๆ ร่วงลงมา", 2.2),
}
# {move: (share of the length the copies' starts are spread over, share of it each copy moves for, whether each
# copy stays where it ended until the flurry's end)}
TIMING = {"burst": (0.0, 1.0, False), "rain": (0.5, 0.5, False), "fountain": (0.4, 0.6, False),
          "bounce": (0.35, 0.5, True), "rise": (0.4, 0.6, False), "scatter": (0.5, 0.5, True),
          "popper": (0.12, 0.88, False)}
# the pictures each move's sample in moves() is drawn with
MOVE_SAMPLES = {"burst": ("money_with_wings", "coin", "dollar_banknote"), "rain": ("coin", "dollar_banknote"),
                "fountain": ("coin", "gem_stone"), "bounce": ("coin",), "rise": ("red_heart", "sparkles"),
                "scatter": ("fire", "star", "hundred_points"), "popper": ("confetti", "star")}
FLURRY_COUNTS = (1, 40)
FLURRY_LENGTHS = (0.8, 6.0)
FLURRY_SIZES = (40, 400)
FLURRY_PICTURES = 6
GROW = (0.75, 1.2)
APPEAR, LEAVE = 0.15, 0.3
CONFETTI = "confetti"
CONFETTI_COLOURS = ("#FFC83D", "#FF4F9A", "#3DA5FF", "#34D399", "#A78BFA", "#FF7A1A")
CONFETTI_SIZE = (20, 50)
SAMPLE_BOX = (60, 200, 960, 1520)
SAMPLE_GROUND = "0x23262E"


def _jitter(k, salt):
    """A share from 0 to 1 that looks random but is the same for copy k every time (no random())."""
    v = math.sin((k + 1) * 12.9898 + salt * 78.233) * 43758.5453
    return v - math.floor(v)


def _halton(k, base):
    """The k-th number (from 1) of the Halton sequence in base: spots that spread evenly however many there are."""
    share, step = 0.0, 1.0 / base
    while k:
        share += step * (k % base)
        k //= base
        step /= base
    return share


def confetti_png(colour):
    """The path of a paper piece in colour (#RRGGBB), CONFETTI_SIZE px, made once in work/shapes."""
    from . import layouts
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    w, h = CONFETTI_SIZE
    return layouts._shape("confetti_%s" % colour[1:].lower(), w, h,
                          "format=rgba,geq=r=%d:g=%d:b=%d:a=255[o]" % (r, g, b))


def _path(move, k, count, u, since, inner):
    """(x, y, angle) expressions of copy k's centre on the frame and its turn in radians: u its share of the way
    (0 to 1), since the seconds since it started, inner the box its centre stays in."""
    x0, y0, x1, y1 = inner
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    j = [_jitter(k, salt) for salt in range(8)]
    spin = (1 if j[3] > 0.5 else -1) * (0.6 + 0.8 * j[4])
    lane = min(1.0, max(0.0, (k + 0.5 + 0.6 * (j[6] - 0.5)) / count))
    if move == "burst":
        a = 2 * math.pi * k / count + (j[1] - 0.5) * 0.9
        far = 0.65 + 0.35 * j[2]
        out = "(1-pow(1-%s,3))" % u
        return ("%s+%s*%s" % (_f(cx), _f(w / 2 * far * math.cos(a)), out),
                "%s+%s*%s+%s*%s*%s" % (_f(cy), _f(h / 2 * 0.75 * far * math.sin(a)), out, _f(h / 2 * 0.25), u, u),
                "%s*%s" % (_f(spin * 2 * math.pi * 0.35), u))
    if move == "rain":
        sway = min(30.0, 0.06 * w)
        return ("%s+%s*sin(2*PI*(0.8*%s+%s))" % (_f(x0 + sway + (w - 2 * sway) * lane), _f(sway), u, _f(j[5])),
                "%s+%s*(0.35*%s+0.65*%s*%s)" % (_f(y0), _f(h), u, u, u), "%s*%s" % (_f(spin * 2 * math.pi * 0.25), u))
    if move == "fountain":
        return ("%s+%s*(1-pow(1-%s,2))" % (_f(cx), _f(w / 2 * (2 * lane - 1)), u),
                "%s-%s*4*%s*(1-%s)" % (_f(y1), _f(h * (0.65 + 0.35 * j[1])), u, u),
                "%s*%s" % (_f(spin * 2 * math.pi * 0.4), u))
    if move == "bounce":
        p = "(2.5*pow(%s,1.6))" % u
        return (_f(x0 + w * lane), "%s-%s*abs(cos(PI*%s))*exp(-1.2*%s)" % (_f(y1), _f(h), p, p),
                "%s*%s" % (_f((j[4] - 0.5) * 1.2), u))
    if move == "rise":
        sway = min(40.0, 0.08 * w)
        wave = "sin(2*PI*(0.6*%s+%s))" % (u, _f(j[5]))
        return ("%s+%s*%s" % (_f(x0 + sway + (w - 2 * sway) * lane), _f(sway), wave),
                "%s-%s*%s" % (_f(y1), _f(h), u), "0.25*%s" % wave)
    if move == "scatter":
        return (_f(x0 + w * _halton(k + 1, 2)), _f(y0 + h * _halton(k + 1, 3)),
                "0.15*sin(2*PI*(0.5*%s+%s))" % (since, _f(j[5])))
    sway = min(30.0, 0.06 * w)
    side = 1 if (k + k // 2) % 2 == 0 else -1
    mates = [m for m in range(count) if (m + m // 2) % 2 == (0 if side == 1 else 1)]
    aim = math.radians(15 + 65 * (mates.index(k) + 0.1 + 0.8 * j[2]) / len(mates))
    far = min((w - 2 * sway) / math.cos(aim), h / math.sin(aim)) * (0.55 + 0.45 * j[1])
    start = x0 + sway if side == 1 else x1 - sway
    out = "(1-pow(1-min(1,%s/0.35),3))" % u
    q = "clip((%s-0.35)/0.65,0,1)" % u
    return ("%s+%s*%s+%s*sin(2*PI*(1.5*%s+%s))*%s" % (_f(start), _f(side * far * math.cos(aim)), out, _f(sway), q,
                                                     _f(j[5]), q),
            "%s-%s*(%s-0.45*%s*%s)" % (_f(y1), _f(far * math.sin(aim)), out, q, q),
            "%s*%s" % (_f(spin * 2 * math.pi * 0.5), u))


def flurry(pngs, at, box, move="burst", count=8, length=None, size=150):
    """Copies of one to six pictures moving together inside box (x, y, w, h on the 1080x1920 frame) from `at`, by
    one of the MOVES: each copy its own way (its own path, start and spin, and 0.75 to 1.2 times size px on its
    longer side), the same every time, from t only. No copy ever reaches outside box, spinning included. pngs is a
    PNG's path or a list of them; CONFETTI ("confetti") in it stands for paper pieces in six colours, CONFETTI_SIZE
    px whatever size says, which flutter as they move. Copy k is picture k of the list, round and round. length is
    the move's own when None. Outside at to at + length every frame is the input's own."""
    items = [pngs] if isinstance(pngs, str) else pngs
    if not isinstance(items, (list, tuple)) or not 1 <= len(items) <= FLURRY_PICTURES:
        raise ValueError("pngs %r is not one to %d pictures: give a PNG's path or a list of them, %s for paper pieces."
                         % (pngs, FLURRY_PICTURES, CONFETTI))
    if move not in MOVES:
        raise ValueError("move %r is not a flurry's move: use one of %s." % (move, ", ".join(MOVES)))
    if isinstance(count, bool) or not isinstance(count, int) or not FLURRY_COUNTS[0] <= count <= FLURRY_COUNTS[1]:
        raise ValueError("count %r is not a number of copies: use a whole number from %d to %d."
                         % ((count,) + FLURRY_COUNTS))
    at = _number("at", at, 0)
    length = MOVES[move][2] if length is None else _number("length", length)
    if not FLURRY_LENGTHS[0] <= length <= FLURRY_LENGTHS[1]:
        raise ValueError("length %s is not a flurry's length: use %s to %s seconds, or leave it out for the %s move's "
                         "%s s." % (_f(length), _f(FLURRY_LENGTHS[0]), _f(FLURRY_LENGTHS[1]), move, _f(MOVES[move][2])))
    size = _number("size", size)
    if not FLURRY_SIZES[0] <= size <= FLURRY_SIZES[1]:
        raise ValueError("size %s is not a picture size: use %d to %d px for its longer side, such as 150."
                         % ((_f(size),) + FLURRY_SIZES))
    x, y, w, h = _frame_box(box)
    copies = []
    for k in range(count):
        item = items[k % len(items)]
        if item == CONFETTI:
            png = confetti_png(CONFETTI_COLOURS[(k // len(items)) % len(CONFETTI_COLOURS)])
            copies.append((png, CONFETTI_SIZE, True))
        else:
            pw, ph = _png_size(item)
            fit = size / max(pw, ph)
            copies.append((item, (max(2, round(pw * fit)), max(2, round(ph * fit))), False))
    reach = max(math.ceil(math.hypot(*base) / 2 * GROW[1]) for _, base, _ in copies) + 2
    if w < 2 * reach + 2 or h < 2 * reach + 2:
        raise ValueError("box %r is too small for these pictures to move in: give it at least %d x %d px, or smaller "
                         "pictures." % (tuple(box), 2 * reach + 2, 2 * reach + 2))
    inner = (x + reach, y + reach, x + w - reach, y + h - reach)
    files, uses = [], collections.OrderedDict()
    for k, (png, base, _) in enumerate(copies):
        uses.setdefault((png, base), []).append(k)
        if png not in files:
            files.append(png)
    parts, picked = [], {}
    for (png, base), ks in uses.items():
        labels = ["_p%d" % k for k in ks]
        for k, label in zip(ks, labels):
            picked[k] = label
        parts.append("[f%d]format=rgba,scale=%d:%d%s" % (files.index(png), base[0], base[1], ",split=%d%s" % (
            len(ks), "".join("[%s]" % label for label in labels)) if len(ks) > 1 else "[%s]" % labels[0]))
    share, moving, held = TIMING[move]
    order = sorted(range(count), key=lambda k: _jitter(k, 11))
    chain, last = [], "in"
    for k, (png, base, flutter) in enumerate(copies):
        start = at + share * length * (order.index(k) / (count - 1) if count > 1 else 0)
        end = at + length if held else start + moving * length
        seen = end - start
        appear, leave = min(APPEAR, 0.3 * seen), min(LEAVE, 0.4 * seen)
        u = "clip((t-%s)/%s,0,1)" % (_f(start), _f(moving * length))
        since = "(t-%s)" % _f(start)
        px, py, angle = _path(move, k, count, u, since, inner)
        grow = _f(GROW[0] + (GROW[1] - GROW[0]) * _jitter(k, 5))
        if move == "scatter":
            grow = "%s*%s" % (grow, _popping(since))
        elif move in ("burst", "fountain", "popper"):
            grow = "%s*min(1,0.3+0.7*%s/%s)" % (grow, since, _f(appear))
        wide = "%s*max(0.15,abs(cos(2*PI*(2.5*%s+%s))))" % (grow, since, _f(_jitter(k, 7))) if flutter else grow
        fades = "fade=t=in:st=%s:d=%s:alpha=1," % (_f(start), _f(appear)) if move in ("rain", "bounce", "rise") else ""
        # c=black@0, not none: none leaves the corners as the last frame drew them, so a spinning copy smears
        chain.append("[%s]rotate=a='%s':c=black@0:ow='hypot(iw,ih)':oh=ow,"
                     "scale=w='max(2,trunc(iw*%s))':h='max(2,trunc(ih*%s))':eval=frame,"
                     "%sfade=t=out:st=%s:d=%s:alpha=1[_q%d]"
                     % (picked[k], angle, wide, grow, fades, _f(end - leave), _f(leave), k))
        after = "out" if k == count - 1 else "_o%d" % k
        chain.append("[%s][_q%d]overlay=x='%s-w/2':y='%s-h/2':eval=frame:shortest=1:format=yuv420:"
                     "enable='between(t,%s,%s)'[%s]" % (last, k, px, py, _f(start), _f(end), after))
        last = after
    return Fragment(";".join(parts + chain), files)


def moves():
    """The moves people and Muse pick a flurry's from: [{"name", "en", "th", "length", "clip"}], clip a 270 x 480
    MP4 at 30 fps with no sound (play it looping) of that move drawn with MOVE_SAMPLES' stickers on a dark ground,
    made once in work/moves."""
    return [{"name": name, "en": en, "th": th, "length": length, "clip": _move_clip(name)}
            for name, (en, th, length) in MOVES.items()]


def _move_clip(name):
    from . import recording
    pictures = [CONFETTI if p == CONFETTI else sticker_png(p) for p in MOVE_SAMPLES[name]]
    count = 24 if CONFETTI in pictures else 8 if name == "scatter" else 10
    graph, files = join([flurry(pictures, 0.2, SAMPLE_BOX, name, count)], "src", "o", 1)
    seconds = 0.2 + MOVES[name][2] + 0.3
    drawing = "[0:v]format=yuv420p[src];%s;[o]scale=270:480,format=yuv420p[v]" % graph
    key = hashlib.sha1(("%s|%s|%s" % (drawing, "|".join(files), _f(seconds))).encode("utf-8")).hexdigest()[:10]
    folder = os.path.join(recording.WORK, "moves")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "%s_%s.mp4" % (name, key))
    if os.path.isfile(path):
        return path
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
           "color=c=%s:s=%dx%d:r=%d:d=%s" % (SAMPLE_GROUND, FRAME_W, FRAME_H, FPS, _f(seconds))]
    for f in files:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-i", f]
    fd, part = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".part.mp4", dir=folder)
    os.close(fd)
    try:
        subprocess.run(cmd + ["-filter_complex", drawing, "-map", "[v]", "-an", "-c:v", "libx264", "-preset",
                              "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4",
                              part], check=True, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        os.remove(part)
        raise
    try:
        os.replace(part, path)
    except PermissionError:
        # made by another request at the same moment and open now (Windows): the same clip, so keep it
        os.remove(part)
        if not os.path.isfile(path):
            raise
    return path


# ---- sound -------------------------------------------------------------------------------------------------------

def _edge(length, attack, release):
    """1 inside a sound of length seconds, rising from 0 over its first attack seconds and falling back to 0 over its
    last release seconds, so the sound starts and ends at silence and never clicks."""
    return "min(1,t/%s)*min(1,max(0,(%s-t)/%s))" % (_f(attack), _f(length), _f(release))


def _hash(i):
    """A number from -1 to 1 that looks random but is always the same for the same i."""
    return "(2*(sin(%s*12.9898)*43758.5453-floor(sin(%s*12.9898)*43758.5453))-1)" % (i, i)


def _noise(p):
    """Noise along p: the hashed values at each whole p, joined smoothly, so the faster p runs, the brighter the
    noise. st() and ld() only hold numbers within one sample's sum; nothing is carried to the next sample."""
    return ("(st(1,%s);st(2,floor(ld(1)));st(3,ld(1)-ld(2));st(3,ld(3)*ld(3)*(3-2*ld(3)));%s+(%s-%s)*ld(3))"
            % (p, _hash("ld(2)"), _hash("(ld(2)+1)"), _hash("ld(2)")))


# The sounds are made here from t alone (t is 0 at the sound's start), so they need no file and no licence, and the
# same clip always sounds the same. {name: (seconds, expression, filters after it)}. Each one's loudest 400 ms sit
# under the voice's -14 LUFS: thump about -16, whoosh and ding -18, pop -20 (a pop is shorter than the meter's 400 ms).
SOUNDS = {
    # a deep hit for a punch-in or a shake: a sine falling from 155 to 45 Hz as it dies away, with a 4 ms tap at its
    # front, so a phone's small speaker hears it land
    "thump": (0.45, "%s*(0.57*exp(-t/0.11)*sin(2*PI*(45*t+4.4*(1-exp(-t/0.04))))+0.115*exp(-t/0.004)*%s)"
              % (_edge(0.45, 0.002, 0.03), _hash("floor(t*24000)")), ""),
    # air rushing past for a fast move or a flash: noise that swells and brightens to its middle, then fades
    "whoosh": (0.5, "%s*0.26*pow(sin(2*PI*t),2)*%s"
               % (_edge(0.5, 0.005, 0.05), _noise("500*t+5500*(t/2-sin(4*PI*t)/(8*PI))")), ",highpass=f=250"),
    # a bubble for a sticker or a big word: a short sine rising from 280 to 900 Hz
    "pop": (0.14, "%s*0.56*exp(-t/0.03)*sin(2*PI*(900*t-9.3*(1-exp(-t/0.015))))" % _edge(0.14, 0.001, 0.02), ""),
    # a small bell for a price or a number on screen: E6 and two partials, the higher ones dying first
    "ding": (1.0, "%s*0.177*(exp(-t/0.3)*sin(2*PI*1318.5*t)+0.45*exp(-t/0.18)*sin(2*PI*2637*t)"
             "+0.18*exp(-t/0.1)*sin(2*PI*3955.5*t))" % _edge(1.0, 0.003, 0.05), ""),
}
RATE = 48000
SOUND_LEVELS = (-20, 6)
DUCK_DOWN, DUCK_UP, DUCK_BRIDGE = 0.25, 0.5, 1.5


def sound(kind, at, level=0):
    """The clip's sound with the sound `kind` (one of SOUNDS) added from `at`, level dB louder than it is made (below
    0 is quieter): [in] and [out] are sound, 48 kHz stereo. Every other sample of the clip's sound is left as it was,
    and a sound that runs past the clip's end is cut there."""
    if kind not in SOUNDS:
        raise ValueError("sound %r is not a sound: use one of %s." % (kind, ", ".join(SOUNDS)))
    at, level = _number("at", at, 0), _number("level", level)
    if not SOUND_LEVELS[0] <= level <= SOUND_LEVELS[1]:
        raise ValueError("level %s is not a sound's level: use %d to %d dB, 0 for the sound as it is made."
                         % (_f(level), SOUND_LEVELS[0], SOUND_LEVELS[1]))
    seconds, expression, after = SOUNDS[kind]
    return Fragment("aevalsrc=exprs='%s':s=%d:c=stereo:d=%s%s%s,adelay=delays=%dS:all=1[_s];"
                    "[in][_s]amix=inputs=2:duration=first:normalize=0[out]"
                    % (expression, RATE, _f(seconds), after, ",volume=%sdB" % _f(level) if level else "",
                       round(at * RATE)), [])


def duck(spans, db=-12):
    """The sound lowered by db (below 0) while someone speaks, spans [(start, end)] in clip seconds: music under a
    voice. It goes down along a half cosine over the 0.25 s before each span, so the first word is clear, and back
    up over the 0.5 s after it; spans less than 1.5 s apart are one, so it does not rise for a breath. The level
    follows t in steps of 1 ms, too small to click. No spans leaves the sound as it was."""
    db = _number("db", db)
    if not -40 <= db <= -1:
        raise ValueError("db %s is not a duck: use -40 to -1 dB, such as -12." % _f(db))
    if not isinstance(spans, (list, tuple)):
        raise ValueError("spans %r is not a list: give [(start, end)] in seconds of the clip." % (spans,))
    windows = []
    for k, span in enumerate(spans):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError("span %d %r is not (start, end) in seconds of the clip." % (k, span))
        windows.append(_window(*span))
    merged = []
    for start, end in sorted(windows):
        if merged and start - merged[-1][1] < DUCK_BRIDGE:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if not merged:
        return Fragment("[in]anull[out]", [])
    down = "+".join("if(between(t,%s,%s),if(lt(t,%s),(1-cos(PI*(t-%s)/%s))/2,if(gt(t,%s),(1+cos(PI*(t-%s)/%s))/2,1)),0)"
                    % (_f(a - DUCK_DOWN), _f(b + DUCK_UP), _f(a), _f(a - DUCK_DOWN), _f(DUCK_DOWN), _f(b), _f(b),
                       _f(DUCK_UP)) for a, b in merged)
    return Fragment("[in]asetnsamples=n=%d:p=0,volume=volume='1-%s*(%s)':eval=frame[out]"
                    % (RATE // 1000, _f(1 - 10 ** (db / 20)), down), [])
