"""Effects: the pieces of FFmpeg filter graph that draw on the clip, each one exactly what its numbers say.

Every piece returns a Fragment: filter graph text with exactly one [in] and one [out], its own inner labels written
[_a], [_b]..., and any extra file it needs (a sticker picture) as [f0], [f1]... in the order of its files. join()
chains the fragments of one stage, so the renderer never has to know what is inside them.

Times are seconds of the clip (0 is its first frame). An expression uses only t, the frame's time: never n, never
random(), and never a filter that remembers earlier frames, so a frame drawn on its own (the preview starts the graph
at that frame's time) is the same frame a full render draws.
"""
import collections
import math
import os
import re
import struct

Fragment = collections.namedtuple("Fragment", "graph files")

LABEL = re.compile(r"\[(in|out|_[a-z0-9]+|f[0-9]+)\]")
FRAME_W, FRAME_H, FPS = 1080, 1920, 30
MOST_GAP = 0.5
STILL = 0.5
POP = ((0.0, 0.6), (0.15, 1.1), (0.25, 1.0))
FADE = 0.15
PNG_SIGNATURE = bytes((137, 80, 78, 71, 13, 10, 26, 10))
BLUR = "boxblur=luma_radius='max(1,min(w,h)/6)':luma_power=3:chroma_radius='max(1,min(cw,ch)/6)':chroma_power=3"


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
        raise ValueError("%s %r is not a number: give it in %s." % (name, value, "seconds of the clip" if name in (
            "start", "end", "at", "length") else "pixels"))
    if low is not None and value < low:
        raise ValueError("%s %s is below %s: use %s from %s." % (name, _f(value), _f(low), name, _f(low)))
    return float(value)


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
    margin = _number("margin", margin, 0)
    if margin > 1:
        raise ValueError("margin %s is more than 1: use a share of the box from 0 to 1, such as 0.15." % margin)
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
        if track and t - track[-1][0] > MOST_GAP:
            raise ValueError("points %d and %d are %.2f s apart (%s to %s s), more than %s s: split the track at "
                             "the gap." % (k - 1, k, t - track[-1][0], _f(track[-1][0]), _f(t), MOST_GAP))
        track.append((t, x + w / 2, y + h / 2, w, h))
    w = 2 * math.ceil(max(p[3] for p in track) * (1 + margin) / 2)
    h = 2 * math.ceil(max(p[4] for p in track) * (1 + margin) / 2)
    path = _simplify([(t, cx, cy) for t, cx, cy, _w, _h in track])
    cx, cy = _along(path, 1), _along(path, 2)
    left = "clip(2*floor((%s-%s)/2),0,%s)" % (cx, _f(w / 2), "%s")
    top = "clip(2*floor((%s-%s)/2),0,%s)" % (cy, _f(h / 2), "%s")
    when = "gte(t,%s)*lte(t,%s)" % (_f(track[0][0]), _f(track[-1][0]))
    return Fragment(_blurred(w, h, left % "iw-ow", top % "ih-oh", left % "W-w", top % "H-h", when), [])


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
    return "(%s)" % "+".join(terms)


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


def flash(at, length=0.15, strength=0.8):
    """A white flash at `at`: the frame jumps brighter by strength (0 to 1) and fades back over length seconds.
    Outside the window every frame is the input's own."""
    at, length, when = _hit(at, length)
    strength = _number("strength", strength)
    if not 0 < strength <= 1:
        raise ValueError("strength %s is not a flash strength: use a share from above 0 to 1, such as 0.8."
                         % _f(strength))
    fade = "(%s*(1-(t-%s)/%s))" % (_f(strength), _f(at), _f(length))
    return Fragment("[in]eq=brightness='%s':saturation='1-%s':eval=frame:enable='%s'[out]" % (fade, fade, when), [])


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
    u = "(t-%s)" % _f(at)
    (t0, s0), (t1, s1), (t2, s2) = POP
    grow = "if(lt(%s,%s),%s,if(lt(%s,%s),%s+%s*(%s-%s)/%s,if(lt(%s,%s),%s+%s*(%s-%s)/%s,%s)))" % (
        u, _f(t0), _f(s0), u, _f(t1), _f(s0), _f(s1 - s0), u, _f(t0), _f(t1 - t0),
        u, _f(t2), _f(s1), _f(s2 - s1), u, _f(t1), _f(t2 - t1), _f(s2))
    when = "gte(t,%s)*lt(t,%s)" % (_f(at), _f(at + length))
    return Fragment("[f0]format=rgba,scale=w='max(2,trunc(%s*%s))':h='max(2,trunc(%s*%s))':eval=frame,"
                    "fade=t=out:st=%s:d=%s:alpha=1[_s];"
                    "[in][_s]overlay=x='%s-w/2':y='%s-h/2':eval=frame:shortest=1:format=auto:enable='%s'[out]"
                    % (_f(base_w), grow, _f(base_h), grow, _f(at + length - FADE), _f(FADE),
                       _f(x + w / 2), _f(y + h / 2), when), [png])


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
