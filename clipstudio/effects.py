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
import re

Fragment = collections.namedtuple("Fragment", "graph files")

LABEL = re.compile(r"\[(in|out|_[a-z0-9]+|f[0-9]+)\]")
FRAME_W, FRAME_H, FPS = 1080, 1920, 30
MOST_GAP = 0.5
STILL = 0.5
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


def _centre(centre):
    if (not isinstance(centre, (tuple, list)) or len(centre) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in centre)):
        raise ValueError("centre %r is not a point: give (x, y) on the %dx%d frame." % (centre, FRAME_W, FRAME_H))
    cx, cy = centre
    if not (0 <= cx <= FRAME_W and 0 <= cy <= FRAME_H):
        raise ValueError("centre %r is off the %dx%d frame: use x from 0 to %d and y from 0 to %d."
                         % (tuple(centre), FRAME_W, FRAME_H, FRAME_W, FRAME_H))
    return cx, cy
