"""The person cut out of the footage, for the effects that need to know which pixels are a person: the footage in
black and white around them, words behind them, a white edge around them.

The person's mask is made by the cut_out tool and kept as state['mask']: {"path", "person", "fps", "size", "from",
"to"}, an FFV1 file of grey frames on the recording's pixel grid (255 the person, 0 not, soft edges between), frame k
at recording second from + k / fps. frame_mask() maps it once onto the clip, through the same pieces, the same layout
and the same slow pushes as the picture, so each effect reads one grey frame per clip frame, at the frame's size,
lined up with the person as the clip shows them.

Each effect is a Fragment (effects.join) with the frame mask as its file [f0]: the renderer opens that file as a
video from clip second 0 (from the preview frame's 1/30 s step for a preview), not as a looped picture. Outside an
effect's start to end every pixel is the frame's own. The background (backdrop) has its picture as [f1] and holds the
whole clip. The renderer draws them before the camera hits, so the person,
the words behind them and their edge move together with a shake or a punch-in.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

from . import effects, layouts, recording

W, H, FPS = effects.FRAME_W, effects.FRAME_H, effects.FPS
FRAME_MASKS = 2
PUSHED = ("card", "whole", "bottom")
SHAPED = {"card": lambda w, h: layouts.rounded(w, h, layouts.CARD_RADIUS),
          "circle": lambda w, h: layouts.rounded(w, h, w / 2)}
EASE = 0.2
OUTLINE = (8, 2, 30)
MAKE_SECONDS = 600


def _f(v):
    return effects._f(v)


def _checked(mask, size):
    """(path, fps, start, end) of a mask entry for a recording of size, or ValueError naming what is wrong."""
    if not isinstance(mask, dict):
        raise ValueError("mask %r is not a person's mask: run cut_out first, which keeps one as state['mask']."
                         % (mask,))
    path = mask.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise ValueError("mask path %r is not a file: run cut_out again to make the person's mask." % (path,))
    fps = mask.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not 1 <= fps <= 240:
        raise ValueError("mask fps %r is not a frame rate: use the recording's frames per second, 1 to 240." % (fps,))
    if list(mask.get("size") or ()) != list(size):
        raise ValueError("mask size %r is not the recording's size %r: the mask must be on the recording's pixel grid."
                         % (mask.get("size"), list(size)))
    start, end = mask.get("from"), mask.get("to")
    if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (start, end))
            or start < 0 or end <= start):
        raise ValueError("mask from %r to %r is not a span: give recording seconds, from 0 and to after from."
                         % (start, end))
    return path, float(fps), float(start), float(end)


def frame_mask(mask, pieces, layout, length, pushes=()):
    """The path of the person's mask on the clip, made once in work/masks: an FFV1 file of grey 1080x1920 frames at
    30 fps, frame n at clip second n / 30, length seconds long, white where the clip shows the person. Each part of
    the layout maps its piece of the recording to its place on the frame as the picture does (the card's corners and
    the face circle cut the same way, a following part moving as the picture's does, the main part pushed in by the
    same slow pushes, a talk layout's shots cut
    between its people as the picture's are); the frame's design shows no person. Where the clip shows a moment the
    mask holds no frame for, it is black."""
    path, fps, start, end = _checked(mask, layout["size"])
    seen = os.stat(path)
    key = hashlib.sha1(json.dumps([FRAME_MASKS, os.path.abspath(path), seen.st_size, seen.st_mtime_ns, fps, start, end,
                                   pieces, layout["kind"], layout["parts"], layout.get("shots"), list(pushes),
                                   round(length, 3)],
                                  sort_keys=True).encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(recording.WORK, "masks")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, "frame_%s.mkv" % key)
    if os.path.isfile(target):
        return target
    graph, files = _graph(pieces, layout, length, pushes, start, end)
    cmd = ["-i", path]
    for f in files:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-i", f]
    return _make(cmd, graph, length, target)


def speaking_mask(mask, pieces, layout, length, pushes=(), told_apart=False):
    """The path of the mask of whoever speaks, on the clip as frame_mask() lays one person's (made once in
    work/masks): each person's own mask (mask["each"], from cut_out on a talk) white only while they speak
    (talk.speaking, from the layout's shots), all of them together, so the grey look keeps the speaker's colour and
    everyone's when they talk together. Tarm 10-30 on four people talking: one person kept in colour through the
    whole clip "มันดูตลก". frame_mask()'s one person when the mask has no one's own or the layout no shots.
    Only with told_apart, the clip's people told apart by their voices. Mouths alone get it wrong, and colour on the
    wrong person shows: Tarm 10-31 on four men in EP0, it stayed on the man in glasses while another spoke. So without
    it, everyone SAM followed keeps their colour (mask["everyone"]) and only the room turns."""
    from . import talk
    if mask.get("everyone") and not told_apart:
        return frame_mask(dict(mask, path=mask["everyone"]), pieces, layout, length, pushes)
    each = mask.get("each") or []
    if layout["kind"] != "talk" or not layout.get("shots") or not any(each):
        return frame_mask(mask, pieces, layout, length, pushes)
    turns = talk.speaking(layout["shots"], len(each))
    laid, gates = [], []
    for n, path in enumerate(each, 1):
        spoken = ["between(t,%s,%s)" % (_f(t["from"]), _f(t["to"])) for t in turns if n in t["who"]]
        if path and spoken:
            laid.append(frame_mask(dict(mask, path=path), pieces, layout, length, pushes))
            gates.append("+".join(spoken))
    folder = os.path.join(recording.WORK, "masks")
    key = hashlib.sha1(json.dumps([FRAME_MASKS, laid, gates, round(length, 3)]).encode("utf-8")).hexdigest()[:16]
    target = os.path.join(folder, "speak_%s.mkv" % key)
    if os.path.isfile(target):
        return target
    if not laid:
        return _make(["-f", "lavfi", "-i", "color=c=black:s=%dx%d:r=%d:d=%s" % (W, H, FPS, _f(length))],
                     "[0:v]format=gray[o]", length, target)
    chains, cmd = [], []
    for k, (path, gate) in enumerate(zip(laid, gates)):
        cmd += ["-i", path]
        chains.append("[%d:v]format=gray,lut=c0=0:enable='not(%s)'[g%d]" % (k, gate, k))
    last = "g0"
    for k in range(1, len(laid)):
        chains.append("[%s][g%d]blend=all_mode=lighten:shortest=1[l%d]" % (last, k, k))
        last = "l%d" % k
    chains.append("[%s]format=gray[o]" % last)
    return _make(cmd, ";".join(chains), length, target)


def _make(inputs, graph, length, target):
    """target, an FFV1 file of grey frames at FPS, length seconds of graph's [o] from inputs (FFmpeg's input options),
    written to a part file and moved in."""
    folder = os.path.dirname(target)
    os.makedirs(folder, exist_ok=True)
    fd, part = tempfile.mkstemp(prefix=os.path.basename(target) + ".", suffix=".part.mkv", dir=folder)
    os.close(fd)
    try:
        done = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y"] + inputs
                              + ["-filter_complex", graph, "-map", "[o]", "-t", "%.3f" % length, "-r", str(FPS),
                                 "-c:v", "ffv1", "-pix_fmt", "gray", "-f", "matroska", part],
                              capture_output=True, text=True, errors="replace", timeout=MAKE_SECONDS)
    except (subprocess.SubprocessError, OSError) as e:
        os.remove(part)
        raise RuntimeError("FFmpeg could not lay the person's mask on the clip (%s)." % e)
    if done.returncode != 0 or not os.path.getsize(part):
        os.remove(part)
        raise RuntimeError("FFmpeg could not lay the person's mask on the clip: %s"
                           % (done.stderr.strip().splitlines() or ["no reason"])[-1])
    try:
        os.replace(part, target)
    except PermissionError:
        # made by another request at the same moment and open now (Windows): the same file, so keep it
        os.remove(part)
        if not os.path.isfile(target):
            raise
    return target


def _graph(pieces, layout, length, pushes, start, end):
    """(graph text from the mask, input 0, to [o], the shape pictures it opens from input 1 on)."""
    w, h = layout["size"]
    spans = [(a, b, max(a, start), min(b, end)) for a, b in pieces]
    used = [k for k, (_a, _b, lo, hi) in enumerate(spans) if hi > lo]
    chains = []
    if used:
        heads = "".join("[m%d]" % k for k in used)
        chains.append("[0:v]format=gray,setpts=PTS-STARTPTS%s" % (",split=%d%s" % (len(used), heads)
                                                                  if len(used) > 1 else heads))
    for k, (a, b, lo, hi) in enumerate(spans):
        if hi > lo:
            chains.append("[m%d]trim=start=%s:end=%s,setpts=PTS-STARTPTS,fps=%d,tpad=start_duration=%s:"
                          "stop_duration=%s:color=black[p%d]" % (k, _f(lo - start), _f(hi - start), FPS, _f(lo - a),
                                                                 _f(b - hi), k))
        else:
            chains.append("color=c=black:s=%dx%d:r=%d:d=%s,format=gray[p%d]" % (w, h, FPS, _f(b - a), k))
    chains.append("%sconcat=n=%d:v=1:a=0[cat]" % ("".join("[p%d]" % k for k in range(len(spans))), len(spans)))
    if layout["kind"] == "talk":
        talked, files = _talk(layout)
        return ";".join(chains + talked), files
    parts = layout["parts"]
    chains.append("[cat]%s" % ("split=%d%s" % (len(parts), "".join("[c%d]" % j for j in range(len(parts))))
                              if len(parts) > 1 else "null[c0]"))
    files = []
    for j, part in enumerate(parts):
        (sx, sy, sw, sh), (dx, dy, dw, dh) = part["src"], part["dst"]
        if part.get("follow"):
            from . import focus
            drawn = "[c%d]crop=w=%d:h=%d:x='%s':y='%s',scale=%d:%d" % (
                j, sw, sh, focus.expression(part["follow"], "x"), focus.expression(part["follow"], "y"), dw, dh)
        else:
            drawn = "[c%d]crop=%d:%d:%d:%d,scale=%d:%d" % (j, sw, sh, sx, sy, dw, dh)
        if part["name"] in PUSHED:
            for p in pushes:
                cx, cy = effects._centre(p.get("centre", (W / 2, H / 2)))
                inside = (min(max(cx - dx, 0), dw), min(max(cy - dy, 0), dh))
                drawn += "," + effects.push_filters(p["start"], p["end"], p.get("zoom", 1.08), inside, (dw, dh))
        if part["name"] in SHAPED:
            files.append(SHAPED[part["name"]](dw, dh))
            drawn += "[s%d];[%d:v]format=gray[r%d];[s%d][r%d]blend=all_mode=multiply:shortest=1" % (
                j, len(files), j, j, j)
        chains.append(drawn + ",pad=%d:%d:%d:%d:color=black,format=gray[q%d]" % (W, H, dx, dy, j))
    last = "q0"
    for j in range(1, len(parts)):
        chains.append("[%s][q%d]blend=all_mode=lighten:shortest=1[l%d]" % (last, j, j))
        last = "l%d" % j
    chains.append("[%s]format=gray[o]" % last)
    return ";".join(chains), files


def _talk(layout):
    """(the chains from [cat] to [o] for a talk layout, the shape pictures they open from input 1 on), cut as
    talk.graph cuts the picture: in a person's shot their crop fills the frame (a shot of everyone holds the crop
    before it), and in a shot of everyone the frame shows no person but the card, its corners rounded, shows them
    all. The cuts switch on t alone, as the picture's do."""
    from . import talk
    parts = {p["name"]: p for p in layout["parts"]}
    shots = layout["shots"]
    held, filled = parts["person 1"]["src"], []
    for s in shots:
        held = talk.crop_of(parts, s) if s["show"] else held
        filled.append(held)
    one = "crop=w=%d:h=%d:x='%s':y='%s',scale=%d:%d,format=gray" % (
        filled[0][2], filled[0][3], talk._switch(shots, [c[0] for c in filled], filled[-1][0]),
        talk._switch(shots, [c[1] for c in filled], filled[-1][1]), W, H)
    together = [s for s in shots if s["show"] == 0]
    if not together:
        return ["[cat]%s[o]" % one], []
    (sx, sy, sw, sh), (cx, cy, cw, ch) = parts["everyone"]["src"], parts["everyone"]["dst"]
    when = "+".join("between(T,%s,%s)" % (_f(s["from"]), _f(s["to"] - 0.001)) for s in together)
    return (["[cat]split[t_a][t_c]", "[t_a]%s[t_one]" % one,
             "[t_c]crop=%d:%d:%d:%d,scale=%d:%d,format=gray[t_s];[1:v]format=gray[t_r];"
             "[t_s][t_r]blend=all_mode=multiply:shortest=1,pad=%d:%d:%d:%d:color=black,format=gray[t_all]"
             % (sw, sh, sx, sy, cw, ch, W, H, cx, cy),
             "[t_one][t_all]blend=all_expr='if(%s,B,A)':shortest=1,format=gray[o]" % when],
            [layouts.rounded(cw, ch, layouts.CARD_RADIUS)])


def _window(start, end):
    start, end = effects._window(start, end)
    if end - start < 2 * EASE:
        raise ValueError("start %s to end %s is too short for a cut-out: give it at least %s s, so it can ease in "
                         "and out." % (_f(start), _f(end), _f(2 * EASE)))
    return start, end


def _person(mask, label):
    """Graph text: the frame [label] with the person's mask [f0] as its alpha, to [_p]."""
    return "[%s]format=yuva420p[_c];[f0]format=gray[_m];[_c][_m]alphamerge[_p]" % label


def backdrop(mask, png, despill=False):
    """The background png (a 1080x1920 picture) wherever the clip does not show the person, through the whole clip.
    mask is frame_mask()'s path. despill takes the cloth's green off the person's edges and hair, for a mask cut off
    a green cloth."""
    clean = "despill=type=green," if despill else ""
    return effects.Fragment("[in]%sformat=yuva420p[_c];[f0]format=gray[_m];[_c][_m]alphamerge[_p];"
                            "[f1]scale=%d:%d,format=yuv420p[_g];[_g][_p]overlay=0:0:shortest=1:format=yuv420[out]"
                            % (clean, W, H), [mask, png])


def grey(mask, start, end):
    """From start to end the footage turns black and white around the person, who keeps their colour: the colour
    drains away over EASE s and comes back over the last EASE s. mask is frame_mask()'s path."""
    start, end = _window(start, end)
    ease = "min(1,min(t-%s,%s-t)/%s)" % (_f(start), _f(end), _f(EASE))
    when = "between(t,%s,%s)" % (_f(start), _f(end))
    return effects.Fragment("[in]split[_a][_b];[_a]hue=s='1-%s':enable='%s'[_g];%s;"
                            "[_g][_p]overlay=0:0:shortest=1:format=yuv420:enable='%s'[out]"
                            % (ease, when, _person(mask, "_b"), when), [mask])


def outline(mask, start, end, width=OUTLINE[0]):
    """From start to end a white edge, about width px wide, runs around the person, fading in and out over EASE s.
    mask is frame_mask()'s path."""
    start, end = _window(start, end)
    width = effects._number("width", width)
    if not OUTLINE[1] <= width <= OUTLINE[2]:
        raise ValueError("width %s is not an outline's width: use %d to %d px, such as %d."
                         % ((_f(width),) + OUTLINE[1:] + OUTLINE[:1]))
    when = "between(t,%s,%s)" % (_f(start), _f(end))
    return effects.Fragment(
        "[in]split[_a][_b];[_b]lutyuv=y=235:u=128:v=128,format=yuva420p[_w];[f0]format=gray,split[_m][_n];"
        "[_m]gblur=sigma=%s,lut=y='255*gt(val,12)'[_d];[_d][_n]blend=all_mode=subtract:shortest=1[_e];"
        "[_w][_e]alphamerge,fade=t=in:st=%s:d=%s:alpha=1,fade=t=out:st=%s:d=%s:alpha=1[_r];"
        "[_a][_r]overlay=0:0:shortest=1:format=yuv420:enable='%s'[out]"
        % (_f(width / 2), _f(start), _f(EASE), _f(end - EASE), _f(EASE), when), [mask])


def behind(words, mask, windows):
    """The words of words (the filter text that draws an ASS file, text._ass_filter) drawn behind the person: the
    words go on the frame, then the person, cut out by mask (frame_mask()'s path), back over them, during windows
    [(start, end)] (when those words show) only, so every other frame is the frame's own."""
    if not windows:
        raise ValueError("windows %r has no times: give [(start, end)] when the words behind the person show."
                         % (windows,))
    when = "+".join("between(t,%s,%s)" % tuple(_f(v) for v in effects._window(a, b)) for a, b in windows)
    return effects.Fragment("[in]split[_a][_b];[_a]%s[_w];%s;[_w][_p]overlay=0:0:shortest=1:format=yuv420:"
                            "enable='%s'[out]" % (words, _person(mask, "_b"), when), [mask])
