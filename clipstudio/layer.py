"""The clip's words as a layer the page lays over the video, so a change to the words shows without making the video
again (lane 5, 10-28).

words() draws what the export burns in from the same ASS file (render.ass) with the same libass, at the page's size,
as a few transparent pictures and a time index [{"from", "to", "file"}] in clip seconds: the words on screen do not
change between "from" and "to". Where libass may change what it draws (an event starts or ends, a transform, a fade,
a karaoke word) a new step starts; each step is drawn at its last frame, so a word that pops, fades or lights shows as
a step rather than a smooth move.

A picture is exact, not near: libass draws each step twice, over black and over white, and the difference of the two
is how much of the ground shows through (alpha); the colour is what is left over black. Laid over any ground with
plain alpha that gives what libass draws over that ground, to a level or two of rounding (libass rounds each of its
layers, so a blurred glow's edge can be up to 5 levels off on a few pixels), glows and soft shadows included
(libass's own alpha drawing was measured 44 to 70 levels off on their edges).

A picture is named by what it shows (the ASS styles, and each event on screen with its time into the event and to
its end), so a second call with nothing changed draws nothing, and a fix to one caption draws only the steps where
that caption shows.
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile

from . import effects, render, text

SIZE = (540, 960)
FPS = render.FPS
LAYERS = 1
FULL = (1080, 1920)
DRAW_SECONDS = 300
CLOCK = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d\d)")
DIALOGUE = re.compile(r"^Dialogue: (\d+),([^,]+),([^,]+),(.*)$")
TRANSFORM = re.compile(r"\\t\((-?\d+),(-?\d+),")
FADE = re.compile(r"\\fad\((\d+),(\d+)\)")
KARAOKE = re.compile(r"\\k[fo]?(\d+)")
# straight colour from the picture over black (x) and over white (y): x * 255 / alpha, alpha = 255 - (y - x)
COLOUR = "if(lt(255-(y-x),1),0,min(255,x*255/(255-(y-x))))"
ALPHA = "max(0,min(255,255-(y-x)))"
PRORES = ["-c:v", "prores_ks", "-profile:v", "4444", "-qscale:v", "4", "-alpha_bits", "8", "-pix_fmt", "yuva444p10le", "-vendor", "apl0", "-an"]


def _seconds(clock):
    h, m, s, cs = (int(v) for v in CLOCK.fullmatch(clock).groups())
    return ((h * 60 + m) * 60 + s) + cs / 100


def events(ass_text):
    """The ASS file's events in file order: [(start, end, line)], line the Dialogue line itself."""
    out = []
    for line in ass_text.splitlines():
        m = DIALOGUE.match(line)
        if m:
            out.append((_seconds(m.group(2)), _seconds(m.group(3)), line))
    return out


def changes(start, end, line, smooth=False):
    """The clip seconds inside an event where what it draws may change: its start and end, the edges of its \\t
    transforms and \\fad fades, and the edges of its karaoke words (\\k, \\kf, \\ko, in order). smooth adds every
    frame of a transform or a fade, so it moves as the export draws it."""
    out = {start, end}
    spans = [(start + int(a) / 1000, start + int(b) / 1000) for a, b in TRANSFORM.findall(line)]
    for a, b in FADE.findall(line):
        spans += [(start, start + int(a) / 1000), (end - int(b) / 1000, end)]
    for a, b in spans:
        out |= {a, b}
        if smooth:
            out |= {n / FPS for n in range(math.ceil(a * FPS), math.ceil(b * FPS))}
    at = start
    for cs in KARAOKE.findall(line):
        at += int(cs) / 100
        out.add(at)
    return {t for t in out if start <= t <= end}


def steps(ass_text, length, smooth=False):
    """[(from, to, frame)]: the stretches of the clip (0 to length s) in which no event changes what it draws, each
    with the frame number it is drawn at (its last frame), only stretches holding a frame and some event."""
    evs = events(ass_text)
    cuts = sorted({0.0, float(length)} | {t for s, e, line in evs for t in changes(s, e, line, smooth) if 0 < t < length})
    out = []
    for a, b in zip(cuts, cuts[1:]):
        first, last = math.ceil(a * FPS - 1e-6), math.ceil(b * FPS - 1e-6) - 1
        if last < first or not any(s <= last / FPS < e for s, e, _ in evs):
            continue
        out.append((round(a, 3), round(b, 3), last))
    return out


def _moves(line):
    return bool(TRANSFORM.search(line) or FADE.search(line) or KARAOKE.search(line))


def _key(head, evs, frame, size, fonts):
    """The name of what libass draws at frame: the styles, the fonts, the size and every event on screen then, with
    how far into it and how far from its end the frame is when it moves (its animations are timed from those); an
    event that holds still is named by what it draws alone, so the same words at another time are the same picture."""
    t = frame / FPS
    shown = [(line.split(",", 3)[3],) + ((round(t - s, 3), round(e - t, 3)) if _moves(line) else ())
             for s, e, line in evs if s <= t < e]
    return hashlib.sha1(json.dumps([LAYERS, list(size), head, fonts, shown], ensure_ascii=False)
                        .encode("utf-8")).hexdigest()[:20]


def _any(terms):
    """terms joined by + as a balanced tree: FFmpeg refuses a flat sum of more than about a hundred (measured: 90
    parse, 110 do not), and a tree of a thousand parses."""
    if len(terms) == 1:
        return terms[0]
    half = len(terms) // 2
    return "(%s+%s)" % (_any(terms[:half]), _any(terms[half:]))


def _draw(path, fonts, frames, size, names, folder):
    """Draws frames (frame numbers) of the ASS file at path into folder as names[i].png, each a transparent picture
    of what libass draws then, in one FFmpeg run."""
    w, h = size
    pick = _any(["eq(n\\,%d)" % n for n in frames])
    seconds = (max(frames) + 2) / FPS
    words = text._ass_filter(path, fonts)
    grounds = ";".join("color=c=%s:s=%dx%d:r=%d:d=%.3f,format=rgb24,select='%s',%s,format=gbrp[%s]"
                       % (c, w, h, FPS, seconds, pick, words, label) for c, label in (("black", "b"), ("white", "w")))
    # each colour divided by its own channel's alpha, the picture's alpha from green: measured closer on a glow's
    # edge than every colour divided by green's alpha (libass rounds each of its layers, so the channels differ)
    graph = ("%s;[b]split[b1][b2];[w]split[w1][w2];[b1][w1]lut2=c0='%s':c1='%s':c2='%s'[c];"
             "[b2]extractplanes=g[bg];[w2]extractplanes=g[wg];[bg][wg]lut2=c0='%s'[a];"
             "[c][a]alphamerge,format=rgba[o]" % (grounds, COLOUR, COLOUR, COLOUR, ALPHA))
    work = tempfile.mkdtemp(prefix="layer.", dir=folder)
    try:
        done = subprocess.run([render._ffmpeg(), "-v", "error", "-y", "-filter_complex", graph, "-map", "[o]",
                               "-fps_mode", "passthrough", "-f", "image2", os.path.join(work, "%05d.png")],
                              capture_output=True, text=True, errors="replace", timeout=DRAW_SECONDS)
        made = sorted(os.listdir(work))
        if done.returncode != 0 or len(made) != len(frames):
            raise RuntimeError("FFmpeg could not draw the words layer: %s" % (
                (done.stderr.strip().splitlines() or ["%d of %d pictures" % (len(made), len(frames))])[-1]))
        for name, got in zip(names, made):
            render.settle(os.path.join(work, got), os.path.join(folder, name + ".png"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def words(state, default="clean", size=SIZE, part=None, smooth=False):
    """The words layer of the clip: [{"from", "to", "file"}] in clip seconds, from 0 to the clip's length, file a
    transparent PNG of size (w, h) in work/layers with the titles, big words and captions as the export burns them
    in at that size (the words behind the person are not in it: they are part of the picture). [] when the clip has
    no words on screen. Steps that show the same picture are one entry. part as for render.ass, smooth as for changes.
    RuntimeError when FFmpeg cannot draw."""
    path = render.ass(state, default, part=part)
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        ass_text = f.read()
    head = ass_text.split("[Events]")[0]
    fonts = render.fonts(state, default)
    evs = events(ass_text)
    folder = render.folder("layers")
    index, todo = [], {}
    for a, b, frame in steps(ass_text, state["length"], smooth):
        name = _key(head, evs, frame, size, fonts)
        if not os.path.isfile(os.path.join(folder, name + ".png")):
            todo.setdefault(name, frame)
        file = os.path.join(folder, name + ".png")
        if index and index[-1]["file"] == file and abs(index[-1]["to"] - a) < 1e-6:
            index[-1]["to"] = b
        else:
            index.append({"from": a, "to": b, "file": file})
    if todo:
        names = sorted(todo, key=todo.get)
        _draw(path, fonts, [todo[n] for n in names], size, names, folder)
    return index


def _frame(seconds):
    """The first frame at or after seconds, which words() rounds to a thousandth."""
    return math.ceil(seconds * FPS - 0.02)


def _clear():
    """The path of a picture of nothing at the clip's full size."""
    clear = os.path.join(render.folder("plates"), "clear.png")
    if not os.path.isfile(clear):
        subprocess.run([render._ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black@0:s=%dx%d,format=rgba" % FULL,
                        "-frames:v", "1", "-update", "1", clear], capture_output=True, timeout=DRAW_SECONDS, check=True)
    return clear


def _steps_video(index, first, last):
    """The words of index (words()) from frame first up to frame last as a video with an alpha channel: its path in
    work/plates, made the first time. Each frame's picture is fed to FFmpeg in turn: FFmpeg's concat list stretched a
    step of one frame to 1/25 s, and on EP22 the captions ran 2.8 s late by the end of 39 s."""
    folder = render.folder("plates")
    key = hashlib.sha1(json.dumps([LAYERS, first, last, index], sort_keys=True).encode("utf-8")).hexdigest()[:16]
    target = os.path.join(folder, key + ".mov")
    if os.path.isfile(target):
        return target
    part = "%s.%d.part.mov" % (target[:-4], os.getpid())
    at, shows = first, []
    for step in index:
        a, b = max(first, _frame(step["from"])), min(last, _frame(step["to"]))
        if a > at:
            shows.append((_clear(), a - at))
            at = a
        if b > at:
            shows.append((step["file"], b - at))
            at = b
    if last > at:
        shows.append((_clear(), last - at))
    with tempfile.TemporaryFile() as said:
        feed = subprocess.Popen([render._ffmpeg(), "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png", "-i", "-"]
                                + PRORES + [part], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=said)
        try:
            with feed.stdin:
                for file, frames in shows:
                    with open(file, "rb") as f:
                        picture = f.read()
                    for _ in range(frames):
                        feed.stdin.write(picture)
        except OSError:
            pass
        feed.wait(DRAW_SECONDS)
        said.seek(0)
        fault = said.read().decode("utf-8", "replace").strip().splitlines()
    if feed.returncode != 0:
        if os.path.exists(part):
            os.remove(part)
        raise RuntimeError("FFmpeg could not make a layer of the clip: %s" % (fault or ["no reason"])[-1])
    render.settle(part, target)
    return target


def captions_video(state, default="clean"):
    """The clip's captions alone at the clip's full size, over nothing, as a video with an alpha channel for a track of
    their own in Premiere (ProRes 4444, which it reads): its path in work/plates, made the first time, or None when the
    clip has no captions. RuntimeError when FFmpeg cannot make it."""
    index = words(state, default, FULL, render.ONLY_CAPTIONS, smooth=True)
    return _steps_video(index, 0, _frame(state["length"])) if index else None


def _word_clip(state, default, kind, name, part):
    """A clip (clips()) of the words of part, from their first frame to their last, or None when they show in no frame."""
    index = words(state, default, FULL, part, smooth=True)
    if not index:
        return None
    first, last = _frame(index[0]["from"]), _frame(index[-1]["to"])
    return {"kind": kind, "name": name, "path": _steps_video(index, first, last), "at": first / FPS, "length": (last - first) / FPS}


def _alone(fragment, length):
    """The path of fragment (effects.py, timed from second 0) drawn over nothing for length seconds, as a video with an
    alpha channel in work/plates, made the first time."""
    graph, files = effects.join([fragment], "clear", "done", 0)
    folder = render.folder("plates")
    target = os.path.join(folder, hashlib.sha1(json.dumps([LAYERS, length, graph, files]).encode("utf-8")).hexdigest()[:16] + ".mov")
    if os.path.isfile(target):
        return target
    part = "%s.%d.part.mov" % (target[:-4], os.getpid())
    script = target[:-4] + ".graph"
    with open(script, "w", encoding="utf-8", newline="\n") as f:
        f.write("color=c=black@0:s=%dx%d:r=%d:d=%.3f,format=rgba[clear];%s" % (FULL + (FPS, length, graph)))
    try:
        cmd = [render._ffmpeg(), "-v", "error", "-y"]
        for file in files:
            cmd += render._picture_input(file)
        subprocess.run(cmd + ["-/filter_complex", script, "-map", "[done]", "-frames:v", str(_frame(length))] + PRORES + [part],
                       capture_output=True, text=True, errors="replace", timeout=DRAW_SECONDS, check=True)
        render.settle(part, target)
        return target
    except subprocess.CalledProcessError as e:
        raise RuntimeError("FFmpeg could not make a layer of the clip: %s" % ((e.stderr or "").strip().splitlines() or ["no reason"])[-1])
    finally:
        for left in (part, script):
            if os.path.exists(left):
                os.remove(left)


def _effect_clip(kind, name, at, length, fragment):
    return {"kind": kind, "name": name, "path": _alone(fragment, length), "at": at, "length": length}


def clips(state, default="clean", each=None):
    """All that sits over the picture and comes off it, each thing as a video of its own over nothing, as the plate
    leaves them out (render.after_source): [{"kind", "name", "path", "at", "length"}] from the bottom up, in clip
    seconds, kind one of "pictures", "stickers", "titles", "captions" and "flashes". each(done, total) is called before
    each one is made, and may raise to stop. RuntimeError when FFmpeg cannot make one."""
    todo = []
    for k in state.get("pictures") or []:
        if not render.under_glass(state, k):
            todo.append(lambda k=k: _effect_clip("pictures", "Picture", k["at"], k["length"],
                                                 effects.picture(k["png"], 0, k["length"], k["box"])))
    for k in state.get("stickers") or []:
        todo.append(lambda k=k: _effect_clip("stickers", "Sticker " + os.path.splitext(os.path.basename(k["png"]))[0], k["at"], k["length"],
                                             effects.sticker(k["png"], 0, k["length"], k["box"])))
    for f in state.get("flurries") or []:
        move = f.get("move", "burst")
        length = f.get("length") or effects.MOVES[move][2]
        todo.append(lambda f=f, move=move, length=length: _effect_clip(
            "stickers", "Flurry " + move, f["at"], length,
            effects.flurry(f["pngs"], 0, f["box"], move, f.get("count", 8), f.get("length"), f.get("size", 150), over="rgb")))
    for k, t in enumerate(state["titles"]):
        if not t.get("behind"):
            todo.append(lambda k=k, t=t: _word_clip(state, default, "titles", "Title: " + t["text"], ("title", k)))
    hits = state.get("hits") or []
    for k, h in enumerate(hits):
        if h["kind"] == "word" and not h.get("behind"):
            todo.append(lambda k=k, h=h: _word_clip(state, default, "titles", "Big word: " + h["word"], ("word", k)))
    if state.get("channel"):
        todo.append(lambda: _word_clip(state, default, "titles", "Channel name", render.ONLY_CHANNEL))
    if state.get("captions"):
        def captions():
            path = captions_video(state, default)
            return path and {"kind": "captions", "name": "Captions", "path": path, "at": 0.0, "length": _frame(state["length"]) / FPS}
        todo.append(captions)
    for h in hits:
        if h["kind"] == "flash":
            todo.append(lambda h=h: _effect_clip("flashes", "Flash", h["at"], effects.FLASH_LENGTH, effects.flash_alone()))
    out = []
    for done, make in enumerate(todo):
        if each:
            each(done, len(todo))
        out.append(make())
    return [c for c in out if c]
