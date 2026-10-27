"""The clip drawn from its project list: one FFmpeg filter graph from the recording to the 1080x1920 frame.

    recording piece(s)  ->  [src] at clip time t  ->  source stage (blurs)  ->  layout (layouts.py)
                        ->  frame stage (camera hits, then the words, then flashes)

The preview frame is the same graph started at the frame's time: the piece is opened at the matching recording
second and its timestamps are set so t is still clip time, so a title from 2 s to 5 s shows in the preview at 3 s
exactly as it will in the finished clip.
"""
import hashlib
import json
import os
import shutil
import subprocess

from . import effects, layouts, project, recording, text

PREVIEW_WIDTH = 540
FRAME_SECONDS = 60
WORD_SECONDS = 0.8
HIT_KINDS = ("flash", "shake", "punch", "word")


class NotDrawn(Exception):
    """The frame could not be drawn. The sentence says why."""


class Stopped(Exception):
    """The person pressed Stop while the clip was being made: export() raises it and leaves no file."""


def folder(name):
    path = os.path.join(recording.WORK, name)
    os.makedirs(path, exist_ok=True)
    return path


def style_of(state, default="clean"):
    """The text style the clip uses: the last style entry, else the house style the recipe guessed."""
    e = state["style"]
    if e:
        return text.style(e["name"], e.get("font"), e.get("colour"))
    return text.STYLES.get(default) or text.STYLES["clean"]


def _hits(state):
    hits = state.get("hits") or []
    for h in hits:
        if h.get("kind") not in HIT_KINDS:
            raise ValueError("hit kind %r is not a hit: use one of %s." % (h.get("kind"), ", ".join(HIT_KINDS)))
    return hits


def ass(state, default="clean"):
    """The path of the clip's ASS file (written once per content) with its titles, captions and big words, or None
    when there are no words on screen."""
    captions = state.get("captions")
    words = [h for h in _hits(state) if h["kind"] == "word"]
    if not (state["titles"] or captions or words):
        return None
    st = style_of(state, default)
    events = []
    for t in state["titles"]:
        events += text.title(t["text"], t["start"], t["end"], t["box"], st, t.get("key"))
    if captions:
        events += text.captions(captions["lines"], st, captions["band"])
    for h in words:
        events += text.big_word(h["word"], h["at"], WORD_SECONDS, h["centre"], st)
    body = text.document([st], events)
    path = os.path.join(folder("ass"), hashlib.sha1(body.encode("utf-8")).hexdigest()[:16] + ".ass")
    if not os.path.isfile(path):
        with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        os.replace(path + ".part", path)
    return path


def after_source(state, src, out, first_input, default="clean"):
    """(graph text from [src] to [out], extra files) in three stages:
    - source stage, on the recording's picture: the blurs, so a blurred box stays blurred wherever the layout shows
      it, the card's blurred background included;
    - the layout, to the 1080x1920 frame;
    - frame stage: the camera hits (shake, punch-in), then the words (titles, captions and big words, one ASS file),
      so the words hold still while the picture moves, then the flashes last, so a flash lights the words too.
    A drawing that cannot be made raises ValueError with the sentence for Muse."""
    blurs = [effects.blur_box(b["box"], b["start"], b["end"]) for b in state.get("blurs") or []]
    seen, files = effects.join(blurs, src, "seen", first_input)
    lay, more = layouts.graph(state["clip"]["layout"], "seen", "drawn", first_input + len(files))
    files = files + more
    hits = _hits(state)
    camera = {"shake": lambda h: effects.shake(h["at"]),
              "punch": lambda h: effects.punch_in(h["at"], centre=h["centre"])}
    fragments = [camera[h["kind"]](h) for h in hits if h["kind"] in camera]
    words = ass(state, default)
    if words:
        fragments.append(text.burn(words))
    fragments += [effects.flash(h["at"]) for h in hits if h["kind"] == "flash"]
    stage, more = effects.join(fragments, "drawn", out, first_input + len(files))
    return ";".join((seen, lay, stage)), files + more


def frame(rec, state, t, default="clean", width=PREVIEW_WIDTH):
    """The clip's frame at clip second t, width pixels wide: the jpg's path in work/clipframes, made once."""
    clip = state["clip"]
    if not clip:
        raise NotDrawn("There is no clip yet, so there is no frame to draw.")
    at = project.to_clip(clip["pieces"], t)
    if at is None:
        raise NotDrawn("%.2f s is outside the clip, which is %.2f s long." % (t, state["length"]))
    try:
        graph, files = after_source(state, "src", "whole", 1, default)
    except ValueError as e:
        raise NotDrawn(str(e))
    graph =("[0:v]setpts=PTS-STARTPTS+%.3f/TB[src];%s;[whole]scale=%d:-2[fin]" % (t, graph, width))
    key = hashlib.sha1(json.dumps([rec["path"], rec["size"], rec["mtime"], round(at[1], 3), graph, files],
                                  ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    target = os.path.join(folder("clipframes"), key + ".jpg")
    if os.path.isfile(target):
        return target
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-ss", "%.3f" % at[1], "-i", rec["path"]]
    for f in files:
        cmd += ["-itsoffset", "%.3f" % t, "-loop", "1", "-framerate", "30", "-i", f]
    cmd += ["-filter_complex", graph, "-map", "[fin]", "-frames:v", "1", "-q:v", "3", "-f", "image2",
            target + ".part.jpg"]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=FRAME_SECONDS)
    except (subprocess.SubprocessError, OSError) as e:
        raise NotDrawn("FFmpeg could not draw the frame (%s)." % e)
    if done.returncode != 0 or not os.path.isfile(target + ".part.jpg"):
        raise NotDrawn("FFmpeg could not draw the frame: %s" % (done.stderr.strip().splitlines() or ["no reason"])[-1])
    os.replace(target + ".part.jpg", target)
    return target
