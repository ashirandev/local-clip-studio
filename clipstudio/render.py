"""The clip drawn from its project list: one FFmpeg filter graph from the recording to the 1080x1920 frame.

    recording piece(s)  ->  [src] at clip time t  ->  source stage (blurs)  ->  layout (layouts.py)
                        ->  frame stage (camera hits, stickers, then the words, then flashes)

The preview frame is the same graph started at the frame's time: the piece is opened at the matching recording
second and its timestamps are set so t is still clip time, so a title from 2 s to 5 s shows in the preview at 3 s
exactly as it will in the finished clip. export() joins the pieces at 30 fps from clip second 0 and draws them with
the same stages.
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading

from . import effects, layouts, project, recording, text

PREVIEW_WIDTH = 540
FRAME_SECONDS = 60
WORD_SECONDS = 0.8
HIT_KINDS = ("flash", "shake", "punch", "word")
FPS, RATE = 30, 48000
LUFS, PEAK, RANGE = -14, -1.5, 11
PRESET, CRF = "medium", 19
EXPORT_SECONDS = 600


class NotDrawn(Exception):
    """The frame could not be drawn. The sentence says why."""


class Stopped(Exception):
    """The person pressed Stop while the clip was being made: export() raises it and leaves no file."""


def _ffmpeg():
    return shutil.which("ffmpeg") or "ffmpeg"


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
        events += text.captions(captions["lines"], st, captions["band"], lit=captions.get("lit", False))
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
    - source stage, on the recording's picture: the blurs (a fixed box, or a face's track of [t, x, y, w, h] points in
      clip seconds), so a blurred box stays blurred wherever the layout shows it, the card's blurred background
      included;
    - the layout, to the 1080x1920 frame;
    - frame stage: the camera hits (shake, punch-in), the stickers, then the words (titles, captions and big words,
      one ASS file), so the words hold still while the picture moves and stay on top of a sticker, then the flashes
      last, so a flash lights the words too.
    A drawing that cannot be made raises ValueError with the sentence for Muse."""
    blurs = [effects.blur_track(b["track"]) if b.get("track") else effects.blur_box(b["box"], b["start"], b["end"])
             for b in state.get("blurs") or []]
    seen, files = effects.join(blurs, src, "seen", first_input)
    lay, more = layouts.graph(state["clip"]["layout"], "seen", "drawn", first_input + len(files))
    files = files + more
    hits = _hits(state)
    camera = {"shake": lambda h: effects.shake(h["at"]),
              "punch": lambda h: effects.punch_in(h["at"], centre=h["centre"])}
    fragments = [camera[h["kind"]](h) for h in hits if h["kind"] in camera]
    fragments += [effects.sticker(k["png"], k["at"], k["length"], k["box"]) for k in state.get("stickers") or []]
    words = ass(state, default)
    if words:
        fragments.append(text.burn(words))
    fragments += [effects.flash(h["at"]) for h in hits if h["kind"] == "flash"]
    stage, more = effects.join(fragments, "drawn", out, first_input + len(files))
    return ";".join((seen, lay, stage)), files + more


def frame(rec, state, t, default="clean", width=PREVIEW_WIDTH):
    """The clip's frame at clip second t, width pixels wide: the jpg's path in work/clipframes, made once. The
    extra pictures (masks, shadows, stickers) are opened from second t, so FFmpeg does not read every picture frame
    before it: a frame late in a long clip draws as fast as the first."""
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


# ---- the finished clip ---------------------------------------------------------------------------------------

def _sound(pieces):
    """Graph text from the first sound track of each piece's input to [snd], the pieces joined. Each input is
    opened at its piece's start, so its timestamps start at 0."""
    return "%sconcat=n=%d:v=0:a=1,aresample=%d[snd]" % ("".join("[%d:a:0]" % k for k in range(len(pieces))),
                                                         len(pieces), RATE)


def _inputs(rec, pieces, sound_only=False):
    """The recording opened once per piece, at the piece's start and for its length, so a piece deep in a long
    recording is reached by seeking, not by decoding everything before it."""
    cmd = []
    for a, b in pieces:
        cmd += ["-ss", "%.3f" % a, "-t", "%.3f" % (b - a)] + (["-vn"] if sound_only else []) + ["-i", rec["path"]]
    return cmd


def loudness(rec, pieces):
    """How loud the joined pieces are, measured for loudnorm's second pass ({"input_i", "input_tp", "input_lra",
    "input_thresh", "target_offset"}), or None when the recording has no sound or the pieces are silent."""
    if not rec.get("sound_tracks"):
        return None
    graph = _sound(pieces) + ";[snd]loudnorm=I=%s:TP=%s:LRA=%s:print_format=json[o]" % (LUFS, PEAK, RANGE)
    cmd = [_ffmpeg(), "-hide_banner", "-nostats"] + _inputs(rec, pieces, True) + [
        "-filter_complex", graph, "-map", "[o]", "-f", "null", "-"]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=EXPORT_SECONDS)
    except (subprocess.SubprocessError, OSError) as e:
        raise NotDrawn("FFmpeg could not measure the clip's sound (%s)." % e)
    found = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", done.stderr)
    if done.returncode != 0 or not found:
        raise NotDrawn("FFmpeg could not measure the clip's sound: %s"
                       % (done.stderr.strip().splitlines() or ["no reason"])[-1])
    measured = json.loads(found.group(0))
    try:
        if not math.isfinite(float(measured["input_i"])):
            return None
    except (KeyError, ValueError):
        return None
    return measured


def _level(measured):
    """The second loudnorm pass: the same gain for the whole clip (linear), so the sound is not pumped."""
    if not measured:
        return ""
    return (",loudnorm=I=%s:TP=%s:LRA=%s:measured_I=%s:measured_TP=%s:measured_LRA=%s:measured_thresh=%s:offset=%s:"
            "linear=true,aresample=%d" % (LUFS, PEAK, RANGE, measured["input_i"], measured["input_tp"],
                                          measured["input_lra"], measured["input_thresh"], measured["target_offset"],
                                          RATE))


def export(rec, state, out_path, on_progress=None, on_start=None, default="clean"):
    """Makes the finished clip at out_path and returns out_path once the file is there: H.264 and AAC, 30 fps,
    1080x1920, the sound at -14 LUFS, the index at the front of the file so it plays while it downloads. The pieces
    are joined, then drawn by the same stages as the preview frame. The file is written next to out_path and only
    replaces it when FFmpeg has finished. on_progress(seconds done) follows FFmpeg; on_start(stop) is given a
    function that ends FFmpeg at once, after which export raises Stopped and leaves no file."""
    clip = state["clip"]
    if not clip:
        raise NotDrawn("There is no clip yet, so there is nothing to make.")
    pieces, length = clip["pieces"], state["length"]
    count = len(pieces)
    try:
        drawn, files = after_source(state, "src", "fin", count, default)
    except ValueError as e:
        raise NotDrawn(str(e))
    sound = rec.get("sound_tracks")
    measured = loudness(rec, pieces) if sound else None
    streams = "".join("[%d:v:0]" % k + ("[%d:a:0]" % k if sound else "") for k in range(count))
    joined = "%sconcat=n=%d:v=1:a=%d[cat]%s;[cat]fps=%d[src]" % (
        streams, count, 1 if sound else 0, "[catsnd]" if sound else "", FPS)
    if sound:
        joined += ";[catsnd]aresample=%d%s[snd]" % (RATE, _level(measured))
    graph = joined + ";" + drawn
    work = folder("exports")
    script = os.path.join(work, hashlib.sha1(graph.encode("utf-8")).hexdigest()[:16] + ".graph")
    with open(script, "w", encoding="utf-8", newline="\n") as f:
        f.write(graph)
    part = out_path + ".part"
    log_path = os.path.join(work, os.path.basename(out_path) + ".log")
    cmd = [_ffmpeg(), "-hide_banner", "-nostats", "-y", "-progress", "pipe:1"] + _inputs(rec, pieces)
    for f in files:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-i", f]
    cmd += ["-/filter_complex", script, "-map", "[fin]"] + (["-map", "[snd]"] if sound else []) + [
        "-t", "%.3f" % length, "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF), "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", str(RATE), "-movflags", "+faststart", "-f", "mp4",
        part]
    stopped = threading.Event()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL, text=True,
                                    encoding="utf-8", errors="replace")
        except OSError as e:
            raise NotDrawn("FFmpeg could not be started to make the clip (%s)." % e)

        def stop():
            stopped.set()
            try:
                proc.kill()
            except OSError:
                pass

        if on_start:
            on_start(stop)
        with proc.stdout as progress:
            for line in progress:
                key, _, value = line.strip().partition("=")
                if key == "out_time_us" and on_progress and value.isdigit():
                    on_progress(min(length, int(value) / 1e6))
        proc.wait()
    if stopped.is_set() or proc.returncode != 0 or not os.path.isfile(part) or os.path.getsize(part) == 0:
        if os.path.isfile(part):
            os.remove(part)
        if stopped.is_set():
            raise Stopped("Stopped: the clip was not made.")
        with open(log_path, encoding="utf-8", errors="replace") as f:
            last = [x for x in f.read().strip().splitlines() if x.strip()] or ["no reason"]
        raise NotDrawn("FFmpeg could not make the clip: %s" % last[-1])
    os.replace(part, out_path)
    if on_progress:
        on_progress(length)
    return out_path
