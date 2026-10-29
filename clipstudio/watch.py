"""Muse Wonder watches a stretch of the recording, picture and sound together, and points out what words alone miss.

The hearing writes down what is said, and the looks are pictures a few seconds apart, so a scream, a laugh, a fall, a
death screen or a silent stare reaches Muse only when somebody says something about it. Muse Wonder takes video with
its sound (Meta's Model API, video_url), so it is asked to watch: each moment a viewer would laugh or react at, with
its second, what happens, and the one studio effect that would land there. Measured 10-29 on 30 s of Tarm's own
stream at 360 pixels and 10 frames a second (1.46 MB): 34.6 s, 2,735 tokens read and 2,375 written, about $0.06; it
found the yelp at a hit (shake), the death from behind (thump), the laugh at the death screen, and the age going from
38 to 44 in silence (the big word "44"), none of which was said.

The stretch is cut small (WATCH_HEIGHT pixels, WATCH_FPS frames a second, mono sound), at most LONGEST seconds, and
kept in work/watch by recording and times, so the same stretch is cut once. It goes to Meta with the person's own key,
the same key as Muse Spark: nothing is sent without one.
"""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import threading

from . import effects, recording, render

LONGEST = 60
SHORTEST = 2
WATCH_HEIGHT = 360
WATCH_FPS = 10
CUT_SECONDS = 300
ANSWER_TOKENS = 6000
MOST_MOMENTS = 8
PAST = 0.5
HITS = ("flash", "shake", "punch", "word")
EFFECTS = HITS + tuple(effects.SOUNDS) + ("flurry", "none")
KINDS = ("scream", "laugh", "fail", "surprise", "reaction", "silent joke", "action")
SCHEMA = {"type": "object", "properties": {"moments": {"type": "array", "maxItems": MOST_MOMENTS, "items": {
    "type": "object", "properties": {
        "at": {"type": "number", "description": "The second in this video."},
        "kind": {"type": "string", "enum": list(KINDS)},
        "what": {"type": "string", "description": "What happens, in one line."},
        "effect": {"type": "string", "enum": list(EFFECTS)},
        "word": {"type": "string", "description": "For the effect word: the word to pop in, else empty."}},
    "required": ["at", "kind", "what", "effect", "word"]}}}, "required": ["moments"]}
ASK = ("This is %(length)d seconds of %(name)s, with its sound. Watch it and listen. List the moments a viewer would "
       "laugh or react at that the words alone would miss: a scream, a laugh, a fail, a surprise, a reaction, a "
       "silent joke, a sudden action. Leave out what is only said. For each give the second in this video, what "
       "happens in one line in %(language)s, and the one effect that would land there: flash (a white flash), shake "
       "(the picture shakes), punch (a quick zoom in), word (a big word pops in: give the word, short, in the "
       "language spoken), a sound (%(sounds)s), flurry (stickers flying), or none. At most %(most)d moments, the "
       "strongest first; none at all when nothing happens.")


class WatchError(Exception):
    pass


def folder():
    return os.path.join(recording.WORK, "watch")


def name(rec, start, end):
    key = "%s|%d|%s|%.2f|%.2f|%d|%d" % (rec["path"], rec["size"], rec["mtime"], start, end, WATCH_HEIGHT, WATCH_FPS)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".mp4"


def cut(rec, start, end, on_start=None):
    """The small video of start to end, cut once: its path, or WatchError with a sentence."""
    target = os.path.join(folder(), name(rec, start, end))
    if os.path.isfile(target):
        return target
    tool = shutil.which("ffmpeg")
    if not tool:
        raise WatchError("FFmpeg is not installed, so nothing could be cut for Muse Wonder to watch.")
    os.makedirs(folder(), exist_ok=True)
    part = "%s.%d-%d.part.mp4" % (target[:-4], os.getpid(), threading.get_ident())
    cmd = [tool, "-v", "error", "-y", "-ss", "%.2f" % start, "-i", rec["path"], "-t", "%.2f" % (end - start),
           "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale=-2:%d,fps=%d" % (WATCH_HEIGHT, WATCH_FPS),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ac", "1", "-b:a", "48k", "-movflags", "+faststart", part]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if on_start:
            on_start(proc.kill)
        _, err = proc.communicate(timeout=CUT_SECONDS)
        if proc.returncode or not os.path.isfile(part) or os.path.getsize(part) == 0:
            raise WatchError("ffmpeg could not cut %s-%s for Muse Wonder (%s)." % (
                recording.clock(start), recording.clock(end), (err or b"").decode("utf-8", "replace").strip()[:200]
                or "it stopped"))
        render.settle(part, target)
        return target
    except subprocess.TimeoutExpired:
        proc.kill()
        raise WatchError("Cutting %s-%s took over %d s, so it was stopped." % (
            recording.clock(start), recording.clock(end), CUT_SECONDS))
    finally:
        if os.path.exists(part):
            os.remove(part)


def question(path, length, title, language):
    """The message that asks Muse Wonder to watch the video at path."""
    with open(path, "rb") as f:
        video = "data:video/mp4;base64," + base64.b64encode(f.read()).decode("ascii")
    words = ASK % {"length": round(length), "name": title, "language": language or "the language spoken",
                   "sounds": ", ".join(effects.SOUNDS), "most": MOST_MOMENTS}
    return [{"role": "user", "content": [{"type": "text", "text": words},
                                         {"type": "video_url", "video_url": {"url": video}}]}]


def moments(answer, start, end):
    """Muse Wonder's answer as moments of the recording: [{"at": recording second, "kind", "what", "effect", "word"}],
    in time order, each inside start to end; a moment without a second, or past the end of the video it was sent, is
    left out rather than guessed. WatchError when the answer is not the JSON asked for."""
    try:
        got = json.loads(answer)
        listed = got["moments"]
    except (ValueError, KeyError, TypeError):
        raise WatchError("Muse Wonder's answer was not the list asked for, so nothing it saw can be used.")
    out = []
    for m in listed[:MOST_MOMENTS]:
        if not isinstance(m, dict) or not isinstance(m.get("at"), (int, float)) or isinstance(m.get("at"), bool):
            continue
        if not 0 <= m["at"] <= end - start + PAST:
            continue
        at = round(start + min(float(m["at"]), end - start - PAST), 2)
        effect = m.get("effect") if m.get("effect") in EFFECTS else "none"
        word = str(m.get("word") or "").strip()[:30]
        out.append({"at": at, "kind": m.get("kind") if m.get("kind") in KINDS else "reaction",
                    "what": " ".join(str(m.get("what") or "").split())[:200], "effect": effect,
                    "word": word if effect == "word" else ""})
    return sorted(out, key=lambda m: m["at"])
