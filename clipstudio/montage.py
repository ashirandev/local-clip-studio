"""A montage: short clips (a trip, an event) made into one clip, in the order Muse chooses and for the reasons it gives.

The studio makes every clip from one recording, so a montage is made into one first. Muse sees each clip (three
pictures from it, when it was shot and what is heard in it) and says which pieces go in, in what order, why, and what
the words on screen say over each. The program joins those pieces into one upright recording, and the layout, the
words, the captions, the render and the checks work on it as on any recording.

When a clip was shot is read from the camera's own tag (com.apple.quicktime.creationdate, with its time zone) before
the file's creation_time: the seven Akihabara clips saved from an app (10-27) all carry the minute they were saved,
14:07, as their creation_time, and the days they were shot, 10 and 11 March, only in the camera's tag.
"""
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading

import cv2
import numpy as np

from . import recording

W, H = 1080, 1920
FPS = 30
RATE = 48000
FILL = 0.74
BLUR, DIM = 30, -0.12
FADE = 0.04
STRIP_HEIGHT, STRIP_LOWEST = 360, 280
MANY = 20
STRIP_AT = (0.15, 0.5, 0.85)
ONE_PICTURE = 1.5
GAP = 8
CAMERA_TAG = "com.apple.quicktime.creationdate"
SAME_TIME = 2.0
JOIN_SECONDS = 600
STILL_NOISE = 0.003
FROZEN = 1.0
PRESET, CRF = "veryfast", 18


class MontageError(Exception):
    """A sentence for Muse: what could not be done and why."""


class Stopped(Exception):
    """The person pressed Stop while the clips were joined: join() leaves no file."""


def _tags(path):
    tool = shutil.which("ffprobe")
    if not tool:
        return {}
    try:
        out = subprocess.run([tool, "-v", "error", "-show_entries", "format_tags", "-of", "json", path],
                             capture_output=True, timeout=recording.PROBE_SECONDS, check=True).stdout
        return (json.loads(out.decode("utf-8", "replace")).get("format") or {}).get("tags") or {}
    except (subprocess.SubprocessError, ValueError, OSError):
        return {}


def _when(text):
    try:
        return datetime.datetime.fromisoformat((text or "").strip())
    except ValueError:
        return None


def shot(tags):
    """(when the clip was shot, with its time zone, or None; "camera", "file" or None) from a file's tags."""
    for key, how in ((CAMERA_TAG, "camera"), ("creation_time", "file")):
        t = _when(tags.get(key))
        if t and t.tzinfo:
            return t, how
    return None, None


def gather(recs, tags=_tags):
    """The clips, numbered in the order they were shot: [{n, rec, shot, how}]. A file's own time shared with another
    clip's to within SAME_TIME s is when they were saved together, so it says nothing (how "saved", shot None). Clips
    with no time come last, by name."""
    found = []
    for rec in recs:
        t, how = shot(tags(rec["path"]))
        found.append({"rec": rec, "shot": t, "how": how})
    for c in found:
        if c["how"] == "file" and any(o is not c and o["how"] in ("file", "saved") and
                                      abs((o["shot"] - c["shot"]).total_seconds()) <= SAME_TIME for o in found):
            c["how"] = "saved"
    for c in found:
        if c["how"] == "saved":
            c["shot"] = None
    found.sort(key=lambda c: (0, c["shot"].timestamp(), c["rec"]["name"]) if c["shot"] else (1, 0, c["rec"]["name"]))
    for n, c in enumerate(found, 1):
        c["n"] = n
    return found


def when(c):
    """When a clip was shot, as Muse reads it, in the camera's own time zone."""
    if not c["shot"]:
        return "time not known" if c["how"] != "saved" else "time not known (the file holds when it was saved)"
    return c["shot"].strftime("shot %Y-%m-%d %H:%M:%S") + ("" if c["how"] == "camera" else " (the file's time)")


def strip_times(seconds):
    """The seconds of a clip its strip shows."""
    if seconds < ONE_PICTURE:
        return [round(seconds / 2, 2)]
    return [round(seconds * k, 2) for k in STRIP_AT]


def strip(c, height=STRIP_HEIGHT):
    """One picture of clip c for Muse: frames from it side by side, height pixels tall, each marked with the second it
    is at, and the clip's number in the corner. Returns the jpg's path in work/montage, or None when no frame could be
    taken. A picture costs Muse about one token for every 650 of its pixels, read on the processor (measured 10-27:
    19 strips 480 px tall were 11,800 tokens and 142 s, a third of its 32,768), so the strips are no taller than they
    must be to tell what happens in a clip."""
    rec = c["rec"]
    times = strip_times(rec["seconds"])
    key = hashlib.sha1(("%s|%d|%s|%d|%s|%d" % (rec["path"], rec["size"], rec["mtime"], c["n"], times, height)
                        ).encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(recording.WORK, "montage")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, key + ".jpg")
    if os.path.isfile(target):
        return target
    frames = []
    for t in times:
        name = recording.frame(rec, at=t)
        if not name:
            return None
        img = cv2.imdecode(np.fromfile(os.path.join(recording.WORK, "frames", name), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)
        _mark(img, "%.1f s" % t, (8, height - 12), 0.6, 2)
        frames.append(img)
    row = [frames[0]]
    for img in frames[1:]:
        row += [np.full((height, GAP, 3), 16, np.uint8), img]
    out = np.hstack(row)
    _mark(out, str(c["n"]), (12, 46), 1.3, 3)
    ok, data = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    data.tofile(target + ".part")
    os.replace(target + ".part", target)
    return target


def _mark(img, text, at, scale, thick):
    """White text on a dark box, readable on any picture."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = at
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + base + 4), (16, 16, 16), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def stills(rec):
    """The stretches of rec where the picture does not move for FROZEN s or more: [(from, to)] in its seconds. A held
    picture is a frozen screen (Tarm, 10-27: "ไม่เอา จอแช่"). None of the 31 phone clips of 10-27 holds
    still that long; a picture held on purpose, or a freeze frame, does."""
    key = hashlib.sha1(("%s|%d|%s|%g|%g" % (rec["path"], rec["size"], rec["mtime"], STILL_NOISE, FROZEN)
                        ).encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(recording.WORK, "montage")
    os.makedirs(folder, exist_ok=True)
    kept = os.path.join(folder, key + ".stills.json")
    try:
        with open(kept, encoding="utf-8") as f:
            return [tuple(s) for s in json.load(f)]
    except (OSError, ValueError):
        pass
    tool = shutil.which("ffmpeg")
    if not tool:
        raise MontageError("FFmpeg is not installed, so the clips cannot be looked at.")
    try:
        run = subprocess.run([tool, "-hide_banner", "-nostats", "-i", rec["path"], "-map", "0:v:0", "-vf",
                              "scale=270:-2,freezedetect=n=%g:d=%g" % (STILL_NOISE, FROZEN), "-f", "null", "-"],
                             capture_output=True, timeout=JOIN_SECONDS)
    except (subprocess.SubprocessError, OSError) as e:
        raise MontageError("FFmpeg could not look at %s (%s)." % (rec["name"], e))
    said = run.stderr.decode("utf-8", "replace")
    starts = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", said)]
    ends = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", said)]
    found = [(round(a, 2), round(ends[i] if i < len(ends) else rec["seconds"], 2)) for i, a in enumerate(starts)]
    with open(kept + ".part", "w", encoding="utf-8") as f:
        json.dump(found, f)
    os.replace(kept + ".part", kept)
    return found


def fills(width, height):
    """True when cropping a width x height picture to 9:16 keeps at least FILL of it, so it fills the frame as it is;
    else it is lying, and fills the frame cropped upright around its centre (upright()), or is shown whole over a
    blurred copy of itself when a piece has no centre."""
    aspect, own = W / float(H), width / float(height)
    return min(own, aspect) / max(own, aspect) >= FILL


def upright(rec, centre):
    """FFmpeg's filter cropping a lying picture upright to fill the W x H frame: its full height, and the width the
    frame takes of it, around centre (0 to 1 of its width), kept inside the picture."""
    wide = int(round(rec["width"] * H / float(rec["height"]) / 2)) * 2
    x = int(round(min(max(centre * wide - W / 2.0, 0), wide - W)))
    return "scale=%d:%d,crop=%d:%d:%d:0" % (wide, H, W, H, x)


def starts(pieces):
    """The second of the joined recording each piece starts at."""
    out, at = [], 0.0
    for p in pieces:
        out.append(round(at, 3))
        at += p["to"] - p["from"]
    return out


def graph(pieces):
    """(FFmpeg's input arguments, the filter graph) joining pieces [{rec, from, to[, centre]}] in order into one upright
    W x H recording at FPS, its sound at RATE in stereo. A lying piece with a centre is cropped upright around it; one
    without is shown whole over a blurred copy of itself. A clip without sound gets silence, and each piece's sound
    fades in and out over FADE s, so a cut does not click."""
    args, chains, joined = [], [], ""
    for i, p in enumerate(pieces):
        rec, d = p["rec"], round(p["to"] - p["from"], 3)
        args += ["-ss", "%.3f" % p["from"], "-t", "%.3f" % d, "-i", rec["path"]]
        cover = "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d" % (W, H, W, H)
        if fills(rec["width"], rec["height"]):
            chains.append("[%d:v:0]%s,setsar=1[p%d]" % (i, cover, i))
        elif p.get("centre") is not None:
            chains.append("[%d:v:0]%s,setsar=1[p%d]" % (i, upright(rec, p["centre"]), i))
        else:
            chains.append("[%d:v:0]split[w%d][b%d];[b%d]%s,gblur=sigma=%d,eq=brightness=%g[g%d];"
                          "[w%d]scale=%d:%d:force_original_aspect_ratio=decrease[s%d];"
                          "[g%d][s%d]overlay=(W-w)/2:(H-h)/2,setsar=1[p%d]"
                          % (i, i, i, i, cover, BLUR, DIM, i, i, W, H, i, i, i, i))
        chains.append("[p%d]fps=%d,format=yuv420p,trim=duration=%.3f,setpts=PTS-STARTPTS[v%d]" % (i, FPS, d, i))
        sound = ("[%d:a:0]aresample=%d,aformat=channel_layouts=stereo,apad" % (i, RATE) if rec.get("sound_tracks")
                 else "anullsrc=r=%d:cl=stereo" % RATE)
        chains.append("%s,atrim=duration=%.3f,afade=t=in:d=%g,afade=t=out:st=%.3f:d=%g,asetpts=PTS-STARTPTS[a%d]"
                      % (sound, d, FADE, max(0.0, d - FADE), FADE, i))
        joined += "[v%d][a%d]" % (i, i)
    chains.append("%sconcat=n=%d:v=1:a=1[v][a]" % (joined, len(pieces)))
    return args, ";".join(chains)


def target(pieces, name):
    """Where the joined recording of pieces is kept: the same pieces in the same order are joined once."""
    key = hashlib.sha1(json.dumps([[p["rec"]["path"], p["rec"]["size"], p["rec"]["mtime"], round(p["from"], 3),
                                    round(p["to"], 3), p.get("centre")] for p in pieces]).encode("utf-8"))
    key = key.hexdigest()[:10]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:40] or "clips"
    return os.path.join(recording.WORK, "montage", "%s-%s.mp4" % (safe, key))


def _made_of_path(path):
    return path + ".pieces.json"


def made_of(path):
    """The pieces the joined recording at path was made of, in order: [{"path", "name", "from", "to", "centre",
    "width", "height", "seconds", "fps", "sound_tracks", "clip", "why"}], seconds of each piece's own file; None for a
    recording that was not joined here, or was joined before this was written down. An export to an editor
    (premiere.py) points at those files, so each piece can be made longer from its own file."""
    try:
        with open(_made_of_path(path), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_made_of(pieces, out):
    if made_of(out) is not None:
        return
    rows = [{"path": p["rec"]["path"], "name": p["rec"]["name"], "from": p["from"], "to": p["to"],
             "centre": p.get("centre"), "width": p["rec"]["width"], "height": p["rec"]["height"],
             "seconds": p["rec"]["seconds"], "fps": p["rec"].get("fps"), "sound_tracks": p["rec"].get("sound_tracks"),
             "clip": p.get("clip"), "why": p.get("why")} for p in pieces]
    part = _made_of_path(out) + ".part"
    with open(part, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    os.replace(part, _made_of_path(out))


def join(pieces, out, on_start=None):
    """Joins pieces into out and returns out once the file is there and as long as the pieces together. The file is
    written next to out and only replaces it when FFmpeg has finished, with what it was made of beside it (made_of).
    on_start(stop) is given a function that ends FFmpeg at once, after which join raises Stopped and leaves no file."""
    if os.path.isfile(out):
        _write_made_of(pieces, out)
        return out
    tool = shutil.which("ffmpeg")
    if not tool:
        raise MontageError("FFmpeg is not installed, so the clips cannot be joined.")
    args, chain = graph(pieces)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    script = out + ".graph"
    with open(script, "w", encoding="utf-8", newline="\n") as f:
        f.write(chain)
    part = out + ".part"
    cmd = [tool, "-hide_banner", "-nostats", "-y"] + args + [
        "-/filter_complex", script, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", PRESET, "-crf",
        str(CRF), "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", str(RATE),
        "-movflags", "+faststart", "-f", "mp4", part]
    stopped = threading.Event()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    except OSError as e:
        raise MontageError("FFmpeg could not be started to join the clips (%s)." % e)

    def stop():
        stopped.set()
        try:
            proc.kill()
        except OSError:
            pass

    if on_start:
        on_start(stop)
    try:
        _, err = proc.communicate(timeout=JOIN_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, err = proc.communicate()
    try:
        if stopped.is_set():
            raise Stopped("Stopped: the clips were not joined.")
        if proc.returncode != 0 or not os.path.isfile(part):
            last = [x for x in err.decode("utf-8", "replace").splitlines() if x.strip()] or ["no reason"]
            raise MontageError("FFmpeg could not join the clips: %s" % last[-1])
        want = sum(p["to"] - p["from"] for p in pieces)
        try:
            got = recording.probe(part)["seconds"]
        except recording.NotOpened as e:
            raise MontageError("The joined clips could not be read back (%s)." % e)
        if abs(got - want) > 0.2:
            raise MontageError("The joined clips came out %.2f s long, not %.2f s, so they were not kept." % (got, want))
        os.replace(part, out)
        _write_made_of(pieces, out)
    finally:
        if os.path.isfile(part):
            os.remove(part)
    return out
