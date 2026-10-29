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

from . import effects, recording

W, H = 1080, 1920
PUSH, PUSH_SHORTEST = 1.08, 1.0
CIRCLE, RING, FACE_ROOM = (760, 800, 260), 6, 1.6
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
OVERVIEW_FROM = 180
OVERVIEW_AT = (0.1, 0.35, 0.6, 0.85)
OVERVIEW_SIDE = 400
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
    target = _sheet_path("%s|%d|%s|%d|%s|%d" % (rec["path"], rec["size"], rec["mtime"], c["n"], times, height))
    if os.path.isfile(target):
        return target
    out = _row([(rec, t, "%.1f s" % t) for t in times], height)
    if out is None:
        return None
    _mark(out, str(c["n"]), (12, 46), 1.3, 3)
    return _saved(out, target)


def pieces_sheet(pieces, height=STRIP_HEIGHT):
    """One picture of the pieces a clip is joined from, in order: the frame at the middle of each, from its own clip,
    side by side, height pixels tall, each marked with its clip's number and the second. The path of the jpg in
    work/montage, or None when a frame could not be taken. Seen 10-29 on a trip of seven clips: every piece Muse chose
    was 6 s from the first second its strip showed, so it never saw five of the six seconds it put in."""
    shots = [(p["rec"], round((p["from"] + p["to"]) / 2, 1), "%d: %.1f s" % (p["clip"], (p["from"] + p["to"]) / 2))
             for p in pieces]
    if not shots:
        return None
    target = _sheet_path("pieces|%s|%d" % ("|".join("%s|%d|%s|%s" % (r["path"], r["size"], r["mtime"], t)
                                                    for r, t, _ in shots), height))
    if os.path.isfile(target):
        return target
    out = _row(shots, height)
    return None if out is None else _saved(out, target)


def _sheet_path(key):
    folder = os.path.join(recording.WORK, "montage")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".jpg")


def _row(shots, height):
    """Frames side by side, height pixels tall, each marked: shots are (recording, second, mark). None when a frame
    could not be taken."""
    frames = []
    for rec, t, mark in shots:
        name = recording.frame(rec, at=t)
        if not name:
            return None
        img = cv2.imdecode(np.fromfile(os.path.join(recording.WORK, "frames", name), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)
        _mark(img, mark, (8, height - 12), 0.6, 2)
        frames.append(img)
    row = [frames[0]]
    for img in frames[1:]:
        row += [np.full((height, GAP, 3), 16, np.uint8), img]
    return np.hstack(row)


def _saved(out, target):
    ok, data = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    data.tofile(target + ".part")
    os.replace(target + ".part", target)
    return target


def overview(rec):
    """For a recording of OVERVIEW_FROM s or more, one picture for Muse of four moments across it, two by two, each
    marked with its time: (the jpg's path in work/montage, [their seconds]), or None. One picture from near the start
    can mislead: on 10-29 an hour-long travel vlog opened on two people in a room, Muse took it for a talk, ranked
    its stretches by how loud they were and cut friends laughing on a bridge with nothing clear said."""
    if rec["seconds"] < OVERVIEW_FROM:
        return None
    times = [round(rec["seconds"] * k, 1) for k in OVERVIEW_AT]
    key = hashlib.sha1(("overview|%s|%d|%s|%s|%d" % (rec["path"], rec["size"], rec["mtime"], times, OVERVIEW_SIDE)
                        ).encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(recording.WORK, "montage")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, key + ".jpg")
    if os.path.isfile(target):
        return target, times
    tiles = []
    for t in times:
        name = recording.frame(rec, at=t)
        if not name:
            return None
        img = cv2.imdecode(np.fromfile(os.path.join(recording.WORK, "frames", name), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        size = tiles[0].shape[1::-1] if tiles else (max(1, round(w * OVERVIEW_SIDE / max(w, h))),
                                                    max(1, round(h * OVERVIEW_SIDE / max(w, h))))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        _mark(img, recording.clock(t), (8, size[1] - 12), 0.6, 2)
        tiles.append(img)
    h, w = tiles[0].shape[:2]
    gap_row = np.full((h, GAP, 3), 16, np.uint8)
    rows = [np.hstack([tiles[0], gap_row, tiles[1]]), np.hstack([tiles[2], gap_row, tiles[3]])]
    out = np.vstack([rows[0], np.full((GAP, rows[0].shape[1], 3), 16, np.uint8), rows[1]])
    ok, data = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    data.tofile(target + ".part")
    os.replace(target + ".part", target)
    return target, times


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


def _wide(rec, centre):
    """(the lying picture's width scaled to the frame's height, the left edge of the upright crop around centre)."""
    wide = int(round(rec["width"] * H / float(rec["height"]) / 2)) * 2
    return wide, int(round(min(max(centre * wide - W / 2.0, 0), wide - W)))


def upright(rec, centre):
    """FFmpeg's filter cropping a lying picture upright to fill the W x H frame: its full height, and the width the
    frame takes of it, around centre (0 to 1 of its width), kept inside the picture."""
    wide, x = _wide(rec, centre)
    return "scale=%d:%d,crop=%d:%d:%d:0" % (wide, H, W, H, x)


def shown(rec, centre):
    """(x, width) of the lying picture's own pixels that the upright crop around centre shows."""
    wide, x = _wide(rec, centre)
    k = rec["height"] / float(H)
    return x * k, W * k


def _on(rec, centre, focus):
    """Where the middle of focus (a box on the 0-1000 grid of the picture) lands in the upright frame."""
    wide, x = _wide(rec, centre)
    return (focus[0] + focus[2]) / 2000.0 * wide - x, (focus[1] + focus[3]) / 2000.0 * H


def _face_square(rec, face):
    """[x, y, side, side] of the picture around a face box [x, y, w, h], FACE_ROOM times the face's size, kept
    inside the picture, even numbers for FFmpeg."""
    fx, fy, fw, fh = face
    side = min(max(fw, fh) * FACE_ROOM, rec["width"], rec["height"])
    x = min(max(fx + fw / 2.0 - side / 2.0, 0), rec["width"] - side)
    y = min(max(fy + fh / 2.0 - side / 2.0, 0), rec["height"] - side)
    return [int(v) // 2 * 2 for v in (x, y, side, side)]


def _round(src, d, out):
    """Graph text from [src] to [out]: the picture squeezed to d x d and cut round."""
    return ("[%s]scale=%d:%d,format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
            "a='255*lte(hypot(X-(W-1)/2,Y-(H-1)/2),W/2)'[%s]" % (src, d, d, out))


def _circle(i, rec, face, base, out):
    """Graph text drawing the face of piece i, from [f<i>], in a white-ringed circle at CIRCLE over [base] to
    [out], so the person stays in the clip while the product fills the frame."""
    x, y, side, _ = _face_square(rec, face)
    cx, cy, d = CIRCLE
    return ";".join([
        "[f%d]crop=%d:%d:%d:%d,split[fa%d][fb%d]" % (i, side, side, x, y, i, i),
        "[fb%d]drawbox=c=white:t=fill[fw%d]" % (i, i),
        _round("fw%d" % i, d + 2 * RING, "ring%d" % i),
        _round("fa%d" % i, d, "disk%d" % i),
        "[%s][ring%d]overlay=%d:%d[cr%d]" % (base, i, cx - RING, cy - RING, i),
        "[cr%d][disk%d]overlay=%d:%d[%s]" % (i, i, cx, cy, out)])


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
            crop = upright(rec, p["centre"])
            if p.get("push") and p.get("focus") and d >= PUSH_SHORTEST:
                crop += "," + effects.push_filters(0, d, PUSH, _on(rec, p["centre"], p["focus"]), (W, H))
            if p.get("face"):
                chains.append("[%d:v:0]split[m%d][f%d]" % (i, i, i))
                chains.append("[m%d]%s,setsar=1[c%d]" % (i, crop, i))
                chains.append(_circle(i, rec, p["face"], "c%d" % i, "p%d" % i))
            else:
                chains.append("[%d:v:0]%s,setsar=1[p%d]" % (i, crop, i))
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
                                    round(p["to"], 3), p.get("centre")] + ([p.get("focus"), p.get("face")] if p.get(
                                        "push") or p.get("face") else []) for p in pieces]).encode("utf-8"))
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
