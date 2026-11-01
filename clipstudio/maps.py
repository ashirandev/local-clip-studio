"""The map of a recording: where to look first in a long recording, before anything in it has been looked at.

Two things are measured for every second:
- the sound: how loud it is (RMS, in dB), and then how far that rises over the minutes around it, because a stream
  that is loud all the time has no loud moments, only a loud level
- the picture: how much it changes (ffmpeg's scene score between one keyframe and the one before)

One ffmpeg run reads the recording once. The sound is read at 8,000 samples a second, one value a second. Only the
picture's keyframes are decoded (an encoder writes one every 1 to 4 seconds), shrunk to 128x72 grey, and each second
gets the change of the keyframe gap it falls in. Decoding only the keyframes is what makes it fast: on the first 10
minutes of a 1080p60 stream it took 3.5 s, against 27 s with every frame decoded.

The measurements are kept per file in work/maps (by path, size and time), so the second map of a recording is
instant. The ranking is made fresh each time from what is kept, with the weights of the recipe in use.
"""
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import threading
import time

from . import recording

VERSION = 1
SOUND = ("aresample=8000,aformat=channel_layouts=mono,asetnsamples=n=8000:p=0,"
         "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=RMS_level,"
         "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=sound.txt")
PICTURE = "scale=128:72,format=gray,select='gte(scene\\,0)',metadata=mode=print:key=lavfi.scene_score:file=picture.txt"
EXTRA_INPUT = []
AROUND = 90
LEAD = 0.7
QUIET_DB = -90.0
FLAT_DB = 3.0
USUAL = 0.3


class MapError(Exception):
    """The recording could not be mapped. The sentence says why."""


class Stopped(Exception):
    """The person pressed Stop while the map was being made."""


def folder():
    return os.path.join(recording.WORK, "maps")


def key(rec):
    st = os.stat(rec["path"])
    return hashlib.sha1(("%d|%s|%d|%s%s" % (VERSION, rec["path"], st.st_size, st.st_mtime, recording.tracks_key(rec))).encode("utf-8")
                        ).hexdigest()[:16]


def kept(rec):
    """The measurements kept for this recording, or None when it has not been mapped as it is now."""
    try:
        with open(os.path.join(folder(), key(rec) + ".json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def measure(rec, on_progress=None, on_start=None):
    """Reads the recording once and keeps {seconds, sound: [dB or None a second], keyframes: [[t, change]], took}.
    on_progress(seconds_read) is called as it goes; on_start(stop) hands over a function that stops it."""
    tool = shutil.which("ffmpeg")
    if not tool:
        raise MapError("FFmpeg is not installed, so the studio cannot map recordings.")
    os.makedirs(folder(), exist_ok=True)
    work = tempfile.mkdtemp(prefix="making-", dir=folder())
    try:
        graph, outputs = [], []
        if rec.get("sound_tracks"):
            chain, pad = recording.mixed(rec)
            graph.append("%s%s%s[sound]" % (chain, pad, SOUND))
            outputs += ["-map", "[sound]", "-f", "null", "-"]
        graph.append("[0:v:0]%s[picture]" % PICTURE)
        outputs += ["-map", "[picture]", "-f", "null", "-"]
        started = time.time()
        with open(os.path.join(work, "errors.txt"), "wb") as errors:
            proc = subprocess.Popen([tool, "-hide_banner", "-nostats", "-v", "error", "-progress", "pipe:1",
                                     "-skip_frame:v:0", "nokey"] + EXTRA_INPUT + ["-i", rec["path"],
                                     "-filter_complex", ";".join(graph)] + outputs,
                                    cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)
            stopped = threading.Event()

            def stop():
                stopped.set()
                proc.kill()

            if on_start:
                on_start(stop)
            for raw in proc.stdout:
                name, _, value = raw.decode("ascii", "replace").strip().partition("=")
                if name == "out_time_us" and on_progress and value.isdigit():
                    on_progress(int(value) / 1e6)
            proc.stdout.close()
            proc.wait()
        if stopped.is_set():
            raise Stopped()
        if proc.returncode != 0:
            with open(os.path.join(work, "errors.txt"), encoding="utf-8", errors="replace") as f:
                said = f.read().strip().splitlines()
            raise MapError("ffmpeg could not read %s (%s)." % (rec["name"], said[-1] if said else
                                                                 "exit code %d" % proc.returncode))
        seconds = max(1, int(math.ceil(rec["seconds"])))
        sound = [None] * seconds
        for t, value in _printed(os.path.join(work, "sound.txt")):
            if 0 <= int(t) < seconds:
                sound[int(t)] = value
        keyframes = [[round(t, 3), round(v or 0.0, 5)] for t, v in _printed(os.path.join(work, "picture.txt"))]
        if not keyframes:
            raise MapError("%s has no picture ffmpeg could read." % rec["name"])
        m = {"version": VERSION, "name": rec["name"], "seconds": seconds, "sound": sound, "keyframes": keyframes,
             "took": round(time.time() - started, 1)}
        target = os.path.join(folder(), key(rec) + ".json")
        with open(target + ".part", "w", encoding="utf-8") as f:
            json.dump(m, f)
        os.replace(target + ".part", target)
        return m
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _printed(path):
    """(time, value) pairs from ffmpeg's metadata print file. A value that is not a number (-inf) comes back None."""
    if not os.path.exists(path):
        return []
    pairs, t = [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("frame:"):
                t = float(line.rsplit("pts_time:", 1)[1])
            elif line.startswith("lavfi.") and t is not None:
                try:
                    value = float(line.split("=", 1)[1])
                except ValueError:
                    value = None
                pairs.append((t, value if value is not None and math.isfinite(value) else None))
    return pairs


def rank(m, weights, window, top):
    """(The stretches that stand out, best first: [{start, end, peak, rise, louder, busier}], what the map says
    about the whole recording). A second ranks by how much louder and how much busier it is than the rest of the
    recording (shares, 0 to 1, so neither signal can swamp the other), weighted by the recipe. The sound rises
    against the minutes around it, but never against a quieter level than the recording's usual quiet, so silence
    before a stream starts does not make its first words a peak. Each stretch is window seconds long with most of
    it before its peak, so what led up to the peak is in it."""
    n = m["seconds"]
    loud, heard = _loud(m)
    gap = [0.0] * n
    last = 0
    for t, change in m["keyframes"]:
        end = min(n, max(last + 1, int(math.ceil(t)) + 1))
        for s in range(last, end):
            gap[s] = change
        last = end
    busy = [sum(gap[max(0, t - 2):t + 3]) / 5 for t in range(n)]
    louder = _shares(loud) if heard else [0.0] * n
    busier = _shares(busy)
    score = [weights.get("sound", 0) * a + weights.get("screen", 0) * b for a, b in zip(louder, busier)]
    out = []
    for peak in sorted(range(n), key=lambda s: -score[s]):
        if len(out) >= top:
            break
        start = max(0, min(peak - int(round(window * LEAD)), n - window))
        end = min(n, start + window)
        if any(start < s["end"] and s["start"] < end for s in out):
            continue
        out.append({"start": start, "end": end, "peak": peak, "rise": round(loud[peak], 1),
                    "louder": int(100 * louder[peak]) if heard else None, "busier": int(100 * busier[peak])})
    return out, {"heard": heard, "flat": heard and max(loud) < FLAT_DB, "loudest_rise": round(max(loud), 1)
                 if heard else None, "every": _keyframe_gap(m)}


def loudest(m, start, end):
    """(The second between start and end where the sound rises most, how far it rises in dB), or None when the
    recording has no sound."""
    loud, heard = _loud(m)
    span = range(max(0, int(start)), min(m["seconds"], int(math.ceil(end))))
    if not heard or not span:
        return None
    t = max(span, key=lambda s: loud[s])
    return t, round(loud[t], 1)


def _loud(m):
    """(How far each second's sound rises over the minutes around it, as a 3 s mean in dB; whether there is sound)."""
    n = m["seconds"]
    heard = [v for v in m["sound"] if v is not None]
    floor = min(heard) if heard else QUIET_DB
    sound = [floor if v is None else v for v in m["sound"]]
    usual = _quantile(sound, USUAL)
    rise = [sound[t] - max(usual, statistics.median(sound[max(0, t - AROUND):t + AROUND + 1])) for t in range(n)]
    return [sum(rise[max(0, t - 1):t + 2]) / 3 for t in range(n)], bool(heard)


def _quantile(xs, q):
    ordered = sorted(xs)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _shares(xs):
    """For each value, the share of all the values below it (0 to 1). Ties share the lowest rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out, below = [0.0] * len(xs), 0
    for k, i in enumerate(order):
        if k and xs[i] != xs[order[k - 1]]:
            below = k
        out[i] = below / len(xs)
    return out


def _keyframe_gap(m):
    times = [t for t, _ in m["keyframes"]]
    gaps = [b - a for a, b in zip(times, times[1:])]
    return round(statistics.median(gaps), 1) if gaps else None
