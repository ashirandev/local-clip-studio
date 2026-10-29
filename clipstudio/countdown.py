"""A countdown: the best moments of one recording in the order a viewer counts them down, the best last.

The person asks for a top 5 (or 3, or 4) in their own words; Muse proposes the moments, puts them in that order and
gives the words over each. The program cuts a piece around the peak of each moment and joins the pieces into one
recording the shape of the one they came from. The clip is made from it as from one moment: the layout, the captions,
the render and the checks work on it as on any recording, and each piece carries its rank and words as a title.

Only the person's words open it (asked): a countdown is another clip than the one best moment, so Muse never picks it
by itself (lane 4, 10-29: the best moment asked for in beat 1 must not turn into a top 5).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading

from . import montage, recording, screen

FEWEST, MOST = 3, 5
BEFORE, AFTER = 7.0, 4.0
DEATH_SHOWN = 1.5
LEAD = 3.0
SHORTEST, LONGEST = 3.0, 20.0
JOIN_SECONDS = 600
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "หนึ่ง": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9,
           "สิบ": 10}
NUMBER = "(\\d{1,2}|%s)" % "|".join(NUMBERS)
# "top 5", "Top5", "ท็อป 5", "ทอปห้า", "5 อันดับ", "ห้าอันดับ"; a number is needed, as "ท็อปฟอร์ม" (top form) and
# "อันดับแรก" (first of all) ask for no countdown, and "แล็ปท็อป 5 เครื่อง" (five laptops) is no top 5.
COUNTED = (re.compile("(?:\\btop|(?<!ป)ท[็๊]?อป)\\s*-?\\s*" + NUMBER, re.I), re.compile(NUMBER + "\\s*อันดับ"))
UNCOUNTED = re.compile("\\bcount\\s*-?\\s*down|นับถอยหลัง|จัดอันดับ", re.I)


class CountdownError(Exception):
    """A sentence for Muse: what could not be done and why."""


class Stopped(Exception):
    """The person pressed Stop while the pieces were joined: join() leaves no file."""


def asked(texts):
    """How many moments the person asked to count down in texts, their own messages oldest first: the number of the
    latest that asks for a countdown, 0 when it names none ("นับถอยหลังช่วงเด็ด"), None when none asks."""
    for t in reversed(texts):
        t = " ".join(t.split())
        for pattern in COUNTED:
            found = pattern.search(t)
            if found:
                n = found.group(1).lower()
                return int(n) if n.isdigit() else NUMBERS[n]
        if UNCOUNTED.search(t):
            return 0
    return None


def menu_runs(readings, least):
    """[[start, end]]: the stretches of readings (screen.readings) that show the game's menu, such as a death screen:
    readings one after another with a word of screen.MENU read surely, least or more of them in one. A run starts at
    its first word, at most LEAD s before the screen shows least: Sifu's death screen shows DEATH 1 to 2 s before
    Give Up (10-29), and a hint such as Continue can stay on the screen all through play. It ends at the next
    reading without a word, or at the last reading."""
    runs, run, full = [], None, None
    for r in readings:
        n = len({re.sub("[^A-Za-z]", "", line["text"]).upper() for line in r["lines"]
                 if line["sure"] >= screen.SURE} & set(screen.MENU))
        if n:
            if run is None:
                run, full = [r["at"], r["at"]], None
            run[1] = r["at"]
            if n >= least and full is None:
                full = r["at"]
        elif run is not None:
            if full is not None:
                runs.append([max(run[0], full - LEAD), r["at"]])
            run = None
    if run is not None and full is not None:
        runs.append([max(run[0], full - LEAD), run[1]])
    return runs


def piece(moment, runs=(), earliest=None):
    """[from, to] of the recording for a moment's piece, or None when no play fits before the game's menu. With no
    run of the menu (menu_runs): BEFORE s before the moment's peak (the picture it showed) to AFTER s after, slid
    inside the moment. With runs starting from earliest (the moment's start when None) to the moment's end, the one
    the peak is on, else the first after the peak, else the latest before it: the piece ends DEATH_SHOWN s into that
    run and starts BEFORE + AFTER s before that, not before earliest nor inside the run before, so it shows the play
    that led to the death and the death landing. A game moment's loudest second is often the laugh at the death
    screen after the fight: on the Sifu countdown (10-29) 21 of its 30 pictures 2 s apart were death, skill and age
    screens. A peak in the play between two deaths leads to the later one: moment 2 of the Sifu top 5 (10-30) showed
    35:47, after the death at 35:37, and its fight ended in the death at 35:56."""
    a0, b0 = moment["start"], moment["end"]
    peak = moment["t"] if moment.get("t") is not None else (a0 + b0) / 2.0
    lo = a0 if earliest is None else earliest
    on = [k for k, r in enumerate(runs) if lo <= r[0] <= peak <= r[1]]
    before = [k for k, r in enumerate(runs) if lo <= r[0] <= peak]
    after = [k for k, r in enumerate(runs) if peak < r[0] <= b0]
    if not before and not after:
        a = max(a0, min(peak - BEFORE, b0 - BEFORE - AFTER))
        return [round(a, 2), round(min(b0, a + BEFORE + AFTER), 2)]
    k = on[-1] if on else after[0] if after else before[-1]
    b = runs[k][0] + DEATH_SHOWN
    a = max([lo, b - BEFORE - AFTER] + [r[1] for r in runs[:k]])
    return [round(a, 2), round(b, 2)] if b - a >= SHORTEST else None


def starts(spans):
    """The second of the joined recording each piece starts at."""
    out, at = [], 0.0
    for a, b in spans:
        out.append(round(at, 3))
        at += b - a
    return out


def graph(rec, spans):
    """(FFmpeg's input arguments, the filter graph) joining the spans [[from, to]] of rec in order, at rec's own size
    (made even, as H.264 needs) and montage.FPS, with rec's first sound track at montage.RATE in stereo: the render
    and the hearing read that one. Each piece's sound fades in and out over montage.FADE s, so a cut does not click."""
    w, h = rec["width"] // 2 * 2, rec["height"] // 2 * 2
    args, chains, joined = [], [], ""
    for i, (a, b) in enumerate(spans):
        d = round(b - a, 3)
        args += ["-ss", "%.3f" % a, "-t", "%.3f" % d, "-i", rec["path"]]
        chains.append("[%d:v:0]scale=%d:%d,setsar=1,fps=%d,format=yuv420p,trim=duration=%.3f,setpts=PTS-STARTPTS[v%d]"
                      % (i, w, h, montage.FPS, d, i))
        sound = ("[%d:a:0]aresample=%d,aformat=channel_layouts=stereo,apad" % (i, montage.RATE)
                 if rec.get("sound_tracks") else "anullsrc=r=%d:cl=stereo" % montage.RATE)
        chains.append("%s,atrim=duration=%.3f,afade=t=in:d=%g,afade=t=out:st=%.3f:d=%g,asetpts=PTS-STARTPTS[a%d]"
                      % (sound, d, montage.FADE, max(0.0, d - montage.FADE), montage.FADE, i))
        joined += "[v%d][a%d]" % (i, i)
    chains.append("%sconcat=n=%d:v=1:a=1[v][a]" % (joined, len(spans)))
    return args, ";".join(chains)


def target(rec, spans):
    """Where the countdown of these spans of rec is kept: the same spans in the same order are joined once."""
    key = hashlib.sha1(json.dumps([rec["path"], rec["size"], rec["mtime"], [[round(a, 3), round(b, 3)]
                                                                          for a, b in spans]]).encode("utf-8"))
    name = os.path.splitext(rec["name"])[0]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:40] or "countdown"
    return os.path.join(recording.WORK, "countdown", "%s-%s.mp4" % (safe, key.hexdigest()[:10]))


def join(rec, spans, out, on_start=None):
    """Joins the spans of rec into out and returns out once the file is there and as long as the spans together. The
    file is written next to out and only replaces it when FFmpeg has finished. on_start(stop) is given a function that
    ends FFmpeg at once, after which join raises Stopped and leaves no file. Nothing is written beside it about the
    pieces: an export to an editor (premiere.py) would take that for a montage's upright pieces."""
    if os.path.isfile(out):
        return out
    tool = shutil.which("ffmpeg")
    if not tool:
        raise CountdownError("FFmpeg is not installed, so the moments cannot be joined.")
    args, chain = graph(rec, spans)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    script = out + ".graph"
    with open(script, "w", encoding="utf-8", newline="\n") as f:
        f.write(chain)
    part = out + ".part"
    cmd = [tool, "-hide_banner", "-nostats", "-y"] + args + [
        "-/filter_complex", script, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", montage.PRESET, "-crf",
        str(montage.CRF), "-pix_fmt", "yuv420p", "-r", str(montage.FPS), "-c:a", "aac", "-b:a", "192k", "-ar",
        str(montage.RATE), "-movflags", "+faststart", "-f", "mp4", part]
    stopped = threading.Event()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    except OSError as e:
        raise CountdownError("FFmpeg could not be started to join the moments (%s)." % e)

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
            raise Stopped("Stopped: the moments were not joined.")
        if proc.returncode != 0 or not os.path.isfile(part):
            last = [x for x in err.decode("utf-8", "replace").splitlines() if x.strip()] or ["no reason"]
            raise CountdownError("FFmpeg could not join the moments: %s" % last[-1])
        want = sum(b - a for a, b in spans)
        try:
            got = recording.probe(part)["seconds"]
        except recording.NotOpened as e:
            raise CountdownError("The joined moments could not be read back (%s)." % e)
        if abs(got - want) > 0.2:
            raise CountdownError("The joined moments came out %.2f s long, not %.2f s, so they were not kept."
                                 % (got, want))
        os.replace(part, out)
    finally:
        if os.path.isfile(part):
            os.remove(part)
    return out
