"""Recordings: where they are kept, finding one, and reading what it is. A recording is only ever read.

Nothing is copied or uploaded. A dropped file is found where it already is: a browser does not give the path of a
dropped file, only its name, size and time, so the studio looks for that file in the folders recordings are kept in
(LCS_VIDEOS, else every drive). What the recording is (length, picture size, sound tracks) comes from ffprobe; one picture from it,
made by ffmpeg, is kept in work/frames to show on the page.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("LCS_WORK") or os.path.join(ROOT, "work")
INBOX = os.environ.get("LCS_INBOX") or os.path.join(ROOT, "From phone")
VIDEO_TYPES = (".mp4", ".mkv", ".mov", ".m4v", ".webm", ".ts", ".flv", ".avi")
SMALLEST = 1_000_000
MOST_FILES_LOOKED_AT = 50_000
SWEEP_SECONDS = 60
FRESH_SECONDS = 30
FIXED_DRIVE = 3
SKIP_FOLDERS = {"windows", "program files", "program files (x86)", "programdata", "appdata", "recovery", "perflogs",
                "system volume information", "windowsapps", "node_modules", "site-packages"}
SAME_TIME = 2.0
PROBE_SECONDS = 30
NUMBER = re.compile(r"^\d+(\.\d+)?$")


class NotOpened(Exception):
    """The recording cannot be opened. The sentence says why and what to do."""


def folders():
    """The folders recordings are kept in: LCS_VIDEOS (separated by ; on Windows, : elsewhere), else every drive
    (drives()), so a person who never named a folder finds their videos wherever they keep them. The folder a phone
    sends to (INBOX) is always one of them."""
    named = os.environ.get("LCS_VIDEOS", "")
    found = [f.strip() for f in named.split(os.pathsep) if f.strip()] or drives()
    tops = [os.path.realpath(f) for f in found if os.path.isdir(f)]
    inbox = os.path.realpath(INBOX)
    if os.path.isdir(inbox) and not any(inside(inbox, t) for t in tops):
        tops.append(inbox)
    return tops


def inside(real, top):
    """True when the real path real is top or inside it. A drive root ends in its separator (C:\\), so the
    separator is not added twice: with it doubled, nothing on a drive was inside it (10-29)."""
    return real == top or real.startswith(top.rstrip(os.sep) + os.sep)


def drives():
    """Every fixed drive on Windows (C:\\, D:\\, ...), else the home folder."""
    if os.name != "nt":
        return [os.path.expanduser("~")]
    import ctypes
    import string
    k = ctypes.windll.kernel32
    return [d + ":\\" for i, d in enumerate(string.ascii_uppercase)
            if k.GetLogicalDrives() >> i & 1 and k.GetDriveTypeW(d + ":\\") == FIXED_DRIVE]


THAI_MONTHS = ("มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม", "กันยายน",
               "ตุลาคม", "พฤศจิกายน", "ธันวาคม")


def walk(words="", fresh=False):
    """Every recording in the folders whose path or recorded day holds all the words: [(mtime, size, path)], newest
    first. The day is written the ways a person asks for it (2026-10-12, 12 Oct, October 12, ตุลาคม, Monday): Muse
    sees "recorded Mon 12 Oct" in every list, and on 10-28 it took 11 searches to find a recording by its day when
    only the names were searched. When the folders are every drive, one sweep is kept FRESH_SECONDS (fresh: sweep
    again now)."""
    wanted = words.lower().split()
    tops = folders()
    every = not os.environ.get("LCS_VIDEOS", "").strip()
    if not every or fresh or _swept.get("tops") != tops or time.time() - _swept.get("at", 0) > FRESH_SECONDS:
        _swept.update(sweep(tops), tops=tops, at=time.time())
    return [(m, s, p) for m, s, p, top in _swept["found"]
            if not wanted or all(w in os.path.relpath(p, top).lower() + " " + _day(m) for w in wanted)]


_swept = {}


def sweep(tops):
    """{"found": [(mtime, size, path, top)] newest first, "complete": False when it stopped after SWEEP_SECONDS}:
    every recording of SMALLEST or more in tops, leaving out system and program folders (SKIP_FOLDERS), hidden ones
    and the studio's own work folder (its clips are not recordings). Reads names and sizes only; measured 10-28 on
    five drives: 4.8 s for 413,000 files."""
    found, started, work = [], time.time(), os.path.normcase(os.path.realpath(WORK))
    for top in tops:
        for here, dirs, files in os.walk(top):
            if time.time() - started > SWEEP_SECONDS:
                found.sort(reverse=True)
                return {"found": found, "complete": False}
            dirs[:] = [d for d in dirs if not d.startswith(("_", ".", "$")) and d.lower() not in SKIP_FOLDERS
                       and os.path.normcase(os.path.join(here, d)) != work]
            for name in files:
                if not name.lower().endswith(VIDEO_TYPES):
                    continue
                path = os.path.join(here, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size >= SMALLEST:
                    found.append((st.st_mtime, st.st_size, path, top))
    found.sort(reverse=True)
    return {"found": found, "complete": True}


def _day(mtime):
    t = time.localtime(mtime)
    return (time.strftime("%Y-%m-%d %d %b %B %b %d %B %d %a %A", t) + " " + THAI_MONTHS[t.tm_mon - 1]).lower()


MUSIC_TYPES = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma") + VIDEO_TYPES
MOST_MUSIC_SHOWN = 8


def music_folders():
    """The folders the person's music may come from: LCS_MUSIC (separated like LCS_VIDEOS), else ~/Music, and the
    folders recordings are kept in. FFmpeg opens whatever path it is given, so no other file can be named."""
    named = os.environ.get("LCS_MUSIC", "")
    found = [f.strip() for f in named.split(os.pathsep) if f.strip()] or [os.path.expanduser("~/Music")]
    tops = [os.path.realpath(f) for f in found if os.path.isdir(f)] + folders()
    return list(dict.fromkeys(tops))


def music_files(words=""):
    """The music and sound files in the music folders whose path holds all the words: [path], newest first."""
    wanted = words.lower().split()
    found, looked = [], 0
    for top in music_folders():
        for here, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if not d.startswith(("_", "."))]
            for name in files:
                looked += 1
                path = os.path.join(here, name)
                if name.lower().endswith(MUSIC_TYPES) and all(w in os.path.relpath(path, top).lower() for w in wanted):
                    try:
                        found.append((os.stat(path).st_mtime, path))
                    except OSError:
                        pass
            if looked > MOST_FILES_LOOKED_AT:
                break
    return [p for _, p in sorted(set(found), reverse=True)]


def find_music(name):
    """The real path of the person's music: name is its path, or words from its name that fit exactly one file.
    Otherwise NotOpened, naming the files that fit, or the newest ones."""
    tops = music_folders()
    where = ", ".join(tops) or "none"
    if os.path.isabs(name) or os.sep in name or "/" in name:
        real = os.path.realpath(name)
        if not any(inside(real, top) for top in tops):
            raise NotOpened("%s is not inside the folders music may come from (%s). Ask the person to put it in one "
                            "of them." % (name, where))
        if not real.lower().endswith(MUSIC_TYPES) or not os.path.isfile(real):
            raise NotOpened("%s is not a music or sound file on this computer." % name)
        return real
    hits = music_files(name)
    if len(hits) == 1:
        return hits[0]
    shown = hits or music_files()
    listed = "".join("\n- %s" % p for p in shown[:MOST_MUSIC_SHOWN])
    if hits:
        raise NotOpened('%d files fit "%s"; name one by its path:%s' % (len(hits), name, listed))
    if shown:
        raise NotOpened('No music file fits "%s" in %s. The newest there:%s' % (name, where, listed))
    raise NotOpened("There is no music or sound file in %s. The studio has no music of its own: ask the person for "
                    "theirs." % where)


def sound_seconds(path):
    """How long a music or sound file plays, in seconds, or None when ffprobe cannot tell."""
    tool = shutil.which("ffprobe")
    try:
        out = subprocess.run([tool, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                             capture_output=True, timeout=PROBE_SECONDS, check=True).stdout
        return float(json.loads(out.decode("utf-8", "replace"))["format"]["duration"])
    except (subprocess.SubprocessError, ValueError, KeyError, OSError, TypeError):
        return None


def locate(name, size, mtime=None):
    """The path of a dropped file, found by its name and size (and its time, when two files share both)."""
    for fresh in (False, True):
        hits = [(abs((m - mtime) if mtime else 0), p) for m, s, p in walk(fresh=fresh)
                if os.path.basename(p).lower() == name.lower() and s == size]
        if hits:
            break
    if not hits:
        raise NotOpened("%s is not in the folders recordings are kept in (%s). Move it into one of them, or add its "
                        "folder to LCS_VIDEOS and start the studio again." % (name, ", ".join(folders()) or "none"))
    hits.sort()
    if mtime and hits[0][0] > SAME_TIME and len(hits) > 1:
        raise NotOpened("There are %d files named %s with that size, and none has the dropped file's time. Open it "
                        "from the list of newest recordings instead." % (len(hits), name))
    return hits[0][1]


def check_path(path):
    """The real path when it is a recording inside the folders, else NotOpened. The page cannot name other files."""
    real = os.path.realpath(path)
    if not any(inside(real, top) for top in folders()):
        raise NotOpened("%s is not inside the folders recordings are kept in (%s)." % (path, ", ".join(folders())))
    if not real.lower().endswith(VIDEO_TYPES) or not os.path.isfile(real):
        raise NotOpened("%s is not a recording on this computer." % path)
    return real


def check_folder(path):
    """The real path when it is a folder inside the folders recordings are kept in, else NotOpened."""
    real = os.path.realpath(path)
    if not any(inside(real, top) for top in folders()):
        raise NotOpened("%s is not inside the folders recordings are kept in (%s)." % (path, ", ".join(folders())))
    if not os.path.isdir(real):
        raise NotOpened("%s is not a folder on this computer." % path)
    return real


def clips_in(folder):
    """The recordings directly in folder, by name."""
    return sorted(os.path.join(folder, n) for n in os.listdir(folder)
                  if n.lower().endswith(VIDEO_TYPES) and os.path.isfile(os.path.join(folder, n)))


def probe(path):
    """{path, name, size, mtime, seconds, width, height, fps, sound_tracks, codec} as the viewer sees the picture."""
    tool = shutil.which("ffprobe")
    if not tool:
        raise NotOpened("FFmpeg is not installed, so the studio cannot read recordings. Install FFmpeg (ffmpeg and "
                        "ffprobe on the PATH), then start the studio again.")
    try:
        out = subprocess.run([tool, "-v", "error", "-show_entries",
                              "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate"
                              ":stream_side_data=rotation", "-of", "json", path],
                             capture_output=True, timeout=PROBE_SECONDS, check=True).stdout
        info = json.loads(out.decode("utf-8", "replace"))
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        raise NotOpened("%s could not be read as a video (%s)." % (os.path.basename(path), _first_line(e)))
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    seconds = float((info.get("format") or {}).get("duration") or 0)
    if not video or seconds <= 0:
        raise NotOpened("%s has no picture or no length, so it is not a recording the studio can use."
                        % os.path.basename(path))
    width, height = video.get("width") or 0, video.get("height") or 0
    turn = next((d.get("rotation") for d in video.get("side_data_list") or [] if "rotation" in d), 0)
    if abs(int(turn or 0)) % 180 == 90:
        width, height = height, width
    st = os.stat(path)
    return {"path": path, "name": os.path.basename(path), "size": st.st_size, "mtime": st.st_mtime,
            "seconds": round(seconds, 2), "width": width, "height": height, "fps": _fps(video.get("r_frame_rate")),
            "sound_tracks": sum(1 for s in streams if s.get("codec_type") == "audio"),
            "codec": video.get("codec_name") or ""}


def frame(rec, at=None):
    """One picture at second at (default: a tenth of the way in), 540 pixels on its long side: the file name in
    work/frames, or None."""
    tool = shutil.which("ffmpeg")
    if not tool:
        return None
    at = rec["seconds"] / 10 if at is None else at
    key = hashlib.sha1(("%s|%d|%s|%.2f" % (rec["path"], rec["size"], rec["mtime"], at)).encode("utf-8")
                       ).hexdigest()[:16]
    folder = os.path.join(WORK, "frames")
    os.makedirs(folder, exist_ok=True)
    name = key + ".jpg"
    target = os.path.join(folder, name)
    if not os.path.isfile(target):
        temp = os.path.join(folder, "%s.%d.part.jpg" % (key, threading.get_ident()))
        try:
            subprocess.run([tool, "-v", "error", "-y", "-ss", "%.2f" % at, "-i", rec["path"],
                            "-frames:v", "1", "-vf", "scale='if(gt(iw,ih),540,-2)':'if(gt(iw,ih),-2,540)'",
                            temp], capture_output=True, timeout=PROBE_SECONDS, check=True)
            try:
                os.replace(temp, target)
            except PermissionError:
                if not os.path.isfile(target):
                    raise
        except (subprocess.SubprocessError, OSError):
            return None
        finally:
            if os.path.exists(temp):
                os.remove(temp)
    return name


def describe(rec):
    """One line about an open recording, for Muse and for the page."""
    return "%s · %s long · %dx%d%s · %d sound track%s · %s · recorded %s · %s" % (
        rec["name"], clock(rec["seconds"]), rec["width"], rec["height"],
        " %g fps" % rec["fps"] if rec["fps"] else "", rec["sound_tracks"], "" if rec["sound_tracks"] == 1 else "s",
        size_text(rec["size"]), when(rec["mtime"]), rec["path"])


def clock(seconds):
    s = int(round(seconds))
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60) if s >= 3600 else "%d:%02d" % (s // 60, s % 60)


def seconds_of(text):
    """Seconds from a time written as clock() writes it (1:51:44, 35:43) or as plain seconds (6704, 6704.5);
    None when it is not a time."""
    parts = str(text).strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(NUMBER.match(p) for p in parts):
        return None
    numbers = [float(p) for p in parts]
    if any(n >= 60 for n in numbers[1:]):
        return None
    total = 0.0
    for n in numbers:
        total = total * 60 + n
    return total


def size_text(n):
    return "%.1f GB" % (n / 1e9) if n >= 1e9 else "%d MB" % round(n / 1e6)


def when(t):
    fmt = "%a %d %b %H:%M" if time.localtime(t).tm_year == time.localtime().tm_year else "%d %b %Y %H:%M"
    return time.strftime(fmt, time.localtime(t))


def _fps(rate):
    try:
        top, _, bottom = (rate or "").partition("/")
        return round(float(top) / float(bottom or 1), 2) if float(top) else 0
    except ValueError:
        return 0


def _first_line(e):
    text = getattr(e, "stderr", None)
    text = text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text or e)
    return (text.strip().splitlines() or [str(e)])[0][:200]
