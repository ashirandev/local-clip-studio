"""Rough previews: each proposed moment cut to a small video the page can play, while Muse goes on working.

A preview is 480 pixels on its short side, at most 30 frames a second, H.264 with AAC sound, with its index at the
front so the page can play it at once. It is rough on purpose: it shows which stretch a card means, and the clip
itself is made later from the recording. One preview is cut at a time, best moment first, in a thread of its own, so
the cards show at once and each one plays when its preview is ready. Previews are kept in work/previews by recording
and times, so the same stretch is cut once.
"""
import hashlib
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

from . import recording, render

SHORT_SIDE = 480
MOST_FPS = 30
CUT_SECONDS = 300
_todo = queue.Queue()
_worker = []
_lock = threading.Lock()
_small_lock = threading.Lock()


def folder():
    return os.path.join(recording.WORK, "previews")


def name(rec, start, end):
    key = "%s|%d|%s|%.2f|%.2f" % (rec["path"], rec["size"], rec["mtime"], start, end)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".mp4"


def cut(chat, job, rec, start, end):
    """Queues a preview of start to end and returns its name. The chat gets a "preview" event when it is ready, or
    with a sentence when it could not be cut."""
    preview = name(rec, start, end)
    _todo.put((chat, job, rec, start, end, preview))
    with _lock:
        if not _worker:
            _worker.append(threading.Thread(target=_work, name="previews", daemon=True))
            _worker[0].start()
    return preview


def _work():
    """Cuts the queued previews one by one, for as long as the studio runs. A preview that fails in a way nobody
    can be told about (its chat's files are gone) is skipped, so it never stops the previews after it."""
    while True:
        chat, job, rec, start, end, preview = _todo.get()
        try:
            started = time.time()
            target = os.path.join(folder(), preview)
            error = None if os.path.isfile(target) else _cut(rec, start, end, target)
            chat.emit("preview", job=job, name=preview, ok=error is None, error=error,
                      seconds=round(time.time() - started, 1))
        except Exception as e:
            print("previews: %s could not be told about (%s)" % (preview, e), file=sys.stderr, flush=True)


def small(path, name):
    """(the folder, the file name) of a finished clip's copy at a preview's size, made the first time it is asked for, for a
    phone that reviews it over the network; None when it could not be made. Tarm 11-01: 1080 by 1920 is too much on his
    phone even on Wi-Fi."""
    little = "small-" + name
    with _small_lock:
        if os.path.isfile(os.path.join(folder(), little)) or not _cut({"path": path}, 0.0, CUT_SECONDS, os.path.join(folder(), little)):
            return folder(), little
    return None


def _cut(rec, start, end, target):
    """None when the preview is written, else a sentence saying why not. It is written to a part file first, so a
    preview is either whole or not there."""
    tool = shutil.which("ffmpeg")
    if not tool:
        return "FFmpeg is not installed, so no preview could be cut."
    os.makedirs(folder(), exist_ok=True)
    part = "%s.%d-%d.part.mp4" % (target[:-4], os.getpid(), threading.get_ident())
    try:
        subprocess.run([tool, "-v", "error", "-y", "-ss", "%.2f" % start, "-i", rec["path"], "-t", "%.2f" % (end - start),
                        "-map", "0:v:0", "-map", "0:a:0?",
                        "-vf", "scale='if(gt(iw,ih),-2,%d)':'if(gt(iw,ih),%d,-2)'" % (SHORT_SIDE, SHORT_SIDE),
                        "-fpsmax", str(MOST_FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-movflags", "+faststart",
                        part], capture_output=True, timeout=CUT_SECONDS, check=True)
        if os.path.getsize(part) == 0:
            return "ffmpeg wrote an empty preview of %s-%s." % (recording.clock(start), recording.clock(end))
        render.settle(part, target)
        return None
    except subprocess.TimeoutExpired:
        return "Cutting the preview of %s-%s took over %d s, so it was stopped." % (
            recording.clock(start), recording.clock(end), CUT_SECONDS)
    except (subprocess.SubprocessError, OSError) as e:
        return "ffmpeg could not cut %s-%s (%s)." % (recording.clock(start), recording.clock(end),
                                                    recording._first_line(e))
    finally:
        if os.path.exists(part):
            os.remove(part)
