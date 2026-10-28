"""Reading the screen: the words on a recording's picture, read at full size by RapidOCR on the processor, and the
private ones among them, found by patterns, never by a model's guess: an email address, a phone number, a card or ID
number, a key or an account number, a password written after its label.

A stretch is looked at CHECKS times a second on a small grey picture. It is read at its first picture, then again when
the picture has changed since the last reading, or FORCE seconds after it, never twice within GAP seconds: a screen
mostly holds still, and reading one picture takes a second or two. Which pictures are read depends on the pictures
alone, so READERS of them are read at once (measured 10-27 on 28 processor threads: 2.8 s a picture one at a time,
0.85 s four at a time). What was read is kept per file and stretch in work/screen, so reading it again is instant.
The proposed moments are read ahead, one at a time in a thread of their own, while Muse goes on working: a clip made
from one of them finds it read, or waits for the reading under way instead of starting another.

The words a stretch shows are also what names are checked against: a speech recogniser mishears a name such as
Eaglet, and the screen has it written. Only words from lines that hold nothing private are given out, so a private
word never reaches Muse.

A private line is hidden whole: the reader gives a box per line, not per letter, and the whole line is the box that is
sure to cover it. It is hidden from the reading before the first one that saw it until the first one that no longer
does, since it came and went somewhere in between.
"""
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import recording

VERSION = 1
LONG_SIDE = 1920
CHECKS = 2
SMALL = (192, 108)
LEVEL = 24
CHANGED = 0.02
GAP = 1.0
FORCE = 3.0
PAD = 0.35
SAME = 0.5
READERS = 4
SURE = 0.9
MOST_WORDS = 40
WORDS_FROM = 3
NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.-][A-Za-z0-9]+)*")
ENDINGS = {"com", "net", "org", "io", "ai", "co", "th", "dev", "app", "me", "tv", "gg", "info", "shop", "site", "xyz",
           "us", "uk", "jp"}
NOT_NAMES = ENDINGS | {"www"}
_SITES = threading.Lock()
_lock = threading.Lock()
_reader = []
_now = {}
_todo = queue.Queue()
_worker = []

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
HANDLE = re.compile(r"(?<![A-Za-z0-9._%+-])@[A-Za-z0-9][A-Za-z0-9._-]{2,}")
DIGITS = re.compile(r"(?<![A-Za-z0-9/._#=&?])\+?\d[\d -]{7,}\d(?![A-Za-z0-9/_#&])")
TOKEN = re.compile(r"[A-Za-z0-9_-]{16,}")
PASSWORD = re.compile(r"(?i)\b(?:pass(?:word|wd|code)?|secret|token|api[ _-]?key)\b\s*[:=]\s*[^\s:=]{4,}")
PREFIXES = ("sk-", "sk_", "pk_", "rk_", "hf_", "ghp_", "gho_", "github_pat_", "xoxb-", "xoxp-", "AKIA", "AIza", "eyJ",
            "whsec_")
KEY_LONG = 16
KEY_MIXED = 6
KINDS = ("an email address", "a card number", "an ID number", "a phone number", "a key or an account number",
         "a password", "a person's handle")


class ScreenError(Exception):
    """The screen could not be read. The sentence says why."""


class Stopped(Exception):
    """The person pressed Stop while the screen was being read."""


def private(text):
    """Which of KINDS a line of the screen holds, or None."""
    if EMAIL.search(text):
        return "an email address"
    for run in DIGITS.findall(text):
        digits = re.sub(r"\D", "", run)
        groups = [len(g) for g in re.split(r"[ -]+", run.lstrip("+"))]
        if len(digits) == 13 and groups in ([13], [1, 4, 5, 2, 1]) and _thai_id(digits):
            return "an ID number"
        if 13 <= len(digits) <= 19 and (len(groups) == 1 or all(g == 4 for g in groups[:-1])) and _luhn(digits):
            return "a card number"
        if (run.startswith("0") and 9 <= len(digits) <= 10) or (run.startswith("+") and 10 <= len(digits) <= 13):
            return "a phone number"
    for token in TOKEN.findall(text):
        if token.startswith(PREFIXES) or any(len(part) >= KEY_LONG and _mixed(part) >= KEY_MIXED
                                             for part in re.split(r"[-_]", token)):
            return "a key or an account number"
    if PASSWORD.search(text):
        return "a password"
    if HANDLE.search(text):
        return "a person's handle"
    return None


def _mixed(part):
    """How often a run of letters and digits switches between the two: a key is random, so it switches often; a
    model's name or a misread line of Thai switches once or twice."""
    return sum(1 for x, y in zip(part, part[1:]) if x.isdigit() != y.isdigit())


def _luhn(digits):
    total = 0
    for k, d in enumerate(int(c) for c in reversed(digits)):
        total += d if k % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2)
    return total % 10 == 0


def _thai_id(digits):
    return (11 - sum(int(d) * (13 - k) for k, d in enumerate(digits[:12])) % 11) % 10 == int(digits[12])


def _engine():
    """The one RapidOCR the studio keeps; it reads several pictures at once."""
    with _lock:
        if not _reader:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError:
                raise ScreenError("RapidOCR is not installed, so the screen cannot be read. Run: pip install -r "
                                  "requirements.txt, then start the studio again.")
            _reader.append(RapidOCR())
    return _reader[0]


def _read(picture):
    """[(corner points, text, sure)] for one picture."""
    found, _ = _engine()(picture, use_cls=False)
    return found or []


def _key(rec, a, b):
    return hashlib.sha1(("%d|%s|%d|%s|%.2f|%.2f" % (VERSION, rec["path"], rec["size"], rec["mtime"], a, b))
                        .encode("utf-8")).hexdigest()[:16]


def kept(rec, a, b):
    """The readings kept for this stretch, or None."""
    try:
        with open(os.path.join(recording.WORK, "screen", _key(rec, a, b) + ".json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _keep(rec, a, b, readings_):
    folder = os.path.join(recording.WORK, "screen")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, _key(rec, a, b) + ".json")
    with open(target + ".part", "w", encoding="utf-8") as f:
        json.dump(readings_, f, ensure_ascii=False)
    os.replace(target + ".part", target)


def ahead(rec, a, b):
    """Queues a to b to be read in the background. Never raises: a stretch that cannot be read there is read again
    when it is asked for."""
    _todo.put((rec, a, b))
    with _lock:
        if not _worker:
            _worker.append(threading.Thread(target=_work, name="screen-ahead", daemon=True))
            _worker[0].start()


def _work():
    while True:
        rec, a, b = _todo.get()
        try:
            readings(rec, a, b)
        except Exception:
            pass


def readings(rec, a, b, on_start=None):
    """[{"at": recording second, "lines": [{"text", "box": [x, y, w, h] in pixels of the recording, "sure"}]}] from a
    to b, in time order. A stretch being read already is waited for, not read twice. on_start(stop) is given a function
    that ends the reading or the wait at once (then Stopped is raised)."""
    key = _key(rec, a, b)
    while True:
        found = kept(rec, a, b)
        if found is not None:
            return found
        with _lock:
            busy = _now.get(key)
            if busy is None:
                mine = _now[key] = threading.Event()
        if busy is None:
            break
        gave_up = []
        if on_start:
            on_start(lambda: gave_up.append(True))
        while not busy.wait(0.25):
            if gave_up:
                raise Stopped()
    try:
        return _read_stretch(rec, a, b, on_start)
    finally:
        with _lock:
            _now.pop(key, None)
        mine.set()


def _read_stretch(rec, a, b, on_start):
    tool = shutil.which("ffmpeg")
    if not tool:
        raise ScreenError("FFmpeg is not installed, so the screen cannot be read.")
    scale = min(1.0, LONG_SIDE / max(rec["width"], rec["height"]))
    w, h = max(2, int(rec["width"] * scale) // 2 * 2), max(2, int(rec["height"] * scale) // 2 * 2)
    proc = subprocess.Popen([tool, "-v", "error", "-ss", "%.2f" % a, "-t", "%.2f" % (b - a), "-i", rec["path"],
                             "-map", "0:v:0", "-vf", "fps=%d,scale=%d:%d" % (CHECKS, w, h), "-f", "rawvideo",
                             "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    stopped = []

    def stop():
        stopped.append(True)
        proc.kill()

    if on_start:
        on_start(stop)
    picked, last, size, k = [], None, w * h * 3, 0
    back = (rec["width"] / float(w), rec["height"] / float(h))
    room = threading.BoundedSemaphore(READERS * 2)
    pool = ThreadPoolExecutor(READERS, thread_name_prefix="screen")
    try:
        while True:
            data = proc.stdout.read(size)
            if len(data) < size or stopped:
                break
            at, k = a + k / float(CHECKS), k + 1
            picture = np.frombuffer(data, np.uint8).reshape(h, w, 3)
            small = cv2.cvtColor(cv2.resize(picture, SMALL, interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2GRAY).astype(np.int16)
            if last is not None:
                since = at - last[0]
                changed = float((np.abs(small - last[1]) > LEVEL).mean())
                if since < GAP or changed < CHANGED and (since < FORCE or changed == 0):
                    continue
            room.acquire()
            reading = pool.submit(_read, picture)
            reading.add_done_callback(lambda _: room.release())
            picked.append((at, reading))
            last = (at, small)
    finally:
        proc.stdout.close()
        proc.wait()
        pool.shutdown(wait=True, cancel_futures=bool(stopped))
    if stopped:
        raise Stopped()
    out = [{"at": round(at, 2), "lines": _lines(reading.result(), back)} for at, reading in picked]
    _keep(rec, a, b, out)
    return out


def _lines(found, back):
    out = []
    for points, text, sure in found:
        xs, ys = [p[0] * back[0] for p in points], [p[1] * back[1] for p in points]
        out.append({"text": text, "box": [round(min(xs)), round(min(ys)), round(max(xs) - min(xs)),
                                          round(max(ys) - min(ys))], "sure": round(float(sure), 2)})
    return out


def _pictures(rec, times):
    """Readings of single pictures at these recording seconds, read at once."""
    tool = shutil.which("ffmpeg")
    if not tool:
        raise ScreenError("FFmpeg is not installed, so the screen cannot be read.")
    scale = min(1.0, LONG_SIDE / max(rec["width"], rec["height"]))
    w, h = max(2, int(rec["width"] * scale) // 2 * 2), max(2, int(rec["height"] * scale) // 2 * 2)
    back = (rec["width"] / float(w), rec["height"] / float(h))

    def one(at):
        data = subprocess.run([tool, "-v", "error", "-ss", "%.2f" % at, "-i", rec["path"], "-map", "0:v:0",
                               "-frames:v", "1", "-vf", "scale=%d:%d" % (w, h), "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "pipe:1"], capture_output=True).stdout
        if len(data) < w * h * 3:
            return {"at": round(at, 2), "lines": []}
        return {"at": round(at, 2), "lines": _lines(_read(np.frombuffer(data[:w * h * 3], np.uint8)
                                                          .reshape(h, w, 3)), back)}

    with ThreadPoolExecutor(READERS, thread_name_prefix="screen") as pool:
        return list(pool.map(one, times))


def words(rec, a, b):
    """The words written on the screen from a to b that may be names, most seen first: Latin words of three letters
    or more, read surely, from lines that hold nothing private. From the stretch's readings when it was read, else from
    WORDS_FROM pictures read now."""
    target = os.path.join(recording.WORK, "screen", _key(rec, a, b) + ".words.json")
    try:
        with open(target, encoding="utf-8") as f:
            found = json.load(f)
        _keep_sites(rec, found)
        return found
    except (OSError, ValueError):
        pass
    read = kept(rec, a, b)
    if read is None:
        read = _pictures(rec, [a + (b - a) * (k + 0.5) / WORDS_FROM for k in range(WORDS_FROM)])
    count = {}
    for r in read:
        for line in r["lines"]:
            if line["sure"] < SURE or private(line["text"]):
                continue
            for word in NAME.findall(line["text"]):
                if len(word) >= 3:
                    count[word] = count.get(word, 0) + 1
    out = [w for w, _ in sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))][:MOST_WORDS]
    _keep_sites(rec, count)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target + ".part", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(target + ".part", target)
    return out


def _site(word):
    """Whether word is written as a web address, labels joined by dots that end in a known ending (ashiran.com,
    th.jobsdb.com); not a file name (CV.pdf, llama.cpp)."""
    parts = word.lower().split(".")
    return len(parts) >= 2 and parts[-1] in ENDINGS and all(parts)


def _sites_file(rec):
    return os.path.join(recording.WORK, "screen", hashlib.sha1(("sites|%d|%s|%d|%s" % (
        VERSION, rec["path"], rec["size"], rec["mtime"])).encode("utf-8")).hexdigest()[:16] + ".sites.json")


def _read_sites(rec):
    try:
        with open(_sites_file(rec), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _keep_sites(rec, found):
    """Adds the web addresses among found to the recording's own list."""
    new = [w for w in found if _site(w)]
    if not new:
        return
    with _SITES:
        known = _read_sites(rec)
        more = [w for w in dict.fromkeys(new) if w not in known]
        if not more:
            return
        target = _sites_file(rec)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target + ".part", "w", encoding="utf-8") as f:
            json.dump(known + more, f, ensure_ascii=False)
        os.replace(target + ".part", target)


def sites(rec):
    """The web addresses read on the screen anywhere in the recording so far, each followed by its names (ashiran.com,
    ashiran): an address is spelled one way, so a name said in a moment is written as the screen writes it even when
    the address shows at another second. Only addresses: measured 10-28 against every word read on every screen, the
    plain words (Content, Company, roles) would have changed Context, Compact and homes; the addresses changed 7 words,
    all rightly (ashilan.do.com, Achilan, JobDB)."""
    out = []
    for address in _read_sites(rec):
        for w in [address] + address.split(".")[:-1]:
            if len(w) >= 4 and w.lower() not in NOT_NAMES and w not in out:
                out.append(w)
    return out


def _padded(box, rec):
    """A line's box with room around it, kept inside the picture."""
    x, y, w, h = box
    pad = max(2.0, h * PAD)
    x1, y1 = max(0.0, x - pad), max(0.0, y - pad)
    x2, y2 = min(float(rec["width"]), x + w + pad), min(float(rec["height"]), y + h + pad)
    return [round(x1), round(y1), round(x2 - x1), round(y2 - y1)]


def _overlap(a, b):
    """How much of the smaller box the two share, from 0 to 1."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    smaller = min(a[2] * a[3], b[2] * b[3])
    return 0.0 if w <= 0 or h <= 0 or smaller <= 0 else w * h / float(smaller)


def _union(a, b):
    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
    return [x1, y1, max(a[0] + a[2], b[0] + b[2]) - x1, max(a[1] + a[3], b[1] + b[3]) - y1]


def _within(read, a, b):
    """The readings of a stretch that hold for a to b inside it: the one showing at a, moved to a, and those after."""
    before = [r for r in read if r["at"] <= a]
    return ([dict(before[-1], at=a)] if before else []) + [r for r in read if a < r["at"] <= b]


def hidden(rec, pieces, on_start=None, span=None):
    """[{"kind", "box": [x, y, w, h] in pixels of the recording, with room around the line, "start", "end": clip
    seconds}]: each private line the clip's pieces show, and for how long. A line that moves a little between two
    readings is one line, its box grown to cover both places. span is the stretch the pieces were cut from, read once
    (and read ahead while the moments were proposed), so a clip with its silences cut is not read piece by piece."""
    whole = readings(rec, span[0], span[1], on_start) if span else None
    out, at = [], 0.0
    for a, b in pieces:
        before, shown = at, []
        for r in (_within(whole, a, b) if span else readings(rec, a, b, on_start)):
            t = at + r["at"] - a
            seen = [(private(line["text"]), _padded(line["box"], rec)) for line in r["lines"]]
            seen = [s for s in seen if s[0]]
            still = []
            for item in shown:
                match = next((s for s in seen if s[0] == item["kind"] and _overlap(s[1], item["box"]) >= SAME), None)
                if match:
                    seen.remove(match)
                    still.append(dict(item, box=_union(item["box"], match[1])))
                else:
                    out.append(dict(item, end=round(t, 2)))
            shown = still + [{"kind": kind, "box": box, "start": round(before, 2)} for kind, box in seen]
            before = t
        out += [dict(item, end=round(at + b - a, 2)) for item in shown]
        at += b - a
    return sorted(out, key=lambda item: (item["start"], item["box"][1]))


def covered(item, blurs):
    """True when one fixed blur hides the whole of a private line for the whole time it shows."""
    x, y, w, h = item["box"]
    return any(b.get("box") and b["box"][0] <= x and b["box"][1] <= y and b["box"][0] + b["box"][2] >= x + w and
               b["box"][1] + b["box"][3] >= y + h and b["start"] <= item["start"] + 0.01 and
               b["end"] >= item["end"] - 0.01 for b in blurs)
