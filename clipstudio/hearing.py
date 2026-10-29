"""Hearing: what is said in a stretch of the recording, written down by Meta's Omnilingual ASR, then misheard words
fixed by Muse from the context, with every number locked.

The hearing worker (ears.py) runs inside the studio, on the processor, with onnxruntime: no Docker. The studio starts
it when the studio starts, so the model is loaded before Muse is asked anything. Its model is in MODELS
(LCS_HEARING_MODELS, else work/models/hearing): Omnilingual's CTC model as ONNX, made once by
hearing/export_onnx.py, with letters.json. LCS_HEARING_URL points at a hearing worker run some other way; then the
studio does not start one.

What was heard is kept per file and stretch in work/heard, so hearing a stretch again is instant. Each phrase has
its own time in the recording, from the model's own frames.

Names: a recogniser mishears a name it never learned (Eaglet came out as "egret"). When the screen shows words at
the time, they are given with the question, and a line may change by as much more as the names it takes from them.

The fix: Muse is asked, apart from the job's conversation (llama-server keeps that conversation's cache, measured
10-27: 2,136 tokens reused after a question in between), to write each line as it was most likely said. The numbers
are taken out first (<N1>, <N2>...), so Muse can only carry them back, never change them. A line keeps what was
heard when its answer loses or adds a number, or changes more than MOST_CHANGED of it. Lines are asked a few at a
time, and the lines whose answer failed are asked once more.
"""
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.parse

from . import ears, mind, recording, thai

VERSION = 2
OWN_URL = os.environ.get("LCS_HEARING_URL", "")
URL = OWN_URL or "http://127.0.0.1:8802"
MODELS = os.environ.get("LCS_HEARING_MODELS") or os.path.join(recording.WORK, "models", "hearing")
HEAR_SECONDS = 600
LINES_PER_QUESTION = 12
MOST_CHANGED = 0.4
FIX_TOKENS = 3000
_state = {"why": None}


class HearingError(Exception):
    """Nothing could be heard. The sentence says why."""


class Stopped(Exception):
    """The person pressed Stop while the stretch was being heard."""


def start():
    """Starts the hearing worker in the background unless one already answers. Never raises: status() says what is
    wrong."""
    threading.Thread(target=_start, name="hearing-start", daemon=True).start()


def _start():
    if OWN_URL or _health() is not None:
        return
    if not ears.model_file(MODELS) or not os.path.isfile(os.path.join(MODELS, "letters.json")):
        _state["why"] = ("The hearing model is not downloaded yet, so the studio cannot hear what is said: it goes in "
                         "%s (%s and letters.json)." % (MODELS, " or ".join(ears.MODEL_FILES)))
        return
    u = urllib.parse.urlsplit(URL)
    try:
        ears.serve(MODELS, u.hostname, u.port)
    except OSError as e:
        _state["why"] = "The hearing worker could not start on %s (%s)." % (URL, e)


def _health():
    """The worker's /health answer, or None when nothing answers."""
    if not mind.listens(URL):
        return None
    try:
        conn = http.client.HTTPConnection(urllib.parse.urlsplit(URL).hostname, urllib.parse.urlsplit(URL).port,
                                          timeout=3)
        conn.request("GET", "/health")
        return json.loads(conn.getresponse().read().decode("utf-8"))
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        conn.close()


def status():
    """{ok, label, where, url, name} or {ok: False, ..., why} with a sentence."""
    out = {"label": "Hearing", "where": "Omnilingual ASR on the processor", "url": URL}
    got = _health()
    if got and got.get("ok"):
        return dict(out, ok=True, name="%s, %s, %s threads" % (got.get("model"), got.get("dtype"), got.get("threads")))
    if got and got.get("loading"):
        return dict(out, ok=False, why="The hearing model is loading; this takes 2 to 10 s.")
    if got and got.get("error"):
        return dict(out, ok=False, why="The hearing model could not load (%s)." % got["error"])
    return dict(out, ok=False, why=_state["why"] or "The hearing worker is not answering at %s." % URL)


def key(rec, start, end):
    st = os.stat(rec["path"])
    return hashlib.sha1(("%d|%s|%d|%s|%.2f|%.2f" % (VERSION, rec["path"], st.st_size, st.st_mtime, start, end))
                        .encode("utf-8")).hexdigest()[:16]


def heard(rec, start, end):
    """What was heard and kept for this stretch, or None."""
    try:
        with open(os.path.join(recording.WORK, "heard", key(rec, start, end) + ".json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def keep(rec, start, end, said):
    folder = os.path.join(recording.WORK, "heard")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, key(rec, start, end) + ".json")
    with open(target + ".part", "w", encoding="utf-8") as f:
        json.dump(said, f, ensure_ascii=False)
    os.replace(target + ".part", target)


def hear(rec, start, end, on_start=None):
    """[{from, to, text}] said between start and end, in seconds of the recording, the Thai marks put back.
    on_start(stop) hands over a function that ends the hearing at once (then Stopped is raised)."""
    tool = shutil.which("ffmpeg")
    if not tool:
        raise HearingError("FFmpeg is not installed, so the studio cannot hear the recording.")
    stopped = threading.Event()
    parts = {}

    def stop():
        stopped.set()
        for thing in parts.values():
            try:
                thing()
            except OSError:
                pass

    if on_start:
        on_start(stop)
    try:
        sound = subprocess.Popen([tool, "-v", "error", "-ss", "%.2f" % start, "-t", "%.2f" % (end - start), "-i",
                                  rec["path"], "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        parts["sound"] = sound.kill
        pcm, _ = sound.communicate()
        if stopped.is_set():
            raise Stopped()
        if sound.returncode != 0 or not pcm:
            raise HearingError("ffmpeg could not read the sound of %s at %s." % (rec["name"], recording.clock(start)))
        u = urllib.parse.urlsplit(URL)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=HEAR_SECONDS)
        parts["ask"] = conn.close
        conn.request("POST", "/hear", pcm, {"Content-Type": "application/octet-stream"})
        answer = conn.getresponse()
        got = json.loads(answer.read().decode("utf-8"))
        conn.close()
    except (OSError, ValueError, http.client.HTTPException) as e:
        if stopped.is_set():
            raise Stopped()
        raise HearingError("The hearing worker did not answer (%s). %s" % (e, status().get("why") or ""))
    if stopped.is_set():
        raise Stopped()
    if answer.status != 200:
        raise HearingError("The hearing worker could not hear it (%s)." % got.get("error"))
    return [{"from": round(start + p["from"], 2), "to": round(start + p["to"], 2), "text": thai.repair(p["text"]),
             "sure": p.get("sure", 1.0)} for p in got["phrases"] if p["text"].strip()]


def fix(ask, lines, kind, before=(), names=()):
    """(each line as Muse fixed it, or as heard; a sentence for each line kept as heard, else None).
    ask(messages, schema) asks Muse and returns its answer's words; before is lines heard earlier, as context; names
    are words written on the screen at the time."""
    out, kept = list(lines), [None] * len(lines)
    todo = list(range(len(lines)))
    for attempt in range(2):
        failed = []
        for first in range(0, len(todo), LINES_PER_QUESTION):
            batch = todo[first:first + LINES_PER_QUESTION]
            locked = []
            for i in batch:
                locked.append(thai.lock(lines[i], first=1 + sum(len(numbers) for _, numbers in locked)))
            try:
                answer = json.loads(ask(_question(kind, before, [text for text, _ in locked], names),
                                        _schema(len(batch))))
                written = answer["lines"]
                if len(written) != len(batch) or not all(isinstance(w, str) for w in written):
                    raise ValueError("%d lines came back for %d" % (len(written), len(batch)))
            except (ValueError, KeyError, TypeError) as e:
                for i in batch:
                    kept[i] = "Muse's answer could not be read (%s)" % str(e)[:80]
                failed += batch
                continue
            for i, (text, numbers), w in zip(batch, locked, written):
                back = thai.unlock(w.strip(), numbers)
                if back is None:
                    kept[i] = "a number was lost or added"
                elif _changed(lines[i].replace(" ", ""), back.replace(" ", "")) > MOST_CHANGED + sum(
                        len(n) for n in named(lines[i], back, names)) / float(max(1, len(back.replace(" ", "")))):
                    kept[i] = "more than %d%% of it changed" % (100 * MOST_CHANGED)
                else:
                    out[i], kept[i] = back, None
                    continue
                failed.append(i)
        todo = failed
        if not todo:
            break
    return out, kept


def named(heard, text, names):
    """The names from the screen that text has and heard did not: what a fix took from the screen."""
    return [n for n in names if n.lower() in text.lower() and n.lower() not in heard.lower()]


def _question(kind, before, lines, names=()):
    rules = ("You fix what a speech recogniser wrote down from a %s recording. It mishears words. Write each "
             "line as the speaker most likely said it, "
             "using the lines around it as context. Rules: change only words that are wrong; keep a line as it is "
             "when it is right or when you cannot tell; never add what was not said; keep every <N1>, <N2>... "
             "exactly once, where it belongs, and never write a number yourself. Answer with JSON: "
             '{"lines": [...]}, the same number of lines, in the same order.' % kind)
    said = ""
    if names:
        said += ("Written on the screen at the time (when the speaker says one of these, spell it as written here): "
                 "%s\n\n" % ", ".join(names))
    if before:
        said += "Heard just before (context only, do not answer these):\n" + "\n".join(before[-8:]) + "\n\n"
    said += "Lines to fix:\n" + "\n".join("%d. %s" % (i, line) for i, line in enumerate(lines, 1))
    return [{"role": "system", "content": rules}, {"role": "user", "content": said}]


def _schema(n):
    return {"type": "object", "required": ["lines"], "properties": {
        "lines": {"type": "array", "minItems": n, "maxItems": n, "items": {"type": "string"}}}}


def _changed(a, b):
    """How much of a was changed to make b: edit distance over the longer length, 0 to 1."""
    if not a and not b:
        return 0.0
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1] / max(len(a), len(b))

