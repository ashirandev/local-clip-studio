"""The person's mask: which pixels of the recording are the person, frame by frame, found on this computer by Meta's
SAM 3.1 and kept as state['mask'] for the cut-out looks and the words behind the person (cutout.py).

SAM runs in its own install, apart from the studio: SAM's code (github.com/facebookresearch/sam3) and its PyTorch in
a Python of their own, named by LCS_SAM_PYTHON, and the weights file named by LCS_SAM_WEIGHTS (see the README and
cutting/cut.py). The studio runs cutting/cut.py with that Python for one stretch and reads what it prints.

SAM and Muse Glimmer do not fit on a 16 GB graphics card together (measured 10-28: SAM 5.7 to 6.6 GB at 10 frames a
second, Glimmer 13.6 GB). So Glimmer steps off the card first: its llama-server, in router mode, unloads it (1.4 s),
SAM runs, and Glimmer is loaded again (15 s), also when SAM fails or Stop is pressed. A llama-server that runs one
model without a preset cannot step aside, so then the cut-out is refused with a sentence before SAM starts. When
Muse is not on this computer (nothing answers at Glimmer's address, or it is another machine's), the card is SAM's.

SAM follows the person at FPS pictures a second, and the renderer holds each mask until the next. Measured 10-28 on
30 s of a talk and of a game stream: a 10 fps mask held for three frames matched SAM's own 30 fps masks with IoU .988
and .994, as close as one frame is to the next, at a third of the time: 0.23 to 0.25 s a picture whatever its size,
plus 24 s to load SAM, so 70 to 74 s for 30 s. The person was never lost, and the face finder's face was inside the
mask in every frame where it found one.

Masks are kept per recording, stretch and face in work/masks, so asking again is instant and Glimmer stays. For a
talk, each person the layout follows gets their own mask from the same run (name.each/n.mkv), so the one speaking
can keep their colour (cutout.speaking_mask).

A computer with no SAM of its own, as anyone's who came for the API, sends the stretch to Meta's SAM 3.1 on the Model
API when the caller says a piece of the video may go (send), with the same Meta key and the same worker in the
studio's own Python, so Glimmer stays on the card. LCS_SAM=meta chooses it where SAM is set up too, LCS_SAM=local
never sends. Same masks, same files: the rest of the studio does not know which one ran.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import mind, recording

VERSION = 2
FPS = 10
API = os.environ.get("LCS_SAM_URL", mind.NAMED["meta"])
MOST_SECONDS = 120
WORKER = os.path.join(recording.ROOT, "cutting", "cut.py")
PYTHON = os.environ.get("LCS_SAM_PYTHON", "")
WEIGHTS = os.environ.get("LCS_SAM_WEIGHTS", "")
STEP_ASIDE_SECONDS = 60
COME_BACK_SECONDS = 300
LOAD_SECONDS = 120
SECONDS_A_PICTURE = 2.0
HERE = ("127.0.0.1", "localhost", "::1")
_lock = threading.Lock()


class Label:
    key, label, model_id = "sam", "Meta SAM 3.1", "sam-3.1"


SAM = Label()


class PersonError(Exception):
    """No mask could be made. The sentence says why."""


class NotSetUp(PersonError):
    """SAM is not on this computer, as on anyone's who uses the API alone."""


class Stopped(Exception):
    """The person pressed Stop while SAM was working."""


def status():
    """{ok, label, where} or {ok: False, ..., why} with a sentence."""
    out = {"label": "Cut-out", "where": "SAM 3.1 on this computer's graphics card"}
    why = _not_set_up()
    return dict(out, ok=False, why=why) if why else dict(out, ok=True, name=os.path.basename(WEIGHTS))


def _not_set_up():
    if not PYTHON or not os.path.isfile(PYTHON):
        return ("SAM 3.1 is not set up: LCS_SAM_PYTHON %r is not a Python. Install SAM in a Python of its own as the "
                "README shows, and set LCS_SAM_PYTHON to that python.exe." % PYTHON)
    if not WEIGHTS or not os.path.isfile(WEIGHTS):
        return ("SAM 3.1 is not set up: LCS_SAM_WEIGHTS %r is not a file. Download sam3.1_multiplex_fp16.safetensors "
                "as the README shows, and set LCS_SAM_WEIGHTS to it." % WEIGHTS)
    return None


def _goes_to_meta(send):
    """True when this stretch goes to Meta's SAM 3.1: the caller says it may, and SAM is not set up here or LCS_SAM=meta
    chooses it; never with LCS_SAM=local."""
    wanted = os.environ.get("LCS_SAM", "").strip().lower()
    return bool(send) and wanted != "local" and (wanted == "meta" or bool(_not_set_up()))


def _checked(rec, start, end, face, people=None):
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (start, end)) or not (
            0 <= start < end <= rec["seconds"] + 0.01):
        raise ValueError("from %r to %r is not a stretch of %s: give recording seconds from 0 to %s, to after from."
                         % (start, end, rec["name"], rec["seconds"]))
    if end - start > MOST_SECONDS:
        raise ValueError("from %r to %r is %.1f s: the cut-out follows a person for at most %d s at a time, so cut "
                         "the stretch the effect needs." % (start, end, end - start, MOST_SECONDS))
    if people is not None and (not isinstance(people, (list, tuple)) or not people):
        raise ValueError("people %r is not a list of face boxes: give [[x, y, w, h], ...], one for each person the "
                         "layout follows." % (people,))
    face = None if face is None else _box(rec, face)
    return round(float(start), 3), round(float(end), 3), face, people and [_box(rec, p) for p in people]


def _box(rec, face):
    ok = (isinstance(face, (list, tuple)) and len(face) == 4
          and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in face)
          and face[2] > 0 and face[3] > 0 and 0 <= face[0] + face[2] / 2.0 <= rec["width"]
          and 0 <= face[1] + face[3] / 2.0 <= rec["height"])
    if not ok:
        raise ValueError("face %r is not a face box on %s: give [x, y, w, h] in its pixels, the middle inside "
                         "its %dx%d picture." % (face, rec["name"], rec["width"], rec["height"]))
    return [round(float(v), 1) for v in face]


def key(rec, start, end, face, people=None, what=None):
    st = os.stat(rec["path"])
    w = os.stat(WEIGHTS) if WEIGHTS and os.path.isfile(WEIGHTS) else None
    return hashlib.sha1(json.dumps([VERSION, rec["path"], st.st_size, st.st_mtime, start, end, FPS, face,
                                    w and w.st_size] + ([people] if people else []) + ([what] if what else []))
                        .encode("utf-8")).hexdigest()[:16]


def _each(folder, name, people):
    """[the path of the n-th person's own mask, or None when SAM found nobody at their face], or None without
    people."""
    if not people:
        return None
    paths = [os.path.join(folder, name + ".each", "%d.mkv" % n) for n in range(1, len(people) + 1)]
    return [p if os.path.isfile(p) else None for p in paths]


def cut_out(rec, start, end, face=None, on_progress=None, on_start=None, people=None, what=None, send=False,
            on_paid=None):
    """state['mask'] for the person from start to end (recording seconds): {"path", "person", "fps", "size", "from",
    "to"}, person being the face box the person was picked by (None: the person with the most pixels). face is
    [x, y, w, h] in the recording's pixels (the layout's face camera), so when SAM finds several people the one
    holding it is kept. With people, the face boxes of a talk's people in their order, the mask has "each" too: the
    path of each one's own mask, or None for one SAM did not find; and "everyone": the path of the mask of all the
    people SAM followed, a face or not. With what, SAM finds that instead of a person,
    such as "white game controller" (the product a seller holds), and keeps the one with the most pixels over the
    stretch when there are several. on_progress(done, of) as SAM follows them;
    on_start(stop) hands over a function that ends the work at once (then Stopped is raised). send says a piece of the
    video may go to Meta (_goes_to_meta), and on_paid(usd) gets what that cost. PersonError with a sentence when no
    mask could be made."""
    start, end, face, people = _checked(rec, start, end, face, people)
    hosted = _goes_to_meta(send)
    why = None if hosted else _not_set_up()
    if why:
        raise NotSetUp(why)
    folder = os.path.join(recording.WORK, "masks")
    name = key(rec, start, end, face, people, what)
    path = os.path.join(folder, name + ".mkv")
    made = {"path": path, "person": face, "fps": FPS, "size": [rec["width"], rec["height"]], "from": start,
            "to": end}

    def given():
        if not people:
            return made
        everyone = os.path.join(folder, name + ".each", "everyone.mkv")
        return dict(made, each=_each(folder, name, people), everyone=everyone if os.path.isfile(everyone) else None)

    if os.path.isfile(path):
        return given()
    os.makedirs(folder, exist_ok=True)
    stopped = threading.Event()
    running = {}

    def stop():
        stopped.set()
        if running.get("worker"):
            try:
                running["worker"].kill()
            except OSError:
                pass

    if on_start:
        on_start(stop)
    with _lock:
        if os.path.isfile(path):
            return given()
        if stopped.is_set():
            raise Stopped()
        aside = None if hosted else _step_aside()
        try:
            if stopped.is_set():
                raise Stopped()
            _run(rec, start, end, face, path, folder, name, running, stopped, on_progress, people, what, hosted,
                 on_paid)
        except BaseException as e:
            if aside:
                _come_back(*aside, after=e)
            raise
        if aside:
            _come_back(*aside)
    return given()


def _run(rec, start, end, face, path, folder, name, running, stopped, on_progress, people=None, what=None,
         hosted=False, on_paid=None):
    frames = os.path.join(folder, name + ".pictures")
    part = path + ".part"
    each, each_part = os.path.join(folder, name + ".each"), os.path.join(folder, name + ".each.part")
    _clear(frames)
    os.makedirs(frames)
    job = {"recording": rec["path"], "from": start, "to": end, "fps": FPS, "size": [rec["width"], rec["height"]],
           "face": face, "weights": WEIGHTS, "frames": frames, "out": part}
    if what:
        job["prompt"] = what
    if hosted:
        job["hosted"] = API
    if people:
        _clear(each_part)
        os.makedirs(each_part)
        job.update(people=people, each=each_part)
    last = {}
    try:
        with open(os.path.join(folder, name + ".log"), "w", encoding="utf-8") as log:
            worker = subprocess.Popen([sys.executable if hosted else PYTHON, "-X", "utf8", WORKER],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                      cwd=os.path.dirname(WORKER))
            running["worker"] = worker
            if stopped.is_set():
                worker.kill()
            most = LOAD_SECONDS + SECONDS_A_PICTURE * FPS * (end - start)

            def late():
                running["late"] = most
                worker.kill()

            timer = threading.Timer(most, late)
            timer.start()
            try:
                worker.stdin.write(json.dumps(job).encode("utf-8"))
                worker.stdin.close()
                for line in worker.stdout:
                    try:
                        said = json.loads(line.decode("utf-8"))
                    except ValueError:
                        continue
                    if "done" in said and on_progress:
                        on_progress(said["done"], said["of"])
                    if "ok" in said:
                        last = said
                worker.wait()
            except OSError:
                worker.kill()
                worker.wait()
            finally:
                timer.cancel()
                worker.stdout.close()
        if last:
            with open(os.path.join(folder, name + ".log"), "a", encoding="utf-8") as log:
                log.write(json.dumps(last) + "\n")
        if stopped.is_set():
            raise Stopped()
        if running.get("late"):
            raise PersonError("SAM took more than %d s for %.1f s of the recording, so it was stopped; its log is %s."
                              % (running["late"], end - start, os.path.join(folder, name + ".log")))
        if not last.get("ok"):
            raise PersonError(last.get("error") or "SAM stopped without saying why (exit code %s); its log is %s."
                              % (worker.returncode, os.path.join(folder, name + ".log")))
        if not os.path.isfile(part):
            raise PersonError("SAM said it was done, but wrote no mask.")
        if on_paid and last.get("usd"):
            on_paid(last["usd"])
        if people:
            _clear(each)
            os.replace(each_part, each)
        os.replace(part, path)
    finally:
        _clear(frames)
        _clear(each_part)
        if os.path.isfile(part):
            os.remove(part)


def _clear(frames):
    if os.path.isdir(frames):
        for f in os.listdir(frames):
            os.remove(os.path.join(frames, f))
        os.rmdir(frames)


def _glimmer():
    """(llama-server's address without /v1, model name) of Muse Glimmer when it is on this computer, else None."""
    local = mind.MINDS["local"]
    root = local.url[:-3] if local.url.endswith("/v1") else local.url
    if urllib.parse.urlsplit(root).hostname not in HERE:
        return None
    return root, local.model_id


def _ask(root, path, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(root + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def _where(root, model):
    """(Glimmer's state on the router: "loaded", "unloaded", "loading"..., the port of the llama-server it runs in)
    or (None, None) when nothing answers there. A new load runs in a new llama-server on a new port. PersonError when
    the server is not in router mode."""
    try:
        got = _ask(root, "/models")
    except (OSError, ValueError):
        return None, None
    row = next((m for m in got.get("data") or [] if m.get("id") == model), None)
    status = (row or {}).get("status") or {}
    if not status.get("value"):
        raise PersonError("Muse Glimmer runs on a llama-server without router mode, so it cannot step off the "
                          "graphics card for SAM. Start it from a preset with --models-preset, as the README shows.")
    args = status.get("args") or []
    port = args[args.index("--port") + 1] if "--port" in args[:-1] else None
    return status["value"], port


def _wait(root, model, done, seconds):
    """True once done(state, port) holds, False after seconds."""
    until = time.time() + seconds
    while time.time() < until:
        if done(*_where(root, model)):
            return True
        time.sleep(0.25)
    return False


def _step_aside():
    """Unloads Glimmer from the graphics card when it is there: (root, model) to load it again after SAM, also when
    it was not loaded, since the studio needs it next; None when Muse is not on this computer."""
    where = _glimmer()
    if not where:
        return None
    root, model = where
    state, port = _where(root, model)
    if state is None:
        return None
    if state == "unloaded":
        return where
    try:
        _ask(root, "/models/unload", {"model": model}, timeout=STEP_ASIDE_SECONDS)
    except (OSError, ValueError) as e:
        raise PersonError("Muse Glimmer could not step off the graphics card for SAM (%s)." % e)
    if not _wait(root, model, lambda st, p: st == "unloaded" or p not in (None, port), STEP_ASIDE_SECONDS):
        _come_back(root, model)
        raise PersonError("Muse Glimmer did not step off the graphics card within %d s, so SAM did not start."
                          % STEP_ASIDE_SECONDS)
    if _where(root, model)[0] != "unloaded":
        raise PersonError("Muse Glimmer was loaded again at once, by a question sent to it, so the graphics card was "
                          "not free and SAM did not start. Start llama-server with --no-models-autoload, as the README "
                          "shows, so that only the studio loads it.")
    return root, model


def _come_back(root, model, after=None):
    """Loads Glimmer again, on the whole graphics card. When a question loaded it while SAM held the card, llama-server
    put part of it off the card (measured 10-28: 5.2 tokens a second instead of 47.5), so it is loaded once more.
    PersonError with a sentence when it is not back within COME_BACK_SECONDS, after the sentence of the PersonError
    that came first, if one did."""
    try:
        state, beside = _where(root, model)
        if state in ("loaded", "loading"):
            _wait(root, model, lambda st, p: st == "loaded", COME_BACK_SECONDS)
            _ask(root, "/models/unload", {"model": model}, timeout=STEP_ASIDE_SECONDS)
            _wait(root, model, lambda st, p: st == "unloaded" or p not in (None, beside), STEP_ASIDE_SECONDS)
        else:
            beside = None
        try:
            _ask(root, "/models/load", {"model": model}, timeout=COME_BACK_SECONDS)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
        if _wait(root, model, lambda st, p: st == "loaded" and (beside is None or p != beside),
                 COME_BACK_SECONDS):
            return
        why = "it was not loaded within %d s" % COME_BACK_SECONDS
    except (OSError, ValueError, PersonError) as e:
        why = str(e)
    first = "%s Then " % after if isinstance(after, PersonError) else ""
    raise PersonError(first + "Muse Glimmer could not come back onto the graphics card after SAM (%s). Any mask made "
                      "is kept; asking for it again returns it at once." % why)
