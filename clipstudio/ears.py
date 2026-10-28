"""The hearing worker: Meta's Omnilingual ASR CTC model, turned into ONNX once (hearing/export_onnx.py) and run by
onnxruntime on the processor, so the graphics card stays with Muse and nothing needs Docker. The studio runs it in
its own process, behind a small HTTP server that hearing.py talks to:

    POST /hear    body: 16 kHz mono 16-bit little-endian PCM, at most 15 minutes
                  -> {"phrases": [{"from": s, "to": s, "text": "...", "sure": 0-1}], "seconds": time taken}
    GET  /health  -> {"ok": true, "model": ..., "dtype": ..., "threads": n} once the model is loaded,
                     else {"ok": false, "loading": true} or {"ok": false, "error": "..."}

Its folder holds the model (MODEL_FILES, the first one there) and letters.json, what each token writes when
Omnilingual's own decoder decodes it alone.

Omnilingual hears at most 40 s at a time, so longer audio is cut into pieces of at most 30 s, each cut at the
quietest tenth of a second from 20 s on: between words, not through one. Each piece is given to the model as
Omnilingual's pipeline gives it (given).

Times come from the model itself. The CTC model decides one token for every 20 ms of sound; the decoding is greedy,
keeping the frame where each letter starts. The model writes a space where the speaker pauses (Thai has no spaces
between words), so a phrase is what lies between two spaces. It starts where its first letter starts and ends one
frame after its last.

How sure the model is of a letter is the mean, over the frames it holds, of the probability the model gave its
choice in each frame. A phrase's sure is the mean over its letters (measured 10-27 on Thai FLEURS, clean and mixed
with game sound: it tells a garbled phrase from a good one with AUC 0.91-0.94; the least sure letter did not).
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

MODEL_FILES = ("model.int8.onnx", "model.onnx")
THREADS = int(os.environ.get("LCS_HEARING_THREADS", "8"))
RATE = 16000
SHORTEST_PIECE, LONGEST_PIECE = 20.0, 30.0
TENTH = RATE // 10
MOST_SECONDS = 900
EPS = 1e-5


def cut_points(audio):
    """Sample offsets [0, ..., len(audio)] that cut audio into pieces of at most LONGEST_PIECE seconds, each cut at
    the quietest tenth of a second between SHORTEST_PIECE and LONGEST_PIECE after the cut before it."""
    points, start = [0], 0
    while len(audio) - start > LONGEST_PIECE * RATE:
        first = start + int(SHORTEST_PIECE * RATE)
        n = int((LONGEST_PIECE - SHORTEST_PIECE) * RATE) // TENTH
        energy = (audio[first:first + n * TENTH].reshape(n, TENTH) ** 2).mean(axis=1)
        start = first + int(np.argmin(energy)) * TENTH + TENTH // 2
        points.append(start)
    points.append(len(audio))
    return points


def given(piece):
    """The samples the model is given for piece (16-bit samples / 32768), as Omnilingual's pipeline makes them from
    a wav file: zero mean, and unit variance unless the piece is near silent (EPS; measured 10-28 against what the
    pipeline gave the model: equal within 1e-6)."""
    return ((piece - piece.mean()) / np.sqrt(piece.var() + EPS)).astype(np.float32)


def letters_of(logits, letters):
    """[(frame, letter, sure)] from a piece's logits (frames x tokens): each run of one token is one letter, at the
    frame where it starts, sure being the mean of the probability of the chosen token over the run."""
    ids = logits.argmax(axis=-1)
    top = 1.0 / np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(axis=-1)
    starts = [0] + [f for f in range(1, len(ids)) if ids[f] != ids[f - 1]] if len(ids) else []
    ends = starts[1:] + [len(ids)]
    return [(f, letters[int(ids[f])], float(top[f:e].mean())) for f, e in zip(starts, ends)]


def phrases(letters, offset, frame_seconds):
    """[(frame, letter, sure)] of one piece -> [{from, to, text, sure}] in seconds of the whole audio. A space
    ends a phrase; the CTC blank between letters decodes to nothing and ends nothing."""
    out, now, sure = [], None, []
    for frame, letter, p in letters:
        if not letter:
            continue
        if not letter.strip():
            now = None
            continue
        t = offset + frame * frame_seconds
        if now is None:
            now, sure = {"from": round(t, 2), "to": 0, "text": ""}, []
            out.append(now)
        now["text"] += letter.strip()
        now["to"] = round(t + frame_seconds, 2)
        sure.append(p)
        now["sure"] = round(sum(sure) / len(sure), 3)
    return out


def model_file(folder):
    """The model file in folder, or None."""
    return next((os.path.join(folder, n) for n in MODEL_FILES if os.path.isfile(os.path.join(folder, n))), None)


class Ears:
    def __init__(self, folder):
        self.folder, self.session, self.letters, self.error, self.name = folder, None, None, None, None
        self.lock = threading.Lock()

    def load(self):
        try:
            import onnxruntime
            path = model_file(self.folder)
            if not path:
                raise OSError("no %s in %s" % (" or ".join(MODEL_FILES), self.folder))
            with open(os.path.join(self.folder, "letters.json"), encoding="utf-8") as f:
                told = json.load(f)
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = THREADS
            self.letters = told["letters"]
            self.session = onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])
            self.name = "%s, %s" % (told.get("card", "Omnilingual ASR"), os.path.basename(path))
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, str(e)[:300])
        print("hearing: %s" % ("ready, %s, %d threads" % (self.name, THREADS) if self.session else self.error),
              flush=True)

    def hear(self, pcm):
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        points = cut_points(audio)
        out = []
        with self.lock:
            for a, b in zip(points, points[1:]):
                if b - a < TENTH:
                    continue
                logits = self.session.run(["logits"], {"x": given(audio[a:b])[None, :]})[0][0]
                out += phrases(letters_of(logits, self.letters), a / RATE, (b - a) / RATE / max(1, len(logits)))
        return out


def handler(ears):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self.send(404, {"error": "There is nothing at %s." % self.path})
            if ears.session:
                return self.send(200, {"ok": True, "model": ears.name, "dtype": "onnx", "threads": THREADS})
            return self.send(503, {"ok": False, "error": ears.error} if ears.error else {"ok": False, "loading": True})

        def do_POST(self):
            if self.path != "/hear":
                return self.send(404, {"error": "There is nothing at %s." % self.path})
            if not ears.session:
                return self.send(503, {"error": ears.error or "The model is still loading."})
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= MOST_SECONDS * RATE * 2:
                return self.send(400, {"error": "Send 16 kHz mono 16-bit sound, at most %d minutes."
                                                % (MOST_SECONDS // 60)})
            pcm = self.rfile.read(size)
            started = time.time()
            try:
                said = ears.hear(pcm)
            except Exception as e:
                return self.send(500, {"error": "%s: %s" % (type(e).__name__, str(e)[:300])})
            self.send(200, {"phrases": said, "seconds": round(time.time() - started, 2)})

    return Handler


def serve(folder, host, port):
    """Starts the worker on host:port in the background and loads the model from folder. Returns the Ears."""
    ears = Ears(folder)
    server = ThreadingHTTPServer((host, port), handler(ears))
    threading.Thread(target=server.serve_forever, name="hearing-server", daemon=True).start()
    threading.Thread(target=ears.load, name="hearing-load", daemon=True).start()
    return ears
