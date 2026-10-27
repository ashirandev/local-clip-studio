"""The hearing worker: Meta's Omnilingual ASR behind a small HTTP server, in a Linux container (its fairseq2 runs on
Linux only). It runs on the processor, so the graphics card stays with Muse. The studio starts it and talks to it.

    POST /hear    body: 16 kHz mono 16-bit little-endian PCM, at most 15 minutes
                  -> {"phrases": [{"from": s, "to": s, "text": "..."}], "seconds": time taken}
    GET  /health  -> {"ok": true, "model": ..., "dtype": ..., "threads": n} once the model is loaded,
                     else {"ok": false, "loading": true} or {"ok": false, "error": "..."}

Omnilingual hears at most 40 s at a time, so longer audio is cut into pieces of at most 30 s, each cut at the
quietest tenth of a second from 20 s on: between words, not through one.

Times come from the model itself. The CTC model decides one token for every 20 ms of sound; the decoding here is
Omnilingual's own greedy decoding, keeping the frame where each letter starts. The model writes a space where the
speaker pauses (Thai has no spaces between words), so a phrase is what lies between two spaces. It starts where its
first letter starts and ends one frame after its last.

It stops by itself after LCS_HEARING_IDLE seconds with no request. The studio asks /health every minute while it
runs, so it lives as long as the studio and its memory is given back when the studio is gone.
"""
import io
import json
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

CARD = os.environ.get("LCS_HEARING_CARD", "omniASR_CTC_1B_v2")
DTYPE = os.environ.get("LCS_HEARING_DTYPE", "float32")
THREADS = int(os.environ.get("LCS_HEARING_THREADS", "8"))
IDLE = float(os.environ.get("LCS_HEARING_IDLE", "1800"))
PORT = 8802
RATE = 16000
SHORTEST_PIECE, LONGEST_PIECE = 20.0, 30.0
TENTH = RATE // 10
MOST_SECONDS = 900


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


def phrases(letters, offset, frame_seconds):
    """[(frame, letter)] of one piece -> [{from, to, text}] in seconds of the whole audio. A space ends a phrase; the
    CTC blank between letters decodes to nothing and ends nothing."""
    out, now = [], None
    for frame, letter in letters:
        if not letter:
            continue
        if not letter.strip():
            now = None
            continue
        t = offset + frame * frame_seconds
        if now is None:
            now = {"from": round(t, 2), "to": 0, "text": ""}
            out.append(now)
        now["text"] += letter.strip()
        now["to"] = round(t + frame_seconds, 2)
    return out


def wav_bytes(audio):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return out.getvalue()


class Ears:
    def __init__(self):
        self.pipe, self.error = None, None
        self.lock = threading.Lock()
        self.last = time.time()

    def load(self):
        try:
            import torch
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

            torch.set_num_threads(THREADS)

            class Timed(ASRInferencePipeline):
                """Omnilingual's own greedy CTC decoding, which also keeps the frame where each letter starts."""

                def _apply_model_wav2vec2asr(self, batch):
                    from fairseq2.nn.batch_layout import BatchLayout
                    layout = BatchLayout(batch.source_seqs.shape, seq_lens=batch.source_seq_lens,
                                         device=batch.source_seqs.device)
                    logits, out = self.model(batch.source_seqs, layout)
                    ids = torch.argmax(logits, dim=-1)
                    texts, self.timed = [], []
                    for i in range(ids.shape[0]):
                        n = int(out.seq_lens[i])
                        seq = ids[i][:n]
                        new = torch.ones(n, dtype=torch.bool, device=seq.device)
                        new[1:] = seq[1:] != seq[:-1]
                        texts.append(self.token_decoder(seq[new]))
                        frames = torch.nonzero(new).flatten().tolist()
                        self.timed.append((n, [(f, self.token_decoder(seq[f:f + 1])) for f in frames]))
                    return texts

            self.torch = torch
            self.pipe = Timed(model_card=CARD, device="cpu", dtype=getattr(torch, DTYPE))
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, str(e)[:300])
        print("hearing: %s" % ("ready, %s %s, %d threads" % (CARD, DTYPE, THREADS) if self.pipe else self.error),
              flush=True)

    def hear(self, pcm):
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        points = cut_points(audio)
        out = []
        with self.lock, self.torch.inference_mode():
            for a, b in zip(points, points[1:]):
                if b - a < TENTH:
                    continue
                self.pipe.transcribe([wav_bytes(audio[a:b])], batch_size=1)
                frames, letters = self.pipe.timed[0]
                out += phrases(letters, a / RATE, (b - a) / RATE / max(1, frames))
            self.last = time.time()
        return out


EARS = Ears()


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
        EARS.last = time.time()
        if self.path != "/health":
            return self.send(404, {"error": "There is nothing at %s." % self.path})
        if EARS.pipe:
            return self.send(200, {"ok": True, "model": CARD, "dtype": DTYPE, "threads": THREADS})
        return self.send(503, {"ok": False, "error": EARS.error} if EARS.error else {"ok": False, "loading": True})

    def do_POST(self):
        EARS.last = time.time()
        if self.path != "/hear":
            return self.send(404, {"error": "There is nothing at %s." % self.path})
        if not EARS.pipe:
            return self.send(503, {"error": EARS.error or "The model is still loading."})
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= MOST_SECONDS * RATE * 2:
            return self.send(400, {"error": "Send 16 kHz mono 16-bit sound, at most %d minutes." % (MOST_SECONDS // 60)})
        pcm = self.rfile.read(size)
        started = time.time()
        try:
            said = EARS.hear(pcm)
        except Exception as e:
            return self.send(500, {"error": "%s: %s" % (type(e).__name__, str(e)[:300])})
        self.send(200, {"phrases": said, "seconds": round(time.time() - started, 2)})


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    EARS.load()
    while time.time() - EARS.last < IDLE or EARS.lock.locked():
        time.sleep(5)
    print("hearing: nothing asked for %d s, stopping" % IDLE, flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
