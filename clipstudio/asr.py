"""Muse Voice Transcribe: Meta's speech to text on the Model API, as the studio's hearing when the hearing on this
computer is not set up (or LCS_HEARING=meta chooses it), with the person's own Meta key.

It answers with turns, each with its start and end, and no time for each word (measured 10-29 on 87 s of our own
Thai review: 3 turns, one of them 51 s long, in 17.5 s). A caption needs its words near their time, so the sound is
cut first at its pauses into pieces of at most PIECE seconds, and each piece is its own question, several at a time:
a phrase is then never longer than its piece. A piece with no sound in it is not sent. The answer is written Thai,
marks and all, so it is not repaired as Omnilingual's is.
"""
import concurrent.futures
import io
import json
import os
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np

MODEL = os.environ.get("LCS_VOICE_MODEL", "muse-voice-transcribe-1.0")
RATE = 16000
FRAME = 0.05
PIECE = 10.0
LONGEST_PIECE = 25.0
QUIET_GAP = 0.3
LONG_QUIET = 1.0
AIR = 0.15
QUIET_OVER = 6.0
QUIET_MOST = -35.0
AT_ONCE = 6
TIMEOUT = 120
PIECE_TRIES = 2
USD_PER_HOUR = 0.18


class Label:
    key, label, model_id = "voice", "Muse Voice Transcribe", MODEL


VOICE = Label()


class VoiceError(Exception):
    pass


def url():
    return os.environ.get("LCS_SPARK_URL", "https://api.meta.ai/v1").rstrip("/") + "/asr/transcribe"


def secret():
    return os.environ.get("META_API_KEY", "")


def pieces(pcm, rate=RATE):
    """[(start s, end s)] of the 16-bit mono sound pcm: cut at the middle of pauses of QUIET_GAP s or more, and at
    both ends of a pause of LONG_QUIET s or more (AIR s inside it), each piece at most PIECE s where a pause allows it
    and never more than LONGEST_PIECE s; a piece that is all quiet, such as a long pause, is left out. A pause is a
    stretch QUIET_OVER dB or less over the quietest fifth of the sound, and under QUIET_MOST."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    step = int(rate * FRAME)
    n = len(samples) // step
    if n == 0:
        return []
    frames = samples[:n * step].reshape(n, step)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) / 32768 + 1e-9)
    floor = min(float(np.percentile(db, 20)) + QUIET_OVER, QUIET_MOST)
    quiet = db <= floor
    cuts, run = [], 0
    for i, q in enumerate(list(quiet) + [False]):
        if q:
            run += 1
            continue
        if run * FRAME >= LONG_QUIET:
            cuts += [(i - run) * FRAME + AIR, i * FRAME - AIR]
        elif run * FRAME >= QUIET_GAP:
            cuts.append((i - run / 2) * FRAME)
        run = 0
    out, start, end = [], 0.0, n * FRAME
    while end - start > PIECE:
        inside = [c for c in cuts if start + 1.0 < c <= start + PIECE]
        later = [c for c in cuts if start + PIECE < c <= start + LONGEST_PIECE]
        cut = inside[-1] if inside else later[0] if later else start + LONGEST_PIECE
        if cut >= end:
            break
        out.append((start, cut))
        start = cut
    out.append((start, end))
    return [(round(a, 2), round(b, 2)) for a, b in out
            if not quiet[int(a / FRAME):max(int(a / FRAME) + 1, int(b / FRAME))].all()]


class PartHeard(VoiceError):
    """Some pieces of the sound were heard and some were not: said and usd as hear returns them, lost the pieces not
    heard, [(from, to)] in seconds of the sound."""

    def __init__(self, sentence, said, usd, lost):
        super().__init__(sentence)
        self.said, self.usd, self.lost = said, usd, lost


def _wav(pcm, rate=RATE):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _answer(pcm, rate, languages, mode):
    """Muse Voice Transcribe's answer to the sound pcm in mode. VoiceError with a sentence when it could not be asked.
    languages, by their English names, are what it should expect to hear (languageBias): it steers, it does not force.
    Its answer sometimes holds a broken Thai byte: that costs the one letter, as 10-31 it lost the hearing of 8 of 261
    looks and EP 22's joke with them (lane 4)."""
    boundary = uuid.uuid4().hex
    asked = {"model": MODEL, "audioEncoding": "WAV", "mode": mode}
    if languages:
        asked["languageBias"] = list(languages)
    request = json.dumps(asked)
    body = b"".join([
        ("--%s\r\nContent-Disposition: form-data; name=\"request\"\r\nContent-Type: application/json\r\n\r\n"
         % boundary).encode("ascii"), request.encode("utf-8"), b"\r\n",
        ("--%s\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav"
         "\r\n\r\n" % boundary).encode("ascii"), _wav(pcm, rate), b"\r\n", ("--%s--\r\n" % boundary).encode("ascii")])
    req = urllib.request.Request(url(), data=body, headers={
        "Authorization": "Bearer " + secret(), "Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        said = e.read()[:300].decode("utf-8", "replace")
        raise VoiceError("Muse Voice Transcribe refused the sound (%d%s)." % (
            e.code, ": the Meta key was not accepted" if e.code in (401, 403) else ", " + said if said else ""))
    except (OSError, ValueError) as e:
        raise VoiceError("Muse Voice Transcribe did not answer (%s)." % e)


def ask(pcm, rate=RATE, languages=()):
    """[(start s, end s, text)] Muse Voice Transcribe heard in the sound pcm, by its turns (_answer)."""
    got = _answer(pcm, rate, languages, "ENDPOINTING")
    turns = [(t.get("startMs", 0) / 1000, t.get("endMs", 0) / 1000, (t.get("transcript") or "").strip())
             for t in got.get("turns") or []]
    turns = [t for t in turns if t[2]]
    if not turns and (got.get("transcript") or "").strip():
        turns = [(0.0, len(pcm) / 2 / rate, got["transcript"].strip())]
    return turns


def turns(pcm, rate=RATE, languages=()):
    """[(start s, end s, speaker)] of the sound pcm, one voice's turn each, and the US dollars it cost, from Muse Voice
    Transcribe's DIARIZATION mode. One request, as its labels hold only inside one (clips to 10 minutes). Its words are
    worse than the pieces' (EP0 10-31), so the turns' edges are all that is kept."""
    got = _answer(pcm, rate, languages, "DIARIZATION")
    return ([(t.get("startMs", 0) / 1000, t.get("endMs", 0) / 1000, t["speaker"]) for t in got.get("turns") or []
             if t.get("speaker")], len(pcm) / 2 / rate / 3600 * USD_PER_HOUR)


def hear(pcm, rate=RATE, stopped=None, languages=()):
    """[{from, to, text}] said in the sound pcm, in seconds of it, and the US dollars it cost: each piece asked,
    AT_ONCE at a time, expecting languages (ask), up to PIECE_TRIES times. stopped() True leaves the pieces not yet
    sent unsent. A piece that still fails costs that piece: PartHeard holds the rest. VoiceError when none was
    heard; sound with nothing in it asks nothing, and nothing was said."""
    if not secret():
        raise VoiceError("Muse Voice Transcribe needs your Meta API key, and none is set.")
    cut = pieces(pcm, rate)
    width = 2 * rate

    failed = {}

    def one(span):
        a, b = span
        for _ in range(PIECE_TRIES):
            if stopped and stopped():
                break
            try:
                turns = ask(pcm[int(a * width):int(b * width)], rate, languages)
            except VoiceError as e:
                failed[span] = str(e)
                continue
            failed.pop(span, None)
            return [(a + x, min(b, a + y) if y > x else b, text) for x, y, text in turns]
        return []

    with concurrent.futures.ThreadPoolExecutor(AT_ONCE) as pool:
        heard = [t for got in pool.map(one, cut) for t in got]
    usd = sum(b - a for a, b in cut) / 3600 * USD_PER_HOUR
    said = [{"from": round(a, 2), "to": round(b, 2), "text": text} for a, b, text in heard]
    if failed and len(failed) == len(cut):
        raise VoiceError(next(iter(failed.values())))
    if failed:
        raise PartHeard(next(iter(failed.values())), said, usd, sorted(failed))
    return said, usd
