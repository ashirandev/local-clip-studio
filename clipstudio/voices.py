"""Voices: who says each line, told apart by how high the voice is, and matched to the faces by Muse looking.

Who speaks was first found from mouths moving (talk.py). On the Q&A with คุณอีฟ (10-27: 1280x720, faces about 220 px
tall, a phone in his hand under his chin) it showed him for 20 s while she told the story: her mouth and his moved
alike, and his hand and phone moved under his chin. Pixel change in a 60 px mouth told nothing: under his voice the
two mouths scored 1.12 and 1.24, under hers 1.17 and 1.17, with a lips-only box too, and their movement followed the
loudness of the speech no better (correlations 0.0 to 0.19). The voices told them apart at once: every line he said
sat at 97 to 137 Hz, every line she said at 192 to 314 Hz.

So each line the hearing wrote down gets the pitch of its voice (the median of its voiced frames, found by
autocorrelation between LOWEST and HIGHEST Hz), and the lines are split into the two voices that keep each side's
pitches closest together, when the two sides sit at least APART apart (a ratio of their medians). Voices too close
to tell apart, such as two men alike, are not split, and who speaks is left to the mouths.

Which voice is which face: Muse is shown frames from inside each voice's longest lines, each person's number drawn
over their head, and asked who is speaking in each; a voice goes to the person named in most of its pictures, and
the two voices must go to two people. Measured on the Q&A with 4 pictures a voice and no numbers drawn: 6 of 8
pictures right, and both voices right by the vote.
"""
import base64
import json
import os
import shutil
import subprocess

import cv2
import numpy as np

from . import recording

RATE = 16000
FRAME, HOP = 0.04, 0.01
LOWEST, HIGHEST = 60, 400
VOICED = 0.45
QUIET = 0.01
SHORTEST = 0.15
APART = 1.35
PICTURES = 3
SAID = 5.0
ASK = ("These pictures are from a video of %d people talking, each person's number drawn over their head. Each "
       "picture was taken while one of them was speaking. For each picture, say the number of the person who is "
       "speaking: whose mouth is open and shaped by words, not just smiling. Give 0 when you cannot tell. Answer with "
       "one number per picture, in order.")
SCHEMA = {"type": "object", "properties": {"speaking": {"type": "array", "items": {"type": "integer"}}},
          "required": ["speaking"], "additionalProperties": False}


class VoiceError(Exception):
    """The voices could not be told apart or matched. The sentence says why."""


def pitch(samples, rate=RATE):
    """The voice's pitch in Hz (the median of the voiced frames), or None when fewer than SHORTEST s are voiced."""
    n, hop = int(FRAME * rate), int(HOP * rate)
    lo, hi = rate // HIGHEST, rate // LOWEST
    window = np.hanning(n)
    found = []
    for i in range(0, len(samples) - n, hop):
        frame = samples[i:i + n] * window
        if np.sqrt(np.mean(frame ** 2)) < QUIET:
            continue
        ac = np.correlate(frame, frame, "full")[n - 1:]
        ac = ac / (ac[0] + 1e-9)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > VOICED:
            found.append(rate / k)
    if len(found) * HOP < SHORTEST:
        return None
    return float(np.median(found))


def sound(rec, a, b):
    """The recording's sound from a to b, mono at RATE, as floats."""
    tool = shutil.which("ffmpeg")
    if not tool:
        raise VoiceError("FFmpeg is not installed, so the voices cannot be heard.")
    try:
        raw = subprocess.run([tool, "-v", "error", "-ss", "%.3f" % a, "-t", "%.3f" % (b - a), "-i", rec["path"],
                              "-map", "0:a:0", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
                             capture_output=True, check=True, timeout=120).stdout
    except (subprocess.SubprocessError, OSError) as e:
        raise VoiceError("FFmpeg could not read the sound (%s)." % e)
    return np.frombuffer(raw, np.float32)


def pitches(rec, said):
    """[(line, pitch or None)] for the lines heard ({"from", "to"} in seconds of the recording)."""
    if not said:
        return []
    a, b = min(p["from"] for p in said), max(p["to"] for p in said)
    wave = sound(rec, a, b)
    return [(p, pitch(wave[int((p["from"] - a) * RATE):int((p["to"] - a) * RATE)])) for p in said]


def split(heard):
    """The pitch between two voices, or None when the lines' pitches do not part into two that sit APART apart. The
    split is the one that leaves each side's pitches closest together (on a log scale): the widest gap alone cut off
    one laugh, at 314 Hz, from the rest of the voice it was in."""
    found = sorted(f for _, f in heard if f)
    if len(found) < 2:
        return None
    logs = np.log(found)
    k = min(range(len(found) - 1), key=lambda i: np.var(logs[:i + 1]) * (i + 1) + np.var(logs[i + 1:]) *
            (len(found) - i - 1))
    low, high = found[:k + 1], found[k + 1:]
    if np.median(high) / np.median(low) < APART:
        return None
    return float(np.sqrt(found[k] * found[k + 1]))


def lines_of(heard, cut):
    """[(line, "low" or "high" or None)]: each line's voice, by its pitch against the cut."""
    return [(p, None if f is None else "high" if f > cut else "low") for p, f in heard]


def moments_to_show(voiced, each=PICTURES):
    """{voice: [seconds of the recording]}: the middle of each voice's longest lines, up to `each`."""
    out = {}
    for voice in ("low", "high"):
        mine = sorted((p for p, v in voiced if v == voice), key=lambda p: p["from"] - p["to"])[:each]
        out[voice] = [round((p["from"] + p["to"]) / 2, 2) for p in mine]
    return out


def picture(rec, at, persons, name):
    """A frame at `at` with each person's number drawn over their head; the jpg's path in work/voices."""
    tool = shutil.which("ffmpeg")
    folder = os.path.join(recording.WORK, "voices")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".jpg")
    raw = subprocess.run([tool or "ffmpeg", "-v", "error", "-ss", "%.3f" % at, "-i", rec["path"], "-frames:v", "1",
                          "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True, timeout=60).stdout
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    for p in persons:
        x, y, w, h = p["box"]
        cx, top = int(x + w / 2), int(max(28, y - 0.45 * h))
        cv2.circle(img, (cx, top), 24, (16, 16, 16), -1)
        cv2.putText(img, str(p["n"]), (cx - 11, top + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3,
                    cv2.LINE_AA)
    scale = 896.0 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    data.tofile(path)
    return path


def match(ask, rec, voiced, persons):
    """{"low": person, "high": person}: which person each voice is, by Muse looking at frames from inside each
    voice's lines. ask(content, schema) returns the answer's words. When Muse settles one voice and not the other, the
    other voice is the other person: seen 10-28 on the Q&A, where one voice said two short lines, one at the clip's
    end, and the clip fell back to the mouths and showed him for most of what she said. VoiceError when neither
    voice is settled, or both are the same person."""
    if len(persons) != 2:
        raise VoiceError("Voices are matched for two people, and the program found %d." % len(persons))
    shown = moments_to_show(voiced)
    content = [{"type": "text", "text": ASK % len(persons)}]
    order = []
    for voice, times in shown.items():
        for at in times:
            path = picture(rec, at, persons, "%s-%d-%s" % (os.path.basename(rec["path"]).replace(".", "_"),
                                                           int(at * 100), voice))
            if path:
                order.append(voice)
                content.append({"type": "text", "text": "Picture %d:" % len(order)})
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}})
    if not all(v in order for v in ("low", "high")):
        raise VoiceError("Not every voice has a picture to match it with.")
    try:
        said = json.loads(ask(content, SCHEMA))["speaking"]
    except (ValueError, KeyError, TypeError):
        raise VoiceError("Muse's answer could not be read.")
    names = {p["n"] for p in persons}
    votes = {"low": {}, "high": {}}
    for voice, n in zip(order, said):
        if n in names:
            votes[voice][n] = votes[voice].get(n, 0) + 1
    out, unsure = {}, []
    for voice, count in votes.items():
        best = max(count.values()) if count else 0
        top = [n for n, c in count.items() if c == best]
        if not count:
            unsure.append("Muse could not tell who speaks in the %s voice's pictures." % voice)
        elif len(top) > 1 or best * 2 <= sum(count.values()):
            unsure.append("Muse's answers for the %s voice do not agree." % voice)
        else:
            out[voice] = top[0]
    if not out:
        raise VoiceError(" ".join(unsure))
    if len(out) == 1:
        settled, = out
        out["high" if settled == "low" else "low"] = next(n for n in names if n != out[settled])
    if out["low"] == out["high"]:
        raise VoiceError("Muse named the same person for both voices.")
    return out


def speakers(heard, cut, people):
    """[{"from", "to", "who"}]: each line heard, in seconds of the recording, with the person whose voice it is, or
    None when its pitch was not found."""
    return [{"from": p["from"], "to": p["to"], "who": people.get(v) if v else None} for p, v in lines_of(heard, cut)]
