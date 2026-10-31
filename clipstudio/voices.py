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

Three or more people, or two whose voices are too alike, need more than pitch. Muse Voice Transcribe's DIARIZATION
gives the turns (turns_of), and Muse Wonder names each one from a small video of it with its sound, each person's number
near their head (numbered, WHO_ASK). Mouths could not do it: on EP0's four men the label of the same voice went to
person 2, 1 and 3 in three runs (10-31), while Wonder gave the same man on all 6 turns, 18 of 18 answers, and Tarm's ear
agreed with all 6 ("EP0 ถูกต้อง!"). Each turn is named by itself, as one label took in up to three men. About $0.07 a
turn, so on the switch in Options only, kept in work/voices, and at most MOST_TURNS a clip.

Numbers from the faces found are not enough: on EP0 the man turned away had no face and no number, so Wonder gave his 7 s
turn to person 1 (Tarm: "ถ่ายเน้นที่ Donut เยอะมาก"). Tarm's idea, 10-31: show Muse a picture and let it say who is there.
Muse Spark counts the heads in one picture (HEADS_ASK, $0.006): 4 people on 9 asks of 9 across 3 frames, the same order
each time, each box on or near the head, though up to 15% off, so good for a number and for a shot this wide, and a face
found stays the better box (talk.with_heads). Its few words on each person go in Wonder's question: with the numbers alone
Wonder gave EP0's turns to men 1 and 3, with Spark's words to men 4 and 2, as Tarm's ear has them (10-31, 2 of 2).

A turn from the diarisation can hold two people: 32 s on EP0 took in man 3 and then man 1, and the one name stood for both,
so a turn over TURN_LONG s is named in pieces, cut where heard lines end. A clip's end counts more than its middle, so the
last turn is named whatever its length: with 8 turns the cap ate EP0's ending, man 2's gesture that Tarm praised. A turn
under TURN_SHORT s gives Wonder little to go on, and EP0's of 3.4 s came back man 2 on one run and man 3 on the next, so it
is asked again and a third time when the two differ.
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
TURN_LEAST = 1.0
TURN_SENT = 20.0
TURN_LONG = 12.0
TURN_SHORT = 5.0
SHORT_ASKS = 3
MOST_TURNS = 10
NUMBERED_LONGEST = 960
WHO_SURE = 0.5
WHO_TOKENS = 600
WHO_ASK = ("%d people in this video have a yellow number near their head%s. Someone else may be there with no number. One "
           "person does most of the talking in this stretch. Watch and listen: whose mouth moves with the words, whose "
           "hands and face go with them. Someone who only laughs or nods is not talking. Which number is talking? Give "
           "the number, or 0 when it is someone with no number; how sure you are from 0 to 1; and one short line on what "
           "you saw.")
WHO_SCHEMA = {"type": "object", "properties": {"who": {"type": "integer"}, "sure": {"type": "number"},
                                               "why": {"type": "string"}}, "required": ["who", "sure", "why"]}
HEADS_TOKENS = 800
HEADS_ASK = ("How many real people are in this picture? Toys and figures on a shelf are not people. Count a person turned "
             "away or cut off by the edge too. List them from left to right. For each give head: the box around their head "
             "as [left, top, right, bottom], each from 0 to 1000 of the picture's width or height, and looks: a few words "
             "on what they wear and where they sit.")
HEADS_SCHEMA = {"type": "object", "properties": {"people": {"type": "array", "items": {
    "type": "object", "properties": {"head": {"type": "array", "items": {"type": "number"}}, "looks": {"type": "string"}},
    "required": ["head", "looks"]}}}, "required": ["people"]}


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


def in_pieces(a, b, ends):
    """a to b cut into pieces of at most TURN_LONG s, at the last of ends (seconds, sorted) each can reach, or at
    TURN_LONG itself where no line ends in reach. A last sliver goes with the piece before it."""
    out, at = [], a
    while b - at > TURN_LONG:
        reach = [e for e in ends if at + TURN_LEAST <= e <= at + TURN_LONG]
        cut = reach[-1] if reach else at + TURN_LONG
        out.append([at, cut])
        at = cut
    if out and b - at < TURN_LEAST:
        out[-1][1] = b
    else:
        out.append([at, b])
    return out


def turns_of(found, start, said=()):
    """The turns to name, [{"from", "to"}] in seconds of the recording and in order: found ([(start s, end s, speaker)]
    in seconds of the stretch from start) with one speaker's neighbours joined, a long one in pieces at the ends of the
    lines said ({"to"} in seconds of the recording), those under TURN_LEAST s left out, the MOST_TURNS longest, and the
    last of all among them."""
    joined = []
    for a, b, label in sorted(found):
        if joined and joined[-1][2] == label:
            joined[-1][1] = b
        else:
            joined.append([a, b, label])
    ends = sorted(p["to"] - start for p in said)
    long_ = [t for a, b, _ in joined for t in in_pieces(a, b, ends) if t[1] - t[0] >= TURN_LEAST]
    kept = sorted(long_, key=lambda t: t[0] - t[1])[:MOST_TURNS]
    if long_ and long_[-1] not in kept:
        kept[-1] = long_[-1]
    return [{"from": round(start + a, 2), "to": round(start + b, 2)} for a, b in sorted(kept)]


def numbers(rec, persons, size, path):
    """A picture of size (w, h) with each person's number near their head, the rest see-through, at path."""
    w, h = size
    k = w / float(rec["width"])
    img = np.zeros((h, w, 4), np.uint8)
    for p in persons:
        x, y, bw, bh = [v * k for v in p.get("head") or p["box"]]
        cx, top = int(x + bw / 2), int(max(22, y - 10 if p.get("head") else y - 0.45 * bh))
        cv2.circle(img, (cx, top), 20, (16, 16, 16, 230), -1)
        cv2.putText(img, str(p["n"]), (cx - 10, top + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 230, 255, 255), 2, cv2.LINE_AA)
    ok, data = cv2.imencode(".png", img)
    if not ok:
        raise VoiceError("The numbers could not be drawn.")
    data.tofile(path)


def numbered(rec, a, b, persons, name):
    """A small video of a to b with its sound and each person's number near their head: its path in work/voices, made
    once. VoiceError with a sentence when it cannot be made."""
    folder = os.path.join(recording.WORK, "voices")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".mp4")
    if os.path.isfile(path):
        return path
    k = NUMBERED_LONGEST / float(max(rec["width"], rec["height"]))
    size = (int(rec["width"] * k / 2) * 2, int(rec["height"] * k / 2) * 2)
    tags = os.path.join(folder, name + ".png")
    numbers(rec, persons, size, tags)
    part = path + ".part.mp4"
    try:
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-ss", "%.3f" % a, "-t", "%.3f" % (b - a),
                        "-i", rec["path"], "-i", tags, "-filter_complex", "[0:v]scale=%d:%d[v];[v][1:v]overlay=0:0" % size,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-b:a", "96k", part], capture_output=True, check=True, timeout=120)
        os.replace(part, path)
    except (subprocess.SubprocessError, OSError) as e:
        raise VoiceError("FFmpeg could not make the video that shows who speaks (%s)." % e)
    finally:
        if os.path.isfile(part):
            os.remove(part)
    return path


def heads_question(path):
    """The message that asks Muse who is in the picture at path."""
    with open(path, "rb") as f:
        picture_ = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
    return [{"role": "user", "content": [{"type": "text", "text": HEADS_ASK},
                                         {"type": "image_url", "image_url": {"url": picture_}}]}]


def heads_answer(text, rec):
    """The heads in Muse's answer from the left, [{"box": [x, y, w, h] in pixels of the recording, "looks": its words}],
    or None when it cannot be read. A box with no size or off the picture is left out."""
    try:
        said = [((float(a), float(b), float(c), float(d)), str(p.get("looks") or "")) for p, (a, b, c, d) in
                ((p, p["head"]) for p in json.loads(text)["people"])]
    except (ValueError, KeyError, TypeError):
        return None
    w, h = rec["width"] / 1000.0, rec["height"] / 1000.0
    return sorted(({"box": [round(a * w, 1), round(b * h, 1), round((c - a) * w, 1), round((d - b) * h, 1)], "looks": looks}
                   for (a, b, c, d), looks in said if 0 <= a < c <= 1000 and 0 <= b < d <= 1000),
                  key=lambda head: head["box"])


def who_question(path, persons):
    """The message that asks Muse Wonder who talks in the video at path."""
    with open(path, "rb") as f:
        video = "data:video/mp4;base64," + base64.b64encode(f.read()).decode("ascii")
    told = "; ".join("%d, %s" % (p["n"], p["looks"]) if p.get("looks") else str(p["n"]) for p in persons)
    ask = WHO_ASK % (len(persons), ": " + told if any(p.get("looks") for p in persons) else "")
    return [{"role": "user", "content": [{"type": "text", "text": ask}, {"type": "video_url", "video_url": {"url": video}}]}]


def who_answer(text, persons):
    """The person Muse Wonder named in its answer, or None when it named nobody here or was not sure."""
    try:
        got = json.loads(text)
        who, sure = got["who"], float(got["sure"])
    except (ValueError, KeyError, TypeError):
        return None
    return who if who in {p["n"] for p in persons} and sure >= WHO_SURE else None


def agreed(answers):
    """The person most of answers name, or None when no two agree. One answer stands by itself."""
    top = max(set(answers), key=answers.count) if answers else None
    return top if len(answers) == 1 or answers.count(top) > 1 else None


def speakers(heard, cut, people):
    """[{"from", "to", "who"}]: each line heard, in seconds of the recording, with the person whose voice it is, or
    None when its pitch was not found."""
    return [{"from": p["from"], "to": p["to"], "who": people.get(v) if v else None} for p, v in lines_of(heard, cut)]
