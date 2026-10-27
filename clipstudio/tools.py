"""The agent's hands: every tool Muse can call, each with its input checked before it runs.

A tool is a plain function with a one-line label for the page, a description for Muse and a JSON Schema for its
input. Every tool also gets two fields from the loop:
- why: one line in the person's language, shown on the page next to the step
- plan_step: which step of Muse's plan the call works on, so the page can tick the plan

Input is checked here, not trusted: the hosted API does not promise that arguments match the schema. Wrong input is
refused with a sentence that says what to send instead, so Muse can fix it and call again.
"""
import json
import math
import os
import re
import shutil
import subprocess
import time

from . import (effects, faces, hearing, layouts, maps, mind, previews, project, recording, render, screen, spoken,
               talk, text, thai, tracks, words)
from .recipes import RECIPES, SIGNALS

WHY = {"type": "string",
       "description": "One short line, in the language the person writes in, telling the person why you take "
                      "this step."}
PLAN_STEP = {"type": "integer", "minimum": 1,
             "description": "Which step of your plan this call works on (1 is the first). Leave it out if there is "
                            "no plan."}


class Refused(Exception):
    """The call cannot run as sent. The sentence says what to send instead."""


class Tool:
    def __init__(self, name, label, summary, params, required, run, changes=False):
        self.name, self.label, self.summary, self.params = name, label, summary, params
        self.required, self.run, self.changes = tuple(required), run, changes

    def schema(self):
        props = dict(self.params, why=WHY, plan_step=PLAN_STEP)
        return {"type": "function", "function": {
            "name": self.name, "description": self.summary,
            "parameters": {"type": "object", "properties": props, "required": list(self.required) + ["why"]}}}

    def check(self, args, plan_size):
        """(why, plan_step, the other fields with their types fixed), or Refused with a sentence."""
        allowed = set(self.params) | {"why", "plan_step"}
        extra = sorted(set(args) - allowed)
        if extra:
            raise Refused("%s has no field named %s. Its fields are: %s."
                          % (self.name, ", ".join(extra), ", ".join(sorted(allowed))))
        why = args.get("why")
        if not isinstance(why, str) or not why.strip():
            raise Refused("Say why in one short line (the why field), then call %s again." % self.name)
        step = args.get("plan_step")
        if step is not None:
            step = _value("plan_step", step, PLAN_STEP)
            if plan_size and step > plan_size:
                raise Refused("Your plan has %d steps, so plan_step %d is not one of them." % (plan_size, step))
        missing = [k for k in self.required if args.get(k) is None]
        if missing:
            raise Refused("%s needs %s." % (self.name, " and ".join(missing)))
        clean = {k: _value(k, args[k], spec) for k, spec in self.params.items() if args.get(k) is not None}
        return why.strip(), (step if plan_size else None), clean


def _value(name, value, spec):
    """The value in the type its schema asks for, or Refused. Numbers sent as text are accepted, and so is text
    sent as a number."""
    kind = spec.get("type")
    if kind == "integer":
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool):
            raise Refused("%s must be a whole number." % name)
        low, high = spec.get("minimum"), spec.get("maximum")
        if (low is not None and value < low) or (high is not None and value > high):
            raise Refused("%s must be from %s to %s; %d is outside that." % (name, low, high if high else "up", value))
        return value
    if kind == "boolean":
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            value = value.strip().lower() == "true"
        if not isinstance(value, bool):
            raise Refused("%s must be true or false." % name)
        return value
    if kind == "number":
        if isinstance(value, str):
            try:
                value = float(value.strip().rstrip("s").strip())
            except ValueError:
                raise Refused("%s must be a number." % name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise Refused("%s must be a number." % name)
        low, high = spec.get("minimum"), spec.get("maximum")
        if (low is not None and value < low) or (high is not None and value > high):
            raise Refused("%s must be from %g to %s; %g is outside that." % (name, low, "%g" % high if high is not None
                                                                              else "up", value))
        return float(value)
    if kind == "array" and (spec.get("items") or {}).get("type") == "integer":
        if isinstance(value, str):
            try:
                value = json.loads(value if value.strip().startswith("[") else "[%s]" % value)
            except ValueError:
                pass
        if isinstance(value, int) and not isinstance(value, bool):
            value = [value]
        if not isinstance(value, list):
            raise Refused("%s must be a list of whole numbers, such as [2]." % name)
        return [_value("%s's number %d" % (name, i), v, spec["items"]) for i, v in enumerate(value, 1)]
    if kind == "array" and (spec.get("items") or {}).get("type") == "number":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        n = spec.get("minItems")
        if not isinstance(value, list) or len(value) != n:
            raise Refused("%s must be a list of %d numbers: %s" % (name, n, spec.get("description", "")))
        return [_value("%s's number %d" % (name, i), v, spec["items"]) for i, v in enumerate(value, 1)]
    if kind == "string":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str):
            raise Refused("%s must be text." % name)
        if spec.get("enum") and value.strip() not in spec["enum"]:
            raise Refused("%s must be one of: %s." % (name, ", ".join(spec["enum"])))
        if len(value) > spec.get("maxLength", 2000):
            raise Refused("%s is too long; keep it under %d characters." % (name, spec.get("maxLength", 2000)))
        return value.strip()
    if kind == "object":
        return _fields(name, value, spec)
    if kind == "array" and (spec.get("items") or {}).get("type") == "object":
        if not isinstance(value, list):
            raise Refused("%s must be a list." % name)
        if not spec.get("minItems", 0) <= len(value) <= spec.get("maxItems", 100):
            raise Refused("%s must have %d to %d items; it has %d."
                          % (name, spec.get("minItems", 0), spec.get("maxItems", 100), len(value)))
        return [_fields("%s %d" % (name.rstrip("s"), i), v, spec["items"]) for i, v in enumerate(value, 1)]
    if kind == "array":
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            raise Refused("%s must be a list of short lines of text." % name)
        if not spec.get("minItems", 0) <= len(value) <= spec.get("maxItems", 100):
            raise Refused("%s must have %d to %d lines; it has %d."
                          % (name, spec.get("minItems", 0), spec.get("maxItems", 100), len(value)))
        return [v.strip() for v in value]
    return value


def _fields(name, value, spec):
    """One object inside a list, its fields in the types its schema asks for, or Refused. An object sent as JSON
    text is read."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    fields = spec["properties"]
    if not isinstance(value, dict):
        raise Refused("%s must be an object with the fields %s." % (name, ", ".join(fields)))
    extra = sorted(set(value) - set(fields))
    if extra:
        raise Refused("%s has no field named %s. Its fields are: %s." % (name, ", ".join(extra), ", ".join(fields)))
    missing = [k for k in spec.get("required", []) if value.get(k) is None]
    if missing:
        raise Refused("%s needs %s." % (name, " and ".join(missing)))
    return {k: _value("%s's %s" % (name, k), value[k], s) for k, s in fields.items() if value.get(k) is not None}


def set_plan(job, why, steps):
    had = len(job.plan)
    job.set_plan(steps, why)
    lines = "\n".join("%d. %s" % (i, s) for i, s in enumerate(steps, 1))
    head = ("Plan changed: it had %d steps and now has %d." % (had, len(steps)) if had
            else "Plan set with %d step%s." % (len(steps), "" if len(steps) == 1 else "s"))
    return "%s\n%s\nWork through it now; give each call the plan_step it belongs to." % (head, lines)


def find_recordings(job, why, words="", newest=5):
    folders = recording.folders()
    if not folders:
        raise Refused("No folder of recordings was found. Ask the person to set LCS_VIDEOS to the folders their "
                      "recordings are in, then start the studio again.")
    found = recording.walk(words)
    where = ", ".join(folders)
    if not found:
        about = 'has "%s" in its name or folders' % words if words.split() else "is there"
        return "No recording %s in %s.%s" % (about, where, _names_here(folders))
    shown = found[:newest]
    about = ' for "%s"' % words if words.split() else ""
    lines = ["%d recordings%s in %s; the %d newest:" % (len(found), about, where, len(shown))]
    for i, (mtime, size, path) in enumerate(shown, 1):
        lines.append("%d. %s · %s · recorded %s · %s" % (i, os.path.basename(path), recording.size_text(size),
                                                       recording.when(mtime), path))
    return "\n".join(lines)


def open_recording(job, why, path):
    try:
        real = recording.check_path(path)
        open_now = job.chat.recording
        if open_now and os.path.normcase(open_now["path"]) == os.path.normcase(real):
            return "It is already open: %s\nIts picture is already in this conversation. Go on with the work." % (
                recording.describe(open_now))
        rec = recording.probe(real)
    except recording.NotOpened as e:
        raise Refused(str(e))
    had, picture = job.chat.open(rec, job=job.n)
    head = "Opened: " + recording.describe(rec)
    if had and had["path"] != rec["path"]:
        head += "\nIt replaces %s, which was open before." % had["name"]
    if picture:
        job.attach(picture, "A picture from %s, a tenth of the way in." % rec["name"])
        head += "\nA picture from it comes next, so you can see what it is."
    return head + "\nThe recording is only read, never changed."


def _names_here(folders):
    """How the recordings here are named, so Muse can tell the person what to say instead."""
    names, subfolders = [], []
    for top in folders:
        try:
            entries = sorted(os.scandir(top), key=lambda e: e.stat().st_mtime, reverse=True)
        except OSError:
            continue
        names += [e.name for e in entries if e.is_file() and e.name.lower().endswith(recording.VIDEO_TYPES)][:3]
        subfolders += [e.name for e in entries if e.is_dir() and not e.name.startswith(("_", "."))][:6]
    out = ""
    if names:
        out += " The newest files here are named like: %s." % ", ".join(names[:4])
    if subfolders:
        out += " Folders: %s." % ", ".join(subfolders[:8])
    return out


MAP_ANYTHING = {"sound": 1.0, "screen": 1.0}
STRETCH_DEFAULT = 30


def _open(job, doing):
    """The open recording, or Refused when there is none or its file has gone."""
    rec = job.chat.recording
    if not rec:
        raise Refused("No recording is open, so there is nothing to %s. Open one first." % doing)
    if not os.path.isfile(rec["path"]):
        raise Refused("%s is no longer at %s. Tell the person, and ask where it is now." % (rec["name"], rec["path"]))
    return rec


def map_recording(job, why, stretch_s=None, top=6):
    rec = _open(job, "map")
    clip = job.chat.clip
    recipe = RECIPES.get(clip["recipe"]) if clip else None
    by = recipe["map"] if recipe else MAP_ANYTHING
    stretch = stretch_s or (clip["length_s"] if clip else STRETCH_DEFAULT)
    m = maps.kept(rec)
    if m:
        head = "Mapped earlier, so this was instant."
    else:
        try:
            m = job.measure(rec)
        except maps.Stopped:
            return "Not mapped: the person pressed Stop. Nothing was kept."
        except maps.MapError as e:
            raise Refused(str(e))
        took = "%.1f s" % m["took"] if m["took"] < 60 else recording.clock(m["took"])
        head = "Mapped in %s and kept, so mapping it again is instant." % took
    stretches, about = maps.rank(m, by, stretch, top)
    lines = ["%s %s (%s): how loud every second is against the minutes around it, and how much the picture "
             "changes (read at each keyframe, about every %s s here)."
             % (head, rec["name"], recording.clock(rec["seconds"]), about["every"])]
    if not about["heard"]:
        lines.append("It has no sound ffmpeg could read, so only the picture ranks.")
    elif about["flat"]:
        lines.append("The sound barely rises anywhere (at most %.1f dB over the minutes around it), so loudness "
                     "says little in this recording." % about["loudest_rise"])
    ranked = "Ranked by %s%s." % (", ".join("%s %g" % (s, by[s]) for s in SIGNALS),
                                  " (the %s recipe)" % recipe["name"] if recipe else "")
    if m["seconds"] <= stretch:
        lines.append("%s The recording is no longer than one stretch of %d s, so all of it is one stretch:"
                     % (ranked, stretch))
    else:
        lines.append("%s Stretches of %d s, each starting %d s before its peak so what led up to the peak is in it. "
                     "Best first:" % (ranked, stretch, round(stretch * maps.LEAD)))
    for i, s in enumerate(stretches, 1):
        sound = ("the sound rises %+.1f dB over the minutes around it (louder than %d%% of the recording)"
                 % (s["rise"], s["louder"]) if s["louder"] is not None else "no sound")
        lines.append("%d. %s-%s, peak %s: %s; the picture is busier than %d%% of it."
                     % (i, recording.clock(s["start"]), recording.clock(s["end"]), recording.clock(s["peak"]), sound,
                        s["busier"]))
    lines.append("Nothing in these stretches has been looked at yet: look at a stretch (its start and end as written "
                 "here) before you propose it.")
    return "\n".join(lines)


LOOK_FRAMES = 4
SHORTEST_LOOK, LONGEST_LOOK = 2, 600
MOST_MOMENTS = 3


def _time(name, text):
    t = recording.seconds_of(text)
    if t is None:
        raise Refused('%s must be a time as the map writes it, like 1:51:44 or 35:43, or in seconds, like 6704; "%s" '
                      'is not one.' % (name, text))
    return t


def look_at_piece(job, why, start, end, frames=LOOK_FRAMES):
    rec = _open(job, "look at")
    clock, length = recording.clock, rec["seconds"]
    a, b = _time("start", start), _time("end", end)
    if a >= length:
        raise Refused("start %s is after the end of the recording, which is %s long." % (clock(a), clock(length)))
    if b <= a:
        raise Refused("end %s must come after start %s." % (clock(b), clock(a)))
    fixed = ""
    if b > length:
        b, fixed = length, "The end was after the end of the recording, so this look ends there, at %s." % clock(length)
    if b - a < SHORTEST_LOOK:
        raise Refused("A look needs at least %d s from start to end; %s-%s is %.1f s."
                      % (SHORTEST_LOOK, clock(a), clock(b), b - a))
    if b - a > LONGEST_LOOK:
        raise Refused("A look covers at most %d minutes; %s-%s is %s. Look at a shorter stretch, such as one the map "
                      "lists." % (LONGEST_LOOK // 60, clock(a), clock(b), clock(b - a)))
    a, b, looks = round(a, 2), round(b, 2), job.chat.looks
    for seen in looks:
        if (seen["start"], seen["end"], len(seen["pictures"])) == (a, b, frames):
            raise Refused("Look %d already showed %s-%s in %d pictures, and they are earlier in this conversation. Look "
                          "somewhere else (earlier, wider, or the next stretch of the map), or propose the moments you "
                          "have looked at." % (seen["n"], clock(a), clock(b), frames))
    look = {"n": len(looks) + 1, "start": a, "end": b, "pictures": [], "loudest": None, "said": None}
    try:
        said = _what_is_said(job, rec, look)
    except hearing.Stopped:
        return "Not looked at: the person pressed Stop."
    for i in range(frames):
        t = round(a + (i + 0.5) * (b - a) / frames, 2)
        picture = recording.frame(rec, at=t)
        if not picture:
            raise Refused("ffmpeg could not take a picture from %s at %s, so nothing was looked at. Tell the person."
                          % (rec["name"], clock(t)))
        look["pictures"].append({"t": t, "frame": picture})
    for i, p in enumerate(look["pictures"], 1):
        job.attach(os.path.join(recording.WORK, "frames", p["frame"]),
                   "Look %d, picture %d of %d, at %s:" % (look["n"], i, frames, clock(p["t"])))
    lines = ["Look %d: %s-%s (%d s). Its %d pictures come next, each with its time."
             % (look["n"], clock(a), clock(b), round(b - a), frames)]
    if fixed:
        lines.append(fixed)
    m = maps.kept(rec)
    peak = maps.loudest(m, a, b) if m else None
    if peak:
        look["loudest"] = peak[0]
        lines.append("The sound rises most at %s, %+.1f dB over the minutes around it." % (clock(peak[0]), peak[1]))
    lines += said
    lines.append("Next: in your next why, say what you saw and heard, and what led up to the peak. Then look earlier "
                 "if the start "
                 "of the action is missing, look wider if you need more, look at the next stretch of the map, or "
                 "propose the moments you have looked at.")
    looks.append(look)
    job.show(look=look)
    return "\n".join(lines)


def _what_is_said(job, rec, look):
    """Lines for Muse about what is said in the look's stretch, as heard. The phrases go into look["said"].
    Misheard words are fixed later, only for the moments Muse proposes (measured 10-27: the fix took 20-42 s a look
    on Glimmer, the hearing 5.6 s, and most looks are not picked)."""
    a, b, clock = look["start"], look["end"], recording.clock
    if not rec["sound_tracks"]:
        return ["The recording has no sound, so nothing is said."]
    said = hearing.heard(rec, a, b)
    if said is None:
        started = time.time()
        try:
            said = job.hear(rec, a, b)
        except hearing.HearingError as e:
            return ["What is said here could not be heard: %s" % e]
        look["took"] = round(time.time() - started, 1)
        hearing.keep(rec, a, b, said)
    look["said"] = said
    if not said:
        return ["No words were heard here."]
    return (["What is said here, by time, as heard (some words may be misheard; the numbers are as heard; a line "
             "marked (not sure) is one the hearing was not sure of, so do not build on it):"] +
            ["%s %s%s" % (clock(p["from"]), p["text"], " (not sure)" if _unsure(p) else "") for p in said])


def _fixed_said(job, rec, look, a, b):
    """The lines heard from a to b with misheard words fixed by Muse from the context and, when the recipe reads the
    screen, from the words written on it: [{from, to, text, heard, kept, named}], where kept says why a line stays as
    heard and named lists the names the fix took from the screen."""
    said = [p for p in look.get("said") or [] if a <= p["from"] < b]
    if not said:
        return []
    heard = [p["text"] for p in said]
    before = [p["text"] for p in look["said"] if p["from"] < a]
    clip = job.chat.clip
    names = []
    if _reads_screen(job):
        try:
            names = screen.words(rec, a, b)
        except screen.ScreenError:
            pass
    try:
        fixed, kept = hearing.fix(job.ask_aside, heard, clip["title"] if clip else "video", before, names)
    except mind.MindError as e:
        fixed, kept = heard, ["Muse could not be asked (%s)" % e] * len(heard)
    return [dict(p, text=text, heard=p["text"], kept=why, named=hearing.named(p["text"], text, names))
            for p, text, why in zip(said, fixed, kept)]


def propose_moments(job, why, moments):
    rec = _open(job, "propose")
    clock, looks = recording.clock, job.chat.looks
    if not looks:
        raise Refused("You have not looked at any stretch yet. Look first: a moment you propose must be one you saw.")
    picked = []
    for i, m in enumerate(moments, 1):
        if not 1 <= m["look"] <= len(looks):
            raise Refused("Moment %d names look %d, but your looks are: %s. Name one of them, or look first."
                          % (i, m["look"], "; ".join("%d. %s-%s" % (s["n"], clock(s["start"]), clock(s["end"]))
                                                      for s in looks)))
        look = looks[m["look"] - 1]
        a = _time("moment %d's start" % i, m["start"]) if m.get("start") else look["start"]
        b = _time("moment %d's end" % i, m["end"]) if m.get("end") else look["end"]
        if round(a) < round(look["start"]) or round(b) > round(look["end"]) or b <= a:
            raise Refused("Moment %d (%s-%s) is not inside look %d (%s-%s). Give a start and an end inside what you "
                          "saw, or look at that stretch first." % (i, clock(a), clock(b), look["n"],
                                                                   clock(look["start"]), clock(look["end"])))
        a, b = max(a, look["start"]), min(b, look["end"])
        if not SHORTEST <= round(b - a) <= LONGEST:
            raise Refused("Moment %d is %d s long; a clip is %d to %d s. Give it a start and an end inside look %d "
                          "(%s-%s)." % (i, round(b - a), SHORTEST, LONGEST, look["n"], clock(look["start"]),
                                        clock(look["end"])))
        pictures = look["pictures"]
        if m.get("picture") and m["picture"] > len(pictures):
            raise Refused("Look %d has %d pictures, so moment %d cannot show picture %d."
                          % (look["n"], len(pictures), i, m["picture"]))
        aim = look["loudest"] if look["loudest"] is not None and a <= look["loudest"] <= b else (a + b) / 2
        shown = pictures[m["picture"] - 1] if m.get("picture") else min(pictures, key=lambda p: abs(p["t"] - aim))
        for other in picked:
            if a < other["end"] and other["start"] < b:
                raise Refused("Moments %d and %d overlap (%s-%s and %s-%s). Each moment must be its own stretch: move "
                              "one, or keep only the better." % (other["n"], i, clock(other["start"]),
                                                                  clock(other["end"]), clock(a), clock(b)))
        picked.append({"n": i, "look": look["n"], "start": round(a, 2), "end": round(b, 2), "reason": m["reason"],
                       "frame": shown["frame"], "t": shown["t"]})
    if _reads_screen(job):
        for p in picked:
            screen.ahead(rec, p["start"], p["end"])
    started = time.time()
    try:
        for p in picked:
            p["said"] = _fixed_said(job, rec, looks[p["look"] - 1], p["start"], p["end"])
    except hearing.Stopped:
        return "Not proposed: the person pressed Stop."
    fixing = round(time.time() - started, 1)
    for p in picked:
        p["preview"] = previews.cut(job.chat, job.n, rec, p["start"], p["end"])
    job.chat.moments = picked
    job.show(moments=picked, fixing=fixing)
    lines = ["Proposed %d moment%s. The person sees each as a card with your reason, a picture from your look and "
             "what is said in it, misheard words fixed from the context; a rough video of each is being cut and plays "
             "on its card when it is ready." % (len(picked), "" if len(picked) == 1 else "s")]
    for p in picked:
        lines.append("%d. %s-%s (%d s, from look %d): %s" % (p["n"], clock(p["start"]), clock(p["end"]),
                                                           round(p["end"] - p["start"]), p["look"], p["reason"]))
        lines += ["   %s %s%s" % (clock(s["from"]), s["text"], " (as heard)" if s["kept"] else "") for s in p["said"]]
    lines.append("Go on with moment 1 by yourself and make the clip from it: do not ask the person to choose, and do not "
                 "list the moments again in words. They can tap another card to switch. Nothing is cut, laid out, captioned or blurred until a "
                 "tool's result says it is: never tell the person a clip is ready when no tool made it.")
    return "\n".join(lines)


FACE_MARGIN = 20
SHORTEST_WORDS = 1.0
LISTED = 2.0
MOVING_FACE = 0.25


def _pixels(name, box, rec):
    """A box on Muse's 0-1000 grid of the picture ([x1, y1, x2, y2]) in pixels of the recording ([x, y, w, h])."""
    x1, x2 = sorted(box[0::2])
    y1, y2 = sorted(box[1::2])
    if x2 - x1 < 5 or y2 - y1 < 5:
        raise Refused("%s %s has no size: give [x1, y1, x2, y2] on your 0-1000 grid of the picture, the top-left "
                      "corner first." % (name, box))
    w, h = rec["width"], rec["height"]
    return [x1 * w / 1000.0, y1 * h / 1000.0, (x2 - x1) * w / 1000.0, (y2 - y1) * h / 1000.0]


def _grid(box, rec):
    w, h = rec["width"], rec["height"]
    return "[%d, %d, %d, %d]" % (round(box[0] * 1000 / w), round(box[1] * 1000 / h),
                                 round((box[0] + box[2]) * 1000 / w), round((box[1] + box[3]) * 1000 / h))


def _the_face(job, rec, a, b, face):
    """(the face's box in the recording or None, a sentence for Muse): found by the program, and snapped to what
    is really there when Muse gave a rough box. faces.Stopped when the person pressed Stop."""
    try:
        found = job.find_faces(rec, a, b)
    except faces.NoFaces as e:
        if face:
            return _pixels("face", face, rec), "Faces could not be looked for (%s), so your face box is used as given." % e
        return None, "Faces could not be looked for: %s" % e
    n = len(found["samples"])
    if face:
        rough = _pixels("face", face, rec)
        g = faces.snap(rough, found)
        if g:
            return g["box"], ("The face inside your face box was found in %d of %d pictures, at %s on your grid; the "
                              "layout uses it." % (g["seen"], n, _grid(g["box"], rec)))
        return rough, "No face was found inside your face box in %d pictures, so the box is used as you gave it." % n
    g = faces.steady(found)
    if g:
        return g["box"], ("A face sits at the same place in %d of %d pictures, at %s on your grid; the layout uses it."
                          % (g["seen"], n, _grid(g["box"], rec)))
    best = next(iter(faces.places(found)), None)
    if best and best["seen"] >= MOVING_FACE * n:
        return best["box"], ("The face moves; it is most often at %s on your grid (%d of %d pictures), and the layout "
                             "uses that place." % (_grid(best["box"], rec), best["seen"], n))
    return None, "No face was found in the moment's %d pictures." % n


def _face_times(state, rec):
    """[(clip second, [boxes in the frame to keep words off])] for every picture the face finder took of the clip,
    and the seconds between two pictures. The product Muse named is in every picture."""
    clip, out, every = state["clip"], [], faces.EVERY
    lay = clip["layout"]

    def on_frame(at, box):
        return layouts.to_frame(dict(lay, parts=talk.parts_at(lay, at)) if lay["kind"] == "talk" else lay, box)

    product = layouts.to_frame(lay, clip["product"]) if clip.get("product") else []
    if product:
        out = [(round(k * every, 2), list(product)) for k in range(int(state["length"] / every) + 1)]
    for a, b in clip["pieces"]:
        found = faces.kept(rec, a, b)
        if not found:
            continue
        every = found["every"]
        for t, boxes in faces.between(found, a, b):
            at = project.from_recording(clip["pieces"], t)
            out.append((at, [f for box in boxes for f in on_frame(at, box)] + product))
    return sorted(out, key=lambda x: x[0]), every


def _covered(box, times, every, a, b):
    """The stretches of a-b (clip seconds) where box comes within FACE_MARGIN pixels of a face: [(from, to)]."""
    x, y, w, h = box[0] - FACE_MARGIN, box[1] - FACE_MARGIN, box[2] + 2 * FACE_MARGIN, box[3] + 2 * FACE_MARGIN
    hit = []
    for t, boxes in times:
        if t + every / 2 <= a or t - every / 2 >= b:
            continue
        if any(f[0] < x + w and x < f[0] + f[2] and f[1] < y + h and y < f[1] + f[3] for f in boxes):
            lo, hi = max(a, t - every / 2), min(b, t + every / 2)
            if hit and lo <= hit[-1][1] + 1e-6:
                hit[-1] = (hit[-1][0], hi)
            else:
                hit.append((lo, hi))
    return hit


def _clear(hit, a, b, least=LISTED):
    """The stretches of a-b that hit leaves free, each at least `least` seconds, so a title fits in it."""
    out, at = [], a
    for lo, hi in hit:
        if lo - at >= least:
            out.append((at, lo))
        at = max(at, hi)
    if b - at >= least:
        out.append((at, b))
    return out


def _spans(spans, inward=False):
    """Stretches in tenths of a second. inward rounds each one inside itself, so a free stretch read back and sent
    again is still free."""
    if inward:
        spans = [(math.ceil(lo * 10 - 1e-6) / 10, math.floor(hi * 10 + 1e-6) / 10) for lo, hi in spans]
    return ", ".join("%g-%g s" % (round(lo, 1), round(hi, 1)) for lo, hi in spans)


def _place_lines(state, rec):
    times, every = _face_times(state, rec)
    length, lay = state["length"], state["clip"]["layout"]
    lines = []
    for name, box in lay["places"].items():
        clear = _clear(_covered(box, times, every, 0, length), 0, length)
        where = "%s (y %d to %d)" % (name, box[1], box[1] + box[3])
        if clear == [(0, length)]:
            lines.append("- %s: clear of faces for the whole clip" % where)
        elif clear:
            lines.append("- %s: clear of faces at %s" % (where, _spans(clear, inward=True)))
        else:
            lines.append("- %s: never clear for %g s or more; do not use it" % (where, LISTED))
    return lines


def _clip_state(job):
    state = job.chat.project.state()
    if not state["clip"]:
        raise Refused("There is no clip yet. Make it from one of your moments first.")
    return state


def _show_frame(job, state, rec, t, caption):
    """Sends Muse the clip's frame at t, right after the results. Returns the sentence that says so."""
    try:
        path = render.frame(rec, state, t, job.chat.clip["style"] if job.chat.clip else "clean")
    except render.NotDrawn as e:
        return "The frame could not be drawn: %s" % e
    job.attach(path, caption)
    return "Its frame at %g s comes next as a picture." % round(t, 2)


def make_clip(job, why, moment, layout, crop=None, face=None, product=None):
    rec = _open(job, "make a clip from")
    moments = job.chat.moments
    if not moments:
        raise Refused("No moments are proposed yet. Look at the recording and propose moments first: the clip is made "
                      "from one of them.")
    if not 1 <= moment <= len(moments):
        raise Refused("There are %d moments, so moment %d is not one of them: %s." % (
            len(moments), moment, "; ".join("%d. %s-%s" % (m["n"], recording.clock(m["start"]),
                                                          recording.clock(m["end"])) for m in moments)))
    m = moments[moment - 1]
    a, b = m["start"], m["end"]
    if layout == "talk":
        return _make_talk(job, rec, moment, m)
    try:
        box, face_line = _the_face(job, rec, a, b, face)
    except faces.Stopped:
        return "Not made: the person pressed Stop."
    try:
        lay = layouts.make(layout, (rec["width"], rec["height"]), box, _pixels("crop", crop, rec) if crop else None)
    except ValueError as e:
        raise Refused(str(e))
    item = [round(v) for v in _pixels("product", product, rec)] if product else None
    if item and not layouts.to_frame(lay, item):
        lay["notes"].append("The product you named is not in the clip's picture: choose a crop that holds it.")
    if not item and (job.chat.clip or {}).get("recipe") == "seller":
        lay["notes"].append("This is a seller clip and no product box was named, so nothing keeps words off the "
                            "product. Make the clip again with product: [x1, y1, x2, y2] on your grid around the product "
                            "in the picture.")
    private = _private_lines(job, rec, lay, [[a, b]])
    if private is None:
        return "Not made: the person pressed Stop."
    people = _people(rec, [[a, b]], box)
    state = job.edit({"do": "clip", "moment": moment, "pieces": [[a, b]], "layout": lay, "product": item,
                      "people": [{k: v for k, v in p.items() if k != "crop"} for p in people or []],
                      "speech": _speech(rec, job.chat.looks[m["look"] - 1], [[a, b]])})
    for f in private["shown"]:
        state = job.edit({"do": "blur", "box": f["box"], "start": f["start"], "end": f["end"], "private": f["kind"]})
    lines = ["Made the clip from moment %d (%s-%s, %g s) in the %s layout." % (
        moment, recording.clock(a), recording.clock(b), state["length"], lay["kind"]), face_line] + lay["notes"]
    if private["said"]:
        lines.append(private["said"])
    if lay.get("safe"):
        lines.append(lay["safe"]["said"])
    lines.append("Places for words on screen, kept clear of the faces the program found%s (the band from y %d to %d "
                 "is kept for the captions):" % (" and the product you named" if item else "", lay["captions"][1],
                                                 lay["captions"][1] + lay["captions"][3]))
    lines += _place_lines(state, rec)
    if state["titles"]:
        times, every = _face_times(state, rec)
        for t in state["titles"]:
            hit = _covered(t["box"], times, every, t["start"], t["end"])
            lines.append('Kept from before: the title "%s" at the %s, %g-%g s%s.' % (
                t["text"], t["place"], t["start"], t["end"],
                "; it now covers a face%s at %s: take it off with remove_title, then put it again" % (
                    " or the product" if state["clip"].get("product") else "", _spans(hit)) if hit else ""))
    peak = m.get("t") if m.get("t") is not None else (a + b) / 2
    at = min(max(0.0, peak - a), state["length"] - 0.1)
    lines.append(_show_frame(job, state, rec, at, "The clip at %g s (moment %d, %s layout):" % (
        round(at, 2), moment, lay["kind"])))
    lines += _people_lines(job, people, rec)
    lines.append("Look at it: if a face or the action is cut badly, make the clip again with another layout or crop. "
                 "Then put the title on.")
    job.show(clip=job.chat.clip_view())
    return "\n".join(lines)


def _reads_screen(job):
    clip = job.chat.clip
    return bool(clip) and (RECIPES.get(clip["recipe"]) or {}).get("read") == "screen"


NEAR_NAME = 0.5


def _screen_words(job, rec, state):
    """The words written on the screen in the clip, for a recipe that reads the screen; [] otherwise, or when it cannot
    be read."""
    if not _reads_screen(job):
        return []
    try:
        return [w for a, b in state["clip"]["pieces"] for w in screen.words(rec, a, b)]
    except screen.ScreenError:
        return []


def _misspelled(words, shown):
    """[(a word in words, the screen's word it looks like)]: a name of five letters or more that the screen writes
    another way, such as "Egret" where the screen shows "Eaglet"."""
    out, long_ = [], {}
    for s in shown:
        if len(s) >= 5:
            long_.setdefault(s.lower(), s)
    for w in dict.fromkeys(screen.NAME.findall(words)):
        x = w.lower()
        if len(x) < 5 or any(x in s or s in x for s in long_):
            continue
        near = min((hearing._changed(x, s), s) for s in long_) if long_ else None
        if near and near[0] <= NEAR_NAME:
            out.append((w, long_[near[1]]))
    return out


def _names_said(wrong, then):
    """The sentence for Muse about names written another way than on the screen, or []."""
    if not wrong:
        return []
    return ["Spelled as the screen writes it? %s. If the speaker meant the screen's words, %s." % (
        "; ".join('"%s" is not on the screen, which shows "%s"' % pair for pair in wrong), then)]


def _private_spans(lines):
    return "; ".join("%s at %g-%g s" % (f["kind"], f["start"], f["end"]) for f in lines)


NOTHING_PRIVATE = ("no line holds an email address, a phone number, a card or ID number, a key, an account number or a "
                   "password")


def _private_lines(job, rec, lay, pieces):
    """{"shown": the private lines the screen shows in the clip's picture, "said": the sentence for Muse}, empty when
    the recipe does not read the screen, or None when the person pressed Stop."""
    if not _reads_screen(job):
        return {"shown": [], "said": ""}
    job.working("Reading the screen for anything private")
    try:
        found = screen.hidden(rec, pieces, on_start=job.started_work)
    except screen.Stopped:
        return None
    except screen.ScreenError as e:
        return {"shown": [], "said": "The screen could not be read (%s), so nothing private is hidden: look at the "
                                     "pictures for an email address, a phone number or a key, and blur it." % e}
    finally:
        job.done_working()
    shown = [f for f in found if layouts.to_frame(lay, f["box"])]
    if not shown:
        return {"shown": [], "said": "The program read the screen: %s." % NOTHING_PRIVATE}
    return {"shown": shown, "said": "The program read the screen and hid %d private line%s, each for as long as it "
                                    "shows: %s. The words themselves are not repeated here, so they cannot reach the "
                                    "title or the post." % (len(shown), "" if len(shown) == 1 else "s",
                                                            _private_spans(shown))}


def _private_check(job, rec, state):
    """(ok, found) for the check that every private line the clip shows is hidden for as long as it shows."""
    lay = state["clip"]["layout"]
    job.working("Reading the screen for anything private")
    try:
        found = screen.hidden(rec, state["clip"]["pieces"], on_start=job.started_work)
    except screen.Stopped:
        return False, "not read: the person pressed Stop"
    except screen.ScreenError as e:
        return False, "the screen could not be read (%s)" % e
    finally:
        job.done_working()
    shown = [f for f in found if layouts.to_frame(lay, f["box"])]
    bare = [f for f in shown if not screen.covered(f, state["blurs"])]
    if bare:
        return False, "not hidden: %s" % _private_spans(bare)
    return True, ("%d private line%s, each hidden for as long as it shows" % (len(shown), "" if len(shown) == 1 else
                                                                               "s") if shown else NOTHING_PRIVATE)


def _make_talk(job, rec, moment, m):
    """make_clip in the talk layout: the people found, who speaks when, and the shots that follow them."""
    a, b = m["start"], m["end"]
    try:
        found = job.find_faces(rec, a, b)
    except faces.NoFaces as e:
        raise Refused("The talk layout needs the people's faces, and they could not be looked for: %s" % e)
    except faces.Stopped:
        return "Not made: the person pressed Stop."
    persons = talk.people(found)
    if len(persons) < 2:
        raise Refused("The talk layout follows two or more people talking, and the program found %d sitting still in "
                      "the moment's %d pictures. Choose the face or the subject layout." % (len(persons),
                                                                                         len(found["samples"])))
    heard = job.chat.looks[m["look"] - 1].get("said")
    job.working("Watching who speaks")
    shots, _ = talk.plan(rec, [[a, b]], heard, persons)
    lay = talk.make((rec["width"], rec["height"]), persons, shots)
    people = _people(rec, [[a, b]], None)
    state = job.edit({"do": "clip", "moment": moment, "pieces": [[a, b]], "layout": lay, "product": None,
                      "people": [{k: v for k, v in p.items() if k != "crop"} for p in people or []],
                      "speech": _speech(rec, job.chat.looks[m["look"] - 1], [[a, b]])})
    lines = ["Made the clip from moment %d (%s-%s, %g s) in the talk layout. It follows %d people, numbered from the "
             "left: %s." % (moment, recording.clock(a), recording.clock(b), state["length"], len(persons), "; ".join(
                 "person %d at %s on your grid" % (p["n"], _grid(p["box"], rec)) for p in persons)),
             "Who speaks was measured: whose mouth moves clearly more %s." % (
                 "while words are heard" if heard is not None else "(the moment was never heard, so from movement "
                                                                    "alone)"),
             "Shots: %s." % talk.said(lay["shots"]),
             "If a shot shows the wrong person, or a reaction should be seen, change it with change_shot."]
    lines += lay["notes"] + [lay["safe"]["said"]]
    lines.append("Places for words on screen, kept clear of every person's head (the band from y %d to %d is kept for "
                 "the captions):" % (lay["captions"][1], lay["captions"][1] + lay["captions"][3]))
    lines += _place_lines(state, rec)
    peak = m.get("t") if m.get("t") is not None else (a + b) / 2
    at = min(max(0.0, peak - a), state["length"] - 0.1)
    lines.append(_show_frame(job, state, rec, at, "The clip at %g s (moment %d, talk layout):" % (round(at, 2),
                                                                                                    moment)))
    lines += _people_lines(job, people, rec)
    job.show(clip=job.chat.clip_view())
    return "\n".join(lines)


def change_shot(job, why, start, end, show):
    rec = _open(job, "change a shot of")
    state = _clip_state(job)
    lay = state["clip"]["layout"]
    if lay["kind"] != "talk":
        raise Refused("Only a clip in the talk layout has shots; this one is in the %s layout." % lay["kind"])
    length, count = state["length"], len(lay["people"])
    end = min(end, length)
    if not 0 <= start < length:
        raise Refused("start %g s is outside the clip, which is %g s long: use seconds of the clip." % (start, length))
    if end - start < talk.SHORTEST_TURN / 2:
        raise Refused("A shot of %g-%g s is too short to see; give it at least %g s." % (start, end,
                                                                                       talk.SHORTEST_TURN / 2))
    if not 0 <= show <= count:
        raise Refused("show %d is not one of the people: use 1 to %d (numbered from the left), or 0 for everyone."
                      % (show, count))
    shots = talk.with_shot(lay["shots"], round(start, 2), round(end, 2), show)
    new = talk.make(tuple(lay["size"]), lay["people"], shots)
    state = job.edit({"do": "layout", "part": "shots", "box": None, "layout": new})
    lines = ["%s from %g to %g s. Shots now: %s." % ("Everyone is shown" if show == 0 else "Person %d is shown" % show,
                                                     start, end, talk.said(shots)),
             "Places for words now:"] + _place_lines(state, rec)
    lines.append(_show_frame(job, state, rec, (start + end) / 2, "The clip at the changed shot:"))
    job.show(clip=job.chat.clip_view())
    return "\n".join(lines)


MOST_FONTS_SHOWN = 12


def _letters(job, state):
    """The letters the clip's words are written in: its titles and captions, else what the person typed last."""
    words = [t["text"] for t in state["titles"]] + [l["text"] for l in (state["captions"] or {}).get("lines") or []]
    if not words:
        words = [e["text"] for e in job.chat.events if e["kind"] == "person"][-1:]
    return "".join(sorted(set("".join(words)) - set(" \n\t")))


def _font_entry(font, every):
    """The entry of every (text.fonts()) whose family or one of whose faces is named font, or None."""
    name = " ".join(font.split()).lower()
    return next((f for f in every if f["family"].lower() == name or name in (x.lower() for x in f["faces"])), None)


def _free_fonts(every):
    free = [f["family"] for f in sorted(every, key=lambda f: f["where"] != "studio") if f["free"]]
    return ", ".join(free[:MOST_FONTS_SHOWN]) + (" and %d more" % (len(free) - MOST_FONTS_SHOWN)
                                                  if len(free) > MOST_FONTS_SHOWN else "")


def list_fonts(job, why):
    state = _clip_state(job)
    letters = _letters(job, state)
    every = text.fonts(letters or None)
    free = [f for f in every if f["free"]]
    other = [f for f in every if not f["free"]]
    lines = ["Fonts that can write every letter of the clip's words (%s): %d free to use in a video, %d not known to "
             "be free." % ("the studio's and this computer's" if letters else "any words", len(free), len(other))]
    lines += ["- %s (%s, %s): %s" % (f["family"], f["where"], f["licence"], ", ".join(f["faces"])) for f in free]
    if other:
        lines.append("Not known to be free, so use one only when the person names it: %s."
                     % ", ".join(f["family"] for f in other))
    lines.append("The person can add any free font: they install it on this computer, and it is listed here.")
    return "\n".join(lines)


WORD_PARTS = {"title": "the title", "caption": "the captions", "big": "the big words of a hit"}


PAINT_WORDS = {"text": "the letters", "key": "the key word", "edge": "the outline", "glow": "the glow",
               "shade": "the shadow"}


def _paints(colours):
    return ", ".join("%s in %s" % (PAINT_WORDS[n], colours[n]) for n in text.PAINTS if n in colours)


def set_style(job, why, style, font=None, size=None, colours=None, part=None, person_said=None):
    state = _clip_state(job)
    shown = None
    colours = colours or {}
    if part and not font and not size and not colours:
        raise Refused("A part takes a font, a size or colours of its own: give them for %s, or leave part out."
                      % WORD_PARTS[part])
    if font:
        letters = _letters(job, state)
        every = text.fonts(letters or None)
        entry = _font_entry(font, every)
        if entry is None:
            known = _font_entry(font, text.fonts())
            raise Refused(("%s cannot write every letter of the clip's words." % known["family"] if known else
                           "There is no font %s in the studio or on this computer." % font) +
                          " Free fonts that can: %s. The person can install another free font on this computer."
                          % _free_fonts(every))
        if not entry["free"] and not _asked(job, person_said):
            raise Refused("%s is not known to be free (%s), so it may not be allowed in a video someone posts. Choose "
                          "a free font: %s. Use it only if the person named it themselves; then give person_said, "
                          "their own words." % (entry["family"], entry["licence"], _free_fonts(every)))
        shown = entry["family"]
    before = state["style"] or {}
    every, fonts = before.get("font"), dict(before.get("fonts") or {})
    whole, sizes = before.get("size"), dict(before.get("sizes") or {})
    paints = dict({"text": before["colour"]} if before.get("colour") else {}, **(before.get("colours") or {}))
    part_paints = {k: dict(v) for k, v in (before.get("part_colours") or {}).items()}
    if part:
        fonts.update({part: font} if font else {})
        sizes.update({part: size} if size else {})
        part_paints[part] = dict(part_paints.get(part) or {}, **colours)
    else:
        every, fonts = (font, {}) if font else (every, fonts)
        whole, sizes = (size, {}) if size else (whole, sizes)
        paints.update(colours)
        part_paints = {k: {n: c for n, c in v.items() if n not in colours} for k, v in part_paints.items()}
    part_paints = {k: v for k, v in part_paints.items() if v}
    try:
        drawn = text.look(style, every, None, fonts or None, size=whole, sizes=sizes or None, colours=paints or None,
                          part_colours=part_paints or None)
    except ValueError as e:
        raise Refused(str(e))
    entry = {"do": "style", "name": style, "font": every}
    entry.update({"fonts": fonts} if fonts else {})
    entry.update({"size": whole} if whole else {})
    entry.update({"sizes": sizes} if sizes else {})
    entry.update({"colours": paints} if paints else {})
    entry.update({"part_colours": part_paints} if part_paints else {})
    state = job.edit(entry)
    said = [" with the font %s" % shown if shown else "", " at size %d" % size if size else "",
            " with %s" % _paints(colours) if colours else ""]
    lines = ["The words on screen now use the %s style%s%s." % (style, "".join(said), " for %s only" % WORD_PARTS[part]
                                                                if part and (shown or size or colours) else "")]
    kept = ["the font %s for %s" % (f, WORD_PARTS[k]) for k, f in fonts.items() if not (part and k == part)] + \
           ["size %d for %s" % (n, WORD_PARTS[k]) for k, n in sizes.items() if not (part and k == part)]
    whole_kept = {n: c for n, c in paints.items() if part or n not in colours}
    kept += [_paints(whole_kept)] if whole_kept else []
    for k, v in part_paints.items():
        left = {n: c for n, c in v.items() if not (part == k and n in colours)}
        kept += ["%s for %s" % (_paints(left), WORD_PARTS[k])] if left else []
    if kept:
        lines.append("Kept from before: %s." % ", ".join(kept))
    lines += list(dict.fromkeys(st.note for st in drawn.values() if st.note))
    if state["titles"]:
        t = state["titles"][0]
        rec = _open(job, "show")
        lines.append(_show_frame(job, state, rec, (t["start"] + t["end"]) / 2, "The clip with the %s style:" % style))
    return "\n".join(lines)


def _frame_box(name, grid):
    """[x, y, w, h] on the 1080x1920 frame from [x1, y1, x2, y2] on Muse's 0-1000 grid of the clip's frame."""
    x1, y1, x2, y2 = grid
    if x2 <= x1 or y2 <= y1:
        raise Refused("%s %s is not a box: give [x1, y1, x2, y2] with x2 right of x1 and y2 below y1." % (name, grid))
    return [round(x1 * 1080 / 1000), round(y1 * 1920 / 1000), round((x2 - x1) * 1080 / 1000),
            round((y2 - y1) * 1920 / 1000)]


def _covers(drawn, lay, state):
    """What lies over drawn on a phone, or under it in the clip, as short reasons: what the apps draw on top, and
    the captions band when the clip has captions."""
    out = layouts.unsafe(drawn)
    y, h = lay["captions"][1], lay["captions"][3]
    if state["captions"] and drawn[1] < y + h and y < drawn[1] + drawn[3]:
        out.append("the captions run from y %d to %d" % (y, y + h))
    return out


def _phone(drawn, lay, state, asked):
    """What a phone or the captions cover of drawn, as notes for the person when they asked for the spot; Refused
    when Muse chose it."""
    covers = _covers([round(v) for v in drawn], lay, state)
    if covers and not asked:
        raise Refused("On a phone, %s. Pick one of the layout's places, or a box inside x %d-%d, y %d-%d; give "
                      "person_said only when the person asked for this spot." % (
                          "; ".join(covers), layouts.SAFE[0], layouts.SAFE[0] + layouts.SAFE[2], layouts.SAFE[1],
                          layouts.SAFE_BOTTOM))
    return ["Placed where the person asked. On a phone, %s; tell the person." % "; ".join(covers)] if covers else []


def _off_faces(state, rec, drawn, a, b, asked, what):
    """Notes for drawn from a to b (clip seconds) when the person asked for the spot, saying the faces it covers;
    Refused, with the places and when they are clear, when it would cover a face Muse should keep clear of."""
    times, every = _face_times(state, rec)
    hit = _covered(drawn, times, every, a, b)
    face = "a face%s" % (" or the product" if state["clip"].get("product") else "")
    if hit and not asked:
        raise Refused("The %s would cover %s at %s. Pick another box or time; the layout's places and when they are "
                      "clear:\n%s" % (what, face, _spans(hit), "\n".join(_place_lines(state, rec))))
    if hit:
        return ["It covers %s at %s, where the person asked for it; tell the person." % (face, _spans(hit))], hit
    return [], hit


def _on_grid(box):
    return "[%s]" % ", ".join("%g" % v for v in box)


def _spot(name, state, place, box):
    """The frame box Muse gave: a box on its grid, or one of the layout's places for words."""
    lay = state["clip"]["layout"]
    if box is not None:
        return _frame_box(name, box), "box at %s on your grid" % _on_grid(box)
    if place in lay["places"]:
        return list(lay["places"][place]), "the %s" % place
    raise Refused("Give a box, or one of the layout's places for words: %s." % ", ".join(lay["places"]))


def _asked(job, person_said):
    """True when person_said quotes words the person typed in this chat, False when it is empty; Refused when the
    person never said it, so Muse cannot claim a request nobody made."""
    if not person_said:
        return False
    quote = " ".join(person_said.split()).lower()
    said = [" ".join(e["text"].split()).lower() for e in job.chat.events if e["kind"] == "person"]
    if len(quote) < 3 or not any(quote in s for s in said):
        raise Refused("The person did not say \"%s\" in this chat. person_said must quote their own words asking for "
                      "this; when nobody asked, choose yourself and leave person_said out." % person_said)
    return True


def add_title(job, why, words, start, end, place=None, key=None, box=None, person_said=None):
    asked = _asked(job, person_said)
    rec = _open(job, "put words on")
    state = _clip_state(job)
    try:
        (words,), spelling = _spell(job, [words])
    except hearing.Stopped:
        return "Not put on: the person pressed Stop."
    length, lay = state["length"], state["clip"]["layout"]
    if not 0 <= start < length:
        raise Refused("start %g s is outside the clip, which is %g s long: use a start from 0 to %g, in seconds of "
                      "the clip." % (start, length, round(length - SHORTEST_WORDS, 1)))
    end = min(end, length)
    if end - start < SHORTEST_WORDS:
        raise Refused("The title must stay on screen at least %g s; %g-%g s is shorter." % (SHORTEST_WORDS, start, end))
    if box is None and place is None:
        raise Refused("Give one of the layout's places for words (%s), or a box." % ", ".join(lay["places"]))
    if box is not None:
        band = _frame_box("box", box)
        place = "box at %d, %d" % (band[0], band[1])
    elif place not in lay["places"]:
        raise Refused("The %s layout has no place called %s; its places are: %s. Or give a box." % (
            lay["kind"], place, ", ".join(lay["places"])))
    else:
        band = lay["places"][place]
    st = render.style_of(state, job.chat.clip["style"] if job.chat.clip else "clean")
    try:
        size = text.measure(words, st, "title", band[2])
    except ValueError as e:
        raise Refused(str(e))
    if size.w > band[2] or size.h > band[3]:
        raise Refused("These words take %d lines at the title size (%d px tall), and the %s place holds %d px: use "
                      "fewer words." % (len(size.lines), size.h, place, band[3]))
    fixed = ""
    try:
        text.title(words, start, end, band, st, key)
    except ValueError as e:
        whole = [words[a:b].strip() for a, b in text._atoms(words)]
        holding = [w for w in whole if key and key in w]
        if len(holding) != 1:
            raise Refused(str(e))
        key, fixed = holding[0], 'The key "%s" is part of the word "%s", so that word is the key.' % (key, holding[0])
        text.title(words, start, end, band, st, key)
    drawn = [band[0] + (band[2] - size.w) / 2, band[1] + (band[3] - size.h) / 2, size.w, size.h]
    notes = _phone(drawn, lay, state, asked)
    times, every = _face_times(state, rec)
    hit = _covered(drawn, times, every, start, end)
    if hit and asked:
        notes.append("It covers a face%s at %s, where the person asked for it; tell the person." % (
            " or the product" if state["clip"].get("product") else "", _spans(hit)))
    elif hit:
        others = []
        for name, box in lay["places"].items():
            clear = _clear(_covered(box, times, every, 0, length), 0, length)
            others.append("%s: %s" % (name, _spans(clear, inward=True) if clear else "never clear"))
        stuck = "" if any(not o.endswith("never clear") for o in others) else (
            " No place is clear for %g s or more: the faces%s fill the picture. %sor leave the title off." % (
                LISTED, " and the product" if state["clip"].get("product") else "",
                "Make the clip again with a product box around the product only, " if state["clip"].get("product")
                else ""))
        raise Refused("At the %s, the title would cover a face%s at %s. Clear: %s. Pick a time or a place from these.%s"
                      % (place, " or the product" if state["clip"].get("product") else "", _spans(hit),
                         "; ".join(others), stuck))
    state = job.edit({"do": "title", "text": words, "start": round(start, 2), "end": round(end, 2), "place": place,
                      "box": band, "key": key})
    lines = ['The title "%s" is on screen at the %s from %g to %g s: %d line%s, %d x %d px%s.' % (
        words, place, start, end, len(size.lines), "" if len(size.lines) == 1 else "s", size.w, size.h,
        "" if hit else ", clear of every face the program found%s" % (
            " and the product" if state["clip"].get("product") else ""))] + notes
    if fixed:
        lines.append(fixed)
    if spelling:
        lines.append(spelling)
    lines += _names_said(_misspelled(words, _screen_words(job, rec, state)), "put the title on again")
    lines.append(_show_frame(job, state, rec, (start + end) / 2, 'The clip with the title "%s":' % words))
    job.show(clip=job.chat.clip_view())
    return "\n".join(lines)


CAPTION_LETTERS = 16
HITS = {"flash": "a short white flash", "shake": "the picture shakes and settles", "punch": "a quick zoom in and back",
        "word": "a big word that pops in, such as POW"}


def _main_part(layout):
    """The middle of the part that shows the action: the card, the picture below the face, or the whole frame."""
    parts = {p["name"]: p["dst"] for p in layout["parts"]}
    x, y, w, h = parts.get("card") or parts.get("bottom") or parts.get("whole") or [0, 0, layouts.W, layouts.H]
    return [round(x + w / 2), round(y + h / 2)]


PARTS = ("card", "circle", "top", "bottom", "whole")


def move_part(job, why, part, box):
    rec = _open(job, "move a part of")
    state = _clip_state(job)
    if state["clip"]["layout"]["kind"] == "talk":
        raise Refused("A talk clip has no parts to move: it cuts between the people. Change who is shown when with "
                      "change_shot.")
    band = _frame_box("box", box)
    try:
        lay = layouts.moved(state["clip"]["layout"], part, band)
    except ValueError as e:
        raise Refused(str(e))
    state = job.edit({"do": "layout", "part": part, "box": band, "layout": lay})
    lines = [lay["notes"][-1] if lay["notes"] else "The %s is at %s." % (part, band),
             "Places for words now:"] + _place_lines(state, rec)
    lines.append(_show_frame(job, state, rec, state["length"] / 2, "The clip with the %s moved:" % part))
    job.show(clip=job.chat.clip_view())
    return "\n".join(lines)


LOOKS = {"vivid": "more contrast and richer greens and blues, skin and reds kept: for games and flat footage",
         "warm": "warmer middle tones, black and white kept, so a white product stays white: for faces, food and "
                 "products",
         "film": "soft blacks and whites and calmer colour: for a quiet, calm clip",
         "none": "the footage as it was shot"}
PUSH_ZOOM = 1.08


def set_look(job, why, look):
    rec = _open(job, "colour")
    state = _clip_state(job)
    name = None if look == "none" else look
    if name is not None:
        try:
            effects.look(name)
        except ValueError as e:
            raise Refused(str(e))
    if name == state["look"]:
        raise Refused("The clip already has %s." % ("the %s look" % name if name else "no look"))
    state = job.edit({"do": "look", "name": name})
    lines = ["The footage now has the %s look: %s. Titles, captions and the frame's design keep their colours."
             % (name, LOOKS[name]) if name else "The footage is as it was shot, with no look."]
    lines.append(_show_frame(job, state, rec, state["length"] / 2, "The clip with %s:" % (
        "the %s look" % name if name else "no look")))
    return "\n".join(lines)


def slow_push(job, why, start, end, zoom=PUSH_ZOOM):
    rec = _open(job, "push in on")
    state = _clip_state(job)
    if state["clip"]["layout"]["kind"] == "talk":
        raise Refused("A talk clip cuts between the people, so it has no slow push: the cuts carry the pace.")
    length = state["length"]
    if not 0 <= start < length:
        raise Refused("start %g s is outside the clip, which is %g s long: use seconds of the clip." % (start, length))
    end = min(end, length)
    centre = _main_part(state["clip"]["layout"])
    try:
        effects.slow_push(start, end, zoom, centre)
    except ValueError as e:
        raise Refused(str(e))
    replaced = [p for p in state["pushes"] if p["start"] < end and start < p["end"]]
    state = job.edit({"do": "push", "start": round(start, 2), "end": round(end, 2), "zoom": zoom, "centre": centre})
    lines = ["A slow push in on the main picture from %g to %g s, to %g times its size, easing in and out, then it "
             "cuts back. The frame's design and the words hold still." % (start, end, zoom)]
    if replaced:
        lines.append("It replaces the push at %s." % ", ".join("%g-%g s" % (p["start"], p["end"]) for p in replaced))
    lines.append(_show_frame(job, state, rec, max(start, end - 0.1), "The clip at the end of the push:"))
    return "\n".join(lines)


def _speech(rec, look, pieces):
    """[[clip a, clip b]] while someone speaks in the clip, from every phrase heard in the look (the ones the hearing
    was not sure of too: they are still someone speaking); [] when the recording has no sound; None when the look
    was never heard, so nobody knows."""
    if not rec["sound_tracks"]:
        return []
    if look.get("said") is None:
        return None
    spans, at = [], 0.0
    for a, b in pieces:
        for p in look["said"]:
            start, end = max(a, p["from"]), min(b, p["to"])
            if start < end:
                spans.append([round(at + start - a, 2), round(at + end - a, 2)])
        at += b - a
    return sorted(spans)


SOUNDS = {"thump": "a deep hit, for a punch-in or a shake", "whoosh": "air rushing past, for a flash or a fast move",
          "pop": "a bubble, for a sticker or a big word", "ding": "a small bell, for a price or a number on screen"}


def add_sound(job, why, kind, at, level=0):
    _open(job, "add a sound to")
    state = _clip_state(job)
    length = state["length"]
    if not 0 <= at < length:
        raise Refused("at %g s is outside the clip, which is %g s long: use seconds of the clip." % (at, length))
    try:
        effects.sound(kind, at, level)
    except ValueError as e:
        raise Refused(str(e))
    replaced = [s for s in state["sounds"] if s["kind"] == kind and abs(s["at"] - at) < 0.5]
    job.edit({"do": "sound", "kind": kind, "at": round(at, 2), "level": level})
    line = "A %s at %g s%s: %s. It is heard in the finished file, not in the frames you see." % (
        kind, at, " (%+g dB)" % level if level else "", SOUNDS[kind])
    if replaced:
        line += " It replaces the %s at %g s." % (kind, replaced[0]["at"])
    return line


def set_music(job, why, music=None, start=0, level=render.MUSIC_LEVEL, duck=render.MUSIC_DUCK):
    _open(job, "put music under")
    state = _clip_state(job)
    if not music:
        if not state["music"]:
            raise Refused("The clip has no music to take off.")
        job.edit({"do": "music", "path": None})
        return "The music is off: the clip has its own sound only."
    if state["speech"] is None:
        raise Refused("The program does not know when someone speaks in this clip, because its moment was never "
                      "heard, so the music could not go down under the voice. Make the clip again from a moment you "
                      "looked at.")
    try:
        path = recording.find_music(music)
    except recording.NotOpened as e:
        raise Refused(str(e))
    entry = {"do": "music", "path": path, "from": round(start, 2), "level": level, "duck": duck}
    try:
        measured = render.music_loudness(entry, state["length"])
    except (ValueError, render.NotDrawn) as e:
        raise Refused(str(e))
    if not measured:
        raise Refused("%s is silent from its second %g for the clip's %g s: choose another file or another start."
                      % (os.path.basename(path), start, state["length"]))
    state = job.edit(entry)
    lines = ["Music: %s from its second %g, %g dB under the voice, and %g dB lower while someone speaks (%d times in "
             "this clip). It fades in over its first %g s and out over the clip's last %g s." % (
                 os.path.basename(path), start, -level, -duck, len(state["speech"]), render.MUSIC_IN,
                 render.MUSIC_OUT)]
    seconds = recording.sound_seconds(path)
    if seconds is not None and seconds - start < state["length"]:
        lines.append("It is %g s long from there, shorter than the clip, so it plays again from its start."
                     % round(seconds - start, 1))
    lines.append("It is heard in the finished file, not in the frames you see. It is the person's own music: "
                 "music they have no rights to may be muted where they post.")
    return "\n".join(lines)


UNSURE = 0.88


def _unsure(phrase):
    """True for a phrase the hearing was not sure of. Measured 10-27 on Thai FLEURS, clean and mixed with game
    sound (141 garbled phrases, 399 good): below 0.88 are 73% of the garbled ones and 4% of the good ones. A phrase
    whose fix took a name from the screen is sure: the screen wrote what the hearing could not."""
    return phrase.get("sure", 1.0) < UNSURE and not phrase.get("named")


def add_captions(job, why, on=True, lit=False):
    rec = _open(job, "caption")
    state = _clip_state(job)
    clip, length = state["clip"], state["length"]
    if not on:
        if not state["captions"]:
            raise Refused("The clip has no captions to take off.")
        job.edit({"do": "captions", "lines": [], "band": None})
        return "Captions are off."
    moments = job.chat.moments or []
    said = moments[clip["moment"] - 1].get("said") or [] if clip["moment"] <= len(moments) else []
    phrases, unsure = [], []
    for p in said:
        if _unsure(p):
            unsure.append(p)
            continue
        a = project.from_recording(clip["pieces"], p["from"])
        if a is None:
            continue
        b = project.from_recording(clip["pieces"], p["to"])
        phrases.append({"from": round(a, 2), "to": round(min(length, b if b is not None else length), 2),
                        "text": p["text"]})
    if not phrases:
        raise Refused("Nothing was heard clearly in this moment, so there are no words to caption. Leave the captions "
                      "off, and say so in your answer.")
    lines = text.caption_lines(phrases, CAPTION_LETTERS)
    band = clip["layout"]["captions"]
    state = job.edit({"do": "captions", "lines": lines, "band": [band[1], band[3]], "lit": bool(lit)})
    out = ["Captions on: %d lines of what was said, numbers exactly as heard, in the band from y %d to %d; %s."
           % (len(lines), band[1], band[1] + band[3],
              "each word lights up as it is said" if lit else "plain words, none lit, so the eye stays on the picture")]
    if unsure:
        out.append("Left out %d line%s the hearing was not sure of, because a wrong caption is worse than none: %s."
                   % (len(unsure), "" if len(unsure) == 1 else "s",
                      "; ".join("%s \"%s\"" % (recording.clock(project.from_recording(clip["pieces"], p["from"]) or 0),
                                                p["text"]) for p in unsure[:4])))
    on_camera = _on_camera(rec, clip) if lit else None
    if on_camera:
        out.append("A person is on camera (%s). Lit words pull the eye away from a person who is talking, so the house "
                   "way is plain captions here. Keep them lit only if the person asked for it; otherwise put the "
                   "captions on again with lit false." % on_camera)
    times, every = _face_times(state, rec)
    hit = _covered(band, times, every, 0, length)
    if hit:
        out.append("The captions cover a face at %s. If that matters, make the clip again with a layout that keeps the "
                   "face higher." % _spans(hit))
    out += ["%s %s" % (recording.clock(line["from"]), line["text"].strip()) for line in lines[:8]]
    if len(lines) > 8:
        out.append("... and %d more lines." % (len(lines) - 8))
    first = lines[0]
    out.append(_show_frame(job, state, rec, (first["from"] + first["to"]) / 2, "The clip with captions:"))
    return "\n".join(out)


def _on_camera(rec, clip):
    """Where a face sits still in the clip's pieces, as words for Muse, or None. Only faces already found are used."""
    for a, b in clip["pieces"]:
        found = faces.kept(rec, a, b)
        g = faces.steady(found) if found else None
        if g:
            return "a face in the same place in %d of %d pictures" % (g["seen"], g["of"])
    return None


PEOPLE_SHOWN = 8
SAME_FACE = 0.3


def _people(rec, pieces, face):
    """The other people's faces in the clip, from the faces already found in each piece, most seen first: [{"n",
    "seen", "of", "from", "to" (clip seconds), "box" (recording), "points": [[clip second, x, y, w, h]], "crop"}].
    The face the layout shows, and a face that sits still (the person making the clip), are not among them. None
    when the faces of a piece were not found."""
    out = []
    for a, b in pieces:
        found = faces.kept(rec, a, b)
        if not found:
            return None
        for tr in tracks.tracks(found, rec):
            if tr["steady"] or face and _overlap(tr["box"], face) > SAME_FACE:
                continue
            points = []
            for t, x, y, w, h in tr["points"]:
                at = round(project.from_recording(pieces, min(max(t, a), b)), 3)
                if not points or at > points[-1][0]:
                    points.append([at, x, y, w, h])
            out.append({"seen": tr["seen"], "of": len(found["samples"]), "from": points[0][0],
                        "to": points[-1][0], "box": [round(v) for v in tr["box"]], "points": points,
                        "crop": tr["crop"]})
    out.sort(key=lambda p: -p["seen"])
    for n, p in enumerate(out, 1):
        p["n"] = n
    return out


def _overlap(a, b):
    """Intersection over union of two boxes (x, y, w, h)."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    both = max(0, w) * max(0, h)
    return both / (a[2] * a[3] + b[2] * b[3] - both) if both else 0.0


def _people_lines(job, people, rec):
    """What make_clip tells Muse about the other people, and their faces sent as pictures."""
    if people is None:
        return ["Other people's faces could not be looked for, so none are listed."]
    if not people:
        return ["No other person's face is in the clip."]
    lines = ["Other people's faces in the clip: %d (the face the layout shows is not one of them). Decide about each "
             "with decide_people before you render: blur a real person who is not the one making the clip, such as "
             "someone passing behind; keep a face that belongs, such as a game character or a face on a screen or a "
             "box." % len(people)]
    for p in people:
        shown = p["n"] <= PEOPLE_SHOWN and p.get("crop")
        if p["seen"] == 1:
            lines.append("- person %d: seen once, at %s of the clip, at %s on your grid (half a second: a passer-by, a "
                         "game character or a face on a screen); kept unless you blur it%s" % (
                             p["n"], recording.clock(p["from"]), _grid(p["box"], rec),
                             "; its face comes next as a picture" if shown else ""))
        else:
            lines.append("- person %d: seen in %d of %d pictures, %s-%s of the clip, at %s on your grid; decide before "
                         "you render%s" % (p["n"], p["seen"], p["of"], recording.clock(p["from"]),
                                           recording.clock(p["to"]), _grid(p["box"], rec),
                                           "; its face comes next as a picture" if shown else ""))
        if shown:
            job.attach(p["crop"], "Person %d:" % p["n"])
    return lines


def _undecided(state):
    """The people seen in two pictures or more that nobody has decided about yet."""
    return [p["n"] for p in state["people"] if p["blur"] is None and p["seen"] > 1]


def decide_people(job, why, blur=None, keep=None):
    rec = _open(job, "blur")
    state = _clip_state(job)
    people = {p["n"]: p for p in state["people"]}
    if not people:
        raise Refused("The clip has no other people's faces to decide about.")
    blur, keep = sorted(set(blur or [])), sorted(set(keep or []))
    if not blur and not keep:
        raise Refused("Name at least one person to blur or to keep.")
    both = sorted(set(blur) & set(keep))
    if both:
        raise Refused("Person %s is in both lists: each person is either blurred or kept." % _numbers(both))
    unknown = [n for n in blur + keep if n not in people]
    if unknown:
        raise Refused("There is no person %s; the clip's people are %s." % (_numbers(unknown), _numbers(people)))
    state = job.edit({"do": "people", "blur": blur, "keep": keep})
    lines = ["Person %d is blurred where their face shows, %s-%s of the clip, and the blur moves with the face."
             % (n, recording.clock(people[n]["from"]), recording.clock(people[n]["to"])) for n in blur]
    if keep:
        lines.append("Kept as they are: person %s." % _numbers(keep))
    left = _undecided(state)
    lines.append("Still to decide before you render: person %s." % _numbers(left) if left else
                 "Every person seen in two pictures or more is decided.")
    if blur:
        p = people[blur[0]]
        lines.append(_show_frame(job, state, rec, (p["from"] + p["to"]) / 2,
                                 "The clip with person %d blurred:" % p["n"]))
    return "\n".join(lines)


def _numbers(ns):
    ns = [str(n) for n in sorted(ns)]
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]


BLUR_FACE_SHARE = 0.3
PART_NAMES = {"card": "the card", "circle": "the circle with the face", "top": "the top panel",
              "bottom": "the panel below", "whole": "the frame"}


def _share(box, other):
    """How much of box (0 to 1) other covers."""
    w = min(box[0] + box[2], other[0] + other[2]) - max(box[0], other[0])
    h = min(box[1] + box[3], other[1] + other[3]) - max(box[1], other[1])
    return max(0.0, w) * max(0.0, h) / max(1e-6, box[2] * box[3])


def _where(lay, box):
    """Where a box of the recording shows in the clip, part by part, in frame pixels."""
    out = []
    for part in lay["parts"]:
        f = layouts.to_frame({"parts": [part]}, box)
        if f:
            x, y, w, h = f[0]
            out.append("in %s at x %d-%d, y %d-%d" % (PART_NAMES.get(part["name"], part["name"]), x, x + w, y, y + h))
    return out


def remove_title(job, why, words):
    rec = _open(job, "take words off")
    state = _clip_state(job)
    if not any(t["text"] == words for t in state["titles"]):
        raise Refused("The clip has no title \"%s\"; its titles are: %s." % (
            words, "; ".join('"%s"' % t["text"] for t in state["titles"]) or "none"))
    state = job.edit({"do": "untitle", "text": words})
    return '\n'.join(['The title "%s" is off the clip.' % words,
                      _show_frame(job, state, rec, state["length"] / 2, "The clip without it:")])


def blur_box(job, why, box, start=None, end=None, person_said=None):
    asked = _asked(job, person_said)
    rec = _open(job, "blur")
    state = _clip_state(job)
    length, lay = state["length"], state["clip"]["layout"]
    px = _pixels("box", box, rec)
    face = lay.get("face")
    if face and _share(face, px) > BLUR_FACE_SHARE and not asked:
        raise Refused("That box covers the face at %s on your grid, the face the clip shows, and the face is the point "
                      "of the clip. Give a box around the chat or the names only, leaving the face out." %
                      _grid(face, rec))
    shown = layouts.to_frame(lay, px)
    if not shown:
        raise Refused("The box %s is not in the clip's picture (the %s layout leaves that part out), so nothing in it "
                      "can be seen and it needs no blur." % (box, lay["kind"]))
    a = 0.0 if start is None else start
    b = length if end is None else min(end, length)
    if not 0 <= a < length or b - a < SHORTEST_WORDS:
        raise Refused("A blur runs from a start inside the clip (0 to %g s) to a later end; %g-%g s is not that. Leave "
                      "both out to blur the whole clip." % (round(length - SHORTEST_WORDS, 1), a, b))
    state = job.edit({"do": "blur", "box": [round(v) for v in px], "start": round(a, 2), "end": round(b, 2)})
    where = "; ".join(_where(lay, px))
    lines = ["Blurred %s on your grid from %g to %g s, so nothing in it can be read. In the clip it shows %s."
             % (_grid(px, rec), a, b, where)]
    lines.append(_show_frame(job, state, rec, (a + b) / 2, "The clip with the blur:"))
    return "\n".join(lines)


def add_hit(job, why, at, kind, word=None):
    state = _clip_state(job)
    length = state["length"]
    if not 0 <= at < length:
        raise Refused("at %g s is outside the clip, which is %g s long: use the second of the hit, from 0 to %g, in "
                      "seconds of the clip." % (at, length, round(length - 0.1, 1)))
    if kind == "word":
        if not word:
            raise Refused("A word hit needs its word, such as POW, in the clip's language.")
        try:
            text.big_word(word, at, 0.8, _main_part(state["clip"]["layout"]),
                          render.style_of(state, job.chat.clip["style"] if job.chat.clip else "clean", kind="big"))
        except ValueError as e:
            raise Refused(str(e))
    job.edit({"do": "hit", "kind": kind, "at": round(at, 2), "word": word if kind == "word" else None,
              "centre": _main_part(state["clip"]["layout"])})
    return "Added %s at %g s%s." % (HITS[kind], at, ' ("%s")' % word if kind == "word" else "")


STICKER_LENGTH = 2.0


def add_sticker(job, why, sticker, at, length=STICKER_LENGTH, place=None, box=None, person_said=None):
    asked = _asked(job, person_said)
    rec = _open(job, "put a sticker on")
    state = _clip_state(job)
    clip_length = state["length"]
    if not 0 <= at < clip_length:
        raise Refused("at %g s is outside the clip, which is %g s long: use seconds of the clip." % (at, clip_length))
    try:
        png = effects.sticker_png(sticker)
    except ValueError as e:
        raise Refused(str(e))
    shortest = effects.POP[-1][0] + effects.FADE
    length = min(length, round(clip_length - at, 2))
    if length < shortest:
        raise Refused("At %g s the clip ends %g s later; a sticker needs %g s to pop in and fade out. Put it "
                      "earlier." % (at, round(clip_length - at, 2), shortest))
    band, where = _spot("box", state, place, box)
    try:
        effects.sticker(png, at, length, band)
    except ValueError as e:
        raise Refused(str(e))
    w, h = effects._png_size(png)
    fit = min(band[2] / w, band[3] / h)
    drawn = [round(band[0] + (band[2] - w * fit) / 2), round(band[1] + (band[3] - h * fit) / 2), round(w * fit),
             round(h * fit)]
    notes = _phone(drawn, state["clip"]["layout"], state, asked)
    more, hit = _off_faces(state, rec, drawn, at, at + length, asked, "sticker")
    replaced = [k for k in state["stickers"] if project._same_box(k, {"box": band, "at": at, "length": length},
                                                                  "at", "length")]
    state = job.edit({"do": "sticker", "png": png, "at": round(at, 2), "length": length, "box": band,
                      "name": sticker})
    lines = ["The sticker %s pops in at %g s at %s and stays %g s: %d x %d px%s." % (
        sticker, at, where, length, drawn[2], drawn[3], "" if hit else ", clear of every face the program found")]
    lines += notes + more
    if replaced:
        lines.append("It replaces %s." % ", ".join("the sticker %s at %g s" % (k.get("name") or "picture", k["at"])
                                                     for k in replaced))
    lines.append(_show_frame(job, state, rec, min(at + 0.4, at + length - effects.FADE), "The clip with the sticker:"))
    return "\n".join(lines)


GLASS_TINTS = {"light": "the picture behind lightened, for dark words or busy colours",
               "dark": "the picture behind darkened, for white words on a bright picture"}


def _light(colour):
    """True for a colour (#RRGGBB) a dark background reads best under."""
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 127.5


def add_glass(job, why, box, start, end, tint=None, shine=True, person_said=None):
    asked = _asked(job, person_said)
    rec = _open(job, "put a glass box on")
    state = _clip_state(job)
    length = state["length"]
    words = render.style_of(state, job.chat.clip["style"] if job.chat.clip else "clean", kind="title")
    fits = "dark" if _light(words.colour) else "light"
    chosen, tint = tint is None, tint or fits
    if not 0 <= start < length:
        raise Refused("start %g s is outside the clip, which is %g s long: use seconds of the clip." % (start, length))
    end = min(end, length)
    band = _frame_box("box", box)
    try:
        layouts.glass(band, start, end, tint, shine)
    except ValueError as e:
        raise Refused(str(e))
    notes = _phone(band, state["clip"]["layout"], state, asked)
    more, hit = _off_faces(state, rec, band, start, end, asked, "glass box")
    replaced = [g for g in state["glass"] if project._same_box(g, {"box": band, "start": start, "end": end},
                                                               "start", "end")]
    state = job.edit({"do": "glass", "box": band, "start": round(start, 2), "end": round(end, 2), "tint": tint,
                      "shine": shine})
    lines = ["A %s glass box at %s on your grid from %g to %g s: %s%s, %d x %d px%s." % (
        tint, _on_grid(box), start, end, GLASS_TINTS[tint], ", with a sweep of light as it appears" if shine else "", band[2],
        band[3], "" if hit else ", clear of every face the program found"),
        "Words and stickers stand on it: give add_title the same box to put a title on it."]
    look = words.name.split("~")[0]
    if words.box:
        lines.append("The %s style draws its own box behind the words, so they read on any glass." % look)
    elif chosen:
        lines.append("%s glass, because the %s style's words are %s." % (
            tint.capitalize(), look, "light" if fits == "dark" else "dark"))
    elif tint != fits:
        lines.append("The %s style's words are %s, so they stand on their outline here; %s glass would read better "
                     "under them." % (look, "light" if fits == "dark" else "dark", fits))
    lines += notes + more
    if replaced:
        lines.append("It replaces the glass box at %s." % ", ".join("%g-%g s" % (g["start"], g["end"])
                                                                  for g in replaced))
    lines.append(_show_frame(job, state, rec, (start + end) / 2, "The clip with the glass box:"))
    return "\n".join(lines)


MOST_TAGS = 8


def render_clip(job, why):
    rec = _open(job, "render")
    state = _clip_state(job)
    left = _undecided(state)
    if left:
        raise Refused("Decide about person %s first, with decide_people: blur or keep. Nothing undecided is "
                      "rendered." % _numbers(left))
    done = job.chat.finished()
    if done:
        return ("The finished file is already made from the clip as it is now (%s, %g s), so nothing was rendered "
                "again. Hand it over." % (done["name"], done["length"]))
    sign = project.signature(job.chat.project.entries)
    out = os.path.join(render.folder("clips"), sign + ".mp4")
    started = time.time()
    try:
        job.render(rec, state, out, job.chat.clip["style"] if job.chat.clip else "clean")
    except render.Stopped:
        return "Not rendered: the person pressed Stop. Nothing was kept."
    except render.NotDrawn as e:
        raise Refused("The clip could not be rendered: %s" % e)
    if not os.path.isfile(out):
        raise Refused("The renderer said it was done, but there is no file at %s. This is a fault in the studio; tell "
                      "the person." % out)
    made = {"name": sign + ".mp4", "signature": sign, "seconds": round(time.time() - started, 1),
            "size": os.path.getsize(out), "length": state["length"]}
    job.chat.rendered.append(made)
    job.emit("rendered", file=made)
    job.show(rendered=made)
    return ("Rendered the finished clip in %s: %g s, 1080x1920, %.1f MB. The person can play it on the page now. Hand "
            "it over next." % (recording.clock(made["seconds"]), made["length"], made["size"] / 1e6))


REEL_LONGEST = 90
FIRST = 0.5
CHECK_ROUNDS = 2


def _said(job, state):
    """The lines said in the clip's moment, as fixed from the context."""
    moments, n = job.chat.moments or [], state["clip"]["moment"]
    return [p["text"] for p in (moments[n - 1].get("said") or [] if n <= len(moments) else [])]


def _not_said(words, said):
    """The numbers in words that are not among the numbers in said (spoken.values: digits in any script, Thai and
    English number words)."""
    heard = set().union(*[spoken.values(s) for s in said]) if said else set()
    return [m for m in spoken.DIGITS.findall(words) if spoken.digits(m) - heard]


def _numbers_unread(job, said):
    """A sentence when the clip is said in a language whose number words are not read here, else None."""
    language = (job.chat.clip or {}).get("language") or ""
    if spoken.reads(language, said):
        return None
    return ("the clip is said in %s, and the studio reads number words only in Thai and English: look at what was "
            "said and check the numbers yourself" % (language or "a language it does not read"))


def _start_of(path, sound):
    """(black picture at the start, silent sound at the start), each True when it lasts FIRST s from 0."""
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-nostats", "-t", "1.5", "-i", path, "-vf",
           "blackdetect=d=%g:pix_th=0.10" % FIRST]
    if sound:
        cmd += ["-af", "silencedetect=noise=-50dB:d=%g" % FIRST]
    log = subprocess.run(cmd + ["-f", "null", "-"], capture_output=True, text=True, encoding="utf-8",
                         errors="replace").stderr
    black = re.search(r"black_start:0(?:\.0+)?\b", log) is not None
    silent = re.search(r"silence_start: -?0(?:\.0+)?\b", log) is not None
    return black, silent


def _still_seen(state, clip_file):
    """[(person, clip second)] where a blurred person's face can still be found in the finished file."""
    out, lay = [], state["clip"]["layout"]
    for p in state["people"]:
        if not p["blur"]:
            continue
        points = p["points"]
        for point in (points[len(points) // 4], points[len(points) // 2], points[3 * len(points) // 4]):
            t, box = point[0], point[1:]
            shown = layouts.to_frame(lay, box)
            try:
                found = faces.sample(clip_file, t, t + 0.1)
            except (faces.NoFaces, faces.Stopped):
                continue
            for f in (f for s in found["samples"] for f in s["faces"]):
                x, y = f[0] + f[2] / 2, f[1] + f[3] / 2
                if any(b[0] <= x <= b[0] + b[2] and b[1] <= y <= b[1] + b[3] for b in shown):
                    out.append((p["n"], t))
                    break
    return out


def check_clip(job, why):
    rec = _open(job, "check")
    state = _clip_state(job)
    done = job.chat.finished()
    if not done:
        raise Refused("There is no finished file made from the clip as it is now. Render it first: the checks run on "
                      "the file the person will post.")
    path = os.path.join(render.folder("clips"), done["name"])
    try:
        made = recording.probe(path)
    except recording.NotOpened as e:
        raise Refused("The finished file could not be read: %s" % e)
    kind = (job.chat.clip or {}).get("recipe")
    said, checks = _said(job, state), []

    def check(name, ok, found, who="program"):
        checks.append({"name": name, "ok": ok, "found": found, "who": who})

    lines = (state["captions"] or {}).get("lines") or []
    joined = "".join(said).replace(" ", "")
    unsaid = [line["text"] for line in lines if line["text"].replace(" ", "") not in joined]
    check("Captions are words that were said", not unsaid if lines else None,
          ("%d of %d lines are in what was said" % (len(lines) - len(unsaid), len(lines)) +
           ("; not said: %s" % "; ".join(unsaid[:3]) if unsaid else "")) if lines else "no captions")
    if kind == "seller":
        wrong = [n for t in state["titles"] for n in _not_said(t["text"], said)]
        unread = _numbers_unread(job, said) if wrong else None
        check("Numbers on screen were said", None if unread else not wrong,
              unread or ("not said: %s" % ", ".join(wrong) if wrong else "every number on screen was said"),
              "Muse" if unread else "program")
    else:
        check("Numbers on screen were said", None, "numbers can come from what the screen shows")
    if kind == "game":
        whole = [b for b in state["blurs"] if b.get("box") and b["start"] <= 0.1 and b["end"] >= state["length"] - 0.1]
        check("No viewer's name can be read", bool(whole),
              "%d blur over the whole clip" % len(whole) if whole else "no blur covers the whole clip, so the chat box "
                                                                      "shows")
    else:
        check("No viewer's name can be read", None, "no chat box in this kind of clip")
    if _reads_screen(job):
        check("Nothing private can be read on the screen", *_private_check(job, rec, state))
    if kind == "seller":
        product = state["clip"].get("product")
        inside = bool(product) and bool(layouts.to_frame(state["clip"]["layout"], product))
        check("The product is in the frame", inside if product else None,
              "the product box is inside the clip's picture" if inside else
              "no product box was named, so it was not checked" if not product else
              "the product box is outside the clip's picture")
    else:
        check("The product is in the frame", None, "not a seller's clip")
    check("The length fits a Reel", made["seconds"] <= REEL_LONGEST,
          "%.1f s; a Reel is at most %d s" % (made["seconds"], REEL_LONGEST))
    black, silent = _start_of(path, made["sound_tracks"])
    check("It does not start black or silent", not (black or silent),
          "picture and sound from the first %g s" % FIRST if not (black or silent) else
          " and ".join(w for w, bad in (("black", black), ("silent", silent)) if bad) + " for the first %g s" % FIRST)
    if any(p["blur"] for p in state["people"]):
        seen = _still_seen(state, made)
        check("No blurred face can be found", not seen,
              "; ".join("person %d at %s" % (n, recording.clock(t)) for n, t in seen) if seen else
              "the face finder found no face under any blur")
    job.chat.checks[done["name"]] = checks
    job.emit("checked", file=done["name"], checks=checks)
    due = [c for c in checks if c["ok"] is not None]
    failed = [c for c in due if not c["ok"]]
    out = ["Checked %d of %d on the finished file %s:" % (len(due) - len(failed), len(due), done["name"])]
    out += ["- %s: %s (%s)" % ("passed" if c["ok"] else "FAILED" if c["ok"] is False else
                               "for you to check" if c["who"] == "Muse" else "not needed", c["name"], c["found"])
            for c in checks]
    if kind == "game":
        for k, at in enumerate((0.25, 0.5, 0.75), 1):
            name = recording.frame(made, made["seconds"] * at)
            if name:
                job.attach(os.path.join(recording.WORK, "frames", name), "The finished clip, picture %d of 3:" % k)
        out.append("Three pictures of the finished clip come next. Look for any name, phone number or face that can "
                   "be read and should not be: if you find one, blur it, render again and check again.")
    if failed:
        out.append("Fix what failed, render again and check again, then hand it over.")
    else:
        out.append("Hand it over next.")
    return "\n".join(out)


_odd_thai = words.odd_thai


def _spell(job, texts):
    """(texts with the Thai spelling fixed in a short conversation of Muse's own, a sentence naming each fix or "").
    When Muse cannot be asked, the texts stay as written. Stop raises hearing.Stopped."""
    try:
        fixed, changed = words.spelled(lambda m, s: job.ask_aside(m, s, "Checking the Thai spelling"), texts)
    except mind.MindError:
        return list(texts), ""
    return fixed, " ".join('The program fixed the Thai spelling: "%s" is now "%s".' % (" ".join(a.split()),
                                                                                       " ".join(b.split()))
                           for a, b in changed)


def _tag(word):
    word = "".join(word.split()).lstrip("#")
    return "#" + word if word else ""


def hand_over(job, why, title, caption, hashtags, english=""):
    _clip_state(job)
    done = job.chat.finished()
    if not done:
        raise Refused("There is no finished file made from the clip as it is now, so there is nothing to hand over. "
                      "Render the clip first, then hand it over.")
    checks = job.chat.checks.get(done["name"])
    if checks is None:
        raise Refused("Check the finished file first, with check_clip: the person gets a clip that was checked.")
    failed = [c["name"] for c in checks if c["ok"] is False]
    rounds = sum(1 for e in job.chat.events if e["kind"] == "checked" and e.get("job") == job.n)
    if failed and rounds < CHECK_ROUNDS:
        raise Refused("These checks failed on the finished file: %s. Fix them, render again and check again; after "
                      "%d rounds you may hand it over and say what still fails." % ("; ".join(failed), CHECK_ROUNDS))
    wrong = _misspelled(" ".join([title, caption, english or ""] + list(hashtags)),
                        _screen_words(job, _open(job, "hand over"), _clip_state(job)))
    if wrong and not any(e["kind"] == "misspelled" and e.get("job") == job.n for e in job.chat.events):
        job.emit("misspelled", words=[w for w, _ in wrong])
        raise Refused(" ".join(_names_said(wrong, "write them as the screen does")) + " If the speaker said them as "
                      "you wrote them, hand over again unchanged.")
    if (job.chat.clip or {}).get("recipe") == "seller":
        said = _said(job, _clip_state(job))
        wrong = _not_said(" ".join([title, caption, english or ""]), said)
        if wrong and not _numbers_unread(job, said):
            raise Refused("The post text has numbers that were not said: %s. Use only numbers the seller said."
                          % ", ".join(wrong))
    caption, more = _without_tags(caption)
    english, also = _without_tags(english)
    try:
        (title, caption), spelling = _spell(job, [title, caption])
    except hearing.Stopped:
        return "Not handed over: the person pressed Stop."
    tags = list(dict.fromkeys(t for t in (_tag(w) for w in list(hashtags) + more + also) if t))
    if not tags:
        raise Refused("Give at least one hashtag, such as #gaming.")
    ready = {"file": done["name"], "length": done["length"], "title": title, "caption": caption, "hashtags": tags,
             "english": english, "checks": checks}
    job.emit("ready", **ready)
    job.show(ready=ready)
    return ("Handed over: the person sees the finished clip with its title, the post text and %d hashtags, a Copy all "
            "button and a download. Tell them in one or two short lines; do not repeat the text, it is on the page."
            % len(tags)) + (" " + spelling if spelling else "")


def _without_tags(words):
    """(the text without its lines of hashtags, those hashtags): the page shows the hashtags once, under the text."""
    kept, tags = [], []
    for line in (words or "").splitlines():
        if line.split() and all(w.startswith("#") for w in line.split()):
            tags += line.split()
        else:
            kept.append(line)
    return "\n".join(kept).strip(), tags


def undo(job, why):
    entry = job.chat.undo(job=job.n)
    if entry is None:
        raise Refused("There is nothing to take back: the clip has no changes yet.")
    job.show(clip=job.chat.clip_view())
    return "Took back %s." % describe(entry)


def describe(entry):
    if entry["do"] == "clip":
        return "the clip from moment %d in the %s layout" % (entry["moment"], entry["layout"]["kind"])
    if entry["do"] == "style":
        return "the %s style" % entry["name"] + (" with the font %s" % entry["font"] if entry.get("font") else "") + \
            (" at size %d" % entry["size"] if entry.get("size") else "") + \
            "".join(", %s in %s" % (WORD_PARTS[k], f) for k, f in (entry.get("fonts") or {}).items()) + \
            "".join(", %s at size %d" % (WORD_PARTS[k], n) for k, n in (entry.get("sizes") or {}).items()) + \
            (", %s" % _paints(entry["colours"]) if entry.get("colours") else "") + \
            "".join(", %s for %s" % (_paints(c), WORD_PARTS[k]) for k, c in (entry.get("part_colours") or {}).items())
    if entry["do"] == "captions":
        return ("the lit captions" if entry.get("lit") else "the captions") if entry["lines"] else \
            "turning the captions off"
    if entry["do"] == "blur":
        return "the blur at %g-%g s" % (entry["start"], entry["end"])
    if entry["do"] == "hit":
        return "the %s at %g s" % (entry["kind"], entry["at"])
    if entry["do"] == "sticker":
        return "the sticker %sat %g s" % (entry["name"] + " " if entry.get("name") else "", entry["at"])
    if entry["do"] == "glass":
        return "the %s glass box at %g-%g s" % (entry["tint"], entry["start"], entry["end"])
    if entry["do"] == "layout":
        return "changing the shots" if entry["part"] == "shots" else "moving the %s" % entry["part"]
    if entry["do"] == "untitle":
        return 'taking the title "%s" off' % entry["text"]
    if entry["do"] == "look":
        return "the %s look" % entry["name"] if entry["name"] else "taking the look off"
    if entry["do"] == "push":
        return "the slow push at %g-%g s" % (entry["start"], entry["end"])
    if entry["do"] == "sound":
        return "the %s at %g s" % (entry["kind"], entry["at"])
    if entry["do"] == "music":
        return "the music %s" % os.path.basename(entry["path"]) if entry["path"] else "taking the music off"
    if entry["do"] == "people":
        return "the decision on people (%s)" % "; ".join(
            "%s person %s" % (k, _numbers(entry[k])) for k in ("blur", "keep") if entry[k])
    return 'the title "%s" at %g-%g s' % (entry["text"], entry["start"], entry["end"])


STYLES = {n: text.CARDS[n][0][0].lower() + text.CARDS[n][0][1:].rstrip(".") for n in text.HOUSE_NAMES}
LOOK_LABELS = {"3d": "3D"}
SHORTEST, LONGEST = 10, 90


def load_recipe(job, why, type, length_s=None, style=None, language="", brief=""):
    recipe = RECIPES.get(type.lower().strip())
    if recipe is None:
        raise Refused("There is no recipe for %s yet. The recipes are: %s. Do not use a recipe made for another "
                      "kind of video: tell the person a clip of this kind is not ready yet, and what the studio can "
                      "make now." % (type, _recipe_list()))
    clip = {"recipe": recipe["name"], "title": recipe["title"], "length_s": length_s or recipe["length"],
            "style": style or recipe["style"], "language": language, "brief": brief}
    had = job.chat.clip
    job.set_clip(clip, chips(clip))
    lines = ["Recipe loaded: %s (%s)." % (recipe["title"], recipe["name"]),
             "Your guesses, shown to the person with buttons to change them: about %d s, style %s%s."
             % (clip["length_s"], clip["style"], ", language " + language if language else ""),
             "The person's own words win over this recipe and over your guesses. Do not ask the person to confirm "
             "them: go on with the work. The buttons let them change any guess at any time."]
    if had and had["recipe"] != recipe["name"]:
        lines.append("This recipe replaces the %s recipe read earlier in this chat; follow only this one." %
                     had["recipe"])
    return "\n".join(lines) + "\n\n" + recipe["body"]


def chips(clip):
    """The buttons under Muse's guesses. Each one sends a short message, as if the person had typed it."""
    out = [{"label": r["chip"], "say": "(tapped) %s." % r["chip"]} for r in RECIPES.values()
           if r["name"] != clip["recipe"]]
    length = clip["length_s"]
    if length > SHORTEST:
        shorter = max(SHORTEST, round(length * 2 / 3))
        out.append({"label": "Shorter", "say": "(tapped) Make it shorter, about %d s." % shorter})
    if length < LONGEST:
        longer = min(LONGEST, round(length * 3 / 2))
        out.append({"label": "Longer", "say": "(tapped) Make it longer, about %d s." % longer})
    out += [{"label": LOOK_LABELS.get(s, s.capitalize()), "say": "(tapped) Use the %s style." % s, "look": s,
             "now": s == clip["style"]} for s in STYLES]
    return out


def _recipe_list():
    return ", ".join("%s (%s)" % (r["name"], r["title"]) for r in RECIPES.values())


TOOLS = {t.name: t for t in [
    Tool("set_plan", "Made a plan",
         "Write your plan for this request as a few short steps, in the language the person writes in. The page "
         "shows it and ticks each step as your calls name it in plan_step. Call it again only to change the plan, "
         "and say why.",
         {"steps": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 12,
                    "description": "The steps in order, each a few words."}},
         ["steps"], set_plan),
    Tool("find_recordings", "Looked for your videos",
         "List the recordings on this computer, newest first, with their size, time and full path. Words narrow the "
         "list to files whose name or folder holds every word, in upper or lower case. It only reads the folder "
         "lists, never the files.",
         {"words": {"type": "string", "maxLength": 200,
                    "description": "Words that must all be in the file name or its folders, such as a game or a date. "
                                   "Leave empty for the newest recordings."},
          "newest": {"type": "integer", "minimum": 1, "maximum": 20,
                     "description": "How many to list, newest first (1 to 20, default 5)."}},
         [], find_recordings),
    Tool("open_recording", "Opened the video",
         "Open one recording to make the clip from: its full path, as the list of recordings gives it. It reads "
         "the length, picture size and sound tracks, and the page shows the recording. One recording is open at a "
         "time; opening another replaces it.",
         {"path": {"type": "string", "maxLength": 1000, "description": "The recording's full path."}},
         ["path"], open_recording, changes=True),
    Tool("load_recipe", "Picked the kind of clip",
         "Read the recipe for this kind of video before you plan the clip, and give your guesses for it. The page "
         "shows your guesses with buttons, so the person can change any of them with one tap. Call it again when "
         "the person says it is another kind, then plan again. The kinds with a recipe now: " + _recipe_list() +
         ". If the video is another kind, still name that kind.",
         {"type": {"type": "string", "maxLength": 40,
                   "description": "The kind of video as it really is: a recipe's name when one fits, else your own "
                                  "word for it, such as travel. Never pick a recipe that does not fit."},
          "length_s": {"type": "integer", "minimum": SHORTEST, "maximum": LONGEST,
                       "description": "Your guess for the clip's length in seconds. Leave it out to use the recipe's."},
          "style": {"type": "string", "enum": list(STYLES),
                    "description": "Your guess for the house style: " +
                                   "; ".join("%s (%s)" % kv for kv in STYLES.items()) + "."},
          "language": {"type": "string", "maxLength": 40,
                       "description": "The language spoken in the recording, as far as you can tell, such as Thai."},
          "brief": {"type": "string", "maxLength": 300,
                    "description": "What the person asked for about the clip itself, in their own words (its mood, "
                                   "length or focus), if they said anything. It wins over every guess."}},
         ["type"], load_recipe, changes=True),
    Tool("map_recording", "Mapped the video",
         "Measure the open recording, then list the stretches that stand out, best first: how loud each second is "
         "against the minutes around it, and how much the picture changes, ranked the way the recipe says. Use it "
         "before you look at pieces of a long recording. The first time it reads the whole file (about half a "
         "minute for every hour of a 1080p stream); after that it is instant. Nothing in the list has been looked at.",
         {"stretch_s": {"type": "integer", "minimum": SHORTEST, "maximum": LONGEST,
                        "description": "How long each stretch is, in seconds. Leave it out to use the clip's length."},
          "top": {"type": "integer", "minimum": 1, "maximum": 12,
                  "description": "How many stretches to list (1 to 12, default 6)."}},
         [], map_recording),
    Tool("look_at_piece", "Looked at a piece",
         "Look at one stretch of the open recording: pictures spread evenly from start to end, each its own picture "
         "with its time, sent to you right after the result, and what is said in it, written down with times. Look "
         "at a stretch before you propose it. To see what led up to a peak, look earlier; to see more around it, "
         "look wider (at most 10 minutes).",
         {"start": {"type": "string", "maxLength": 20,
                    "description": "Where the stretch starts, written as the map writes times, like 1:51:23."},
          "end": {"type": "string", "maxLength": 20, "description": "Where it ends, written the same way."},
          "frames": {"type": "integer", "minimum": 2, "maximum": 6,
                     "description": "How many pictures (2 to 6, default %d)." % LOOK_FRAMES}},
         ["start", "end"], look_at_piece, changes=True),
    Tool("propose_moments", "Proposed the moments",
         "Propose the best moments for the clip, best first, each from a look you made. The person sees each as a "
         "card with your reason, a picture from your look and a rough video of the stretch. Then go on with the "
         "best one yourself; do not ask the person to choose.",
         {"moments": {"type": "array", "minItems": 1, "maxItems": MOST_MOMENTS,
                      "description": "1 to %d moments, best first." % MOST_MOMENTS,
                      "items": {"type": "object", "properties": {
                          "look": {"type": "integer", "minimum": 1,
                                   "description": "The number of the look you saw this moment in."},
                          "reason": {"type": "string", "maxLength": 300,
                                     "description": "One line for the person, in their language: what happens, and "
                                                    "what leads up to it, as you saw it."},
                          "start": {"type": "string", "maxLength": 20,
                                    "description": "Where the moment starts, inside that look, like 1:51:30. Leave "
                                                   "it out to start where the look starts."},
                          "end": {"type": "string", "maxLength": 20,
                                  "description": "Where it ends, inside that look. Leave it out to end where the "
                                                 "look ends."},
                          "picture": {"type": "integer", "minimum": 1, "maximum": 6,
                                      "description": "Which picture of that look shows the moment best. Leave it out "
                                                     "for the one nearest the loudest second."}},
                                "required": ["look", "reason"]}}},
         ["moments"], propose_moments, changes=True),
    Tool("make_clip", "Made the clip",
         "Make the vertical clip from one of the moments you proposed, in one of the layouts that the program "
         "draws: " + "; ".join("%s: %s" % kv for kv in dict(layouts.SAID, talk=talk.SAID).items()) + ". The program "
         "finds the face itself "
         "in the moment's pictures and fixes what it safely can, and says what it did. The clip's frame comes back "
         "to you as a picture. Call it again to remake the clip from another moment, layout or crop; the words on "
         "screen stay.",
         {"moment": {"type": "integer", "minimum": 1, "maximum": MOST_MOMENTS,
                     "description": "The number of the moment you proposed."},
          "layout": {"type": "string", "enum": list(layouts.KINDS) + ["talk"], "description": "Which layout."},
          "crop": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0,
                                                                              "maximum": 1000},
                   "description": "[x1, y1, x2, y2] on your 0-1000 grid of the recording's picture: the part to show "
                                  "(the game for card and stack, the product or the action for subject). Leave it out "
                                  "for the whole picture."},
          "face": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0,
                                                                              "maximum": 1000},
                   "description": "[x1, y1, x2, y2] on your 0-1000 grid, roughly around the face to use. Leave it out: "
                                  "the program finds the face. Give it only when the program found none, or the wrong "
                                  "one."},
          "product": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0,
                                                                                 "maximum": 1000},
                      "description": "[x1, y1, x2, y2] on your 0-1000 grid around the product being sold, when there "
                                     "is one, so no words cover it. Leave it out otherwise."}},
         ["moment", "layout"], make_clip, changes=True),
    Tool("set_style", "Chose the look of the words",
         "Choose how the words on screen look: one of the looks (" + "; ".join("%s: %s" % kv for kv in STYLES.items())
         + "), with the font, the size and the colours the person asked for. The person sees every look as a "
         "picture on the page and can tap one. A colour for the letters that would be hard to read gets an "
         "outline that makes it readable; an outline or key colour that is given is kept, and the result says it "
         "may be hard to read. Colours given stay until changed, also in another look.",
         {"style": {"type": "string", "enum": list(STYLES), "description": "The house style."},
          "font": {"type": "string", "maxLength": 80,
                   "description": "A font family or face from list_fonts, free to use in a video. Leave it out for "
                                  "the style's own font."},
          "size": {"type": "integer", "minimum": text.SIZES[0], "maximum": text.SIZES[-1],
                   "description": "How big the words are, %d (smallest) to %d (biggest); %d is the size tuned on "
                                  "phones. Leave it out to keep the size." % (text.SIZES[0], text.SIZES[-1],
                                                                              text.TUNED_SIZE)},
          "colours": {"type": "object", "properties": {
              "text": {"type": "string", "maxLength": 7, "description": "the letters"},
              "key": {"type": "string", "maxLength": 7, "description": "the key word of a title"},
              "edge": {"type": "string", "maxLength": 7,
                       "description": "the outline, or the box behind the words in the sticker look"},
              "glow": {"type": "string", "maxLength": 7,
                       "description": "a soft glow around the letters: the glow look has one, and any look given "
                                      "one gets it"},
              "shade": {"type": "string", "maxLength": 7, "description": "the hard shadow of pop and 3d"}},
                      "description": "The colours the person asked for, each #RRGGBB, such as {\"key\": "
                                     "\"#E11D48\"}: only the ones to change. Leave it out to keep the look's own."},
          "part": {"type": "string", "enum": list(WORD_PARTS),
                   "description": "Give the font, the size or the colours to one part only: " +
                                  "; ".join("%s: %s" % kv for kv in WORD_PARTS.items()) + ". Leave it out for every "
                                  "part."},
          "person_said": {"type": "string", "maxLength": 200,
                          "description": "Only for a font not known to be free that the person named: their own "
                                         "words asking for it, quoted. Leave it out otherwise."}},
         ["style"], set_style, changes=True),
    Tool("list_fonts", "Listed the fonts",
         "List the fonts that can write every letter of the clip's words, the studio's and this computer's, with "
         "whether each is free to use in a video someone posts. Use it when the person wants another font.",
         {}, [], list_fonts),
    Tool("add_title", "Put a title on",
         "Put words on screen from start to end, in one of the layout's places for words, or in a box the person "
         "asked for. The program measures the words as they will be drawn and refuses any time or place you choose "
         "where they would cover a face or sit where a phone covers the clip, with the times and places that are "
         "clear. A title in the same place at the same time replaces the earlier one.",
         {"words": {"type": "string", "maxLength": 80,
                    "description": "The words, in the clip's language. Say what happens without giving the ending away. "
                                   "A number or a claim must be one that was said."},
          "start": {"type": "number", "minimum": 0, "description": "When they appear, in seconds of the clip (0 is "
                                                                   "its first frame)."},
          "end": {"type": "number", "minimum": 0, "description": "When they go, in seconds of the clip."},
          "place": {"type": "string", "maxLength": 20,
                    "description": "One of the layout's places for words, as make_clip listed them, such as top. "
                                   "Leave it out when you give a box."},
          "box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0,
                                                                              "maximum": 1000},
                  "description": "[x1, y1, x2, y2] on a 0-1000 grid of the clip's vertical frame (not the "
                                 "recording), for a spot the person asked for that the places do not have."},
          "person_said": {"type": "string", "maxLength": 200,
                          "description": "Only when the person asked for this exact spot or time: their own words "
                                         "from the chat that ask for it, quoted. The program checks they said "
                                         "them. Then the title goes there even over a face or where a phone covers "
                                         "it, and the result says so; when you choose, those are refused."},
          "key": {"type": "string", "maxLength": 40,
                  "description": "One word of the title to draw in the style's key colour. Leave it out for none."}},
         ["words", "start", "end"], add_title, changes=True),
    Tool("remove_title", "Took a title off",
         "Take a title off the clip, by its words.",
         {"words": {"type": "string", "maxLength": 80, "description": "The title's words, as on the clip."}},
         ["words"], remove_title, changes=True),
    Tool("add_captions", "Put the captions on",
         "Put captions of what was said in the moment on the clip: the words as heard and fixed from the context, "
         "numbers exactly as said, a few words a line. The program places them in the layout's caption band. They "
         "are plain words unless you set lit. Set on to false to take them off.",
         {"on": {"type": "boolean", "description": "true to put captions on (the default), false to take them off."},
          "lit": {"type": "boolean",
                  "description": "true lights up each word as it is said, like karaoke. Use it only when nobody on "
                                 "screen is talking: a voice-over or the words of a song over the picture. When a "
                                 "person on camera explains something, the lit word pulls the eye away from them, "
                                 "so leave it out, unless the person asked for lit words."}},
         [], add_captions, changes=True),
    Tool("blur_box", "Blurred a box",
         "Blur one box of the recording's picture so nothing in it can be read, such as a stream's chat with "
         "viewers' names, or a phone number. The box is fixed on the picture for the whole clip unless you give a "
         "start and an end.",
         {"box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0,
                                                                             "maximum": 1000},
                  "description": "[x1, y1, x2, y2] on your 0-1000 grid of the recording's picture, around everything "
                                 "to hide, with a little room."},
          "start": {"type": "number", "minimum": 0, "description": "From when, in seconds of the clip. Leave it out "
                                                                   "for the start."},
          "end": {"type": "number", "minimum": 0, "description": "Until when, in seconds of the clip. Leave it out for "
                                                                 "the end."},
          "person_said": {"type": "string", "maxLength": 200,
                          "description": "Only when the person asked to blur this, even the face the clip shows: "
                                         "their own words from the chat, quoted. The program checks them."}},
         ["box"], blur_box, changes=True),
    Tool("decide_people", "Decided about people",
         "Decide who of the other people make_clip listed is blurred and who is kept. A blurred face is followed as "
         "it moves. Blur a real person who is not the one making the clip, such as someone passing behind; keep a "
         "face that belongs, such as a game character, or a face on a screen or a box. Every person seen in two "
         "pictures or more must be decided before the clip is rendered; one call can decide them all.",
         {"blur": {"type": "array", "items": {"type": "integer", "minimum": 1},
                   "description": "The numbers of the people to blur, such as [2]."},
          "keep": {"type": "array", "items": {"type": "integer", "minimum": 1},
                   "description": "The numbers of the people to keep as they are."}},
         [], decide_people, changes=True),
    Tool("move_part", "Moved a part of the frame",
         "Only when the person asks: put one part of the frame where they want it, anywhere on the clip, such as "
         "their face circle bottom right, a bigger circle, or the card higher. circle also adds the face circle to "
         "a full-screen layout. The result says where it went and what a phone covers there; tell the person.",
         {"part": {"type": "string", "enum": list(PARTS),
                   "description": "card: the game card; circle: the face circle; top and bottom: the two halves of "
                                  "the stack; whole: the full-screen picture."},
          "box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0,
                                                                              "maximum": 1000},
                  "description": "[x1, y1, x2, y2] on a 0-1000 grid of the clip's vertical frame, where the part "
                                 "goes. A circle stays round, in the middle of the box."}},
         ["part", "box"], move_part, changes=True),
    Tool("change_shot", "Changed a shot",
         "In a talk clip, show one person, or everyone, from start to end: when the program cut to the wrong person, "
         "when a reaction should be seen, or when the person asks. The shots around it are cut to make room.",
         {"start": {"type": "number", "minimum": 0, "description": "From when, in seconds of the clip."},
          "end": {"type": "number", "minimum": 0, "description": "Until when, in seconds of the clip."},
          "show": {"type": "integer", "minimum": 0, "maximum": talk.MOST_PEOPLE,
                   "description": "The person to show, numbered from the left as make_clip listed them, or 0 for "
                                  "everyone together."}},
         ["start", "end", "show"], change_shot, changes=True),
    Tool("set_look", "Chose a colour look",
         "Grade the footage with one colour look: " + "; ".join("%s: %s" % kv for kv in LOOKS.items()) + ". It "
         "colours the footage only; titles, captions and the frame's design keep their colours.",
         {"look": {"type": "string", "enum": list(LOOKS), "description": "Which look."}},
         ["look"], set_look, changes=True),
    Tool("add_sound", "Added a sound",
         "Add a short sound at a moment of the clip: " + "; ".join("%s: %s" % kv for kv in SOUNDS.items()) + ". "
         "Use them sparingly, on the hits that matter, and never over every word. A sound of the same kind within "
         "0.5 s of an earlier one replaces it.",
         {"kind": {"type": "string", "enum": list(SOUNDS), "description": "Which sound."},
          "at": {"type": "number", "minimum": 0, "description": "When it starts, in seconds of the clip: on the hit."},
          "level": {"type": "number", "minimum": effects.SOUND_LEVELS[0], "maximum": effects.SOUND_LEVELS[1],
                    "description": "dB louder (or quieter, below 0) than it is made. Leave it out for 0."}},
         ["kind", "at"], add_sound, changes=True),
    Tool("set_music", "Put music under the clip",
         "Put the person's own music under the clip, or take it off. The studio has no music of its own: use it "
         "only when the person asks for music or gives a file. The music goes down while someone speaks, loops when "
         "shorter than the clip, and fades in and out.",
         {"music": {"type": "string", "maxLength": 300,
                    "description": "The path of the person's music file, or words from its name. Leave it out to "
                                   "take the music off."},
          "start": {"type": "number", "minimum": 0, "description": "The second of the music the clip starts at."},
          "level": {"type": "number", "minimum": render.MUSIC_LEVELS[0], "maximum": render.MUSIC_LEVELS[1],
                    "description": "dB under the voice, such as -6. Leave it out for -6."},
          "duck": {"type": "number", "minimum": -40, "maximum": -1,
                   "description": "dB lower while someone speaks, such as -12. Leave it out for -12."}},
         [], set_music, changes=True),
    Tool("slow_push", "Pushed in slowly",
         "A slow push in on the clip's main picture, like a camera moving closer, for the build-up into the peak. "
         "It eases in and out, then cuts back at the end. One per moment; a zoom of 1.05 to 1.12 reads as a "
         "camera move, more reads as a zoom. A push over the time of an earlier one replaces it.",
         {"start": {"type": "number", "minimum": 0, "description": "When it starts, in seconds of the clip."},
          "end": {"type": "number", "minimum": 0, "description": "When it ends and cuts back, 0.5 to 30 s later: "
                                                                 "on the peak."},
          "zoom": {"type": "number", "minimum": 1.01, "maximum": 1.5,
                   "description": "How much bigger the picture is at the end, such as 1.08."}},
         ["start", "end"], slow_push, changes=True),
    Tool("add_hit", "Added a hit",
         "Put one effect on the second something lands, never before it: " +
         "; ".join("%s: %s" % kv for kv in HITS.items()) + ". Use it on the peak, at most a few in a clip.",
         {"at": {"type": "number", "minimum": 0, "description": "The second of the hit, in seconds of the clip."},
          "kind": {"type": "string", "enum": list(HITS), "description": "Which effect."},
          "word": {"type": "string", "maxLength": 20,
                   "description": "For a word hit: the word, short, in the clip's language."}},
         ["at", "kind"], add_hit, changes=True),
    Tool("add_sticker", "Put a sticker on",
         "Put one of the studio's stickers on the clip: a 3D emoji picture that pops in, holds and fades out, such "
         "as fire on the peak, money_bag on a price, thumbs_up on a good result. One or two in a clip, on the "
         "moments that matter. It keeps its square shape in the middle of the box you give: a box 200 wide and 110 "
         "tall on the grid is about 216 px square. A sticker on the same spot at an overlapping time replaces the "
         "earlier one.",
         {"sticker": {"type": "string", "enum": [s["name"] for s in effects.stickers()],
                      "description": "Which sticker, by its name."},
          "at": {"type": "number", "minimum": 0, "description": "When it pops in, in seconds of the clip."},
          "length": {"type": "number", "minimum": 0.4, "maximum": 10,
                     "description": "How long it stays, in seconds. Leave it out for %g." % STICKER_LENGTH},
          "place": {"type": "string", "maxLength": 20,
                    "description": "One of the layout's places for words, as make_clip listed them. Leave it out "
                                   "when you give a box."},
          "box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0,
                                                                              "maximum": 1000},
                  "description": "[x1, y1, x2, y2] on a 0-1000 grid of the clip's vertical frame, where it goes."},
          "person_said": {"type": "string", "maxLength": 200,
                          "description": "Only when the person asked for this exact spot: their own words from the "
                                         "chat, quoted. Then it goes there even over a face or where a phone covers "
                                         "it, and the result says so."}},
         ["sticker", "at"], add_sticker, changes=True),
    Tool("add_glass", "Put a glass box on",
         "Put a frosted glass box on the clip for a title, a price or a list to stand on, so the words read over a "
         "busy picture: inside its rounded shape the picture is blurred and " +
         "; or ".join("%s: %s" % kv for kv in GLASS_TINTS.items()) + ". It has a soft edge and shadow, fades in and "
         "out, and lies under the words and stickers. Then give add_title the same box.",
         {"box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0,
                                                                             "maximum": 1000},
                  "description": "[x1, y1, x2, y2] on a 0-1000 grid of the clip's vertical frame, a little bigger "
                                 "than the words that stand on it."},
          "start": {"type": "number", "minimum": 0, "description": "When it fades in, in seconds of the clip."},
          "end": {"type": "number", "minimum": 0, "description": "When it is gone, in seconds of the clip."},
          "tint": {"type": "string", "enum": list(GLASS_TINTS),
                   "description": "Leave it out and the program picks the one the style's words read best on."},
          "shine": {"type": "boolean",
                    "description": "true (the default) sweeps a band of light across it once as it appears; false "
                                   "for a quiet box."},
          "person_said": {"type": "string", "maxLength": 200,
                          "description": "Only when the person asked for this exact spot: their own words from the "
                                         "chat, quoted."}},
         ["box", "start", "end"], add_glass, changes=True),
    Tool("render_clip", "Rendered the clip",
         "Render the finished clip from everything on it now: one call that returns when the file exists, with how "
         "long it took. Render after the last change, before you hand it over. Calling it again with no change in "
         "between does nothing.",
         {}, [], render_clip, changes=True),
    Tool("check_clip", "Checked the clip",
         "Check the finished file the person will post, all checks in one call: the captions are words that were "
         "said, numbers on screen were said (seller), no viewer's name can be read (game), the product is in the "
         "frame (seller), the length fits a Reel, it does not start black or silent, and no blurred face can be "
         "found. Each line says what it found. Run it after the render, before you hand the clip over.",
         {}, [], check_clip),
    Tool("hand_over", "Handed over the clip",
         "Give the person the finished clip, ready to post: the page shows the file with your title, the post text "
         "and the hashtags, a Copy all button and a download. It needs a finished file made from the clip as it is "
         "now.",
         {"title": {"type": "string", "maxLength": 100,
                    "description": "The post's title, in the clip's language."},
          "caption": {"type": "string", "maxLength": 600,
                      "description": "The post text, a few short lines in the clip's language. Any number or claim in "
                                     "it must be one that was said."},
          "hashtags": {"type": "array", "minItems": 1, "maxItems": MOST_TAGS,
                       "items": {"type": "string"}, "description": "1 to %d hashtags, such as #gaming." % MOST_TAGS},
          "english": {"type": "string", "maxLength": 600,
                      "description": "The post text in English, when the clip's language is not English."}},
         ["title", "caption", "hashtags"], hand_over, changes=True),
    Tool("undo", "Took back the last change",
         "Take back the last change to the clip: a layout, a moved part, a style, a title, the captions, a blur, a hit, a "
         "look, a push or a decision about people. The "
         "person can also do this on the page.",
         {}, [], undo, changes=True),
]}
SCHEMAS = [t.schema() for t in TOOLS.values()]
