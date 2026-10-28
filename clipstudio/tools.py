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

import numpy as np

from . import (effects, faces, focus, hearing, layouts, maps, mind, montage, pauses, previews, project, recording,
               render, voices,
               screen, spoken, takes, talk, text, thai, tracks, words)
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
    if kind == "array" and (spec.get("items") or {}).get("type") == "array":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        if isinstance(value, list) and value and not isinstance(value[0], (list, str)):
            value = [value]
        if not isinstance(value, list) or len(value) > spec.get("maxItems", 100):
            raise Refused("%s must be a list of at most %d lists of numbers, such as [[600, 0, 1000, 300]]."
                          % (name, spec.get("maxItems", 100)))
        return [_value("%s %d" % (name, i), v, spec["items"]) for i, v in enumerate(value, 1)]
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
        about = 'has "%s" in its name, folders or recorded day' % words if words.split() else "is there"
        return "No recording %s in %s.%s" % (about, where, _names_here(folders))
    shown = found[:newest]
    about = ' for "%s"' % words if words.split() else ""
    lines = ["%d recordings%s in %s; the %d newest:" % (len(found), about, where, len(shown))]
    for i, (mtime, size, path) in enumerate(shown, 1):
        lines.append("%d. %s · %s · recorded %s · %s" % (i, os.path.basename(path), recording.size_text(size),
                                                       recording.when(mtime), path))
    held = {}
    for _, size, path in found:
        held.setdefault(os.path.dirname(path), []).append(size)
    for folder in sorted({os.path.dirname(p) for _, _, p in shown}):
        sizes = held[folder]
        if 2 <= len(sizes) <= MOST_CLIPS and max(sizes) < CLIP_SIZE:
            lines.append("%s holds %d short recordings: to make one clip from several of them, open them together "
                         "with open_clips." % (folder, len(sizes)))
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


MONTAGE = "montage"
MOST_CLIPS = 30
CLIP_SIZE = 1_000_000_000
SHORTEST_PIECE, LONGEST_PIECE = 1.0, 8.0
SHORTEST_FIT = 2.0
OVER_LENGTH = 1.25
LINE_EVERY = 15.0


def open_clips(job, why, folder):
    try:
        real = recording.check_folder(folder)
    except recording.NotOpened as e:
        raise Refused(str(e))
    paths = recording.clips_in(real)
    if len(paths) < 2:
        raise Refused("%s holds %d recording%s; open_clips joins two or more. Open one with open_recording."
                      % (real, len(paths), "" if len(paths) == 1 else "s"))
    if len(paths) > MOST_CLIPS:
        raise Refused("%s holds %d recordings, and a montage is made from at most %d. Ask the person which ones to "
                      "use, and to put them in a folder of their own." % (real, len(paths), MOST_CLIPS))
    recs, unread = [], []
    for path in paths:
        try:
            recs.append(recording.probe(path))
        except recording.NotOpened as e:
            unread.append(str(e))
    if len(recs) < 2:
        raise Refused("Only %d of the recordings in %s could be read: %s" % (len(recs), real, " ".join(unread)))
    clips = montage.gather(recs)
    try:
        for c in clips:
            job.working("Looking at clip %d of %d" % (c["n"], len(clips)))
            c["stills"] = montage.stills(c["rec"])
            c["said"], c["unheard"] = _clip_said(job, c["rec"])
    except hearing.Stopped:
        return "Not opened: the person pressed Stop."
    except montage.MontageError as e:
        raise Refused(str(e))
    finally:
        job.done_working()
    kept = []
    height = montage.STRIP_HEIGHT if len(clips) <= montage.MANY else montage.STRIP_LOWEST
    for c in clips:
        rec = c["rec"]
        picture = montage.strip(c, height)
        if picture:
            job.attach(picture, "Clip %d (%s, %g s): pictures at %s s:" % (
                c["n"], rec["name"], rec["seconds"], ", ".join("%g" % t for t in montage.strip_times(rec["seconds"]))))
        kept.append({"n": c["n"], "rec": rec, "when": montage.when(c), "stills": c["stills"], "said": c["said"],
                     "unheard": c["unheard"], "picture": picture})
    job.chat.clips = {"folder": real, "clips": kept}
    job.show(clips=job.chat.clips)
    lines = ["Opened %d clips from %s, numbered in the order they were shot, by the camera's clock where it wrote one. "
             "A picture of each comes next: frames from it side by side, each marked with its second, and its number "
             "in the corner." % (len(kept), real)]
    lines += ["Not read: %s" % u for u in unread]
    for c in kept:
        lines += _clip_lines(c)
    lines.append("Next: load the %s recipe, then arrange the clips: which go in, in what order, which seconds of "
                 "each, the words on screen over each, and why. Every clip is either used or left out with a reason."
                 % MONTAGE)
    return "\n".join(lines)


def _clip_said(job, rec):
    """(what is said in a whole clip as heard, or None when it has no sound; why it could not be heard, or None)."""
    if not rec["sound_tracks"]:
        return None, None
    said = hearing.heard(rec, 0.0, rec["seconds"])
    if said is None:
        try:
            said = job.hear(rec, 0.0, rec["seconds"])
        except hearing.HearingError as e:
            return [], str(e)
        hearing.keep(rec, 0.0, rec["seconds"], said)
    return said, None


def _clip_lines(c):
    rec = c["rec"]
    shape = "upright" if rec["height"] > rec["width"] else "lying (shown whole over a blurred copy of itself)" if (
        not montage.fills(rec["width"], rec["height"])) else "square"
    lines = ["%d. %s · %s · %g s · %s %dx%d" % (c["n"], rec["name"], c["when"], rec["seconds"], shape, rec["width"],
                                                rec["height"])]
    if c["stills"]:
        lines.append("   The picture does not move at %s: a frozen screen, so no piece may hold it." %
                     ", ".join("%g-%g s" % s for s in c["stills"]))
    if c["said"] is None:
        lines.append("   No sound.")
    elif c["unheard"]:
        lines.append("   What is said could not be heard: %s" % c["unheard"])
    elif not c["said"]:
        lines.append("   No words heard.")
    else:
        sure = [p for p in c["said"] if not _unsure(p)]
        lines += ["   %g s %s" % (round(p["from"], 1), p["text"]) for p in sure]
        if len(sure) < len(c["said"]):
            lines.append("   %d more line%s heard but not sure, such as voices around the camera: left out." % (
                len(c["said"]) - len(sure), "" if len(c["said"]) - len(sure) == 1 else "s"))
    return lines


def arrange_clips(job, why, story, clips, left_out=None):
    opened = job.chat.clips
    if not opened:
        raise Refused("No clips are open. Open a folder of clips with open_clips first.")
    if (job.chat.clip or {}).get("recipe") != MONTAGE:
        raise Refused("Load the %s recipe first: it says how to choose and order the clips." % MONTAGE)
    known = {c["n"]: c for c in opened["clips"]}
    used = [c["clip"] for c in clips]
    both = sorted({c["clip"] for c in left_out or []} & set(used))
    out = {}
    for c in left_out or []:
        if c["clip"] not in both:
            out.setdefault(c["clip"], c)
    left_out = list(out.values())
    named = used + [c["clip"] for c in left_out]
    problems = []
    twice = sorted({n for n in named if named.count(n) > 1})
    if twice:
        problems.append("Clip%s %s named twice: each clip is either used once or left out." % (
            "s" if len(twice) > 1 else "", ", ".join(map(str, twice))))
    strange = sorted({n for n in named if n not in known})
    if strange:
        problems.append("There is no clip %s: the clips are numbered 1 to %d." % (
            ", ".join(map(str, strange)), len(known)))
    missing = sorted(set(known) - set(named))
    if missing:
        problems.append("Clip%s %s %s neither used nor left out. Say for every clip whether it goes in, and why it "
                        "does not when it does not." % ("s" if len(missing) > 1 else "", ", ".join(map(str, missing)),
                                                        "are" if len(missing) > 1 else "is"))
    pieces, lines, fitted = [], [], []
    for k, c in enumerate(clips, 1):
        if c["clip"] not in known:
            continue
        rec = known[c["clip"]]["rec"]
        a = c.get("start", 0.0)
        b = min(c.get("end", rec["seconds"]), rec["seconds"])
        frozen = [s for s in known[c["clip"]]["stills"] if s[0] < b and a < s[1]]
        if b <= a:
            problems.append("Clip %d: start %g s must come before end %g s; the clip is %g s long."
                            % (c["clip"], a, b, rec["seconds"]))
        elif b - a < min(SHORTEST_PIECE, rec["seconds"]) - 0.01:
            problems.append("Clip %d: %g-%g s is under %g s, too short to see. Keep more of it, or leave it out."
                            % (c["clip"], a, b, SHORTEST_PIECE))
        else:
            fit = _moving(a, b, known[c["clip"]]["stills"], LONGEST_PIECE, min(SHORTEST_PIECE, b - a))
            if fit is None:
                problems.append("Clip %d: its picture does not move at %s, and a frozen screen never goes in. Keep "
                                "only seconds outside that, or leave the clip out." % (
                                    c["clip"], ", ".join("%g-%g s" % s for s in frozen)))
            elif abs(fit[0] - a) > 0.01 or abs(fit[1] - b) > 0.01:
                fitted.append("clip %d: %g-%g s of the %g-%g s you gave (%s)" % (
                    c["clip"], fit[0], fit[1], round(a, 2), round(b, 2), "its picture holds still at %s" % ", ".join(
                        "%g-%g s" % s for s in frozen) if frozen else "a piece is at most %g s" % LONGEST_PIECE))
                a, b = fit
        pieces.append({"rec": rec, "from": round(a, 2), "to": round(max(a, b), 2), "clip": c["clip"],
                       "why": c["why"]})
        if c.get("words"):
            lines.append({"piece": len(pieces), "text": c["words"]})
    asked = job.chat.clip["length_s"]
    most = min(LONGEST, asked * OVER_LENGTH)
    length = sum(p["to"] - p["from"] for p in pieces)
    if length > most:
        target = min(asked, most)
        _shortened(pieces, target)
        now = sum(p["to"] - p["from"] for p in pieces)
        if now < length - 0.05:
            fitted.append("the pieces came to %g s, longer than the %g s asked, so the end of each was cut by the "
                          "same share, keeping at least %g s of each" % (round(length, 1), target, SHORTEST_FIT))
        length = now
    if not SHORTEST <= round(length) <= most:
        problems.append("The pieces add up to %g s; this clip is %d to %d s (about %d s was asked). %s" % (
            round(length, 1), SHORTEST, most, asked, "Use fewer clips." if length > most else
            "Keep more seconds of each, or more clips."))
    if clips and not clips[0].get("words"):
        problems.append("The first piece (clip %d) opens the clip: give it words that say where we are or what this "
                        "is." % clips[0]["clip"])
    if len(lines) < round(length / LINE_EVERY):
        problems.append("%d line%s of words for %g s: give at least %d, one for each part of the story, so the "
                        "viewer knows what each part is." % (len(lines), "" if len(lines) == 1 else "s",
                                                             round(length, 1), round(length / LINE_EVERY)))
    if problems:
        raise Refused("Nothing was joined. Fix all of these, then call it again:\n- " + "\n- ".join(problems))
    try:
        texts, spelling = _spell(job, [ln["text"] for ln in lines]) if lines else ([], "")
    except hearing.Stopped:
        return "Not arranged: the person pressed Stop."
    for ln, words in zip(lines, texts):
        ln["text"] = words
        _fits_a_place(words, "the words over clip %d" % pieces[ln["piece"] - 1]["clip"], job.chat.clip["style"])
    out = montage.target(pieces, os.path.basename(opened["folder"]))
    job.working("Joining %d pieces into one clip" % len(pieces))
    try:
        montage.join(pieces, out, on_start=job.started_work)
    except montage.Stopped:
        return "Not arranged: the person pressed Stop."
    except montage.MontageError as e:
        return "The clips could not be joined: %s Tell the person." % e
    finally:
        job.done_working()
    try:
        rec = recording.probe(out)
    except recording.NotOpened as e:
        return "The joined clips could not be read back: %s Tell the person." % e
    job.chat.open(rec, job=job.n)
    at = montage.starts(pieces)
    for ln in lines:
        k = ln["piece"] - 1
        later = [x for x in lines if x["piece"] > ln["piece"]]
        ln["start"], ln["end"] = at[k], (at[later[0]["piece"] - 1] if later else round(rec["seconds"], 2))
    look, moment = _montage_moment(job, rec, pieces, at, story, lines)
    job.chat.looks.append(look)
    job.chat.moments = [moment]
    job.show(look=look, moments=[moment], arranged={"story": story, "pieces": [
        {"clip": p["clip"], "name": p["rec"]["name"], "from": p["from"], "to": p["to"], "at": t, "why": p["why"]}
        for p, t in zip(pieces, at)], "left_out": left_out, "lines": lines})
    said = ["Joined %d clips into one recording, %g s long, in your order: %s" % (
        len(pieces), round(rec["seconds"], 1), story)]
    if both:
        said.append("Clip%s %s %s used, so %s left-out reason was dropped." % (
            "s" if len(both) > 1 else "", ", ".join(map(str, both)), "are" if len(both) > 1 else "is",
            "their" if len(both) > 1 else "its"))
    for p, t in zip(pieces, at):
        said.append("%g s: clip %d, %g-%g s of it: %s" % (t, p["clip"], p["from"], p["to"], p["why"]))
    for c in left_out:
        said.append("Left out: clip %d: %s" % (c["clip"], c["why"]))
    if fitted:
        said.append("The program fitted the pieces: %s." % "; ".join(fitted))
    if spelling:
        said.append(spelling)
    made = make_clip(job, why, 1, "subject")
    return "\n".join(said) + "\n\n" + made


def _moving(a, b, stills, longest, shortest):
    """(start, end): the first stretch from a to b whose picture moves all through (stills: [[x, y]], where it holds
    still), at least shortest s and at most longest s; None when there is none."""
    t = a
    for x, y in sorted(s for s in stills if s[0] < b and a < s[1]):
        if x - t >= shortest - 0.01:
            break
        t = max(t, y)
    else:
        x = b
    if min(x, b) - t < shortest - 0.01:
        return None
    return round(t, 2), round(min(x, b, t + longest), 2)


def _shortened(pieces, target):
    """Cuts the end of each piece by the same share of what it has over SHORTEST_FIT s, so they add up to about
    target s; a piece of SHORTEST_FIT s or less is kept whole."""
    total = sum(p["to"] - p["from"] for p in pieces)
    room = sum(max(0.0, p["to"] - p["from"] - SHORTEST_FIT) for p in pieces)
    share = min(1.0, max(0.0, total - target) / room) if room > 0 else 0.0
    for p in pieces:
        over = max(0.0, p["to"] - p["from"] - SHORTEST_FIT)
        p["to"] = round(p["to"] - over * share, 2)


def _fits_a_place(words, what, style):
    """Refused when words do not fit the largest place for a title in the subject layout."""
    lay = layouts.make("subject", (layouts.W, layouts.H))
    band = max(lay["places"].values(), key=lambda b: b[2] * b[3])
    try:
        size = text.measure(words, render.style_of({"style": None}, style), "title", band[2])
    except ValueError as e:
        raise Refused("%s: %s" % (what, e))
    if size.w > band[2] or size.h > band[3]:
        raise Refused("%s take %d lines at the title size; the place holds %d px of the %d px they need. Use fewer "
                      "words." % (what[0].upper() + what[1:], len(size.lines), band[3], size.h))


def _montage_moment(job, rec, pieces, at, story, lines):
    """The look and the one moment of a joined montage, made by the program from the clips Muse arranged: what is
    said in each piece moved to the joined recording's seconds."""
    said = []
    for p, t in zip(pieces, at):
        for s in hearing.heard(p["rec"], 0.0, p["rec"]["seconds"]) or []:
            a, b = max(s["from"], p["from"]), min(s["to"], p["to"])
            if b > a:
                said.append(dict(s, **{"from": round(t + a - p["from"], 2), "to": round(t + b - p["from"], 2)}))
    pictures = []
    for p, t in zip(pieces, at):
        mid = round(t + (p["to"] - p["from"]) / 2, 2)
        frame = recording.frame(rec, at=mid)
        if frame:
            pictures.append({"t": mid, "frame": frame})
    end = round(rec["seconds"], 2)
    look = {"n": len(job.chat.looks) + 1, "start": 0.0, "end": end, "pictures": pictures, "loudest": None,
            "said": said, "heard": [0.0, end]}
    first = pictures[0] if pictures else {"t": 0.0, "frame": None}
    moment = {"n": 1, "look": look["n"], "start": 0.0, "end": end, "reason": story, "snapped": None,
              "frame": first["frame"], "t": first["t"], "said": _heard_said(look, 0.0, end), "lines": lines,
              "preview": previews.cut(job.chat, job.n, rec, 0.0, end)}
    return look, moment


def _put_lines(job, rec, lines):
    """Puts a montage's words on as titles, each at the first of the layout's places it covers no face in for its
    whole time, else where it covers faces the least, for the first stretch of its time (LISTED s or more) where it
    covers none. Returns sentences for Muse. Seen 10-28: Muse spent four steps moving a line off a face itself, and
    ran out of room before the hand-over."""
    state = _clip_state(job)
    lay = state["clip"]["layout"]
    st = render.style_of(state, job.chat.clip["style"] if job.chat.clip else "clean")
    times, every = _face_times(state, rec)
    out = []
    for ln in lines:
        if any(t["text"] == ln["text"] and t["start"] == ln["start"] for t in state["titles"]):
            continue
        best = None
        for name, band in lay["places"].items():
            size = text.measure(ln["text"], st, "title", band[2])
            if size.w > band[2] or size.h > band[3]:
                continue
            drawn = [band[0] + (band[2] - size.w) / 2, band[1] + (band[3] - size.h) / 2, size.w, size.h]
            hit = _covered(drawn, times, every, ln["start"], ln["end"])
            if best is None or sum(b - a for a, b in hit) < sum(b - a for a, b in best[2]):
                best = (name, band, hit)
            if not hit:
                break
        if best is None:
            out.append('"%s" fits none of the places at the title size: arrange the clips again with fewer words.'
                       % ln["text"])
            continue
        name, band, hit = best
        start, end = ln["start"], ln["end"]
        clear = _clear(hit, start, end) if hit else []
        if clear:
            start, end = clear[0]
            start, end, hit = math.ceil(start * 10) / 10, math.floor(end * 10) / 10, []
        text.title(ln["text"], start, end, band, st, None)
        state = job.edit({"do": "title", "text": ln["text"], "start": start, "end": end, "place": name,
                          "box": band, "key": None})
        out.append('"%s" at the %s from %g to %g s%s.' % (
            ln["text"], name, start, end, ", where it covers a face at %s: move it with remove_title and "
            "add_title" % _spans(hit) if hit else ", clear of every face" if clear == [] else
            ", the first part of its piece where it covers no face"))
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
MOST_MOMENTS = 8


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


SPILL = 6.0


def _what_is_said(job, rec, look):
    """Lines for Muse about what is said in the look's stretch, as heard. The phrases go into look["said"].
    Misheard words are fixed later, only for the moments Muse proposes (measured 10-27: the fix took 20-42 s a look
    on Glimmer, the hearing 5.6 s, and most looks are not picked). The hearing takes SPILL s more on each side
    (look["heard"]), so a line the look's edge falls in is heard whole and a moment's edge can move out of it: on
    10-27 a look ending at 12:29 heard "ว่าตอนนี้นี่ย" as the last line, where the speaker went on to 12:30.4, "ว่าตอนนี้
    เราทำไปถึงไหนแล้วนะครับ", and the clip ended in the middle of the sentence."""
    a, b, clock = look["start"], look["end"], recording.clock
    if not rec["sound_tracks"]:
        return ["The recording has no sound, so nothing is said."]
    ha, hb = round(max(0.0, a - SPILL), 2), round(min(rec["seconds"], b + SPILL), 2)
    said = hearing.heard(rec, ha, hb)
    if said is None:
        started = time.time()
        try:
            said = job.hear(rec, ha, hb)
        except hearing.HearingError as e:
            return ["What is said here could not be heard: %s" % e]
        look["took"] = round(time.time() - started, 1)
        hearing.keep(rec, ha, hb, said)
    look["said"], look["heard"] = said, [ha, hb]
    inside = [p for p in said if p["to"] > a and p["from"] < b]
    if not inside:
        return ["No words were heard here."]
    return (["What is said here, by time, as heard (some words may be misheard; the numbers are as heard; a line "
             "marked (not sure) is one the hearing was not sure of, so do not build on it):"] +
            ["%s %s%s" % (clock(p["from"]), p["text"], " (not sure)" if _unsure(p) else "") for p in inside])


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


LATER = "fixed when a clip is made from this moment"


def _heard_said(look, a, b):
    """The lines heard from a to b as heard, to be fixed when a clip is made from them: Muse serves one question at a
    time, so fixing the other moments' lines first would keep the first clip waiting (measured 10-27: three moments
    took 195 s to propose)."""
    return [dict(p, heard=p["text"], kept=LATER, named=[]) for p in look.get("said") or [] if a <= p["from"] < b]


def _fix_moment(job, rec, m):
    """Fixes the moment's lines when they were kept as heard for later. Stop raises hearing.Stopped."""
    if any(s.get("kept") == LATER for s in m.get("said") or []):
        m["said"] = _fixed_said(job, rec, job.chat.looks[m["look"] - 1], m["start"], m["end"])


AIR = 0.25
SNAP_REACH = 2.0
FINISH_REACH = 4.0


def _snapped(look, a, b):
    """(start, end, sentence or None): a moment's edges moved out of the words they would cut through, to just before
    the line the start falls in when it began within SNAP_REACH, and just after the line the end falls in when it
    finishes within FINISH_REACH (a sentence cut short is worse than an opening), AIR seconds from the words; never
    outside what was heard. A model's own edges sit in a pause about half the time (measured by lane 5 on Tarm's own
    edits: 47%, 77% within 0.6 s), so a clip would often open or close on half a word."""
    na, nb, said = a, b, look.get("said") or []
    lo, hi = look.get("heard") or (look["start"], look["end"])
    for p in said:
        if p["from"] < a < p["to"] and a - p["from"] <= SNAP_REACH:
            na = max(lo, p["from"] - AIR)
        if p["from"] < b < p["to"] and p["to"] - b <= FINISH_REACH:
            nb = min(hi, p["to"] + AIR)
    moved = []
    if na != a:
        moved.append("starts %.1f s earlier, so it opens before a line rather than inside it" % (a - na))
    if nb != b:
        moved.append("ends %.1f s later, so it closes after a line rather than inside it" % (nb - b))
    return round(na, 2), round(nb, 2), " and ".join(moved) or None


DEAD_AIR = 1.0
KEPT_AIR = 0.3


def _pieces(said, a, b, still):
    """(the pieces of the recording a clip from a to b is made of, a sentence for Muse or None): the earlier take of a
    line said again is cut (takes.said_again), and every silence longer than DEAD_AIR is cut down to KEPT_AIR at each
    side (pauses.tighten), the clip's opening and closing silence down to KEPT_AIR, where still(x, y) says nothing
    happens on the picture from x to y: a silence while the answer streams onto the screen is the payoff, not dead
    air. still None keeps the moment whole (a game, where the action is the clip and nobody needs to speak). Measured
    on 10-27's five clips: three opened silent for 0.5 s or more and were not handed over, and one held a first take
    and an 8 s stop before the line was said again."""
    if still is None:
        return [[a, b]], None
    said = [p for p in said if p["to"] > p["from"]]
    again = takes.said_again(said, a, b)
    pieces = takes.without([[a, b]], [x["cut"] for x in again])
    try:
        _, quiet = pauses.tighten(pieces, said, gap=DEAD_AIR, air=KEPT_AIR)
    except ValueError:
        quiet = []
    words = sorted((max(p["from"], x), min(p["to"], y)) for p in said if not pauses.filler(p["text"])
                   for x, y in pieces if p["from"] < y and x < p["to"])
    if words:
        quiet += [[pieces[0][0], words[0][0] - KEPT_AIR], [max(w[1] for w in words) + KEPT_AIR, pieces[-1][1]]]
    quiet = [q for q in quiet if q[1] - q[0] >= 0.05 and still(q[0], q[1])]
    pieces = takes.without(pieces, quiet)
    gone = round((b - a) - sum(y - x for x, y in pieces), 1)
    if gone < 0.1:
        return pieces, None
    parts = ["the first take of a line said again at %s (\"%s\"; the take from %s is kept)" % (
        recording.clock(x["cut"][0]), x["first"], recording.clock(x["cut"][1] + takes.AIR)) for x in again]
    inner = len(pieces) - 1 - len(again)
    if inner > 0:
        parts.append("%d silence%s longer than %g s while nothing moved on the picture"
                     % (inner, "" if inner == 1 else "s", DEAD_AIR))
    return pieces, "The program cut %g s out of the moment: %s." % (gone, "; ".join(
        parts + ["the silence at its edges, so it opens and closes %g s from a word" % KEPT_AIR]))


def _still(rec, recipe, a, b, avoid):
    """still(x, y) for _pieces: True when less than focus.QUIET of the picture outside avoid (the face camera, which
    moves while nobody speaks) changes in any second from x to y; always True for people talking, where a silence
    is dead air whatever moves; None for a game. When the picture cannot be looked at, nothing counts as still."""
    if recipe in ("game", MONTAGE):
        return None
    if recipe == "talk":
        return lambda x, y: True
    try:
        found, _, (sw, sh) = focus.heat(rec, [0, 0, rec["width"], rec["height"]], a, b)
    except focus.FocusError:
        return lambda x, y: False
    keep = np.ones((sh, sw), np.float32)
    if avoid:
        k = sw / float(rec["width"])
        keep[max(0, int(avoid[1] * k)):int(math.ceil((avoid[1] + avoid[3]) * k)),
             max(0, int(avoid[0] * k)):int(math.ceil((avoid[0] + avoid[2]) * k))] = 0
    moved = [float((m * keep).sum()) >= focus.QUIET * sw * sh for m in found]

    def still(x, y):
        first, last = int((x - a) / focus.STEP), int(math.ceil((y - a) / focus.STEP))
        return not any(moved[max(0, first):min(len(moved), last)])
    return still


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
        a, b, snapped = _snapped(look, a, b)
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
                       "snapped": snapped,
                       "frame": shown["frame"], "t": shown["t"]})
    if _reads_screen(job):
        for p in picked:
            screen.ahead(rec, p["start"], p["end"])
    started = time.time()
    try:
        for k, p in enumerate(picked):
            look = looks[p["look"] - 1]
            p["said"] = (_fixed_said(job, rec, look, p["start"], p["end"]) if k == 0 else
                         _heard_said(look, p["start"], p["end"]))
    except hearing.Stopped:
        return "Not proposed: the person pressed Stop."
    fixing = round(time.time() - started, 1)
    renamed = []
    if _reads_screen(job):
        for p in picked:
            try:
                shown = screen.words(rec, p["start"], p["end"])
            except screen.ScreenError:
                continue
            wrong = _misspelled(p["reason"], shown + [w for w in screen.sites(rec) if w not in shown])
            p["reason"] = _as_on_screen(p["reason"], wrong)
            renamed += [w for w in wrong if w not in renamed]
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
        if p["snapped"]:
            lines.append("   The program moved its edges: it %s." % p["snapped"])
        lines += ["   %s %s%s" % (clock(s["from"]), s["text"], " (as heard)" if s["kept"] else "") for s in p["said"]]
    if renamed:
        lines.append("The program wrote the names in your reasons as the screen writes them: %s."
                     % "; ".join('"%s" as "%s"' % w for w in renamed))
    if len(picked) > 1:
        lines.append("Misheard words are fixed in moment 1's lines; the other moments show theirs as heard until a "
                     "clip is made from one of them.")
    lines.append("When the person asked for several clips, make them all in one go, one from each moment. Otherwise "
                 "go on with moment 1 by yourself and make the clip from it: do not ask the person to choose, and do not "
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


def _found_faces(rec, pieces, span=None):
    """[(a, b, the faces found from a to b, or None when nobody looked)]: the faces found over span, the moment the
    clip was made from, as one stretch; or over each piece, for a clip kept from before clips were cut into pieces.
    The faces are looked for over the whole moment, so a piece alone was never looked at: asked piece by piece, a
    clip with its silences cut knew no face, and words could go on one."""
    whole = faces.kept(rec, *span) if span else None
    return [(span[0], span[1], whole)] if whole else [(a, b, faces.kept(rec, a, b)) for a, b in pieces]


def _face_times(state, rec):
    """[(clip second, [boxes in the frame to keep words off])] for every picture the face finder took of the clip,
    and the seconds between two pictures. The product Muse named is in every picture."""
    clip, out, every = state["clip"], [], faces.EVERY
    lay = clip["layout"]

    def on_frame(at, box):
        return layouts.to_frame(lay, box, at)

    product = layouts.to_frame(lay, clip["product"]) if clip.get("product") else []
    if product:
        out = [(round(k * every, 2), list(product)) for k in range(int(state["length"] / every) + 1)]
    for a, b, found in _found_faces(rec, clip["pieces"], clip.get("span")):
        if not found:
            continue
        every = found["every"]
        for t, boxes in faces.between(found, a, b):
            at = project.from_recording(clip["pieces"], t)
            if at is not None:
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


def make_clip(job, why, moment, layout, crop=None, face=None, product=None, follow=True):
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
    try:
        _fix_moment(job, rec, m)
    except hearing.Stopped:
        return "Not made: the person pressed Stop."
    recipe = (job.chat.clip or {}).get("recipe")
    if layout in ("talk", "speaker", "split"):
        return _make_talk(job, rec, moment, m, *_pieces(m.get("said") or [], a, b, _still(rec, "talk", a, b, None)),
                          name=layout)
    try:
        box, face_line = _the_face(job, rec, a, b, face)
    except faces.Stopped:
        return "Not made: the person pressed Stop."
    try:
        lay = layouts.make(layout, (rec["width"], rec["height"]), box, _pixels("crop", crop, rec) if crop else None)
    except ValueError as e:
        raise Refused(str(e))
    pieces, cut = _pieces(m.get("said") or [], a, b, _still(rec, recipe, a, b, layouts.face_room(lay)))
    following = ""
    if lay["kind"] == "card" and follow and not crop and _reads_screen(job):
        lay, following = _follow(rec, lay, pieces, [a, b])
    item = [round(v) for v in _pixels("product", product, rec)] if product else None
    if item and not layouts.to_frame(lay, item):
        lay["notes"].append("The product you named is not in the clip's picture: choose a crop that holds it.")
    if not item and (job.chat.clip or {}).get("recipe") == "seller":
        lay["notes"].append("This is a seller clip and no product box was named, so nothing keeps words off the "
                            "product. Make the clip again with product: [x1, y1, x2, y2] on your grid around the product "
                            "in the picture.")
    private = _private_lines(job, rec, lay, pieces, [a, b])
    if private is None:
        return "Not made: the person pressed Stop."
    people = _people(rec, pieces, box, [a, b])
    state = job.edit({"do": "clip", "moment": moment, "pieces": pieces, "span": [a, b], "layout": lay, "product": item,
                      "people": [{k: v for k, v in p.items() if k != "crop"} for p in people or []],
                      "speech": _speech(rec, job.chat.looks[m["look"] - 1], pieces)})
    for f in private["shown"]:
        state = job.edit({"do": "blur", "box": f["box"], "start": f["start"], "end": f["end"], "private": f["kind"]})
    lines = ["Made the clip from moment %d (%s-%s, %g s) in the %s layout." % (
        moment, recording.clock(a), recording.clock(b), state["length"], lay["kind"]), face_line] + lay["notes"]
    if cut:
        lines.append(cut)
    if following:
        lines.append(following)
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
    if m.get("lines"):
        lines.append("The words over the clips, put on by the program:")
        lines += ["- " + s for s in _put_lines(job, rec, m["lines"])]
        state = job.chat.project.state()
    peak = project.from_recording(pieces, m["t"] if m.get("t") is not None else (a + b) / 2)
    at = min(max(0.0, peak if peak is not None else state["length"] / 2), state["length"] - 0.1)
    lines.append(_show_frame(job, state, rec, at, "The clip at %g s (moment %d, %s layout):" % (
        round(at, 2), moment, lay["kind"])))
    if people and recipe == MONTAGE:
        state = job.edit({"do": "people", "blur": [], "keep": [p["n"] for p in people]})
        lines.append(KEPT_AT_AN_EVENT % len(people))
    else:
        lines += _people_lines(job, people, rec)
    lines.append("Look at it: if a face or the action is cut badly, make the clip again with another layout or crop. "
                 "Then put the title on.")
    job.show(clip=job.chat.clip_view())
    return "\n".join(lines)


KEPT_AT_AN_EVENT = ("%d people's faces are in the clip, all kept as they are: at a trip or an event the people are "
                    "part of the day. Blur someone only when the person asks: show_people numbers the faces at the "
                    "second they mean, and decide_people blurs that number.")


def _follow(rec, lay, pieces, span):
    """(the card layout following the action on the screen, the sentence for Muse), or the layout as it was."""
    card = next(p for p in lay["parts"] if p["name"] == "card")
    whole = card["src"][2]
    try:
        text = [dict(line, at=r["at"]) for r in screen.readings(rec, span[0], span[1]) for line in r["lines"]]
    except screen.ScreenError:
        text = None
    try:
        plan = focus.plan(rec, pieces, [0, 0, rec["width"], rec["height"]], card["dst"][2] / float(card["dst"][3]),
                          layouts.face_room(lay), text, card["dst"][2])
    except focus.FocusError as e:
        return lay, "The card shows the whole screen: the screen could not be looked at (%s)." % e
    if not plan:
        return lay, ("The card shows the whole screen: the screen changes everywhere at once, or nowhere, so no one "
                     "part of it is where the action is.")
    readable = ""
    if plan.get("letters"):
        readable = (" The screen's words are about %g px on a phone%s." % (plan["letters"], "" if plan["letters"] >=
                    focus.PHONE_PX - 0.5 else ", under the %d px floor: the screen writes them smaller than the "
                    "closest zoom can lift" % focus.PHONE_PX))
    size, shots = plan["size"], plan["shots"]
    if len(shots) == 1:
        moves = "holds one place, %s on your grid" % _grid([shots[0]["x"], shots[0]["y"]] + size, rec)
    elif plan.get("on") == "lines":
        moves = "moves where the lines are when the place it holds has lost them, %d places: %s" % (len(shots), "; ".join(
            "%s on your grid from %g s" % (_grid([s["x"], s["y"]] + size, rec), s["from"]) for s in shots))
    else:
        moves = "moves to where the screen changes, %d places: %s" % (len(shots), "; ".join(
            "%s on your grid from %g s" % (_grid([s["x"], s["y"]] + size, rec), s["from"]) for s in shots))
    return layouts.followed(lay, plan), (
        "%s: it shows %d x %d of the %d x %d picture, so the words on the screen are %.1f times as big as on the "
        "card without following, and it %s. It never shows the face camera. If it shows the wrong part, make the clip "
        "again with a crop around what matters, or with follow false for the whole screen.%s" % (
            "Nothing on the screen changes, so the card holds where the most lines are written, from their start"
            if plan.get("on") == "lines" else "The card follows the action on the screen", size[0], size[1],
            rec["width"], rec["height"], whole / float(size[0]), moves, readable))


def _reads_screen(job):
    clip = job.chat.clip
    return bool(clip) and (RECIPES.get(clip["recipe"]) or {}).get("read") == "screen"


NEAR_NAME = 0.4


def _screen_words(job, rec, state):
    """The words written on the screen in the clip, for a recipe that reads the screen; [] otherwise, or when it cannot
    be read."""
    if not _reads_screen(job):
        return []
    try:
        span = state["clip"].get("span")
        found = screen.words(rec, *span) if span else [w for a, b in state["clip"]["pieces"]
                                                        for w in screen.words(rec, a, b)]
    except screen.ScreenError:
        return []
    return found + [w for w in screen.sites(rec) if w not in found]


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
            right = long_[near[1]]
            out.append((w, right[:1].upper() + right[1:] if w[:1].isupper() and right.islower() and "." not in right
                        else right))
    return out


def _as_on_screen(words, wrong):
    """words with each name in wrong ([(as written, as the screen writes it)]) written as the screen writes it."""
    for w, right in wrong:
        words = re.sub(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(w), right, words)
    return words


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


def _private_lines(job, rec, lay, pieces, span=None):
    """{"shown": the private lines the screen shows in the clip's picture, "said": the sentence for Muse}, empty when
    the recipe does not read the screen, or None when the person pressed Stop."""
    if not _reads_screen(job):
        return {"shown": [], "said": ""}
    job.working("Reading the screen for anything private")
    try:
        found = screen.hidden(rec, pieces, on_start=job.started_work, span=span)
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


def _shown_private(job, rec, state):
    """(the private lines the clip's picture shows, None), or (None, why the screen could not be read)."""
    job.working("Reading the screen for anything private")
    try:
        found = screen.hidden(rec, state["clip"]["pieces"], on_start=job.started_work, span=state["clip"].get("span"))
    except screen.Stopped:
        return None, "not read: the person pressed Stop"
    except screen.ScreenError as e:
        return None, "the screen could not be read (%s)" % e
    finally:
        job.done_working()
    return [f for f in found if layouts.to_frame(state["clip"]["layout"], f["box"])], None


NAME_LINE = "a person's handle"


def _names_check(job, rec, state):
    """(ok, found) for the check that no viewer's name can be read in a game stream's clip: every line read on the
    screen with a handle is under a blur for as long as it shows, and one blur stays for the whole clip, as the chat
    box does (a name read without its @ is not known to be a name). Game afc1 10-28: the check passed on a blur over
    the whole clip that lay under the face, while the chat at the top right showed."""
    shown, why = _shown_private(job, rec, state)
    if shown is None:
        return False, why
    names = [f for f in shown if f["kind"] == NAME_LINE]
    bare = [f for f in names if not screen.covered(f, state["blurs"])]
    if bare:
        return False, "names can be read at %s on your grid from %g to %g s, not under a blur" % (
            _grid(_around(bare), rec), min(f["start"] for f in bare), max(f["end"] for f in bare))
    whole = [b for b in state["blurs"] if b.get("box") and b["start"] <= 0.1 and b["end"] >= state["length"] - 0.1]
    if not whole:
        return False, "no blur covers the whole clip, so the chat box shows"
    return True, ("%d line%s with a name, under a blur for as long as it shows" % (
        len(names), "" if len(names) == 1 else "s") if names else
        "%d blur over the whole clip; no name read on the screen" % len(whole))


def _around(lines):
    """The box [x, y, w, h] around the boxes of lines."""
    x1, y1 = min(f["box"][0] for f in lines), min(f["box"][1] for f in lines)
    x2 = max(f["box"][0] + f["box"][2] for f in lines)
    y2 = max(f["box"][1] + f["box"][3] for f in lines)
    return [x1, y1, x2 - x1, y2 - y1]


def _private_check(job, rec, state):
    """(ok, found) for the check that every private line the clip shows is hidden for as long as it shows."""
    shown, why = _shown_private(job, rec, state)
    if shown is None:
        return False, why
    bare = [f for f in shown if not screen.covered(f, state["blurs"])]
    if bare:
        return False, "not hidden: %s" % _private_spans(bare)
    return True, ("%d private line%s, each hidden for as long as it shows" % (len(shown), "" if len(shown) == 1 else
                                                                               "s") if shown else NOTHING_PRIVATE)


def _make_talk(job, rec, moment, m, pieces, cut, name="talk"):
    """make_clip in a layout of people talking: the people found, who speaks when, and the shots that follow them
    (talk); the same with the others on screen in circles (speaker); or two people, one above the other, for the
    whole clip (split)."""
    a, b = m["start"], m["end"]
    try:
        found = job.find_faces(rec, a, b)
    except faces.NoFaces as e:
        raise Refused("The talk layout needs the people's faces, and they could not be looked for: %s" % e)
    except faces.Stopped:
        return "Not made: the person pressed Stop."
    persons = talk.people(found)
    if len(persons) < 2:
        raise Refused("The %s layout follows two or more people talking, and the program found %d sitting still in "
                      "the moment's %d pictures. Choose the face or the subject layout." % (name, len(persons),
                                                                                         len(found["samples"])))
    if name == "split" and len(persons) != 2:
        raise Refused("The split layout shows two people, one above the other, and the program found %d. Choose the "
                      "speaker or the talk layout." % len(persons))
    heard = job.chat.looks[m["look"] - 1].get("said")
    try:
        spoken, voice_line = _voices(job, rec, m, persons)
    except hearing.Stopped:
        return "Not made: the person pressed Stop."
    if name == "split":
        lay = talk.split((rec["width"], rec["height"]), persons)
    else:
        job.working("Watching who speaks")
        shots, _ = talk.plan(rec, pieces, heard, persons, voices=spoken)
        lay = talk.make((rec["width"], rec["height"]), persons, shots, name == "speaker")
    private = _private_lines(job, rec, lay, pieces, [a, b])
    if private is None:
        return "Not made: the person pressed Stop."
    people = _people(rec, pieces, None, [a, b])
    state = job.edit({"do": "clip", "moment": moment, "pieces": pieces, "span": [a, b], "layout": lay, "product": None,
                      "people": [{k: v for k, v in p.items() if k != "crop"} for p in people or []],
                      "speech": _speech(rec, job.chat.looks[m["look"] - 1], pieces), "voices": spoken})
    for f in private["shown"]:
        state = job.edit({"do": "blur", "box": f["box"], "start": f["start"], "end": f["end"], "private": f["kind"]})
    lines = ["Made the clip from moment %d (%s-%s, %g s) in the %s layout. It follows %d people, numbered from the "
             "left: %s.%s" % (moment, recording.clock(a), recording.clock(b), state["length"], name, len(persons),
                              "; ".join("person %d at %s on your grid" % (p["n"], _grid(p["box"], rec))
                                        for p in persons),
                              {"speaker": " Everyone is on screen all the time: the one shown fills the frame, the "
                                          "others are in circles at the top.",
                               "split": " Person 1 is in the top half and person 2 in the lower half for the whole "
                                        "clip, so both are always on screen."}.get(name, ""))]
    if voice_line:
        lines.append(voice_line)
    if name != "split":
        lines += ["Who speaks was measured: %s." % (
                      "from the voices; the mouths only find the listener's reactions" if spoken else
                      "whose mouth moves clearly more while words are heard" if heard is not None else
                      "whose mouth moves clearly more (the moment was never heard, so from movement alone)"),
                  "Shots: %s." % talk.said(lay["shots"]),
                  "If a shot shows the wrong person, or a reaction should be seen, change it with change_shot."]
    if cut:
        lines.insert(1, cut)
    if private["said"]:
        lines.append(private["said"])
    lines += lay["notes"] + [lay["safe"]["said"]]
    lines.append("Places for words on screen, kept clear of every person's head (the band from y %d to %d is kept for "
                 "the captions):" % (lay["captions"][1], lay["captions"][1] + lay["captions"][3]))
    lines += _place_lines(state, rec)
    peak = project.from_recording(pieces, m["t"] if m.get("t") is not None else (a + b) / 2)
    at = min(max(0.0, peak if peak is not None else state["length"] / 2), state["length"] - 0.1)
    lines.append(_show_frame(job, state, rec, at, "The clip at %g s (moment %d, %s layout):" % (round(at, 2),
                                                                                                  moment, name)))
    lines += _people_lines(job, people, rec)
    job.show(clip=job.chat.clip_view())
    return "\n".join(lines)


def _voices(job, rec, m, persons):
    """(who says each of the moment's lines, [{"from", "to", "who"}] in seconds of the recording, or None; a sentence
    for Muse). Two people only: their voices are told apart by how high they are, and Muse matches each voice to a
    face from pictures. Stop raises hearing.Stopped."""
    said = [p for p in m.get("said") or [] if p["to"] - p["from"] >= voices.SHORTEST]
    if len(persons) != 2 or len(said) < 2:
        return None, ""
    job.working("Hearing how high each voice is")
    try:
        heard = voices.pitches(rec, said)
    except voices.VoiceError as e:
        return None, "The voices could not be heard (%s), so who speaks is found from mouths moving." % e
    finally:
        job.done_working()
    cut = voices.split(heard)
    if cut is None:
        return None, ("The voices are too alike to tell apart by how high they are, so who speaks is found from mouths "
                      "moving.")
    voiced = voices.lines_of(heard, cut)
    try:
        people = voices.match(lambda content, schema: job.ask_aside([{"role": "user", "content": content}], schema,
                                                                    "Telling the voices apart"), rec, voiced, persons)
    except voices.VoiceError as e:
        return None, ("Two voices were heard, but which one is whose face is not settled: %s Who speaks is found from "
                      "mouths moving." % e)
    tones = {v: round(float(np.median([f for (p, f), (_, w) in zip(heard, voiced) if w == v]))) for v in people}
    return voices.speakers(heard, cut, people), (
        "Two voices were heard, one at about %d Hz and one at about %d Hz, and matched to the faces from pictures: "
        "person %d speaks in the lower voice, person %d in the higher one. Who says each line is taken from the voices."
        % (tones["low"], tones["high"], people["low"], people["high"]))


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
    if show == 0 and lay.get("seats"):
        raise Refused("In the speaker layout everyone is on screen all the time; show the person who should fill the "
                      "frame, 1 to %d." % count)
    shots = talk.with_shot(lay["shots"], round(start, 2), round(end, 2), show)
    new = talk.make(tuple(lay["size"]), lay["people"], shots, bool(lay.get("seats")))
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
SUBTITLES = {"English": ("english", "en", "eng", "อังกฤษ", "ภาษาอังกฤษ"), "Thai": ("thai", "th", "ไทย", "ภาษาไทย")}
SENTENCE_GAP = 0.7
SENTENCE_SHORT = 1.5
SENTENCE_LONG = 5.0
READS = 20
SCREEN_NAMES = 30
SUBTITLE_SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
                   "required": ["lines"]}
SUBTITLE_ASK = ("These are the spoken sentences of a short vertical clip (%s), in %s, heard by speech recognition (a "
                "word may be misheard), in the order they are said, each with how long it is on screen. Write the %s "
                "subtitle for each sentence, so a viewer who does not know %s follows the clip: natural spoken %s, "
                "short, the meaning and not word for word; keep every number the sentence says, and add none; keep "
                "names written in Latin letters exactly as they are%s; when a sentence is only a filler sound, write "
                "it short (\"Yeah.\", \"Huh?\"). One %s line for each sentence, in the same order.\n\n%s")
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
READ_SURE = 0.8


def _unsure(phrase):
    """True for a phrase the hearing was not sure of. Measured 10-27 on Thai FLEURS, clean and mixed with game
    sound (141 garbled phrases, 399 good): below 0.88 are 73% of the garbled ones and 4% of the good ones. A phrase
    whose fix took a name from the screen is sure: the screen wrote what the hearing could not.

    Two people talking in a room are heard less surely than FLEURS: in the talk clip of 10-28, 10 of 22 lines fell
    below UNSURE and the captions went quiet from 0:00 to 0:14, though Muse's fix had made most of them read right.
    So a Thai phrase from READ_SURE up, once fixed from the context and with no odd bits left, is sure enough. Of the
    179 unsure Thai phrases of every chat (10-28), 51 are such phrases and 46 of them read right; below 0.75 most
    were guesses, such as a number word made from a bit of a syllable."""
    sure = phrase.get("sure", 1.0)
    if sure >= UNSURE or phrase.get("named"):
        return False
    fixed = "heard" in phrase and phrase.get("kept") != LATER
    return not (fixed and sure >= READ_SURE and re.search("[ก-๛]", phrase["text"])
                and not words.odd_thai(phrase["text"]))


def _in_clip(pieces, p, length):
    """(from, to) in clip seconds of a phrase heard from p["from"] to p["to"] of the recording, or None when it starts
    where the clip has no picture. A phrase whose end was cut out ends with its piece, not at the clip's end."""
    a = project.from_recording(pieces, p["from"])
    if a is None:
        return None
    b = project.from_recording(pieces, p["to"])
    if b is None:
        at = 0.0
        for x, y in pieces:
            if x <= p["from"] <= y:
                b = at + y - x
                break
            at += y - x
    return round(a, 2), round(min(length, b), 2)


def _subtitles(name):
    """The language named for the captions, as SUBTITLES writes it, or None when none is named."""
    named = " ".join((name or "").lower().split())
    if not named:
        return None
    for language, names in SUBTITLES.items():
        if named in names:
            return language
    raise Refused("The studio writes captions in %s, the languages its fonts draw; %s is not one of them. Tell the "
                  "person so." % (" or ".join(SUBTITLES), name))


def _spoken(job, texts):
    """The language the texts are said in: Thai when they are written in Thai letters, else the recipe's guess."""
    if any(re.search("[ก-๛]", t) for t in texts):
        return "Thai"
    named = " ".join(((job.chat.clip or {}).get("language") or "").split())
    return next((k for k, v in SUBTITLES.items() if named.lower() in v), named or "English")


def _sentences(phrases):
    """The phrases joined into the sentences a subtitle is written for: a phrase joins the one just before it in what
    was said when the gap is at most SENTENCE_GAP s and the one before is shorter than SENTENCE_SHORT s or the two
    last at most SENTENCE_LONG s together. [{"from", "to", "text", "first"}], first = the index of its first phrase."""
    out = []
    for k, p in enumerate(phrases):
        last = out[-1] if out else None
        if last and p["n"] == phrases[k - 1]["n"] + 1 and p["from"] - last["to"] <= SENTENCE_GAP and (
                last["to"] - last["from"] < SENTENCE_SHORT or p["to"] - last["from"] <= SENTENCE_LONG):
            last["to"], last["text"] = p["to"], last["text"] + " " + p["text"]
        else:
            out.append({"from": p["from"], "to": p["to"], "text": p["text"], "first": k})
    return out


def _translated(job, rec, state, units, language, said_in):
    """Muse's subtitle in language for each unit, asked apart from the job, or None when twice the lines it wrote
    did not line up one for one with the units. When the clip's captions already hold each sentence in language,
    translated from the same words and keeping its numbers, they are kept: a tap on the page (Plain, Lit, Pop) never
    asks Muse, and Muse putting the same captions on lit is not asked to translate them again. A line that lost a
    number is asked again, so putting the captions on again is how Muse fixes the check that found it."""
    caps = (state["captions"] or {}).get("lines") or []
    held = {}
    if (state["captions"] or {}).get("language") == language:
        for k in {line["phrase"] for line in caps}:
            mine = [line for line in caps if line["phrase"] == k]
            held[(round(mine[0]["from"], 2), round(mine[-1]["to"], 2), mine[0].get("said"))] = \
                "".join(line["text"] for line in mine).strip()
    kept = [held.get((u["from"], u["to"], u["text"])) for u in units]
    if all(kept) and not any(_numbers_differ(u["text"], s) for u, s in zip(units, kept)):
        return kept
    kind = (job.chat.clip or {}).get("title") or "a video"
    names = [w for w in dict.fromkeys(_screen_words(job, rec, state)) if len(w) >= 4][:SCREEN_NAMES]
    listing = "\n".join("%d. (%.1f s) %s" % (k, u["to"] - u["from"], u["text"]) for k, u in enumerate(units, 1))
    shown = "; the screen writes these words this way: %s" % ", ".join(names) if names else ""
    ask = SUBTITLE_ASK % (kind, said_in, language, said_in, language, shown, language, listing)
    for _ in range(2):
        try:
            got = json.loads(job.ask_aside([{"role": "user", "content": ask}], SUBTITLE_SCHEMA,
                                           "Writing the %s subtitles" % language))["lines"]
        except mind.MindError as e:
            raise Refused("Muse writes the %s captions, and Muse could not be asked: %s" % (language, e))
        except (ValueError, KeyError, TypeError):
            continue
        if isinstance(got, list) and len(got) == len(units) and all(isinstance(s, str) and s.strip() for s in got):
            return [" ".join(s.split()) for s in got]
    return None


def _said_numbers(said):
    """[{the values one number may be}] for each number said, leaving out 1, which one language says as a number
    and another as "a". A Thai number word counts only where it stands as a word of its own, and one that ends in
    หนึ่ง after a unit is said for both readings. Eaglet 10-28: the check failed two right translations, "Okay, you can
    go check it in" for สามารถ... (สามารถ, can, starts with สาม, three) and "It's back to one hundred thousand now" for
    แสนหนึ่ง (said for 100,000 as often as for 100,001)."""
    from pythainlp.tokenize import word_tokenize
    out = [{v} for v in spoken.digits(said) | spoken.english(said)]
    if re.search("[ก-๛]", said):
        words.odd_thai(said)
        bounds, at = {0}, 0
        for w in word_tokenize(said, engine="newmm", keep_whitespace=True, custom_dict=words._WORDS[1]):
            at += len(w)
            bounds.add(at)
        for m in thai.NUMBER.finditer(said):
            if m.group(0)[:1].isdigit() or m.start() not in bounds or m.end() not in bounds:
                continue
            v = thai.value(m.group(0))
            out.append({v, v - 1} if m.group(0).endswith("หนึ่ง") and m.group(0) != "หนึ่ง" else {v})
    return [c for c in out if c != {1}]


def _numbers_differ(said, translated):
    """True when the translated line loses a number said or says one that was not."""
    heard, told = _said_numbers(said), spoken.values(translated) - {1}
    return any(not c & told for c in heard) or bool(told - set().union(*heard))


def add_captions(job, why, on=True, lit=False, pop=False, language=""):
    rec = _open(job, "caption")
    state = _clip_state(job)
    if lit and pop:
        raise Refused("Captions are lit or popped, not both: a popped word already moves as it is said. Choose one.")
    target = _subtitles(language or (job.chat.clip or {}).get("subtitles") or (state["captions"] or {}).get("language"))
    if language and job.chat.clip is not None:
        job.chat.clip["subtitles"] = target
    clip, length = state["clip"], state["length"]
    if not on:
        if not state["captions"]:
            raise Refused("The clip has no captions to take off.")
        job.edit({"do": "captions", "lines": [], "band": None})
        return "Captions are off."
    moments = job.chat.moments or []
    said = moments[clip["moment"] - 1].get("said") or [] if clip["moment"] <= len(moments) else []
    phrases, unsure, heard = [], [], []
    for n, p in enumerate(said):
        if _unsure(p):
            unsure.append(p)
            continue
        at = _in_clip(clip["pieces"], p, length)
        if at:
            phrases.append({"from": at[0], "to": at[1], "text": p["text"], "n": n})
            heard.append(p)
    if not phrases:
        raise Refused("Nothing was heard clearly in this moment, so there are no words to caption. Leave the captions "
                      "off, and say so in your answer.")
    translated, first = None, list(range(len(phrases)))
    try:
        texts, spelling = _spell(job, [p["text"] for p in phrases])
        for phrase, p, words_ in zip(phrases, heard, texts):
            phrase["text"] = p["text"] = words_
        said_in = _spoken(job, texts)
        if target and target != said_in:
            units = _sentences(phrases)
            translated = _translated(job, rec, state, units, target, said_in)
    except hearing.Stopped:
        return "No captions: the person pressed Stop."
    shown = [{"from": u["from"], "to": u["to"], "text": s} for u, s in zip(units, translated)] if translated else \
        phrases
    typed = _typed(state["captions"])
    for piece in shown:
        piece["text"] = typed.get((piece["from"], piece["to"]), piece["text"])
    lines = text.caption_lines(shown, CAPTION_LETTERS)
    for line in lines:
        if translated:
            line["said"] = units[line["phrase"]]["text"]
        if (shown[line["phrase"]]["from"], shown[line["phrase"]]["to"]) in typed:
            line["typed"] = True
    if translated:
        first = [u["first"] for u in units]
    band = clip["layout"]["captions"]
    entry = {"do": "captions", "lines": lines, "band": [band[1], band[3]], "lit": bool(lit), "pop": bool(pop)}
    if translated:
        entry["language"] = target
    bands, voiced = clip["layout"].get("bands"), clip.get("voices")
    if bands and voiced:
        who = {(round(v["from"], 2), round(v["to"], 2)): v["who"] for v in voiced}
        owners = [who.get((round(p["from"], 2), round(p["to"], 2))) for p in said if not _unsure(p)
                  and _in_clip(clip["pieces"], p, length)]
        for line in lines:
            k = first[line["phrase"]]
            line["who"] = owners[k] if k < len(owners) else None
        entry["bands"] = {n: [b[1], b[3]] for n, b in bands.items()}
    state = job.edit(entry)
    under = ("each under the face of the person who says it" if entry.get("bands") and any(
        line.get("who") for line in lines) else "in the band from y %d to %d" % (band[1], band[1] + band[3]))
    out = ["Captions on: %d lines %s, %s; %s."
           % (len(lines), "in %s, translated from the %d sentences said in %s, numbers kept" % (
               target, len(units), said_in) if translated else "of what was said, numbers exactly as heard", under,
              "each word lights up as it is said" if lit else
              "each word pops in at its place as it is said and stays, so once said the line looks plain" if pop else
              "plain words, none lit, so the eye stays on the picture")]
    if spelling:
        out.append(spelling)
    if translated:
        out.append("The clip is for viewers who read %s: write the title, the words on screen and the post in %s "
                   "too." % (target, target))
        fast = max(((len(s) / max(0.1, u["to"] - u["from"]), u, s) for u, s in zip(units, translated)),
                   key=lambda f: f[0])
        if fast[0] > READS:
            out.append("The fastest line is read at %d letters a second (%s \"%s\"); most viewers read up to %d."
                       % (round(fast[0]), recording.clock(fast[1]["from"]), fast[2], READS))
    elif target and target != said_in:
        out.append("The %s lines Muse wrote did not line up one for one with the sentences said, twice, so the "
                   "captions are in %s as said. Tell the person, or put the captions on again." % (target, said_in))
    if any(line.get("typed") for line in lines):
        out.append("The words the person typed on the page for %d of the lines are kept as they typed them."
                   % sum(1 for line in lines if line.get("typed")))
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


def _typed(captions):
    """{(from, to): the words} of each caption phrase the person typed on the page, by the phrase's time in the clip:
    captions put on again keep the person's words."""
    lines = (captions or {}).get("lines") or []
    out = {}
    for k in {line["phrase"] for line in lines if line.get("typed")}:
        mine = [line for line in lines if line["phrase"] == k]
        out[(round(mine[0]["from"], 2), round(mine[-1]["to"], 2))] = "".join(line["text"] for line in mine)
    return out


def _on_camera(rec, clip):
    """Where a face sits still in the clip's pieces, as words for Muse, or None. Only faces already found are used."""
    for _, _, found in _found_faces(rec, clip["pieces"], clip.get("span")):
        g = faces.steady(found) if found else None
        if g:
            return "a face in the same place in %d of %d pictures" % (g["seen"], g["of"])
    return None


PEOPLE_SHOWN = 8
PEOPLE_LISTED = 12
CROWD_SHOWN = 4
SAME_FACE = 0.3


def _people(rec, pieces, face, span=None):
    """The other people's faces in the clip, from the faces already found in each piece, most seen first: [{"n",
    "seen", "of", "from", "to" (clip seconds), "box" (recording), "points": [[clip second, x, y, w, h]], "crop"}].
    The face the layout shows, and a face that sits still (the person making the clip), are not among them. None
    when the faces of a piece were not found."""
    out = []
    for a, b, found in _found_faces(rec, pieces, span):
        if not found:
            return None
        for tr in tracks.tracks(found, rec):
            if tr["steady"] or face and _overlap(tr["box"], face) > SAME_FACE:
                continue
            points = []
            for t, x, y, w, h in tr["points"]:
                at = project.from_recording(pieces, min(max(t, a), b))
                if at is not None and (not points or at > points[-1][0]):
                    points.append([round(at, 3), x, y, w, h])
            if not points:
                continue
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
    listed = sorted(people, key=lambda p: (-p["seen"], p["n"]))[:PEOPLE_LISTED]
    rest = [p for p in people if p not in listed]
    if rest:
        twice = sum(1 for p in rest if p["seen"] > 1)
        lines.append("A crowd: the %d seen most are listed; the other %d (%s) are decided together with others "
                     "(blur or keep)." % (len(listed), len(rest), "%d seen in two pictures or more, %d seen once" % (
                         twice, len(rest) - twice) if twice else "each seen once, for half a second"))
    pictured = {p["n"] for p in listed[:CROWD_SHOWN]} if rest else {p["n"] for p in people if p["n"] <= PEOPLE_SHOWN}
    for p in sorted(listed, key=lambda p: p["n"]):
        shown = p["n"] in pictured and p.get("crop")
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


def decide_people(job, why, blur=None, keep=None, others=None):
    rec = _open(job, "blur")
    state = _clip_state(job)
    people = {p["n"]: p for p in state["people"]}
    if not people:
        raise Refused("The clip has no other people's faces to decide about.")
    blur, keep = sorted(set(blur or [])), sorted(set(keep or []))
    if others:
        rest = sorted(set(people) - set(blur) - set(keep))
        blur, keep = (sorted(blur + rest), keep) if others == "blur" else (blur, sorted(keep + rest))
    if not blur and not keep:
        raise Refused("Name at least one person to blur or to keep.")
    both = sorted(set(blur) & set(keep))
    if both:
        raise Refused("Person %s is in both lists: each person is either blurred or kept." % _numbers(both))
    unknown = [n for n in blur + keep if n not in people]
    if unknown:
        raise Refused("There is no person %s; the clip's people are %s." % (_numbers(unknown), _numbers(people)))
    state = job.edit({"do": "people", "blur": blur, "keep": keep})
    if len(blur) > PEOPLE_LISTED:
        lines = ["%d people are blurred where their faces show, and each blur moves with its face." % len(blur)]
    else:
        lines = ["Person %d is blurred where their face shows, %s-%s of the clip, and the blur moves with the face."
                 % (n, recording.clock(people[n]["from"]), recording.clock(people[n]["to"])) for n in blur]
    if len(keep) > PEOPLE_LISTED:
        lines.append("Kept as they are: %d people." % len(keep))
    elif keep:
        lines.append("Kept as they are: person %s." % _numbers(keep))
    left = _undecided(state)
    lines.append(("Still to decide before you render: %s." % _people_named(left)) if left else
                 "Every person seen in two pictures or more is decided.")
    if blur:
        p = people[blur[0]]
        lines.append(_show_frame(job, state, rec, (p["from"] + p["to"]) / 2,
                                 "The clip with person %d blurred:" % p["n"]))
    return "\n".join(lines)


NEAR = 0.5


def show_people(job, why, at):
    rec = _open(job, "look at the people in")
    state = _clip_state(job)
    length = state["length"]
    if not 0 <= at <= length:
        raise Refused("at %g s is outside the clip, which is %g s long: give the second in seconds of the clip, from 0 "
                      "to %g." % (at, length, round(length, 1)))
    seen = []
    for p in state["people"]:
        point = min(p["points"], key=lambda q: abs(q[0] - at))
        if abs(point[0] - at) <= NEAR:
            seen.append({"n": p["n"], "box": point[1:], "blur": p["blur"]})
    if not seen:
        near = sorted(state["people"], key=lambda p: min(abs(q[0] - at) for q in p["points"]))[:3]
        raise Refused("No face the program found shows at %g s.%s" % (at, " The nearest: %s." % "; ".join(
            "person %d at %s-%s" % (p["n"], recording.clock(p["from"]), recording.clock(p["to"])) for p in near)
            if near else " The clip has no other people's faces."))
    where = project.to_clip(state["clip"]["pieces"], at)
    path = _people_picture(rec, where[1] if where else at, seen, "%s-%d" % (job.chat.id, int(at * 100)))
    if not path:
        return "The picture at %g s could not be drawn, so the faces cannot be shown." % at
    job.attach(path, "The picture at %g s of the clip, each face the program found in a numbered box:" % round(at, 2))
    lines = ["%d face%s at %g s, numbered in the picture that comes next:" % (len(seen), "" if len(seen) == 1 else "s",
                                                                               round(at, 2))]
    for s in sorted(seen, key=lambda s: s["n"]):
        p = next(q for q in state["people"] if q["n"] == s["n"])
        lines.append("- person %d: %s-%s of the clip, %s" % (s["n"], recording.clock(p["from"]), recording.clock(p["to"]),
                                                              "blurred" if s["blur"] else "kept" if s["blur"] is False
                                                              else "not decided yet"))
    lines.append("Match the person's words (what they wear, where they stand) to a number, then blur or keep that "
                 "number with decide_people. A person seen in other seconds may have another number there.")
    return "\n".join(lines)


def faces_at(state, t):
    """The other people's faces at clip second t, where the clip's frame shows them: [{"n", "box" (frame pixels),
    "blur"}], for the page, where the person taps a face to blur it."""
    if not state["clip"]:
        return []
    out = []
    for p in state["people"]:
        point = min(p["points"], key=lambda q: abs(q[0] - t))
        shown = layouts.to_frame(state["clip"]["layout"], point[1:], t) if abs(point[0] - t) <= NEAR else []
        if shown:
            out.append({"n": p["n"], "box": [round(v) for v in shown[0]], "blur": p["blur"]})
    return out


def _people_picture(rec, at, seen, name):
    """The recording's picture at `at` with a numbered box around each face in seen ({"n", "box"} in recording
    pixels), 896 px on its long side; the jpg's path in work/people, or None."""
    import cv2
    folder = os.path.join(recording.WORK, "people")
    os.makedirs(folder, exist_ok=True)
    raw = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-ss", "%.3f" % at, "-i", rec["path"],
                          "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True,
                         timeout=60).stdout
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
    if img is None:
        return None
    k = 896.0 / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    for s in seen:
        x, y, w, h = [int(v * k) for v in s["box"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 220, 255), 2)
        label = str(s["n"])
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        top = max(th + 8, y - 4)
        cv2.rectangle(img, (x, top - th - 8), (x + tw + 10, top + base - 2), (16, 16, 16), -1)
        cv2.putText(img, label, (x + 5, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    path = os.path.join(folder, name + ".jpg")
    ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    data.tofile(path)
    return path


def _people_named(ns):
    """"person 3" or "person 2 and 5", or for a crowd only how many, and that others decides them together."""
    if len(ns) > PEOPLE_LISTED:
        return "%d people (others: blur or keep decides everyone not named)" % len(ns)
    return "person %s" % _numbers(ns)


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
        shown = _shown_private(job, rec, state)[0] if (job.chat.clip or {}).get("recipe") == "game" else None
        names = [f for f in shown or [] if f["kind"] == NAME_LINE]
        raise Refused("That box covers the face at %s on your grid, the face the clip shows, and the face is the point "
                      "of the clip. Give a box around the chat or the names only, leaving the face out.%s" % (
                          _grid(face, rec), " The viewers' names were read on the screen at %s on your grid." %
                          _grid(_around(names), rec) if names else ""))
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
        raise Refused("Decide about %s first, with decide_people: blur or keep. Nothing undecided is rendered."
                      % _people_named(left))
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
SILENCE_FLOOR = "-50dB"
OPENING = "It does not start black or silent"
OPENING_LOOK = 8
OPEN_AIR = 0.2
SOUND_HOLD = 0.25
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
    """(black picture at the start, silent sound at the start), each True when it lasts FIRST s from 0. A stretch
    that starts later is not the start: read as text, "silence_start: 0.951771" once passed for one at 0 (10-28),
    and two clips that opened on words were not handed over."""
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-nostats", "-t", "1.5", "-i", path, "-vf",
           "blackdetect=d=%g:pix_th=0.10" % FIRST]
    if sound:
        cmd += ["-af", "silencedetect=noise=%s:d=%g" % (SILENCE_FLOOR, FIRST)]
    log = subprocess.run(cmd + ["-f", "null", "-"], capture_output=True, text=True, encoding="utf-8",
                         errors="replace").stderr
    black = any(float(v) <= 0.05 for v in re.findall(r"black_start:(-?[\d.]+)", log))
    silent = any(float(v) <= 0.05 for v in re.findall(r"silence_start: (-?[\d.]+)", log))
    return black, silent


def _first_sound(path):
    """The second at which the file's sound first rises above SILENCE_FLOOR and stays up for SOUND_HOLD s, as the
    opening check hears it: 0 when it does from the start, None when it does not within OPENING_LOOK s. A shorter
    sound, such as a click or a breath, is not a start: measured 10-28, one 2.3 s into a screen recording's silence
    moved a clip there, and it still opened on 0.5 s of silence."""
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-nostats", "-t", str(OPENING_LOOK), "-i", path, "-vn",
           "-af", "silencedetect=noise=%s:d=0.05" % SILENCE_FLOOR, "-f", "null", "-"]
    log = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    starts = [float(s) for s in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(s) for s in re.findall(r"silence_end: ([\d.]+)", log)]
    if not starts or starts[0] > 0.05:
        return 0.0
    for k, end in enumerate(ends):
        if (starts[k + 1] if k + 1 < len(starts) else OPENING_LOOK) - end >= SOUND_HOLD:
            return end
    return None


def _sound_at(state, path):
    """(seconds into the clip, second in the recording) of the finished file's first sound, or None when it is not
    in the clip's first piece."""
    first, piece = _first_sound(path), state["clip"]["pieces"][0]
    if not first or first >= piece[1] - piece[0]:
        return None
    return first, piece[0] + first


def _open_on_sound(job, moment):
    """Moves the moment's start to OPEN_AIR s before the finished clip's first sound, when that sound comes within its
    first piece and leaves FIRST s or more after it, and returns the seconds it moved; else None. Measured 10-28 on a
    screen recording: two clips opened on 0.5 s of silence or more, so they failed the opening check, and Muse looked
    again, took back a title of another clip, made both again as they were, and ran out of memory."""
    state = _clip_state(job)
    found = _sound_at(state, os.path.join(render.folder("clips"), job.chat.finished()["name"]))
    if not found or state["length"] - found[0] < FIRST:
        return None
    m = job.chat.moments[moment - 1]
    m["start"] = round(found[1] - OPEN_AIR, 2)
    return round(found[0] - OPEN_AIR, 1)


def _still_seen(state, clip_file):
    """[(person, clip second)] where a blurred person's face can still be found in the finished file."""
    out, lay = [], state["clip"]["layout"]
    for p in state["people"]:
        if not p["blur"]:
            continue
        points = p["points"]
        for point in (points[len(points) // 4], points[len(points) // 2], points[3 * len(points) // 4]):
            t, box = point[0], point[1:]
            shown = layouts.to_frame(lay, box, t)
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
    unsaid = [line["text"] for line in lines if line.get("said", line["text"]).replace(" ", "") not in joined]
    typed = {line["text"] for line in lines if line.get("typed")}
    unsaid = [t for t in unsaid if t not in typed]
    check("Captions are words that were said", not unsaid if lines else None,
          ("%d of %d lines are in what was said%s" % (len(lines) - len(unsaid), len(lines), " or are the person's "
                                                       "own words, typed on the page" if typed else "") +
           ("; not said: %s" % "; ".join(unsaid[:3]) if unsaid else "")) if lines else "no captions")
    sentences = {}
    for line in lines:
        if "said" in line and not line.get("typed"):
            sentences.setdefault(line["phrase"], [line["said"], ""])[1] += line["text"]
    if sentences:
        off = ['"%s" for "%s"' % (t.strip(), s) for s, t in sentences.values()
               if _numbers_differ(s, t)]
        check("Translated captions keep the numbers", not off,
              "%d of %d sentences keep every number said" % (len(sentences) - len(off), len(sentences)) +
              ("; not kept: %s; put the captions on again and those lines are translated again"
               % "; ".join(off[:3]) if off else ""))
    if kind == "seller":
        wrong = [n for t in state["titles"] for n in _not_said(t["text"], said)]
        unread = _numbers_unread(job, said) if wrong else None
        check("Numbers on screen were said", None if unread else not wrong,
              unread or ("not said: %s" % ", ".join(wrong) if wrong else "every number on screen was said"),
              "Muse" if unread else "program")
    else:
        check("Numbers on screen were said", None, "numbers can come from what the screen shows")
    if kind == "game":
        check("No viewer's name can be read", *_names_check(job, rec, state))
    else:
        check("No viewer's name can be read", None, "a handle on the screen is checked with the private lines"
              if _reads_screen(job) else "no chat box in this kind of clip")
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
    heard = _sound_at(state, path) if silent and not black else None
    check(OPENING, not (black or silent),
          "picture and sound from the first %g s" % FIRST if not (black or silent) else
          " and ".join(w for w, bad in (("black", black), ("silent", silent)) if bad) + " for the first %g s" % FIRST +
          ("; the first sound is %g s in, at %g s in the recording: propose the moment again from %g s and make "
           "the clip again" % (round(heard[0], 1), round(heard[1], 1), round(heard[1] - OPEN_AIR, 1)) if heard
           else ""))
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
    shown = 0
    for k, share in enumerate(REVIEWED, 1):
        name = recording.frame(made, made["seconds"] * share)
        if name:
            shown += 1
            job.attach(os.path.join(recording.WORK, "frames", name), "%s %d of %d:" % (FINISHED, k, len(REVIEWED)))
    if shown:
        out.append(LOOK_AT_IT % (shown, " or the product" if kind == "seller" else "",
                                 "; a viewer's name, a phone number or a face in the chat that can be read" if
                                 kind == "game" else ""))
    if failed:
        out.append("Fix what failed, render again and check again, then hand it over.")
    else:
        out.append("Hand it over next.")
    return "\n".join(out)


REVIEWED = (0.12, 0.37, 0.62, 0.87)
FINISHED = "The finished clip, picture"
LOOK_AT_IT = ("%d pictures of the finished clip come next, as a viewer sees it. Before you hand it over, look at each "
              "for: a face blurred that should show, or one the person asked to blur that shows; words over a "
              "face%s; words cut off at an edge or too small to read; a head or the action cut off by the frame%s. "
              "Say in your next why what you saw. If something is wrong, fix it, render again and check again.")

_odd_thai = words.odd_thai


def _spell(job, texts):
    """(texts with the Thai spelling fixed in a short conversation of Muse's own, a sentence naming each fix or "").
    When Muse cannot be asked, the texts stay as written. Stop raises hearing.Stopped."""
    try:
        fixed, changed = words.spelled(lambda m, s: job.ask_aside(m, s, "Checking the Thai spelling"), texts,
                                       every=True, kept=job.proofread)
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
    tags = list(dict.fromkeys(t for t in (_tag(w) for w in list(hashtags) + more + also) if t))
    try:
        (title, caption, *tags), spelling = _spell(job, [title, caption] + tags)
    except hearing.Stopped:
        return "Not handed over: the person pressed Stop."
    tags = list(dict.fromkeys(t for t in (_tag(w) for w in tags) if t))
    if not tags:
        raise Refused("Give at least one hashtag, such as #gaming.")
    ready = {"file": done["name"], "length": done["length"], "title": title, "caption": caption, "hashtags": tags,
             "english": english, "checks": checks}
    job.emit("ready", **ready)
    job.show(ready=ready)
    return ("Handed over: the person sees the finished clip with its title, the post text and %d hashtags, a Copy all "
            "button and a download. Tell them in one or two short lines; do not repeat the text, it is on the page."
            % len(tags)) + (" " + spelling if spelling else "")


LAYOUT_OF = {"explain": "card", "game": "card", "talk": "talk", "seller": "subject"}
BATCH_WHY = "one of the clips the person asked for"


def make_clips(job, why, clips, blur=None):
    rec = _open(job, "make clips from")
    moments = job.chat.moments or []
    if not moments:
        raise Refused("No moments are proposed yet. Look at the recording and propose moments first: each clip is made "
                      "from one of them.")
    numbers = [c["moment"] for c in clips]
    wrong = [n for n in numbers if not 1 <= n <= len(moments)]
    if wrong:
        raise Refused("There are %d moments, so %s %s not one of them." % (
            len(moments), _numbers(wrong), "is" if len(wrong) == 1 else "are"))
    if len(set(numbers)) < len(numbers):
        raise Refused("Each moment makes one clip; moment %s is given twice." % _numbers(
            sorted({n for n in numbers if numbers.count(n) > 1})))
    spelled = []
    for c in clips:
        m = moments[c["moment"] - 1]
        shown = _screen_words(job, rec, {"clip": {"pieces": [[m["start"], m["end"]]]}})
        wrong = _misspelled(" ".join([c["title"], c.get("words") or "", c["caption"]] + list(c["hashtags"])), shown)
        if wrong:
            for key in ("title", "words", "caption"):
                if c.get(key):
                    c[key] = _as_on_screen(c[key], wrong)
            c["hashtags"] = [_as_on_screen(h, wrong) for h in c["hashtags"]]
            spelled.append("clip of moment %d: %s" % (c["moment"], ", ".join('"%s" as "%s"' % w for w in wrong)))
    recipe = (job.chat.clip or {}).get("recipe")
    if recipe == "game" and not blur:
        raise Refused("Nothing was made. A game stream's clips show its chat box with the viewers' names: give the "
                      "chat box in blur, on your 0-1000 grid of the pictures from your looks, and every clip hides it. "
                      "Leave the face camera out of the box.")
    pictures, lines, first, looked = len(job.pictures), [], None, []
    for k, c in enumerate(clips, 1):
        if job.stopped:
            lines.append("The person pressed Stop before clip %d; it and the clips after it were not made." % k)
            break
        job.working("Making clip %d of %d, from moment %d" % (k, len(clips), c["moment"]))
        job.chat.project = project.Project()
        lines.append(_one_clip(job, rec, k, c, c.get("layout") or LAYOUT_OF.get(recipe, "subject"), blur or []))
        seen = [p for p in job.pictures[pictures:] if p[1].startswith(FINISHED)][-len(REVIEWED):]
        del job.pictures[pictures:]
        looked += [(path, "Clip %d: %s" % (k, caption)) for path, caption in seen]
        if first is None and job.chat.project.entries:
            first = list(job.chat.project.entries)
    job.pictures += looked
    if first:
        job.chat.project = project.Project(first)
        job.show(clip=job.chat.clip_view())
    done = sum(1 for line in lines if " handed over" in line and "not handed over" not in line)
    return "\n".join(["Made %d clip%s, %d handed over; each is reported on its own:" % (
        len(lines), "" if len(lines) == 1 else "s", done)] + lines + ([
        "The program wrote names as the screen shows them: %s." % "; ".join(spelled)] if spelled else []) + [
        "Each clip's finished file comes next in %d pictures, as a viewer sees it: look at them as check_clip asks "
        "and say what you saw. When one shows something wrong, make that clip again with make_clip and fix it." %
        len(REVIEWED),
        "The person sees a ready card for each clip handed over. The clip on the page now is the first one: change it "
        "with your tools, and tell the person in one or two lines which clips are ready and what failed in the "
        "others."])


def _title_where_clear(job, rec, words):
    """Puts the title up in the earliest stretch of a layout place that covers no face, the longer first when two
    start together, and says where and when; None when no stretch takes it. Seen 10-28 on a Q&A where both talkers
    fill the frame in turn: two of three clips went out with no title, as no place was clear for the whole clip."""
    state = _clip_state(job)
    lay, length = state["clip"]["layout"], state["length"]
    times, every = _face_times(state, rec)
    stretches = sorted((a, a - b, name) for name, box in lay["places"].items()
                       for a, b in _clear(_covered(box, times, every, 0, length), 0, length))
    for a, minus, name in stretches:
        start, end = math.ceil(a * 10) / 10, math.floor((a - minus) * 10) / 10
        try:
            add_title(job, BATCH_WHY, words, start, end, place=name)
        except Refused:
            continue
        return "at the %s from %g to %g s" % (name, start, end)
    return None


def _one_clip(job, rec, k, c, layout, blur, opened=None):
    """One clip of make_clips, from its moment to the hand-over: the sentence reporting on it. Its words are
    proofread together first, so the title on the clip is spelled as the post text is. Each box in blur is
    blurred for the whole clip. When the only check that fails is the silent opening, the moment is moved to its
    first sound (_open_on_sound) and the clip is made once more; opened is the seconds it moved. Seen 10-27 on a 4-hour game stream: six clips were made without the chat box blurred,
    none passed the check for viewers' names, and Muse ran out of room fixing the first one by hand."""
    n, head = c["moment"], "Clip %d (moment %d)" % (k, c["moment"])
    try:
        _spell(job, [c["title"], c.get("words") or "", c["caption"]] + list(c["hashtags"]))
        made = make_clip(job, BATCH_WHY, n, layout)
        if made.startswith("Not made"):
            return "%s: not made, the person pressed Stop." % head
        state = _clip_state(job)
        head = "Clip %d (moment %d, %s-%s, %g s, %s layout)" % (k, n, recording.clock(state["clip"]["pieces"][0][0]),
                                                               recording.clock(state["clip"]["pieces"][-1][1]),
                                                               state["length"], state["clip"]["layout"]["kind"])
        notes, others = [], _undecided(state)
        if opened:
            notes.append("it opened on silence, so the program started it %g s later, %g s before its first sound"
                         % (opened, OPEN_AIR))
        for box in blur:
            try:
                blur_box(job, BATCH_WHY, box)
            except Refused as e:
                notes.append("the box %s was not blurred: %s" % (box, e))
        if others:
            decide_people(job, BATCH_WHY, keep=others)
            notes.append("kept %d other face%s as %s" % (len(others), "" if len(others) == 1 else "s",
                                                          "they are" if len(others) > 1 else "it is"))
        try:
            add_captions(job, BATCH_WHY)
        except Refused as e:
            notes.append("no captions: %s" % e)
        words, why_not = c.get("words") or c["title"], ""
        for place in ("top", "middle"):
            try:
                add_title(job, BATCH_WHY, words, 0, _clip_state(job)["length"], place=place)
                break
            except Refused as e:
                why_not = str(e)
        else:
            placed = _title_where_clear(job, rec, words)
            notes.append("the title is on the clip only %s, where it covers no face" % placed if placed else
                         "no title on the clip: %s" % why_not)
        made = render_clip(job, BATCH_WHY)
        if made.startswith("Not rendered"):
            return "%s: not rendered, the person pressed Stop." % head
        check_clip(job, BATCH_WHY)
        done = job.chat.finished()
        checks = job.chat.checks.get(done["name"]) or []
        failed = [ch["name"] for ch in checks if ch["ok"] is False]
        later = _open_on_sound(job, n) if failed == [OPENING] and opened is None else None
        if later:
            return _one_clip(job, rec, k, c, layout, blur, opened=later)
        if failed:
            return "%s: rendered but not handed over, these checks failed: %s%s." % (
                head, "; ".join(failed), "; " + "; ".join(notes) if notes else "")
        handed = hand_over(job, BATCH_WHY, c["title"], c["caption"], c["hashtags"], c.get("english") or "")
        if handed.startswith("Not handed"):
            return "%s: not handed over, the person pressed Stop." % head
    except Refused as e:
        return "%s: not handed over: %s" % (head, e)
    due = [ch for ch in checks if ch["ok"] is not None]
    return "%s: handed over, %d of %d checks passed%s." % (head, len(due), len(due),
                                                           "; " + "; ".join(notes) if notes else "")


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
        return ("the lit captions" if entry.get("lit") else "the popped captions" if entry.get("pop") else
                "the captions") if entry["lines"] else \
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


def load_recipe(job, why, type, length_s=None, style=None, language="", brief="", subtitles=""):
    subtitles = _subtitles(subtitles)
    recipe = RECIPES.get(type.lower().strip())
    if recipe is None:
        raise Refused("There is no recipe for %s yet. The recipes are: %s. Do not use a recipe made for another "
                      "kind of video: tell the person a clip of this kind is not ready yet, and what the studio can "
                      "make now." % (type, _recipe_list()))
    if recipe["name"] == MONTAGE and not job.chat.clips:
        raise Refused("The %s recipe is for several clips opened together with open_clips; this chat has %s. Pick "
                      "the recipe for the kind of recording that is open." % (
                          MONTAGE, "one recording open" if job.chat.recording else "no clips open"))
    clip = {"recipe": recipe["name"], "title": recipe["title"], "length_s": length_s or recipe["length"],
            "style": style or recipe["style"], "language": language, "brief": brief, "subtitles": subtitles}
    had = job.chat.clip
    job.set_clip(clip, chips(clip))
    lines = ["Recipe loaded: %s (%s)." % (recipe["title"], recipe["name"]),
             "Your guesses, shown to the person with buttons to change them: about %d s, style %s%s%s."
             % (clip["length_s"], clip["style"], ", language " + language if language else "",
                ", captions in " + subtitles if subtitles else ""),
             "The person's own words win over this recipe and over your guesses. Do not ask the person to confirm "
             "them: go on with the work. The buttons let them change any guess at any time."]
    if had and had["recipe"] != recipe["name"]:
        lines.append("This recipe replaces the %s recipe read earlier in this chat; follow only this one." %
                     had["recipe"])
    return "\n".join(lines) + "\n\n" + recipe["body"]


def chips(clip):
    """The buttons under Muse's guesses. Each one sends a short message, as if the person had typed it."""
    out = [{"label": r["chip"], "say": "(tapped) %s." % r["chip"]} for r in RECIPES.values()
           if r["name"] not in (clip["recipe"], MONTAGE)]
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
         "list to files whose name, folder or recorded day holds every word, in upper or lower case. It only reads "
         "the folder lists, never the files.",
         {"words": {"type": "string", "maxLength": 200,
                    "description": "Words that must all be in the file name, its folders or the day it was recorded, "
                                   "such as a game, or a day written 2026-10-12, 12 Oct or 12 ตุลาคม. Leave empty for "
                                   "the newest recordings."},
          "newest": {"type": "integer", "minimum": 1, "maximum": 20,
                     "description": "How many to list, newest first (1 to 20, default 5)."}},
         [], find_recordings),
    Tool("open_recording", "Opened the video",
         "Open one recording to make the clip from: its full path, as the list of recordings gives it. It reads "
         "the length, picture size and sound tracks, and the page shows the recording. One recording is open at a "
         "time; opening another replaces it.",
         {"path": {"type": "string", "maxLength": 1000, "description": "The recording's full path."}},
         ["path"], open_recording, changes=True),
    Tool("open_clips", "Opened the clips",
         "Open every recording in one folder together, to make one clip from several of them: a trip, an outing or "
         "an event shot as many short clips. The program numbers them in the order they were shot, hears each, and "
         "sends you a picture of each with frames from its start, middle and end.",
         {"folder": {"type": "string", "maxLength": 1000, "description": "The folder's full path."}},
         ["folder"], open_clips, changes=True),
    Tool("arrange_clips", "Put the clips in order",
         "Say which of the open clips go into the clip, in what order, which seconds of each, the words on screen "
         "over each, and why; and why each clip left out is left out. The program joins the pieces into one "
         "recording in that order, makes the clip from it and puts the words on. A piece is at most %g s: the "
         "program keeps that much of a longer one from its start, skips where the picture holds still, and cuts "
         "the end of every piece by the same share when together they are longer than asked. Call it again to "
         "change anything." % LONGEST_PIECE,
         {"story": {"type": "string", "maxLength": 300,
                    "description": "One line for the person, in their language: what the clip tells, from its first "
                                   "picture to its last."},
          "clips": {"type": "array", "minItems": 1, "maxItems": MOST_CLIPS,
                    "description": "The clips that go in, in the order they are shown.",
                    "items": {"type": "object", "properties": {
                        "clip": {"type": "integer", "minimum": 1, "maximum": MOST_CLIPS,
                                 "description": "The clip's number."},
                        "start": {"type": "number", "minimum": 0,
                                  "description": "The second of the clip its piece starts at. Leave it out for its "
                                                 "start."},
                        "end": {"type": "number", "minimum": 0,
                                "description": "The second of the clip its piece ends at. Leave it out for its end."},
                        "words": {"type": "string", "maxLength": 80,
                                  "description": "A short line on screen from this piece until the next piece with "
                                                 "words, in the person's language. Leave it out where the picture "
                                                 "says it all."},
                        "why": {"type": "string", "maxLength": 200,
                                "description": "One line for the person, in their language: why this piece goes "
                                               "here."}},
                              "required": ["clip", "why"]}},
          "left_out": {"type": "array", "maxItems": MOST_CLIPS,
                       "description": "Every clip that does not go in, each with why.",
                       "items": {"type": "object", "properties": {
                           "clip": {"type": "integer", "minimum": 1, "maximum": MOST_CLIPS,
                                    "description": "The clip's number."},
                           "why": {"type": "string", "maxLength": 200,
                                   "description": "One line for the person, in their language: why it is left out."}},
                                 "required": ["clip", "why"]}}},
         ["story", "clips"], arrange_clips, changes=True),
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
          "subtitles": {"type": "string", "maxLength": 40,
                        "description": "Only when the person wants the captions in another language than the one "
                                       "spoken, such as English subtitles for a clip said in Thai: that language. "
                                       "Leave it out for captions of the words as said."},
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
         "draws: " + "; ".join("%s: %s" % kv for kv in dict(layouts.SAID, talk=talk.SAID, speaker=talk.SPEAKER_SAID,
                                                          split=talk.SPLIT_SAID).items()) + ". The program "
         "finds the face itself "
         "in the moment's pictures and fixes what it safely can, and says what it did. The clip's frame comes back "
         "to you as a picture. Call it again to remake the clip from another moment, layout or crop; the words on "
         "screen stay.",
         {"moment": {"type": "integer", "minimum": 1, "maximum": MOST_MOMENTS,
                     "description": "The number of the moment you proposed."},
          "layout": {"type": "string", "enum": list(layouts.KINDS) + ["talk", "speaker", "split"],
                     "description": "Which layout."},
          "follow": {"type": "boolean",
                     "description": "For a card of a screen: true (the default) lets the program zoom the card to "
                                    "where the screen changes and move with it, so its words can be read on a "
                                    "phone; false shows the whole screen."},
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
         "are plain words unless you set lit or pop. Set on to false to take them off.",
         {"on": {"type": "boolean", "description": "true to put captions on (the default), false to take them off."},
          "lit": {"type": "boolean",
                  "description": "true lights up each word as it is said, like karaoke. Use it only when nobody on "
                                 "screen is talking: a voice-over or the words of a song over the picture. When a "
                                 "person on camera explains something, the lit word pulls the eye away from them, "
                                 "so leave it out, unless the person asked for lit words."},
          "pop": {"type": "boolean",
                  "description": "true pops each word in at its own place as it is said, and the line stays once "
                                 "said, so nothing is lit and the eye can go back to the person. Use it when the "
                                 "person asks for words that pop or appear as they are said. Not with lit."},
          "language": {"type": "string", "maxLength": 40,
                       "description": "The language to write the captions in, such as English, when the person wants "
                                      "subtitles in another language than the one spoken: the program has each "
                                      "sentence translated and keeps its numbers. Leave it out to keep what the "
                                      "recipe was told."}},
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
         "face that belongs, such as a game character, or a face on a screen or a box. At a trip or an event every "
         "face starts kept: blur one only when the person asks. Every person seen in two pictures or more must be "
         "decided before the clip is rendered; one call can decide them all.",
         {"blur": {"type": "array", "items": {"type": "integer", "minimum": 1},
                   "description": "The numbers of the people to blur, such as [2]."},
          "keep": {"type": "array", "items": {"type": "integer", "minimum": 1},
                   "description": "The numbers of the people to keep as they are."},
          "others": {"type": "string", "enum": ["blur", "keep"],
                     "description": "What happens to everyone not named in blur or keep, such as a crowd."}},
         [], decide_people, changes=True),
    Tool("show_people", "Looked at the people",
         "Shows the clip's picture at one second with a numbered box on every face the program found then. Use it "
         "when the person names someone by what they look like or where they stand (the man in white at 12 s), to "
         "find that person's number for decide_people.",
         {"at": {"type": "number", "minimum": 0, "description": "The second of the clip, such as 12."}},
         ["at"], show_people),
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
    Tool("make_clips", "Made the clips",
         "Make a finished clip from each of several moments you proposed, in one go, when the person asks for more "
         "than one clip or for every good moment. For each, the program makes the clip in the recipe's layout (or the "
         "one you give), hides what is private, puts the captions and your title on, renders it, checks it and hands "
         "it over; then it reports on each clip, handed over or why not. Give each clip its title and post text as at "
         "hand-over. Names must be spelled as the screen shows them.",
         {"clips": {"type": "array", "minItems": 2, "maxItems": MOST_MOMENTS, "items": {
             "type": "object", "required": ["moment", "title", "caption", "hashtags"], "properties": {
                 "moment": {"type": "integer", "minimum": 1, "maximum": MOST_MOMENTS,
                            "description": "The moment's number, as you proposed it."},
                 "layout": {"type": "string", "enum": list(layouts.KINDS) + ["talk", "speaker", "split"],
                            "description": "Leave it out for the recipe's layout."},
                 "title": {"type": "string", "maxLength": 100,
                           "description": "The post's title, in the clip's language; also the words on the clip "
                                          "unless you give words."},
                 "words": {"type": "string", "maxLength": 60,
                           "description": "Shorter words for the title on the clip, when the post's title is long."},
                 "caption": {"type": "string", "maxLength": 600,
                             "description": "The post text, a few short lines in the clip's language."},
                 "hashtags": {"type": "array", "minItems": 1, "maxItems": MOST_TAGS, "items": {"type": "string"}},
                 "english": {"type": "string", "maxLength": 600,
                             "description": "The post text in English, when the clip's language is not English."}}},
                   "description": "2 to %d clips, one for each moment." % MOST_MOMENTS},
          "blur": {"type": "array", "maxItems": 4, "items": {
              "type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0, "maximum": 1000}},
                   "description": "Boxes [x1, y1, x2, y2] on your 0-1000 grid of the recording's picture to blur in "
                                  "every clip for its whole length, such as a stream's chat box, which stays in one "
                                  "place. A game stream must give its chat box here."}},
         ["clips"], make_clips, changes=True),
    Tool("undo", "Took back the last change",
         "Take back the last change to the clip: a layout, a moved part, a style, a title, the captions, a blur, a hit, a "
         "look, a push or a decision about people. The "
         "person can also do this on the page.",
         {}, [], undo, changes=True),
]}
SCHEMAS = [t.schema() for t in TOOLS.values()]
