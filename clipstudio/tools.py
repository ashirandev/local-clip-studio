"""The agent's hands: every tool Muse can call, each with its input checked before it runs.

A tool is a plain function with a one-line label for the page, a description for Muse and a JSON Schema for its
input. Every tool also gets two fields from the loop:
- why: one line in the person's language, shown on the page next to the step
- plan_step: which step of Muse's plan the call works on, so the page can tick the plan

Input is checked here, not trusted: the hosted API does not promise that arguments match the schema. Wrong input is
refused with a sentence that says what to send instead, so Muse can fix it and call again.
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import traceback
import types

import numpy as np

from . import (asr, channels, cover, cutout, effects, faces, focus, green, hearing, layouts, maps, mind, montage,
               pauses, person, picture, pitch,
               countdown, previews,
               project, recording, render, review, sends, spent, voices,
               screen, spoken, takes, talk, text, thai, tracks, watch, words)
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
    if kind == "array" and (spec.get("items") or {}).get("type") == "string":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = [v for v in value.split(",") if v.strip()]
        if isinstance(value, str):
            value = [value]
        low, high = spec.get("minItems", 0), spec.get("maxItems", 100)
        if not isinstance(value, list) or not low <= len(value) <= high:
            raise Refused("%s must be a list of %d to %d names." % (name, low, high))
        return [_value("%s, name %d," % (name, i), v, spec["items"]) for i, v in enumerate(value, 1)]
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
    found = recording.walk(words) or recording.walk(words, fresh=True)
    where = ", ".join(folders) + ("" if recording._swept.get("complete", True) else
                                  " (the sweep stopped after %d s, so some may be missing)" % recording.SWEEP_SECONDS)
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
    seen = montage.overview(rec) if picture else None
    if seen:
        job.attach(seen[0], "Four moments across %s. %s" % (rec["name"], montage.OVERVIEW_SAID % ", ".join(
            recording.clock(t) for t in seen[1])))
        head += "\nA picture of four moments across it comes next, so you can see what it is."
    elif picture:
        job.attach(picture, "A picture from %s, a tenth of the way in." % rec["name"])
        head += "\nA picture from it comes next, so you can see what it is."
    if round(rec["seconds"]) < SHORTEST:
        head += "\n" + TOO_SHORT % ("It", round(rec["seconds"], 1), SHORTEST)
    return head + "\nThe recording is only read, never changed."


TOO_SHORT = ("%s is only %g s long, shorter than the %d s a clip needs, so no part of it can be a clip on its own. Tell "
             "the person, and offer to join it with the other recordings of its folder (open_clips), or to choose a "
             "longer recording.")


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
# A piece of a long clip given no seconds starts this much before the middle picture of the clip's strip, one Muse saw,
# so the picture stays in it however short it is cut to fit (SHORTEST_FIT at the least). Seen 10-29 on a trip seen
# through glasses: seven pieces given no seconds all started at 0 s, the first one looking at the sky. The close, the last
# piece on the clip shot last, goes by the closing picture instead: 10-31 Glimmer named clip 7 as the close 3 of 3 with no
# seconds, and its piece came from 61.7 s of 125 s, not the group facing the camera at 113-119 s (lane 4).
MIDDLE_LEAD = 1.0
OVER_LENGTH = 1.25
LINE_EVERY = 15.0
# A montage whose pieces are this much talk is told by its captions: no line of words is asked for each part, and a
# line stays over its own piece only, the way the reviews Tarm follows put a hook on the first seconds and nothing
# more (Extreme IT, 10-28). C's yeti cut kept its hook over 12 s and a title over every part.
TALKED = 0.6


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
            c["claps"] = takes.heard_claps(c["rec"]) if c["said"] else []
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
        last = c is clips[-1]
        picture = montage.strip(c, height, last)
        if picture:
            job.attach(picture, "Clip %d (%s, %g s): pictures at %s s:" % (
                c["n"], rec["name"], rec["seconds"],
                ", ".join("%g" % t for t in montage.strip_times(rec["seconds"], last))))
        kept.append({"n": c["n"], "rec": rec, "when": montage.when(c), "stills": c["stills"], "said": c["said"],
                     "unheard": c["unheard"], "claps": c["claps"], "picture": picture})
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
    talking = [c["n"] for c in kept if _talking(c) and c["rec"]["seconds"] > LONGEST_PIECE]
    if talking:
        lines.append("Someone talks through clip%s %s: choose the lines that tell the story, and give each piece the "
                     "seconds around them. A clip someone talks through may give more than one piece."
                     % ("s" if len(talking) > 1 else "", ", ".join(map(str, talking))))
    return "\n".join(lines)


TALKING = 0.3
TALKING_LINES = 3


def _talking(c):
    """True when someone talks through clip c (open_clips): words cover TALKING of it or more and at least
    TALKING_LINES of its lines are sure. Measured on Tarm's yeti review takes (10-28): words cover 43-80% of the five
    he talks through; the three short clips around them have no sure line."""
    said = [p for p in c.get("said") or [] if not pauses.filler(p["text"])]
    return (len([p for p in said if not _unsure(p)]) >= TALKING_LINES
            and sum(p["to"] - p["from"] for p in said) >= TALKING * c["rec"]["seconds"])


def _clip_said(job, rec):
    """(what is said in a whole clip as heard, or None when it has no sound; why it could not be heard, or None)."""
    if not rec["sound_tracks"]:
        return None, None
    said = hearing.heard(rec, 0.0, rec["seconds"], job.languages())
    if said is None:
        try:
            said = job.hear(rec, 0.0, rec["seconds"])
        except hearing.HearingError as e:
            return [], str(e)
        hearing.keep(rec, 0.0, rec["seconds"], said, job.languages())
    return said, None


def _clip_lines(c):
    rec = c["rec"]
    shape = "upright" if rec["height"] > rec["width"] else (
        "lying (cropped upright to fill the frame, around its face or the focus you give it)"
        if not montage.fills(rec["width"], rec["height"]) else "square")
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
        listed, cuts = _listed(c)
        if cuts is not None:
            lines.append("   Someone talks through it: give each piece of it a start and an end around the lines you "
                         "keep. The program cuts the stretches marked cut here, and silences over %g s." % DEAD_AIR)
        if any(less for _, less in listed):
            lines.append("   A line marked ~ was heard less surely: some of its words may be misheard.")
        rows = [(p["from"], "   %g s %s%s" % (round(p["from"], 1), "~ " if less else "", p["text"]))
                for p, less in listed]
        rows += [(x, "   %g-%g s cut: %s" % (round(x, 1), round(y, 1), why)) for x, y, why in cuts or []]
        lines += [r for _, r in sorted(rows, key=lambda r: r[0])]
        held = sum(1 for p in c["said"] if _unsure(p)) - sum(1 for _, less in listed if less)
        if held > 0:
            lines.append("   %d more line%s heard but not sure, such as voices around the camera: left out." % (
                held, "" if held == 1 else "s"))
    return lines


def _listed(c):
    """([(phrase, less sure)] of clip c (open_clips) shown to Muse, the take cuts of it or None): its sure lines, and in
    a clip someone talks through also those heard from READ_SURE up, outside the take cuts. Seen 10-28 in Tarm's yeti
    review: his verdict and the price ("ราคาไม่ถึงสองพัน", heard 0.83 to 0.86) were among 14 lines of IMG_3376 held
    back as not sure, so neither Muse ever saw the price; the voices around the camera in those clips were all heard
    below 0.8."""
    talking = _talking(c)
    cuts = _take_cuts(c) if talking and c["rec"]["seconds"] > LONGEST_PIECE else None
    out = []
    for p in c["said"] or []:
        if any(x <= p["from"] < y for x, y, _ in cuts or []):
            continue
        if not _unsure(p):
            out.append((p, False))
        elif talking and p.get("sure", 1.0) >= READ_SURE:
            out.append((p, True))
    return out, cuts


TAKE_CUT = {"again": "a take given up at a clap, said again after it",
            "unsure": "talk off the script before a clap",
            "slate": "a clap before the first word",
            "said again": "a take said again just after it"}


def _take_cuts(c):
    """[[from, to, why]] that the program cuts from any piece of talking clip c that holds them: takes given up at a
    clap or said again, and slates (takes.at_claps, takes.said_again), as _pieces cuts them."""
    said = [p for p in c["said"] or [] if p["to"] > p["from"]]
    clapped = takes.at_claps(said, c.get("claps") or [], unsure=_unsure, slates=True)
    again = [dict(x, why="said again") for x in takes.said_again(said, 0.0, c["rec"]["seconds"])
             if not any(x["cut"][0] < y["cut"][1] and y["cut"][0] < x["cut"][1] for y in clapped)]
    return sorted(x["cut"] + [TAKE_CUT[x["why"]]] for x in clapped + again)


def _each_piece(asked, count):
    """How long a piece given no seconds runs: the length asked shared by the pieces, from SHORTEST_FIT to
    LONGEST_PIECE. Glimmer's montage 10-30 (164839-7a16) gave no seconds for any of 7 clips: each ran LONGEST_PIECE,
    one rhythm, 50 s for the 45 s asked."""
    return round(min(LONGEST_PIECE, max(SHORTEST_FIT, float(asked) / max(1, count))), 1)


def arrange_clips(job, why, story, clips, left_out=None, hook="", products=None):
    opened = job.chat.clips
    if not opened:
        raise Refused("No clips are open. Open a folder of clips with open_clips first.")
    if (job.chat.clip or {}).get("recipe") != MONTAGE:
        raise Refused("Load the %s recipe first: it says how to choose and order the clips." % MONTAGE)
    hook = " ".join((hook or "").split())
    if not clips:
        clips, left_out = _the_cut(opened, hook, left_out)
    elif hook and not clips[0].get("words"):
        clips = [dict(clips[0], words=hook)] + list(clips[1:])
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
    talking = {n for n, c in known.items() if _talking(c) and c["rec"]["seconds"] > LONGEST_PIECE}
    twice = sorted({n for n in named if named.count(n) > 1 and not (n in talking and n not in out)})
    if twice:
        problems.append("Clip%s %s named twice: each clip is either used once or left out%s." % (
            "s" if len(twice) > 1 else "", ", ".join(map(str, twice)),
            " (only a clip someone talks through may give more than one piece)" if talking else ""))
    strange = sorted({n for n in named if n not in known})
    if strange:
        problems.append("There is no clip %s: the clips are numbered 1 to %d." % (
            ", ".join(map(str, strange)), len(known)))
    missing = sorted(set(known) - set(named))
    if missing:
        problems.append("Clip%s %s %s neither used nor left out. Say for every clip whether it goes in, and why it "
                        "does not when it does not." % ("s" if len(missing) > 1 else "", ", ".join(map(str, missing)),
                                                        "are" if len(missing) > 1 else "is"))
    pieces, lines, fitted, taken = [], [], [], {}
    each = _each_piece(job.chat.clip["length_s"], len(clips))
    for k, c in enumerate(clips, 1):
        if c["clip"] not in known:
            continue
        clip = known[c["clip"]]
        rec = clip["rec"]
        a = c.get("start", 0.0)
        b = min(c.get("end", rec["seconds"]), rec["seconds"])
        talk = c["clip"] in talking
        if not talk and "start" not in c and "end" not in c and rec["seconds"] > LONGEST_PIECE:
            close = k == len(clips) and c["clip"] == max(known)
            seen = round(rec["seconds"] * montage.LAST_AT, 2) if close else montage.strip_times(rec["seconds"])[
                len(montage.STRIP_AT) // 2]
            a = round(max(0.0, min(seen - MIDDLE_LEAD, rec["seconds"] - each)), 2)
            b = round(a + each, 2)
            fitted.append("clip %d: no seconds were given, so its piece starts at %g s, by the %s picture of its "
                          "strip (%g s), and runs %g s, %s; give a start and an end for other seconds" % (
                              c["clip"], a, "closing" if close else "middle", seen, each,
                              "the longest a piece runs" if each >= LONGEST_PIECE else
                              "the %g s asked shared by the %d pieces" % (job.chat.clip["length_s"], len(clips))))
        frozen = [s for s in clip["stills"] if s[0] < b and a < s[1]]
        if talk and ("start" not in c or "end" not in c):
            problems.append("Clip %d is someone talking for %g s: give its piece a start and an end around the lines "
                            "you keep, at the seconds the list gives them; a piece of it never starts at 0 s by "
                            "itself." % (c["clip"], rec["seconds"]))
            b = min(b, a + LONGEST_PIECE)
        elif b <= a:
            problems.append("Clip %d: start %g s must come before end %g s; the clip is %g s long."
                            % (c["clip"], a, b, rec["seconds"]))
        elif b - a < min(SHORTEST_PIECE, rec["seconds"]) - 0.01:
            problems.append("Clip %d: %g-%g s is under %g s, too short to see. Keep more of it, or leave it out."
                            % (c["clip"], a, b, SHORTEST_PIECE))
        elif talk:
            given = (a, b)
            a, b, _ = _snapped({"said": clip["said"] or [], "heard": (0.0, rec["seconds"])}, a, b)
            if any(x < b and a < y for x, y in taken.get(c["clip"], [])):
                problems.append("Clip %d: %g-%g s overlaps a piece of it you already use; a line goes in once."
                                % ((c["clip"],) + given))
            taken.setdefault(c["clip"], []).append((a, b))
            level = takes.heard_level(rec) if clip["said"] else None
            parts, cut = _on_quiet(rec, _pieces(clip["said"] or [], a, b, _still(rec, None, a, b, None),
                                                clip.get("claps") or [], slates=True, of="the piece", level=level), level)
            parts = takes.without(parts, clip["stills"])
            if not parts:
                problems.append("Clip %d: nothing is said at %g-%g s once the takes given up and the silences are cut. "
                                "Keep lines the list shows." % (c["clip"], a, b))
                continue
            if cut:
                fitted.append("clip %d, %g-%g s: %s" % (c["clip"], a, b, cut[0].lower() + cut[1:].rstrip(".")))
            first = len(pieces) + 1
            pieces += [{"rec": rec, "from": round(x, 2), "to": round(y, 2), "clip": c["clip"], "why": c["why"],
                        "talk": True, "focus": c.get("focus"), "whole": c.get("whole")} for x, y in parts]
            if c.get("words"):
                lines.append({"piece": first, "last": len(pieces), "text": c["words"]})
            continue
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
                       "why": c["why"], "focus": c.get("focus"), "whole": c.get("whole")})
        if c.get("words"):
            lines.append({"piece": len(pieces), "last": len(pieces), "text": c["words"]})
    boxes = {b["clip"]: b["box"] for b in products or [] if b.get("clip")}
    own = {b["piece"]: b["box"] for b in products or [] if b.get("piece")}
    odd = sorted(set(own) - set(range(1, len(pieces) + 1)))
    if odd or any(not (b.get("clip") or b.get("piece")) for b in products or []):
        problems.append("Each of products names a clip or a piece: the pieces are numbered 1 to %d%s." % (
            len(pieces), " (there is no piece %s)" % ", ".join(map(str, odd)) if odd else ""))
    for k, p in enumerate(pieces, 1):
        if k in own:
            p["focus"] = own[k]
        elif not p.get("focus") and p["clip"] in boxes:
            p["focus"] = boxes[p["clip"]]
    review = bool(opened.get("review")) or _a_review(job, None)
    lying = [(k, p) for k, p in enumerate(pieces, 1) if review and not p.get("whole")
             and not montage.fills(p["rec"]["width"], p["rec"]["height"])]
    bare = sorted({p["clip"] for p in pieces if review and not montage.fills(p["rec"]["width"], p["rec"]["height"])
                   and (p.get("whole") or not p.get("focus"))})
    seen = [[p["clip"], p["from"], p["to"]] for _, p in lying]
    if any(k not in own for k, _ in lying) and opened.get("pieces_shown") != seen:
        opened["pieces_shown"] = seen
        for k, p in lying:
            t = round((p["from"] + p["to"]) / 2.0, 2)
            picture = recording.frame(p["rec"], at=t)
            if picture:
                job.attach(os.path.join(recording.WORK, "frames", picture), "Piece %d: clip %d at %g s, the middle "
                           "of its %g-%g s:" % (k, p["clip"], t, p["from"], p["to"]))
        problems.append(PIECE_BOXES % ", ".join(str(k) for k, _ in lying))
        bare = sorted({p["clip"] for p in pieces if p.get("whole") and p["clip"] in bare})
    if bare:
        problems.append(PRODUCT_LEADS % ("clips %s lie" % ", ".join(map(str, bare)) if len(bare) > 1 else
                                         "clip %d lies" % bare[0], "their" if len(bare) > 1 else "its"))
    middled = set()
    for p in pieces:
        if not montage.fills(p["rec"]["width"], p["rec"]["height"]) and not p.get("whole"):
            p["centre"] = _upright_centre(p["rec"], p["from"], p["to"], p.get("focus"))
            if p["centre"] is None:
                p["centre"] = 0.5
                middled.add(p["clip"])
    if middled:
        fitted.append("no face was found in clip%s %s, so %s cropped upright around the middle of the picture: give "
                      "focus where the product or the action is" % ("s" if len(middled) > 1 else "", ", ".join(
                          map(str, sorted(middled))), "they are" if len(middled) > 1 else "it is"))
    asked = job.chat.clip["length_s"]
    most = min(LONGEST, asked * OVER_LENGTH)
    length = sum(p["to"] - p["from"] for p in pieces)
    said_length = sum(p["to"] - p["from"] for p in pieces if p.get("talk"))
    if length > most:
        target = min(asked, most)
        _shortened([p for p in pieces if not p.get("talk")], max(0.0, target - said_length))
        now = sum(p["to"] - p["from"] for p in pieces)
        if now < length - 0.05:
            fitted.append("the pieces came to %g s, longer than the %g s asked, so the end of each piece without "
                          "talk was cut by the same share, keeping at least %g s of each" % (
                              round(length, 1), target, SHORTEST_FIT))
        length = now
    if not SHORTEST <= round(length) <= most:
        problems.append("The pieces add up to %g s; this clip is %d to %d s (about %d s was asked). %s" % (
            round(length, 1), SHORTEST, most, asked, (
                "Keep fewer lines of the talking clips (%g s of them, never cut inside a line)." % round(said_length, 1)
                if said_length > most * 0.5 else "Use fewer clips.") if length > most else
            "Keep more seconds of each, or more clips."))
    if clips and not clips[0].get("words"):
        problems.append("The first piece (clip %d) opens the clip: give it words that say where we are or what this "
                        "is." % clips[0]["clip"])
    talked = said_length >= TALKED * length
    if not talked and len(lines) < round(length / LINE_EVERY):
        problems.append("%d line%s of words for %g s: give at least %d, one for each part of the story, so the "
                        "viewer knows what each part is." % (len(lines), "" if len(lines) == 1 else "s",
                                                             round(length, 1), round(length / LINE_EVERY)))
    thai = _thai_for_english(job, [ln["text"] for ln in lines])
    if thai:
        problems.append("The person asked for the words on the clip in English, so a viewer who reads no Thai can "
                        "follow: write these in English: %s." % "; ".join('"%s"' % t for t in thai))
    if problems:
        raise Refused("Nothing was joined. Fix all of these, then call it again:\n- " + "\n- ".join(problems))
    try:
        texts, spelling = _spell(job, [ln["text"] for ln in lines]) if lines else ([], "")
    except hearing.Stopped:
        return "Not arranged: the person pressed Stop."
    for ln, words in zip(lines, texts):
        ln["text"] = words
        _fits_a_place(words, "the words over clip %d" % pieces[ln["piece"] - 1]["clip"], job.chat.clip["style"])
    led = [p for p in pieces if review and p.get("focus") and p.get("centre") is not None]
    for p in led:
        p["push"] = True
        p["face"] = _circle_face(p["rec"], p["from"], p["to"], p["centre"])
    if led:
        circled = sorted({p["clip"] for p in led if p["face"]})
        fitted.append("in this product review each lying piece fills the frame around the product you named and "
                      "pushes in on it slowly%s" % (
                          "; the face, outside that, rides in a small circle at the right (clip%s %s)" % (
                              "s" if len(circled) > 1 else "", ", ".join(map(str, circled))) if circled else ""))
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
    if job.chat.clip is not None:
        job.chat.clip["talked"] = talked
    at = montage.starts(pieces)
    for ln in lines:
        k = ln["piece"] - 1
        later = [x["piece"] for x in lines if x["piece"] > ln["piece"]]
        if talked:
            later = [ln.pop("last") + 1] if ln["last"] < len(pieces) else []
        ln.pop("last", None)
        ln["start"], ln["end"] = at[k], (at[later[0] - 1] if later else round(rec["seconds"], 2))
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
    sheet = montage.pieces_sheet(pieces, montage.STRIP_HEIGHT if len(pieces) <= montage.MANY else montage.STRIP_LOWEST)
    if sheet:
        job.attach(sheet, "The pieces in order, a picture from the middle of each, marked with its clip and second:")
        said.append("A picture from the middle of each piece comes next, in order, marked with its clip and second: "
                    "check that each shows what its why and the words over it say. If one does not, arrange again "
                    "with other seconds of that clip.")
    made = make_clip(job, why, 1, "subject")
    if talked:
        try:
            made += "\n\nCaptions put on by the program, as a talked review has them, one line at a time: take them off " \
                    "only if the person asks.\n" + add_captions(job, why, on=True)
        except Refused:
            pass
    return "\n".join(said) + "\n\n" + made


def _the_cut(opened, hook, left_out):
    """(clips, left_out) for arrange_clips from the cut the program made of a review (_review_cut), hook over its first
    piece; Muse's own reasons for leaving a clip out are kept. Refused when there is no cut or no hook."""
    cut = (opened.get("review") or {}).get("cut")
    if not cut:
        raise Refused("Give the clips that go in, in order: the program made no cut of these clips for you to use.")
    if not hook:
        raise Refused("Give hook: the words over the first piece of the program's cut, 2 to 6 words from what is said "
                      "in it, such as the question the clip answers.")
    clips = [dict({"clip": p["clip"], "start": p["start"], "end": p["end"], "why": review.SAYS[p["job"]]},
                  **({"words": hook} if k == 0 else {})) for k, p in enumerate(cut["clips"])]
    mine = {c["clip"]: c for c in left_out or []}
    return clips, [mine.get(c["clip"], c) for c in cut["left_out"]]


PRODUCT_LEADS = ("This is a product review, so the product leads (Tarm 10-29: the product stands out most, the face "
                 "may ride in a circle): %s on %s side, so give each one the box around the product and "
                 "the hands showing it, as products or as a piece's focus, on your 0-1000 grid of that clip's picture, "
                 "and never whole. The program fills the upright frame around it, pushes in on it and puts the face "
                 "in a small circle.")
# On the yeti review (10-29) the box drawn on clip 7's strip missed the mic in the piece 20 s later: the product
# moves in a take, so each lying piece's box is drawn on a picture of that piece.
PIECE_BOXES = ("This is a product review, so the product leads (Tarm 10-29: the product stands out most, the face "
               "may ride in a circle), and it moves within a take: a box drawn on a clip's strip misses it in a "
               "piece from another second. A picture of each piece that lies on its side comes next, at its middle "
               "(pieces %s). Give products again with one {piece, box} for each: the box around the product and the "
               "hands showing it, on your 0-1000 grid of that piece's own picture. The program fills the upright "
               "frame around it, pushes in on it and puts the face in a small circle.")
CIRCLE_SEEN = 0.5
# The most of the face's width the upright crop may show for the face to ride in a circle too: on the yeti review
# (10-29) a face 46% inside the crop, its middle just outside, showed twice, half at the edge and whole in the circle.
CIRCLE_SHOWN = 0.25


def _circle_face(rec, a, b, centre):
    """The face box to show in a circle beside the product, in pixels of the recording: the face seen most from a to
    b, when it is seen in at least CIRCLE_SEEN of the pictures, whole inside the picture (a face cut by the top edge
    was a shoulder on the yeti takes) and at most CIRCLE_SHOWN of its width inside what the upright crop around
    centre shows; else None."""
    try:
        best = faces.places(faces.sample(rec, a, b))
    except (faces.NoFaces, faces.Stopped):
        return None
    if not best or best[0]["seen"] < CIRCLE_SEEN * best[0]["of"]:
        return None
    x, y, w, h = best[0]["box"]
    edge = 0.01 * rec["height"]
    if x < edge or y < edge or x + w > rec["width"] - edge or y + h > rec["height"] - edge:
        return None
    left, wide = montage.shown(rec, centre)
    inside = max(0.0, min(x + w, left + wide) - max(x, left))
    return None if inside > CIRCLE_SHOWN * w else [round(v, 1) for v in (x, y, w, h)]


def _upright_centre(rec, a, b, focus):
    """Where a lying piece is cropped upright to fill the frame, 0 to 1 of its width: the middle of the focus Muse gave
    on its 0-1000 grid, else the face seen most from a to b, else None (no face, or faces cannot be looked for)."""
    if focus:
        return (focus[0] + focus[2]) / 2000.0
    try:
        best = faces.places(faces.sample(rec, a, b))
    except (faces.NoFaces, faces.Stopped):
        return None
    if not best:
        return None
    x, _, w, _ = best[0]["box"]
    return (x + w / 2.0) / rec["width"]


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
        raise Refused("%s take %d lines at the title size; the place holds %d of the %d they need on your grid. Use "
                      "fewer words." % (what[0].upper() + what[1:], len(size.lines), layouts.grid(band[3], "y"),
                                        layouts.grid(size.h, "y", up=True)))


def _montage_moment(job, rec, pieces, at, story, lines):
    """The look and the one moment of a joined montage, made by the program from the clips Muse arranged: what is
    said in each piece moved to the joined recording's seconds."""
    said = []
    for p, t in zip(pieces, at):
        for s in hearing.heard(p["rec"], 0.0, p["rec"]["seconds"], job.languages()) or []:
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


# A montage's words keep off the people, not only their faces: on the horse ride (10-30) the face finder saw the
# rider in 2 of 84 pictures, in the top place, and the hook went to the middle, over their body. The body under a
# face stands BODY_WIDE times its width across and BODY_DOWN times its height below it.
BODY_WIDE, BODY_DOWN = 3.0, 6.0


def _with_bodies(times):
    """times (from _face_times) with each box grown to the body under it."""
    return [(t, [[f[0] - f[2] * (BODY_WIDE - 1) / 2, f[1], f[2] * BODY_WIDE, f[3] * (BODY_DOWN + 1)] for f in boxes])
            for t, boxes in times]


def _put_lines(job, rec, lines):
    """Puts a montage's words on as titles, each at the first of the layout's places it covers no person in for its
    whole time, else where it covers people the least, for the first stretch of its time (LISTED s or more) where it
    covers none, grown as far as it stays clear (_grown). Returns sentences for Muse. Seen 10-28: Muse spent four
    steps moving a line off a face itself, and ran out of room before the hand-over."""
    state = _clip_state(job)
    lay = state["clip"]["layout"]
    default = job.chat.clip["style"] if job.chat.clip else "clean"
    st = render.style_of(state, default)
    times, every = _face_times(state, rec)
    times = _with_bodies(times)
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
        grow = 1 if hit else _grown(state, default, ln["text"], band, times, every, start, end)[0]
        state = job.edit(dict({"do": "title", "text": ln["text"], "start": start, "end": end, "place": name,
                               "box": band, "key": None}, **({"grow": grow} if grow > 1 else {})))
        out.append('"%s" at the %s from %g to %g s%s.' % (
            ln["text"], name, start, end, ", where it covers a person at %s: move it with remove_title and "
            "add_title" % _spans(hit) if hit else ", clear of every person" if clear == [] else
            ", the first part of its piece where it covers no person"))
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
MOST_MOMENTS = 25
MOST_AT_ONCE = 8
NEAR_ENOUGH = 0.75
FEW_MOMENTS = ("The person chose to see %d clips and you proposed %d. Look at more stretches across the recording and propose "
               "them all in one call. If it does not hold that many, call it again with what you have.")


def _too_few(got, wanted):
    """Far under what the person chose. Tarm 11-01: of 20, a number near 20 is fine, more or fewer: they should not wait for
    an exact count, as they pick from the cards anyway."""
    return got < wanted * NEAR_ENOUGH


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


def watch_piece(job, why, start, end, person_said=None):
    """Muse Wonder watches start to end, picture and sound (watch.py), and its moments come back to Muse as
    suggestions: each with its second in the recording and in the clip, what happens, and the tool call that would put
    its effect there. Whichever Muse runs the chat calls it; the watching is always Muse Wonder's, with the person's
    Meta key. With Muse on this computer nothing is sent unless the person asked for it in their own words
    (person_said) or turned on the switch in Options (sends.py)."""
    asked = _asked(job, person_said)
    rec = _open(job, "watch")
    clock = recording.clock
    a, b = _time("start", start), _time("end", end)
    if a >= rec["seconds"]:
        raise Refused("start %s is after the end of the recording, which is %s long." % (clock(a), clock(rec["seconds"])))
    a, b = round(a, 2), round(min(b, rec["seconds"]), 2)
    if b - a < watch.SHORTEST:
        raise Refused("Give a stretch of at least %d s to watch; %s-%s is %.1f s." % (watch.SHORTEST, clock(a),
                                                                                    clock(b), b - a))
    if b - a > watch.LONGEST:
        raise Refused("Muse Wonder watches at most %d s at a time, and %s-%s is %d s. Watch the clip's own stretch, "
                      "or the %d s around the moment you mean." % (watch.LONGEST, clock(a), clock(b), round(b - a),
                                                                   watch.LONGEST))
    wonder = mind.MINDS.get("wonder")
    if not wonder or not wonder.secret():
        raise Refused("Muse Wonder watches with the person's own Meta API key, and none is set, so nothing was sent. "
                      "Go on from your looks and what was heard. Tell the person once that Options > API takes the "
                      "key.")
    if _here(job) and not asked and not sends.allowed():
        raise Refused("Not watched, and nothing was sent: you work on this computer, so no piece of the person's "
                      "video goes to Meta unless they ask for it in their own words (quote them in person_said) or "
                      "turn on Meta sees pieces in Options. Go on from your looks and what was heard, without asking "
                      "them about it.")
    if not asked and not job.chat.moments and _told_once(job, "watched_early"):
        raise Refused("Muse Wonder has watched once already and no moment is proposed yet. Each watch is paid: propose "
                      "from your looks and what was heard, then have it watch the clip's own stretch for its effects.")
    started = time.time()
    try:
        job.working("Cutting %s-%s for Muse Wonder to watch" % (clock(a), clock(b)))
        try:
            path = watch.cut(rec, a, b, on_start=job.started_work)
        finally:
            job.done_working()
        if job.stopped:
            return "Not watched: the person pressed Stop."
        got = job.ask_other("wonder", watch.question(path, b - a, rec["name"], "English"), watch.SCHEMA,
                            "Muse Wonder is watching %s-%s with its sound" % (clock(a), clock(b)), watch.ANSWER_TOKENS)
        found = watch.moments(got["text"], a, b)
    except hearing.Stopped:
        return "Not watched: the person pressed Stop."
    except (watch.WatchError, mind.MindError) as e:
        return "Muse Wonder did not watch %s-%s: %s Go on from your looks." % (clock(a), clock(b), e)
    took, usd = round(time.time() - started, 1), (got.get("engine") or {}).get("usd")
    job.show(watched={"start": a, "end": b, "moments": found})
    head = "Muse Wonder watched %s-%s with its sound (%g s%s)" % (clock(a), clock(b), took,
                                                                 ", $%.3f" % usd if usd is not None else "")
    if not found:
        return head + " and saw nothing a viewer would react to that the words do not already carry."
    clip = (job.chat.project.state()["clip"] or {}) if job.chat.project else {}
    lines = [head + ". What it saw, which words alone miss, with the effect it would put there:"]
    for m in found:
        at = project.from_recording(clip["pieces"], m["at"]) if clip else None
        where = "%s (%s)" % (_tenth(m["at"]), "%.1f s of the clip" % at if at is not None else "outside the clip")
        lines.append("- %s, %s: %s %s" % (where, m["kind"], m["what"], _watch_effect(m, at)))
    lines.append("These are suggestions: check each against your looks, keep the ones that fit, and put them on at the "
                 "clip seconds given. A moment outside the clip can be taken in with change_ends.")
    return "\n".join(lines)


def _watch_effect(m, at):
    """The tool call that puts m's effect on the clip, as words for Muse."""
    if m["effect"] == "none":
        return "(no effect)"
    t = "%.1f" % at if at is not None else "its second once it is in the clip"
    if m["effect"] in watch.HITS:
        return "→ add_hit kind %s%s at %s" % (m["effect"], ' word "%s"' % m["word"] if m["effect"] == "word" else "", t)
    if m["effect"] == "flurry":
        return "→ add_flurry at %s" % t
    return "→ add_sound kind %s at %s" % (m["effect"], t)


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
    said = hearing.heard(rec, ha, hb, job.languages())
    lost = []
    if said is None:
        started = time.time()
        try:
            said = job.hear(rec, ha, hb)
        except hearing.PartHeard as e:
            said, lost = e.said, e.lost
        except hearing.HearingError as e:
            return ["What is said here could not be heard: %s" % e]
        look["took"] = round(time.time() - started, 1)
        if not lost:
            hearing.keep(rec, ha, hb, said, job.languages())
    look["said"], look["heard"] = said, [ha, hb]
    holes = ["The hearing failed from %s to %s, so what is said there is missing here, whatever the lines around it "
             "say: look at that stretch again before you judge it." % (clock(x), clock(y)) for x, y in lost
             if y > a and x < b]
    inside = [p for p in said if p["to"] > a and p["from"] < b]
    if not inside:
        return holes or ["No words were heard here."]
    return (["What is said here, each line with when it begins and when it ends, as heard (some words may be misheard; the "
             "numbers are as heard; a line marked (not sure) is one the hearing was not sure of, so do not build on it):"] +
            ["%s-%s %s%s" % (clock(p["from"]), clock(p["to"]), p["text"], " (not sure)" if _unsure(p) else "")
             for p in inside] + holes)


def _fixed_said(job, rec, look, a, b, said=None, titled=()):
    """The lines heard from a to b (or the lines said, from the look) with misheard words fixed by Muse from the
    context and, when the recipe reads the screen, from the words written on it, and from titled, names the clip's
    title will show: [{from, to, text, heard, kept, named}], where kept says why a line stays as heard and named lists
    the names the fix took from the screen."""
    said = [p for p in look.get("said") or [] if a <= p["from"] < b] if said is None else said
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
    names = names + [n for n in titled if n not in names]
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


def _fix_moment(job, rec, m, titled=()):
    """Fixes the moment's lines when they were kept as heard for later, or when they lack a name of titled, from the
    title of a clip in a batch: the first moment is fixed before any title exists, and a seller's Redmi stayed "Let me"
    beside the title Redmi (11-01, 020751-cdcd). Stop raises hearing.Stopped."""
    said = m.get("said") or []
    lost = [n for n in titled if not any(n.lower() in s["text"].lower() for s in said)]
    if lost or any(s.get("kept") == LATER for s in said):
        m["said"] = _fixed_said(job, rec, job.chat.looks[m["look"] - 1], m["start"], m["end"], titled=lost)


AIR = 0.25
SNAP_REACH = 2.0
FINISH_REACH = 10.0
CARRY_GAP = 0.6
CARRY_REACH = 8.0
CARRY_SHOWN = 30
NUMBER = "(?:\\d|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ|ยี่|(?<!เรียบ)ร้อย|(?<!ผูก)พัน|หมื่น|แสน|ล้าน)"
NUMBER_ENDS = re.compile(NUMBER + "\\W*$")
NUMBER_STARTS = re.compile("^\\W*" + NUMBER)
UNIT_STARTS = re.compile("^\\W*(?:มิลลิ|มิลิ|ลิตร|ลิต|ตาราง|ตร\\.|เมตร|เซน|กิโล|บาท|เปอร์เซ็น|วัตต์|องศา|นาที|ชั่วโมง|วินาที)")


def _carried(before, line):
    """Whether line goes on from before, the line said just ahead of it: a number and what it counts are one thing to
    hear. Measured 10-29 on the 133 recordings heard so far: a line after one ending in a number starts 0.14 s later
    (median; 96% within CARRY_GAP), one starting with a unit 0.18 s (all 15), any other line 0.28 s. A line that is
    only a number goes on from the line before it. A joining word (ซึ่ง, แต่ว่า, แล้วก็) does not count: lines starting
    with one come 0.40 s after the line before, as many after a breath as any other (62% within CARRY_GAP)."""
    text = " ".join(line["text"].split())
    return bool(NUMBER_ENDS.search(before["text"]) or UNIT_STARTS.search(text)
                or (NUMBER_STARTS.search(text) and NUMBER_ENDS.search(text)))


def _only_closes(line):
    """Whether line is nothing but a word that closes the sentence before it. A clip does not open on one: lane 5 10-30,
    3 of 3 explain clips of gold EP1 opened on "นะครับ"."""
    return bool(ONLY_CLOSES.match(" ".join(line["text"].split())))


def _asks_after(lines, last):
    """The line of lines that asks to buy within ASK_GAP after last, a line with the price, or None."""
    return next((p for p in lines if last and PRICE.search(last["text"]) and 0 <= p["from"] - last["to"] <= ASK_GAP
                 and ASKS.search(p["text"])), None)


def _snapped(look, a, b, price=False):
    """(start, end, sentence or None): a moment's edges moved out of the words they would cut through, to just before
    the line the start falls in when it began within SNAP_REACH, and just after the line the end falls in when it
    finishes within FINISH_REACH and the clip still runs at most LONGEST, else to just before that line: a clip never
    stops inside a sentence. Tarm 10-30 on EP 0's talk, cut at 8:21 with 4.4 s of an answer left (FINISH_REACH was 4):
    "ตัดห้วนเกินไป มันยังไม่จบ". Of 2,589 lines heard so far, 2.3% run past 10 s. AIR seconds from the words; never
    outside what was heard. A model's own edges sit in a pause about half the time (measured by lane 5 on Tarm's own
    edits: 47%, 77% within 0.6 s), so a clip would often open or close on half a word. A start in the pause before a
    line that goes on from the line ahead of it (_carried) moves to that line's start, within CARRY_REACH: seen 10-29
    on a seller's review, a moment at 1:20 opened on "มิลลิลิตรต่อวัน" (millilitres a day), the 750 and what it
    measured said in the 4 s before. With price (a seller), an end that the price follows within PRICE_REACH, in what
    was heard, moves to just after the price: on Tarm's own green-screen seller (10-30) Muse ended the controller's
    moment at 0:39 and 0:36 both times, and its price was said at 0:41."""
    na, nb, said = a, b, look.get("said") or []
    lo, hi = look.get("heard") or (look["start"], look["end"])
    for p in said:
        if p["from"] < a < p["to"] and a - p["from"] <= SNAP_REACH:
            na = max(lo, p["from"] - AIR)
        if p["from"] < b < p["to"] and p["to"] - b <= FINISH_REACH and p["to"] + AIR - na <= LONGEST:
            nb = min(hi, p["to"] + AIR)
        elif p["from"] < b < p["to"] and p["from"] - AIR - na >= SHORTEST:
            nb = max(lo, p["from"] - AIR)
    lines = sorted((p for p in said if p["to"] > p["from"]), key=lambda p: p["from"])
    k = next((i for i, p in enumerate(lines) if p["from"] >= na), None)
    opened = lines[k] if k is not None else None
    while k and lines[k]["from"] - lines[k - 1]["to"] <= CARRY_GAP and a - lines[k - 1]["from"] <= CARRY_REACH \
            and lines[k - 1]["from"] - AIR >= lo and nb - lines[k - 1]["from"] + AIR <= LONGEST \
            and _carried(lines[k - 1], lines[k]):
        k -= 1
    closing = None
    if opened is not None and lines[k] is opened and _only_closes(opened) and k + 1 < len(lines) and \
            nb - lines[k + 1]["from"] + AIR >= SHORTEST:
        closing, na = opened, round(max(opened["to"], lines[k + 1]["from"] - AIR), 2)
    priced = next((p for p in lines if price and nb - AIR <= p["from"] <= nb + PRICE_REACH and p["to"] <= hi
                   and PRICE.search(p["text"]) and p["to"] + AIR - na <= LONGEST), None)
    if priced:
        nb = round(priced["to"] + AIR, 2)
    last = next((p for p in reversed(lines) if p["from"] < nb), None)
    asked = _asks_after([p for p in lines if p["to"] <= hi and p["to"] + AIR - na <= LONGEST], last) if price else None
    if asked:
        nb = round(asked["to"] + AIR, 2)
    moved = []
    if closing:
        moved.append("starts %.1f s later, so it does not open on \"%s\", which closes what was said before it"
                     % (na - a, " ".join(closing["text"].split())))
    elif opened is not None and lines[k] is not opened:
        na = round(lines[k]["from"] - AIR, 2)
        moved.append("starts %.1f s earlier, so it opens where the words start: \"%s\" goes on from what is said "
                     "before it" % (a - na, " ".join(opened["text"].split())[:CARRY_SHOWN]))
    elif na != a:
        moved.append("starts %.1f s earlier, so it opens before a line rather than inside it" % (a - na))
    if asked:
        moved.append("ends %.1f s later, so the call to buy said right after the price is in the clip" % (nb - b))
    elif priced:
        moved.append("ends %.1f s later, so the price said right after it is in the clip" % (nb - b))
    elif nb > b:
        moved.append("ends %.1f s later, so it closes after a line rather than inside it" % (nb - b))
    elif nb < b:
        moved.append("ends %.1f s earlier, so it closes before a line it would cut through rather than inside it"
                     % (b - nb))
    return round(na, 2), round(nb, 2), " and ".join(moved) or None


PRICE = re.compile("บาท|฿")
PRICE_REACH = 6.0
ASKS = re.compile("สนใจ|อยากได้|(?<![a-z])(?:CF|DM)(?![a-z])|ซีเอฟ|สี่เอฟ|ดีเอ็ม|ทักมา|ทักแชท|อินบ็อกซ์", re.I)
ASK_GAP = 1.5
ASK_OVER = 3.0
ONLY_CLOSES = re.compile("^(?:นะครับ|นะคะ|ครับผม|ครับ|ค่ะ|คะ|นะ)$")
CLOSES_FIRST = re.compile("^\\s*(?:นะครับ|นะคะ|ครับผม|ครับ|ค่ะ|คะ|นะ)\\s+\\S")
FINER_OVER = 6.0
FINER_PAST, AT_THE_EDGE = 4.0, 0.5
FINER_KINDS = ("explain", "seller", "talk")
FINER = ("Nothing was proposed yet. A heard line is a stretch of sound, not a sentence, and these lines at the edges of "
         "your moments were one stretch each, so the program heard them again piece by piece, each with when it begins "
         "and when it ends:\n%s\nPropose again: start each moment at the time the piece that opens a new thought "
         "begins, and end it at the time the piece that finishes a thought ends, so the clip does not stop inside a "
         "sentence. A piece that still finishes or agrees with what came before is the close of that thought, not a "
         "start.")


def _fine(t):
    """t in recording seconds as clock() writes it, to the hundredth, for a line shorter than a second."""
    m, s = divmod(round(t, 2), 60)
    h, m = divmod(int(m), 60)
    return "%d:%02d:%05.2f" % (h, m, s) if h else "%d:%05.2f" % (m, s)


def _kept_or_heard(job, rec, a, b):
    """What is said from a to b, heard now unless it was kept. Raises what job.hear raises."""
    said = hearing.heard(rec, a, b, job.languages())
    if said is None:
        said = job.hear(rec, a, b)
        hearing.keep(rec, a, b, said, job.languages())
    return said


def _finer(job, rec, look, line):
    """Cuts line of look["said"] at its dips (takes.dips) and hears each piece, and the pieces take its place: they
    are returned, or [] when it stays as it is, with no dips in it. line is tried once, so asking cannot go on. The
    last line of a look that runs into the edge of what was heard stopped because the hearing did, so it is heard
    FINER_PAST s on, and the look has heard that far: a moment may end there. Lane 5's showcase, 10-31: the last piece
    ended on "แต่ว่าถ้าบ้านไหนไม่ได้ชื้นนะครับ" at 1:25.62, and the sentence ended at 1:28."""
    line["finer"] = True
    reach, end = look.get("heard"), line["to"]
    if reach and line is look["said"][-1] and reach[1] - end < AT_THE_EDGE:
        end = round(min(rec["seconds"], reach[1] + FINER_PAST), 2)
    cuts = takes.dips(rec, line["from"], end)
    edges = [line["from"]] + cuts + [end]
    pieces = []
    for a, b in zip(edges, edges[1:]) if cuts else ():
        heard = _kept_or_heard(job, rec, a, b)
        if heard:
            pieces.append({"from": a, "to": b, "text": " ".join(p["text"] for p in heard),
                           "sure": min(p.get("sure", 1.0) for p in heard), "finer": True})
    if len(pieces) < 2:
        return []
    at = look["said"].index(line)
    look["said"][at:at + 1] = pieces
    if reach:
        reach[1] = max(reach[1], end)
    return pieces


def _finer_edges(job, rec, look, a, b):
    """For a moment from a to b of a clip made of what is said: the line its start or its end falls in, when it runs
    over FINER_OVER s or opens on a word that closes what came before, heard again piece by piece (_finer), and on a
    seller's clip the line after the price that asks to buy, over ASK_OVER s, as the clip takes it in (_snapped). Returns
    [a sentence for Muse for each line that was]. Lane 4 10-30: a heard line is a sound piece, not a sentence, and Muse
    can only choose where a line begins or ends; 53% of what Muse Voice Transcribe heard was in lines over 8 s. Spark
    10-31 on the seller: "กดสี่เอฟไว้ได้เลยเนาะ" and the next product's opening were one line of 7 s, and the
    controller's clip took the balm's opening. A look hears only so far, so in another run that line was 4.2 s in the
    look of the controller's clip and the same opening went in (lane 5, 035133-c553); the ask alone takes 2.5 s. Stop
    raises hearing.Stopped; a line that cannot be heard again stays as it is."""
    recipe = (job.chat.clip or {}).get("recipe")
    if recipe not in FINER_KINDS:
        return []
    said = look.get("said") or []
    opens = next((p for p in said if p["to"] > a), None)
    closes = next((p for p in reversed(said) if p["from"] < b), None)
    asks = _asks_after(said, closes) if recipe == "seller" else None
    out = []
    for side, line, over in (("start", opens, FINER_OVER), ("end", closes, FINER_OVER), ("end", asks, ASK_OVER)):
        if not line or line.get("finer") or not (line["to"] - line["from"] > over or (
                side == "start" and CLOSES_FIRST.match(line["text"]))):
            continue
        try:
            pieces = _finer(job, rec, look, line)
        except hearing.HearingError:
            continue
        if pieces:
            out.append("The moment's %s at %s:\n%s" % (side, _fine(a if side == "start" else b), "\n".join(
                "  %s-%s %s" % (_fine(p["from"]), _fine(p["to"]), p["text"]) for p in pieces)))
    return out


DEAD_AIR = 1.0
KEPT_AIR = 0.3
QUIET_UNDER = 10.0


def _pieces(said, a, b, still, claps=(), slates=False, of="the moment", edges=True, level=None):
    """(the pieces of the recording a clip from a to b is made of, a sentence for Muse or None): the earlier take of a
    line said again is cut (takes.said_again), and so is a take given up at a clap from claps (takes.heard_claps), with
    the clap, or the clap alone when it is a slate and slates (takes.at_claps); every silence longer than DEAD_AIR is
    cut down to KEPT_AIR at each
    side (pauses.tighten), the clip's opening and closing silence down to KEPT_AIR, where still(x, y) says nothing
    happens on the picture from x to y: a silence while the answer streams onto the screen is the payoff, not dead
    air. still None keeps the moment whole (a game, where the action is the clip and nobody needs to speak). Measured
    on 10-27's five clips: three opened silent for 0.5 s or more and were not handed over, and one held a first take
    and an 8 s stop before the line was said again. edges False keeps the opening and the closing as they are: the
    person put them there (change_ends). level, from takes.heard_level, keeps what is heard where nothing is said
    (_sounds): a silence is cut only where it is quiet."""
    if still is None:
        return [[a, b]], None
    said = [p for p in said if p["to"] > p["from"]]
    clapped = [x for x in takes.at_claps(said, [c for c in claps if a <= c[0] < b], unsure=_unsure, slates=slates)
               if x["cut"][0] < b and a < x["cut"][1]]
    again = [x for x in takes.said_again(said, a, b)
             if not any(x["cut"][0] < y["cut"][1] and y["cut"][0] < x["cut"][1] for y in clapped)]
    pieces = takes.without([[a, b]], [x["cut"] for x in again + clapped])
    if not pieces:
        return [], None
    try:
        _, quiet = pauses.tighten(pieces, said, gap=DEAD_AIR, air=KEPT_AIR)
    except ValueError:
        quiet = []
    words = sorted((max(p["from"], x), min(p["to"], y)) for p in said if not pauses.filler(p["text"])
                   for x, y in pieces if p["from"] < y and x < p["to"])
    if words and edges:
        quiet += [[pieces[0][0], words[0][0] - KEPT_AIR], [max(w[1] for w in words) + KEPT_AIR, pieces[-1][1]]]
    if not edges:
        quiet = [q for q in quiet if pieces[0][0] < q[0] and q[1] < pieces[-1][1]]
    quiet = [q for q in quiet if q[1] - q[0] >= 0.05 and still(q[0], q[1])]
    outer = pieces[0][0], pieces[-1][1]
    sounds = _sounds(level, said, a, b) if level and words else []
    heard = [part for q in quiet for part in _unheard(q, sounds)]
    held = round(sum(y - x for x, y in takes.without(pieces, heard))
                 - sum(y - x for x, y in takes.without(pieces, quiet)), 1)
    quiet = heard
    pieces = takes.without(pieces, quiet)
    gone = round((b - a) - sum(y - x for x, y in pieces), 1)
    if gone < 0.1:
        return pieces, None
    parts = ["the first take of a line said again at %s (\"%s\"; the take from %s is kept)" % (
        recording.clock(x["cut"][0]), x["first"], recording.clock(x["cut"][1] + takes.AIR)) for x in again]
    parts += [CLAPPED[x["why"]] % {"at": recording.clock(x["clap"]), "said": x["first"]} for x in clapped]
    inner = len(pieces) - 1 - len(again) - len(clapped)
    if inner > 0:
        parts.append("%d silence%s longer than %g s while nothing moved on the picture"
                     % (inner, "" if inner == 1 else "s", DEAD_AIR))
    if any(q[0] <= outer[0] or q[1] >= outer[1] for q in quiet):
        parts.append("the silence at its edges, so it opens and closes %g s from a word or a sound" % KEPT_AIR)
    return pieces, "The program cut %g s out of %s: %s.%s" % (gone, of, "; ".join(parts), (
        " It kept %g s where nobody speaks but the sound is as loud as the talk, a laugh, applause or a reaction: "
        "that is part of the moment, not silence." % held) if held >= 0.1 else "")


EDGE_REACH = 1.0
LAST_REACH = 4.0
PICTURE_ENDS = ("game", "sport")
EDGE_BED = 6.0
EDGE_FILE = 0.05
MOVED_CUTS = "The program moved %d cut%s, at most %g s, into the nearest quiet sound, so none falls inside a word."
LOUD_CUT = "The cut at %s lands on loud sound: nothing is quiet within %g s."


def _quiet_under(level):
    """The level in dB a sound is quiet under, from level, takes.heard_level's: QUIET_UNDER under what the loudest
    quarter of the stretch starts at, as the talk is there, with or without lines heard."""
    return float(np.percentile(level[1], 75)) - QUIET_UNDER


def _on_quiet(rec, made, level, last_end=False):
    """made, what _pieces gave (the pieces, its sentence), with every cut moved outward into the nearest quiet stretch
    within EDGE_REACH (takes.quiet_near, or over a bed of music or a crowd the nearest dip of EDGE_BED), KEPT_AIR from the
    sound on the side the clip keeps, and the sentence saying so. A cut never moves into what the clip keeps, so it
    cannot cut off a word it had: a start only moves earlier and an end only later (lane 5 heard it 10-31, an end moved
    back past the พัน of "สามพัน"). A cut with neither near stays where it is and the sentence
    says that; a cut already in the quiet, and the first and last moments of the recording, stay. Pieces that meet
    become one. Tarm 10-30: "ตัวตัดต่อ อยากให้ตัดเฉพาะตอนเสียงเงียบ แบบ wave เสียงมันเบาแล้ว ไม่เอาตัดกลางคำ", then, as a
    sound engineer: "the program should cut at the silent". Heard lines say where words are only to a fraction of a
    second: of the 140 cuts of the day's chats, 85 sat within 10 dB of the talk. With last_end, the clip's last end on
    loud sound looks on LAST_REACH s for a real pause, and no shorter gap or dip, as those lie between words: an end on
    loud sound breaks Tarm's rule, and EP 22 has a line ending at 26:19.30 with the sound at talk level until 26:22.36.
    Of 75 last ends this moves 4, and a clip at its longest stays."""
    pieces, said = made
    if not pieces or not level or not level[1]:
        return made
    under, loud = _quiet_under(level), []

    def land(t, opening):
        if t <= EDGE_FILE or t >= rec["seconds"] - EDGE_FILE:
            return t
        run = takes.quiet_near(rec, t, EDGE_REACH, under, bed=EDGE_BED, side=-1 if opening else 1)
        if not run:
            loud.append(t)
            return t
        if run[0] <= t <= run[1]:
            return t
        return round(max(run[0], run[1] - KEPT_AIR) if opening else min(run[1], run[0] + KEPT_AIR), 2)

    landed, moved = [], []
    for x, y in pieces:
        nx, ny = land(x, True), land(y, False)
        moved += [abs(new - old) for new, old in ((nx, x), (ny, y)) if abs(new - old) >= 0.01]
        if landed and nx <= landed[-1][1]:
            landed[-1][1] = max(landed[-1][1], ny)
        else:
            landed.append([nx, ny])
    if sum(y - x for x, y in landed) > LONGEST:
        return made
    run = takes.quiet_near(rec, landed[-1][1], LAST_REACH, under, side=1, pause_only=True) if (
        last_end and landed and landed[-1][1] in loud) else None
    if run:
        end = landed[-1][1]
        past = round(min(run[1], run[0] + KEPT_AIR), 2)
        if sum(y - x for x, y in landed) + past - end <= LONGEST:
            landed[-1][1] = past
            loud.remove(end)
            moved.append(past - end)
    told = ([MOVED_CUTS % (len(moved), "" if len(moved) == 1 else "s", round(max(moved), 2))] if moved else []) + [
        LOUD_CUT % (t, EDGE_REACH) for t in dict.fromkeys(_tenth(t) for t in loud)]
    return landed, " ".join(([said] if said else []) + told) or None


def _reaches_on(recipe, m):
    """True when the last end of the clip of recipe from moment m may look on for a pause (_on_quiet): not when the
    picture ended it, as a game's death or a goal's celebration does, nor when the person gave it."""
    return recipe not in PICTURE_ENDS and not m.get("ends")


def _loud_cuts(rec, cuts, level):
    """Those of cuts, seconds of the recording, that no quiet stretch holds, the first and last of the file apart."""
    under = _quiet_under(level)

    def held(t):
        run = takes.quiet_near(rec, t, takes.QUIET_LEAST, under)
        return run and run[0] <= t <= run[1]

    return [t for t in cuts if EDGE_FILE < t < rec["seconds"] - EDGE_FILE and not held(t)]


def _cuts_check(rec, state, kind):
    """(ok, found, who) for the check that the clip's cuts sit in quiet sound. _on_quiet moved those it could, so one
    left on loud sound had nothing quiet near: only a person can hear whether it is right."""
    if kind == MONTAGE:
        return None, "a montage cuts on the picture, under its music", "program"
    pieces = state["clip"]["pieces"]
    level = takes.heard_level(rec, pieces[0][0], pieces[-1][1])
    if not level:
        return None, "the sound could not be read", "program"
    loud = _loud_cuts(rec, sum(pieces, []), level)
    if loud:
        return None, "on sound as loud as the talk at %s of the recording, with nothing quiet near: tell the person" % (
            ", ".join(_tenth(t) for t in loud)), "Muse"
    return True, "every cut is in quiet sound", "program"


def _sounds(level, said, a, b):
    """[[from, to]] of the recording from a to b where the sound is less than QUIET_UNDER dB under the talk's, each
    with KEPT_AIR at its sides. level is (start, [dB every takes.LEVEL_STEP s]); the talk's level is the median
    where words are heard, and a step is loud when it and the mean of the 3 steps around it both are, so a key or a
    click in a pause stays quiet and a loud step never makes its quiet neighbour loud. A filler (pauses.filler) is
    a pause, however loud. Lane 5, 10-30: with Muse Voice Transcribe hearing, a talk clip of a story lost the 11 s of
    laughing after its last line, at -26 to -30 dB against the talk's -25 to -33. On every heard look of the chats,
    the stretches between words sat either within 6 dB of the talk (a laugh, a crowd, music under an episode) or 13 dB
    and more under it (Tarm's screen recordings 18 to 73)."""
    start, db = level
    step = takes.LEVEL_STEP
    db = np.asarray(db, np.float64)
    t = start + (np.arange(len(db)) + 0.5) * step
    inside = (t >= a) & (t < b)
    spoken, filled = np.zeros(len(db), bool), np.zeros(len(db), bool)
    for p in said:
        span = (t >= p["from"]) & (t < p["to"])
        if pauses.filler(p["text"]):
            filled |= span
        else:
            spoken |= span
    if not (spoken & inside).any():
        return []
    talk = float(np.median(db[spoken & inside]))
    mean = 10 * np.log10(np.convolve(10 ** (db / 10), np.ones(3) / 3, mode="same"))
    loud = np.r_[False, (np.minimum(db, mean) >= talk - QUIET_UNDER) & inside & ~filled, False]
    turns = np.flatnonzero(np.diff(loud.astype(np.int8)))
    return [[round(float(start + i * step - KEPT_AIR), 2), round(float(start + j * step + KEPT_AIR), 2)]
            for i, j in zip(turns[::2], turns[1::2])]


def _unheard(q, sounds):
    """The parts of the silence q that no sound covers: all of it when none is heard in it, else each part left long
    enough to be one of the silences cut (DEAD_AIR less the air at its sides)."""
    hit = [s for s in sounds if s[0] < q[1] and q[0] < s[1]]
    if not hit:
        return [q]
    return [p for p in takes.without([q], hit) if p[1] - p[0] >= DEAD_AIR - 2 * KEPT_AIR]


CLAPPED = {"again": "a take given up at the clap at %(at)s (\"%(said)s\"), said again after it",
           "unsure": "talk off the script before the clap at %(at)s (\"%(said)s\")",
           "slate": "the clap at %(at)s before the first word"}


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


# The event of a job whose person asked to see the moments first ("find the highlights, don't make anything yet",
# Tarm 10-30): its propose_moments ends the work, and make_clip refuses in it; a tap on a card is a new job.
WAITING = "waiting"


def _waits(job):
    """True when the person asked job to stop at the moments (WAITING)."""
    return any(e["kind"] == WAITING and e.get("job") == job.n for e in job.chat.events)


# A moment that stops where its look stops was never seen to end: 94 of 127 moments on 10-30 ended within 3 s of the
# look's end (lane 4). Recipe lines did not move Glimmer's pick, so the program
# looks on once, EDGE seconds from the edge, and Muse keeps the moment or moves it. A game's end is its death, so a game
# is only looked back from, when its moment runs under the recipe's length: all of Sifu's runs took one look opening a
# second after "ตรงนี้อีกแล้ววะ", and 35:52-36:11 shipped as 13 s of 30, three deaths at that spot unseen (lane 4). Once a
# job, and not for a moment near its length: 27 s of 30 sent back twice left the deaths for a stretch with none.
EDGE = 3.0
SEEN_ON, SEEN_BACK = 30.0, 25.0
GAME_SHORT = 0.85
COLD_RECIPES = ("talk", "explain")
EDGE_RECIPES = ("talk", "vlog", "explain", "game")


# A talk or a vlog's first moment under this share of the clip's length is told so, once a job, whatever its looks saw.
# Lane 4 10-31, EP 0 rows since the edit-first rubric: under 60% of the recipe's 60 s, 6 of 6 at 7 or lower; at 36 s or
# more, 11 of 12 at 7.5 or higher. Glimmer kept 18:21 in 4 of 9 talk runs, and seen_past left it no second word.
# A game moment within this of its look at both ends took the look whole, and is looked back from like a short one:
# Glimmer 8 of 13, the last 7 all 35:43-36:13, Spark 0 of 17, and both game 8s hold the death at 35:34 (lane 4 10-31).
WHOLE_LOOK = 1.0
SHORT_RECIPES = ("talk", "vlog")
FAR_SHORT = 0.6
FAR_SHORT_SAID = ("Not proposed yet: moment %d runs %g s of the %g s this clip can hold. A stranger needs how it starts and "
                  "how it ends: move the start back to where the story begins and the end on to where it pays off, looking "
                  "first if you have not seen them. If it is whole as it is, propose it again and it stands.")
# Past this many seconds after the end, the look's loudest second is named too. Lane 4 10-31 on 6c861e0: vlog asked 26:14
# and 26:05 with "The sound rises most at 26:20" already printed, was told to move the end on, and moved the start. Of the
# 6 far-short steps it names the rise on those 2 only. A rise is not always a laugh, so Muse is told, and answers.
RISE_PAST = 3.0
RISE_SAID = " The sound rises most at %s, after this end: if that is a laugh or the answer to what was said, end after it."
# On a vlog it is said on its own too, whatever the moment's length, and the start moves with the end: 101550-5bd0 ran
# 25:14 to 26:02, 48 s, too long to be told, and ends alone do not make 8: 25:19-26:22 got 6.5, the 8s start 25:43-25:52
# and run under 40 s. Measured on the moment's own look (lane 4 10-31): vlog Glimmer 6 of 11, Spark 4 of 18, talk 0 of 32.
RISE_ALONE = ("Not proposed yet: moment %d ends at %s, and the sound rises most at %s, after this end. If that is a laugh or "
              "the answer to what was said, end after it, and move the start on so the clip stays near its %g s. If it "
              "lands where it is, propose it again and it stands.")


def _seen_past(job, why, rec, i, m, look, a, b):
    """What Muse sees past the edges of moment i, m as Muse gave it (a to b inside look), when a start or an end Muse
    gave stops at one and no look has shown what is there, else what it reads again (_read_cold). Once for a look, so
    Muse's second answer stands."""
    clip = job.chat.clip or {}
    game = clip.get("recipe") == "game"
    if clip.get("recipe") not in EDGE_RECIPES:
        return None
    rise = look.get("loudest")
    past = rise is not None and rise - b > RISE_PAST
    if clip["recipe"] in SHORT_RECIPES and b - a < FAR_SHORT * clip["length_s"] and not _told_once(job, "far_short"):
        return FAR_SHORT_SAID % (i, round(b - a), clip["length_s"]) + (
            RISE_SAID % recording.clock(rise) if past and not _told_once(job, "rise_told") else "")
    if clip["recipe"] == "vlog" and past and not _told_once(job, "rise_told"):
        return RISE_ALONE % (i, recording.clock(b), recording.clock(rise), clip["length_s"])
    if look.get("seen_past"):
        return None
    whole = a - look["start"] <= WHOLE_LOOK and look["end"] - b <= WHOLE_LOOK
    before = ((b - a < GAME_SHORT * clip["length_s"] or whole) and not _told_once(job, "seen_back")) if game else (
        m.get("start") and a - look["start"] <= EDGE)
    after = not game and m.get("end") and look["end"] - b <= EDGE
    if not before and not after:
        return _read_cold(look, i, m, a) if clip["recipe"] in COLD_RECIPES else None
    clock = recording.clock
    start = max(0.0, look["start"] - SEEN_BACK) if before else look["start"]
    end = min(rec["seconds"], look["end"] + SEEN_ON) if after else look["end"]
    looks = job.chat.looks
    look["seen_past"] = True
    if any(s["start"] <= start and end <= s["end"] for s in looks):
        return None
    had = len(looks)
    seen = look_at_piece(job, why, clock(start), clock(end))
    if len(looks) == had:
        return seen
    looks[-1]["seen_past"] = True
    if game:
        return ("Not proposed yet: moment %d runs %g s of the %g s this clip can hold, and you have not seen what led up "
                "to it. The program looked back for you.\n%s\nPropose again from look %d: move the start back to where "
                "what leads to the moment begins, if that tells it better, and keep the end on the peak. A look's end is "
                "where the look stopped, not the peak: end just after the reaction." % (
                    i, round(b - a), clip["length_s"], seen, looks[-1]["n"]))
    edges = " and ".join(([clock(look["start"])] if before else []) + ([clock(look["end"])] if after else []))
    return ("Not proposed yet: moment %d sits at the edge of look %d (%s), so you have not seen what is said on the other "
            "side. The program looked on for you.\n%s\nPropose again from look %d: keep the moment's start and end if it "
            "really starts and lands there, or move them to where the thought starts and to after its point and the "
            "reaction." % (i, look["n"], edges, seen, looks[-1]["n"]))


def _read_cold(look, i, m, a):
    """What Muse reads again when moment i, m as Muse gave it, starts at a inside what Muse saw: the words a stranger
    would miss, else None. Once for a look. Glimmer started EP 0's story at 18:21, 3 of 3, mid-story with its look
    open from 18:00, and two recipe lines did not move it; shown the words before that start it moved to 17:48 and
    17:49 (Tarm: the moment first, a good first draft). A vlog is left out: an episode talks on, so it ran EP 22's
    joke back to 25:19 and 25:26, 2 of 2, past where it starts."""
    said = [p for p in look.get("said") or [] if p["to"] > p["from"]]
    missed = [p for p in said if a - SEEN_BACK < p["to"] <= a + AIR]
    opens = next((p for p in said if p["from"] >= a - AIR), None)
    if not m.get("start") or not missed or opens is None:
        return None
    look["seen_past"] = True
    clock = recording.clock
    lines = "\n".join("%s %s" % (clock(p["from"]), " ".join(p["text"].split())) for p in missed)
    return ("Not proposed yet: a stranger who never saw this recording starts moment %d at %s, on \"%s\", and has not "
            "heard what is said just before it:\n%s\nPropose again: keep the start only if a stranger follows the first "
            "line with nothing before it. If it leans on anything above (a person, a place, what happened, a story already "
            "going), move the start back to the line where this story or thought begins." % (
                i, clock(a), " ".join(opens["text"].split())[:80], lines))


# Glimmer took the 3 loudest stretches of Tarm's eFootball match and found 1 goal of 4, 3 of 3: the crowd rises only for
# its own side. Spark read the score across its looks and found all 4, 3 of 3. So when the looks have seen little of a
# match, the program puts pictures of the whole of it in front of Muse, once, and Muse reads the score in them itself.
ACROSS_PICTURES = 8
ACROSS_SEEN = 0.25
ACROSS_LEAST = 180.0


def _read_across(job, rec):
    """What Muse is told, with ACROSS_PICTURES pictures of the whole recording, before the moments of a sports match are
    proposed from looks that saw under ACROSS_SEEN of it, else None. Once for a clip."""
    clip, clock = job.chat.clip or {}, recording.clock
    seen = sum(s["end"] - s["start"] for s in job.chat.looks)
    if clip.get("recipe") != "sport" or clip.get("read_across") or rec["seconds"] < ACROSS_LEAST \
            or seen >= ACROSS_SEEN * rec["seconds"]:
        return None
    clip["read_across"] = True
    gap = rec["seconds"] / ACROSS_PICTURES
    for k in range(ACROSS_PICTURES):
        t = (k + 0.5) * gap
        frame = recording.frame(rec, at=t)
        if frame:
            job.attach(os.path.join(recording.WORK, "frames", frame), "The match at %s:" % clock(t))
    return ("Not proposed yet: your looks saw little of the match, and the crowd does not rise for a goal against its side. "
            "%d pictures across the whole match come next, one every %s. Read the score in each: where it differs between "
            "two pictures, a goal lies between them, so look at that whole stretch, from the earlier picture on, until "
            "you see it. The last picture's score says how many goals there are: propose every one."
            % (ACROSS_PICTURES, clock(gap)))


# Spark Open wrote 0-1, 0-2, 0-3, 1-3 over a match the board has at 0-1, 0-2, 1-2, 1-3 (11-01, 022837-3062): just after a goal
# the picture is a replay with no board, so it worked the score out. As each attack starts the board shows: asked cold, it
# read the board right 7 of 8 times at those seconds, and 4 of 4 from 30 s after the last goal.
# Glimmer's sport held 4, 3 and 2 of 4 goals across runs (11-01): 30 s after its own last goal the board agreed with it.
# 20 and 30 s from the end of that match are a replay with no board, so there are three near the end.
BOARD_SIDE, BOARD_AFTER, BOARD_END = 1024, (30.0, 45.0), (60.0, 40.0, 20.0)
READ_BOARD = ("Nothing was joined yet. The pictures that come next show the match as each piece's attack starts and after the "
              "last goal. Read the score on the board in each picture yourself: the change from one piece's picture to the "
              "next is that piece's goal, and it tells which side scored. A picture with no board tells nothing. Write each "
              "piece's words with the score the board shows after its goal, in the board's order: the left number is the left "
              "team's goals in every piece, whoever leads. Then call make_countdown again, with final set to the score in the "
              "last picture near the end of the recording that shows a board: if it is more than your last piece's, a goal is "
              "missing, so look for it and propose it first.")


def _board_pictures(job, rec, times):
    for t in times:
        frame = recording.frame(rec, at=t, side=BOARD_SIDE)
        if frame:
            job.attach(os.path.join(recording.WORK, "frames", frame), "The match at %s:" % recording.clock(t))


def _read_the_board(job, rec, spans):
    """READ_BOARD, with pictures of the match at the start of each of spans, after the last and near the end of the recording,
    at most ACROSS_PICTURES."""
    last = spans[-1][1]
    after = [last + wait for wait in BOARD_AFTER] + [rec["seconds"] - wait for wait in BOARD_END]
    after = [t for t in after if last < t < rec["seconds"]]
    _board_pictures(job, rec, ([a for a, _ in spans] + sorted(set(after)))[-ACROSS_PICTURES:])
    return READ_BOARD


# A seller shows one product after another, and a batch holds only what Muse's looks saw. Lane 5 10-31, 153416-7e05, 5 products
# in 2:20: three looks of 30 s from the map missed the balm, and the Redmi and the Samsung went into one clip, graded 6.
PRODUCT_RECIPES = ("seller",)
ALL_SEEN_LONGEST, UNSEEN_LEAST, OPENING_ROOM = 600.0, 8.0, 3.0
NOT_SEEN = ("Not proposed yet: a seller shows one product after another, and your looks had not seen %s of this recording, so "
            "the program looked there for you.\n%s\nPropose again, a moment for each product you have seen, from these looks "
            "too. If they show no other product, propose again and it stands.")
NEXT_PRODUCT = re.compile(r"สินค้าชิ้น(?:ที่|ต่อไป|สุดท้าย)|ชิ้นต่อไป|ชิ้นสุดท้าย|ต่อไปเป็น|next (?:product|item)|last (?:product|item)", re.I)
ONE_PRODUCT = ("Not proposed yet: a clip holds one product, and %s. Start the moment at that line or end it before that line, "
               "and give the other product its own moment. If it is still the same product, propose again and it stands.")


def _not_looked(looks, seconds):
    """The stretches [(from, to)] of a recording of seconds, UNSEEN_LEAST s or more, that no look saw."""
    gaps, at = [], 0.0
    for a, b in sorted((s["start"], s["end"]) for s in looks):
        if a - at >= UNSEEN_LEAST:
            gaps.append((at, a))
        at = max(at, b)
    return gaps + ([(at, seconds)] if seconds - at >= UNSEEN_LEAST else [])


def _all_seen(job, why, rec, moments):
    """What Muse sees of a seller's recording that no look has shown, when it proposes several moments, else None. Once a
    job, and not on a long live, where looking at everything is no answer."""
    if (job.chat.clip or {}).get("recipe") not in PRODUCT_RECIPES or len(moments) < 2 or rec["seconds"] > ALL_SEEN_LONGEST:
        return None
    gaps = _not_looked(job.chat.looks, rec["seconds"])
    if not gaps or _told_once(job, "all_seen"):
        return None
    clock = recording.clock
    seen = [look_at_piece(job, why, clock(a), clock(b)) for a, b in gaps]
    return NOT_SEEN % (" and ".join("%s-%s" % (clock(a), clock(b)) for a, b in gaps), "\n".join(seen))


def _two_products(job, looks, picked):
    """Each of a seller's moments with another product's opening inside it, past its first OPENING_ROOM s, as sentences."""
    if (job.chat.clip or {}).get("recipe") not in PRODUCT_RECIPES:
        return []
    clock, out = recording.clock, []
    for p in picked:
        line = next((s for s in looks[p["look"] - 1].get("said") or [] if p["start"] + OPENING_ROOM < s["from"] < p["end"]
                     and NEXT_PRODUCT.search(s["text"])), None)
        if line:
            out.append('moment %d (%s-%s) has another product\'s opening at %s: "%s"' % (
                p["n"], clock(p["start"]), clock(p["end"]), clock(line["from"]), line["text"]))
    return out


NO_RECIPE = ("Not proposed yet: no recipe is loaded. Your load_recipe came before the recording was open, so it was not "
             "done. Call load_recipe now for what you saw, then propose these moments again.")


def _no_recipe(job):
    """True once a job when its load_recipe was refused for coming before the recording and none was loaded since. 10-31:
    045116-b599 and 045946-c52f did so on Glimmer and made a clip with no recipe: a slide cut at both sides, and a seller's
    phone never in the frame while the product check said this was not a seller's clip."""
    return (not job.chat.clip and any(e["kind"] == "recipe_blind" and e.get("job") == job.n for e in job.chat.events)
            and not _told_once(job, "no_recipe"))


def propose_moments(job, why, moments, then="make"):
    rec = _open(job, "propose")
    clock, looks = recording.clock, job.chat.looks
    if not looks:
        raise Refused("You have not looked at any stretch yet. Look first: a moment you propose must be one you saw.")
    if _no_recipe(job):
        raise Refused(NO_RECIPE)
    wanted = job.chat.wanted
    if wanted:
        then = "wait"
        if _too_few(len(moments), wanted) and not _told_once(job, "few_moments"):
            raise Refused(FEW_MOMENTS % (wanted, len(moments)))
    picked, gave, finer = [], [], []
    for i, m in enumerate(moments, 1):
        if not 1 <= m["look"] <= len(looks):
            raise Refused("Moment %d names look %d, but your looks are: %s. Name one of them, or look first."
                          % (i, m["look"], "; ".join("%d. %s-%s" % (s["n"], clock(s["start"]), clock(s["end"]))
                                                      for s in looks)))
        look = looks[m["look"] - 1]
        a = _time("moment %d's start" % i, m["start"]) if m.get("start") else look["start"]
        b = _time("moment %d's end" % i, m["end"]) if m.get("end") else look["end"]
        lo, hi = look.get("heard") or (look["start"], look["end"])
        if round(a) < round(lo) or round(b) > round(hi) or b <= a:
            raise Refused("Moment %d (%s-%s) is not inside look %d (%s-%s). Give a start and an end inside what you "
                          "saw, or look at that stretch first." % (i, clock(a), clock(b), look["n"],
                                                                   clock(look["start"]), clock(look["end"])))
        a, b = max(a, lo), min(b, hi)
        gave.append((a, b))
        seen = (_read_across(job, rec) or _seen_past(job, why, rec, i, m, look, a, b)) if i == 1 and not wanted else None
        if seen:
            return seen
        try:
            finer += [] if wanted else _finer_edges(job, rec, look, a, b)
        except hearing.Stopped:
            return "Not proposed: the person pressed Stop."
        a, b, snapped = _snapped(look, a, b, (job.chat.clip or {}).get("recipe") == "seller")
        for other, (x, y) in zip(picked, gave):
            if y <= gave[-1][0] and a < other["end"]:
                a = other["end"]
            elif gave[-1][1] <= x and other["start"] < b:
                b = other["start"]
            elif a < other["end"] and other["start"] < b:
                raise Refused("Moments %d and %d overlap (%s-%s and %s-%s). Each moment must be its own stretch: move "
                              "one, or keep only the better." % (other["n"], i, clock(other["start"]),
                                                                  clock(other["end"]), clock(a), clock(b)))
        if round(rec["seconds"]) < SHORTEST:
            raise Refused(TOO_SHORT % (rec["name"], round(rec["seconds"], 1), SHORTEST) + " Do not propose it again.")
        if round(look["end"] - look["start"]) < SHORTEST:
            raise Refused("Look %d (%s-%s) is shorter than the %d s a clip needs, so no moment fits inside it. Look at "
                          "a stretch of %d s or more first." % (look["n"], clock(look["start"]), clock(look["end"]),
                                                               SHORTEST, SHORTEST))
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
        picked.append({"n": i, "look": look["n"], "start": round(a, 2), "end": round(b, 2), "reason": m["reason"],
                       "snapped": snapped,
                       "frame": shown["frame"], "t": shown["t"]})
    unseen = _all_seen(job, why, rec, moments)
    if unseen:
        return unseen
    two = _two_products(job, looks, picked)
    if two and not _told_once(job, "two_products"):
        raise Refused(ONE_PRODUCT % "; ".join(two))
    if finer:
        raise Refused(FINER % "\n".join(finer))
    if _reads_screen(job) or (job.chat.clip or {}).get("recipe") == "game":
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
    unclear = _unclear(job, picked[0], looks) if then == "make" else None
    if unclear:
        raise Refused(unclear)
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
    if then == "wait":
        job.emit(WAITING)
        job.chat.wanted = None
        lines.append("The person asked to see the moments first, so no clip is made in this job: answer now, in one or "
                     "two short lines, that the cards are there to look at, and stop. They tap a card when they want a "
                     "clip from it. Do not list the moments again in words.")
        return "\n".join(lines)
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
        return None, ("No face was found inside your face box in %d pictures, so the layout goes without it: a circle "
                      "needs a face in it." % n)
    g = faces.steady(found)
    if g:
        return g["box"], ("A face sits at the same place in %d of %d pictures, at %s on your grid; the layout uses it."
                          % (g["seen"], n, _grid(g["box"], rec)))
    best = next(iter(faces.places(found)), None)
    if best and best["seen"] >= MOVING_FACE * n:
        return best["box"], ("The face moves; it is most often at %s on your grid (%d of %d pictures), and the layout "
                             "uses that place." % (_grid(best["box"], rec), best["seen"], n))
    if best:
        return None, ("No face stays in the picture for the layout to follow: faces come and go, the one seen most in "
                      "%d of the moment's %d pictures." % (best["seen"], n))
    return None, "No face was found in the moment's %d pictures." % n


VLOG_WHOLE = "A vlog keeps its whole picture: the card has no face circle, as nobody looks for one."


def _vlog_trim(job, rec, a, b):
    """(the part of a vlog's picture its card shows, or None for the whole picture, a sentence for Muse): its sides
    cut as far as layouts.trimmed() lets the faces seen from a to b stay. faces.Stopped when the person pressed
    Stop."""
    try:
        found = job.find_faces(rec, a, b)
    except faces.NoFaces as e:
        return None, VLOG_WHOLE + " Faces could not be looked for (%s), so its sides are not cut." % e
    boxes = [f[:4] for s in found["samples"] for f in s["faces"] if f[4] >= faces.SCORE]
    part, share = layouts.trimmed((rec["width"], rec["height"]), boxes)
    if part[2] >= rec["width"]:
        return None, VLOG_WHOLE
    return part, ("A vlog keeps its whole height and the subtitles an editor centres: the card shows %s on your grid, "
                  "its sides cut as far as it still holds %d%% of the faces the whole picture shows in the moment, so "
                  "the people are %.1f times as big. The card has no face circle, as nobody looks for one. To show "
                  "the whole picture, make the clip again with crop [0, 0, 1000, 1000]."
                  % (_grid(part, rec), int(round(share * 100)), rec["width"] / float(part[2])))


PRODUCT_NEAR = 0.5
PRODUCT_FILLS = 0.5


def _found_faces(rec, pieces, span=None):
    """[(a, b, the faces found from a to b, or None when nobody looked)]: the faces found over span, the moment the
    clip was made from, as one stretch; or over each piece, for a clip kept from before clips were cut into pieces.
    The faces are looked for over the whole moment, so a piece alone was never looked at: asked piece by piece, a
    clip with its silences cut knew no face, and words could go on one."""
    whole = faces.kept(rec, *span) if span else None
    return [(span[0], span[1], whole)] if whole else [(a, b, faces.kept(rec, a, b)) for a, b in pieces]


def _face_times(state, rec):
    """[(clip second, [boxes in the frame to keep words off])] for every picture the face finder took of the clip,
    and the seconds between two pictures. The product Muse named is in every picture, where its box is. On a frame that
    follows it, it is where SAM saw it at that second, and nowhere where SAM did not; with no pictures from SAM, in the
    frame's middle. As the box at every place that frame goes it filled both places for words, and 2 of 4 clips of
    Tarm's seller kept a title for under 2 s (10-31); as one box in the frame's middle, held in hands that move, it
    filled them too, and 2 of 5 clips of the seller had no title and no price (11-01, 005549-f1e8: top clear 0 s, now 9.5
    and 6.5 s). A product over PRODUCT_FILLS of the frame cannot be kept off, as that iPad at 78%: the title goes on it."""
    clip, out, every = state["clip"], [], faces.EVERY
    lay = clip["layout"]
    seen = focus.thing_boxes(lay["thing"], clip["pieces"], (rec["width"], rec["height"])) if lay.get("thing") and \
        clip.get("product") else []

    def on_frame(at, box):
        return layouts.to_frame(lay, box, at)

    def product(at):
        if not clip.get("product"):
            return []
        if not lay.get("thing"):
            return layouts.to_frame(lay, clip["product"])
        if seen:
            t, box = min(seen, key=lambda s: abs(s[0] - at))
            return [b for b in on_frame(at, box) if b[2] * b[3] < PRODUCT_FILLS * layouts.W * layouts.H] \
                if abs(t - at) <= PRODUCT_NEAR else []
        whole = next(p for p in layouts.parts_at(lay, at) if p["name"] == "whole")
        x, y, w, h = clip["product"]
        return on_frame(at, [whole["src"][0] + (whole["src"][2] - w) / 2.0, y, w, h])

    if clip.get("product"):
        out = [(round(k * every, 2), list(product(k * every))) for k in range(int(state["length"] / every) + 1)]
    for a, b, found in _found_faces(rec, clip["pieces"], clip.get("span")):
        if not found:
            continue
        every = found["every"]
        for t, boxes in faces.between(found, a, b):
            at = project.from_recording(clip["pieces"], t)
            if at is not None:
                out.append((at, [f for box in boxes for f in on_frame(at, box)] + product(at)))
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


# The sizes a title clear of faces tries, biggest first: each new size is a libass measure of about 0.2 s, so a
# montage's seven labels take a second or two, not fifteen.
GROWS = (render.TITLE_GROW, 1.4, 1.2)


def _grown(state, default, words, band, times, every, a, b, readable=None):
    """(grow, its Measure, the box it draws in) for a title that covers no face at its own size in band from a to b:
    the first of GROWS at which band still holds it and it still covers no face, or (1, None, None) when none does or
    the person chose a size. Over captions shown one line at a time render draws it in their letters as tall. A title
    behind the person grows with no faces given, since they stand in front of it: on the PS5 seller (10-30) no place
    was clear of the face and the product, and the title went behind at its own size, 477 x 62 px, under the hands.
    There readable(box) says whether the person still leaves the words readable: grown about its centre a title can
    reach further over them ("Go now" beside a person's edge: 32% hidden at its own size, 39% at TITLE_GROW)."""
    own = render.style_of(state, default, "title").sizes[0]
    for grow in GROWS:
        st = render.grown_style(state, default, grow)
        if st.sizes[0] <= own:
            break
        size = text.measure(words, st, "title", band[2])
        drawn = [band[0] + (band[2] - size.w) / 2, band[1] + (band[3] - size.h) / 2, size.w, size.h]
        if size.w <= band[2] and size.h <= band[3] and not _covered(drawn, times, every, a, b) and (
                readable is None or readable(drawn)):
            return grow, size, drawn
    return 1, None, None


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
    product = _product_box(state)
    lines = []
    for name, box in lay["places"].items():
        clear = _clear(_covered(box, times, every, 0, length), 0, length)
        where = "%s (%s)" % (name, layouts.on_grid(box[1], box[1] + box[3], "y"))
        opening = None if clear else _longest(0, length, every, lambda t: max(
            (f for f in _free_x(box, times, every, 0, t, product) if f[1] - f[0] >= PART_LEAST),
            key=lambda f: f[1] - f[0], default=None))
        if clear == [(0, length)]:
            lines.append("- %s: clear of faces for the whole clip" % where)
        elif clear:
            lines.append("- %s: clear of faces at %s" % (where, _spans(clear, inward=True)))
        elif opening:
            lines.append("- %s: never clear across its width, but from 0 to %g s its part from %s is: a title from the "
                         "first frame given this place goes there when it fits" % (
                             where, math.floor(opening[0] * 10) / 10,
                             layouts.on_grid(math.ceil(opening[1][0]), math.floor(opening[1][1]), "x")))
        else:
            lines.append("- %s: never clear for %g s or more; do not use it" % (where, LISTED))
    return lines


# The narrowest part of a place worth naming: "นั่งไกล เสียงยังชัด" stands in it in two lines at the title size.
PART_LEAST = 300


def _product_box(state):
    clip = state["clip"]
    return layouts.to_frame(clip["layout"], clip["product"]) if clip.get("product") else []


def _free_x(band, times, every, a, b, product=()):
    """[(x from, x to)] of band, its whole height, that no face or product box (FACE_MARGIN around it) crosses from a
    to b. None of a band the product's box crosses: the product moves in its shot and its box is where it mostly is
    (on the seller beat, 10-29, the dehumidifier was carried over the part of the middle place left of its box)."""
    x, y, w, h = band
    if any(p[1] < y + h and y < p[1] + p[3] for p in product):
        return []
    taken = sorted((f[0] - FACE_MARGIN, f[0] + f[2] + FACE_MARGIN) for t, boxes in times
                   if a - every / 2 < t < b + every / 2 for f in boxes
                   if f[1] - FACE_MARGIN < y + h and y < f[1] + f[3] + FACE_MARGIN)
    free, at = [], x
    for lo, hi in taken:
        if lo > at and at < x + w:
            free.append((at, min(lo, x + w)))
        at = max(at, hi)
    if at < x + w:
        free.append((at, x + w))
    return free


def _longest(a, b, every, fits):
    """(t, what fits(t) gave) for the latest t from a + LISTED to b, a step of every apart, at which fits(t) still
    gives something, or None: what fits from a to t only shrinks as t grows."""
    ts = [t for t in (round(a + LISTED + k * every, 2) for k in range(int(max(0, b - a - LISTED) / every) + 1))
          if t < b] + [b]
    lo, hi, best = 0, len(ts) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        got = fits(ts[mid])
        if got:
            best, lo = (ts[mid], got), mid + 1
        else:
            hi = mid - 1
    return best


def _clear_part(band, words, st, key, times, every, a, b, product=()):
    """(box, size): the widest part of band, its whole height, that no face or product crosses from a to b and that
    holds the words at the title size; None when no part does. On the seller beat (10-29) the face filled the middle
    of the top place, so a title centred in it covered the face while the left of it was clear, and the clip went
    out with no title."""
    for lo, hi in sorted(_free_x(band, times, every, a, b, product), key=lambda f: f[0] - f[1]):
        part = [int(math.ceil(lo)), band[1], int(math.floor(hi) - math.ceil(lo)), band[3]]
        try:
            size = text.measure(words, st, "title", part[2])
            text.title(words, a, b, part, st, key)
        except ValueError:
            continue
        if size.w <= part[2] and size.h <= part[3]:
            return part, size
    return None


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


def make_clip(job, why, moment, layout, crop=None, face=None, product=None, follow=True, moved=False, titled=()):
    rec = _open(job, "make a clip from")
    if _waits(job):
        raise Refused("Not made: the person asked to see the moments first, and they are on the cards. Answer them now "
                      "and stop; they tap a card when they want a clip from it.")
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
    if not moved and job.n is not None and why != BATCH_WHY:
        job.chat.album_to(moment)
    made = {"layout": layout, "crop": crop, "face": face, "product": product, "follow": follow}
    kept = {"made": made, "moved": True} if moved else {"made": made}
    try:
        _fix_moment(job, rec, m, titled)
    except hearing.Stopped:
        return "Not made: the person pressed Stop."
    recipe = (job.chat.clip or {}).get("recipe")
    unasked = None if moved else (_layout_unasked(job, recipe, [layout]) or _menu_start(job, rec, a)
                                  or _one_goal(job, why, moments))
    if unasked:
        raise Refused("Not made. " + unasked)
    level = None if recipe == MONTAGE else takes.heard_level(rec, a, b)
    if layout in ("talk", "speaker", "split"):
        return _make_talk(job, rec, moment, m, *_on_quiet(rec, _pieces(
            m.get("said") or [], a, b, _still(rec, "talk", a, b, None), takes.heard_claps(rec, a, b),
            edges=not m.get("ends"), level=level), level, _reaches_on(recipe, m)), name=layout, kept=kept)
    trim = None
    try:
        if recipe == "vlog" and layout == "card" and not face:
            box = None
            trim, face_line = _vlog_trim(job, rec, a, b) if not crop else (None, VLOG_WHOLE)
        else:
            box, face_line = _the_face(job, rec, a, b, face)
    except faces.Stopped:
        return "Not made: the person pressed Stop."
    try:
        lay = layouts.make(layout, (rec["width"], rec["height"]), box,
                           trim or (_pixels("crop", crop, rec) if crop else None),
                           layouts.VLOG_TALLEST if trim else None, tall=recipe == "game")
    except ValueError as e:
        raise Refused(str(e))
    still = _still(rec, recipe, a, b, layouts.face_room(lay))
    pieces, cut = _on_quiet(rec, _pieces(m.get("said") or [], a, b, still, takes.heard_claps(rec, a, b) if still else [],
                                         edges=not m.get("ends"), level=level), level, _reaches_on(recipe, m))
    menus = None
    if recipe == "game":
        try:
            pieces, menus = _menus_cut(job, rec, pieces, a, b)
        except screen.Stopped:
            return "Not made: the person pressed Stop."
    following = ground = ""
    if lay["kind"] == "card" and follow and not crop and not trim and _reads_screen(job):
        lay, following = _follow(rec, lay, pieces, [a, b])
    if lay["kind"] == "subject" and recipe == "sport" and follow and not crop:
        lay, following = _follow_play(job, rec, lay, pieces)
    if lay["kind"] == "card" and _reads_screen(job):
        lay, ground = _plain(job, rec, lay, a, b)
    item = [round(v) for v in _pixels("product", product, rec)] if product else None
    if item and not layouts.to_frame(lay, item):
        lay["notes"].append("The product you named is not in the clip's picture: choose a crop that holds it.")
    if not item and (job.chat.clip or {}).get("recipe") == "seller":
        lay["notes"].append("This is a seller clip and no product box was named, so nothing keeps words off the "
                            "product. Make the clip again with product: [x1, y1, x2, y2] on your grid around the product "
                            "in the picture.")
    if (job.chat.clip or {}).get("recipe") == "seller" and lay["kind"] == "subject" and rec["width"] > rec["height"]:
        lay["notes"].append(FOLLOW_SAYS)
    private = _private_lines(job, rec, lay, pieces, [a, b])
    if private is None:
        return "Not made: the person pressed Stop."
    people = _people(rec, pieces, box, [a, b], job=job)
    state = job.edit(dict({"do": "clip", "moment": moment, "pieces": pieces, "span": [a, b], "layout": lay,
                           "product": item, "people": [{k: v for k, v in p.items() if k != "crop"} for p in people or []],
                           "speech": _speech(rec, job.chat.looks[m["look"] - 1], pieces)}, **kept))
    for f in private["shown"]:
        state = job.edit({"do": "blur", "box": f["box"], "start": f["start"], "end": f["end"], "private": f["kind"]})
    channel = _channel_on(job)
    state = job.chat.project.state()
    lines = ["Made the clip from moment %d (%s-%s, %g s) in the %s layout." % (
        moment, recording.clock(a), recording.clock(b), state["length"], lay["kind"]), face_line] + lay["notes"]
    card = [p["dst"][1] for p in lay["parts"] if p["name"] == "card"]
    if recipe == "explain" and card and not any(p["name"] == "circle" for p in lay["parts"]):
        lines.append(EMPTY_ABOVE % layouts.on_grid(0, min(card), "y"))
    try:
        if green.find(rec["path"], a, b, (rec["width"], rec["height"]), samples=2):
            lines.append(GREEN_SAYS)
    except green.GreenError as e:
        lines.append("A green cloth is behind the person, but: %s" % e)
    if channel:
        lines.append(channel)
    if cut:
        lines.append(cut)
    if menus:
        lines.append(menus)
    if following:
        lines.append(following)
    if ground:
        lines.append(ground)
    if private["said"]:
        lines.append(private["said"])
    if lay.get("safe"):
        lines.append(lay["safe"]["said"])
    lines.append("Places for words on screen on your grid, kept clear of the faces the program found%s (the band "
                 "from %s is kept for the captions):" % (" and the product you named" if item else "", layouts.on_grid(
                     lay["captions"][1], lay["captions"][1] + lay["captions"][3], "y", inward=False)))
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
    if people and recipe in EVENT_RECIPES and not _blur_asked(job):
        state = job.edit({"do": "people", "blur": [], "keep": [p["n"] for p in people]})
        lines.append(KEPT_AT_AN_EVENT % len(people))
    else:
        lines += _people_lines(job, people, rec)
    lines.append("Look at it: if a face or the action is cut badly, make the clip again with another layout or crop. "
                 "Then put the title on.")
    job.show(clip=job.chat.clip_view())
    return "\n".join(lines)


def _frame_kept(job, rec, old, new):
    """After change_ends made the clip again from how it was made: the upright frame of a subject clip where it was,
    and the sentence that says so, or None. A frame that followed a product follows it again over the new ends from
    SAM's first pass (the layout's thing, focus.thing), and a frame placed elsewhere (the person's, on the Frame tab)
    is put back there, as the shots of a talk clip that were given a place are. 10-30, Glimmer's green seller: the ends
    moved, the frame went back to the face, the title had no clear place left, and the clip went out without one."""
    before, after = old["layout"], new["layout"]
    if before["kind"] == after["kind"] == "talk":
        shots = talk.crops_kept(before, old["pieces"], after, new["pieces"])
        if shots == after["shots"]:
            return None
        job.edit({"do": "layout", "part": "shots", "box": None, "layout": dict(after, shots=shots)})
        return "The shots whose frame was moved stay where they were put over the recording."
    whole = next((p for p in before["parts"] if p["name"] == "whole"), None)
    if before["kind"] != "subject" or after["kind"] != "subject" or not whole:
        return None
    thing = before.get("thing")
    if thing:
        face = after.get("face")
        plan = focus.thing(thing, new["pieces"], (rec["width"], rec["height"]), layouts.W / float(layouts.H),
                           face[0] + face[2] / 2.0 if face else None)
        if not plan:
            return ("SAM's pass did not see the %s in the new ends, so the frame no longer follows it: call "
                    "follow_product again if it should." % thing["what"])
        job.edit({"do": "layout", "part": "product", "box": None,
                  "layout": dict(layouts.followed(after, plan, "whole"), thing=thing)})
        return "The frame follows the %s again over the new ends, from SAM's first pass." % thing["what"]
    if whole.get("follow"):
        return None if any(p.get("follow") for p in after["parts"]) else (
            "The frame no longer follows what it followed: call follow_product again if it should.")
    parts = [dict(p, src=list(whole["src"])) if p["name"] == "whole" else p for p in after["parts"]]
    if parts == after["parts"]:
        return None
    job.edit({"do": "layout", "part": "frame", "box": None, "layout": dict(after, parts=parts)})
    return "The frame stays where it was put over the recording."


FOLLOW_SAYS = ("This is a seller clip from a landscape recording, so the frame stays where the face is while the product "
               "moves in the hands: next, call follow_product with the product in a few English words, so the frame "
               "follows the product. The seller sells the product, not their face.")
GREEN_SAYS = ("A green cloth is behind the person: put a new background behind them with put_background, a place that "
              "suits what the clip is about.")
# An explainer's card with no face circle leaves the frame above it to the title alone: Spark loaded the recipe's
# Pictures line and still left that third empty (10-30, 154248-dd25), so make_clip says where it is and what fills it.
EMPTY_ABOVE = ("With no face circle, the frame above the card (%s on your grid) holds only the title: when the words "
               "name a thing a picture can show, such as gold bars or a price going up, put a picture of it there with "
               "add_picture at the second it is said, 1 to 3 in a clip.")
EVENT_RECIPES = (MONTAGE, "sport")
KEPT_AT_AN_EVENT = ("%d people's faces are in the clip, all kept as they are: at a trip, an event or a match the people are "
                    "part of the day. Blur someone only when the person asks: show_people numbers the faces at the "
                    "second they mean, and decide_people blurs that number.")


def change_ends(job, why, start=None, end=None):
    """The clip's moment with the start or the end moved where the person wants it, made again as it was made (the
    layout, crop, face and product make_clip was given), its edges where they were put, or in the nearest quiet sound
    when that is inside a word (_on_quiet), and the result says where they went. What sits on the footage
    stays on it (project.state, a moved clip), so a cut-off start takes its captions with it, and a start taken
    earlier is heard first when the look never heard it. The new moment is one more in the list, so Undo puts the
    clip back as it was and the moment it was made from is still there."""
    rec = _open(job, "change the ends of")
    state = _clip_state(job)
    clip, tenth = state["clip"], _tenth
    moments, looks = job.chat.moments or [], job.chat.looks
    if not clip.get("made"):
        raise Refused("This clip's ends cannot be moved: it was made before the program kept how it was made. Make the "
                      "clip again, then move its ends.")
    if not 1 <= clip["moment"] <= len(moments):
        raise Refused("The moment this clip was made from is no longer proposed, so its ends cannot be moved. Make the "
                      "clip again from one of the moments.")
    if start is None and end is None:
        raise Refused("Give the new start, the new end, or both.")
    m = moments[clip["moment"] - 1]
    now_a, now_b = clip["pieces"][0][0], clip["pieces"][-1][1]
    a = _time("start", start) if start is not None else now_a
    b = _time("end", end) if end is not None else now_b
    a, b = round(max(0.0, a), 2), round(min(rec["seconds"], b), 2)
    if b <= a:
        raise Refused("The end, %s, must come after the start, %s." % (tenth(b), tenth(a)))
    if not SHORTEST <= round(b - a) <= LONGEST:
        raise Refused("From %s to %s is %d s, and a clip is %d to %d s. The clip runs from %s to %s now." % (
            tenth(a), tenth(b), round(b - a), SHORTEST, LONGEST, tenth(now_a), tenth(now_b)))
    if abs(a - now_a) < 0.05 and abs(b - now_b) < 0.05:
        raise Refused("The clip already starts at %s and ends at %s." % (tenth(now_a), tenth(now_b)))
    look, heard_line = looks[m["look"] - 1], None
    lo, hi = look.get("heard") or (look["start"], look["end"])
    wider = None
    if a < lo or b > hi:
        wider = dict(look, n=len(looks) + 1, start=round(min(look["start"], a), 2), end=round(max(look["end"], b), 2))
        try:
            said_lines = _what_is_said(job, rec, wider)
        except hearing.Stopped:
            return "Not moved: the person pressed Stop."
        if said_lines and said_lines[0].startswith("What is said here could not be heard"):
            heard_line = said_lines[0]
        look = wider
    said = _ends_said(job, rec, look, m, [x for x in moments if x["look"] == m["look"]], a, b)
    moment = dict(m, n=len(moments) + 1, look=look["n"], start=a, end=b, said=said, ends=True, snapped=None)
    moment.pop("preview", None)
    if wider:
        looks.append(wider)
    moments.append(moment)
    job.chat.emit("ends", job=job.n, moments=moments, look=wider)
    made = clip["made"]
    said_back = make_clip(job, why, moment["n"], made["layout"], made.get("crop"), made.get("face"),
                          made.get("product"), made.get("follow", True), moved=True)
    if job.chat.project.state()["clip"]["moment"] != moment["n"]:
        return said_back
    pieces = job.chat.project.state()["clip"]["pieces"]
    asked, a, b = (a, b), pieces[0][0], pieces[-1][1]
    lines = ["Moved the clip's ends: it now runs from %s to %s of the recording (it ran from %s to %s)." % (
        tenth(a), tenth(b), tenth(now_a), tenth(now_b))]
    shifted = [round(abs(x - y), 2) for x, y in zip((a, b), asked) if abs(x - y) >= 0.01]
    if shifted:
        lines.append(MOVED_CUTS % (len(shifted), "" if len(shifted) == 1 else "s", max(shifted)))
    level = takes.heard_level(rec, a, b)
    loud = _loud_cuts(rec, [a, b], level) if level else []
    lines += [LOUD_CUT % (t, EDGE_REACH) for t in dict.fromkeys(tenth(t) for t in loud)]
    framed = _frame_kept(job, rec, clip, job.chat.project.state()["clip"])
    if framed:
        lines.append(framed)
    if heard_line:
        lines.append(heard_line)
    caps = state["captions"]
    new_words = [p for p in said if not now_a <= p["from"] < now_b]
    if caps and new_words and not caps.get("language"):
        try:
            lines.append(add_captions(job, why, lit=bool(caps.get("lit")), pop=bool(caps.get("pop"))))
        except Refused as e:
            lines.append("The captions were not put on again: %s" % e)
    elif caps and new_words:
        lines.append("The captions are in %s, and the seconds the clip took in have none yet: put the captions on "
                     "again so they are written for them too." % caps["language"])
    return "\n".join(lines + [said_back])


def _ends_said(job, rec, look, m, same_look, a, b):
    """The moment's lines from a to b: the ones already fixed (in m, or in another moment from the same look), and
    the new ones before and after them fixed now, or kept as heard when Muse cannot be asked (a tap on the page)."""
    fixed = {round(p["from"], 2): p for x in same_look for p in x.get("said") or [] if p.get("kept") != LATER}
    known = [p for p in m.get("said") or [] if a <= p["from"] < b]
    first = known[0]["from"] if known else b
    last = known[-1]["to"] - AIR if known else a
    fresh = [fixed.get(round(p["from"], 2), p) for p in look.get("said") or [] if a <= p["from"] < b and (
        p["to"] <= first + AIR or p["from"] >= last)]
    unfixed = [p for p in fresh if "heard" not in p]
    if unfixed:
        done = {round(p["from"], 2): p for p in _fixed_said(job, rec, look, a, b, said=unfixed)}
        fresh = [done.get(round(p["from"], 2), p) for p in fresh]
    return sorted(known + fresh, key=lambda p: p["from"])


def _tenth(t):
    """A time of the recording to the tenth of a second, as 1:20.4 or 1:02:03.5."""
    t = round(t, 1)
    h, rest = divmod(t, 3600)
    return ("%d:%02d:%04.1f" % (h, rest // 60, rest % 60)) if h else "%d:%04.1f" % (rest // 60, rest % 60)


def _follow_play(job, rec, lay, pieces):
    """(the upright frame following a match's play, the sentence for Muse), or the layout as it was (pitch.follow)."""
    whole = next(p for p in lay["parts"] if p["name"] == "whole")
    job.working("Following the play")
    try:
        plan = pitch.follow(rec, pieces, whole["src"][2], whole["src"][1])
    finally:
        job.done_working()
    if not plan:
        return lay, ("The frame holds the middle of the picture: only a game that marks the player in control can be "
                     "followed, and a fixed frame misses what happens at either end. The card shows the whole pitch.")
    plan["size"] = list(whole["src"][2:])
    return layouts.followed(lay, plan, "whole"), ("The frame follows the play: the game's mark over the player in control "
                                                  "is in %d%% of the match's pictures." % round(plan["seen"] * 100))


def _follow(rec, lay, pieces, span):
    """(the card layout following the action on the screen, the sentence for Muse), or the layout as it was."""
    card = next(p for p in lay["parts"] if p["name"] == "card")
    whole = card["src"][2]
    try:
        text = [dict(line, at=r["at"]) for r in screen.readings(rec, span[0], span[1]) for line in r["lines"]]
    except screen.ScreenError:
        text = None
    if focus.unread(text):
        return _slide(rec, lay, pieces, text)
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


def _slide(rec, lay, pieces, text):
    """(the card layout showing the written part of a slide whose lines cannot be read, or the layout as it was, the
    sentence for Muse): the card does not follow lines it cannot read (focus.unread), so it never cuts them."""
    said = ("The screen's words are in a script the reader does not know (%d of %d lines read surely), so the card "
            "does not follow them" % (sum(1 for line in text if line.get("sure", 0) >= focus.SURE), len(text)))
    try:
        box = focus.content(rec, pieces)
    except focus.FocusError as e:
        return lay, "%s: it shows the whole screen, as the screen could not be looked at (%s)." % (said, e)
    if not box:
        return lay, "%s: the screen is written on to its edges, or it is not a slide, so the card shows it whole." % said
    size = (rec["width"], rec["height"])
    part = layouts.framed(box, size)
    try:
        slide = layouts.make("card", size, lay.get("face"), part, tall=bool(lay.get("tall")))
    except ValueError:
        return lay, "%s: it shows the whole screen." % said
    (card,) = [p for p in slide["parts"] if p["name"] == "card"]
    return slide, ("%s: it shows the part of the slide that is written on, %s on your grid, whole and %.1f times as "
                   "big as the whole screen. If it leaves out what matters, make the clip again with a crop around "
                   "it." % (said, _grid(card["src"], rec), rec["width"] / float(card["src"][2])))


# A card sits on a plain ground (layouts.PLAIN) when its screen holds WORDY lines or more at a time: the gold lesson's
# slides 8 to 12, a tool's window 50 and more, a camera 0.3.
WORDY = 3


def _plain(job, rec, lay, a, b):
    """(the card layout on a plain ground when the screen from a to b is full of words, the sentence for Muse), or
    the layout as it was and ""."""
    job.working("Reading the screen's words")
    try:
        looks = screen.readings(rec, a, b)
    except screen.ScreenError:
        return lay, ""
    finally:
        job.done_working()
    if not looks:
        return lay, ""
    many = sum(len(r["lines"]) for r in looks) / float(len(looks))
    if many < WORDY:
        return lay, ""
    return dict(lay, ground="plain"), (
        "The screen holds about %d lines of words at a time, so the card sits on a plain ground of the picture's "
        "colours instead of its blurred copy, which would show those words big behind the card and the title."
        % round(many))


# Lane 4 10-31, 12 match reels with words: Glimmer read the board right 7 of 7, and 4 of those went false when it wrote
# the words again, leader first (094114-7e80: Hull 1-0, Hull 2-0, Arsenal level at 1-2). All 4 go down or stay, Spark 0 of 5.
# 071525-6738 on 11-01: 2-0, 1-2, 1-3 refused once, so 2-0, 1-2, 3-1 stood on the next call, under a title with the wrong winner.
REEL_SCORE = re.compile(r"(?<![\d:])(\d{1,2})\s*-\s*(\d{1,2})(?![\d:])")
REEL_LEVEL = ("เสมอ", "level", "equal", "draw")
SCORE_TRIES = 3
REEL_NO_WORDS = "GOAL"
SCORES_OFF = ("The scores in these words read %s, and a match's score only goes up, one side at a time, in the board's "
              "order: the left number is the left team's goals in every piece, whoever leads. Copy each piece's score off "
              "the board. If these are two matches, or the numbers are not scores, leave them out of the words. Try %d of %d: after "
              "the last the program takes them out.")
SCORES_OUT = ("The scores in the words still could not be a match's, so the program took them out of the words. In the title "
              "and the post text, write only a score you read on the board.")
FINAL_ASKED = ("Give final: the score in the last picture near the end of the recording that shows a board, as the board shows "
               "it. With no board in those pictures, leave it out and call it again.")
# 080148-9959 on 11-01: told of the gap before 0-2, Glimmer took a look of 4 pictures across 5 minutes and called 4:15 the goal, a
# later stretch at the same score, and 1-3 went missing as nothing here knew the final score. The goal was at 3:22.
GOAL_LEFT_OUT = ("Nothing was joined yet. The board reads %s, more goals than your pieces before it show, so a goal between %s "
                 "and %s of the recording is not in the clip. The pictures that come next show that stretch, each with its "
                 "time: the goal lies between the last picture with the lower score and the first with the higher, and a "
                 "later stretch at the same score is not the goal. Look there, propose it from the play that led to it, and "
                 "join it in match order. If no picture's score changes, or the person asked for fewer goals, call "
                 "make_countdown again and it is joined.")
AHEAD = "The last score, %s, in the board's order has %s: the title and the post text say the same."
# 090608-4e21 on 11-01: 0-2, 1-2, 1-3 is right, but "Arsenal ตีเสมอ 1-2" was refused as a score going back, so Glimmer changed
# nothing, the numbers came out on the last try and only the wrong words stood, with no score left for final or a gap to work on.
LEVEL_OFF = ("Nothing was joined yet. These words call the teams level on a score that is not level: %s. The score is fine: keep "
             "it and write again what happened. If the words come back the same, the program puts %s and the score in their place.")
LEVEL_OUT = ("The program put %s and the score over the words that called the teams level at %s, as the score is not level. "
             "In the title and the post text, do not call them level there.")


def _reel_scores(lines):
    """(left, right, the line in lower case) of each of a match reel's lines with a score, in order."""
    return [(int(m[1]), int(m[2]), line.lower()) for line, m in ((s, REEL_SCORE.search(s)) for s in lines) if m]


def _goals_left_out(lines, spans, final=None, seconds=0.0):
    """[(the score, the start, the end)] of the stretches of the recording that a match reel's own scores say hold a goal
    left out: before a piece two or more goals above the piece before it, as a match starts at 0-0, and after the last
    piece when final, the score near the end of the recording of seconds, is above it."""
    gaps, goals, after = [], 0, 0.0
    for line, (a, b) in zip(lines, spans):
        for left, right, _ in _reel_scores([line]):
            if left + right > goals + 1:
                gaps.append(("%d-%d" % (left, right), after, a))
            goals = left + right
        after = b
    for left, right, _ in _reel_scores([final or ""]):
        if goals and left + right > goals:
            gaps.append(("%d-%d" % (left, right), after, seconds))
    return gaps


def _ahead(lines):
    """AHEAD for a match reel's last score, or None with no score."""
    scores = _reel_scores(lines)
    if not scores:
        return None
    a, b = scores[-1][:2]
    return AHEAD % ("%d-%d" % (a, b), "the teams level" if a == b else "the %s team ahead" % ("left" if a > b else "right"))


def _scores_off(lines):
    """The scores in a match reel's lines in order when they cannot be a match's, else None: a later score lower on
    either side or the same as the one before it. Numbers only: a wrong team on a right number passes."""
    scores = _reel_scores(lines)
    back = any(c < a or d < b or (c, d) == (a, b) for (a, b, _), (c, d, _) in zip(scores, scores[1:]))
    return ", then ".join("%d-%d" % s[:2] for s in scores) if back else None


def _level_off(lines):
    """Those of a match reel's lines with a level word on a score that is not level."""
    return [line for line in lines for a, b, low in _reel_scores([line]) if a != b and any(w in low for w in REEL_LEVEL)]


def _one_goal(job, why, moments):
    """The sentence refusing, once a job, a clip of one of a match's moments when Muse proposed more, or None. On Tarm's
    eFootball match Spark joined the goals 3 of 3; Glimmer proposed 3 goals and made a clip of the first, 2 of 2. A card
    the person tapped is that clip, so job.n None is left alone."""
    if (job.chat.clip or {}).get("recipe") != "sport" or len(moments) < 2 or job.n is None or why == BATCH_WHY \
            or _told_once(job, "one_goal"):
        return None
    return ("A match's goals go in one clip, in match order: join these %d moments with make_countdown, each piece with "
            "its start and its end. If the person asked for one goal, make it again and it is made." % len(moments))


def _menu_start(job, rec, a):
    """The sentence refusing, once a job, a game clip that opens on the game's menu, or None. Tarm's beat 1, 10-29: the
    clip started on SIFU's death screen (DEATH, GiveUp, UNLOCK) after the fight it came from, so it never showed the
    death; the look's sound peak was the laugh after it."""
    if (job.chat.clip or {}).get("recipe") != "game":
        return None
    job.working("Reading the screen at the clip's start")
    try:
        shown = screen.menu(rec, a)
    except screen.ScreenError:
        return None
    finally:
        job.done_working()
    if len(shown) < MENU_LINES or _told_once(job, "menu_start"):
        return None
    return ("The clip would open on the game's menu: at %s the screen shows %s, and a viewer scrolls past a menu in "
            "the first second. Start on the play that led to it: look at %s-%s and propose the moment from there, with "
            "only its start moved and its end where it was, so the reaction stays in. If the menu is the point, make "
            "it again and it is made." % (recording.clock(a), ", ".join(shown),
                                                                     recording.clock(max(0, a - LOOK_BACK)),
                                                                     recording.clock(a)))


# A game clip shows the game's menu (a death screen, its skill tree) for countdown.DEATH_SHOWN s, the death landing,
# and goes on where play starts again: the best moment of beat 1 (10-29) held SIFU's death and skill screens in 5 of
# its 12 pictures. A menu shown for less than that and MENU_CUT_LEAST more is left whole.
MENU_CUT_LEAST = 0.5


def _menus_cut(job, rec, pieces, a, b):
    """(pieces without the game's menu after its first countdown.DEATH_SHOWN s, the sentence for Muse or None) for a
    game clip from a to b. The menu shows while MENU_LINES or more of its words are read surely, from the first reading
    that shows them to the next that does not, or to b: one word is a hint play keeps on the screen (Continue). A menu
    the clip opens on is _menu_start's. Stop raises screen.Stopped."""
    job.working("Reading the screen from %s to %s for the game's menu" % (recording.clock(a), recording.clock(b)))
    try:
        looks = screen.readings(rec, a, b, on_start=job.started_work)
    except screen.ScreenError:
        return pieces, None
    finally:
        job.done_working()
    shown, start, words = [], None, set()
    for k, r in enumerate(looks):
        now = _menu_words(r)
        if len(now) >= MENU_LINES:
            if start is None:
                start, words = r["at"], set()
            words |= now
        elif start is not None:
            shown.append([start, _menu_end(looks[k:], words, b)])
            start = None
    if start is not None:
        shown.append([start, b])
    shown = [[x, y] for x, y in shown if a < x and y - x - countdown.DEATH_SHOWN >= MENU_CUT_LEAST]
    if not shown:
        return pieces, None
    kept = takes.without(pieces, [[x + countdown.DEATH_SHOWN, y] for x, y in shown])
    gone = round(sum(y - x for x, y in pieces) - sum(y - x for x, y in kept), 1)
    return kept, ("The program cut %g s of the game's menu out of the clip: %s. The clip shows the first %g s of "
                  "each, the death landing, and goes on where play starts again." % (
                      gone, "; ".join("the menu at %s for %g s" % (recording.clock(x), round(y - x, 1))
                                      for x, y in shown), countdown.DEATH_SHOWN))


def _menu_words(reading):
    return {screen._menu_word(line["text"]) for line in reading["lines"] if line["sure"] >= screen.SURE} - {None}


def _menu_end(rest, words, b):
    """Where play comes back after a menu: from the first of the readings rest, the first that no longer shows the menu
    whole, the first reading with none of the menu's own words, at most MENU_TAIL s on. Sifu's RISE stays alone for 3 s
    while the age goes up and the picture fades in (10-30: beat 1's menus at 35:40 and 36:01, and lane 5's game clip,
    where those seconds were 3 of 18 s that were not play); a word the menu never showed, such as Continue, a hint play
    keeps on the screen, does not hold it."""
    y = rest[0]["at"]
    for r in rest:
        if r["at"] >= y + MENU_TAIL:
            return y + MENU_TAIL
        if not _menu_words(r) & words:
            return r["at"]
    return min(b, y + MENU_TAIL)


def _labels_copied(job, rec, state, texts, what):
    """The sentence refusing, once a job for each what, words on a game clip that copy a label the game writes on its
    screen, or None. Beat 1, 10-29: the title and the post said "ตาย AGE 35 แล้วรีสตาร์ททันที", the screen's Age32
    and the ending, against the recipe's own rule for the title."""
    if (job.chat.clip or {}).get("recipe") != "game":
        return None
    clip = state["clip"]
    a, b = clip.get("span") or (clip["pieces"][0][0], clip["pieces"][-1][1])
    try:
        shown = set(screen.labels(rec, a, b))
    except screen.ScreenError:
        return None
    copied = sorted({w.upper() for t in texts for w in re.findall(r"[A-Za-z]{3,}", t or "")} & shown)
    if not copied or _told_once(job, "labels_" + what):
        return None
    return ("%s copies what the game writes on its screen (%s): a viewer reads it there already, and it tells what "
            "happens before it does. Say it in words a viewer would use, without giving the ending away." % (
                what, ", ".join(copied)))


def _reads_screen(job):
    clip = job.chat.clip
    return bool(clip) and (RECIPES.get(clip["recipe"]) or {}).get("read") == "screen"


NEAR_NAME = 0.4
OWN_WORDS = ("talk", "vlog")
OWN_HALF = 0.5


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
    another way, such as "Elmero" where the screen shows "Elmoro"."""
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


CAPITAL_NAME = re.compile(r"[A-Z0-9]{3,}|[A-Za-z0-9]*[a-z][A-Z][A-Za-z0-9]*")
HEARD_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9฀-๿]*")
SHOWN_WORDS = 12
UNSEEN_THEN = "Write each name as the screen shows it, or leave it out; if the person gave you the name, %s unchanged."


def _unseen_names(job, rec, a, b, text, said=()):
    """([names], [the screen's words from a to b]): the names in text written in capitals (ASTI) or with a capital
    inside (SampleChannel) that the screen does not show from a to b (recording seconds), that were not said (said:
    the lines heard there) and that the person did not write, whatever the recipe (a tap on the page, a job with no
    number, is the person's own words and is not asked). Seen 10-28 on a travel show's
    sponsor read: the pack showed Jabs, PDRN and ASTAXANTHIN, Muse wrote ASTI on three clips' titles and posts, and
    every check passed. ([], []) when the screen cannot be read."""
    names = [w for w in dict.fromkeys(screen.NAME.findall(text)) if CAPITAL_NAME.fullmatch(w)]
    if not names or job.n is None:
        return [], []
    try:
        shown = screen.words(rec, a, b)
    except screen.ScreenError:
        return [], []
    known = " ".join([rec["name"]] + list(said) + [e["text"] for e in job.chat.events if e["kind"] == "person"])
    seen = {w.lower() for w in shown + screen.sites(rec) + screen.NAME.findall(known)}
    heard = {_skeleton(w) for w in HEARD_LATIN.findall(" ".join(said))}
    return [n for n in names if not any(n.lower() == s or n.lower() in s or len(s) >= 4 and s in n.lower()
                                        for s in seen) and not (len(_skeleton(n)) >= 3 and _skeleton(n) in heard)], shown


def _skeleton(word):
    """A Latin word as the hearing may get it wrong: lower case, any Thai letter in it left out, its vowels after the
    first letter left out and a doubled letter once, so iPad, ipd and iแpd are all ipd (lane 5 10-30: a seller's
    "iPad" was heard so, and its title was refused as a name nobody said), while ASTI (ast) is still not ASTAXANTHIN
    (astxnthn)."""
    w = re.sub(r"[^a-z0-9]", "", word.lower())
    return re.sub(r"(.)+", r"", w[:1] + re.sub("[aeiou]", "", w[1:]))


def _unseen_said(names, shown):
    return '%s %s not on the screen there, not said and not written by the person; the screen there shows %s' % (
        ", ".join('"%s"' % n for n in names), "is" if len(names) == 1 else "are",
        ", ".join(shown[:SHOWN_WORDS]) if shown else "no words")


def _times_told(job, kind):
    """How many times this job was refused for kind."""
    return sum(e["kind"] == kind and e.get("job") == job.n for e in job.chat.events)


def _told_once(job, kind):
    """True when this job was already refused for kind; else marks that it is, and False."""
    if _times_told(job, kind):
        return True
    job.emit(kind)
    return False


WATCHED_RECIPES = ("game",)
MENU_LINES = 2
MENU_TAIL = 4
LOOK_BACK = 30


def _watch_first(job, state):
    """Refuses, once a job, an effect on a game clip that Muse Wonder has not watched yet when it can: the effects
    then land on what a viewer reacts to, which the words and a look's four pictures miss. Measured 10-29 on Tarm's
    beat 1 (Glimmer, the SIFU stream): a shake and a sticker at 7.5 s, from the words alone, and watch_piece never
    called."""
    if job.n is None or (job.chat.clip or {}).get("recipe") not in WATCHED_RECIPES:
        return
    wonder = mind.MINDS.get("wonder")
    if not wonder or not wonder.secret() or (_here(job) and not sends.allowed()):
        return
    if any(e["kind"] == "step" and e.get("tool") == "watch_piece" and not e.get("refused") for e in job.chat.events):
        return
    if _told_once(job, "watch_first"):
        return
    a, b = state["clip"]["pieces"][0][0], state["clip"]["pieces"][-1][1]
    raise Refused("Not added. Have Muse Wonder watch the clip first: watch_piece from %s to %s (at most %d s at a "
                  "time). It sees and hears what a viewer reacts to, which the words miss, and names the effect for "
                  "each second; then put the effects where it says, this one too if it fits."
                  % (_tenth(a), _tenth(b), watch.LONGEST))


FACES_ONLY = ("face", "talk", "speaker", "split")
SCREENS = ("card", "stack")
FACE_ASKED = "their face to fill the frame, in words such as 'only my face' or 'just me, big', or the recording has no " \
             "screen to show"
LAYOUT_RULES = {
    "explain": (FACES_ONLY, "A tutorial's clip shows the screen being explained: the card layout keeps the screen and "
                            "puts the face in a circle above it", FACE_ASKED),
    "game": (FACES_ONLY, "A game clip shows the game: the card layout keeps it and puts the face in a circle above it",
             FACE_ASKED),
    "talk": (SCREENS + ("speaker", "split"), "A talk's clip cuts between its faces: talk, or face for one person "
                                             "(Tarm 10-29: cut between faces, not faces in circles)",
             "the screen or everyone on the screen at once, in words such as 'show the screen' or 'keep everyone on "
             "screen', or what they talk about is on a screen"),
    "vlog": (FACES_ONLY + ("stack", "subject"), "A vlog's clip keeps the whole picture as its editor framed it: the "
                                                "card layout, so the subtitles, titles and logo already in it stay "
                                                "whole and nothing is lost at a cut",
             "a face or one part of the picture to fill the frame, in words such as 'just her face' or 'zoom in on "
             "the food'"),
    "seller": (FACES_ONLY + SCREENS, "A seller's clip keeps the product whole in the frame: the subject layout fills "
                                     "the frame around it, and follow_product moves the frame with it. A card leaves "
                                     "out the part round the face, and a stack or a split cuts the frame in two, so a "
                                     "product held up by the face is cut or gone (Tarm 10-30, on a seller's ad that "
                                     "split the screen as she raised the tumbler)",
               "their face alone, a screen or the frame in two, in words such as 'just my face' or 'split the "
               "screen'"),
}


def _layout_unasked(job, recipe, kinds):
    """The sentence refusing, once a job, a layout the recipe's kind of video does not take by default (Tarm 10-28: a
    tutorial with a screen is a card, a talk or a story is faces, unless the person says otherwise; lane 5 asked for 'a
    clip of me talking' on a software tutorial and the face filled the frame, the page it explained gone), or None. A
    tap on the page is the person's own choice, and so is the layout their saved look keeps for this kind."""
    rule = LAYOUT_RULES.get(recipe)
    if not rule or job.n is None or not set(kinds) & set(rule[0]) - {_saved(recipe)[0]}:
        return None
    if _told_once(job, "layout_asked"):
        return None
    return "%s. Use %s only when the person asked for %s; then make it again unchanged." % (
        rule[1], " or ".join(sorted(set(kinds) & set(rule[0]))), rule[2])


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


def _no_chat(job):
    """True when Muse said this recording shows no chat box (no_chat_box), since it was opened."""
    told = next((e for e in reversed(job.chat.events) if e["kind"] in ("no_chat", "opened")), None)
    return bool(told) and told["kind"] == "no_chat"


def _names_check(job, rec, state):
    """(ok, found) for the check that no viewer's name can be read in a game stream's clip: every line read on the
    screen with a handle is under a blur for as long as it shows, and one blur stays for the whole clip, as the chat
    box does (a name read without its @ is not known to be a name). Game afc1 10-28: the check passed on a blur over
    the whole clip that lay under the face, while the chat at the top right showed. A recording with no chat box has
    nothing to blur, so Muse says so (no_chat_box) and names read on the screen still fail it: lane 5's 022940-2e20, a
    console match with no chat, went round this check by blurring the whole picture and ran out of steps."""
    shown, why = _shown_private(job, rec, state)
    if shown is None:
        return False, why
    names = [f for f in shown if f["kind"] == NAME_LINE]
    bare = [f for f in names if not screen.covered(f, state["blurs"])]
    if bare:
        return False, "names can be read at %s on your grid from %g to %g s, not under a blur" % (
            _grid(_around(bare), rec), min(f["start"] for f in bare), max(f["end"] for f in bare))
    whole = [b for b in state["blurs"] if b.get("box") and b["start"] <= 0.1 and b["end"] >= state["length"] - 0.1]
    if not whole and _no_chat(job):
        return True, "Muse saw no chat box, and no name was read on the screen", "Muse"
    if not whole:
        return False, "no blur covers the whole clip, so a chat box would show: blur it, or say the recording has none"
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


def _make_talk(job, rec, moment, m, pieces, cut, name="talk", kept=None):
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
                      "the moment's %d pictures. Choose the face or the subject layout. If the people move about, "
                      "as in a video someone edited, this is not a talk: load the vlog recipe, which keeps the whole "
                      "picture." % (name, len(persons), len(found["samples"])))
    if name == "split" and len(persons) != 2:
        raise Refused("The split layout shows two people, one above the other, and the program found %d. Choose the "
                      "speaker or the talk layout." % len(persons))
    heard = job.chat.looks[m["look"] - 1].get("said")
    try:
        persons, count_line = (persons, None) if name == "split" else _count_people(job, rec, found, m, persons)
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
    people = _people(rec, pieces, None, [a, b], shown=[p["box"] for p in persons])
    state = job.edit(dict({"do": "clip", "moment": moment, "pieces": pieces, "span": [a, b], "layout": lay,
                           "product": None, "people": [{k: v for k, v in p.items() if k != "crop"} for p in people or []],
                           "speech": _speech(rec, job.chat.looks[m["look"] - 1], pieces), "voices": spoken},
                          **(kept or {})))
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
    lines += filter(None, [count_line, voice_line])
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
    lines.append("Places for words on screen on your grid, kept clear of every person's head (the band from %s is "
                 "kept for the captions):" % layouts.on_grid(lay["captions"][1], lay["captions"][1] + lay["captions"][3],
                                                             "y", inward=False))
    lines += _place_lines(state, rec)
    peak = project.from_recording(pieces, m["t"] if m.get("t") is not None else (a + b) / 2)
    at = min(max(0.0, peak if peak is not None else state["length"] / 2), state["length"] - 0.1)
    lines.append(_show_frame(job, state, rec, at, "The clip at %g s (moment %d, %s layout):" % (round(at, 2),
                                                                                                  moment, name)))
    lines += _people_lines(job, people, rec)
    job.show(clip=job.chat.clip_view())
    return "\n".join(lines)


def _count_people(job, rec, found, m, persons):
    """(persons with whoever Muse Spark counts in one picture of the moment, kept in work/voices; a sentence for Muse or
    None). On the switch in Options only, as Muse Wonder's naming is. Stop raises hearing.Stopped."""
    if not _wonder_names():
        return persons, None
    key = hashlib.sha1(json.dumps([rec["path"], rec["size"], rec["mtime"], m["start"], m["end"],
                                   [p["box"] for p in persons]], sort_keys=True).encode("utf-8")).hexdigest()[:16]
    kept = os.path.join(recording.WORK, "voices", key + "_heads.json")
    try:
        with open(kept, encoding="utf-8") as f:
            heads = json.load(f)
    except (OSError, ValueError):
        at = max(found["samples"], key=lambda s: len(s["faces"]))["t"]
        path = voices.picture(rec, at, [], key + "_heads")
        try:
            got = path and job.ask_other("spark", voices.heads_question(path), voices.HEADS_SCHEMA,
                                         "Muse Spark is counting the people", voices.HEADS_TOKENS)
        except mind.MindError as e:
            return persons, "Muse Spark could not count the people (%s), so only the %d faces found have numbers." % (
                e, len(persons))
        heads = voices.heads_answer(got["text"], rec) if got else None
        if heads is None:
            return persons, "Muse Spark's count could not be read, so only the %d faces found have numbers." % len(persons)
        with open(kept + ".part", "w", encoding="utf-8") as f:
            json.dump(heads, f)
        os.replace(kept + ".part", kept)
    out = talk.with_heads(persons, heads)
    return out, (None if len(out) == len(persons) else
                 "Muse Spark counted %d people in one picture, and the faces found were %d: the others are followed where "
                 "it saw their heads, so look at their shots." % (len(heads), len(persons)))


def _voices(job, rec, m, persons):
    """(who says each turn, [{"from", "to", "who"}] in seconds of the recording, or None; a sentence for Muse): two
    people by how high they are, and otherwise, with the switch on, each turn named by Muse Wonder. Stop raises
    hearing.Stopped."""
    spoken, line = _two_voices(job, rec, m, persons)
    return (spoken, line) if spoken or not _wonder_names() else _named_voices(job, rec, m, persons, line)


def _wonder_names():
    """True when Muse Wonder may name who speaks: Muse Voice Transcribe hears, so the person's key is there and their
    switch is on."""
    wonder = mind.MINDS.get("wonder")
    return hearing.engine() == "meta" and bool(wonder and wonder.secret())


def _named_voices(job, rec, m, persons, line):
    """_voices by Muse Voice Transcribe's turns and Muse Wonder naming each one (voices.py), kept in work/voices so the
    clip made again over the same moment costs nothing. hearing.engine() is "meta", so the person's switch is on."""
    a, b = m["start"], m["end"]
    key = hashlib.sha1(json.dumps([rec["path"], rec["size"], rec["mtime"], a, b, [p["box"] for p in persons],
                                   [p.get("looks") for p in persons], voices.MOST_TURNS, voices.TURN_LONG],
                                  sort_keys=True).encode("utf-8")).hexdigest()[:16]
    kept = os.path.join(recording.WORK, "voices", key + ".json")
    try:
        with open(kept, encoding="utf-8") as f:
            return json.load(f), "Who speaks was named by Muse Wonder earlier, turn by turn."
    except (OSError, ValueError):
        pass
    job.working("Muse Voice Transcribe is telling the turns apart")
    try:
        sound = np.clip(voices.sound(rec, a, b), -1, 1)
        found, usd = asr.turns((sound * 32767).astype("<i2").tobytes(), voices.RATE, job.languages())
    except (voices.VoiceError, asr.VoiceError) as e:
        return None, " ".join(filter(None, [line, "Who speaks is found from mouths moving (%s)." % e]))
    finally:
        job.done_working()
    spent.add(os.path.join(recording.WORK, "spent.jsonl"), job.chat.id, job.n, asr.VOICE, {"usd": round(usd, 6)})
    named, cost = [], 0.0
    for k, turn in enumerate(voices.turns_of(found, a, m.get("said") or [])):
        try:
            path = voices.numbered(rec, turn["from"], min(turn["to"], turn["from"] + voices.TURN_SENT), persons,
                                   "%s_%d" % (key, k))
            answers = []
            while not answers or (len(answers) < voices.SHORT_ASKS and turn["to"] - turn["from"] < voices.TURN_SHORT
                                  and voices.agreed(answers + [None]) is None):
                got = job.ask_other("wonder", voices.who_question(path, persons), voices.WHO_SCHEMA,
                                    "Muse Wonder is watching who speaks, turn %d" % (k + 1), voices.WHO_TOKENS)
                cost += (got.get("engine") or {}).get("usd") or 0
                answers.append(voices.who_answer(got["text"], persons))
        except (voices.VoiceError, mind.MindError) as e:
            return None, " ".join(filter(None, [line, "Who speaks is found from mouths moving (%s)." % e]))
        named.append(dict(turn, who=voices.agreed(answers)))
    if not any(t["who"] for t in named):
        return None, " ".join(filter(None, [line, "Muse Wonder could not name who speaks, so it is found from mouths "
                                                  "moving."]))
    with open(kept + ".part", "w", encoding="utf-8") as f:
        json.dump(named, f)
    os.replace(kept + ".part", kept)
    return named, ("Muse Wonder named who speaks in %d of %d turns from their voices and faces ($%.2f)." % (
        sum(1 for t in named if t["who"]), len(named), cost + usd))


def _two_voices(job, rec, m, persons):
    """_voices for two people: their voices are told apart by how high they are, and Muse matches each voice to a face
    from pictures."""
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


def change_shot(job, why, start, end, show, centre=None):
    rec = _open(job, "change a shot of")
    state = _clip_state(job)
    lay = state["clip"]["layout"]
    if lay["kind"] != "talk":
        raise Refused("Only a clip in the talk layout has shots; this one is in the %s layout." % lay["kind"])
    length, count = state["length"], len(lay["people"])
    end = min(end, length)
    if not 0 <= start < length:
        raise Refused("start %g s is outside the clip, which is %g s long: use seconds of the clip." % (start, length))
    if end <= start:
        raise Refused("end %g s must come after start %g s." % (end, start))
    if not 0 <= show <= count:
        raise Refused("show %d is not one of the people: use 1 to %d (numbered from the left), or 0 for everyone."
                      % (show, count))
    if show == 0 and lay.get("seats"):
        raise Refused("In the speaker layout everyone is on screen all the time; show the person who should fill the "
                      "frame, 1 to %d." % count)
    if show == 0 and centre is not None:
        raise Refused("A shot of everyone shows them all on a card, so it has no frame to move: leave centre out, or "
                      "show one person.")
    crop = talk.crop_at(lay, show, centre) if centre is not None else None
    shots = talk.with_shot(lay["shots"], round(start, 2), round(end, 2), show, crop)
    short = next((s for s in shots if s["to"] - s["from"] < talk.SHORTEST_TURN / 2 and s not in lay["shots"]), None)
    if short:
        raise Refused("A shot of %g-%g s is too short to see; give it at least %g s."
                      % (short["from"], short["to"], talk.SHORTEST_TURN / 2))
    new = talk.make(tuple(lay["size"]), lay["people"], shots, bool(lay.get("seats")))
    state = job.edit({"do": "layout", "part": "shots", "box": None, "layout": new})
    lines = ["%s from %g to %g s. Shots now: %s." % ("Everyone is shown" if show == 0 else "Person %d is shown" % show,
                                                     start, end, talk.said(shots)),
             "Places for words now:"] + _place_lines(state, rec)
    lines.append(_show_frame(job, state, rec, (start + end) / 2, "The clip at the changed shot:"))
    job.show(clip=job.chat.clip_view())
    return "\n".join(lines)


MOST_FONTS_SHOWN = 12


def _letters_of(state):
    """The letters of a clip's titles and captions."""
    words = [t["text"] for t in state["titles"]] + [l["text"] for l in (state["captions"] or {}).get("lines") or []]
    return "".join(sorted(set("".join(words)) - set(" \n\t")))


def _letters(job, state):
    """The letters the clip's words are written in: its titles and captions, else what the person typed last."""
    return _letters_of(state) or "".join(sorted(set("".join(
        [e["text"] for e in job.chat.events if e["kind"] == "person"][-1:])) - set(" \n\t")))


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
    """What lies over drawn on a phone, or under it in the clip, as short reasons: what the apps draw on top, the
    captions band when the clip has captions, and the row kept for the channel's name (layouts.NAME_TOP), on every
    clip, since a name saved later goes there."""
    out = layouts.unsafe(drawn)
    y, h = lay["captions"][1], lay["captions"][3]
    if state["captions"] and drawn[1] < y + h and y < drawn[1] + drawn[3]:
        out.append("the captions run from %s on your grid" % layouts.on_grid(y, y + h, "y", inward=False))
    if drawn[1] < layouts.SAFE_BOTTOM and drawn[1] + drawn[3] > layouts.NAME_TOP:
        out.append("the row kept for the channel's name runs from %s on your grid" % layouts.on_grid(
            layouts.NAME_TOP, layouts.SAFE_BOTTOM, "y", inward=False))
    return out


def _phone(drawn, lay, state, asked):
    """What a phone or the captions cover of drawn, as notes for the person when they asked for the spot; Refused
    when Muse chose it."""
    covers = _covers([round(v) for v in drawn], lay, state)
    if covers and not asked:
        raise Refused("On a phone, %s. Pick one of the layout's places, or a box inside %s, %s on your grid; give "
                      "person_said only when the person asked for this spot." % (
                          "; ".join(covers), layouts.on_grid(layouts.SAFE[0], layouts.SAFE[0] + layouts.SAFE[2], "x"),
                          layouts.on_grid(layouts.SAFE[1], layouts.NAME_TOP, "y")))
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


def _thai_for_english(job, texts):
    """The texts Muse wrote in Thai when the person asked for the words on the clip in English, so a viewer who reads
    no Thai can follow; words the person typed themselves pass. The captions are not among them: they say what was
    said, and add_captions translates them."""
    if _subtitles((job.chat.clip or {}).get("subtitles")) != "English":
        return []
    said = " ".join(" ".join(e["text"].split()) for e in job.chat.events if e["kind"] == "person")
    return [t for t in texts if re.search("[ก-๛]", t) and " ".join(t.split()) not in said]


# Glimmer gave the person's whole opening ask as person_said, and a title sat on a forehead for 32 s (11-01, 031834-b64b).
LONG_ASK = 60
WHOLE_ASK = ("That is the person's ask for the clip, not their word on this one thing, so it does not stand. Quote only the "
             "words that ask for this; when nobody asked for it, choose yourself and leave person_said out.")


def _asked(job, person_said):
    """True when person_said quotes words the person typed in this chat or their own tap made the call, False when it
    is empty; Refused when the person never said it, so Muse cannot claim a request nobody made."""
    if getattr(job, "asked", False):
        return True
    if not person_said:
        return False
    quote = " ".join(person_said.split()).lower()
    said = [" ".join(e["text"].split()).lower() for e in job.chat.events if e["kind"] == "person"]
    whole = next((s for s in said if quote in s), None)
    if len(quote) < 3 or whole is None:
        raise Refused("The person did not say \"%s\" in this chat. person_said must quote their own words asking for "
                      "this; when nobody asked, choose yourself and leave person_said out." % person_said)
    if PLACES.search(quote) or (len(whole) > LONG_ASK and len(quote) > len(whole) / 2):
        raise Refused(WHOLE_ASK)
    return True


# Glimmer's own word รีสไตน์ reached a Sifu title on 11-01, and its spelling check, asked to keep every word, kept it. Of
# 100 titles that night the dictionary flagged 11: 5 wrong, 6 right (จอย, โกล, now in the dictionary), so Muse may send
# a word it means again.
ODD_TITLE = ("%s: no dictionary has %s in this title, and a viewer reads the title first. If it is a slip or a word "
             "you made up, write a word they know. If it is a name or a word people use, send the same title again.")
# Glimmer proofreading Glimmer mended the title and left หน้ฝน and ใช้งานง่ย in the post (11-01, 054318-527c).
ODD_POST = ("Not handed over: no dictionary has %s in the post text, and a person who presses Copy all posts it as it is. Mend "
            "the slips. A name or a word people use can stay: send the same text again.")


def add_title(job, why, words, start, end, place=None, key=None, box=None, person_said=None, behind=False):
    asked = _asked(job, person_said)
    rec = _open(job, "put words on")
    state = _clip_state(job)
    if behind and not asked and (job.chat.clip or {}).get("recipe") == "seller":
        raise Refused("A seller's title stays in front: behind the person it stands behind the hands and the product, "
                      "and nobody can read it. Put it on without behind.")
    if behind and not state["mask"]:
        raise Refused("A title goes behind the person once the person is found: call cut_out first, then put the title "
                      "on again with behind true.")
    if _thai_for_english(job, [words]):
        raise Refused("The person asked for the words on the clip in English, so a viewer who reads no Thai can "
                      "follow: write \"%s\" in English." % words)
    _other_apps(words)
    _latin_in_thai(words)
    copied = None if asked else _labels_copied(job, rec, state, [words], "The title")
    if copied:
        raise Refused("Not put on. " + copied)
    written = words
    try:
        (words,), spelling = _spell(job, [words])
    except hearing.Stopped:
        return "Not put on: the person pressed Stop."
    pieces = state["clip"]["pieces"]
    names, shown = _unseen_names(job, rec, min(p[0] for p in pieces), max(p[1] for p in pieces), words,
                                 _said(job, state))
    if names and not _told_once(job, "unseen"):
        raise Refused("%s. %s" % (_unseen_said(names, shown), UNSEEN_THEN % "put the title on again"))
    odd = (_odd_thai(words) or _split_up(written, words)) if job.n is not None and not asked else []
    if odd and not _told_once(job, "odd_title"):
        raise Refused(ODD_TITLE % ("Not put on", ", ".join('"%s"' % bit for bit in odd)))
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
        place = "box at %s on your grid" % _grid(band, FRAME)
    elif place not in lay["places"]:
        raise Refused("The %s layout has no place called %s; its places are: %s. Or give a box." % (
            lay["kind"], place, ", ".join(lay["places"])))
    else:
        band = lay["places"][place]
    default = job.chat.clip["style"] if job.chat.clip else "clean"
    st = render.style_of(state, default)
    try:
        size = text.measure(words, st, "title", band[2])
    except ValueError as e:
        raise Refused(str(e))
    if size.w > band[2] or size.h > band[3]:
        raise Refused("These words take %d lines at the title size (%d tall on your grid), and the %s place holds %d: "
                      "use fewer words." % (len(size.lines), layouts.grid(size.h, "y", up=True), place,
                                            layouts.grid(band[3], "y")))
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
    crossed = None
    if behind:
        frames = _person_frames(state, start, end)
        crossed = _crossing(frames, drawn, start, end)
        if not _crosses(crossed, start, end):
            raise Refused(_not_crossed(frames, drawn, place, start, end, crossed, lay, state, size))
        hidden = _hidden(frames, drawn)
        if (hidden > HIDDEN_MOST).mean() > 0.5:
            raise Refused(_too_hidden(frames, place, start, end, hidden, words, lay, state, size))
    times, every = _face_times(state, rec)
    hit = _covered(drawn, times, every, start, end)
    product = _product_box(state)
    part = None if asked or behind or not hit else _clear_part(band, words, st, key, times, every, start, end,
                                                               product)
    if part:
        band, size = part
        drawn = [band[0] + (band[2] - size.w) / 2, band[1] + (band[3] - size.h) / 2, size.w, size.h]
        moved = "Centred, it would cover a face%s at %s, so it stands in the part of the %s that is clear: %s on " \
                "your grid." % (" or the product" if state["clip"].get("product") else "", _spans(hit), place,
                                layouts.on_grid(band[0], band[0] + band[2], "x"))
        hit = []
    grow = 1
    if behind or not hit:
        grow, bigger, at = _grown(state, default, words, band, [] if behind else times, every, start, end,
                                  (lambda box: (_hidden(frames, box) > HIDDEN_MOST).mean() <= 0.5) if behind else None)
        if bigger:
            size, drawn = bigger, at
        if bigger and behind:
            crossed, hidden, hit = (_crossing(frames, drawn, start, end), _hidden(frames, drawn),
                                    _covered(drawn, times, every, start, end))
    notes = _phone(drawn, lay, state, asked) + ([moved] if part else [])
    over = _effects_over(state, drawn, start, end)
    if over and job.n is not None and not asked:
        raise Refused(UNDER_EFFECT % ", ".join(over))
    if over:
        notes.append("It stands under %s, where the person asked for it; tell the person." % ", ".join(over))
    if crossed is not None:
        notes.append("Behind the person: they cross the words %s and hide about %d%% of their width." % (
            _when(crossed, start, end), round(100 * float(np.median(hidden)))))
    if hit and behind:
        notes.append("It crosses a face at %s, and the person stands in front of it." % _spans(hit))
    elif hit and asked:
        notes.append("It covers a face%s at %s, where the person asked for it; tell the person." % (
            " or the product" if state["clip"].get("product") else "", _spans(hit)))
    elif hit:
        others = []
        for name, box in lay["places"].items():
            clear = _clear(_covered(box, times, every, 0, length), 0, length)
            opening = _longest(start, end, every, lambda t: _clear_part(
                box, words, st, key, times, every, start, t, product))
            if opening and any(lo <= start + 1e-6 and opening[0] <= hi + 1e-6 for lo, hi in clear):
                opening = None
            part = opening and "its part at %s, %s on your grid at %g-%g s" % (
                layouts.on_grid(opening[1][0][0], opening[1][0][0] + opening[1][0][2], "x"),
                layouts.on_grid(opening[1][0][1], opening[1][0][1] + opening[1][0][3], "y"), start,
                math.floor(opening[0] * 10) / 10)
            others.append("%s: %s" % (name, ", or ".join(filter(None, [_spans(clear, inward=True), part])) if (
                clear or part) else "never clear"))
        stuck = "" if any(not o.endswith("never clear") for o in others) else (
            " No place is clear for %g s or more: the faces%s fill the picture. %sor leave the title off." % (
                LISTED, " and the product" if state["clip"].get("product") else "",
                "Make the clip again with a product box around the product only, " if state["clip"].get("product")
                else ""))
        raise Refused("At the %s, the title would cover a face%s at %s. Clear: %s. Pick a time or a place from these.%s"
                      % (place, " or the product" if state["clip"].get("product") else "", _spans(hit),
                         "; ".join(others), stuck))
    entry = {"do": "title", "text": words, "start": round(start, 2), "end": round(end, 2), "place": place,
             "box": band, "key": key}
    if behind:
        entry["behind"] = True
    if grow > 1:
        entry["grow"] = grow
    state = job.edit(entry)
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
# Captions over a talked review: one line at a time, of 2 or 3 Thai words, drawn big (render._caption_style), as in
# the reviews Tarm follows (Extreme IT, 10-28); 12 Thai letters at that size measure about 575 of the band's 830 px.
ONE_LETTERS = 12
KEYS_ASK = ("These are the caption lines of a short clip, in order. For each line, give the one word a viewer must not "
            "miss, such as a number with its unit, a name or the thing the line is about, copied exactly as it is "
            "written in the line, or \"\" when no word stands out. Give back one key for each line, in order.\n\n%s")
KEYS_SCHEMA = {"type": "object", "properties": {"keys": {"type": "array", "items": {"type": "string"}}},
               "required": ["keys"], "additionalProperties": False}
ENGLISH_WORDS = {"the", "this", "that", "from", "with", "make", "please", "for", "and", "of", "to", "my", "it", "is",
                 "again", "add", "use", "show", "give", "can", "you"}
SUBTITLES = {"English": ("english", "en", "eng", "อังกฤษ", "ภาษาอังกฤษ"), "Thai": ("thai", "th", "ไทย", "ภาษาไทย")}
# The languages told by their letters alone; Japanese before Chinese, as its kana say Japanese and its kanji do not.
SCRIPTS = (("Thai", re.compile("[฀-๿]")), ("Japanese", re.compile("[぀-ヿ]")),
           ("Korean", re.compile("[가-힯]")), ("Chinese", re.compile("[一-鿿]")),
           ("Arabic", re.compile("[؀-ۿ]")), ("Hindi", re.compile("[ऀ-ॿ]")))
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
NUMBER_ASK = ("These lines do not carry the numbers their sentences say:\n%s\nGive each line the numbers its sentence "
              "says, and no others. Where a word is not a number in that sentence (the speech recognition may have "
              "misheard it), write the line as it was. Answer with all %d lines again, in the same order.")
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
    _watch_first(job, state)
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


def set_sound_tracks(job, why, tracks):
    """Which of the recording's sound tracks are in the clip, by their numbers from 1. Every track is in it until this
    is called. Tarm 11-01: Henri records the game and his voice on separate tracks and wants them told apart."""
    rec = _open(job, "choose the sound tracks of")
    _clip_state(job)
    streams = recording.streams_of(rec)
    count = len(streams)
    if count < 2:
        raise Refused("The recording has %s, so there is no track to choose." % (
            "one sound track" if count else "no sound"))
    if not tracks or any(not 1 <= n <= count for n in tracks):
        raise Refused("The recording has sound tracks 1 to %d: give at least one of those." % count)
    job.edit({"do": "tracks", "on": sorted({streams[n - 1] for n in tracks})})
    return "The clip now uses sound track%s %s of the %d. Render it again to hear it in the finished file." % (
        "" if len(set(tracks)) == 1 else "s", ", ".join(map(str, sorted(set(tracks)))), count)


def mute_sound(job, why, start, end, back=False):
    """Mutes the recording's own sound from clip second start to end, or brings it back over them (back): a count-in,
    another person's voice, a noise. The mute is kept as the stretches of the recording those seconds play
    (project.to_recording_spans), so it stays on what was said there when the clip is cut again. Tarm 10-28, after
    Eve's "1 2 3" opened his review: the person picks the seconds and the sound there goes."""
    rec = _open(job, "mute the sound of")
    state = _clip_state(job)
    length = state["length"]
    if not rec.get("sound_tracks"):
        raise Refused("The recording has no sound, so there is nothing to mute.")
    e = min(end, length)
    if not 0 <= start < e:
        raise Refused("start %g s and end %g s do not make a time inside the clip, which is %g s long."
                      % (start, end, length))
    if e - start < MUTE_SHORTEST:
        raise Refused("%g to %g s is shorter than %g s: give the start and end of what should not be heard."
                      % (start, e, MUTE_SHORTEST))
    before, had = state["mutes"], _caption_words(state)
    if back and not any(a < e and start < b for a, b in before):
        raise Refused("The sound is not muted anywhere from %g to %g s. %s" % (start, e, _muted_now(before)))
    start, e = round(start, 2), round(e, 2)
    spans = project.to_recording_spans(state["clip"]["pieces"], start, e)
    state = job.edit({"do": "mute", "spans": spans, "start": start, "end": e, "back": bool(back)})
    if back:
        return "The clip's own sound is back from %g to %g s. %s" % (start, e, _muted_now(state["mutes"]))
    out = ["The clip's own sound is muted from %g to %g s (%s of the recording): the picture plays on, and the music "
           "and the studio's sounds still play there. It is heard in the finished file and the page's video. %s"
           % (start, e, ", ".join("%s to %s" % (recording.clock(a), recording.clock(b)) for a, b in spans),
              _muted_now(state["mutes"]))]
    gone = had - _caption_words(state)
    if gone:
        out.append("%d caption word%s said there %s off the captions." % (gone, "" if gone == 1 else "s",
                                                                     "is" if gone == 1 else "are"))
    if start <= project.SLIVER and not state["music"]:
        out.append("The clip now opens without sound for %g s: music under it would fill that (set_music), when the "
                   "person has some." % e)
    return "\n".join(out)


MUTE_SHORTEST = 0.1


def _muted_now(mutes):
    return "Muted now: %s." % ", ".join("%g-%g s" % (a, b) for a, b in mutes) if mutes else "Nothing is muted now."


def _caption_words(state):
    return sum(len(ln["words"]) for ln in (state["captions"] or {}).get("lines") or [])


WORDS_ARE_THE_CLIP = ("talk", "explain")


def _unclear(job, first, looks):
    """The sentence refusing to go on with moment 1 when the clip's words are its captions (WORDS_ARE_THE_CLIP) and
    the hearing was sure of none of its lines, with where the looks hold lines it was sure of; None otherwise, and from
    the second time in a job, so a recording heard badly all through still gets its clip. Glimmer, EP 0 10-30: the
    talk recipe ranks by loudness, the loudest stretches were four people laughing over each other, every line was
    not sure, and the clip went out with no captions."""
    if (job.chat.clip or {}).get("recipe") not in WORDS_ARE_THE_CLIP or "unclear" in job.warned:
        return None
    if not first["said"] or any(not _unsure(s) for s in first["said"]):
        return None
    job.warned.add("unclear")
    clear = ["look %d at %s" % (look["n"], ", ".join(recording.clock(p["from"]) for p in sure[:4]))
             for look in looks for sure in [[p for p in look.get("said") or [] if not _unsure(p)]] if sure]
    return ("The hearing was sure of none of the %d line%s said in moment 1 (%s-%s): people talking over each other "
            "or laughing. In this kind of clip the words are the clip, and a line the hearing was not sure of is not "
            "captioned, so this one would go out with no words. %s If the person asked for this stretch, propose it "
            "again and it goes on." % (
                len(first["said"]), "" if len(first["said"]) == 1 else "s", recording.clock(first["start"]),
                recording.clock(first["end"]),
                "Propose a stretch where one person speaks at a time; lines were heard clearly in %s." % "; ".join(clear)
                if clear else "None of your looks has a line heard clearly either: look at other stretches of the map "
                "first, such as a quieter one where one person speaks at a time."))


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


def _by_line(phrase, at):
    """An unsure phrase at clip seconds at as [(from, to, words, unsure)]: cut into its caption lines when only an odd
    bit of Thai keeps it from being sure (_unsure: fixed, READ_SURE up), so the lines without one go on the video and
    only the line with it is a draft; else the phrase whole, unsure. 10-30, a seller's 13 s line read right but for
    ไจโรสโคป (gyroscope), which no dictionary has: the whole line went to the drafts, and the clip had no captions for
    its first 15 s."""
    fixed = "heard" in phrase and phrase.get("kept") != LATER
    if not (fixed and phrase.get("sure", 1.0) >= READ_SURE and re.search("[ก-๛]", phrase["text"])):
        return [(at[0], at[1], phrase["text"], True)]
    lines = text.caption_lines([{"from": at[0], "to": at[1], "text": phrase["text"]}], CAPTION_LETTERS)
    return [(round(x["from"], 2), round(x["to"], 2), x["text"], bool(words.odd_thai(x["text"]))) for x in lines]


def _in_clip(pieces, p, length):
    """(from, to) in clip seconds of a phrase heard from p["from"] to p["to"] of the recording, or None when it starts
    where the clip has no picture, or plays in it for less than a frame. A phrase whose end was cut out ends with its
    piece, not at the clip's end. 10-30 seller: one heard from the first piece's very end came to 28.69-28.69 s, and
    the whole caption step broke on it."""
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
    a, b = round(a, 2), round(min(length, b), 2)
    return (a, b) if b - a >= 1 / text.FPS else None


def _subtitles(name):
    """The language named for the captions, as SUBTITLES writes it or else as it was named, or None when none is. Any
    language Muse can write goes: libass draws letters the clip's font lacks from the computer's own fonts (10-31:
    Chinese, Korean, Arabic joined from right to left, Hindi, Russian, Vietnamese, Greek, on Windows)."""
    named = " ".join((name or "").lower().split())
    if not named:
        return None
    for language, names in SUBTITLES.items():
        if named in names:
            return language
    if not re.fullmatch(r"[^\W\d_]+( [^\W\d_]+)*", named):
        raise Refused("%r is not the name of a language. Give one, such as Spanish." % name)
    return " ".join(name.split()).title()


def _asked_in(job):
    """The language of the person's last words, when the studio writes captions in it and the words leave no doubt,
    else None: a clip whose speaker speaks another language gets its captions translated with nobody asking (Tarm
    10-31, "โอเค ลองเลยนะ"). A path in the words says nothing of it. Latin letters alone do not say English: a person
    writing Spanish over a Spanish video would get English captions (Tarm: "ถ้า user ใช้ภาษาอื่นจะมีปัญหาปะ"), so
    English takes one of its own small words, and with none the captions stay as they were said."""
    said = [PLACES.sub(" ", e["text"]) for e in job.chat.events if e["kind"] == "person"]
    words = said[-1] if said else ""
    return next((name for name, letters in SCRIPTS if letters.search(words)), None) or (
        "English" if ENGLISH_WORDS & set(re.findall(r"[a-z']+", words.lower())) else None)


def _spoken(job, texts):
    """The language the texts are said in: the one their letters say, else the recipe's guess."""
    found = next((name for name, letters in SCRIPTS if any(letters.search(t) for t in texts)), None)
    if found:
        return found
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
    """(Muse's subtitle in language for each unit, the units whose line leaves out a number Muse was shown), asked
    apart from the job, or None when twice the lines it wrote did not line up one for one with the units. A line
    that leaves out a number said is asked for once more with the number named, since the hearing can mishear a
    word as a number (a tutorial, 10-28: "นะครับสามก่อนเข้าไปดู", where สามารถ, can, was heard as สาม, three, and four
    lines translated it rightly as "you can go see it" and failed the check each time). What Muse writes then
    stands, and check_clip names the line for the person. When the clip's captions already hold each sentence in
    language, translated from the same words, they are kept: a tap on the page (Plain, Lit, Pop) never asks Muse,
    and Muse putting the same captions on lit is not asked to translate them again. A held line that lost a number
    Muse was never shown is asked again, so putting the captions on again is how Muse fixes the check."""
    caps = (state["captions"] or {}).get("lines") or []
    held = {}
    if (state["captions"] or {}).get("language") == language:
        for k in {line["phrase"] for line in caps}:
            mine = [line for line in caps if line["phrase"] == k]
            held[(round(mine[0]["from"], 2), round(mine[-1]["to"], 2), mine[0].get("said"))] = \
                ("".join(line["text"] for line in mine).strip(), bool(mine[0].get("no_number")))
    kept = [held.get((u["from"], u["to"], u["text"])) for u in units]
    if all(kept) and not any(_numbers_differ(u["text"], s) and not shown for u, (s, shown) in zip(units, kept)):
        return [s for s, _ in kept], {k for k, (_, shown) in enumerate(kept) if shown}
    kind = (job.chat.clip or {}).get("title") or "a video"
    names = [w for w in dict.fromkeys(_screen_words(job, rec, state)) if len(w) >= 4][:SCREEN_NAMES]
    listing = "\n".join("%d. (%.1f s) %s" % (k, u["to"] - u["from"], u["text"]) for k, u in enumerate(units, 1))
    shown = "; the screen writes these words this way: %s" % ", ".join(names) if names else ""
    talk = [{"role": "user", "content": SUBTITLE_ASK % (kind, said_in, language, said_in, language, shown, language,
                                                         listing)}]
    lines = _subtitle_lines(job, talk, len(units), language, 2)
    if lines is None:
        return None
    lost = [k for k, (u, s) in enumerate(zip(units, lines)) if _numbers_differ(u["text"], s)]
    if not lost:
        return lines, set()
    named = "\n".join('%d. "%s" says %s; your line: "%s"' % (
        k + 1, units[k]["text"], " and ".join("/".join("%g" % v for v in sorted(c)) for c in _said_numbers(
            units[k]["text"])) or "no number", lines[k]) for k in lost)
    talk += [{"role": "assistant", "content": json.dumps({"lines": lines}, ensure_ascii=False)},
             {"role": "user", "content": NUMBER_ASK % (named, len(units))}]
    again = _subtitle_lines(job, talk, len(units), language, 1)
    if again is None:
        return lines, set()
    for k in lost:
        lines[k] = again[k]
    return lines, {k for k in lost if _numbers_differ(units[k]["text"], lines[k])}


def _subtitle_lines(job, talk, count, language, tries):
    """The count lines Muse writes for talk, or None when in tries asks they never came one for each sentence."""
    for _ in range(tries):
        try:
            got = json.loads(job.ask_aside(talk, SUBTITLE_SCHEMA, "Writing the %s subtitles" % language))["lines"]
        except mind.MindError as e:
            raise Refused("Muse writes the %s captions, and Muse could not be asked: %s" % (language, e))
        except (ValueError, KeyError, TypeError):
            continue
        if isinstance(got, list) and len(got) == count and all(isinstance(s, str) and s.strip() for s in got):
            return [" ".join(s.split()) for s in got]
    return None


def _said_numbers(said):
    """[{the values one number may be}] for each number said, leaving out 1, which one language says as a number
    and another as "a". A Thai number word counts only where it stands as a word of its own, and one that ends in
    หนึ่ง after a unit is said for both readings. A tutorial, 10-28: the check failed two right translations, "Okay, you can
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


def _caption_keys(job, lines, before):
    """Each caption line's key, the one word a viewer must not miss, drawn in the key colour: Muse names them, apart
    from the job, and a line keeps one only when it is a run of its whole words (text._caption_span), as Muse wrote it
    or with Glimmer's slips put back. A tap on the page never asks Muse, so a line keeps the key a line of the same
    words had (before: the lines the captions had). The look people know from caption apps, asked for by Tarm 10-29
    for the judges, who often watch with the sound off. Returns how many lines have a key. Stop raises
    hearing.Stopped."""
    kept = {ln["text"]: ln["key"] for ln in before if ln.get("key")}
    try:
        got = json.loads(job.ask_aside([{"role": "user", "content": KEYS_ASK % json.dumps(
            [ln["text"].strip() for ln in lines], ensure_ascii=False)}], KEYS_SCHEMA,
            "Choosing the key word of each caption line"))["keys"]
    except (mind.MindError, ValueError, KeyError, TypeError):
        got = None
    if not isinstance(got, list) or len(got) != len(lines):
        got = [kept.get(ln["text"]) for ln in lines]
    for ln, key in zip(lines, got):
        for k in (key, words.unslipped(key)) if isinstance(key, str) else ():
            if text._caption_span(ln["text"].strip(), k):
                ln["key"] = k.strip()
                break
    return sum(1 for ln in lines if ln.get("key"))


def _talked(clip, phrases, length, translated=False):
    """Whether the clip is talked through, so its captions come one line at a time, drawn big: as arrange_clips found
    it for clips joined, else when what is said fills TALKED of it, as there. Seen 10-29 on a seller's review talked
    through 30 s: the captions were two small lines, as only a clip joined by arrange_clips was ever found talked. A
    game keeps them small, under the action, and so do captions translated: ONE_LETTERS was measured on Thai, and an
    English line cut at it leaves a word alone ("Easy to set", "up."). phrases are all that is said, the lines left as
    drafts too: EP0's talk (10-30) was said through, its sure lines 26.7 s of 45.3 and its drafts 16.6 s more, and
    counted on the sure ones alone it got two small lines."""
    if "talked" in clip:
        return bool(clip["talked"])
    return not translated and clip.get("recipe") != "game" and \
        sum(p["to"] - p["from"] for p in phrases) >= TALKED * length


def _off_own_words(job, rec, state, lines):
    """(the caption lines to draw, the sentence for Muse or None): in a kind of video an editor may have written words
    on (OWN_WORDS), a line said while the video's own words show in the clip's frame for at least half of it is left
    out, since two sets of words fight (EP0 10-30, a talk clip: the editor's names and reactions, such as "พี่ตูมตาม",
    sat over ours). A line the person typed stays. (None, None) when the person pressed Stop."""
    if (job.chat.clip or {}).get("recipe") not in OWN_WORDS:
        return lines, None
    job.working("Looking for words the video already shows")
    try:
        found = screen.own_words(rec, state["clip"]["pieces"], on_start=job.started_work)
    except screen.Stopped:
        return None, None
    except screen.ScreenError as e:
        return lines, ("The picture could not be looked at for words of its own (%s): if it shows subtitles of its "
                       "own, take the captions off." % e)
    finally:
        job.done_working()
    lay = state["clip"]["layout"]
    kept, out, seen = [], [], []
    for line in lines:
        mid = (line["from"] + line["to"]) / 2
        shows = [w for w in found if _shows_whole(lay, w["box"], mid)]
        over = sum(max(0.0, min(line["to"], w["end"]) - max(line["from"], w["start"])) for w in shows)
        if over >= OWN_HALF * (line["to"] - line["from"]) and not line.get("typed"):
            out.append(line)
            seen += [w for w in shows if w not in seen]
        else:
            kept.append(line)
    if not out:
        return lines, None
    return kept, ("The video shows words of its own in the clip's frame (an editor's subtitles or names) at %s, so the "
                  "%d caption line%s said then %s left out: two sets of words fight. The other %d %s drawn." % (
                      ", ".join("%g-%g s" % (w["start"], w["end"]) for w in sorted(seen, key=lambda w: w["start"])),
                      len(out), "" if len(out) == 1 else "s", "is" if len(out) == 1 else "are", len(kept),
                      "is" if len(kept) == 1 else "are"))


def _shows_whole(lay, box, at):
    """True when at least half of box (pixels of the recording) is in one part of the clip's frame at clip second at,
    both across and down: a sliver at a part's edge cannot be read."""
    for part in layouts.parts_at(lay, at):
        sx, sy, sw, sh = part["src"]
        across = min(box[0] + box[2], sx + sw) - max(box[0], sx)
        down = min(box[1] + box[3], sy + sh) - max(box[1], sy)
        if across >= box[2] * OWN_HALF and down >= box[3] * OWN_HALF:
            return True
    return False


HIDDEN_SAYS = ("The captions are hidden with the eye on the clip: they show again from there. The person hid "
               "them, so leave them hidden; tell the person, if they asked for captions. When they ask to see them "
               "again, add_captions with hidden false shows them as they were.")


CAPTIONS_OFF = ("The person's saved look, \"%s\", keeps the captions off, so leave them off. If the person asked for "
                "captions in this chat, call add_captions again.")


def _eye(job, state, hidden):
    """add_captions with hidden: what the eye on the clip does, every caption hidden and kept as it is, the lines the
    person typed too, or shown again as it was. Tarm 10-30: every tool on the page can be asked for in the chat."""
    if bool(state["hidden"]) == hidden:
        return "The captions are already %s." % ("hidden" if hidden else "showing")
    if hidden and not state["captions"]:
        raise Refused("The clip has no captions to hide.")
    job.edit({"do": "eye", "hidden": hidden})
    return ("Every caption is hidden and kept as it is: add_captions with hidden false shows them again." if hidden
            else "The captions show again, as they were.")


STRAY_SECONDS = 2.0


def _stray(p, said):
    """True when Voice Transcribe wrote one foreign word on a laugh or a breath: "Yeah," for 1.8 s and "Do" among EP0's
    Thai lines (10-31), and "have" before. Letters in capitals, such as OK or MT4, stay."""
    return (p["to"] - p["from"] < STRAY_SECONDS and bool(re.fullmatch(r"[A-Za-z][a-z']*[,.!?]*", p["text"].strip())) and
            2 * sum(1 for q in said if any(letters.search(q["text"]) for _, letters in SCRIPTS)) >= len(said))


def add_captions(job, why, on=True, lit=False, pop=None, language="", hidden=None):
    rec = _open(job, "caption")
    state = _clip_state(job)
    if hidden is not None:
        return _eye(job, state, hidden)
    if state["hidden"]:
        raise Refused(HIDDEN_SAYS)
    if on and job.n is not None and _captions_off() and not _told_once(job, "captions_off"):
        raise Refused(CAPTIONS_OFF % _captions_off())
    if lit and pop:
        raise Refused("Captions are lit or popped, not both: a popped word already moves as it is said. Choose one.")
    target = _subtitles(language or (job.chat.clip or {}).get("subtitles") or (state["captions"] or {}).get(
        "language")) or _asked_in(job)
    clip, length = state["clip"], state["length"]
    if not on:
        if not state["captions"]:
            raise Refused("The clip has no captions to take off.")
        job.edit({"do": "captions", "lines": [], "band": None})
        return "Captions are off."
    moments = job.chat.moments or []
    said = moments[clip["moment"] - 1].get("said") or [] if clip["moment"] <= len(moments) else []
    typed = _typed(state["captions"])
    phrases, doubtful, heard = [], [], []
    for n, p in enumerate(said):
        at = _in_clip(clip["pieces"], p, length)
        if not at or (_stray(p, said) and not _unsure(p)):
            continue
        if _unsure(p) and (at[0], at[1]) not in typed:
            for a, b, line, odd in _by_line(p, at):
                if odd:
                    doubtful.append({"from": a, "to": b, "text": line})
                else:
                    phrases.append({"from": a, "to": b, "text": line, "n": n})
                    heard.append(dict(p, text=line))
        else:
            phrases.append({"from": at[0], "to": at[1], "text": p["text"], "n": n})
            heard.append(p)
    if not phrases and not doubtful:
        raise Refused("Nothing was heard clearly in this moment, so there are no words to caption. Leave the captions "
                      "off, and say so in your answer.")
    band = clip["layout"]["captions"]
    if not phrases:
        one = _talked(job.chat.clip or {}, doubtful, length)
        entry = {"do": "captions", "lines": [], "band": [band[1], band[3]], "lit": bool(lit),
                 "pop": bool(one and not lit if pop is None else pop),
                 "drafts": text.caption_lines(doubtful, ONE_LETTERS if one else CAPTION_LETTERS)}
        if one:
            entry["one"] = True
        job.edit(entry)
        return "No line was heard clearly, so no captions are drawn on the video. " + _drafts_said(doubtful)
    translated, first, unnumbered = None, list(range(len(phrases))), set()
    try:
        texts, spelling = _spell(job, [p["text"] for p in phrases], heard=True)
        for phrase, p, words_ in zip(phrases, heard, texts):
            phrase["text"] = p["text"] = words_
        said_in = _spoken(job, texts)
        if target and target != said_in:
            units = _sentences(phrases)
            translated, unnumbered = _translated(job, rec, state, units, target, said_in) or (None, set())
    except hearing.Stopped:
        return "No captions: the person pressed Stop."
    shown = [{"from": u["from"], "to": u["to"], "text": s} for u, s in zip(units, translated)] if translated else \
        phrases
    for piece in shown:
        piece["text"] = typed.get((piece["from"], piece["to"]), piece["text"])
    one = _talked(job.chat.clip or {}, phrases + doubtful, length, bool(translated))
    lines = text.caption_lines(shown, ONE_LETTERS if one else CAPTION_LETTERS)
    for line in lines:
        if translated:
            line["said"] = units[line["phrase"]]["text"]
        if line["phrase"] in unnumbered:
            line["no_number"] = True
        if (shown[line["phrase"]]["from"], shown[line["phrase"]]["to"]) in typed:
            line["typed"] = True
    if not translated:
        lines, own = _off_own_words(job, rec, state, lines)
        if lines is None:
            return "No captions: the person pressed Stop."
        if own and not lines:
            job.edit({"do": "captions", "lines": [], "band": None})
            return own + " So no captions are drawn."
    if pop is None:
        pop = one and not lit
    try:
        keyed = 0 if lit else _caption_keys(job, lines, (state["captions"] or {}).get("lines") or [])
    except hearing.Stopped:
        return "No captions: the person pressed Stop."
    if translated:
        first = [u["first"] for u in units]
    entry = {"do": "captions", "lines": lines, "band": [band[1], band[3]], "lit": bool(lit), "pop": bool(pop)}
    if one:
        entry["one"] = True
    if doubtful:
        entry["drafts"] = text.caption_lines(doubtful, ONE_LETTERS if one else CAPTION_LETTERS)
    if translated:
        entry["language"] = target
    bands, voiced = clip["layout"].get("bands"), clip.get("voices")
    if bands and voiced:
        who = {(round(v["from"], 2), round(v["to"], 2)): v["who"] for v in voiced}
        owners = [who.get((round(p["from"], 2), round(p["to"], 2))) for p in heard]
        for line in lines:
            k = first[line["phrase"]]
            line["who"] = owners[k] if k < len(owners) else None
        entry["bands"] = {n: [b[1], b[3]] for n, b in bands.items()}
    state = job.edit(entry)
    if language and job.chat.clip is not None and (translated or target == said_in):
        job.chat.clip["subtitles"] = target
    under = ("each under the face of the person who says it" if entry.get("bands") and any(
        line.get("who") for line in lines) else "in the band from %s on your grid" % layouts.on_grid(
            band[1], band[1] + band[3], "y", inward=False))
    out = ["Captions on: %d lines %s, %s; %s."
           % (len(lines), "in %s, translated from the %d sentences said in %s, %s" % (
               target, len(units), said_in, "numbers kept" if not unnumbered else
               "numbers kept but in %d where you were shown the number and left it out, which the check will name to "
               "the person" % len(unnumbered)) if translated else "of what was said, numbers exactly as heard", under,
              "each word lights up as it is said" if lit else
              "each line shows whole on its see-through box, as a news clip's subtitles do: the look's box does not "
              "let words pop in" if pop and render.whole_lines(state, job.chat.clip["style"] if job.chat.clip else
                                                                 "clean") else
              "each word pops in at its place as it is said and stays, so once said the line looks plain" if pop else
              "plain words, none lit, so the eye stays on the picture")]
    if keyed:
        out.append("In %d of the %d lines the word a viewer must not miss is in the key colour." % (keyed, len(lines)))
    if not translated and own:
        out.append(own)
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
    if doubtful:
        out.append(_drafts_said(doubtful))
    on_camera = _on_camera(rec, clip) if lit else None
    if on_camera:
        out.append("A person is on camera (%s). Lit words pull the eye away from a person who is talking, so the house "
                   "way is plain captions here. Keep them lit only if the person asked for it; otherwise put the "
                   "captions on again with lit false." % on_camera)
    times, every = _face_times(state, rec)
    product = _product_box(state)
    hit = _covered(band, [(t, [b for b in boxes if b not in product]) for t, boxes in times], every, 0, length)
    if hit:
        out.append("The captions cover a face at %s. If that matters, make the clip again with a layout that keeps the "
                   "face higher." % _spans(hit))
    if product and _covered(band, [(t, product) for t, _ in times], every, 0, length):
        out.append("The captions cross the product's box, so they cover the product where it mostly stands. If that "
                   "matters, make the clip again with a layout that keeps the product higher.")
    out += ["%s %s" % (recording.clock(line["from"]), line["text"].strip()) for line in lines[:8]]
    if len(lines) > 8:
        out.append("... and %d more lines." % (len(lines) - 8))
    first = lines[0]
    out.append(_show_frame(job, state, rec, (first["from"] + first["to"]) / 2, "The clip with captions:"))
    return "\n".join(out)


def _drafts_said(doubtful):
    """What Muse is told of the lines the hearing was not sure of: drafts under the clip, never drawn on the video
    until the person keeps or fixes them there, for a wrong caption is worse than none. Tarm 10-30: a clip is made 80%
    of the way and the person does the last 20% before posting, and a talk of four people laughing had no caption at
    all, so they typed every line."""
    one = len(doubtful) == 1
    return ("%d line%s the hearing was not sure of %s under the clip as %s, not on the video, because a wrong "
            "caption is worse than none: %s. The person keeps or fixes each draft there before posting: say so in "
            "your answer." % (len(doubtful), "" if one else "s", "is" if one else "are", "a draft" if one else "drafts",
                              "; ".join("%s \"%s\"" % (recording.clock(p["from"]), p["text"]) for p in doubtful[:4])))


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


def _people(rec, pieces, face, span=None, shown=(), job=None):
    """The other people's faces in the clip, from the faces already found in each piece, most seen first: [{"n",
    "seen", "of", "from", "to" (clip seconds), "box" (recording), "points": [[clip second, x, y, w, h]], "crop"}].
    The face the layout shows, the faces of a talk layout (shown: the people it follows, the clip itself), and a
    face that sits still (the person making the clip), are not among them. None when the faces of a piece were not
    found. 10-29, the brother's podcast: the three hosts were listed as other people, and Glimmer blurred them all.
    With a job, faces that move are looked for again, finer, unless they are the players of a game."""
    out = []
    for a, b, found in _found_faces(rec, pieces, span):
        if not found:
            return None
        if job and (job.chat.clip or {}).get("recipe") not in PICTURE_ENDS:
            found = tracks.finer(rec, a, b, found, on_start=getattr(job, "started_work", None))
        for tr in tracks.tracks(found, rec):
            if tr["steady"] or face and _overlap(tr["box"], face) > SAME_FACE or any(
                    _overlap(tr["box"], box) > SAME_FACE for box in shown):
                continue
            points = []
            for t, x, y, w, h in tr["points"]:
                at = project.from_recording(pieces, min(max(t, a), b))
                if at is not None and (not points or at > points[-1][0]):
                    points.append([round(at, 3), x, y, w, h])
            if not points:
                continue
            out.append({"seen": tr["seen"], "of": round(len(found["samples"]) * found["every"] / faces.EVERY),
                        "from": points[0][0],
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
    lines = ["Other people's faces in the clip: %d (the faces the layout shows are not among them). Decide about each "
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


def _undecided(state, job=None):
    """The people seen in two pictures or more that nobody has decided about yet; with a job of Muse's, when the person
    asked for faces to be blurred, everyone: seen once is half a second, and a rider so told was on screen for 0.9 s of a
    montage, plainly, because Muse named 4 of 11 and render let the rest go by (11-01, 061212-faed)."""
    everyone = job is not None and job.n is not None and _blur_asked(job)
    return [p["n"] for p in state["people"] if p["blur"] is None and (everyone or p["seen"] > 1)]


# 10-31 050337-bff5: Glimmer blurred all four people on a talk's sofa, the one telling the story too.
TALK_BLUR = ("Not blurred. A talk is its people: keep them. If the person asked for this blur, give person_said with their "
             "words.")
# Glimmer blurred all 9 people of a cooking episode "for privacy" (11-01, 005953-ecc0).
ALL_BLUR = ("Not blurred. That is everyone in the clip, and a clip of blurred faces has nobody in it. Keep the people the clip "
            "is about, and blur only someone passing behind. If the person asked for this blur, give person_said with their "
            "words.")


BLUR_MANY = 3
# Measured on the horse trip (lane 5, 11-01, 040740-b50f): 19% of 139 faces covered whole, as a phone on a horse moves more between
# finds than a face is wide. A person who hides a stranger and trusts "the blur moves with the face" posts the face.
BLUR_SLIPS = ("With a moving camera a face can slip out from under its blur between finds, so do not tell the person it is hidden: "
              "tell them to check it before they post.")


BLUR_SAID = re.compile(r"(?<!ไม่ต้อง)(?<!ไม่)(?<!อย่า)เบลอ\S{0,6}(?:หน้า(?!จอ)|คน)|"
                       r"(?<!not )(?<!n't )(?<!no )\bblur\b[^.!?\n]{0,40}\b(?:faces?|people|others?|strangers?)\b", re.I)
BLUR_ASKED = ("Not decided: the person asked for this in the chat: \"%s\". Blur the people they meant, by number. Send this "
              "again unchanged only if they have since said to keep everyone.")


def _blur_asked(job):
    """The person's own words in this chat asking for faces to be blurred, or None. Reads their messages, not the screen.
    054756-9aba (11-01): the ask said เบลอหน้าคนอื่นในคลิป, Spark Open kept all 9 for the feel of the trip, and the answer did
    not say so; at a trip every face starts kept, so no decision was even asked for."""
    for e in reversed(job.chat.events):
        said = BLUR_SAID.search(e["text"]) if e["kind"] == "person" else None
        if said:
            return said.group(0)
    return None


def decide_people(job, why, blur=None, keep=None, others=None, person_said=None):
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
    asked = None if blur or job.n is None or why == BATCH_WHY else _blur_asked(job)
    if asked and not _told_once(job, "blur_asked"):
        raise Refused(BLUR_ASKED % asked)
    recipe = (job.chat.clip or {}).get("recipe")
    crowd = len(blur) == len(people) >= BLUR_MANY and recipe != "seller"
    # Sent again unasked, Glimmer's blur of 5 of a talk's 6 faces once went through (11-01, 021005-389c): it needs their words.
    if job.n is not None and blur and (recipe == "talk" or crowd and not _blur_asked(job)) and not _asked(job, person_said):
        raise Refused(TALK_BLUR if recipe == "talk" else ALL_BLUR)
    state = job.edit({"do": "people", "blur": blur, "keep": keep})
    if len(blur) > PEOPLE_LISTED:
        lines = ["%d people are blurred where their faces were found, and each blur follows those finds." % len(blur)]
    else:
        lines = ["Person %d is blurred where their face was found, %s-%s of the clip, and the blur follows those finds."
                 % (n, recording.clock(people[n]["from"]), recording.clock(people[n]["to"])) for n in blur]
    if blur:
        lines.append(BLUR_SLIPS)
    if len(keep) > PEOPLE_LISTED:
        lines.append("Kept as they are: %d people." % len(keep))
    elif keep:
        lines.append("Kept as they are: person %s." % _numbers(keep))
    left = _undecided(state, job)
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
BLUR_MOST = 0.5
BLUR_WIDE = ("That blur covers most of the picture for the whole clip, so nothing would be left to watch. Blur only what "
             "must not be read, such as the chat or the names.")
PART_NAMES = {"card": "the card", "circle": "the circle with the face", "top": "the top panel",
              "bottom": "the panel below", "whole": "the frame"}


def _share(box, other):
    """How much of box (0 to 1) other covers."""
    w = min(box[0] + box[2], other[0] + other[2]) - max(box[0], other[0])
    h = min(box[1] + box[3], other[1] + other[3]) - max(box[1], other[1])
    return max(0.0, w) * max(0.0, h) / max(1e-6, box[2] * box[3])


def _where(lay, box):
    """Where a box of the recording shows in the clip, part by part, on Muse's grid of the frame."""
    out = []
    for part in lay["parts"]:
        f = layouts.to_frame({"parts": [part]}, box)
        if f:
            x, y, w, h = f[0]
            out.append("in %s at %s, %s on your grid of the clip" % (
                PART_NAMES.get(part["name"], part["name"]), layouts.on_grid(x, x + w, "x", inward=False),
                layouts.on_grid(y, y + h, "y", inward=False)))
    return out


EFFECTS = ("hit", "sticker", "flurry", "sound", "picture", "push", "title")
ON_THE_PICTURE = ("sticker", "flurry", "picture", "title")


def change_effect(job, why, what, called, at, to=None, box=None, person_said=None):
    _open(job, "change an effect on")
    state = _clip_state(job)
    key, kept = ("start", {"push": "pushes", "title": "titles"}[what]) if what in project.LASTING else (
        "at", {"hit": "hits", "sticker": "stickers", "flurry": "flurries", "sound": "sounds", "picture": "pictures"}[what])
    found = [x for x in state[kept] if abs(x[key] - at) < 0.01 and project.effect_name(what, x) == called]
    if not found:
        raise Refused("The clip has no %s \"%s\" at %g s; its %s: %s." % (what, called, at, kept, ", ".join(
            "%s at %g s" % (project.effect_name(what, x), x[key]) for x in state[kept]) or "none"))
    effect, length = found[0], state["length"]
    if box is not None:
        if what not in ON_THE_PICTURE:
            raise Refused("A %s has no spot on the picture to move." % what)
        if max(box) > 1000:
            raise Refused("The box %s runs off the frame: the grid ends at 1000." % _on_grid(box))
        if job.n is not None and not _asked(job, person_said):
            raise Refused("A box here goes over faces and titles unchecked, so it is only for a spot the person asked for: "
                          "give person_said with their words. To choose a spot yourself, add the %s again there." % what)
        band = _frame_box("box", box)
        job.edit(dict({"do": "effect", "of": what, "name": called, "at": effect[key], "to": effect[key], "box": band},
                      **({"place": "box at %s on your grid" % _grid(band, FRAME)} if what == "title" else {})))
        return "The %s \"%s\" is at %s on your grid now." % (what, called, _on_grid(box))
    if to is not None:
        lasts = effect["end"] - effect["start"] if what in project.LASTING else effect.get("length", 0)
        if not 0 <= to < length or to + lasts > length:
            raise Refused("It would run off the clip: a %s of %g s fits from 0 to %g s of the %g s clip." % (
                what, lasts, round(length - lasts, 2), length))
    job.edit({"do": "effect", "of": what, "name": called, "at": effect[key], "to": None if to is None else round(to, 2)})
    if to is None:
        return "The %s \"%s\" at %g s is off the clip." % (what, called, effect[key])
    return "The %s \"%s\" moved from %g s to %g s." % (what, called, effect[key], to)


def remove_title(job, why, words):
    rec = _open(job, "take words off")
    state = _clip_state(job)
    if not any(t["text"] == words for t in state["titles"]):
        raise Refused("The clip has no title \"%s\"; its titles are: %s." % (
            words, "; ".join('"%s"' % t["text"] for t in state["titles"]) or "none"))
    state = job.edit({"do": "untitle", "text": words})
    return '\n'.join(['The title "%s" is off the clip.' % words,
                      _show_frame(job, state, rec, state["length"] / 2, "The clip without it:")])


def _whole_names(job, rec, state, px, face):
    """(the box to blur, notes): in a game, px grown to cover the whole of each line with a viewer's name it reaches
    into, as long as it still leaves the face clear. Game 10-30 (174316-0f5e): check_clip found names at
    [781, 48, 1000, 324], Glimmer blurred [780, 0, 1000, 250], then [750, 0, 1000, 400]: three renders and checks
    for a bottom edge."""
    if (job.chat.clip or {}).get("recipe") != "game":
        return px, []
    names = [f for f in _shown_private(job, rec, state)[0] or [] if f["kind"] == NAME_LINE and _share(f["box"], px) > 0]
    if not names or all(screen.covered(f, [{"box": px, "start": 0, "end": state["length"]}]) for f in names):
        return px, []
    grown = _around(names + [{"box": px}])
    if face and _share(face, grown) > BLUR_FACE_SHARE:
        return px, []
    return grown, ["The box reached into %d line%s with a viewer's name, so it was grown to cover %s whole." % (
        len(names), "" if len(names) == 1 else "s", "it" if len(names) == 1 else "them")]


def _blurs_the_picture(px, rec):
    return px[2] * px[3] > BLUR_MOST * rec["width"] * rec["height"]


def _blur_wide(job):
    return BLUR_WIDE + (" A recording with no chat box needs no blur: say so with no_chat_box."
                        if (job.chat.clip or {}).get("recipe") == "game" else "")


# Montage 035936-7931 (11-01): 4 boxes of 2 to 2.5 s on riders, each shown at its middle, and two stood on the trees by their end.
STILL_BOX = ("The box stays where it is while the picture moves, so what moves out of it shows again. The frame next is its last: "
             "if it no longer covers what it should, give a shorter span, and a new box for the rest.")
BLUR_LAST = 0.1


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
    if b - a >= length - 0.1 and _blurs_the_picture(px, rec) and not asked:
        raise Refused(_blur_wide(job))
    px, widened = _whole_names(job, rec, state, px, face)
    state = job.edit({"do": "blur", "box": [round(v) for v in px], "start": round(a, 2), "end": round(b, 2)})
    where = "; ".join(_where(lay, px))
    lines = ["Blurred %s on your grid from %g to %g s, so nothing in it can be read. In the clip it shows %s."
             % (_grid(px, rec), a, b, where)] + widened
    span = start is not None or end is not None
    if span:
        lines.append(STILL_BOX)
    lines.append(_show_frame(job, state, rec, b - BLUR_LAST if span else (a + b) / 2, "The clip with the blur:"))
    return "\n".join(lines)


def no_chat_box(job, why):
    _open(job, "check")
    if (job.chat.clip or {}).get("recipe") != "game":
        raise Refused("Only a game stream is checked for a chat box.")
    job.emit("no_chat")
    return ("Noted: this recording shows no chat box, so no blur is asked for. The screen is still read, and a viewer's "
            "name read on it fails the check.")


def add_hit(job, why, at, kind, word=None):
    state = _clip_state(job)
    _watch_first(job, state)
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
WORDS_UNDER = 0.05


def _sticker_drawn(png, band):
    """[x, y, w, h] on the frame where a sticker given band is drawn: as large as fits, centred in it."""
    w, h = effects._png_size(png)
    fit = min(band[2] / w, band[3] / h)
    return [round(band[0] + (band[2] - w * fit) / 2), round(band[1] + (band[3] - h * fit) / 2), round(w * fit),
            round(h * fit)]


def _on_words(drawn, box):
    """True when drawn covers WORDS_UNDER or more of the words standing in box (frame px)."""
    w = min(drawn[0] + drawn[2], box[0] + box[2]) - max(drawn[0], box[0])
    h = min(drawn[1] + drawn[3], box[1] + box[3]) - max(drawn[1], box[1])
    return w > 0 and h > 0 and w * h >= WORDS_UNDER * box[2] * box[3]


def _over_titles(state, default, drawn, a, b):
    """The titles drawn (frame px) from a to b covers WORDS_UNDER or more of: [(title, where its letters stand)].
    Game 10-30 (174316-0f5e): a sticker at the top went on over the title at 7 s, the star over "หลาย"."""
    return [(t, box) for t, box in render.title_boxes(state, default)
            if t["start"] < b and t["end"] > a and _on_words(drawn, box)]


UNDER_EFFECT = ("The title would stand under %s, and words on the clip are never covered. Give it another place, a box or a time "
                "clear of that, or take the effect off with change_effect and add it again in a clear spot.")


def _effects_over(state, words, a, b):
    """What is already on the clip from a to b that would stand over words (frame px), as it is drawn over them: talk 035009-dfe9
    (11-01) put a sticker on at 21.5 s and then its title, and the sticker sat on the first letter for 2 s."""
    return ["the sticker %s at %g s" % (k.get("name") or "picture", k["at"]) for k in state["stickers"]
            if k["at"] < b and k["at"] + k["length"] > a and _on_words(_sticker_drawn(k["png"], k["box"]), words)] + [
        "the %s flurry at %g s" % (f["move"], f["at"]) for f in state["flurries"]
        if f["at"] < b and f["at"] + f["length"] > a and _on_words(f["box"], words)]


# The first is the strip between a card's title and its picture, where a burst covers neither.
EFFECT_BOXES = ((120, 235, 880, 400), (300, 700, 700, 880), (300, 420, 700, 600), (60, 620, 400, 780), (600, 620, 940, 780),
                (340, 230, 660, 440), (60, 230, 380, 440), (620, 230, 940, 440))


def _clear_box(state, rec, default, at, length, drawn):
    """The first of EFFECT_BOXES clear of titles, faces and what a phone covers from at, for a refusal to name, or None:
    told only to give a box, Glimmer tried a card's one place 7 times and no goal of sport 032940-bc6c got its burst (11-01)."""
    times, every = _face_times(state, rec)
    for box in EFFECT_BOXES:
        spot = drawn(_frame_box("box", box))
        if not (_over_titles(state, default, spot, at, at + length) or _covered(spot, times, every, at, at + length)
                or _covers([round(v) for v in spot], state["clip"]["layout"], state)):
            return box
    return None


def _clear_box_said(refused, state, rec, default, at, length, drawn):
    """The refusal's sentence and the box clear at that second: on one goal of sport 033344-78ec Spark Open went from the
    title to the buttons to a face, 7 tries, and left that goal with no burst (11-01)."""
    box = _clear_box(state, rec, default, at, length, drawn)
    return "%s%s" % (refused, "\nClear of all that at %g s: the box %s." % (at, _on_grid(box)) if box else "")


def _title_in_the_way(what, state, rec, default, under, at, length, drawn):
    """Refused for a sticker or a flurry that would cover the titles under: where they are, and the layout's places
    that are clear at that time (drawn(place) is where it would stand there), or a box that is when none is."""
    clear = [name for name, spot in state["clip"]["layout"]["places"].items()
             if not _over_titles(state, default, drawn(spot), at, at + length)]
    box = None if clear else _clear_box(state, rec, default, at, length, drawn)
    raise Refused("The %s would cover the title %s: words on the clip are never covered. %s, or a time the title is not "
                  "shown." % (what, "; ".join('"%s" (%s, from %g to %g s)' % (
                      t["text"], ", ".join(layouts.on_grid(box[i], box[i] + box[i + 2], "xy"[i], inward=False)
                                           for i in (0, 1)), t["start"], t["end"]) for t, box in under),
                      "Clear of it at %g s: %s" % (at, ", ".join(clear)) if clear else
                      "No place is clear of it at %g s: give %s" % (
                          at, "the box %s" % _on_grid(box) if box else "a box away from it")))


def add_sticker(job, why, sticker, at, length=STICKER_LENGTH, place=None, box=None, person_said=None):
    asked = _asked(job, person_said)
    rec = _open(job, "put a sticker on")
    state = _clip_state(job)
    _watch_first(job, state)
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
    drawn = _sticker_drawn(png, band)
    default = job.chat.clip["style"] if job.chat.clip else "clean"
    under = _over_titles(state, default, drawn, at, at + length)
    if under and not asked:
        _title_in_the_way("sticker", state, rec, default, under, at, length, lambda spot: _sticker_drawn(png, spot))
    try:
        notes = _phone(drawn, state["clip"]["layout"], state, asked)
        more, hit = _off_faces(state, rec, drawn, at, at + length, asked, "sticker")
    except Refused as e:
        raise Refused(_clear_box_said(e, state, rec, default, at, length, lambda spot: _sticker_drawn(png, spot)))
    if under:
        notes.append("It covers the title %s, where the person asked for it; tell the person." % ", ".join(
            '"%s"' % t["text"] for t, _ in under))
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


FLURRY_SIZE = 150


def _flurry_size(pngs, box, move, count, length):
    """The pictures' size in a flurry inside box: FLURRY_SIZE px on their longer side, or less, 10 px at a time
    down to the smallest a flurry draws, so the copies move inside the box; Refused with the drawing's sentence when
    even that is too big, or when anything else is wrong."""
    error = None
    for size in range(FLURRY_SIZE, effects.FLURRY_SIZES[0] - 1, -10):
        try:
            effects.flurry(pngs, 0, box, move, count, length, size)
            return size
        except ValueError as e:
            if "too small" not in str(e):
                raise Refused(str(e))
            error = e
    raise Refused(str(error))


PICTURE_LENGTH = 3.0
PICTURE_LEAST = 480
PICTURE_MOST = 960


def add_picture(job, why, about, at, length=PICTURE_LENGTH):
    """Muse Image draws a picture of what is said (picture.make, from the words about alone) and it covers the top of
    the frame from `at` for length seconds, down to the highest face (or the product) on screen then and never past
    a card's top, at most to the middle of the frame, its lower edge fading into the footage, under the titles and
    captions. Tarm 10-30: a talk that shows a new picture on top as each thing is named holds the eye ("ตอนที่พูดถึง
    เรื่องผีก็มีภาพผี"), and an ad's picture over the top, above what it explains: "ถ้าเราสามารถใช้โมเดลของเมต้าทำภาพได้
    เลยก็จบ"; on a card with no face the picture ran to the middle of the frame, over the top of the slide."""
    rec = _open(job, "put a picture on")
    state = _clip_state(job)
    _watch_first(job, state)
    clip_length = state["length"]
    if not 0 <= at < clip_length:
        raise Refused("at %g s is outside the clip, which is %g s long: use seconds of the clip." % (at, clip_length))
    length = min(length, round(clip_length - at, 2))
    if length < 2 * effects.PICTURE_FADE:
        raise Refused("At %g s the clip ends %g s later; a picture needs %g s to come in and go out. Put it earlier."
                      % (at, round(clip_length - at, 2), 2 * effects.PICTURE_FADE))
    times, every = _face_times(state, rec)
    tops = [b[1] for t, boxes in times if at - every <= t <= at + length for b in boxes]
    card = [p["dst"][1] for p in state["clip"]["layout"]["parts"] if p["name"] == "card"]
    bottom = layouts._even(min(tops + card + [PICTURE_MOST]))
    if bottom < PICTURE_LEAST:
        raise Refused("From %g to %g s a face or the product comes up to y %d on your grid, so a picture above them "
                      "would be too short to see. Put it where they sit lower, or leave it out."
                      % (at, round(at + length, 2), layouts.grid(bottom, "y", up=False)))
    box = [0, 0, layouts.W, bottom]
    job.working("%s is drawing %s" % (picture.LABEL, " ".join(about.split())[:60]))
    try:
        got = picture.make(about)
        png = picture.fitted(got["path"], box[2], box[3])
    except picture.PictureError as e:
        got = {"usd": e.usd, "error": str(e)}
    finally:
        job.done_working()
    if got["usd"]:
        job.paid({"engine": {"read_new": None, "reused": None, "written": None, "usd": got["usd"]}},
                 types.SimpleNamespace(key="image", label=picture.LABEL, model_id=cover.model()))
    if got.get("error"):
        return "No picture: %s Tell the person, or go on without it." % got["error"]
    replaced = [k for k in state["pictures"] if k["at"] < at + length and at < k["at"] + k["length"]]
    state = job.edit({"do": "picture", "png": png, "at": round(at, 2), "length": length, "box": box,
                      "about": " ".join(about.split())})
    said = " ".join(about.split())
    lines = ["Kept the picture of %s (made before, at no cost)." % said if got["reused"] else
             "%s drew the picture of %s in %g s, for $%g." % (picture.LABEL, said, got["seconds"], got["usd"]),
             "It covers the top of the frame from %g to %g s, down to just above the %s (%s on your grid), its "
             "lower edge fading into the footage, under the titles and captions."
             % (round(at, 2), round(at + length, 2), "card" if card and bottom == layouts._even(min(card)) else
                "people", layouts.on_grid(0, bottom, "y", inward=False))]
    if replaced:
        lines.append("It replaces %s." % ", ".join("the picture of %s at %g s" % (k["about"], k["at"])
                                                     for k in replaced))
    lines.append(_show_frame(job, state, rec, round(at + length / 2, 2), "The clip with the picture:"))
    return "\n".join(lines)


# The examples in put_background's description. Glimmer gave the first word for word for an iPad seller twice
# (10-30, 155100-78b2 and 165330-f308): a gaming desk behind a tablet sold in a kitchen.
BACKDROP_EXAMPLES = ("a tidy gaming desk with soft blue light", "a bright kitchen counter")
EXAMPLE_PLACE = ("\"%s\" is the example in this tool's description, written for no clip in particular. Describe a place that "
                 "suits what this clip sells or is about, from what you saw in it. If the person asked for these very words, "
                 "put it on again with them.")


def _example_place(job, about):
    """True once a job when about is the tool's own example: 153416-7e05 copied it, and a batch found out after 12 minutes."""
    if " ".join(about.lower().split()).strip(" .") not in BACKDROP_EXAMPLES or "backdrop example" in job.warned:
        return False
    job.warned.add("backdrop example")
    return True


def put_background(job, why, about):
    """A new background behind the person, drawn by Muse Image for the words about (green.backdrop, from the words
    alone), wherever the clip does not show them, through the whole clip. A person in front of a green cloth comes off
    it by its colour (green.mask: FFmpeg on this computer, no GPU), once for the clip's stretch of the recording, even
    after SAM found them: the cloth's cut keeps what they hold, and SAM's person leaves out the product in their hands
    (live 10-30). With no cloth, the person found by SAM (cut_out) is used. The clip's layout stays as it is, so a
    seller clip keeps the
    product big: Tarm 10-30, "โปรแกรมต้องฉลาดพอจะรู้ว่าต้องโฟกัสสินค้ามากกว่าคน". Tarm 10-30, on recording a seller in
    front of a green cloth: "ตัดผ้าเขียวออก แล้วใส่พื้นหลังให้เหมาะสมด้วยโมเดลของเมต้า"."""
    rec = _open(job, "put a background behind")
    state = _clip_state(job)
    if _example_place(job, about):
        raise Refused(EXAMPLE_PLACE % about.strip())
    clip = state["clip"]
    a, b = min(p[0] for p in clip["pieces"]), max(p[1] for p in clip["pieces"])
    held = state["mask"]
    covered = held and held["from"] <= a + 0.01 and b - 0.01 <= held["to"]
    lines = []
    if covered and held.get("green"):
        lines.append("The person was already cut off the green cloth for this clip, so it was not done again.")
    else:
        job.working("Looking for a green cloth behind the person")
        try:
            mask = green.mask(rec, a, b)
        except green.GreenError as e:
            if not covered:
                raise Refused("%s Or find the person with cut_out (SAM 3.1), then put the background on again." % e)
            mask = None
        finally:
            job.done_working()
        if mask:
            state = job.edit(dict(mask, do="mask"))
            lines.append("The person was cut off the green cloth by its colour, on this computer with no GPU, with "
                         "what they hold; the cloth covers %d%% of the frame's edge, and anything it misses apart from "
                         "the person stays out." % round(mask["green"]["edge"] * 100))
        elif covered:
            lines.append("The person SAM 3.1 found for this clip is kept in front: what they hold is in front only "
                         "where SAM counted it as part of them.")
        else:
            raise Refused("No green cloth is behind the person, so the person must be found with SAM 3.1 first: call "
                          "cut_out, then put_background again.")
    job.working("%s is drawing the background" % picture.LABEL)
    try:
        got = green.backdrop(about)
    except picture.PictureError as e:
        got = {"usd": e.usd, "error": str(e)}
    finally:
        job.done_working()
    if got["usd"]:
        job.paid({"engine": {"read_new": None, "reused": None, "written": None, "usd": got["usd"]}},
                 types.SimpleNamespace(key="image", label=picture.LABEL, model_id=cover.model()))
    if got.get("error"):
        return "\n".join(lines + ["No background: %s Tell the person, or go on without it." % got["error"]])
    said = " ".join(about.split())
    state = job.edit({"do": "backdrop", "png": got["path"], "about": said})
    lines += ["Kept the background of %s (made before, at no cost)." % said if got["reused"] else
              "%s drew the background, %s, in %g s, for $%g." % (picture.LABEL, said, got["seconds"], got["usd"]),
              "It is behind the person wherever the clip does not show them, through the whole clip; the person, "
              "what they hold and the words stay in front.",
              _show_frame(job, state, rec, round(state["length"] / 2, 2), "The clip with the background:")]
    return "\n".join(lines)


def _sam_goes(job):
    """What person.cut_out needs to send the stretch to Meta's SAM 3.1, where the computer has no SAM: a Meta key, and,
    as for Muse Wonder, the switch in Options unless Muse is on Meta's servers already. Nothing without them."""
    if not asr.secret() or (_here(job) and not sends.allowed()):
        return {}
    return {"send": True, "on_paid": lambda usd: spent.add(os.path.join(recording.WORK, "spent.jsonl"), job.chat.id,
                                                           job.n, person.SAM, {"usd": round(usd, 6)})}


SAM_FOUND_NONE = "SAM found no "
AFTER_THE_NOUN = re.compile(r"\s+(?:with|in|on|and|of|for|from|near|inside)\s+")


def _thing_noun(what):
    """The thing itself in Muse's words, for SAM's second try: Muse called the blue Redmi a "black smartphone" and a tablet
    a "tablet with dog case", and SAM found neither, so 2 of 5 clips lost their frame and their title (11-01, 024201-3215);
    "blue smartphone" had followed it in 71% of the pictures on 005549-f1e8."""
    return AFTER_THE_NOUN.split(what, maxsplit=1)[0].split()[-1]


def follow_product(job, why, what):
    """SAM 3.1 finds what (the product, in English words) through the clip (person.cut_out with what as its prompt),
    and the vertical frame moves with it (focus.thing), so a product held out to the side stays in the picture. Tarm
    10-30, on a seller clip whose frame sat on his face while the controller he sold was at its edge:
    "โปรแกรมต้องฉลาดพอจะรู้ว่าต้องโฟกัสสินค้ามากกว่าคน ... ไม่ได้ขายหน้าตา"."""
    rec = _open(job, "follow the product in")
    state = _clip_state(job)
    clip = state["clip"]
    lay = clip["layout"]
    if lay["kind"] != "subject":
        raise Refused("Only the subject layout's frame can follow the product; this clip is in the %s layout: make it "
                      "again in the subject layout." % lay["kind"])
    if rec["height"] >= rec["width"]:
        raise Refused("The recording is already as tall as the frame, so the frame shows it whole and has nowhere to "
                      "move: frame the product with crop when you make the clip.")
    a, b = min(p[0] for p in clip["pieces"]), max(p[1] for p in clip["pieces"])
    face, plan = lay.get("face"), None
    said = " ".join(what.split())
    names = list(dict.fromkeys([said, _thing_noun(said)]))
    for said in names:
        job.working("SAM 3.1 is finding the %s: starting" % said)
        try:
            found = person.cut_out(rec, a, b, on_start=job.started_work, what=said,
                                   on_progress=lambda done, of: job.working(
                                       "SAM 3.1 is following the %s: picture %d of %d" % (said, done, of)),
                                   **_sam_goes(job))
        except ValueError as e:
            raise Refused(str(e))
        except person.Stopped:
            return "Not followed: the person pressed Stop."
        except person.NotSetUp:
            return _hold_product(job, rec, state, said)
        except person.PersonError as e:
            if said != names[-1] and str(e).startswith(SAM_FOUND_NONE):
                continue
            return "The %s could not be followed: %s Tell the person, or leave the frame where it is." % (said, e)
        finally:
            job.done_working()
        plan = focus.thing(found, clip["pieces"], (rec["width"], rec["height"]), layouts.W / float(layouts.H),
                           face[0] + face[2] / 2.0 if face else None)
        if plan:
            break
    if not plan:
        return ("SAM 3.1 did not see the %s anywhere in the clip, so the frame stays where it is. Name it as it looks "
                "in the picture, such as \"white game controller\"." % said)
    new = dict(layouts.followed(lay, plan, "whole"),
               thing=dict({k: found[k] for k in ("path", "fps", "from", "to")}, what=said))
    state = job.edit({"do": "layout", "part": "product", "box": None, "layout": new})
    shots = plan["shots"]
    lines = ["SAM 3.1 followed the %s through the clip: it is in %d%% of the pictures." % (said,
                                                                                          round(plan["seen"] * 100)),
             "The frame %s." % ("holds one place, %s on your grid, where the %s is" % (
                 _grid([shots[0]["x"], 0] + plan["size"], rec), said) if len(shots) == 1 else
                 "moves with it, %d places: %s" % (len(shots), "; ".join(
                     "%s on your grid from %g s" % (_grid([s["x"], 0] + plan["size"], rec), s["from"])
                     for s in shots)))]
    if plan["seen"] < THING_SEEN:
        lines.append("It is missing from most of the clip: where it is not seen the frame holds, so look at the "
                     "frames, and name it another way if SAM found the wrong thing.")
    cut = _product_cut(state["clip"], rec)
    if cut:
        lines.append(PRODUCT_CUT % (said, _spans(cut)))
    lines.append("Places for words now:")
    lines += _place_lines(state, rec)
    at = round((cut[0][0] + cut[0][1]) / 2 if cut else state["length"] / 2, 2)
    lines.append(_show_frame(job, state, rec, at, "The clip following the %s%s:" % (
        said, ", where the frame cuts it" if cut else "")))
    return "\n".join(lines)


# Touching the frame's edge was enough to pass "The product is in the frame": Glimmer's balm 10-31 passed with his face
# on all 6 sampled frames.
PRODUCT_SHOWN = 0.5


def _hold_product(job, rec, state, said):
    """follow_product with no SAM on this computer, as on anyone's who uses the API alone: the frame moves once, to the
    product box Muse named, and holds there. Spark 10-31 on Tarm's green-cloth seller with no SAM: the frame stayed on
    his face, no title found a place on any of the 4 clips, 3 failed the product check, and the 30 steps ran out with 1
    handed over."""
    clip = state["clip"]
    if not clip.get("product"):
        return ("SAM 3.1 is not on this computer, so nothing can follow the %s. Make the clip again with product: [x1, "
                "y1, x2, y2] on your grid around it, and the frame holds that place." % said)
    plan = focus.held(clip["product"], (rec["width"], rec["height"]), layouts.W / float(layouts.H))
    state = job.edit({"do": "layout", "part": "product", "box": None,
                      "layout": layouts.followed(clip["layout"], plan, "whole")})
    lines = ["With no SAM on this computer the frame does not follow the %s: it holds one place, %s on your grid, around "
             "your product box, for the whole clip." % (said, _grid([plan["shots"][0]["x"], 0] + plan["size"], rec)),
             "Places for words now:"]
    lines += _place_lines(state, rec)
    lines.append(_show_frame(job, state, rec, round(state["length"] / 2, 2), "The clip held on the %s:" % said))
    return "\n".join(lines)


PRODUCT_CUT = ("The frame cuts the %s or has left it at %s of the clip: SAM sees it there, but most of it is outside "
               "the frame. The frame there comes next: if the product is lost, tell the person what the frame cannot "
               "hold.")


def _product_cut(clip, rec):
    """focus.cut_off for the thing the clip's frame follows (follow_product), or None when it follows none."""
    lay = clip["layout"]
    if not lay.get("thing"):
        return None
    return focus.cut_off(lay["thing"], clip["pieces"], (rec["width"], rec["height"]), lambda t: next(
        p["src"] for p in layouts.parts_at(lay, t) if p["name"] == "whole"))


THING_SEEN = 0.5
# The check fails only where the frame loses the product this long at a time. On Glimmer's tablet (10-30, chat
# 152449-7de9) a 0.5 s cut, the frame catching up, failed it, and Muse moved the clip's ends to fix it, which made the
# clip again without its follow, background and title, and ran out of steps before it handed anything over.
PRODUCT_LOST = 1.0


def add_flurry(job, why, stickers, at, move="burst", count=8, length=None, place=None, box=None, person_said=None):
    asked = _asked(job, person_said)
    rec = _open(job, "put a flurry on")
    state = _clip_state(job)
    _watch_first(job, state)
    clip_length = state["length"]
    if not 0 <= at < clip_length:
        raise Refused("at %g s is outside the clip, which is %g s long: use seconds of the clip." % (at, clip_length))
    try:
        pngs = [n if n == effects.CONFETTI else effects.sticker_png(n) for n in stickers]
    except ValueError as e:
        raise Refused(str(e))
    if move not in effects.MOVES:
        raise Refused("move %r is not a flurry's move: use one of %s." % (move, ", ".join(effects.MOVES)))
    own = effects.MOVES[move][2] if length is None else length
    length = min(own, round(clip_length - at, 2))
    if length < effects.FLURRY_LENGTHS[0]:
        raise Refused("At %g s the clip ends %g s later; a flurry needs %g s to move. Put it earlier."
                      % (at, round(clip_length - at, 2), effects.FLURRY_LENGTHS[0]))
    band, where = _spot("box", state, place, box)
    size = _flurry_size(pngs, band, move, count, length)
    default = job.chat.clip["style"] if job.chat.clip else "clean"
    under = _over_titles(state, default, band, at, at + length)
    if under and not asked:
        _title_in_the_way("flurry", state, rec, default, under, at, length, lambda spot: spot)
    try:
        notes = _phone(band, state["clip"]["layout"], state, asked)
        more, hit = _off_faces(state, rec, band, at, at + length, asked, "flurry")
    except Refused as e:
        raise Refused(_clear_box_said(e, state, rec, default, at, length, lambda spot: spot))
    replaced = [f for f in state["flurries"] if project._same_box(f, {"box": band, "at": at, "length": length},
                                                                  "at", "length")]
    state = job.edit({"do": "flurry", "names": list(stickers), "pngs": pngs, "at": round(at, 2), "length": length,
                      "box": band, "move": move, "count": count, "size": size})
    said = effects.MOVES[move][0]
    lines = ["A %s flurry of %d copies of %s from %g to %g s in %s%s: they %s" % (
        move, count, " and ".join(stickers), at, round(at + length, 2), where,
        "" if hit else ", clear of every face the program found", said[0].lower() + said[1:])]
    if size < FLURRY_SIZE:
        lines.append("The pictures are %d px, not %d, so they move inside the box; give a bigger box for bigger "
                     "ones." % (size, FLURRY_SIZE))
    if length < own:
        lines.append("It ends with the clip, so the move takes %g s instead of %g s." % (length, own))
    lines += notes + more
    if replaced:
        lines.append("It replaces %s." % ", ".join("the %s flurry at %g s" % (f["move"], f["at"]) for f in replaced))
    lines.append(_show_frame(job, state, rec, at + length / 2.0, "The clip halfway through the flurry:"))
    return "\n".join(lines)


GLASS_TINTS ={"light": "the picture behind lightened, for dark words or busy colours",
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


# The house standard's five (recipes/house.md): the yeti review of 10-29 was handed over with eight.
MOST_TAGS = 5


# Glimmer gave EP 0's talk no title (11-01, 021005-389c): every recipe has one.
NO_TITLE = (" No title is on the clip, and a viewer scrolling past reads the title first: put one on with add_title and render "
            "again, unless the person asked for none.")
# Spark Open marked 1 of 4 goals (11-01, 033344-78ec) and Glimmer none (032940-bc6c): the recipe marks every goal alike.
NO_BURST = " No burst yet on %s, and every goal gets the same mark: put each on as its ball goes in and render again."


def _no_burst(job, state):
    """NO_BURST for the goals of a match's clip, one title each, with no flurry while their title is shown; or nothing."""
    bare = [t for t in state["titles"] if not any(f["at"] < t["end"] and f["at"] + f["length"] > t["start"]
                                                  for f in state["flurries"])]
    return NO_BURST % ", ".join('"%s"' % t["text"] for t in bare) if bare and (job.chat.clip or {}).get("recipe") == "sport" else ""


def render_clip(job, why):
    rec = _open(job, "render")
    state = _clip_state(job)
    left = _undecided(state, None if why == BATCH_WHY else job)
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
            "size": os.path.getsize(out), "length": state["length"], "moment": state["clip"]["moment"]}
    job.chat.rendered.append(made)
    job.emit("rendered", file=made)
    job.show(rendered=made)
    return ("Rendered the finished clip in %s: %g s, 1080x1920, %.1f MB. The person can play it on the page now. Hand "
            "it over next." % (recording.clock(made["seconds"]), made["length"], made["size"] / 1e6)) + (
        NO_TITLE if (job.chat.clip or {}).get("recipe") and not state["titles"] and why != BATCH_WHY else _no_burst(job, state))


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


NAMED_NUMBER = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]*[a-z][A-Z][A-Za-z]*[ -]?|[A-Z]{1,2})\d{1,2}(?![\d,.])"
                          r"(?!\s*(?:(?i:baht|thb)|บาท|฿|%))")


def _not_said(words, said):
    """The numbers in words that are not among the numbers in said (spoken.values: digits in any script, Thai and
    English number words). A number in what can only be a product's name, a word with a capital inside it or capitals
    glued to the digits, is left out: 10-31 053131-51de, the name check took PlayStation 5 in place of PS5, this one failed
    its 5, and 8 steps went on it. Any other word is left in, as "Only 5 left" and "Flash Sale 50%" are claims: a name
    wrongly refused costs steps, a claim wrongly passed goes on a seller's post (lane 1)."""
    heard = set().union(*[spoken.values(s) for s in said]) if said else set()
    return [m for m in spoken.DIGITS.findall(NAMED_NUMBER.sub("", words)) if spoken.digits(m) - heard]


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
            sentences.setdefault(line["phrase"], [line["said"], "", line.get("no_number")])[1] += line["text"]
    if sentences:
        differ = [('"%s" for "%s"' % (t.strip(), s), shown) for s, t, shown in sentences.values()
                  if _numbers_differ(s, t)]
        off = [d for d, shown in differ if not shown]
        read = [d for d, shown in differ if shown]
        check("Translated captions keep the numbers", not off,
              "%d of %d sentences keep every number said" % (len(sentences) - len(differ), len(sentences)) +
              ("; not kept: %s; put the captions on again and those lines are translated again"
               % "; ".join(off[:3]) if off else "") +
              ("; Muse was shown the number and wrote it as no number, which the person can check: %s"
               % "; ".join(read[:3]) if read else ""))
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
    cut = _product_cut(state["clip"], rec) if kind == "seller" else None
    if cut is not None:
        thing = state["clip"]["layout"]["thing"]["what"]
        lost = [(a, b) for a, b in cut if b - a >= PRODUCT_LOST]
        check("The product is in the frame", False if lost else None if cut else True,
              "the frame cuts the %s or has left it at %s, while SAM sees it%s" % (
                  thing, _spans(cut), "" if lost else "; each under %g s, the frame catching up with the hands: look "
                                                      "at those seconds" % PRODUCT_LOST) if cut else
              "SAM followed the %s, and the frame holds it whenever it is seen" % thing, "program" if lost or not cut
              else "Muse")
    elif kind == "seller" and state["clip"]["layout"]["kind"] in ("talk", "speaker", "split"):
        check("The product is in the frame", None, "the person asked for a layout that cuts between faces")
    elif kind == "seller":
        product = state["clip"].get("product")
        inside = bool(product) and layouts.shown_share(state["clip"]["layout"], product) >= PRODUCT_SHOWN
        check("The product is in the frame", inside,
              "the product box is inside the clip's picture" if inside else
              "no product box was named and the frame follows nothing, so nothing shows the product is in the picture: "
              "make the clip again with its product, or call follow_product" if not product else
              "under half of the product box is inside the clip's picture: frame it again around the product, or "
              "choose the card layout to show all of it")
    else:
        check("The product is in the frame", None, "not a seller's clip")
    check("The length fits a Reel", made["seconds"] <= REEL_LONGEST,
          "%.1f s; a Reel is at most %d s" % (made["seconds"], REEL_LONGEST))
    check("The cuts are in quiet sound", *_cuts_check(rec, state, kind))
    black, silent = _start_of(path, made["sound_tracks"])
    heard = _sound_at(state, path) if silent and not black else None
    muted = silent and not black and state["mutes"] and state["mutes"][0][0] <= project.SLIVER
    check(OPENING, None if muted else not (black or silent),
          "silent for the first %g s, where the sound is muted (0-%g s) as asked: music under it would fill it"
          % (FIRST, state["mutes"][0][1]) if muted else
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
    moment = state["clip"]["moment"]
    failed = {c["name"] for c in _checked_before(job, moment) if c["ok"] is False}
    for c in checks:
        if c["ok"] and c["name"] in failed:
            c["fixed"] = True
    job.chat.checks[done["name"]] = checks
    job.emit("checked", file=done["name"], moment=moment, checks=checks)
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
    elif not _here(job):
        out.append("Make the cover next (make_cover), then hand it over.")
    else:
        out.append("Hand it over next.")
    return "\n".join(out)


def _here(job):
    """True when the job's Muse runs on this computer (no key), so nothing of the person's leaves it unasked."""
    return not getattr(job.brain, "key_env", None)


def _cover_second(job, state):
    """The clip second the cover is drawn from: the peak of the moment the clip was made from, else its middle."""
    clip, moments = state["clip"], job.chat.moments or []
    n = clip.get("moment")
    m = moments[n - 1] if isinstance(n, int) and 1 <= n <= len(moments) else {}
    peak = project.from_recording(clip["pieces"], m["t"]) if m.get("t") is not None else None
    return min(max(0.0, peak if peak is not None else state["length"] / 2), state["length"] - 0.1)


def make_cover(job, why, title, again=False, asked=False):
    rec, state = _open(job, "make a cover for"), _clip_state(job)
    done = job.chat.finished()
    if not done:
        raise Refused("There is no finished file made from the clip as it is now. Render the clip first, then make "
                      "its cover.")
    if _here(job) and not asked and not sends.allowed():
        raise Refused("You work on this computer, so no picture of the person's leaves it unasked: %s draws the cover "
                      "on Meta's servers, with their key, for $%.2f. Make it only when the person asks for a cover or "
                      "taps Make the cover; otherwise hand the clip over without one, and never say it has a cover." % (
                          cover.LABEL, cover.USD))
    at = _cover_second(job, state)
    try:
        frame = render.frame(rec, state, at, (job.chat.clip or {}).get("style") or "clean", width=cover.WIDTH,
                             words=False)
    except render.NotDrawn as e:
        raise Refused("The frame for the cover could not be drawn: %s" % e)
    job.working("%s is drawing the cover" % cover.LABEL)
    try:
        got = cover.make(frame, title, _spoken(job, [title]), again=again)
    except cover.CoverError as e:
        got = {"usd": e.usd, "error": str(e)}
    finally:
        job.done_working()
    if got["usd"]:
        job.paid({"engine": {"read_new": None, "reused": None, "written": None, "usd": got["usd"]}},
                 types.SimpleNamespace(key="image", label=cover.LABEL, model_id=cover.model()))
    if got.get("error"):
        raise Refused(got["error"])
    name = os.path.basename(got["path"])
    job.emit("cover", file=done["name"], cover=name, title=" ".join(title.split()), usd=got["usd"])
    job.attach(got["path"], "The cover, as the person sees it:")
    return ("%s the cover from the clip at %g s with the words \"%s\"%s. It comes next as a picture: check that the "
            "words are spelled as you wrote them and whole, and that faces and the product are as shot. If not, make "
            "it again with again: true; then hand the clip over, and the person gets the cover with it." % (
                "Kept" if got["reused"] else "%s drew" % cover.LABEL, round(at, 2), " ".join(title.split()),
                " (made before, at no cost)" if got["reused"] else " in %g s, for $%.2f" % (got["seconds"],
                                                                                           got["usd"])))


def _checked_before(job, moment):
    """The checks of the last time this recording's clip from moment was checked, or []: a check that failed then
    and passes now is marked fixed, so the page shows a red line turn green."""
    for e in reversed(job.chat.events):
        if e["kind"] == "opened":
            return []
        if e["kind"] == "checked" and e.get("moment") == moment:
            return e["checks"]
    return []


REVIEWED = (0.12, 0.37, 0.62, 0.87)
FINISHED = "The finished clip, picture"
LOOK_AT_IT = ("%d pictures of the finished clip come next, as a viewer sees it. Before you hand it over, look at each "
              "for: a face blurred that should show, or one the person asked to blur that shows; words over a "
              "face%s; words cut off at an edge or too small to read; a head or the action cut off by the frame%s. "
              "Say in your next why what you saw. If something is wrong, fix it, render again and check again.")

_odd_thai = words.odd_thai
_split_up = words.split_up


def _spell(job, texts, heard=False):
    """(texts with the Thai spelling fixed in a short conversation of Muse's own, a sentence naming each fix or "").
    Texts Muse wrote have Glimmer's slips put back first (words.unslipped), which Muse's check passed: ป๊อปคัลเจอร
    in a trip clip's title (10-29), as the dictionary has the bare syllables and not the right word. Words heard are
    not: on 1,800 heard lines the swaps changed 15, most of them wrong (รองว้ย to รองวัย), as hearing slips are not
    Glimmer's. The ์ goes back after Muse's check too, whose rules would keep it dropped again: the dictionary flags
    คัลเจอร์ and not คัลเจอร. When Muse cannot be asked, the texts stay as they are then. Stop raises hearing.Stopped."""
    first = list(texts) if heard else [words.unslipped(t) for t in texts]
    try:
        fixed, _ = words.spelled(lambda m, s: job.ask_aside(m, s, "Checking the Thai spelling"), first,
                                 every=True, kept=job.proofread)
    except mind.MindError:
        fixed = first
    fixed = fixed if heard else [words.marked(t) for t in fixed]
    return fixed, " ".join('The program fixed the Thai spelling: "%s" is now "%s".' % (" ".join(a.split()),
                                                                                       " ".join(b.split()))
                           for a, b in zip(texts, fixed) if isinstance(a, str) and a != b)


def _tag(word):
    word = "".join(word.split()).lstrip("#")
    return "#" + word if word else ""


# A folder or file of this computer, as a drive (H:\Phone), a share (\\host) or a home folder; not a web address,
# whose "s://" a letter comes before. Seen 10-28: Glimmer's post read "จากโฟลเดอร์ H:\Phone\yeti".
PLACES = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?!/)\S*|\\\\[^\s\\]+\S*|(?:/Users|/home)/\S+")
# Another company's short-video app, or a tag that belongs to one (#fyp is TikTok's For You page). The studio makes clips
# for Reels on Instagram and Facebook, so Muse's post and its words on the clip name none (Tarm 10-28: "# ไม่ควรมี
# tiktok", "Reel IG Facebook ใช้ได้"). Words the speaker said are theirs: captions are not checked.
OTHER_APPS = re.compile(r"(?i)tik\s*-?\s*tok|ต[ิี][๊๋]?ก\s*-?\s*ต[็๊๋]?อ[็๊๋]?ก|ท[ิี][๊๋]?ก\s*ท[็๊๋]?อ[็๊๋]?ก|douyin|"
                        r"you\s*tube|ยูทู[บป]|#\s*(?:yt)?shorts\b|capcut|snapchat|kuaishou|\bkwai\b|likee|lemon8|"
                        r"#\s*fyp\w*|#\s*foryou(?:page)?\b")


# A small Latin letter or two inside a Thai word is a slip, not a name: Glimmer wrote "ซื้oมาขายไป" in a seller's post
# title and twice in its post (10-30, 164011-1c04), and every check passed. Capitals are left alone, as TV or AI sit
# between Thai words as they are.
SLIPPED = re.compile(r"[ก-๛]+[a-z]{1,2}[ก-๛]+")


def _latin_in_thai(*texts):
    """Refused naming each Thai word with a small Latin letter or two inside it, or nothing."""
    found = list(dict.fromkeys(m.group(0) for m in SLIPPED.finditer(" ".join(t for t in texts if t))))
    if found:
        raise Refused("%s %s a Latin letter inside a Thai word, a slip a viewer reads as a typo. Write %s in Thai "
                      "letters only, then try again." % (", ".join('"%s"' % w for w in found),
                                                         "has" if len(found) == 1 else "have",
                                                         "it" if len(found) == 1 else "them"))


def _other_apps(*texts):
    """Refused naming another company's app in texts, or nothing."""
    named = list(dict.fromkeys(m.group(0) for m in OTHER_APPS.finditer(" ".join(t for t in texts if t))))
    if named:
        raise Refused("This names %s, which belongs to another company's app. The studio makes clips for Reels on "
                      "Instagram and Facebook, so the title, the post and the hashtags name no other app and use none "
                      "of its tags. Write them without it." % ", ".join('"%s"' % n for n in named))


SCORE = re.compile(r"(?<![\d:.-])(\d{1,2})\s*[-\u2013]\s*(\d{1,2})(?![\d:.-])")
IN_A_ROW = re.compile(r"(\d+)\s*(?:ประตู|ลูก)\s*รวด|(\d+)\s*(?:goals?\s+in\s+a\s+row|straight\s+goals?|unanswered\s+goals?)", re.I)
SCORE_SLIP = ("Not handed over: %s does not fit this match. By the clip's own titles the goals went %s. Write the title and the post "
              "from those scores; if what you wrote is true of the match, hand over again unchanged.")


def _score_slips(state, post):
    """(what post says that the goals' titles do not, the scores they give): a score that never stood, or more goals in a row
    than came. 033344-78ec (11-01) read every score right and posted 4 ประตูรวด for a side on a game of 0-1, 0-2, 1-2, 1-3."""
    titles = [[tuple(int(n) for n in m.groups()) for m in SCORE.finditer(t["text"])]
              for t in sorted(state["titles"], key=lambda t: t["start"])]
    went = [found[-1] for found in titles if found]
    if len(went) < 2:
        return [], []
    known = {s for found in titles for s in found} | {s[::-1] for found in titles for s in found}
    most, run, side, was = 0, 0, None, (0, 0)
    for now in went:
        home = now[0] > was[0]
        run = run + 1 if home == side else 1
        most, side, was = max(most, run), home, now
    return (['"%s"' % m.group(0) for m in SCORE.finditer(post) if tuple(int(n) for n in m.groups()) not in known] +
            ['"%s"' % m.group(0) for m in IN_A_ROW.finditer(post) if int(m.group(1) or m.group(2)) > most],
            ["%d-%d" % s for s in went])


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
    if failed and rounds < CHECK_ROUNDS and not job.finishing:
        raise Refused("These checks failed on the finished file: %s. Fix them, render again and check again; after "
                      "%d rounds you may hand it over and say what still fails." % ("; ".join(failed), CHECK_ROUNDS))
    rec, state = _open(job, "hand over"), _clip_state(job)
    # Glimmer read past NO_TITLE in render_clip's answer twice (021005-389c, 050402-c12f on 11-01); a refusal it obeys.
    if (job.chat.clip or {}).get("recipe") and not state["titles"] and why != BATCH_WHY and not job.finishing \
            and not _told_once(job, "no_title"):
        raise Refused("Not handed over." + NO_TITLE + " If they did, hand over again unchanged.")
    where = [m.group(0) for m in PLACES.finditer(" ".join([title, caption, english or ""] + list(hashtags)))]
    if where:
        raise Refused("The post names %s, a place on this computer: viewers must never see where the files are kept. "
                      "Write the title and the post without it." % ", ".join('"%s"' % w for w in where))
    _other_apps(title, caption, english, *[_tag(h) for h in hashtags])
    _latin_in_thai(title, caption, *hashtags)
    copied = _labels_copied(job, rec, state, [title, caption, english], "The title or the post")
    if copied:
        raise Refused("Not handed over. " + copied)
    wrong = _misspelled(" ".join([title, caption, english or ""] + list(hashtags)), _screen_words(job, rec, state))
    if wrong and not any(e["kind"] == "misspelled" and e.get("job") == job.n for e in job.chat.events):
        job.emit("misspelled", words=[w for w, _ in wrong])
        raise Refused(" ".join(_names_said(wrong, "write them as the screen does")) + " If the speaker said them as "
                      "you wrote them, hand over again unchanged.")
    pieces = state["clip"]["pieces"]
    names, shown = _unseen_names(job, rec, min(p[0] for p in pieces), max(p[1] for p in pieces), " ".join(
        [title, caption, english or ""] + list(hashtags)), _said(job, state))
    if names and not _told_once(job, "unseen"):
        raise Refused("%s. %s" % (_unseen_said(names, shown), UNSEEN_THEN % "hand over again"))
    if (job.chat.clip or {}).get("recipe") == "sport":
        slips, went = _score_slips(state, " ".join([title, caption, english or ""]))
        if slips and not _told_once(job, "score_post"):
            raise Refused(SCORE_SLIP % (", ".join(slips), ", ".join(went)))
    if (job.chat.clip or {}).get("recipe") == "seller":
        said = _said(job, _clip_state(job))
        wrong = _not_said(" ".join([title, caption, english or ""]), said)
        if wrong and not _numbers_unread(job, said):
            raise Refused("The post text has numbers that were not said: %s. Use only numbers the seller said; one "
                          "that was not said comes out, it is not written in words instead." % ", ".join(wrong))
    caption, more = _without_tags(caption)
    english, also = _without_tags(english)
    tags = list(dict.fromkeys(t for t in (_tag(w) for w in list(hashtags) + more + also) if t))
    if len(tags) > MOST_TAGS:
        raise Refused("%d hashtags (%s): give at most %d, the first three about what the viewer is looking for or "
                      "struggling with, because the apps show the first three, then two about the topic or the "
                      "channel. Each more says less about this clip." % (len(tags), " ".join(tags), MOST_TAGS))
    written = title
    try:
        (title, caption, *tags), spelling = _spell(job, [title, caption] + tags)
    except hearing.Stopped:
        return "Not handed over: the person pressed Stop."
    odd = [] if why == BATCH_WHY or job.n is None or job.finishing else _odd_thai(title) or _split_up(written, title)
    if odd and not _told_once(job, "odd_post"):
        raise Refused(ODD_TITLE % ("Not handed over", ", ".join('"%s"' % bit for bit in odd)))
    odd = [] if why == BATCH_WHY or job.n is None or job.finishing else _odd_thai(caption)
    if odd and not _told_once(job, "odd_caption"):
        raise Refused(ODD_POST % ", ".join('"%s"' % bit for bit in odd))
    tags = list(dict.fromkeys(t for t in (_tag(w) for w in tags) if t))
    if not tags:
        raise Refused("Give at least one hashtag, such as #gaming.")
    mine = _tag(((channels.in_use() or {}).get("handle") or "").lstrip("@"))
    if mine and mine.lower() not in {t.lower() for t in tags}:
        tags.append(mine)
    made = job.chat.cover_of(done["name"])
    ready = {"file": done["name"], "moment": state["clip"]["moment"], "length": done["length"], "title": title,
             "caption": caption, "hashtags": tags,
             "english": english, "checks": checks, "work": job.chat.worked(job.n, job.steps),
             "cover": made}
    job.emit("ready", **ready)
    job.show(ready=ready)
    if failed and job.finishing:
        job.emit("note", text="Muse ran out of steps, so this clip was handed over with %s: %s." % (
            "a check that failed" if len(failed) == 1 else "checks that failed", "; ".join(failed)))
    pieces = _clip_state(job)["clip"]["pieces"]
    return ("Handed over: the person sees the finished clip%s with its title, the post text and %d hashtags, a Copy "
            "all button and a download. Tell them in one or two short lines; do not repeat the text, it is on the "
            "page. The clip is %g s, from %s to %s of the recording: when you say its length or its times, say these, "
            "not the ones asked for." % (" and its cover" if made else "", len(tags), round(done["length"], 2),
                                         recording.clock(min(p[0] for p in pieces)),
                                         recording.clock(max(p[1] for p in pieces)))) + (
        " " + spelling if spelling else "") + (
        " It went over out of steps, and these checks still fail on it: say so: %s." % "; ".join(failed)
        if failed and job.finishing else "")


LAYOUT_OF = {"explain": "card", "game": "card", "talk": "talk", "seller": "subject", "vlog": "card"}
BATCH_WHY = "one of the clips the person asked for"


# 082547-8596 on 11-01: 5 clips, 4 handed over and the Samsung held back; Glimmer made all 5 again, 13 minutes, and its memory
# filled with the job's calls and answers, so the job ended with no last word to the person.
HANDED_ALREADY = ("Nothing was made. This job handed over the clip of moment %s already, and a clip takes minutes: leave it out and "
                  "make only the clips still held back. To change a clip that was handed over, open it with open_clip, or make "
                  "it again with make_clip.")


def _handed_here(job, clips):
    """The moments of clips that at least half of a clip this job handed over shares, or at least half of the moment. It goes by
    the seconds of the recording, as proposing again numbers the moments anew, and a clip can be much shorter than its moment."""
    handed = []
    for place in job.chat.album:
        ready, clip = job.chat.ready_of(place), place["project"].state()["clip"]
        if ready and ready.get("job") == job.n and clip:
            handed.append(clip["pieces"])

    def shared(m, pieces):
        both = sum(max(0.0, min(b, m["end"]) - max(a, m["start"])) for a, b in pieces)
        return 2 * both >= min(m["end"] - m["start"], sum(b - a for a, b in pieces))

    return [c["moment"] for c in clips if any(shared(job.chat.moments[c["moment"] - 1], p) for p in handed)]


def make_clips(job, why, clips, blur=None):
    rec = _open(job, "make clips from")
    moments = job.chat.moments or []
    if not moments:
        raise Refused("No moments are proposed yet. Look at the recording and propose moments first: each clip is made "
                      "from one of them.")
    if len(clips) < 2 and not job.chat.album:
        raise Refused("clips must have 2 to %d items; it has 1. For one clip, make_clip." % MOST_AT_ONCE)
    numbers = [c["moment"] for c in clips]
    wrong = [n for n in numbers if not 1 <= n <= len(moments)]
    if wrong:
        raise Refused("There are %d moments, so %s %s not one of them." % (
            len(moments), _numbers(wrong), "is" if len(wrong) == 1 else "are"))
    if len(set(numbers)) < len(numbers):
        raise Refused("Each moment makes one clip; moment %s is given twice." % _numbers(
            sorted({n for n in numbers if numbers.count(n) > 1})))
    handed = _handed_here(job, clips)
    if handed:
        raise Refused(HANDED_ALREADY % _numbers(handed))
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
    unseen = []
    for c in clips:
        m = moments[c["moment"] - 1]
        names, shown = _unseen_names(job, rec, m["start"], m["end"], " ".join(
            [c["title"], c.get("words") or "", c["caption"]] + list(c["hashtags"])), [p["text"] for p in m.get("said")
                                                                                       or []])
        if names:
            unseen.append("Clip of moment %d: %s." % (c["moment"], _unseen_said(names, shown)))
    if unseen and not _told_once(job, "unseen"):
        raise Refused("Nothing was made. %s %s" % (" ".join(unseen), UNSEEN_THEN % "make the clips again"))
    recipe = (job.chat.clip or {}).get("recipe")
    usual = _saved(recipe)[0] or LAYOUT_OF.get(recipe, "subject")
    unasked = _layout_unasked(job, recipe, [c.get("layout") or usual for c in clips])
    if unasked:
        raise Refused("Nothing was made. " + unasked)
    if any(_blurs_the_picture(_pixels("blur", box, rec), rec) for box in blur or ()):
        raise Refused("Nothing was made. " + _blur_wide(job))
    if recipe == "game" and not blur and not _no_chat(job):
        raise Refused("Nothing was made. A game stream's clips show its chat box with the viewers' names: give the "
                      "chat box in blur, on your 0-1000 grid of the pictures from your looks, and every clip hides it. "
                      "Leave the face camera out of the box. If your looks show no chat box, say so with no_chat_box.")
    if recipe == "seller":
        wide = rec["width"] > rec["height"]
        bare = [c["moment"] for c in clips if not c.get("product") or wide and not c.get("what")]
        if bare:
            raise Refused("Nothing was made. A seller's clip shows its product: give the clip of moment %s its product, "
                          "the box around it in the picture%s." % (_numbers(bare), ", and what, the product in a few "
                                                                   "English words, so the frame follows it" if wide
                                                                   else ""))
        plain = [c["moment"] for c in clips if not c.get("background") and _cloth(
            rec, moments[c["moment"] - 1]["start"], moments[c["moment"] - 1]["end"])]
        if plain:
            raise Refused("Nothing was made. A green cloth is behind the person in moment %s: give each clip a "
                          "background, a place that suits its product, so the cloth does not show." % _numbers(plain))
        copied = next((c["background"] for c in clips if _example_place(job, c.get("background") or "")), None)
        if copied:
            raise Refused("Nothing was made. " + EXAMPLE_PLACE % copied.strip())
    pictures, lines, made, looked = len(job.pictures), [], [], []
    job.chat.hold_open_clip()
    since = job.emit("batch")["id"]
    for k, c in enumerate(clips, 1):
        if job.stopped:
            lines.append("The person pressed Stop before clip %d; it and the clips after it were not made." % k)
            break
        job.working("Making clip %d of %d, from moment %d" % (k, len(clips), c["moment"]))
        job.chat.project = project.Project()
        lines.append(_one_clip(job, rec, k, c, c.get("layout") or usual, blur or []))
        seen = [p for p in job.pictures[pictures:] if p[1].startswith(FINISHED)][-len(REVIEWED):]
        del job.pictures[pictures:]
        looked += [(path, "Clip %d: %s" % (k, caption)) for path, caption in seen]
        if job.chat.project.entries:
            made.append({"moment": c["moment"], "title": c["title"], "project": job.chat.project})
    job.pictures += looked
    job.chat.fill_album(made, since)
    if made:
        job.show(clip=job.chat.clip_view())
    done = sum(1 for line in lines if " handed over" in line and "not handed over" not in line)
    look = channels.in_use()
    return "\n".join(["Made %d clip%s, %d handed over; each is reported on its own:" % (
        len(lines), "" if len(lines) == 1 else "s", done)] + lines + ([
        'Every clip is in the "%s" channel look the person saved: %s.' % (look["name"], look_said(look))] if look
                                                                          else []) + ([
        "The program wrote names as the screen shows them: %s." % "; ".join(spelled)] if spelled else []) + [
        "Each clip's finished file comes next in %d pictures, as a viewer sees it: look at them as check_clip asks "
        "and say what you saw. When one shows something wrong, make that clip again with make_clip and fix it." %
        len(REVIEWED),
        "The person sees the clips as an album above the clip and a ready card for each one handed over. The first "
        "clip is open: your tools change the clip that is open, and open_clip opens another. Tell the person in one or "
        "two lines which clips are ready and what failed in the others."])


def match_album(job, why, parts):
    _open(job, "change")
    _clip_state(job)
    if len(job.chat.album) < 2:
        raise Refused("There is no other clip to change: the album holds the clips made in one go.")
    return job.chat.match_album(parts, job.n)


def undo_album(job, why):
    return job.chat.unmatch_album(job.n)


def _changed_clips(chat):
    """The album's places whose finished video is no longer the clip as it is now."""
    return [k for k, c in enumerate((chat.album_now() or {"clips": []})["clips"]) if c["state"] == "changed"]


def render_album(job, why):
    chat = job.chat
    todo = _changed_clips(chat)
    if not todo:
        raise Refused("No clip of the album has changed since its video was made, so there is nothing to make again.")
    back, said = chat.album_at, []
    for k in todo:
        if k != chat.album_at:
            chat.open_album_clip(k)
        try:
            done = render_clip(job, why)
        except Refused as e:
            done = str(e)
        said.append("Clip %d: %s" % (k + 1, done))
        if done.startswith("Not rendered"):
            break
    if back != chat.album_at:
        chat.open_album_clip(back)
    return "\n".join(said)


def set_tagline(job, why, words, spot=None):
    _open(job, "put a tagline on")
    state = _clip_state(job)
    drawn = " ".join((words or "").split())
    job.edit({"do": "channel", "name": drawn, "spot": spot or (state["channel"] or {}).get("spot")})
    return 'The tagline "%s" is on the clip.' % drawn if drawn else "The tagline is off the clip."


def open_clip(job, why, clip):
    """Opens a clip of the album, as a tap on it does: every tool changes the clip that is open, and the one left keeps
    its edits and its undo. Tarm 10-30: "ทุกเครื่องมือ ต้องแชทได้"."""
    album = job.chat.album
    if not album:
        raise Refused("There is no album: it holds the clips made in one go, and there is only the clip that is open.")
    if not 1 <= clip <= len(album):
        raise Refused("The album has %d clips, so clip %d is not one of them." % (len(album), clip))
    if clip - 1 != job.chat.album_at:
        job.chat.open_album_clip(clip - 1)
    return "Clip %d of %d is open. %s" % (clip, len(album), job.chat.clip_line())


TITLE_HOLDS = 3.0


def _title_order(stretches, last=()):
    """stretches ([(from, to, place)]) in the order to try them: the earliest of those a viewer can read a title in,
    TITLE_HOLDS s or more, then the longer of the others; the places in last after the others of their kind."""
    return sorted(stretches, key=lambda s: (s[1] - s[0] < TITLE_HOLDS, s[2] in last,
                                            s[0] if s[1] - s[0] >= TITLE_HOLDS else s[0] - s[1]))


def _title_where_clear(job, rec, words):
    """Puts the title up in a stretch of a layout place that covers no face (_title_order), and says where and when;
    None when no stretch takes it. Seen 10-28 on a Q&A where both talkers fill the frame in turn: two of three clips went
    out with no title, as no place was clear for the whole clip. A seller's clip has its product in the middle: 073756-0c52
    on 11-01 put the iPad's title on the iPad's screen, so the middle comes after any stretch at the top a viewer can read."""
    state = _clip_state(job)
    lay, length = state["clip"]["layout"], state["length"]
    times, every = _face_times(state, rec)
    stretches = _title_order([(a, b, name) for name, box in lay["places"].items()
                              for a, b in _clear(_covered(box, times, every, 0, length), 0, length)],
                             ("middle",) if (job.chat.clip or {}).get("recipe") == "seller" else ())
    for a, b, name in stretches:
        start, end = math.ceil(a * 10) / 10, math.floor(b * 10) / 10
        try:
            add_title(job, BATCH_WHY, words, start, end, place=name)
        except Refused:
            continue
        return "at the %s from %g to %g s" % (name, start, end)
    return None


def _cloth(rec, a, b):
    """True when a green cloth that can be told from the person is behind them from a to b in the recording."""
    try:
        return bool(green.find(rec["path"], a, b, (rec["width"], rec["height"]), samples=2))
    except green.GreenError:
        return False


def _on_the_product(job, rec, c, notes):
    """A seller's clip of make_clips, just made, gets what one made by hand gets: the frame follows its product when
    the recording has room to move in (follow_product), and its background goes behind a person on a green cloth
    (put_background). Notes what came of each; False when the person pressed Stop. Spark 10-30, 205011-1a6e: four clips
    of Tarm's green-cloth seller came back "handed over, 4 of 4 checks" with a face filling the frame, the product out
    of the picture, the raw cloth and no place for the title."""
    clip = _clip_state(job)["clip"]
    if c.get("what") and clip["layout"]["kind"] == "subject" and rec["width"] > rec["height"]:
        said = follow_product(job, BATCH_WHY, c["what"])
        if said.startswith("Not followed"):
            return False
        notes.append(said.splitlines()[0].rstrip("."))
    a, b = min(p[0] for p in clip["pieces"]), max(p[1] for p in clip["pieces"])
    if c.get("background") and _cloth(rec, a, b):
        said = put_background(job, BATCH_WHY, c["background"])
        notes.append(next((line for line in said.splitlines() if line.startswith("No background")),
                          "the background is %s" % " ".join(c["background"].split())).rstrip("."))
    elif c.get("background"):
        notes.append("no green cloth is behind the person, so the background was left out")
    return True


TITLE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _one_clip(job, rec, k, c, layout, blur, opened=None):
    """One clip of make_clips, from its moment to the hand-over: the sentence reporting on it. Its words are
    proofread together first, so the title on the clip is spelled as the post text is. A seller's clip is framed on
    its product first (_on_the_product). Each box in blur is
    blurred for the whole clip. When the only check that fails is the silent opening, the moment is moved to its
    first sound (_open_on_sound) and the clip is made once more; opened is the seconds it moved. Seen 10-27 on a 4-hour game stream: six clips were made without the chat box blurred,
    none passed the check for viewers' names, and Muse ran out of room fixing the first one by hand."""
    n, head = c["moment"], "Clip %d (moment %d)" % (k, c["moment"])
    try:
        _spell(job, [c["title"], c.get("words") or "", c["caption"]] + list(c["hashtags"]))
        titled = TITLE_NAME.findall(" ".join([c["title"], c.get("words") or ""])) if (
            job.chat.clip or {}).get("recipe") == "seller" else ()
        made = make_clip(job, BATCH_WHY, n, layout, product=c.get("product"), titled=list(dict.fromkeys(titled)))
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
        if (job.chat.clip or {}).get("recipe") == "seller" and not _on_the_product(job, rec, c, notes):
            return "%s: not made, the person pressed Stop." % head
        for box in blur:
            try:
                blur_box(job, BATCH_WHY, box)
            except Refused as e:
                notes.append("the box %s was not blurred: %s" % (box, e))
        if others:
            decide_people(job, BATCH_WHY, keep=others)
            notes.append("kept %d other face%s as %s" % (len(others), "" if len(others) == 1 else "s",
                                                          "they are" if len(others) > 1 else "it is"))
        look = channels.in_use() or {}
        try:
            if _captions_off():
                notes.append("no captions, as the person's saved look keeps them off")
            else:
                add_captions(job, BATCH_WHY, lit=look.get("captions") == "lit", pop=look.get("captions") == "pop")
        except Refused as e:
            notes.append("no captions: %s" % e)
        words, why_not = c.get("words") or c["title"], ""
        for place in dict.fromkeys(([look["title_place"]] if look.get("title_place") else []) + ["top", "middle"]):
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
    except Exception as e:
        traceback.print_exc()
        return "%s: broke (%s), a fault in the studio: make this clip by hand with make_clip." % (head, e)
    due = [ch for ch in checks if ch["ok"] is not None]
    return "%s: handed over, %d of %d checks passed%s." % (head, len(due), len(due),
                                                           "; " + "; ".join(notes) if notes else "")


def make_countdown(job, why, moments, story, final=None):
    rec, proposed, looks = _counted_from(job)
    n = countdown.asked([e["text"] for e in job.chat.events if e["kind"] == "person"])
    recipe = (job.chat.clip or {}).get("recipe")
    reel = n is None and recipe == "sport"
    if reel:
        n = 0
    if n is None:
        raise Refused("Nobody asked for a countdown in their own words (a top 5, 5 อันดับ, นับถอยหลัง), so make the clip "
                      "from the best moment with make_clip instead.")
    if recipe not in ("game", "sport"):
        raise Refused("A countdown is made from a game stream for now: make the clip from the best moment with "
                      "make_clip, or load the game recipe when this is one.")
    if 0 < n < countdown.FEWEST:
        raise Refused("A top %d is no countdown: make the clip from the best moment with make_clip." % n)
    want = min(n, countdown.MOST) if n else None
    if not proposed:
        raise Refused("No moments are proposed yet: look at the recording and propose the %s best, then count them "
                      "down." % (want or "%d to %d" % (countdown.FEWEST, countdown.MOST)))
    problems = []
    if want and len(moments) != want:
        problems.append("The person asked for a top %d%s: give %d moments, the best last." % (
            n, ", and a countdown holds at most %d" % countdown.MOST if n > countdown.MOST else "", want))
    numbers = [c["moment"] for c in moments]
    twice = sorted({k for k in numbers if numbers.count(k) > 1})
    if twice:
        problems.append("Moment%s %s named twice: each moment goes in once." % (
            "s" if len(twice) > 1 else "", ", ".join(map(str, twice))))
    strange = sorted({k for k in numbers if not 1 <= k <= len(proposed)})
    if strange:
        problems.append("There is no moment %s: you proposed %d. Propose the others first." % (
            ", ".join(map(str, strange)), len(proposed)))
    if not reel and len(moments) < countdown.FEWEST:
        problems.append("A countdown holds at least %d moments." % countdown.FEWEST)
    if reel:
        moments = sorted(moments, key=lambda c: proposed[c["moment"] - 1]["start"] if 1 <= c["moment"] <= len(proposed)
                         else 0.0)
        if any(c.get("start") is None or c.get("end") is None for c in moments):
            problems.append("Give each piece its start, where the attack starts, and its end, when the celebration ends: "
                            "only you saw where the goal is in the moment.")
    clock, spans, fitted, heard_in = recording.clock, [], [], []
    chosen = [proposed[k - 1] for k in numbers if 1 <= k <= len(proposed)]
    for c in moments:
        if not 1 <= c["moment"] <= len(proposed):
            continue
        m = proposed[c["moment"] - 1]
        if c.get("start") is None and c.get("end") is None:
            look = looks[m["look"] - 1]
            lo = max([(look.get("heard") or [look["start"]])[0], m["start"] - countdown.BEFORE - countdown.AFTER] +
                     [p["end"] for p in chosen if p["end"] <= m["start"]])
            try:
                runs = _menu_runs(job, rec, lo, m["end"])
            except screen.Stopped:
                return "Not made: the person pressed Stop."
            got = countdown.piece(m, runs, lo)
            if got is None:
                problems.append("Moment %d: the game's menu shows at %s with under %g s of play before it: propose it "
                                "from the play that led there." % (c["moment"], ", ".join(clock(r[0]) for r in runs),
                                                                   countdown.SHORTEST))
                continue
            a, b = got
            if any(abs(b - r[0] - countdown.DEATH_SHOWN) < 0.01 for r in runs):
                fitted.append("moment %d's piece ends %g s after the game's menu shows at %s, so it shows the play "
                              "that led there" % (c["moment"], countdown.DEATH_SHOWN,
                                                  clock(b - countdown.DEATH_SHOWN)))
        else:
            a = _time("moment %d's start" % c["moment"], c["start"]) if c.get("start") else m["start"]
            b = _time("moment %d's end" % c["moment"], c["end"]) if c.get("end") else m["end"]
            if reel:
                saw = next((s for s in looks if s["start"] <= a and b <= s["end"]), None)
                if saw:
                    m = dict(m, look=saw["n"], start=min(a, m["start"]), end=max(b, m["end"]))
                else:
                    a, b = max(a, m["start"]), min(b, m["end"])
            wrong = []
            if round(a, 2) < m["start"] or round(b, 2) > m["end"]:
                wrong.append("runs outside the moment (%s-%s)" % (clock(m["start"]), clock(m["end"])))
            if not countdown.SHORTEST <= round(b - a, 2) <= countdown.LONGEST:
                wrong.append("is %g s long, and a piece is %g to %g s" % (round(b - a, 1), countdown.SHORTEST,
                                                                          countdown.LONGEST))
            if wrong:
                problems.append("Moment %d: its piece %s-%s %s; %s." % (
                    c["moment"], clock(a), clock(b), " and ".join(wrong), "give it a start and an end around the goal"
                    if reel else "leave start and end out for the %g s around its peak" % (
                        countdown.BEFORE + countdown.AFTER)))
                continue
        a, b, moved = _snapped(looks[m["look"] - 1], a, b)
        if moved:
            fitted.append("moment %d's piece %s" % (c["moment"], moved))
        spans.append([a, b])
        heard_in.append(looks[m["look"] - 1])
    total = sum(b - a for a, b in spans)
    if not SHORTEST <= round(total) <= LONGEST:
        problems.append("The pieces add up to %g s; a clip is %d to %d s: %s." % (
            round(total, 1), SHORTEST, LONGEST, "give shorter pieces or fewer moments" if total > LONGEST else
            "give longer pieces"))
    thai = _thai_for_english(job, [c["words"] for c in moments])
    if thai:
        problems.append("The person asked for the words on the clip in English, so a viewer who reads no Thai can "
                        "follow: write these in English: %s." % "; ".join('"%s"' % t for t in thai))
    if reel and not problems and not _told_once(job, "reel_board"):
        raise Refused(_read_the_board(job, rec, spans))
    off = _scores_off([c["words"] for c in moments]) if reel else None
    tries = _times_told(job, "reel_scores")
    if off and tries < SCORE_TRIES:
        job.emit("reel_scores")
        problems.append(SCORES_OFF % (off, tries + 1, SCORE_TRIES))
    if reel and not final and _times_told(job, "reel_board") and not _told_once(job, "reel_final"):
        problems.append(FINAL_ASKED)
    if problems:
        raise Refused("Nothing was joined. Fix all of these, then call it again:\n- " + "\n- ".join(problems))
    if off:
        moments = [dict(c, words=" ".join(REEL_SCORE.sub(" ", c["words"]).split()) or REEL_NO_WORDS) for c in moments]
    for score, a, b in _goals_left_out([c["words"] for c in moments], spans, final, rec["seconds"]) if reel else ():
        if not _told_once(job, "reel_goal_left_" + score):
            _board_pictures(job, rec, [a + (b - a) * (k + 1) / (ACROSS_PICTURES + 1) for k in range(ACROSS_PICTURES)])
            raise Refused(GOAL_LEFT_OUT % (score, clock(a), clock(b)))
    level = _level_off([c["words"] for c in moments]) if reel else []
    if level and not _told_once(job, "reel_level"):
        raise Refused(LEVEL_OFF % ("; ".join('"%s"' % line for line in level), REEL_NO_WORDS))
    moments = [dict(c, words="%s %s" % (REEL_NO_WORDS, REEL_SCORE.search(c["words"])[0])) if c["words"] in level else c
               for c in moments]
    menu = None if reel else _menu_start(job, rec, spans[0][0])
    if menu:
        raise Refused("Nothing was joined. " + menu)
    try:
        texts, spelling = _spell(job, [" ".join(c["words"].split()) for c in moments])
    except hearing.Stopped:
        return "Not made: the person pressed Stop."
    count = len(spans)
    lines = texts if reel else ["%d. %s" % (count - k, words) for k, words in enumerate(texts)]
    for c, line in zip(moments, lines):
        _fits_a_place(line, "the words over moment %d" % c["moment"], job.chat.clip["style"])
    picked = [proposed[c["moment"] - 1] for c in moments]
    at = countdown.starts(spans)
    said = []
    for heard, (a, b), t in zip(heard_in, spans, at):
        for s in heard.get("said") or []:
            x, y = max(s["from"], a), min(s["to"], b)
            if y > x:
                said.append(dict(s, **{"from": round(t + x - a, 2), "to": round(t + y - a, 2)}))
    peaks = [round(t + min(max(0.0, (m["t"] if m.get("t") is not None else (a + b) / 2) - a), b - a), 2)
             for m, (a, b), t in zip(picked, spans, at)]
    out = countdown.target(rec, spans)
    job.working("Joining %d moments into the %s" % (count, "clip" if reel else "countdown"))
    try:
        countdown.join(rec, spans, out, on_start=job.started_work)
    except countdown.Stopped:
        return "Not made: the person pressed Stop."
    except countdown.CountdownError as e:
        return "The countdown could not be joined: %s Tell the person." % e
    finally:
        job.done_working()
    try:
        joined = recording.probe(out)
    except recording.NotOpened as e:
        return "The joined countdown could not be read back: %s Tell the person." % e
    job.chat.counted = {"out": out, "rec": rec, "moments": proposed, "looks": looks}
    job.chat.open(joined, job=job.n)
    end = round(joined["seconds"], 2)
    titles = [{"piece": k + 1, "text": line, "start": t, "end": at[k + 1] if k + 1 < count else end}
              for k, (line, t) in enumerate(zip(lines, at))]
    look, moment = _countdown_moment(job, joined, said, peaks, story, titles)
    job.chat.looks.append(look)
    job.chat.moments = [moment]
    job.show(look=look, moments=[moment])
    head = ["Joined %d moments into one recording, %g s long, %s: %s" % (
        count, round(end, 1), "in match order" if reel else "counting down to the best", story)]
    for m, (a, b), t, line in zip(picked, spans, at, lines):
        head.append('%g s: "%s", moment %d, %s-%s of the recording' % (t, line, m["n"], clock(a), clock(b)))
    if n > countdown.MOST:
        head.append("The person asked for a top %d; a countdown holds at most %d moments, so this is a top %d: tell "
                    "the person." % (n, countdown.MOST, count))
    if fitted:
        head.append("The program fitted the pieces: %s." % "; ".join(fitted))
    if spelling:
        head.append(spelling)
    if off:
        head.append(SCORES_OUT)
    if level:
        head.append(LEVEL_OUT % (REEL_NO_WORDS, ", ".join(REEL_SCORE.search(line)[0] for line in level)))
    ahead = _ahead(lines + [final or ""]) if reel else None
    if ahead:
        head.append(ahead)
    try:
        made = make_clip(job, why, 1, "card")
    except Refused as e:
        made = "The clip is not made yet: %s" % e
    return "\n".join(head) + "\n\n" + made


def _menu_runs(job, rec, a, b):
    """The runs of the game's menu (countdown.menu_runs) on the screen from a to b, [] when it cannot be read. Stop
    raises screen.Stopped. Measured on the Sifu stream (10-29), about 35 s to read 36 s, a game running beside it."""
    job.working("Reading the screen from %s to %s for the game's death screen" % (recording.clock(a),
                                                                                 recording.clock(b)))
    try:
        return countdown.menu_runs(screen.readings(rec, a, b, on_start=job.started_work), MENU_LINES)
    except screen.ScreenError:
        return []
    finally:
        job.done_working()


def _counted_from(job):
    """(the recording, its proposed moments, its looks) a countdown is cut from: the open recording's or, when the open
    one is the countdown made last, those of the recording it was cut from, so it can be counted down again in another
    order or with other words."""
    kept = job.chat.counted
    if kept and job.chat.recording and job.chat.recording["path"] == kept["out"]:
        rec = kept["rec"]
        if not os.path.isfile(rec["path"]):
            raise Refused("%s is no longer at %s. Tell the person, and ask where it is now." % (
                rec["name"], rec["path"]))
        return rec, kept["moments"], kept["looks"]
    return _open(job, "count down"), job.chat.moments or [], job.chat.looks


def _countdown_moment(job, rec, said, peaks, story, lines):
    """The look and the one moment of a joined countdown: what is said in each piece, moved to the joined seconds, and
    a picture at each piece's peak; the moment shows the last, the best."""
    pictures = []
    for t in peaks:
        frame = recording.frame(rec, at=t)
        if frame:
            pictures.append({"t": t, "frame": frame})
    end = round(rec["seconds"], 2)
    look = {"n": len(job.chat.looks) + 1, "start": 0.0, "end": end, "pictures": pictures, "loudest": None,
            "said": said, "heard": [0.0, end]}
    best = pictures[-1] if pictures else {"t": end / 2, "frame": None}
    moment = {"n": 1, "look": look["n"], "start": 0.0, "end": end, "reason": story, "snapped": None,
              "frame": best["frame"], "t": best["t"], "said": _heard_said(look, 0.0, end), "lines": lines,
              "preview": previews.cut(job.chat, job.n, rec, 0.0, end)}
    return look, moment


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


# 072526-70cc on 11-01: the ask said ให้ทุกคนคงสีไว้, Glimmer left everyone out, and one man stood in colour among grey people,
# as in the take Tarm withdrew (193425-85fa). It is set, not refused: a refusal after a local SAM run would cost about 20 min.
COLOUR_SAID = re.compile(r"สี|colou?r", re.I)
ALL_IN_COLOUR = re.compile(r"(?<!ไม่ให้)(?<!ไม่ต้องให้)(?<!อย่าให้)ทุกคน[^\n.]{0,16}?สี(?!ขาว|ดำ|เทา)|(?:คง|เก็บ|มี)สี[^\n.]{0,8}?ทุกคน|"
                           r"(?<!not )\b(?:everyone|everybody)\b[^.!?\n]{0,30}(?<!no )(?<!without )\bcolou?r\b|"
                           r"(?<!no )(?<!without )\bcolou?r\b[^.!?\n]{0,30}(?<!not )\b(?:everyone|everybody)\b", re.I)


def _all_in_colour(job):
    """True when the person's newest words about colour say everyone keeps theirs. Reads their messages, not the screen."""
    for e in reversed(job.chat.events):
        if e["kind"] == "person" and COLOUR_SAID.search(e["text"]):
            return bool(ALL_IN_COLOUR.search(e["text"]))
    return False


def cut_out(job, why, look=None, start=None, end=None, everyone=False):
    """Finds the person through the clip with SAM 3.1 (person.cut_out), once for the clip's stretch of the recording,
    so titles can go behind them and the clip can take a cut-out look. The mask is kept: asking again with another
    look or time does not run SAM again."""
    rec = _open(job, "cut the person out of")
    everyone = everyone or (look == "grey" and _all_in_colour(job))
    state = _clip_state(job)
    clip, length = state["clip"], state["length"]
    a, b = min(p[0] for p in clip["pieces"]), max(p[1] for p in clip["pieces"])
    held = state["mask"]
    people = [p["box"] for p in clip["layout"].get("people") or []] if clip["layout"]["kind"] == "talk" else []
    people = people if len(people) > 1 else None
    said = []
    if held and held["from"] <= a + 0.01 and b - 0.01 <= held["to"] and (held.get("each") or not people):
        said.append("The person was already found for this clip, so SAM did not run again.")
    else:
        job.working("SAM 3.1 is finding the person: starting")
        try:
            mask = person.cut_out(rec, a, b, face=clip["layout"].get("face"), on_start=job.started_work,
                                  on_progress=lambda done, of: job.working(
                                      "SAM 3.1 is finding the person: picture %d of %d" % (done, of)), people=people,
                                  **_sam_goes(job))
        except ValueError as e:
            raise Refused(str(e))
        except person.Stopped:
            return "Not cut out: the person pressed Stop."
        except person.PersonError as e:
            return "The person could not be cut out: %s Tell the person." % e
        finally:
            job.done_working()
        state = job.edit(dict(mask, do="mask"))
        found = sum(1 for p in mask.get("each") or [] if p)
        said.append("SAM 3.1 found %s through the clip, %s to %s of the recording, %d pictures a second."
                    % ("%d of the %d people" % (found, len(people)) if people else "the person",
                       recording.clock(a), recording.clock(b), mask["fps"]))
    says = CUTOUT_LOOK_SAYS
    if any((state["mask"] or {}).get("each") or []):
        says = CUTOUT_ALL_SAYS if everyone and state["mask"].get("everyone") else CUTOUT_EVERYONE_SAYS if state[
            "mask"].get("everyone") and not clip.get("voices") else CUTOUT_SPEAKER_SAYS
    if look:
        s = 0.0 if start is None else length + start if start < 0 else start
        e = length if end is None else length + end if end < 0 else min(end, length)
        if not 0 <= s < e:
            raise Refused("start %g s and end %g s do not make a time inside the clip, which is %g s long."
                          % (s, e, length))
        s, e = round(s, 2), round(e, 2)
        job.edit({"do": "cutout", "look": look, "start": s, "end": e, "everyone": bool(everyone)})
        said.append("The %s look is on from %g to %g s of the %g s clip%s: %s." % (
            look, s, e, length, "" if e >= length - 0.05 else ", so the clip's last %g s are without it"
            % round(length - e, 2), says[look]))
    said.append(_person_said(state, render.style_of(state, job.chat.clip["style"] if job.chat.clip else "clean")))
    return "\n".join(said)


CUTOUT_LOOK_SAYS = {"grey": "the footage around the person is black and white, and the person keeps their colour",
                    "outline": "a white edge goes around the person"}
CUTOUT_SPEAKER_SAYS = {"grey": "the footage is black and white around whoever speaks, who keeps their colour while "
                               "they speak",
                       "outline": "a white edge goes around whoever speaks, while they speak"}
CUTOUT_EVERYONE_SAYS = {"grey": "only the room is black and white and everyone keeps their colour, as their voices do "
                                "not tell who speaks and colour on one person could land on the wrong one; say so if "
                                "the person asked for the speaker alone",
                        "outline": "a white edge goes around everyone, as their voices do not tell who speaks; say so "
                                   "if the person asked for the speaker alone"}
CUTOUT_ALL_SAYS = {"grey": "only the room is black and white and everyone keeps their colour, as the person asked",
                   "outline": "a white edge goes around everyone, as the person asked"}
BEHIND_SHARE = 0.05
HIDDEN_MOST = 1 / 3.0
PERSON_STEP = 5
PERSON_SCALE = 8
HAIR = 0.3
SAFE_ROWS = (layouts.SAFE[1], layouts.NAME_TOP)
LIKES_LEFT = 890
FRAME = {"width": effects.FRAME_W, "height": effects.FRAME_H}


def _person_frames(state, start, end):
    """Where the clip shows the person from clip second start to end, everyone in a talk, read from the mask the
    renderer draws with (cutout.people_mask, made once): [pictures, rows, columns] of True, PERSON_STEP pictures a
    second at 1/PERSON_SCALE of the frame's size."""
    clip = state["clip"]
    try:
        path = cutout.people_mask(state["mask"], clip["pieces"], clip["layout"], state["length"],
                                  state.get("pushes") or [])
    except (ValueError, RuntimeError) as e:
        raise Refused(str(e))
    w, h = effects.FRAME_W // PERSON_SCALE, effects.FRAME_H // PERSON_SCALE
    raw = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-ss", "%.3f" % start, "-i", path, "-t",
                          "%.3f" % max(end - start, 1.0 / PERSON_STEP), "-vf",
                          "fps=%d,scale=%d:%d:flags=area" % (PERSON_STEP, w, h), "-f", "rawvideo", "-pix_fmt", "gray",
                          "-"], capture_output=True, timeout=300).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[:n * w * h], np.uint8).reshape(n, h, w) > 127


def _cells(frames, box):
    """frames inside box ([x, y, w, h] on the frame): [pictures, rows, columns]."""
    k = PERSON_SCALE
    x0, y0 = max(0, int(box[0] // k)), max(0, int(box[1] // k))
    x1, y1 = int(math.ceil((box[0] + box[2]) / k)), int(math.ceil((box[1] + box[3]) / k))
    return frames[:, y0:y1, x0:x1]


def _hidden(frames, box):
    """How much of box's width the person hides in each picture, 0 to 1: the share of its columns they cover at
    least half of, where the letters there cannot be read."""
    inside = _cells(frames, box)
    if not inside.size:
        return np.zeros(len(frames))
    return (inside.mean(axis=1) >= 0.5).mean(axis=1)


def _crossing(frames, box, start, end):
    """The stretches of start-end (clip seconds, frames read from start) where the person covers at least
    BEHIND_SHARE of box ([x, y, w, h] on the frame): [(from, to)]."""
    inside = _cells(frames, box)
    out = []
    if not inside.size:
        return out
    for i, on in enumerate(inside.mean(axis=(1, 2)) >= BEHIND_SHARE):
        if not on:
            continue
        lo, hi = start + i / PERSON_STEP, min(end, start + (i + 1) / PERSON_STEP)
        if out and lo <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def _crosses(spans, start, end):
    """Whether the person crosses the words long enough to be seen in front of them: SHORTEST_WORDS, or the whole
    time when that is shorter."""
    return sum(b - a for a, b in spans) >= min(SHORTEST_WORDS, end - start) - 1e-6


def _when(spans, start, end):
    if sum(b - a for a, b in spans) >= end - start - 1.0 / PERSON_STEP:
        return "from %g to %g s" % (start, end)
    more = len(spans) - 4
    return "at %s%s" % (_spans(spans[:4]), " and %d more times" % more if more > 0 else "")


def _person_box(frames):
    """[x, y, w, h] on the frame around where the person is in at least half the pictures (anywhere they are, when
    they move too much for that), or None when no picture shows them."""
    if not len(frames):
        return None
    for seen in (frames.mean(axis=0) >= 0.5, frames.any(axis=0)):
        ys, xs = np.nonzero(seen)
        if len(ys):
            k = PERSON_SCALE
            return [int(xs.min()) * k, int(ys.min()) * k, (int(xs.max()) + 1 - int(xs.min())) * k,
                    (int(ys.max()) + 1 - int(ys.min())) * k]
    return None


def _behind_band(frames, lay, state, start, end, tall, wide=None):
    """A band for words behind the person, [x, y, w, h] on the frame: as tall as the layout's first place for words
    and as wide, or ending left of the like buttons when that is covered, with a line of words tall px high and wide
    px wide (the band's width when None) in its middle sitting on the person's head, so the top of the head cuts the
    lower HAIR of the letters and the words can still be read; inside the rows a phone shows. None when such words
    would not cross the person, would be hidden too much, or a phone or the captions would cover them."""
    who = _person_box(frames)
    if not who:
        return None
    x, _y, w, h = next(iter(lay["places"].values()), [65, 0, 950, 190])
    top = who[1] - (1 - HAIR) * tall
    y = round(min(max(top - (h - tall) / 2, SAFE_ROWS[0]), SAFE_ROWS[1] - h))
    for band in ([x, y, w, h], [x, y, LIKES_LEFT - x, h]):
        line = band[2] if wide is None else wide
        if line > band[2]:
            continue
        drawn = [band[0] + (band[2] - line) / 2, y + (h - tall) / 2, line, tall]
        if (_crosses(_crossing(frames, drawn, start, end), start, end)
                and (_hidden(frames, drawn) > HIDDEN_MOST).mean() <= 0.5
                and not _covers([round(v) for v in drawn], lay, state)):
            return band
    return None


def _where_said(who):
    return "%s on your grid" % _grid(who, FRAME)


def _person_said(state, style):
    """Where the clip shows the person, and which places for words cross them, for Muse to put a title behind them
    (style: the title's text style, for the height of one line)."""
    lay, length = state["clip"]["layout"], state["length"]
    try:
        frames = _person_frames(state, 0, length)
    except Refused as e:
        return ("A title put on with behind true goes behind the person; where they are on the frame could not be "
                "read: %s" % e)
    who = _person_box(frames)
    if not who:
        return ("No picture of the clip shows the person, so a title behind them would not go behind anyone: make "
                "the clip again in a layout that shows them, then call cut_out.")
    crossing = []
    for name, box in lay["places"].items():
        spans = _crossing(frames, box, 0, length)
        if _crosses(spans, 0, length):
            crossing.append("%s %s" % (name, _when(spans, 0, length)))
    band = _behind_band(frames, lay, state, 0, length, text.measure("Title", style, "title", effects.FRAME_W).h)
    return ("On the frame the person is at %s. A title goes behind them only where they cross it, and it must still "
            "be read, so the top of their head should cut only the bottom of the letters: %s%s." % (
                _where_said(who), "the places for words that cross them: %s" % "; ".join(crossing) if crossing else
                "no place for words crosses them, so give add_title a box", "" if not band else
                "; for one line of words on top of their head, a box such as %s" % _grid(band, FRAME)))


def _not_crossed(frames, drawn, place, start, end, spans, lay, state, size):
    """Why a title put on with behind true is refused: the person does not cross its words long enough."""
    who = _person_box(frames)
    if not who:
        return ("From %g to %g s no picture shows the person, so the title would go behind nobody: pick a time when "
                "the clip shows them, or put it on without behind." % (start, end))
    band = _behind_band(frames, lay, state, start, end, size.h, size.w)
    return ("Behind the person, the title must cross them for at least %g s, and at the %s they cross its words (%s "
            "on your grid) for %g s of %g to %g s; they are at %s. %s, or put it on without behind." % (
                min(SHORTEST_WORDS, end - start), place, layouts.on_grid(drawn[1], drawn[1] + drawn[3], "y",
                                                                         inward=False),
                round(sum(b - a for a, b in spans), 1), start, end, _where_said(who),
                "Give a box such as %s, on top of their head" % _grid(band, FRAME) if band else
                "Give a box across those rows"))


def _too_hidden(frames, place, start, end, hidden, words, lay, state, size):
    """Why a title put on with behind true is refused: the person hides so much of its words they cannot be read."""
    band = _behind_band(frames, lay, state, start, end, size.h, size.w)
    return ('Behind the person, the words must still be read, and at the %s the person hides more than a third of '
            'their width for %g s of %g to %g s (about %d%% most of the time), so a viewer cannot read "%s". %s, or '
            'put it on without behind.' % (
                place, round(float((hidden > HIDDEN_MOST).sum()) / PERSON_STEP, 1), start, end,
                round(100 * float(np.median(hidden))), words,
                "Put it higher, so the top of their head cuts only the bottom of the letters: a box such as %s"
                % _grid(band, FRAME) if band else "Put it where the person crosses only the bottom of the letters"))


def look_of(state, default, name, handle="", made=None):
    """A channel look made from the clip as it is now (state), for channels.save: its style entry or the house style
    default, its colour look, how its captions show (off when they are hidden with the eye), the place of its first
    title, its drawn name's spot, and for the kind of clip made is (the chat's recipe) its layout and its length,
    beside what the look kept for other kinds. Tarm 10-30: "ทำให้ user save template ได้ไปเลย เช่น Enable / disable
    caption"."""
    style, caps = state["style"], state["captions"]
    places = state["clip"]["layout"]["places"]
    before = channels.get(name) or {}
    kind = (made or {}).get("recipe")
    return {"name": name, "handle": handle,
            "style": {k: v for k, v in style.items() if k != "do"} if style else {"name": default, "font": None},
            "look": state["look"],
            "captions": ("off" if state["hidden"] else None if not caps else "lit" if caps.get("lit") else
                         "pop" if caps.get("pop") else "plain"),
            "title_place": next((t["place"] for t in state["titles"] if t["place"] in places), None),
            "spot": (state["channel"] or {}).get("spot"),
            "layouts": dict(before.get("layouts") or {}, **({kind: state["clip"]["layout"]["kind"]} if kind else {})),
            "lengths": dict(before.get("lengths") or {}, **({kind: made["length_s"]} if kind else {}))}


def _saved(kind):
    """(the layout, the length) the look in use keeps for the kind of clip, each None when it keeps none."""
    look = channels.in_use() or {}
    return (look.get("layouts") or {}).get(kind), (look.get("lengths") or {}).get(kind)


def _captions_off():
    """The look in use's name when it keeps the captions off, else None."""
    look = channels.in_use() or {}
    return look["name"] if look.get("captions") == "off" else None


def channel_entries(look, entries, every=False):
    """The project entries that put look on a clip whose entries so far are entries: its style, its colour look and
    its drawn name, each only where the clip has no entry of that kind yet, so what the person chose on this clip
    stays; with every, all of them (the person picked the look for this clip)."""
    has = set() if every else {e["do"] for e in entries}
    out = []
    if look.get("style") and "style" not in has:
        out.append(dict(look["style"], do="style"))
    if "look" not in has and (look.get("look") or every):
        out.append({"do": "look", "name": look.get("look")})
    if "channel" not in has and (look.get("handle") or every):
        out.append({"do": "channel", "name": look.get("handle") or "", "spot": look.get("spot")})
    return out


def channel_unfit(state, default):
    """The sentence why the clip's channel name (state["channel"]) cannot be drawn in its captions' letters at its
    spot (text.channel_box: letters the font lacks, or wider than the spot), or None when it can or there is none."""
    channel = state["channel"]
    if not channel:
        return None
    try:
        text.channel_box(channel["name"], render.style_of(state, default, "caption"), channel.get("spot"))
    except ValueError as e:
        return str(e)
    return None


def look_said(look, muse=True):
    """What a channel look puts on a clip: for Muse, with the call that puts its captions on; else for the person."""
    parts = [describe(dict(look["style"], do="style"))] if look.get("style") else []
    parts += ["the %s colour look" % look["look"]] if look.get("look") else []
    hint = "" if not muse else " (add_captions%s)" % (" with %s true" % look["captions"]
                                                      if look.get("captions") != "plain" else "")
    parts += (["no captions"] if look.get("captions") == "off" else
              ["%s captions%s" % (look["captions"], hint)] if look.get("captions") else [])
    parts += ["titles at the %s" % look["title_place"]] if look.get("title_place") else []
    parts += ["the name %s drawn on the clip" % look["handle"]] if look.get("handle") else []
    return ", ".join(parts) or "nothing yet"


ALBUM_PARTS = {"tagline": "Tagline", "circle": "Face circle", "captions": "Captions", "words": "Style of the words",
               "colour": "Colour"}
ALBUM_TOOLS = ("match_album", "undo_album", "render_album", "set_tagline")


def _circle_box(state):
    return next((p["dst"] for p in state["clip"]["layout"]["parts"] if p["name"] == "circle"), None)


def _captions_way(state):
    caps = state["captions"]
    return ("off" if state["hidden"] or not caps else "lit" if caps.get("lit") else "pop" if caps.get("pop")
            else "plain")


def _font_gaps(style, letters):
    """The fonts of a style entry that cannot write letters."""
    fonts = [f for f in [style.get("font")] + list((style.get("fonts") or {}).values()) if f]
    every = text.fonts(letters or None) if fonts else []
    return [f for f in fonts if _font_entry(f, every) is None]


def album_entries(source, entries, parts, default):
    """(the entries that put parts of the clip whose state is source on the clip whose entries are entries, [what could
    not go on it, each a sentence]). Tarm 10-30: the tagline and where the face circle sits go across an album of one
    stream, the words of a headline stay each clip's own, so no part here touches a title's words, a clip's moment, or
    its effects. A part both clips already share adds nothing, so the clip's video stays current."""
    target = project.Project(entries).state()
    out, left = [], []
    if "tagline" in parts:
        mine, theirs = source["channel"] or {}, target["channel"] or {}
        if (mine.get("name", ""), mine.get("spot")) != (theirs.get("name", ""), theirs.get("spot")):
            out.append({"do": "channel", "name": mine.get("name", ""), "spot": mine.get("spot")})
    if "words" in parts:
        style = dict(source["style"]) if source["style"] else {"do": "style", "name": default, "font": None}
        gaps = _font_gaps(style, _letters_of(target))
        if gaps:
            left.append("%s cannot write every letter of its words" % gaps[0])
        elif style != (target["style"] or {"do": "style", "name": default, "font": None}):
            out.append(style)
    if "colour" in parts and source["look"] != target["look"]:
        out.append({"do": "look", "name": source["look"]})
    if "captions" in parts:
        way, now = _captions_way(source), _captions_way(target)
        if way == "off" and now != "off":
            out.append({"do": "eye", "hidden": True})
        elif way != "off":
            eye = [{"do": "eye", "hidden": False}] if target["hidden"] else []
            caps = project.Project(entries + eye).state()["captions"]
            if not caps:
                left.append("it has no captions yet")
            else:
                out += eye
                if (way == "lit", way == "pop") != (bool(caps.get("lit")), bool(caps.get("pop"))):
                    out.append(dict(caps, lit=way == "lit", pop=way == "pop"))
    box = _circle_box(source) if "circle" in parts else None
    if box and _circle_box(target) != box:
        try:
            out.append({"do": "layout", "part": "circle", "box": list(box),
                        "layout": layouts.moved(target["clip"]["layout"], "circle", box)})
        except ValueError:
            left.append("its layout has no face circle")
    return out, left


def _channel_on(job):
    """Puts the channel look in use on the clip just made, each part the clip has no entry of yet: the sentence for
    Muse, or None when no look is in use."""
    look = channels.in_use()
    entries = channel_entries(look, job.chat.project.entries) if look else []
    unfit = job.chat.unfit(*entries)
    name = next((e["name"] for e in entries if e["do"] == "channel"), (job.chat.project.state()["channel"] or {}).get(
        "name"))
    if unfit:
        entries = [{"do": "channel", "name": "", "spot": None}] + [e for e in entries if e["do"] != "channel"]
    for entry in entries:
        job.edit(entry)
    said = ['The clip is in the "%s" channel look the person saved: %s. Keep to it unless the person asks for '
            'something else; what they changed on this clip themselves stays.' % (look["name"], look_said(look))
            ] if look else []
    said += ["The name %s is not drawn on this clip: %s" % (name, unfit)] if unfit else []
    return " ".join(said) or None


def save_channel_look(job, why, channel, handle=""):
    state = _clip_state(job)
    drawn = " ".join((handle or "").split())
    unfit = drawn and job.chat.unfit({"do": "channel", "name": drawn, "spot": (state["channel"] or {}).get("spot")})
    if unfit:
        raise Refused("Nothing was saved: %s" % unfit)
    try:
        look = channels.save(look_of(state, job.chat.clip["style"] if job.chat.clip else "clean", channel, handle,
                                     job.chat.clip))
        channels.use(look["name"])
    except ValueError as e:
        raise Refused(str(e))
    if look["handle"] and (state["channel"] or {}).get("name") != look["handle"]:
        job.edit({"do": "channel", "name": look["handle"], "spot": look["spot"]})
    job.show(clip=job.chat.clip_view())
    return ('Saved the "%s" channel look: %s. Every new clip starts in it, in this chat and the next ones, until the '
            'person picks another look or none on the page.' % (look["name"], look_said(look)))


def use_channel_look(job, why, channel):
    _clip_state(job)
    look = channels.get(channel) if channel else None
    unfit = look and job.chat.unfit(*channel_entries(look, [], every=True))
    if unfit:
        raise Refused('The "%s" look was not put on: %s' % (channel, unfit))
    try:
        channels.use(channel or None)
    except ValueError as e:
        raise Refused(str(e))
    if not channel:
        if _clip_state(job)["channel"]:
            job.edit({"do": "channel", "name": "", "spot": None})
        return "No channel look is in use now: new clips start in the recipe's guesses."
    look = channels.get(channel)
    for entry in channel_entries(look, [], every=True):
        job.edit(entry)
    job.show(clip=job.chat.clip_view())
    return 'The clip and every new clip are in the "%s" channel look: %s.' % (channel, look_said(look))


def describe(entry):
    if entry["do"] == "eye":
        return "hiding the captions" if entry["hidden"] else "showing the captions again"
    if entry["do"] == "mask":
        return "cutting the person off the green cloth" if entry.get("green") else "finding the person with SAM 3.1"
    if entry["do"] == "backdrop":
        return "the background of %s" % entry["about"] if entry.get("png") else "taking the background off"
    if entry["do"] == "cutout":
        return "the %s cut-out look at %g-%g s" % (entry["look"], entry["start"], entry["end"]) if entry["look"] \
            else "taking the cut-out look off at %g-%g s" % (entry["start"], entry["end"])
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
                "the captions") if entry["lines"] else "the caption drafts" if entry.get("drafts") else \
            "turning the captions off"
    if entry["do"] == "blur":
        return "the blur at %g-%g s" % (entry["start"], entry["end"])
    if entry["do"] == "hit":
        return "the %s at %g s" % (entry["kind"], entry["at"])
    if entry["do"] == "sticker":
        return "the sticker %sat %g s" % (entry["name"] + " " if entry.get("name") else "", entry["at"])
    if entry["do"] == "picture":
        return "the picture of %s at %g s" % (entry["about"], entry["at"])
    if entry["do"] == "glass":
        return "the %s glass box at %g-%g s" % (entry["tint"], entry["start"], entry["end"])
    if entry["do"] == "flurry":
        return "the %s flurry of %s at %g s" % (entry["move"], " and ".join(entry["names"]), entry["at"])
    if entry["do"] == "layout":
        return {"shots": "changing the shots", "product": "following the product"}.get(entry["part"]) or \
            "moving the %s" % entry["part"]
    if entry["do"] == "untitle":
        return 'taking the title "%s" off' % entry["text"]
    if entry["do"] == "effect":
        if entry.get("box"):
            return "moving the %s %s to another spot on the picture" % (entry["of"], entry["name"])
        return "%s the %s %s at %g s%s" % ("taking off" if entry.get("to") is None else "moving", entry["of"], entry["name"],
                                           entry["at"], "" if entry.get("to") is None else " to %g s" % entry["to"])
    if entry["do"] == "look":
        return "the %s look" % entry["name"] if entry["name"] else "taking the look off"
    if entry["do"] == "channel":
        return "the channel name %s on the clip" % entry["name"] if entry["name"] else \
            "taking the channel name off the clip"
    if entry["do"] == "push":
        return "the slow push at %g-%g s" % (entry["start"], entry["end"])
    if entry["do"] == "sound":
        return "the %s at %g s" % (entry["kind"], entry["at"])
    if entry["do"] == "music":
        return "the music %s" % os.path.basename(entry["path"]) if entry["path"] else "taking the music off"
    if entry["do"] == "mute":
        return "%s at %g-%g s" % ("bringing the sound back" if entry.get("back") else "the mute", entry["start"],
                                  entry["end"])
    if entry["do"] == "tracks":
        return "the choice of sound tracks"
    if entry["do"] == "people":
        return "the decision on people (%s)" % "; ".join(
            "%s person %s" % (k, _numbers(entry[k])) for k in ("blur", "keep") if entry[k])
    return 'the title "%s" at %g-%g s' % (entry["text"], entry["start"], entry["end"])


STYLES = {n: text.CARDS[n][0][0].lower() + text.CARDS[n][0][1:].rstrip(".") for n in text.HOUSE_NAMES}
LOOK_LABELS = {"3d": "3D"}
SHORTEST, LONGEST = 10, 90


# The recipe is picked by what the recording shows: on 10-30 Glimmer loaded one before opening anything, a seller's
# for EP 22, a cooking vlog, and made it a seller's clip, with no captions and no product.
BLIND = ("Open the recording first (open_recording, or open_clips for a folder) and look at its pictures: the recipe is "
         "picked by what it shows, not by the words of the ask.")


LENGTH_SAID = re.compile("(?:" + NUMBER + "|[.,]|ครึ่ง)+\\s*(?:วินาที|วิ(?!ด)|นาที|s\\b|secs?\\b|seconds?\\b|mins?\\b|"
                         "minutes?\\b)|\\b(?:an?|one|two|three|half an?)\\s+minutes?\\b|\\b(?:ten|fifteen|twenty|"
                         "thirty|forty|forty-five|fifty|sixty|ninety)\\s+seconds?\\b", re.I)


def _length_said(job):
    """True when the person named a length in this chat, in their own words or with the Shorter and Longer buttons
    ("about 20 s"). Lane 5, 10-30, over every chat's first load_recipe where the person named none: Glimmer passed
    length_s in 106 of 119 and 49 of them were shorter than the recipe's (vlog 30 for 40, talk 45 for 60), so the
    recipes' lengths never reached it, and EP 22's vlog ended 2 s before the line its talk built to."""
    return any(LENGTH_SAID.search(e["text"]) for e in job.chat.events if e["kind"] == "person")


def load_recipe(job, why, type, length_s=None, style=None, language="", brief="", subtitles=""):
    subtitles = _subtitles(subtitles)
    recipe = RECIPES.get(type.lower().strip())
    if recipe is None:
        raise Refused("There is no recipe for %s yet. The recipes are: %s. Do not use a recipe made for another "
                      "kind of video: tell the person a clip of this kind is not ready yet, and what the studio can "
                      "make now." % (type, _recipe_names()))
    if not (job.chat.recording or job.chat.clips):
        job.emit("recipe_blind")
        raise Refused(BLIND)
    if recipe["name"] == MONTAGE and not job.chat.clips:
        raise Refused("The %s recipe is for several clips opened together with open_clips; this chat has %s. Pick "
                      "the recipe for the kind of recording that is open." % (
                          MONTAGE, "one recording open" if job.chat.recording else "no clips open"))
    asked = None
    if recipe["name"] != MONTAGE and _joining(job):
        asked, recipe = recipe, RECIPES[MONTAGE]
    layout_saved, length_saved = _saved(recipe["name"])
    usual = length_saved or recipe["length"]
    guessed = length_s if length_s and length_s != usual and not _length_said(job) else None
    clip = {"recipe": recipe["name"], "title": recipe["title"],
            "length_s": usual if guessed else length_s or usual,
            "style": style or recipe["style"], "language": language, "brief": brief, "subtitles": subtitles}
    had = job.chat.clip
    job.set_clip(clip, chips(clip))
    lines = ["Recipe loaded: %s (%s)." % (recipe["title"], recipe["name"]),
             "Your guesses, shown to the person with buttons to change them: about %d s, style %s%s%s."
             % (clip["length_s"], clip["style"], ", language " + language if language else "",
                ", captions in " + subtitles if subtitles else ""),
             "The person's own words win over this recipe and over your guesses. Do not ask the person to confirm "
             "them: go on with the work. The buttons let them change any guess at any time."]
    if guessed:
        lines.append("The person named no length, so the clip is %s %d s, not the %d s you gave: a clip runs as long "
                     "as what it shows needs, and the Shorter and Longer buttons are theirs."
                     % ("their saved look's" if length_saved else "the recipe's", usual, guessed))
    keeps = (["the %s layout" % layout_saved] if layout_saved else []) + (
        ["about %d s" % length_saved] if length_saved else []) + (["no captions"] if _captions_off() else [])
    if keeps:
        lines.append("The person's saved look keeps this kind of clip to %s. Keep to it unless they ask for something "
                     "else." % ", ".join(keeps))
    if had and had["recipe"] != recipe["name"]:
        lines.append("This recipe replaces the %s recipe read earlier in this chat; follow only this one." %
                     had["recipe"])
    if asked:
        lines.append("Clips from a folder are open, so this is the %s recipe, not the %s one you asked for: the clips "
                     "are joined first, with arrange_clips. A %s told over several takes is joined the same way, by "
                     "its lines." % (MONTAGE, asked["name"], asked["title"][0].lower() + asked["title"][1:]))
    cut = _review_cut(job) if recipe["name"] == MONTAGE and _a_review(job, asked) else []
    return "\n".join(lines) + "\n\n" + recipe["body"] + ("\n\n" + "\n".join(cut) if cut else "")


# Spark wrote 13,869 tokens, its thinking with them, labelling the 170 lines of the yeti takes (10-28), and in the
# studio the same question at 16,384 came back empty.
REVIEW_TOKENS = 32000


REVIEW_WORDS = re.compile(r"รีวิว|แกะกล่อง|ป้ายยา|review|unbox", re.IGNORECASE)


def _a_review(job, asked):
    """True when the clips are a product review: Muse loaded the seller recipe, or the person's own words in this chat
    say review. Which recipe Muse loads for the same folder changes from run to run; the person's words do not."""
    return bool(asked and asked["name"] == "seller") or any(
        e["kind"] == "person" and REVIEW_WORDS.search(e.get("text") or "") for e in job.chat.events)


def _review_cut(job):
    """Sentences for Muse with the cut the program makes of a product review told over several takes (review.py):
    Muse is asked aside what each line does in a review, and the program puts the lines in the review order inside
    the length. [] when no take is talked through."""
    opened = job.chat.clips
    lines, clips = [], []
    for c in opened["clips"]:
        talking = _talking(c) and c["rec"]["seconds"] > LONGEST_PIECE
        clips.append({"n": c["n"], "talking": talking})
        for p, _ in _listed(c)[0] if talking else []:
            if not pauses.filler(p["text"]):
                lines.append({"n": len(lines) + 1, "clip": c["n"], "from": p["from"], "to": p["to"],
                              "text": p["text"]})
    if not lines:
        return []
    yourself = ("so arrange the takes yourself, in the review order: open on the line that shows it working, then "
                "what it is, the one thing it does well, what it works with, what there is to see, the verdict and "
                "the price, and the ask to comment.")
    messages, schema = review.question(lines)
    try:
        read = review.read(job.ask_aside(messages, schema, "Reading what each line of the review does",
                                         REVIEW_TOKENS), lines)
    except hearing.Stopped:
        return ["The person pressed Stop while the program read the lines, " + yourself]
    except (mind.MindError, ValueError, KeyError, TypeError, AttributeError) as e:
        return ["The program could not learn what each line does (%s), %s" % (str(e).rstrip(".") or "no answer",
                                                                                yourself)]
    cut = review.draft(clips, lines, read["jobs"], job.chat.clip["length_s"], read["hook"])
    if not cut["clips"] or not review.is_review(read["jobs"]):
        return ["No line said in the takes reads as part of a review (what it is, what it does, the verdict), " +
                yourself]
    opened["review"] = dict(read, cut=cut)
    said = {ln["n"]: ln["text"] for ln in lines}
    out = ["The program asked what each of the %d lines said in the takes does in a review, and made the cut in the "
           "review order (show it working, what it is, its best point, what it works with, what there is to see, "
           "the verdict and the price, the ask), %g s of lines:" % (len(lines), cut["length"])]
    out += ["%d. clip %d, %g-%g s, %s: %s" % (k, p["clip"], p["start"], p["end"], p["job"], " ".join(
        said[n] for n in p["lines"])) for k, p in enumerate(cut["clips"], 1)]
    out += ["Left out: clip %d, %s." % (c["clip"], c["why"]) for c in cut["left_out"]]
    out.append("To make it, call arrange_clips with clips left empty ([]) and hook: the words over its first piece, 2 "
               "to 6 words from what is said in it, such as the question the clip answers. The program puts the "
               "pieces and the left out clips in as listed. Give clips of your own only where the person asked for "
               "something else.")
    return out


def _joining(job):
    """True while a folder of clips is open and what is open is not another recording: no recording yet, or the one
    arrange_clips joined from them. Seen 10-28 in Tarm's yeti review, twice: Muse loaded the seller recipe for a
    review in eight takes, arrange_clips refused it, and one run gave up on the folder for a single take."""
    rec = job.chat.recording
    return bool(job.chat.clips) and (rec is None or os.path.dirname(os.path.abspath(rec["path"])) == os.path.abspath(
        os.path.join(recording.WORK, "montage")))


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


def _recipe_names():
    return ", ".join("%s (%s)" % (r["name"], r["title"]) for r in RECIPES.values())


def _recipe_list():
    """Each recipe with when to pick it: with its title alone, Glimmer took three recipes for one ask on EP 22 (11-01)."""
    return " ".join("%s: %s" % (r["name"], r["description"]) for r in RECIPES.values())


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
         "the end of every piece by the same share when together they are longer than asked. A clip someone talks "
         "through is cut by its lines instead: give each piece of it a start and an end around the lines you keep; "
         "the program cuts the takes said again or given up at a clap, and the silences. Call it again to change "
         "anything." % LONGEST_PIECE,
         {"story": {"type": "string", "maxLength": 300,
                    "description": "One line for the person, in their language: what the clip tells, from its first "
                                   "picture to its last."},
          "clips": {"type": "array", "minItems": 0, "maxItems": MOST_CLIPS,
                    "description": "The clips that go in, in the order they are shown. Leave it empty ([]) to use "
                                   "the cut the program made of a review, as it is.",
                    "items": {"type": "object", "properties": {
                        "clip": {"type": "integer", "minimum": 1, "maximum": MOST_CLIPS,
                                 "description": "The clip's number."},
                        "start": {"type": "number", "minimum": 0,
                                  "description": "The second of the clip its piece starts at. Leave it out for its "
                                                 "start, except in a clip someone talks through."},
                        "end": {"type": "number", "minimum": 0,
                                "description": "The second of the clip its piece ends at. Leave it out for its end."},
                        "words": {"type": "string", "maxLength": 80,
                                  "description": "A short line on screen from this piece until the next piece with "
                                                 "words, in the person's language. Leave it out where the picture "
                                                 "says it all."},
                        "focus": {"type": "array", "minItems": 4, "maxItems": 4,
                                  "items": {"type": "number", "minimum": 0, "maximum": 1000},
                                  "description": "[x1, y1, x2, y2] on your 0-1000 grid of this clip's picture: what "
                                                 "must stay in the frame when a lying picture is cropped upright to "
                                                 "fill it, such as the product and the hands showing it. Leave it out "
                                                 "to keep the face the program finds, except in a product review, "
                                                 "where the product leads."},
                        "whole": {"type": "boolean",
                                  "description": "true shows a lying picture whole over a blurred copy of itself, "
                                                 "small in the middle of the frame: only for a screen or a wide view "
                                                 "where every part matters. Leave it out to fill the frame."},
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
                                 "required": ["clip", "why"]}},
          "hook": {"type": "string", "maxLength": 80,
                   "description": "The words over the first piece, in the person's language: with clips left empty, "
                                  "over the first piece of the program's cut. 2 to 6 words that make someone scrolling "
                                  "past stay."},
          "products": {"type": "array", "maxItems": 2 * MOST_CLIPS,
                       "description": "In a product review, the product in each piece that lies on its side: that "
                                      "piece fills the upright frame around it. Give a piece's box on the picture of "
                                      "that piece the program shows; a clip's box is for every piece of it without "
                                      "one. Needed for the program's cut, whose pieces you do not list.",
                       "items": {"type": "object", "properties": {
                           "piece": {"type": "integer", "minimum": 1,
                                     "description": "The piece's number, as the program's pictures of the pieces "
                                                    "give it."},
                           "clip": {"type": "integer", "minimum": 1, "maximum": MOST_CLIPS,
                                    "description": "The clip's number, for a box on that clip's strip."},
                           "box": {"type": "array", "minItems": 4, "maxItems": 4,
                                   "items": {"type": "number", "minimum": 0, "maximum": 1000},
                                   "description": "[x1, y1, x2, y2] on your 0-1000 grid of that piece's picture (or "
                                                  "the clip's): the product and the hands showing it."}},
                                 "required": ["box"]}}},
         ["story", "clips"], arrange_clips, changes=True),
    Tool("load_recipe", "Picked the kind of clip",
         "Read the recipe for this kind of video before you plan the clip, and give your guesses for it. The page "
         "shows your guesses with buttons, so the person can change any of them with one tap. Call it again when "
         "the person says it is another kind, or the pictures you looked at show it is, then plan again. The kinds "
         "with a recipe now: " + _recipe_list() + " If the video is another kind, still name that kind.",
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
    Tool("watch_piece", "Watched it with Muse Wonder",
         "Have Muse Wonder watch a stretch of up to %d s with its sound, and point out what the words alone miss: a "
         "scream, a laugh, a fail, a surprise, a silent stare, each with its second and the effect that would land "
         "there. Call it on the clip's own stretch before you put effects on a game, a reaction or any clip whose fun "
         "is in the picture and the sound. It sends that stretch, small, to Meta with the person's own key, and costs "
         "a few cents. When you work on this computer it sends only what the person asked for in their own words, "
         "unless they turned on Meta sees pieces in Options." % watch.LONGEST,
         {"start": {"type": "string", "maxLength": 20, "description": "Where the stretch starts, like 1:51:23."},
          "end": {"type": "string", "maxLength": 20, "description": "Where it ends, written the same way."},
          "person_said": {"type": "string", "maxLength": 200,
                          "description": "Only when the person asked for the clip to be watched with its sound, or "
                                         "for Muse Wonder: their own words from the chat, quoted. The program checks "
                                         "they said them."}},
         ["start", "end"], watch_piece),
    Tool("propose_moments", "Proposed the moments",
         "Propose the best moments for the clip, best first, each from a look you made. The person sees each as a "
         "card with your reason, a picture from your look and a rough video of the stretch. Then go on with the "
         "best one yourself; do not ask the person to choose, unless they asked to see the moments first.",
         {"then": {"type": "string", "enum": ["make", "wait"],
                   "description": "wait when the person asked only to find or see the moments and not for a clip yet "
                                  "(\"just find the highlights\", \"don't make anything yet\", \"ยังไม่ต้องทำ บอกช่วงที่"
                                  "น่าสนใจ\"): the work stops at the cards. make (the default) when they asked for a "
                                  "clip, or gave no words."},
          "moments": {"type": "array", "minItems": 1, "maxItems": MOST_MOMENTS,
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
         {"moment": {"type": "integer", "minimum": 1,
                     "description": "The number of the moment you proposed, or of the one change_ends made."},
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
    Tool("change_ends", "Moved the clip's ends",
         "Move where the clip starts or ends in the recording when the person asks: cut off its start or its end, or "
         "take in more before or after it ('start earlier', 'cut the last bit', 'end after the laugh'). The program "
         "makes the clip again from the same moment in the same layout, crop and face, with its edges where you put "
         "them, or in the nearest quiet sound when that is inside a word; the captions, blurs, sounds and effects stay "
         "on what is said and shown, and the words on screen "
         "stay at their time in the clip. The person can do the same from the Trim tab under the clip.",
         {"start": {"type": "string", "maxLength": 20,
                    "description": "The new start, a time of the recording like 1:51:30 or 6704.5. Leave it out to keep "
                                   "the start."},
          "end": {"type": "string", "maxLength": 20,
                  "description": "The new end, a time of the recording. Leave it out to keep the end."}},
         [], change_ends, changes=True),
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
         "whether each is free to use in a video someone posts. Call it whenever the person asks which fonts there "
         "are or which can write their words, and when they want another font. The list depends on this computer, so "
         "never answer it from memory.",
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
                  "description": "One word of the title to draw in the style's key colour. Leave it out for none."},
          "behind": {"type": "boolean",
                     "description": "true puts the title behind the person, or everyone the clip shows in a talk, "
                                    "so their head and body cross in front of "
                                    "the words, as on a magazine cover. It needs cut_out first, whose result says "
                                    "where the person is on the frame: the words must cross them there. Then a title "
                                    "may cross a face: the face stays in front."}},
         ["words", "start", "end"], add_title, changes=True),
    Tool("cut_out", "Found the person",
         "Find the person through the whole clip with SAM 3.1, Meta's model that follows a person frame by frame, "
         "so a title can go behind them (add_title with behind true) and the clip can take a cut-out look. It takes "
         "about a minute: Muse steps off the graphics card while SAM works, and comes back after. The person is "
         "found once; calling it again only changes the look. Use it when the person asks for words behind them or "
         "a cut-out look, or when a title behind the person makes a talk clip stand out.",
         {"look": {"type": "string", "enum": list(render.CUTOUT_LOOKS),
                   "description": "grey: the footage around the person goes black and white, and the person keeps "
                                  "their colour. outline: a white edge around the person. In a talk both follow "
                                  "whoever speaks when their voices tell who that is, and else take in everyone. "
                                  "Leave it out to only find the person, for a title behind them."},
          "everyone": {"type": "boolean", "description": "In a talk, true when the person said everyone keeps their "
                                                         "colour or the look goes on everyone: the look then does not "
                                                         "follow whoever speaks."},
          "start": {"type": "number", "description": "When the look starts, in seconds of the clip; a negative "
                                                     "number counts back from the clip's end, so -5 is its last 5 "
                                                     "seconds. Leave it out for the start."},
          "end": {"type": "number", "description": "When the look ends, in seconds of the clip; a negative number "
                                                   "counts back from the clip's end. Leave it out for the end."}},
         [], cut_out, changes=True),
    Tool("save_channel_look", "Saved the channel look",
         "Save the clip's look as the person's channel look: the words' style, the colour look, how the captions "
         "show, where the title goes, and the channel's name drawn on the clip. Every new clip, in this chat and the "
         "next ones, starts in it. Use it only when the person asks to keep this look for their channel, and call it "
         "before you say it is saved: nothing is kept without this call.",
         {"channel": {"type": "string", "maxLength": channels.LONGEST_NAME,
                      "description": "The channel's name, as the person says it, such as Sample Channel."},
          "handle": {"type": "string", "maxLength": channels.LONGEST_HANDLE,
                     "description": "The name to draw small on every clip, such as @yourchannel, when the person "
                                    "gives one. Leave it out for none."}},
         ["channel"], save_channel_look, changes=True),
    Tool("use_channel_look", "Chose the channel look",
         "Put one of the person's saved channel looks on the clip and on every new clip, or none. Use it when the "
         "person names a channel whose look they saved.",
         {"channel": {"type": "string", "maxLength": channels.LONGEST_NAME,
                      "description": "The saved look's name; an empty name means no channel look."}},
         ["channel"], use_channel_look, changes=True),
    Tool("remove_title", "Took a title off",
         "Take a title off the clip, by its words.",
         {"words": {"type": "string", "maxLength": 80, "description": "The title's words, as on the clip."}},
         ["words"], remove_title, changes=True),
    Tool("add_captions", "Put the captions on",
         "Put captions of what was said in the moment on the clip: the words as heard and fixed from the context, "
         "numbers exactly as said, a few words a line. The program places them in the layout's caption band, and "
         "asks you apart for the one word in each line a viewer must not miss, which it draws in the key colour. "
         "When the clip is talked through, they come one big line at a time and each word pops in as it is said; "
         "otherwise they are plain words unless you set lit or pop. Set on to false to take them off.",
         {"on": {"type": "boolean", "description": "true to put captions on (the default), false to take them off."},
          "lit": {"type": "boolean",
                  "description": "true lights up each word as it is said, like karaoke. Use it only when nobody on "
                                 "screen is talking: a voice-over or the words of a song over the picture. When a "
                                 "person on camera explains something, the lit word pulls the eye away from them, "
                                 "so leave it out, unless the person asked for lit words."},
          "pop": {"type": "boolean",
                  "description": "true pops each word in at its own place as it is said, and the line stays once "
                                 "said, so nothing is lit and the eye can go back to the person. Leave it out and "
                                 "the words pop in when the clip is talked through; set it true when the person asks "
                                 "for words that pop or appear as they are said, false when they ask for plain "
                                 "lines. Not with lit."},
          "language": {"type": "string", "maxLength": 40,
                       "description": "The language to write the captions in, any language, such as English or "
                                      "Spanish, when the person wants subtitles in another language than the one "
                                      "spoken, or wrote to you in another language than the one spoken: the program "
                                      "has each sentence translated and keeps its numbers. Leave it out to keep what "
                                      "the recipe was told."},
          "hidden": {"type": "boolean",
                     "description": "true hides every caption and keeps it, the lines the person typed too, as the eye "
                                    "on the clip does; false shows them again as they were. Use it when the person "
                                    "asks to hide or show the captions; on false takes them off for good."}},
         [], add_captions, changes=True),
    Tool("no_chat_box", "Said there is no chat box",
         "Only when your looks show no chat box in a game recording, such as a console game with no stream: say so, and "
         "no blur is asked for. Never blur the whole picture instead.",
         {}, [], no_chat_box, changes=True),
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
         "Decide who of the other people make_clip listed is blurred and who is kept. A blurred face is followed "
         "where the finder saw it. Blur a real person who is not the one making the clip, such as someone passing behind; keep a "
         "face that belongs, such as a game character, or a face on a screen or a box. At a trip or an event every "
         "face starts kept: blur one only when the person asks. Every person seen in two pictures or more must be "
         "decided before the clip is rendered; one call can decide them all.",
         {"blur": {"type": "array", "items": {"type": "integer", "minimum": 1},
                   "description": "The numbers of the people to blur, such as [2]."},
          "keep": {"type": "array", "items": {"type": "integer", "minimum": 1},
                   "description": "The numbers of the people to keep as they are."},
          "others": {"type": "string", "enum": ["blur", "keep"],
                     "description": "What happens to everyone not named in blur or keep, such as a crowd."},
          "person_said": {"type": "string", "maxLength": 200,
                          "description": "Only when the person asked for this blur: their own words from the chat, "
                                         "quoted."}},
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
         "In a talk clip, show one person from start to end: when the program cut to the wrong person, or when a "
         "reaction should be seen. Show everyone only when the person asks for it. The shots around it are cut to make "
         "room. With centre, the shot's frame moves too, when the person asks to see more of something beside the "
         "face; it keeps its size.",
         {"start": {"type": "number", "minimum": 0, "description": "From when, in seconds of the clip."},
          "end": {"type": "number", "minimum": 0, "description": "Until when, in seconds of the clip."},
          "show": {"type": "integer", "minimum": 0, "maximum": talk.MOST_PEOPLE,
                   "description": "The person to show, numbered from the left as make_clip listed them, or 0 for "
                                  "everyone together, when the person asks."},
          "centre": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": 0,
                                                                              "maximum": 1000},
                     "description": "[x, y] on your 0-1000 grid of the recording: the point the shot's frame sits "
                                    "around. Leave it out and the shot keeps the place it has."}},
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
    Tool("set_sound_tracks", "Chose the sound tracks",
         "Choose which of the recording's sound tracks are in the clip, by their numbers from 1. A recording can hold "
         "several, such as a game on one and a voice on another. All of them are in the clip until you choose. Use it "
         "when the person asks for one track, or to leave one out, such as no game sound.",
         {"tracks": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 1},
                     "description": "The numbers of the tracks to keep, from 1."}},
         ["tracks"], set_sound_tracks, changes=True),
    Tool("mute_sound", "Muted the sound",
         "Mute the clip's own sound from one second of the clip to another: a count-in such as 1 2 3, someone else "
         "talking, a noise. The picture plays on; the music and the studio's sounds still play there, and the "
         "captions leave out the words said there. back true brings the sound back over that time. The mute stays on "
         "what was said there when the clip is cut again. Use it when the person asks to cut, mute or take out the "
         "sound of a part, or when what is heard there is not part of the clip, such as a voice counting in before "
         "the speaker starts.",
         {"start": {"type": "number", "minimum": 0, "description": "Where the mute starts, in seconds of the clip."},
          "end": {"type": "number", "minimum": 0, "description": "Where it ends, in seconds of the clip."},
          "back": {"type": "boolean", "description": "true brings the sound back from start to end. Leave it out "
                                                     "to mute."}},
         ["start", "end"], mute_sound, changes=True),
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
    Tool("change_effect", "Changed an effect",
         "Take one effect off the clip, or move it to another second: a hit, sticker, flurry, sound, picture or slow "
         "push, by its name and second as the clip's list says them. Use it when the person asks for one effect to go "
         "or to come earlier or later. It keeps everything else about the effect. With box, a sticker, flurry, picture "
         "or title goes to that spot on the picture, as the person asked, and nothing else about it changes.",
         {"what": {"type": "string", "enum": list(EFFECTS), "description": "Which kind of effect."},
          "called": {"type": "string", "maxLength": 300,
                     "description": "A hit's or sound's kind, a sticker's name, a flurry's stickers joined by +, a "
                                    "picture's words, a title's words, or push."},
          "at": {"type": "number", "minimum": 0, "description": "The second it is at now, in seconds of the clip."},
          "to": {"type": "number", "minimum": 0,
                 "description": "The second to move it to. Leave it out to take the effect off."},
          "box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0},
                  "description": "[x1, y1, x2, y2] on the 0-1000 grid of the clip's frame: the new spot on the picture."},
          "person_said": {"type": "string", "maxLength": 200,
                          "description": "With box: the person's own words asking for that spot, quoted exactly."}},
         ["what", "called", "at"], change_effect, changes=True),
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
    Tool("add_picture", "Drew a picture",
         "Have Muse Image draw a picture of something the clip names, and show it over the top of the frame, above "
         "the people, its lower edge fading into the footage, under the titles and captions: a ghost while they "
         "tell a ghost story, a crowd on phones while they talk about millions of viewers. A new picture each time "
         "a new thing is named holds the eye through a talk; 1 to 3 in a clip, each on the second its thing is "
         "said. Only your words go to Meta, never the footage. It costs $%g a picture; the same words again cost "
         "nothing. A picture at an overlapping time replaces the earlier one." % picture.USD,
         {"about": {"type": "string", "minLength": 3, "maxLength": 300,
                    "description": "What the picture shows, in English, as a scene: the thing named and where it is, "
                                   "such as \"a pale ghost peeking from a dark doorway of an old wooden house at "
                                   "night\". No real or famous person, no brand, no writing."},
          "at": {"type": "number", "minimum": 0, "description": "When it comes in, in seconds of the clip: when its "
                                                                 "thing is said."},
          "length": {"type": "number", "minimum": 0.6, "maximum": 10,
                     "description": "How long it stays, in seconds. Leave it out for %g." % PICTURE_LENGTH}},
         ["about", "at"], add_picture, changes=True),
    Tool("put_background", "Put a background behind the person",
         "Have Muse Image draw a new background and put it behind the person wherever the clip does not show them, "
         "through the whole clip: for a person in front of a green cloth (cut off it by its colour on this computer, "
         "no GPU, in seconds), or after cut_out has found the person with SAM. The layout stays as it is, so the "
         "product stays as big as you framed it. Only your words go to Meta, never the footage. It costs $%g; the "
         "same words again cost nothing. Make the clip first; after changing its ends, put it on again."
         % picture.USD,
         {"about": {"type": "string", "minLength": 3, "maxLength": 300,
                    "description": "The place behind the person, in English, suited to what the clip sells or is "
                                   "about, such as \"%s\" or \"%s\". No people, no brand, no writing."
                                   % BACKDROP_EXAMPLES}},
         ["about"], put_background, changes=True),
    Tool("follow_product", "Follow the product",
         "Have SAM 3.1 find the product through the clip and move the vertical frame with it, so a product held out "
         "to the side or moved around stays in the picture: for a seller in a landscape recording, made in the "
         "subject layout. The frame holds where the product is not seen. It runs on this computer's graphics card, "
         "about a second for every few seconds of the clip.",
         {"what": {"type": "string", "minLength": 3, "maxLength": 80,
                   "description": "The product as it looks in the picture, in a few English words SAM can find, "
                                  "such as \"white game controller\" or \"yellow balm jar\". No brand."}},
         ["what"], follow_product, changes=True),
    Tool("add_flurry", "Send stickers flying",
         "Send many copies of the studio's stickers moving together for a moment: money bursting out on a payout, "
         "coins raining on a win, hearts rising on a thank-you, confetti at the end. The moves: " +
         "; ".join("%s: %s" % (name, m[0].rstrip(".")) for name, m in effects.MOVES.items()) + ". Every copy stays "
         "inside the box, so give a big one clear of faces (in a small one the pictures shrink to fit). One in a "
         "clip, on the peak. A flurry on the same spot at an overlapping time replaces the earlier one.",
         {"stickers": {"type": "array", "minItems": 1, "maxItems": effects.FLURRY_PICTURES,
                       "items": {"type": "string", "enum": [s["name"] for s in effects.stickers()] + [effects.CONFETTI]},
                       "description": "1 to %d of the studio's stickers by name, or %s for paper pieces in six "
                                      "colours; the copies take them in turn." % (effects.FLURRY_PICTURES,
                                                                                  effects.CONFETTI)},
          "at": {"type": "number", "minimum": 0, "description": "When it starts, in seconds of the clip."},
          "move": {"type": "string", "enum": list(effects.MOVES), "description": "How they move. Leave it out for "
                                                                                 "burst."},
          "count": {"type": "integer", "minimum": effects.FLURRY_COUNTS[0], "maximum": effects.FLURRY_COUNTS[1],
                    "description": "How many copies. Leave it out for 8; 12 to 20 for rain or confetti."},
          "length": {"type": "number", "minimum": effects.FLURRY_LENGTHS[0], "maximum": effects.FLURRY_LENGTHS[1],
                     "description": "How long the move takes, in seconds. Leave it out for the move's own."},
          "place": {"type": "string", "maxLength": 20,
                    "description": "One of the layout's places for words, as make_clip listed them. Leave it out "
                                   "when you give a box."},
          "box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0,
                                                                              "maximum": 1000},
                  "description": "[x1, y1, x2, y2] on a 0-1000 grid of the clip's vertical frame: where they move."},
          "person_said": {"type": "string", "maxLength": 200,
                          "description": "Only when the person asked for this exact spot: their own words from the "
                                         "chat, quoted. Then it goes there even over a face or where a phone covers "
                                         "it, and the result says so."}},
         ["stickers", "at"], add_flurry, changes=True),
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
    Tool("make_cover", "Made the cover",
         "Have Muse Image draw the clip's cover, the picture the post shows before it plays: the finished clip's "
         "frame at its peak, without its words, with your words written on it, the people and the product kept as "
         "shot. It costs $%.2f on Meta's API with the person's key; the same words again cost nothing. Run it after "
         "check_clip, before hand_over." % cover.USD,
         {"title": {"type": "string", "maxLength": 60,
                    "description": "The words on the cover, a few words in the clip's language for a stranger "
                                   "scrolling past: what they get or what bothers them, no jargon."},
          "again": {"type": "boolean", "description": "true draws a new cover for the same words, when the person "
                                                      "asks or the last one has a word misspelled or cut off."},
          "asked": {"type": "boolean", "description": "true only when the person asked for a cover in their own "
                                                      "words or tapped Make the cover."}},
         ["title"], make_cover),
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
         "hand-over. Names must be spelled as the screen shows them. A seller's clips each need their product, and "
         "their background on a green cloth: the program then follows the product and puts the background on.",
         {"clips": {"type": "array", "minItems": 1, "maxItems": MOST_AT_ONCE, "items": {
             "type": "object", "required": ["moment", "title", "caption", "hashtags"], "properties": {
                 "moment": {"type": "integer", "minimum": 1, "maximum": MOST_MOMENTS,
                            "description": "The moment's number, as you proposed it."},
                 "layout": {"type": "string", "enum": list(layouts.KINDS) + ["talk", "speaker", "split"],
                            "description": "Leave it out for the recipe's layout."},
                 "product": {"type": "array", "minItems": 4, "maxItems": 4, "items": {
                     "type": "number", "minimum": 0, "maximum": 1000},
                             "description": "A seller's clip must give this: [x1, y1, x2, y2] on your 0-1000 grid "
                                            "around this clip's product in the picture."},
                 "what": {"type": "string", "minLength": 3, "maxLength": 80,
                          "description": "A seller's clip from a landscape recording must give this: the product as it "
                                         "looks in the picture, in a few English words SAM can find, no brand. The "
                                         "frame follows it."},
                 "background": {"type": "string", "minLength": 3, "maxLength": 300,
                                "description": "A seller's clip with a green cloth behind the person must give this: "
                                               "the place behind them in English, suited to this clip's product. No "
                                               "people, no brand, no writing. The same words again cost nothing."},
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
                   "description": "2 to %d clips, one for each moment." % MOST_AT_ONCE},
          "blur": {"type": "array", "maxItems": 4, "items": {
              "type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0, "maximum": 1000}},
                   "description": "Boxes [x1, y1, x2, y2] on your 0-1000 grid of the recording's picture to blur in "
                                  "every clip for its whole length, such as a stream's chat box, which stays in one "
                                  "place. A game stream must give its chat box here."}},
         ["clips"], make_clips, changes=True),
    Tool("open_clip", "Opened a clip of the album",
         "Open another of the clips made in one go, in the album above the clip, when the person wants to change it: "
         "your tools change the clip that is open, and the one you leave keeps its changes.",
         {"clip": {"type": "integer", "minimum": 1, "maximum": MOST_MOMENTS,
                   "description": "The clip's place in the album, 1 for the first."}},
         ["clip"], open_clip, changes=True),
    Tool("match_album", "Used this clip's look on the album",
         "Put parts of the look of the clip that is open on every other clip of the album, when the person wants one "
         "look across it. Each clip keeps the words of its own titles, its moment and its effects. Open the clip whose "
         "look it is first. One press: undo_album takes it all back. Their videos stay as they were until render_album.",
         {"parts": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string",
                                                                                   "enum": list(ALBUM_PARTS)},
                    "description": "Which parts to put on the others: tagline (the words drawn on the clip), circle "
                                   "(where the face circle sits), captions (on or off, and lit or pop), words (the "
                                   "font, size and colours of titles and captions), colour (the colour look)."}},
         ["parts"], match_album, changes=True),
    Tool("undo_album", "Took the album's look back",
         "Take back the last match_album on every clip it changed that was not changed since.",
         {}, [], undo_album, changes=True),
    Tool("render_album", "Made the album's videos again",
         "Make the video again of every clip of the album changed since its video was made, one after the other, and "
         "open the clip that was open. Each takes as long as render_clip.",
         {}, [], render_album, changes=True),
    Tool("set_tagline", "Set the tagline",
         "Draw a short line of words on the clip that is open, or take it off with empty words. To have it on every "
         "clip of the album, follow with match_album and tagline.",
         {"words": {"type": "string", "maxLength": 30, "description": "The words, empty to take them off."},
          "spot": {"type": "string", "enum": list(channels.SPOTS), "description": "Where the words sit; leave it out "
                                                                                  "to keep where they are."}},
         ["words"], set_tagline, changes=True),
    Tool("make_countdown", "Made the countdown",
         "Only when the person asked in their own words for a countdown of the best moments (a top 5, ท็อป 5, 5 อันดับ, "
         "นับถอยหลัง): one clip of %d to %d of the moments you proposed, in countdown order, the best last. The "
         "program cuts %g s around the peak of each (or the seconds you give), joins them into one recording and makes "
         "the clip from it in the card layout, each piece with its rank and your words as a title (\"5. ...\" down to "
         "\"1. ...\"). In a sports match, also unasked: the goals, in match order, your words with no rank, each "
         "piece the seconds you give, at most %g s. For the one best moment, make_clip; for a clip from each moment, "
         "make_clips." % (countdown.FEWEST, countdown.MOST, countdown.BEFORE + countdown.AFTER, countdown.LONGEST),
         {"moments": {"type": "array", "minItems": 2, "maxItems": countdown.MOST, "items": {
             "type": "object", "required": ["moment", "words"], "properties": {
                 "moment": {"type": "integer", "minimum": 1, "maximum": MOST_MOMENTS,
                            "description": "The moment's number, as you proposed it."},
                 "words": {"type": "string", "maxLength": 40,
                           "description": "A few words over its piece, after its rank, in the clip's language: what "
                                          "happens, without giving the ending away."},
                 "start": {"type": "string", "maxLength": 20,
                           "description": "Where its piece starts, inside the moment, like 1:51:30. Leave start and "
                                          "end out for the seconds around its peak."},
                 "end": {"type": "string", "maxLength": 20,
                         "description": "Where its piece ends, inside the moment."}}},
                     "description": "The moments in countdown order: the first is the lowest rank, the last the best."},
          "story": {"type": "string", "maxLength": 200,
                    "description": "What is counted down, in one line in the person's language, such as the five "
                                   "closest fights of the stream."},
          "final": {"type": "string", "maxLength": 20,
                    "description": "In a sports match: the score on the board near the end of the recording, as the "
                                   "board shows it."}},
         ["moments", "story"], make_countdown, changes=True),
    Tool("undo", "Took back the last change",
         "Take back the last change to the clip: a layout, a moved part, a style, a title, the captions, a blur, a hit, a "
         "look, a push, a sound, a mute or a decision about people. The "
         "person can also do this on the page. When the person says undo, ย้อนกลับ or เอาคืน and names no other change, "
         "call it at once, with no question, and never make the change again by hand.",
         {}, [], undo, changes=True),
]}
SCHEMAS = [t.schema() for t in TOOLS.values()]
# make_countdown refuses unless a person asked for a countdown in their own words, and offering it anyway moved the
# moment Glimmer made the SIFU clip from (10-30: 1:51 in 3 of 3 runs before it was offered, in 0 of 3 after), so it
# is offered only once someone asked.
UNASKED = [s for s in SCHEMAS if s["function"]["name"] != "make_countdown"]
FEATURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.json")
FEATURE_GROUPS = ("Clips", "Layout", "Captions", "Text and titles", "Effects", "Colour", "Blur", "Cut-out", "Sound",
                  "Channel", "Check and post")


def features():
    """Everything the studio can do, for the page's Features menu: [{"name": a group, "features": [{"tool", "name",
    "says", "ask", "show"}]}], groups by the kind of work in FEATURE_GROUPS order, from features.json, which has a
    line or more for every tool Muse has (a test holds it to TOOLS, so the menu cannot fall behind the tools). show
    names the small moving picture the page draws for it, so a person sees what it does before reading (Tarm 10-28:
    "people use their eyes first")."""
    with open(FEATURES, encoding="utf-8") as f:
        rows = json.load(f)
    groups = FEATURE_GROUPS + tuple(sorted({r["group"] for r in rows} - set(FEATURE_GROUPS)))
    return [{"name": g, "features": [{k: r[k] for k in ("tool", "name", "says", "ask", "show")} for r in rows
                                     if r["group"] == g]} for g in groups if any(r["group"] == g for r in rows)]
