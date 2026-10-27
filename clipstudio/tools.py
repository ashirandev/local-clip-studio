"""The agent's hands: every tool Muse can call, each with its input checked before it runs.

A tool is a plain function with a one-line label for the page, a description for Muse and a JSON Schema for its
input. Every tool also gets two fields from the loop:
- why: one line in the person's language, shown on the page next to the step
- plan_step: which step of Muse's plan the call works on, so the page can tick the plan

Input is checked here, not trusted: the hosted API does not promise that arguments match the schema. Wrong input is
refused with a sentence that says what to send instead, so Muse can fix it and call again.
"""
import json
import os

from . import maps, previews, recording
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
    look = {"n": len(looks) + 1, "start": a, "end": b, "pictures": [], "loudest": None}
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
    lines.append("What is said here has not been heard yet, so judge by the pictures%s."
                 % (" and the sound" if peak else ""))
    lines.append("Next: in your next why, say what you saw and what led up to the peak. Then look earlier if the start "
                 "of the action is missing, look wider if you need more, look at the next stretch of the map, or "
                 "propose the moments you have looked at.")
    looks.append(look)
    job.show(look=look)
    return "\n".join(lines)


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
    for p in picked:
        p["preview"] = previews.cut(job.chat, job.n, rec, p["start"], p["end"])
    job.chat.moments = picked
    job.show(moments=picked)
    lines = ["Proposed %d moment%s. The person sees each as a card with your reason and a picture from your look; a "
             "rough video of each is being cut and plays on its card when it is ready."
             % (len(picked), "" if len(picked) == 1 else "s")]
    lines += ["%d. %s-%s (%d s, from look %d): %s" % (p["n"], clock(p["start"]), clock(p["end"]),
                                                     round(p["end"] - p["start"]), p["look"], p["reason"])
              for p in picked]
    lines.append("Go on with moment 1 by yourself: do not ask the person to choose, and do not list the moments again "
                 "in words. They can tap another card to switch. Nothing is cut, laid out, captioned or blurred until a "
                 "tool's result says it is: never tell the person a clip is ready when no tool made it.")
    return "\n".join(lines)


STYLES = {"clean": "plain text, quiet and clear", "pop": "bold colours and quick hits",
          "calm": "slow and relaxed, a few words at a time"}
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
    out += [{"label": s.capitalize(), "say": "(tapped) Use the %s style." % s} for s in STYLES if s != clip["style"]]
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
         "with its time, sent to you right after the result. Look at a stretch before you propose it. To see what "
         "led up to a peak, look earlier; to see more around it, look wider (at most 10 minutes).",
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
]}
SCHEMAS = [t.schema() for t in TOOLS.values()]
