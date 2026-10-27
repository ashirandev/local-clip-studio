"""The agent's hands: every tool Muse can call, each with its input checked before it runs.

A tool is a plain function with a one-line label for the page, a description for Muse and a JSON Schema for its
input. Every tool also gets two fields from the loop:
- why: one line in the person's language, shown on the page next to the step
- plan_step: which step of Muse's plan the call works on, so the page can tick the plan

Input is checked here, not trusted: the hosted API does not promise that arguments match the schema. Wrong input is
refused with a sentence that says what to send instead, so Muse can fix it and call again.
"""
import os

from . import recording
from .recipes import RECIPES

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
    """The value in the type its schema asks for, or Refused. Numbers sent as text are accepted."""
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
        if not isinstance(value, str):
            raise Refused("%s must be text." % name)
        if spec.get("enum") and value.strip() not in spec["enum"]:
            raise Refused("%s must be one of: %s." % (name, ", ".join(spec["enum"])))
        if len(value) > spec.get("maxLength", 2000):
            raise Refused("%s is too long; keep it under %d characters." % (name, spec.get("maxLength", 2000)))
        return value.strip()
    if kind == "array":
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            raise Refused("%s must be a list of short lines of text." % name)
        if not spec.get("minItems", 0) <= len(value) <= spec.get("maxItems", 100):
            raise Refused("%s must have %d to %d lines; it has %d."
                          % (name, spec.get("minItems", 0), spec.get("maxItems", 100), len(value)))
        return [v.strip() for v in value]
    return value


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
    Tool("find_recordings", "Looked for recordings",
         "List the recordings on this computer, newest first, with their size, time and full path. Words narrow the "
         "list to files whose name or folder holds every word, in upper or lower case. It only reads the folder "
         "lists, never the files.",
         {"words": {"type": "string", "maxLength": 200,
                    "description": "Words that must all be in the file name or its folders, such as a game or a date. "
                                   "Leave empty for the newest recordings."},
          "newest": {"type": "integer", "minimum": 1, "maximum": 20,
                     "description": "How many to list, newest first (1 to 20, default 5)."}},
         [], find_recordings),
    Tool("open_recording", "Opened the recording",
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
]}
SCHEMAS = [t.schema() for t in TOOLS.values()]
