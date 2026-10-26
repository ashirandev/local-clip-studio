"""The agent's hands: every tool Muse can call, each with its input checked before it runs.

A tool is a plain function with a one-line label for the page, a description for Muse and a JSON Schema for its
input. Every tool also gets two fields from the loop:
- why: one line in the person's language, shown on the page next to the step
- plan_step: which step of Muse's plan the call works on, so the page can tick the plan

Input is checked here, not trusted: the hosted API does not promise that arguments match the schema. Wrong input is
refused with a sentence that says what to send instead, so Muse can fix it and call again.
"""
import os
import time

WHY = {"type": "string",
       "description": "One short line, in the language the person writes in, telling the person why you take "
                      "this step."}
PLAN_STEP = {"type": "integer", "minimum": 1,
             "description": "Which step of your plan this call works on (1 is the first). Leave it out if there is "
                            "no plan."}
VIDEO_TYPES = (".mp4", ".mkv", ".mov", ".m4v", ".webm", ".ts", ".flv", ".avi")
SMALLEST_RECORDING = 1_000_000
MOST_FILES_LOOKED_AT = 50_000


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


def video_folders():
    """The folders recordings are kept in: LCS_VIDEOS (separated by ; on Windows, : elsewhere), else ~/Videos."""
    named = os.environ.get("LCS_VIDEOS", "")
    folders = [f.strip() for f in named.split(os.pathsep) if f.strip()] or [os.path.expanduser("~/Videos")]
    return [f for f in folders if os.path.isdir(f)]


def find_recordings(job, why, words="", newest=5):
    folders = video_folders()
    if not folders:
        raise Refused("No folder of recordings was found. Ask the person to set LCS_VIDEOS to the folders their "
                      "recordings are in, then start the studio again.")
    wanted = words.lower().split()
    found, looked = [], 0
    for top in folders:
        for here, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if not d.startswith(("_", "."))]
            for name in files:
                looked += 1
                if not name.lower().endswith(VIDEO_TYPES):
                    continue
                path = os.path.join(here, name)
                if wanted and not all(w in os.path.relpath(path, top).lower() for w in wanted):
                    continue
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size >= SMALLEST_RECORDING:
                    found.append((st.st_mtime, st.st_size, path))
            if looked > MOST_FILES_LOOKED_AT:
                break
    where = ", ".join(folders)
    if not found:
        about = 'has "%s" in its name or folders' % words if wanted else "is there"
        return "No recording %s in %s.%s" % (about, where, _names_here(folders))
    found.sort(reverse=True)
    shown = found[:newest]
    about = ' for "%s"' % words if wanted else ""
    lines = ["%d recordings%s in %s; the %d newest:" % (len(found), about, where, len(shown))]
    for i, (mtime, size, path) in enumerate(shown, 1):
        lines.append("%d. %s · %.1f GB · recorded %s · %s" % (i, os.path.basename(path), size / 1e9, _when(mtime),
                                                              path))
    return "\n".join(lines)


def _names_here(folders):
    """How the recordings here are named, so Muse can tell the person what to say instead."""
    names, subfolders = [], []
    for top in folders:
        try:
            entries = sorted(os.scandir(top), key=lambda e: e.stat().st_mtime, reverse=True)
        except OSError:
            continue
        names += [e.name for e in entries if e.is_file() and e.name.lower().endswith(VIDEO_TYPES)][:3]
        subfolders += [e.name for e in entries if e.is_dir() and not e.name.startswith(("_", "."))][:6]
    out = ""
    if names:
        out += " The newest files here are named like: %s." % ", ".join(names[:4])
    if subfolders:
        out += " Folders: %s." % ", ".join(subfolders[:8])
    return out


def _when(t):
    fmt = "%a %d %b %H:%M" if time.localtime(t).tm_year == time.localtime().tm_year else "%d %b %Y %H:%M"
    return time.strftime(fmt, time.localtime(t))


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
]}
SCHEMAS = [t.schema() for t in TOOLS.values()]
