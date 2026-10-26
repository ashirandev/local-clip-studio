"""The agent loop: Muse plans, calls one tool at a time, reads the result and goes on until the work is done.

What the loop keeps, whatever Muse writes:
- The conversation only grows at its end. The system message and the tool list are the same bytes for the whole
  chat, so llama-server reuses everything it has already read and reads only the newest messages. Each step's
  engine line shows how much was reused.
- One tool call is one numbered step. Each step goes to the job log with its why, its input, its result, its
  seconds and its engine numbers.
- Every tool result ends with the state of the job ("Now: ..."), so the newest thing Muse reads is the current state.
- The repeat guard refuses a call only when the same call already got the same result twice and nothing has changed
  since. A call with fixed input, or any call after a change, runs.
- The loop ends the job, not Muse: when Muse answers in words (after one reminder of any plan steps it left open),
  when it keeps repeating a refused call, or after MAX_STEPS steps.

The chat is written to work/chats as it happens: <chat>.events.jsonl is the job log the page is drawn from, and
<chat>.messages.jsonl is exactly what Muse read.
"""
import json
import os
import secrets
import threading
import time
import traceback

from . import mind, words
from .tools import SCHEMAS, TOOLS, Refused

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("LCS_WORK") or os.path.join(ROOT, "work")
AGENT_TEMPERATURE = 0.5
AGENT_MAX_TOKENS = 4096
MAX_STEPS = 30
GUARD_STOPS = 3
SYSTEM = """You are the editor in Local Chat Studio. People bring long recordings (selling lives, game streams, travel \
clips) and you turn them into short vertical clips ready to post.

You work as an agent. You call one tool at a time, read what it returns, and go on by yourself until the work is \
done. Nobody approves your steps, so never stop to ask for something a tool can find out. When a couple of tries \
do not find what the person means, say what you did find and ask them.

- When a request needs more than one tool call, call set_plan first with a few short steps. Every later call says \
which plan step it works on (plan_step).
- Every call has a why: one short line, in the language the person writes in, telling the person why you take this \
step.
- When a tool refuses, its answer says what to fix. Fix the input and call again.
- Do not repeat a call that already gave you its answer.
- Each tool result ends with a line that starts with "Now:". It is the current state; trust it over anything older.
- When the work is done, or only the person can decide, answer briefly, in the language the person writes in. Say \
only what your tools actually did, in plain words, without tool or field names.
- A question that needs no tool gets a plain answer, with no plan."""


class Busy(Exception):
    """The chat already has a job running."""


class Chat:
    """One conversation: the messages Muse reads, the events the page draws, and the job running now."""

    def __init__(self):
        self.id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        self.messages, self.events = [], []
        self.changed = 0
        self.jobs = 0
        self.job = None
        self.cond = threading.Condition()
        folder = os.path.join(WORK, "chats")
        os.makedirs(folder, exist_ok=True)
        self.log_path = os.path.join(folder, self.id + ".events.jsonl")
        self.talk_path = os.path.join(folder, self.id + ".messages.jsonl")
        self.add({"role": "system", "content": SYSTEM})

    def add(self, message):
        """Adds a message at the end. Nothing earlier is ever changed."""
        self.messages.append(message)
        _append(self.talk_path, message)

    def emit(self, kind, **data):
        with self.cond:
            event = dict(data, id=len(self.events), kind=kind, t=round(time.time(), 3))
            self.events.append(event)
            self.cond.notify_all()
        _append(self.log_path, event)
        return event

    def since(self, n, timeout):
        """The events from number n on, waiting up to timeout seconds for one if there are none yet."""
        with self.cond:
            if len(self.events) <= n:
                self.cond.wait(timeout)
            return self.events[n:]

    def busy(self):
        return bool(self.job and self.job.thread.is_alive())

    def say(self, words, mind_key):
        with self.cond:
            if self.busy():
                raise Busy("Muse is still working on your last message. Wait for it to finish.")
            self.jobs += 1
            self.job = Job(self, mind.get(mind_key), words)
            self.job.thread.start()
        return self.job


class Job:
    """One message from the person and everything Muse does about it."""

    def __init__(self, chat, brain, words):
        self.chat, self.brain, self.words, self.n = chat, brain, words, chat.jobs
        self.plan, self.plan_shown = [], "[]"
        self.steps = self.refused = self.fixed = self.guard_stops = 0
        self.seen = {}
        self.last_refused = {}
        self.thread = threading.Thread(target=self.run, name="job-%d" % self.n, daemon=True)

    def emit(self, kind, **data):
        return self.chat.emit(kind, job=self.n, **data)

    def run(self):
        chat, started, end, nudged = self.chat, time.time(), "answered", False
        chat.add({"role": "user", "content": self.words})
        chat.changed += 1
        self.emit("person", text=self.words)
        self.emit("job", mind=self.brain.label)
        try:
            while True:
                if self.steps >= MAX_STEPS:
                    end = "too many steps"
                    self.emit("note", text="Muse took %d steps on this message, the most one message may take, so the "
                                           "studio stopped it here. Say what to do next." % self.steps)
                    break
                self.emit("wait", step=self.steps + 1)
                got = self.brain.ask(chat.messages, tools=SCHEMAS, temperature=AGENT_TEMPERATURE,
                                     max_tokens=AGENT_MAX_TOKENS)
                calls = got["tool_calls"]
                got["text"] = words.tidy(got["text"])
                got["message"]["content"] = words.tidy(got["message"]["content"])
                for i, call in enumerate(calls):
                    call.setdefault("id", "call_%d_%d" % (self.steps + 1, i))
                    if isinstance(call.get("function"), dict):
                        call["function"]["arguments"] = words.tidy_arguments(call["function"].get("arguments"))
                chat.add(got["message"])
                if not calls:
                    left = self.finish_plan()
                    if left and not nudged:
                        nudged = True
                        self.emit("answer", final=False, text=got["text"], thinking=got["thinking"],
                                  engine=got["engine"], mind=self.brain.label)
                        chat.add({"role": "user", "content": "[studio] Your plan has steps no call has named yet: %s. "
                                  "Do them now, or answer the person and say why they are not needed."
                                  % "; ".join("%d. %s" % s for s in left)})
                        self.emit("note", text="Muse stopped with plan steps still open, so the studio asked it to go "
                                               "on or say why.")
                        continue
                    self.emit("answer", final=True, text=got["text"], thinking=got["thinking"],
                              engine=got["engine"], mind=self.brain.label)
                    break
                for i, call in enumerate(calls):
                    self.steps += 1
                    shown, text = self.call(call)
                    chat.add({"role": "tool", "tool_call_id": call["id"], "content": text + "\n\n" + self.state()})
                    first = i == 0
                    self.emit("step", step=self.steps, said=got["text"] if first else "",
                              thinking=got["thinking"] if first else "", engine=got["engine"] if first else None,
                              **shown)
                    self.show_plan()
                if self.guard_stops >= GUARD_STOPS:
                    end = "repeating"
                    self.emit("note", text="Muse kept repeating a call that had already given its answer, so the "
                                           "studio stopped this job. Say what to do next.")
                    break
        except mind.MindError as e:
            end = "mind"
            self.emit("note", text=str(e))
        except Exception as e:
            end = "fault"
            self.emit("note", text="The studio broke while working (%s). The details are in the job log." % e,
                      fault=traceback.format_exc())
        self.emit("end", steps=self.steps, refused=self.refused, fixed=self.fixed,
                  seconds=round(time.time() - started, 1), why=end)

    def call(self, call):
        """Runs one tool call through the input check and the repeat guard. Returns (fields for the page, text)."""
        fn = call.get("function") or {}
        name, raw = fn.get("name") or "", fn.get("arguments")
        tool = TOOLS.get(name)
        shown = {"tool": name, "label": tool.label if tool else name, "why": "", "input": raw}
        try:
            if tool is None:
                raise Refused("There is no tool named %s. The tools are: %s." % (name or "(none)", ", ".join(TOOLS)))
            why, step, clean = tool.check(_arguments(raw), len(self.plan))
        except Refused as e:
            return self.refuse(shown, name, str(e))
        shown.update(why=why, input=clean, plan_step=step)
        key = name + json.dumps(clean, sort_keys=True, ensure_ascii=False)
        seen = self.seen.get(key)
        if seen and seen["changed"] == self.chat.changed and seen["times"] >= 2:
            self.guard_stops += 1
            shown["guard"] = True
            return self.refuse(shown, name, "This exact call already ran twice, got the same result both times, and "
                                            "nothing has changed since. Its result was:\n%s\nDo something different, "
                                            "or tell the person what is stuck." % seen["result"])
        self.guard_stops = 0
        if step:
            self.tick(step)
        t0 = time.time()
        try:
            text, ran = tool.run(self, why, **clean), True
        except Refused as e:
            text, ran = str(e), False
        except Exception as e:
            text, ran = "%s broke (%s). This is a fault in the studio, not in your input." % (name, e), False
            shown["fault"] = traceback.format_exc()
        shown["seconds"] = round(time.time() - t0, 2)
        if ran and tool.changes:
            self.chat.changed += 1
        if seen and seen["changed"] == self.chat.changed and seen["result"] == text:
            seen["times"] += 1
        else:
            self.seen[key] = {"result": text, "times": 1, "changed": self.chat.changed}
        if not ran:
            return self.refuse(shown, name, text)
        if self.last_refused.get(name):
            self.fixed += 1
        self.last_refused[name] = False
        shown.update(result=text, refused=False)
        return shown, text

    def refuse(self, shown, name, sentence):
        self.refused += 1
        self.last_refused[name] = True
        shown.update(result=sentence, refused=True)
        return shown, sentence

    def set_plan(self, steps, why):
        changed = bool(self.plan)
        self.plan = [{"text": s, "state": "todo"} for s in steps]
        self.show_plan(why=why, changed=changed)

    def tick(self, step):
        for i, item in enumerate(self.plan, 1):
            if i < step:
                item["state"] = "done"
            elif i == step:
                item["state"] = "now"
            elif item["state"] == "now":
                item["state"] = "todo"

    def finish_plan(self):
        """Marks the step being worked on as done. Returns the steps no call has named: [(number, text)]."""
        for item in self.plan:
            if item["state"] == "now":
                item["state"] = "done"
        self.show_plan()
        return [(i, item["text"]) for i, item in enumerate(self.plan, 1) if item["state"] == "todo"]

    def show_plan(self, why=None, changed=False):
        now = json.dumps(self.plan, ensure_ascii=False)
        if now != self.plan_shown or why:
            self.plan_shown = now
            self.emit("plan", items=self.plan, why=why, changed=changed)

    def state(self):
        parts = ["Now: step %d of this job" % self.steps]
        if self.plan:
            done = sum(item["state"] == "done" for item in self.plan)
            line = "plan: %d of %d steps done" % (done, len(self.plan))
            now = [(i, item["text"]) for i, item in enumerate(self.plan, 1) if item["state"] == "now"]
            if now:
                line += ", working on step %d (%s)" % now[0]
            parts.append(line)
        else:
            parts.append("no plan")
        return " · ".join(parts)


def _arguments(raw):
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    try:
        args = json.loads(raw)
    except ValueError as e:
        raise Refused("The input was not readable JSON (%s). Send the fields as one JSON object." % e)
    if not isinstance(args, dict):
        raise Refused("The input must be one JSON object with named fields.")
    return args


def _append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


CHATS = {}
_chats_lock = threading.Lock()


def current():
    """The chat the page shows: the newest one, made on first use."""
    with _chats_lock:
        if not CHATS:
            chat = Chat()
            CHATS[chat.id] = chat
        return list(CHATS.values())[-1]


def find(chat_id):
    return CHATS.get(chat_id)
