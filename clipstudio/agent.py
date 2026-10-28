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
- The person can steer at any time. Words typed while Muse works cancel the question in flight and go in as the
  newest message, so Muse answers them next; Stop cancels it and ends the job. Everything before the cut is still
  reused by llama-server.
- Muse's thinking is shown while it is written (Chat.live), but only the finished step goes into the job log.

The chat is written to work/chats as it happens: <chat>.events.jsonl is the job log the page is drawn from, and
<chat>.messages.jsonl is exactly what Muse read. An earlier chat opens again from these two files.
"""
import base64
import json
import os
import re
import secrets
import threading
import time
import traceback

from . import faces, hearing, maps, mind, previews, project, recording, render, text, words
from .tools import (CAPTION_LETTERS, LOOK_LABELS, LOOKS, SCHEMAS, STYLES, TOOLS, Refused, _undecided, describe,
                    faces_at)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("LCS_WORK") or os.path.join(ROOT, "work")
AGENT_TEMPERATURE = 0.5
AGENT_MAX_TOKENS = 4096
SPELL_TOKENS = 8192
MAX_STEPS = 30
GUARD_STOPS = 3
REPLY_ROOM = 2048
PICTURE_TOKENS = 256
ROOM_TRIES = 4
TAKEN_OUT = ("(A picture was here. It was taken out to make room in Muse's memory: the tool that showed it shows it "
             "again.)")
SHOW_THINKING_EVERY = 0.4
CHATS_LISTED = 20
CHAT_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
TAPPED = "the person tapped it on the page"
CAPTION_CHOICES = {"off": {"on": False}, "plain": {}, "lit": {"lit": True}, "pop": {"pop": True}}
COLOUR_LOOKS = ("none",) + tuple(n for n in LOOKS if n != "none")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(])")
SYSTEM = """You are the editor in Local Clip Studio. People bring long recordings (selling lives, game streams, travel \
clips) and you turn them into short vertical clips ready to post.

You work as an agent. You call one tool at a time, read what it returns, and go on by yourself until the work is \
done. Nobody approves your steps, so never stop to ask for something a tool can find out. When a couple of tries \
do not find what the person means, say what you did find and ask them.

- When a request needs more than one tool call, call set_plan first with a few short steps. Every later call says \
which plan step it works on (plan_step).
- Every call has a why: one short line, in the language the person writes in, telling the person why you take this \
step.
- When a tool refuses, its answer says what to fix. Fix the input and call again.
- A clip is done only when the person has it: after the last change, render it, check it, then hand it over with \
a title, the post text and hashtags. Plan those three steps too.
- Do not repeat a call that already gave you its answer.
- Each tool result ends with a line that starts with "Now:". It is the current state; trust it over anything older.
- When the work is done, or only the person can decide, answer briefly, in the language the person writes in. Say \
only what your tools actually did, in plain words, without tool or field names. Write plain text, not markdown.
- A question that needs no tool gets a plain answer, with no plan.
- A message that starts with "(tapped)" is a choice the person made with a button under your guesses. Act on it \
like their own words, and keep answering in the language they type in.
- A message that starts with "(opened)" means the person opened that recording to make a clip from it. Start at \
once: pick its kind, then plan and work. Do not ask anything first; the person can correct any guess with a tap. \
Judge the kind by the picture that comes with it and by the person's words, not by the file name."""


class Missing(Exception):
    """What the page named is not on the clip, or not as the page showed it. The sentence says so."""


class Made(Exception):
    """The page's hand made the tool's edit. What the tool does after its edit is for Muse."""


class Allowed(Exception):
    """A trial hand reached the tool's edit: the tool would make it."""


class Chat:
    """One conversation: the messages Muse reads, the events the page draws, and the job running now."""

    def __init__(self, chat_id=None):
        self.id = chat_id or time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        self.messages, self.events = [], []
        self.changed = self.jobs = 0
        self.job = None
        self.live, self.live_rev = None, 0
        self.clip = self.recording = None
        self.looks, self.moments = [], None
        self.clips = None
        self.project = project.Project()
        self.rendered = []
        self.checks = {}
        self.told = []
        self.spelling = None
        self.making = None
        self.cond = threading.Condition()
        folder = os.path.join(WORK, "chats")
        os.makedirs(folder, exist_ok=True)
        self.log_path = os.path.join(folder, self.id + ".events.jsonl")
        self.talk_path = os.path.join(folder, self.id + ".messages.jsonl")
        if chat_id is None:
            self.add({"role": "system", "content": SYSTEM})

    @classmethod
    def load(cls, chat_id):
        """A chat written earlier, read back from work/chats exactly as Muse read it."""
        chat = cls(chat_id)
        chat.messages = _read(chat.talk_path)
        chat.events = _read(chat.log_path)
        chat.jobs = max([e.get("job") or 0 for e in chat.events] or [0])
        ended = {e.get("job") for e in chat.events if e["kind"] == "end"}
        for n in sorted({e["job"] for e in chat.events if e.get("job")} - ended):
            steps = sum(1 for e in chat.events if e.get("job") == n and e["kind"] == "step")
            chat.emit("end", job=n, steps=steps, refused=0, fixed=0, seconds=0, why="closed")
        guesses = [e for e in chat.events if e["kind"] == "guesses"]
        chat.clip = guesses[-1]["clip"] if guesses else None
        opened = [e for e in chat.events if e["kind"] == "opened"]
        chat.recording = opened[-1]["recording"] if opened else None
        path = None
        for e in chat.events:
            if e["kind"] == "opened":
                if e["recording"]["path"] != path:
                    chat.looks, chat.moments, chat.project, chat.rendered = [], None, project.Project(), []
                    chat.checks = {}
                path = e["recording"]["path"]
            elif e["kind"] == "edit":
                chat.project.add(e["entry"])
            elif e["kind"] == "undo":
                chat.project.undo()
            elif e["kind"] == "rendered":
                chat.rendered.append(e["file"])
            elif e["kind"] == "checked":
                chat.checks[e["file"]] = e["checks"]
            elif e["kind"] == "step":
                if e.get("clips"):
                    chat.clips = e["clips"]
                if e.get("look"):
                    chat.looks.append(e["look"])
                if e.get("moments"):
                    chat.moments = e["moments"]
        chat.catch_up_previews()
        return chat

    def catch_up_previews(self):
        """A preview the studio was closed before it was cut gets its event now: ready if its file is there."""
        told = {e["name"] for e in self.events if e["kind"] == "preview"}
        for e in [e for e in self.events if e["kind"] == "step" and e.get("moments")]:
            for m in e["moments"]:
                if m["preview"] not in told:
                    told.add(m["preview"])
                    ok = os.path.isfile(os.path.join(previews.folder(), m["preview"]))
                    self.emit("preview", job=e["job"], name=m["preview"], ok=ok, seconds=0,
                              error=None if ok else "The studio was closed before this preview was cut.")

    def add(self, message):
        """Adds a message at the end. Nothing earlier is ever changed."""
        self.messages.append(message)
        _append(self.talk_path, message)

    def emit(self, kind, **data):
        with self.cond:
            event = dict(data, id=len(self.events), kind=kind, t=round(time.time(), 3))
            self.events.append(event)
            _append(self.log_path, event)
            self.cond.notify_all()
        return event

    def since(self, n, rev, timeout):
        """(events from number n on, Muse's live thinking if it changed since rev), waiting up to timeout seconds."""
        with self.cond:
            if len(self.events) <= n and (self.live is None or self.live_rev == rev):
                self.cond.wait(timeout)
            live = dict(self.live, rev=self.live_rev) if self.live and self.live_rev != rev else None
            return self.events[n:], live

    def busy(self):
        return bool(self.job and not self.job.done)

    def say(self, words, mind_key, pictures=(), shown=None):
        """The person's message: a new job, or, while Muse works, words that cut in on the job running now.
        Pictures (jpg paths) go to Muse with the words, in the same message. shown is what the page shows instead
        of words, when Muse is told more than the person said. What the person changed on the page since Muse last
        worked goes first, so Muse never works from an old clip."""
        shown = words if shown is None else shown
        with self.cond:
            told, self.told = self.told, []
        if told:
            words = "[studio] Since your last answer: %s. Now: %s\n\n%s" % ("; ".join(told), self.clip_line(), words)
        content = _with_pictures(words, pictures)
        with self.cond:
            job = self.job
            if job and not job.done:
                self.emit("person", job=job.n, text=shown, cut_in=True)
                job.cut_in(content)
                return job
            if self.spelling:
                self.spelling.cancel()
            self.jobs += 1
            self.job = Job(self, mind.get(mind_key), content)
            self.emit("person", job=self.jobs, text=shown)
            self.job.thread.start()
            return self.job

    def stop(self):
        """Stops the job running now and the video the page is making, at once. True when either was running."""
        job, making = self.job, self.making
        stopped = bool(job and job.stop())
        return bool(making and making.stop()) or stopped

    def edit(self, entry, job=None):
        """Adds one change to the clip's project list; the page redraws the clip. Returns the clip's state."""
        with self.cond:
            self.project.add(entry)
            self.changed += 1
        self.emit("edit", job=job, entry=entry, clip=self.clip_view(), said=describe(entry))
        return self.project.state()

    def set_person(self, n, blur):
        """The person's tap on a face: person n blurred, or kept as they are, and told to Muse before its next job.
        False when the clip has no person n."""
        if not any(p["n"] == n for p in self.project.state()["people"]):
            return False
        self.edit({"do": "people", "blur": [n] if blur else [], "keep": [] if blur else [n]})
        with self.cond:
            self.told.append("the person %s person %d by tapping their face" % ("blurred" if blur else "kept", n))
        return True

    def faces(self, t):
        return faces_at(self.project.state(), t)

    def tap(self, name, trial=False, **args):
        """Runs Muse's tool name for a tap on the page, through the tool's own input check (see Hand). Returns
        (None, the entry it made) when it made its edit, or with trial would, and (the first sentence of its refusal,
        None) when it refused."""
        tool, hand = TOOLS[name], Hand(self, trial)
        try:
            why, _, clean = tool.check(dict(args, why=TAPPED), 0)
            said = tool.run(hand, why, **clean)
        except (Made, Allowed):
            return None, hand.entry
        except Refused as e:
            said = str(e)
        return _first_sentence(said), None

    def panel(self):
        """What the page shows beside and under the clip, or {"clip": False} before there is one: each choice of
        captions, text style and colour look, with the one the clip has now and the reason a choice is off (its tool
        is tried, changing nothing, so the page and Muse never disagree); the words on the clip by their time; and
        whether the video can be made again. Where the captions sit is not offered: no tool moves them (the layout
        places them, and move_part moves the picture's parts only)."""
        state = self.project.state()
        if not state["clip"]:
            return {"clip": False}
        caps = state["captions"]
        now = {"captions": "off" if not caps else "lit" if caps.get("lit") else "pop" if caps.get("pop") else "plain",
               "style": state["style"]["name"] if state["style"] else (self.clip or {}).get("style", "clean"),
               "look": state["look"] or "none"}

        def choices(group, names, label, tool, args):
            return [{"name": n, "label": label(n), "now": n == now[group],
                     "why": None if n == now[group] else self.tap(tool, True, **args(n))[0]} for n in names]

        return {"clip": True, "length": state["length"],
                "captions": choices("captions", CAPTION_CHOICES, str.capitalize, "add_captions", CAPTION_CHOICES.get),
                "styles": choices("style", STYLES, lambda n: LOOK_LABELS.get(n, n.capitalize()), "set_style",
                                  lambda n: {"style": n}),
                "looks": choices("look", COLOUR_LOOKS, str.capitalize, "set_look", lambda n: {"look": n}),
                "words": self.words_on(state), "again": bool(self.rendered) and self.finished() is None,
                "making": self.making is not None}

    def words_on(self, state):
        """The words on the clip as blocks the page places by their time: every title, and every phrase of the
        captions (its lines joined give it back whole), "typed" when the person wrote it on the page."""
        out = [{"kind": "title", "n": k, "text": t["text"], "start": t["start"], "end": t["end"]}
               for k, t in enumerate(state["titles"])]
        lines = (state["captions"] or {}).get("lines") or []
        for k in sorted({ln["phrase"] for ln in lines}):
            mine = [ln for ln in lines if ln["phrase"] == k]
            out.append({"kind": "caption", "n": k, "text": "".join(ln["text"] for ln in mine).strip(),
                        "start": mine[0]["from"], "end": mine[-1]["to"],
                        "typed": any(ln.get("typed") for ln in mine)})
        return sorted(out, key=lambda b: (b["start"], b["kind"]))

    def set_captions(self, mode):
        """The person's tap on Off, Plain, Lit or Pop beside the clip: add_captions's own edit, made at once without
        asking Muse (so Undo takes it back), and told to Muse before its next job. Returns None, or the sentence why
        not."""
        self._clip_now()
        if mode not in CAPTION_CHOICES:
            return "Choose Off, Plain, Lit or Pop for the captions."
        refused, entry = self.tap("add_captions", **CAPTION_CHOICES[mode])
        if entry:
            self._tell("the person %s on the page" % ("turned the captions off" if mode == "off" else
                                                      "put on %s" % describe(entry)))
        return refused

    def set_style(self, name):
        """The person's tap on one of the house styles of the words: set_style's own edit, as set_captions."""
        self._clip_now()
        if name not in STYLES:
            return "Choose one of the text styles: %s." % ", ".join(LOOK_LABELS.get(n, n.capitalize()) for n in STYLES)
        refused, entry = self.tap("set_style", style=name)
        if entry:
            self._tell("the person chose the %s style on the page" % name)
        return refused

    def set_look(self, name):
        """The person's tap on a colour look for the footage: set_look's own edit, as set_captions."""
        self._clip_now()
        if name not in COLOUR_LOOKS:
            return "Choose None, Vivid, Warm or Film for the colour look."
        refused, entry = self.tap("set_look", look=name)
        if entry:
            self._tell("the person %s on the page" % ("took the colour look off" if name == "none" else
                                                      "chose the %s look" % name))
        return refused

    def set_words(self, kind, n, was, words):
        """The person's own words for one block of the words under the clip. A title goes on again through
        add_title, in its place and time, so its checks hold (it must fit the place and keep off faces) and it
        replaces the old one. A caption phrase is broken into lines as add_captions breaks it, over the phrase's own
        time, and marked typed: what the person typed is what they want, so the check "Captions are words that were
        said" does not fail it, and add_captions keeps it when the captions go on again. Told to Muse before its next
        job. Returns None, or the sentence why not; Missing when the block is not on the clip as the page showed it."""
        state = self._clip_now()
        block = next((b for b in self.words_on(state) if b["kind"] == kind and b["n"] == n), None)
        if block is None or block["text"] != was:
            raise Missing("Those words are not on the clip as the page showed them. Look at the clip again.")
        words = " ".join(str(words).split())
        if not words:
            return "Type the words first."
        if words == was:
            return None
        if kind == "title":
            t = state["titles"][n]
            spot = {"place": t["place"]} if t["place"] in state["clip"]["layout"]["places"] else \
                {"box": _grid_box(t["box"])}
            refused, entry = self.tap("add_title", words=words, start=t["start"], end=t["end"],
                                      key=t["key"] if t.get("key") and t["key"] in words else None, **spot)
            if refused:
                return refused
        else:
            caps = state["captions"]
            mine = [ln for ln in caps["lines"] if ln["phrase"] == n]
            typed = text.caption_lines([{"from": mine[0]["from"], "to": mine[-1]["to"], "text": words}],
                                       CAPTION_LETTERS)
            for ln in typed:
                ln.update(phrase=n, typed=True)
                if "who" in mine[0]:
                    ln["who"] = mine[0]["who"]
            self.edit(dict(caps, lines=[ln for ln in caps["lines"] if ln["phrase"] < n] + typed +
                           [ln for ln in caps["lines"] if ln["phrase"] > n]))
        self._tell('the person typed their own words for the %s at %g-%g s on the page: "%s" is now "%s"'
                   % (kind, block["start"], block["end"], was, words))
        return None

    def make_again(self):
        """The person's tap on "Make the video again": render_clip's own render of the clip as it is now, run in the
        background without asking Muse, its progress shown as a render's is. render_clip's checks hold (nobody left
        undecided, nothing made when the file is already the clip as it is now), and its file is the one check_clip
        and hand_over take, as from Muse. Returns None once it has started, or the sentence why not."""
        state = self._clip_now()
        if self.busy():
            return "Muse is working on the clip now. Wait until it is done, or press Stop."
        if self.making:
            return "The video is being made now."
        if _undecided(state):
            return "Decide about each person in the clip first: tap a face to blur it, and tap it again to show it."
        refused, _ = self.tap("render_clip", trial=True)
        if refused:
            return refused
        hand = Hand(self)
        with self.cond:
            if self.busy() or self.making:
                return "Muse is working on the clip now. Wait until it is done, or press Stop."
            self.making = hand
        threading.Thread(target=self._make, args=(hand,), name="make-again", daemon=True).start()
        return None

    def _make(self, hand):
        try:
            said = TOOLS["render_clip"].run(hand, TAPPED)
        except Refused as e:
            said = str(e)
        except Exception as e:
            said = "The studio broke while making the video (%s)." % e
        finally:
            with self.cond:
                self.making, hand.done = None, True
        if hand.shown_extra.get("rendered"):
            self._tell("the person made the video again on the page")
        else:
            self.emit("not_made", error=_first_sentence(said))

    def _clip_now(self):
        state = self.project.state()
        if not state["clip"]:
            raise Missing("There is no clip yet. Ask Muse to make one first.")
        return state

    def _tell(self, line):
        with self.cond:
            self.told.append(line)

    def undo(self, job=None):
        """Takes the last change back, or returns None when there is none. The person's undo is told to Muse
        before its next job (or in the state line of the job running now)."""
        with self.cond:
            entry = self.project.undo()
            if entry is None:
                return None
            self.changed += 1
            if job is None:
                self.told.append("the person took back %s" % describe(entry))
        self.emit("undo", job=job, entry=entry, clip=self.clip_view(), said=describe(entry))
        return entry

    def finished(self):
        """The finished file made from the clip as it is now, or None: {"name", "signature", "seconds", "size",
        "length"}. A change after the render makes it stale, and undoing that change makes it current again."""
        sign = project.signature(self.project.entries)
        return next((f for f in reversed(self.rendered) if f["signature"] == sign
                     and os.path.isfile(os.path.join(render.folder("clips"), f["name"]))), None)

    def clip_view(self):
        """The clip as the page draws it: its length, layout and words on screen, or None before it is made."""
        state = self.project.state()
        if not state["clip"]:
            return None
        return {"length": state["length"], "layout": state["clip"]["layout"]["kind"],
                "moment": state["clip"]["moment"],
                "titles": [{"text": t["text"], "start": t["start"], "end": t["end"]} for t in state["titles"]],
                "people": len(state["people"]), "changes": len(self.project.entries)}

    def clip_line(self):
        """One line on the clip for Muse's state: what it is made of, or that none is made."""
        state = self.project.state()
        clip = state["clip"]
        if not clip:
            return "no clip made yet"
        a, b = clip["pieces"][0][0], clip["pieces"][-1][1]
        st = state["style"]["name"] if state["style"] else (self.clip or {}).get("style", "clean")
        words = "; ".join('"%s" %g-%g s at the %s' % (t["text"], t["start"], t["end"], t["place"])
                          for t in state["titles"])
        extras = [("lit captions on" if state["captions"].get("lit") else "popped captions on"
                   if state["captions"].get("pop") else "captions on") if state["captions"] else "no captions"]
        if state["blurs"]:
            extras.append("%d blur%s" % (len(state["blurs"]), "" if len(state["blurs"]) == 1 else "s"))
        if state["people"]:
            people = state["people"]
            left = sum(1 for p in people if p["blur"] is None and p["seen"] > 1)
            extras.append("other people: %d blurred, %d kept%s" % (
                sum(1 for p in people if p["blur"]), sum(1 for p in people if p["blur"] is False or
                                                         p["blur"] is None and p["seen"] == 1),
                ", %d still to decide before rendering" % left if left else ""))
        if state["look"]:
            extras.append("%s look" % state["look"])
        if state["pushes"]:
            extras.append("slow push " + ", ".join("%g-%g s" % (p["start"], p["end"]) for p in state["pushes"]))
        if state["hits"]:
            extras.append("hits: " + ", ".join("%s at %g s" % (h["kind"], h["at"]) for h in state["hits"]))
        if state["glass"]:
            extras.append("glass boxes: " + ", ".join("%s %g-%g s" % (g["tint"], g["start"], g["end"])
                                                      for g in state["glass"]))
        if state["stickers"]:
            extras.append("stickers: " + ", ".join("%s at %g s" % (k.get("name") or "a picture", k["at"])
                                                   for k in state["stickers"]))
        if state["sounds"]:
            extras.append("sounds: " + ", ".join("%s at %g s" % (s["kind"], s["at"]) for s in state["sounds"]))
        if state["music"]:
            extras.append("music %s" % os.path.basename(state["music"]["path"]))
        done = self.finished()
        if done:
            made = "finished file %s, made from the clip as it is now" % done["name"]
        elif self.rendered:
            made = ("the last finished file is older than the last change: render again before you hand it over, and "
                    "never say the clip is ready until then")
        else:
            made = ("no finished file yet: the person sees only a preview on the page, so never say the clip is "
                    "ready")
        return ("clip: moment %d (%s-%s, %g s), %s layout, %s style, %s, %s · %s"
                % (clip["moment"], recording.clock(a), recording.clock(b), state["length"], clip["layout"]["kind"],
                   st, "words on screen: " + words if words else "no words on screen", ", ".join(extras), made))

    def open(self, rec, job=None):
        """Opens rec as this chat's recording (one at a time) and shows it.
        Returns (the recording open before, the path of a picture from rec or None)."""
        with self.cond:
            had, self.recording = self.recording, rec
            if not had or had["path"] != rec["path"]:
                self.looks, self.moments, self.project, self.rendered = [], None, project.Project(), []
                self.checks = {}
            self.changed += 1
        frame = recording.frame(rec)
        self.emit("opened", job=job, recording=rec, frame=frame, line=recording.describe(rec))
        return had, (os.path.join(recording.WORK, "frames", frame) if frame else None)

    def open_dropped(self, rec, words, mind_key, language=""):
        """The person opened rec from the page: it opens, and Muse is told, with any words typed with it and one
        picture from it. The file name can mislead (the review clip is called trip_...), the picture does not.
        Until the person types, the browser's language is the only clue to the language they read."""
        _, picture = self.open(rec)
        opened, said = "(opened) " + recording.describe(rec), ("\n" + words if words else "")
        told = opened
        if language:
            told += "\nThe person's browser is set to %s; answer in that language until they write." % language
        return self.say(told + said, mind_key, [picture] if picture else (), shown=opened + said)


class Job:
    """One message from the person and everything Muse does about it, including words typed while it works."""

    def __init__(self, chat, brain, content):
        self.chat, self.brain, self.n = chat, brain, chat.jobs
        self.inbox = [content]
        self.pictures = []
        self.proofread = {}
        self.asking = None
        self.stopped = self.done = False
        self.plan, self.plan_shown = [], "[]"
        self.guesses = None
        self.steps = self.refused = self.fixed = self.guard_stops = 0
        self.seen = {}
        self.last_refused = {}
        self.refused_last = False
        self.shown_live = 0.0
        self.stop_work = None
        self.shown_extra = {}
        self.thread = threading.Thread(target=self.run, name="job-%d" % self.n, daemon=True)

    def emit(self, kind, **data):
        return self.chat.emit(kind, job=self.n, **data)

    def cut_in(self, content):
        """Called by Chat.say with the lock held: the words go in next, and the question in flight is cancelled."""
        self.inbox.append(content)
        if self.asking:
            self.asking.cancel()

    def stop(self):
        with self.chat.cond:
            if self.done:
                return False
            self.stopped = True
            asking, work = self.asking, self.stop_work
        if asking:
            asking.cancel()
        if work:
            work()
        return True

    def take_inbox(self):
        with self.chat.cond:
            said, self.inbox = self.inbox, []
        for content in said:
            self.chat.add({"role": "user", "content": content})
            self.chat.changed += 1

    def attach(self, picture, caption):
        """A tool's picture for Muse. Pictures never go in a tool result, so the loop sends them right after the
        results, as one message from the studio."""
        self.pictures.append((picture, caption))

    def show(self, **fields):
        """Fields the tool running now adds to its step on the page, such as the pictures of a look."""
        self.shown_extra.update(fields)

    def working(self, line):
        """Shows how far a tool that takes a while has got. Like Muse's thinking, it is shown, not logged."""
        with self.chat.cond:
            self.chat.live = {"job": self.n, "step": self.steps, "work": line}
            self.chat.live_rev += 1
            self.chat.cond.notify_all()

    def measure(self, rec):
        """Maps rec for a tool: the page shows how far it has got, and Stop ends ffmpeg at once."""
        total, shown = max(1.0, rec["seconds"]), [0.0]

        def progress(done):
            if time.time() - shown[0] >= SHOW_THINKING_EVERY:
                shown[0] = time.time()
                self.working("Mapping the video: %s of %s (%d%%)" % (recording.clock(min(done, total)),
                                                                     recording.clock(total), min(99, 100 * done / total)))

        self.working("Mapping the video: starting")
        try:
            return maps.measure(rec, on_progress=progress, on_start=self.started_work)
        finally:
            self.done_working()

    def hear(self, rec, start, end):
        """Hears start to end for a tool: the page says so, and Stop ends it at once (hearing.Stopped)."""
        self.working("Writing down what is said in %s-%s" % (recording.clock(start), recording.clock(end)))
        try:
            return hearing.hear(rec, start, end, on_start=self.started_work)
        finally:
            self.done_working()

    def find_faces(self, rec, start, end):
        """The faces in start-end for a tool: the page says so, and Stop ends it at once (faces.Stopped)."""
        self.working("Finding the faces in %s-%s" % (recording.clock(start), recording.clock(end)))
        try:
            return faces.sample(rec, start, end, on_start=self.started_work)
        finally:
            self.done_working()

    def edit(self, entry):
        return self.chat.edit(entry, job=self.n)

    def render(self, rec, state, out, default):
        """Renders the clip into out for a tool: the page shows how far it has got, and Stop ends FFmpeg at once."""
        total, shown = max(0.1, state["length"]), [0.0]

        def progress(done):
            if time.time() - shown[0] >= SHOW_THINKING_EVERY:
                shown[0] = time.time()
                self.working("Rendering the clip: %.0f of %.0f s (%d%%)" % (min(done, total), total,
                                                                              min(99, 100 * done / total)))

        self.working("Rendering the clip: starting")
        try:
            return render.export(rec, state, out, on_progress=progress, on_start=self.started_work,
                                 default=default)
        finally:
            self.done_working()

    def ask_aside(self, messages, schema, doing="Fixing misheard words from the context"):
        """Asks Muse apart from this job's conversation (the conversation and its cache stay as they are) and
        returns the words of the answer. Stop cancels it (hearing.Stopped)."""
        asking = mind.Asking()
        self.working(doing)
        self.started_work(asking.cancel)
        try:
            return self.brain.ask(messages, temperature=0.2, max_tokens=hearing.FIX_TOKENS, asking=asking,
                                  schema=schema)["text"]
        except mind.Cancelled:
            raise hearing.Stopped()
        finally:
            self.done_working()

    def started_work(self, stop):
        """A tool's work that takes a while has begun: Stop ends it with stop(), at once if Stop came first."""
        with self.chat.cond:
            self.stop_work, stopped = stop, self.stopped
        if stopped:
            stop()

    def done_working(self):
        with self.chat.cond:
            self.stop_work = None
            if self.chat.live and self.chat.live["job"] == self.n:
                self.chat.live = None

    def on_delta(self, kind, piece):
        chat = self.chat
        with chat.cond:
            if not chat.live or chat.live["job"] != self.n:
                return
            chat.live[kind] += piece
            if time.time() - self.shown_live >= SHOW_THINKING_EVERY:
                self.shown_live = time.time()
                chat.live_rev += 1
                chat.cond.notify_all()

    def run(self):
        chat, started, end, nudged = self.chat, time.time(), "answered", False
        self.emit("job", mind=self.brain.label)
        try:
            while True:
                self.take_inbox()
                if self.stopped:
                    end = "stopped"
                    break
                if self.steps >= MAX_STEPS:
                    end = "too many steps"
                    self.emit("note", text="Muse took %d steps on this message, the most one message may take, so the "
                                           "studio stopped it here. Say what to do next." % self.steps)
                    break
                with chat.cond:
                    if self.inbox or self.stopped:
                        continue
                    self.asking = mind.Asking()
                    chat.live = {"job": self.n, "step": self.steps + 1, "thinking": "", "text": ""}
                self.emit("wait", step=self.steps + 1)
                try:
                    got = self.ask()
                except mind.Cancelled:
                    self.emit("cut", step=self.steps + 1, by="stop" if self.stopped else "person")
                    continue
                finally:
                    with chat.cond:
                        chat.live = None
                calls = got["tool_calls"]
                got["text"] = words.plain(words.tidy(got["text"]))
                got["message"]["content"] = words.plain(words.tidy(got["message"]["content"]))
                for i, call in enumerate(calls):
                    call.setdefault("id", "call_%d_%d" % (self.steps + 1, i))
                    if isinstance(call.get("function"), dict):
                        call["function"]["arguments"] = words.tidy_arguments(call["function"].get("arguments"))
                if calls:
                    got["message"]["tool_calls"] = [_sendable(call) for call in calls]
                chat.add(got["message"])
                if not calls:
                    left = self.finish_plan()
                    if left and not nudged and not self.refused_last:
                        nudged = True
                        self.emit("answer", final=False, text=got["text"], thinking=got["thinking"],
                                  engine=got["engine"], mind=self.brain.label)
                        chat.add({"role": "user", "content": "[studio] Your plan has steps no call has named yet: %s. "
                                  "Do them now, or answer the person and say why they are not needed."
                                  % "; ".join("%d. %s" % s for s in left)})
                        self.emit("note", text="Muse stopped with plan steps still open, so the studio asked it to go "
                                               "on or say why.")
                        continue
                    with chat.cond:
                        more = bool(self.inbox) and not self.stopped
                        self.emit("answer", final=not more, text=got["text"], thinking=got["thinking"],
                                  engine=got["engine"], mind=self.brain.label)
                        if not more:
                            self.close(end, started)
                            return
                    continue
                for i, call in enumerate(calls):
                    if self.stopped:
                        chat.add({"role": "tool", "tool_call_id": call["id"],
                                  "content": "Not run: the person pressed Stop.\n\n" + self.state()})
                        continue
                    self.steps += 1
                    shown, text = self.call(call)
                    chat.add({"role": "tool", "tool_call_id": call["id"], "content": text + "\n\n" + self.state()})
                    first = i == 0
                    self.emit("step", step=self.steps, said=got["text"] if first else "",
                              thinking=got["thinking"] if first else "", engine=got["engine"] if first else None,
                              **shown)
                    self.show_plan()
                    if self.guesses:
                        self.emit("guesses", **self.guesses)
                        self.guesses = None
                if self.pictures:
                    pictures, self.pictures = self.pictures, []
                    chat.add({"role": "user", "content": _studio_pictures(pictures)})
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
        with chat.cond:
            self.close(end, started)

    def ask(self):
        """Muse's next answer. When the conversation and an answer no longer fit in Muse's memory (mind.TooLong),
        the oldest pictures are taken out of it (make_room) and Muse is asked again, so a job is not ended by its own
        length. Measured 10-28: a montage of 19 clips filled Glimmer's 32,768 tokens 20 tokens before its hand-over,
        and a screen recording filled them while Muse wrote a call, which was cut off and could not be read."""
        tries = 0
        while True:
            try:
                return self.brain.ask(self.chat.messages, tools=SCHEMAS, temperature=AGENT_TEMPERATURE,
                                      max_tokens=AGENT_MAX_TOKENS, on_delta=self.on_delta, asking=self.asking)
            except mind.TooLong as e:
                tries += 1
                with self.chat.cond:
                    if self.stopped:
                        raise mind.Cancelled()
                    took = make_room(self.chat.messages, e.asked + REPLY_ROOM - e.room) if tries <= ROOM_TRIES else 0
                    if not took:
                        raise mind.MindError(
                            "%s's memory is full: this chat holds more than it can read at once (%d tokens), and no "
                            "picture is left to take out. Start a new chat for the next clip." % (self.brain.label,
                                                                                                 e.room))
                    self.asking = mind.Asking()
                self.emit("note", text="%s's memory was nearly full, so the studio took out the %d oldest picture%s "
                                       "it had already looked at, and asked again." % (
                                           self.brain.label, took, "" if took == 1 else "s"))

    def close(self, end, started):
        """Ends the job (the caller holds the lock). Words typed too late to be answered still go into the chat."""
        self.done = True
        for w in self.inbox:
            self.chat.add({"role": "user", "content": w})
        self.inbox = []
        self.emit("end", steps=self.steps, refused=self.refused, fixed=self.fixed,
                  seconds=round(time.time() - started, 1), why=end)
        if end == "answered":
            self.speller = threading.Thread(target=self.spell_shown, name="spell-%d" % self.n, daemon=True)
            self.speller.start()

    def spell_shown(self):
        """After the job, the reasons, moment cards and answers it showed, with their Thai spelling fixed in one question of their
        own (words.spelled), as a "spelled" event the page redraws them from. Glimmer answers one question at a time,
        so the person's next message cancels it: it never keeps Muse waiting. What Muse reads back is not changed."""
        spots, texts, answers = [], [], 0
        for e in [e for e in self.chat.events if e.get("job") == self.n]:
            if e["kind"] == "step":
                for field in ("why", "said"):
                    if e.get(field):
                        spots.append({"kind": "step", "step": e["step"], "field": field})
                        texts.append(e[field])
                for i, m in enumerate(e.get("moments") or []):
                    if m.get("reason"):
                        spots.append({"kind": "moment", "step": e["step"], "index": i})
                        texts.append(m["reason"])
            elif e["kind"] == "answer":
                if e.get("text"):
                    spots.append({"kind": "answer", "index": answers})
                    texts.append(e["text"])
                answers += 1
        if not any(words.odd_thai(t) for t in texts):
            return
        asking = mind.Asking()
        with self.chat.cond:
            if self.chat.job is not self:
                return
            self.chat.spelling = asking
        try:
            fixed, _ = words.spelled(lambda m, s: self.brain.ask(m, temperature=0.2, max_tokens=SPELL_TOKENS,
                                                                 asking=asking, schema=s)["text"], texts)
        except (mind.Cancelled, mind.MindError):
            return
        finally:
            with self.chat.cond:
                if self.chat.spelling is asking:
                    self.chat.spelling = None
        fixes = [dict(spot, text=after) for spot, before, after in zip(spots, texts, fixed) if after != before]
        if fixes:
            self.emit("spelled", fixes=fixes)

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
        self.shown_extra = {}
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
        shown.update(self.shown_extra)
        if self.plan and not step and name != "set_plan":
            text += ("\n(This call named no plan_step, so the plan on the page did not move. Give every call the plan "
                     "step it works on.)")
        if self.last_refused.get(name):
            self.fixed += 1
        self.last_refused[name] = False
        self.refused_last = False
        shown.update(result=text, refused=False)
        return shown, text

    def refuse(self, shown, name, sentence):
        self.refused += 1
        self.last_refused[name] = True
        self.refused_last = True
        shown.update(result=sentence, refused=True)
        return shown, sentence

    def set_clip(self, clip, chips):
        """Muse's recipe and guesses for the clip. They belong to the chat, and the page shows them after the step."""
        self.chat.clip = clip
        self.guesses = {"clip": clip, "chips": chips}

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
        """Muse answered in words: the step being worked on is done, and so is a last step that only the answer
        could do ("compare them", "tell the person"). Returns the steps no call has named: [(number, text)]."""
        for item in self.plan:
            if item["state"] == "now":
                item["state"] = "done"
        if self.plan and all(item["state"] == "done" for item in self.plan[:-1]):
            self.plan[-1]["state"] = "done"
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
        clip = self.chat.clip
        if clip:
            parts.append("recipe %s: about %d s, %s style%s" % (clip["recipe"], clip["length_s"], clip["style"],
                                                              ", " + clip["language"] if clip["language"] else ""))
        else:
            parts.append("no recipe yet")
        rec = self.chat.recording
        parts.append("open: %s (%s long)" % (rec["name"], recording.clock(rec["seconds"])) if rec
                     else "no recording open")
        if self.chat.looks:
            parts.append("%d look%s made" % (len(self.chat.looks), "" if len(self.chat.looks) == 1 else "s"))
        if self.chat.moments:
            parts.append("moments proposed: " + ", ".join(
                "%d. %s-%s" % (m["n"], recording.clock(m["start"]), recording.clock(m["end"]))
                for m in self.chat.moments))
        parts.append(self.chat.clip_line())
        return " · ".join(parts)


class Hand(Job):
    """The page's hand: a tap runs the very tool Muse calls, so the page gets the tool's checks, its refusals and its
    project edit, and Muse is never asked (a spelling question gets no answer, so the words stay as typed). What a
    tool does after its edit is for Muse (a frame to look at, a warning to pass on), so the hand stops the tool at
    its edit (Made). A trial hand stops it there before the edit (Allowed) and changes nothing: that is how the page
    knows a choice the tool would refuse, and why."""

    def __init__(self, chat, trial=False):
        self.chat, self.trial, self.n, self.steps = chat, trial, None, 0
        self.proofread, self.pictures, self.shown_extra = {}, [], {}
        self.asking = self.stop_work = self.entry = None
        self.stopped = self.done = False

    def edit(self, entry):
        if self.trial:
            raise Allowed()
        self.entry = entry
        self.chat.edit(entry)
        raise Made()

    def ask_aside(self, messages, schema, doing=""):
        raise mind.MindError("The page does not ask Muse.")

    def render(self, rec, state, out, default):
        if self.trial:
            raise Allowed()
        return Job.render(self, rec, state, out, default)


def _first_sentence(said):
    """The first sentence of a tool's answer, for the page: the rest tells Muse what to do next."""
    return SENTENCE_END.split(said.strip(), 1)[0].split("\n", 1)[0]


def _grid_box(box):
    """[x1, y1, x2, y2] on the 0-1000 grid of the clip's frame, from [x, y, w, h] in its pixels: add_title takes a
    box this way and gives back the same pixels."""
    x, y, w, h = box
    return [x * 1000 / 1080, y * 1000 / 1920, (x + w) * 1000 / 1080, (y + h) * 1000 / 1920]


def make_room(messages, need):
    """Takes pictures out of the conversation, oldest first, until about need tokens are free, and returns how many
    it took; each leaves TAKEN_OUT in its place. A picture is counted as PICTURE_TOKENS, fewer than one takes
    (measured 10-28: 280 to 360), so enough come out at once. The newest message with pictures keeps them: Muse may
    not have looked at them yet."""
    holding = [m for m in messages if isinstance(m.get("content"), list) and
               any(p.get("type") == "image_url" for p in m["content"])]
    took = 0
    for m in holding[:-1]:
        parts = []
        for p in m["content"]:
            if p.get("type") == "image_url" and took * PICTURE_TOKENS < need:
                parts.append({"type": "text", "text": TAKEN_OUT})
                took += 1
            else:
                parts.append(p)
        m["content"] = parts
        if took * PICTURE_TOKENS >= need:
            break
    return took


def _sendable(call):
    """The call as it goes back into the conversation: arguments that are not readable JSON become {}, as
    llama-server reads every earlier call again and answers 500 to one it cannot read. The call itself is still
    refused with what was wrong."""
    fn = call.get("function")
    if not isinstance(fn, dict):
        return call
    try:
        json.loads(fn.get("arguments") or "{}")
        return call
    except (TypeError, ValueError):
        return dict(call, function=dict(fn, arguments="{}"))


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


def _with_pictures(words, pictures):
    """A user message's content: the words alone, or the words and each picture (a jpg path) inline."""
    if not pictures:
        return words
    return [{"type": "text", "text": words}] + [_picture(path) for path in pictures]


def _studio_pictures(pictures):
    """The studio's message with the pictures tools made, each right after its caption, so Muse reads which
    picture is which (one picture holding several frames broke Muse's 0-1000 grid)."""
    content = []
    for i, (path, caption) in enumerate(pictures):
        content += [{"type": "text", "text": ("[studio] " if i == 0 else "") + caption}, _picture(path)]
    return content


def _picture(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}


def _append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


CHATS = {}
_chats_lock = threading.Lock()
_shown = []


def _show(chat):
    """Makes chat the one every page shows. A job still running in the chat left behind is stopped, and the pages
    still showing that chat are woken so they follow (the server says which chat is open; an address cannot)."""
    left = CHATS.get(_shown[-1]) if _shown and _shown[-1] != chat.id else None
    CHATS[chat.id] = chat
    _shown[:] = [chat.id]
    if left:
        left.stop()
        with left.cond:
            left.cond.notify_all()
    return chat


def shown():
    """The id of the chat every page should show."""
    return _shown[-1] if _shown else None


def current():
    """The chat the page shows, made on first use."""
    with _chats_lock:
        return CHATS[_shown[-1]] if _shown else _show(Chat())


def new():
    with _chats_lock:
        return _show(Chat())


def open_chat(chat_id):
    """An earlier chat, from memory or from work/chats; None when there is no such chat."""
    if not CHAT_ID.match(chat_id or ""):
        return None
    with _chats_lock:
        chat = CHATS.get(chat_id)
        if chat is None:
            if not os.path.isfile(os.path.join(WORK, "chats", chat_id + ".messages.jsonl")):
                return None
            chat = Chat.load(chat_id)
        return _show(chat)


def find(chat_id):
    return CHATS.get(chat_id)


def earlier(limit=CHATS_LISTED):
    """The newest chats on disk that have at least one message: [{id, when, first, messages}]."""
    folder = os.path.join(WORK, "chats")
    try:
        names = [n for n in os.listdir(folder) if n.endswith(".events.jsonl") and CHAT_ID.match(n[:-13])]
    except OSError:
        return []
    names.sort(key=lambda n: os.path.getmtime(os.path.join(folder, n)), reverse=True)
    out = []
    for name in names:
        events = _read(os.path.join(folder, name))
        said = [e["text"] for e in events if e["kind"] == "person"]
        opened = [e["recording"]["name"] for e in events if e["kind"] == "opened"]
        if said:
            out.append({"id": name[:-13], "when": round(os.path.getmtime(os.path.join(folder, name))),
                        "first": said[0][:100], "messages": len(said), "recording": opened[-1] if opened else ""})
        if len(out) >= limit:
            break
    return out
