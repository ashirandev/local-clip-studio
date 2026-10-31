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
- A message that only talks starts no work: when Muse's first answer to a typed message calls a tool, one short
  question apart (needs.py) asks whether the message asked for work, and if it did not, Muse is asked again with no
  tools at all, so it answers in words and nothing runs.
- The loop ends the job, not Muse: when Muse answers in words (after one reminder of any plan steps it left open),
  when it keeps repeating a refused call, or after MAX_STEPS steps. A job whose last call handed the clip over is
  done, so at MAX_STEPS Muse is asked once more, with no tools, for its closing words (SIFU on Spark, 10-29: the
  hand-over was step 30, and the page said the studio had stopped it). STEPS_WARNED steps before the end of a job with
  nothing handed over, Muse is told once to finish with what it has: on lane 5's rounds of 10-30, four jobs reached
  step 30 with no clip, one of them with a clip rendered. A job at MAX_STEPS with a clip the person has not got then
  gets FINISH_STEPS more, with only what it takes to give it: check_clip and hand_over, and render_clip while no file
  is made from the clip as it is, after decide_people if anyone is undecided, as nothing undecided is rendered. A check
  that still fails goes over with the clip and is said on the page. 5 of 91 runs of 10-30 and 31, 4 of them Glimmer's,
  ended at step 30 with their file made from the clip as it was, and 2 of the 5 had just passed their checks; 3 of 21
  Glimmer runs of 10-31 ended there with no such file, a clip never rendered, one changed after its render and one
  made again after it (lane 5). A clip already handed over as it is gives no more steps. The steps are for the render,
  the check, the hand-over and one refusal: of 122 hand-overs, 118 came within 4 steps of their render, most with the
  cover in them, and no cover is made on these steps.
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

from . import (asr, channels, chatname, countdown, faces, hearing, layer, layouts, maps, mind, montage, needs, previews,
               project, recording, render, spent, text, words)
from .tools import (CAPTION_LETTERS, LONGEST, LOUD_CUT, MOVED_CUTS, ONE_LETTERS, LOOK_LABELS, LOOKS, SCHEMAS, SHORTEST,
                    STYLES, TOOLS, UNASKED, ALBUM_PARTS, ALBUM_TOOLS, Refused, _changed_clips, _circle_box, _spoken, _subtitles,
                    _undecided, _waits, album_entries, channel_entries, channel_unfit, describe, faces_at,
                    features, look_of, look_said)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("LCS_WORK") or os.path.join(ROOT, "work")
AGENT_TEMPERATURE = 0.5
AGENT_MAX_TOKENS = 4096
SPELL_TOKENS = 8192
MAX_STEPS = 30
STEPS_WARNED = 6
FINISH_STEPS = 4
FINISH_TOOLS = ("check_clip", "hand_over")
GUARD_STOPS = 3
REPLY_ROOM = 2048
PICTURE_TOKENS = 256
ROOM_TRIES = 4
ROOM_FULL = 0.85
ROOM_AFTER = 0.7
# Once no picture is left, long old results are cut, down to RESULT_HEAD characters: 10-31 130454-90c1 filled Glimmer's 49,152
# tokens at step 25 with its reel joined and no picture left, 58,000 of its 63,000 characters tool results, two recipes among
# them. The studio keeps what they said (the looks, the moments), and each tool gives it again. The newest RESULT_KEPT
# messages stay whole, and the newest recipe, which the job still follows. English runs about CHARS_TOKEN characters a token
# and Thai fewer, so the count errs on cutting enough.
LONG_RESULT, RESULT_HEAD, RESULT_KEPT, CHARS_TOKEN = 1500, 200, 12, 4
RECIPE_SAID = "Recipe loaded:"
ROOMS = {}
TAKEN_OUT = ("(A picture was here. It was taken out to make room in Muse's memory: the tool that showed it shows it "
             "again.)")
SHOW_THINKING_EVERY = 0.4
CHATS_LISTED = 20
CHAT_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
TAPPED = "the person tapped it on the page"
CAPTION_CHOICES = {"off": {"on": False}, "plain": {"pop": False}, "lit": {"lit": True}, "pop": {"pop": True}}
TALK_KINDS = ("talk", "speaker", "split")
LAYOUT_LABELS = {"card": "Card", "stack": "Stack", "face": "Face", "subject": "Fill", "talk": "Talk",
                 "speaker": "Speaker", "split": "Split"}
COLOUR_LOOKS = ("none",) + tuple(n for n in LOOKS if n != "none")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(])")
PREVIEW_WAIT = 1.0
TAP_SAID = "(tapped) "
COVER_SAID = re.compile(r"(?i)\bcovers?\b|หน้าปก|ภาพปก|ปกพร้อม|ปกคลิป|ปกให้|ปกของ|ปกสวย")
NO_COVER = ("[studio] Your answer speaks of the clip's cover, but no cover is made for the finished clip. Answer the "
            "person again without saying it has one; if they want one, they can tap Make the cover or ask for it.")
LANDED = (MOVED_CUTS[:MOVED_CUTS.index("%")], LOUD_CUT[:LOUD_CUT.index("%")])
LAST_STEPS = ("[studio] %d steps are left on this message and nothing is handed over yet. Finish with what you have "
              "now: make the clip if there is none, then render it, check it and hand it over. A rough clip the person "
              "can fix is better than none.")
FINISH = ("[studio] No steps are left for anything else on this message, and the clip is not handed over yet. You have "
          "%d more, for %s only, in that order, leaving out what is done. Hand the clip over with its title, post and "
          "hashtags. A check that still fails goes over with it, and the page says which.")
FINISH_ONLY = "Only %s are left on this message: the person gets the clip as it is now."
FINISH_CALL = "[studio] Words alone do not give the person the clip. Call the next of %s now."
CUT_OUT = (" (The rest of this result was cut to make room in Muse's memory: the studio still has it, and the tool that gave "
           "it gives it again.)")
OPEN_SAID = "(opened) "
AS_SAID = "said"
WORDS_GONE = "Those words are not on the clip as the page showed them. Look at the clip again."
SCRIPTS = (("Thai", re.compile("[฀-๿]")), ("Japanese", re.compile("[぀-ヿ]")),
           ("Korean", re.compile("[가-힯]")), ("Chinese", re.compile("[一-鿿]")),
           ("Arabic", re.compile("[؀-ۿ]")), ("Hindi", re.compile("[ऀ-ॿ]")))
VOICE_NAMES = {"Chinese": "Mandarin Chinese"}
SYSTEM = """You are the editor in Local Clip Studio. People bring long recordings (selling lives, game streams, travel \
clips) and you turn them into short vertical clips ready to post as Reels on Instagram and Facebook. When you say \
where a clip goes, name those; never name another company's app.

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
- When the person only talks (a greeting, thanks, how they feel, small talk, or a question you can answer from \
what you already know), answer like a person, briefly and warmly, with no tool and no plan. If they may want \
something done, ask what; do not start it.
- Under a finished clip the page has a Premiere project button: the clip as a project an editor finishes in Premiere \
Pro, each piece still part of the person's own file. When the person wants to edit further in their own editor, tell \
them to press it.
- A message that starts with "(tapped)" is a choice the person made with a button under your guesses. Act on it \
like their own words, and keep answering in the language they type in.
- A message that starts with "(opened)" means the person opened that recording to make a clip from it. Start at \
once: pick its kind, then plan and work. Do not ask anything first; the person can correct any guess with a tap. \
Judge the kind by the picture that comes with it and by the person's words, not by the file name."""


def _placed(lines, new):
    """The caption lines with a new phrase's lines among them by time, every phrase numbered again in that order."""
    phrases = [[ln for ln in lines if ln["phrase"] == k] for k in sorted({ln["phrase"] for ln in lines})] + [new]
    phrases.sort(key=lambda mine: mine[0]["from"])
    return [dict(ln, phrase=k) for k, mine in enumerate(phrases) for ln in mine]


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
        self.clips = self.counted = None
        self.project = project.Project()
        self.album, self.album_at = [], 0
        self.rendered = []
        self.checks = {}
        self.told = []
        self.held = 0
        self.spelling = None
        self.making = self.basing = None
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
                    chat.album, chat.album_at = [], 0
                path = e["recording"]["path"]
            elif e["kind"] == "batch":
                chat.hold_open_clip()
            elif e["kind"] == "album":
                chat.put_in_album([dict(c, project=project.Project(c["entries"], c["acts"])) for c in e["clips"]],
                                  e["since"])
            elif e["kind"] == "picked":
                chat.to_album_place(e["at"], e.get("new"))
            elif e["kind"] == "matched":
                for change in e["changes"]:
                    for entry in change["entries"]:
                        chat.album[change["at"]]["project"].add(entry, e["act"])
            elif e["kind"] == "unmatched":
                for at in e["back"]:
                    chat.album[at]["project"].undo()
            elif e["kind"] == "edit":
                chat.project.add(e["entry"], e.get("act"))
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
            elif e["kind"] == "ends":
                chat.moments = e["moments"]
                if e.get("look"):
                    chat.looks.append(e["look"])
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
        if not shown.startswith((TAP_SAID, OPEN_SAID)):
            words += written_in(shown)
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
            typed = None if shown.startswith((TAP_SAID, OPEN_SAID)) else shown
            self.job = Job(self, mind.get(mind_key), content, said=typed)
            self.emit("person", job=self.jobs, text=shown)
            self.job.thread.start()
            return self.job

    def stop(self):
        """Stops the job running now and what the page is making (the video, the preview), at once. True when any
        of them was running."""
        job, making, basing = self.job, self.making, self.basing
        stopped = bool(job and job.stop())
        stopped = bool(making and making.stop()) or stopped
        return bool(basing and basing.stop()) or stopped

    def unfit(self, *entries):
        """Why the clip's channel name could not be drawn on it after entries (tools.channel_unfit): a sentence, or
        None."""
        after = project.Project(self.project.entries + list(entries)).state()
        return channel_unfit(after, self.clip["style"] if self.clip else "clean")

    def refusal(self, entry):
        """Why entry may not go on the clip, or None: a style or channel entry that would leave a channel name the
        clip cannot draw, so a clip's render never fails on it."""
        if entry.get("do") not in ("channel", "style") or self.unfit():
            return None
        unfit = self.unfit(entry)
        return unfit and "The channel's name would not fit on the clip then: %s" % unfit

    def edit(self, entry, job=None, act=None):
        """Adds one change to the clip's project list; the page redraws the clip. Returns the clip's state. Refused
        when self.refusal(entry) says why not. act names the action it is part of (one tool call, one tap), so Undo
        takes the action back whole (Project.undo)."""
        with self.cond:
            said = self.refusal(entry)
            if said:
                raise Refused(said)
            self.project.add(entry, act)
            self.changed += 1
        self.emit("edit", job=job, entry=entry, act=act, clip=self.clip_view(), said=describe(entry))
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
        except Exception as e:
            said = "The studio broke while trying that (%s)." % e
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

        captions = choices("captions", CAPTION_CHOICES, str.capitalize, "add_captions", CAPTION_CHOICES.get)
        return {"clip": True, "length": state["length"], "captions": captions,
                "eye": {"has": bool(state["captions"] or state["hidden"]), "hidden": bool(state["hidden"])},
                "languages": self.languages(state, next(c["why"] for c in captions if c["name"] == "plain")),
                "styles": choices("style", STYLES, lambda n: LOOK_LABELS.get(n, n.capitalize()), "set_style",
                                  lambda n: {"style": n}),
                "looks": choices("look", COLOUR_LOOKS, str.capitalize, "set_look", lambda n: {"look": n}),
                "words": self.words_on(state), "again": bool(self.rendered) and self.finished() is None,
                "making": self.making is not None, "channels": self.channels_now(state),
                "sound": {"mutes": state["mutes"], "has": self.has_sound()}, "ends": self.ends(state),
                "frame": self.framing(state), "layouts": self.layouts_now(state), "album": self.album_now(),
                "circle": self.circle_now(state)}

    def layouts_now(self, state):
        """The layouts the Layout tab offers this clip, [{"name", "label", "now", "why", "picture"}] (picture: the
        page may draw the clip in it with /api/clipframe's layout), or None for clips joined from several videos,
        whose layout Muse chooses. A recording lying down takes every layout of layouts.KINDS, one standing up only
        the face and the subject (a card or a stack would show its face twice); people talking take the talk
        layouts too. A layout the recording cannot take is refused when tapped, with make_clip's own reason."""
        clip = state["clip"]
        if self.clips or not self.recording or clip.get("moment") is None:
            return None
        kind, (w, h) = clip["layout"]["kind"], clip["layout"]["size"]
        names = list(layouts.KINDS) if w > h else ["face", "subject"]
        if kind in TALK_KINDS or (self.clip or {}).get("recipe") == "talk":
            names += list(TALK_KINDS)
        return [{"name": n, "label": LAYOUT_LABELS[n], "now": n == kind, "why": None,
                 "picture": self._shape(state, n) is not None} for n in names]

    def layout_for(self, name):
        """The clip's layout drawn as name would draw it, for the picture of name in the Layout tab, or None. It is
        made from what the clip has already found (the face), so it costs no looking; the tap makes the clip again
        and finds more (the action a game's card follows, the product)."""
        return self._shape(self.project.state(), name)

    def _shape(self, state, name):
        clip = state["clip"]
        if not clip:
            return None
        lay = clip["layout"]
        if name == lay["kind"]:
            return lay
        if name not in layouts.KINDS:
            return None
        try:
            return layouts.make(name, lay["size"], lay.get("face"), None,
                                tall=(self.clip or {}).get("recipe") == "game")
        except ValueError:
            return None

    def set_layout(self, name):
        """The person's tap on a layout in the Layout tab: the clip made again from its moment in that layout, by
        make_clip's own work run in the background as the Trim tab's is, without asking Muse, as a moved clip: what
        sits on the footage stays on the same seconds of the recording when the pieces come out otherwise (a game's
        menus cut since it was made); one Undo takes it back, and Muse is told. Tarm 10-30: "ให้คนเลือก layout เองง่ายที่สุด ดูเหมือนเราจะทำให้มัน
        อัตโนมัติ 100% ไม่ได้". Returns None once it has started, or the sentence why not."""
        state = self._clip_now()
        offered = self.layouts_now(state)
        if offered is None:
            return "A clip joined from several videos keeps the layout Muse chose: ask Muse to change it."
        if name not in [c["name"] for c in offered]:
            return "Choose one of the layouts: %s." % ", ".join(c["label"] for c in offered)
        if name == state["clip"]["layout"]["kind"]:
            return None
        hand = Hand(self, page="layout", whole=True)
        with self.cond:
            if self.making:
                return "The video is being made now. Change the layout when it is done."
            if self.busy():
                return "Muse is working on the clip now. Wait until it is done, or press Stop."
            self.making = hand
        threading.Thread(target=self._relayout, args=(hand, name, state["clip"]), name="change-layout",
                         daemon=True).start()
        return None

    def _relayout(self, hand, name, clip):
        made = clip.get("made") or {}
        args = dict({k: made[k] for k in ("face", "product") if made.get(k)}, moment=clip["moment"], layout=name)
        if made.get("follow") is False:
            args["follow"] = False
        before = sum(e["do"] == "clip" for e in self.project.entries)
        try:
            said = TOOLS["make_clip"].run(hand, TAPPED, moved=True, **args)
            said = None if sum(e["do"] == "clip" for e in self.project.entries) > before else said
        except Refused as e:
            said = str(e)
        except (hearing.Stopped, faces.Stopped):
            said = "Stopped."
        except Exception as e:
            said = "The studio broke while making the clip in the %s layout (%s)." % (LAYOUT_LABELS[name], e)
        finally:
            hand.done_working()
            with self.cond:
                self.making, hand.done = None, True
        if said:
            self.emit("not_relaid", error=_first_sentence(said))
            return
        self._tell("the person chose the %s layout on the page: the clip was made again in it from moment %d, its "
                   "words and effects kept" % (name, clip["moment"]))
        self.emit("relaid", clip=self.clip_view())

    def channels_now(self, state):
        """The channel looks the page offers: each saved look with what it puts on a clip, the one in use, and the
        name drawn on this clip with its spot; a spot the name does not fit in says why."""
        mine = state["channel"]
        return {"saved": [{"name": c["name"], "handle": c["handle"], "says": look_said(c, muse=False)}
                          for c in channels.listed()],
                "in_use": (channels.in_use() or {}).get("name"), "handle": mine["name"] if mine else "",
                "spots": [{"name": s, "label": s.capitalize(),
                           "now": bool(mine) and (mine.get("spot") or channels.SPOTS[0]) == s,
                           "why": _first_sentence(self.unfit(dict(mine, spot=s)) or "") or None if mine else None}
                          for s in channels.SPOTS]}

    def languages(self, state, refused):
        """The captions' language choices: as said, which the page puts on itself (add_captions in the language
        the words are said in needs no translation), and English, which only Muse can write: its tap is a message
        to Muse, as a chip under Muse's guesses is. The one marked now is the captions' language (none: as said);
        a language Muse put on that is neither is shown too, so the page says what the clip has. refused is why
        Plain captions would be refused, which holds for every language."""
        now = (state["captions"] or {}).get("language")
        said = refused
        if not said and now:
            try:
                _subtitles(self._said_in(state))
            except Refused as e:
                said = _first_sentence(str(e))
        out = [{"name": AS_SAID, "label": "As said", "now": not now, "why": None if not now else said}]
        for name in ["English"] + ([now] if now and now != "English" else []):
            out.append({"name": name, "label": name, "now": name == now, "why": None if name == now else refused,
                        "say": TAP_SAID + "Put the captions in %s." % name})
        return out

    def words_on(self, state):
        """The words on the clip as blocks the page places by their time: every title, and every phrase of the
        captions (its lines joined give it back whole), "typed" when the person wrote it on the page, and every
        draft: a phrase the hearing was not sure of, not on the video until the person keeps or fixes it."""
        out = [{"kind": "title", "n": k, "text": t["text"], "start": t["start"], "end": t["end"]}
               for k, t in enumerate(state["titles"])]
        for kind, key in (("caption", "lines"), ("draft", "drafts")):
            lines = (state["captions"] or {}).get(key) or []
            for k in sorted({ln["phrase"] for ln in lines}):
                mine = [ln for ln in lines if ln["phrase"] == k]
                out.append({"kind": kind, "n": k, "text": "".join(ln["text"] for ln in mine).strip(),
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

    def set_eye(self, hidden):
        """The person's tap on the eye over the clip: every caption hidden, or shown again as it was (the lines they
        typed too), as one change Undo takes back, and told to Muse before its next job. Returns None, or the sentence
        why not."""
        state = self._clip_now()
        if bool(state["hidden"]) == hidden:
            return None
        refused, entry = self.tap("add_captions", hidden=hidden)
        if not entry:
            return refused
        self._tell("the person %s with the eye on the page" % (
            "hid every caption (kept as they are, off the video until they show them again)" if hidden else
            "showed the captions again"))
        return None

    def set_style(self, name):
        """The person's tap on one of the house styles of the words: set_style's own edit, as set_captions."""
        self._clip_now()
        if name not in STYLES:
            return "Choose one of the text styles: %s." % ", ".join(LOOK_LABELS.get(n, n.capitalize()) for n in STYLES)
        refused, entry = self.tap("set_style", style=name)
        if entry:
            self._tell("the person chose the %s style on the page" % name)
        return refused

    def use_channel(self, name):
        """The person's tap on a saved channel look, or on None: the look goes on the clip at once, every part of it,
        and on every new clip; None stops it for new clips and takes the drawn name off this one. Told to Muse
        before its next job. Returns None, or the sentence why not."""
        state = self._clip_now()
        look = channels.get(name) if name else None
        off = [{"do": "channel", "name": "", "spot": None}] if state["channel"] else []
        entries = channel_entries(look, [], every=True) if look else off
        unfit = self.unfit(*entries)
        if unfit:
            return "That look's name does not fit on this clip: %s" % unfit
        try:
            channels.use(name or None)
        except ValueError as e:
            return str(e)
        for entry in entries:
            self.edit(entry)
        self._tell('the person put the "%s" channel look on the clip on the page' % name if name else
                   "the person stopped using a channel look on the page")
        return None

    def save_channel(self, name, handle):
        """The person's tap on Save under their name for the channel: the clip's look is kept as that channel's
        look and is in use from now on; the name to draw goes on this clip. Returns None, or the sentence why not."""
        state = self._clip_now()
        drawn = " ".join((handle or "").split())
        unfit = drawn and self.unfit({"do": "channel", "name": drawn, "spot": (state["channel"] or {}).get("spot")})
        if unfit:
            return "Nothing was saved: %s" % unfit
        try:
            look = channels.save(look_of(state, (self.clip or {}).get("style", "clean"), name, handle, self.clip))
            channels.use(look["name"])
        except ValueError as e:
            return str(e)
        if (state["channel"] or {}).get("name", "") != look["handle"]:
            self.edit({"do": "channel", "name": look["handle"], "spot": look["spot"]})
        self._tell("the person saved the clip's look as the \"%s\" channel look on the page: %s" % (
            look["name"], look_said(look)))
        return None

    def set_tagline(self, words, spot):
        """The person's tag line typed in the Album tab: set_tagline's own edit, on the clip open now. Returns None, or
        the sentence why not."""
        self._clip_now()
        refused, entry = self.tap("set_tagline", **dict({"words": words}, **({"spot": spot} if spot else {})))
        if entry:
            self._tell("the person set the tagline of the clip to \"%s\" on the page" % entry["name"])
        return refused

    def circle_now(self, state):
        """The corners the Layout tab offers the clip's face circle, [{"name", "label", "now"}], or None with no
        circle."""
        box = _circle_box(state)
        lay = state["clip"]["layout"]
        return [{"name": c, "label": c.capitalize(), "now": layouts.corner(lay, c) == box}
                for c in layouts.CORNERS] if box else None

    def set_circle(self, name):
        """The person's tap on a corner for the face circle: the same edit as move_part's, at the corner's exact box.
        Returns None, or the sentence why not."""
        state = self._clip_now()
        if name not in layouts.CORNERS:
            return "Choose a corner: %s." % ", ".join(layouts.CORNERS)
        if not _circle_box(state):
            return "This clip has no face circle."
        box = layouts.corner(state["clip"]["layout"], name)
        self.edit({"do": "layout", "part": "circle", "box": box, "layout": layouts.moved(state["clip"]["layout"], "circle", box)})
        self._tell("the person moved the face circle to the %s on the page" % name)
        return None

    def set_channel_spot(self, spot):
        """The person's tap on where the channel's name sits on the clip. Returns None, or the sentence why not."""
        state = self._clip_now()
        if spot not in channels.SPOTS:
            return "Choose where the name goes: %s." % ", ".join(channels.SPOTS)
        if not state["channel"]:
            return "Save the clip's look with a name to draw on it first."
        try:
            self.edit({"do": "channel", "name": state["channel"]["name"], "spot": spot})
        except Refused as e:
            return str(e)
        look = channels.in_use()
        if look and look["handle"] == state["channel"]["name"]:
            channels.save(dict(look, spot=spot))
        return None

    def ends(self, state):
        """Where the clip starts and ends in the recording, for the Trim tab: {"start", "end", "pieces", "recording",
        "shortest", "longest", "why"}, why the sentence when its ends cannot be moved, else None."""
        clip, rec = state["clip"], self.recording or {}
        n, moments = clip["moment"], self.moments or []
        why = None if clip.get("made") and 1 <= n <= len(moments) else (
            "Ask Muse to make the clip again first: this clip was made before its ends could be moved.")
        return {"start": clip["pieces"][0][0], "end": clip["pieces"][-1][1], "pieces": clip["pieces"],
                "recording": rec.get("seconds"), "shortest": SHORTEST, "longest": LONGEST, "why": why}

    def set_ends(self, start, end):
        """The person's new start and end of the clip in the recording, from the Trim tab: change_ends's own work,
        run in the background as a render is, without asking Muse (the new words it hears stay as heard). Returns None
        once it has started, or the sentence why not."""
        state = self._clip_now()
        why = self.ends(state)["why"]
        if why:
            return why
        hand = Hand(self, page="ends", whole=True)
        with self.cond:
            if self.making:
                return "The video is being made now. Move the ends when it is done."
            if self.busy():
                return "Muse is working on the clip now. Wait until it is done, or press Stop."
            self.making = hand
        threading.Thread(target=self._move, args=(hand, start, end), name="change-ends", daemon=True).start()
        return None

    def _move(self, hand, start, end):
        before, landed = self.project.state()["clip"], ""
        try:
            said = TOOLS["change_ends"].run(hand, TAPPED, start="%.2f" % start, end="%.2f" % end)
            landed = " ".join(line for line in (said or "").splitlines() if line.startswith(LANDED))
            said = None if self.project.state()["clip"]["moment"] != before["moment"] else said
        except Refused as e:
            said = str(e)
        except hearing.Stopped:
            said = "Stopped."
        except Exception as e:
            said = "The studio broke while moving the clip's ends (%s)." % e
        finally:
            hand.done_working()
            with self.cond:
                self.making, hand.done = None, True
        if said:
            self.emit("not_moved", error=_first_sentence(said))
            return
        clip = self.project.state()["clip"]
        self._tell("the person moved the clip's ends on the page: it now runs from %s to %s of the recording"
                   % (recording.clock(clip["pieces"][0][0]), recording.clock(clip["pieces"][-1][1])))
        self.emit("moved", clip=self.clip_view(), said=landed)

    def framing(self, state):
        """Where the upright frame sits over a lying recording, for the Frame tab: {"picture": a frame of the clip's
        middle (recording.frame), "size": the recording's [w, h], "x" and "width": the frame's left edge and width in
        its pixels, "follows": True while it moves by itself, "why": None}, or {"why": the sentence} when there
        is no frame to move. Tarm 10-30: "ซูมเป็นแนวตั้ง แต่ขยับกรอบไปชิดซ้าย"."""
        clip, rec = state["clip"], self.recording or {}
        whole = next((p for p in clip["layout"]["parts"] if p["name"] == "whole"), None)
        if clip["layout"]["kind"] != "subject" or not whole:
            return {"why": "Only a clip whose picture fills the frame (the subject layout) has a frame to move: ask "
                           "Muse to make it in the subject layout."}
        if not rec.get("width") or whole["src"][2] >= rec["width"] - 1:
            return {"why": "The frame already shows the recording's whole width, so there is nowhere to move it."}
        _, at = project.to_clip(clip["pieces"], state["length"] / 2) or (0, clip["pieces"][0][0])
        return {"picture": recording.frame(rec, at), "size": [rec["width"], rec["height"]],
                "x": round(whole["src"][0]), "width": round(whole["src"][2]), "follows": bool(whole.get("follow")),
                "why": None}

    def set_frame(self, x):
        """The person's place for the upright frame, from the Frame tab: x is its left edge in the recording's pixels.
        It holds there through the whole clip (a frame that followed the product stops following, and the product it
        followed is dropped, so check_clip and Muse take the person's place as it is), as one change Undo takes
        back, and Muse is told. Returns None, or the sentence why not."""
        state = self._clip_now()
        now = self.framing(state)
        if now["why"]:
            return now["why"]
        x = round(min(max(0.0, x), self.recording["width"] - now["width"]))
        if x == now["x"] and not now["follows"]:
            return None
        lay = state["clip"]["layout"]
        parts = [dict({k: v for k, v in p.items() if k != "follow"}, src=[x] + list(p["src"][1:]))
                 if p["name"] == "whole" else p for p in lay["parts"]]
        self.edit({"do": "layout", "part": "frame", "box": None,
                   "layout": dict({k: v for k, v in lay.items() if k != "thing"}, parts=parts)})
        w = float(self.recording["width"])
        self._tell("the person moved the upright frame on the page: it now shows %d%% to %d%% of the recording's "
                   "width, and holds there%s" % (round(100 * x / w), round(100 * (x + now["width"]) / w),
                                                 " (it no longer follows anything)" if now["follows"] else ""))
        return None

    def has_sound(self):
        """True when the clip's recording has sound to mute."""
        return bool((self.recording or {}).get("sound_tracks"))

    def set_mute(self, start, end, back=False):
        """The person's seconds to mute under the clip, or to bring the sound back over: mute_sound's own edit, as
        set_captions."""
        self._clip_now()
        refused, entry = self.tap("mute_sound", start=start, end=end, back=back)
        if entry:
            self._tell("the person %s from %g to %g s of the clip on the page"
                       % ("brought the sound back" if back else "muted the sound", entry["start"], entry["end"]))
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
        said" does not fail it, and add_captions keeps it when the captions go on again. A draft, kept as it is or
        changed, becomes such a phrase, among the captions by its time. Told to Muse before its next job. Returns
        None, or the sentence why not; Missing when the block is not on the clip as the page showed it."""
        state = self._clip_now()
        block = next((b for b in self.words_on(state) if b["kind"] == kind and b["n"] == n), None)
        if block is None or block["text"] != was:
            raise Missing(WORDS_GONE)
        words = " ".join(str(words).split())
        if not words:
            return "Type the words first."
        if words == was and kind != "draft":
            return None
        if kind == "title":
            t = state["titles"][n]
            refused, entry = self.tap("add_title", words=words, start=t["start"], end=t["end"],
                                      key=t["key"] if t.get("key") and t["key"] in words else None,
                                      **_spot(state, t))
            if refused:
                return refused
        else:
            caps = state["captions"]
            mine = [ln for ln in caps["drafts" if kind == "draft" else "lines"] if ln["phrase"] == n]
            typed = text.caption_lines([{"from": mine[0]["from"], "to": mine[-1]["to"], "text": words}],
                                       ONE_LETTERS if caps.get("one") else CAPTION_LETTERS)
            for ln in typed:
                ln.update(phrase=n, typed=True)
                for key in ("who", "said"):
                    if key in mine[0]:
                        ln[key] = mine[0][key]
            if kind == "draft":
                self.edit(dict(caps, lines=_placed(caps["lines"], typed),
                               drafts=[ln for ln in caps["drafts"] if ln["phrase"] != n]))
                self._tell('the person %s the caption draft at %g-%g s on the page, so it is on the video now: "%s"'
                           % ("kept" if words == was else 'changed "%s" in' % was, block["start"], block["end"], words))
                return None
            self.edit(dict(caps, lines=[ln for ln in caps["lines"] if ln["phrase"] < n] + typed +
                           [ln for ln in caps["lines"] if ln["phrase"] > n]))
        self._tell('the person typed their own words for the %s at %g-%g s on the page: "%s" is now "%s"'
                   % (kind, block["start"], block["end"], was, words))
        return None

    def set_title_time(self, n, was, start, end):
        """The person's drag of a title's ends under the clip: it shows from start to end s of the clip. It goes on
        again through add_title with its own words, place and key, so its checks hold (long enough on screen, off
        faces). A title of the same place it would run into is not replaced, as a new title there would replace it:
        the drag is refused with that title's time. Told to Muse before its next job. Returns None, or the sentence
        why not; Missing when the title is not on the clip as the page showed it."""
        state = self._clip_now()
        titles = state["titles"]
        if not 0 <= n < len(titles) or titles[n]["text"] != was:
            raise Missing(WORDS_GONE)
        t = titles[n]
        start, end = round(start, 2), round(end, 2)
        if abs(start - t["start"]) < 0.05 and abs(end - t["end"]) < 0.05:
            return None
        hit = next((o for k, o in enumerate(titles) if k != n and project.same_spot(o, dict(t, start=start, end=end))),
                   None)
        if hit:
            return 'It would run into "%s", which shows in the same place from %g to %g s: keep it %s.' % (
                hit["text"], hit["start"], hit["end"],
                "after %g s" % hit["end"] if hit["start"] < t["start"] else "before %g s" % hit["start"])
        refused, entry = self.tap("add_title", words=t["text"], start=start, end=end, key=t.get("key"),
                                  **_spot(state, t))
        if refused:
            return refused
        self._tell('the person dragged the ends of the title "%s" on the page: it shows from %g to %g s now, not '
                   '%g-%g s' % (t["text"], entry["start"], entry["end"], t["start"], t["end"]))
        return None

    def make_again(self):
        """The person's tap on "Make the video again": render_clip's own render of the clip as it is now, run in the
        background without asking Muse, its progress shown as a render's is. render_clip's checks hold (nobody left
        undecided, nothing made when the file is already the clip as it is now), and its file is the one check_clip
        and hand_over take, as from Muse. Returns None once it has started, or the sentence why not."""
        if _undecided(self._clip_now()):
            return "Decide about each person in the clip first: tap a face to blur it, and tap it again to show it."
        refused, _ = self.tap("render_clip", trial=True)
        if refused:
            return refused
        hand = Hand(self)
        with self.cond:
            if self.making:
                return "The video is being made now."
            if self.busy():
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

    def set_captions_language(self, name):
        """The person's tap on As said beside the clip: add_captions's own edit in the language the words are said
        in, which needs no translation, so Muse is not asked; lit or popped stays as it is. English is not put on
        here: only Muse writes it, so its tap is a message to Muse. Returns None, or the sentence why not."""
        state = self._clip_now()
        if name != AS_SAID:
            return "Only As said changes here; Muse writes the captions in another language when you tap it."
        caps = state["captions"]
        if not caps or not caps.get("language"):
            return None
        refused, entry = self.tap("add_captions", lit=bool(caps.get("lit")), pop=bool(caps.get("pop")),
                                  language=self._said_in(state))
        if entry:
            self._tell("the person put the captions back in the words as said, on the page")
        return refused

    def _said_in(self, state):
        """The language the clip's moment is said in, as add_captions names it."""
        moments, n = self.moments or [], state["clip"]["moment"]
        said = moments[n - 1].get("said") or [] if n <= len(moments) else []
        return _spoken(Hand(self, True), [p["text"] for p in said])

    def layer(self):
        """The clip's words as the layer the page lays over its video (layer.words): [{"from", "to", "file"}] in clip
        seconds. Missing before there is a clip; RuntimeError when FFmpeg cannot draw it."""
        return layer.words(self._clip_now(), self.clip["style"] if self.clip else "clean")

    def base(self):
        """The page's video of the clip as it is now, without its words over the picture (render.base): {"url",
        "from_cache"} when it is there within PREVIEW_WAIT s; else {"url": None, "making": True} while it is made in
        the background, as make_again makes the video (its progress shown, Stop ends it), and a "base" event says
        when it is made, or why not. One is made at a time; asked while one is made, the page asks again when its
        event comes. A change to the words alone keeps the same file, so only a change to the picture or the sound
        waits (lane 5 measured a new one at 9 to 17 s, 10-28). {"error"} when it could not be made."""
        state = self._clip_now()
        hand, got, done = Hand(self, page="preview"), {}, threading.Event()
        with self.cond:
            if self.basing:
                return {"url": None, "making": True}
            self.basing = hand
        total, shown = max(0.1, state["length"]), [0.0]

        def started(stop):
            got["ran"] = True
            hand.started_work(stop)

        def progress(at):
            if time.time() - shown[0] >= SHOW_THINKING_EVERY:
                shown[0] = time.time()
                hand.working("Making the preview: %.0f of %.0f s (%d%%)" % (min(at, total), total,
                                                                            min(99, 100 * at / total)))

        def make():
            try:
                got["url"] = "/api/base/" + os.path.basename(render.base(
                    self.recording, state, on_progress=progress, on_start=started,
                    default=self.clip["style"] if self.clip else "clean"))
            except render.Stopped:
                got["error"] = "The preview was not made: Stop was pressed."
            except render.NotDrawn as e:
                got["error"] = "The preview could not be made: %s" % e
            except Exception as e:
                got["error"] = "The studio broke while making the preview (%s)." % e
            finally:
                hand.done_working()
                with self.cond:
                    self.basing, hand.done, got["done"] = None, True, True
                    late = got.get("late")
            if late:
                self.emit("base", job=None, **{k: got[k] for k in ("url", "error") if k in got})
            done.set()

        threading.Thread(target=make, name="preview", daemon=True).start()
        done.wait(PREVIEW_WAIT)
        with self.cond:
            if not got.get("done"):
                got["late"] = True
                return {"url": None, "making": True}
        if "error" in got:
            return {"error": got["error"]}
        return {"url": got["url"], "from_cache": not got.get("ran")}

    def worked(self, n, steps):
        """The line on the Ready to post card that shows how little the person did for the clip job n made: Muse's
        time on the job from its start, the messages the person typed in the chat so far, their taps (a chip under
        Muse's guesses, a video picked, a change made on the page), and the job's steps."""
        started = next((e["t"] for e in self.events if e["kind"] == "job" and e.get("job") == n), time.time())
        said = [e["text"] for e in self.events if e["kind"] == "person"]
        typed = sum(1 for s in said if not s.startswith(TAP_SAID) and not (s.startswith(OPEN_SAID) and "\n" not in s))
        taps = sum(1 for s in said if s.startswith((TAP_SAID, OPEN_SAID))) + sum(
            1 for e in self.events if e.get("job") is None and e["kind"] in ("edit", "undo", "rendered"))
        seconds = round(time.time() - started)
        line = "Muse worked %s · you typed %s%s · %s" % (
            recording.clock(seconds), _count(typed, "message") if typed else "nothing",
            " and tapped %s" % ("once" if taps == 1 else "%d times" % taps) if taps else "", _count(steps, "step"))
        return {"seconds": seconds, "messages": typed, "taps": taps, "steps": steps, "line": line}

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

    def cover_of(self, name):
        """The newest cover made in this chat for the finished file name, or None."""
        return next((e["cover"] for e in reversed(self.events) if e["kind"] == "cover" and e.get("file") == name),
                    None)

    def hold_open_clip(self):
        """Before make_clips: the clip open now keeps its place in the album, the album's first when there is none
        yet, so the clips made next do not take its edits away."""
        clip = self.project.state()["clip"]
        if clip and not self.album:
            self.album, self.album_at = [{"moment": clip["moment"], "title": "", "project": self.project}], 0
        self.project = project.Project()

    def put_in_album(self, made, since):
        """The clips made in one go, each {"moment", "title", "project"}, go in the album, one from a moment the album
        holds in that clip's place, and the first of them opens. With none, the clip open before is open again. since
        is the event before they were made: a hand-over older than that was of the clip a place held before."""
        for c in made:
            place = dict({k: c[k] for k in ("moment", "title", "project")}, since=since)
            at = next((k for k, had in enumerate(self.album) if had["moment"] == c["moment"]), None)
            if at is None:
                self.album.append(place)
            else:
                self.album[at] = place
        if made:
            self.album_at = next(k for k, had in enumerate(self.album) if had["moment"] == made[0]["moment"])
        if self.album:
            self.project = self.album[self.album_at]["project"]

    def fill_album(self, made, since):
        """put_in_album, logged so a chat read back has each clip with its own edits and the same one open. Tarm 10-30:
        "ช่วยดูเรื่อง preview ให้มันออกมาเป็น album จะได้เลือกแก้ไขต่อได้ง่าย". make_clips kept only the first clip's edits
        then, and the others could only be watched."""
        with self.cond:
            self.put_in_album(made, since)
        self.emit("album", at=self.album_at, since=since, clips=[
            {"moment": c["moment"], "title": c["title"], "entries": c["project"].entries, "acts": c["project"].acts}
            for c in made], **self.opened_clip())

    def to_album_place(self, at, new=None):
        """Opens the album's place at, after adding one for the moment new. What was done to the clip left stays in its
        place, its undo too."""
        if new is not None:
            self.album.append({"moment": new, "title": "", "project": project.Project()})
        self.album_at, self.project = at, self.album[at]["project"]

    def open_album_clip(self, at):
        with self.cond:
            self.to_album_place(at)
        self.emit("picked", at=at, **self.opened_clip())

    def album_to(self, moment):
        """Before make_clip makes a clip from moment: with an album, that moment's clip opens, or a new place for it, so
        the clip open now is not made over."""
        if not self.album:
            return
        at = next((k for k, place in enumerate(self.album) if moment in self._moments_of(place)), None)
        if at is None:
            with self.cond:
                self.to_album_place(len(self.album), moment)
            self.emit("picked", at=self.album_at, new=moment, **self.opened_clip())
        elif at != self.album_at:
            self.open_album_clip(at)

    def pick_clip(self, at):
        """The person's tap on a clip of the album: it opens in the editor. Returns None, or the sentence why not."""
        if not 0 <= at < len(self.album):
            return "That clip is not in the album."
        with self.cond:
            if self.making:
                return "The video is being made now. Open another clip when it is done."
            if self.busy():
                return "Muse is working on the clip now. Wait until it is done, or press Stop."
        if at != self.album_at:
            self.open_album_clip(at)
            self._tell("the person opened clip %d of the album on the page, so that clip is the one open now" % (at + 1))
        return None

    def _last_match(self):
        """The act of the last album-wide look that was not taken back, or None."""
        for e in reversed(self.events):
            if e["kind"] in ("album", "opened", "unmatched"):
                return None
            if e["kind"] == "matched":
                return e["act"]
        return None

    def match_album(self, parts, job=None):
        """Puts parts (tools.ALBUM_PARTS) of the clip open now on the album's other clips, each as one action of that
        clip, so its own Undo takes it back and the clip left open is not disturbed. A name the clip could not draw
        stays off it and is said. Returns the sentence of what changed."""
        default = (self.clip or {}).get("style", "clean")
        act, changes, left = "%s.album.%s" % (self.id, secrets.token_hex(4)), [], []
        with self.cond:
            source = self.project.state()
            for at, place in enumerate(self.album):
                if at == self.album_at:
                    continue
                entries, why = album_entries(source, place["project"].entries, parts, default)
                kept = []
                for entry in entries:
                    after = project.Project(place["project"].entries + kept + [entry]).state()
                    unfit = entry["do"] in ("channel", "style") and channel_unfit(after, default)
                    if unfit:
                        why.append(_first_sentence(unfit))
                    else:
                        kept.append(entry)
                for entry in kept:
                    place["project"].add(entry, act)
                if kept:
                    changes.append({"at": at, "entries": kept})
                left += ["Clip %d kept its own: %s." % (at + 1, w.rstrip(".")) for w in why]
            self.changed += 1
        left += ["This clip has no face circle, so no circle moved."] if "circle" in parts and not _circle_box(source) else []
        said = " ".join(["The look is on %d clip%s of the album." % (len(changes), "" if len(changes) == 1 else "s")
                         if changes else "Every other clip already has this look."] + left)
        if changes:
            self.emit("matched", job=job, act=act, changes=changes, said=said)
        return said

    def unmatch_album(self, job=None):
        """Takes back the last match_album from each clip whose last action it still is: a clip changed since keeps
        that change, and is said. Returns the sentence of what happened."""
        act = self._last_match()
        if act is None:
            return "There is no album-wide look to take back."
        with self.cond:
            back = [at for at, place in enumerate(self.album) if place["project"].acts[-1:] == [act]]
            stayed = [at for at, place in enumerate(self.album) if at not in back and act in place["project"].acts]
            for at in back:
                self.album[at]["project"].undo()
            self.changed += 1
        said = "The look was taken back from %d clip%s." % (len(back), "" if len(back) == 1 else "s")
        said += " Clip%s %s changed since, so take it back there with Undo." % (
            "" if len(stayed) == 1 else "s", ", ".join(str(at + 1) for at in stayed)) if stayed else ""
        self.emit("unmatched", job=job, back=back, said=said, **self.opened_clip())
        return said

    def _album_ready(self):
        """Why the album cannot be changed by a tap now, or None."""
        if len(self.album) < 2:
            return "There is no album: it holds the clips made in one go."
        if self.making:
            return "The video is being made now. Wait until it is done."
        if self.busy():
            return "Muse is working on the clip now. Wait until it is done, or press Stop."
        return None

    def album_look(self, parts):
        """The person's press of Use this clip's look on all clips. Returns None, or the sentence why not."""
        self._clip_now()
        if not parts or any(p not in ALBUM_PARTS for p in parts):
            return "Tick what to put on the other clips: %s." % ", ".join(ALBUM_PARTS.values())
        with self.cond:
            why = self._album_ready()
        if why:
            return why
        self._tell("the person used this clip's look on the whole album on the page: %s" % self.match_album(parts))
        return None

    def album_back(self):
        """The person's press of Take back. Returns None, or the sentence why not."""
        with self.cond:
            why = self._album_ready()
        if why:
            return why
        if self._last_match() is None:
            return "There is no album-wide look to take back."
        self._tell("the person took the album-wide look back on the page: %s" % self.unmatch_album())
        return None

    def make_album(self):
        """The person's press of Update all videos: render_album in the background, as make_again. Returns None once it
        has started, or the sentence why not."""
        hand = Hand(self)
        with self.cond:
            why = self._album_ready()
            if why:
                return why
            if not _changed_clips(self):
                return "No clip of the album has changed since its video was made."
            self.making = hand
        threading.Thread(target=self._make_album, args=(hand,), name="make-album", daemon=True).start()
        return None

    def _make_album(self, hand):
        try:
            said = TOOLS["render_album"].run(hand, TAPPED)
        except Refused as e:
            said = str(e)
        except Exception as e:
            said = "The studio broke while making the videos (%s)." % e
        finally:
            with self.cond:
                self.making, hand.done = None, True
        self._tell("the person made the album's changed videos again on the page")
        self.emit("album_made", said=_album_said(said), **self.opened_clip())

    def _moments_of(self, place):
        clip = place["project"].state()["clip"]
        return {place["moment"], clip["moment"] if clip else None}

    def _ready_of(self, place):
        """What was handed over last with the clip in this place of the album, or None."""
        moments = self._moments_of(place)
        return next((e for e in reversed(self.events) if e["kind"] == "ready" and e.get("moment") in moments
                     and e["id"] > place.get("since", 0)), None)

    def opened_clip(self):
        """The clip open now, for the page: {"clip": clip_view, "file": its finished file or None, "ready": what was
        handed over with it or None, "title": its place's title or ""}."""
        place = self.album[self.album_at] if self.album else None
        return {"clip": self.clip_view(), "file": self.finished(), "ready": self._ready_of(place) if place else None,
                "title": place["title"] if place else ""}

    def album_now(self):
        """The album for the strip above the clip, or None with fewer than two clips: {"at", "clips": [{"moments",
        "title", "length", "file", "state"}]}. A clip's file is its newest finished one; its state is "ready" when that
        file is the clip as it is now and a hand-over was made, "changed" when it was changed since, else "unfinished"."""
        if len(self.album) < 2:
            return None
        clips = []
        for place in self.album:
            sign = project.signature(place["project"].entries)
            moments = sorted(m for m in self._moments_of(place) if m is not None)
            done = next((f for f in reversed(self.rendered) if f["signature"] == sign), None)
            newest = done or next((f for f in reversed(self.rendered) if f.get("moment") in moments), None)
            place["file"] = newest["name"] if newest else place.get("file")
            ready = self._ready_of(place)
            clips.append({"moments": moments,
                          "title": (ready or {}).get("title") or place["title"],
                          "length": place["project"].state()["length"], "file": place.get("file"),
                          "state": "changed" if place.get("file") and not done else "ready" if ready else "unfinished"})
        return {"at": self.album_at, "clips": clips, "matched": self._last_match() is not None,
                "changed": sum(c["state"] == "changed" for c in clips),
                "parts": [{"name": k, "label": v} for k, v in ALBUM_PARTS.items()]}

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
        caps = state["captions"]
        extras = [("lit captions on" if caps.get("lit") else "popped captions on" if caps.get("pop") else
                   "captions on") if caps and caps["lines"] else "no captions"]
        drafts = len({ln["phrase"] for ln in (caps or {}).get("drafts") or []})
        if drafts:
            extras.append("%d caption draft%s the person has not kept yet" % (drafts, "" if drafts == 1 else "s"))
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
        if state["pictures"]:
            extras.append("pictures drawn: " + ", ".join("%s at %g s" % (k["about"], k["at"])
                                                         for k in state["pictures"]))
        if state["backdrop"]:
            extras.append("background drawn: " + state["backdrop"]["about"])
        if state["flurries"]:
            extras.append("flurries: " + ", ".join("%s %s at %g s" % (" and ".join(f["names"]), f["move"], f["at"])
                                                   for f in state["flurries"]))
        if state["sounds"]:
            extras.append("sounds: " + ", ".join("%s at %g s" % (s["kind"], s["at"]) for s in state["sounds"]))
        if state["music"]:
            extras.append("music %s" % os.path.basename(state["music"]["path"]))
        if state["mutes"]:
            extras.append("sound muted " + ", ".join("%g-%g s" % (a, b) for a, b in state["mutes"]))
        done = self.finished()
        if done:
            made = "finished file %s, made from the clip as it is now" % done["name"]
            cover = self.cover_of(done["name"])
            made += ("; its cover %s is made" % cover if cover else
                     "; no cover made for it: never say it has a cover")
        elif self.rendered:
            made = ("the last finished file is older than the last change: render again before you hand it over, and "
                    "never say the clip is ready until then")
        else:
            made = ("no finished file yet: the person sees only a preview on the page, so never say the clip is "
                    "ready")
        album = "clip %d of %d in the album (open_clip opens another), " % (
            self.album_at + 1, len(self.album)) if len(self.album) > 1 else ""
        return ("%sclip: moment %d (%s-%s, %g s), %s layout, %s style, %s, %s · %s"
                % (album, clip["moment"], recording.clock(a), recording.clock(b), state["length"],
                   clip["layout"]["kind"], st, "words on screen: " + words if words else "no words on screen",
                   ", ".join(extras), made))

    def open(self, rec, job=None):
        """Opens rec as this chat's recording (one at a time) and shows it.
        Returns (the recording open before, the path of a picture from rec or None)."""
        with self.cond:
            had, self.recording = self.recording, rec
            if not had or had["path"] != rec["path"]:
                self.looks, self.moments, self.project, self.rendered = [], None, project.Project(), []
                self.checks = {}
                self.album, self.album_at = [], 0
            self.changed += 1
        frame = recording.frame(rec)
        self.emit("opened", job=job, recording=rec, frame=frame, line=recording.describe(rec))
        return had, (os.path.join(recording.WORK, "frames", frame) if frame else None)

    def open_dropped(self, rec, words, mind_key, language=""):
        """The person opened rec from the page: it opens, and Muse is told, with any words typed with it and one
        picture from it. The file name can mislead (the review clip is called trip_...), the picture does not.
        Until the person types, the browser's language is the only clue to the language they read."""
        _, picture = self.open(rec)
        opened, said = OPEN_SAID + recording.describe(rec), ("\n" + words if words else "")
        told = opened
        seen = montage.overview(rec)
        if seen:
            picture = seen[0]
            told += "\n" + montage.OVERVIEW_SAID % ", ".join(recording.clock(t) for t in seen[1])
        told += _reader(words, language)
        return self.say(told + said, mind_key, [picture] if picture else (), shown=opened + said)

    def open_sent(self, folder, n, words, mind_key, language=""):
        """The person sent n videos from their phone, into folder: Muse is told, to open them together."""
        shown = OPEN_SAID + "%d videos sent from the phone, in %s" % (n, folder)
        told = shown + "\nOpen them together with open_clips." + _reader(words, language)
        said = "\n" + words if words else ""
        return self.say(told + said, mind_key, (), shown=shown + said)


class Job:
    """One message from the person and everything Muse does about it, including words typed while it works."""

    def __init__(self, chat, brain, content, said=None):
        """said is the person's own words when they typed the message (not a tap, not an opened recording): only
        then may the message turn out to be talk (needs.py)."""
        self.chat, self.brain, self.n = chat, brain, chat.jobs
        self.said, self.sorted, self.talk = said, False, False
        self.inbox = [content]
        self.pictures = []
        self.proofread = {}
        self.asking = None
        self.stopped = self.done = False
        self.plan, self.plan_shown, self.worked = [], "[]", set()
        self.guesses = None
        self.steps = self.refused = self.fixed = self.guard_stops = 0
        self.handed = False
        self.finishing = 0
        self.seen = {}
        self.last_refused = {}
        self.warned = set()
        self.refused_last = False
        self.shown_live = 0.0
        self.stop_work = self.act = None
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
            return hearing.hear(rec, start, end, on_start=self.started_work, on_paid=lambda usd: spent.add(
                os.path.join(WORK, "spent.jsonl"), self.chat.id, self.n, asr.VOICE, {"usd": round(usd, 6)}),
                languages=self.languages())
        finally:
            self.done_working()

    def languages(self):
        """The languages the hearing expects in this chat (heard_in)."""
        return heard_in([e["text"] for e in self.chat.events if e["kind"] == "person"])

    def find_faces(self, rec, start, end):
        """The faces in start-end for a tool: the page says so, and Stop ends it at once (faces.Stopped)."""
        self.working("Finding the faces in %s-%s" % (recording.clock(start), recording.clock(end)))
        try:
            return faces.sample(rec, start, end, on_start=self.started_work)
        finally:
            self.done_working()

    def edit(self, entry):
        return self.chat.edit(entry, job=self.n, act=self.act)

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

    def ask_aside(self, messages, schema, doing="Fixing misheard words from the context", room=None):
        """Asks Muse apart from this job's conversation (the conversation and its cache stay as they are) and
        returns the words of the answer, in at most room tokens (hearing.FIX_TOKENS when None). Stop cancels it
        (hearing.Stopped)."""
        asking = mind.Asking()
        self.working(doing)
        self.started_work(asking.cancel)
        try:
            return self.paid(self.brain.ask(messages, temperature=0.2, max_tokens=room or hearing.FIX_TOKENS,
                                            asking=asking, schema=schema))["text"]
        except mind.Cancelled:
            raise hearing.Stopped()
        finally:
            self.done_working()

    def paid(self, got, brain=None):
        """Writes what an answer cost into the studio's ledger (spent.py) and gives the answer back."""
        spent.add(os.path.join(WORK, "spent.jsonl"), self.chat.id, self.n, brain or self.brain, got["engine"])
        return got

    def ask_other(self, key, messages, schema, doing, room):
        """Asks another Muse (mind key) apart from this job's conversation, for what only it can do, such as Muse
        Wonder watching a video with its sound: {text, engine, ...}, its cost in the ledger under that Muse. Stop
        cancels it (hearing.Stopped); a Muse that is not set up raises MindError with its sentence."""
        other = mind.MINDS.get(key)
        if other is None:
            raise mind.MindError("There is no Muse called %s." % key)
        asking = mind.Asking()
        self.working(doing)
        self.started_work(asking.cancel)
        try:
            return self.paid(other.ask(messages, temperature=0.2, max_tokens=room, asking=asking, schema=schema),
                             other)
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

    def false_cover(self, text):
        """True when Muse's answer speaks of the clip's cover and no cover was made for the finished file. Beat 1,
        10-29: make_cover was refused on Glimmer, and the answer still told Tarm "และปกพร้อมโพสต์ให้แล้ว"."""
        if not COVER_SAID.search(text or ""):
            return False
        done = self.chat.finished()
        return not (done and self.chat.cover_of(done["name"]))

    def run(self):
        chat, started, end, nudged = self.chat, time.time(), "answered", False
        closing = covered = warned = False
        self.emit("job", mind=self.brain.label)
        try:
            while True:
                self.take_inbox()
                if self.stopped:
                    end = "stopped"
                    break
                if self.steps >= MAX_STEPS and not closing:
                    if self.handed:
                        closing = True
                    elif not self.finishing and self.to_give() and not (_waits(self) or self.gave()):
                        left = self.finish_tools()
                        self.finishing = self.steps + FINISH_STEPS
                        chat.add({"role": "user", "content": FINISH % (FINISH_STEPS, ", ".join(left))})
                        self.emit("note", text="Muse has used its %d steps, so the studio left it only %s and the "
                                               "hand-over, to give you the clip as it is." % (
                                                   self.steps, "the check" if left == FINISH_TOOLS
                                                   else "the render, the check"))
                    elif not self.finishing or self.steps >= self.finishing:
                        end = "too many steps"
                        self.emit("note", text="Muse took %d steps on this message, the most one message may take, so "
                                               "the studio stopped it here. Say what to do next." % self.steps)
                        break
                if self.steps >= MAX_STEPS - STEPS_WARNED and not (warned or closing or _waits(self) or self.gave()):
                    warned = True
                    chat.add({"role": "user", "content": LAST_STEPS % (MAX_STEPS - self.steps)})
                with chat.cond:
                    if self.inbox or self.stopped:
                        continue
                    self.asking = mind.Asking()
                    chat.live = {"job": self.n, "step": self.steps + 1, "thinking": "", "text": ""}
                self.emit("wait", step=self.steps + 1)
                try:
                    got = self.ask(tools=None) if closing else self.ask(self.offered())
                    if got["tool_calls"] and self.only_talk():
                        got = self.ask(tools=None)
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
                if not calls and self.finishing and not self.handed:
                    self.steps += 1
                    self.emit("answer", final=False, text=words.unslipped(got["text"]), thinking=got["thinking"],
                              engine=got["engine"], mind=self.brain.label)
                    chat.add({"role": "user", "content": FINISH_CALL % ", ".join(self.finish_tools())})
                    continue
                if not calls:
                    left = self.finish_plan()
                    if left and not nudged and not self.refused_last and not closing and not _waits(self):
                        nudged = True
                        self.emit("answer", final=False, text=words.unslipped(got["text"]),
                                  thinking=got["thinking"], engine=got["engine"], mind=self.brain.label)
                        chat.add({"role": "user", "content": "[studio] Your plan has steps no call has named yet: %s. "
                                  "Do them now, or answer the person and say why they are not needed."
                                  % "; ".join("%d. %s" % s for s in left)})
                        self.emit("note", text="Muse stopped with plan steps still open, so the studio asked it to go "
                                               "on or say why.")
                        continue
                    if not covered and self.false_cover(got["text"]):
                        covered = True
                        self.emit("answer", final=False, text=words.unslipped(got["text"]),
                                  thinking=got["thinking"], engine=got["engine"], mind=self.brain.label)
                        chat.add({"role": "user", "content": NO_COVER})
                        self.emit("note", text="Muse spoke of a cover, and no cover is made, so the studio asked it "
                                               "to answer again.")
                        continue
                    with chat.cond:
                        more = bool(self.inbox) and not self.stopped
                        self.emit("answer", final=not more, text=words.unslipped(got["text"]),
                                  thinking=got["thinking"], engine=got["engine"], mind=self.brain.label,
                                  talk=self.talk)
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
                    self.emit("step", step=self.steps, said=words.unslipped(got["text"]) if first else "",
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

    def gave(self):
        """True when this job has handed a clip over."""
        return any(e["kind"] == "ready" and e.get("job") == self.n for e in self.chat.events)

    def to_give(self):
        """True when there is a clip and the person has not got it as it is now."""
        done = self.chat.finished()
        if done:
            return not any(e["kind"] == "ready" and e.get("file") == done["name"] for e in self.chat.events)
        return bool(self.chat.project.state()["clip"])

    def finish_tools(self):
        """What a job out of steps may still call, in order: render_clip while no file is made from the clip as it is,
        and decide_people before it while anyone is undecided, as the render refuses them."""
        if self.chat.finished():
            return FINISH_TOOLS
        return (("decide_people",) if _undecided(self.chat.project.state()) else ()) + ("render_clip",) + FINISH_TOOLS

    def offered(self):
        """The tools Muse is offered at this step: only finish_tools on a job out of steps; make_countdown only in a
        sports match, which joins its goals with it, or once a person asked for a countdown (UNASKED)."""
        if self.finishing:
            left = self.finish_tools()
            return [s for s in SCHEMAS if s["function"]["name"] in left]
        said = [e["text"] for e in self.chat.events if e["kind"] == "person"]
        match = (self.chat.clip or {}).get("recipe") == "sport"
        tools = SCHEMAS if match or countdown.asked(said) is not None else UNASKED
        left = set(() if self.chat.album else ALBUM_TOOLS)
        if (self.chat.clip or {}).get("recipe") != "game" or not self.chat.moments:
            left.add("no_chat_box")
        return [s for s in tools if s["function"]["name"] not in left]

    def only_talk(self):
        """True when the person's typed message only talks, so Muse answers it in words with no tools offered. Asked
        once a job, before any step, and only when Muse's first answer calls a tool: an answer in words needs no
        question. A question that cannot be answered leaves the job as work, as it was before this existed."""
        if self.said is None or self.sorted or self.steps:
            return False
        self.sorted = True
        messages, schema = needs.question(needs.last_answer(self.chat.messages[:-1]), self.said)
        with self.chat.cond:
            if self.stopped or self.inbox:
                raise mind.Cancelled()
            self.asking = mind.Asking()
            self.chat.live = {"job": self.n, "step": 1, "work": "Reading whether you asked for work or are just "
                                                               "talking"}
            self.chat.live_rev += 1
            self.chat.cond.notify_all()
        try:
            said = self.paid(self.brain.ask(messages, temperature=0.0, max_tokens=needs.ROOM, asking=self.asking,
                                            schema=schema))["text"]
        except mind.MindError:
            return False
        finally:
            with self.chat.cond:
                self.chat.live = {"job": self.n, "step": 1, "thinking": "", "text": ""}
        self.talk = needs.read(said) == "talk"
        if self.talk:
            with self.chat.cond:
                if self.stopped or self.inbox:
                    raise mind.Cancelled()
                self.asking = mind.Asking()
        return self.talk

    def ask(self, tools=SCHEMAS):
        """Muse's next answer. When the conversation and an answer no longer fit in Muse's memory (mind.TooLong),
        the oldest pictures are taken out of it (make_room) and Muse is asked again, so a job is not ended by its own
        length. Measured 10-28: a montage of 19 clips filled Glimmer's 32,768 tokens 20 tokens before its hand-over,
        and a screen recording filled them while Muse wrote a call, which was cut off and could not be read.
        Pictures come out down to ROOM_AFTER of the memory, not just enough to fit, and once a Muse's memory is known
        (ROOMS, from its refusal), before a question when the last one came over ROOM_FULL of it: Glimmer reads the
        conversation again from the first picture taken out, so taking just enough made it read it again on every
        turn near the edge, and 10-30 it died 10 s into such a reading (router log, 0xC0000409). What pictures do not
        free comes from long old results (cut_results)."""
        room = ROOMS.get(self.brain.label)
        if room and self.chat.held + REPLY_ROOM > room * ROOM_FULL:
            with self.chat.cond:
                took, cut = free_room(self.chat.messages, self.chat.held + REPLY_ROOM - int(room * ROOM_AFTER))
            if took or cut:
                self.emit("note", text="%s's memory was nearly full, so the studio took out %s before asking." % (
                    self.brain.label, _taken(took, cut)))
        tries = 0
        while True:
            try:
                got = self.paid(self.brain.ask(self.chat.messages, tools=tools, temperature=AGENT_TEMPERATURE,
                                               max_tokens=AGENT_MAX_TOKENS, on_delta=self.on_delta,
                                               asking=self.asking))
                used = got["engine"]
                self.chat.held = sum(used.get(k) or 0 for k in ("read_new", "reused", "written"))
                return got
            except mind.TooLong as e:
                tries += 1
                ROOMS[self.brain.label] = e.room
                with self.chat.cond:
                    if self.stopped:
                        raise mind.Cancelled()
                    took, cut = (free_room(self.chat.messages, e.asked + REPLY_ROOM - int(e.room * ROOM_AFTER))
                                 if tries <= ROOM_TRIES else (0, 0))
                    if not (took or cut):
                        raise mind.MindError(
                            "%s's memory is full: this chat holds more than it can read at once (%d tokens), and "
                            "nothing is left to take out. Start a new chat for the next clip." % (self.brain.label,
                                                                                                  e.room))
                    self.asking = mind.Asking()
                self.emit("note", text="%s's memory was nearly full, so the studio took out %s, and asked again." % (
                    self.brain.label, _taken(took, cut)))

    def close(self, end, started):
        """Ends the job (the caller holds the lock). Words typed too late to be answered still go into the chat."""
        self.done = True
        for w in self.inbox:
            self.chat.add({"role": "user", "content": w})
        self.inbox = []
        self.emit("end", steps=self.steps, refused=self.refused, fixed=self.fixed,
                  seconds=round(time.time() - started, 1), why=end)
        if end == "answered":
            self.speller = threading.Thread(target=self.after_answer, name="spell-%d" % self.n, daemon=True)
            self.speller.start()

    def after_answer(self):
        """What runs once a job is answered, one question after the other, since Glimmer answers one at a time: the
        spelling of what the job showed, then the chat's name."""
        self.spell_shown()
        self.name_chat()

    def name_chat(self):
        """Names the chat once, after its first job that did any work, with a "named" event the list of earlier chats
        reads (chatname.py), from what was typed in the jobs that did work: small talk before them named a look at
        the yeti folder "รวมคลิปฝนตกทั้งวัน" (10-28). Like the spelling pass, the person's next message cancels it; a
        later job asks again."""
        events = self.chat.events
        if not self.steps or any(e["kind"] == "named" for e in events):
            return
        worked = {e["job"] for e in events if e["kind"] == "step"}
        said = [e["text"] for e in events if e["kind"] == "person" and e.get("job") in worked
                and not e["text"].startswith((TAP_SAID, OPEN_SAID))]
        answer = [e["text"] for e in events if e["kind"] == "answer" and e.get("job") == self.n and e.get("text")]
        titles = [e["title"] for e in events if e["kind"] == "ready" and e.get("title")]
        messages, schema = chatname.question(said, answer[-1] if answer else "", titles[-1] if titles else "")
        asking = mind.Asking()
        with self.chat.cond:
            if self.chat.job is not self:
                return
            self.chat.spelling = asking
        try:
            name = chatname.read(self.paid(self.brain.ask(messages, temperature=0.3, max_tokens=chatname.ROOM,
                                                          asking=asking, schema=schema))["text"])
        except (mind.Cancelled, mind.MindError):
            return
        finally:
            with self.chat.cond:
                if self.chat.spelling is asking:
                    self.chat.spelling = None
        if name:
            self.emit("named", title=name)

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
            fixed, _ = words.spelled(lambda m, s: self.paid(self.brain.ask(
                m, temperature=0.2, max_tokens=SPELL_TOKENS, asking=asking, schema=s))["text"], texts)
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
            if self.finishing and name not in self.finish_tools():
                raise Refused(FINISH_ONLY % ", ".join(self.finish_tools()))
            why, step, clean = tool.check(_arguments(raw), len(self.plan))
        except Refused as e:
            return self.refuse(shown, name, str(e))
        shown.update(why=words.unslipped(why), input=_named(clean, self.chat.clips), plan_step=step)
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
        self.act = "%s.%d.%d" % (self.chat.id, self.n, self.steps)
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
        self.handed = name == "hand_over"
        if step:
            self.worked.add(step)
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
        self.plan, self.worked = [{"text": s, "state": "todo"} for s in steps], set()
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
        """Muse answered in words: the step being worked on is done when a call on it went through, and so is a last
        step that only the answer could do ("compare them", "tell the person"). A step whose every call was refused
        is not done (10-29: English captions refused three times showed both steps ticked). Returns the steps no call
        has named or none went through: [(number, text)]."""
        for i, item in enumerate(self.plan, 1):
            if item["state"] == "now":
                item["state"] = "done" if i in self.worked else "todo"
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
    knows a choice the tool would refuse, and why. page names what the hand makes for the page (the video, the
    preview), so the page shows each one's progress in its own place."""

    def __init__(self, chat, trial=False, page="video", whole=False):
        self.chat, self.trial, self.page, self.n, self.steps = chat, trial, page, None, 0
        self.whole, self.act = whole, "%s.page.%s" % (chat.id, secrets.token_hex(4))
        self.proofread, self.pictures, self.shown_extra = {}, [], {}
        self.asking = self.stop_work = self.entry = None
        self.stopped = self.done = False

    def working(self, line):
        with self.chat.cond:
            self.chat.live = {"job": None, "step": 0, "work": line, "page": self.page}
            self.chat.live_rev += 1
            self.chat.cond.notify_all()

    def edit(self, entry):
        said = self.chat.refusal(entry)
        if said:
            raise Refused(said)
        if self.trial:
            raise Allowed()
        self.entry = self.entry or entry
        state = self.chat.edit(entry, act=self.act)
        if self.whole:
            return state
        raise Made()

    def ask_aside(self, messages, schema, doing="", room=None):
        raise mind.MindError("The page does not ask Muse.")

    def render(self, rec, state, out, default):
        if self.trial:
            raise Allowed()
        return Job.render(self, rec, state, out, default)


def _album_said(said):
    """render_album's sentences, each for Muse, as the page says them: how many were made, and each that was not."""
    lines = said.split("\n")
    made = [line for line in lines if ": Rendered " in line]
    return " ".join((["Updated %d video%s." % (len(made), "" if len(made) == 1 else "s")] if made else []) +
                    [line for line in lines if line not in made])


def _named(value, opened):
    """A step's input for the page, each {"clip": n} given the file name of clip n of the clips opened together, so
    the step reads as their names (Tarm 10-28: it read "[object Object]")."""
    names = {c["n"]: c["rec"]["name"] for c in (opened or {}).get("clips") or []}
    if isinstance(value, list):
        return [_named(v, opened) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: _named(v, opened) for k, v in value.items()}
    if value.get("clip") in names:
        out["name"] = names[value["clip"]]
    return out


def _first_sentence(said):
    """The first sentence of a tool's answer, for the page: the rest tells Muse what to do next."""
    return SENTENCE_END.split(said.strip(), 1)[0].split("\n", 1)[0]


def _count(n, one):
    return "%d %s%s" % (n, one, "" if n == 1 else "s")


def _grid_box(box):
    """[x1, y1, x2, y2] on the 0-1000 grid of the clip's frame, from [x, y, w, h] in its pixels: add_title takes a
    box this way and gives back the same pixels."""
    x, y, w, h = box
    return [x * 1000 / 1080, y * 1000 / 1920, (x + w) * 1000 / 1080, (y + h) * 1000 / 1920]


def _spot(state, t):
    """Where title t goes on again through add_title: its place while the layout has it, else its own box."""
    return {"place": t["place"]} if t["place"] in state["clip"]["layout"]["places"] else {"box": _grid_box(t["box"])}


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


def cut_results(messages, need):
    """Cuts long tool results to their first RESULT_HEAD characters, oldest first, until about need tokens are free, and
    returns how many it cut; each ends in CUT_OUT. The newest RESULT_KEPT messages stay whole, and so does the newest
    recipe: the job still follows it, while a recipe replaced by another is only memory."""
    recipe = next((m for m in reversed(messages) if m.get("role") == "tool" and str(m.get("content")).startswith(RECIPE_SAID)),
                  None)
    freed = cut = 0
    for m in messages[:-RESULT_KEPT]:
        c = m.get("content")
        if freed >= need:
            break
        if m.get("role") == "tool" and isinstance(c, str) and len(c) >= LONG_RESULT and m is not recipe:
            m["content"] = c[:RESULT_HEAD] + CUT_OUT
            freed += (len(c) - RESULT_HEAD - len(CUT_OUT)) / CHARS_TOKEN
            cut += 1
    return cut


def free_room(messages, need):
    """(pictures taken out, results cut) for about need tokens: pictures first, as Muse has looked at them, then results."""
    took = make_room(messages, need)
    return took, cut_results(messages, need - took * PICTURE_TOKENS) if need > took * PICTURE_TOKENS else 0


def _taken(took, cut):
    """What free_room took, for the page."""
    said = ["the %d oldest picture%s it had already looked at" % (took, "" if took == 1 else "s")] if took else []
    return " and ".join(said + (["the rest of %d long old result%s" % (cut, "" if cut == 1 else "s")] if cut else []))


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


def written_in(words):
    """A line for Muse naming the language the person typed words in, when its letters say it ("" otherwise).
    The rule in SYSTEM was not enough: on 10-29 Tarm typed Thai four times after a recording opened with the browser
    set to English, and Glimmer answered every one in English."""
    for name, letters in SCRIPTS:
        if letters.search(words or ""):
            return "\n[studio] The person wrote this in %s: answer in %s." % (name, name)
    return ""


def heard_in(texts):
    """The languages Muse Voice Transcribe is told to expect, by the names it knows: each one the person typed in, by
    its letters (SCRIPTS), and English, which games and talk mix in; () when they typed only Latin letters, so it
    finds the language itself. Measured 10-30 on our own footage where it had written Vietnamese and Chinese over
    Thai: 3 of 6 hearings had such a line told nothing, 0 of 6 told Thai and English."""
    names = [VOICE_NAMES.get(name, name) for name, letters in SCRIPTS if any(letters.search(t or "") for t in texts)]
    return tuple(names + ["English"]) if names else ()


def _reader(words, language):
    """What Muse is told about the language to answer in when a recording opens: the words typed with it, else the
    browser's language until the person writes."""
    if written_in(words):
        return written_in(words)
    if language:
        return "\nThe person's browser is set to %s; answer in that language until they write." % language
    return ""


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
    data = base64.b64encode(render.read_settled(path)).decode("ascii")
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
    """The newest chats on disk that have at least one message: [{id, when, first, messages, recording, title,
    minds, clips, usd}], title being the name Muse gave the chat (Job.name_chat), else the title of its newest finished
    clip, else "", minds each Muse that answered in it, [{label, local}], local when it runs on this computer (no key),
    clips each finished clip it handed over, oldest first, [{file, title, length}], and usd what its answers cost on a
    mind that is not on this computer: 0 when it used none, None when it did but its answers were never priced. Tarm
    10-28: the list read "yeti-c149d0193c.mp4" eight times over; 10-29: say which chat made which clip, and its cost."""
    folder = os.path.join(WORK, "chats")
    try:
        names = [n for n in os.listdir(folder) if n.endswith(".events.jsonl") and CHAT_ID.match(n[:-13])]
    except OSError:
        return []
    names.sort(key=lambda n: os.path.getmtime(os.path.join(folder, n)), reverse=True)
    costs = spent.by_chat(os.path.join(WORK, "spent.jsonl"))
    out = []
    for name in names:
        events = _read(os.path.join(folder, name))
        said = [e["text"] for e in events if e["kind"] == "person"]
        opened = [e["recording"]["name"] for e in events if e["kind"] == "opened"]
        if said:
            minds = list(dict.fromkeys(e["mind"] for e in events if e["kind"] == "job" and e.get("mind")))
            titles = [e["title"] for e in events if e["kind"] == "named"] or [
                e["title"] for e in events if e["kind"] == "ready" and e.get("title")]
            listed = [{"label": m, "local": not any(x.label == m and x.key_env for x in mind.MINDS.values())}
                      for m in minds]
            clips, seen = [], set()
            for e in events:
                if e["kind"] == "ready" and e.get("file") and e["file"] not in seen:
                    seen.add(e["file"])
                    clips.append({"file": e["file"], "title": e.get("title") or "", "length": e.get("length")})
            cid = name[:-13]
            paid = any(not m["local"] for m in listed)
            out.append({"id": cid, "when": round(os.path.getmtime(os.path.join(folder, name))),
                        "first": said[0][:100], "messages": len(said), "recording": opened[-1] if opened else "",
                        "title": titles[-1] if titles else "", "minds": listed, "clips": clips,
                        "usd": round(costs[cid], 4) if cid in costs else (None if paid else 0.0)})
        if len(out) >= limit:
            break
    return out
