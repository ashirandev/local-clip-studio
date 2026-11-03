"""A run packed as plain files plays without the studio: every picture and video its events name is in the folder,
the page asks for them where they sit, and no folder of the person's computer is in it."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, previews, recording, render, replay  # noqa: E402

CHAT = "20261027-150945-5680"
FRAME, PREVIEW, CLIP = "0123456789abcdef.jpg", "1123456789abcdef.mp4", "2123456789abcdef.mp4"
COVER = "3123456789abcdef.jpg"


class Pack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.was = agent.WORK, recording.WORK
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        os.makedirs(os.path.join(recording.WORK, "chats"))
        for folder, name, body in ((os.path.join(recording.WORK, "frames"), FRAME, b"jpg"),
                                   (previews.folder(), PREVIEW, b"preview"),
                                   (render.folder("clips"), CLIP, b"clip"),
                                   (render.folder("clips"), CLIP[:-4] + ".jpg", b"poster"),
                                   (render.folder("covers"), COVER, b"cover")):
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, name), "wb") as f:
                f.write(body)
        rec = {"path": "C:\\Users\\me\\Videos\\Reviews\\dry.mov", "name": "dry.mov"}
        self.events = [
            {"job": None, "recording": rec, "frame": FRAME, "line": "dry.mov · C:\\Users\\me\\Videos\\Reviews\\dry.mov",
             "id": 0, "kind": "opened", "t": 100.0},
            {"job": 1, "text": "a clip for posting", "id": 1, "kind": "person", "t": 100.5},
            {"job": 1, "mind": "Muse Glimmer", "id": 2, "kind": "job", "t": 101.0},
            {"job": 1, "name": PREVIEW, "id": 3, "kind": "preview", "t": 160.0},
            {"job": 1, "file": CLIP, "result": "Saved C:\\Users\\me\\Videos\\Reviews\\dry.mov as a clip", "id": 4,
             "kind": "rendered", "t": 220.0, "cover": COVER},
            {"job": 1, "step": 1, "tool": "render_clip", "id": 6, "kind": "step", "t": 221.0},
            {"job": 1, "steps": 3, "seconds": 241.0, "why": "answered", "id": 5, "kind": "end", "t": 342.0}]
        self.write(self.events)
        self.out = os.path.join(self.tmp, "replay")

    def tearDown(self):
        agent.WORK, recording.WORK = self.was
        agent.CHATS.pop(CHAT, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, events):
        base = os.path.join(recording.WORK, "chats", CHAT)
        with open(base + ".events.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e) + "\n" for e in events)
        with open(base + ".messages.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "user", "content": "a clip for posting"}) + "\n")

    def read(self, *name):
        with open(os.path.join(self.out, *name), "rb") as f:
            return f.read()

    def test_every_picture_and_video_the_run_names_is_in_the_folder_where_the_page_asks_for_it(self):
        done = replay.pack(CHAT, self.out, card="RTX 5070 Ti 16 GB")
        self.assertEqual(self.read("api", "frame", FRAME), b"jpg")
        self.assertEqual(self.read("api", "preview", PREVIEW), b"preview")
        self.assertEqual(self.read("api", "clip", CLIP), b"clip")
        self.assertEqual(self.read("api", "clip", CLIP[:-4] + ".jpg"), b"poster")
        self.assertEqual(self.read("api", "cover", COVER), b"cover", "lane 5 11-03: a Muse Image cover read as gone")
        page = self.read("app.js").decode("utf-8")
        self.assertNotIn('"/api/', page)
        self.assertIn('"api/frame/"', page)
        self.assertLess(self.read("index.html").index(b"replay.js"), self.read("index.html").index(b'src="app.js"'))
        self.assertTrue(os.path.isfile(os.path.join(self.out, "replay.js")))
        data = json.loads(self.read("api", "replay.json"))
        self.assertEqual([e["kind"] for e in data["events"]], [e["kind"] for e in self.events])
        self.assertEqual((data["chat"], data["took"], data["card"]), (CHAT, 241, "RTX 5070 Ti 16 GB"))
        self.assertEqual(done["events"], 7)

    def test_the_words_layer_and_the_base_video_play_from_the_folder_as_the_studio_sent_them(self):
        png, mp4 = "0123456789abcdef0123.png", "3123456789abcdef.mp4"
        for folder, name, body in ((render.folder("layers"), png, b"layer"), (render.folder("bases"), mp4, b"base")):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(body)
        layer = {"ok": True, "size": [540, 960], "steps": [{"from": 0.0, "to": 2.5, "url": "/api/layer/" + png}]}
        had = replay._layer_and_base
        replay._layer_and_base = lambda get, chat: (layer, {"ok": True, "url": "/api/base/" + mp4, "from_cache": True})
        try:
            replay.pack(CHAT, self.out, card="")
        finally:
            replay._layer_and_base = had
        self.assertEqual(self.read("api", "layer", png), b"layer")
        self.assertEqual(self.read("api", "base", mp4), b"base")
        data = json.loads(self.read("api", "replay.json"))
        self.assertEqual(data["layer"]["steps"][0]["url"], "api/layer/" + png, "an address the page uses from any folder")
        self.assertEqual(data["base"]["url"], "api/base/" + mp4)

    def test_no_folder_of_the_persons_computer_is_in_the_replay(self):
        replay.pack(CHAT, self.out, card="")
        data = self.read("api", "replay.json").decode("utf-8")
        self.assertNotIn("Users", data)
        self.assertNotIn("Reviews", data)
        data = json.loads(data)
        self.assertEqual(data["events"][0]["recording"]["path"], "dry.mov")
        self.assertEqual(data["events"][4]["result"], "Saved dry.mov as a clip")

    def test_the_replay_names_only_the_muse_that_ran_and_where_it_ran(self):
        """Lane 5 11-03, EP0 on Spark 1.3: the chip read Glimmer, the studio's default, and the banner put the run on
        the card."""
        replay.pack(CHAT, self.out, card="RTX 5070 Ti 16 GB")
        data = json.loads(self.read("api", "replay.json"))
        self.assertEqual([m["label"] for m in data["health"]["minds"]], ["Muse Glimmer"])
        self.assertEqual((data["health"]["default"], data["card"], data["api"]), ("local", "RTX 5070 Ti 16 GB", []))
        self.events[2]["mind"] = "Muse Spark"
        self.write(self.events)
        other = os.path.join(self.tmp, "api_run")
        replay.pack(CHAT, other, card="RTX 5070 Ti 16 GB")
        with open(os.path.join(other, "api", "replay.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([(m["label"], m["ok"]) for m in data["health"]["minds"]], [("Muse Spark", True)])
        self.assertEqual((data["health"]["default"], data["card"], data["api"]), ("spark", "", ["Muse Spark"]))
        with open(os.path.join(other, "replay.js"), encoding="utf-8") as f:
            said = f.read()
        self.assertIn('" with " + d.api.join(" and ") + " through an API"', said)
        self.assertIn('+ api + "; it plays here faster', said, "the banner says it")

    def test_a_mind_stopped_before_it_answered_is_not_credited(self):
        """Lane 4 11-03, Vlog 7c21: job 1 on Glimmer was stopped 30 s in with no step, the rest ran on Baseten, and the
        banner put the run on the card."""
        self.events[2]["mind"] = "Muse Spark"
        stopped = [{"job": 0, "text": "make a vlog", "id": 10, "kind": "person", "t": 90.0},
                   {"job": 0, "mind": "Muse Glimmer", "id": 11, "kind": "job", "t": 90.5},
                   {"job": 0, "steps": 0, "seconds": 30.0, "why": "stopped", "id": 12, "kind": "end", "t": 99.0}]
        self.write(stopped + self.events)
        replay.pack(CHAT, self.out, card="RTX 5070 Ti 16 GB")
        data = json.loads(self.read("api", "replay.json"))
        self.assertEqual([m["label"] for m in data["health"]["minds"]], ["Muse Spark"])
        self.assertEqual((data["card"], data["api"]), ("", ["Muse Spark"]))

    def test_the_replay_answers_what_its_own_run_cost(self):
        """Lane 5 11-03: the page asked api/spent about 8 times and got a 404 each time. The day and the month are the
        run's: what else the computer ran that day is not the replay's to show."""
        rows = [{"t": 200.0, "chat": CHAT, "job": 1, "label": "Muse Spark", "usd": 0.01},
                {"t": 300.0, "chat": CHAT, "job": 1, "label": "Muse Spark", "usd": 0.02},
                {"t": 250.0, "chat": "20261027-150000-0000", "job": 1, "label": "Muse Wonder", "usd": 0.5}]
        with open(os.path.join(agent.WORK, "spent.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        replay.pack(CHAT, self.out, card="")
        got = json.loads(self.read("api", "replay.json"))["spent"]
        for key in ("chat", "today", "month", "all"):
            self.assertAlmostEqual(got[key], 0.03, msg=key)
        self.assertEqual([(j["job"], round(j["usd"], 2), j["answers"], j["minds"]) for j in got["jobs"]],
                         [(1, 0.03, 2, ["Muse Spark"])])
        self.assertEqual(list(got["minds"]), ["Muse Spark"])
        self.assertEqual((got["ok"], got["rate"]), (True, {"currency": "USD", "per_usd": 1.0}))
        page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "replay.js")
        with open(page, encoding="utf-8") as f:
            self.assertIn('if (at[1] === "spent" && d.spent) return answer(d.spent);', f.read())

    def test_a_run_whose_pictures_are_gone_or_whose_job_has_not_ended_is_not_packed(self):
        os.remove(os.path.join(previews.folder(), PREVIEW))
        with self.assertRaises(replay.Refused) as said:
            replay.pack(CHAT, self.out, card="")
        self.assertIn(PREVIEW, str(said.exception))
        self.write(self.events[:-1])
        with self.assertRaises(replay.Refused) as said:
            replay.pack(CHAT, os.path.join(self.tmp, "other"), card="")
        self.assertIn("has not finished", str(said.exception))
        with open(os.path.join(recording.WORK, "chats", CHAT + ".events.jsonl"), encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), len(self.events) - 1, "packing never writes an end into a run that is going on")


STEPPER = r"""
const fs = require("fs");
const vm = require("vm");
class El {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.attrs = {};
    this.heard = {};
    this.textContent = "";
    this.disabled = false;
    this.style = {};
    const set = new Set();
    this.classList = { add: (c) => set.add(c), remove: (c) => set.delete(c), contains: (c) => set.has(c),
      toggle: (c, on) => (on === undefined ? (set.has(c) ? set.delete(c) : set.add(c)) : on ? set.add(c) : set.delete(c)) };
  }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(k, fn) { (this.heard[k] = this.heard[k] || []).push(fn); }
  append(...k) { this.children.push(...k); }
  prepend(...k) { this.children.unshift(...k); }
  click() { (this.heard.click || []).forEach((fn) => fn({})); }
}
const keys = {};
const chat = new El("div");
const document = { createElement: (t) => new El(t), head: new El("head"), querySelector: () => chat,
  querySelectorAll: () => [], addEventListener: (k, fn) => { keys[k] = fn; } };
const events = ["opened", "person", "job", "step", "plan", "step", "step", "end"].map((kind, t) => ({ kind, t }));
const got = [];
const timers = [];
const box = { document, console, JSON, Object, Math, Date, Promise, Infinity, String, Number, Set, URL, Response: function () {},
  location: { href: "http://x/", hash: process.argv[3] || "", reload: () => got.push("reload") }, setTimeout: (fn) => (timers.push(fn), timers.length), clearTimeout: () => {} };
const heard = {};
box.window = { addEventListener: (k, fn) => { heard[k] = fn; }, fetch: () => Promise.resolve({ json: () => Promise.resolve({ events, took: 7, made: "3 Nov 2026", card: "", api: [] }) }) };
box.window.location = box.location;
vm.createContext(box);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), box);
let source = null;
box.room = { chat: "c" };
box.show = () => {
  if (source) source.close();
  source = new box.window.EventSource();
  source.onmessage = (m) => got.push(JSON.parse(m.data).kind);
};
const tick = () => new Promise((r) => setTimeout(r, 5));
(async () => {
  const said = {};
  const [bar, go] = chat.children;
  const [back, pause, forth, where] = go.children;
  const end = bar.children[2];
  const look = () => [where.textContent, got.splice(0).join(" "), pause.textContent, back.disabled, forth.disabled];
  if (box.location.hash) {
    box.show();
    await tick();
    said.opened = look();
    for (const hash of ["#step=1", "#step=9", "#top"]) {
      box.location.hash = hash;
      heard.hashchange({});
      await tick();
      said[hash] = look();
    }
    let kept = 0;
    box.location.hash = "#step=1";
    heard.popstate({ state: null, stopImmediatePropagation: () => (kept += 1) });
    heard.popstate({ state: { screen: "chat" }, stopImmediatePropagation: () => (kept += 10) });
    box.location.hash = "#top";
    heard.popstate({ state: null, stopImmediatePropagation: () => (kept += 100) });
    said.kept = kept;
    return console.log(JSON.stringify(said));
  }
  box.show();
  await tick();
  said.start = look();
  pause.click();
  timers[timers.length - 1]();
  said.paused = look();
  forth.click();
  await tick();
  said.next = look();
  forth.click();
  await tick();
  said.next2 = look();
  back.click();
  await tick();
  said.back = look();
  const pressed = { prevented: 0, preventDefault() { this.prevented += 1; }, stopImmediatePropagation() {}, key: "ArrowRight", target: { closest: () => null } };
  keys.keydown(pressed);
  await tick();
  said.arrow = look().concat(pressed.prevented);
  const typing = { prevented: 0, preventDefault() { this.prevented += 1; }, stopImmediatePropagation() {}, key: "ArrowLeft", target: { closest: () => ({}) } };
  keys.keydown(typing);
  await tick();
  said.typing = look().concat(typing.prevented);
  end.click();
  await tick();
  said.end = look();
  console.log(JSON.stringify(said));
})();
"""


THINKER = r"""
const fs = require("fs");
const vm = require("vm");
let clock = 1000;
const timers = [];
const el = () => ({ children: [], heard: {}, style: {}, textContent: "", disabled: false,
  classList: { toggle() {}, add() {}, remove() {}, contains: () => false }, setAttribute() {},
  addEventListener(k, fn) { (this.heard[k] = this.heard[k] || []).push(fn); },
  append(...k) { this.children.push(...k); }, prepend(...k) { this.children.unshift(...k); },
  click() { (this.heard.click || []).forEach((fn) => fn({})); } });
const chat = el();
const thought = { open: false };
const row = { querySelector: (q) => (q === "details.thought" ? thought : null) };
let rows = [];
const document = { createElement: el, head: el(), querySelector: () => chat, querySelectorAll: () => rows,
  addEventListener() {} };
const words = "x".repeat(300);
const events = [{ kind: "opened" }, { kind: "job", job: 1 }, { kind: "wait", job: 1 },
  { kind: "step", job: 1, step: 1, thinking: words }, { kind: "wait", job: 1 },
  { kind: "step", job: 1, step: 2, thinking: "", said: "y".repeat(50) }, { kind: "step", job: 1, step: 3, thinking: "" },
  { kind: "end", job: 1 }].map((e) => Object.assign(e, { t: 0 }));
const box = { document, console, JSON, Object, Math, Promise, Infinity, String, Number, Set, URL, Response: function () {},
  Date: { now: () => clock }, location: { href: "http://x/", hash: process.argv[3] || "", reload() {} },
  setTimeout: (fn, ms) => (timers.push({ fn, ms: ms || 0 }), timers.length),
  clearTimeout: (n) => { if (timers[n - 1]) timers[n - 1].fn = null; } };
box.window = { addEventListener() {}, fetch: () => Promise.resolve({ json: () => Promise.resolve({ events, took: 7,
  made: "3 Nov 2026", card: "", api: [] }) }) };
box.window.location = box.location;
vm.createContext(box);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), box);
const got = [];
const opened = [];
const at = {};
box.room = { chat: "c", jobs: { 1: { rows: { contains: (r) => r === row } } } };
box.setOpen = (j, on) => opened.push(on);
box.show = () => {
  const source = new box.window.EventSource();
  source.onmessage = (m) => {
    const e = JSON.parse(m.data);
    got.push(e.kind + (e.step || ""));
    if (e.kind === "step") rows = [row];
    if (e.kind === "step" && e.step === 1) at.step = clock;
  };
  source.addEventListener("think", (m) => {
    const n = JSON.parse(m.data).thinking.length;
    got.push("think:" + n);
    if (n === 300 && at.whole === undefined) at.whole = clock;
  });
};
const due = (until) => {
  for (let k = 0; k < 1000; k++) {
    const next = timers.find((t) => t.fn);
    if (!next || (until && until())) return;
    const fn = next.fn;
    next.fn = null;
    clock += next.ms;
    fn();
  }
};
(async () => {
  const [, go] = chat.children;
  const pause = go.children[1];
  box.show();
  await new Promise((r) => setTimeout(r, 5));
  const said = {};
  if (process.argv[4] === "pause") {
    due(() => got.some((g) => g.startsWith("think:") && g !== "think:0"));
    pause.click();
    due();
    said.paused = got.splice(0);
    pause.click();
    await new Promise((r) => setTimeout(r, 5));
    due();
    said.played = got.splice(0);
  } else {
    due();
    said.got = got.splice(0);
  }
  said.held = at.step - at.whole;
  said.open = thought.open;
  said.setOpen = opened;
  console.log(JSON.stringify(said));
})();
"""


def _thinker(*args):
    import subprocess
    page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "replay.js")
    run = subprocess.run(["node", "-e", THINKER, "-", page] + list(args), capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    if run.returncode:
        raise AssertionError(run.stderr[-2000:])
    return json.loads(run.stdout.strip().splitlines()[-1])


@unittest.skipUnless(shutil.which("node"), "node is not on this computer")
class Thinking(unittest.TestCase):
    """Tarm 11-03 through lane 5: "replay ขอแบบอ่าน thinking ได้ด้วยเหมือน กรรมการนั่งเรนเดอร์เองเลย". A step's thinking types
    into the job's live box before the step lands, as the run showed it."""

    def test_each_steps_thinking_types_in_before_the_step_lands(self):
        got = _thinker("", "play")["got"]
        typed = [g for g in got[:got.index("step1")] if g.startswith("think:")]
        self.assertEqual(got[:3], ["opened", "job", "wait"])
        self.assertGreater(len(typed), 20, "it types, it does not land at once")
        self.assertEqual(typed[-1], "think:300")
        self.assertEqual([int(g[6:]) for g in typed], sorted(int(g[6:]) for g in typed))
        after = got[got.index("step1"):]
        self.assertEqual([g for g in after if not g.startswith("think:")], ["step1", "wait", "step2", "step3", "end"])
        self.assertEqual(after[after.index("step2") - 1], "think:50", "Spark on Meta's API sends no thinking: its words type")
        self.assertEqual(after[after.index("step2") + 1], "step3", "a step with neither types nothing")
        self.assertGreaterEqual(_thinker("", "play")["held"], 1000, "the whole thinking stays a moment before the step")

    def test_a_pause_shows_the_thinking_whole_and_play_goes_on_without_typing_it_again(self):
        said = _thinker("", "pause")
        first = next(k for k, g in enumerate(said["paused"]) if g.startswith("think:") and g != "think:0")
        self.assertEqual(said["paused"][first + 1:], ["think:300"], "a pause shows it whole at once")
        self.assertNotIn("step1", said["paused"])
        self.assertEqual(said["played"][:2], ["step1", "wait"], "the step lands at once, its thinking not typed again")
        self.assertEqual([g for g in said["played"] if not g.startswith("think:")], ["step1", "wait", "step2", "step3", "end"])

    def test_a_step_stopped_on_opens_how_muse_decided(self):
        said = _thinker("#step=1", "play")
        self.assertEqual(said["got"], ["opened", "job", "wait", "step1", "wait"], "going to a step types nothing")
        self.assertTrue(said["open"])
        self.assertEqual(said["setOpen"], [True])


@unittest.skipUnless(shutil.which("node"), "node is not on this computer")
class Steps(unittest.TestCase):
    """Tarm 11-03: "ส่วน replay ควรมีปุ่ม next หรืออะไรแบบนี้". A judge goes step by step, back and forth, or lets it play."""

    @classmethod
    def setUpClass(cls):
        import subprocess
        page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "replay.js")
        run = subprocess.run(["node", "-e", STEPPER, "-", page], capture_output=True, text=True, encoding="utf-8", timeout=60)
        if run.returncode:
            raise AssertionError(run.stderr[-2000:])
        cls.said = json.loads(run.stdout.strip().splitlines()[-1])

    def test_it_plays_by_itself_until_paused(self):
        self.assertEqual(self.said["start"], ["Step 0 of 3", "opened", "Pause", True, False])
        self.assertEqual(self.said["paused"], ["Step 0 of 3", "", "Play", True, False])

    def test_next_shows_one_step_and_what_follows_it_and_stops_before_the_next(self):
        self.assertEqual(self.said["next"], ["Step 1 of 3", "person job step plan", "Play", False, False])
        self.assertEqual(self.said["next2"], ["Step 2 of 3", "step", "Play", False, False])

    def test_back_plays_the_run_again_to_one_step_before(self):
        self.assertEqual(self.said["back"], ["Step 1 of 3", "opened person job step plan", "Play", False, False])

    def test_the_arrows_step_too_but_not_while_typing(self):
        self.assertEqual(self.said["arrow"], ["Step 2 of 3", "step", "Play", False, False, 1])
        self.assertEqual(self.said["typing"], ["Step 2 of 3", "", "Play", False, False, 0])

    def test_the_step_controls_stay_at_the_top_while_the_run_scrolls(self):
        page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "replay.js")
        with open(page, encoding="utf-8") as f:
            self.assertIn(".replay-go{position:sticky;top:0;", f.read())

    def test_show_the_end_plays_the_rest(self):
        self.assertEqual(self.said["end"], ["Step 3 of 3", "step end", "Play", False, True])


@unittest.skipUnless(shutil.which("node"), "node is not on this computer")
class StepAddress(unittest.TestCase):
    """Tarm 11-03 through lane 5: beside each time in the video's chapters, a link that opens the replay at that step.
    Step n is the n-th step event of the chat, every job together, in the order they were written."""

    @classmethod
    def setUpClass(cls):
        import subprocess
        page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "replay.js")
        run = subprocess.run(["node", "-e", STEPPER, "-", page, "#step=2"], capture_output=True, text=True,
                             encoding="utf-8", timeout=60)
        if run.returncode:
            raise AssertionError(run.stderr[-2000:])
        cls.said = json.loads(run.stdout.strip().splitlines()[-1])

    def test_an_address_with_a_step_opens_the_replay_paused_there_as_next_pressed_up_to_it_would(self):
        self.assertEqual(self.said["opened"], ["Step 2 of 3", "opened person job step plan step", "Play", False, False])

    def test_another_step_on_the_open_page_plays_the_run_again_up_to_it(self):
        self.assertEqual(self.said["#step=1"], ["Step 1 of 3", "opened person job step plan", "Play", False, False])

    def test_a_step_past_the_end_shows_the_end(self):
        self.assertEqual(self.said["#step=9"], ["Step 3 of 3", "opened person job step plan step step end", "Play",
                                                False, True])

    def test_an_address_that_is_not_a_step_changes_nothing(self):
        self.assertEqual(self.said["#top"], ["Step 3 of 3", "", "Play", False, True])

    def test_a_new_step_address_keeps_the_chat_on_screen(self):
        """The page goes home on a history move with no screen of its own, and a new hash is one: real Chrome 11-03,
        the run played to step 2 behind the home screen."""
        self.assertEqual(self.said["kept"], 1)



if __name__ == "__main__":
    unittest.main()
