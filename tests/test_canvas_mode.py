"""The Layout sheet as a canvas: the clip stays still, a part is dragged or sized on the picture, Pan moves what a part
or a shot shows, and the crop is what the render cuts. The page runs from web/app.js itself in node on a small
stand-in. python -m unittest discover -s tests -p test_canvas_mode.py"""
import json
import os
import re
import shutil
import subprocess
import unittest
from unittest import mock

import test_editor
from clipstudio import agent, layouts, server

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

STAND_IN = r"""
const fs = require("fs");
const vm = require("vm");

function any(own) {
  const t = Object.assign(function () {}, own);
  return new Proxy(t, {
    get(o, k) {
      if (k in o) return o[k];
      if (k === Symbol.toPrimitive) return () => 0;
      if (k === Symbol.iterator) return function* () {};
      if (typeof k === "symbol" || k === "then") return undefined;
      return (o[k] = any());
    },
    apply: () => any(),
    construct: () => any(),
  });
}

class Node {
  constructor(tag) {
    this.tag = tag;
    this.className = "";
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.textContent = "";
    this.attrs = {};
    this.heard = {};
    const node = this;
    this.classList = {
      add: (...k) => k.forEach((c) => node.classList.toggle(c, true)),
      remove: (...k) => k.forEach((c) => node.classList.toggle(c, false)),
      contains: (c) => node.className.split(" ").includes(c),
      toggle: (c, on) => {
        const have = node.className.split(" ").filter((x) => x && x !== c);
        node.className = (on ? have.concat(c) : have).join(" ");
      },
    };
  }
  addEventListener(k, fn) { (this.heard[k] = this.heard[k] || []).push(fn); }
  removeEventListener() {}
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] === undefined ? null : this.attrs[k]; }
  append(...kids) { kids.forEach((k) => this.appendChild(k)); }
  appendChild(k) { if (k instanceof Node) this.children.push(k); return k; }
  prepend(...kids) { this.children.unshift(...kids.filter((k) => k instanceof Node)); }
  replaceChildren(...kids) { this.children = kids.filter((k) => k instanceof Node); }
  remove() {}
  closest() { return null; }
  get tagName() { return this.tag.toUpperCase(); }
  get firstElementChild() { return this.children[0] || new Node("i"); }
  focus() {}
  getBoundingClientRect() { return this.rect || { left: 0, top: 0, width: 100, height: 100 }; }
  querySelectorAll() { return []; }
  querySelector(q) { return this.pick ? this.pick(q) : null; }
}

function load(page) {
  const found = {};
  const on = {};
  const hear = (name) => (k, fn) => ((on[name + " " + k] = on[name + " " + k] || []).push(fn));
  const later = [];
  const posted = [];
  const video = { paused: false, ended: false, duration: 20, currentTime: 2, classList: { contains: () => false }, heard: {}, addEventListener(k, fn) { (this.heard[k] = this.heard[k] || []).push(fn); }, play() { this.paused = false; return Promise.resolve(); }, pause() { this.paused = true; } };
  const element = (id) => {
    if (found[id]) return found[id];
    const n = new Node("div");
    if (id === "frame") n.pick = (q) => (q.startsWith("video") && !video.none ? video : null);
    if (id === "watch" || id === "sheet" || id === "canvasTools" || id === "partBoxes") n.hidden = true;
    if (id === "canvasDrag" || id === "canvasPan") n.closest = () => found[id];
    if (id === "lanes") n.pick = (q) => (q === ".playhead" ? new Node("i") : null);
    return (found[id] = new Proxy(n, {
      get(o, k) {
        if (k in o) return typeof o[k] === "function" ? o[k].bind(o) : o[k];
        if (typeof k === "symbol" || k === "then") return undefined;
        return (o[k] = any());
      },
    }));
  };
  const box = {
    Event: class { constructor(type) { this.type = type; } },
    console, Math, Number, String, Object, Array, JSON, Promise, Date, Set, Map, RegExp, Infinity, NaN, parseInt, parseFloat, isNaN, encodeURIComponent, decodeURIComponent,
    document: any({ getElementById: element, addEventListener: hear("document"), createElement: (tag) => new Node(tag), createElementNS: (ns, tag) => new Node(tag), querySelectorAll: () => [], querySelector: () => null, body: any() }),
    window: any({ addEventListener: hear("window"), removeEventListener: (k, fn) => { on["window " + k] = (on["window " + k] || []).filter((f) => f !== fn); }, matchMedia: () => any({ matches: false }) }),
    history: any(), navigator: any(), localStorage: any({ getItem: () => null }), sessionStorage: any({ getItem: () => null }),
    fetch: (path, opts) => {
      if (!opts || opts.method !== "POST") return new Promise(() => {});
      posted.push({ path, body: JSON.parse(opts.body) });
      return Promise.resolve({ json: () => Promise.resolve(box.answer || { ok: true }) });
    },
    EventSource: function () { return any(); }, ResizeObserver: function () { return any(); }, IntersectionObserver: function () { return any(); }, MutationObserver: function () { return any(); },
    getComputedStyle: () => ({ display: "block" }), requestAnimationFrame: () => 0, cancelAnimationFrame: () => {}, setTimeout: (fn) => (later.push(fn), 0), clearTimeout: () => {}, setInterval: () => 0, clearInterval: () => {},
    Intl: any(), URL: any(), Image: function () { return any(); }, FormData: function () { return any(); }, CSS: any(), performance: any({ now: () => 0 }), innerWidth: 1280, innerHeight: 800, scrollTo: () => {},
  };
  vm.createContext(box);
  vm.runInContext(page + "\n;globalThis.t = { room, clipAt, openSheet, closeSheet, togglePlay, wirePlayer, drawFrameParts, showClip, hidePanel, playButton, bigPlay, fullButton, watchBox, clipBox, sheet, stepClip, movePlayhead };", box);
  const t = box.t;
  t.room.chat = "c";
  t.room.screen = "clip";
  t.room.panel = { length: 20, parts: [], crops: [], frame: { why: "no frame" } };
  const key = (k) => {
    const e = { key: k, target: { closest: () => null }, repeat: false, ctrlKey: false, altKey: false, metaKey: false, shiftKey: false, prevented: false, preventDefault() { this.prevented = true; } };
    on["document keydown"].forEach((fn) => fn(e));
    return e.prevented;
  };
  const click = (id) => element(id).heard.click.forEach((fn) => fn({ target: element(id) }));
  const boxes = () => element("partBoxes").children;
  const down = (node, x, y, inner) => node.heard.pointerdown.forEach((fn) => fn({ clientX: x, clientY: y, target: inner || node, preventDefault() {} }));
  const move = (x, y) => (on["window pointermove"] || []).slice().forEach((fn) => fn({ clientX: x, clientY: y }));
  const up = () => (on["window pointerup"] || []).slice().forEach((fn) => fn({}));
  const settle = () => new Promise((r) => setImmediate(r));
  return { t, box, video, element, later, posted, key, click, boxes, down, move, up, settle };
}

const CARD = { name: "card", box: [50, 400, 950, 750], src: [100, 0, 1400, 1080], size: [1920, 1080], follows: false };
const CIRCLE = { name: "circle", box: [650, 700, 950, 870], src: [1600, 400, 200, 200], size: [1920, 1080], follows: false };
const part = (c) => ({ name: c.name, box: c.box });
const SHOTS = { shots: [{ from: 0, to: 4, show: 2, x: 590, y: 10, moved: false }, { from: 4, to: 8, show: 0, x: null, y: null, moved: false }], people: [], size: [1920, 1080], width: 576, height: 1026, why: null };
const panel = (over) => Object.assign({ length: 20, parts: [part(CARD), part(CIRCLE)], crops: [CARD, CIRCLE], frame: { why: "no frame" } }, over);
const pressed = (p) => ({ drag: p.element("canvasDrag").getAttribute("aria-pressed"), pan: p.element("canvasPan").getAttribute("aria-pressed") });
const state = (p) => ({ canvas: p.t.room.canvas, tools: !p.element("canvasTools").hidden, boxes: p.element("partBoxes").hidden ? [] : p.boxes().map((n) => n.children.find((k) => k.tag === "span").textContent), drag: p.element("canvasDrag").disabled, pan: p.element("canvasPan").disabled, says: p.element("canvasSays").textContent, row: !p.element("partRow").hidden });
const open = (p, d) => {
  p.t.room.panel = d;
  p.t.openSheet("layout");
};

const said = {};
const page = fs.readFileSync(process.argv[2], "utf8");
const steps = [];
const each = (name, fn) => steps.push(async () => { said[name] = await fn(load(page)); });

each("still", (p) => {
  p.t.wirePlayer(p.video);
  open(p, panel());
  const stopped = p.video.paused;
  const at = p.video.currentTime;
  const clip = p.t.clipBox.classList.contains("canvas");
  const off = { play: p.t.playButton.disabled, big: p.t.bigPlay.hidden, full: p.t.fullButton.hidden };
  p.video.heard.click.forEach((fn) => fn({}));
  p.click("play");
  p.click("bigPlay");
  const clicks = !p.video.paused;
  p.key(" ");
  const space = !p.video.paused;
  p.key("ArrowRight");
  const stepped = p.video.currentTime - at;
  const still = p.video.paused;
  p.t.closeSheet();
  const back = { play: p.t.playButton.disabled, big: p.t.bigPlay.hidden, full: p.t.fullButton.hidden, clip: p.t.clipBox.classList.contains("canvas"), canvas: p.t.room.canvas };
  p.click("bigPlay");
  return { stopped, clip, off, clicks, space, stepped, still, back, plays: !p.video.paused };
});
each("sheets", (p) => {
  const shown = {};
  p.t.room.panel = panel();
  ["frame", "style", "layout"].forEach((name) => {
    p.t.openSheet(name);
    shown[name] = state(p);
  });
  p.t.closeSheet();
  shown.closed = state(p);
  return shown;
});
each("older clip", (p) => {
  const w = p.t.watchBox;
  w.hidden = false;
  p.t.clipBox.classList.add("watching");
  open(p, panel());
  return { watching: !w.hidden, played: p.t.clipBox.classList.contains("watching") };
});
each("no video", (p) => {
  p.video.none = true;
  p.t.room.made = { clip: { length: 20 }, version: 1 };
  open(p, panel());
  return { tools: state(p).tools, play: p.t.playButton.disabled };
});
each("switch", (p) => {
  open(p, panel());
  const first = pressed(p);
  const dragging = state(p);
  p.element("canvasTools").heard.click.forEach((fn) => fn({ target: p.element("canvasPan") }));
  const second = pressed(p);
  const panning = state(p);
  p.t.closeSheet();
  p.t.openSheet("layout");
  return { first, dragging, second, panning, again: pressed(p) };
});
each("whole picture", (p) => {
  open(p, panel({ crops: [] }));
  return state(p);
});
each("pan runs out", (p) => {
  open(p, panel());
  p.t.room.tool = "pan";
  p.t.room.panel = panel({ crops: [] });
  p.t.drawFrameParts(p.t.room.panel);
  return pressed(p);
});
each("one part pans", (p) => {
  open(p, panel({ crops: [CIRCLE] }));
  p.t.room.tool = "pan";
  p.t.drawFrameParts(p.t.room.panel);
  return state(p);
});
each("follows", (p) => {
  open(p, panel({ parts: [part(CARD)], crops: [Object.assign({}, CARD, { follows: true })] }));
  p.t.room.tool = "pan";
  p.t.drawFrameParts(p.t.room.panel);
  return state(p);
});
each("talk", (p) => {
  p.video.currentTime = 1;
  open(p, panel({ parts: [], crops: [], frame: SHOTS }));
  const first = Object.assign(state(p), { pressed: pressed(p) });
  p.video.currentTime = 5;
  p.t.movePlayhead(5);
  const everyone = state(p);
  p.video.currentTime = 1;
  p.t.movePlayhead(1);
  return { first, everyone, back: state(p) };
});
each("pan a part", async (p) => {
  open(p, panel());
  p.t.room.tool = "pan";
  p.t.drawFrameParts(p.t.room.panel);
  const node = p.boxes().find((n) => n.children.some((k) => k.textContent === "Card"));
  node.rect = { left: 0, top: 0, width: 200, height: 100 };
  p.down(node, 50, 50);
  const pic = node.children.find((k) => k.tag === "img");
  const at = { width: pic.style.width, height: pic.style.height, left: pic.style.left, top: pic.style.top, src: pic.src };
  p.move(40, 50);
  const moved = pic.style.left;
  p.t.drawFrameParts(p.t.room.panel);
  const kept = p.boxes().includes(node);
  p.up();
  await p.settle();
  return { at, moved, kept, posted: p.posted };
});
each("pan stops at the edge", async (p) => {
  open(p, panel());
  p.t.room.tool = "pan";
  p.t.drawFrameParts(p.t.room.panel);
  const node = p.boxes().find((n) => n.children.some((k) => k.textContent === "Circle"));
  p.down(node, 50, 50);
  p.move(-5000, 5000);
  p.up();
  await p.settle();
  return p.posted;
});
each("a tap pans nothing", async (p) => {
  open(p, panel());
  p.t.room.tool = "pan";
  p.t.drawFrameParts(p.t.room.panel);
  const node = p.boxes()[0];
  p.down(node, 50, 50);
  p.up();
  await p.settle();
  return { posted: p.posted, back: state(p).boxes };
});
each("pan a shot", async (p) => {
  p.video.currentTime = 1;
  open(p, panel({ parts: [], crops: [], frame: SHOTS }));
  const node = p.boxes()[0];
  node.rect = { left: 0, top: 0, width: 288, height: 513 };
  p.down(node, 50, 50);
  p.move(0, 50);
  p.up();
  await p.settle();
  return p.posted;
});
each("drag a part", async (p) => {
  open(p, panel());
  const node = p.boxes().find((n) => n.children.some((k) => k.textContent === "Card"));
  p.down(node, 50, 50);
  p.move(53, 50);
  p.t.drawFrameParts(p.t.room.panel);
  const kept = p.boxes().includes(node);
  p.up();
  await p.settle();
  return { posted: p.posted, kept };
});
each("size a part", async (p) => {
  open(p, panel());
  const node = p.boxes().find((n) => n.children.some((k) => k.textContent === "Circle"));
  const corner = node.children.find((k) => k.tag === "i");
  p.down(node, 50, 50, corner);
  p.move(55, 50);
  p.up();
  await p.settle();
  return p.posted;
});
each("refused", async (p) => {
  open(p, panel());
  p.box.answer = { ok: false, error: "box is too small to show a part" };
  const node = p.boxes()[0];
  p.down(node, 50, 50);
  p.move(60, 50);
  p.up();
  await p.settle();
  await p.settle();
  p.t.drawFrameParts(p.t.room.panel);
  const why = p.element("partWhy");
  const refused = { text: why.textContent, hidden: why.hidden };
  p.box.answer = null;
  const again = p.boxes()[0];
  p.down(again, 50, 50);
  p.move(60, 50);
  p.up();
  await p.settle();
  await p.settle();
  p.t.drawFrameParts(p.t.room.panel);
  return { refused, cleared: why.hidden };
});
each("making a layout", (p) => {
  open(p, panel());
  p.t.room.work.layout = true;
  p.t.drawFrameParts(p.t.room.panel);
  return state(p);
});
each("the clip goes away", (p) => {
  open(p, panel());
  p.t.room.panel = null;
  p.t.hidePanel();
  return state(p);
});
each("the first change keeps the moment", (p) => {
  p.video.currentTime = 7.5;
  p.t.clipAt.value = 10;
  p.t.showClip({ id: 1, kind: "edit", entry: { do: "layout", part: "card" }, clip: { length: 20 } });
  return Number(p.t.clipAt.value);
});

(async () => {
  for (const step of steps) await step();
  console.log(JSON.stringify(said));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def said_by_the_page():
    run = subprocess.run(["node", "-", os.path.join(WEB, "app.js")], input=STAND_IN, capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    if run.returncode:
        raise AssertionError(run.stderr[-3000:])
    return json.loads(run.stdout.strip().splitlines()[-1])


@unittest.skipUnless(shutil.which("node"), "node runs the page")
class Canvas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.said = said_by_the_page()

    def test_with_layout_open_nothing_plays_the_clip(self):
        s = self.said["still"]
        self.assertTrue(s["stopped"], "the clip stops when Layout opens")
        self.assertTrue(s["clip"])
        self.assertEqual(s["off"], {"play": True, "big": True, "full": True})
        self.assertFalse(s["clicks"], "not the picture, the play button or the big one")
        self.assertFalse(s["space"], "and not Space")

    def test_the_arrows_still_walk_the_clip_and_do_not_start_it(self):
        s = self.said["still"]
        self.assertAlmostEqual(s["stepped"], 0.25)
        self.assertTrue(s["still"])

    def test_leaving_layout_brings_play_back(self):
        s = self.said["still"]
        self.assertEqual(s["back"], {"play": False, "big": False, "full": False, "clip": False, "canvas": False})
        self.assertTrue(s["plays"])

    def test_the_parts_show_on_the_picture_on_layout_only(self):
        """Henri's click did nothing: the hint sat on the Layout sheet and the boxes showed on the Frame sheet."""
        s = self.said["sheets"]
        self.assertEqual(s["layout"]["boxes"], ["Card", "Circle"], "the biggest under the smallest")
        self.assertTrue(s["layout"]["tools"])
        for name in ("frame", "style", "closed"):
            self.assertEqual(s[name]["boxes"], [], name)
            self.assertFalse(s[name]["tools"], name)
            self.assertFalse(s[name]["canvas"], name)

    def test_layout_goes_back_to_the_newest_clip(self):
        self.assertEqual(self.said["older clip"], {"watching": False, "played": False})

    def test_a_still_picture_is_a_canvas_too(self):
        self.assertEqual(self.said["no video"], {"tools": True, "play": True})

    def test_drag_is_the_default_and_pan_is_one_click_away(self):
        s = self.said["switch"]
        self.assertEqual(s["first"], {"drag": "true", "pan": "false"})
        self.assertEqual(s["second"], {"drag": "false", "pan": "true"})
        self.assertEqual(s["again"], {"drag": "true", "pan": "false"}, "each time Layout opens")
        self.assertIn("corner", s["dragging"]["says"])
        self.assertTrue(s["dragging"]["row"])
        self.assertIn("It stays where you let go, and Undo takes it back.", s["panning"]["says"])
        self.assertFalse(s["panning"]["row"], "Bigger and Up move the part, not its picture")

    def test_a_layout_with_nothing_to_pan_says_so_and_the_button_is_off(self):
        s = self.said["whole picture"]
        self.assertTrue(s["pan"])
        self.assertFalse(s["drag"])
        self.assertIn("nothing to pan", s["says"])

    def test_when_nothing_is_left_to_pan_the_tool_goes_back_to_drag(self):
        self.assertEqual(self.said["pan runs out"], {"drag": "true", "pan": "false"})

    def test_pan_shows_only_the_parts_that_can_pan_and_names_the_rest(self):
        s = self.said["one part pans"]
        self.assertEqual(s["boxes"], ["Circle"])
        self.assertIn("Card already shows the whole picture.", s["says"])

    def test_a_part_that_moves_by_itself_says_pan_holds_it(self):
        self.assertIn("Card moves by itself now, and holds still once you pan it.", self.said["follows"]["says"])

    def test_a_talk_clip_pans_the_shot_on_screen(self):
        s = self.said["talk"]
        self.assertEqual(s["first"]["boxes"], ["Person 2"])
        self.assertEqual((s["first"]["drag"], s["first"]["pan"]), (True, False), "it has no parts to drag")
        self.assertEqual(s["first"]["pressed"], {"drag": "false", "pan": "true"})
        self.assertIn("from 0:00 to 0:04", s["first"]["says"], "it says which stretch changes")
        self.assertEqual(s["everyone"]["boxes"], [])
        self.assertTrue(s["everyone"]["pan"])
        self.assertIn("This shot shows everyone", s["everyone"]["says"])
        self.assertEqual(s["back"]["boxes"], ["Person 2"], "the buttons follow the clip's moment")

    def test_pan_moves_the_picture_under_the_hand(self):
        s = self.said["pan a part"]
        self.assertAlmostEqual(float(s["at"]["width"][:-1]), 100 * 1920 / 1400)
        self.assertAlmostEqual(float(s["at"]["height"][:-1]), 100.0)
        self.assertAlmostEqual(float(s["at"]["left"][:-1]), -100 * 100 / 1400)
        self.assertIn("/api/shotframe?chat=c&t=2.00", s["at"]["src"], "the recording at this moment")
        self.assertAlmostEqual(float(s["moved"][:-1]), -100 * 170 / 1400, msg="10 of 200 px is 70 of 1400")
        self.assertTrue(s["kept"], "a redraw does not take the box from under the hand")
        self.assertEqual(s["posted"], [{"path": "/api/part", "body": {"chat": "c", "part": "card", "at": [170, 0]}}])

    def test_pan_stops_at_the_edge_of_the_recording(self):
        self.assertEqual(self.said["pan stops at the edge"][0]["body"]["at"], [1720, 0])

    def test_a_tap_pans_nothing(self):
        self.assertEqual(self.said["a tap pans nothing"]["posted"], [])
        self.assertTrue(self.said["a tap pans nothing"]["back"])

    def test_a_shot_pans_with_its_own_stretch(self):
        self.assertEqual(self.said["pan a shot"], [{"path": "/api/shot", "body": {
            "chat": "c", "start": 0, "end": 4, "show": 2, "x": 690, "y": 10}}])

    def test_drag_and_the_corner_still_move_and_size_a_part(self):
        self.assertEqual(self.said["drag a part"]["posted"], [{"path": "/api/part", "body": {
            "chat": "c", "part": "card", "box": [80, 400, 980, 750]}}])
        self.assertTrue(self.said["drag a part"]["kept"], "a redraw does not take the box from under the hand")
        self.assertEqual(self.said["size a part"][0]["body"]["box"], [650, 700, 1000, 898])

    def test_a_refused_move_keeps_its_reason_on_screen_until_the_next_move(self):
        r = self.said["refused"]
        self.assertEqual(r["refused"], {"text": "box is too small to show a part", "hidden": False})
        self.assertTrue(r["cleared"])

    def test_nothing_moves_while_a_layout_is_being_made(self):
        s = self.said["making a layout"]
        self.assertEqual(s["boxes"], [])
        self.assertTrue(s["drag"] and s["pan"])
        self.assertIn("being made", s["says"])

    def test_no_clip_leaves_no_boxes_and_no_buttons(self):
        s = self.said["the clip goes away"]
        self.assertEqual((s["boxes"], s["tools"]), ([], False))

    def test_the_first_change_after_a_finished_video_keeps_its_moment(self):
        self.assertEqual(self.said["the first change keeps the moment"], 7.5)


class Page(unittest.TestCase):
    def read(self, name):
        with open(os.path.join(WEB, name), encoding="utf-8") as f:
            return f.read()

    def test_the_two_buttons_sit_with_the_picture(self):
        tools_at = self.read("index.html").index('id="canvasTools"')
        self.assertLess(self.read("index.html").index('class="frame-wrap"'), tools_at)
        self.assertLess(tools_at, self.read("index.html").index('id="watch"'))

    def test_the_canvas_stays_above_the_backdrop_that_closes_the_sheet(self):
        """position: fixed makes its own stacking context: without a z-index on .clip the backdrop takes every press."""
        style = self.read("style.css")
        back = int(re.search(r"\.s-back \{[^}]*z-index: (\d+)", style).group(1))
        phone = re.search(r"\.clip\.canvas \{ position: fixed;[^}]*\}", style).group(0)
        self.assertGreater(int(re.search(r"z-index: (\d+)", phone).group(1)), back)
        tools_rule = re.search(r"^\.canvas-tools \{[^}]*\}", style, re.M).group(0)
        self.assertGreater(int(re.search(r"z-index: (\d+)", tools_rule).group(1)), back)

    def test_under_the_canvas_on_a_phone(self):
        style = self.read("style.css")
        block = style[style.index("@media (max-width: 999px), (max-aspect-ratio: 1/1)"):]
        self.assertRegex(block[:400], r"\.canvas-tools \{[^}]*top: 100%")


def card(follow=False):
    src = {"name": "card", "src": [100, 0, 1400, 1080], "dst": [60, 700, 960, 740]}
    if follow:
        src["follow"] = [{"from": 0, "x": 100, "y": 0}, {"from": 5, "x": 400, "y": 0}]
    return {"kind": "card", "size": [1920, 1080], "parts": [src, {"name": "circle", "src": [1500, 300, 200, 200], "dst": [700, 1400, 300, 300]}]}


class Panned(unittest.TestCase):
    def test_only_what_the_part_shows_moves(self):
        lay = card()
        new = layouts.panned(lay, "card", 300, 0)
        self.assertEqual(new["parts"][0]["src"], [300, 0, 1400, 1080])
        self.assertEqual(new["parts"][0]["dst"], lay["parts"][0]["dst"])
        self.assertEqual(new["parts"][1], lay["parts"][1])
        self.assertEqual(lay["parts"][0]["src"][0], 100, "the old layout stays for Undo")

    def test_it_stays_on_the_recording(self):
        self.assertEqual(layouts.panned(card(), "circle", 5000, -40)["parts"][1]["src"], [1720, 0, 200, 200])
        self.assertEqual(layouts.panned(card(), "circle", -5, 5000)["parts"][1]["src"], [0, 880, 200, 200])

    def test_a_part_that_followed_holds_where_it_is_put(self):
        new = layouts.panned(card(follow=True), "card", 250, 0)
        self.assertNotIn("follow", new["parts"][0])
        self.assertEqual(layouts.parts_at(new, 9.0)[0]["src"][0], 250, "at every second of the clip")

    def test_the_product_goes_with_the_whole_frame_and_stays_for_a_circle(self):
        lay = {"kind": "subject", "size": [1920, 1080], "thing": {"what": "a cup"},
               "parts": [{"name": "whole", "src": [600, 0, 608, 1080], "dst": [0, 0, 1080, 1920], "follow": [{"x": 600, "y": 0}]},
                         {"name": "circle", "src": [10, 10, 100, 100], "dst": [700, 1400, 300, 300]}]}
        self.assertNotIn("thing", layouts.panned(lay, "whole", 100, 0))
        self.assertIn("thing", layouts.panned(lay, "circle", 100, 0))

    def test_a_part_it_does_not_have_is_said(self):
        with self.assertRaisesRegex(ValueError, "use one of card, circle"):
            layouts.panned(card(), "top", 0, 0)

    def test_the_cut_is_where_it_was_put(self):
        new = layouts.panned(card(), "card", 320, 0)
        self.assertIn("crop=1400:1080:320:0,", layouts._scaled(new["parts"][0]))


class Panning(unittest.TestCase):
    """The real tools on a real clip, with test_editor's small recording."""

    setUpClass = classmethod(test_editor.Tools.setUpClass.__func__)
    tearDownClass = classmethod(test_editor.Tools.tearDownClass.__func__)
    setUp = test_editor.Tools.setUp
    tearDown = test_editor.Tools.tearDown
    run_job = test_editor.Tools.run_job
    to_moment = test_editor.Tools.to_moment

    def a_clip(self):
        script = self.to_moment()
        self.run_job(script[0], [test_editor.call("load_recipe", type="game", why="a game")], *script[1:],
                     [test_editor.call("make_clip", moment=1, layout="subject", why="the clip")], "Done.")
        return self.chat.project.state()["clip"]["layout"]

    def part(self, name):
        return next(p for p in self.chat.project.state()["clip"]["layout"]["parts"] if p["name"] == name)

    def test_a_pan_is_one_change_undo_takes_back_and_muse_is_told(self):
        before = self.a_clip()
        whole = next(p for p in before["parts"] if p["name"] == "whole")
        changes = len(self.chat.project.entries)
        self.assertIsNone(self.chat.pan_part("whole", 40, 0))
        self.assertEqual(self.part("whole")["src"][0], 40)
        self.assertEqual(self.part("whole")["src"][2:], whole["src"][2:])
        self.assertIn("panned the whole", self.chat.told[-1])
        self.chat.undo()
        self.assertEqual(self.part("whole")["src"], whole["src"])
        self.assertEqual(len(self.chat.project.entries), changes)

    def test_a_pan_to_where_it_is_changes_nothing(self):
        before = self.a_clip()
        x, y = next(p for p in before["parts"] if p["name"] == "whole")["src"][:2]
        told = len(self.chat.told)
        self.assertIsNone(self.chat.pan_part("whole", x, y))
        self.assertEqual(len(self.chat.told), told)

    def test_the_panel_offers_the_crop_of_what_can_pan(self):
        self.a_clip()
        d = self.chat.panel()
        self.assertEqual([c["name"] for c in d["crops"]], ["whole"])
        c = d["crops"][0]
        self.assertEqual(c["box"], [0, 0, 1000, 1000])
        self.assertEqual(c["size"], [320, 180])
        self.assertEqual(c["src"][2:], [round(v) for v in self.part("whole")["src"][2:]])
        self.assertFalse(c["follows"])
        self.assertEqual(d["parts"], [{"name": "whole", "box": [0, 0, 1000, 1000]}], "the parts stay as they were")

    def test_a_part_that_shows_the_whole_recording_is_not_offered(self):
        state = {"clip": {"layout": {"kind": "card", "size": [1920, 1080], "parts": [
            {"name": "card", "src": [0, 0, 1920, 1080], "dst": [60, 700, 960, 540]},
            {"name": "circle", "src": [1500, 300, 200, 200], "dst": [700, 1400, 300, 300]}]}}}
        self.assertEqual([c["name"] for c in self.chat.crops_now(state)], ["circle"])

    def test_a_split_clip_pans_its_people_though_it_has_no_parts_to_drag(self):
        state = {"clip": {"layout": {"kind": "split", "size": [1920, 1080], "parts": [
            {"name": "person 1", "src": [0, 0, 700, 800], "dst": [0, 0, 1080, 900]},
            {"name": "person 2", "src": [900, 0, 700, 800], "dst": [0, 920, 1080, 900]}]}}}
        self.assertEqual([c["name"] for c in self.chat.crops_now(state)], ["person 1", "person 2"])
        self.assertEqual(self.chat.parts_now(state), [])

    def test_a_talk_clip_pans_shots_and_offers_no_crops(self):
        state = {"clip": {"layout": {"kind": "talk", "size": [1920, 1080], "parts": [
            {"name": "person 1", "src": [0, 0, 576, 1026], "dst": [0, 0, 1080, 1920]}]}}}
        self.assertEqual(self.chat.crops_now(state), [])

    def test_the_server_takes_a_place_and_refuses_anything_else(self):
        self.a_clip()
        for body, ok in (({"part": "whole", "at": [30, 0]}, True), ({"part": "whole", "at": [30]}, False),
                         ({"part": "whole", "at": ["a", 0]}, False), ({"part": "whole", "at": [True, 0]}, False),
                         ({"part": "whole", "at": [1e12, 0]}, False), ({"part": "top", "at": [0, 0]}, False)):
            answer = self.post(body)
            self.assertEqual(answer.get("ok"), ok, (body, answer))
            if not ok:
                self.assertTrue(answer.get("error"))

    def post(self, body):
        got = {}
        page = server.Handler.__new__(server.Handler)
        page.send_json = lambda data, code=None: got.update(data)
        page.change_clip(self.chat, "/api/part", body)
        return got


if __name__ == "__main__":
    unittest.main()
