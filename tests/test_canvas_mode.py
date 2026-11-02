"""The Layout sheet as a canvas: the clip stays still, a part is dragged or sized on the picture, Pan moves what a part
or a shot shows, and the crop is what the render cuts. Any element tapped gets a frame to drag the same way. The page
runs from web/app.js itself in node on a small stand-in. python -m unittest discover -s tests -p test_canvas_mode.py"""
import json
import os
import re
import shutil
import subprocess
import unittest
from unittest import mock

import test_editor
from clipstudio import agent, layouts, render, server, text

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
  vm.runInContext(page + "\n;globalThis.t = { room, clipAt, openSheet, closeSheet, togglePlay, wirePlayer, drawFrameParts, showClip, hidePanel, playButton, bigPlay, fullButton, watchBox, clipBox, sheet, stepClip, movePlayhead, tapBlock, tapEffect, pickWords, pickSpot };", box);
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
const TITLE = { kind: "title", n: 0, text: "Big words", start: 0, end: 10, box: [60, 130, 940, 230] };
const SAID = { kind: "caption", n: 0, text: "hello", start: 1, end: 3, box: [60, 700, 940, 780] };
const LATER = { kind: "caption", n: 1, text: "there", start: 3, end: 5, box: [60, 700, 940, 780] };
const UNSURE = { kind: "draft", n: 7, text: "maybe", start: 6, end: 7 };
const FIRE = { of: "sticker", name: "fire", label: "Fire", start: 2, end: 4, box: [100, 300, 300, 420] };
const POP = { of: "sound", name: "pop", label: "Pop", start: 2, end: 2.5 };
const elements = (p) => {
  p.t.wirePlayer(p.video);
  p.video.paused = true;
  p.video.currentTime = 3;
  p.t.room.panel = panel({ words: [TITLE, SAID, LATER, UNSURE], effects: [FIRE, POP] });
};
const framed = (p) => {
  const shown = !p.element("partBoxes").hidden;
  const n = shown ? p.boxes() : [];
  return { n: n.length, at: n[0] ? [n[0].style.left, n[0].style.top, n[0].style.width, n[0].style.height] : null, corner: Boolean(n[0] && n[0].children.some((k) => k.tag === "i")), big: p.t.bigPlay.hidden, full: p.t.fullButton.hidden };
};
each("a caption tapped", (p) => {
  elements(p);
  p.t.tapBlock(SAID);
  const first = Object.assign(framed(p), { cap: p.t.room.cap, seat: p.video.currentTime, words: !p.t.pickWords.hidden, sheet: p.t.sheet.hidden });
  p.t.tapBlock(LATER);
  const other = p.t.room.cap;
  p.t.tapBlock(LATER);
  return { first, other, second: { sheet: p.t.sheet.hidden, editing: p.t.room.editing } };
});
each("a draft tapped", (p) => {
  elements(p);
  p.t.tapBlock(UNSURE);
  return Object.assign(framed(p), { editing: p.t.room.editing });
});
each("a caption dragged", async (p) => {
  elements(p);
  p.t.tapBlock(SAID);
  p.down(p.boxes()[0], 50, 70);
  p.move(70, 50);
  const at = p.boxes()[0].style.top;
  p.up();
  await p.settle();
  return { at, posted: p.posted, dragged: p.t.room.dragged };
});
each("a title dragged", async (p) => {
  elements(p);
  p.t.tapBlock(TITLE);
  p.down(p.boxes()[0], 50, 20);
  p.move(50, 40);
  p.up();
  await p.settle();
  return Object.assign(framed(p), { posted: p.posted });
});
each("a sticker dragged and sized", async (p) => {
  elements(p);
  p.t.tapEffect(FIRE);
  const first = framed(p);
  p.down(p.boxes()[0], 20, 40);
  p.move(30, 50);
  p.up();
  await p.settle();
  p.later.forEach((fn) => fn());
  const node = p.boxes()[0];
  p.down(node, 30, 42, node.children.find((k) => k.tag === "i"));
  p.move(40, 42);
  p.up();
  await p.settle();
  return { first, posted: p.posted };
});
each("a hand that shakes", async (p) => {
  elements(p);
  p.t.tapEffect(FIRE);
  const node = p.boxes()[0];
  p.down(node, 20, 40);
  p.move(22, 41);
  p.up();
  await p.settle();
  return Object.assign(framed(p), { posted: p.posted, same: p.boxes()[0] === node });
});
each("a sound has no frame", (p) => {
  elements(p);
  p.t.tapEffect(POP);
  return framed(p);
});
each("play takes the frame off", (p) => {
  elements(p);
  p.t.tapBlock(SAID);
  p.click("play");
  const button = Object.assign(framed(p), { cap: p.t.room.cap, playing: !p.video.paused });
  p.video.paused = true;
  p.t.tapEffect(FIRE);
  p.key(" ");
  const space = Object.assign(framed(p), { fx: p.t.room.fx, playing: !p.video.paused });
  p.video.pause();
  p.video.heard.pause.forEach((fn) => fn({}));
  return { button, space, again: framed(p) };
});
each("the picture tapped", (p) => {
  elements(p);
  const tap = (x, y) => p.video.heard.click.forEach((fn) => fn({ clientX: x, clientY: y }));
  tap(20, 35);
  const sticker = { fx: p.t.room.fx, playing: !p.video.paused, frame: framed(p) };
  p.t.room.panel.effects = [Object.assign({}, FIRE, { name: "small", box: [100, 150, 300, 210] }), POP];
  tap(20, 18);
  const small = p.t.room.fx;
  p.t.room.panel.effects = [FIRE, POP];
  p.t.clipBox.classList.remove("big");
  const area = p.element("partBoxes");
  const tapBox = (x, y) => area.heard.click.forEach((fn) => { const e = { clientX: x, clientY: y, target: area, currentTarget: area }; fn(e); });
  p.video.currentTime = 2;
  tapBox(50, 74);
  const caption = { cap: p.t.room.cap, fx: p.t.room.fx };
  p.video.currentTime = 8;
  tapBox(50, 74);
  const gone = Object.assign(framed(p), { cap: p.t.room.cap, playing: !p.video.paused });
  tap(50, 74);
  return { sticker, small, caption, gone, plays: !p.video.paused };
});
each("the picture tapped while it plays", (p) => {
  elements(p);
  p.video.paused = false;
  p.video.heard.click.forEach((fn) => fn({ clientX: 20, clientY: 35 }));
  return { fx: p.t.room.fx || null, paused: p.video.paused };
});
each("full screen plays on a tap", (p) => {
  elements(p);
  p.t.clipBox.classList.add("big");
  p.video.heard.click.forEach((fn) => fn({ clientX: 20, clientY: 35 }));
  return { fx: p.t.room.fx || null, playing: !p.video.paused };
});
each("layout takes over from a frame", (p) => {
  elements(p);
  p.t.room.panel.parts = [part(CARD)];
  p.t.room.panel.crops = [CARD];
  p.t.tapBlock(SAID);
  p.t.openSheet("layout");
  return { boxes: state(p).boxes };
});
each("up and down on the bar", async (p) => {
  elements(p);
  p.t.tapBlock(SAID);
  p.t.pickSpot.heard.click.forEach((fn) => fn({ target: { closest: () => ({ dataset: { spot: "up" } }) } }));
  await p.settle();
  return p.posted;
});
each("a caption that stays under a face", (p) => {
  elements(p);
  p.t.room.panel.words = [Object.assign({}, SAID, { box: undefined })];
  p.t.tapBlock(p.t.room.panel.words[0]);
  return Object.assign(framed(p), { says: p.element("pickSays").textContent, spot: p.element("pickSpot").hidden });
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

    def test_a_tapped_caption_gets_a_frame_and_the_big_play_steps_aside(self):
        s = self.said["a caption tapped"]["first"]
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["at"], ["6%", "70%", "88%", "8%"])
        self.assertFalse(s["corner"], "captions have no size to drag")
        self.assertTrue(s["big"], "the big Play is off the frame")
        self.assertTrue(s["full"])
        self.assertEqual(s["cap"], {"n": 0})
        self.assertEqual(s["seat"], 1, "the clip goes where the words show")
        self.assertTrue(s["words"], "Change words is one tap away")
        self.assertTrue(s["sheet"], "and the sheet stays out of the way")

    def test_a_second_tap_on_the_same_caption_changes_its_words(self):
        s = self.said["a caption tapped"]
        self.assertEqual(s["other"], {"n": 1})
        self.assertFalse(s["second"]["sheet"])
        self.assertEqual(s["second"]["editing"]["n"], 1)

    def test_a_draft_is_not_on_the_video_so_it_opens_its_words(self):
        s = self.said["a draft tapped"]
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["editing"]["kind"], "draft")

    def test_captions_drag_up_and_down_only_and_all_go_together(self):
        s = self.said["a caption dragged"]
        self.assertEqual(s["at"], "50%")
        self.assertEqual(s["posted"], [{"path": "/api/part", "body": {"chat": "c", "part": "captions", "box": [60, 500, 940, 580]}}])
        self.assertTrue(s["dragged"], "the click after the drag is not a tap")

    def test_a_title_drags_through_the_tool_that_moves_it(self):
        s = self.said["a title dragged"]
        self.assertEqual(s["posted"], [{"path": "/api/effect", "body": {"chat": "c", "of": "title", "name": "Big words", "at": 0, "box": [60, 330, 940, 430]}}])
        self.assertFalse(s["corner"])
        self.assertEqual(s["n"], 1, "the frame stays after the move")

    def test_a_sticker_drags_and_its_corner_sizes_it(self):
        s = self.said["a sticker dragged and sized"]
        self.assertTrue(s["first"]["corner"])
        self.assertEqual([p["path"] for p in s["posted"]], ["/api/effect", "/api/effect"])
        self.assertEqual(s["posted"][0]["body"], {"chat": "c", "of": "sticker", "name": "fire", "at": 2, "box": [200, 400, 400, 520]})
        self.assertEqual(s["posted"][1]["body"]["box"], [100, 300, 400, 480])

    def test_a_hand_that_shakes_moves_nothing(self):
        s = self.said["a hand that shakes"]
        self.assertEqual(s["posted"], [])
        self.assertEqual(s["n"], 1)
        self.assertTrue(s["same"], "and the frame is there for the next press")

    def test_what_has_no_spot_on_the_picture_has_no_frame(self):
        s = self.said["a sound has no frame"]
        self.assertEqual(s["n"], 0)
        self.assertFalse(s["big"], "and Play stays")

    def test_play_or_space_takes_the_frame_off_and_brings_play_back(self):
        s = self.said["play takes the frame off"]
        self.assertEqual((s["button"]["n"], s["button"]["cap"], s["button"]["playing"]), (0, None, True))
        self.assertEqual((s["space"]["n"], s["space"]["fx"], s["space"]["playing"]), (0, None, True))
        self.assertFalse(s["again"]["big"])

    def test_a_tap_on_the_picture_picks_what_stands_there_and_plays_nothing(self):
        s = self.said["the picture tapped"]
        self.assertEqual(s["sticker"]["fx"], "sticker|fire|2")
        self.assertEqual(s["small"], "sticker|small|2", "the smallest of what stands there, the sticker over the title")
        self.assertFalse(s["sticker"]["playing"])
        self.assertEqual(s["sticker"]["frame"]["n"], 1)
        self.assertEqual(s["caption"], {"cap": {"n": 0}, "fx": None})

    def test_a_tap_beside_everything_takes_the_frame_off_and_the_next_one_plays(self):
        s = self.said["the picture tapped"]
        self.assertEqual((s["gone"]["n"], s["gone"]["cap"], s["gone"]["playing"]), (0, None, False))
        self.assertTrue(s["plays"])

    def test_while_it_plays_a_tap_only_stops_it(self):
        self.assertEqual(self.said["the picture tapped while it plays"], {"fx": None, "paused": True})

    def test_full_screen_only_watches(self):
        self.assertEqual(self.said["full screen plays on a tap"], {"fx": None, "playing": True})

    def test_layout_shows_its_own_boxes(self):
        self.assertEqual(self.said["layout takes over from a frame"]["boxes"], ["Card"])

    def test_up_and_down_on_the_bar_move_the_captions_too(self):
        self.assertEqual(self.said["up and down on the bar"], [{"path": "/api/part", "body": {"chat": "c", "part": "captions", "box": [60, 660, 940, 740]}}])

    def test_captions_under_each_face_say_why_they_have_no_frame(self):
        s = self.said["a caption that stays under a face"]
        self.assertEqual(s["n"], 0)
        self.assertIn("under each person's face", s["says"])
        self.assertTrue(s["spot"])

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
        self.assertIn("Drag the picture", s["panning"]["says"])
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

    def test_full_screen_keeps_the_clip_inside_the_window_on_a_wide_screen(self):
        style = self.read("style.css")
        self.assertIn(".clip.big .frame { width: min(100vw, calc(100dvh * 9 / 16)); height: min(100dvh, calc(100vw * 16 / 9));", style)

    def test_a_mouse_gets_a_slider_under_the_tools_and_a_finger_swipes(self):
        style = self.read("style.css")
        self.assertRegex(style, r"\.tools \{[^}]*scrollbar-width: none;")
        self.assertRegex(style, r"@media \(hover: hover\) and \(pointer: fine\) \{\s*\.tools \{ scrollbar-width: thin;")

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

    def with_captions(self):
        self.a_clip()
        lay = self.chat.project.state()["clip"]["layout"]["captions"]
        lines = text.caption_lines([{"from": 0.5, "to": 1.5, "text": "hello there"}], 30)
        self.chat.edit({"do": "captions", "lines": lines, "band": [lay[1], lay[3]], "lit": False, "pop": False})
        return lay

    def drawn_at(self):
        with open(render.ass(self.chat.project.state()), encoding="utf-8") as f:
            return next(line for line in f if line.startswith("Dialogue") and "_caption" in line).split("pos(")[1].split(")")[0]

    def test_a_caption_says_where_it_stands(self):
        lay = self.with_captions()
        box = next(b for b in self.chat.panel()["words"] if b["kind"] == "caption")["box"]
        self.assertEqual([round(v, 3) for v in box[1::2]], [round(v, 3) for v in agent._grid_box(lay)[1::2]])

    def test_the_frame_of_captions_stands_in_the_middle_where_their_words_do(self):
        self.with_captions()
        state = self.chat.project.state()
        lay = dict(state["clip"]["layout"], captions=[65, 400, 830, 200])
        box = next(b for b in self.chat.words_on(dict(state, clip=dict(state["clip"], layout=lay))) if b["kind"] == "caption")["box"]
        self.assertAlmostEqual(box[0] + box[2], 1000)

    def test_the_captions_move_as_one_undo_takes_them_back_and_muse_is_told(self):
        lay = self.with_captions()
        before = self.drawn_at()
        self.assertIsNone(self.chat.set_part("captions", [0, 300, 1000, 400]))
        state = self.chat.project.state()
        self.assertEqual(state["captions"]["band"], [576, lay[3]])
        self.assertEqual(state["clip"]["layout"]["captions"], [lay[0], 576, lay[2], lay[3]], "so captions made again stay there")
        self.assertIn("moved the captions", self.chat.told[-1])
        x0, y0 = [float(v) for v in before.split(",")]
        x1, y1 = [float(v) for v in self.drawn_at().split(",")]
        self.assertEqual(x1, x0, "they stay in the middle of the frame")
        self.assertAlmostEqual(y1 - y0, 576 - lay[1], delta=1)
        self.chat.undo()
        self.assertEqual(self.chat.project.state()["captions"]["band"], [lay[1], lay[3]])
        self.assertEqual(self.drawn_at(), before)

    def test_words_changed_after_a_move_stay_where_they_were_put(self):
        lay = self.with_captions()
        self.chat.set_part("captions", [0, 300, 1000, 400])
        self.assertIsNone(self.chat.set_words("caption", 0, "hello there", "hello you"))
        self.assertEqual(self.chat.project.state()["captions"]["band"], [576, lay[3]])

    def test_the_captions_stay_on_the_frame(self):
        lay = self.with_captions()
        self.assertIsNone(self.chat.set_part("captions", [0, 990, 1000, 1000]))
        self.assertEqual(self.chat.project.state()["captions"]["band"], [1920 - lay[3], lay[3]])

    def test_titles_keep_clear_of_where_the_captions_went(self):
        self.with_captions()
        self.chat.set_part("captions", [0, 300, 1000, 400])
        self.assertEqual(self.chat.project.state()["clip"]["layout"]["captions"][1], 576)

    def test_no_captions_nothing_to_move(self):
        self.a_clip()
        self.assertIn("no captions", self.chat.set_part("captions", [0, 300, 1000, 400]))

    def test_captions_under_each_face_stay_there(self):
        self.with_captions()
        caps = self.chat.project.state()["captions"]
        self.chat.edit(dict(caps, bands={"1": [400, 200], "2": [1200, 200]}))
        self.assertIn("under the face", self.chat.set_part("captions", [0, 300, 1000, 400]))
        self.assertNotIn("box", next(b for b in self.chat.panel()["words"] if b["kind"] == "caption"))

    def post(self, body):
        got = {}
        page = server.Handler.__new__(server.Handler)
        page.send_json = lambda data, code=None: got.update(data)
        page.change_clip(self.chat, "/api/part", body)
        return got


if __name__ == "__main__":
    unittest.main()
