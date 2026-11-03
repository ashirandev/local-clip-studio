"""The cards of a run last in the chat, a preview with its sound, and the bar on the clip's side, run from web/app.js
itself in node on a small stand-in for the page. python -m unittest tests.test_cards_and_bar"""
import json
import os
import re
import shutil
import subprocess
import unittest

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

STAND_IN = r"""
const fs = require("fs");
const vm = require("vm");

class Node {
  constructor(tag) {
    this.tag = tag;
    this.className = "";
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.hidden = false;
    this.value = "";
    this.textContent = "";
    this.attrs = {};
    this.heard = {};
    const node = this;
    this.classList = {
      add: (...k) => k.forEach((c) => node.classList.toggle(c, true)),
      remove: (...k) => k.forEach((c) => node.classList.toggle(c, false)),
      contains: (c) => node.className.split(" ").includes(c),
      toggle: (c, on = !node.classList.contains(c)) => {
        const have = node.className.split(" ").filter((x) => x && x !== c);
        node.className = (on ? have.concat(c) : have).join(" ");
      },
    };
  }
  addEventListener(k, fn) { (this.heard[k] = this.heard[k] || []).push(fn); }
  removeEventListener() {}
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] === undefined ? null : this.attrs[k]; }
  removeAttribute(k) { delete this.attrs[k]; }
  append(...kids) { kids.forEach((k) => this.appendChild(k)); }
  appendChild(k) { if (k instanceof Node) { k.up = this; this.children.push(k); } return k; }
  click() { this.clicked = (this.clicked || 0) + 1; }
  press(target) { (this.heard.click || []).forEach((fn) => fn({ target: target || this })); }
  play() { this.played = true; return Promise.resolve(); }
  prepend(...kids) { this.children.unshift(...kids.filter((k) => k instanceof Node)); }
  replaceChildren(...kids) {
    this.children = kids.filter((k) => k instanceof Node);
    this.children.forEach((k) => { k.up = this; });
  }
  remove() { if (this.up) this.up.children = this.up.children.filter((k) => k !== this); }
  closest(q) { return this.closeTo === q ? this : null; }
  focus() {}
  scrollIntoView() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100, bottom: 100, right: 100 }; }
  walk(fn) { this.children.forEach((k) => { fn(k); k.walk(fn); }); }
  querySelectorAll(q) {
    const out = [];
    const cls = q.replace(/^\./, "");
    this.walk((k) => { if (k.classList.contains(cls) || k.tag === q) out.push(k); });
    return out;
  }
  querySelector(q) { return this.querySelectorAll(q)[0] || null; }
}

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

function load(page, answer) {
  const found = {};
  const body = new Node("body");
  const made = [];
  const element = (id) => (found[id] = found[id] || new Proxy(new Node("div"), {
    get(o, k) {
      if (k in o) return typeof o[k] === "function" ? o[k].bind(o) : o[k];
      if (typeof k === "symbol" || k === "then") return undefined;
      return (o[k] = any());
    },
  }));
  const document = {
    getElementById: element, addEventListener: () => {}, removeEventListener: () => {}, body, documentElement: body, title: "Studio",
    scrollingElement: { scrollTop: 0, scrollHeight: 0 },
    createElement: (tag) => { const n = new Node(tag); made.push(n); return n; },
    createElementNS: (ns, tag) => new Node(tag),
    querySelectorAll: (q) => element("log").querySelectorAll(q),
    querySelector: (q) => element("log").querySelector(q),
  };
  const asked = [], ran = [], heard = {};
  const box = {
    console, Math, Number, String, Object, Array, JSON, Promise, Date, Set, Map, RegExp, Infinity, NaN, parseInt, parseFloat, isNaN, encodeURIComponent, decodeURIComponent, document,
    window: any({ lcsReplay: false, addEventListener: (k, fn) => { heard[k] = fn; }, matchMedia: () => any({ matches: false }) }),
    history: any(), navigator: any(), localStorage: any({ getItem: () => null }), sessionStorage: any({ getItem: () => null }),
    fetch: (u) => (asked.push(u), answer ? Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(answer) }) : new Promise(() => {})), EventSource: function () { return any(); }, ResizeObserver: function () { return any(); }, IntersectionObserver: function () { return any(); }, MutationObserver: function () { return any(); },
    getComputedStyle: () => ({ overflowY: "auto", display: "block" }), matchMedia: () => any({ matches: false }), requestAnimationFrame: () => 0, cancelAnimationFrame: () => {}, setTimeout: (fn, ms) => { if (ms === 1500) ran.push(fn); return 0; }, clearTimeout: () => {}, setInterval: () => 0, clearInterval: () => {},
    Intl: any(), URL: any(), Image: function () { return any(); }, FormData: function () { return any(); }, CSS: any(), performance: any({ now: () => 0 }), innerWidth: 1200, innerHeight: 800,
  };
  vm.createContext(box);
  vm.runInContext(page + "\n;globalThis.t = { room, job, tile, drawMoments, drawPlan, showMoment, showStatus, onThinking, pageWork, endLine, on, setScreen, chatView, askPremiere, showClip, fillPreview, setBlur, drawFaces, drawMinds, drawCost, loadKey, showPart, showPicture, showOver, showFinished, showBase, api: (name) => { apiNow = name; } };", box);
  const t = box.t;
  t.room.chat = "c";
  let id = 0;
  const event = (kind, over) => t.on(Object.assign({ id: id++, kind, job: 1 }, over));
  return { t, element, made, event, asked, ran, heard, box };
}

const page = fs.readFileSync(process.argv[2], "utf8");
const said = {};
const each = (name, fn) => { said[name] = fn(load(page)); };
const MOMENTS = [1, 2, 3].map((n) => ({ n, preview: "p" + n + ".mp4", frame: "f.jpg", t: n, start: n, end: n + 5, reason: "why", said: [] }));
const bar = (p) => ({
  hidden: p.element("jobBar").hidden, says: p.element("jobBarSays").textContent, step: p.element("jobBarStep").textContent,
  width: p.element("jobBarFill").style.width, now: p.element("jobBarTrack").attrs["aria-valuenow"], done: p.element("jobBar").classList.contains("done"),
});
const plan = (done, now, todo) => ({ items: [
  ...Array(done).fill(0).map((_, k) => ({ text: "Done " + k, state: "done" })),
  ...Array(now).fill(0).map((_, k) => ({ text: "Cut the clip", state: "now" })),
  ...Array(todo).fill(0).map((_, k) => ({ text: "Later " + k, state: "todo" })),
] });

const MINDS = [["local", "Muse Glimmer", true, false], ["spark", "Muse Spark", false, true], ["wonder", "Muse Wonder", false, true],
  ["modal", "Muse Spark Open", false, false], ["baseten", "Muse Spark Open (Baseten)", false, false], ["fireworks", "Muse Spark Open (Fireworks)", false, false]]
  .map(([key, label, local, ok]) => ({ key, label, local, ok, name: label, where: "on servers", why: "no key", url: "u" }));
each("faces", (p) => {
  p.t.room.made = { version: 1, clip: { people: 2, length: 10 } };
  const faces = () => p.asked.filter((u) => String(u).includes("/api/faces")).length;
  p.t.drawFaces(2, p.t.room.made, "a");
  const closed = faces();
  p.t.setBlur(true);
  const open = faces();
  return { closed, open, wide: p.element("sheet").classList.contains("canvas") };
});
each("blurSheet", (p) => {
  const sheet = p.element("sheet");
  sheet.hidden = false;
  p.t.showPart("blur");
  const open = p.t.room.blur;
  sheet.hidden = true;
  p.t.showPart("blur");
  const shut = p.t.room.blur;
  sheet.hidden = false;
  p.t.showPart("trim");
  return { open, shut, other: p.t.room.blur };
});
each("bigPlay", (p) => {
  const v = new Node("video");
  v.paused = true;
  v.currentTime = 1;
  p.element("frame").appendChild(v);
  p.t.room.made = { version: 1, clip: { people: 2, length: 10 } };
  p.element("sheet").hidden = true;
  p.t.showPart("trim");
  const before = p.element("bigPlay").hidden;
  p.t.setBlur(true);
  return { before, after: p.element("bigPlay").hidden };
});
each("blurTab", (p) => {
  p.t.showClip({ id: 1, clip: { people: 2, length: 10, lanes: [] } });
  const people = p.element("blurTab").hidden;
  p.t.showClip({ id: 2, clip: { people: 0, length: 10, lanes: [] } });
  return { people, nobody: p.element("blurTab").hidden, line: p.element("clipLine").textContent };
});
each("options", (p) => {
  p.t.room.minds = MINDS;
  p.t.drawMinds();
  const folded = p.element("minds").children.length;
  const machine = p.element("machine").children.length;
  p.element("apiOther").press();
  return { folded, machine, opened: p.element("minds").children.length, button: p.element("apiOther").hidden, tiles: p.element("apiSeg").hidden };
});
each("readyStays", (p) => {
  p.t.room.minds = MINDS.map((m) => (m.key === "modal" ? Object.assign({}, m, { ok: true }) : m));
  p.t.drawMinds();
  return p.element("minds").children.length;
});
each("cost", (p) => {
  p.t.room.minds = MINDS;
  const box = p.element("costBox");
  p.t.drawCost({ jobs: [], chat: 0, today: 0, month: 1, minds: { "Muse Spark": 1 }, prices: "the list", rate: {} }, box);
  const names = box.children.map((c) => (c.children[0] ? c.children[0].textContent : ""));
  return { names, note: box.children[box.children.length - 1].textContent };
});
each("order", (p) => {
  p.event("job", { mind: "Muse Spark" });
  const j = p.t.job(1);
  p.t.drawMoments(j, MOMENTS);
  p.event("job", { job: 2, mind: "Muse Spark" });
  const log = p.element("log");
  const later = { inLog: log.children.includes(j.momentsBox), last: j.momentsBox.classList.contains("last"), inCard: j.card.children.includes(j.momentsBox) };
  const k = p.t.job(2);
  p.t.drawMoments(k, MOMENTS);
  return { later, older: j.momentsBox.classList.contains("last"), newer: k.momentsBox.classList.contains("last") };
});
each("pick", (p) => {
  p.event("job", { mind: "Muse Spark" });
  const j = p.t.job(1);
  p.t.drawMoments(j, MOMENTS);
  const chips = () => j.momentsBox.querySelectorAll(".chip").map((b) => b.hidden);
  const before = chips();
  p.event("waiting");
  const after = chips();
  const q = load(page);
  q.event("job", { mind: "Muse Spark" });
  const k = q.t.job(1);
  q.event("waiting");
  q.t.drawMoments(k, MOMENTS);
  return { before, after, first: k.momentsBox.querySelectorAll(".chip").map((b) => b.hidden) };
});
each("sound", (p) => {
  p.t.room.cut["p2.mp4"] = { ok: true, name: "p2.mp4" };
  p.t.showMoment(MOMENTS[1]);
  const videos = p.made.filter((n) => n.tag === "video");
  return { videos: videos.length, muted: videos.map((v) => v.muted === true) };
});
each("run", (p) => {
  p.t.showStatus();
  const start = bar(p);
  p.event("job", { mind: "Muse Spark" });
  const none = bar(p);
  p.event("plan", plan(0, 1, 4));
  const first = bar(p);
  p.event("plan", plan(2, 1, 2));
  const two = bar(p);
  p.t.onThinking({ job: 1, work: "Rendering the clip: 12 of 24 s (50%)" });
  const within = bar(p);
  p.event("plan", plan(2, 1, 7));
  const replanned = bar(p);
  p.event("plan", plan(10, 0, 0));
  const all = bar(p);
  p.event("end", { why: "answered", steps: 5, seconds: 60 });
  const end = bar(p);
  return { start, none, first, two, within, replanned, all, end };
});
each("clipReady", (p) => {
  p.event("job", { mind: "Muse Spark" });
  p.t.job(1).handed = true;
  p.event("end", { why: "answered", steps: 5, seconds: 60 });
  return bar(p);
});
each("cards", (p) => {
  p.event("job", { mind: "Muse Spark" });
  p.event("waiting");
  p.event("end", { why: "answered", steps: 5, seconds: 60 });
  return bar(p);
});
each("stopped", (p) => {
  p.event("job", { mind: "Muse Spark" });
  p.event("plan", plan(2, 1, 2));
  p.event("end", { why: "stopped", steps: 3, seconds: 60 });
  return bar(p);
});
each("video", (p) => {
  p.t.pageWork("video", true);
  const started = bar(p);
  p.t.onThinking({ job: null, work: "Rendering the clip: 13 of 24 s (51%)" });
  const moving = bar(p);
  p.t.pageWork("video", false);
  return { started, moving, ended: bar(p), line: p.t.room.workLine };
});
each("nextJob", (p) => {
  p.event("job", { mind: "Muse Spark" });
  p.t.onThinking({ job: 1, work: "Mapping the video: 1:00 of 2:00 (50%)" });
  p.event("step", { step: 1, tool: "map", ok: true, label: "Mapped", text: "" });
  p.event("plan", plan(1, 1, 2));
  return bar(p);
});
each("open", (p) => {
  p.t.room.screen = "home";
  p.t.chatView.scrollHeight = 900;
  p.t.setScreen("clip");
  return { top: p.t.chatView.scrollTop };
});
const stopAtCards = (p) => {
  p.event("job", { mind: "Muse Spark" });
  const j = p.t.job(1);
  p.event("waiting");
  p.t.drawMoments(j, MOMENTS.map((m) => Object.assign({}, m)));
  p.event("end", { why: "answered", steps: 5, seconds: 60 });
  return j;
};
const strip = (p) => p.element("basketRow").children;
const stripOn = (p) => p.element("clip").classList.contains("basket-on");
each("stripOpens", (p) => {
  const j = stopAtCards(p);
  return { on: stripOn(p), n: strip(p).length, says: p.element("basketSays").textContent, back: p.element("basketBack").hidden,
    uses: strip(p).map((t) => t.children[1].hidden), first: strip(p)[0].className };
});
each("stripOverClip", (p) => {
  p.t.room.made = { clip: { length: 10, parts: [] }, version: 2 };
  stopAtCards(p);
  return { on: stripOn(p) };
});
each("stripWatch", (p) => {
  const j = stopAtCards(p);
  p.t.room.cut["p2.mp4"] = { ok: true, name: "p2.mp4" };
  strip(p)[1].children[0].press();
  const videos = p.made.filter((n) => n.tag === "video");
  return { on: stripOn(p), showing: p.t.room.showing.n, picked: strip(p).map((t) => t.className), played: videos[videos.length - 1].played === true };
});
each("stripUse", (p) => {
  const j = stopAtCards(p);
  strip(p)[2].children[1].press();
  return { on: stripOn(p), showing: p.t.room.showing.n };
});
each("stripClip", (p) => {
  stopAtCards(p);
  p.t.showClip({ clip: { length: 10, parts: [], people: false }, id: 9 });
  return { on: stripOn(p) };
});
each("chatTile", (p) => {
  p.event("job", { mind: "Muse Spark" });
  const j = p.t.job(1);
  p.t.drawMoments(j, MOMENTS.map((m) => Object.assign({}, m)));
  const before = stripOn(p);
  const tiles = j.momentsBox.children;
  tiles[1].press();
  const after = stripOn(p);
  const showing = p.t.room.showing.n;
  p.element("clip").classList.remove("basket-on");
  const chip = tiles[2].querySelector(".chip");
  chip.closeTo = ".chip";
  tiles[2].press(chip);
  return { before, after, showing, chipOpened: stripOn(p) };
});
each("stripBack", (p) => {
  stopAtCards(p);
  const none = p.element("basketBack").hidden;
  p.t.room.made = { clip: { length: 10, parts: [], people: false }, version: 3 };
  p.t.room.basket && p.element("clip").classList.add("basket-on");
  strip(p)[1].children[0].press();
  const shown = p.element("basketBack").hidden;
  (p.element("basketBack").heard.click || []).forEach((fn) => fn({}));
  return { none, shown, on: stripOn(p), showing: p.t.room.showing };
});
each("spelled", (p) => {
  p.event("job", { mind: "Muse Spark" });
  const j = p.t.job(1);
  const moments = MOMENTS.map((m) => Object.assign({}, m));
  p.t.drawMoments(j, moments);
  moments[0].why.textContent = "fixed";
  p.event("waiting");
  p.event("end", { why: "answered", steps: 5, seconds: 60 });
  return { why: strip(p)[0].children[0].children[2].textContent };
});
each("tileReady", (p) => {
  p.event("job", { mind: "Muse Spark" });
  const j = p.t.job(1);
  p.t.drawMoments(j, MOMENTS.map((m) => Object.assign({}, m)));
  const tile = j.momentsBox.children[0];
  p.t.fillPreview(tile, { ok: true, name: "p1.mp4" });
  p.t.fillPreview(tile, { ok: true, name: "p1.mp4" });
  const media = tile.children[0];
  return { wait: media.querySelectorAll(".moment-wait").length, kids: media.children.map((n) => n.tag), inline: p.made.some((n) => n.tag === "video" && n.controls) };
});
each("premiereBar", (p) => {
  p.t.pageWork("premiere", true);
  p.t.onThinking({ job: null, page: "premiere", work: "Making the Premiere files: 19 of 38 s (50%)" });
  return { bar: bar(p), status: p.element("status").textContent };
});
each("premiereReload", (p) => {
  p.t.onThinking({ job: null, page: "premiere", work: "Making the Premiere files: 19 of 38 s (50%)" });
  return { work: p.t.room.work.premiere, says: bar(p).says };
});
const saved = (p) => p.made.filter((n) => n.tag === "a" && n.clicked).length;
each("premiereEnd", (p) => {
  p.t.pageWork("premiere", true);
  p.t.room.wantPremiere = true;
  p.t.on({ id: 1, kind: "layers", job: null });
  return { saved: saved(p), want: p.t.room.wantPremiere, work: p.t.room.work.premiere };
});
each("premiereReplay", (p) => {
  p.t.on({ id: 1, kind: "layers", job: null });
  return { saved: saved(p) };
});
each("premiereFailed", (p) => {
  p.t.pageWork("premiere", true);
  p.t.room.wantPremiere = true;
  p.t.on({ id: 1, kind: "layers", job: null, error: "No." });
  return { saved: saved(p), work: p.t.room.work.premiere };
});
const press = async (answer) => {
  const p = load(page, answer);
  await p.t.askPremiere();
  return { saved: saved(p), want: p.t.room.wantPremiere, work: Boolean(p.t.room.work.premiere) };
};
(async () => {
  const keyed = load(page, { here: true, url: "", minds: MINDS, apis: [{ name: "meta", label: "Meta", saved: false, key: "k", page: "p", icon: "spark" }, { name: "modal", label: "Modal", saved: true, key: "k", page: "p", icon: "stack" }] });
  keyed.t.room.minds = MINDS;
  await keyed.t.loadKey();
  said.otherSaved = { tiles: keyed.element("apiSeg").hidden, minds: keyed.element("minds").children.length };
  said.pressReady = await press({ ok: true, ready: true });
  said.pressMaking = await press({ ok: true, making: true });
  said.pressRefused = await press({ ok: false, error: "No." });
  Object.defineProperty(Node.prototype, "tagName", { get() { return this.tag.toUpperCase(); } });
  const pics = load(page);
  const box = pics.element("frame");
  const fire = (n, kind) => (n.heard[kind] || []).forEach((fn) => fn());
  const shown = () => box.children.map((n) => n.name);
  const picture = (name) => Object.assign(new Node("img"), { name });
  const [first, next, last] = ["first", "next", "last"].map(picture);
  pics.t.showPicture(first);
  said.picture = { empty: shown() };
  pics.t.showPicture(next);
  said.picture.waits = shown();
  pics.t.showPicture(last);
  fire(next, "load");
  said.picture.stale = shown();
  fire(last, "load");
  said.picture.done = shown();
  const film = Object.assign(new Node("video"), { name: "film" });
  pics.t.showOver(film, []);
  said.picture.under = [shown(), film.classList.contains("waiting")];
  fire(film, "loadeddata");
  said.picture.over = [shown(), film.classList.contains("waiting")];
  const late = (show) => {
    const opened = load(page);
    const frame = opened.element("frame");
    const held = Object.assign(new Node("img"), { name: "held" }), slow = Object.assign(new Node("img"), { name: "slow" });
    opened.t.showPicture(held);
    opened.t.showPicture(slow);
    show(opened.t);
    fire(slow, "load");
    return frame.children.map((n) => n.tag);
  };
  const wide = load(page);
  const stage = wide.element("clip");
  const seat = wide.element("clipAt");
  wide.t.room.made = { clip: { length: 39 }, version: 1 };
  seat.value = 5;
  wide.t.showBase("/api/base/a.mp4");
  const quick = wide.element("frame").querySelector("video");
  seat.value = 0.05;
  quick.duration = 39;
  fire(quick, "loadedmetadata");
  said.picture.startsAt = quick.currentTime;
  quick.enters = 0;
  quick.webkitEnterFullscreen = () => quick.enters++;
  fire(wide.element("full"), "click");
  said.picture.big = [stage.classList.contains("big"), quick.enters];
  fire(wide.element("full"), "click");
  said.picture.bigAgain = stage.classList.contains("big");
  fire(wide.element("full"), "click");
  wide.t.showFinished({ name: "a.mp4" });
  said.picture.bigGone = stage.classList.contains("big");
  const fresh = load(page);
  fresh.element("clipAt").value = 7;
  fresh.t.showClip({ id: 1, kind: "edit", entry: { do: "clip" }, clip: { length: 39 } });
  said.picture.opensAt = Number(fresh.element("clipAt").value);
  said.picture.lateOnFinished = late((t) => t.showFinished({ name: "a.mp4" }));
  said.picture.lateOnMoment = late((t) => {
    t.room.cut = { "m.mp4": { ok: true } };
    t.showMoment({ n: 1, preview: "m.mp4", frame: "f.jpg", start: 0, end: 5 });
  });
  const phone = load(page);
  const screen = phone.element("frame");
  const still = Object.assign(new Node("img"), { name: "still" }), played = Object.assign(new Node("video"), { name: "played" });
  screen.replaceChildren(still);
  phone.t.showOver(played, []);
  said.picture.timer = phone.ran.length;
  phone.ran.forEach((fn) => fn());
  said.picture.phone = [screen.children.map((n) => n.name), played.classList.contains("waiting")];
  const broken = load(page);
  const glass = broken.element("frame");
  const lost = Object.assign(new Node("video"), { name: "lost" });
  glass.replaceChildren(Object.assign(new Node("img"), { name: "still" }));
  broken.t.showOver(lost, []);
  (lost.heard.error || []).forEach((fn) => fn());
  said.picture.broken = lost.classList.contains("waiting");
  const noted = async (api, sends) => {
    const p = load(page, { ok: true, url: "u", mind: { ok: true, label: "Muse Spark" } });
    await new Promise((r) => setImmediate(r));
    p.t.api(api);
    p.t.room.sends = sends;
    await p.element("apiForm").heard.submit[0]({ preventDefault: () => {} });
    await new Promise((r) => setImmediate(r));
    return p.element("apiNote").textContent;
  };
  said.keyNote = [await noted("meta", false), await noted("meta", true), await noted("baseten", false)];
  const turned = load(page, { ok: true, url: "u", turned: true, sends: false, mind: { ok: true, label: "Muse Spark" } });
  await new Promise((r) => setTimeout(r, 20));
  turned.t.api("meta");
  turned.t.room.sends = false;
  await turned.element("apiForm").heard.submit[0]({ preventDefault: () => {} });
  said.keyTurned = [turned.element("apiNote").textContent, turned.t.room.sends];
  {
  const tick = () => new Promise((r) => setTimeout(r, 5));
  {
  const w = load(page, { ok: true, jobs: [], rate: { currency: "USD" } });
  await tick();
  const beats = [];
  w.box.setInterval = (fn) => (beats.push(fn), beats.length);
  w.box.clearInterval = () => {};
  const now = Date.now() / 1000;
  w.event("job", { mind: "Muse Spark" });
  w.event("wait", { step: 9, t: now - 3 });
  const j = w.t.room.jobs[1];
  said.work = { thinking: j.meta.textContent };
  w.t.onThinking({ job: 1, step: 8, work: "Muse Wonder is watching who speaks, turn 5", since: now - 252 });
  said.work.working = j.meta.textContent;
  const timers = beats.length;
  w.t.onThinking({ job: 1, step: 8, work: "Muse Wonder is watching who speaks, turn 6", since: now - 252 });
  said.work.sameClock = beats.length === timers;
  said.work.line = j.tick.textContent;
  const before = w.asked.length;
  for (let k = 0; k < 15; k++) beats[beats.length - 1]();
  said.work.costWhileWorking = w.asked.slice(before).filter((u) => u.startsWith("/api/spent")).length;
  const atStep = w.asked.length;
  w.event("step", { step: 9, tool: "speakers", why: "who speaks", seconds: 300 });
  said.work.costAtStep = w.asked.slice(atStep).filter((u) => u.startsWith("/api/spent")).length;
  w.event("wait", { step: 10, t: Date.now() / 1000 });
  said.work.next = j.meta.textContent;
  }
  const dragged = { stopped: 0, preventDefault() { this.stopped += 1; } };
  load(page).heard.dragstart(dragged);
  said.dragStopped = dragged.stopped;
  const chat = { id: "a", title: "Trip", first: "cut", clips: [], minds: [] };
  const parts = (p) => {
    p.t.room.rate = { currency: "USD", rate: 1 };
    const box = p.t.tile(chat);
    const get = (k) => box.querySelector("." + k);
    return { box, bin: get("tile-bin"), ask: get("tile-ask"), yes: get("tile-yes"), no: get("tile-no") };
  };
  const deleting = load(page, { ok: true, chat: "new", chats: [] });
  await tick();
  deleting.t.room.chat = "b";
  const tile = parts(deleting);
  said.del = { first: [tile.ask.hidden, tile.bin.hidden, tile.bin.attrs["aria-label"]], words: tile.ask.children[0].textContent };
  const before = deleting.asked.length;
  tile.bin.press();
  said.del.asked = [tile.ask.hidden, tile.bin.hidden, deleting.asked.length - before];
  tile.no.press();
  said.del.kept = [tile.ask.hidden, tile.bin.hidden, deleting.asked.length - before];
  tile.bin.press();
  tile.yes.press();
  await tick();
  said.del.sent = deleting.asked.slice(before);
  said.del.off = tile.yes.disabled && tile.no.disabled;
  const shown = load(page, { ok: true, chat: "new", chats: [] });
  await tick();
  shown.t.room.chat = "a";
  const here = parts(shown);
  const from = shown.asked.length;
  here.bin.press();
  here.yes.press();
  await tick();
  said.del.shown = shown.asked.slice(from);
  const refused = load(page, { ok: false, error: "This chat is working. Stop it first, then delete it.", chat: "c", chats: [] });
  await tick();
  refused.t.room.chat = "a";
  const stays = parts(refused);
  const toasts = refused.made.length;
  stays.bin.press();
  stays.yes.press();
  await tick();
  said.del.refused = refused.made.slice(toasts).filter((n) => n.className === "toast").map((n) => n.textContent);
  const working = load(page);
  working.t.room.chat = "a";
  working.t.room.open = new Set([1]);
  said.del.working = parts(working).bin === null;
  }
  const named = load(page);
  const opened = (name) => named.event("opened", { recording: { name, seconds: 60 } });
  opened("sifu.mkv");
  named.event("named", { title: "Sifu" });
  opened("sifu.mkv");
  said.name = [named.element("chatName").textContent];
  opened("pots.mkv");
  said.name.push(named.element("chatName").textContent);
  console.log(JSON.stringify(said));
})();
"""


@unittest.skipUnless(shutil.which("node"), "node is not on this computer")
class Page(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run = subprocess.run(["node", "-e", STAND_IN, "-", os.path.join(WEB, "app.js")], capture_output=True, text=True,
                             encoding="utf-8", timeout=60)
        if run.returncode:
            raise AssertionError(run.stderr[-2000:])
        cls.said = json.loads(run.stdout.strip().splitlines()[-1])

    def test_the_quick_preview_starts_where_the_line_is_and_keeps_its_words_full_screen(self):
        """iPhone 11-02: a phone loads a video only when it is played, so the seat kept from when the page opened moved
        it to the middle then; and the phone's own full screen shows the video without the words laid over it."""
        s = self.said["picture"]
        self.assertEqual(s["opensAt"], 0, "a new clip opens at its start, so Play plays it from there")
        self.assertEqual(s["startsAt"], 0.05, "where the line is now, not where it was")
        self.assertEqual(s["big"], [True, 0], "the page grows the video with its words; the phone's own player has none")
        self.assertFalse(s["bigAgain"])
        self.assertFalse(s["bigGone"], "a finished video has its words, and its own full screen")

    def test_a_still_on_its_way_does_not_land_over_a_video(self):
        """EP0 on a phone 11-02: opened again, the slow still of a step before came in 12 s later and took the player's
        finished video away, so Play could not be pressed."""
        pictures = self.said["picture"]
        self.assertEqual(pictures["lateOnFinished"], ["video"])
        self.assertEqual(pictures["lateOnMoment"], ["video"])

    def test_the_picture_stays_until_the_next_one_can_be_seen(self):
        """Lane 5 11-01: 0.5 to 4.9 s of black after a drop, while the next picture was on its way."""
        s = self.said["picture"]
        self.assertEqual(s["empty"], ["first"], "nothing to keep, so it shows at once")
        self.assertEqual(s["waits"], ["first"])
        self.assertEqual(s["stale"], ["first"], "an answer overtaken by a newer ask is not shown")
        self.assertEqual(s["done"], ["last"])
        self.assertEqual(s["under"], [["last", "film"], True], "the video loads under the picture")
        self.assertEqual(s["over"], [["film"], False])
        self.assertEqual(s["phone"], [["played"], False], "a phone loads no video until it is played, so the wait ends")
        self.assertFalse(s["broken"], "a video that cannot load says so itself, on the page")

    def test_a_meta_key_saved_with_the_switch_off_says_what_is_missing(self):
        """Lane 5 11-02: a key added later gave no Wonder and no cover, and the note only said it was ready."""
        off, on, other = self.said["keyNote"]
        self.assertIn("Meta sees pieces is off", off)
        self.assertEqual(on, "Muse Spark is ready.")
        self.assertEqual(other, "Muse Spark is ready.")

    def test_a_meta_key_that_turned_the_switch_on_says_so_who_pays_and_how_to_stop(self):
        note, on = self.said["keyTurned"]
        self.assertTrue(on, "the switch below shows it at once")
        self.assertIn("Meta sees pieces is now on", note)
        self.assertIn("this key pays", note)
        self.assertIn("Turn it off below", note)

    def test_the_chat_name_goes_when_the_chat_opens_another_recording(self):
        """Lane 5 11-01: a clip of a chicken pot under the name of the Sifu clip before it."""
        self.assertEqual(self.said["name"], ["Sifu", "New clip"])

    def test_the_cards_stay_last_in_the_chat_so_nobody_scrolls_back_up(self):
        s = self.said["order"]["later"]
        self.assertTrue(s["inLog"], "a run's own box would leave its cards above the next run")
        self.assertTrue(s["last"], "a clip made from one card is followed by the others")
        self.assertFalse(s["inCard"], "a folded card would take the basket with it")

    def test_a_chat_opened_from_home_lands_on_its_end_where_the_cards_are(self):
        self.assertEqual(self.said["open"]["top"], 900)

    def test_only_the_newest_cards_are_last(self):
        self.assertFalse(self.said["order"]["older"])
        self.assertTrue(self.said["order"]["newer"])

    def test_muses_pick_can_be_tapped_when_the_run_stopped_at_the_cards(self):
        s = self.said["pick"]
        self.assertEqual(s["before"], [True, False, False], "while Muse makes its pick the card has nothing to tap")
        self.assertEqual(s["after"], [False, False, False])
        self.assertEqual(s["first"], [False, False, False], "the stop can come before the cards")

    def test_a_preview_on_the_big_player_has_its_sound(self):
        s = self.said["sound"]
        self.assertEqual(s["videos"], 1)
        self.assertEqual(s["muted"], [False], "the player has no sound button, so a muted preview stays silent")

    def test_nothing_shows_before_the_first_run(self):
        self.assertTrue(self.said["run"]["start"]["hidden"])

    def test_the_bar_counts_the_steps_of_the_plan(self):
        s = self.said["run"]
        self.assertFalse(s["none"]["hidden"])
        self.assertEqual(s["none"]["says"], "In progress · 0%")
        self.assertEqual(s["first"]["says"], "In progress · 0%")
        self.assertEqual(s["two"]["says"], "In progress · 40%")
        self.assertEqual(s["two"]["width"], "40%")
        self.assertEqual(s["two"]["now"], "40")
        self.assertEqual(s["two"]["step"], "Step 3 of 5: Cut the clip")

    def test_the_bar_moves_inside_a_long_step(self):
        s = self.said["run"]["within"]
        self.assertEqual(s["says"], "In progress · 50%")
        self.assertEqual(s["step"], "Rendering the clip: 12 of 24 s (50%)")

    def test_a_longer_plan_never_takes_the_percent_back(self):
        self.assertEqual(self.said["run"]["replanned"]["says"], "In progress · 50%")

    def test_only_the_end_of_the_work_reads_a_hundred(self):
        s = self.said["run"]
        self.assertEqual(s["all"]["says"], "In progress · 99%")
        self.assertFalse(s["all"]["done"])
        self.assertEqual(s["end"]["says"], "Finished · 100%")
        self.assertEqual(s["end"]["width"], "100%")
        self.assertTrue(s["end"]["done"])

    def test_the_end_says_what_to_do_next(self):
        self.assertEqual(self.said["clipReady"]["step"], "Your clip is ready.")
        self.assertEqual(self.said["cards"]["step"], "Tap a card to make that clip.")

    def test_a_stopped_run_is_not_called_finished(self):
        s = self.said["stopped"]
        self.assertEqual(s["says"], "Stopped.")
        self.assertFalse(s["done"])
        self.assertEqual(s["width"], "0%")

    def test_making_the_video_shows_on_the_bar(self):
        s = self.said["video"]
        self.assertEqual(s["started"]["says"], "In progress · 0%")
        self.assertEqual(s["moving"]["says"], "In progress · 51%")
        self.assertTrue(s["ended"]["hidden"], "with no run behind it the bar goes away")
        self.assertEqual(s["line"], "")

    def test_a_finished_steps_percent_does_not_leak_into_the_next_step(self):
        s = self.said["nextJob"]
        self.assertEqual(s["says"], "In progress · 25%")
        self.assertEqual(s["step"], "Step 2 of 4: Cut the clip")


    def test_making_the_premiere_files_shows_on_the_bar_and_the_status(self):
        s = self.said["premiereBar"]
        self.assertEqual(s["bar"]["says"], "In progress · 50%")
        self.assertEqual(s["bar"]["step"], "Making the Premiere files: 19 of 38 s (50%)")
        self.assertEqual(s["status"], "Making the Premiere files. Press Stop to end it.")

    def test_a_page_opened_while_they_are_made_still_shows_it(self):
        self.assertEqual(self.said["premiereReload"], {"work": True, "says": "In progress · 50%"})

    def test_the_zip_is_saved_when_the_files_are_made_and_only_then(self):
        self.assertEqual(self.said["premiereEnd"], {"saved": 1, "want": False, "work": False})
        self.assertEqual(self.said["premiereReplay"]["saved"], 0, "opening the chat again saves nothing")
        self.assertEqual(self.said["premiereFailed"], {"saved": 0, "work": False})

    def test_a_press_saves_at_once_when_the_files_are_there_or_waits_for_them(self):
        self.assertEqual(self.said["pressReady"]["saved"], 1)
        self.assertEqual(self.said["pressMaking"], {"saved": 0, "want": True, "work": True})
        self.assertEqual(self.said["pressRefused"]["saved"], 0)
        self.assertFalse(self.said["pressRefused"]["work"])


    def test_a_run_stopped_at_the_cards_has_them_where_the_timeline_goes(self):
        s = self.said["stripOpens"]
        self.assertTrue(s["on"])
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["says"], "3 clips to choose from. Tap one to watch it.")
        self.assertTrue(s["back"], "no timeline to go back to yet")
        self.assertEqual(s["uses"], [False, False, False])
        self.assertIn(" on", s["first"], "the one on the big player is marked")

    def test_a_clip_on_the_timeline_stays_there_until_the_cards_are_pressed(self):
        self.assertFalse(self.said["stripOverClip"]["on"])

    def test_a_card_there_plays_on_the_big_player_and_the_cards_stay(self):
        s = self.said["stripWatch"]
        self.assertTrue(s["on"])
        self.assertEqual(s["showing"], 2)
        self.assertTrue(s["played"])
        self.assertEqual([" on" in c for c in s["picked"]], [False, True, False])

    def test_using_a_card_brings_the_timeline_back(self):
        self.assertEqual(self.said["stripUse"], {"on": False, "showing": 3})

    def test_a_new_clip_brings_the_timeline_back(self):
        self.assertFalse(self.said["stripClip"]["on"])

    def test_a_press_on_the_cards_in_the_chat_opens_them_but_not_on_use_this_one(self):
        s = self.said["chatTile"]
        self.assertFalse(s["before"], "a run that goes on to make its clip keeps the timeline's place empty for it")
        self.assertTrue(s["after"])
        self.assertEqual(s["showing"], 2)
        self.assertFalse(s["chipOpened"])

    def test_back_goes_to_the_timeline_only_when_there_is_one(self):
        s = self.said["stripBack"]
        self.assertTrue(s["none"])
        self.assertFalse(s["shown"])
        self.assertFalse(s["on"])
        self.assertIsNone(s["showing"], "the clip is back on the big player")

    def test_the_words_fixed_on_a_card_are_the_words_there(self):
        self.assertEqual(self.said["spelled"]["why"], "fixed")

    def test_a_card_in_the_chat_is_a_picture_to_press_and_the_big_player_has_the_sound(self):
        s = self.said["tileReady"]
        self.assertEqual(s["wait"], 0)
        self.assertEqual(s["kids"], ["img", "svg"], "one play mark, however many times it is told")
        self.assertFalse(s["inline"], "a player in the card as well would be two ways to watch and two sounds")


@unittest.skipUnless(shutil.which("node"), "node is not on this computer")
class TileDelete(unittest.TestCase):
    def test_a_chat_goes_only_after_a_yes_and_keep_takes_the_question_back(self):
        """Tarm 11-03: there was no way to delete a chat. Deleting what a person made asks first."""
        s = Page.said["del"]
        self.assertEqual(s["first"], [True, False, "Delete Trip"])
        self.assertEqual(s["words"], "Delete this chat? Its finished videos stay in work/clips.", "what stays, and where")
        self.assertEqual(s["asked"], [False, True, 0], "the bin only asks")
        self.assertEqual(s["kept"], [True, False, 0], "Keep sends nothing")
        self.assertEqual(s["sent"][0], "/api/chat/delete")
        self.assertTrue(s["off"], "a second tap does not send it twice")
        self.assertNotIn("/api/chat", s["sent"], "the page stays on the chat it shows")
        self.assertIn("/api/chats", s["sent"], "Home is drawn again")

    def test_the_page_follows_when_the_chat_it_shows_is_deleted(self):
        self.assertEqual(Page.said["del"]["shown"][:2], ["/api/chat/delete", "/api/chat"])

    def test_a_refused_delete_says_why_and_a_chat_at_work_has_no_bin(self):
        self.assertEqual(Page.said["del"]["refused"], ["This chat is working. Stop it first, then delete it."])
        self.assertTrue(Page.said["del"]["working"])

    def test_the_bin_stands_on_the_picture_and_is_big_enough_to_tap(self):
        with open(os.path.join(WEB, "style.css"), encoding="utf-8") as f:
            style = f.read()
        self.assertIn(".tile-over { position: absolute; inset: 0 0 auto 0; aspect-ratio: 9 / 16;", style)
        self.assertIn(".tile-bin::before { content: \"\"; position: absolute; inset: -6px; }", style)
        self.assertIn("background: var(--stop); border-color: var(--stop); color: var(--on-stop);", style)
        with open(os.path.join(WEB, "index.html"), encoding="utf-8") as f:
            self.assertIn('id="i-bin"', f.read())


class WorkClock(unittest.TestCase):
    def test_while_a_tool_works_the_card_says_so_and_its_clock_runs(self):
        """Tarm 11-03 on the EP0 take: "ตัวเลขนับวินาทีมันค้างนะ" - the card read "Thinking about step 9 · 3 s" for minutes
        while Muse Wonder worked, so he stopped a job that was alive."""
        s = Page.said["work"]
        self.assertEqual(s["thinking"], "Thinking about step 9 · 3 s")
        self.assertEqual(s["working"], "Working on step 9 · 4:12", "the time since the work began, from the studio")
        self.assertTrue(s["sameClock"], "a new line of the same work keeps the clock it has")
        self.assertEqual(s["line"], "Muse Wonder is watching who speaks, turn 6")
        self.assertEqual(s["next"], "Thinking about step 10 · 0 s")

    def test_what_the_chat_cost_moves_during_a_run(self):
        """The header read $0 with $0.87 spent, as it was asked only when the job ended."""
        s = Page.said["work"]
        self.assertEqual(s["costWhileWorking"], 1, "every 15 s while a tool works")
        self.assertEqual(s["costAtStep"], 1, "and when each step ends")


class Pointer(unittest.TestCase):
    def test_a_picture_in_the_page_cannot_be_dragged_in_as_a_file(self):
        """Tarm 11-03: dragging a style picture said "3d.png is not a video the studio reads". A file from outside starts
        no dragstart here, so it still drops."""
        self.assertEqual(Page.said["dragStopped"], 1)

    def test_every_sideways_row_shows_a_mouse_a_bar_and_nothing_later_takes_it_back(self):
        """Tarm 11-03: "ลองเช็คว่ามีอีกไหม ไม่งั้นใน pc เลื่อนไม่ได้". A touch keeps the clean row."""
        with open(os.path.join(WEB, "style.css"), encoding="utf-8") as f:
            style = f.read()
        hidden = re.findall(r"^\.([\w-]+) \{[^}]*overflow-x: (?:auto|scroll)[^}]*scrollbar-width: none", style, re.M)
        self.assertEqual(sorted(hidden), sorted(["w-strip", "chips", "look-cards", "look", "tools", "panel-row", "feature-tabs"]))
        block = style[style.index("@media (hover: hover) and (pointer: fine) {\n  .tools"):]
        self.assertNotIn("scrollbar-width: none", block, "a rule after the bar would hide it again")
        for name in hidden:
            self.assertRegex(block, r"\.%s[,\s][^}]*\{ scrollbar-width: thin;|\.%s \{ scrollbar-width: thin;" % (name, name))


class English(unittest.TestCase):
    def test_everything_a_person_reads_is_in_english(self):
        """Tarm 11-02: "ใช้ภาษาอังกฤษให้หมดเลย กรรมการเป็นฝรั่งทั้งหมด". Thai stays in what the program understands."""
        root = os.path.dirname(WEB)
        for name in ("web/app.js", "web/index.html", "web/replay.js", "clipstudio/features.json", "README.md"):
            with open(os.path.join(root, name), encoding="utf-8") as f:
                self.assertEqual(re.findall("[\u0e00-\u0e7f]+", f.read()), [], name)


class Plain(unittest.TestCase):
    def test_faces_are_asked_for_only_under_blur(self):
        s = Page.said["faces"]
        self.assertEqual(s["closed"], 0, "a clip on the player has no boxes on its faces")
        self.assertEqual(s["open"], 1)
        self.assertTrue(s["wide"], "the sheet leaves the picture in sight, as on Layout")

    def test_the_big_play_button_stays_off_the_faces(self):
        self.assertEqual(Page.said["bigPlay"], {"before": False, "after": True})

    def test_the_faces_show_while_blur_is_open_and_not_after(self):
        self.assertEqual(Page.said["blurSheet"], {"open": True, "shut": False, "other": False})

    def test_the_blur_tab_shows_when_the_clip_has_people(self):
        s = Page.said["blurTab"]
        self.assertEqual((s["people"], s["nobody"], s["line"]), (False, True, ""))

    def test_options_show_the_muses_meta_gives_and_the_rest_one_press_on(self):
        s = Page.said["options"]
        self.assertEqual(s["folded"], 3)
        self.assertEqual(s["machine"], 6, "About this computer still says all of them")
        self.assertEqual((s["opened"], s["button"], s["tiles"]), (6, True, False))

    def test_a_muse_that_is_ready_is_never_folded_away(self):
        self.assertEqual(Page.said["readyStays"], 4)

    def test_a_saved_key_for_another_api_opens_them(self):
        self.assertEqual(Page.said["otherSaved"], {"tiles": False, "minds": 6})

    def test_the_cost_sheet_lists_only_what_was_used_and_this_computer(self):
        s = Page.said["cost"]
        self.assertIn("Muse Glimmer", s["names"], "this computer stays: it is free")
        self.assertIn("Muse Spark", s["names"])
        self.assertEqual(sum("Open" in n for n in s["names"]), 0)
        self.assertEqual(s["note"], "Priced by the list. An answer cut short by Stop or by new words is not counted.")

    def test_no_sentence_on_the_page_runs_past_25_words(self):
        with open(os.path.join(WEB, "app.js"), encoding="utf-8") as f:
            page = f.read()
        long = [s for s in re.findall(r'"((?:[^"\\\n]|\\.)*)"', page) if len(s.split()) > 25]
        self.assertEqual(long, [])

    def test_every_face_box_and_icon_button_gets_its_name_on_hover(self):
        with open(os.path.join(WEB, "app.js"), encoding="utf-8") as f:
            page = f.read()
        self.assertIn('document.addEventListener("pointerover"', page)
        self.assertIn('if (b && !b.textContent.trim()) b.title = b.getAttribute("aria-label");', page)


class Button(unittest.TestCase):
    def test_update_video_is_the_strong_colour_and_grey_only_while_it_cannot_be_pressed(self):
        with open(os.path.join(WEB, "index.html"), encoding="utf-8") as f:
            page = f.read()
        with open(os.path.join(WEB, "style.css"), encoding="utf-8") as f:
            style = f.read()
        self.assertRegex(page, r'class="pill dark again go" id="againButton"')
        self.assertIn(".clip .pill.go { background: var(--st-accent);", style)
        self.assertRegex(style, r"\.clip \.pill\.again:disabled \{[^}]*background: var\(--st-raise\)")

    def test_the_cards_have_a_scrollbar_and_take_the_timelines_place(self):
        with open(os.path.join(WEB, "style.css"), encoding="utf-8") as f:
            style = f.read()
        self.assertRegex(re.search(r"\.basket-row \{[^}]*\}", style).group(0), r"overflow-x: scroll")
        self.assertRegex(re.search(r"\.moments \{[^}]*\}", style).group(0), r"scrollbar-width: auto; scrollbar-color")
        self.assertIn(".clip.basket-on .timeline, .clip.basket-on .pick-bar { display: none; }", style)
        self.assertRegex(re.search(r"\.basket \{[^}]*\}", style).group(0), r"display: none")
        self.assertIn(".clip.basket-on .basket { display: grid; }", style)

    def test_last_means_last_on_the_screen(self):
        with open(os.path.join(WEB, "style.css"), encoding="utf-8") as f:
            style = f.read()
        self.assertIn(".moments.last { order: 1; }", style)
        self.assertRegex(re.search(r"\.log \{[^}]*\}", style).group(0), r"display: flex; flex-direction: column")

    def test_the_bar_sits_on_the_clips_side_and_stays_in_sight(self):
        with open(os.path.join(WEB, "index.html"), encoding="utf-8") as f:
            page = f.read()
        with open(os.path.join(WEB, "style.css"), encoding="utf-8") as f:
            style = f.read()
        clip = page[page.index('id="clip"'):]
        self.assertLess(clip.index('id="jobBar"'), clip.index('id="timeline"'))
        self.assertRegex(re.search(r"\.job-bar \{[^}]*\}", style).group(0), r"position: sticky")


if __name__ == "__main__":
    unittest.main()
