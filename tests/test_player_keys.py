"""Space, the arrows and a press on the timeline, run from web/app.js itself in node with a small stand-in for the
page. python -m unittest tests.test_player_keys"""
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

function target(...matches) {
  return { closest: (q) => (q.split(", ").some((m) => matches.includes(m)) ? { closest: (p) => (matches.includes(p) ? {} : null) } : null) };
}

function load(page) {
  const found = {};
  const on = {};
  const hear = (name) => (k, fn) => ((on[name + " " + k] = on[name + " " + k] || []).push(fn));
  const shown = { display: "block" };
  const video = { paused: true, ended: false, duration: 20, currentTime: 5, classList: { contains: () => false }, play() { this.paused = false; return Promise.resolve(); }, pause() { this.paused = true; } };
  const later = [];
  const element = (id) => {
    if (found[id]) return found[id];
    const own = { addEventListener: hear(id), dataset: {}, style: {}, hidden: false, value: "" };
    if (id === "frame") own.querySelector = (q) => (q === "video" || (q === "video.base" && video.base) ? (video.none ? null : video) : any());
    if (id === "lanes") own.getBoundingClientRect = () => ({ left: 0, width: 200 });
    if (id === "clipAt") own.dispatchEvent = (e) => on["clipAt " + e.type].forEach((fn) => fn(e));
    if (id === "watch") own.hidden = true;
    return (found[id] = any(own));
  };
  const box = {
    Event: class { constructor(type) { this.type = type; } },
    console, Math, Number, String, Object, Array, JSON, Promise, Date, Set, Map, RegExp, Infinity, NaN, parseInt, parseFloat, isNaN, encodeURIComponent, decodeURIComponent,
    document: any({ getElementById: element, addEventListener: hear("document"), createElement: () => any(), body: any() }),
    window: any({ addEventListener: hear("window"), matchMedia: () => any({ matches: false }) }),
    history: any(), navigator: any(), localStorage: any({ getItem: () => null }), sessionStorage: any({ getItem: () => null }),
    fetch: () => new Promise(() => {}), EventSource: function () { return any(); }, ResizeObserver: function () { return any(); }, IntersectionObserver: function () { return any(); }, MutationObserver: function () { return any(); },
    getComputedStyle: () => shown, requestAnimationFrame: () => 0, cancelAnimationFrame: () => {}, setTimeout: (fn) => (later.push(fn), 0), clearTimeout: () => {}, setInterval: () => 0, clearInterval: () => {},
    Intl: any(), URL: any(), Image: function () { return any(); }, FormData: function () { return any(); }, CSS: any(), performance: any({ now: () => 0 }), innerWidth: 1280, innerHeight: 800, scrollTo: () => {},
  };
  vm.createContext(box);
  vm.runInContext(page + "\n;globalThis.t = { room, clipAt, watchVideo, watchBox, tapBlock, tapShot, seek, KEY_STEP };", box);
  const key = (k, over) => {
    const e = Object.assign({ key: k, target: target("body"), repeat: false, ctrlKey: false, altKey: false, metaKey: false, shiftKey: false, prevented: false, preventDefault() { this.prevented = true; } }, over);
    on["document keydown"].forEach((fn) => fn(e));
    return e.prevented;
  };
  const press = (x, over) => on["timeline pointerdown"].forEach((fn) => fn(Object.assign({ clientX: x, target: target(), preventDefault() {} }, over)));
  const move = (x) => (on["window pointermove"] || []).slice().forEach((fn) => fn({ clientX: x }));
  const up = () => {
    (on["window pointerup"] || []).slice().forEach((fn) => fn({}));
    on["window pointermove"] = [];
    on["window pointerup"] = [];
  };
  const mouse = () => on["window pointerdown"].forEach((fn) => fn({}));
  box.t.room.panel = { length: 20, frame: { shots: [] } };
  box.t.room.screen = "clip";
  return { t: box.t, video, shown, key, press, move, up, mouse, later };
}

const page = fs.readFileSync(process.argv[2], "utf8");
const said = {};
const each = (name, fn) => {
  const p = load(page);
  said[name] = fn(p);
};

each("space", (p) => {
  const prevented = p.key(" ");
  const playing = !p.video.paused;
  p.key(" ", { repeat: true });
  const held = !p.video.paused;
  p.key(" ");
  return { prevented, playing, held, again: p.video.paused };
});
each("arrows", (p) => {
  p.video.paused = false;
  const prevented = p.key("ArrowRight");
  const after = p.video.currentTime;
  const stopped = p.video.paused;
  p.key("ArrowLeft");
  p.key("ArrowLeft");
  const back = p.video.currentTime;
  p.video.currentTime = 0.1;
  p.key("ArrowLeft");
  const floor = p.video.currentTime;
  p.video.currentTime = 19.9;
  p.key("ArrowRight");
  return { prevented, after, stopped, back, floor, ceiling: p.video.currentTime, step: p.t.KEY_STEP };
});
each("typing", (p) => {
  const out = {};
  ["input", "textarea", "select", "[contenteditable]"].forEach((m) => {
    out[m] = [" ", "ArrowLeft", "ArrowRight"].map((k) => p.key(k, { target: target(m) })).some(Boolean) || !p.video.paused || p.video.currentTime !== 5;
  });
  return out;
});
each("modifiers", (p) => {
  const out = {};
  ["ctrlKey", "altKey", "metaKey", "shiftKey"].forEach((m) => {
    out[m] = p.key("ArrowRight", { [m]: true }) || p.key(" ", { [m]: true }) || !p.video.paused || p.video.currentTime !== 5;
  });
  return out;
});
each("hidden clip", (p) => {
  p.shown.display = "none";
  return p.key(" ") || p.key("ArrowRight") || !p.video.paused || p.video.currentTime !== 5;
});
each("button", (p) => {
  p.mouse();
  const byMouse = p.key(" ", { target: target("button") });
  const played = !p.video.paused;
  p.key("Tab");
  const byTab = p.key(" ", { target: target("button") });
  const arrow = p.key("ArrowRight", { target: target("button") });
  p.mouse();
  const again = p.key(" ", { target: target("button") });
  return { byMouse, played, byTab, stillPlaying: !p.video.paused || !arrow, arrow, again };
});
each("own controls", (p) => {
  const space = p.key(" ", { target: target("video[controls]") });
  const arrow = p.key("ArrowRight", { target: target("video[controls]") });
  return { space, played: !p.video.paused, arrow };
});
each("still picture", (p) => {
  p.video.none = true;
  p.t.room.made = { clip: { length: 20 }, version: 1 };
  p.t.clipAt.value = 5;
  const waiting = p.later.length;
  const prevented = p.key("ArrowRight");
  const after = Number(p.t.clipAt.value);
  const drawn = p.later.length > waiting;
  p.t.clipAt.value = 19.9;
  p.key("ArrowRight");
  return { prevented, after, drawn, ceiling: Number(p.t.clipAt.value) };
});
each("nothing to play", (p) => {
  p.video.none = true;
  return p.key(" ") || p.key("ArrowRight");
});
each("no length yet", (p) => {
  p.video.duration = NaN;
  p.key("ArrowRight");
  return p.video.currentTime;
});
each("watching", (p) => {
  p.t.watchBox.hidden = false;
  const w = p.t.watchVideo;
  w.paused = true;
  w.duration = 20;
  w.currentTime = 5;
  w.play = function () { this.paused = false; return Promise.resolve(); };
  w.pause = function () { this.paused = true; };
  p.key(" ");
  const playing = !w.paused;
  p.key("ArrowRight");
  return { playing, stopped: w.paused, at: w.currentTime, main: p.video.currentTime, mainPaused: p.video.paused };
});
each("click stops", (p) => {
  p.video.paused = false;
  p.press(100);
  const out = { stopped: p.video.paused, at: p.video.currentTime };
  p.up();
  return out;
});
each("click stops on a made clip", (p) => {
  p.video.paused = false;
  p.video.base = true;
  p.t.room.made = { clip: { length: 20 }, version: 1 };
  p.press(100);
  p.up();
  return { stopped: p.video.paused, at: p.video.currentTime };
});
each("a preview keeps playing", (p) => {
  p.video.paused = false;
  p.t.room.showing = { n: 1 };
  p.press(100);
  p.up();
  return !p.video.paused;
});
each("click on a block", (p) => {
  p.press(40, { target: target(".block") });
  const dragged = p.t.room.dragged;
  p.up();
  return { dragged, at: p.video.currentTime };
});
each("small shake on a block", (p) => {
  p.press(40, { target: target(".block") });
  p.move(42);
  const dragged = p.t.room.dragged;
  p.up();
  return { dragged, at: p.video.currentTime };
});
each("drag on a block", (p) => {
  p.press(40, { target: target(".block") });
  p.move(100);
  const out = { dragged: p.t.room.dragged, at: p.video.currentTime };
  const before = p.t.room.pick;
  p.t.tapBlock({ kind: "title", n: 1, text: "x", start: 1, end: 2 });
  out.picked = p.t.room.pick !== before;
  p.t.tapShot({ from: 1, to: 2 });
  out.frameSheet = p.t.room.shotAt !== undefined;
  p.up();
  p.later.forEach((fn) => fn());
  out.after = p.t.room.dragged;
  return out;
});
each("effect keeps its own drag", (p) => {
  p.press(40, { target: target(".block", ".lane.effect") });
  p.move(100);
  const out = { at: p.video.currentTime };
  p.up();
  return out;
});
console.log(JSON.stringify(said));
"""


@unittest.skipUnless(shutil.which("node"), "node is not on this computer")
class Keys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run = subprocess.run(["node", "-e", STAND_IN, "-", os.path.join(WEB, "app.js")], capture_output=True, text=True,
                             encoding="utf-8", timeout=60)
        if run.returncode:
            raise AssertionError(run.stderr[-2000:])
        cls.said = json.loads(run.stdout.strip().splitlines()[-1])

    def test_space_plays_and_pauses_once_for_each_press(self):
        s = self.said["space"]
        self.assertTrue(s["prevented"], "Space must not scroll the page too")
        self.assertTrue(s["playing"])
        self.assertTrue(s["held"], "a held key is one press")
        self.assertTrue(s["again"])

    def test_the_arrows_stop_the_clip_and_step_a_quarter_of_a_second_inside_it(self):
        a = self.said["arrows"]
        self.assertEqual(a["step"], 0.25)
        self.assertTrue(a["prevented"])
        self.assertTrue(a["stopped"])
        self.assertAlmostEqual(a["after"], 5.25)
        self.assertAlmostEqual(a["back"], 4.75)
        self.assertEqual(a["floor"], 0)
        self.assertAlmostEqual(a["ceiling"], 19.95, msg="the last step must not end the video")

    def test_keys_stay_with_whatever_the_person_is_typing_in(self):
        self.assertEqual([k for k, moved in self.said["typing"].items() if moved], [])

    def test_a_held_modifier_leaves_the_key_to_the_browser(self):
        self.assertEqual([k for k, moved in self.said["modifiers"].items() if moved], [])

    def test_keys_do_nothing_while_the_clip_is_not_on_screen(self):
        self.assertFalse(self.said["hidden clip"])

    def test_a_button_pressed_with_the_mouse_gives_space_to_the_clip_and_one_reached_with_tab_keeps_it(self):
        b = self.said["button"]
        self.assertTrue(b["byMouse"])
        self.assertTrue(b["played"])
        self.assertFalse(b["byTab"], "Space presses a button the person moved to with Tab")
        self.assertTrue(b["arrow"], "an arrow means nothing to a button")
        self.assertTrue(b["again"])

    def test_a_video_with_its_own_controls_keeps_space_or_it_would_be_pressed_twice(self):
        o = self.said["own controls"]
        self.assertFalse(o["space"])
        self.assertFalse(o["played"])
        self.assertTrue(o["arrow"])

    def test_the_arrows_walk_the_still_picture_while_there_is_no_video_yet(self):
        s = self.said["still picture"]
        self.assertTrue(s["prevented"])
        self.assertAlmostEqual(s["after"], 5.25)
        self.assertTrue(s["drawn"], "the picture must follow")
        self.assertAlmostEqual(s["ceiling"], 19.95)
        self.assertFalse(self.said["nothing to play"], "with no clip the page keeps its own keys")
        self.assertEqual(self.said["no length yet"], 5, "a video still loading has no place to step to")

    def test_the_keys_reach_the_older_clip_being_watched(self):
        w = self.said["watching"]
        self.assertTrue(w["playing"])
        self.assertTrue(w["stopped"])
        self.assertAlmostEqual(w["at"], 5.25)
        self.assertEqual(w["main"], 5)
        self.assertTrue(w["mainPaused"])

    def test_a_click_on_the_timeline_moves_the_clip_there_and_it_stops(self):
        c = self.said["click stops"]
        self.assertTrue(c["stopped"])
        self.assertAlmostEqual(c["at"], 10)
        c = self.said["click stops on a made clip"]
        self.assertTrue(c["stopped"])
        self.assertAlmostEqual(c["at"], 10.05, delta=0.01)
        self.assertTrue(self.said["a preview keeps playing"], "the timeline does not reach a moment's preview")

    def test_only_a_click_counts_on_a_block(self):
        c = self.said["click on a block"]
        self.assertFalse(c["dragged"])
        self.assertEqual(c["at"], 5)
        self.assertFalse(self.said["small shake on a block"]["dragged"], "a hand that shakes still clicks")
        d = self.said["drag on a block"]
        self.assertTrue(d["dragged"])
        self.assertAlmostEqual(d["at"], 10)
        self.assertFalse(d["picked"], "a drag must not pick the words")
        self.assertFalse(d["frameSheet"], "a drag must not open the shot")
        self.assertFalse(d["after"], "the next click must count")

    def test_an_effect_block_keeps_its_own_drag(self):
        self.assertEqual(self.said["effect keeps its own drag"]["at"], 5)


class Page(unittest.TestCase):
    def test_a_press_on_the_timeline_cannot_start_a_text_selection(self):
        with open(os.path.join(WEB, "style.css"), encoding="utf-8") as f:
            style = f.read()
        rule = re.search(r"^\.timeline \{(.*?)\}", style, re.M).group(1)
        self.assertRegex(rule, r"(^|[ ;])user-select: none")
        self.assertIn("-webkit-user-select: none", rule)

    def test_the_place_in_the_clip_can_hold_a_quarter_of_a_second(self):
        with open(os.path.join(WEB, "index.html"), encoding="utf-8") as f:
            page = f.read()
        step = float(re.search(r'id="clipAt"[^>]* step="([0-9.]+)"', page).group(1))
        self.assertAlmostEqual((0.25 / step) % 1, 0)


if __name__ == "__main__":
    unittest.main()
