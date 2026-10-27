"use strict";

const log = document.getElementById("log");
const statusLine = document.getElementById("status");
const composer = document.getElementById("composer");
const text = document.getElementById("text");
const send = document.getElementById("send");
const stopButton = document.getElementById("stop");
const newButton = document.getElementById("new");
const earlier = document.getElementById("earlier");
const earlierList = document.getElementById("earlierList");
const mindsBox = document.getElementById("minds");
const machineBox = document.getElementById("machine");
const settings = document.getElementById("settings");
const detailsBox = document.getElementById("details");
const darkBox = document.getElementById("dark");
const phone = window.matchMedia("(max-width: 760px)");
const clipBox = document.getElementById("clip");
const clipOpen = document.getElementById("clipOpen");
const clipHide = document.getElementById("clipHide");
const clipBack = document.getElementById("clipBack");
const clipPill = document.getElementById("clipPill");
const welcome = document.getElementById("welcome").cloneNode(true);
const frameBox = document.getElementById("frame");
const emptyFrame = frameBox.cloneNode(true).childNodes;
const clipLine = document.getElementById("clipLine");
const clipTitle = document.getElementById("clipTitle");
const clipTools = document.getElementById("clipTools");
const clipAt = document.getElementById("clipAt");
const clipAtLine = document.getElementById("clipAtLine");
const clipUndo = document.getElementById("clipUndo");
const fileInput = document.getElementById("file");
const dropCover = document.getElementById("dropCover");
const idle = text.placeholder;

const room = { mind: localStore("mind") || "local", minds: [], chat: null, source: null, last: -1, jobs: {}, open: new Set(), timer: null, cut: {}, showing: null };

function localStore(key, value) {
  try {
    if (value === undefined) return localStorage.getItem("lcs." + key);
    localStorage.setItem("lcs." + key, value);
  } catch (e) {
    return null;
  }
}

function el(tag, cls, words) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (words !== undefined && words !== null) e.textContent = words;
  return e;
}

function scrollDown() {
  log.scrollTop = log.scrollHeight;
}

function clearWelcome() {
  const w = document.getElementById("welcome");
  if (w) w.remove();
}

function bubble(kind, words, into) {
  clearWelcome();
  const div = el("div", "msg " + kind, words);
  (into || log).appendChild(div);
  scrollDown();
  return div;
}

function number(n) {
  return n === null || n === undefined ? null : Number(n).toLocaleString("en-US");
}

function engineLine(e, label) {
  const parts = label ? [label] : [];
  if (e.read_new !== null && e.read_new !== undefined) parts.push("read " + number(e.read_new) + " tokens new");
  if (e.reused) parts.push("reused " + number(e.reused));
  if (e.read_per_s) parts.push("reading " + number(e.read_per_s) + "/s");
  if (e.write_per_s) parts.push("writing " + e.write_per_s + "/s");
  parts.push(e.seconds + " s");
  return parts.join(" · ");
}

function fold(summary, body, cls) {
  const d = el("details", "detail");
  d.append(el("summary", null, summary), el("p", cls, body));
  return d;
}

function inputLine(input) {
  if (!input) return "";
  if (typeof input === "string") return input;
  return Object.entries(input)
    .map(([k, v]) => k + ": " + (Array.isArray(v) ? v.join(" / ") : v))
    .join(" · ");
}

function clock(seconds) {
  const s = Math.round(seconds);
  const mm = String(Math.floor((s % 3600) / 60)).padStart(s >= 3600 ? 2 : 1, "0");
  return (s >= 3600 ? Math.floor(s / 3600) + ":" : "") + mm + ":" + String(s % 60).padStart(2, "0");
}

function when(seconds) {
  return new Date(seconds * 1000).toLocaleString("en-GB", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

function job(n) {
  if (!room.jobs[n]) {
    clearWelcome();
    const box = el("section", "job");
    box.setAttribute("aria-label", "Muse's work on message " + n);
    const plan = el("div", "plan");
    plan.hidden = true;
    const flow = el("div", "flow");
    const live = el("div", "live");
    live.hidden = true;
    const liveLine = el("p", "live-line");
    const liveThinking = el("p", "live-thinking detail");
    live.append(liveLine, liveThinking);
    const foot = el("p", "foot");
    foot.hidden = true;
    box.append(plan, flow, live, foot);
    log.appendChild(box);
    room.jobs[n] = { box, plan, flow, live, liveLine, liveThinking, foot, mind: "", steps: {}, answers: [] };
  }
  return room.jobs[n];
}

function drawPlan(j, e) {
  j.plan.textContent = "";
  j.plan.hidden = false;
  j.plan.appendChild(el("p", "plan-head", e.changed ? "Muse changed its plan" : "Muse's plan"));
  if (e.why) j.plan.appendChild(el("p", "plan-why", e.why));
  const list = el("ol", "plan-items");
  const said = { done: "done", now: "working on it", todo: "not started" };
  e.items.forEach((item) => {
    const li = el("li", item.state);
    const mark = el("span", "mark");
    mark.setAttribute("aria-hidden", "true");
    li.append(mark, el("span", null, item.text), el("span", "sr", " (" + said[item.state] + ")"));
    list.appendChild(li);
  });
  j.plan.appendChild(list);
}

function drawStep(j, e) {
  const div = el("div", "step" + (e.refused ? " refused" : ""));
  const head = el("p", "step-head");
  const label = e.look ? "Looked at " + clock(e.look.start) + "–" + clock(e.look.end) : e.label || e.tool;
  head.append(el("span", "n", String(e.step)), el("span", "label", label));
  div.appendChild(head);
  const said = e.said ? el("p", "said", e.said) : null;
  if (said) div.appendChild(said);
  const why = el("p", "why", e.why || "Muse gave no reason for this step.");
  div.appendChild(why);
  j.steps[e.step] = { why: why, said: said };
  if (e.look) div.appendChild(lookStrip(e.look));
  if (e.look && e.look.said && e.look.said.length) div.appendChild(saidFold(e.look.said));
  if (e.look && e.look.took) div.appendChild(el("p", "engine detail", "Heard in " + e.look.took + " s"));
  if (e.moments && e.fixing) div.appendChild(el("p", "engine detail", "Misheard words fixed by Muse in " + e.fixing + " s"));
  if (e.refused) div.appendChild(el("p", "result simple", "That did not work, so Muse tried again."));
  const input = inputLine(e.input);
  if (input) div.appendChild(el("p", "input detail", input));
  const lines = String(e.result || "").split("\n");
  const first = e.refused ? (e.guard ? "Refused, already answered: " : "Refused: ") + lines[0] : lines[0];
  div.appendChild(el("p", "result detail", first));
  if (lines.length > 1) div.appendChild(fold("Full result", lines.slice(1).join("\n"), "more"));
  if (e.thinking) div.appendChild(fold("How Muse decided", e.thinking, "thinking"));
  if (e.engine) div.appendChild(el("p", "engine detail", engineLine(e.engine, j.mind)));
  j.flow.appendChild(div);
  if (e.moments) drawMoments(j, e.moments);
}

function lookStrip(look) {
  const row = el("div", "look");
  look.pictures.forEach((p, i) => {
    const fig = el("figure");
    const img = el("img");
    img.src = "/api/frame/" + p.frame;
    img.alt = "Picture " + (i + 1) + " of " + look.pictures.length + ", at " + clock(p.t);
    img.loading = "lazy";
    fig.append(img, el("figcaption", null, clock(p.t)));
    row.appendChild(fig);
  });
  return row;
}

function saidFold(said) {
  const d = el("details", "said-fold");
  d.appendChild(el("summary", null, "What was said (" + plural(said.length, "line") + ")"));
  const list = el("ul", "said-lines");
  said.forEach((p) => {
    const li = el("li");
    li.append(el("span", "at", clock(p.from)), el("span", null, p.text));
    if (p.kept) li.appendChild(el("span", "heard detail", "As heard, not fixed: " + p.kept + "."));
    else if (p.heard && p.heard !== p.text) li.appendChild(el("span", "heard detail", "Heard as: " + p.heard));
    list.appendChild(li);
  });
  d.appendChild(list);
  return d;
}

function span(m) {
  return clock(m.start) + "–" + clock(m.end);
}

function drawMoments(j, moments) {
  document.querySelectorAll(".moments .chip").forEach((b) => (b.disabled = true));
  const div = el("div", "moments");
  div.appendChild(el("p", "moments-head", "The moments Muse found"));
  const list = el("ol", "moment-list");
  moments.forEach((m) => {
    const li = el("li", "moment");
    li.dataset.preview = m.preview;
    const media = el("div", "moment-media");
    const img = el("img");
    img.src = "/api/frame/" + m.frame;
    img.alt = "A picture from moment " + m.n + ", at " + clock(m.t);
    media.append(img, el("span", "moment-wait", "Cutting a preview…"));
    const body = el("div", "moment-body");
    body.append(el("p", "moment-head", "Moment " + m.n + " · " + span(m) + " · " + Math.round(m.end - m.start) + " s"), el("p", "moment-why", m.reason));
    if (m.said && m.said.length) body.appendChild(saidFold(m.said));
    if (m.n === 1) {
      body.appendChild(el("p", "moment-pick", "Muse starts with this one"));
    } else {
      const b = el("button", "chip", "Use this one instead");
      b.type = "button";
      b.addEventListener("click", () => {
        showMoment(m);
        say("(tapped) Use moment " + m.n + " (" + clock(m.start) + "-" + clock(m.end) + ") instead.");
      });
      body.appendChild(b);
    }
    li.append(media, body);
    list.appendChild(li);
    if (room.cut[m.preview]) fillPreview(li, room.cut[m.preview]);
  });
  div.appendChild(list);
  j.flow.appendChild(div);
  showMoment(moments[0]);
}

function fillPreview(li, e) {
  const media = li.querySelector(".moment-media");
  if (!e.ok) {
    media.querySelector(".moment-wait").textContent = "No preview: " + e.error;
    return;
  }
  const v = el("video");
  v.src = "/api/preview/" + e.name;
  v.poster = media.querySelector("img").src;
  v.controls = true;
  v.preload = "metadata";
  v.playsInline = true;
  media.replaceChildren(v);
}

function showMoment(m) {
  room.showing = m;
  const ready = room.cut[m.preview];
  if (ready && ready.ok) {
    const v = el("video");
    v.src = "/api/preview/" + m.preview;
    v.muted = true;
    v.autoplay = true;
    v.loop = true;
    v.playsInline = true;
    v.controls = true;
    v.setAttribute("aria-label", "A rough preview of moment " + m.n);
    frameBox.replaceChildren(v);
  } else {
    const img = el("img");
    img.src = "/api/frame/" + m.frame;
    img.alt = "A picture from moment " + m.n;
    frameBox.replaceChildren(img);
  }
  clipTitle.textContent = "Moment " + m.n;
  clipPill.textContent = "Show moment " + m.n;
  clipLine.textContent = span(m) + " of your video. " + (ready && ready.ok ? "A rough preview" : "Cutting a rough preview") + "; your clip shows here instead once Muse makes it.";
  popClip();
}

function drawAnswer(j, e) {
  const div = bubble("muse", null, j.flow);
  const words = el("p", "words", e.text);
  div.appendChild(words);
  j.answers.push(words);
  if (e.thinking) div.appendChild(fold("How Muse decided", e.thinking, "thinking"));
  if (e.engine) div.appendChild(el("p", "engine detail", engineLine(e.engine, e.mind)));
}

function drawGuesses(j, e) {
  document.querySelectorAll(".guesses .chip").forEach((b) => (b.disabled = true));
  const c = e.clip;
  const div = el("div", "guesses");
  div.appendChild(el("p", "guesses-head", "Muse's guesses"));
  const facts = [c.title, "about " + c.length_s + " s", c.style + " style"];
  if (c.language) facts.push(c.language);
  div.appendChild(el("p", "facts", facts.join(" · ")));
  if (c.brief) div.appendChild(el("p", "brief", "What you asked for: " + c.brief));
  const row = el("div", "chips");
  e.chips.forEach((chip) => {
    const b = el("button", "chip", chip.label);
    b.type = "button";
    b.addEventListener("click", () => say(chip.say));
    row.appendChild(b);
  });
  div.appendChild(row);
  j.flow.appendChild(div);
}

function plural(n, one) {
  return n + " " + one + (n === 1 ? "" : "s");
}

function stopTimer() {
  if (room.timer) clearInterval(room.timer);
  room.timer = null;
}

function hideLive(j) {
  stopTimer();
  j.live.hidden = true;
  j.liveThinking.textContent = "";
}

function setBusy() {
  const busy = room.open.size > 0;
  stopButton.hidden = !busy;
  stopButton.disabled = false;
  text.placeholder = busy ? "Type here to change what Muse is doing" : idle;
  showStatus();
}

const TAPPED = "(tapped) ";
const OPENED = "(opened) ";

function personWords(text) {
  if (text.startsWith(TAPPED)) return ["you tapped", "Tapped: " + text.slice(TAPPED.length)];
  if (text.startsWith(OPENED)) {
    const lines = text.slice(OPENED.length).split("\n");
    const parts = lines[0].split(" · ");
    return ["you opened", ["Opened: " + parts[0] + " · " + parts[1]].concat(lines.slice(1)).join("\n")];
  }
  return ["you", text];
}

function clipView(view) {
  clipBox.dataset.view = view;
  clipPill.hidden = view !== "tucked";
}

function popClip() {
  if (clipBox.dataset.view === "full") return;
  if (phone.matches) {
    clipView("tucked");
    return;
  }
  clipView("pip");
  clipBox.classList.remove("popping");
  void clipBox.offsetWidth;
  clipBox.classList.add("popping");
}

function clearClip() {
  frameBox.replaceChildren(...[...emptyFrame].map((n) => n.cloneNode(true)));
  clipLine.textContent = "";
  clipTitle.textContent = "Your clip";
  clipTools.hidden = true;
  room.made = null;
  clipView("none");
}

function showClip(e) {
  const clip = e.clip;
  if (!clip) {
    room.made = null;
    clipTools.hidden = true;
    clipLine.textContent = "The clip was taken back. Ask Muse to make it again.";
    return;
  }
  room.made = { clip, version: e.id };
  room.showing = null;
  clipAt.max = clip.length;
  const entry = e.kind === "edit" ? e.entry : null;
  if (entry && entry.do === "title") clipAt.value = (entry.start + entry.end) / 2;
  else if (entry && entry.do === "clip") clipAt.value = Math.min(clip.length / 2, clip.length - 0.1);
  drawClipFrame();
  const titles = clip.titles.length;
  clipTitle.textContent = "Your clip";
  clipPill.textContent = "Show your clip";
  clipLine.textContent = "Moment " + clip.moment + " · " + clip.length + " s · the " + clip.layout + " layout · " + (titles ? titles + (titles === 1 ? " title" : " titles") : "no words yet") + ". Slide to see any second of it.";
  clipTools.hidden = false;
  popClip();
}

function showFinished(file) {
  room.made = null;
  room.showing = null;
  const v = el("video");
  v.src = "/api/clip/" + file.name;
  v.poster = "/api/clip/" + file.name.replace(/\.mp4$/, ".jpg");
  v.controls = true;
  v.playsInline = true;
  v.preload = "metadata";
  v.setAttribute("aria-label", "Your finished clip");
  frameBox.replaceChildren(v);
  clipTools.hidden = true;
  clipTitle.textContent = "Your clip";
  clipPill.textContent = "Show your clip";
  clipLine.textContent = "The finished clip · " + file.length + " s · " + (file.size / 1e6).toFixed(1) + " MB.";
  popClip();
}

function drawChecked(j, e) {
  const due = e.checks.filter((c) => c.ok !== null);
  const passed = due.filter((c) => c.ok).length;
  const div = el("div", "checks" + (passed === due.length ? " all" : ""));
  div.appendChild(el("p", "checks-head", "Checked " + passed + " of " + due.length + " on the finished file"));
  const list = el("ul", "checks-list");
  e.checks.forEach((c) => {
    const li = el("li", c.ok === null ? "skip" : c.ok ? "pass" : "fail");
    const mark = el("span", "mark", c.ok === null ? "–" : c.ok ? "✓" : "✗");
    mark.setAttribute("aria-hidden", "true");
    const words = el("span", "what");
    words.append(el("span", "sr", c.ok === null ? (c.who === "Muse" ? "Left to Muse: " : "Not needed: ") : c.ok ? "Passed: " : "Failed: "), el("b", null, c.name), el("small", null, c.found + (c.who === "program" ? "" : " · " + c.who)));
    li.append(mark, words);
    list.appendChild(li);
  });
  div.appendChild(list);
  j.flow.appendChild(div);
}

function drawReady(j, e) {
  const div = el("div", "ready");
  div.appendChild(el("p", "ready-head", "Ready to post"));
  const v = el("video");
  v.src = "/api/clip/" + e.file;
  v.poster = "/api/clip/" + e.file.replace(/\.mp4$/, ".jpg");
  v.controls = true;
  v.playsInline = true;
  v.preload = "metadata";
  v.setAttribute("aria-label", "Your finished clip");
  const body = el("div", "ready-body");
  body.append(el("p", "ready-title", e.title), el("p", "ready-caption", e.caption), el("p", "ready-tags", e.hashtags.join(" ")));
  if (e.english) body.appendChild(el("p", "ready-english", e.english));
  const all = [e.title, e.caption, e.hashtags.join(" ")].concat(e.english ? [e.english] : []).join("\n\n");
  const copy = el("button", "chip", "Copy all");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(all);
      copy.textContent = "Copied";
    } catch (err) {
      copy.textContent = "Select the text to copy it";
    }
  });
  const save = el("a", "chip", "Download the clip");
  save.href = "/api/clip/" + e.file + "?download=1";
  const row = el("div", "ready-actions");
  row.append(copy, save);
  body.appendChild(row);
  div.append(v, body);
  j.flow.appendChild(div);
}

function drawClipFrame() {
  const made = room.made;
  if (!made) return;
  const t = Math.min(Number(clipAt.value), Math.max(0, made.clip.length - 0.05));
  clipAtLine.textContent = t.toFixed(1) + " s";
  const img = el("img");
  img.src = "/api/clipframe?chat=" + encodeURIComponent(room.chat) + "&t=" + t.toFixed(2) + "&v=" + made.version;
  img.alt = "Your clip at " + t.toFixed(1) + " seconds";
  frameBox.replaceChildren(img);
}

clipAt.addEventListener("input", () => {
  clearTimeout(room.frameTimer);
  room.frameTimer = setTimeout(drawClipFrame, 120);
});

clipUndo.addEventListener("click", async () => {
  clipUndo.disabled = true;
  try {
    const d = await post("/api/undo", { chat: room.chat });
    if (!d.ok) clipLine.textContent = d.error || "Nothing was taken back.";
  } catch (err) {
    clipLine.textContent = "The studio is not answering. Start it again with: python app.py";
  }
  clipUndo.disabled = false;
});

function showRecording(e) {
  const r = e.recording;
  if (e.frame) {
    const img = el("img");
    img.src = "/api/frame/" + e.frame;
    img.alt = "A picture from " + r.name;
    frameBox.replaceChildren(img);
  } else {
    frameBox.replaceChildren(el("p", "hint", "Opened " + r.name));
  }
  clipTitle.textContent = "Your video";
  clipPill.textContent = "Show your video";
  clipLine.textContent = r.name + " · " + clock(r.seconds) + " long. This is the video you opened. Your clip shows here instead once Muse makes it.";
  popClip();
}

const ENDS = {
  stopped: "Stopped",
  closed: "The studio was closed during this job",
  mind: "Ended: Muse could not answer",
  fault: "Ended: the studio broke",
  repeating: "Ended: Muse kept repeating itself",
  "too many steps": "Ended: too many steps",
};

function on(e) {
  if (e.id <= room.last) return;
  room.last = e.id;
  if (e.kind === "person") {
    const [kind, words] = personWords(e.text);
    bubble(kind, words, e.cut_in && room.jobs[e.job] ? room.jobs[e.job].flow : null);
    return;
  }
  if (e.kind === "opened") {
    showRecording(e);
    return;
  }
  if (e.kind === "edit" || e.kind === "undo") {
    showClip(e);
    return;
  }
  if (e.kind === "preview") {
    room.cut[e.name] = e;
    document.querySelectorAll(".moment").forEach((li) => {
      if (li.dataset.preview === e.name) fillPreview(li, e);
    });
    if (room.showing && room.showing.preview === e.name) showMoment(room.showing);
    return;
  }
  const j = job(e.job);
  if (e.kind === "job") {
    j.mind = e.mind;
    room.open.add(e.job);
    setBusy();
  } else if (e.kind === "plan") {
    drawPlan(j, e);
  } else if (e.kind === "guesses") {
    drawGuesses(j, e);
  } else if (e.kind === "wait") {
    stopTimer();
    const tick = () => {
      j.liveLine.textContent = "Muse is thinking about step " + e.step + " · " + Math.max(0, Math.round(Date.now() / 1000 - e.t)) + " s";
    };
    tick();
    j.live.hidden = false;
    j.box.appendChild(j.live);
    room.timer = setInterval(tick, 1000);
  } else if (e.kind === "step") {
    hideLive(j);
    drawStep(j, e);
  } else if (e.kind === "answer") {
    hideLive(j);
    drawAnswer(j, e);
  } else if (e.kind === "rendered") {
    showFinished(e.file);
  } else if (e.kind === "checked") {
    drawChecked(j, e);
  } else if (e.kind === "spelled") {
    e.fixes.forEach((f) => {
      const spot = f.kind === "step" ? (j.steps[f.step] || {})[f.field] : j.answers[f.index];
      if (spot) spot.textContent = f.text;
    });
  } else if (e.kind === "ready") {
    drawReady(j, e);
  } else if (e.kind === "cut") {
    hideLive(j);
    bubble("note", e.by === "stop" ? "Stopped while Muse was thinking about step " + e.step + "." : "Muse stopped thinking about step " + e.step + " to read your message.", j.flow);
  } else if (e.kind === "note") {
    bubble("note", e.text, j.flow);
  } else if (e.kind === "end") {
    hideLive(j);
    const parts = [ENDS[e.why] || null, plural(e.steps, "step")].filter(Boolean);
    if (e.refused) parts.push(e.refused + " refused" + (e.fixed ? ", " + e.fixed + " fixed by Muse" : ""));
    if (e.seconds) parts.push(e.seconds + " s");
    const plain = (ENDS[e.why] || "Muse finished") + (e.seconds ? " in " + Math.max(1, Math.round(e.seconds)) + " s" : "") + ".";
    j.foot.replaceChildren(el("span", "simple", plain), el("span", "detail", parts.join(" · ")));
    j.foot.hidden = false;
    room.open.delete(e.job);
    setBusy();
  }
  scrollDown();
}

function onThinking(live) {
  const j = room.jobs[live.job];
  if (!j) return;
  if (live.work) {
    stopTimer();
    j.liveLine.textContent = live.work;
    j.liveThinking.textContent = "";
    j.live.hidden = false;
    j.box.appendChild(j.live);
    scrollDown();
    return;
  }
  if (j.live.hidden) return;
  const words = (live.thinking + (live.text ? "\n" + live.text : "")).trim();
  j.liveThinking.textContent = words.length > 1500 ? "…" + words.slice(-1500) : words;
  j.liveThinking.scrollTop = j.liveThinking.scrollHeight;
  scrollDown();
}

function current() {
  return room.minds.find((m) => m.key === room.mind) || room.minds[0];
}

function showStatus() {
  const m = current();
  if (room.open.size) {
    statusLine.textContent = "Muse is working on it. Type to change what it does, or press Stop.";
  } else if (m) {
    statusLine.textContent = m.ok ? m.label + " is ready, " + m.where + "." : m.label + " is not ready yet. Open Settings to see why.";
  }
}

function drawMinds() {
  mindsBox.textContent = "";
  machineBox.textContent = "";
  room.minds.forEach((m) => {
    const b = el("button", "mind");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(m.key === room.mind));
    b.disabled = !m.ok;
    const dot = el("span", "dot " + (m.ok ? "ok" : "off"));
    dot.setAttribute("aria-hidden", "true");
    b.append(dot, el("span", null, m.label), el("small", null, m.ok ? m.where.charAt(0).toUpperCase() + m.where.slice(1) : "Not ready"));
    b.addEventListener("click", () => {
      room.mind = m.key;
      localStore("mind", m.key);
      drawMinds();
    });
    mindsBox.appendChild(b);
    machineBox.appendChild(el("p", null, m.label + ": " + (m.ok ? m.name + " at " + m.url : m.why)));
  });
  const h = room.hearing;
  if (h) machineBox.appendChild(el("p", null, h.label + ": " + (h.ok ? h.name + " at " + h.url : h.why)));
  showStatus();
}

function showDetails(on) {
  document.body.classList.toggle("details", on);
  detailsBox.checked = on;
}

detailsBox.addEventListener("change", () => {
  localStore("details", detailsBox.checked ? "1" : "0");
  showDetails(detailsBox.checked);
});

document.addEventListener("click", (e) => {
  [settings, earlier].forEach((menu) => {
    if (menu.open && !menu.contains(e.target)) menu.open = false;
  });
});

clipOpen.addEventListener("click", () => clipView("full"));
clipHide.addEventListener("click", () => clipView("tucked"));
clipBack.addEventListener("click", () => clipView(phone.matches ? "tucked" : "pip"));
clipPill.addEventListener("click", () => clipView("full"));

function showDark(on) {
  if (on) document.documentElement.dataset.theme = "dark";
  else delete document.documentElement.dataset.theme;
  darkBox.checked = on;
}

darkBox.addEventListener("change", () => {
  localStore("theme", darkBox.checked ? "dark" : "light");
  showDark(darkBox.checked);
});

async function health() {
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    const d = await r.json();
    room.minds = d.minds || [];
    room.hearing = d.hearing || null;
    if (!room.minds.some((m) => m.key === room.mind && m.ok)) {
      const ready = room.minds.find((m) => m.key === d.default && m.ok) || room.minds.find((m) => m.ok);
      room.mind = ready ? ready.key : d.default;
    }
    document.body.classList.toggle("away", d.here === false);
    drawMinds();
  } catch (e) {
    statusLine.textContent = "The studio is not answering. Start it again with: python app.py";
  }
}

function show(chatId) {
  if (room.source) room.source.close();
  stopTimer();
  room.chat = chatId;
  room.last = -1;
  room.jobs = {};
  room.open = new Set();
  room.cut = {};
  room.showing = null;
  log.replaceChildren(welcome.cloneNode(true));
  clearClip();
  fillNewest();
  setBusy();
  const source = new EventSource("/api/events?chat=" + encodeURIComponent(chatId));
  source.onmessage = (m) => on(JSON.parse(m.data));
  source.addEventListener("think", (m) => onThinking(JSON.parse(m.data)));
  source.addEventListener("moved", () => {
    source.close();
    connect();
  });
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) setTimeout(connect, 2000);
  };
  room.source = source;
}

async function connect(chatId) {
  try {
    const r = await fetch("/api/chat" + (chatId ? "?id=" + encodeURIComponent(chatId) : ""), { cache: "no-store" });
    const d = await r.json();
    if (!d.ok) {
      bubble("note", d.error);
      return;
    }
    if (d.chat !== room.chat || !room.source || room.source.readyState === EventSource.CLOSED) show(d.chat);
  } catch (e) {
    statusLine.textContent = "The studio is not answering. Start it again with: python app.py";
    setTimeout(connect, 3000);
  }
}

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}

async function say(words) {
  try {
    const d = await post("/api/say", { chat: room.chat, message: words, mind: room.mind });
    if (d.ok) return true;
    bubble("note", d.error || "Something went wrong.");
  } catch (err) {
    bubble("note", "The studio is not answering. Start it again with: python app.py");
  }
  return false;
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const words = text.value.trim();
  if (!words || !room.chat) return;
  text.value = "";
  if (!(await say(words))) text.value = words;
  text.focus();
});

stopButton.addEventListener("click", async () => {
  stopButton.disabled = true;
  try {
    await post("/api/stop", { chat: room.chat });
  } catch (err) {
    bubble("note", "The studio is not answering. Start it again with: python app.py");
    stopButton.disabled = false;
  }
});

newButton.addEventListener("click", async () => {
  try {
    const d = await post("/api/chat/new", {});
    if (d.ok) show(d.chat);
  } catch (err) {
    bubble("note", "The studio is not answering. Start it again with: python app.py");
  }
  earlier.open = false;
  text.focus();
});

earlier.addEventListener("toggle", async () => {
  if (!earlier.open) return;
  earlierList.replaceChildren(el("p", "earlier-empty", "Looking…"));
  try {
    const r = await fetch("/api/chats", { cache: "no-store" });
    const d = await r.json();
    earlierList.textContent = "";
    if (!d.chats.length) earlierList.appendChild(el("p", "earlier-empty", "No earlier chats yet."));
    d.chats.forEach((c) => {
      const b = el("button", "earlier-chat" + (c.id === room.chat ? " here" : ""));
      b.type = "button";
      b.append(el("span", "first", c.recording || c.first), el("span", "meta", when(c.when)));
      b.addEventListener("click", () => {
        earlier.open = false;
        connect(c.id);
      });
      earlierList.appendChild(b);
    });
  } catch (err) {
    earlierList.replaceChildren(el("p", "earlier-empty", "The studio is not answering."));
  }
});

async function fillNewest() {
  const box = log.querySelector(".welcome .newest");
  if (!box) return;
  try {
    const r = await fetch("/api/recordings", { cache: "no-store" });
    const d = await r.json();
    box.textContent = "";
    if (!d.recordings.length) box.appendChild(el("p", "earlier-empty", "No videos found in " + (d.folders.join(", ") || "any folder") + "."));
    d.recordings.forEach((rec) => {
      const b = el("button", "recording");
      b.type = "button";
      b.append(el("span", "first", rec.name), el("span", "meta", [rec.length, rec.size, rec.when].filter(Boolean).join(" · ")));
      b.addEventListener("click", () => openRecording({ path: rec.path }));
      box.appendChild(b);
    });
  } catch (err) {
    box.replaceChildren(el("p", "earlier-empty", "The studio is not answering."));
  }
}

async function openRecording(body) {
  const words = text.value.trim();
  try {
    const d = await post("/api/open", Object.assign({ chat: room.chat, words: words, mind: room.mind, language: navigator.language || "" }, body));
    if (d.ok) {
      text.value = "";
      return;
    }
    bubble("note", d.error || "Something went wrong.");
  } catch (err) {
    bubble("note", "The studio is not answering. Start it again with: python app.py");
  }
}

function openFile(file) {
  if (file) openRecording({ name: file.name, size: file.size, mtime: file.lastModified / 1000 });
}

log.addEventListener("click", (e) => {
  if (e.target.closest(".pick")) fileInput.click();
});

fileInput.addEventListener("change", () => {
  openFile(fileInput.files[0]);
  fileInput.value = "";
});

function carriesFiles(e) {
  return e.dataTransfer && [...e.dataTransfer.types].includes("Files");
}

window.addEventListener("dragover", (e) => {
  if (!carriesFiles(e)) return;
  e.preventDefault();
  dropCover.hidden = false;
});

window.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) dropCover.hidden = true;
});

window.addEventListener("drop", (e) => {
  if (!carriesFiles(e)) return;
  e.preventDefault();
  dropCover.hidden = true;
  openFile(e.dataTransfer.files[0]);
});

text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

showDetails(localStore("details") === "1");
showDark(localStore("theme") === "dark");
health();
connect();
