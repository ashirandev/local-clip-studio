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
const welcome = document.getElementById("welcome").cloneNode(true);
const frameBox = document.getElementById("frame");
const emptyFrame = frameBox.cloneNode(true).childNodes;
const clipLine = document.getElementById("clipLine");
const clipTitle = document.getElementById("clipTitle");
const fileInput = document.getElementById("file");
const dropCover = document.getElementById("dropCover");
const idle = text.placeholder;

const room = { mind: localStore("mind") || "local", minds: [], chat: null, source: null, last: -1, jobs: {}, open: new Set(), timer: null };

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
  const d = el("details");
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
    const liveThinking = el("p", "live-thinking");
    live.append(liveLine, liveThinking);
    const foot = el("p", "foot");
    foot.hidden = true;
    box.append(plan, flow, live, foot);
    log.appendChild(box);
    room.jobs[n] = { box, plan, flow, live, liveLine, liveThinking, foot, mind: "" };
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
  head.append(el("span", "n", String(e.step)), el("span", "label", e.label || e.tool));
  div.appendChild(head);
  if (e.said) div.appendChild(el("p", "said", e.said));
  div.appendChild(el("p", "why", e.why || "Muse gave no reason for this call."));
  const input = inputLine(e.input);
  if (input) div.appendChild(el("p", "input", input));
  const lines = String(e.result || "").split("\n");
  const first = e.refused ? (e.guard ? "Refused, already answered: " : "Refused: ") + lines[0] : lines[0];
  div.appendChild(el("p", "result", first));
  if (lines.length > 1) div.appendChild(fold("Full result", lines.slice(1).join("\n"), "more"));
  if (e.thinking) div.appendChild(fold("How Muse decided", e.thinking, "thinking"));
  if (e.engine) div.appendChild(el("p", "engine", engineLine(e.engine, j.mind)));
  j.flow.appendChild(div);
}

function drawAnswer(j, e) {
  const div = bubble("muse", null, j.flow);
  div.appendChild(el("p", "words", e.text));
  if (e.thinking) div.appendChild(fold("How Muse decided", e.thinking, "thinking"));
  if (e.engine) div.appendChild(el("p", "engine", engineLine(e.engine, e.mind)));
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
  text.placeholder = busy ? "Type to steer Muse while it works" : idle;
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

function clearClip() {
  frameBox.replaceChildren(...[...emptyFrame].map((n) => n.cloneNode(true)));
  clipLine.textContent = "";
  clipTitle.textContent = "Your clip";
}

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
  clipTitle.textContent = "Your recording";
  clipLine.textContent = r.name + " · " + clock(r.seconds) + " long. This is the recording you opened; your clip will show here instead once Muse makes it.";
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
    j.foot.textContent = parts.join(" · ");
    j.foot.hidden = false;
    room.open.delete(e.job);
    setBusy();
  }
  scrollDown();
}

function onThinking(live) {
  const j = room.jobs[live.job];
  if (!j || j.live.hidden) return;
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
    statusLine.textContent = "Muse is working on it" + (m ? " · " + m.label : "") + " · type to steer it, or Stop";
  } else if (m) {
    statusLine.textContent = m.ok ? "Ready · " + m.name + " " + m.where : m.why;
  }
}

function drawMinds() {
  mindsBox.textContent = "";
  room.minds.forEach((m) => {
    const b = el("button", "mind");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(m.key === room.mind));
    const dot = el("span", "dot " + (m.ok ? "ok" : "off"));
    dot.setAttribute("aria-hidden", "true");
    b.append(dot, document.createTextNode(m.label));
    b.title = m.ok ? m.name + " " + m.where : m.why;
    b.addEventListener("click", () => {
      room.mind = m.key;
      localStore("mind", m.key);
      drawMinds();
    });
    mindsBox.appendChild(b);
  });
  showStatus();
}

async function health() {
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    const d = await r.json();
    room.minds = d.minds || [];
    if (!room.minds.some((m) => m.key === room.mind)) room.mind = d.default;
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
      b.append(el("span", "first", c.first), el("span", "meta", [c.recording, when(c.when), plural(c.messages, "message")].filter(Boolean).join(" · ")));
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
    if (!d.recordings.length) box.appendChild(el("p", "earlier-empty", "No recordings found in " + (d.folders.join(", ") || "any folder") + "."));
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

health();
connect();
