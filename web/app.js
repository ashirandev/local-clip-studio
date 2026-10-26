"use strict";

const log = document.getElementById("log");
const statusLine = document.getElementById("status");
const composer = document.getElementById("composer");
const text = document.getElementById("text");
const send = document.getElementById("send");
const mindsBox = document.getElementById("minds");
const welcome = document.getElementById("welcome").cloneNode(true);

const room = { mind: localStore("mind") || "local", minds: [], chat: null, source: null, last: -1, jobs: {}, busy: false, timer: null };

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

function job(n) {
  if (!room.jobs[n]) {
    clearWelcome();
    const box = el("section", "job");
    box.setAttribute("aria-label", "Muse's work on message " + n);
    const plan = el("div", "plan");
    plan.hidden = true;
    const flow = el("div", "flow");
    const live = el("p", "live");
    live.hidden = true;
    const foot = el("p", "foot");
    foot.hidden = true;
    box.append(plan, flow, live, foot);
    log.appendChild(box);
    room.jobs[n] = { box, plan, flow, live, foot, mind: "" };
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
  if (e.plan_step) head.appendChild(el("span", "ps", "plan step " + e.plan_step));
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

function plural(n, one) {
  return n + " " + one + (n === 1 ? "" : "s");
}

function setBusy(busy) {
  room.busy = busy;
  send.disabled = busy;
  text.setAttribute("aria-busy", String(busy));
  showStatus();
}

function stopTimer() {
  if (room.timer) clearInterval(room.timer);
  room.timer = null;
}

function on(e) {
  if (e.id <= room.last) return;
  room.last = e.id;
  if (e.kind === "person") {
    bubble("you", e.text);
    return;
  }
  const j = job(e.job);
  if (e.kind === "job") {
    j.mind = e.mind;
    setBusy(true);
  } else if (e.kind === "plan") {
    drawPlan(j, e);
  } else if (e.kind === "wait") {
    stopTimer();
    const tick = () => {
      j.live.textContent = "Muse is thinking about step " + e.step + " · " + Math.max(0, Math.round(Date.now() / 1000 - e.t)) + " s";
    };
    tick();
    j.live.hidden = false;
    room.timer = setInterval(tick, 1000);
  } else if (e.kind === "step") {
    stopTimer();
    j.live.hidden = true;
    drawStep(j, e);
  } else if (e.kind === "answer") {
    stopTimer();
    j.live.hidden = true;
    drawAnswer(j, e);
  } else if (e.kind === "note") {
    bubble("note", e.text, j.flow);
  } else if (e.kind === "end") {
    stopTimer();
    j.live.hidden = true;
    const parts = [plural(e.steps, "step")];
    if (e.refused) parts.push(e.refused + " refused" + (e.fixed ? ", " + e.fixed + " fixed by Muse" : ""));
    parts.push(e.seconds + " s");
    j.foot.textContent = parts.join(" · ");
    j.foot.hidden = false;
    setBusy(false);
  }
  scrollDown();
}

function current() {
  return room.minds.find((m) => m.key === room.mind) || room.minds[0];
}

function showStatus() {
  const m = current();
  if (room.busy) {
    statusLine.textContent = "Muse is working on it" + (m ? " · " + m.label : "");
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

async function connect() {
  try {
    const r = await fetch("/api/chat", { cache: "no-store" });
    const d = await r.json();
    if (d.chat !== room.chat) {
      if (room.source) room.source.close();
      stopTimer();
      room.chat = d.chat;
      room.last = -1;
      room.jobs = {};
      log.replaceChildren(welcome.cloneNode(true));
      setBusy(false);
    }
    const source = new EventSource("/api/events?chat=" + encodeURIComponent(room.chat));
    source.onmessage = (m) => on(JSON.parse(m.data));
    source.onerror = () => {
      if (source.readyState === EventSource.CLOSED) setTimeout(connect, 2000);
    };
    room.source = source;
  } catch (e) {
    statusLine.textContent = "The studio is not answering. Start it again with: python app.py";
    setTimeout(connect, 3000);
  }
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const words = text.value.trim();
  if (!words || room.busy || !room.chat) return;
  text.value = "";
  send.disabled = true;
  try {
    const r = await fetch("/api/say", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat: room.chat, message: words, mind: room.mind }),
    });
    const d = await r.json();
    if (!d.ok) {
      bubble("note", d.error || "Something went wrong.");
      text.value = words;
      send.disabled = room.busy;
    }
  } catch (err) {
    bubble("note", "The studio is not answering. Start it again with: python app.py");
    text.value = words;
    send.disabled = room.busy;
  }
  text.focus();
});

text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

health();
connect();
