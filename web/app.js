"use strict";

const log = document.getElementById("log");
const statusLine = document.getElementById("status");
const composer = document.getElementById("composer");
const text = document.getElementById("text");
const send = document.getElementById("send");
const mindsBox = document.getElementById("minds");

const room = { mind: localStore("mind") || "local", minds: [], history: [], busy: false };

function localStore(key, value) {
  try {
    if (value === undefined) return localStorage.getItem("lcs." + key);
    localStorage.setItem("lcs." + key, value);
  } catch (e) {
    return null;
  }
}

function clearWelcome() {
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.remove();
}

function bubble(kind, words) {
  clearWelcome();
  const div = document.createElement("div");
  div.className = "msg " + kind;
  if (words) div.textContent = words;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function number(n) {
  return n === null || n === undefined ? null : Number(n).toLocaleString("en-US");
}

function engineLine(e, label) {
  const parts = [label];
  if (e.read_new !== null && e.read_new !== undefined) parts.push("read " + number(e.read_new) + " tokens new");
  if (e.reused) parts.push("reused " + number(e.reused));
  if (e.read_per_s) parts.push("reading " + number(e.read_per_s) + "/s");
  if (e.write_per_s) parts.push("writing " + e.write_per_s + "/s");
  parts.push(e.seconds + " s");
  return parts.join(" · ");
}

function answer(d) {
  const div = bubble("muse");
  const p = document.createElement("p");
  p.className = "words";
  p.textContent = d.answer;
  div.appendChild(p);
  if (d.thinking) {
    const fold = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = "How Muse decided";
    const body = document.createElement("p");
    body.className = "thinking";
    body.textContent = d.thinking;
    fold.append(sum, body);
    div.appendChild(fold);
  }
  const eng = document.createElement("p");
  eng.className = "engine";
  eng.textContent = engineLine(d.engine || {}, d.mind);
  div.appendChild(eng);
  log.scrollTop = log.scrollHeight;
}

function current() {
  return room.minds.find((m) => m.key === room.mind) || room.minds[0];
}

function showStatus() {
  const m = current();
  if (!m) return;
  statusLine.textContent = m.ok ? "Ready · " + m.name + " " + m.where : m.why;
}

function drawMinds() {
  mindsBox.textContent = "";
  room.minds.forEach((m) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "mind";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(m.key === room.mind));
    const dot = document.createElement("span");
    dot.className = "dot " + (m.ok ? "ok" : "off");
    dot.setAttribute("aria-hidden", "true");
    b.append(dot, document.createTextNode(m.label));
    b.title = m.ok ? m.name + " " + m.where : m.why;
    b.addEventListener("click", () => {
      room.mind = m.key;
      localStore("mind", m.key);
      drawMinds();
      showStatus();
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

function setBusy(busy) {
  room.busy = busy;
  send.disabled = busy;
  text.setAttribute("aria-busy", String(busy));
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const words = text.value.trim();
  if (!words || room.busy) return;
  bubble("you", words);
  text.value = "";
  setBusy(true);
  const wait = bubble("note", "Muse is thinking…");
  const started = Date.now();
  const timer = setInterval(() => {
    wait.textContent = "Muse is thinking… " + Math.round((Date.now() - started) / 1000) + " s";
  }, 1000);
  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: words, history: room.history, mind: room.mind }),
    });
    const d = await r.json();
    wait.remove();
    if (d.ok) {
      answer(d);
      room.history.push({ role: "user", content: words }, { role: "assistant", content: d.answer });
    } else {
      bubble("note", d.error || "Something went wrong.");
    }
  } catch (err) {
    wait.remove();
    bubble("note", "The studio is not answering. Start it again with: python app.py");
  } finally {
    clearInterval(timer);
    setBusy(false);
    text.focus();
  }
});

text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

health();
