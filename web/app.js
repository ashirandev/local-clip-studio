"use strict";

const log = document.getElementById("log");
const statusLine = document.getElementById("status");
const composer = document.getElementById("composer");
const text = document.getElementById("text");

function say(kind, words) {
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.remove();
  const div = document.createElement("div");
  div.className = "msg " + kind;
  div.textContent = words;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function health() {
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    const d = await r.json();
    statusLine.textContent = d.ok ? "Ready" : "The studio is not ready.";
  } catch (e) {
    statusLine.textContent = "The studio is not answering. Start it again with: python app.py";
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const words = text.value.trim();
  if (!words) return;
  say("you", words);
  text.value = "";
  say("note", "Muse is not connected in this build yet.");
});

text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

health();
