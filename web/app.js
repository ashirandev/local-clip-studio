"use strict";

const $ = (id) => document.getElementById(id);
const app = $("app");
const log = $("log");
const chatView = $("chatView");
const statusLine = $("status");
const composer = $("composer");
const text = $("text");
const send = $("send");
const stopButton = $("stop");
const newButton = $("new");
const grid = $("grid");
const gridEmpty = $("gridEmpty");
const monthChip = $("month");
const mindsBox = $("minds");
const machineBox = $("machine");
const museChip = $("museChip");
const museWhere = $("museWhere");
const chatMuse = $("chatMuse");
const chatName = $("chatName");
const chatSub = $("chatSub");
const toHome = $("toHome");
const toClip = $("toClip");
const toChat = $("toChat");
const MONEY = { TH: "THB", JP: "JPY", KR: "KRW", CN: "CNY", TW: "TWD", HK: "HKD", SG: "SGD", MY: "MYR", ID: "IDR", VN: "VND", PH: "PHP", IN: "INR", GB: "GBP", AU: "AUD", NZ: "NZD", CA: "CAD", CH: "CHF", BR: "BRL", MX: "MXN" };
const EURO = ["AT", "BE", "CY", "DE", "EE", "ES", "FI", "FR", "GR", "HR", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PT", "SI", "SK"];
const darkScheme = window.matchMedia("(prefers-color-scheme: dark)");
const clipBox = $("clip");
const watchBox = $("watch");
const watchVideo = $("watchVideo");
const watchBack = $("watchBack");
const clipBack = $("clipBack");
const clipCost = $("clipCost");
const clipSave = $("clipSave");
const welcome = $("welcome").cloneNode(true);
const frameBox = $("frame");
const emptyFrame = frameBox.cloneNode(true).childNodes;
const bigPlay = $("bigPlay");
const fullButton = $("full");
const eyeButton = $("eye");
const playButton = $("play");
const timeLine = $("time");
const clipLine = $("clipLine");
const clipTitle = $("clipTitle");
const clipAt = $("clipAt");
const clipAtLine = $("clipAtLine");
const clipUndo = $("clipUndo");
const partTabs = $("partTabs");
const panelNote = $("panelNote");
const lanes = $("lanes");
const wordList = $("wordList");
const wordEdit = $("wordEdit");
const wordLabel = $("wordLabel");
const wordText = $("wordText");
const wordSave = $("wordSave");
const wordCancel = $("wordCancel");
const wordNote = $("wordNote");
const againButton = $("againButton");
const againLine = $("againLine");
const albumBox = $("album");
const previewLine = $("previewLine");
const fileInput = $("file");
const dropCover = $("dropCover");
const sheet = $("sheet");
const sheetBack = $("sheetBack");
const sheetHead = $("sheetHead");
const sheetIcon = $("sheetIcon");
const costBox = $("costBox");
const outRow = $("outRow");
const postBox = $("postBox");
const featuresList = $("featuresList");
const mindWhy = $("mindWhy");
const partsBox = $("partsBox");
const setUpButton = $("setUp");
const setupLine = $("setupLine");
const PART_ICONS = { ffmpeg: "film", engine: "gear", glimmer: "pc", hearing: "wave" };
const apiForm = $("apiForm");
const apiUrl = $("apiUrl");
const apiKey = $("apiKey");
const apiNote = $("apiNote");
const apiSeg = $("apiSeg");
const apiGet = $("apiGet");
const apiMore = $("apiMore");
let apis = [];
let apiNow = "";
const phoneBox = $("phoneBox");
const sendsBox = $("sendsBox");
const sendsLine = $("sendsLine");
const phoneLine = $("phoneLine");
const phoneCodes = $("phoneCodes");
const phoneFiles = $("phoneFiles");
const attached = $("attached");
const idle = text.placeholder;

const SHEETS = { album: ["film", "Album"], layout: ["stack", "Layout"], trim: ["cut", "Trim"], frame: ["move", "Frame"], caps: ["cc", "Captions"], style: ["aa", "Style"], look: ["drop", "Colour"], sound: ["sound", "Sound"], words: ["words", "Words"], channel: ["at", "Channel"], out: ["out", "Save"], cost: ["cash", "Cost"], options: ["sliders", "Options"], features: ["spark", "Features"] };
const EDITS = ["layout", "trim", "frame", "caps", "style", "look", "sound", "words", "channel", "album"];
const TOOL_ICONS = { add_captions: "cc", add_flurry: "spark", add_glass: "aa", add_hit: "sound", add_sound: "sound", add_sticker: "sticker", add_title: "aa", arrange_clips: "cut", blur_box: "blur", change_ends: "cut", change_shot: "people", check_clip: "shield", cut_out: "people", decide_people: "blur", find_recordings: "search", hand_over: "check", list_fonts: "aa", load_recipe: "tag", look_at_piece: "eye", make_clip: "cut", make_clips: "cut", map_recording: "wave", match_album: "film", move_part: "move", mute_sound: "mute", no_chat_box: "blur", open_clip: "film", open_clips: "folder", open_recording: "film", propose_moments: "star", remove_title: "aa", render_album: "gear", render_clip: "gear", save_channel_look: "at", set_look: "drop", set_music: "sound", set_plan: "list", set_style: "aa", set_tagline: "at", show_people: "people", slow_push: "move", undo: "undo", undo_album: "undo", use_channel_look: "at", watch_piece: "eye" };

const room = { mind: localStore("mind") || "local", minds: [], rate: { currency: "USD", per_usd: 1 }, costs: [], chat: null, source: null, last: -1, jobs: {}, open: new Set(), timer: null, cut: {}, showing: null, work: { video: false, preview: false, ends: false, layout: false }, screen: "", list: [], title: "", ready: null, spent: null, minded: [], paid: false, mute: {}, trim: null, sends: false };

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

function ico(name, cls) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("class", "ico" + (cls ? " " + cls : ""));
  s.setAttribute("aria-hidden", "true");
  const u = document.createElementNS("http://www.w3.org/2000/svg", "use");
  u.setAttribute("href", "#i-" + name);
  s.appendChild(u);
  return s;
}

function iconButton(cls, label, name) {
  const b = el("button", cls);
  b.type = "button";
  b.setAttribute("aria-label", label);
  b.appendChild(ico(name));
  return b;
}

function toast(words) {
  let t = document.querySelector(".toast");
  if (!t) {
    t = el("div", "toast");
    t.setAttribute("role", "status");
    document.body.appendChild(t);
  }
  t.textContent = words;
  t.hidden = false;
  clearTimeout(room.toastTimer);
  room.toastTimer = setTimeout(() => (t.hidden = true), 1800);
}

function chatScroller() {
  return getComputedStyle(chatView).overflowY === "auto" ? chatView : document.scrollingElement;
}

function reading() {
  return room.pinned !== null && room.pinned !== undefined && chatScroller().scrollTop < room.pinned - 48;
}

function scrollDown(always) {
  if (app.dataset.screen !== "chat" && app.dataset.screen !== "clip") return;
  if (getComputedStyle(chatView).overflowY !== "auto" && app.dataset.screen !== "chat") return;
  if (!always && reading()) return;
  const scroller = chatScroller();
  if (scroller === chatView) chatView.scrollTop = chatView.scrollHeight;
  else window.scrollTo(0, document.scrollingElement.scrollHeight);
  room.pinned = scroller.scrollTop;
}

function clearWelcome() {
  const w = $("welcome");
  if (w) w.remove();
}

function bubble(kind, words, into) {
  clearWelcome();
  const div = el("div", "msg " + kind, words);
  (into || log).appendChild(div);
  scrollDown();
  return div;
}

function note(words, into) {
  const div = bubble("note", null, into);
  div.append(ico("warn", "sm"), el("span", null, words));
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
  if (e.usd) parts.push(money(e.usd));
  return parts.join(" · ");
}

function myCurrency() {
  const chosen = localStore("money");
  if (chosen) return chosen;
  return localCurrency();
}

function localCurrency() {
  try {
    const region = new Intl.Locale(navigator.language || "en").maximize().region;
    return MONEY[region] || (EURO.includes(region) ? "EUR" : "USD");
  } catch (err) {
    return "USD";
  }
}

function money(usd) {
  const r = room.rate;
  const v = usd * r.per_usd;
  const shown = new Intl.NumberFormat(navigator.language || "en-US", { style: "currency", currency: r.currency, minimumFractionDigits: 2, maximumFractionDigits: v && Math.abs(v) < 1 ? 4 : 2 }).format(v);
  return r.currency === "USD" ? shown : shown + " ($" + usd.toFixed(usd && usd < 1 ? 4 : 2) + ")";
}

function moneyShort(usd) {
  const r = room.rate;
  if (usd === null || usd === undefined) return new Intl.NumberFormat(navigator.language || "en-US", { style: "currency", currency: r.currency, maximumFractionDigits: 0 }).format(0).replace(/0/, "–");
  const v = usd * r.per_usd;
  const digits = v === 0 ? 0 : Math.abs(v) < 0.1 ? 3 : 2;
  return new Intl.NumberFormat(navigator.language || "en-US", { style: "currency", currency: r.currency, minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v);
}

function costRow(what, usd, more) {
  const row = el("div", "cost-row");
  const name = el("span", "what", what);
  if (more) name.appendChild(el("small", null, more));
  row.append(name, el("span", "amount", money(usd)));
  return row;
}

function drawCost(d, box, job) {
  const mine = d.jobs.find((j) => j.job === job);
  const rows = [];
  if (mine) rows.push(costRow("This clip", mine.usd, mine.minds.join(", ") + " · " + plural(mine.answers, "answer")));
  if (room.chat) rows.push(costRow("This chat", d.chat));
  rows.push(costRow("Today", d.today), costRow("This month", d.month));
  if (d.jobs.length > 1) {
    rows.push(el("p", "pop-head", "Each message"));
    d.jobs.forEach((j) => rows.push(costRow("Message " + j.job, j.usd, j.minds.join(", ") + " · " + plural(j.answers, "answer"))));
  }
  rows.push(el("p", "pop-head", "Each Muse, in all"));
  room.minds.forEach((m) => rows.push(costRow(m.label, d.minds[m.label] || 0, m.key === "local" ? "On your own graphics card: free" : m.where.charAt(0).toUpperCase() + m.where.slice(1))));
  const notes = ["Each answer is priced by " + d.prices + ". A question cut short by Stop or by new words is not counted: the provider says what an answer cost only at its end."];
  if (d.unpriced) notes.push(plural(d.unpriced, "answer") + " had no known price and " + (d.unpriced === 1 ? "is" : "are") + " not counted.");
  if (d.rate.source) notes.push(d.rate.currency + " at " + d.rate.per_usd.toFixed(2) + " to the US dollar, the rate of " + d.rate.day + " from " + d.rate.source + ".");
  else if (d.rate.why) notes.push(d.rate.why);
  rows.push(el("p", "cost-note", notes.join(" ")));
  box.replaceChildren(...rows);
}

function mindIcons(minds) {
  return (minds || []).map((m) => ico(m.local ? "pc" : "cloud", "sm"));
}

function isLocal(label) {
  const m = room.minds.find((x) => x.label === label);
  return m ? m.local : /glimmer/i.test(label || "");
}

function drawCosts(d) {
  room.spent = d;
  room.rate = d.rate;
  const month = new Date().toLocaleString("en", { month: "short" });
  monthChip.replaceChildren(ico("cash", "sm"), moneyShort(d.month), el("small", null, month));
  monthChip.setAttribute("aria-label", "Spent this month: " + moneyShort(d.month) + ". Show what Muse cost");
  monthChip.hidden = false;
  if (room.chat) {
    const usd = room.open.size || d.jobs.length || !room.paid ? d.chat : null;
    chatSub.replaceChildren(...mindIcons(room.minded), moneyShort(usd));
    clipCost.replaceChildren(...mindIcons(room.minded), moneyShort(usd));
    clipCost.setAttribute("aria-label", "What this chat cost: " + moneyShort(usd) + ". Show the costs");
    clipCost.hidden = false;
  }
  Object.values(room.jobs).forEach((j) => {
    const mine = d.jobs.find((x) => x.job === j.n);
    if (!j.mind) return;
    j.cost.replaceChildren(ico(isLocal(j.mind) ? "pc" : "cloud", "sm"), moneyShort(mine ? mine.usd : isLocal(j.mind) ? 0 : null));
    j.cost.hidden = false;
  });
  room.costs.forEach((c) => {
    const mine = d.jobs.find((x) => x.job === c.job);
    c.tag.replaceChildren(ico("cash", "sm"), moneyShort(mine ? mine.usd : 0));
  });
  if (!sheet.hidden && room.part === "cost") drawCost(d, costBox, room.ready ? room.ready.job : undefined);
}

async function refreshCost() {
  try {
    const r = await fetch("/api/spent?currency=" + myCurrency() + (room.chat ? "&chat=" + encodeURIComponent(room.chat) : ""), { cache: "no-store" });
    drawCosts(await r.json());
  } catch (err) {
    if (room.part === "cost") costBox.replaceChildren(el("p", "earlier-empty", "The studio is not answering."));
  }
}

function fold(summary, body, cls, always) {
  const d = el("details", always ? "thought" : "detail");
  const s = el("summary");
  s.append(ico("think", "sm"), summary);
  d.append(s, el("p", cls, body));
  return d;
}

function plainValue(v) {
  if (v === null || v === undefined) return "none";
  if (Array.isArray(v)) return v.map(plainValue).join(" / ");
  if (typeof v === "object" && v.name) return String(v.name);
  if (typeof v === "object" && v.clip !== undefined) return "clip " + v.clip;
  if (typeof v === "object") return Object.entries(v).map(([k, w]) => k + " " + plainValue(w)).join(", ");
  return String(v);
}

function inputLine(input) {
  if (!input) return "";
  if (typeof input === "string") return input;
  return Object.entries(input)
    .map(([k, v]) => k + ": " + plainValue(v))
    .join(" · ");
}

function clock(seconds) {
  const s = Math.round(seconds);
  const mm = String(Math.floor((s % 3600) / 60)).padStart(s >= 3600 ? 2 : 1, "0");
  return (s >= 3600 ? Math.floor(s / 3600) + ":" : "") + mm + ":" + String(s % 60).padStart(2, "0");
}

function plural(n, one) {
  return n + " " + one + (n === 1 ? "" : "s");
}

function setScreen(name, push) {
  if (room.screen === name) return;
  const from = room.screen;
  if (from === "chat") room.left = { at: chatScroller().scrollTop, reading: reading() };
  room.screen = name;
  app.dataset.screen = name;
  closeSheet();
  if (push !== false) history.pushState({ screen: name }, "");
  if (name === "home") {
    loadHome();
    window.scrollTo(0, 0);
  } else if (name === "chat") {
    if (from !== "clip") scrollDown(true);
    else if (chatScroller() === chatView) scrollDown(false);
    else if (room.left && room.left.reading) window.scrollTo(0, room.left.at);
    else scrollDown(true);
  } else {
    window.scrollTo(0, 0);
  }
  if (name !== "clip") {
    const v = playerVideo();
    if (v && !v.paused && getComputedStyle(clipBox).display === "none") v.pause();
    if (getComputedStyle(clipBox).display === "none") watchVideo.pause();
  }
}

window.addEventListener("popstate", (e) => setScreen((e.state && e.state.screen) || "home", false));

toHome.addEventListener("click", () => setScreen("home"));
toClip.addEventListener("click", () => setScreen("clip"));
toChat.addEventListener("click", () => setScreen("chat"));
clipBack.addEventListener("click", () => setScreen("chat"));

function job(n) {
  if (!room.jobs[n]) {
    clearWelcome();
    const box = el("section", "job");
    box.setAttribute("aria-label", "Muse's work on message " + n);
    const card = el("article", "w-card");
    const head = el("div", "w-head");
    const open = el("button", "w-open");
    open.type = "button";
    open.setAttribute("aria-expanded", "false");
    const av = el("span", "w-av");
    av.appendChild(ico("muse", "sm"));
    const st = el("span", "w-st");
    const state = el("b", null, "Working");
    const meta = el("small", null, "");
    st.append(state, meta);
    const cost = el("span", "w-cost");
    cost.hidden = true;
    open.append(av, st, cost, ico("down", "w-chev"));
    const replay = iconButton("round w-replay", "Play how Muse worked again", "replay");
    replay.hidden = true;
    head.append(open, replay);
    const plan = el("div", "w-plan");
    plan.setAttribute("aria-hidden", "true");
    plan.hidden = true;
    const now = el("p", "w-now");
    now.hidden = true;
    const tick = el("p", "w-tick");
    const think = el("p", "w-think");
    think.setAttribute("aria-label", "What Muse is thinking now");
    const moments = el("div", "moments");
    moments.hidden = true;
    const strip = el("div", "w-strip");
    const rows = el("ol", "w-steps");
    card.classList.add("open");
    open.setAttribute("aria-expanded", "true");
    const chips = el("div", "w-chips");
    chips.hidden = true;
    card.append(head, plan, now, tick, think, moments, strip, rows, chips);
    const flow = el("div", "job");
    box.append(card, flow);
    log.appendChild(box);
    const j = { n, box, card, open, state, meta, cost, replay, plan, now, tick, think, momentsBox: moments, strip, rows, chips, flow, mind: "", steps: {}, answers: [], moments: {}, count: 0, refused: 0, ended: false, items: [] };
    open.addEventListener("click", () => setOpen(j, !card.classList.contains("open")));
    replay.addEventListener("click", () => replayJob(j));
    room.jobs[n] = j;
  }
  return room.jobs[n];
}

function setOpen(j, on) {
  j.card.classList.toggle("open", on);
  j.rows.hidden = !on;
  j.open.setAttribute("aria-expanded", String(on));
}

function drawPlan(j, e) {
  j.items = e.items;
  j.plan.replaceChildren(...e.items.map((item) => el("i", item.state)));
  j.plan.hidden = !e.items.length;
  const now = e.items.find((item) => item.state === "now") || e.items.find((item) => item.state === "todo");
  j.now.textContent = now ? (e.changed ? "New plan · " : "") + now.text : "";
  j.now.hidden = !j.now.textContent;
  j.now.title = e.why || "";
  const said = { done: "done", now: "working on it", todo: "not started" };
  j.plan.setAttribute("aria-label", "Muse's plan: " + e.items.map((item) => item.text + " (" + said[item.state] + ")").join(", "));
}

function drawStep(j, e) {
  j.count += 1;
  if (e.refused) j.refused += 1;
  const label = e.look ? "Looked at " + clock(e.look.start) + "–" + clock(e.look.end) : e.label || e.tool;
  const icon = TOOL_ICONS[e.tool] || "dash";
  const li = el("li", "w-row");
  const mark = el("span", "w-ic" + (e.refused ? " bad" : ""));
  mark.appendChild(ico(icon, "sm"));
  li.append(mark, el("b", null, label), e.refused ? el("span", "no", "Refused") : el("span", "sec", e.seconds >= 1 ? Math.round(e.seconds) + " s" : ""));
  const said = e.said ? el("p", "said", e.said) : null;
  if (said) li.appendChild(said);
  const why = el("p", "why", e.why || "Muse gave no reason for this step.");
  li.appendChild(why);
  j.steps[e.step] = { why: why, said: said };
  if (e.look) li.appendChild(lookStrip(e.look));
  if (e.look && e.look.said && e.look.said.length) li.appendChild(saidFold(e.look.said));
  if (e.look && e.look.took) li.appendChild(el("p", "engine detail", "Heard in " + e.look.took + " s"));
  if (e.moments && e.fixing) li.appendChild(el("p", "engine detail", "Misheard words fixed by Muse in " + e.fixing + " s"));
  if (e.refused) li.appendChild(el("p", "result simple", "That did not work, so Muse tried again."));
  const input = inputLine(e.input);
  if (input) li.appendChild(el("p", "input detail", input));
  const lines = String(e.result || "").split("\n");
  const first = e.refused ? (e.guard ? "Refused, already answered: " : "Refused: ") + lines[0] : lines[0];
  li.appendChild(el("p", "result detail", first));
  if (lines.length > 1) li.appendChild(fold("Full result", lines.slice(1).join("\n"), "more"));
  if (e.thinking) li.appendChild(fold("How Muse decided", e.thinking, "thinking", true));
  if (e.engine) li.appendChild(el("p", "engine detail", engineLine(e.engine, j.mind)));
  j.rows.appendChild(li);
  const at = j.rows.children.length - 1;
  const b = el("button", "w-ic" + (e.refused ? " bad" : "") + (j.ended ? "" : " new"));
  b.type = "button";
  b.setAttribute("aria-label", "Step " + e.step + ": " + label + (e.refused ? ", refused" : ""));
  b.appendChild(ico(icon, "sm"));
  b.addEventListener("click", () => {
    setOpen(j, true);
    const row = j.rows.children[at];
    if (!row) return;
    row.classList.remove("flash");
    void row.offsetWidth;
    row.classList.add("flash");
    row.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
  j.strip.appendChild(b);
  j.strip.scrollLeft = j.strip.scrollWidth;
  if (!j.ended) j.meta.textContent = label + " · step " + e.step;
  j.tick.textContent = "";
  j.think.textContent = "";
  if (e.moments) j.moments[e.step] = drawMoments(j, e.moments);
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
  document.querySelectorAll(".moment .chip").forEach((b) => (b.disabled = true));
  const box = j.momentsBox;
  box.replaceChildren();
  box.hidden = false;
  const reasons = [];
  moments.forEach((m) => {
    const tile = el("div", "moment");
    tile.dataset.preview = m.preview;
    const media = el("div", "moment-media");
    const img = el("img");
    img.src = "/api/frame/" + m.frame;
    img.alt = "A picture from moment " + m.n + ", at " + clock(m.t);
    media.append(img, el("span", "moment-wait", "Cutting a preview…"));
    const why = el("p", "moment-why", m.reason);
    reasons.push(why);
    tile.append(media, el("p", "moment-head", m.n + " · " + span(m) + " · " + Math.round(m.end - m.start) + " s"), why);
    if (m.said && m.said.length) {
      const said = saidFold(m.said);
      said.classList.add("detail");
      tile.appendChild(said);
    }
    if (m.n === 1) {
      tile.appendChild(el("p", "moment-pick", "Muse's pick"));
    } else {
      const b = el("button", "chip", "Use this one");
      b.type = "button";
      b.addEventListener("click", () => {
        showMoment(m);
        say("(tapped) Use moment " + m.n + " (" + clock(m.start) + "-" + clock(m.end) + ") instead.");
      });
      tile.appendChild(b);
    }
    box.appendChild(tile);
    if (room.cut[m.preview]) fillPreview(tile, room.cut[m.preview]);
  });
  showMoment(moments[0]);
  return reasons;
}

function fillPreview(tile, e) {
  const media = tile.querySelector(".moment-media");
  if (!e.ok) {
    const wait = media.querySelector(".moment-wait");
    if (wait) wait.textContent = "No preview: " + e.error;
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
  room.wantBase = false;
  room.base = null;
  previewLine.hidden = true;
  const ready = room.cut[m.preview];
  if (ready && ready.ok) {
    const v = el("video");
    v.src = "/api/preview/" + m.preview;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.setAttribute("aria-label", "A rough preview of moment " + m.n);
    frameBox.replaceChildren(v);
    wirePlayer(v);
  } else {
    const img = el("img");
    img.src = "/api/frame/" + m.frame;
    img.alt = "A picture from moment " + m.n;
    frameBox.replaceChildren(img);
  }
  thumb("/api/frame/" + m.frame);
  clipTitle.textContent = "Moment " + m.n + " · " + span(m);
  clipLine.textContent = ready && ready.ok ? "" : "Cutting a rough preview…";
  updatePlayer();
}

function drawAnswer(j, e) {
  const div = bubble("muse", null, j.flow);
  const words = el("p", "words", e.text);
  div.appendChild(words);
  j.answers.push(words);
  if (e.thinking) div.appendChild(fold("How Muse decided", e.thinking, "thinking", true));
  if (e.engine) div.appendChild(el("p", "engine detail", engineLine(e.engine, e.mind)));
}

function drawGuesses(j, e) {
  document.querySelectorAll(".w-chips .chip, .w-chips .look-card").forEach((b) => (b.disabled = true));
  const c = e.clip;
  const facts = [c.title, "about " + c.length_s + " s", c.style + " style"];
  if (c.language) facts.push(c.language);
  const row = el("div", "chips");
  const looks = el("div", "look-cards");
  e.chips.forEach((chip) => {
    const b = el("button", chip.look ? "look-card" : "chip", chip.look ? null : chip.label);
    b.type = "button";
    if (chip.look) {
      const pic = el("img", "look-pic");
      pic.src = "/api/look/" + chip.look + ".png";
      pic.alt = "";
      b.append(pic, el("span", null, chip.now ? chip.label + " · now" : chip.label));
      if (chip.now) b.setAttribute("aria-current", "true");
    }
    if (!chip.now) b.addEventListener("click", () => say(chip.say));
    (chip.look ? looks : row).appendChild(b);
  });
  j.chips.replaceChildren(el("p", "facts", facts.join(" · ") + (c.brief ? " · “" + c.brief + "”" : "")), row);
  if (looks.children.length) j.chips.appendChild(looks);
  j.chips.hidden = false;
}

function stopTimer() {
  if (room.timer) clearInterval(room.timer);
  room.timer = null;
}

function setBusy() {
  const busy = room.open.size > 0;
  stopButton.hidden = !(busy || room.work.video || room.work.preview || room.work.ends || room.work.layout);
  stopButton.disabled = false;
  text.placeholder = room.attached ? "What to do? Or just send" : busy ? "Change what Muse is doing" : idle;
  showStatus();
}

function pageWork(what, on) {
  room.work[what] = on;
  setBusy();
}

const TAPPED = "(tapped) ";
const OPENED = "(opened) ";

function personBubble(words, into) {
  const div = bubble("you", null, into);
  if (words.startsWith(TAPPED)) {
    const tap = el("span", "tap");
    tap.appendChild(ico("tap", "sm"));
    div.append(tap, words.slice(TAPPED.length));
  } else if (words.startsWith(OPENED)) {
    const lines = words.slice(OPENED.length).split("\n");
    const parts = lines[0].split(" · ");
    const tap = el("span", "tap");
    tap.appendChild(ico("film", "sm"));
    div.append(tap, [parts[0] + " · " + parts[1]].concat(lines.slice(1)).join("\n"));
  } else {
    div.textContent = words;
  }
}

function thumb(src) {
  if (!src) {
    toClip.hidden = true;
    return;
  }
  const img = el("img");
  img.src = src;
  img.alt = "";
  toClip.replaceChildren(img);
  toClip.hidden = false;
}

function clearClip() {
  stopWatching();
  room.pinned = null;
  frameBox.replaceChildren(...[...emptyFrame].map((n) => n.cloneNode(true)));
  clipLine.textContent = "";
  clipTitle.textContent = "Your clip";
  room.made = null;
  room.panel = null;
  room.wantBase = false;
  room.base = null;
  room.steps = [];
  room.againError = false;
  room.ready = null;
  room.clipName = "";
  room.finished = null;
  room.mute = {};
  room.pick = null;
  room.pickWhy = "";
  room.frameX = null;
  againLine.textContent = "";
  previewLine.hidden = true;
  clipSave.hidden = true;
  clipCost.hidden = true;
  drawAlbum(null);
  hidePanel();
  thumb(null);
  updatePlayer();
}

function showClip(e) {
  const clip = e.clip;
  if (!clip) {
    room.made = null;
    clipLine.textContent = "The clip was taken back. Ask Muse to make it again.";
    updatePlayer();
    return;
  }
  room.made = { clip, version: e.id };
  room.showing = null;
  room.wantBase = true;
  clipAt.max = clip.length;
  const entry = e.kind === "edit" ? e.entry : null;
  if (entry && entry.do === "title") clipAt.value = (entry.start + entry.end) / 2;
  else if (entry && entry.do === "clip") clipAt.value = Math.min(clip.length / 2, clip.length - 0.1);
  if (entry && entry.do === "clip") {
    room.mute = {};
    room.trim = null;
    room.frameX = null;
  }
  drawClipFrame();
  clipTitle.textContent = room.ready ? room.ready.title : room.clipName || "Your clip";
  clipLine.textContent = clip.people ? "Tap a face to blur it, and tap it again to show it." : "";
  thumb("/api/clipframe?chat=" + encodeURIComponent(room.chat) + "&t=" + (clip.length / 2).toFixed(1) + "&width=120&v=" + e.id);
  updatePlayer();
}

function showFinished(file) {
  room.made = null;
  room.showing = null;
  leaveBase();
  const v = el("video");
  v.src = "/api/clip/" + file.name;
  v.poster = "/api/clip/" + file.name.replace(/\.mp4$/, ".jpg");
  v.playsInline = true;
  v.preload = "metadata";
  v.setAttribute("aria-label", "Your finished clip");
  frameBox.replaceChildren(v);
  wirePlayer(v);
  clipLine.textContent = "";
  thumb(v.poster);
  updatePlayer();
}

function playerVideo() {
  return frameBox.querySelector("video");
}

function wirePlayer(v) {
  ["play", "pause", "ended", "loadedmetadata", "timeupdate"].forEach((k) => v.addEventListener(k, () => {
    if (k === "timeupdate" && !v.classList.contains("base")) movePlayhead(v.currentTime);
    updatePlayer();
  }));
  v.addEventListener("click", togglePlay);
}

function updatePlayer() {
  const v = playerVideo();
  const playing = Boolean(v && !v.paused && !v.ended);
  playButton.disabled = !v;
  playButton.replaceChildren(ico(playing ? "pause" : "play"));
  playButton.setAttribute("aria-label", playing ? "Pause" : "Play");
  bigPlay.hidden = !v || playing;
  fullButton.hidden = !v;
  clipUndo.disabled = !room.panel;
  if (v && v.duration) timeLine.replaceChildren(el("b", null, clock(v.currentTime)), " / " + clock(v.duration));
  else if (room.made) timeLine.replaceChildren(el("b", null, clock(Number(clipAt.value) || 0)), " / " + clock(room.made.clip.length));
  else timeLine.textContent = "";
}

function togglePlay() {
  const v = playerVideo();
  if (!v) return;
  if (v.paused || v.ended) v.play().catch(() => {});
  else v.pause();
}

playButton.addEventListener("click", togglePlay);
bigPlay.addEventListener("click", togglePlay);

fullButton.addEventListener("click", () => {
  const v = playerVideo();
  if (!v) return;
  try {
    if (v.requestFullscreen) v.requestFullscreen().catch(() => v.webkitEnterFullscreen && v.webkitEnterFullscreen());
    else if (v.webkitEnterFullscreen) v.webkitEnterFullscreen();
  } catch (err) {
    toast("This browser cannot show it full screen.");
  }
});

const CHECKED = { pass: "Passed", fail: "Failed", skip: "Not needed", muse: "For Muse", fixed: "Fixed" };

function drawChecked(j, e) {
  const due = e.checks.filter((c) => c.ok !== null);
  const passed = due.filter((c) => c.ok).length;
  const fixed = e.checks.filter((c) => c.fixed).length;
  j.checks = { passed, due: due.length };
  const box = el("details", "checks " + (passed === due.length ? "all" : "fail"));
  box.open = passed !== due.length;
  const head = el("summary");
  head.append(ico("shield", "sm"), "Checked " + passed + " of " + due.length + (fixed ? " · " + fixed + " fixed" : ""));
  const list = el("ul", "checks-list");
  e.checks.forEach((c) => {
    const how = c.fixed ? "fixed" : c.ok === null ? (c.who === "Muse" ? "muse" : "skip") : c.ok ? "pass" : "fail";
    const li = el("li", how);
    const words = el("span", "what");
    words.append(el("b", null, c.name), el("small", null, CHECKED[how] + " · " + c.found + (c.who === "program" ? "" : " · " + c.who)));
    li.append(ico(c.ok === null ? "dash" : c.ok ? "check" : "x", "sm"), words);
    list.appendChild(li);
  });
  box.append(head, list);
  j.flow.appendChild(box);
}

function drawAlbum(a) {
  albumBox.hidden = !a;
  albumBox.replaceChildren();
  if (!a) return;
  a.clips.forEach((c, k) => {
    const b = el("button", "album-clip" + (k === a.at ? " on" : ""));
    b.type = "button";
    const words = c.state === "changed" ? "Changed" : c.state === "unfinished" ? "Not ready" : "";
    b.setAttribute("aria-pressed", String(k === a.at));
    b.setAttribute("aria-label", "Clip " + (k + 1) + ": " + c.title + (words ? ". " + words : ""));
    b.title = c.title;
    if (c.file) {
      const img = el("img");
      img.src = "/api/clip/" + c.file.replace(/\.mp4$/, ".jpg");
      img.alt = "";
      b.appendChild(img);
    }
    b.appendChild(el("span", "album-n", String(k + 1)));
    if (words) b.appendChild(el("span", "album-state", words));
    b.addEventListener("click", () => pickClip(k));
    albumBox.appendChild(b);
    if (k === a.at) b.scrollIntoView({ block: "nearest", inline: "nearest" });
  });
}

async function pickClip(k) {
  stopWatching();
  try {
    const d = await post("/api/album", { chat: room.chat, n: k });
    if (!d.ok) toast(d.error || "That clip could not be opened.");
  } catch (err) {
    toast("The studio is not answering. Start it again with: python app.py");
  }
}

function showPicked(e) {
  if (!e.clip) return;
  stopWatching();
  room.ready = e.ready || null;
  room.clipName = e.title || "";
  room.finished = e.file ? e.file.name : null;
  room.mute = {};
  room.trim = null;
  room.frameX = null;
  room.againError = false;
  showClip(e);
  if (e.file) {
    room.wantBase = false;
    showFinished(e.file);
  }
  clipSave.hidden = !room.ready;
  againLine.textContent = "";
  loadPanel();
}

function openClip(e) {
  const a = room.panel && room.panel.album;
  const k = a ? a.clips.findIndex((c) => c.moments.includes(e.moment)) : -1;
  if (k >= 0) {
    pickClip(k);
    setScreen("clip");
    return;
  }
  if (e.file !== room.finished || room.open.size) watch(e);
  else stopWatching();
  setScreen("clip");
}

function watch(e) {
  watchVideo.src = "/api/clip/" + e.file;
  watchVideo.poster = "/api/clip/" + e.file.replace(/\.mp4$/, ".jpg");
  watchVideo.setAttribute("aria-label", "Your clip: " + e.title);
  watchBox.hidden = false;
  clipBox.classList.add("watching");
}

function stopWatching() {
  if (watchBox.hidden) return;
  watchVideo.pause();
  watchVideo.removeAttribute("src");
  watchVideo.load();
  watchBox.hidden = true;
  clipBox.classList.remove("watching");
}

watchBack.addEventListener("click", stopWatching);

function drawReady(j, e) {
  room.ready = e;
  const div = el("div", "ready");
  const img = el("img");
  img.src = "/api/clip/" + e.file.replace(/\.mp4$/, ".jpg");
  img.alt = "";
  img.addEventListener("click", () => openClip(e));
  const pics = el("div", "ready-pics");
  pics.appendChild(img);
  if (e.cover) {
    const link = el("a");
    link.href = "/api/cover/" + e.cover;
    link.target = "_blank";
    link.rel = "noopener";
    link.title = "The cover, drawn by Muse Image";
    const cover = el("img");
    cover.src = link.href;
    cover.alt = "Cover: " + e.title;
    link.appendChild(cover);
    pics.appendChild(link);
  }
  const body = el("div", "ready-body");
  const title = el("p", "ready-title", e.title);
  title.addEventListener("click", () => openClip(e));
  const meta = el("div", "ready-meta");
  if (e.length) {
    const len = el("span", "tag");
    len.append(ico("film", "sm"), clock(e.length));
    meta.appendChild(len);
  }
  if (j.checks && j.checks.due) {
    const ok = j.checks.passed === j.checks.due;
    const c = el("span", "tag " + (ok ? "ok" : "bad"));
    c.append(ico("shield", "sm"), j.checks.passed + "/" + j.checks.due);
    c.setAttribute("aria-label", "Checks: " + j.checks.passed + " of " + j.checks.due + " passed");
    meta.appendChild(c);
  }
  const cost = el("span", "tag");
  cost.append(ico("cash", "sm"), "…");
  meta.appendChild(cost);
  room.costs.push({ job: e.job, tag: cost });
  body.append(title, meta);
  if (e.work) body.appendChild(el("p", "ready-work", e.work.line));
  const go = el("div", "ready-go");
  const open = iconButton("round go", "Open the clip: " + e.title, "play");
  open.addEventListener("click", () => openClip(e));
  const save = iconButton("round", "Save the clip or the post", "save");
  save.addEventListener("click", () => {
    room.ready = e;
    openSheet("out");
  });
  const remake = iconButton("round", e.cover ? "Make the cover again" : "Make the cover", e.cover ? "again" : "spark");
  remake.title = e.cover ? "Make the cover again" : "Make the cover with Muse Image";
  remake.addEventListener("click", () => say(TAPPED + (e.cover ? "Make the cover again." : "Make the cover.")));
  go.append(open, save, remake);
  div.append(pics, body, go);
  j.flow.appendChild(div);
  clipTitle.textContent = e.title;
  clipSave.hidden = false;
  refreshCost();
}

function drawOut() {
  const e = room.ready;
  outRow.replaceChildren();
  postBox.replaceChildren();
  if (!e) return;
  const save = el("a");
  save.href = "/api/clip/" + (room.finished || e.file) + "?download=1";
  save.append(ico("save"), "Video");
  const edit = el("a");
  edit.href = "/api/premiere?chat=" + encodeURIComponent(room.chat);
  edit.title = "The clip as it is now, for Premiere Pro: each piece points at your own file, so an editor can drag its edges";
  edit.append(ico("lanes"), "Premiere");
  const all = [e.title, e.caption, e.hashtags.join(" ")].concat(e.english ? [e.english] : []).join("\n\n");
  const copy = el("button");
  copy.type = "button";
  copy.append(ico("copy"), "Copy post");
  copy.addEventListener("click", () => copyText(all));
  outRow.append(save, edit, copy);
  postBox.append(el("p", "t", e.title), el("p", null, e.caption), el("p", "h", e.hashtags.join(" ")));
  if (e.english) postBox.appendChild(el("p", "e", e.english));
}

async function copyText(words) {
  try {
    await navigator.clipboard.writeText(words);
    toast("Copied");
    return;
  } catch (err) {
    const t = el("textarea");
    t.value = words;
    t.style.position = "fixed";
    t.style.opacity = "0";
    document.body.appendChild(t);
    t.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (e) {
      ok = false;
    }
    t.remove();
    toast(ok ? "Copied" : "Select the words to copy them");
  }
}

function drawClipFrame() {
  const made = room.made;
  if (!made) return;
  const t = Math.min(Number(clipAt.value), Math.max(0, made.clip.length - 0.05));
  clipAtLine.textContent = t.toFixed(1) + " s";
  updatePlayer();
  const v = frameBox.querySelector("video.base");
  if (v) {
    if (Math.abs(v.currentTime - t) > 0.04) v.currentTime = t;
    return;
  }
  const img = el("img");
  img.src = "/api/clipframe?chat=" + encodeURIComponent(room.chat) + "&t=" + t.toFixed(2) + "&v=" + made.version;
  img.alt = "Your clip at " + t.toFixed(1) + " seconds";
  frameBox.replaceChildren(img);
  room.facesAt = made.version + "@" + t.toFixed(2);
  if (made.clip.people) drawFaces(t, made, room.facesAt);
}

async function drawFaces(t, made, asked) {
  let d;
  try {
    const r = await fetch("/api/faces?chat=" + encodeURIComponent(room.chat) + "&t=" + t.toFixed(2), { cache: "no-store" });
    d = await r.json();
  } catch (err) {
    return;
  }
  if (!d.ok || !d.faces.length || room.made !== made || room.facesAt !== asked) return;
  const layer = el("div", "faces");
  d.faces.forEach((f) => {
    const b = el("button", "face" + (f.blur ? " blurred" : ""));
    b.type = "button";
    b.style.left = (100 * f.box[0] / d.size[0]) + "%";
    b.style.top = (100 * f.box[1] / d.size[1]) + "%";
    b.style.width = (100 * f.box[2] / d.size[0]) + "%";
    b.style.height = (100 * f.box[3] / d.size[1]) + "%";
    b.setAttribute("aria-label", (f.blur ? "Show the face of person " : "Blur person ") + f.n);
    b.appendChild(el("span", "face-n", String(f.n)));
    b.addEventListener("click", async () => {
      b.disabled = true;
      try {
        const res = await post("/api/person", { chat: room.chat, n: f.n, blur: !f.blur });
        if (!res.ok) clipLine.textContent = res.error || "That face could not be changed.";
      } catch (err) {
        clipLine.textContent = "The studio is not answering. Start it again with: python app.py";
      }
      b.disabled = false;
    });
    layer.appendChild(b);
  });
  frameBox.appendChild(layer);
}

clipAt.addEventListener("input", () => {
  movePlayhead(Number(clipAt.value));
  clearTimeout(room.frameTimer);
  room.frameTimer = setTimeout(drawClipFrame, 120);
});

function hidePanel() {
  lanes.replaceChildren();
  wordList.replaceChildren();
  pickBar.hidden = true;
  againButton.hidden = true;
  partTabs.querySelectorAll(".tool[data-show]").forEach((b) => (b.disabled = true));
  closeWords();
}

function loadPanel() {
  clearTimeout(room.panelTimer);
  room.panelTimer = setTimeout(async () => {
    const chat = room.chat;
    let d;
    try {
      const r = await fetch("/api/panel?chat=" + encodeURIComponent(chat), { cache: "no-store" });
      d = await r.json();
    } catch (err) {
      return;
    }
    if (chat !== room.chat) return;
    if (!d.ok || !d.clip) {
      room.panel = null;
      drawAlbum(null);
      drawEye(null);
      hidePanel();
      updatePlayer();
      return;
    }
    room.panel = d;
    drawAlbum(d.album);
    drawPanel(d);
    drawWords(d);
    drawAgain(d);
    updatePlayer();
    if (room.wantBase && room.made) refreshPreview();
  }, 150);
}

async function refreshPreview() {
  const chat = room.chat;
  const ask = (room.previewAsk || 0) + 1;
  room.previewAsk = ask;
  let words, base;
  try {
    [words, base] = await Promise.all([
      fetch("/api/layer?chat=" + encodeURIComponent(chat), { cache: "no-store" }).then((r) => r.json()),
      post("/api/base", { chat: chat }),
    ]);
  } catch (err) {
    return;
  }
  if (chat !== room.chat || ask !== room.previewAsk || !room.wantBase || !room.made) return;
  room.steps = words.ok ? words.steps : [];
  room.steps.forEach((step) => {
    const pic = new Image();
    pic.src = step.url;
  });
  if (base.ok && base.url) {
    pageWork("preview", false);
    previewLine.hidden = true;
    showBase(base.url);
  } else {
    leaveBase();
    drawClipFrame();
    pageWork("preview", Boolean(base.ok && base.making));
    previewLine.textContent = base.ok ? "Making the preview…" : "No quick preview: " + (base.error || "it could not be made.");
    previewLine.hidden = false;
  }
}

function showBase(url) {
  let v = frameBox.querySelector("video.base");
  if (!v || room.base !== url) {
    const at = Number(clipAt.value);
    v = el("video", "base");
    v.src = url;
    v.playsInline = true;
    v.preload = "auto";
    v.setAttribute("aria-label", "Your clip");
    const pic = el("img", "layer");
    pic.alt = "";
    pic.hidden = true;
    v.addEventListener("loadedmetadata", () => {
      v.currentTime = Math.min(at, Math.max(0, v.duration - 0.05));
    });
    v.addEventListener("timeupdate", () => {
      clipAt.value = v.currentTime;
      clipAtLine.textContent = v.currentTime.toFixed(1) + " s";
      movePlayhead(v.currentTime);
      drawLayer();
    });
    v.addEventListener("seeked", () => {
      drawLayer();
      baseFaces(v);
    });
    v.addEventListener("pause", () => baseFaces(v));
    v.addEventListener("play", () => {
      frameBox.querySelectorAll(".faces").forEach((f) => f.remove());
      follow(v);
    });
    frameBox.replaceChildren(v, pic);
    wirePlayer(v);
    room.base = url;
  }
  drawLayer();
  updatePlayer();
}

function leaveBase() {
  room.base = null;
  frameBox.querySelectorAll("video.base, img.layer").forEach((n) => n.remove());
}

function drawLayer() {
  const v = frameBox.querySelector("video.base");
  const pic = frameBox.querySelector("img.layer");
  if (!v || !pic) return;
  const t = v.currentTime;
  const step = (room.steps || []).find((s) => s.from <= t && t < s.to);
  pic.hidden = !step;
  if (step && pic.dataset.url !== step.url) {
    pic.src = step.url;
    pic.dataset.url = step.url;
  }
}

function follow(v) {
  const tick = () => {
    if (!frameBox.contains(v)) return;
    drawLayer();
    if (!v.paused && !v.ended) next();
  };
  const next = () => (v.requestVideoFrameCallback ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(tick));
  next();
}

function baseFaces(v) {
  const made = room.made;
  frameBox.querySelectorAll(".faces").forEach((f) => f.remove());
  if (!made || !made.clip.people || !v.paused) return;
  room.facesAt = made.version + "@" + v.currentTime.toFixed(2);
  drawFaces(v.currentTime, made, room.facesAt);
}

function drawChoices(row, why, choices, act, picture) {
  row.textContent = "";
  row.classList.toggle("picture-row", Boolean(picture));
  const off = {};
  choices.forEach((c) => {
    const b = el("button", picture ? "choice" : row.classList.contains("seg") ? null : "chip", picture ? null : c.label);
    b.type = "button";
    b.setAttribute("aria-pressed", String(c.now));
    if (picture) b.append(picture(c), el("span", null, c.label));
    if (c.why) {
      b.disabled = true;
      b.setAttribute("aria-describedby", why.id);
      (off[c.why] = off[c.why] || []).push(c.label);
    } else if (!c.now) {
      b.addEventListener("click", () => act(c));
    }
    row.appendChild(b);
  });
  const reasons = Object.entries(off).map(([words, labels]) => labels.join(", ") + ": " + words);
  why.textContent = reasons.join("\n");
  why.hidden = !reasons.length;
}

function showPart(name) {
  room.part = name;
  partTabs.querySelectorAll(".tool[data-show]").forEach((b) => b.setAttribute("aria-pressed", String(!sheet.hidden && b.dataset.show === name)));
  document.querySelectorAll("#panel [data-part]").forEach((e) => e.classList.toggle("part-off", e.dataset.part !== name));
}

partTabs.addEventListener("click", (e) => {
  const b = e.target.closest(".tool[data-show]");
  if (!b) return;
  if (!sheet.hidden && room.part === b.dataset.show) closeSheet();
  else openSheet(b.dataset.show);
});

function drawEye(d) {
  const eye = d && d.eye;
  eyeButton.hidden = !eye || !eye.has;
  if (!eye) return;
  eyeButton.setAttribute("aria-pressed", eye.hidden ? "true" : "false");
  eyeButton.setAttribute("aria-label", eye.hidden ? "Show the captions" : "Hide the captions");
  eyeButton.title = eye.hidden ? "Captions hidden: tap to show them" : "Hide every caption";
  eyeButton.replaceChildren(ico(eye.hidden ? "eye-off" : "eye"));
}

eyeButton.addEventListener("click", async () => {
  if (!room.panel || !room.panel.eye) return;
  eyeButton.disabled = true;
  try {
    const d = await post("/api/captions", { chat: room.chat, hidden: !room.panel.eye.hidden });
    if (!d.ok) panelNote.textContent = d.error || "That could not be changed.";
  } catch (err) {
    panelNote.textContent = "The studio is not answering. Start it again with: python app.py";
  }
  eyeButton.disabled = false;
  loadPanel();
});

function drawPanel(d) {
  drawEye(d);
  partTabs.querySelectorAll(".tool[data-show]").forEach((b) => (b.disabled = false));
  partTabs.querySelector('[data-show="words"]').hidden = !d.words.length;
  partTabs.querySelector('[data-show="channel"]').hidden = !d.channels;
  partTabs.querySelector('[data-show="sound"]').hidden = !(d.sound && d.sound.has);
  partTabs.querySelector('[data-show="frame"]').hidden = !d.frame || Boolean(d.frame.why);
  partTabs.querySelector('[data-show="layout"]').hidden = !d.layouts;
  partTabs.querySelector('[data-show="album"]').hidden = !d.album;
  if (d.album) drawAlbumPanel(d);
  const circle = $("circleRow");
  circle.hidden = $("circleHead").hidden = !d.circle;
  if (d.circle) drawChoices(circle, $("circleWhy"), d.circle.map((c) => Object.assign({ why: null }, c)), (x) => choose(circle, "/api/circle", { spot: x.name }), false);
  const caps = $("capsRow");
  const langs = $("langRow");
  const styles = $("styleRow");
  const looks = $("lookRow");
  const at = Number(clipAt.value) || 0;
  const lookPic = (c) => {
    const pic = el("img", "look-pic frame-pic");
    pic.src = "/api/clipframe?chat=" + encodeURIComponent(room.chat) + "&t=" + at.toFixed(1) + "&look=" + c.name + "&width=240";
    pic.alt = "";
    return pic;
  };
  const stylePic = (c) => {
    const pic = el("img", "look-pic");
    pic.src = "/api/look/" + c.name + ".png";
    pic.alt = "";
    return pic;
  };
  drawChoices(caps, $("capsWhy"), d.captions, (c) => choose(caps, "/api/captions", { mode: c.name }), (c) => mini("cap-" + c.name, c.label));
  drawChoices(langs, $("langWhy"), d.languages, (c) => (c.say ? askMuse(langs, c) : choose(langs, "/api/captions", { language: c.name })), null);
  drawChoices(styles, $("styleWhy"), d.styles, (c) => choose(styles, "/api/style", { style: c.name }), stylePic);
  drawChoices(looks, $("lookWhy"), d.looks, (c) => choose(looks, "/api/colour", { look: c.name }), lookPic);
  if (d.layouts) drawLayouts(d.layouts, at);
  if (d.channels) drawChannels(d.channels);
  drawSound(d.sound);
  drawTrim();
  drawFrame();
}

const muteFrom = $("muteFrom");
const muteTo = $("muteTo");
const muteGo = $("muteGo");

function clipNow() {
  const v = frameBox.querySelector("video");
  if (room.made) return Number(clipAt.value) || 0;
  return v ? v.currentTime : 0;
}

function drawMute() {
  const m = room.mute || {};
  muteFrom.lastChild.textContent = m.from === undefined ? "From here" : "From " + m.from.toFixed(1) + " s";
  muteTo.lastChild.textContent = m.to === undefined ? "To here" : "To " + m.to.toFixed(1) + " s";
  const ok = m.from !== undefined && m.to !== undefined && Math.abs(m.to - m.from) >= 0.1;
  muteFrom.disabled = muteTo.disabled = false;
  muteGo.disabled = !ok;
  muteGo.lastChild.textContent = ok ? "Mute " + Math.min(m.from, m.to).toFixed(1) + "–" + Math.max(m.from, m.to).toFixed(1) + " s" : "Mute";
  drawMuteMarks();
}

function drawSound(sound) {
  const row = $("mutedRow");
  row.textContent = "";
  ((sound && sound.mutes) || []).forEach(([a, b]) => {
    const back = el("button", "chip");
    back.type = "button";
    back.append(ico("mute", "sm"), a.toFixed(1) + "–" + b.toFixed(1) + " s", ico("again", "sm"));
    back.setAttribute("aria-label", "Bring back the sound from " + a.toFixed(1) + " to " + b.toFixed(1) + " s");
    back.addEventListener("click", () => mute(a, b, true));
    row.appendChild(back);
  });
  drawMute();
}

async function mute(start, end, back) {
  document.querySelectorAll("#muteRow button, #mutedRow button").forEach((b) => (b.disabled = true));
  panelNote.textContent = "";
  try {
    const d = await post("/api/mute", { chat: room.chat, start, end, back });
    if (d.ok) room.mute = {};
    else panelNote.textContent = d.error || "That could not be muted.";
  } catch (err) {
    panelNote.textContent = "The studio is not answering. Start it again with: python app.py";
  }
  loadPanel();
}

muteFrom.addEventListener("click", () => {
  room.mute = Object.assign({}, room.mute, { from: clipNow() });
  drawMute();
});

muteTo.addEventListener("click", () => {
  room.mute = Object.assign({}, room.mute, { to: clipNow() });
  drawMute();
});

muteGo.addEventListener("click", () => {
  const m = room.mute || {};
  mute(Math.max(0, Math.min(m.from, m.to)), Math.max(m.from, m.to), false);
});

const trimSays = $("trimSays");
const trimWhy = $("trimWhy");
const trimGo = $("trimGo");
const trimReset = $("trimReset");
const TRIM_STEP = 2;
const TITLE_SHORTEST = 1;
const pickBar = $("pickBar");
const pickSays = $("pickSays");
const pickBack = $("pickBack");
const pickWords = $("pickWords");
const pickGo = $("pickGo");

function tenths(t) {
  const m = Math.floor(t / 60);
  const s = t - 60 * m;
  return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1);
}

function onRecording(t) {
  const p = room.panel.ends.pieces;
  let at = 0;
  for (const [a, b] of p) {
    if (t <= at + (b - a)) return a + Math.max(0, t - at);
    at += b - a;
  }
  return p[p.length - 1][1];
}

function inClip(r) {
  const p = room.panel.ends.pieces;
  let at = 0;
  for (const [a, b] of p) {
    if (r < a) return at;
    if (r <= b) return at + r - a;
    at += b - a;
  }
  return at;
}

function trimNow() {
  const e = room.panel.ends;
  return room.trim || { start: e.start, end: e.end };
}

function setTrim(k, v) {
  const e = room.panel.ends;
  const t = Object.assign({}, trimNow());
  t[k] = Math.round(Math.max(0, Math.min(e.recording || Infinity, v)) * 100) / 100;
  room.trim = Math.abs(t.start - e.start) < 0.05 && Math.abs(t.end - e.end) < 0.05 ? null : t;
  trimWhy.hidden = true;
  drawTrim();
}

function drawTrim() {
  const e = room.panel && room.panel.ends;
  if (!e) return;
  const t = trimNow();
  const moved = Boolean(room.trim);
  const parts = [];
  if (moved && t.start < e.start) parts.push("starts " + (e.start - t.start).toFixed(1) + " s earlier");
  if (moved && t.start > e.start) parts.push("cuts " + (t.start - e.start).toFixed(1) + " s off the start");
  if (moved && t.end > e.end) parts.push("ends " + (t.end - e.end).toFixed(1) + " s later");
  if (moved && t.end < e.end) parts.push("cuts " + (e.end - t.end).toFixed(1) + " s off the end");
  trimSays.textContent = moved ? "New: " + tenths(t.start) + " to " + tenths(t.end) + " of your video (" + (t.end - t.start).toFixed(1) + " s): " + parts.join(", ") + "." : "The clip plays " + tenths(e.start) + " to " + tenths(e.end) + " of your video. Tap the timeline to choose a spot, then start or end the clip there, or take in " + TRIM_STEP + " s more at a time.";
  const off = Boolean(e.why) || room.work.ends;
  $("trimStart").disabled = $("trimEnd").disabled = off;
  $("trimEarlier").disabled = off || t.start <= 0;
  $("trimLater").disabled = off || Boolean(e.recording && t.end >= e.recording);
  const span = t.end - t.start;
  const fits = span >= e.shortest - 0.5 && span <= e.longest + 0.5;
  trimGo.disabled = off || !moved || !fits;
  trimReset.hidden = !moved;
  trimGo.lastChild.textContent = moved && !fits ? (span < e.shortest ? "Too short" : "Too long") : "Apply";
  if (e.why) {
    trimWhy.textContent = e.why;
    trimWhy.hidden = false;
  } else if (moved && !fits) {
    trimWhy.textContent = "A clip is " + e.shortest + " to " + e.longest + " s long.";
    trimWhy.hidden = false;
  }
  drawTrimMarks();
  drawPick();
}

function edgeAt(k) {
  const e = room.panel.ends;
  const t = trimNow();
  if (k === "start") return t.start < e.start ? t.start - e.start : inClip(t.start);
  return t.end > e.end ? room.panel.length + t.end - e.end : inClip(t.end);
}

function drawTrimMarks() {
  const track = lanes.querySelector(".lane.film .track");
  if (!track || !room.panel || !room.panel.ends) return;
  track.querySelectorAll(".trim-span, .trim-more, .grip").forEach((n) => n.remove());
  const e = room.panel.ends;
  const length = room.panel.length;
  const t = trimNow();
  const shade = (a, b) => {
    const s = el("i", "trim-span");
    s.style.left = (100 * a / length) + "%";
    s.style.width = (100 * Math.max(0, b - a) / length) + "%";
    track.appendChild(s);
  };
  if (t.start > e.start) shade(0, inClip(t.start));
  if (t.end < e.end) shade(inClip(t.end), length);
  const at = (k) => 100 * Math.max(0, Math.min(length, edgeAt(k))) / length;
  const more = (k, by) => {
    if (Math.abs(by) < 0.05) return;
    const b = el("b", "trim-more", (by > 0 ? "+" : "−") + Math.abs(by).toFixed(1) + " s");
    if (k === "start") b.style.left = "calc(" + at(k) + "% + 18px)";
    else b.style.right = "calc(" + (100 - at(k)) + "% + 18px)";
    track.appendChild(b);
  };
  more("start", e.start - t.start);
  more("end", t.end - e.end);
  if (e.why || room.work.ends) return;
  ["start", "end"].forEach((k) => {
    const g = el("i", "grip " + k);
    g.dataset.edge = k;
    g.style.left = at(k) + "%";
    track.appendChild(g);
  });
}

function trimDrag(e, k) {
  const box = e.target.closest(".track").getBoundingClientRect();
  const ends = room.panel.ends;
  const length = room.panel.length;
  const per = length / box.width;
  const from = edgeAt(k);
  const x0 = e.clientX;
  const out = k === "start" ? -1 : 1;
  let x = x0;
  let grown = 0;
  let last = 0;
  let shown = -1;
  let frame = 0;
  const place = () => {
    let v = from + (x - x0) * per + out * grown;
    const far = k === "start" ? -ends.start : length + (ends.recording || Infinity) - ends.end;
    if (out * (v - far) > 0) {
      grown -= out * (v - far);
      v = far;
    }
    if (k === "start") v = Math.min(v, edgeAt("end") - 1);
    else v = Math.max(v, edgeAt("start") + 1);
    if (v < 0) setTrim(k, ends.start + v);
    else if (v > length) setTrim(k, ends.end + v - length);
    else {
      setTrim(k, onRecording(v));
      if (Math.abs(v - shown) > 0.05) seek((shown = v));
    }
  };
  const grow = (now) => {
    const past = out * (x - (k === "start" ? box.left : box.right));
    if (past > 0 && last) grown += (now - last) / 1000 * Math.min(6, 0.5 + past / 10);
    last = now;
    if (room.dragged) place();
    frame = requestAnimationFrame(grow);
  };
  const move = (ev) => {
    x = ev.clientX;
    if (Math.abs(x - x0) > 3) room.dragged = true;
  };
  const up = () => {
    cancelAnimationFrame(frame);
    setTimeout(() => (room.dragged = false), 0);
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    window.removeEventListener("pointercancel", up);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
  window.addEventListener("pointercancel", up);
  frame = requestAnimationFrame(grow);
}

const trimAt = $("trimAt");
const trimAtLine = $("trimAtLine");

function showTrimAt(t) {
  if (room.panel) trimAt.max = room.panel.length;
  trimAt.value = t;
  trimAtLine.textContent = tenths(Number(trimAt.value));
}

trimAt.addEventListener("input", () => seek(Number(trimAt.value)));

$("trimEarlier").addEventListener("click", () => setTrim("start", trimNow().start - TRIM_STEP));
$("trimLater").addEventListener("click", () => setTrim("end", trimNow().end + TRIM_STEP));
$("trimStart").addEventListener("click", () => setTrim("start", onRecording(clipNow())));
$("trimEnd").addEventListener("click", () => setTrim("end", onRecording(clipNow())));

function trimBack() {
  room.trim = null;
  trimWhy.hidden = true;
  drawTrim();
}

trimReset.addEventListener("click", trimBack);
pickBack.addEventListener("click", trimBack);
trimGo.addEventListener("click", trimApply);
pickGo.addEventListener("click", trimApply);

async function trimApply() {
  const t = room.trim;
  if (!t) return;
  trimGo.disabled = pickGo.disabled = true;
  trimWhy.hidden = true;
  try {
    const d = await post("/api/ends", { chat: room.chat, start: t.start, end: t.end });
    if (d.ok) {
      pageWork("ends", true);
      againLine.textContent = "Making the clip again…";
    } else {
      trimWhy.textContent = d.error || "The clip could not be changed.";
      trimWhy.hidden = false;
    }
  } catch (err) {
    trimWhy.textContent = "The studio is not answering. Start it again with: python app.py";
    trimWhy.hidden = false;
  }
  drawTrim();
}

const framePick = $("framePick");
const framePic = $("framePic");
const frameWindow = $("frameWindow");
const frameSays = $("frameSays");
const frameWhy = $("frameWhy");
const frameGo = $("frameGo");
const frameReset = $("frameReset");
const FRAME_STEP = 0.05;

function frameNow() {
  const f = room.panel.frame;
  return room.frameX === null || room.frameX === undefined ? f.x : room.frameX;
}

function setFrame(x) {
  const f = room.panel && room.panel.frame;
  if (!f || f.why) return;
  const v = Math.round(Math.max(0, Math.min(f.size[0] - f.width, x)));
  room.frameX = v === f.x && !f.follows ? null : v;
  frameWhy.hidden = true;
  drawFrame();
}

function drawFrame() {
  const f = room.panel && room.panel.frame;
  if (!f || f.why) return;
  const x = frameNow();
  const moved = room.frameX !== null && room.frameX !== undefined;
  const share = (v) => Math.round((100 * v) / f.size[0]);
  if (f.picture && framePic.dataset.name !== f.picture) {
    framePic.src = "/api/frame/" + f.picture;
    framePic.dataset.name = f.picture;
  }
  framePick.hidden = !f.picture;
  framePick.style.aspectRatio = f.size[0] + " / " + f.size[1];
  frameWindow.style.left = (100 * x) / f.size[0] + "%";
  frameWindow.style.width = (100 * f.width) / f.size[0] + "%";
  frameSays.textContent = moved ? "New: the clip shows " + share(x) + "% to " + share(x + f.width) + "% of your video's width" + (f.follows ? ", held there instead of following the product." : ".") : f.follows ? "The frame follows the product now. Drag it, or tap Left, Middle or Right, to hold it in one place." : "Drag the frame over your video, or tap Left, Middle or Right, to choose what the clip shows.";
  $("frameLeft").disabled = x <= 0;
  $("frameRight").disabled = x >= f.size[0] - f.width;
  frameGo.disabled = !moved;
  frameReset.hidden = !moved;
}

framePick.addEventListener("pointerdown", (e) => {
  const f = room.panel && room.panel.frame;
  if (!f || f.why) return;
  const box = framePick.getBoundingClientRect();
  const at = ((e.clientX - box.left) / box.width) * f.size[0];
  const x = frameNow();
  const grab = at >= x && at <= x + f.width ? at - x : f.width / 2;
  framePick.setPointerCapture(e.pointerId);
  framePick.classList.add("held");
  const follow = (ev) => setFrame(((ev.clientX - box.left) / box.width) * f.size[0] - grab);
  follow(e);
  const done = () => {
    framePick.classList.remove("held");
    framePick.removeEventListener("pointermove", follow);
    framePick.removeEventListener("pointerup", done);
    framePick.removeEventListener("pointercancel", done);
  };
  framePick.addEventListener("pointermove", follow);
  framePick.addEventListener("pointerup", done);
  framePick.addEventListener("pointercancel", done);
});

$("frameLeft").addEventListener("click", () => setFrame(frameNow() - FRAME_STEP * room.panel.frame.size[0]));
$("frameRight").addEventListener("click", () => setFrame(frameNow() + FRAME_STEP * room.panel.frame.size[0]));
$("frameMiddle").addEventListener("click", () => setFrame((room.panel.frame.size[0] - room.panel.frame.width) / 2));
frameReset.addEventListener("click", () => {
  room.frameX = null;
  frameWhy.hidden = true;
  drawFrame();
});
frameGo.addEventListener("click", frameApply);

async function frameApply() {
  if (room.frameX === null || room.frameX === undefined) return;
  frameGo.disabled = true;
  frameWhy.hidden = true;
  try {
    const d = await post("/api/frame", { chat: room.chat, x: room.frameX });
    if (d.ok) {
      room.frameX = null;
    } else {
      frameWhy.textContent = d.error || "The frame could not be moved.";
      frameWhy.hidden = false;
    }
  } catch (err) {
    frameWhy.textContent = "The studio is not answering. Start it again with: python app.py";
    frameWhy.hidden = false;
  }
  drawFrame();
}

function pickedTitle() {
  if (!room.pick || !room.panel) return null;
  const same = room.panel.words.filter((b) => b.kind === "title" && b.text === room.pick.was);
  return same.length === 1 ? same[0] : same.find((b) => b.n === room.pick.n) || null;
}

function drawPick() {
  const b = pickedTitle();
  const e = room.panel && room.panel.ends;
  const trimming = Boolean(!b && e && room.trim);
  pickWords.hidden = !b;
  pickBack.hidden = pickGo.hidden = !trimming;
  pickBar.hidden = !b && !trimming;
  if (b) {
    pickSays.textContent = room.pickWhy || b.start.toFixed(1) + "–" + b.end.toFixed(1) + " s · drag its ends";
  } else if (trimming) {
    pickSays.textContent = trimWhy.hidden ? "New: " + (room.trim.end - room.trim.start).toFixed(1) + " s" : trimWhy.textContent;
    pickGo.disabled = trimGo.disabled;
    pickGo.lastChild.textContent = trimGo.lastChild.textContent;
  }
}

function drawPicked() {
  const b = pickedTitle();
  if (b) room.pick.n = b.n;
  lanes.querySelectorAll(".lane.title .block").forEach((node) => {
    const on = Boolean(b) && Number(node.dataset.n) === b.n;
    node.classList.toggle("picked", on);
    node.setAttribute("aria-pressed", String(on));
    node.querySelectorAll(".grip").forEach((g) => g.remove());
    if (!on) return;
    ["start", "end"].forEach((k) => {
      const g = el("i", "grip " + k);
      g.dataset.edge = k;
      node.appendChild(g);
    });
  });
  drawPick();
}

function tapBlock(b) {
  if (room.dragged) return;
  room.pickWhy = "";
  if (b.kind === "title" && !(pickedTitle() && pickedTitle().n === b.n)) {
    room.pick = { n: b.n, was: b.text };
    seek(b.start);
    drawPicked();
    return;
  }
  if (b.kind !== "title") room.pick = null;
  drawPicked();
  openWords(b);
}

pickWords.addEventListener("click", () => {
  const b = pickedTitle();
  if (b) openWords(b);
});

function titleDrag(e, k, node) {
  const b = pickedTitle();
  if (!b) return;
  const box = node.parentNode.getBoundingClientRect();
  const length = room.panel.length;
  const x0 = e.clientX;
  const t = { start: b.start, end: b.end };
  const move = (ev) => {
    if (Math.abs(ev.clientX - x0) > 3) room.dragged = true;
    if (!room.dragged) return;
    const v = Math.round(Math.max(0, Math.min(length, b[k] + (ev.clientX - x0) / box.width * length)) * 10) / 10;
    if (k === "start") t.start = Math.min(v, t.end - TITLE_SHORTEST);
    else t.end = Math.max(v, t.start + TITLE_SHORTEST);
    node.style.left = (100 * t.start / length) + "%";
    node.style.width = (100 * (t.end - t.start) / length) + "%";
    pickSays.textContent = t.start.toFixed(1) + "–" + t.end.toFixed(1) + " s · let go to keep it";
    seek(t[k]);
  };
  const up = async () => {
    const dragged = room.dragged;
    setTimeout(() => (room.dragged = false), 0);
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    window.removeEventListener("pointercancel", up);
    if (!dragged || (t.start === b.start && t.end === b.end)) return;
    pickSays.textContent = "Moving the words…";
    try {
      const d = await post("/api/title/time", { chat: room.chat, n: b.n, was: b.text, start: t.start, end: t.end });
      room.pickWhy = d.ok ? "" : d.error || "The words could not be moved.";
    } catch (err) {
      room.pickWhy = "The studio is not answering. Start it again with: python app.py";
    }
    loadPanel();
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
  window.addEventListener("pointercancel", up);
}

const channelSave = $("channelSave");
const channelName = $("channelName");
const channelHandle = $("channelHandle");

function drawChannels(c) {
  const row = $("channelRow");
  const spots = $("spotRow");
  const looks = [{ name: "", label: "None", now: !c.in_use, why: null }].concat(c.saved.map((s) => ({ name: s.name, label: s.name, now: s.name === c.in_use, why: null })));
  drawChoices(row, $("channelWhy"), looks, (x) => choose(row, "/api/channel", { name: x.name }), false);
  const using = c.saved.find((s) => s.name === c.in_use);
  $("channelSays").textContent = using ? "New clips start in this look: " + using.says + "." : c.saved.length ? "No channel look is in use: new clips start in Muse's guesses." : "No channel look yet. Make a clip look the way your channel does, then save it here.";
  spots.hidden = !c.handle;
  drawChoices(spots, $("spotWhy"), c.spots.map((s) => Object.assign({ why: null }, s)), (x) => choose(spots, "/api/channel", { spot: x.name }), false);
  if (using && document.activeElement !== channelName && document.activeElement !== channelHandle && !channelName.value) {
    channelName.value = using.name;
    channelHandle.value = using.handle;
  }
}

channelSave.addEventListener("submit", (e) => {
  e.preventDefault();
  const name = channelName.value.trim();
  if (!name) {
    panelNote.textContent = "Give the channel a name first.";
    channelName.focus();
    return;
  }
  choose(channelSave, "/api/channel/save", { name: name, handle: channelHandle.value.trim() });
});

async function askMuse(row, c) {
  row.querySelectorAll("button").forEach((b) => (b.disabled = true));
  panelNote.textContent = (await say(c.say)) ? "Muse is putting the captions in " + c.label + ". It writes them, so this takes a little while." : "";
  loadPanel();
}

function drawLayouts(choices, at) {
  const row = $("layoutRow");
  const tile = () => {
    const none = el("div", "layout-none");
    none.appendChild(ico("stack"));
    return none;
  };
  const pic = (c) => {
    if (!c.picture) return tile();
    const img = el("img", "look-pic layout-pic");
    img.addEventListener("error", () => img.replaceWith(tile()));
    img.src = "/api/clipframe?chat=" + encodeURIComponent(room.chat) + "&t=" + at.toFixed(1) + "&layout=" + c.name + "&width=180";
    img.alt = "";
    return img;
  };
  drawChoices(row, $("layoutWhy"), choices, chooseLayout, pic);
  if (room.work.layout) row.querySelectorAll("button").forEach((b) => (b.disabled = true));
}

async function chooseLayout(c) {
  const row = $("layoutRow");
  const why = $("layoutWhy");
  row.querySelectorAll("button").forEach((b) => (b.disabled = true));
  why.hidden = false;
  why.textContent = "Making the clip in the " + c.label + " layout…";
  try {
    const d = await post("/api/layout", { chat: room.chat, layout: c.name });
    if (!d.ok) {
      why.textContent = d.error || "The layout could not be changed.";
      loadPanel();
      return;
    }
    pageWork("layout", true);
  } catch (err) {
    why.textContent = "The studio is not answering. Start it again with: python app.py";
  }
}

async function choose(row, path, body) {
  row.querySelectorAll("button").forEach((b) => (b.disabled = true));
  panelNote.textContent = "";
  try {
    const d = await post(path, Object.assign({ chat: room.chat }, body));
    if (!d.ok) panelNote.textContent = d.error || "That could not be changed.";
  } catch (err) {
    panelNote.textContent = "The studio is not answering. Start it again with: python app.py";
  }
  loadPanel();
}

function drawAlbumPanel(d) {
  const a = d.album;
  const box = $("albumParts");
  if (!box.children.length) {
    a.parts.forEach((p) => {
      const tick = el("input");
      tick.type = "checkbox";
      tick.checked = true;
      tick.value = p.name;
      const label = el("label");
      label.append(tick, p.label);
      box.appendChild(label);
    });
  }
  const c = d.channels;
  const words = $("taglineWords");
  if (document.activeElement !== words) words.value = c.handle;
  const spots = $("tagSpotRow");
  spots.hidden = !c.handle;
  drawChoices(spots, $("tagSpotWhy"), c.spots.map((s) => Object.assign({ why: null }, s)), (x) => choose(spots, "/api/channel", { spot: x.name }), false);
  const stopped = d.making;
  $("albumUse").disabled = stopped;
  const back = $("albumBack");
  back.hidden = !a.matched;
  back.disabled = stopped;
  const make = $("albumMake");
  make.hidden = !a.changed;
  make.disabled = stopped;
  make.textContent = a.changed === 1 ? "Update 1 video" : "Update all " + a.changed + " videos";
}

function albumNote(words) {
  const says = $("albumSays");
  says.hidden = !words;
  says.textContent = words || "";
}

async function albumPress(path, body) {
  albumNote("");
  try {
    const d = await post(path, Object.assign({ chat: room.chat }, body));
    if (!d.ok) albumNote(d.error || "That could not be done.");
    return d.ok;
  } catch (err) {
    albumNote("The studio is not answering. Start it again with: python app.py");
    return false;
  } finally {
    loadPanel();
  }
}

$("albumUse").addEventListener("click", () => {
  const parts = [...$("albumParts").querySelectorAll("input:checked")].map((t) => t.value);
  albumPress("/api/album/look", { parts: parts });
});

$("albumBack").addEventListener("click", () => albumPress("/api/album/back", {}));

$("albumMake").addEventListener("click", async () => {
  if (await albumPress("/api/album/make", {})) pageWork("video", true);
});

$("taglineForm").addEventListener("submit", (e) => {
  e.preventDefault();
  albumPress("/api/tagline", { words: $("taglineWords").value.trim() });
});

function rows(blocks) {
  const ends = [];
  return blocks.map((b) => {
    let k = ends.findIndex((end) => end <= b.start);
    if (k < 0) k = ends.push(0) - 1;
    ends[k] = b.end;
    return k;
  });
}

function filmLane(d) {
  const lane = el("div", "lane film");
  lane.setAttribute("aria-hidden", "true");
  const track = el("div", "track");
  const src = room.ready ? "/api/clip/" + room.ready.file.replace(/\.mp4$/, ".jpg") : "/api/clipframe?chat=" + encodeURIComponent(room.chat) + "&t=" + (d.length / 2).toFixed(1) + "&width=90";
  track.style.backgroundImage = "url(\"" + src + "\")";
  lane.append(ico("film", "sm lane-ico"), track);
  return lane;
}

function drawWords(d) {
  lanes.textContent = "";
  lanes.appendChild(filmLane(d));
  const kinds = [["title", "Words on screen", "aa"], ["caption", "Captions", "cc"]];
  kinds.forEach(([kind, name, icon]) => {
    const blocks = d.words.filter((b) => b.kind === kind || (kind === "caption" && b.kind === "draft"));
    if (!blocks.length) return;
    const lane = el("div", "lane " + kind);
    lane.setAttribute("role", "group");
    lane.setAttribute("aria-label", name);
    const track = el("div", "track");
    const row = rows(blocks);
    track.style.height = (Math.max(...row) + 1) * 30 + "px";
    blocks.forEach((b, i) => {
      const button = el("button", "block" + (b.typed ? " typed" : "") + (b.kind === "draft" ? " draft" : ""), b.text);
      button.type = "button";
      button.title = b.text;
      button.style.left = (100 * b.start / d.length) + "%";
      button.style.width = (100 * Math.max(0, Math.min(b.end, d.length) - b.start) / d.length) + "%";
      button.style.top = row[i] * 30 + "px";
      button.dataset.n = b.n;
      button.setAttribute("aria-label", wordName(b) + " from " + b.start.toFixed(1) + " to " + b.end.toFixed(1) + " s: " + b.text + (b.kind === "draft" ? ". Not on the video until you keep or fix it." : b.typed ? ". Your own words." : ""));
      button.addEventListener("click", () => tapBlock(b));
      track.appendChild(button);
    });
    lane.append(ico(icon, "sm lane-ico"), track);
    lanes.appendChild(lane);
  });
  const head = el("div", "playhead");
  head.setAttribute("aria-hidden", "true");
  lanes.appendChild(head);
  drawWordList(d);
  drawMuteMarks();
  drawTrimMarks();
  drawPicked();
  movePlayhead(room.made ? Number(clipAt.value) : 0);
  if (room.editing && !d.words.some((b) => b.kind === room.editing.kind && b.n === room.editing.n && b.text === room.editing.was)) closeWords();
}

function drawMuteMarks() {
  const track = lanes.querySelector(".lane.film .track");
  if (!track || !room.panel) return;
  track.querySelectorAll(".mute-span, .mute-mark").forEach((n) => n.remove());
  const length = room.panel.length;
  ((room.panel.sound && room.panel.sound.mutes) || []).forEach(([a, b]) => {
    const s = el("i", "mute-span");
    s.style.left = (100 * a / length) + "%";
    s.style.width = (100 * (b - a) / length) + "%";
    track.appendChild(s);
  });
  ["from", "to"].forEach((k) => {
    const t = (room.mute || {})[k];
    if (t === undefined) return;
    const m = el("i", "mute-mark");
    m.style.left = (100 * t / length) + "%";
    track.appendChild(m);
  });
}

function wordName(b) {
  return b.kind === "title" ? "Words on screen" : b.kind === "draft" ? "Caption draft" : "Caption";
}

function drawWordList(d) {
  wordList.replaceChildren();
  d.words.forEach((b) => {
    const r = el("button", "word-row" + (b.typed ? " typed" : "") + (b.kind === "draft" ? " draft" : ""));
    r.type = "button";
    r.append(ico(b.kind === "title" ? "aa" : "cc", "sm"), el("span", "at", clock(b.start)), el("span", "w", b.text));
    if (b.kind === "draft") r.append(el("span", "tag", "Not sure"));
    r.append(ico("words", "sm"));
    r.setAttribute("aria-label", "Change " + (b.kind === "title" ? "the words on screen" : b.kind === "draft" ? "the caption draft, not on the video yet," : "the caption") + " at " + b.start.toFixed(1) + " s: " + b.text);
    r.addEventListener("click", () => openWords(b));
    wordList.appendChild(r);
  });
}

function movePlayhead(t) {
  const head = lanes.querySelector(".playhead");
  if (!head || !room.panel) return;
  const share = Math.max(0, Math.min(1, t / room.panel.length));
  head.style.left = (share * 100) + "%";
  showTrimAt(t);
  if (room.made) updatePlayer();
}

function seek(t) {
  const v = frameBox.querySelector("video");
  if (room.made) {
    clipAt.value = Math.max(0, Math.min(t + 0.05, room.made.clip.length - 0.05));
    movePlayhead(Number(clipAt.value));
    drawClipFrame();
  } else if (v && !room.showing) {
    v.currentTime = t;
    movePlayhead(t);
  }
}

$("timeline").addEventListener("pointerdown", (e) => {
  room.dragged = false;
  if (!room.panel) return;
  const grip = e.target.closest(".grip");
  if (grip) {
    e.preventDefault();
    if (grip.closest(".block")) titleDrag(e, grip.dataset.edge, grip.closest(".block"));
    else trimDrag(e, grip.dataset.edge);
    return;
  }
  if (e.target.closest(".block")) return;
  if (room.pick) {
    room.pick = null;
    room.pickWhy = "";
    drawPicked();
  }
  const at = (ev) => {
    const r = lanes.getBoundingClientRect();
    seek(Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)) * room.panel.length);
  };
  at(e);
  const up = () => {
    window.removeEventListener("pointermove", at);
    window.removeEventListener("pointerup", up);
  };
  window.addEventListener("pointermove", at);
  window.addEventListener("pointerup", up);
});

function openWords(b) {
  seek(b.start);
  room.editing = { kind: b.kind, n: b.n, was: b.text };
  if (sheet.hidden || room.part !== "words") openSheet("words");
  wordLabel.textContent = wordName(b) + ", " + b.start.toFixed(1) + " to " + b.end.toFixed(1) + " s" + (b.kind === "draft" ? " · the hearing was not sure, so it is not on the video: Save puts it on, as it is or as you change it" : b.typed ? " · your own words" : "");
  wordText.value = b.text;
  wordNote.textContent = "";
  wordEdit.hidden = false;
  wordText.focus();
}

function closeWords() {
  room.editing = null;
  wordEdit.hidden = true;
  wordNote.textContent = "";
}

wordEdit.addEventListener("submit", async (e) => {
  e.preventDefault();
  const editing = room.editing;
  if (!editing) return;
  wordSave.disabled = true;
  try {
    const d = await post("/api/words", { chat: room.chat, kind: editing.kind, n: editing.n, was: editing.was, words: wordText.value });
    if (d.ok) closeWords();
    else wordNote.textContent = d.error || "Those words could not be saved.";
  } catch (err) {
    wordNote.textContent = "The studio is not answering. Start it again with: python app.py";
  }
  wordSave.disabled = false;
});

wordCancel.addEventListener("click", closeWords);

wordText.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeWords();
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    wordEdit.requestSubmit();
  }
});

function drawAgain(d) {
  pageWork("video", d.making);
  againButton.hidden = !(d.again || d.making);
  againButton.disabled = d.making;
  if (d.making && !againLine.textContent) againLine.textContent = "Making the video…";
  if (!d.making && !room.againError) againLine.textContent = d.again ? "You changed the clip after the video was made." : "";
}

againButton.addEventListener("click", async () => {
  againButton.disabled = true;
  room.againError = false;
  againLine.textContent = "Making the video…";
  try {
    const d = await post("/api/remake", { chat: room.chat });
    if (d.ok) pageWork("video", true);
    if (!d.ok) {
      room.againError = true;
      againLine.textContent = d.error || "The video could not be made.";
      againButton.disabled = false;
    }
  } catch (err) {
    againLine.textContent = "The studio is not answering. Start it again with: python app.py";
    againButton.disabled = false;
  }
});

clipUndo.addEventListener("click", async () => {
  clipUndo.disabled = true;
  try {
    const d = await post("/api/undo", { chat: room.chat });
    if (!d.ok) clipLine.textContent = d.error || "Nothing was taken back.";
  } catch (err) {
    clipLine.textContent = "The studio is not answering. Start it again with: python app.py";
  }
  clipUndo.disabled = !room.panel;
});

function showRecording(e) {
  const r = e.recording;
  room.wantBase = false;
  room.base = null;
  if (e.frame) {
    const img = el("img");
    img.src = "/api/frame/" + e.frame;
    img.alt = "A picture from " + r.name;
    frameBox.replaceChildren(img);
    thumb("/api/frame/" + e.frame);
  } else {
    frameBox.replaceChildren(el("p", "hint", "Opened " + r.name));
  }
  clipTitle.textContent = r.name;
  clipLine.textContent = clock(r.seconds) + " long. Your clip shows here once Muse makes it.";
  updatePlayer();
}

const ENDS = {
  answered: "Done",
  stopped: "Stopped",
  closed: "Ended: the studio was closed",
  mind: "Ended: Muse could not answer",
  fault: "Ended: the studio broke",
  repeating: "Ended: Muse kept repeating itself",
  "too many steps": "Paused: too many steps",
};

function endJob(j, e) {
  stopTimer();
  j.ended = true;
  j.card.classList.remove("run");
  j.state.textContent = ENDS[e.why] || "Done";
  const parts = [plural(e.steps, "step")];
  if (e.seconds) parts.push(clock(e.seconds));
  if (e.refused) parts.push(e.refused + " refused" + (e.fixed ? ", " + e.fixed + " fixed" : ""));
  j.meta.textContent = parts.join(" · ");
  j.tick.textContent = "";
  j.think.textContent = "";
  j.replay.hidden = j.count < 2;
  const left = j.items.find((item) => item.state !== "done");
  j.now.textContent = left ? left.text : "";
  j.now.hidden = !left;
}

function replayJob(j) {
  if (!j.ended || j.card.classList.contains("run")) return;
  const icons = [...j.strip.children];
  const words = [...j.rows.children].map((row) => (row.querySelector(".why") || {}).textContent || "");
  const labels = [...j.rows.children].map((row) => (row.querySelector("b") || {}).textContent || "");
  const state = j.state.textContent;
  const meta = j.meta.textContent;
  const plan = [...j.plan.children].map((i) => i.className);
  const moments = j.momentsBox.hidden;
  icons.forEach((b) => (b.hidden = true));
  j.momentsBox.hidden = true;
  j.card.classList.add("run");
  j.state.textContent = "Working";
  let n = 0;
  const one = () => {
    if (n >= icons.length) {
      j.card.classList.remove("run");
      j.state.textContent = state;
      j.meta.textContent = meta;
      j.tick.textContent = "";
      [...j.plan.children].forEach((i, k) => (i.className = plan[k]));
      j.momentsBox.hidden = moments;
      return;
    }
    const b = icons[n];
    b.hidden = false;
    b.classList.remove("new");
    void b.offsetWidth;
    b.classList.add("new");
    j.strip.scrollLeft = j.strip.scrollWidth;
    j.meta.textContent = labels[n] + " · step " + (n + 1);
    j.tick.textContent = words[n];
    const share = (n + 1) / icons.length;
    [...j.plan.children].forEach((i, k, all) => (i.className = (k + 1) / all.length <= share ? "done" : k / all.length <= share ? "now" : ""));
    if (/moment/i.test(labels[n]) && j.momentsBox.children.length) j.momentsBox.hidden = false;
    n += 1;
    setTimeout(one, b.classList.contains("bad") ? 650 : 420);
  };
  one();
}

function on(e) {
  if (e.id <= room.last) return;
  room.last = e.id;
  if (e.kind === "person") {
    personBubble(e.text, e.cut_in && room.jobs[e.job] ? room.jobs[e.job].flow : null);
    return;
  }
  if (e.kind === "opened") {
    showRecording(e);
    return;
  }
  if (e.kind === "named") {
    room.title = e.title;
    chatName.textContent = e.title;
    return;
  }
  if (e.kind === "edit" || e.kind === "undo") {
    showClip(e);
    room.againError = false;
    loadPanel();
    return;
  }
  if (e.kind === "album" || e.kind === "picked") {
    showPicked(e);
    return;
  }
  if (e.kind === "matched") {
    albumNote(e.said);
    loadPanel();
    return;
  }
  if (e.kind === "unmatched" || e.kind === "album_made") {
    albumNote(e.said);
    if (e.kind === "album_made") pageWork("video", false);
    showPicked(e);
    return;
  }
  if (e.kind === "rendered") {
    room.wantBase = false;
    room.finished = e.file.name;
    showFinished(e.file);
    againLine.textContent = "";
    if (e.job === null) pageWork("video", false);
    loadPanel();
    return;
  }
  if (e.kind === "not_made") {
    room.againError = true;
    againLine.textContent = "The video was not made: " + e.error;
    pageWork("video", false);
    loadPanel();
    return;
  }
  if (e.kind === "relaid" || e.kind === "not_relaid") {
    pageWork("layout", false);
    againLine.textContent = "";
    const why = $("layoutWhy");
    why.textContent = e.kind === "relaid" ? "" : "The layout was not changed: " + e.error;
    why.hidden = e.kind === "relaid";
    loadPanel();
    return;
  }
  if (e.kind === "moved" || e.kind === "not_moved") {
    pageWork("ends", false);
    againLine.textContent = "";
    if (e.kind === "moved") room.trim = null;
    trimWhy.textContent = e.kind === "moved" ? e.said || "" : "The clip was not changed: " + e.error;
    trimWhy.hidden = e.kind === "moved" && !e.said;
    loadPanel();
    return;
  }
  if (e.kind === "base") {
    pageWork("preview", false);
    if (e.error) previewLine.textContent = "No quick preview: " + e.error;
    loadPanel();
    return;
  }
  if (e.kind === "preview") {
    room.cut[e.name] = e;
    document.querySelectorAll(".moment").forEach((tile) => {
      if (tile.dataset.preview === e.name) fillPreview(tile, e);
    });
    if (room.showing && room.showing.preview === e.name) showMoment(room.showing);
    return;
  }
  if (e.job === null || e.job === undefined) return;
  const j = job(e.job);
  if (e.kind === "job") {
    j.mind = e.mind;
    j.card.classList.add("run");
    j.cost.replaceChildren(ico(isLocal(e.mind) ? "pc" : "cloud", "sm"));
    j.cost.hidden = false;
    if (!room.minded.some((m) => m.label === e.mind)) room.minded.push({ label: e.mind, local: isLocal(e.mind) });
    if (!isLocal(e.mind)) room.paid = true;
    room.open.add(e.job);
    setBusy();
  } else if (e.kind === "plan") {
    drawPlan(j, e);
  } else if (e.kind === "guesses") {
    drawGuesses(j, e);
  } else if (e.kind === "wait") {
    stopTimer();
    const tick = () => {
      j.meta.textContent = "Thinking about step " + e.step + " · " + Math.max(0, Math.round(Date.now() / 1000 - e.t)) + " s";
    };
    tick();
    room.timer = setInterval(tick, 1000);
  } else if (e.kind === "step") {
    stopTimer();
    drawStep(j, e);
  } else if (e.kind === "answer") {
    stopTimer();
    drawAnswer(j, e);
  } else if (e.kind === "checked") {
    drawChecked(j, e);
  } else if (e.kind === "spelled") {
    e.fixes.forEach((f) => {
      const spot = f.kind === "step" ? (j.steps[f.step] || {})[f.field] : f.kind === "moment" ? (j.moments[f.step] || [])[f.index] : j.answers[f.index];
      if (spot) spot.textContent = f.text;
    });
  } else if (e.kind === "ready") {
    drawReady(j, e);
  } else if (e.kind === "cut") {
    stopTimer();
    note(e.by === "stop" ? "Stopped while Muse was thinking about step " + e.step + "." : "Muse stopped thinking about step " + e.step + " to read your message.", j.flow);
  } else if (e.kind === "note") {
    note(e.text, j.flow);
  } else if (e.kind === "end") {
    endJob(j, e);
    room.open.delete(e.job);
    setBusy();
    refreshCost();
  }
  scrollDown();
}

function onThinking(live) {
  if (live.job === null) {
    if (live.work && live.page === "layout") {
      if (!room.work.layout) pageWork("layout", true);
      againLine.textContent = live.work;
      $("layoutWhy").hidden = false;
      $("layoutWhy").textContent = live.work;
    } else if (live.work && live.page === "ends") {
      if (!room.work.ends) pageWork("ends", true);
      againLine.textContent = live.work;
    } else if (live.work && live.page === "preview") {
      if (!room.work.preview) pageWork("preview", true);
      previewLine.hidden = false;
      previewLine.textContent = live.work;
    } else if (live.work) {
      againButton.hidden = false;
      againButton.disabled = true;
      againLine.textContent = live.work;
    }
    return;
  }
  const j = room.jobs[live.job];
  if (!j || j.ended) return;
  if (live.work) {
    stopTimer();
    j.tick.textContent = live.work;
    j.think.textContent = "";
    scrollDown();
    return;
  }
  const words = (live.thinking + (live.text ? "\n" + live.text : "")).trim();
  j.tick.textContent = "";
  j.think.textContent = words.length > 1500 ? "…" + words.slice(-1500) : words;
  j.think.scrollTop = j.think.scrollHeight;
  scrollDown();
}

function current() {
  return room.minds.find((m) => m.key === room.mind) || room.minds[0];
}

function showStatus() {
  const m = current();
  if (room.open.size) {
    statusLine.textContent = "Muse is working on it. Type to change what it does, or press Stop.";
  } else if (room.work.ends) {
    statusLine.textContent = "Making the clip again with its new start and end. Press Stop to end it.";
  } else if (room.work.video || room.work.preview) {
    statusLine.textContent = "Making the " + (room.work.video ? "video" : "preview") + ". Press Stop to end it.";
  } else if (m) {
    statusLine.textContent = m.ok ? m.label + " is ready, " + m.where + "." : m.label + " is not ready yet. Open Options to see why.";
  }
}

function mindName(m) {
  return m.label.replace(/^Muse /, "");
}

function drawChip() {
  const m = current();
  [museChip, chatMuse].forEach((chip) => chip.replaceChildren());
  if (!m) return;
  const dot = el("span", "dot " + (m.ok ? "ok" : "off"));
  dot.setAttribute("aria-hidden", "true");
  museChip.append(dot, ico(m.local ? "pc" : "cloud", "sm"), el("b", null, mindName(m)), el("span", null, m.ok ? (m.local ? (room.sends ? "on this computer, Meta sees pieces" : "on this computer") : "API, with your key") : "not ready"));
  chatMuse.append(ico(m.local ? "pc" : "cloud", "sm"), mindName(m));
  [museChip, chatMuse].forEach((chip) => {
    chip.setAttribute("aria-label", "Muse: " + m.label + ", " + (m.local ? (room.sends ? "on this computer, and Meta's models may see and hear pieces" : "on this computer") : "through an API, which costs money") + ". Change");
    chip.title = m.label + ", " + m.where;
  });
  museWhere.textContent = m.ok ? m.where.charAt(0).toUpperCase() + m.where.slice(1) : "Not ready yet";
}

function drawMinds() {
  mindsBox.textContent = "";
  machineBox.textContent = "";
  drawChip();
  room.minds.forEach((m) => {
    const b = el("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(m.key === room.mind));
    if (!m.ok) b.setAttribute("aria-disabled", "true");
    b.append(ico(m.local ? "pc" : "cloud"), mindName(m), el("small", null, m.ok ? (m.local ? "This computer · free" : "API · costs money") : "Not ready"));
    b.title = m.ok ? m.where : m.why;
    b.addEventListener("click", () => {
      mindWhy.textContent = m.ok ? "" : m.why;
      mindWhy.hidden = m.ok;
      if (!m.ok) return;
      room.mind = m.key;
      localStore("mind", m.key);
      drawMinds();
    });
    mindsBox.appendChild(b);
    machineBox.appendChild(el("p", null, m.label + ": " + (m.ok ? m.name + " at " + m.url : m.why)));
  });
  const h = room.hearing;
  if (h) machineBox.appendChild(el("p", null, h.label + ": " + (h.ok ? h.name + " at " + h.url : h.why)));
  drawSeg($("stepsSeg"), [["simple", "Simple"], ["detailed", "Detailed"]], document.body.classList.contains("details") ? "detailed" : "simple", (k) => {
    localStore("details", k === "detailed" ? "1" : "0");
    showDetails(k === "detailed");
  });
  drawSeg($("themeSeg"), [["auto", "Auto", "auto"], ["light", "Light", "sun"], ["dark", "Dark", "moon"]], localStore("theme") || "auto", (k) => {
    localStore("theme", k);
    showTheme();
  });
  const mine = localCurrency();
  const choices = [["USD", "$ USD"]].concat(mine !== "USD" ? [[mine, mine]] : []);
  drawSeg($("moneySeg"), choices, myCurrency(), (k) => {
    localStore("money", k);
    refreshCost();
    loadHome();
  });
  showStatus();
}

function drawSeg(box, choices, now, act) {
  box.replaceChildren();
  choices.forEach(([key, label, icon]) => {
    const b = el("button");
    b.type = "button";
    if (icon) b.appendChild(ico(icon, "sm"));
    b.append(label);
    b.setAttribute("aria-pressed", String(key === now));
    b.addEventListener("click", () => {
      box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      act(key);
    });
    box.appendChild(b);
  });
}

function showDetails(on) {
  document.body.classList.toggle("details", on);
  Object.values(room.jobs).forEach((j) => setOpen(j, on || j.card.classList.contains("open")));
}

function showTheme() {
  const k = localStore("theme") || "auto";
  const dark = k === "dark" || (k === "auto" && darkScheme.matches);
  if (dark) document.documentElement.dataset.theme = "dark";
  else delete document.documentElement.dataset.theme;
}

darkScheme.addEventListener("change", showTheme);

function openSheet(part) {
  const fromClip = app.dataset.screen === "clip" || EDITS.includes(part) || part === "out";
  sheet.classList.toggle("dark", fromClip);
  sheet.classList.toggle("tall", part === "features" || part === "options");
  sheetBack.classList.toggle("clear", EDITS.includes(part));
  const [icon, title] = SHEETS[part];
  sheetIcon.firstElementChild.setAttribute("href", "#i-" + icon);
  sheetHead.textContent = title;
  sheet.hidden = false;
  sheetBack.classList.add("on");
  showPart(part);
  panelNote.textContent = "";
  if (part !== "words") closeWords();
  if (part === "cost") {
    if (room.spent) drawCost(room.spent, costBox, room.ready ? room.ready.job : undefined);
    refreshCost();
  } else if (part === "out") {
    drawOut();
  } else if (part === "options") {
    mindWhy.hidden = true;
    drawMinds();
    loadParts();
    loadKey();
    loadPhone();
  } else if (part === "features") {
    loadFeatures();
  }
  sheet.scrollTop = 0;
}

function closeSheet() {
  if (sheet.hidden) return;
  sheet.hidden = true;
  sheetBack.classList.remove("on");
  closeWords();
  showPart(room.part);
}

$("sheetClose").addEventListener("click", closeSheet);
sheetBack.addEventListener("click", closeSheet);

document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-sheet]");
  if (t) {
    if (!sheet.hidden && room.part === t.dataset.sheet) closeSheet();
    else openSheet(t.dataset.sheet);
    return;
  }
  const idea = e.target.closest("[data-idea]");
  if (idea) {
    text.value = idea.dataset.idea;
    text.focus();
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !sheet.hidden && document.activeElement !== wordText) closeSheet();
});

async function health() {
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    const d = await r.json();
    room.minds = d.minds || [];
    room.hearing = d.hearing || null;
    room.sends = d.sends === true;
    if (!room.minds.some((m) => m.key === room.mind && m.ok)) {
      const ready = room.minds.find((m) => m.key === d.default && m.ok) || room.minds.find((m) => m.ok);
      room.mind = ready ? ready.key : d.default;
    }
    document.body.classList.toggle("away", d.here === false);
    $("add").setAttribute("aria-label", d.here === false ? "Send photos and videos from this phone" : "Open a video");
    $("homeSetUp").hidden = !room.minds.length || room.minds.some((m) => m.ok);
    drawMinds();
    refreshCost();
  } catch (e) {
    statusLine.textContent = "The studio is not answering. Start it again with: python app.py";
  }
}

function listed(chatId) {
  return room.list.find((c) => c.id === chatId);
}

function show(chatId) {
  if (room.source) room.source.close();
  stopTimer();
  room.chat = chatId;
  room.last = -1;
  room.jobs = {};
  room.open = new Set();
  room.costs = [];
  room.cut = {};
  room.showing = null;
  room.minded = [];
  room.paid = false;
  room.work = { video: false, preview: false, ends: false, layout: false };
  const known = listed(chatId);
  room.title = known ? known.title || known.first : "";
  chatName.textContent = room.title || "New clip";
  chatSub.replaceChildren();
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
  refreshCost();
}

async function connect(chatId) {
  try {
    const r = await fetch("/api/chat" + (chatId ? "?id=" + encodeURIComponent(chatId) : ""), { cache: "no-store" });
    const d = await r.json();
    if (!d.ok) {
      toast(d.error);
      return false;
    }
    if (d.chat !== room.chat || !room.source || room.source.readyState === EventSource.CLOSED) show(d.chat);
    return true;
  } catch (e) {
    statusLine.textContent = "The studio is not answering. Start it again with: python app.py";
    setTimeout(connect, 3000);
    return false;
  }
}

async function loadKey() {
  if (window.lcsReplay) return;
  let d;
  try {
    d = await fetch("/api/key", { cache: "no-store" }).then((r) => r.json());
  } catch (e) {
    return;
  }
  apiForm.hidden = !d.here;
  apis = d.apis || [];
  const saved = apis.find((a) => a.url && a.url === d.url);
  const now = apis.some((a) => a.name === apiNow) ? apiNow : saved ? saved.name : apis.length ? apis[0].name : "";
  fillApi(now);
  drawSends(d);
}

function drawApis() {
  apiSeg.replaceChildren();
  apis.forEach((a) => {
    const b = el("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(a.name === apiNow));
    b.append(ico(a.icon), el("span", null, a.label), el("small", null, a.saved ? "Key saved" : a.sub));
    b.addEventListener("click", () => fillApi(a.name));
    apiSeg.appendChild(b);
  });
}

function fillApi(name) {
  const a = apis.find((x) => x.name === name);
  if (!a) return;
  apiNow = a.name;
  drawApis();
  const mine = a.url && !(a.own && a.url === a.address) ? a.url : "";
  apiUrl.placeholder = a.address;
  apiUrl.value = mine;
  apiUrl.hidden = a.own && !mine;
  apiMore.hidden = !apiUrl.hidden;
  apiKey.value = "";
  apiKey.placeholder = a.saved ? a.key + " (saved)" : a.key;
  apiGet.href = a.page;
  apiNote.classList.remove("bad");
  apiNote.textContent = "";
}

apiMore.addEventListener("click", () => {
  apiUrl.hidden = false;
  apiMore.hidden = true;
  apiUrl.focus();
});

function drawSends(d) {
  sendsBox.hidden = !d.here;
  if (!d.here) return;
  room.sends = d.sends === true;
  drawSeg($("sendsSeg"), [["off", "Off"], ["on", "On", "eye"]], room.sends ? "on" : "off", async (k) => {
    let got;
    try {
      got = await post("/api/sends", { on: k === "on" });
    } catch (e) {
      got = { ok: false, error: "The studio is not answering. Start it again with: python app.py" };
    }
    if (!got.ok) {
      sendsLine.classList.add("bad");
      sendsLine.textContent = got.error;
      return;
    }
    drawSends({ here: true, sends: got.sends });
    drawChip();
    health();
  });
  sendsLine.classList.remove("bad");
  sendsLine.textContent = room.sends ? "On: with Muse on this computer, Muse Wonder may watch pieces of up to a minute with their sound and Muse Image may draw a cover from a frame, and Muse Voice Transcribe hears what is said: on Meta's servers, with your key." : "Off: with Muse on this computer, nothing of yours goes to Meta unless you ask for it in your message.";
}

apiForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  let d;
  try {
    d = await post("/api/key", { api: apiNow, url: apiUrl.value, key: apiKey.value });
  } catch (err) {
    d = { ok: false, error: "The studio is not answering. Start it again with: python app.py" };
  }
  const ok = d.ok && d.mind.ok;
  apiNote.classList.toggle("bad", !ok);
  apiNote.textContent = !d.ok ? d.error : ok ? d.mind.label + " is ready." : d.mind.why;
  if (!d.ok) return;
  const a = apis.find((x) => x.name === apiNow);
  if (a) {
    a.saved = true;
    a.url = d.url;
    drawApis();
  }
  apiKey.value = "";
  apiKey.placeholder = (a ? a.key : "API key") + " (saved)";
  health();
});

async function loadPhone() {
  if (window.lcsReplay || sheet.hidden || room.part !== "options") return;
  let d;
  try {
    d = await fetch("/api/phone", { cache: "no-store" }).then((r) => r.json());
  } catch (e) {
    return;
  }
  drawPhone(d);
}

function drawPhone(d) {
  phoneBox.hidden = !d.ok;
  if (!d.ok) return;
  drawSeg($("phoneSeg"), [["off", "Off"], ["on", "On", "phone"]], d.on ? "on" : "off", async (k) => {
    phoneLine.textContent = k === "on" ? "Turning Phone on…" : "";
    let got;
    try {
      got = await post("/api/phone", { on: k === "on" });
    } catch (e) {
      got = { ok: false, error: "The studio is not answering. Start it again with: python app.py" };
    }
    if (!got.ok) {
      phoneLine.classList.add("bad");
      phoneLine.textContent = got.error;
      return;
    }
    drawPhone(got);
    loadPhone();
  });
  $("phoneSeg").querySelectorAll("button").forEach((b) => { b.disabled = d.fixed; });
  phoneLine.classList.toggle("bad", !!d.why);
  phoneLine.textContent = d.why || (d.on ? "A phone on the same Wi-Fi or Tailscale opens the studio at an address below: scan its code with the phone's camera. Windows may ask once whether Python may use the network: choose Allow." : "Off: only this computer can use the studio.");
  const open = d.on ? d.addresses.filter((a) => a.listening) : [];
  const shown = open.map((a) => a.address).join(" ");
  if (phoneCodes.dataset.shown !== shown) {
    phoneCodes.dataset.shown = shown;
    phoneCodes.replaceChildren(...open.map((a) => {
      const fig = el("figure", "phone-code");
      const img = el("img");
      img.src = "/api/phone/qr.svg?address=" + encodeURIComponent(a.address);
      img.alt = "Code that opens " + a.url;
      const cap = el("figcaption");
      cap.append(el("b", null, a.kind === "Wi-Fi" ? "Same Wi-Fi" : a.kind), el("span", null, a.url));
      fig.append(img, cap);
      return fig;
    }));
  }
}

function stamp(t) {
  const two = (n) => String(n).padStart(2, "0");
  return t.getFullYear() + "-" + two(t.getMonth() + 1) + "-" + two(t.getDate()) + " " + two(t.getHours()) + "." + two(t.getMinutes()) + "." + two(t.getSeconds());
}

function upload(file, batch, progress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", "/api/upload?name=" + encodeURIComponent(file.name) + "&batch=" + encodeURIComponent(batch) + "&when=" + Math.round(file.lastModified / 1000));
    x.setRequestHeader("Content-Type", "application/octet-stream");
    x.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) progress(e.loaded / e.total);
    });
    x.addEventListener("load", () => {
      try {
        resolve(JSON.parse(x.responseText));
      } catch (e) {
        reject(e);
      }
    });
    x.addEventListener("error", () => reject(new Error("network")));
    x.send(file);
  });
}

async function sendFiles(files) {
  const list = [...files];
  if (!list.length) return;
  if (app.dataset.screen === "home") setScreen("chat");
  const batch = stamp(new Date());
  const line = bubble("note sent");
  const words = el("span");
  line.append(ico("phone", "sm"), words);
  const sent = [];
  const failed = [];
  for (let i = 0; i < list.length; i++) {
    const f = list[i];
    const head = "Sending " + (list.length > 1 ? (i + 1) + " of " + list.length + ", " : "") + f.name;
    words.textContent = head;
    try {
      const d = await upload(f, batch, (part) => {
        words.textContent = head + " · " + Math.round(part * 100) + "%";
      });
      if (d.ok) sent.push(Object.assign({ name: f.name, size: f.size }, d));
      else failed.push(d.error);
    } catch (e) {
      failed.push(f.name + " did not reach the computer. Keep this page open while it sends, then send it again.");
    }
  }
  const videos = sent.filter((d) => d.kind === "video");
  const photos = sent.length - videos.length;
  const said = [sent.length + " of " + list.length + " sent to " + (sent.length ? sent[0].folder : "the computer") + "."];
  if (photos) said.push((photos === 1 ? "1 photo is" : photos + " photos are") + " kept there; Muse makes clips from videos.");
  words.textContent = said.join(" ");
  failed.forEach((why) => note(why));
  if (videos.length === 1) attach({ path: videos[0].path }, videos[0].name, sizeText(videos[0].size));
  else if (videos.length > 1) attach({ folder: videos[0].folder }, videos.length + " videos", "from your phone");
}

function attach(body, name, meta) {
  const off = el("button", "off");
  off.type = "button";
  off.setAttribute("aria-label", "Take " + name + " off");
  off.append(ico("x", "sm"));
  off.addEventListener("click", detach);
  room.attached = { body: body, name: name };
  attached.replaceChildren(ico("film", "sm"), el("span", "name", name), el("span", "meta", meta || ""), off);
  attached.hidden = false;
  setBusy();
  text.focus();
}

function detach() {
  room.attached = null;
  attached.hidden = true;
  attached.replaceChildren();
  setBusy();
}

function sizeText(bytes) {
  return bytes < 1e9 ? Math.round(bytes / 1e6) + " MB" : (bytes / 1e9).toFixed(1) + " GB";
}

async function loadParts() {
  if (window.lcsReplay) return;
  let d;
  try {
    d = await fetch("/api/parts", { cache: "no-store" }).then((r) => r.json());
  } catch (e) {
    return;
  }
  room.parts = d;
  drawParts();
  clearTimeout(room.partsTimer);
  if (d.setup.running) {
    room.setting = true;
    room.partsTimer = setTimeout(loadParts, 1000);
  } else if (room.setting) {
    room.setting = false;
    health();
  }
}

function drawParts() {
  const d = room.parts;
  const s = d.setup;
  partsBox.hidden = d.parts.every((p) => p.ok) && !s.running && !s.error && !d.engine;
  $("cardName").textContent = d.card ? d.card.name.replace(/^NVIDIA (GeForce )?/, "") + " · " + d.card.gb + " GB" : "";
  const said = new Set();
  $("parts").replaceChildren(...d.parts.map((p) => {
    const row = el("div", "part");
    row.append(ico(PART_ICONS[p.key]), el("span", null, p.label), p.ok ? ico("check", "sm ok") : p.size ? el("small", null, sizeText(p.size)) : ico("warn", "sm bad"));
    if (!p.ok && p.why && !said.has(p.why)) row.appendChild(el("p", "why", p.why));
    said.add(p.why);
    row.title = p.what;
    return row;
  }));
  const bar = $("setupBar");
  bar.hidden = !s.running;
  bar.firstElementChild.style.width = s.total ? Math.min(100, (100 * s.done) / s.total) + "%" : "0";
  setupLine.classList.toggle("bad", !s.running && !!(s.error || d.engine));
  if (s.running) {
    const part = d.parts.find((p) => p.key === s.part);
    const doing = { check: "Checking", unpack: "Unpacking", copy: "Copying" }[s.step] || "Downloading";
    setupLine.textContent = doing + " " + (part ? part.label : "") + " · " + sizeText(s.done) + " of " + sizeText(s.total);
  } else {
    setupLine.textContent = s.error || d.engine || "";
  }
  setUpButton.hidden = s.running || !d.needed;
  setUpButton.replaceChildren(ico("save", "sm"), "Set up · " + sizeText(d.needed));
}

setUpButton.addEventListener("click", async () => {
  setUpButton.disabled = true;
  let d;
  try {
    d = await post("/api/parts", {});
  } catch (e) {
    d = { ok: false, error: "The studio is not answering. Start it again with: python app.py" };
  }
  setUpButton.disabled = false;
  if (d.ok) return loadParts();
  setupLine.classList.add("bad");
  setupLine.textContent = d.error;
});

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}

async function say(words) {
  try {
    const d = await post("/api/say", { chat: room.chat, message: words, mind: room.mind });
    if (d.ok) return true;
    note(d.error || "Something went wrong.");
  } catch (err) {
    note("The studio is not answering. Start it again with: python app.py");
  }
  return false;
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const words = text.value.trim();
  if (room.attached) {
    openRecording(room.attached.body);
    return;
  }
  if (!words || !room.chat) return;
  text.value = "";
  fitText();
  if (!(await say(words))) text.value = words;
  text.focus();
});

stopButton.addEventListener("click", async () => {
  stopButton.disabled = true;
  try {
    await post("/api/stop", { chat: room.chat });
  } catch (err) {
    note("The studio is not answering. Start it again with: python app.py");
    stopButton.disabled = false;
  }
});

newButton.addEventListener("click", async () => {
  try {
    const d = await post("/api/chat/new", {});
    if (d.ok) show(d.chat);
  } catch (err) {
    toast("The studio is not answering. Start it again with: python app.py");
    return;
  }
  setScreen("chat");
  text.focus();
});

async function openChat(c) {
  if (c.id !== room.chat && !(await connect(c.id))) return;
  room.title = c.title || c.first;
  chatName.textContent = room.title;
  setScreen(c.clips && c.clips.length ? "clip" : "chat");
}

async function loadHome() {
  let d, s;
  try {
    [d, s] = await Promise.all([
      fetch("/api/chats", { cache: "no-store" }).then((r) => r.json()),
      fetch("/api/spent?currency=" + myCurrency(), { cache: "no-store" }).then((r) => r.json()),
    ]);
  } catch (err) {
    gridEmpty.textContent = "The studio is not answering. Start it again with: python app.py";
    gridEmpty.hidden = false;
    return;
  }
  room.list = d.chats || [];
  if (s && s.rate) {
    room.rate = s.rate;
    const month = new Date().toLocaleString("en", { month: "short" });
    monthChip.replaceChildren(ico("cash", "sm"), moneyShort(s.month), el("small", null, month));
    monthChip.setAttribute("aria-label", "Spent this month: " + moneyShort(s.month) + ". Show what Muse cost");
    monthChip.hidden = false;
  }
  grid.replaceChildren(...room.list.map(tile));
  gridEmpty.textContent = "No clips yet. Tap + to make your first one.";
  gridEmpty.hidden = room.list.length > 0;
  const known = listed(room.chat);
  if (known && !room.title) {
    room.title = known.title || known.first;
    chatName.textContent = room.title;
  }
}

function tile(c) {
  const clips = c.clips || [];
  const b = el("button", "tile" + (c.id === room.chat ? " here" : ""));
  b.type = "button";
  const shot = el("span", "shot");
  const last = clips[clips.length - 1];
  if (last) {
    const img = el("img");
    img.src = "/api/clip/" + last.file.replace(/\.mp4$/, ".jpg");
    img.alt = "";
    img.loading = "lazy";
    shot.appendChild(img);
  } else {
    const none = el("span", "none");
    none.appendChild(ico("bubble"));
    shot.appendChild(none);
  }
  const minds = el("span", "badge minds");
  minds.append(...mindIcons(c.minds));
  if (minds.children.length) shot.appendChild(minds);
  const usd = c.usd === undefined ? null : c.usd;
  shot.appendChild(el("span", "badge cost", moneyShort(usd)));
  if (clips.length > 1) {
    const many = el("span", "badge many");
    many.append(ico("stack", "sm"), String(clips.length));
    shot.appendChild(many);
  }
  if (c.id === room.chat && room.open.size) shot.appendChild(el("span", "badge busy", "Working"));
  const name = c.title || c.first || c.recording || "A chat";
  b.append(shot, el("span", "tile-name", name));
  b.setAttribute("aria-label", name + ", " + (clips.length || "no") + (clips.length === 1 ? " clip" : " clips") + ", " + ((c.minds || []).map((m) => mindName(m) + (m.local ? " on this computer" : " through an API")).join(" and ") || "no Muse yet") + ", " + (usd === null ? "cost not known" : moneyShort(usd)));
  b.addEventListener("click", () => openChat(c));
  return b;
}

const features = { loaded: false };

async function loadFeatures() {
  if (features.loaded) return;
  featuresList.replaceChildren(el("p", "earlier-empty", "Looking…"));
  try {
    const r = await fetch("/api/features", { cache: "no-store" });
    const d = await r.json();
    const tabs = el("div", "feature-tabs");
    tabs.setAttribute("role", "group");
    tabs.setAttribute("aria-label", "Kinds of work");
    const gridBox = el("div", "feature-grid");
    const pick = el("div", "feature-pick");
    pick.hidden = true;
    const says = el("p");
    const tryIt = el("button", "pill go");
    tryIt.type = "button";
    tryIt.append(ico("send", "sm"), "Try");
    pick.append(says, tryIt);
    const choose = (g, chip) => {
      tabs.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c === chip)));
      pick.hidden = true;
      gridBox.replaceChildren(...g.features.map((f) => {
        const item = el("button", "feature");
        item.type = "button";
        item.setAttribute("aria-pressed", "false");
        item.append(mini(f.show, f.name + ": " + f.says), el("span", null, f.name));
        item.addEventListener("click", () => {
          gridBox.querySelectorAll(".feature").forEach((x) => x.setAttribute("aria-pressed", String(x === item)));
          says.textContent = f.says;
          tryIt.onclick = () => {
            closeSheet();
            if (app.dataset.screen !== "chat") setScreen("chat");
            text.value = f.ask;
            fitText();
            text.focus();
          };
          pick.hidden = false;
        });
        return item;
      }));
    };
    d.groups.forEach((g) => {
      const chip = el("button", "chip", g.name);
      chip.type = "button";
      chip.addEventListener("click", () => choose(g, chip));
      tabs.appendChild(chip);
    });
    featuresList.replaceChildren(tabs, gridBox, pick);
    if (d.groups.length) choose(d.groups[0], tabs.firstChild);
    features.loaded = true;
  } catch (err) {
    featuresList.replaceChildren(el("p", "earlier-empty", "The studio is not answering."));
  }
}

const MINI = {
  find: { parts: "rows fx", fx: "🔎" },
  open: { parts: "scr fx", fx: "▶️" },
  recipe: { parts: "rows fx", fx: "🏷️" },
  map: { parts: "wave fx", fx: "📈" },
  watch: { parts: "scr fx", fx: "👀" },
  cover: { parts: "who ttl", ttl: "Cover" },
  moments: { parts: "wave fx", fx: "⭐" },
  plan: { parts: "rows" },
  many: { parts: "reel" },
  folder: { parts: "reel fx", fx: "📁" },
  story: { parts: "reel" },
  countdown: { parts: "reel fx", fx: "🏆" },
  card: { parts: "scr ring cap" },
  stack: { parts: "ring scr" },
  face: { parts: "who cap" },
  subject: { parts: "box fx", fx: "👟" },
  move: { parts: "scr ring" },
  shot: { parts: "two who" },
  "cap-plain": { parts: "who cap" },
  "cap-lit": { parts: "who cap" },
  "cap-pop": { parts: "who cap" },
  "style-clean": { parts: "who ttl" },
  "style-pop": { parts: "who ttl" },
  "style-sticker": { parts: "who ttl" },
  "style-glow": { parts: "who ttl" },
  "style-3d": { parts: "who ttl" },
  "style-classic": { parts: "who ttl" },
  "style-font": { parts: "who ttl cap" },
  title: { parts: "who ttl" },
  fonts: { parts: "ttl", ttl: "Aa กข" },
  untitle: { parts: "who ttl" },
  flash: { parts: "who" },
  shake: { parts: "who" },
  punch: { parts: "who" },
  word: { parts: "who fx", fx: "WOW" },
  push: { parts: "who" },
  sticker: { parts: "who fx", fx: "🔥" },
  glass: { parts: "who box ttl", ttl: "฿990" },
  "look-vivid": { parts: "who" },
  "look-warm": { parts: "who" },
  "look-film": { parts: "who" },
  behind: { parts: "ttl who", ttl: "WOW" },
  "cut-bw": { parts: "who" },
  "cut-outline": { parts: "who" },
  sound: { parts: "who wave fx", fx: "💥" },
  music: { parts: "who wave fx", fx: "🎵" },
  mute: { parts: "who wave fx", fx: "🔇" },
  trim: { parts: "reel fx", fx: "✂️" },
  "blur-box": { parts: "rows box" },
  people: { parts: "two who fx", fx: "2" },
  "blur-faces": { parts: "two who" },
  channel: { parts: "who tag" },
  "channel-switch": { parts: "who tag", tag: "@sineysociety" },
  render: { parts: "scr bar" },
  check: { parts: "rows" },
  post: { parts: "tag fx", tag: "#reels #game", fx: "📋" },
  undo: { parts: "who ttl fx", fx: "↶" },
};

const PANEL_MINI = { "cap-off": { parts: "who" } };

function mini(show, said) {
  const m = MINI[show] || PANEL_MINI[show] || { parts: "fx", fx: "✨" };
  const box = el("div", "mini");
  box.dataset.show = show;
  box.setAttribute("role", "img");
  box.setAttribute("aria-label", said);
  box.append(el("i", "sky"));
  if (show.startsWith("flurry-")) {
    const v = el("video", "move");
    v.src = "/api/move/" + show.slice(7) + ".mp4";
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    v.autoplay = !still;
    v.preload = still ? "auto" : "metadata";
    if (still) v.addEventListener("loadedmetadata", () => { v.currentTime = v.duration / 2; }, { once: true });
    box.append(v);
    return box;
  }
  m.parts.split(" ").forEach((p) => {
    if (p === "cap") {
      const c = el("b", "cap");
      ["we", "did", "it"].forEach((w, k) => c.append(k ? " " : "", el("span", null, w)));
      box.append(c);
    } else if (p === "rows" || p === "wave" || p === "reel") {
      const r = el("i", p);
      for (let k = 0; k < (p === "wave" ? 9 : 3); k += 1) r.append(el("i"));
      box.append(r);
    } else {
      const words = { ttl: m.ttl || "Boss down", fx: m.fx, tag: m.tag || "@yourchannel" }[p];
      box.append(el(words === undefined ? "i" : "b", p, words));
    }
  });
  return box;
}

async function fillNewest() {
  const box = log.querySelector(".welcome .newest");
  if (!box) return;
  try {
    const r = await fetch("/api/recordings", { cache: "no-store" });
    const d = await r.json();
    box.querySelectorAll(".recording:not(.pick):not(.from-phone), .earlier-empty").forEach((n) => n.remove());
    if (!d.recordings.length) box.appendChild(el("p", "earlier-empty", "No videos found in " + (d.folders.join(", ") || "any folder") + "."));
    d.recordings.forEach((rec) => {
      const b = el("button", "recording");
      b.type = "button";
      b.append(ico("film"), el("span", "first", rec.name), el("span", "meta", [rec.length, rec.size, rec.when].filter(Boolean).join(" · ")));
      b.addEventListener("click", () => attach({ path: rec.path }, rec.name, rec.length));
      box.appendChild(b);
    });
  } catch (err) {
    box.appendChild(el("p", "earlier-empty", "The studio is not answering."));
  }
}

async function openRecording(body) {
  const words = text.value.trim();
  try {
    const d = await post("/api/open", Object.assign({ chat: room.chat, words: words, mind: room.mind, language: navigator.language || "" }, body));
    if (d.ok) {
      text.value = "";
      fitText();
      detach();
      return;
    }
    note(d.error || "Something went wrong.");
  } catch (err) {
    note("The studio is not answering. Start it again with: python app.py");
  }
}

function openFile(file) {
  if (file) attach({ name: file.name, size: file.size, mtime: file.lastModified / 1000 }, file.name, sizeText(file.size));
}

$("add").addEventListener("click", () => {
  if (document.body.classList.contains("away")) phoneFiles.click();
  else fileInput.click();
});

log.addEventListener("click", (e) => {
  if (e.target.closest(".pick")) fileInput.click();
  if (e.target.closest(".from-phone")) phoneFiles.click();
});

phoneFiles.addEventListener("change", () => {
  const files = [...phoneFiles.files];
  phoneFiles.value = "";
  sendFiles(files);
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
  if (app.dataset.screen === "home") setScreen("chat");
  openFile(e.dataTransfer.files[0]);
});

function fitText() {
  text.style.height = "auto";
  text.style.height = Math.min(text.scrollHeight + 2, window.innerHeight * 0.4) + "px";
}

text.addEventListener("input", fitText);

text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

showDetails(localStore("details") === "1");
showTheme();
history.replaceState({ screen: window.lcsReplay ? "chat" : "home" }, "");
setScreen(window.lcsReplay ? "chat" : "home", false);
health();
connect();
