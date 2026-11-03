(() => {
  window.lcsReplay = true;
  const NOTE ="This is a replay of a real run, so nothing runs here. Get the studio to make clips from your own videos.";
  const realFetch = window.fetch.bind(window);
  const data = realFetch("api/replay.json", { cache: "no-store" }).then((r) => r.json());
  let skip = false;
  let playing = null;
  let want = null;
  let paused = false;
  let startAt = null;
  let seen = 0;
  let total = 0;

  const answer = (body) => new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });

  window.fetch = async (url, opts) => {
    const at = new URL(String(url), location.href).pathname.split("/api/");
    if (at.length < 2 || at[1] === "replay.json") return realFetch(url, opts);
    const d = await data;
    if (at[1] === "base" && d.base && d.base.url) return answer(d.base);
    if (opts && opts.method === "POST") return answer({ ok: false, error: NOTE });
    if (at[1] === "layer") return answer(d.layer && d.layer.ok ? d.layer : { ok: false, error: NOTE });
    if (at[1] === "health") return answer(d.health);
    if (at[1] === "chat") return answer({ ok: true, chat: d.chat, busy: false });
    if (at[1] === "chats") return answer({ ok: true, chats: [d.listed] });
    if (at[1] === "recordings") return answer({ ok: true, folders: [], recordings: [] });
    if (at[1] === "panel") return answer(d.panel);
    if (at[1] === "features") return answer({ ok: true, groups: d.features || [] });
    if (at[1] === "faces") return answer({ ok: true, faces: [], size: [1080, 1920] });
    return realFetch(url, opts);
  };

  class Replay {
    constructor() {
      this.readyState = Replay.OPEN;
      this.onmessage = null;
      this.onerror = null;
      this.timer = 0;
      this.next = 0;
      seen = 0;
      if (startAt !== null) {
        want = startAt;
        startAt = null;
      }
      playing = this;
      data.then((d) => {
        this.events = d.events;
        this.play(d.events);
      });
    }

    addEventListener() {}

    close() {
      this.readyState = Replay.CLOSED;
      clearTimeout(this.timer);
    }

    play(events) {
      clearTimeout(this.timer);
      total = events.filter((e) => e.kind === "step").length;
      while (this.readyState !== Replay.CLOSED && this.next < events.length) {
        if (want !== null && seen >= want && events[this.next].kind === "step") {
          want = null;
          return count();
        }
        if (paused && want === null && !skip) return count();
        const e = Object.assign({}, events[this.next], { t: Date.now() / 1000 });
        this.next += 1;
        if (e.kind === "step") seen += 1;
        if (this.onmessage) this.onmessage({ data: JSON.stringify(e) });
        if (this.next < events.length && !skip && want === null) {
          count();
          const gap = Math.max(0, events[this.next].t - events[this.next - 1].t);
          this.timer = setTimeout(() => this.play(events), 1000 * Math.min(gap, 0.25 + gap / 8, 2.5));
          return;
        }
      }
      want = null;
      count();
    }
  }
  Replay.CONNECTING = 0;
  Replay.OPEN = 1;
  Replay.CLOSED = 2;
  window.EventSource = Replay;

  const style = document.createElement("style");
  style.textContent = ".replay{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;margin:0;padding:10px 16px;background:var(--you);color:var(--text);border-bottom:1px solid var(--line);font-size:14px;line-height:1.4}.replay span{flex:1 1 260px}.replay.done .end{display:none}.composer,.status{display:none!important}";
  document.head.append(style);
  const bar = document.createElement("p");
  bar.className = "replay";
  bar.setAttribute("role", "note");
  const line = document.createElement("span");
  line.textContent = "A replay of a real run.";
  const pill = (words, cls, label) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "pill " + cls;
    b.textContent = words;
    if (label) b.setAttribute("aria-label", label);
    return b;
  };
  const where = document.createElement("b");
  where.className = "replay-step";
  const pause = pill("Pause", "pause");
  const back = pill("Back", "back", "One step back");
  const forth = pill("Next", "next", "The next step");
  const count = () => {
    const ended = Boolean(playing && playing.next >= ((playing.events || []).length || Infinity));
    where.textContent = "Step " + seen + " of " + total;
    pause.textContent = paused || ended ? "Play" : "Pause";
    back.disabled = seen === 0;
    forth.disabled = ended;
    bar.classList.toggle("done", ended);
  };
  const run = () => data.then((d) => {
    if (playing) {
      playing.events = d.events;
      playing.play(d.events);
    }
  });
  const toNext = () => {
    paused = true;
    want = seen + 1;
    run();
  };
  const toBack = () => {
    paused = true;
    startAt = Math.max(0, seen - 1);
    if (typeof show === "function" && typeof room === "object") show(room.chat);
  };
  pause.addEventListener("click", () => {
    if (pause.textContent === "Play" && playing && playing.events && playing.next >= playing.events.length) return location.reload();
    paused = !paused;
    want = null;
    if (!paused) run();
    count();
  });
  back.addEventListener("click", toBack);
  forth.addEventListener("click", toNext);
  document.addEventListener("keydown", (e) => {
    if (e.target && e.target.closest && e.target.closest("input, textarea, select, [contenteditable]")) return;
    if (e.key === "ArrowRight") toNext();
    else if (e.key === "ArrowLeft") toBack();
    else return;
    e.preventDefault();
    e.stopImmediatePropagation();
  }, true);
  const again = document.createElement("button");
  again.type = "button";
  again.className = "pill";
  again.textContent = "Play again";
  again.addEventListener("click", () => location.reload());
  const end = document.createElement("button");
  end.type = "button";
  end.className = "pill end";
  end.textContent = "Show the end";
  end.addEventListener("click", () => {
    skip = true;
    want = null;
    paused = false;
    run();
  });
  bar.append(line, where, back, pause, forth, again, end);
  document.querySelector(".chat").prepend(bar);

  data.then((d) => {
    const minutes = Math.floor(d.took / 60);
    const took = (minutes ? minutes + " min " : "") + (d.took % 60) + " s";
    const api = d.api && d.api.length ? (d.card ? " and" : "") + " with " + d.api.join(" and ") + " through an API" : "";
    line.textContent = "A replay of a real run made on " + d.made + ". Muse took " + took + (d.card ? " on one " + d.card : "") + api + "; it plays here faster, and nothing runs.";
  });
})();
