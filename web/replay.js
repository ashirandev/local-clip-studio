(() => {
  const NOTE = "This is a replay of a real run, so nothing runs here. Get the studio to make clips from your own videos.";
  const realFetch = window.fetch.bind(window);
  const data = realFetch("api/replay.json", { cache: "no-store" }).then((r) => r.json());
  let skip = false;
  let playing = null;

  const answer = (body) => new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });

  window.fetch = async (url, opts) => {
    const at = new URL(String(url), location.href).pathname.split("/api/");
    if (at.length < 2 || at[1] === "replay.json") return realFetch(url, opts);
    const d = await data;
    if (opts && opts.method === "POST") return answer({ ok: false, error: NOTE });
    if (at[1] === "health") return answer(d.health);
    if (at[1] === "chat") return answer({ ok: true, chat: d.chat, busy: false });
    if (at[1] === "chats") return answer({ ok: true, chats: [d.listed] });
    if (at[1] === "recordings") return answer({ ok: true, folders: [], recordings: [] });
    if (at[1] === "panel") return answer(d.panel);
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
      playing = this;
      data.then((d) => this.play(d.events));
    }

    addEventListener() {}

    close() {
      this.readyState = Replay.CLOSED;
      clearTimeout(this.timer);
    }

    play(events) {
      clearTimeout(this.timer);
      while (this.readyState !== Replay.CLOSED && this.next < events.length) {
        const e = Object.assign({}, events[this.next], { t: Date.now() / 1000 });
        this.next += 1;
        if (this.onmessage) this.onmessage({ data: JSON.stringify(e) });
        if (this.next < events.length && !skip) {
          const gap = Math.max(0, events[this.next].t - events[this.next - 1].t);
          this.timer = setTimeout(() => this.play(events), 1000 * Math.min(gap, 0.25 + gap / 8, 2.5));
          return;
        }
      }
      if (this.next >= events.length) bar.classList.add("done");
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
  const again = document.createElement("button");
  again.type = "button";
  again.className = "ghost";
  again.textContent = "Play again";
  again.addEventListener("click", () => location.reload());
  const end = document.createElement("button");
  end.type = "button";
  end.className = "ghost end";
  end.textContent = "Show the end";
  end.addEventListener("click", () => {
    skip = true;
    if (playing) data.then((d) => playing.play(d.events));
  });
  bar.append(line, again, end);
  document.querySelector(".chat").prepend(bar);

  data.then((d) => {
    const minutes = Math.floor(d.took / 60);
    const took = (minutes ? minutes + " min " : "") + (d.took % 60) + " s";
    line.textContent = "A replay of a real run made on " + d.made + ". Muse took " + took + (d.card ? " on one " + d.card : "") + "; it plays here faster, and nothing runs.";
  });
})();
