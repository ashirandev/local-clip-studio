"""One real run as a folder of plain files that plays in any browser, with no studio, card or key behind it: the page
as it is, the chat's events in their order at a quicker pace, and every picture and video they name. The files are
fetched from the studio's own server, so each one is what the page was sent; the folder each recording sat in is
left out, and the page says on every view that it is a replay. A tap that would change the clip answers that
nothing runs here.

python -m clipstudio.replay <chat> <folder>, then serve the folder (python -m http.server -d <folder>) or put it on
any static host. It is made on the computer the run was made on, which it names."""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from . import __version__, agent, recording, server

NAMED = re.compile(r"^[0-9a-f]{16}\.(?:jpg|mp4)$")
FOLDER = re.compile(r'(?<![\w\\])(?:[A-Za-z]:\\{1,2}|\\{2,4})(?:[^\\\r\n"<>|:*?]+\\{1,2})+')
API = '"/api/'
BASE_SECONDS = 180
SHIM = '<script src="replay.js"></script>\n<script src="app.js"></script>'


class Refused(Exception):
    pass


class _Quiet(server.Handler):
    """The studio's own handler, without a line for each file the pack looks for in the wrong folder first."""

    def log_request(self, code="-", size="-"):
        pass


def _named(value, out):
    """Every file name such as 0123456789abcdef.jpg anywhere in value, and every look named as a value of "look"."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "look" and isinstance(v, str):
                out.add(v + ".png")
            _named(v, out)
    elif isinstance(value, list):
        for v in value:
            _named(v, out)
    elif isinstance(value, str) and NAMED.match(value):
        out.add(value)
    return out


def _unfoldered(value):
    """value with the folders taken off every Windows path in it, so F:\\OBS\\a.mkv reads a.mkv."""
    if isinstance(value, dict):
        return {k: _unfoldered(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unfoldered(v) for v in value]
    if isinstance(value, str):
        return FOLDER.sub("", value)
    return value


def _layer_and_base(get, chat_id):
    """The clip's words layer and its base video as the studio answers the page for them ({} for a studio or a chat
    without them), waiting while the base video is made."""
    layer = json.loads(get("/api/layer?chat=" + chat_id) or b"{}")
    until = time.time() + BASE_SECONDS
    while True:
        made = json.loads(get("/api/base", {"chat": chat_id}) or b"{}")
        if not made.get("making") or made.get("url") or time.time() > until:
            return (layer if layer.get("ok") else {}), (made if made.get("url") else {})
        time.sleep(1)


def _api_paths(value, out):
    """Every address under /api/ the studio wrote into value, such as /api/layer/<name>.png."""
    if isinstance(value, dict):
        for v in value.values():
            _api_paths(v, out)
    elif isinstance(value, list):
        for v in value:
            _api_paths(v, out)
    elif isinstance(value, str) and value.startswith("/api/"):
        out.add(value)
    return out


def _relative(value):
    """value with every /api/ address made relative, so the replay plays from any folder of a web host."""
    if isinstance(value, dict):
        return {k: _relative(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_relative(v) for v in value]
    return value[1:] if isinstance(value, str) and value.startswith("/api/") else value


def _card():
    """The graphics card of this computer, as nvidia-smi names it, or "" without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return ""
    if not out:
        return ""
    name, _, mib = out[0].partition(",")
    name = re.sub(r"^NVIDIA (GeForce )?", "", name.strip())
    return "%s %d GB" % (name, round(int(mib) / 1024)) if mib.strip().isdigit() else name


def pack(chat_id, folder, card=None):
    """Writes the replay of chat_id into folder, which must be empty or new. Returns what it wrote: {"files",
    "bytes", "events", "took"}. Refused when the chat is not there or a picture or video it names is gone, since a
    replay with holes in it would show a run that did not happen that way."""
    events_path = os.path.join(recording.WORK, "chats", "%s.events.jsonl" % chat_id)
    if not agent.CHAT_ID.match(chat_id or "") or not os.path.isfile(events_path):
        raise Refused("There is no chat %s." % chat_id)
    if os.path.isdir(folder) and os.listdir(folder):
        raise Refused("%s has files in it already. Name a new folder." % folder)
    events = agent._read(events_path)
    if {e["job"] for e in events if e["kind"] == "job"} - {e["job"] for e in events if e["kind"] == "end"}:
        raise Refused("Muse has not finished in chat %s. Pack it once the job has ended." % chat_id)
    http = ThreadingHTTPServer(("127.0.0.1", 0), _Quiet)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % http.server_port

    def get(path, body=None):
        req = urllib.request.Request(base + path, None if body is None else json.dumps(body).encode("utf-8"),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError:
            return None

    try:
        if json.loads(get("/api/chat?id=" + chat_id) or b"{}").get("chat") != chat_id:
            raise Refused("The studio could not open chat %s." % chat_id)
        panel = json.loads(get("/api/panel?chat=" + chat_id) or b'{"ok": false}')
        health = json.loads(get("/api/health"))
        layer, made = _layer_and_base(get, chat_id)
        files, gone = {}, []
        for path in sorted(_api_paths([layer, made], set())):
            body = get(path)
            if body is None:
                gone.append(path)
            else:
                files[path[1:]] = body
        cards = {c["name"] + ".png" for c in panel.get("styles") or []}
        for name in sorted(_named([events, panel], cards)):
            if name.endswith(".png"):
                body = get("/api/look/" + name)
                if body is not None:
                    files["api/look/" + name] = body
                continue
            tries = (["api/frame/", "api/cover/"] if name.endswith(".jpg") else ["api/preview/", "api/clip/"])
            for where in tries:
                body = get("/" + where + name)
                if body is not None:
                    files[where + name] = body
                    if where == "api/clip/":
                        poster = get("/api/clip/" + name[:-4] + ".jpg")
                        if poster is not None:
                            files["api/clip/" + name[:-4] + ".jpg"] = poster
                    break
            else:
                gone.append(name)
        if gone:
            raise Refused("The run names %d file(s) that are gone: %s. Pack a newer run."
                          % (len(gone), ", ".join(gone[:5])))
    finally:
        http.shutdown()
        http.server_close()
    said = [e["text"] for e in events if e["kind"] == "person"]
    opened = [e["recording"]["name"] for e in events if e["kind"] == "opened"]
    began = {e["job"]: e["t"] for e in events if e["kind"] == "job"}
    took = round(sum(e["t"] - began[e["job"]] for e in events if e["kind"] == "end" and e["job"] in began))
    ran = list(dict.fromkeys(e["mind"] for e in events if e["kind"] == "job" and e.get("mind")))
    minds = [dict(m, ok=True) for m in health.get("minds") or [] if m.get("label") in ran]
    shown = dict(health, here=True, **({"minds": minds, "default": minds[0]["key"]} if minds else {}))
    on_card = any(m.get("local") for m in minds) or not minds
    data = _unfoldered({
        "chat": chat_id, "events": events, "panel": panel, "version": __version__, "layer": _relative(layer),
        "base": _relative(made),
        "health": shown,
        "listed": {"id": chat_id, "when": round(events[-1]["t"]) if events else 0, "first": said[0][:100] if said
                   else "", "messages": len(said), "recording": opened[-1] if opened else ""},
        "made": time.strftime("%d %b %Y", time.localtime(events[0]["t"])).lstrip("0") if events else "",
        "took": took, "card": (_card() if card is None else card) if on_card else "",
        "api": [m["label"] for m in minds if not m.get("local")], "features": agent.features()})
    files["api/replay.json"] = json.dumps(data, ensure_ascii=False).encode("utf-8")
    for name in sorted(os.listdir(server.WEB)):
        with open(os.path.join(server.WEB, name), "rb") as f:
            body = f.read()
        if name == "app.js":
            body = body.decode("utf-8")
            if API not in body:
                raise Refused("The page asks the studio for nothing under /api/, so the replay cannot stand in.")
            body = body.replace(API, '"api/').encode("utf-8")
        elif name == "index.html":
            body = body.decode("utf-8")
            if '<script src="app.js"></script>' not in body:
                raise Refused("The page does not load app.js the way the replay expects.")
            body = body.replace('<script src="app.js"></script>', SHIM).encode("utf-8")
        files[name] = body
    for name, body in files.items():
        full = os.path.join(folder, *name.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full + ".part", "wb") as f:
            f.write(body)
        os.replace(full + ".part", full)
    return {"files": len(files), "bytes": sum(len(b) for b in files.values()), "events": len(events), "took": took}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.exit("python -m clipstudio.replay <chat> <folder>")
    try:
        done = pack(argv[0], argv[1])
    except Refused as e:
        sys.exit(str(e))
    print("%d files, %.1f MB, %d events of a %d s run in %s. Serve it with: python -m http.server -d %s"
          % (done["files"], done["bytes"] / 1e6, done["events"], done["took"], argv[1], argv[1]))


if __name__ == "__main__":
    main()
