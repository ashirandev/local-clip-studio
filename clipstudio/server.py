"""The web server: the page, and the small JSON API the page talks to.

Standard library only. It listens on 127.0.0.1, so the recordings it opens never leave this computer. LCS_HOST=0.0.0.0
lets a phone on the same Wi-Fi use it too; then anyone on that network can open the videos in the folders.
Two guards hold either way: a request must name the studio by an address or "localhost" (a web page cannot reach it
through a name it owns), and a POST must be JSON (another web page cannot send JSON here without asking first).
- GET /api/health: the two minds and whether each can answer, and whether the page is on this computer
- GET /api/chat[?id=<chat>]: the chat the page shows (or an earlier one, opened again), and whether Muse is working
- POST /api/chat/new: a new, empty chat
- GET /api/chats: the earlier chats, newest first
- POST /api/say {chat, message, mind}: the person's message. It starts a job, or cuts in on the job running now
- POST /api/stop {chat}: stops the job running now, at once
- GET /api/recordings: the newest recordings, to open with one tap
- POST /api/open {chat, path} or {chat, name, size, mtime}: opens a recording where it is (a dropped file is found
  by its name, size and time) and tells Muse, with any words typed with it
- GET /api/frame/<name>.jpg: a picture from an open recording
- GET /api/events?chat=<id>: every event of the chat from the start, then each new one as it happens, and Muse's
  thinking while it is written (Server-Sent Events, so a reload redraws the whole chat and a running job carries on).
  Every page shows the chat the server has open: when another chat opens, the stream says "moved" and the page
  follows, so neither an address nor a second tab can leave a page working on another chat.
"""
import ipaddress
import json
import os
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl

from . import __version__, agent, mind, recording

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
HOST = os.environ.get("LCS_HOST", "127.0.0.1")
PORT = int(os.environ.get("LCS_PORT", "8899"))
# Windows can map .js to text/plain from the registry, so the types the page needs are fixed here.
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml",
         ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".mp4": "video/mp4"}
MAX_BODY = 1_000_000
KEEP_ALIVE = 15
GONE = "That chat is not open any more. Reload the page."
NEWEST = 5
FRAME = re.compile(r"^/api/frame/([0-9a-f]{16}\.jpg)$")


class Handler(BaseHTTPRequestHandler):
    server_version = "local-clip-studio/" + __version__

    def do_GET(self):
        if not self.named_well():
            return
        path, _, query = self.path.partition("?")
        query = dict(parse_qsl(query))
        if path == "/api/health":
            self.send_json({"ok": True, "version": __version__, "default": mind.DEFAULT, "here": self.here(),
                            "minds": [m.status() for m in mind.MINDS.values()]})
        elif path == "/api/chat":
            chat = agent.open_chat(query["id"]) if query.get("id") else agent.current()
            if chat is None:
                self.send_json({"ok": False, "error": "There is no chat %s." % query["id"]}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json({"ok": True, "chat": chat.id, "busy": chat.busy()})
        elif path == "/api/chats":
            self.send_json({"ok": True, "chats": agent.earlier()})
        elif path == "/api/events":
            self.send_events(query)
        elif path == "/api/recordings":
            self.send_json({"ok": True, "folders": recording.folders(), "recordings": newest()})
        elif FRAME.match(path):
            self.send_frame(FRAME.match(path).group(1))
        elif path.startswith("/api/"):
            self.send_json({"ok": False, "error": "There is nothing at %s." % path}, HTTPStatus.NOT_FOUND)
        else:
            self.send_file(path)

    def do_POST(self):
        if not self.named_well():
            return
        path = self.path.split("?", 1)[0]
        if path not in ("/api/say", "/api/stop", "/api/chat/new", "/api/open"):
            self.send_json({"ok": False, "error": "There is nothing at %s." % path}, HTTPStatus.NOT_FOUND)
            return
        body = self.read_json()
        if body is None:
            return
        if path == "/api/chat/new":
            self.send_json({"ok": True, "chat": agent.new().id})
            return
        chat = agent.find(str(body.get("chat") or ""))
        if chat is None:
            self.send_json({"ok": False, "error": GONE}, HTTPStatus.NOT_FOUND)
        elif path == "/api/stop":
            self.send_json({"ok": True, "stopped": chat.stop()})
        elif path == "/api/open":
            self.open_recording(chat, body)
        else:
            words = str(body.get("message") or "").strip()
            if not words:
                self.send_json({"ok": False, "error": "Type a message first."}, HTTPStatus.BAD_REQUEST)
                return
            job = chat.say(words, body.get("mind"))
            self.send_json({"ok": True, "chat": chat.id, "job": job.n})

    def open_recording(self, chat, body):
        try:
            if body.get("path"):
                where = recording.check_path(str(body["path"]))
            else:
                size, mtime = body.get("size"), body.get("mtime")
                if not body.get("name") or not isinstance(size, int):
                    raise recording.NotOpened("Send the dropped file's name and size, or a path.")
                where = recording.locate(str(body["name"]), size, mtime if isinstance(mtime, (int, float)) else None)
            rec = recording.probe(where)
        except recording.NotOpened as e:
            self.send_json({"ok": False, "error": str(e)}, HTTPStatus.NOT_FOUND)
            return
        job = chat.open_dropped(rec, str(body.get("words") or "").strip(), body.get("mind"),
                                str(body.get("language") or "")[:20])
        self.send_json({"ok": True, "chat": chat.id, "job": job.n, "path": rec["path"]})

    def send_frame(self, name):
        full = os.path.join(recording.WORK, "frames", name)
        if not os.path.isfile(full):
            self.send_json({"ok": False, "error": "There is no picture %s." % name}, HTTPStatus.NOT_FOUND)
            return
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def send_events(self, query):
        chat = agent.find(query.get("chat", ""))
        if chat is None:
            self.send_json({"ok": False, "error": GONE}, HTTPStatus.NOT_FOUND)
            return
        last = self.headers.get("Last-Event-ID")
        n, rev = (int(last) + 1 if last and last.isdigit() else 0), -1
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                if agent.shown() not in (None, chat.id):
                    self.wfile.write(("event: moved\ndata: %s\n\n" % json.dumps({"chat": agent.shown()}))
                                     .encode("utf-8"))
                    self.wfile.flush()
                    return
                batch, live = chat.since(n, rev, KEEP_ALIVE)
                for event in batch:
                    line = "id: %d\ndata: %s\n\n" % (event["id"], json.dumps(event, ensure_ascii=False))
                    self.wfile.write(line.encode("utf-8"))
                n += len(batch)
                if live:
                    rev = live["rev"]
                    self.wfile.write(("event: think\ndata: %s\n\n" % json.dumps(live, ensure_ascii=False))
                                     .encode("utf-8"))
                if not batch and not live:
                    self.wfile.write(b": still here\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def named_well(self):
        """False after answering, when the request names the studio by a web name such as evil.example."""
        name = (self.headers.get("Host") or "").strip().lower()
        if name.startswith("["):
            name = name[1:].split("]", 1)[0]
        elif name.count(":") == 1:
            name = name.split(":", 1)[0]
        if name == "localhost":
            return True
        try:
            ipaddress.ip_address(name)
            return True
        except ValueError:
            self.send_json({"ok": False, "error": "Open the studio by this computer's address, not by the name %s."
                                                  % (name or "(none)")}, HTTPStatus.FORBIDDEN)
            return False

    def here(self):
        """True when the page is on this computer, so a file dropped on it is on this computer too."""
        return ipaddress.ip_address(self.client_address[0].split("%", 1)[0]).is_loopback

    def read_json(self):
        """The request body as a dict, or None after answering with a sentence."""
        if (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower() != "application/json":
            self.send_json({"ok": False, "error": "Send the request as JSON."}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return None
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            size = -1
        if not 0 < size <= MAX_BODY:
            self.send_json({"ok": False, "error": "The request was empty or too big."}, HTTPStatus.BAD_REQUEST)
            return None
        try:
            body = json.loads(self.rfile.read(size).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_json({"ok": False, "error": "The request was not readable JSON."}, HTTPStatus.BAD_REQUEST)
            return None
        if not isinstance(body, dict):
            self.send_json({"ok": False, "error": "The request must be a JSON object."}, HTTPStatus.BAD_REQUEST)
            return None
        return body

    def send_file(self, path):
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        web = os.path.realpath(WEB)
        full = os.path.realpath(os.path.join(web, name))
        kind = TYPES.get(os.path.splitext(full)[1].lower())
        if not full.startswith(web + os.sep) or not os.path.isfile(full) or not kind:
            self.send_json({"ok": False, "error": "There is nothing at %s." % path}, HTTPStatus.NOT_FOUND)
            return
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_request(self, code="-", size="-"):
        if str(int(code) if isinstance(code, HTTPStatus) else code)[:1] in ("4", "5"):
            super().log_request(code, size)


def newest():
    """The newest recordings, with their length when ffprobe can read it."""
    out = []
    for mtime, size, path in recording.walk()[:NEWEST]:
        item = {"path": path, "name": os.path.basename(path), "size": recording.size_text(size),
                "when": recording.when(mtime), "length": ""}
        try:
            item["length"] = recording.clock(recording.probe(path)["seconds"])
        except recording.NotOpened:
            pass
        out.append(item)
    return out


def main():
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        sys.exit("Port %d on this computer is taken (%s). Set LCS_PORT to a free port and start again."
                 % (PORT, e.strerror))
    print("Local Clip Studio %s: http://127.0.0.1:%d/" % (__version__, PORT), flush=True)
    if not ipaddress.ip_address(HOST).is_loopback:
        print("On a phone on the same Wi-Fi, open http://<this computer's address>:%d/" % PORT, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
