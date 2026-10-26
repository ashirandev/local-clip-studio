"""The web server: the page, and the small JSON API the page talks to.

Standard library only. It listens on 127.0.0.1, so the recordings it opens never leave this computer.
"""
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, mind

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
HOST = "127.0.0.1"
PORT = int(os.environ.get("LCS_PORT", "8899"))
# Windows can map .js to text/plain from the registry, so the types the page needs are fixed here.
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml",
         ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".mp4": "video/mp4"}
SYSTEM = ("You are the editor in Local Chat Studio. People drop in long recordings (selling lives, game streams, "
          "travel clips) and you turn them into short vertical clips ready to post. Answer in the language the person "
          "writes in, briefly and in plain words.")
MAX_BODY = 1_000_000
HISTORY = 20


class Handler(BaseHTTPRequestHandler):
    server_version = "local-chat-studio/" + __version__

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self.send_json({"ok": True, "version": __version__, "default": mind.DEFAULT,
                            "minds": [m.status() for m in mind.MINDS.values()]})
        elif path.startswith("/api/"):
            self.send_json({"ok": False, "error": "There is nothing at %s." % path}, HTTPStatus.NOT_FOUND)
        else:
            self.send_file(path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/ask":
            self.send_json({"ok": False, "error": "There is nothing at %s." % path}, HTTPStatus.NOT_FOUND)
            return
        body = self.read_json()
        if body is None:
            return
        talk = [m for m in body.get("history") or [] if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str)][-HISTORY:]
        words = str(body.get("message") or "").strip()
        if not words:
            self.send_json({"ok": False, "error": "Type a message first."}, HTTPStatus.BAD_REQUEST)
            return
        messages = [{"role": "system", "content": SYSTEM}] + talk + [{"role": "user", "content": words}]
        brain = mind.get(body.get("mind"))
        try:
            got = brain.ask(messages)
        except mind.MindError as e:
            self.send_json({"ok": False, "error": str(e)}, HTTPStatus.BAD_GATEWAY)
            return
        self.send_json({"ok": True, "mind": brain.label, "answer": got["text"], "thinking": got["thinking"],
                        "engine": got["engine"]})

    def read_json(self):
        """The request body as a dict, or None after answering with a sentence."""
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


def main():
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        sys.exit("Port %d on this computer is taken (%s). Set LCS_PORT to a free port and start again."
                 % (PORT, e.strerror))
    print("Local Chat Studio %s: http://%s:%d/" % (__version__, HOST, PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
