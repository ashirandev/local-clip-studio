"""The web server: the page, and the small JSON API the page talks to.

Standard library only. It listens on 127.0.0.1, so the recordings it opens never leave this computer.
"""
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
HOST = "127.0.0.1"
PORT = int(os.environ.get("LCS_PORT", "8899"))
# Windows can map .js to text/plain from the registry, so the types the page needs are fixed here.
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml",
         ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".mp4": "video/mp4"}


class Handler(BaseHTTPRequestHandler):
    server_version = "local-chat-studio/" + __version__

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self.send_json({"ok": True, "version": __version__})
        elif path.startswith("/api/"):
            self.send_json({"ok": False, "error": "There is nothing at %s." % path}, HTTPStatus.NOT_FOUND)
        else:
            self.send_file(path)

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
