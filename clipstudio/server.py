"""The web server: the page, and the small JSON API the page talks to.

Standard library only. It listens on 127.0.0.1, so the recordings it opens never leave this computer. LCS_HOST=0.0.0.0
lets a phone on the same Wi-Fi use it too; then anyone on that network can open the videos in the folders.
Two guards hold either way: a request must name the studio by an address or "localhost" (a web page cannot reach it
through a name it owns), and a POST must be JSON (another web page cannot send JSON here without asking first).
- GET /api/health: the minds and whether each can answer, whether the hearing worker is ready, and whether
  the page is on this computer
- GET /api/chat[?id=<chat>]: the chat the page shows (or an earlier one, opened again), and whether Muse is working
- POST /api/chat/new: a new, empty chat
- GET /api/chats: the earlier chats, newest first
- GET /api/spent?chat=<id>&currency=<code>: what Muse cost, in US dollars by the providers' list prices: each
  message of the chat, the chat, today, this month, each mind; and the day's rate for the person's own money
- POST /api/say {chat, message, mind}: the person's message. It starts a job, or cuts in on the job running now
- POST /api/stop {chat}: stops the job running now, at once
- GET /api/recordings: the newest recordings, to open with one tap
- POST /api/open {chat, path} or {chat, name, size, mtime}: opens a recording where it is (a dropped file is found
  by its name, size and time) and tells Muse, with any words typed with it
- GET /api/frame/<name>.jpg: a picture from an open recording
- GET /api/look/<name>.png: the picture of one look of the words, for the person to pick from
- GET /api/preview/<name>.mp4: the rough video of a proposed moment, in the byte range the browser asks for, so it
  can play from any second
- GET /api/panel?chat=<id>: what the page shows beside and under the clip: the choices of captions, text style and
  colour look (the one the clip has, and why a choice is off), the words on the clip by their time, and whether the
  video can be made again
- POST /api/captions {chat, mode}, /api/style {chat, style}, /api/colour {chat, look}: a tap beside the clip. The
  program makes the edit Muse's tool makes, at once, without asking Muse; Undo takes it back, and Muse is told
- POST /api/words {chat, kind, n, was, words}: the person's own words for a title or a caption phrase
- POST /api/mute {chat, start, end, back}: the clip's own sound muted from start to end s of the clip, or brought back
- POST /api/remake {chat}: makes the video again from the clip as it is now, as render_clip does, without Muse.
  Each answers {"ok": true}, or {"ok": false, "error"} with 404 when the clip or the words are not there and 400
  with the reason the change was refused. /api/captions {chat, language: "said"} puts the captions back in the words
  as said; English is a message to Muse, which writes it
- GET /api/layer?chat=<id>: the clip's words as see-through pictures by their time, {"size", "steps": [{"from",
  "to", "url"}]}, so the page lays them over the base video and a change of words shows without a render
- GET /api/layer/<name>.png: one of those pictures, from work/layers only
- POST /api/base {chat}: the base video of the clip as it is now, the clip without its words: {"url", "from_cache"}
  at once when it is there, else {"url": null, "making": true} while it is made in the background (a "base" event
  says when). A POST, as it starts work: another web page cannot start it
- GET /api/base/<name>.mp4: the base video, in the byte range the browser asks for
- GET /api/events?chat=<id>: every event of the chat from the start, then each new one as it happens, and Muse's
  thinking while it is written (Server-Sent Events, so a reload redraws the whole chat and a running job carries on).
  Every page shows the chat the server has open: when another chat opens, the stream says "moved" and the page
  follows, so neither an address nor a second tab can leave a page working on another chat.
"""
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, quote

from . import __version__, agent, effects, hearing, layer, mind, premiere, previews, recording, render, spent, text

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
PREVIEW = re.compile(r"^/api/preview/([0-9a-f]{16}\.mp4)$")
CLIP = re.compile(r"^/api/clip/([0-9a-f]{16}\.mp4)$")
MOVE = re.compile(r"^/api/move/([a-z]+)\.mp4$")
POSTER = re.compile(r"^/api/clip/([0-9a-f]{16})\.jpg$")
LOOK = re.compile(r"^/api/look/([0-9a-z]+)\.png$")
LAYER = re.compile(r"^/api/layer/([0-9a-f]{20}\.png)$")
BASE = re.compile(r"^/api/base/([0-9a-f]{16}\.mp4)$")
POSTER_AT = 1.0
READ_TRIES, READ_WAIT = 10, 0.05
RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
CHANGES = ("/api/captions", "/api/style", "/api/colour", "/api/words", "/api/remake", "/api/channel",
           "/api/channel/save", "/api/mute")


class Handler(BaseHTTPRequestHandler):
    server_version = "local-clip-studio/" + __version__

    def do_GET(self):
        if not self.named_well():
            return
        path, _, query = self.path.partition("?")
        query = dict(parse_qsl(query))
        if path == "/api/health":
            self.send_json({"ok": True, "version": __version__, "default": mind.DEFAULT, "here": self.here(),
                            "minds": [m.status() for m in mind.MINDS.values()], "hearing": hearing.status()})
        elif path == "/api/features":
            self.send_json({"ok": True, "groups": agent.features()})
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
        elif PREVIEW.match(path):
            self.send_video(previews.folder(), PREVIEW.match(path).group(1), "preview")
        elif CLIP.match(path):
            self.send_video(render.folder("clips"), CLIP.match(path).group(1), "clip",
                            download=query.get("download") == "1")
        elif POSTER.match(path):
            self.send_poster(POSTER.match(path).group(1))
        elif MOVE.match(path):
            self.send_move(MOVE.match(path).group(1))
        elif path == "/api/clipframe":
            self.send_clip_frame(query)
        elif path == "/api/faces":
            self.send_faces(query)
        elif path == "/api/panel":
            chat = agent.find(query.get("chat", ""))
            if chat is None:
                self.send_json({"ok": False, "error": GONE}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(dict(chat.panel(), ok=True))
        elif path == "/api/spent":
            got = spent.totals(os.path.join(agent.WORK, "spent.jsonl"), query.get("chat"))
            self.send_json(dict(got, ok=True, prices=mind.PRICED, rate=spent.rate(
                query.get("currency"), os.path.join(agent.WORK, "rates.json"))))
        elif path == "/api/layer":
            self.send_layer(agent.find(query.get("chat", "")))
        elif path == "/api/premiere":
            self.send_premiere(agent.find(query.get("chat", "")))
        elif LAYER.match(path):
            self.send_picture(render.folder("layers"), LAYER.match(path).group(1))
        elif BASE.match(path):
            self.send_video(render.folder("bases"), BASE.match(path).group(1), "preview")
        elif LOOK.match(path):
            self.send_look(LOOK.match(path).group(1))
        elif path.startswith("/api/"):
            self.send_json({"ok": False, "error": "There is nothing at %s." % path}, HTTPStatus.NOT_FOUND)
        else:
            self.send_file(path)

    def do_POST(self):
        if not self.named_well():
            return
        path = self.path.split("?", 1)[0]
        if path not in ("/api/say", "/api/stop", "/api/chat/new", "/api/open", "/api/undo", "/api/person",
                        "/api/base") + CHANGES:
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
        elif path == "/api/undo":
            entry = chat.undo()
            if entry is None:
                self.send_json({"ok": False, "error": "There is nothing to take back."})
            else:
                self.send_json({"ok": True, "undone": agent.describe(entry)})
        elif path == "/api/open":
            self.open_recording(chat, body)
        elif path == "/api/person":
            n = body.get("n")
            if not isinstance(n, int) or isinstance(n, bool) or not chat.set_person(n, bool(body.get("blur"))):
                self.send_json({"ok": False, "error": "The clip has no person %s." % n}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json({"ok": True})
        elif path in CHANGES:
            self.change_clip(chat, path, body)
        elif path == "/api/base":
            self.send_base(chat)
        else:
            words = str(body.get("message") or "").strip()
            if not words:
                self.send_json({"ok": False, "error": "Type a message first."}, HTTPStatus.BAD_REQUEST)
                return
            job = chat.say(words, body.get("mind"))
            self.send_json({"ok": True, "chat": chat.id, "job": job.n})

    def change_clip(self, chat, path, body):
        """A tap beside or under the clip: {"ok": true} once the change is made (for /api/remake, once the video is
        being made), else 404 when the clip or the words are not there and 400 with the reason it was refused."""
        try:
            if path == "/api/captions" and "language" in body:
                why = chat.set_captions_language(str(body.get("language") or ""))
            elif path == "/api/captions":
                why = chat.set_captions(str(body.get("mode") or ""))
            elif path == "/api/style":
                why = chat.set_style(str(body.get("style") or ""))
            elif path == "/api/colour":
                why = chat.set_look(str(body.get("look") or ""))
            elif path == "/api/remake":
                why = chat.make_again()
            elif path == "/api/channel" and "spot" in body:
                why = chat.set_channel_spot(str(body.get("spot") or ""))
            elif path == "/api/channel":
                why = chat.use_channel(str(body.get("name") or ""))
            elif path == "/api/channel/save":
                why = chat.save_channel(str(body.get("name") or ""), str(body.get("handle") or ""))
            elif path == "/api/mute":
                start, end = body.get("start"), body.get("end")
                why = chat.set_mute(start, end, body.get("back") is True) if all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in (start, end)) else                     "Send the start and the end of the part to mute, in seconds of the clip."
            elif body.get("kind") not in ("title", "caption") or not isinstance(body.get("n"), int) or \
                    isinstance(body.get("n"), bool):
                why = "Send which words to change: a title or a caption, and its number."
            else:
                why = chat.set_words(body["kind"], body["n"], str(body.get("was") or ""), str(body.get("words") or ""))
        except agent.Missing as e:
            self.send_json({"ok": False, "error": str(e)}, HTTPStatus.NOT_FOUND)
            return
        if why:
            self.send_json({"ok": False, "error": why}, HTTPStatus.BAD_REQUEST)
        else:
            self.send_json({"ok": True})

    def send_layer(self, chat):
        """The clip's words layer: {"size", "steps": [{"from", "to", "url"}]}, or 404 with why there is none."""
        try:
            if chat is None:
                raise agent.Missing(GONE)
            steps = chat.layer()
        except agent.Missing as e:
            self.send_json({"ok": False, "error": str(e)}, HTTPStatus.NOT_FOUND)
            return
        except (RuntimeError, ValueError) as e:
            self.send_json({"ok": False, "error": "The words could not be drawn: %s" % e}, HTTPStatus.NOT_FOUND)
            return
        self.send_json({"ok": True, "size": list(layer.SIZE), "steps": [
            {"from": s["from"], "to": s["to"], "url": "/api/layer/" + os.path.basename(s["file"])} for s in steps]})

    def send_base(self, chat):
        """The base video of the clip as it is now (Chat.base): 404 before there is a clip, 400 when it could not be
        made."""
        try:
            got = chat.base()
        except agent.Missing as e:
            self.send_json({"ok": False, "error": str(e)}, HTTPStatus.NOT_FOUND)
            return
        if "error" in got:
            self.send_json({"ok": False, "error": got["error"]}, HTTPStatus.BAD_REQUEST)
        else:
            self.send_json(dict(got, ok=True))

    def send_picture(self, folder, name):
        """A words picture the page lays over the video, by its name: nothing from the request but a name of hex
        digits reaches the disk, so only files in folder are sent. A name is what the picture shows, so it never
        changes."""
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            self.send_json({"ok": False, "error": "There is no words picture %s." % name}, HTTPStatus.NOT_FOUND)
            return
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

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

    def send_look(self, name):
        """The picture of the look called name, made the first time any look's picture is asked for."""
        card = next((c for c in text.cards() if c["name"] == name), None)
        if card is None:
            self.send_json({"ok": False, "error": "There is no look %s." % name}, HTTPStatus.NOT_FOUND)
            return
        with open(card["png"], "rb") as f:
            body = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def send_poster(self, sign):
        """A picture of the finished clip POSTER_AT seconds in, for its player to show before it plays. It is made
        from the file the first time it is asked for."""
        folder = render.folder("clips")
        video, full = os.path.join(folder, sign + ".mp4"), os.path.join(folder, sign + ".jpg")
        if not os.path.isfile(full) and os.path.isfile(video):
            part = "%s.%d.part.jpg" % (full, threading.get_ident())
            subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-ss", "%g" % POSTER_AT, "-i",
                            video, "-frames:v", "1", "-vf", "scale=540:-2", "-q:v", "3", part], capture_output=True)
            if os.path.isfile(part):
                render.settle(part, full)
        if not os.path.isfile(full):
            self.send_json({"ok": False, "error": "There is no finished clip %s." % sign}, HTTPStatus.NOT_FOUND)
            return
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def send_clip_frame(self, query):
        """The chat's clip drawn at second t, as it stands now; with look (one of agent.COLOUR_LOOKS), as it would be
        in that colour look, for its choice beside the clip; width (120 to render.PREVIEW_WIDTH) makes it smaller."""
        chat = agent.find(str(query.get("chat") or ""))
        look = query.get("look")
        try:
            t = float(query.get("t") or 0)
            width = min(max(int(query.get("width") or render.PREVIEW_WIDTH), 120), render.PREVIEW_WIDTH)
        except ValueError:
            t, width = -1.0, render.PREVIEW_WIDTH
        if chat is None or not chat.recording or t < 0 or look not in (None,) + agent.COLOUR_LOOKS:
            self.send_json({"ok": False, "error": "Send a chat that has a clip, a time from 0 and a colour look of %s."
                            % ", ".join(agent.COLOUR_LOOKS)}, HTTPStatus.NOT_FOUND)
            return
        state = chat.project.state()
        if look:
            state = dict(state, look=None if look == "none" else look)
        try:
            path = render.frame(chat.recording, state, t, chat.clip["style"] if chat.clip else "clean",
                                width=width - width % 2)
        except render.NotDrawn as e:
            self.send_json({"ok": False, "error": str(e)}, HTTPStatus.NOT_FOUND)
            return
        body = read_settled(path)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_faces(self, query):
        """The other people's faces at second t of the chat's clip, in frame pixels, for the page to tap."""
        chat = agent.find(str(query.get("chat") or ""))
        try:
            t = float(query.get("t") or 0)
        except ValueError:
            t = -1.0
        if chat is None or t < 0:
            self.send_json({"ok": False, "error": "Send a chat and a time from 0."}, HTTPStatus.NOT_FOUND)
            return
        self.send_json({"ok": True, "faces": chat.faces(t), "size": [1080, 1920]})

    def send_move(self, name):
        """The looping sample of a flurry move (effects.moves()) for its tile in the Features menu, made once."""
        if name not in effects.MOVES:
            self.send_json({"ok": False, "error": "There is no move %s." % name}, HTTPStatus.NOT_FOUND)
            return
        clip = next(m["clip"] for m in effects.moves() if m["name"] == name)
        self.send_video(os.path.dirname(clip), os.path.basename(clip), "move")

    def send_video(self, folder, name, what, download=False):
        """A video in the range the browser asks for. download asks the browser to save it as a file."""
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            self.send_json({"ok": False, "error": "There is no %s %s." % (what, name)}, HTTPStatus.NOT_FOUND)
            return
        size = os.path.getsize(full)
        start, end = 0, size - 1
        asked = RANGE.match(self.headers.get("Range") or "")
        if asked and asked.group(1):
            start, end = int(asked.group(1)), min(int(asked.group(2) or end), end)
        elif asked and asked.group(2):
            start = max(0, size - int(asked.group(2)))
        asked = bool(asked and (asked.group(1) or asked.group(2)))
        if start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", "bytes */%d" % size)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        with open(full, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if asked else HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="clip-%s"' % name)
        if asked:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def send_premiere(self, chat):
        """The clip as it is now as a Premiere Pro project (premiere.py), a zip to save; 404 with the reason when the
        chat or its clip is not there."""
        if chat is None:
            self.send_json({"ok": False, "error": GONE}, HTTPStatus.NOT_FOUND)
            return
        titles = [e["title"] for e in chat.events if e["kind"] in ("ready", "named") and e.get("title")]
        try:
            name, body = premiere.bundle(titles[-1] if titles else "", chat.recording, chat.project.state())
        except premiere.NotExported as e:
            self.send_json({"ok": False, "error": str(e)}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", "attachment; filename=\"clip-premiere.zip\"; filename*=UTF-8''%s"
                         % quote(name))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
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


def read_settled(path, tries=READ_TRIES):
    """The file's bytes. On Windows a file that another request is moving into place refuses to open for a moment
    (seen 10-28: PermissionError on clip frames while the page and Muse asked for the same frame), so it is tried
    again a few times before the error stands."""
    for k in range(tries):
        try:
            with open(path, "rb") as f:
                return f.read()
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(READ_WAIT)


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
    keys = mind.load_keys()
    mind.MINDS.update(mind.minds())
    hearing.start()
    print("Local Clip Studio %s: http://127.0.0.1:%d/" % (__version__, PORT), flush=True)
    if keys:
        print("Keys read from LCS_KEYS: %s" % ", ".join(keys), flush=True)
    if not ipaddress.ip_address(HOST).is_loopback:
        print("On a phone on the same Wi-Fi, open http://<this computer's address>:%d/" % PORT, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
