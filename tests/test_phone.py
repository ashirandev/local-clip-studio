"""A phone on the same Wi-Fi or Tailscale that sends photos and videos, with made-up files in temporary folders:
python -m unittest discover -s tests"""
import http.client
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import phone, recording, server  # noqa: E402

BATCH = "2026-10-29 11.40.00"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Phone(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patches = [mock.patch.object(recording, "WORK", os.path.join(self.tmp, "work")),
                        mock.patch.object(recording, "INBOX", os.path.join(self.tmp, "From phone"))]
        for p in self.patches:
            p.start()
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.away = mock.patch.object(server.Handler, "here", lambda h: False)

    def tearDown(self):
        server.stop_phones()
        self.http.shutdown()
        self.http.server_close()
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def ask(self, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=30)
        sent = dict(headers or {})
        data = raw
        if body is not None:
            data = json.dumps(body)
            sent["Content-Type"] = "application/json"
        conn.request(method, path, data, sent)
        r = conn.getresponse()
        got = r.read().decode("utf-8", "replace")
        conn.close()
        return r, got

    def send(self, name, data, kind="application/octet-stream"):
        return self.ask("POST", "/api/upload?name=%s&batch=%s&when=1761700000" % (
            name.replace(" ", "%20").replace("/", "%2F"), BATCH.replace(" ", "%20")),
            headers={"Content-Type": kind}, raw=data)

    def inbox(self):
        folder = os.path.join(recording.INBOX, BATCH)
        return sorted(os.listdir(folder)) if os.path.isdir(folder) else []

    def test_a_phone_uses_the_studio_with_nothing_to_set_up(self):
        with self.away:
            r, page = self.ask("GET", "/")
            self.assertEqual(r.status, 200)
            self.assertIn("Send from this phone", page)
            self.assertEqual(self.ask("GET", "/api/health")[0].status, 200)
            self.assertEqual(self.ask("GET", "/api/recordings")[0].status, 200)

    def test_a_phone_sends_a_video_that_arrives_whole_under_its_own_name(self):
        data = os.urandom(300_000)
        with self.away:
            r, got = self.send("../../Windows/IMG 3374.MOV", data)
            self.assertEqual(r.status, 200, got)
            d = json.loads(got)
            self.assertEqual((d["name"], d["kind"]), ("IMG 3374.mov", "video"))
            r, got = self.send("IMG 3374.MOV", data[:1000])
        self.assertEqual(json.loads(got)["name"], "IMG 3374 (2).mov", "a second file of that name keeps the first")
        self.assertEqual(self.inbox(), ["IMG 3374 (2).mov", "IMG 3374.mov"])
        with open(d["path"], "rb") as f:
            self.assertEqual(f.read(), data)
        self.assertEqual(int(os.path.getmtime(d["path"])), 1761700000)
        with mock.patch.dict(os.environ, {"LCS_VIDEOS": self.tmp + os.sep + "elsewhere"}):
            os.makedirs(os.environ["LCS_VIDEOS"])
            self.assertIn(os.path.realpath(recording.INBOX), recording.folders(), "named folders and the phone's")
            self.assertEqual(recording.check_path(d["path"]), os.path.realpath(d["path"]))

    def test_a_recording_on_a_drive_the_studio_reads_whole_is_inside_it(self):
        clip = os.path.join(self.tmp, "trip.mp4")
        with open(clip, "wb") as f:
            f.write(b"x")
        root = os.path.splitdrive(os.path.realpath(self.tmp))[0] + os.sep
        with mock.patch.object(recording, "folders", lambda: [root]):
            self.assertEqual(recording.check_path(clip), os.path.realpath(clip))
            self.assertEqual(recording.check_folder(self.tmp), os.path.realpath(self.tmp))

    def test_a_send_that_stops_partway_leaves_nothing_behind(self):
        with self.away:
            s = socket.create_connection(("127.0.0.1", self.http.server_port))
            s.sendall(("POST /api/upload?name=IMG_1.MOV&batch=%s HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                       "Content-Type: application/octet-stream\r\nContent-Length: 100000\r\n\r\n"
                       % BATCH.replace(" ", "%20")).encode("ascii") + b"x" * 40_000)
            s.shutdown(socket.SHUT_WR)
            got = b""
            while True:
                piece = s.recv(4096)
                if not piece:
                    break
                got += piece
            s.close()
        answer = got.decode("utf-8", "replace")
        self.assertIn("400", answer.split("\r\n", 1)[0])
        self.assertIn("stopped after 40000 of 100000 bytes", answer)
        self.assertEqual(self.inbox(), [])

    def test_only_photos_and_videos_and_only_as_octet_stream(self):
        with self.away:
            r, _ = self.send("IMG_1.MOV", b"x" * 100, kind="text/plain")
            self.assertEqual(r.status, 415, "a form on another web page cannot send it")
            r, got = self.send("setup.exe", b"x" * 100)
            self.assertEqual(r.status, 400)
            self.assertIn("is not a photo or a video", got)
            r, got = self.send("IMG_2.HEIC", b"x" * 100)
            self.assertEqual(json.loads(got)["kind"], "photo")
        self.assertEqual(self.inbox(), ["IMG_2.heic"])

    def test_a_phone_cannot_turn_phone_off_or_set_the_key(self):
        with self.away:
            for method, path, body in (("GET", "/api/phone", None), ("POST", "/api/phone", {"on": False}),
                                       ("GET", "/api/phone/qr.svg?address=127.0.0.1", None)):
                r, _ = self.ask(method, path, body)
                self.assertEqual(r.status, 403, path)
        self.assertTrue(phone.wanted(), "Phone is on until the computer turns it off")

    def test_phone_on_listens_on_the_phone_addresses_and_off_stops_at_once(self):
        port = free_port()
        with mock.patch.object(server, "PORT", port), \
                mock.patch.object(phone, "addresses", lambda: [("127.0.0.2", "Wi-Fi")]):
            r, got = self.ask("POST", "/api/phone", {"on": True})
            d = json.loads(got)
            self.assertEqual((d["on"], d["addresses"][0]["listening"]), (True, True))
            conn = http.client.HTTPConnection("127.0.0.2", port, timeout=10)
            conn.request("GET", "/style.css")
            self.assertEqual(conn.getresponse().status, 200)
            conn.close()
            with mock.patch.object(server.segno, "make", wraps=server.segno.make) as made:
                r, svg = self.ask("GET", "/api/phone/qr.svg?address=127.0.0.2")
                self.assertEqual(made.call_args[0][0], "http://127.0.0.2:%d/" % port)
            self.assertEqual(r.status, 200)
            self.assertIn("<svg", svg)
            r, got = self.ask("POST", "/api/phone", {"on": False})
            self.assertFalse(json.loads(got)["on"])
            self.assertFalse(phone.wanted(), "kept for the next start")
            with self.assertRaises(OSError):
                socket.create_connection(("127.0.0.2", port), timeout=2).close()

    def test_videos_sent_together_are_for_muse_to_open_together(self):
        folder = os.path.join(recording.INBOX, BATCH)
        os.makedirs(folder)
        for name in ("IMG_1.MOV", "IMG_2.MOV", "IMG_3.HEIC"):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(b"x")
        chat = mock.Mock(id="c1")
        chat.open_sent.return_value = mock.Mock(n=4)
        with mock.patch.object(server.agent, "find", lambda c: chat):
            r, got = self.ask("POST", "/api/open", {"chat": "c1", "folder": folder, "words": "ทำคลิปเที่ยว"})
        self.assertEqual((r.status, json.loads(got)["job"]), (200, 4))
        self.assertEqual(chat.open_sent.call_args[0][:3], (os.path.realpath(folder), 2, "ทำคลิปเที่ยว"))


if __name__ == "__main__":
    unittest.main()
