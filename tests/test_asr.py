"""Muse Voice Transcribe as the hearing (asr.py, hearing.engine), with a fake Meta server that keeps what it was sent:
nothing goes to Meta.

    python -m unittest test_asr
"""
import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, asr, hearing, recording, sends  # noqa: E402
from test_agent_loop import ScriptedMind  # noqa: E402

RATE = 16000


def sound(*parts):
    """16-bit mono PCM: each part (seconds, loud) a tone or near silence."""
    out = bytearray()
    for seconds, loud in parts:
        for i in range(int(seconds * RATE)):
            out += struct.pack("<h", int((9000 if loud else 3) * math.sin(i * 0.07)))
    return bytes(out)


class Meta:
    """A fake /asr/transcribe: one turn per question, 0.1 s in from each end, its words saying how long it was."""

    def __init__(self, status=200):
        self.got, self.status = [], status
        meta = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                kind = self.headers["Content-Type"]
                boundary = kind.split("boundary=")[1].encode()
                parts = {}
                for part in body.split(b"--" + boundary)[1:-1]:
                    head, _, data = part.partition(b"\r\n\r\n")
                    name = head.split(b'name="')[1].split(b'"')[0].decode()
                    parts[name] = data[:-2]
                meta.got.append({"auth": self.headers.get("Authorization"), "request": json.loads(parts["request"]),
                                 "audio": parts["audio"]})
                if meta.status != 200:
                    self.send_response(meta.status)
                    self.end_headers()
                    return
                with wave.open(io.BytesIO(parts["audio"])) as w:
                    ms = round(1000 * w.getnframes() / w.getframerate())
                answer = json.dumps({"transcript": "x", "turns": [
                    {"turnId": 0, "startMs": 100, "endMs": ms - 100, "transcript": "พูด %d ms" % ms}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d/v1" % self.http.server_port

    def close(self):
        self.http.shutdown()
        self.http.server_close()


class Pieces(unittest.TestCase):
    def test_the_sound_is_cut_at_its_pauses_and_a_quiet_piece_is_not_sent(self):
        pcm = sound((3, True), (0.6, False), (6, True), (0.6, False), (8, True), (12, False), (30, True))
        cut = asr.pieces(pcm)
        self.assertAlmostEqual(cut[0][0], 0.0)
        self.assertAlmostEqual(cut[0][1], 9.9, delta=0.1, msg="the last pause inside 10 s")
        self.assertTrue(all(b - a <= asr.LONGEST_PIECE + 0.01 for a, b in cut))
        self.assertFalse(any(a < 29.9 and b > 18.5 for a, b in cut), "the 12 s pause from 18.2 s is not sent")
        self.assertAlmostEqual(cut[-1][1], 60.2, delta=0.1, msg="the last piece ends where the sound ends")
        self.assertIn(25.0, [round(b - a, 2) for a, b in cut], "30 s with no pause is cut at 25 s")
        self.assertEqual(asr.pieces(sound((15, True))), [(0.0, 15.0)], "15 s with no pause is one piece, to its end")


class Hears(unittest.TestCase):
    def setUp(self):
        self.meta = Meta()
        self.env = mock.patch.dict(os.environ, {"LCS_SPARK_URL": self.meta.url, "META_API_KEY": "a key"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.meta.close()

    def test_each_piece_is_asked_as_wav_and_its_turns_come_back_at_their_seconds(self):
        said, usd = asr.hear(sound((3, True), (0.6, False), (6, True)))
        self.assertEqual(len(self.meta.got), 1, "9.6 s is one piece")
        got = self.meta.got[0]
        self.assertEqual(got["auth"], "Bearer a key")
        self.assertEqual(got["request"], {"model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV",
                                          "mode": "ENDPOINTING"})
        with wave.open(io.BytesIO(got["audio"])) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 16000))
        self.assertEqual(said, [{"from": 0.1, "to": 9.5, "text": "พูด 9600 ms"}])
        self.assertAlmostEqual(usd, 9.6 / 3600 * 0.18, places=6)
        said, _ = asr.hear(sound((8, True), (0.6, False), (8, True)))
        self.assertEqual([(p["from"], p["to"]) for p in said], [(0.1, 8.2), (8.4, 16.5)],
                         "a second piece's turns start where it starts")

    def test_no_key_and_a_refused_key_are_said(self):
        with mock.patch.dict(os.environ, {"META_API_KEY": ""}):
            with self.assertRaisesRegex(asr.VoiceError, "needs your Meta API key"):
                asr.hear(sound((2, True)))
        self.assertEqual(self.meta.got, [])
        self.meta.status = 401
        with self.assertRaisesRegex(asr.VoiceError, r"refused the sound \(401: the Meta key was not accepted\)"):
            asr.hear(sound((2, True)))


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
class Engine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.meta = Meta()
        self.models, hearing.MODELS = hearing.MODELS, os.path.join(self.tmp, "no models")
        self.work = agent.WORK, recording.WORK
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        self.env = mock.patch.dict(os.environ, {"LCS_SPARK_URL": self.meta.url, "META_API_KEY": "a key",
                                                "LCS_HEARING": ""})
        self.env.start()
        self.path = os.path.join(self.tmp, "talk.mp4")
        import subprocess
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=300:d=6", "-f",
                        "lavfi", "-i", "color=c=black:s=64x64:d=6", "-shortest", "-c:a", "aac", self.path], check=True)

    def tearDown(self):
        self.env.stop()
        hearing.MODELS = self.models
        agent.WORK, recording.WORK = self.work
        self.meta.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_without_the_model_here_meta_hears_only_with_the_switch_on_and_its_cost_is_written(self):
        self.assertEqual(hearing.engine(), "local", "until the person turns the switch on, no sound goes to Meta by itself")
        with mock.patch.object(hearing, "_health", lambda: None):
            self.assertIn("Or turn on Meta sees pieces in Options", hearing.status()["why"])
        sends.allow(True)
        self.assertEqual(hearing.engine(), "meta")
        self.assertTrue(hearing.status()["ok"])
        self.assertEqual(hearing.status()["where"], "Muse Voice Transcribe on Meta's servers, with your own key")
        with mock.patch.dict(os.environ, {"LCS_HEARING": "local"}):
            self.assertEqual(hearing.engine(), "local", "LCS_HEARING keeps the one here")
        with mock.patch.dict(os.environ, {"META_API_KEY": ""}):
            self.assertEqual(hearing.engine(), "local", "no key, nothing goes to Meta")
        with mock.patch.object(hearing, "URL", "http://127.0.0.1:1"):
            self.assertEqual(hearing.engine(), "local", "a hearing worker of one's own is here, so a test's sound "
                                                        "never goes to Meta")
        rec = {"path": self.path, "name": "talk.mp4"}
        chat = agent.Chat()
        chat.jobs += 1
        job = agent.Job(chat, ScriptedMind(), "hear it")
        said = job.hear(rec, 2.0, 6.0)
        self.assertEqual(said[0]["from"], 2.1, "in seconds of the recording")
        self.assertEqual(said[0]["sure"], 1.0)
        with open(os.path.join(recording.WORK, "spent.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual([(r["mind"], r["label"]) for r in rows], [("voice", "Muse Voice Transcribe")])
        self.assertGreater(rows[0]["usd"], 0)
        local_key = None
        with mock.patch.dict(os.environ, {"LCS_HEARING": "local"}):
            local_key = hearing.key(rec, 2.0, 6.0)
        self.assertNotEqual(hearing.key(rec, 2.0, 6.0), local_key, "what each hearing heard is kept apart")
