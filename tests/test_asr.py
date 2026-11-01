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
        self.stray = self.fails = self.turns = None
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
                with wave.open(io.BytesIO(parts["audio"])) as w:
                    ms = round(1000 * w.getnframes() / w.getframerate())
                status = 500 if meta.fails and meta.fails(ms, len(meta.got)) else meta.status
                if status != 200:
                    self.send_response(status)
                    self.end_headers()
                    return
                turns = [{"turnId": 0, "startMs": 100, "endMs": ms - 100, "transcript": "พูด %d ms" % ms}]
                answer = json.dumps({"transcript": "x", "turns": meta.turns(ms, len(meta.got)) if meta.turns else turns},
                                    ensure_ascii=False).encode()
                if meta.stray:
                    answer = answer.replace("พูด".encode(), "พู".encode() + meta.stray + "ด".encode())
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

    def test_the_turns_of_a_stretch_are_one_request_each_with_its_voice(self):
        """Labels hold only inside one transcription, so the whole stretch goes as one."""
        found, usd = asr.turns(sound((40, True)))
        self.assertEqual(len(self.meta.got), 1)
        self.assertEqual(self.meta.got[0]["request"]["mode"], "DIARIZATION")
        self.assertEqual(found, [], "a turn with no speaker says nothing of who")
        self.assertAlmostEqual(usd, 40 / 3600 * 0.18, places=6)
        answer = {"turns": [{"startMs": 100, "endMs": 2100, "speaker": "A", "transcript": "x"}]}
        with mock.patch.object(asr, "_answer", lambda *a: answer):
            self.assertEqual(asr.turns(sound((3, True)))[0], [(0.1, 2.1, "A")])

    def test_a_broken_byte_in_the_answer_costs_only_that_byte(self):
        """Lane 4 10-31: "byte 0xad at position 831" lost the hearing of a whole look, 8 of 261 looks, and EP 22's joke."""
        self.meta.stray = b"\xad"
        said, _ = asr.hear(sound((3, True)))
        self.assertEqual([p["text"] for p in said], ["พูด 3000 ms"])

    def test_a_piece_that_fails_once_is_asked_again(self):
        self.meta.fails = lambda ms, n: n == 1
        said, _ = asr.hear(sound((3, True)))
        self.assertEqual((len(self.meta.got), [p["text"] for p in said]), (2, ["พูด 3000 ms"]))

    def test_a_piece_that_keeps_failing_costs_that_piece_and_the_rest_is_kept(self):
        self.meta.fails = lambda ms, n: ms > 6000
        with self.assertRaises(asr.PartHeard) as got:
            asr.hear(sound((8, True), (0.6, False), (5, True)))
        part = got.exception
        self.assertEqual([p["text"] for p in part.said], ["พูด 5300 ms"], "the other piece")
        self.assertEqual([(round(a, 1), round(b, 1)) for a, b in part.lost], [(0.0, 8.3)])
        self.assertEqual(len(self.meta.got), 3, "the failed piece twice, the other once")
        self.assertAlmostEqual(part.usd, 13.6 / 3600 * 0.18, places=5)
        self.assertIn("refused the sound (500)", str(part))

    def test_when_nothing_was_heard_it_is_not_a_part(self):
        self.meta.fails = lambda ms, n: True
        with self.assertRaises(asr.VoiceError) as got:
            asr.hear(sound((8, True), (0.6, False), (5, True)))
        self.assertNotIsInstance(got.exception, asr.PartHeard)

    def test_a_stretch_with_no_sound_in_it_asks_nothing_and_nothing_was_said(self):
        """183215-ea60: "propose_moments broke ()", as next() of no failures is a StopIteration with no words in it."""
        self.assertEqual(asr.hear(bytes(RATE * 2 * 5)), ([], 0))
        self.assertEqual(self.meta.got, [])

    def holes(self, *more):
        """A fake that hears two lines around a hole in a 25 s piece, and more when it is asked about that hole again."""
        def turns(ms, n):
            line = lambda a, b, words: {"startMs": a, "endMs": b, "transcript": words}
            return [line(0, 2000, "ก่อน"), line(12000, 14000, "หลัง")] if n == 1 else [line(*m) for m in more]

        self.meta.turns = turns

    def test_a_hole_with_sound_in_it_is_asked_again_with_the_sound_around_it_and_its_own_lines_are_kept(self):
        """081254-7a4b on 11-01: 10.9 s of laughing talk with no caption, as Voice gave nothing back after a cut inside it."""
        self.holes((0, 1000, "ก่อนอีกที"), (8000, 11000, "ในรู"), (17000, 19000, "หลังอีกที"))
        said, usd = asr.hear(sound((14.5, True)))
        self.assertEqual([p["text"] for p in said], ["ก่อน", "ในรู", "หลัง"], "only what falls in the hole, in order")
        self.assertEqual([(p["from"], p["to"]) for p in said], [(0.0, 2.0), (8.0, 11.0), (12.0, 14.0)])
        with wave.open(io.BytesIO(self.meta.got[1]["audio"])) as w:
            self.assertEqual(w.getnframes() / w.getframerate(), 13.0, "7 s before 2-12 s, as far as the sound goes, and 1 s after")
        self.assertAlmostEqual(usd, 27.5 / 3600 * 0.18, places=6)

    def test_a_hole_late_in_the_sound_is_asked_with_7_s_before_it(self):
        self.meta.turns = lambda ms, n: [{"startMs": 0, "endMs": 10000, "transcript": "ก่อน"},
                                         {"startMs": 18000, "endMs": 20000, "transcript": "หลัง"}] if n == 1 else []
        with mock.patch.object(asr, "pieces", return_value=[(0.0, 25.0)]):
            asr.hear(sound((25, True)))
        with wave.open(io.BytesIO(self.meta.got[1]["audio"])) as w:
            self.assertEqual(w.getnframes() / w.getframerate(), 16.0, "from 3 s, 7 s before the hole, to 19 s, 1 s after it")

    def test_a_line_is_held_to_its_hole(self):
        self.holes((1000, 4000, "ข้ามขอบ"))
        said, _ = asr.hear(sound((14.5, True)))
        self.assertEqual([(p["text"], p["from"], p["to"]) for p in said][1], ("ข้ามขอบ", 2.0, 4.0))

    def test_a_short_gap_and_a_quiet_one_are_not_asked_about(self):
        def turns(ms, n):
            return [{"startMs": 0, "endMs": 2000, "transcript": "ก่อน"}, {"startMs": 6500, "endMs": 8000, "transcript": "หลัง"}]

        self.meta.turns = turns
        asr.hear(sound((8.5, True)))
        self.assertEqual(len(self.meta.got), 1, "under 5 s")
        self.meta.got.clear()

        def apart(ms, n):
            return [{"startMs": 0, "endMs": 2000, "transcript": "ก่อน"}, {"startMs": 22000, "endMs": 24000, "transcript": "หลัง"}]

        self.meta.turns = apart
        with mock.patch.object(asr, "pieces", return_value=[(0.0, 25.0)]):
            asr.hear(sound((2, True), (20, False), (3, True)))
        self.assertEqual(len(self.meta.got), 1, "a pause, not a hole")

    def test_only_the_longest_holes_are_asked_about(self):
        heard = [(float(k * 20), float(k * 20 + 1 + k % 3), "x") for k in range(9)]
        quiet = asr._quiet(sound((200, True)))
        holes = asr._holes(heard, quiet)
        self.assertEqual(len(holes), asr.MOST_HOLES)
        self.assertEqual(holes, sorted(holes), "in order")
        self.assertEqual(min(y - x for x, y in holes), 18.0, "the long ones")

    def test_a_hole_asked_about_that_fails_leaves_what_was_heard(self):
        self.holes()
        self.meta.fails = lambda ms, n: n > 1
        said, usd = asr.hear(sound((14.5, True)))
        self.assertEqual([p["text"] for p in said], ["ก่อน", "หลัง"])

    def test_a_piece_that_failed_is_not_also_a_hole(self):
        self.meta.fails = lambda ms, n: ms > 6000
        with self.assertRaises(asr.PartHeard):
            asr.hear(sound((3, True), (0.6, False), (8, True), (0.6, False), (3, True)))
        self.assertEqual(len(self.meta.got), 4, "the failed piece twice, the others once, and no hole")

    def test_stop_leaves_a_hole_unasked(self):
        self.holes((8000, 11000, "ในรู"))
        said, _ = asr.hear(sound((14.5, True)), stopped=lambda: len(self.meta.got) > 0)
        self.assertEqual([p["text"] for p in said], ["ก่อน", "หลัง"])
        self.assertEqual(len(self.meta.got), 1)

    def test_stop_asks_nothing_again(self):
        self.meta.fails = lambda ms, n: True
        asked = []

        def stopped():
            asked.append(1)
            return len(asked) > 1

        with self.assertRaises(asr.VoiceError):
            asr.hear(sound((3, True)), stopped=stopped)
        self.assertEqual(len(self.meta.got), 1)

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

    def test_meta_hears_only_with_the_switch_on_and_its_cost_is_written(self):
        self.assertEqual(hearing.engine(), "local", "until the person turns the switch on, no sound goes to Meta by itself")
        with mock.patch.object(hearing, "_health", lambda: None):
            self.assertIn("Or turn on Meta sees pieces in Options", hearing.status()["why"])
        sends.allow(True)
        self.assertEqual(hearing.engine(), "meta")
        self.assertTrue(hearing.status()["ok"])
        self.assertEqual(hearing.status()["where"], "Muse Voice Transcribe on Meta's servers, with your own key")
        os.makedirs(hearing.MODELS)
        with open(os.path.join(hearing.MODELS, "letters.json"), "w") as f:
            f.write("[]")
        with mock.patch.object(hearing.ears, "model_file", lambda folder: os.path.join(folder, "model.int8.onnx")):
            self.assertEqual(hearing.engine(), "meta", "the hearing model here too: on Thai talk Meta's hears clearly "
                                                       "what it garbles (EP 0, 10-30)")
            sends.allow(False)
            self.assertEqual(hearing.engine(), "local")
            sends.allow(True)
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

    def test_the_hearing_expects_the_languages_the_person_types_in_and_keeps_them_apart(self):
        """Lane 5, 10-30: told nothing, Muse Voice Transcribe wrote 5 of 47 lines of Thai talk in Vietnamese or
        Chinese, on laughs and game noise, and the caption check passed them against the same hearing."""
        sends.allow(True)
        rec = {"path": self.path, "name": "talk.mp4"}
        chat = agent.Chat()
        chat.jobs += 1
        job = agent.Job(chat, ScriptedMind(), "cut it")
        chat.emit("person", job=None, text="cut a clip please")
        job.hear(rec, 0.0, 6.0)
        self.assertNotIn("languageBias", self.meta.got[-1]["request"], "Latin letters only: it finds the language")
        chat.emit("person", job=None, text="ตัดคลิปให้หน่อย")
        self.assertEqual(job.languages(), ("Thai", "English"))
        job.hear(rec, 0.0, 6.0)
        self.assertEqual(self.meta.got[-1]["request"]["languageBias"], ["Thai", "English"])
        self.assertNotEqual(hearing.key(rec, 0.0, 6.0, job.languages()), hearing.key(rec, 0.0, 6.0))
        with mock.patch.dict(os.environ, {"LCS_HEARING": "local"}):
            self.assertEqual(hearing.key(rec, 0.0, 6.0, job.languages()), hearing.key(rec, 0.0, 6.0),
                             "the hearing here is told no language")
        self.assertEqual(agent.heard_in(["漢字", "and ไทย"]), ("Thai", "Mandarin Chinese", "English"))
