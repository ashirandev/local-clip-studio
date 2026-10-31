"""The person's mask, against a fake SAM worker and a fake llama-server router: python -m unittest discover -s tests"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import cutout, mind, person, recording  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
MODEL = "muse-glimmer"

# The fake worker speaks cut.py's language: the job on stdin, one JSON line per step on stdout. LCS_FAKE_SAM says how
# it behaves, and it writes "worker start" and "worker end" into the shared event file, as the router writes its own.
FAKE = r'''
import json, os, subprocess, sys, time, urllib.request
job = json.loads(sys.stdin.read())
mode, events = os.environ["LCS_FAKE_SAM"], os.environ["LCS_FAKE_EVENTS"]
def note(what):
    with open(events, "a") as f:
        f.write(what + "\n")
def say(**what):
    print(json.dumps(what), flush=True)
note("worker start")
if job.get("prompt"):
    note("find " + job["prompt"])
say(stage="pictures")
if mode == "silent":
    note("worker end")
    sys.exit(3)
if mode == "fail":
    with open(job["out"], "wb") as f:
        f.write(b"half a mask")
    say(ok=False, error="SAM found no person in the stretch.")
    note("worker end")
    sys.exit(1)
if mode == "hang":
    time.sleep(60)
if mode == "glimmer":
    req = urllib.request.Request(os.environ["LCS_FAKE_ROUTER"] + "/v1/chat/completions", b"{}",
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()
n = int(round((job["to"] - job["from"]) * job["fps"]))
for k in range(10, n + 1, 10):
    say(done=k, of=n)
w, h = job["size"]
def white(out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=white:s=%dx%d:r=%g:d=%g,format=gray"
                    % (w, h, job["fps"], n / float(job["fps"])), "-c:v", "ffv1", "-pix_fmt", "gray", "-f", "matroska",
                    out], check=True)
white(job["out"])
each = {}
for k in range(1, len(job.get("people") or []) + 1):
    each[str(k)] = None if str(k) in os.environ.get("LCS_FAKE_LOST", "").split(",") else k
    if each[str(k)] is not None:
        white(os.path.join(job["each"], "%d.mkv" % k))
if job.get("each"):
    white(os.path.join(job["each"], "everyone.mkv"))
note("worker end")
say(ok=True, frames=n, people=[{"id": 0, "sure": 0.97}], kept=0, each=each, seconds={})
'''


class Router:
    """A fake llama-server in router mode: Glimmer loaded or unloaded, each load in a new llama-server on a new port.
    router=False answers like a server that runs one model; autoload=True loads Glimmer again at once after an
    unload, as a question sent at that moment would; a question (POST /v1/chat/completions) loads it when it is not
    loaded; load_fails makes loads fail."""

    def __init__(self, events, router=True, autoload=False, load_fails=False):
        self.events, self.router, self.autoload, self.load_fails = events, router, autoload, load_fails
        self.state, self.port = "loaded", 50001
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, data):
                body = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path != "/models":
                    return self.send(404, {"error": "no"})
                row = {"id": MODEL, "object": "model"}
                if fake.router:
                    row["status"] = {"value": fake.state, "args": ["llama-server", "--port", str(fake.port)]} \
                        if fake.state == "loaded" else {"value": fake.state}
                self.send(200, {"data": [row]})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if not fake.router:
                    return self.send(404, {"error": "File Not Found"})
                if self.path == "/models/unload":
                    fake.note("unload")
                    fake.state = "unloaded"
                    if fake.autoload:
                        fake.load("load by a question")
                    return self.send(200, {"success": True})
                if self.path == "/models/load":
                    if fake.load_fails:
                        fake.note("load failed")
                        return self.send(500, {"error": {"message": "out of memory"}})
                    if fake.state == "loaded":
                        return self.send(400, {"error": {"code": 400, "message": "model is already running"}})
                    fake.load("load")
                    return self.send(200, {"success": True})
                if self.path == "/v1/chat/completions":
                    if fake.state != "loaded":
                        fake.load("load by a question")
                    return self.send(200, {"choices": []})
                self.send(404, {"error": "no"})

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.http.server_port

    def note(self, what):
        with open(self.events, "a") as f:
            f.write(what + "\n")

    def load(self, what):
        self.note(what)
        self.state, self.port = "loaded", self.port + 1

    def close(self):
        self.http.shutdown()
        self.http.server_close()


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class CutOut(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lcs_person_")
        self.events = os.path.join(self.tmp, "events.txt")
        open(self.events, "w").close()
        worker = os.path.join(self.tmp, "fake_cut.py")
        with open(worker, "w", encoding="utf-8") as f:
            f.write(FAKE)
        weights = os.path.join(self.tmp, "sam3.1_multiplex_fp16.safetensors")
        with open(weights, "wb") as f:
            f.write(b"weights")
        video = os.path.join(self.tmp, "talk.mkv")
        with open(video, "wb") as f:
            f.write(b"a recording")
        self.rec = {"path": video, "name": "talk.mkv", "seconds": 20.0, "width": 64, "height": 36}
        self.router = Router(self.events)
        os.environ["LCS_FAKE_SAM"], os.environ["LCS_FAKE_EVENTS"] = "ok", self.events
        os.environ["LCS_FAKE_ROUTER"] = self.router.url
        for patch in (mock.patch.object(person, "PYTHON", sys.executable), mock.patch.object(person, "WORKER", worker),
                      mock.patch.object(person, "WEIGHTS", weights),
                      mock.patch.object(recording, "WORK", os.path.join(self.tmp, "work")),
                      mock.patch.object(mind.MINDS["local"], "url", self.router.url + "/v1"),
                      mock.patch.object(mind.MINDS["local"], "model_id", MODEL),
                      mock.patch.object(person, "STEP_ASIDE_SECONDS", 3),
                      mock.patch.object(person, "COME_BACK_SECONDS", 3)):
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.router.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def said(self):
        with open(self.events) as f:
            return f.read().split("\n")[:-1]

    def masks(self):
        folder = os.path.join(self.tmp, "work", "masks")
        return sorted(os.listdir(folder)) if os.path.isdir(folder) else []

    def test_glimmer_steps_off_the_card_while_sam_runs_and_comes_back(self):
        progress = []
        made = person.cut_out(self.rec, 2, 5, face=[20, 10, 8, 10], on_progress=lambda d, n: progress.append((d, n)))
        self.assertEqual(self.said(), ["unload", "worker start", "worker end", "load"])
        self.assertEqual(self.router.state, "loaded")
        self.assertEqual({k: made[k] for k in ("person", "fps", "size", "from", "to")},
                         {"person": [20.0, 10.0, 8.0, 10.0], "fps": 10, "size": [64, 36], "from": 2.0, "to": 5.0})
        self.assertEqual(cutout._checked(made, [64, 36]), (made["path"], 10.0, 2.0, 5.0))
        frames = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames",
                                 "-of", "csv=p=0", made["path"]], capture_output=True, text=True).stdout.strip()
        self.assertEqual(frames, "30")
        self.assertEqual(progress, [(10, 30), (20, 30), (30, 30)])
        self.assertEqual([n for n in self.masks() if not n.endswith(".log")], [os.path.basename(made["path"])])

    def test_a_mask_made_once_is_given_back_at_once_and_glimmer_stays(self):
        first = person.cut_out(self.rec, 2, 5)
        open(self.events, "w").close()
        started = time.time()
        self.assertEqual(person.cut_out(self.rec, 2, 5), first)
        self.assertLess(time.time() - started, 0.5)
        self.assertEqual(self.said(), [])
        self.assertNotEqual(person.cut_out(self.rec, 2, 5, face=[20, 10, 8, 10])["path"], first["path"])
        got = []
        with person._lock:
            asking = threading.Thread(target=lambda: got.append(person.cut_out(self.rec, 2, 5)), daemon=True)
            asking.start()
            asking.join(2)
        self.assertEqual(got, [first], "a kept mask does not wait for another cut-out to finish")

    def real_video(self):
        """The stretch that goes up to Meta must be a real video, not the empty file the fake SAM here is given."""
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=64x36:r=10:d=6", "-c:v",
                        "libx264", "-pix_fmt", "yuv420p", self.rec["path"]], check=True)

    def test_with_no_sam_here_the_stretch_goes_to_metas_sam_and_comes_back_a_mask(self):
        """API computers have no card and no SAM: Tarm's seller 10-31 lost 3 of 4 clips for it."""
        self.real_video()
        api = FakeSam(picture=(64, 36))
        self.addCleanup(api.close)
        with mock.patch.object(person, "PYTHON", ""), mock.patch.object(person, "API", api.url), \
                mock.patch.object(person, "WORKER", os.path.join(ROOT, "cutting", "cut.py")), \
                mock.patch.dict(os.environ, {"META_API_KEY": "the key"}):
            with self.assertRaises(person.NotSetUp):
                person.cut_out(self.rec, 2, 5)
            with mock.patch.dict(os.environ, {"LCS_SAM": "local"}), self.assertRaises(person.NotSetUp):
                person.cut_out(self.rec, 2, 5, send=True)
            paid = []
            made = person.cut_out(self.rec, 2, 5, what="white game controller", send=True, on_paid=paid.append)
        self.assertEqual(self.said(), [], "Glimmer stays on the card: nothing runs here")
        self.assertTrue(os.path.isfile(made["path"]), "the same file as SAM's here")
        self.assertEqual(cutout._checked(made, [64, 36])[1:], (10.0, 2.0, 5.0))
        self.assertEqual(paid, [50 * 0.0002], "30 pictures, and 50 are the least billed")
        self.assertEqual(api.prompts, ["white game controller"])
        self.assertEqual(api.keys, {"Bearer the key"})
        self.assertEqual(api.deleted, ["file-1"], "the video does not stay on Meta's side")

    def test_a_refusal_from_metas_sam_says_so(self):
        self.real_video()
        api = FakeSam(picture=(64, 36), refuse=403)
        self.addCleanup(api.close)
        with mock.patch.object(person, "PYTHON", ""), mock.patch.object(person, "API", api.url), \
                mock.patch.object(person, "WORKER", os.path.join(ROOT, "cutting", "cut.py")), \
                mock.patch.dict(os.environ, {"META_API_KEY": "the key"}):
            with self.assertRaisesRegex(person.PersonError, r"refused the video \(403: the Meta key was not accepted\)"):
                person.cut_out(self.rec, 2, 5, send=True)

    def test_each_person_of_a_talk_gets_their_own_mask_from_the_same_run(self):
        """Tarm 10-30: in a talk the one speaking keeps their colour, so each person needs a mask of their own."""
        os.environ["LCS_FAKE_LOST"] = "2"
        self.addCleanup(os.environ.pop, "LCS_FAKE_LOST", None)
        people = [[10, 10, 8, 10], [40, 10, 8, 10]]
        made = person.cut_out(self.rec, 2, 5, people=people)
        self.assertEqual(self.said(), ["unload", "worker start", "worker end", "load"], "one SAM run for everyone")
        self.assertTrue(os.path.isfile(made["each"][0]))
        self.assertEqual(made["each"][1:], [None], "SAM found nobody at person 2's face")
        self.assertTrue(os.path.isfile(made["everyone"]), "and everyone it followed stands in one mask of their own")
        self.assertEqual(cutout._checked(dict(made, path=made["each"][0]), [64, 36])[1:], (10.0, 2.0, 5.0))
        open(self.events, "w").close()
        self.assertEqual(person.cut_out(self.rec, 2, 5, people=people), made, "given back at once")
        self.assertEqual(self.said(), [])
        alone = person.cut_out(self.rec, 2, 5)
        self.assertNotIn("each", alone)
        self.assertNotIn("everyone", alone)
        self.assertNotEqual(alone["path"], made["path"], "a mask without each person's own is another mask")
        self.assertEqual([n for n in self.masks() if "part" in n or "pictures" in n], [], "nothing half made is left")
        with self.assertRaisesRegex(ValueError, r"^people \[\] is not a list of face boxes"):
            person.cut_out(self.rec, 2, 5, people=[])
        with self.assertRaisesRegex(ValueError, r"^face \[1, 2\] is not a face box on talk\.mkv"):
            person.cut_out(self.rec, 2, 5, people=[[10, 10, 8, 10], [1, 2]])

    def test_sam_finds_the_product_named_instead_of_a_person_in_a_mask_of_its_own(self):
        """Tarm 10-30, on a seller clip: "โปรแกรมต้องฉลาดพอจะรู้ว่าต้องโฟกัสสินค้ามากกว่าคน"."""
        made = person.cut_out(self.rec, 2, 5, what="white game controller")
        self.assertEqual(self.said(), ["unload", "worker start", "find white game controller", "worker end", "load"])
        self.assertNotEqual(person.cut_out(self.rec, 2, 5)["path"], made["path"], "the person's mask is another")
        open(self.events, "w").close()
        self.assertEqual(person.cut_out(self.rec, 2, 5, what="white game controller"), made, "given back at once")
        self.assertEqual(self.said(), [])

    def test_glimmer_comes_back_when_sam_fails_and_the_sentence_says_why(self):
        os.environ["LCS_FAKE_SAM"] = "fail"
        with self.assertRaisesRegex(person.PersonError, "^SAM found no person in the stretch.$"):
            person.cut_out(self.rec, 2, 5)
        self.assertEqual(self.said(), ["unload", "worker start", "worker end", "load"])
        self.assertEqual([n for n in self.masks() if not n.endswith(".log")], [])

    def test_glimmer_comes_back_when_sam_dies_without_a_word(self):
        os.environ["LCS_FAKE_SAM"] = "silent"
        with self.assertRaisesRegex(person.PersonError, r"SAM stopped without saying why \(exit code 3\); its log"):
            person.cut_out(self.rec, 2, 5)
        self.assertEqual(self.said()[-1], "load")

    def test_stop_ends_sam_at_once_and_glimmer_comes_back(self):
        os.environ["LCS_FAKE_SAM"] = "hang"
        stops = []

        def press():
            until = time.time() + 20
            while "worker start" not in self.said() and time.time() < until:
                time.sleep(0.05)
            stops[0]()

        threading.Thread(target=press, daemon=True).start()
        started = time.time()
        with self.assertRaises(person.Stopped):
            person.cut_out(self.rec, 2, 5, on_start=stops.append)
        self.assertLess(time.time() - started, 15)
        self.assertEqual(self.said(), ["unload", "worker start", "load"])
        self.assertEqual([n for n in self.masks() if not n.endswith(".log")], [])

    def test_a_slow_sam_is_stopped_and_glimmer_comes_back(self):
        os.environ["LCS_FAKE_SAM"] = "hang"
        with mock.patch.object(person, "LOAD_SECONDS", 1), mock.patch.object(person, "SECONDS_A_PICTURE", 0.01):
            with self.assertRaisesRegex(person.PersonError, "SAM took more than 1 s for 3.0 s of the recording"):
                person.cut_out(self.rec, 2, 5)
        self.assertEqual(self.said(), ["unload", "worker start", "load"])

    def test_glimmer_loaded_by_a_question_while_sam_ran_is_loaded_again_on_a_free_card(self):
        os.environ["LCS_FAKE_SAM"] = "glimmer"
        person.cut_out(self.rec, 2, 5)
        self.assertEqual(self.said(), ["unload", "worker start", "load by a question", "worker end", "unload", "load"])
        self.assertEqual((self.router.state, self.router.port), ("loaded", 50003))

    def test_when_a_question_loads_glimmer_again_at_once_sam_does_not_start(self):
        self.router.autoload = True
        with self.assertRaisesRegex(person.PersonError, "loaded again at once.*--no-models-autoload"):
            person.cut_out(self.rec, 2, 5)
        self.assertNotIn("worker start", self.said())
        self.assertEqual(self.router.state, "loaded")

    def test_a_llama_server_without_router_mode_is_refused_before_sam_runs(self):
        self.router.router = False
        with self.assertRaisesRegex(person.PersonError, "without router mode.*--models-preset"):
            person.cut_out(self.rec, 2, 5)
        self.assertEqual(self.said(), [])

    def test_glimmer_off_the_card_is_loaded_after_and_with_no_muse_here_sam_runs_alone(self):
        self.router.state = "unloaded"
        person.cut_out(self.rec, 2, 5)
        self.assertEqual(self.said(), ["worker start", "worker end", "load"])
        self.assertEqual(self.router.state, "loaded")
        self.router.close()
        open(self.events, "w").close()
        person.cut_out(self.rec, 6, 8)
        with mock.patch.object(mind.MINDS["local"], "url", "http://192.0.2.1:8863/v1"):
            person.cut_out(self.rec, 9, 11)
        self.assertEqual(self.said(), ["worker start", "worker end"] * 2)

    def test_a_muse_on_another_machine_is_left_alone(self):
        with mock.patch.object(person, "HERE", ("localhost",)):
            person.cut_out(self.rec, 2, 5)
        self.assertEqual(self.said(), ["worker start", "worker end"])
        self.assertEqual(self.router.state, "loaded")

    def test_a_glimmer_that_cannot_come_back_says_so_after_what_sam_said(self):
        os.environ["LCS_FAKE_SAM"] = "fail"
        self.router.load_fails = True
        with self.assertRaisesRegex(person.PersonError, r"^SAM found no person in the stretch\. Then Muse Glimmer "
                                                        r"could not come back onto the graphics card after SAM"):
            person.cut_out(self.rec, 2, 5)
        os.environ["LCS_FAKE_SAM"] = "ok"
        with self.assertRaisesRegex(person.PersonError, "^Muse Glimmer could not come back.*Any mask made is kept"):
            person.cut_out(self.rec, 2, 5)
        self.router.load_fails = False
        open(self.events, "w").close()
        person.cut_out(self.rec, 2, 5)
        self.assertEqual(self.said(), [])

    def test_what_cannot_be_cut_says_what_to_send(self):
        for start, end, said in ((5, 2, "from 5 to 2 is not a stretch of talk.mkv: give recording seconds from 0 to "
                                           "20.0, to after from."),
                                 (-1, 2, "from -1 to 2 is not a stretch"), (2, 30, "from 2 to 30 is not a stretch"),
                                 (True, 2, "from True to 2 is not a stretch")):
            with self.assertRaisesRegex(ValueError, "^" + said.replace("(", r"\(").replace(".", r"\.")):
                person.cut_out(self.rec, start, end)
        long_rec = dict(self.rec, seconds=600.0)
        with self.assertRaisesRegex(ValueError, "^from 0 to 121 is 121.0 s: the cut-out follows a person for at most "
                                                "120 s at a time"):
            person.cut_out(long_rec, 0, 121)
        for face in ([1, 2, 3], [10, 10, 0, 5], [100, 10, 8, 8], "face", [1, 2, 3, True]):
            with self.assertRaisesRegex(ValueError, r"^face .* is not a face box on talk\.mkv: give \[x, y, w, h\] in "
                                                    r"its pixels, the middle inside its 64x36 picture\.$"):
                person.cut_out(self.rec, 2, 5, face=face)
        with mock.patch.object(person, "PYTHON", os.path.join(self.tmp, "none.exe")):
            with self.assertRaisesRegex(person.PersonError, "SAM 3.1 is not set up: LCS_SAM_PYTHON"):
                person.cut_out(self.rec, 2, 5)
            self.assertFalse(person.status()["ok"])
        with mock.patch.object(person, "WEIGHTS", ""):
            with self.assertRaisesRegex(person.PersonError, "SAM 3.1 is not set up: LCS_SAM_WEIGHTS"):
                person.cut_out(self.rec, 2, 5)
        self.assertEqual(self.said(), [])
        self.assertTrue(person.status()["ok"])


def sam_line(at, objects):
    """One line as the API writes it: objects are (id, x, y, w, h), all of each box filled."""
    from meta_sam_parser._mask_codec import _encode_raster
    parts = ["%d<|box;x1=%d;y1=%d;x2=%d;y2=%d;w=64;h=36|><|mask;x=0;y=0;data=%d,%d,%s|>" % (
        i, x, y, x + w - 1, y + h - 1, h, w, _encode_raster(bytes([1]) * (w * h), w, h)) for i, x, y, w, h in objects]
    return "<%df>%s" % (at, ",".join(parts))


class FakeSam:
    """Meta's Model API as SAM 3.1 uses it: a file goes up, an answer streams, the file is deleted."""

    def __init__(self, picture, refuse=None):
        self.prompts, self.keys, self.deleted = [], set(), []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, body, kind="application/json"):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_DELETE(self):
                fake.deleted.append(self.path.rsplit("/", 1)[1])
                self.send(b"{}")

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                fake.keys.add(self.headers["Authorization"])
                if refuse:
                    self.send_response(refuse)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path.endswith("/files"):
                    self.send(json.dumps({"id": "file-1", "status": "uploaded"}).encode())
                else:
                    fake.prompts.append(json.loads(body)["input"][0]["content"][0]["text"])
                    events = [{"type": "response.output_text.delta", "delta": sam_line(k, [(3, 10, 5, 8, 6)]) + "\n"}
                              for k in range(30)]
                    events.append({"type": "response.completed", "response": {"usage": {"video_frames_processed": 30}}})
                    self.send(("".join("data: %s\n\n" % json.dumps(e) for e in events) + "data: [DONE]\n\n").encode(),
                              "text/event-stream")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d/v1" % self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def worker_module():
    spec = importlib.util.spec_from_file_location("cut", os.path.join(ROOT, "cutting", "cut.py"))
    cut = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cut)
    return cut


@unittest.skipUnless(HAS_NUMPY, "needs numpy")
class Worker(unittest.TestCase):
    def test_the_person_holding_the_face_is_kept_else_the_biggest(self):
        import numpy as np
        cut = worker_module()

        def box(x0, x1):
            m = np.zeros((36, 64), bool)
            m[5:30, x0:x1] = True
            return np.packbits(m, axis=-1)

        masks = {0: [box(0, 30)] * 4, 1: [None, box(40, 50), box(40, 50), None]}
        self.assertEqual(cut.pick(masks, [42, 10, 6, 8], (128, 72), (64, 36)), 0)
        self.assertEqual(cut.pick(masks, [84, 20, 12, 16], (128, 72), (64, 36)), 1)
        self.assertEqual(cut.pick(masks, None, (128, 72), (64, 36)), 0)
        with self.assertRaisesRegex(ValueError, r"^None of the 2 people SAM found holds the face at \[120, 2, 4, 4\]\.$"):
            cut.pick(masks, [120, 2, 4, 4], (128, 72), (64, 36))
        with self.assertRaisesRegex(ValueError, "^SAM found no person in the stretch.$"):
            cut.pick({}, None, (128, 72), (64, 36))
        with self.assertRaisesRegex(ValueError, "^SAM found no white game controller in the stretch.$"):
            cut.pick({}, None, (128, 72), (64, 36), "white game controller")
        self.assertEqual(cut.holders(masks, [[42, 10, 6, 8], [84, 20, 12, 16], [120, 2, 4, 4]], (128, 72), (64, 36)),
                         {"1": 0, "2": 1, "3": None}, "each face its own person; a face nobody holds, none")
        self.assertEqual(cut.holders(masks, [], (128, 72), (64, 36)), {})

    def test_a_person_sam_lost_and_found_again_stays_one_mask(self):
        import numpy as np
        cut = worker_module()

        def box(x0, x1):
            m = np.zeros((36, 64), bool)
            m[5:30, x0:x1] = True
            return np.packbits(m, axis=-1)

        a, b, c = box(0, 10), box(20, 30), box(40, 50)
        empty = np.packbits(np.zeros((36, 64), bool), axis=-1)
        masks = {0: [a, a, empty, None], 2: [None, None, b, b], 3: [c, None, None, c]}
        got = cut.joined(masks, 2)
        self.assertTrue(got[0] is a and got[1] is a, "the pictures before SAM found him again come from his first id")
        self.assertTrue(got[2] is b and got[3] is b, "the kept id stands where it has him")
        beside = cut.joined({2: [a, None], 3: [b, b]}, 2)
        self.assertIs(beside[0], a)
        self.assertIsNone(beside[1], "an id seen in a picture beside the kept one is someone else")
        brief = cut.joined({2: [a] * 6 + [None] * 4, 3: [None] * 5 + [b] * 5}, 2)
        self.assertTrue(all(m is a for m in brief[:6]) and all(m is b for m in brief[6:]),
                        "two ids of one person may touch for a moment as SAM hands over, and the kept one stands there")
        crumb = box(0, 1)
        lost = cut.joined({0: [a] * 6 + [None, crumb] + [None] * 12, 1: [None] * 6 + [b] * 6 + [None] * 8,
                           2: [None] * 12 + [c] * 8}, 2)
        self.assertIs(lost[7], b, "a crumb the first id left behind gives way to the person")
        self.assertTrue(all(m is not None for m in lost), "three ids of one person leave no picture empty")

    def test_everyone_sam_followed_stands_in_one_mask(self):
        import numpy as np
        cut = worker_module()
        a, b = np.zeros((4, 16), bool), np.zeros((4, 16), bool)
        a[1, :4], b[2, 8:] = True, True
        pa, pb = np.packbits(a, axis=-1), np.packbits(b, axis=-1)
        got = cut.everyone({0: [pa, pa, None, None], 3: [pb, None, pb, None]})
        np.testing.assert_array_equal(np.unpackbits(got[0], axis=-1)[:, :16], a | b, "both in the picture they share")
        np.testing.assert_array_equal(got[1], pa)
        np.testing.assert_array_equal(got[2], pb)
        self.assertIsNone(got[3], "a picture with nobody in it stays empty")

    def test_the_answers_masks_stand_where_their_boxes_are(self):
        import numpy as np
        cut = worker_module()
        at, found = cut.found_in(sam_line(4, [(7, 10, 5, 8, 6)]), (64, 36))
        self.assertEqual(at, 4)
        got = np.unpackbits(found[7], axis=-1)[:, :64]
        self.assertEqual(int(got.sum()), 48)
        self.assertEqual(got[5:11, 10:18].min(), 1, "the mask at its box")
        self.assertEqual(cut.found_in("not a frame", (64, 36)), (None, {}))
        at, found = cut.found_in(sam_line(1, [(1, 60, 30, 8, 8)]), (64, 36))
        self.assertEqual(int(np.unpackbits(found[1], axis=-1)[:, :64].sum()), 4 * 6, "a box past the edge is cut there")

    def test_the_worker_writes_the_person_joined_up(self):
        import types

        import numpy as np
        cut = worker_module()
        first, later = np.zeros((36, 64), bool), np.zeros((36, 64), bool)
        first[5:30, 0:10], later[5:30, 20:50] = True, True

        class Sam:
            def handle_request(self, ask):
                return {"session_id": "s", "outputs": {"out_obj_ids": np.array([0]), "out_probs": np.array([0.9])}}

            def handle_stream_request(self, ask):
                for n, (i, m) in enumerate([(0, first), (2, later), (2, later)]):
                    yield {"frame_index": n, "outputs": {"out_obj_ids": np.array([i]), "out_binary_masks": [m]}}

        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)

        def pictures(*a, **k):
            for n in range(3):
                open(os.path.join(folder, "%05d.jpg" % n), "wb").close()
            return types.SimpleNamespace(returncode=0, stderr="")

        with mock.patch.object(cut.subprocess, "run", pictures), mock.patch.object(cut, "load", lambda w: Sam()), \
                mock.patch.object(cut, "write") as write, mock.patch.object(cut, "say"):
            got = cut.cut({"recording": "r", "from": 0, "to": 1, "fps": 3, "size": [64, 36], "face": None, "weights": "w",
                           "frames": folder, "out": "o"})
        self.assertEqual(got["kept"], 2)
        self.assertTrue(all(m is not None for m in write.call_args[0][0]), "no picture is left without the person")
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
        with mock.patch.object(cut.subprocess, "run", pictures), mock.patch.object(cut, "load", lambda w: Sam()),                 mock.patch.object(cut, "write") as write, mock.patch.object(cut, "say"):
            cut.cut({"recording": "r", "from": 0, "to": 1, "fps": 3, "size": [64, 36], "face": None, "weights": "w",
                     "frames": folder, "out": "o", "people": [[0, 5, 10, 25]], "each": "people"})
        self.assertIn(os.path.join("people", "everyone.mkv"), [c[0][1] for c in write.call_args_list],
                      "a talk's run writes everyone it followed")

    def test_pictures_are_at_most_1920_on_the_long_side_and_even(self):
        cut = worker_module()
        self.assertEqual(cut.picture_size((3840, 2160)), (1920, 1080))
        self.assertEqual(cut.picture_size((2160, 3840)), (1080, 1920))
        self.assertEqual(cut.picture_size((1920, 1080)), (1920, 1080))
        self.assertEqual(cut.picture_size((1001, 563)), (1002, 564))


if __name__ == "__main__":
    unittest.main()
