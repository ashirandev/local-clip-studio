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

    def test_each_person_of_a_talk_gets_their_own_mask_from_the_same_run(self):
        """Tarm 10-30: in a talk the one speaking keeps their colour, so each person needs a mask of their own."""
        os.environ["LCS_FAKE_LOST"] = "2"
        self.addCleanup(os.environ.pop, "LCS_FAKE_LOST", None)
        people = [[10, 10, 8, 10], [40, 10, 8, 10]]
        made = person.cut_out(self.rec, 2, 5, people=people)
        self.assertEqual(self.said(), ["unload", "worker start", "worker end", "load"], "one SAM run for everyone")
        self.assertTrue(os.path.isfile(made["each"][0]))
        self.assertEqual(made["each"][1:], [None], "SAM found nobody at person 2's face")
        self.assertEqual(cutout._checked(dict(made, path=made["each"][0]), [64, 36])[1:], (10.0, 2.0, 5.0))
        open(self.events, "w").close()
        self.assertEqual(person.cut_out(self.rec, 2, 5, people=people), made, "given back at once")
        self.assertEqual(self.said(), [])
        alone = person.cut_out(self.rec, 2, 5)
        self.assertNotIn("each", alone)
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

    def test_pictures_are_at_most_1920_on_the_long_side_and_even(self):
        cut = worker_module()
        self.assertEqual(cut.picture_size((3840, 2160)), (1920, 1080))
        self.assertEqual(cut.picture_size((2160, 3840)), (1080, 1920))
        self.assertEqual(cut.picture_size((1920, 1080)), (1920, 1080))
        self.assertEqual(cut.picture_size((1001, 563)), (1002, 564))


if __name__ == "__main__":
    unittest.main()
