"""Setting up this computer's parts, with small made-up files from a server on this computer: python -m unittest
discover -s tests"""
import hashlib
import http.client
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import hearing, parts, server  # noqa: E402

BIG = {"name": "NVIDIA GeForce RTX 5070 Ti", "mib": 16303, "cc": (12, 0), "cuda": (13, 0)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Files(BaseHTTPRequestHandler):
    """Serves self.server.files by path. ranges=False answers every request whole; cut=n closes after n bytes."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        data = self.server.files.get(self.path)
        if data is None:
            self.send_error(404)
            return
        asked = self.headers.get("Range")
        self.server.ranges_asked.append(asked)
        start = int(asked[6:-1]) if asked and self.server.ranges else 0
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        body = data[start:]
        if self.server.cut is not None:
            body = body[:self.server.cut]
        self.wfile.write(body)
        self.close_connection = True


class Fetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Files)
        self.http.files, self.http.ranges_asked, self.http.ranges, self.http.cut = {}, [], True, None
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.blob = os.urandom(300_000)
        self.http.files["/blob.bin"] = self.blob
        self.url = "http://127.0.0.1:%d/blob.bin" % self.http.server_port
        self.dest = os.path.join(self.tmp, "models", "blob.bin")

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_cut_off_download_carries_on_from_where_it_stopped(self):
        os.makedirs(os.path.dirname(self.dest))
        with open(self.dest + ".part", "wb") as f:
            f.write(self.blob[:100_000])
        parts.fetch(self.url, len(self.blob), sha(self.blob), self.dest)
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.blob)
        self.assertEqual(self.http.ranges_asked, ["bytes=100000-"])
        self.assertFalse(os.path.exists(self.dest + ".part"))

    def test_a_connection_that_closes_early_keeps_what_came_and_says_to_press_set_up_again(self):
        self.http.cut = 120_000
        with self.assertRaises(parts.PartError) as said:
            parts.fetch(self.url, len(self.blob), sha(self.blob), self.dest)
        self.assertIn("Press Set up again: it carries on from where it stopped", str(said.exception))
        self.assertEqual(os.path.getsize(self.dest + ".part"), 120_000)
        self.assertFalse(os.path.exists(self.dest))
        self.http.cut = None
        parts.fetch(self.url, len(self.blob), sha(self.blob), self.dest)
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.blob)

    def test_a_file_that_does_not_match_is_deleted_and_never_used(self):
        with self.assertRaises(parts.PartError) as said:
            parts.fetch(self.url, len(self.blob), sha(b"something else"), self.dest)
        self.assertIn("did not match the file the studio expects, so it was deleted", str(said.exception))
        self.assertFalse(os.path.exists(self.dest))
        self.assertFalse(os.path.exists(self.dest + ".part"))

    def test_a_server_that_ignores_the_range_starts_the_file_over(self):
        self.http.ranges = False
        os.makedirs(os.path.dirname(self.dest))
        with open(self.dest + ".part", "wb") as f:
            f.write(b"x" * 50_000)
        parts.fetch(self.url, len(self.blob), sha(self.blob), self.dest)
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.blob)

    def test_a_file_not_at_its_address_yet_says_so_and_nothing_is_kept(self):
        url = "http://127.0.0.1:%d/not-yet.bin" % self.http.server_port
        with self.assertRaises(parts.PartError) as said:
            parts.fetch(url, 10, sha(b"0123456789"), self.dest)
        self.assertEqual(str(said.exception), "blob.bin is not at %s yet, so it was left out." % url)
        self.assertFalse(os.path.exists(self.dest + ".part"))

    def test_a_part_that_fails_does_not_stop_the_parts_after_it(self):
        done = []

        def part(key, kind):
            if key == "hearing":
                raise parts.PartError("model.int8.onnx is not at its address yet, so it was left out.")
            done.append(key)
        with mock.patch.object(parts, "_set_up_part", part), mock.patch.object(parts, "start_glimmer") as started:
            parts._set_up([{"key": "ffmpeg"}, {"key": "hearing"}, {"key": "glimmer"}], "cuda13")
        self.assertEqual(done, ["ffmpeg", "glimmer"])
        self.assertTrue(started.called)
        self.assertEqual(parts.state()["setup"]["error"],
                         "model.int8.onnx is not at its address yet, so it was left out.")
        self.assertFalse(parts.state()["setup"]["running"])

    def test_zips_are_unpacked_beside_their_folder_and_moved_in_whole(self):
        made = io.BytesIO()
        with zipfile.ZipFile(made, "w") as z:
            z.writestr("ffmpeg-8.1.1-essentials_build/bin/ffmpeg.exe", b"ffmpeg")
            z.writestr("ffmpeg-8.1.1-essentials_build/bin/ffprobe.exe", b"ffprobe")
        data = made.getvalue()
        self.http.files["/ffmpeg.zip"] = data
        url = "http://127.0.0.1:%d/ffmpeg.zip" % self.http.server_port
        folder = os.path.join(self.tmp, "tools", "ffmpeg")
        with mock.patch.object(parts, "TOOLS", os.path.join(self.tmp, "tools")):
            parts._unzip_all([(url, len(data), sha(data))], folder, strip=True)
        self.assertEqual(sorted(os.listdir(os.path.join(folder, "bin"))), ["ffmpeg.exe", "ffprobe.exe"])
        self.assertEqual(sorted(os.listdir(os.path.join(self.tmp, "tools"))), ["downloads", "ffmpeg"])
        self.assertEqual(os.listdir(os.path.join(self.tmp, "tools", "downloads")), [], "the zip is not kept")


class Card(unittest.TestCase):
    def test_the_card_and_its_driver_pick_the_llama_cpp_build(self):
        def gpu(mib=16376, cc=(8, 9), cuda=(12, 6)):
            return {"name": "NVIDIA GeForce RTX 4080", "mib": mib, "cc": cc, "cuda": cuda}
        self.assertEqual(parts.build(BIG), ("cuda13", None))
        self.assertEqual(parts.build(gpu()), ("cuda12", None))
        self.assertEqual(parts.build(gpu(cuda=(13, 4))), ("cuda13", None))
        self.assertEqual(parts.build(dict(BIG, cuda=(12, 8)))[0], None, "an RTX 50 card has no CUDA 12.4 kernels")
        self.assertIn("Update the NVIDIA driver", parts.build(gpu(cuda=(12, 2)))[1])
        self.assertEqual(parts.build(gpu(mib=12282))[1],
                         "Muse Glimmer needs a graphics card with 16 GB; the NVIDIA GeForce RTX 4080 has 12 GB.")
        self.assertIn("no NVIDIA graphics card", parts.build(None)[1])

    def test_nvidia_smi_says_the_driver_cuda_both_ways_drivers_write_it(self):
        self.assertEqual(parts.CUDA.search("| NVIDIA-SMI 617.14   KMD Version: 617.14   CUDA UMD Version: 13.4  |")
                         .groups(), ("13", "4"))
        self.assertEqual(parts.CUDA.search("| NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.6  |")
                         .groups(), ("12", "6"))


class Machine(unittest.TestCase):
    """A computer where nothing runs: the local address and the hearing answer nothing, nothing is in work/."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.had = {n: os.environ.pop(n, None) for n in ("LCS_MIND_URL", "LCS_GLIMMER_FILE", "LCS_LLAMA_SERVER")}
        self.patches = [mock.patch.object(parts, "LOCAL_URL", "http://127.0.0.1:%d/v1" % free_port()),
                        mock.patch.object(hearing, "URL", "http://127.0.0.1:%d" % free_port()),
                        mock.patch.object(parts, "WORK", self.tmp),
                        mock.patch.object(parts, "TOOLS", os.path.join(self.tmp, "tools")),
                        mock.patch.object(parts, "LLAMA_DIR", os.path.join(self.tmp, "tools", "llama")),
                        mock.patch.object(parts, "GLIMMER_FILE", os.path.join(self.tmp, "glimmer.gguf")),
                        mock.patch.object(parts, "VISION_FILE", os.path.join(self.tmp, "vision.gguf")),
                        mock.patch.object(parts, "HEARING_DIR", os.path.join(self.tmp, "hearing")),
                        mock.patch.object(parts, "HEARING_FROM", "https://example.invalid/hearing")]
        for p in self.patches:
            p.start()
        self.which = mock.patch.object(parts.shutil, "which", lambda name: None)
        self.which.start()

    def tearDown(self):
        self.which.stop()
        parts.stop_glimmer()
        parts._engine.update(process=None, why=None)
        for p in self.patches:
            p.stop()
        for n, v in self.had.items():
            os.environ.pop(n, None)
            if v is not None:
                os.environ[n] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def found(self, gpu):
        return {p["key"]: p for p in parts.parts(gpu, look=False)}

    def test_a_16_gb_card_downloads_every_part(self):
        got = self.found(BIG)
        self.assertEqual({k: p["ok"] for k, p in got.items()},
                         {"ffmpeg": False, "engine": False, "glimmer": False, "hearing": False})
        self.assertEqual(got["glimmer"]["size"], 13360983072 + 1400328928, "the model and its vision part")
        self.assertEqual(got["engine"]["size"], 155071236 + 423535356, "CUDA 13.4 for an RTX 50 card")
        self.assertEqual(got["hearing"]["size"], 1031964630 + 70913)
        if parts.windows():
            self.assertEqual(got["ffmpeg"]["size"], 109282242)

    def test_glimmer_counts_as_here_only_with_its_vision_part(self):
        small = [(url, 5, digest) for url, _, digest in parts.FILES["glimmer"]]
        with mock.patch.dict(parts.FILES, {"glimmer": small}):
            with open(parts.GLIMMER_FILE, "wb") as f:
                f.write(b"model")
            self.assertFalse(parts._glimmer_here(), "the studio shows Muse pictures, so it needs the vision part")
            with open(parts.VISION_FILE, "wb") as f:
                f.write(b"sight")
            self.assertTrue(parts._glimmer_here())

    def test_setup_leaves_out_what_this_card_cannot_run_and_says_why(self):
        got = self.found(dict(BIG, mib=12282))
        for key in ("engine", "glimmer"):
            self.assertEqual((got[key]["ok"], got[key]["size"]), (False, 0))
            self.assertIn("needs a graphics card with 16 GB", got[key]["why"])
        self.assertGreater(got["hearing"]["size"], 0, "the hearing runs on the processor, so it still comes")

    def test_without_an_address_the_hearing_says_where_its_files_go(self):
        with mock.patch.object(parts, "HEARING_FROM", ""):
            got = self.found(BIG)["hearing"]
        self.assertEqual(got["size"], 0)
        self.assertIn(os.path.join(self.tmp, "hearing"), got["why"])

    def test_a_server_of_the_persons_own_leaves_glimmer_and_llama_cpp_alone(self):
        os.environ["LCS_MIND_URL"] = "http://192.168.1.20:8080/v1"
        got = self.found(BIG)
        self.assertTrue(got["engine"]["ok"] and got["glimmer"]["ok"])
        self.assertFalse(parts.start_glimmer())

    def test_setup_says_how_much_room_it_needs_when_the_drive_is_too_full(self):
        with mock.patch.object(parts, "card", lambda: BIG), \
                mock.patch.object(parts.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100, 90, 10 ** 9)):
            said = parts.set_up()
        self.assertTrue(said.startswith("Setting up needs %s GB free where the studio keeps its files"
                                        % ("17.6" if parts.windows() else "16.9")), said)
        self.assertTrue(said.endswith("; there is 1.0 GB."), said)
        self.assertFalse(parts.state()["setup"]["running"])

    def test_the_studio_starts_glimmer_the_way_the_readme_shows_and_says_when_it_stopped(self):
        for path in (parts.GLIMMER_FILE, parts.VISION_FILE):
            open(path, "wb").close()
        os.environ["LCS_GLIMMER_FILE"] = parts.GLIMMER_FILE
        os.environ["LCS_LLAMA_SERVER"] = sys.executable
        self.assertTrue(parts.start_glimmer())
        ini = os.path.join(self.tmp, "tools", "glimmer.ini")
        with open(ini, encoding="utf-8") as f:
            self.assertEqual(f.read(), "version = 1\n\n[muse-glimmer]\nmodel = %s\nmmproj = %s\nmmproj-offload = "
                                       "true\nc = 49152\nparallel = 1\njinja = true\nfit = on\nload-on-startup = "
                                       "true\n" % (parts.GLIMMER_FILE, parts.VISION_FILE))
        self.assertEqual(parts.glimmer_command("llama-server", ini)[1:],
                         ["--models-preset", ini, "--models-max", "1", "--no-models-autoload", "--host", "127.0.0.1",
                          "--port", parts.LOCAL_URL.rsplit(":", 1)[1].split("/")[0]])
        parts._engine["process"].wait(30)
        self.assertIn("llama-server stopped (exit code 2)", parts.state()["engine"],
                      "Python is no llama-server: it refuses the options and stops, and the page says so")

    def test_nothing_is_started_when_something_already_answers_at_the_address(self):
        open(parts.GLIMMER_FILE, "wb").close()
        os.environ["LCS_GLIMMER_FILE"] = parts.GLIMMER_FILE
        os.environ["LCS_LLAMA_SERVER"] = sys.executable
        port = int(parts.LOCAL_URL.rsplit(":", 1)[1].split("/")[0])
        held = socket.socket()
        held.bind(("127.0.0.1", port))
        held.listen()
        try:
            self.assertFalse(parts.start_glimmer())
            self.assertIsNone(parts._engine["process"])
        finally:
            held.close()


class Page(unittest.TestCase):
    def setUp(self):
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()

    def ask(self, method, body=None, kind="application/json"):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=30)
        conn.request(method, "/api/parts", body, {"Content-Type": kind} if body is not None else {})
        r = conn.getresponse()
        got = (r.status, json.loads(r.read().decode("utf-8")))
        conn.close()
        return got

    def test_the_page_reads_the_parts_and_starts_setup_only_with_json(self):
        status, got = self.ask("GET")
        self.assertEqual((status, got["ok"]), (200, True))
        self.assertEqual([p["key"] for p in got["parts"]], ["ffmpeg", "engine", "glimmer", "hearing"])
        self.assertEqual(self.ask("POST", "go=1", "application/x-www-form-urlencoded")[0], 415)
        with mock.patch.object(parts, "set_up", lambda: "Everything that can be set up here is set up."):
            self.assertEqual(self.ask("POST", "{}"),
                             (200, {"ok": False, "error": "Everything that can be set up here is set up."}))


if __name__ == "__main__":
    unittest.main()
