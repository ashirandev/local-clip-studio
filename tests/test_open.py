"""Opening a recording where it is, on small videos made by ffmpeg: python -m unittest discover -s tests"""
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, mind, recording, server  # noqa: E402
from test_agent_loop import ScriptedMind, call  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def make_video(path, seconds=3, size="320x180", rotate=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=%d:size=%s:rate=30"
                    % (seconds, size), "-f", "lavfi", "-i", "sine=duration=%d" % seconds, "-shortest",
                    "-pix_fmt", "yuv420p", path], check=True)
    if rotate:
        turned = path + ".turned.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-display_rotation", str(rotate), "-i", path, "-c", "copy",
                        turned], check=True)
        os.replace(turned, path)


def snapshot(folder):
    return sorted((os.path.relpath(os.path.join(d, n), folder), os.path.getsize(os.path.join(d, n)),
                   os.path.getmtime(os.path.join(d, n))) for d, _, names in os.walk(folder) for n in names)


class EveryDrive(unittest.TestCase):
    """With no folder named, Muse finds recordings on every drive, as Eaglet finds files (Tarm 10-28: his brother
    should not have to set anything)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.had = os.environ.pop("LCS_VIDEOS", None), recording.WORK, recording.drives, recording.SMALLEST
        self.c, self.d = os.path.join(self.tmp, "C"), os.path.join(self.tmp, "D")
        recording.WORK = os.path.join(self.d, "studio", "work")
        recording.drives, recording.SMALLEST = (lambda: [self.c, self.d]), 0
        recording._swept.clear()

    def tearDown(self):
        videos, recording.WORK, recording.drives, recording.SMALLEST = self.had
        if videos is not None:
            os.environ["LCS_VIDEOS"] = videos
        recording._swept.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def put(self, *parts):
        path = os.path.join(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"0123456789")
        return os.path.realpath(path)

    def test_every_drive_is_swept_without_system_hidden_or_studio_folders_and_a_miss_sweeps_again(self):
        kept = [self.put(self.c, "Users", "me", "Videos", "trip.mp4"), self.put(self.d, "YouTube", "ep1.mkv")]
        for left in [(self.c, "Program Files", "app", "intro.mp4"), (self.c, "Users", "me", "AppData", "cache.mp4"),
                     (self.d, ".hidden", "x.mp4"), (self.d, "$Recycle.Bin", "old.mp4"),
                     (self.d, "studio", "work", "clips", "made.mp4")]:
            self.put(*left)
        self.assertEqual(recording.folders(), [os.path.realpath(self.c), os.path.realpath(self.d)])
        self.assertEqual(sorted(p for _, _, p in recording.walk()), sorted(kept))
        self.assertTrue(recording._swept["complete"])
        self.assertEqual([p for _, _, p in recording.walk("youtube")], [kept[1]])
        later = self.put(self.d, "YouTube", "ep2.mp4")
        self.assertNotIn(later, [p for _, _, p in recording.walk()], "one sweep is kept, not made for every search")
        self.assertEqual(recording.locate("ep2.mp4", 10), later, "a dropped file the kept sweep lacks: swept again")


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Open(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.videos = os.path.join(self.tmp, "videos")
        self.clip = os.path.join(self.videos, "reviews", "my review.mp4")
        make_video(self.clip)
        os.environ["LCS_VIDEOS"] = self.videos
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        self.smallest, recording.SMALLEST = recording.SMALLEST, 0
        self.chat = agent.Chat()

    def tearDown(self):
        recording.SMALLEST = self.smallest
        agent.CHATS.clear()
        agent._shown.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_probe_reads_length_picture_and_sound(self):
        rec = recording.probe(self.clip)
        self.assertAlmostEqual(rec["seconds"], 3, delta=0.1)
        self.assertEqual((rec["width"], rec["height"], rec["sound_tracks"], rec["fps"]), (320, 180, 1, 30))

    def test_a_frame_another_request_is_showing_is_kept_and_no_part_is_left(self):
        rec = recording.probe(self.clip)
        name = recording.frame(rec, at=1)
        target = os.path.join(recording.WORK, "frames", name)
        os.remove(target)
        real = os.replace

        def refused(a, b):
            shutil.copy(a, b)
            raise PermissionError(13, "The process cannot access the file because it is being used", b)

        os.replace = refused
        try:
            self.assertEqual(recording.frame(rec, at=1), name, "the file the other request wrote is the same frame")
        finally:
            os.replace = real
        self.assertEqual(os.listdir(os.path.dirname(target)), [name], "no part file is left behind")

    def test_a_turned_phone_video_is_as_tall_as_it_looks(self):
        phone = os.path.join(self.videos, "phone.mov")
        make_video(phone, rotate=90)
        rec = recording.probe(phone)
        self.assertEqual((rec["width"], rec["height"]), (180, 320))

    def test_a_dropped_file_is_found_by_name_and_size(self):
        other = os.path.join(self.videos, "old", "my review.mp4")
        make_video(other, seconds=5)
        st = os.stat(self.clip)
        for mtime in (st.st_mtime, None):
            found = recording.locate("My Review.mp4", st.st_size, mtime)
            self.assertEqual(os.path.normcase(found), os.path.normcase(os.path.realpath(self.clip)))
        with self.assertRaises(recording.NotOpened) as e:
            recording.locate("not here.mp4", 1234)
        self.assertIn("is not in the folders recordings are kept in", str(e.exception))

    def test_only_recordings_inside_the_folders_open(self):
        outside = os.path.join(self.tmp, "elsewhere", "secret.mp4")
        make_video(outside)
        with self.assertRaises(recording.NotOpened):
            recording.check_path(outside)
        with self.assertRaises(recording.NotOpened):
            recording.check_path(os.path.join(self.videos, "reviews", "..", "..", "elsewhere", "secret.mp4"))
        fake = os.path.join(self.videos, "fake.mp4")
        with open(fake, "w") as f:
            f.write("not a video")
        with self.assertRaises(recording.NotOpened) as e:
            recording.probe(recording.check_path(fake))
        self.assertIn("could not be read as a video", str(e.exception))

    def test_opening_copies_nothing(self):
        before = snapshot(self.videos)
        rec = recording.probe(recording.check_path(self.clip))
        self.chat.open(rec)
        self.assertEqual(snapshot(self.videos), before)
        made = [f for f, _, _ in snapshot(os.path.join(self.tmp, "work")) if not f.startswith("chats")]
        self.assertEqual(len(made), 1)
        self.assertRegex(made[0].replace(os.sep, "/"), r"^frames/[0-9a-f]{16}\.jpg$")

    def test_muse_opens_a_recording_and_the_state_says_so(self):
        brain = ScriptedMind([call("open_recording", path=self.clip, why="the person named it")],
                             [call("find_recordings", why="look")],
                             [call("open_recording", path=self.clip, why="to see it again")], "Opened.")
        self.chat.jobs += 1
        job = self.chat.job = agent.Job(self.chat, brain, "open my review")
        job.run()
        kinds = [e["kind"] for e in self.chat.events]
        self.assertLess(kinds.index("opened"), kinds.index("step"))
        after_open = brain.sent[1][-2:]
        self.assertEqual(after_open[0]["role"], "tool")
        self.assertEqual(after_open[1]["role"], "user")
        self.assertEqual(after_open[1]["content"][0]["text"],
                         "[studio] A picture from my review.mp4, a tenth of the way in.")
        self.assertEqual(after_open[1]["content"][1]["type"], "image_url")
        last = [m for m in brain.sent[-1] if m["role"] == "tool"][-1]["content"]
        self.assertIn("open: my review.mp4 (0:03 long)", last.splitlines()[-1])
        self.assertTrue(last.startswith("It is already open: my review.mp4"))
        self.assertEqual(kinds.count("opened"), 1)
        self.assertEqual(sum(1 for m in brain.sent[-1] if isinstance(m["content"], list)), 1)
        self.assertEqual(agent.Chat.load(self.chat.id).recording["path"], self.chat.recording["path"])

    def test_a_dropped_recording_tells_muse_with_the_words_typed(self):
        brain, real = ScriptedMind("Making it."), mind.get
        mind.get = lambda key: brain
        try:
            rec = recording.probe(self.clip)
            self.chat.open_dropped(rec, "funny, 20 s", None, "th-TH").thread.join(5)
        finally:
            mind.get = real
        said = brain.sent[0][-1]["content"]
        self.assertTrue(said[0]["text"].startswith("(opened) my review.mp4 · 0:03 long · 320x180"))
        self.assertTrue(said[0]["text"].endswith("\nfunny, 20 s"))
        self.assertIn("\nThe person's browser is set to th-TH; answer in that language", said[0]["text"])
        self.assertTrue(said[1]["image_url"]["url"].startswith("data:image/jpeg;base64,/9j/"))
        person = [e for e in self.chat.events if e["kind"] == "person"][0]
        self.assertNotIn("base64", json.dumps(person))
        self.assertEqual(person["text"].split("\n")[1:], ["funny, 20 s"])
        again = agent.Chat.load(self.chat.id)
        self.assertEqual((again.jobs, again.recording["name"]), (1, "my review.mp4"))


class Follow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        self.keep_alive, server.KEEP_ALIVE = server.KEEP_ALIVE, 5
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def tearDown(self):
        server.KEEP_ALIVE = self.keep_alive
        self.http.shutdown()
        self.http.server_close()
        agent.CHATS.clear()
        agent._shown.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_page_on_one_chat_follows_when_another_opens(self):
        first = agent.current()
        url = "http://127.0.0.1:%d/api/events?chat=%s" % (self.http.server_port, first.id)
        lines, moved_at = [], None
        with urllib.request.urlopen(url, timeout=10) as stream:
            time.sleep(0.3)
            asked = time.time()
            second = agent.new()
            for raw in stream:
                lines.append(raw.decode("utf-8").strip())
                if lines[-1].startswith("data:") and len(lines) > 1 and lines[-2] == "event: moved":
                    moved_at = time.time()
                    break
                if time.time() - asked > 3:
                    break
        self.assertIsNotNone(moved_at, lines)
        self.assertLess(moved_at - asked, 1.0)
        self.assertEqual(json.loads(lines[-1][5:]), {"chat": second.id})

    def test_an_idle_stream_waits_instead_of_spinning(self):
        server.KEEP_ALIVE = 0.5
        url = "http://127.0.0.1:%d/api/events?chat=%s" % (self.http.server_port, agent.current().id)
        lines = []
        with urllib.request.urlopen(url, timeout=10) as stream:
            start = time.time()
            for raw in stream:
                lines.append(raw)
                if time.time() - start > 1.2 or len(lines) > 50:
                    break
        self.assertLessEqual(len(lines), 8)

    def ask(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=10)
        try:
            conn.request(method, path, body, headers or {})
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

    def test_each_look_has_a_picture_and_no_other_name_does(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=60)
        try:
            conn.request("GET", "/api/look/glow.png")
            resp = conn.getresponse()
            body = resp.read()
        finally:
            conn.close()
        self.assertEqual((resp.status, resp.getheader("Content-Type")), (200, "image/png"))
        self.assertTrue(body.startswith(b"\x89PNG"))
        for name in ("nope", "..", "GLOW"):
            self.assertEqual(self.ask("GET", "/api/look/%s.png" % name)[0], 404, name)

    def test_the_page_knows_it_is_on_this_computer(self):
        status, health = self.ask("GET", "/api/health")
        self.assertEqual((status, health["here"]), (200, True))

    def test_another_web_page_cannot_start_anything(self):
        before = len(agent.CHATS)
        status, answer = self.ask("POST", "/api/chat/new", "{}", {"Content-Type": "text/plain"})
        self.assertEqual((status, answer["ok"]), (415, False))
        self.assertEqual(len(agent.CHATS), before)
        status, answer = self.ask("POST", "/api/chat/new", "{}", {"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual((status, answer["ok"]), (200, True))

    def test_only_an_address_or_localhost_may_name_the_studio(self):
        for host in ("127.0.0.1:8899", "localhost:8899", "LOCALHOST", "[::1]:8899", "192.168.1.20:8899"):
            self.assertEqual(self.ask("GET", "/api/health", headers={"Host": host})[0], 200, host)
        for host in ("evil.example:8899", "evil.example", "127.0.0.1.evil.example", ""):
            status, answer = self.ask("GET", "/api/health", headers={"Host": host})
            self.assertEqual(status, 403, host)
            self.assertIn("Open the studio by this computer's address", answer["error"])
        status, _ = self.ask("POST", "/api/chat/new", "{}", {"Host": "evil.example", "Content-Type": "application/json"})
        self.assertEqual(status, 403)

    def test_the_page_never_reads_its_own_address(self):
        with open(os.path.join(server.WEB, "app.js"), encoding="utf-8") as f:
            script = f.read()
        self.assertNotIn("location", script)
        self.assertNotIn("URLSearchParams", script)


if __name__ == "__main__":
    unittest.main()
