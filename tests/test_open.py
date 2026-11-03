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
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, mind, montage, previews, recording, render, server, takes  # noqa: E402
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
    """With no folder named, Muse finds recordings on every drive, as Elmoro finds files (Tarm 10-28: his brother
    should not have to set anything)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.had = os.environ.pop("LCS_VIDEOS", None), recording.WORK, recording.drives, recording.SMALLEST
        self.c, self.d = os.path.join(self.tmp, "C"), os.path.join(self.tmp, "D")
        recording.WORK = os.path.join(self.d, "studio", "work")
        recording.drives, recording.SMALLEST = (lambda: [self.c, self.d]), 0
        self.inbox = mock.patch.object(recording, "INBOX", os.path.join(self.d, "studio", "From phone"))
        self.inbox.start()
        recording._swept.clear()

    def tearDown(self):
        self.inbox.stop()
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

    def test_a_memory_card_is_swept_and_an_empty_reader_is_not(self):
        if os.name != "nt":
            self.skipTest("drive letters are for Windows")
        recording.drives = self.had[2]
        kinds = {"C": recording.FIXED_DRIVE, "E": recording.REMOVABLE_DRIVE, "F": recording.REMOVABLE_DRIVE, "Z": 4}
        kernel = mock.Mock()
        kernel.GetLogicalDrives.return_value = sum(1 << (ord(d) - 65) for d in kinds)
        kernel.GetDriveTypeW.side_effect = lambda root: kinds[root[0]]
        with mock.patch("ctypes.windll.kernel32", kernel), mock.patch("os.path.isdir", lambda p: p != "F:\\"):
            self.assertEqual(recording.drives(), ["C:\\", "E:\\"])

    def test_every_drive_is_swept_without_system_hidden_or_studio_folders_and_a_miss_sweeps_again(self):
        kept = [self.put(self.c, "Users", "me", "Videos", "trip.mp4"), self.put(self.d, "Streams", "ep1.mkv")]
        for left in [(self.c, "Program Files", "app", "intro.mp4"), (self.c, "Users", "me", "AppData", "cache.mp4"),
                     (self.d, ".hidden", "x.mp4"), (self.d, "$Recycle.Bin", "old.mp4"),
                     (self.d, "studio", "work", "clips", "made.mp4")]:
            self.put(*left)
        self.assertEqual(recording.folders(), [os.path.realpath(self.c), os.path.realpath(self.d)])
        self.assertEqual(sorted(p for _, _, p in recording.walk()), sorted(kept))
        self.assertTrue(recording._swept["complete"])
        self.assertEqual([p for _, _, p in recording.walk("streams")], [kept[1]])
        later = self.put(self.d, "Streams", "ep2.mp4")
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
        with self.assertRaises(recording.NotSeen) as e:
            recording.locate("not here.mp4", 1234)
        self.assertIn("is not in the folders recordings are kept in", str(e.exception))

    def test_a_dropped_file_that_is_no_video_is_told_so_and_not_sent_in(self):
        with self.assertRaises(recording.NotOpened) as e:
            recording.locate("notes.txt", 1234)
        self.assertNotIsInstance(e.exception, recording.NotSeen)
        self.assertIn("notes.txt is not a video the studio reads (.mp4,", str(e.exception))

    def test_a_phone_gets_a_small_copy_of_a_finished_clip_and_a_save_gets_the_whole_file(self):
        """Tarm 11-01: 1080 by 1920 does not load on his phone, even on Wi-Fi."""
        clips = render.folder("clips")
        os.makedirs(clips, exist_ok=True)
        name, full = "0123456789abcdef.mp4", os.path.join(clips, "0123456789abcdef.mp4")
        make_video(full, seconds=2, size="1080x1920")
        served = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=served.serve_forever, daemon=True).start()
        try:
            def size(tail):
                conn = http.client.HTTPConnection("127.0.0.1", served.server_port, timeout=60)
                conn.request("GET", "/api/clip/" + name + tail)
                r = conn.getresponse()
                got = len(r.read())
                conn.close()
                return r.status, got
            whole = os.path.getsize(full)
            self.assertEqual(size(""), (200, whole), "the computer's page plays the file itself")
            self.assertEqual(size("?download=1"), (200, whole))
            status, little = size("?small=1")
            self.assertEqual(status, 200)
            self.assertLess(little, whole)
            made = recording.probe(os.path.join(previews.folder(), "small-" + name))
            self.assertEqual((made["width"], made["height"]), (480, 854))
            self.assertAlmostEqual(made["seconds"], 2, delta=0.2)
            with mock.patch.object(previews, "_cut", lambda *a: self.fail("made once")):
                self.assertEqual(size("?small=1"), (200, little))
            os.remove(os.path.join(previews.folder(), "small-" + name))
            with mock.patch.object(previews, "_cut", lambda *a: "ffmpeg could not"):
                self.assertEqual(size("?small=1"), (200, whole), "a clip with no small copy still plays")
            os.remove(full)
            self.assertEqual(size("?small=1")[0], 404)
        finally:
            served.shutdown()
            served.server_close()

    def two_tracks(self, name="two tracks.mkv"):
        """A recording with silence on its first sound track, as a game with nothing playing, and a tone on its second, as a voice."""
        path = os.path.join(self.videos, name)
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x180:rate=30", "-f", "lavfi", "-i",
                        "anullsrc=r=48000:cl=stereo", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-map", "0:v", "-map", "1:a",
                        "-map", "2:a", "-t", "3", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", path], check=True)
        return recording.probe(path)

    def test_every_sound_track_is_used_unless_the_person_chose(self):
        """Tarm 11-01: Henri's game was on the first track and his voice on another, so his words were not heard or in the clip."""
        rec = self.two_tracks()
        self.assertEqual((rec["sound_tracks"], recording.tracks(rec)), (2, [0, 1]))
        self.assertEqual(recording.mixed(rec, 3), ("[3:a:0][3:a:1]amix=inputs=2:duration=longest:normalize=0[tracks3];", "[tracks3]"))
        self.assertEqual(recording.mixed(dict(rec, tracks=[1]), 3), ("", "[3:a:1]"))
        self.assertEqual(recording.sound_args(dict(rec, tracks=[1])), ["-map", "0:a:1"])
        self.assertEqual(recording.sound_args({"path": "x"}, either=True), ["-map", "0:a:0?"])
        self.assertEqual(recording.mixed(dict(rec, sound_streams=[0])), ("", "[0:a:0]"), "one track is read as before")

    def tracks_of(self, name, *sounds):
        """A recording with these lavfi sounds as its tracks, and what is used of it."""
        path = os.path.join(self.videos, name)
        inputs = ["-f", "lavfi", "-i", "testsrc=duration=12:size=320x180:rate=10"]
        maps = ["-map", "0:v"]
        for k, s in enumerate(sounds, 1):
            inputs += ["-f", "lavfi", "-i", s]
            maps += ["-map", "%d:a" % k]
        subprocess.run([FFMPEG, "-v", "error", "-y"] + inputs + maps + ["-t", "12", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "pcm_s16le", path], check=True)
        return recording.tracks(recording.probe(path))

    def test_a_track_that_repeats_another_is_left_out_and_the_louder_stays(self):
        """Lane 4 11-01: on the Sifu stream the mic has a track of its own and is in the first one too, so reading both raised it 6 dB."""
        game = "anoisesrc=d=12:c=pink:r=48000:a=0.2:seed=7"
        voice = "sine=frequency=330:duration=12:sample_rate=48000"
        mix = "anoisesrc=d=12:c=pink:r=48000:a=0.2:seed=7,volume=1[n];sine=frequency=330:duration=12:sample_rate=48000[v];[n][v]amix=normalize=0"
        self.assertEqual(self.tracks_of("apart.mkv", game, voice), [0, 1], "Henri's game and voice are both his")
        self.assertEqual(self.tracks_of("same.mkv", voice, voice), [0], "identical tracks")
        self.assertEqual(self.tracks_of("mix first.mkv", mix, voice), [0], "the mic is in the mix")
        self.assertEqual(self.tracks_of("mix second.mkv", voice, mix), [1], "the mix stays wherever it is")
        self.assertEqual(self.tracks_of("silent.mkv", "anullsrc=r=48000:cl=mono", voice), [0, 1], "silence tells nothing")

    def test_a_track_ffmpeg_cannot_read_is_not_used(self):
        """Lane 4 11-01: IMG_4096.mov's second track is Apple's spatial sound, and asking for it stops FFmpeg."""
        done = mock.Mock(stdout=json.dumps({"format": {"duration": "3"}, "streams": [
            {"codec_type": "video", "codec_name": "hevc", "width": 320, "height": 180, "r_frame_rate": "30/1"},
            {"codec_type": "audio", "codec_name": "unknown"}, {"codec_type": "audio", "codec_name": "aac"},
            {"codec_type": "audio"}]}).encode())
        with mock.patch.object(recording.subprocess, "run", lambda *a, **k: done):
            rec = recording.probe(self.clip)
        self.assertEqual((rec["sound_tracks"], rec["sound_streams"], recording.tracks(rec)), (1, [1], [1]))
        self.assertEqual(recording.sound_args(rec), ["-map", "0:a:1"], "numbered among all the audio streams")

    def test_a_voice_on_the_second_track_is_heard_and_is_in_the_clip(self):
        rec = self.two_tracks()
        self.assertGreater(float(abs(takes._decoded(rec, 0, 2)).max()), 0.05, "what the studio listens to")
        self.assertLess(float(abs(takes._decoded(dict(rec, tracks=[0]), 0, 2)).max()), 0.01, "the choice is kept")
        self.assertIsNotNone(render.loudness(rec, [[0.0, 2.0]]), "what the clip is levelled from")
        self.assertIsNone(render.loudness(dict(rec, tracks=[0]), [[0.0, 2.0]]), "the silent track alone is silence")
        self.assertEqual(previews._cut(rec, 0.0, 2.0, os.path.join(self.tmp, "p.mp4")), None)
        made = recording.probe(os.path.join(self.tmp, "p.mp4"))
        self.assertEqual(made["sound_tracks"], 1)
        self.assertGreater(float(abs(takes._decoded(made, 0, 1.5)).max()), 0.05, "the voice is in the preview")

    def test_the_finished_clip_uses_the_tracks_the_person_chose(self):
        rec = self.two_tracks()
        made = {}

        def clip(tracks):
            state = {"clip": {"pieces": [[0.0, 2.0]], "layout": {"kind": "fill", "parts": [
                {"src": [0, 0, 320, 180], "dst": [0, 0, 1080, 1920]}]}}, "length": 2.0, "tracks": tracks}
            with mock.patch.object(render, "_ffmpeg", lambda: FFMPEG):
                made["graph"] = render._sound(dict(rec, tracks=state["tracks"]), state["clip"]["pieces"])
            return made["graph"]
        self.assertIn("[0:a:0][0:a:1]amix", clip(None))
        self.assertNotIn("[0:a:0]", clip([1]))
        self.assertIn("[0:a:1]", clip([1]))

    def test_the_sound_tab_shows_tracks_only_when_there_are_several(self):
        rec = self.two_tracks()
        self.chat.recording = rec
        self.assertEqual(self.chat.tracks_shown({"tracks": None}), [{"n": 0, "on": True}, {"n": 1, "on": True}])
        self.assertEqual(self.chat.tracks_shown({"tracks": [1]}), [{"n": 0, "on": False}, {"n": 1, "on": True}])
        self.chat.recording = dict(rec, sound_tracks=1, sound_streams=[0])
        self.assertEqual(self.chat.tracks_shown({"tracks": None}), [])

    def test_a_camera_file_is_a_recording(self):
        camera = os.path.join(self.videos, "card", "00001.MTS")
        make_video(camera.replace(".MTS", ".mp4"))
        os.rename(camera.replace(".MTS", ".mp4"), camera)
        found = recording.locate("00001.mts", os.path.getsize(camera))
        self.assertEqual(os.path.normcase(found), os.path.normcase(os.path.realpath(camera)))
        for name in ("a.m2ts", "a.mxf", "a.mpg", "a.mpeg", "a.wmv", "a.3gp"):
            self.assertTrue(name.endswith(recording.VIDEO_TYPES), name)

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

    def test_muse_sees_four_moments_of_a_long_recording_it_opens_itself(self):
        """Lane 5, 10-30: Glimmer opened an episode itself, saw one man holding a snack up to a camera at 2:48, a tenth
        of the way in, and loaded the seller recipe in 2 of 4 chats."""
        times = [168.8, 590.8, 1012.8, 1434.8]
        with mock.patch.object(montage, "overview", lambda rec: (
                os.path.join(recording.WORK, "frames", recording.frame(rec)), times)):
            brain = ScriptedMind([call("open_recording", path=self.clip, why="the person named it")], "Opened.")
            self.chat.jobs += 1
            job = self.chat.job = agent.Job(self.chat, brain, "open my review")
            job.run()
        result, shown = brain.sent[1][-2:]
        self.assertIn("A picture of four moments across it comes next", result["content"])
        self.assertEqual(shown["content"][0]["text"], "[studio] Four moments across my review.mp4. The picture shows "
                         "four moments across it, each marked with its time (2:49, 9:51, 16:53, 23:55): judge what "
                         "kind of video it is by all four.")
        self.assertEqual(shown["content"][1]["type"], "image_url")

    def test_a_chosen_number_of_clips_asks_for_rough_cards_and_only_those_counts_do(self):
        """Tarm 11-01: one clip from a long stream is a surprise, and rough cards first spare the API."""
        said = []
        self.chat.say = lambda told, mind_key, pictures=(), shown=None: said.append((told, shown))
        rec = recording.probe(self.clip)
        self.chat.open_dropped(rec, "", "local", "th", 10)
        self.assertEqual(self.chat.wanted, 10)
        self.assertIn(agent.CLIPS_TOLD % (10, 10), said[-1][0])
        self.assertIn("10 clips to choose from", said[-1][1])
        self.assertNotIn(agent.CLIPS_TOLD % (10, 10), said[-1][1], "the person sees their choice, not the instruction")
        for count in (None, 1, 7, "10", True):
            self.chat.open_dropped(rec, "", "local", "th", count)
            self.assertIsNone(self.chat.wanted, count)
            self.assertNotIn("choose from", said[-1][1])

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
        self.assertIn("\n[studio] The person wrote this in English: answer, and write every why, in English", said[0]["text"],
                      "words typed with the drop say the language, before the browser does")
        self.assertIn("\nThe person's browser is set to th-TH; answer in that language", agent._reader("", "th-TH"))
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

    def test_the_page_is_told_which_muse_a_chat_starts_on(self):
        with mock.patch.object(mind, "default", return_value="baseten"):
            self.assertEqual(self.ask("GET", "/api/health")[1]["default"], "baseten")

    def test_a_chat_is_deleted_by_its_id_and_the_page_hears_which_chat_shows_now(self):
        typed = {"Content-Type": "application/json"}
        first = self.ask("POST", "/api/chat/new", "{}", typed)[1]["chat"]
        chat = agent.find(first)
        for path in (chat.log_path, chat.talk_path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("{}\n")
        body = json.dumps({"chat": first})
        self.assertEqual(self.ask("POST", "/api/chat/delete", body, {"Content-Type": "text/plain"})[0], 415)
        self.assertTrue(os.path.isfile(chat.log_path), "another web page cannot delete a chat")
        status, answer = self.ask("POST", "/api/chat/delete", body, typed)
        self.assertEqual((status, answer["ok"], answer["error"]), (200, True, None))
        self.assertNotEqual(answer["chat"], first)
        self.assertEqual(answer["chat"], agent.shown())
        self.assertEqual(self.ask("GET", "/api/chat?id=" + first)[0], 404)
        status, answer = self.ask("POST", "/api/chat/delete", body, typed)
        self.assertEqual((status, answer["ok"]), (200, False))
        self.assertIn("There is no chat", answer["error"])

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
