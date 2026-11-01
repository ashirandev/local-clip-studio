"""The clip as a Premiere Pro project (Final Cut Pro 7 XML + .srt): python -m unittest discover -s tests"""
import http.client
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile
from unittest import mock
import xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, layouts, montage, premiere, server  # noqa: E402


def rec(path, w=1920, h=1080, seconds=120.0, fps=59.94, sound=1):
    return {"path": path, "name": os.path.basename(path), "width": w, "height": h, "seconds": seconds, "fps": fps,
            "sound_tracks": sound}


def state(pieces, layout, **more):
    out = {"clip": {"pieces": pieces, "layout": layout}, "length": round(sum(b - a for a, b in pieces), 2),
           "titles": [], "captions": None, "blurs": [], "hits": [], "stickers": [], "glass": [], "flurries": [],
           "look": None, "sounds": [], "music": None, "mutes": [], "channel": None}
    out.update(more)
    return out


def tracks(xml, kind):
    root = ET.fromstring(xml.split("\n", 2)[2])
    return root.find("sequence/media/%s" % kind).findall("track")


def params(clipitem, effect):
    for f in clipitem.findall("filter/effect"):
        if f.findtext("effectid") == effect:
            return {p.findtext("parameterid"): p.find("value") for p in f.findall("parameter")}
    return None


class Export(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.stream = rec(os.path.join(self.tmp, "my stream & 100%.mkv"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_each_piece_points_at_the_whole_original_file_so_an_editor_can_drag_its_edges(self):
        """Tarm 10-29: "น้องผมต้องลาก timeline ของแต่ละวิดีโอ ยาวขึ้น สั้นลงเพื่อปรับซีนได้"."""
        lay = layouts.make("subject", (1920, 1080), crop=[0, 0, 608, 1080])
        xml = premiere.sequence("Boss down", self.stream, state([[10.0, 14.0], [50.5, 52.0]], lay))
        video = tracks(xml, "video")
        self.assertEqual(len(video), 1)
        items = video[0].findall("clipitem")
        self.assertEqual([c.findtext("name") for c in items], ["my stream & 100%.mkv"] * 2, "named by the file")
        self.assertEqual([(int(c.findtext("start")), int(c.findtext("end"))) for c in items], [(0, 120), (120, 165)],
                         "pieces meet exactly, at 30 fps")
        first = items[0]
        self.assertEqual((first.findtext("rate/timebase"), first.findtext("rate/ntsc")), ("60", "TRUE"))
        self.assertEqual((int(first.findtext("in")), int(first.findtext("out"))), (599, 839), "frames of the file")
        self.assertEqual(int(first.findtext("pproTicksIn")), 10 * premiere.TICKS)
        self.assertEqual(first.findtext("file/pathurl"), "file://localhost/" + premiere._url(
            self.stream["path"]).split("file://localhost/")[1])
        self.assertIn("%3a/", first.findtext("file/pathurl"))
        self.assertIn("my%20stream%20%26%20100%25.mkv", first.findtext("file/pathurl"))
        self.assertEqual(int(first.findtext("file/duration")), 7193, "the whole file: 120 s at 59.94")
        self.assertEqual(items[1].find("file").attrib, {"id": "file-1"}, "the file is written once")
        root = ET.fromstring(xml.split("\n", 2)[2])
        self.assertEqual((root.findtext("sequence/duration"), root.findtext(
            "sequence/media/video/format/samplecharacteristics/width"), root.findtext(
            "sequence/media/video/format/samplecharacteristics/height")), ("165", "1080", "1920"))

    def test_the_frame_comes_over_as_scale_and_position(self):
        left = layouts.make("subject", (1920, 1080), crop=[0, 0, 608, 1080])
        middle = layouts.make("subject", (1920, 1080), crop=[656, 0, 608, 1080])
        for lay, horiz in ((left, 0.6076), (middle, 0.0)):
            item = premiere.items(self.stream, state([[0.0, 2.0]], lay))[0]
            scale, h, v, crop = premiere.motion(item, lay["parts"][0])
            part = lay["parts"][0]
            self.assertAlmostEqual(scale, 100 * max(1080.0 / part["src"][2], 1920.0 / part["src"][3]), places=2,
                                   msg="it fills the frame: no hairline left at an edge")
            self.assertAlmostEqual(h, (1920 * scale / 200 - part["src"][0] * scale / 100 - 540) / 1920, places=4)
            self.assertAlmostEqual(h, horiz, delta=0.01 if horiz else 0.001)
            self.assertAlmostEqual(v, 0.0, delta=0.001)
            self.assertIsNone(crop, "a part that fills the frame needs no crop")
        xml = premiere.sequence("t", self.stream, state([[0.0, 2.0]], left))
        motion = params(tracks(xml, "video")[0].find("clipitem"), "basic")
        self.assertAlmostEqual(float(motion["scale"].text), 177.778, places=2)
        self.assertAlmostEqual(float(motion["center"].findtext("horiz")), 0.6076, places=3)

    def test_a_card_puts_each_part_on_its_own_track_cropped_to_what_it_shows(self):
        lay = layouts.make("card", (1920, 1080), face=[1500, 60, 300, 300], crop=[0, 0, 1400, 1080])
        self.assertEqual([p["name"] for p in lay["parts"]], ["card", "circle"])
        xml = premiere.sequence("t", self.stream, state([[5.0, 9.0]], lay))
        video = tracks(xml, "video")
        self.assertEqual(len(video), 2)
        for track, part in zip(video, lay["parts"]):
            crop = params(track.find("clipitem"), "crop")
            self.assertIsNotNone(crop, part["name"])
            sx, sy, sw, sh = part["src"]
            got = [float(crop[k].text) for k in ("left", "right", "top", "bottom")]
            want = [100.0 * sx / 1920, 100.0 * (1920 - sx - sw) / 1920, 100.0 * sy / 1080, 100.0 * (1080 - sy - sh) / 1080]
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, delta=0.3, msg=part["name"])
        self.assertEqual(len(video[0].find("clipitem").findall("link")), 3, "the card's piece links its sound")
        self.assertEqual(video[1].find("clipitem").findall("link"), [], "the circle is the same moment, unlinked")

    def test_the_sound_is_linked_left_and_right_and_a_silent_file_has_none(self):
        lay = layouts.make("subject", (1920, 1080))
        audio = tracks(premiere.sequence("t", self.stream, state([[0.0, 2.0]], lay)), "audio")
        self.assertEqual([t.find("clipitem/sourcetrack/trackindex").text for t in audio], ["1", "2"])
        video_id = tracks(premiere.sequence("t", self.stream, state([[0.0, 2.0]], lay)), "video")[0].find(
            "clipitem").get("id")
        self.assertIn(video_id, [ln.findtext("linkclipref") for ln in audio[0].find("clipitem").findall("link")])
        quiet = dict(self.stream, sound_tracks=0)
        self.assertEqual(tracks(premiere.sequence("t", quiet, state([[0.0, 2.0]], lay)), "audio"), [])

    def test_a_montage_points_back_at_the_clips_it_was_joined_from(self):
        upright = rec(os.path.join(self.tmp, "IMG_1.MOV"), 1080, 1920, 20.0, 29.97)
        lying = rec(os.path.join(self.tmp, "IMG_2.MOV"), 1920, 1080, 30.0, 59.94)
        joined = rec(os.path.join(self.tmp, "trip-abc.mp4"), 1080, 1920, 7.04, 30)
        montage._write_made_of([{"rec": upright, "from": 2.0, "to": 5.0, "clip": 1, "why": "where we are"},
                                {"rec": lying, "from": 10.0, "to": 14.0, "centre": 0.25, "clip": 2, "why": "the view"}],
                               joined["path"])
        lay = layouts.make("subject", (1080, 1920))
        items = premiere.items(joined, state([[1.0, 7.04]], lay))
        self.assertEqual([(i["name"], i["from"], i["to"], i["at"]) for i in items],
                         [("IMG_1.MOV", 3.0, 5.0, 0.0), ("IMG_2.MOV", 10.0, 14.04, 2.0)],
                         "split where one clip ends; the last takes the joined file's tail from its own file")
        scale, horiz, _, _ = premiere.motion(items[1], lay["parts"][0])
        wide = int(round(1920 * 1920 / 1080.0 / 2)) * 2
        x = int(round(0.25 * wide - 540))
        self.assertAlmostEqual(scale, 100.0 * wide / 1920, places=2)
        self.assertAlmostEqual(horiz, (wide / 2.0 - x - 540) / 1920, places=4, msg="cropped upright around its centre")
        xml = premiere.sequence("trip", joined, state([[1.0, 7.04]], lay))
        root = ET.fromstring(xml.split("\n", 2)[2])
        self.assertEqual([m.findtext("name") for m in root.findall("sequence/marker")], ["where we are", "the view"])
        self.assertEqual([c.findtext("file/name") for c in tracks(xml, "video")[0].findall("clipitem")],
                         ["IMG_1.MOV", "IMG_2.MOV"])

    def test_a_joined_montage_says_what_it_was_made_of_even_when_it_was_joined_before(self):
        clip = rec(os.path.join(self.tmp, "IMG_9.MOV"), 1080, 1920, 20.0, 29.97)
        out = os.path.join(self.tmp, "trip-old.mp4")
        with open(out, "wb") as f:
            f.write(b"joined before made_of was written")
        self.assertIsNone(montage.made_of(out))
        self.assertEqual(montage.join([{"rec": clip, "from": 1.0, "to": 4.0, "clip": 1, "why": "the gate"}], out), out)
        self.assertEqual([(m["path"], m["from"], m["to"], m["why"]) for m in montage.made_of(out)],
                         [(clip["path"], 1.0, 4.0, "the gate")])

    def test_a_photo_in_a_montage_gets_the_length_of_its_piece(self):
        photo = rec(os.path.join(self.tmp, "IMG_3.JPG"), 4284, 5712, 0.04, 25, sound=0)
        joined = rec(os.path.join(self.tmp, "trip-def.mp4"), 1080, 1920, 3.0, 30)
        montage._write_made_of([{"rec": photo, "from": 0.0, "to": 3.0, "clip": 1, "why": "the menu"}], joined["path"])
        item = premiere.items(joined, state([[0.0, 3.0]], layouts.make("subject", (1080, 1920))))[0]
        self.assertEqual((item["from"], item["to"], item["seconds"], item["fps"], item["sound"]), (0.0, 3.0, 3.0, 30,
                                                                                                   False))

    def test_captions_titles_and_what_did_not_come_over(self):
        lay = layouts.make("subject", (1920, 1080))
        lines = [{"from": 0.0, "to": 1.25, "text": "ลองฟังเสียง"}, {"from": 3661.5, "to": 3662.0, "text": "two"}]
        st = state([[0.0, 4.0]], lay, captions={"lines": lines},
                   titles=[{"text": "Yeti Orb", "start": 0.5, "end": 2.0}],
                   blurs=[{"box": [0, 0, 100, 100], "start": 0, "end": 4}])
        self.assertEqual(premiere.srt(st), "1\n00:00:00,000 --> 00:00:01,250\nลองฟังเสียง\n\n"
                                           "2\n01:01:01,500 --> 01:01:02,000\ntwo\n\n")
        name, data = premiere.bundle("Yeti Orb: ไมค์ / ดี?", self.stream, st)
        self.assertEqual(name, "Yeti-Orb-ไมค์-ดี-premiere.zip")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            self.assertEqual(sorted(z.namelist()), ["Yeti-Orb-ไมค์-ดี/README.txt", "Yeti-Orb-ไมค์-ดี/Yeti-Orb-ไมค์-ดี.srt",
                                                    "Yeti-Orb-ไมค์-ดี/Yeti-Orb-ไมค์-ดี.xml"])
            note = z.read("Yeti-Orb-ไมค์-ดี/README.txt").decode("utf-8")
            xml = z.read("Yeti-Orb-ไมค์-ดี/Yeti-Orb-ไมค์-ดี.xml").decode("utf-8")
        self.assertIn("A face or a chat the studio blurred is NOT blurred here", note)
        self.assertNotIn("colour look", note)
        marker = ET.fromstring(xml.split("\n", 2)[2]).find("sequence/marker")
        self.assertEqual((marker.findtext("name"), marker.findtext("in"), marker.findtext("out")), ("Yeti Orb", "15",
                                                                                                    "60"))
        with zipfile.ZipFile(io.BytesIO(premiere.bundle("x", self.stream, state([[0.0, 4.0]], lay))[1])) as z:
            self.assertFalse([n for n in z.namelist() if n.endswith(".srt")], "no captions, no .srt")

    def test_the_music_goes_after_the_sound_tracks(self):
        song = os.path.join(self.tmp, "song.mp3")
        with open(song, "wb") as f:
            f.write(b"0")
        lay = layouts.make("subject", (1920, 1080))
        for clip, first in ((self.stream, 3), (dict(self.stream, sound_tracks=0), 1)):
            xml = premiere.sequence("t", clip, state([[0.0, 4.0]], lay, music={"path": song, "from": 12.0}))
            audio = tracks(xml, "audio")
            music = audio[-1].find("clipitem")
            self.assertEqual((music.findtext("name"), music.findtext("in"), music.findtext("out")),
                             ("song.mp3", "360", "480"))
            self.assertEqual(sorted({int(ln.findtext("trackindex")) for ln in music.findall("link")}),
                             [first, first + 1])

    def test_no_clip_is_said(self):
        with self.assertRaises(premiere.NotExported):
            premiere.sequence("t", self.stream, {"clip": None})


class Finished(unittest.TestCase):
    """Henri 11-01: the picture was black and the timeline incomplete in Premiere, and no one here has Premiere to find out why."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.stream = rec(os.path.join(self.tmp, "stream.mkv"))
        self.path = os.path.join(self.tmp, "clips", "0123456789abcdef.mp4")
        song = os.path.join(self.tmp, "song.mp3")
        with open(song, "wb") as f:
            f.write(b"x")
        self.state = state([[10.0, 12.0], [20.0, 22.0]], layouts.make("subject", (1920, 1080)), music={"path": song, "from": 0})
        self.xml = premiere.sequence("t", self.stream, self.state, self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_finished_clip_is_the_bottom_track_and_the_sound_with_no_scale_to_go_wrong(self):
        video, audio = tracks(self.xml, "video"), tracks(self.xml, "audio")
        whole = video[0].findall("clipitem")
        self.assertEqual(len(whole), 1)
        self.assertEqual(whole[0].findtext("name"), "0123456789abcdef.mp4")
        self.assertEqual([whole[0].findtext(k) for k in ("start", "end", "in", "out", "enabled")], ["0", "120", "0", "120", "TRUE"])
        self.assertEqual(whole[0].findall("filter"), [], "no scale, position or crop")
        self.assertIn("0123456789abcdef.mp4", whole[0].find("file").findtext("pathurl"))
        self.assertEqual(len(audio), 2, "the music is in the finished clip, so it is not there twice")
        self.assertEqual([t.find("clipitem/sourcetrack/trackindex").text for t in audio], ["1", "2"])
        self.assertEqual({t.find("clipitem").findtext("name") for t in audio}, {"0123456789abcdef.mp4"})

    def test_the_pieces_stay_above_it_switched_off_with_their_whole_file(self):
        above = tracks(self.xml, "video")[1:]
        self.assertTrue(above)
        items = [c for t in above for c in t.findall("clipitem")]
        self.assertEqual(len(items), 2 * len(above))
        self.assertEqual({c.findtext("enabled") for c in items}, {"FALSE"})
        self.assertEqual({c.findtext("name") for c in items}, {"stream.mkv"})
        self.assertEqual([ln for c in items for ln in c.findall("link") if ln.findtext("mediatype") == "audio"], [],
                         "a link to sound that is not on the timeline")

    def test_the_captions_are_their_own_layer_just_above_the_clip_switched_off_and_see_through(self):
        caps = os.path.join(self.tmp, "plates", "c.mov")
        pieces = len(tracks(self.xml, "video"))
        video = tracks(premiere.sequence("t", self.stream, self.state, self.path, caps), "video")
        self.assertEqual(len(video), pieces + 1)
        self.assertEqual(video[0].find("clipitem").findtext("name"), "0123456789abcdef.mp4")
        layer = video[1].findall("clipitem")
        self.assertEqual(len(layer), 1)
        self.assertEqual([layer[0].findtext(k) for k in ("name", "enabled", "alphatype", "start", "end", "in", "out")],
                         ["c.mov", "FALSE", "straight", "0", "120", "0", "120"])
        self.assertIn("c.mov", layer[0].find("file").findtext("pathurl"))
        self.assertIsNone(layer[0].find("file/media/audio"), "it has no sound")
        self.assertEqual(layer[0].findall("link"), [])
        self.assertEqual(layer[0].findall("filter"), [])
        self.assertEqual({c.findtext("enabled") for t in video[2:] for c in t.findall("clipitem")}, {"FALSE"})

    def test_the_note_tells_how_to_bring_the_captions_in_when_they_are_a_layer(self):
        lines = {"lines": [{"from": 0.0, "to": 1.0, "text": "สวัสดี"}]}
        marked = dict(self.state, captions=lines)
        with zipfile.ZipFile(io.BytesIO(premiere.bundle("ไมค์ดี", self.stream, marked, self.path, "c.mov")[1])) as z:
            note = z.read("ไมค์ดี/README.txt").decode("utf-8")
            self.assertIn("ไมค์ดี/ไมค์ดี.srt", z.namelist())
        self.assertIn("pick ไมค์ดี.srt and drag it onto the timeline", note)
        self.assertIn("so they never show twice", note)
        self.assertIn("ซับอยู่ในไฟล์ ไมค์ดี.srt", note)
        with zipfile.ZipFile(io.BytesIO(premiere.bundle("ไมค์ดี", self.stream, self.state, self.path)[1])) as z:
            self.assertNotIn(".srt", z.read("ไมค์ดี/README.txt").decode("utf-8"))

    def test_the_note_says_what_is_on_the_timeline_and_warns_of_the_blurs(self):
        name, body = premiere.bundle("ไมค์ดี", self.stream, self.state, self.path)
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            note = z.read("ไมค์ดี/README.txt").decode("utf-8")
        self.assertIn("The bottom video track is your clip", note)
        self.assertIn("is NOT blurred in them", note)
        self.assertIn("แทร็กวิดีโอล่างสุดคือคลิปของคุณ", note)
        self.assertIn('double-click "ไมค์ดี"', note)
        self.assertNotIn("What came over", note)


class Page(unittest.TestCase):
    """The finished clip's "Premiere project" button saves the clip as it is now."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, agent.WORK = agent.WORK, os.path.join(self.tmp, "work")
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        agent.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=30)
        conn.request("GET", path)
        got = conn.getresponse()
        out = (got.status, got.getheader("Content-Disposition"), got.read())
        conn.close()
        return out

    def test_the_button_saves_a_zip_named_by_the_clip(self):
        chat = agent.Chat()
        agent.CHATS[chat.id] = chat
        status, _, body = self.get("/api/premiere?chat=" + chat.id)
        self.assertEqual((status, json.loads(body)["error"]), (404, "There is no clip yet, so there is nothing to export."))
        chat.recording = rec(os.path.join(self.tmp, "IMG_1.MOV"))
        chat.project.add({"do": "clip", "pieces": [[1.0, 3.0]], "layout": layouts.make("subject", (1920, 1080))})
        chat.emit("ready", job=1, title="ไมค์ดี")
        status, _, body = self.get("/api/premiere?chat=" + chat.id)
        self.assertEqual((status, json.loads(body)["error"]), (404, "The Premiere files of the clip as it is now are not made yet. Press Premiere again."))
        plate, caps = os.path.join(self.tmp, "0123456789abcdef.mp4"), os.path.join(self.tmp, "c.mov")
        chat.layers_now = lambda: {"plate": plate, "captions": caps}
        status, named, body = self.get("/api/premiere?chat=" + chat.id)
        self.assertEqual(status, 200)
        self.assertIn("filename*=UTF-8''%E0%B9%84%E0%B8%A1%E0%B8%84%E0%B9%8C%E0%B8%94%E0%B8%B5-premiere.zip", named)
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            self.assertIn("ไมค์ดี/ไมค์ดี.xml", z.namelist())
            xml = z.read("ไมค์ดี/ไมค์ดี.xml")
        self.assertIn(b"0123456789abcdef.mp4", xml, "the clip without its captions goes on the timeline")
        self.assertIn(b"c.mov", xml, "the captions' own layer goes on the timeline")
        self.assertEqual(self.get("/api/premiere?chat=20990101-000000-abcd")[0], 404)

    def post(self, path, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=30)
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        got = conn.getresponse()
        out = (got.status, json.loads(got.read()))
        conn.close()
        return out

    def test_a_press_makes_the_files_and_says_how_it_went(self):
        chat = agent.Chat()
        agent.CHATS[chat.id] = chat
        self.assertEqual(self.post("/api/premiere", {"chat": chat.id})[0], 404)
        for said, status in (({"ready": True}, 200), ({"making": True}, 200), ({"error": "No."}, 400)):
            chat.layers = lambda: said
            got = self.post("/api/premiere", {"chat": chat.id})
            self.assertEqual(got[0], status)
            self.assertEqual(got[1].get("error"), said.get("error"))
            self.assertEqual(got[1].get("ready"), said.get("ready"))
            self.assertEqual(got[1].get("making"), said.get("making"))


class Making(unittest.TestCase):
    """The two files are made in the background with the bar's percent, and belong to the clip as it was then."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, agent.WORK = agent.WORK, os.path.join(self.tmp, "work")
        self.chat = agent.Chat()
        self.chat.recording = rec(os.path.join(self.tmp, "IMG_1.MOV"))
        self.chat.project.add({"do": "clip", "pieces": [[1.0, 3.0]], "layout": layouts.make("subject", (1920, 1080))})
        self.plate = os.path.join(self.tmp, "plate.mp4")
        with open(self.plate, "wb") as f:
            f.write(b"x")
        self.lines = []
        self.chat.emit = lambda kind, **more: self.lines.append((kind, more))

    def tearDown(self):
        agent.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def made(self, plate=None, caps="c.mov"):
        return (mock.patch.object(agent.render, "plate", plate or (lambda *a, **k: self.plate)),
                mock.patch.object(agent.layer, "captions_video", lambda *a, **k: caps))

    def test_the_files_are_kept_for_the_clip_they_were_made_from(self):
        one, two = self.made()
        with one, two:
            self.assertEqual(self.chat.layers(), {"ready": True})
        self.assertEqual(self.chat.layers_now(), {"signature": mock.ANY, "plate": self.plate, "captions": "c.mov"})
        self.chat.project.add({"do": "look", "look": "warm"})
        self.assertIsNone(self.chat.layers_now(), "a change makes them stale")

    def test_files_gone_from_the_disk_are_not_handed_out(self):
        one, two = self.made()
        with one, two:
            self.chat.layers()
        os.remove(self.plate)
        self.assertIsNone(self.chat.layers_now())

    def test_what_went_wrong_is_said(self):
        def broke(*a, **k):
            raise agent.render.NotDrawn("no fonts")

        one, two = self.made(broke)
        with one, two:
            self.assertEqual(self.chat.layers(), {"error": "The Premiere files could not be made: no fonts"})
        self.assertIsNone(self.chat.layers_now())

    def test_stop_ends_it_and_says_so(self):
        def stopped(*a, **k):
            raise agent.render.Stopped("Stopped")

        one, two = self.made(stopped)
        with one, two:
            self.assertEqual(self.chat.layers(), {"error": "The Premiere files were not made: Stop was pressed."})

    def test_a_long_making_says_so_and_its_end_comes_as_an_event(self):
        gate = threading.Event()

        stopped = []

        def slow(*a, **k):
            k["on_start"](lambda: stopped.append(True))
            k["on_progress"](1.0)
            gate.wait(10)
            return self.plate

        one, two = self.made(slow)
        with one, two, mock.patch.object(agent, "PREVIEW_WAIT", 0.05):
            self.assertEqual(self.chat.layers(), {"making": True})
            self.assertEqual(self.chat.layers(), {"making": True}, "one at a time")
            self.assertIn("(50%)", self.chat.live["work"])
            self.assertEqual(self.chat.live["page"], "premiere")
            self.assertTrue(self.chat.stop(), "Stop reaches it")
            self.assertEqual(stopped, [True])
            gate.set()
            for _ in range(200):
                if self.lines:
                    break
                threading.Event().wait(0.01)
        self.assertEqual(self.lines, [("layers", {"job": None})])
        self.assertIsNotNone(self.chat.layers_now())


if __name__ == "__main__":
    unittest.main()
