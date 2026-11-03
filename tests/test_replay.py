"""A run packed as plain files plays without the studio: every picture and video its events name is in the folder,
the page asks for them where they sit, and no folder of the person's computer is in it."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, previews, recording, render, replay  # noqa: E402

CHAT = "20261027-150945-5680"
FRAME, PREVIEW, CLIP = "0123456789abcdef.jpg", "1123456789abcdef.mp4", "2123456789abcdef.mp4"
COVER = "3123456789abcdef.jpg"


class Pack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.was = agent.WORK, recording.WORK
        agent.WORK = recording.WORK = os.path.join(self.tmp, "work")
        os.makedirs(os.path.join(recording.WORK, "chats"))
        for folder, name, body in ((os.path.join(recording.WORK, "frames"), FRAME, b"jpg"),
                                   (previews.folder(), PREVIEW, b"preview"),
                                   (render.folder("clips"), CLIP, b"clip"),
                                   (render.folder("clips"), CLIP[:-4] + ".jpg", b"poster"),
                                   (render.folder("covers"), COVER, b"cover")):
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, name), "wb") as f:
                f.write(body)
        rec = {"path": "C:\\Users\\me\\Videos\\Reviews\\dry.mov", "name": "dry.mov"}
        self.events = [
            {"job": None, "recording": rec, "frame": FRAME, "line": "dry.mov · C:\\Users\\me\\Videos\\Reviews\\dry.mov",
             "id": 0, "kind": "opened", "t": 100.0},
            {"job": 1, "text": "a clip for posting", "id": 1, "kind": "person", "t": 100.5},
            {"job": 1, "mind": "Muse Glimmer", "id": 2, "kind": "job", "t": 101.0},
            {"job": 1, "name": PREVIEW, "id": 3, "kind": "preview", "t": 160.0},
            {"job": 1, "file": CLIP, "result": "Saved C:\\Users\\me\\Videos\\Reviews\\dry.mov as a clip", "id": 4,
             "kind": "rendered", "t": 220.0, "cover": COVER},
            {"job": 1, "steps": 3, "seconds": 241.0, "why": "answered", "id": 5, "kind": "end", "t": 342.0}]
        self.write(self.events)
        self.out = os.path.join(self.tmp, "replay")

    def tearDown(self):
        agent.WORK, recording.WORK = self.was
        agent.CHATS.pop(CHAT, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, events):
        base = os.path.join(recording.WORK, "chats", CHAT)
        with open(base + ".events.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e) + "\n" for e in events)
        with open(base + ".messages.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "user", "content": "a clip for posting"}) + "\n")

    def read(self, *name):
        with open(os.path.join(self.out, *name), "rb") as f:
            return f.read()

    def test_every_picture_and_video_the_run_names_is_in_the_folder_where_the_page_asks_for_it(self):
        done = replay.pack(CHAT, self.out, card="RTX 5070 Ti 16 GB")
        self.assertEqual(self.read("api", "frame", FRAME), b"jpg")
        self.assertEqual(self.read("api", "preview", PREVIEW), b"preview")
        self.assertEqual(self.read("api", "clip", CLIP), b"clip")
        self.assertEqual(self.read("api", "clip", CLIP[:-4] + ".jpg"), b"poster")
        self.assertEqual(self.read("api", "cover", COVER), b"cover", "lane 5 11-03: a Muse Image cover read as gone")
        page = self.read("app.js").decode("utf-8")
        self.assertNotIn('"/api/', page)
        self.assertIn('"api/frame/"', page)
        self.assertLess(self.read("index.html").index(b"replay.js"), self.read("index.html").index(b'src="app.js"'))
        self.assertTrue(os.path.isfile(os.path.join(self.out, "replay.js")))
        data = json.loads(self.read("api", "replay.json"))
        self.assertEqual([e["kind"] for e in data["events"]], [e["kind"] for e in self.events])
        self.assertEqual((data["chat"], data["took"], data["card"]), (CHAT, 241, "RTX 5070 Ti 16 GB"))
        self.assertEqual(done["events"], 6)

    def test_the_words_layer_and_the_base_video_play_from_the_folder_as_the_studio_sent_them(self):
        png, mp4 = "0123456789abcdef0123.png", "3123456789abcdef.mp4"
        for folder, name, body in ((render.folder("layers"), png, b"layer"), (render.folder("bases"), mp4, b"base")):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(body)
        layer = {"ok": True, "size": [540, 960], "steps": [{"from": 0.0, "to": 2.5, "url": "/api/layer/" + png}]}
        had = replay._layer_and_base
        replay._layer_and_base = lambda get, chat: (layer, {"ok": True, "url": "/api/base/" + mp4, "from_cache": True})
        try:
            replay.pack(CHAT, self.out, card="")
        finally:
            replay._layer_and_base = had
        self.assertEqual(self.read("api", "layer", png), b"layer")
        self.assertEqual(self.read("api", "base", mp4), b"base")
        data = json.loads(self.read("api", "replay.json"))
        self.assertEqual(data["layer"]["steps"][0]["url"], "api/layer/" + png, "an address the page uses from any folder")
        self.assertEqual(data["base"]["url"], "api/base/" + mp4)

    def test_no_folder_of_the_persons_computer_is_in_the_replay(self):
        replay.pack(CHAT, self.out, card="")
        data = self.read("api", "replay.json").decode("utf-8")
        self.assertNotIn("Users", data)
        self.assertNotIn("Reviews", data)
        data = json.loads(data)
        self.assertEqual(data["events"][0]["recording"]["path"], "dry.mov")
        self.assertEqual(data["events"][4]["result"], "Saved dry.mov as a clip")

    def test_the_replay_names_only_the_muse_that_ran_and_where_it_ran(self):
        """Lane 5 11-03, EP0 on Spark 1.3: the chip read Glimmer, the studio's default, and the banner put the run on
        the card."""
        replay.pack(CHAT, self.out, card="RTX 5070 Ti 16 GB")
        data = json.loads(self.read("api", "replay.json"))
        self.assertEqual([m["label"] for m in data["health"]["minds"]], ["Muse Glimmer"])
        self.assertEqual((data["health"]["default"], data["card"], data["api"]), ("local", "RTX 5070 Ti 16 GB", []))
        self.events[2]["mind"] = "Muse Spark"
        self.write(self.events)
        other = os.path.join(self.tmp, "api_run")
        replay.pack(CHAT, other, card="RTX 5070 Ti 16 GB")
        with open(os.path.join(other, "api", "replay.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([(m["label"], m["ok"]) for m in data["health"]["minds"]], [("Muse Spark", True)])
        self.assertEqual((data["health"]["default"], data["card"], data["api"]), ("spark", "", ["Muse Spark"]))
        with open(os.path.join(other, "replay.js"), encoding="utf-8") as f:
            said = f.read()
        self.assertIn('" with " + d.api.join(" and ") + " through an API"', said)
        self.assertIn('+ api + "; it plays here faster', said, "the banner says it")

    def test_a_run_whose_pictures_are_gone_or_whose_job_has_not_ended_is_not_packed(self):
        os.remove(os.path.join(previews.folder(), PREVIEW))
        with self.assertRaises(replay.Refused) as said:
            replay.pack(CHAT, self.out, card="")
        self.assertIn(PREVIEW, str(said.exception))
        self.write(self.events[:-1])
        with self.assertRaises(replay.Refused) as said:
            replay.pack(CHAT, os.path.join(self.tmp, "other"), card="")
        self.assertIn("has not finished", str(said.exception))
        with open(os.path.join(recording.WORK, "chats", CHAT + ".events.jsonl"), encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), 5, "packing never writes an end into a run that is going on")


if __name__ == "__main__":
    unittest.main()
