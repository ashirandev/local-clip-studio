"""The endpoint and API key a person types once, with made-up keys in a temporary file: python -m unittest discover
-s tests"""
import http.client
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import mind, parts, server  # noqa: E402

NAMES = ("LCS_SPARK_URL", "LCS_MODAL_URL", "META_API_KEY", "MODAL_KEY", "MODAL_SECRET", "LCS_KEYS")


class Saved(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.file = mock.patch.object(mind, "KEY_FILE", os.path.join(self.tmp, "LocalClipStudio", "keys.env"))
        self.file.start()
        self.had = {n: os.environ.pop(n, None) for n in NAMES}

    def tearDown(self):
        self.file.stop()
        for n, v in self.had.items():
            os.environ.pop(n, None)
            if v is not None:
                os.environ[n] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def read(self):
        with open(mind.KEY_FILE, encoding="utf-8") as f:
            return f.read()

    def test_meta_s_endpoint_and_key_are_kept_in_the_person_s_own_file_and_used_at_once(self):
        self.assertIsNone(mind.save_api(" https://api.meta.ai/v1/ ", " made-up-key "))
        self.assertEqual(self.read(), "META_API_KEY=made-up-key\nLCS_SPARK_URL=https://api.meta.ai/v1\n")
        self.assertEqual((os.environ["META_API_KEY"], os.environ["LCS_SPARK_URL"]),
                         ("made-up-key", "https://api.meta.ai/v1"))
        self.assertEqual(mind.saved_api(), ("https://api.meta.ai/v1", True))
        self.assertTrue(mind.minds()["spark"].status()["ok"])

    def test_a_modal_token_is_split_into_its_key_and_secret(self):
        url = "https://made-up--ep-muse-spark-open.us-west.modal.direct/v1"
        self.assertEqual(mind.save_api(url, "wk-only"),
                         "A Modal proxy token has two parts: give the key and the secret joined by a dot.")
        self.assertFalse(os.path.exists(mind.KEY_FILE), "nothing is kept from a refused save")
        self.assertIsNone(mind.save_api(url, "wk-made-up.ws-made-up"))
        self.assertEqual((os.environ["MODAL_KEY"], os.environ["MODAL_SECRET"], os.environ["LCS_MODAL_URL"]),
                         ("wk-made-up", "ws-made-up", url))
        self.assertEqual(mind.minds()["modal"].secret(), "wk-made-up.ws-made-up")

    def test_a_second_endpoint_replaces_the_first_and_other_lines_stay(self):
        os.makedirs(os.path.dirname(mind.KEY_FILE))
        with open(mind.KEY_FILE, "w", encoding="utf-8") as f:
            f.write("# mine\nLCS_TEST_OTHER=1\n")
        mind.save_api("https://api.meta.ai/v1", "first")
        mind.save_api("https://x--y.modal.direct/v1", "wk-a.ws-b")
        self.assertEqual(self.read(), "# mine\nLCS_TEST_OTHER=1\nMODAL_KEY=wk-a\nMODAL_SECRET=ws-b\n"
                                      "LCS_MODAL_URL=https://x--y.modal.direct/v1\n")

    def test_a_key_is_never_sent_in_the_clear_to_another_computer(self):
        self.assertEqual(mind.save_api("http://192.168.1.20:8080/v1", "k"),
                         "Give the endpoint's address, starting with https://.")
        self.assertIsNone(mind.save_api("http://127.0.0.1:8080/v1", "k"), "this computer's own server is fine")
        self.assertEqual(mind.save_api("https://api.meta.ai/v1", " "), "Give the API key.")

    def test_the_studio_reads_the_saved_file_when_it_starts(self):
        mind.save_api("https://api.meta.ai/v1", "from-the-file")
        for n in NAMES:
            os.environ.pop(n, None)
        self.assertEqual(sorted(mind.load_keys()), ["LCS_SPARK_URL", "META_API_KEY"])
        self.assertEqual(os.environ["META_API_KEY"], "from-the-file")

    def test_setup_asks_once_in_the_console_and_never_shows_the_key(self):
        answers = iter(["ftp://nope", "https://api.meta.ai/v1"])
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertTrue(parts.ask_api(read=lambda q: next(answers), read_secret=lambda q: "made-up-secret"))
        self.assertIn("Give the endpoint's address, starting with https://.", out.getvalue())
        self.assertNotIn("made-up-secret", out.getvalue())
        self.assertEqual(os.environ["META_API_KEY"], "made-up-secret")
        with redirect_stdout(io.StringIO()):
            self.assertFalse(parts.ask_api(read=lambda q: "", read_secret=lambda q: "x"), "Enter skips")


class Page(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.file = mock.patch.object(mind, "KEY_FILE", os.path.join(self.tmp, "keys.env"))
        self.file.start()
        self.had = {n: os.environ.pop(n, None) for n in NAMES}
        self.minds = dict(mind.MINDS)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        mind.MINDS.clear()
        mind.MINDS.update(self.minds)
        self.file.stop()
        for n, v in self.had.items():
            os.environ.pop(n, None)
            if v is not None:
                os.environ[n] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def ask(self, method, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=30)
        conn.request(method, "/api/key", None if body is None else json.dumps(body),
                     {"Content-Type": "application/json"} if body is not None else {})
        r = conn.getresponse()
        raw = r.read().decode("utf-8")
        conn.close()
        return r.status, json.loads(raw), raw

    def test_the_page_saves_the_key_and_reads_back_only_where_it_goes(self):
        status, got, _ = self.ask("POST", {"url": "https://api.meta.ai/v1", "key": "made-up-key"})
        self.assertEqual((status, got["ok"], got["mind"]["key"], got["mind"]["ok"]), (200, True, "spark", True))
        self.assertTrue(mind.MINDS["spark"].status()["ok"], "ready without a restart")
        status, got, raw = self.ask("GET")
        self.assertEqual((got["url"], got["saved"], got["here"]), ("https://api.meta.ai/v1", True, True))
        self.assertNotIn("made-up-key", raw)

    def test_another_device_on_the_network_cannot_set_or_read_the_key(self):
        mind.save_api("https://api.meta.ai/v1", "made-up-key")
        with mock.patch.object(server.Handler, "here", lambda self: False):
            status, got, _ = self.ask("POST", {"url": "https://attacker.example/v1", "key": "theirs"})
            self.assertEqual((status, got["error"]), (403, "Set the key on the computer the studio runs on."))
            self.assertEqual(self.ask("GET")[1], {"ok": True, "here": False, "url": "", "saved": False})
        self.assertEqual(mind.saved_api()[0], "https://api.meta.ai/v1")


class SameComputer(unittest.TestCase):
    def here(self, client, reached):
        handler = mock.Mock(client_address=(client, 50000))
        handler.connection.getsockname.return_value = (reached, 8899)
        return server.Handler.here(handler)

    def test_the_page_opened_by_this_computer_s_network_address_is_this_computer_and_a_phone_is_not(self):
        self.assertTrue(self.here("127.0.0.1", "127.0.0.1"))
        self.assertTrue(self.here("192.0.2.10", "192.0.2.10"), "the studio opened by its own network address")
        self.assertFalse(self.here("192.0.2.20", "192.0.2.10"), "a phone on the same network")
        self.assertTrue(self.here("::1", "::1"))


if __name__ == "__main__":
    unittest.main()
