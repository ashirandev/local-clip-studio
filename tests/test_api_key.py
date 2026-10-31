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

from clipstudio import hearing, mind, parts, recording, sends, server  # noqa: E402

NAMES = ("LCS_SPARK_URL", "LCS_MODAL_URL", "META_API_KEY", "MODAL_KEY", "MODAL_SECRET", "LCS_KEYS", "LCS_BASETEN_URL",
         "BASETEN_API_KEY", "LCS_BASETEN_MODEL", "LCS_FIREWORKS_URL", "FIREWORKS_API_KEY", "LCS_FIREWORKS_MODEL")


class Saved(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.file = mock.patch.object(mind, "KEY_FILE", os.path.join(self.tmp, "LocalClipStudio", "keys.env"))
        self.file.start()
        self.work = mock.patch.object(recording, "WORK", os.path.join(self.tmp, "work"))
        self.work.start()
        self.had = {n: os.environ.pop(n, None) for n in NAMES}

    def tearDown(self):
        self.work.stop()
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

    def test_baseten_and_fireworks_serve_spark_open_with_the_persons_own_key_and_fireworks_says_it_is_not_measured(
            self):
        for key, url, name, model, label, measured in (
                ("baseten", "https://inference.baseten.co/v1", "BASETEN_API_KEY", "meta-models/muse-spark-open-1026",
                 "Baseten", True),
                ("fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY",
                 "accounts/fireworks/models/muse-spark-open", "Fireworks", False)):
            m = mind.minds()[key]
            self.assertEqual((m.url, m.model_id, m.key_env), (url, model, name))
            self.assertEqual("not measured" in m.where, not measured, "run with a key of ours on 10-29: Baseten only")
            self.assertNotEqual(m.label, mind.minds()["modal"].label, "the page tells minds apart by their label")
            gone = m.status()
            self.assertFalse(gone["ok"])
            self.assertIn("Set %s to your own %s API key" % (name, label), gone["why"])
            self.assertIsNone(mind.save_api(url, " made-up-%s " % key))
            self.assertEqual(mind.saved_api(), (url, True))
            m = mind.minds()[key]
            self.assertEqual(m.headers()["Authorization"], "Bearer made-up-" + key)
            self.assertEqual((m.status()["ok"], m.status()["name"]), (True, model))
        self.assertEqual(set(mind.saved_apis()), {"baseten", "fireworks"}, "each API keeps its own key")
        self.assertEqual(mind.minds()["fireworks"].price(), (1.25, 0.15, 4.25))
        self.assertEqual(mind.minds()["baseten"].price(), (1.25, 0.15, 4.25), "Baseten's own page, 10-29")
        self.assertAlmostEqual(mind.minds()["baseten"].cost({"read_new": 26, "reused": 128, "written": 235}),
                               0.00105, places=5)
        os.environ["LCS_FIREWORKS_MODEL"] = "accounts/someone/deployedModels/spark"
        self.assertEqual(mind.minds()["fireworks"].model_id, "accounts/someone/deployedModels/spark",
                         "a judge's own deployment is one setting")
        self.assertIsNone(mind.minds()["fireworks"].price(), "and its price is the judge's own")

    def test_each_api_keeps_its_own_key_a_second_endpoint_of_one_replaces_its_first_and_other_lines_stay(self):
        os.makedirs(os.path.dirname(mind.KEY_FILE))
        with open(mind.KEY_FILE, "w", encoding="utf-8") as f:
            f.write("# mine\nLCS_TEST_OTHER=1\n")
        mind.save_api("https://api.meta.ai/v1", "first")
        mind.save_api("https://x--y.modal.direct/v1", "wk-a.ws-b")
        self.assertEqual(mind.saved_api(), ("https://x--y.modal.direct/v1", True), "the one saved last")
        mind.save_api("https://api.meta.ai/v1", "second")
        self.assertEqual(self.read(), "# mine\nLCS_TEST_OTHER=1\nMODAL_KEY=wk-a\nMODAL_SECRET=ws-b\n"
                                      "LCS_MODAL_URL=https://x--y.modal.direct/v1\nMETA_API_KEY=second\n"
                                      "LCS_SPARK_URL=https://api.meta.ai/v1\n")
        self.assertEqual(mind.saved_apis(), {"meta": {"url": "https://api.meta.ai/v1", "saved": True},
                                             "modal": {"url": "https://x--y.modal.direct/v1", "saved": True}},
                         "Modal's token did not take the place of Meta's key, which Wonder, Image and Voice need")
        self.assertEqual(mind.saved_api(), ("https://api.meta.ai/v1", True))

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
        answers = iter(["ftp://nope", "meta", "n", ""])
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertTrue(parts.ask_api(read=lambda q: next(answers), read_secret=lambda q: "made-up-secret"))
        self.assertIn("Give the endpoint's address, starting with https://.", out.getvalue())
        self.assertIn("Type meta (Meta's API, https://api.meta.ai/v1)", out.getvalue(), "a judge cannot guess it")
        self.assertNotIn("made-up-secret", out.getvalue())
        self.assertEqual(os.environ["META_API_KEY"], "made-up-secret")
        self.assertEqual(mind.saved_api(), ("https://api.meta.ai/v1", True), "the word names Meta's address")
        with redirect_stdout(io.StringIO()):
            self.assertTrue(parts.ask_api(read=lambda q: " Baseten ", read_secret=lambda q: "made-up-baseten"))
        self.assertEqual(mind.saved_api(), ("https://inference.baseten.co/v1", True))
        with redirect_stdout(io.StringIO()):
            self.assertFalse(parts.ask_api(read=lambda q: "", read_secret=lambda q: "x"), "Enter skips")

    def asked(self, *answers, endpoint="meta", secret="made-up-secret"):
        """setup's key question with these answers after the endpoint. -> (what was printed, the questions asked)."""
        left, asked, out = iter((endpoint,) + answers), [], io.StringIO()

        def read(q):
            asked.append(q)
            return next(left, "")
        with redirect_stdout(out):
            self.assertTrue(parts.ask_api(read=read, read_secret=lambda q: secret))
        return out.getvalue(), [q for q in asked[1:] if "Another endpoint" not in q]

    def test_two_keys_at_setup_keep_both_and_the_open_spark_runs_the_chat(self):
        """Tarm's brother 10-31: Spark Open for the chat and Meta's for the rest, new to the app, with nobody beside him."""
        answers, keys = iter(["meta", "", "baseten"]), iter(["made-up-meta", "made-up-baseten"])
        with redirect_stdout(io.StringIO()):
            self.assertTrue(parts.ask_api(read=lambda q: next(answers), read_secret=lambda q: next(keys)))
        self.assertTrue(sends.allowed(), "Enter at the question between them")
        self.assertEqual(hearing.engine(), "meta")
        self.assertEqual(os.environ["META_API_KEY"], "made-up-meta")
        self.assertEqual(os.environ["BASETEN_API_KEY"], "made-up-baseten")
        mind.MINDS.update(mind.minds())
        self.assertEqual(mind.default([m.status() for m in mind.MINDS.values() if m.key != "local"]), "baseten")

    def test_a_chat_starts_on_glimmer_then_the_open_spark_then_metas(self):
        ready = lambda *keys: [{"key": k, "ok": True} for k in keys]
        self.assertEqual(mind.default(ready("local", "spark", "baseten")), "local")
        self.assertEqual(mind.default(ready("spark", "wonder", "baseten")), "baseten")
        self.assertEqual(mind.default(ready("spark", "wonder")), "spark")
        self.assertEqual(mind.default([{"key": "spark", "ok": False}]), "local", "nothing ready: the page says how to set up")

    def test_after_a_meta_key_setup_asks_once_says_what_is_sent_and_enter_turns_it_on(self):
        self.assertFalse(sends.allowed(), "a fresh copy starts off")
        said, questions = self.asked("")
        self.assertEqual(len(questions), 1)
        self.assertIn("Enter for yes", questions[0])
        for what in ("what is said", "pieces of up to a minute", "frames", "with your key", "the hearing stays on this computer"):
            self.assertIn(what, said)
        self.assertTrue(sends.allowed())
        self.assertEqual(hearing.engine(), "meta", "the key and the answer are all a judge needs for the better hearing")
        self.assertEqual(self.asked(endpoint="https://api.meta.ai/v1")[1], [], "on already, so it is not asked again")

    def test_a_no_stays_off_and_so_does_any_answer_that_is_not_a_yes(self):
        for answer in ("n", "no", "N", "later", "ไม่"):
            self.asked(answer)
            self.assertFalse(sends.allowed(), answer)
            self.assertEqual(hearing.engine(), "local", answer)
        for answer in ("y", " YES "):
            sends.allow(False)
            self.asked(answer)
            self.assertTrue(sends.allowed(), answer)

    def test_no_answer_leaves_it_off(self):
        def read(q):
            if "Endpoint" in q:
                return "meta"
            raise EOFError
        with redirect_stdout(io.StringIO()):
            self.assertTrue(parts.ask_api(read=read, read_secret=lambda q: "made-up-secret"), "the key is saved all the same")
        self.assertFalse(sends.allowed())

    def test_another_apis_key_is_not_followed_by_the_question_as_the_hearing_does_not_use_it(self):
        for endpoint, secret in (("baseten", "made-up"), ("fireworks", "made-up"), ("https://x--y.modal.direct/v1", "a.b")):
            self.assertEqual(self.asked(endpoint=endpoint, secret=secret)[1], [], endpoint)
        self.assertFalse(sends.allowed())
        mind.save_api("https://api.meta.ai/v1", "made-up-secret")
        self.assertEqual(self.asked(endpoint="baseten")[1], [], "a Meta key saved earlier is not this key")
        self.assertFalse(sends.allowed())

    def test_setup_with_nobody_at_the_keyboard_never_reaches_the_question(self):
        for argv, stdin in (([], io.StringIO()), (["--no-key"], mock.Mock(isatty=lambda: True))):
            with mock.patch.object(parts, "use_tools"), mock.patch.object(parts, "card", side_effect=SystemExit), \
                    mock.patch.object(parts, "ask_api") as ask, mock.patch.object(sys, "stdin", stdin):
                self.assertRaises(SystemExit, parts.main, argv)
            ask.assert_not_called()
        self.assertFalse(sends.allowed())


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
        for url, key in (("https://inference.baseten.co/v1", "baseten"),
                         ("https://api.fireworks.ai/inference/v1", "fireworks")):
            status, got, _ = self.ask("POST", {"url": url, "key": "made-up-key"})
            self.assertEqual((status, got["mind"]["key"], got["mind"]["ok"]), (200, key, True), url)

    def test_each_api_shows_its_own_faded_fields_and_an_empty_address_saves_its_own(self):
        forms = {a["name"]: a for a in self.ask("GET")[1]["apis"]}
        self.assertEqual({n: (a["address"], a["own"]) for n, a in forms.items()},
                         {"meta": ("https://api.meta.ai/v1", True),
                          "modal": ("https://<your endpoint>.modal.direct/v1", False),
                          "baseten": ("https://inference.baseten.co/v1", True),
                          "fireworks": ("https://api.fireworks.ai/inference/v1", True)})
        self.assertEqual(list(forms), ["meta", "modal", "baseten", "fireworks"], "in the order the page shows them")
        self.assertEqual({n: (a["key"], a["page"]) for n, a in forms.items()},
                         {"meta": ("Meta API key", "https://dev.meta.ai/api-keys/"),
                          "modal": ("Modal proxy token", "https://modal.com/settings/proxy-auth-tokens"),
                          "baseten": ("Baseten API key", "https://app.baseten.co/settings/api_keys"),
                          "fireworks": ("Fireworks API key", "https://app.fireworks.ai/settings/users/api-keys")},
                         "Get a key opens the page each provider's own docs link to")
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html"),
                  encoding="utf-8") as f:
            page = f.read()
        self.assertEqual({n: a["sub"] for n, a in forms.items()},
                         {"meta": "4 Muse models", "modal": "Spark Open", "baseten": "Spark Open",
                          "fireworks": "Spark Open"})
        icons = [a["icon"] for a in forms.values()]
        self.assertEqual(len(set(icons)), 4, "a tile is told apart by its shape too")
        for icon in icons:
            self.assertIn('symbol id="i-%s"' % icon, page, "the page's own icon, never a provider's logo")
        status, got, _ = self.ask("POST", {"api": "baseten", "url": "", "key": "made-up-baseten"})
        self.assertEqual((status, got["url"], got["mind"]["key"], got["mind"]["ok"]),
                         (200, "https://inference.baseten.co/v1", "baseten", True), "what the faded text shows")
        self.assertEqual(self.ask("POST", {"api": "meta", "url": " ", "key": "made-up-meta"})[1]["mind"]["key"], "spark")
        forms = {a["name"]: a for a in self.ask("GET")[1]["apis"]}
        self.assertEqual({n: (a["url"], a["saved"]) for n, a in forms.items()},
                         {"meta": ("https://api.meta.ai/v1", True), "modal": ("", False),
                          "baseten": ("https://inference.baseten.co/v1", True), "fireworks": ("", False)})
        self.assertEqual(self.ask("POST", {"api": "modal", "url": "", "key": "wk-a.ws-b"})[:2],
                         (400, {"ok": False, "error": "Give your Modal endpoint's address."}))
        status, got, _ = self.ask("POST", {"api": "fireworks", "url": "https://api.meta.ai/v1", "key": "made-up"})
        self.assertEqual((status, got["error"]), (400, "That address is not Fireworks's: leave it empty for "
                                                       "Fireworks's own, or pick the API it is for."))
        self.assertEqual(self.ask("POST", {"api": "baseten", "url": "ftp://nope", "key": "k"})[1]["error"],
                         "Give the endpoint's address, starting with https://.")

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
