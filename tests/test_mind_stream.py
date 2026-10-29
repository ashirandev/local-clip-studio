"""The mind's streamed answers, against a small fake server: python -m unittest discover -s tests"""
import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import mind  # noqa: E402

TOOL_CALL = [
    {"choices": [{"delta": {"reasoning_content": "The person wants "}}]},
    {"choices": [{"delta": {"reasoning_content": "the newest files."}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                            "function": {"name": "find_recordings", "arguments": "{\"why\": "}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"look\", \"newest\": 3}"}}]}}]},
    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
     "timings": {"prompt_n": 14, "cache_n": 1088, "prompt_per_second": 55.0, "predicted_n": 40,
                 "predicted_per_second": 52.3}},
]


class Fake(BaseHTTPRequestHandler):
    chunks, pause, status, asked, models, states, loads = TOOL_CALL, 0, 200, [], [], {}, []
    listing, seen = 200, []
    refusal = b'{"error": "the request exceeds the available context size"}'

    def do_GET(self):
        if self.listing != 200:
            self.send_response(self.listing)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        body = json.dumps({"data": [dict({"id": m}, **({"status": {"value": self.states[m]}} if m in self.states
                                                          else {})) for m in self.models]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/models/load":
            name = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"]
            Fake.loads.append((name, len(self.asked)))
            Fake.states[name] = "loading"
            threading.Timer(0.3, Fake.states.__setitem__, (name, "loaded")).start()
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        self.seen.append(dict(self.headers))
        self.asked.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(self.status)
        self.send_header("Content-Type", "text/event-stream" if self.status == 200 else "application/json")
        self.end_headers()
        if self.status != 200:
            self.wfile.write(self.refusal)
            return
        try:
            for i, chunk in enumerate(self.chunks):
                if i == 1:
                    time.sleep(self.pause)
                self.wfile.write(("data: %s\n\n" % json.dumps(chunk)).encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        except OSError:
            pass

    def log_message(self, *args):
        pass


class Stream(unittest.TestCase):
    def setUp(self):
        Fake.chunks, Fake.pause, Fake.status, Fake.refusal = TOOL_CALL, 0, 200, Fake.refusal
        Fake.listing, Fake.seen = 200, []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.mind = mind.Mind("test", "Test mind", "http://127.0.0.1:%d/v1" % self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_pieces_become_one_answer_with_its_engine_numbers(self):
        seen = []
        got = self.mind.ask([{"role": "user", "content": "hi"}], on_delta=lambda k, p: seen.append((k, p)))
        self.assertEqual(seen, [("thinking", "The person wants "), ("thinking", "the newest files.")])
        self.assertEqual(got["thinking"], "The person wants the newest files.")
        self.assertEqual(got["tool_calls"], [{"type": "function", "id": "c1", "function": {
            "name": "find_recordings", "arguments": "{\"why\": \"look\", \"newest\": 3}"}}])
        self.assertEqual(got["message"]["tool_calls"], got["tool_calls"])
        self.assertEqual((got["engine"]["read_new"], got["engine"]["reused"], got["engine"]["write_per_s"]),
                         (14, 1088, 52.3))

    def test_the_local_mind_names_its_model_so_a_router_server_can_route_it(self):
        url = "http://127.0.0.1:%d/v1" % self.server.server_port
        local = mind.Mind("local", "Muse Glimmer", url, model="muse-glimmer")
        Fake.asked = []
        local.ask([{"role": "user", "content": "hi"}])
        self.assertEqual(Fake.asked[-1]["model"], "muse-glimmer")
        Fake.models = ["muse-glimmer", "another"]
        self.assertEqual(local.status()["name"], "muse-glimmer", "a router lists it by its name")
        Fake.models = ["C:/models/Muse-Glimmer-30B-UD-Q3_K_XL.gguf"]
        self.assertEqual(local.status()["name"], "Muse-Glimmer-30B-UD-Q3_K_XL.gguf",
                         "a server of one model lists its file")
        Fake.models = []
        self.assertFalse(local.status()["ok"], "the server is asked, even with a model name")
        self.assertEqual(mind.MINDS["local"].model_id, os.environ.get("LCS_MIND_MODEL", "muse-glimmer"))

    def test_a_router_that_holds_glimmer_unloaded_is_asked_to_load_it_before_the_question(self):
        url = "http://127.0.0.1:%d/v1" % self.server.server_port
        local = mind.Mind("local", "Muse Glimmer", url, model="muse-glimmer")
        Fake.asked, Fake.loads, Fake.models, Fake.states = [], [], ["muse-glimmer"], {"muse-glimmer": "unloaded"}
        try:
            got = local.ask([{"role": "user", "content": "hi"}])
            self.assertEqual(Fake.loads, [("muse-glimmer", 0)], "loaded once, before the question was sent")
            self.assertEqual((len(Fake.asked), got["thinking"]), (1, "The person wants the newest files."))
            local.ask([{"role": "user", "content": "hi"}])
            Fake.states = {}
            local.ask([{"role": "user", "content": "hi"}])
            self.assertEqual(len(Fake.loads), 1, "a loaded model, or a server of one model, is just asked")
        finally:
            Fake.models, Fake.states = [], {}

    def test_cancel_stops_a_question_in_flight_within_a_second(self):
        Fake.pause = 10
        asking, out = mind.Asking(), {}

        def ask():
            try:
                self.mind.ask([{"role": "user", "content": "hi"}], asking=asking)
                out["how"] = "finished"
            except mind.Cancelled:
                out["how"] = "cancelled"
            out["at"] = time.time()

        thread = threading.Thread(target=ask)
        thread.start()
        time.sleep(0.5)
        cancelled_at = time.time()
        asking.cancel()
        thread.join(5)
        self.assertEqual(out.get("how"), "cancelled")
        self.assertLess(out["at"] - cancelled_at, 1.0)

    def test_a_refused_question_gives_the_servers_reason(self):
        Fake.status = 400
        with self.assertRaises(mind.MindError) as e:
            self.mind.ask([{"role": "user", "content": "hi"}])
        self.assertIn("HTTP 400", str(e.exception))
        self.assertIn("exceeds the available context size", str(e.exception))

    def test_a_question_longer_than_the_memory_says_how_long_it_was(self):
        Fake.status, Fake.refusal = 400, json.dumps({"error": {
            "code": 400, "message": "request (32788 tokens) exceeds the available context size (32768 tokens)",
            "type": "exceed_context_size_error", "n_prompt_tokens": 32788, "n_ctx": 32768}}).encode()
        try:
            with self.assertRaises(mind.TooLong) as e:
                self.mind.ask([{"role": "user", "content": "hi"}])
        finally:
            Fake.refusal = b'{"error": "the request exceeds the available context size"}'
        self.assertEqual((e.exception.asked, e.exception.room), (32788, 32768))
        self.assertIn("HTTP 400", str(e.exception))

    def test_an_answer_cut_where_the_memory_filled_is_not_an_answer(self):
        Fake.chunks = TOOL_CALL[2:4] + [{"choices": [{"delta": {}, "finish_reason": "length"}],
                                         "timings": {"prompt_n": 2062, "cache_n": 30591, "predicted_n": 115}}]
        with self.assertRaises(mind.TooLong) as e:
            self.mind.ask([{"role": "user", "content": "hi"}], max_tokens=4096)
        self.assertEqual((e.exception.asked, e.exception.room), (32768, 32768))
        self.assertIn("memory filled while it wrote its answer", str(e.exception))

    def test_thinking_that_runs_out_of_room_says_so(self):
        Fake.chunks = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]},
                       {"choices": [{"delta": {}, "finish_reason": "length"}]}]
        with self.assertRaises(mind.MindError) as e:
            self.mind.ask([{"role": "user", "content": "hi"}], max_tokens=4096)
        self.assertIn("4096 tokens without reaching an answer", str(e.exception))


class Modal(unittest.TestCase):
    NAMES = ("LCS_TEST_MODAL_KEY", "LCS_TEST_MODAL_SECRET", "LCS_MODAL_MODEL")

    def setUp(self):
        Fake.chunks, Fake.pause, Fake.status, Fake.listing, Fake.seen, Fake.models = TOOL_CALL, 0, 200, 200, [], []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.had = {n: os.environ.pop(n, None) for n in self.NAMES}
        self.mind = mind.Shared("modal", "Muse Spark Open", "http://127.0.0.1:%d/v1" % self.server.server_port,
                                model="meta-models/m", key_env=self.NAMES[:2], where="on Modal's servers")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        for n, v in self.had.items():
            os.environ.pop(n, None)
            if v is not None:
                os.environ[n] = v

    def test_the_token_is_its_key_and_its_secret_and_both_are_needed(self):
        why = ("Set LCS_TEST_MODAL_KEY and LCS_TEST_MODAL_SECRET to the key and the secret of your Modal proxy token, "
               "or name a file that holds them in LCS_KEYS, then start the studio again.")
        self.assertEqual(self.mind.status()["why"], why)
        os.environ["LCS_TEST_MODAL_KEY"] = "wk-made-up"
        self.assertEqual(self.mind.status()["why"], why, "a key without its secret is no token")
        with self.assertRaises(mind.MindError) as e:
            self.mind.ask([{"role": "user", "content": "hi"}])
        self.assertEqual(str(e.exception), why)
        self.assertEqual(Fake.seen, [], "nothing is sent without the whole token")
        os.environ["LCS_TEST_MODAL_SECRET"] = "ws-made-up"
        self.mind.ask([{"role": "user", "content": "hi"}])
        self.assertEqual(Fake.seen[-1]["Authorization"], "Bearer wk-made-up.ws-made-up")

    def test_ready_only_when_the_workspace_has_an_endpoint_for_the_model(self):
        os.environ["LCS_TEST_MODAL_KEY"], os.environ["LCS_TEST_MODAL_SECRET"] = "wk-a", "ws-b"
        self.assertEqual(self.mind.status()["why"], "Your Modal workspace has no shared endpoint for meta-models/m "
                                                    "yet. Make one in Modal's dashboard, under Endpoints.")
        Fake.models = ["another/model", "meta-models/m"]
        self.assertEqual((self.mind.status()["ok"], self.mind.status()["name"]), (True, "meta-models/m"))
        Fake.listing = 401
        self.assertIn("Modal refused the proxy token (HTTP 401)", self.mind.status()["why"])

    def test_one_conversation_stays_on_one_machine_and_another_need_not(self):
        os.environ["LCS_TEST_MODAL_KEY"], os.environ["LCS_TEST_MODAL_SECRET"] = "wk-a", "ws-b"
        start = [{"role": "system", "content": "You edit clips."}, {"role": "user", "content": "Clip the yeti."}]
        self.mind.ask(start)
        self.mind.ask(start + [{"role": "assistant", "content": "Looking."}, {"role": "user", "content": "Go on."}])
        self.mind.ask([start[0], {"role": "user", "content": "Clip the trip."}])
        keys = [h.get("Modal-Routing-Affinity-Key") for h in Fake.seen]
        self.assertTrue(keys[0] and keys[0] == keys[1], "the same conversation, the same machine")
        self.assertNotEqual(keys[2], keys[0])
        self.assertEqual(Fake.asked[-1]["model"], "meta-models/m")
        plain = mind.Mind("local", "Muse Glimmer", self.mind.url, model="muse-glimmer")
        plain.ask(start)
        self.assertNotIn("Modal-Routing-Affinity-Key", Fake.seen[-1])

    def test_a_hosted_mind_gets_its_schema_closed_and_room_to_think(self):
        os.environ["LCS_TEST_MODAL_KEY"], os.environ["LCS_TEST_MODAL_SECRET"] = "wk-a", "ws-b"
        Fake.chunks = [{"choices": [{"delta": {"content": "{\"lines\": [\"ok\"]}"}}]},
                       {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        schema = {"type": "object", "required": ["lines"], "properties": {
            "lines": {"type": "array", "items": {"type": "object", "required": ["text"], "properties": {
                "text": {"type": "string"}, "kept": {"type": "string"}}}}}}
        self.mind.ask([{"role": "user", "content": "fix"}], max_tokens=3000, schema=schema)
        sent = Fake.asked[-1]
        closed = sent["response_format"]["json_schema"]["schema"]
        self.assertEqual((closed["additionalProperties"], closed["required"]), (False, ["lines"]))
        item = closed["properties"]["lines"]["items"]
        self.assertEqual((item["additionalProperties"], item["required"]), (False, ["text", "kept"]),
                         "Meta refuses a schema with an object left open (HTTP 400, 10-28)")
        self.assertEqual(sent["max_tokens"], mind.HOSTED_ROOM, "Spark thought 6,050 tokens to fix 3 lines")
        local = mind.Mind("local", "Muse Glimmer", self.mind.url, model="muse-glimmer")
        local.ask([{"role": "user", "content": "fix"}], max_tokens=3000, schema=schema)
        self.assertEqual((Fake.asked[-1]["response_format"]["json_schema"]["schema"], Fake.asked[-1]["max_tokens"]),
                         (schema, 3000), "llama-server takes the schema and the room as they are")

    def test_a_model_named_in_a_key_file_counts_once_the_minds_are_made_again(self):
        self.assertEqual(mind.minds()["modal"].model_id, "meta-models/Muse-Spark-Open-816B-A42B")
        os.environ["LCS_MODAL_MODEL"] = "meta-models/another"
        self.assertEqual(mind.minds()["modal"].model_id, "meta-models/another")
        self.assertEqual(mind.minds()["modal"].key_env, ("MODAL_KEY", "MODAL_SECRET"))

    def test_a_mind_that_takes_few_pictures_is_sent_the_newest_and_the_conversation_keeps_them_all(self):
        def picture(n):
            return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,p%d" % n}}

        talk = [{"role": "user", "content": [{"type": "text", "text": "look"}, picture(1), picture(2)]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [picture(3)]},
                {"role": "user", "content": [{"type": "text", "text": "the sheet"}, picture(4), picture(5)]}]

        def sent():
            return [[p["image_url"]["url"][-2:] if p["type"] == "image_url" else p["text"] for p in m["content"]]
                    for m in Fake.asked[-1]["messages"] if isinstance(m["content"], list)]

        few = mind.Mind("few", "Few pictures", self.mind.url, pictures=3)
        few.ask(talk)
        out = mind.LEFT_OUT % 3
        self.assertEqual(sent(), [["look", out, out], ["p3"], ["the sheet", "p4", "p5"]])
        self.assertEqual(talk[0]["content"][1]["image_url"]["url"][-2:], "p1", "the conversation keeps its pictures")
        mind.Mind("all", "All pictures", self.mind.url).ask(talk)
        self.assertEqual(sent(), [["look", "p1", "p2"], ["p3"], ["the sheet", "p4", "p5"]], "others get them all")
        self.assertEqual({k: m.pictures for k, m in mind.minds().items()},
                         {"local": None, "spark": None, "wonder": None, "modal": None, "baseten": 8,
                          "fireworks": None}, "Baseten refused 9 pictures in one question (10-29)")


if __name__ == "__main__":
    unittest.main()
