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
    refusal = b'{"error": "the request exceeds the available context size"}'

    def do_GET(self):
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


if __name__ == "__main__":
    unittest.main()
