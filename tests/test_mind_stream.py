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
    chunks, pause, status = TOOL_CALL, 0, 200

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(self.status)
        self.send_header("Content-Type", "text/event-stream" if self.status == 200 else "application/json")
        self.end_headers()
        if self.status != 200:
            self.wfile.write(b'{"error": "the request exceeds the available context size"}')
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
        Fake.chunks, Fake.pause, Fake.status = TOOL_CALL, 0, 200
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

    def test_thinking_that_runs_out_of_room_says_so(self):
        Fake.chunks = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]},
                       {"choices": [{"delta": {}, "finish_reason": "length"}]}]
        with self.assertRaises(mind.MindError) as e:
            self.mind.ask([{"role": "user", "content": "hi"}], max_tokens=4096)
        self.assertIn("4096 tokens without reaching an answer", str(e.exception))


if __name__ == "__main__":
    unittest.main()
