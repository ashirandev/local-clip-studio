"""What Muse cost: prices, the studio's ledger, the day's rate and the page's question, with no key and no network:
python -m unittest discover -s tests"""
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, mind, server, spent  # noqa: E402
from test_agent_loop import ScriptedMind, call  # noqa: E402
from test_mind_stream import Fake  # noqa: E402

SPARK_ANSWER = [{"choices": [{"delta": {"content": "Done."}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 20000, "completion_tokens": 1000,
                           "completion_tokens_details": {"reasoning_tokens": 900},
                           "prompt_tokens_details": {"cached_tokens": 15000}}}]


class Priced(ScriptedMind):
    """A scripted mind whose every answer cost one cent."""

    def ask(self, *args, **kwargs):
        got = ScriptedMind.ask(self, *args, **kwargs)
        got["engine"] = dict(got["engine"], written=5, usd=0.01)
        return got


class Prices(unittest.TestCase):
    def setUp(self):
        Fake.chunks, Fake.pause, Fake.status, Fake.listing, Fake.seen = SPARK_ANSWER, 0, 200, 200, []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d/v1" % self.server.server_port
        self.had = os.environ.pop("LCS_TEST_SPENT_KEY", None)
        os.environ["LCS_TEST_SPENT_KEY"] = "made-up"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        os.environ.pop("LCS_TEST_SPENT_KEY", None)
        if self.had is not None:
            os.environ["LCS_TEST_SPENT_KEY"] = self.had

    def test_a_hosted_answer_is_priced_new_cached_and_written_with_its_thinking(self):
        spark = mind.Mind("spark", "Muse Spark", self.url, model="muse-spark-1.3", key_env="LCS_TEST_SPENT_KEY")
        got = spark.ask([{"role": "user", "content": "hi"}])["engine"]
        self.assertEqual((got["read_new"], got["reused"], got["written"]), (5000, 15000, 1000),
                         "of 20,000 tokens read, 15,000 came from the cache; 1,000 written, thinking included")
        self.assertAlmostEqual(got["usd"], (5000 * 1.25 + 15000 * 0.15 + 1000 * 4.25) / 1e6)

    def test_this_computer_costs_nothing_and_an_unknown_price_is_not_guessed(self):
        numbers = {"read_new": 100, "reused": 900, "written": 50}
        self.assertEqual(mind.Mind("local", "Muse Glimmer", self.url, model="muse-glimmer").cost(numbers), 0.0)
        other = mind.Mind("spark", "Muse Spark", self.url, model="muse-spark-9", key_env="LCS_TEST_SPENT_KEY")
        self.assertIsNone(other.cost(numbers))
        self.assertIsNone(other.ask([{"role": "user", "content": "hi"}])["engine"]["usd"])
        self.assertIsNone(mind.MINDS["spark"].cost({"read_new": None, "reused": None, "written": None}),
                          "an answer whose token counts are not known is not priced")
        self.assertEqual(mind.MINDS["modal"].price(), mind.PRICES["meta-models/Muse-Spark-Open-816B-A42B"])


class Ledger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "work", "spent.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_totals_by_message_chat_day_month_and_mind_with_unpriced_answers_apart(self):
        spark, glimmer = mind.MINDS["spark"], mind.MINDS["local"]
        now = time.mktime((2026, 10, 28, 20, 0, 0, 0, 0, -1))
        rows = [(now - 60, "a", 1, spark, 0.25), (now - 30, "a", 1, spark, 0.15), (now - 20, "a", 2, glimmer, 0.0),
                (now - 10, "a", 2, spark, None), (now - 2 * 86400, "b", 1, spark, 1.0),
                (time.mktime((2026, 9, 30, 12, 0, 0, 0, 0, -1)), "c", 1, spark, 4.0)]
        for t, chat, job, brain, usd in rows:
            with mock.patch.object(spent.time, "time", return_value=t):
                spent.add(self.path, chat, job, brain, {"read_new": 1, "reused": 2, "written": 3, "usd": usd})
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("a line cut short by a crash\n")
        got = spent.totals(self.path, "a", now=now)
        self.assertAlmostEqual(got["chat"], 0.40)
        self.assertEqual([(j["job"], round(j["usd"], 2), j["answers"], j["minds"]) for j in got["jobs"]],
                         [(1, 0.40, 2, ["Muse Spark"]), (2, 0.0, 1, ["Muse Glimmer"])])
        self.assertEqual((round(got["today"], 2), round(got["month"], 2), round(got["all"], 2)), (0.40, 1.40, 5.40))
        self.assertEqual({k: round(v, 2) for k, v in got["minds"].items()}, {"Muse Spark": 5.40, "Muse Glimmer": 0.0})
        self.assertEqual(got["unpriced"], 1)
        self.assertEqual(spent.totals(os.path.join(self.tmp, "none.jsonl"))["all"], 0.0)

    def test_every_answer_of_a_job_goes_into_the_ledger(self):
        work, agent.WORK = agent.WORK, os.path.join(self.tmp, "work")
        try:
            chat = agent.Chat()
            chat.jobs += 1
            job = chat.job = agent.Job(chat, Priced([call("set_plan", steps=["look"], why="plan")], "Done."), "hi")
            job.run()
        finally:
            agent.WORK = work
        got = spent.totals(self.path, chat.id)
        self.assertEqual((round(got["chat"], 2), [j["answers"] for j in got["jobs"]]), (0.02, [2]))


class Rates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "rates.json")
        self.fetched = 0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fetch(self):
        self.fetched += 1
        return {"time_last_update_utc": "Wed, 28 Oct 2026 00:02:31 +0000", "rates": {"USD": 1, "THB": 33.77}}

    def test_the_days_rate_is_fetched_once_and_kept_twelve_hours(self):
        with mock.patch.object(spent, "_fetch", self.fetch):
            got = spent.rate("thb", self.path)
            self.assertEqual(got, {"currency": "THB", "per_usd": 33.77, "day": "Wed, 28 Oct 2026",
                                   "source": "open.er-api.com"})
            spent.rate("THB", self.path)
            self.assertEqual(self.fetched, 1)
            with mock.patch.object(spent.time, "time", return_value=time.time() + spent.RATE_KEPT + 1):
                spent.rate("THB", self.path)
            self.assertEqual(self.fetched, 2)
            self.assertEqual(spent.rate("USD", self.path), {"currency": "USD", "per_usd": 1.0})
            self.assertEqual(spent.rate("../x", self.path), {"currency": "USD", "per_usd": 1.0})
            self.assertIn("no rate for XYZ", spent.rate("XYZ", self.path)["why"])

    def test_rates_that_cannot_be_fetched_give_dollars_and_say_why_and_are_not_asked_for_at_every_look(self):
        def fail():
            self.fetched += 1
            raise OSError("no network")
        with mock.patch.object(spent, "_fetch", fail):
            got = spent.rate("THB", self.path)
            spent.rate("THB", self.path)
        self.assertEqual((got["currency"], got["per_usd"]), ("USD", 1.0))
        self.assertIn("could not be fetched from open.er-api.com (no network)", got["why"])
        self.assertEqual(self.fetched, 1)


class Page(unittest.TestCase):
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

    def test_the_page_asks_what_was_spent_in_its_own_money(self):
        spent.add(os.path.join(agent.WORK, "spent.jsonl"), "a", 1, mind.MINDS["spark"], {"usd": 0.39})
        fetched = lambda: {"time_last_update_utc": "Wed, 28 Oct 2026 00:02:31", "rates": {"THB": 33.77}}  # noqa: E731
        with mock.patch.object(spent, "_fetch", fetched):
            conn = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=30)
            conn.request("GET", "/api/spent?chat=a&currency=THB")
            got = json.loads(conn.getresponse().read().decode("utf-8"))
            conn.close()
        self.assertEqual((got["ok"], got["chat"], got["jobs"][0]["job"], got["rate"]["per_usd"]), (True, 0.39, 1, 33.77))
        self.assertEqual((got["prices"], round(got["today"], 2)), (mind.PRICED, 0.39))


if __name__ == "__main__":
    unittest.main()
