"""Keys from a file of the person's own (LCS_KEYS), with made-up keys: python -m unittest discover -s tests"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import mind  # noqa: E402

NAMES = ("LCS_TEST_KEY_A", "LCS_TEST_KEY_B", "LCS_TEST_KEY_C", "LCS_TEST_KEY_D", "LCS_TEST_KEY_E")


class Keys(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.had = {n: os.environ.pop(n, None) for n in NAMES}

    def tearDown(self):
        for n, v in self.had.items():
            os.environ.pop(n, None)
            if v is not None:
                os.environ[n] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_key_file_fills_the_environment_but_never_over_a_key_already_set(self):
        first, second = os.path.join(self.tmp, "a.env"), os.path.join(self.tmp, "b.env")
        with open(first, "w", encoding="utf-8") as f:
            f.write("﻿# a comment\nLCS_TEST_KEY_A=made-up-a\n\nexport LCS_TEST_KEY_B = \"made-up b\"\n"
                    "LCS_TEST_KEY_C='made-up-c'\nnot a line\nLCS_TEST_KEY_D=\n")
        with open(second, "w", encoding="utf-8") as f:
            f.write("LCS_TEST_KEY_E=from-the-file\n")
        os.environ["LCS_TEST_KEY_E"] = "already-set"
        got = mind.load_keys(os.pathsep.join([first, os.path.join(self.tmp, "missing.env"), second]))
        self.assertEqual(got, ["LCS_TEST_KEY_A", "LCS_TEST_KEY_B", "LCS_TEST_KEY_C"], "names only, never values")
        self.assertEqual([os.environ.get(n) for n in NAMES],
                         ["made-up-a", "made-up b", "made-up-c", None, "already-set"])

    def test_no_brain_says_what_to_do_and_keeps_the_address_for_the_log(self):
        """Lane 5 11-02: a first run with no key ended on an address, a Windows error and llama-server."""
        here = mind.Mind("local", "Muse Glimmer", "http://127.0.0.1:9/v1", model="m")
        with contextlib.redirect_stdout(io.StringIO()) as log:
            said = here.unreachable(OSError("refused"))
        self.assertEqual(said, "Muse has no brain to use yet. Open Options and paste an API key, or press Set up for Muse Glimmer.")
        self.assertIn("http://127.0.0.1:9/v1", log.getvalue())

    def test_spark_says_where_its_key_can_come_from(self):
        spark = mind.Mind("spark", "Muse Spark", "https://example.invalid/v1", model="m", key_env="LCS_TEST_KEY_A")
        self.assertEqual(spark.status()["why"], "Paste your Meta API key in Options.", "lane 5 11-02: no key needs no restart")
        os.environ["LCS_TEST_KEY_A"] = "made-up"
        self.assertTrue(spark.status()["ok"])


if __name__ == "__main__":
    unittest.main()
