"""The Features menu shows everything Muse can do: a line for every tool it has, in words a person reads."""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import tools  # noqa: E402


class Features(unittest.TestCase):
    def test_every_tool_muse_has_is_in_the_menu_and_nothing_else_is(self):
        rows = [f for g in tools.features() for f in g["features"]]
        self.assertEqual(sorted(set(tools.TOOLS) - {f["tool"] for f in rows}), [],
                         "a new tool needs its line in features.json, or the menu falls behind the studio")
        self.assertEqual([f["tool"] for f in rows if f["tool"] not in tools.TOOLS], [], "no line for a tool that went")

    def test_each_line_is_plain_words_with_a_request_to_try(self):
        groups = tools.features()
        self.assertEqual([g["name"] for g in groups], list(tools.FEATURE_GROUPS), "grouped by the kind of work")
        for g in groups:
            for f in g["features"]:
                for k in ("name", "says", "ask"):
                    self.assertTrue(f[k].strip(), (f["tool"], k))
                self.assertIsNone(re.search(r"\b[a-z]+_[a-z_]+\b", f["says"] + " " + f["ask"]),
                                  "%s: a tool's name is for Muse, not the person" % f["name"])


    def test_each_line_has_a_picture_the_page_draws(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "app.js"),
                  encoding="utf-8") as f:
            page = f.read()
        drawn = set(re.findall(r'^  "?([a-z0-9-]+)"?: \{ parts:', page[page.index("const MINI = {"):], re.M))
        shows = [f["show"] for g in tools.features() for f in g["features"]]
        self.assertEqual(sorted({s for s in shows if s not in drawn}), [], "a line whose picture the page lacks")
        self.assertEqual(sorted(drawn - set(shows)), [], "a picture no line uses")

if __name__ == "__main__":
    unittest.main()
