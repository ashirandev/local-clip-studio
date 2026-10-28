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
        self.assertEqual([g["name"] for g in groups][:2], ["Finding the moments", "Making clips"])
        for g in groups:
            for f in g["features"]:
                for k in ("name", "says", "ask"):
                    self.assertTrue(f[k].strip(), (f["tool"], k))
                self.assertIsNone(re.search(r"\b[a-z]+_[a-z_]+\b", f["says"] + " " + f["ask"]),
                                  "%s: a tool's name is for Muse, not the person" % f["name"])


if __name__ == "__main__":
    unittest.main()
