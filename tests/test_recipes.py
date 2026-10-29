"""The recipes and the house standard every recipe ends with: python -m unittest discover -s tests"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import recipes  # noqa: E402

RECIPE = """---
name: %s
title: A kind
chip: It's a kind
description: When to pick it.
length: 30
style: clean
map: sound 1
---

# A kind

What a good clip of this kind is.
"""


class House(unittest.TestCase):
    def test_every_recipe_muse_loads_ends_with_the_house_standard_and_the_house_is_no_recipe(self):
        with open(os.path.join(recipes.FOLDER, recipes.HOUSE), encoding="utf-8") as f:
            house = f.read().strip()
        self.assertNotIn("house", recipes.RECIPES)
        self.assertGreaterEqual(len(recipes.RECIPES), 6)
        for name, recipe in recipes.RECIPES.items():
            self.assertTrue(recipe["body"].endswith("\n\n" + house), name)
            self.assertEqual(recipe["body"].count("# The house standard"), 1, name)
        for rule in ("Open on the clip's strongest line", "first word of a sentence", "a stranger scrolling past",
                     "Never a middle dot", "Never name another app", "Five hashtags"):
            self.assertIn(rule, house)

    def test_a_folder_of_recipes_takes_its_own_house_and_a_recipe_without_front_matter_is_refused(self):
        tmp = tempfile.mkdtemp()
        try:
            for name in ("one", "two"):
                with open(os.path.join(tmp, name + ".md"), "w", encoding="utf-8") as f:
                    f.write(RECIPE % name)
            with open(os.path.join(tmp, recipes.HOUSE), "w", encoding="utf-8") as f:
                f.write("\n# The house standard\n\n- Open on the strongest line.\n\n")
            got = recipes.load(tmp)
            self.assertEqual(sorted(got), ["one", "two"])
            self.assertEqual(got["one"]["body"], "# A kind\n\nWhat a good clip of this kind is.\n\n"
                                                 "# The house standard\n\n- Open on the strongest line.")
            with open(os.path.join(tmp, "three.md"), "w", encoding="utf-8") as f:
                f.write("# No front matter\n")
            with self.assertRaises(ValueError):
                recipes.load(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
