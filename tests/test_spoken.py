"""Numbers as they are said: digits in any script, Thai and English number words.

    python -m unittest discover -s tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import spoken  # noqa: E402


class Spoken(unittest.TestCase):
    def test_english_number_words_are_read_as_one_number_each(self):
        cases = {"seventy five square metres": {75}, "it is twenty-one": {21}, "one hundred and twenty": {120},
                 "two thousand five hundred": {2500}, "three hundred thousand": {300000}, "a hundred bucks": {100},
                 "ninety nine point nine nine": {99.99}, "one point five": {1.5}, "zero": {0}}
        for said, want in cases.items():
            self.assertEqual(spoken.values(said), {float(n) for n in want}, said)

    def test_words_that_cannot_follow_each_other_start_the_next_number(self):
        self.assertEqual(spoken.english("one two"), {1.0, 2.0})
        self.assertEqual(spoken.english("one and two"), {1.0, 2.0})
        self.assertEqual(spoken.english("twelve fifteen"), {12.0, 15.0})
        self.assertEqual(spoken.english("I have a cat and a point"), set())

    def test_digits_are_read_in_any_script(self):
        self.assertEqual(spoken.values("๗๕ บาท"), {75.0})
        self.assertEqual(spoken.values("٧٥"), {75.0})
        self.assertEqual(spoken.values("७५"), {75.0})
        self.assertEqual(spoken.values("1,990 baht"), {1990.0})
        self.assertEqual(spoken.values("เจ็ดสิบห้า"), {75.0})

    def test_the_language_says_whether_number_words_are_read(self):
        self.assertTrue(spoken.reads("Thai", []))
        self.assertTrue(spoken.reads("ไทย", []))
        self.assertTrue(spoken.reads("English", []))
        self.assertFalse(spoken.reads("Spanish", ["setenta y cinco"]))
        self.assertTrue(spoken.reads("", ["สวัสดีครับ"]), "Thai letters")
        self.assertTrue(spoken.reads("", ["hello there"]), "Latin letters are read as English")
        self.assertFalse(spoken.reads("", ["七十五平方米"]))


if __name__ == "__main__":
    unittest.main()
