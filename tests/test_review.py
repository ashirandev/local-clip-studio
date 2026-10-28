"""A product review told over several takes, cut by what each line does: python -m unittest discover -s tests"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import mind, review  # noqa: E402


def line(n, clip, a, b, text="..."):
    return {"n": n, "clip": clip, "from": a, "to": b, "text": text}


# Shaped on Tarm's yeti takes (10-28): clip 3 says what it is and what it plugs into, clip 5 the lights and the test
# from a distance, clip 7 the verdict, the price and the link; clip 6 is the settings, step by step.
LINES = [line(1, 3, 4.8, 11.5), line(2, 3, 12.9, 18.0), line(3, 3, 18.4, 21.9), line(4, 3, 39.3, 41.5),
         line(5, 3, 41.6, 44.6), line(6, 5, 18.3, 21.0), line(7, 5, 21.2, 26.4), line(8, 5, 31.6, 34.6),
         line(9, 5, 35.0, 41.6), line(10, 6, 7.0, 18.9), line(11, 7, 55.1, 57.8), line(12, 7, 57.8, 58.4),
         line(13, 7, 58.5, 63.8), line(14, 7, 65.0, 75.3)]
JOBS = {1: "what", 2: "point", 3: "point", 4: "use", 5: "use", 6: "look", 7: "look", 8: "proof", 9: "other",
        10: "other", 11: "verdict", 12: "verdict", 13: "verdict", 14: "ask"}
CLIPS = [{"n": 1, "talking": False}, {"n": 2, "talking": False}, {"n": 3, "talking": True},
         {"n": 4, "talking": True}, {"n": 5, "talking": True}, {"n": 6, "talking": True}, {"n": 7, "talking": True}]


class Asking(unittest.TestCase):
    def test_every_line_is_asked_about_by_its_number_and_the_answer_is_one_of_the_jobs(self):
        messages, schema = review.question([line(1, 3, 4.84, 11.5, "เดี๋ยวนี้ผมจะพาคนมาฟังเสียง"),
                                            line(2, 7, 58.5, 63.8, "ราคาไม่ถึงสองพันเลย")])
        asked = messages[-1]["content"]
        self.assertIn("1. take 3, 4.8 s: เดี๋ยวนี้ผมจะพาคนมาฟังเสียง", asked)
        self.assertIn("2. take 7, 58.5 s: ราคาไม่ถึงสองพันเลย", asked)
        for name in review.NAMES:
            self.assertIn("- %s: " % name, asked)
        self.assertEqual(schema["properties"]["lines"]["items"]["properties"]["job"]["enum"], list(review.NAMES))
        closed = mind._closed(schema)
        self.assertIs(closed["properties"]["lines"]["items"]["additionalProperties"], False,
                      "Meta's strict mode refuses an object that is not closed")

    def test_a_line_skipped_or_given_an_unknown_job_is_other(self):
        got = review.read(json.dumps({"lines": [{"n": 1, "job": "what"}, {"n": 2, "job": "price"},
                                                {"n": 99, "job": "proof"}], "hook": 2}), LINES[:3])
        self.assertEqual(got, {"jobs": {1: "what", 2: "other", 3: "other"}, "hook": 2})
        self.assertIsNone(review.read(json.dumps({"lines": [], "hook": 99}), LINES[:3])["hook"],
                          "a hook that is not one of the lines is no hook")
        with self.assertRaises(ValueError):
            review.read("not json", LINES)


class Kind(unittest.TestCase):
    def test_a_review_says_what_the_product_is_and_its_point_or_verdict(self):
        self.assertTrue(review.is_review(JOBS))
        self.assertTrue(review.is_review({1: "what", 2: "verdict"}))
        self.assertFalse(review.is_review({1: "look", 2: "other", 3: "verdict"}), "nothing says what it is")
        self.assertFalse(review.is_review({1: "what", 2: "look"}), "nothing says why it is worth having")


class Cut(unittest.TestCase):
    def test_lines_next_to_each_other_with_one_job_are_one_stretch_of_at_most_a_piece(self):
        found = review.stretches(LINES, JOBS)
        point = [s for s in found if s["job"] == "point"]
        self.assertEqual([(s["from"], s["to"], s["lines"]) for s in point], [(12.9, 21.9, [2, 3])])
        long = review.stretches([line(1, 3, 0.0, 5.0), line(2, 3, 5.5, 10.0), line(3, 3, 10.2, 12.0)],
                                {1: "what", 2: "what", 3: "what"})
        self.assertEqual([(s["from"], s["to"]) for s in long], [(0.0, 5.0), (5.5, 12.0)],
                         "a run longer than a piece is split, never cut inside a line")
        self.assertEqual(review.stretches([line(1, 3, 0.0, 14.0)], {1: "what"})[0]["to"], 14.0)

    def test_the_cut_follows_the_review_order_and_opens_on_the_proof(self):
        cut = review.draft(CLIPS, LINES, JOBS, 60)
        self.assertEqual([(p["clip"], p["start"], p["job"]) for p in cut["clips"]],
                         [(5, 31.6, "proof"), (3, 4.8, "what"), (3, 12.9, "point"), (3, 39.3, "use"),
                          (5, 18.3, "look"), (7, 55.1, "verdict"), (7, 65.0, "ask")])
        self.assertEqual(cut["length"], 51.1)

    def test_a_cut_too_long_drops_what_there_is_to_see_before_the_verdict(self):
        self.assertEqual([p["job"] for p in review.draft(CLIPS, LINES, JOBS, 40)["clips"]],
                         ["proof", "what", "point", "use", "verdict", "ask"])
        cut = review.draft(CLIPS, LINES, JOBS, 30)
        self.assertEqual([p["job"] for p in cut["clips"]], ["proof", "what", "point", "verdict"])
        self.assertLessEqual(cut["length"], 30 * review.OVER)
        short = review.draft(CLIPS, LINES, JOBS, 10)
        self.assertEqual([p["job"] for p in short["clips"]], ["proof", "what", "verdict"],
                         "the hook, what it is and the verdict make a review, however short")

    def test_the_hook_is_the_proof_nearest_a_few_seconds(self):
        jobs = {**JOBS, 1: "proof"}
        cut = review.draft(CLIPS, LINES, jobs, 60)
        self.assertEqual((cut["clips"][0]["clip"], cut["clips"][0]["start"]), (5, 31.6))

    def test_the_hook_muse_names_opens_alone_even_inside_another_job(self):
        """Spark marked three lines around the test from a distance as proof; the one C opened on is the line said
        while sitting far from the mic (10-28)."""
        jobs = {**JOBS, 9: "proof", 15: "proof"}
        lines = LINES + [line(15, 5, 28.9, 31.1)]
        cut = review.draft(CLIPS, lines, jobs, 60, hook=7)
        self.assertEqual((cut["clips"][0]["job"], cut["clips"][0]["start"], cut["clips"][0]["end"],
                          cut["clips"][0]["lines"]), ("proof", 21.2, 26.4, [7]))
        look = [p for p in cut["clips"] if p["job"] == "look"]
        self.assertEqual([(p["start"], p["end"]) for p in look], [(18.3, 21.0)], "the hook's line is used once")

    def test_a_second_look_comes_from_another_take_and_the_ask_is_the_short_one(self):
        lines = LINES + [line(15, 5, 50.0, 57.0), line(16, 4, 22.0, 25.0), line(17, 5, 39.1, 41.6)]
        cut = review.draft(CLIPS, lines, {**JOBS, 15: "look", 16: "look", 17: "ask"}, 80)
        self.assertEqual([(p["clip"], p["start"]) for p in cut["clips"] if p["job"] == "look"], [(4, 22.0), (5, 18.3)],
                         "the picture changes: the green light of take 4, not more of take 5")
        self.assertEqual([(p["clip"], p["start"]) for p in cut["clips"] if p["job"] == "ask"], [(5, 39.1)])
        short = review.draft(CLIPS, lines, {**JOBS, 15: "look", 16: "look", 17: "ask"}, 40)["clips"]
        self.assertEqual([(p["clip"], p["start"]) for p in short if p["job"] == "look"], [(5, 18.3)],
                         "when the cut is too long the second look goes before any job does")
        self.assertEqual([p["job"] for p in short], ["proof", "what", "point", "use", "look", "verdict", "ask"])

    def test_every_clip_not_used_is_left_out_with_its_reason(self):
        cut = review.draft(CLIPS + [{"n": 8, "talking": True}], LINES + [line(15, 8, 1.0, 3.0)],
                           {**JOBS, 15: "what"}, 60)
        why = {c["clip"]: c["why"] for c in cut["left_out"]}
        self.assertEqual(sorted(why), [1, 2, 4, 6, 8])
        self.assertIn("no one talks through it", why[1])
        self.assertIn("nothing said in it is part of the review", why[6])
        self.assertIn("said better in another take: what", why[8])


if __name__ == "__main__":
    unittest.main()
