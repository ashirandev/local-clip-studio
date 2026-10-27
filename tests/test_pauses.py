"""The pauses cut out of a clip: python -m unittest discover -s tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import pauses, project  # noqa: E402

SAID = [{"from": 1.0, "to": 2.0, "text": "วันนี้"}, {"from": 2.3, "to": 3.0, "text": "ลดราคา"},
        {"from": 5.0, "to": 6.0, "text": "ทั้งร้าน"}, {"from": 6.2, "to": 6.5, "text": "เอ่อ"},
        {"from": 8.0, "to": 9.0, "text": "ซื้อเลย"}]


def length(pieces):
    return sum(b - a for a, b in pieces)


def covered(spans, a, b):
    """True when [a, b] lies inside one of spans."""
    return any(x - 1e-9 <= a and b <= y + 1e-9 for x, y in spans)


class Tighten(unittest.TestCase):
    def test_pauses_go_and_the_words_stay_with_air_around_them(self):
        kept, cuts = pauses.tighten([[0, 10]], SAID)
        self.assertEqual(kept, [[0.8, 3.2], [4.8, 6.2], [7.8, 9.2]], "0.3 s between two words is not a pause; the "
                                                                    "filler at 6.2 is")
        self.assertEqual(cuts, [[0, 0.8], [3.2, 4.8], [6.2, 7.8], [9.2, 10]])

    def test_nothing_is_lost_but_pauses_and_no_word_is_ever_cut(self):
        cases = [([[0, 10]], SAID, 0.7, 0.2), ([[0, 4], [4.5, 10]], SAID, 0.5, 0.1),
                 ([[1.5, 5.5], [7.9, 8.6]], SAID, 1.2, 0.3), ([[0, 10]], SAID, 0.3, 0.0),
                 ([[0.0, 2.1], [2.2, 9.5]], SAID, 2.0, 0.5)]
        for pieces, said, gap, air in cases:
            kept, cuts = pauses.tighten(pieces, said, gap, air)
            self.assertAlmostEqual(length(kept) + length(cuts), length(pieces), 6, (pieces, gap))
            for p in said:
                if pauses.filler(p["text"]):
                    continue
                for a, b in pieces:
                    lo, hi = max(a, p["from"]), min(b, p["to"])
                    if lo < hi:
                        self.assertTrue(covered(kept, lo, hi), "%s (%s to %s) is kept whole in %s"
                                        % (p["text"], lo, hi, kept))
            for x, y in cuts:
                self.assertGreater(y, x)
            starts = [a for a, _ in kept]
            order = [next(k for k, (a, b) in enumerate(pieces) if a - 1e-9 <= s <= b) for s in starts]
            self.assertEqual(order, sorted(order), "the pieces keep the clip's order")

    def test_what_is_left_has_no_pause_longer_than_gap(self):
        for gap, air in ((0.7, 0.2), (0.4, 0.1), (1.5, 0.0)):
            kept, _ = pauses.tighten([[0, 10]], SAID, gap, air)
            words = [(p["from"], p["to"]) for p in SAID if not pauses.filler(p["text"])]
            for a, b in kept:
                edges = sorted({a, b} | {v for x, y in words for v in (x, y) if a < v < b})
                quiet = [(u, v) for u, v in zip(edges, edges[1:])
                         if not any(x <= u and v <= y for x, y in words)]
                self.assertTrue(all(v - u <= gap + 1e-9 for u, v in quiet), (gap, (a, b), quiet))

    def test_a_piece_where_nobody_speaks_is_left_out_and_edges_with_words_stay(self):
        kept, cuts = pauses.tighten([[3.2, 4.8], [7.8, 9.1]], SAID)
        self.assertEqual(kept, [[7.8, 9.1]], "the words reach the piece's end, so its end stays")
        self.assertEqual(cuts, [[3.2, 4.8]])

    def test_a_moment_of_the_old_clip_is_found_in_the_new_one(self):
        old = [[0, 10]]
        kept, _ = pauses.tighten(old, SAID)
        at = project.to_clip(old, 5.5)[1]
        self.assertAlmostEqual(project.from_recording(kept, at), (3.2 - 0.8) + (5.5 - 4.8), 6,
                               "a title on 'ทั้งร้าน' moves with the word")
        self.assertIsNone(project.from_recording(kept, project.to_clip(old, 4.0)[1]), "a moment in a cut pause is gone")

    def test_fillers(self):
        for said in ("เอ่อ", "อ่าาา", "อืมม", "เอ่อ อ่า...", "um, uh", "Hmm", "อ่ะ", "อืฮึ", "เฮ้อ"):
            self.assertTrue(pauses.filler(said), said)
        for said in ("เอ่อ ลดราคา", "อะไร", "อ่านต่อ", "umbrella", "", None):
            self.assertFalse(pauses.filler(said), said)

    def test_refusals_name_the_field_and_what_is_allowed(self):
        for kwargs, said in (({"gap": 0.1}, "gap 0.1 is not a pause length: use 0.3 to 5.0 seconds"),
                             ({"air": 2}, "air 2 is not the air left around words: use 0.0 to 1.0 seconds"),
                             ({"gap": 0.7, "air": 0.4}, "air 0.4 is too much for gap 0.7: keep 2 x air under gap"),
                             ({"pieces": [[3, 1]]}, r"pieces \[\[3, 1\]\] are not pieces"),
                             ({"said": None}, "Nobody has heard this moment yet"),
                             ({"said": [{"from": 1, "to": 2, "text": "อืม"}]}, "Nobody says a word in these pieces")):
            args = dict({"pieces": [[0, 10]], "said": SAID}, **kwargs)
            with self.assertRaisesRegex(ValueError, said):
                pauses.tighten(**args)


if __name__ == "__main__":
    unittest.main()
