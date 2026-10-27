import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import takes  # noqa: E402


def line(a, b, text):
    return {"from": a, "to": b, "text": text}


# Tarm's own retake, 10-27 (12:03-12:29 of a tutorial), as the hearing heard it: the line at 726.4 is said again
# from 738.5 after an 7 s stop, with a name heard two ways.
RETAKE = [line(722.9, 723.5, "และนี่แหละคือ"), line(723.9, 724.2, "ai"), line(724.3, 724.5, "นะครับ"),
          line(724.8, 725.0, "egret"), line(725.3, 725.4, "นะ"),
          line(726.4, 728.4, "สำหรับใครที่สนใจอยากจะดูข้อมูลของ"), line(728.5, 728.7, "eกret"),
          line(728.8, 730.2, "นะครับเข้าไปดูได้ก่อนใน"), line(730.4, 730.7, "aชhilan"), line(730.8, 731.3, "dot com เลย"),
          line(738.5, 741.0, "สำหรับใครที่อยากจะดูข้อมูลของโลขol"), line(741.1, 741.3, "ai"),
          line(741.4, 741.7, "egret"), line(741.7, 742.8, "นะครับสามารถเข้าไปดูได้ใน"), line(743.5, 743.8, "achilan"),
          line(743.9, 744.2, "dot com"), line(744.3, 744.7, "นะครับแล้วก็")]


class SaidAgain(unittest.TestCase):
    def test_the_first_take_of_a_line_said_again_is_cut_up_to_the_new_take(self):
        (found,) = takes.said_again(RETAKE, 719.0, 749.0)
        self.assertEqual(found["cut"], [726.4, 738.5 - takes.AIR])
        self.assertTrue(found["first"].startswith("สำหรับใครที่สนใจ"), "the take starts at its own first word, not at "
                        "the นะ ending the line before: %s" % found["first"])
        self.assertGreaterEqual(found["alike"], takes.ALIKE)

    def test_a_different_line_after_a_stop_is_not_a_take(self):
        said = RETAKE[:10] + [line(738.5, 741.0, "ต่อไปเราจะมาดูหน้าตั้งค่ากันนะครับ"), line(741.2, 743.0, "กดที่ปุ่มนี้")]
        self.assertEqual(takes.said_again(said, 719.0, 749.0), [])

    def test_a_line_said_again_without_a_stop_is_not_a_take(self):
        said = RETAKE[:10] + [dict(p, **{"from": p["from"] - 7.0, "to": p["to"] - 7.0}) for p in RETAKE[10:]]
        self.assertLess(said[10]["from"] - said[9]["to"], takes.RESTART)
        self.assertEqual(takes.said_again(said, 719.0, 749.0), [])

    def test_a_short_word_said_twice_is_emphasis_not_a_take(self):
        said = [line(10.0, 10.6, "ดีมาก"), line(12.0, 12.6, "ดีมาก"), line(12.7, 14.0, "จริงๆ นะครับ")]
        self.assertEqual(takes.said_again(said, 0.0, 20.0), [])

    def test_a_take_further_back_than_within_is_not_looked_for(self):
        said = [dict(p, **{"from": p["from"] + (20.0 if k >= 10 else 0.0), "to": p["to"] + (20.0 if k >= 10 else 0.0)})
                for k, p in enumerate(RETAKE)]
        self.assertEqual(takes.said_again(said, 719.0, 780.0), [])

    def test_only_lines_inside_the_moment_count(self):
        self.assertEqual(takes.said_again(RETAKE, 738.0, 749.0), [], "the first take is before the moment")


class Without(unittest.TestCase):
    def test_cuts_come_out_of_the_pieces(self):
        self.assertEqual(takes.without([[0, 10], [20, 30]], [[2, 4], [8, 22], [29, 40]]),
                         [[0, 2], [4, 8], [22, 29]])
        self.assertEqual(takes.without([[0, 10]], []), [[0, 10]])
        self.assertEqual(takes.without([[0, 10]], [[0, 10]]), [])


if __name__ == "__main__":
    unittest.main()
