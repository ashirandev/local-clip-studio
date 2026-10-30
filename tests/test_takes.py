import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import takes  # noqa: E402


def line(a, b, text, sure=1.0):
    return {"from": a, "to": b, "text": text, "sure": sure}


def unsure(p):
    return p["sure"] < 0.88


# Tarm's review of a microphone, 10-28 (IMG_3375, 1:42-2:03), as the hearing heard it: he loses the thread, claps at
# 116.02 and 116.37, and says it again from 117.75.
GAVE_UP = [line(102.82, 104.84, "แต่สำหรับใครนะครับที่แบบอยากจะลองใช้", 0.95), line(104.96, 106.08, "อfฟ็cอื่นก็ลองได้นะ", 0.88),
           line(106.50, 107.06, "อันนี้จะไม่ใช่", 0.98), line(107.16, 107.34, "efฟ็c", 0.58),
           line(107.40, 108.14, "ขอโทษอันนี้จะเป็น", 0.95), line(108.38, 109.20, "presetของพวก", 0.78),
           line(109.40, 109.54, "eq", 0.89), line(109.78, 110.12, "ต่่าง", 0.90),
           line(110.36, 113.40, "แล้วก็พวกมันก็คือ", 0.87), line(113.48, 113.68, "ffcหน", 0.55),
           line(117.39, 117.61, "เอาละ", 0.86), line(117.75, 118.71, "ซึ่งตรงนี้นะครับก็จะเป็น", 0.92),
           line(118.77, 119.19, "pีset", 0.59),
           line(119.27, 123.23, "ที่ใช้สำหรับการแบบเหมือนฟังว่าเสียงมันชัดเจนแค่ไหนอะไรแบบนี้เนาะ", 0.95)]
# The same take, 1:16-1:31: he finishes a part, talks to himself about the screen, claps at 89.43 and 89.71.
ASIDE = [line(76.33, 77.51, "ตั้งเสียงประมาณนี้", 0.96), line(77.73, 78.55, "โอเคแล้วนะครับ", 0.85),
         line(79.91, 80.05, "โอ้โห", 0.87), line(80.17, 81.73, "จอแกมันซoมมากเลยปะ", 0.84),
         line(84.97, 86.39, "ันมันทำไงวนนี้ย", 0.77), line(87.45, 87.81, "เออไมนห", 0.60), line(88.29, 88.43, "โค", 0.72),
         line(90.36, 91.28, "แต่สำหรับใครที่แบบ", 0.99)]


def sound(seconds, voices=(), claps=(), keys=()):
    """Room noise, a hum of voice from a to b for each (a, b) in voices, a clap at each second in claps and a key
    pressed at each second in keys: a clap is a burst of noise dying in a few ms, a key a smaller, shorter one."""
    rate, rng = takes.RATE, np.random.default_rng(7)
    x = rng.normal(0, 0.0005, int(seconds * rate))
    for a, b in voices:
        t = np.arange(int((b - a) * rate)) / rate
        hum = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 560), 1))
        x[int(a * rate):int(a * rate) + len(t)] += 0.08 * hum * np.minimum(1, np.minimum(t, t[::-1]) / 0.03)
    for when, size, gone in [(c, 0.3, 0.008) for c in claps] + [(k, 0.12, 0.003) for k in keys]:
        n = np.arange(int(0.05 * rate))
        x[int(when * rate):int(when * rate) + len(n)] += rng.normal(0, size, len(n)) * np.exp(-n / (gone * rate))
    return x.astype(np.float32)


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


class Claps(unittest.TestCase):
    def test_a_clap_and_a_double_clap_are_found_among_the_voice(self):
        found = takes.claps(sound(9.0, voices=[(0.5, 2.5), (4.0, 6.0), (7.4, 8.6)], claps=[3.0, 6.6, 6.9]))
        self.assertEqual(len(found), 2, found)
        for (a, b, others), (x, y) in zip(found, [(3.0, 3.0), (6.6, 6.9)]):
            self.assertAlmostEqual(a, x, delta=0.02)
            self.assertAlmostEqual(b, y, delta=0.02)
            self.assertEqual(others, 0, "nothing else cracks near it")

    def test_a_voice_alone_is_no_clap(self):
        self.assertEqual(takes.claps(sound(6.0, voices=[(0.5, 2.0), (2.2, 3.0), (4.0, 5.5)])), [])

    def test_keys_pressed_in_a_run_are_never_alone(self):
        found = takes.claps(sound(6.0, voices=[(0.5, 1.5)], keys=[2.0, 2.2, 2.4, 2.6, 2.8]))
        self.assertTrue(found)
        self.assertTrue(all(others > 0 for _, _, others in found), found)


class AtClaps(unittest.TestCase):
    def test_a_take_given_up_at_a_clap_and_said_again_is_cut_from_its_first_wrong_word(self):
        (found,) = takes.at_claps(GAVE_UP, [[116.02, 116.37, 0]], unsure=unsure)
        self.assertEqual(found["why"], "again")
        self.assertEqual(found["cut"], [106.5, 117.39 - takes.AIR], "from อันนี้จะไม่ใช่ to just before the new take")
        self.assertEqual(takes.said_again(GAVE_UP, 100.0, 125.0), [], "too unlike for a take said again without the "
                                                                       "clap")

    def test_a_crack_with_others_near_it_is_a_key_not_a_clap(self):
        self.assertEqual(takes.at_claps(GAVE_UP, [[116.02, 116.37, 3]], unsure=unsure), [])

    def test_talk_off_the_script_before_a_clap_goes_with_it(self):
        (found,) = takes.at_claps(ASIDE, [[89.43, 89.71, 0]], unsure=unsure)
        self.assertEqual(found["why"], "unsure")
        self.assertEqual(found["cut"], [77.73, 90.36 - takes.AIR], "the finished line before it stays")
        self.assertTrue(found["first"].startswith("โอเคแล้วนะครับ โอ้โห จอแกมันซoมมากเลยปะ"), found["first"])

    def test_a_sure_line_then_a_new_one_keeps_both(self):
        said = [line(1.0, 3.0, "ซื้อได้ที่ร้านเลยนะครับ"), line(4.5, 6.5, "ต่อไปมาดูหน้าตั้งค่ากันนะครับ")]
        self.assertEqual(takes.at_claps(said, [[3.8, 3.8, 0]], unsure=unsure), [])

    def test_a_clap_before_the_first_word_is_a_slate_only_in_a_whole_take(self):
        said = [line(3.6, 6.0, "สวัสดีครับวันนี้มารีวิวไมค์")]
        self.assertEqual(takes.at_claps(said, [[3.07, 3.38, 0]], unsure=unsure), [])
        (found,) = takes.at_claps(said, [[3.07, 3.38, 0]], unsure=unsure, slates=True)
        self.assertEqual((found["why"], found["cut"]), ("slate", [3.02, 3.38 + takes.CLAP_TAIL]))

    def test_a_crack_inside_a_word_is_the_word(self):
        said = [line(1.0, 4.0, "หนึ่งหกล้านสีลายซิงแบบนี้นะครับ"), line(5.5, 7.0, "หนึ่งหกล้านสีลายซิงแบบนี้นะ")]
        self.assertEqual(takes.at_claps(said, [[2.2, 2.2, 0]], unsure=unsure, slates=True), [])


class Dips(unittest.TestCase):
    """Lane 4 10-30: a heard line is a sound piece, not a sentence."""
    rec = {"path": "x.mp4", "seconds": 300.0, "sound_tracks": 1}

    def cuts(self, stretches, start=100.0):
        """takes.dips over sound of stretches [(seconds, loud)], from start."""
        rate = takes.RATE
        sound = np.concatenate([(0.3 if loud else 0.01) * np.sin(np.arange(int(rate * s)) * 0.3) for s, loud in stretches])
        had, takes._decoded = takes._decoded, lambda rec, a, b: sound
        try:
            return takes.dips(self.rec, start, start + sum(s for s, _ in stretches))
        finally:
            takes._decoded = had

    def test_a_line_is_cut_at_the_middle_of_each_dip(self):
        self.assertEqual(self.cuts([(1.0, True), (0.2, False), (1.0, True), (0.4, False), (1.0, True)]), [101.1, 102.4])

    def test_a_dip_too_short_to_be_a_pause_is_not_a_cut(self):
        self.assertEqual(self.cuts([(1.0, True), (0.08, False), (1.0, True)]), [])

    def test_no_piece_is_too_short_to_hear(self):
        self.assertEqual(self.cuts([(0.1, True), (0.14, False), (1.0, True)]), [], "too near the start")
        self.assertEqual(self.cuts([(1.0, True), (0.14, False), (0.1, True)]), [], "too near the end")
        self.assertEqual(self.cuts([(1.0, True), (0.14, False), (0.1, True), (0.14, False), (1.0, True)]), [101.07],
                         "too near the last cut")

    def test_sound_that_cannot_be_read_has_no_dips(self):
        self.assertEqual(takes.dips({"path": "x.mp4", "seconds": 5.0, "sound_tracks": 0}, 0.0, 5.0), [])


class Level(unittest.TestCase):
    def test_the_level_is_one_value_every_tenth_of_a_second(self):
        rate = takes.RATE
        tone = 0.5 * np.sin(np.arange(rate // 2) * 0.07)
        level = takes._levels(np.concatenate([tone, np.zeros(rate // 2)]))
        self.assertEqual(len(level), 10)
        self.assertTrue(all(-10 < v < -8 for v in level[:5]), level)
        self.assertEqual(level[5:], [-100.0] * 5)

    def test_no_sound_has_no_level(self):
        self.assertIsNone(takes.heard_level({"path": "x.mp4", "seconds": 5.0, "sound_tracks": 0}))

    def quiet(self, stretches, t, under=-20.0, bed=None, reach=1.0):
        """quiet_near over a sound built of (seconds, amplitude) stretches, starting at second 0."""
        from unittest import mock
        rate = takes.RATE
        sound = np.concatenate([v * np.sin(np.arange(int(rate * s)) * 0.7) for s, v in stretches])
        rec = {"path": "x.mp4", "seconds": len(sound) / rate, "sound_tracks": 1}

        def decoded(rec, a, b):
            return sound[int(a * rate):int(b * rate)]

        with mock.patch.object(takes, "_decoded", decoded):
            return takes.quiet_near(rec, t, reach, under, bed=bed)

    def test_the_quiet_nearest_a_time_is_found_in_the_sound_itself(self):
        word = (1.0, 0.5)
        self.assertEqual(self.quiet([word, (0.2, 0.0), word, (0.4, 0.0), word], 1.5), [1.0, 1.2], "the nearer of two")
        self.assertEqual(self.quiet([word, (0.2, 0.0), word, (0.4, 0.0), word], 1.9), [2.2, 2.6])
        self.assertEqual(self.quiet([word, (0.2, 0.0), word], 1.1), [1.0, 1.2], "a time in the quiet is in it")

    def test_a_dip_too_short_to_be_a_gap_is_not_quiet(self):
        word = (1.0, 0.5)
        self.assertIsNone(self.quiet([word, (0.04, 0.0), word], 1.0), "the closure of a consonant is inside a word")
        self.assertEqual(self.quiet([word, (0.08, 0.0), word], 1.0), [1.0, 1.08])

    def test_nothing_quiet_within_reach_is_none(self):
        self.assertIsNone(self.quiet([(1.0, 0.5), (0.2, 0.0), (3.0, 0.5)], 3.0))
        self.assertIsNone(self.quiet([(4.0, 0.5)], 2.0))

    def test_over_a_bed_of_sound_the_gap_is_a_dip_against_what_is_around_it(self):
        over = [(1.0, 0.5), (0.2, 0.2), (1.0, 0.5)]
        self.assertIsNone(self.quiet(over, 1.0), "the music under the talk is not quiet")
        self.assertEqual(self.quiet(over, 1.0, bed=6.0), [1.0, 1.2])
        self.assertIsNone(self.quiet([(1.0, 0.5), (0.2, 0.4), (1.0, 0.5)], 1.0, bed=6.0), "a small dip is not a gap")

    def test_no_sound_read_has_no_quiet(self):
        self.assertIsNone(takes.quiet_near({"path": "x.mp4", "seconds": 5.0, "sound_tracks": 0}, 2.0, 1.0, -20.0))


class Without(unittest.TestCase):
    def test_cuts_come_out_of_the_pieces(self):
        self.assertEqual(takes.without([[0, 10], [20, 30]], [[2, 4], [8, 22], [29, 40]]),
                         [[0, 2], [4, 8], [22, 29]])
        self.assertEqual(takes.without([[0, 10]], []), [[0, 10]])
        self.assertEqual(takes.without([[0, 10]], [[0, 10]]), [])


if __name__ == "__main__":
    unittest.main()
