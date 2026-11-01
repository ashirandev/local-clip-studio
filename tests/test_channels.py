"""Channel looks kept on this computer: saved by name (any letters), one in use for the whole studio, and nothing
half-written or wrong kept."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import channels, recording  # noqa: E402


class Channels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.was, recording.WORK = recording.WORK, self.tmp

    def tearDown(self):
        recording.WORK = self.was
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_look_is_kept_by_its_name_and_the_one_in_use_outlives_the_studio(self):
        self.assertEqual((channels.listed(), channels.in_use()), ([], None))
        channels.save({"name": "  ตัวอย่าง  Society ", "handle": "@sample", "look": "warm", "captions": "lit"})
        channels.save({"name": "Ashiran", "style": {"name": "pop", "font": None}, "spot": "bottom right"})
        self.assertEqual([c["name"] for c in channels.listed()], ["Ashiran", "ตัวอย่าง Society"], "spaces tidied")
        channels.use("ตัวอย่าง Society")
        self.assertEqual(channels.in_use()["handle"], "@sample")
        channels.save({"name": "ตัวอย่าง Society", "handle": "@sample2"})
        self.assertEqual(len(channels.listed()), 2, "the same name replaces its look")
        self.assertEqual(channels.in_use()["handle"], "@sample2")
        self.assertEqual(sorted(os.listdir(channels.folder()))[-1], channels.IN_USE, "no part file is left behind")

    def test_what_a_look_cannot_have_is_refused_with_a_sentence(self):
        for look, said in (({"name": ""}, "Give the channel a name of 1 to 40 letters."),
                           ({"name": "x" * 41}, "Give the channel a name of 1 to 40 letters."),
                           ({"name": "A", "handle": "@" + "a" * 30}, "keep it to 30"),
                           ({"name": "A", "spot": "top"}, "use bottom left, bottom middle, bottom right"),
                           ({"name": "A", "captions": "loud"}, "use plain, lit, pop, off, or none")):
            with self.assertRaises(ValueError) as e:
                channels.save(look)
            self.assertIn(said, str(e.exception))
        with self.assertRaises(ValueError) as e:
            channels.use("Nobody")
        self.assertEqual(str(e.exception), "There is no channel look called Nobody. Saved looks: none yet.")
        self.assertEqual(channels.listed(), [])

    def test_a_removed_look_is_no_longer_in_use(self):
        channels.use(channels.save({"name": "Ashiran"})["name"])
        self.assertTrue(channels.remove("Ashiran"))
        self.assertFalse(channels.remove("Ashiran"))
        self.assertIsNone(channels.in_use())
        channels.use(None)
        self.assertIsNone(channels.in_use())


if __name__ == "__main__":
    unittest.main()
