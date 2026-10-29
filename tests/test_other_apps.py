"""Muse's post and words on the clip name no other company's app: python -m unittest discover -s tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import tools  # noqa: E402


class OtherApps(unittest.TestCase):
    """Tarm 10-28: "# ไม่ควรมี tiktok", "Reel IG Facebook ใช้ได้"."""

    def test_another_apps_name_or_tag_is_refused(self):
        for said in ("ลง TikTok", "tik tok", "#TikTokThailand", "ติ๊กต็อก", "ติ๊กต๊อก", "ติกตอก", "ทิกทอก", "YouTube Shorts",
                     "ยูทูบ", "#shorts", "#ytshorts", "#fyp", "#fypシ", "#foryou", "#foryoupage", "CapCut", "Snapchat",
                     "Lemon8", "douyin", "Kwai"):
            with self.assertRaises(tools.Refused, msg=said):
                tools._other_apps("คลิปเด็ด", said)

    def test_meta_apps_and_everyday_words_are_kept(self):
        for said in ("#reels", "#instagram", "#facebook", "#reelsinstagram", "Reels บน IG และ Facebook", "กางเกง shorts",
                     "made for you", "#foryourinfo", "stock ตัวนี้", "ตึกต่อตึก", "#ikea"):
            self.assertIsNone(tools._other_apps(said), said)

    def test_the_page_names_no_other_companys_app(self):
        web = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
        for name in sorted(os.listdir(web)):
            if name.endswith((".html", ".js", ".css", ".json")):
                with open(os.path.join(web, name), encoding="utf-8") as f:
                    found = tools.OTHER_APPS.search(f.read())
                self.assertIsNone(found, "%s says %r (10-29: the feature tile for a post showed #fyp)"
                                  % (name, found and found.group(0)))


if __name__ == "__main__":
    unittest.main()
