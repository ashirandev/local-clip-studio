"""The app's icon: a computer screen with a video timeline, one picture for the page's header and the browser's tab.
Tarm 11-01: "มันดูแล้วไม่รู้ว่าเป็นโปรแกรมตัดต่อวิดีโอที่อยู่ในคอมพิวเตอร์"."""
import os
import re
import unittest
import xml.etree.ElementTree as ET

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def read(name):
    with open(os.path.join(WEB, name), encoding="utf-8") as f:
        return f.read()


class Icon(unittest.TestCase):
    def test_the_header_and_the_tab_show_one_picture(self):
        page = read("index.html")
        self.assertEqual(len(re.findall(r'<link rel="icon" type="image/svg\+xml" href="icon\.svg">', page)), 1)
        self.assertEqual(len(re.findall(r'<img class="mark" src="icon\.svg" alt=""', page)), 1)
        self.assertNotIn('<svg class="mark"', page, "two drawings would drift apart")

    def test_the_picture_is_drawn_in_the_colours_of_the_page(self):
        root = ET.fromstring(read("icon.svg"))
        drawn = {e.get("fill") for e in root.iter() if e.get("fill")}
        self.assertTrue(drawn)
        self.assertLessEqual(drawn, set(re.findall(r"#[0-9A-F]{6}", read("style.css"))))

    def test_the_screen_stands_on_a_tile_and_holds_a_timeline_with_its_playhead(self):
        root = ET.fromstring(read("icon.svg"))
        boxes = [e for e in root.iter() if e.tag.endswith("rect")]
        tile, screen = boxes[0], boxes[1]
        self.assertEqual((tile.get("width"), tile.get("height")), ("36", "36"))
        self.assertGreater(int(screen.get("width")), int(screen.get("height")), "a wide screen, not a phone")
        self.assertEqual(boxes[-1].get("fill"), "#FFD27A", "the playhead crosses the clips on top of them")


if __name__ == "__main__":
    unittest.main()
