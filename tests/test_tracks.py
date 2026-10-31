"""Face tracks from the faces found every half second: one track per person, never swapped, bridged over one missed
picture, smooth, inside the picture, and ready for effects.blur_track: python -m unittest discover -s tests"""
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import effects, recording, tools, tracks  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def found(pictures, size=(1920, 1080), every=0.5, start=10.0):
    """faces.sample()'s shape from a list of pictures, each a list of face boxes."""
    return {"size": list(size), "every": every,
            "samples": [{"t": round(start + k * every, 2), "faces": [list(b) + [0.9] for b in boxes]}
                        for k, boxes in enumerate(pictures)]}


def gaps(points):
    return [b[0] - a[0] for a, b in zip(points, points[1:])]


class Tracks(unittest.TestCase):
    def test_two_people_who_pass_each_other_stay_two_tracks(self):
        walker = [[100 + 40 * k, 300, 80, 80] for k in range(20)]
        sitter = [500, 300, 80, 80]
        for order in ("same", "swapped"):
            pictures = [[w, sitter] if order == "same" or k == 0 else [sitter, w] for k, w in enumerate(walker)]
            got = tracks.tracks(found(pictures))
            self.assertEqual(len(got), 2, order)
            moving = [tr for tr in got if tr["points"][-1][1] > 700][0]
            still = [tr for tr in got if tr is not moving][0]
            self.assertTrue(all(abs(p[1] - 500) < 1 for p in still["points"]), "the sitter's track never moves")
            xs = [p[1] for p in moving["points"]]
            self.assertEqual(xs, sorted(xs), "the walker's track only moves right (%s order)" % order)

    def test_a_face_missed_in_one_picture_is_bridged_and_in_two_the_track_ends(self):
        face = [600, 300, 90, 110]
        once = [[face] if k != 4 else [] for k in range(10)]
        (tr,) = tracks.tracks(found(once))
        self.assertLessEqual(max(gaps(tr["points"])), 0.5 + 1e-9)
        self.assertIn(12.0, [p[0] for p in tr["points"]], "a point is put where the face was missed")
        twice = [[face] if k not in (4, 5) else [] for k in range(10)]
        got = tracks.tracks(found(twice))
        self.assertEqual(len(got), 2, "missed in two pictures, the track ends")
        for tr in got:
            self.assertLessEqual(max(gaps(tr["points"])), 0.5 + 1e-9)
            effects.blur_track([[p[0] - 9.75] + p[1:] for p in tr["points"]])

    def test_a_face_that_jumps_across_the_picture_is_another_person(self):
        got = tracks.tracks(found([[[100, 300, 80, 80]], [[1500, 300, 80, 80]], [[1500, 300, 80, 80]]]))
        self.assertEqual(len(got), 2)
        got = tracks.tracks(found([[[100, 300, 80, 80]], [[110, 300, 200, 200]]]))
        self.assertEqual(len(got), 2, "a face more than half again bigger is another face")

    def test_the_points_are_smoothed(self):
        shaky = [[[600 + (8 if k % 2 else -8), 300, 90, 110]] for k in range(12)]
        (tr,) = tracks.tracks(found(shaky))
        inner = tr["points"][2:-2]
        self.assertLessEqual(max(abs(p[1] - 600) for p in inner), 4.01, [p[1] for p in inner])

    def test_the_track_is_held_half_a_picture_at_both_ends_and_kept_inside_the_picture(self):
        (tr,) = tracks.tracks(found([[[-20, 1000, 90, 110]] for _ in range(4)], size=(1920, 1080)))
        self.assertEqual((tr["from"], tr["to"]), (9.75, 11.75))
        self.assertEqual(tr["points"][0][1:], tr["points"][1][1:])
        for t, x, y, w, h in tr["points"]:
            self.assertTrue(x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080, (x, y, w, h))

    def test_the_face_that_sits_still_is_marked_steady_and_comes_first(self):
        streamer = [1690, 430, 92, 114]
        pictures = [[streamer] + ([[300 + 30 * k, 500, 60, 70]] if 3 <= k < 9 else []) for k in range(20)]
        got = tracks.tracks(found(pictures))
        self.assertEqual([tr["n"] for tr in got], [1, 2])
        self.assertTrue(got[0]["steady"])
        self.assertEqual(got[0]["seen"], 20)
        self.assertFalse(got[1]["steady"])
        self.assertEqual((got[1]["from"], got[1]["seen"]), (11.25, 6))
        self.assertIsNone(got[0]["crop"], "no crop without the recording")


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Fine(unittest.TestCase):
    """Faces on a moving camera, found every tracks.FINE s."""

    def test_a_shaking_face_keeps_its_path_and_a_wider_box(self):
        shaky = [[[300 + (40 if k % 2 else 0), 300, 100, 100]] for k in range(20)]
        (tr,) = tracks.tracks(found(shaky, every=tracks.FINE))
        xs = [p[1] for p in tr["points"][1:-1]]
        self.assertAlmostEqual(xs[1] - xs[0], 40, delta=1)
        self.assertAlmostEqual(tr["points"][5][3], 100 * tracks.WIDE, delta=1)

    def test_a_face_missed_for_most_of_a_second_is_bridged_and_for_longer_the_track_ends(self):
        def gap(missed):
            pictures = [[[300, 300, 100, 100]]] * 5 + [[]] * missed + [[[300, 300, 100, 100]]] * 5
            return tracks.tracks(found(pictures, every=tracks.FINE))
        self.assertEqual(len(gap(18)), 1)
        self.assertEqual(len(gap(24)), 2)

    def test_a_bridged_track_has_no_two_points_too_far_apart_for_the_blur(self):
        pictures = [[[300, 300, 100, 100]]] * 5 + [[]] * 18 + [[[300, 300, 100, 100]]] * 5
        (tr,) = tracks.tracks(found(pictures, every=tracks.FINE))
        self.assertLessEqual(max(gaps(tr["points"])), 0.5)
        effects.blur_tracks([tr["points"]])

    def test_it_is_held_a_quarter_of_a_second_at_both_ends(self):
        (tr,) = tracks.tracks(found([[[300, 300, 100, 100]]] * 20, every=tracks.FINE))
        self.assertAlmostEqual(tr["points"][1][0] - tr["points"][0][0], 0.25, places=2)
        self.assertAlmostEqual(tr["points"][-1][0] - tr["points"][-2][0], 0.25, places=2)

    def test_a_big_patch_found_in_two_pictures_is_no_face(self):
        pictures = [[[100, 300, 80, 80]] + ([[600, 100, 400, 400]] if k in (30, 31) else []) for k in range(60)]
        (tr,) = tracks.tracks(found(pictures, every=tracks.FINE))
        self.assertEqual(tr["box"][2], 80)

    def test_a_big_face_found_for_a_quarter_of_a_second_stays(self):
        (tr,) = tracks.tracks(found([[[600, 100, 400, 400]]] * tracks.SOLID, every=tracks.FINE))
        self.assertEqual(tr["box"][2], 400)

    def test_a_small_face_found_in_one_picture_stays(self):
        (tr,) = tracks.tracks(found([[[600, 100, 200, 200]]], every=tracks.FINE))
        self.assertEqual(tr["box"][2], 200)

    def test_at_half_a_second_a_big_face_in_one_picture_stays(self):
        (tr,) = tracks.tracks(found([[[600, 100, 400, 400]]]))
        self.assertEqual(tr["box"][2], 400)

    def test_seen_counts_half_seconds_as_the_person_was_told(self):
        (tr,) = tracks.tracks(found([[[300, 300, 100, 100]]] * 20, every=tracks.FINE))
        self.assertEqual(tr["seen"], 2)
        (once,) = tracks.tracks(found([[[300, 300, 100, 100]]], every=tracks.FINE))
        self.assertEqual(once["seen"], 1)

    def test_at_half_a_second_a_box_stays_the_size_the_finder_gave(self):
        (tr,) = tracks.tracks(found([[[300, 300, 100, 100]]] * 4))
        self.assertEqual(tr["points"][2][3], 100)

    def test_faces_that_stay_in_their_tracks_are_not_looked_for_again(self):
        still = found([[[100, 300, 80, 80], [600, 300, 80, 80], [1200, 300, 80, 80]]] * 20)
        with mock.patch.object(tracks.faces, "sample") as sample:
            self.assertIs(tracks.finer({}, 0, 10, still), still)
        sample.assert_not_called()

    def test_one_or_two_faces_passing_are_not_looked_for_again(self):
        few = found([[[100, 300, 80, 80]], [[1200, 300, 80, 80]], [], []])
        with mock.patch.object(tracks.faces, "sample") as sample:
            self.assertIs(tracks.finer({}, 0, 2, few), few)
        sample.assert_not_called()

    def test_faces_that_will_not_stay_in_a_track_are_looked_for_every_fine_step(self):
        moving = found([[[100 + 900 * (k % 2), 200 * (k % 5), 80, 80]] for k in range(20)])
        better = found([[[100, 300, 80, 80]]], every=tracks.FINE)
        with mock.patch.object(tracks.faces, "sample", return_value=better) as sample:
            self.assertIs(tracks.finer({"r": 1}, 2, 12, moving, on_start="stop"), better)
        sample.assert_called_once_with({"r": 1}, 2, 12, every=tracks.FINE, on_start="stop")

    def test_when_the_finer_look_fails_the_faces_already_found_stand(self):
        moving = found([[[100 + 900 * (k % 2), 200 * (k % 5), 80, 80]] for k in range(20)])
        with mock.patch.object(tracks.faces, "sample", side_effect=tracks.faces.NoFaces("no")):
            self.assertIs(tracks.finer({}, 0, 10, moving), moving)

    def test_faces_already_fine_are_not_looked_for_again(self):
        fine = found([[[100 + 900 * (k % 2), 200 * (k % 5), 80, 80]] for k in range(20)], every=tracks.FINE)
        with mock.patch.object(tracks.faces, "sample") as sample:
            self.assertIs(tracks.finer({}, 0, 1, fine), fine)
        sample.assert_not_called()


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class Drawings(unittest.TestCase):
    def test_a_finer_look_compares_pictures_half_a_second_apart(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        video = os.path.join(folder, "still.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=30:d=1.5", "-pix_fmt",
                        "yuv420p", video], check=True)
        rec = {"path": video, "size": 1, "mtime": 1, "width": 320, "height": 240, "name": "still.mp4"}
        against = []

        def changed(data, before, w, h, box):
            against.append(before)
            return 9.0

        with mock.patch.object(tracks.faces, "folder", return_value=folder), \
                mock.patch.object(tracks.faces, "_detector", return_value=lambda d, w, h: [[10, 10, 50, 50, 0.9]]), \
                mock.patch.object(tracks.faces, "_changed", changed):
            tracks.faces.sample(rec, 0, 1.5, every=tracks.FINE)
        back = round(tracks.faces.EVERY / tracks.FINE)
        self.assertEqual(sum(1 for b in against if b is None), back)
        self.assertGreater(len(against), back)


class FromTheClip(unittest.TestCase):
    def people(self, recipe, fine):
        job = types.SimpleNamespace(chat=types.SimpleNamespace(clip={"recipe": recipe}), started_work="stop")
        coarse = found([[[100, 300, 80, 80]]] * 20)
        with mock.patch.object(tools, "_found_faces", return_value=[(0.0, 10.0, coarse)]), \
                mock.patch.object(tools.tracks, "_crop"), \
                mock.patch.object(tools.tracks, "finer", return_value=fine) as finer:
            got = tools._people({"width": 1920, "height": 1080}, [[0.0, 10.0]], None, [0.0, 10.0], job=job)
        return got, finer

    def test_the_players_of_a_game_are_not_looked_for_again(self):
        for recipe in ("game", "sport"):
            with self.subTest(recipe):
                self.people(recipe, None)[1].assert_not_called()

    def test_the_people_of_a_trip_are_and_stop_on_the_page_ends_it(self):
        fine = found([[[100 + 3 * k, 300, 80, 80]] for k in range(200)], every=tracks.FINE, start=0.0)
        got, finer = self.people("montage", fine)
        self.assertEqual(finer.call_args.kwargs, {"on_start": "stop"})
        self.assertEqual(len(got), 1)

    def test_seen_and_of_stay_in_half_seconds(self):
        got = self.people("montage", found([[[100 + 3 * k, 300, 80, 80]] for k in range(200)], every=tracks.FINE, start=0.0))[0]
        self.assertEqual((got[0]["seen"], got[0]["of"]), (20, 20))

    def test_a_talk_built_without_a_job_is_not_looked_for_again(self):
        coarse = found([[[100, 300, 80, 80]]] * 20)
        with mock.patch.object(tools, "_found_faces", return_value=[(0.0, 10.0, coarse)]), \
                mock.patch.object(tools.tracks, "_crop"), \
                mock.patch.object(tools.tracks, "finer") as finer:
            tools._people({"width": 1920, "height": 1080}, [[0.0, 10.0]], None, [0.0, 10.0])
        finer.assert_not_called()


class Crops(unittest.TestCase):
    def test_each_person_gets_a_small_picture_of_their_face(self):
        tmp = tempfile.mkdtemp()
        work, recording.WORK = recording.WORK, os.path.join(tmp, "work")
        try:
            path = os.path.join(tmp, "two.mp4")
            subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                            "color=gray:s=640x360:r=30:d=3,drawbox=x=100:y=100:w=60:h=60:color=red:t=fill,"
                            "drawbox=x=450:y=200:w=60:h=60:color=blue:t=fill", "-c:v", "libx264", "-preset",
                            "ultrafast", "-pix_fmt", "yuv420p", path], check=True)
            rec = recording.probe(path)
            got = tracks.tracks(found([[[100, 100, 60, 60], [450, 200, 60, 60]]] * 4, size=(640, 360), start=0.0),
                                rec)
            for tr, colour in zip(sorted(got, key=lambda tr: tr["box"][0]), ("red", "blue")):
                raw = subprocess.run([FFMPEG, "-v", "error", "-i", tr["crop"], "-f", "rawvideo", "-pix_fmt", "rgb24",
                                      "-"], capture_output=True, check=True).stdout
                side = int((len(raw) // 3) ** 0.5)
                self.assertEqual(side, tracks.CROP_HEIGHT)
                mid = (side // 2 * side + side // 2) * 3
                r, g, b = raw[mid:mid + 3]
                self.assertTrue((r > 180 and b < 80) if colour == "red" else (b > 180 and r < 80), (colour, r, g, b))
        finally:
            recording.WORK = work
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
