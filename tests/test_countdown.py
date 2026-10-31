"""A countdown of the best moments of one recording, the best last, on small videos made by ffmpeg:
python -m unittest test_countdown"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clipstudio import agent, countdown, hearing, recording, screen, tools  # noqa: E402
from test_agent_loop import ScriptedMind, call  # noqa: E402
from test_game_screen import MENU, PLAY  # noqa: E402
import test_editor  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
BEAT_ONE = "เอาช่วงที่มันส์ที่สุด"
SELLER = "ทำคลิปสั้นจาก C:\\Users\\hunte\\Videos\\TravelTest\\trip_2026-08-03.mov ให้พร้อมโพสต์"
TRIP = "ทำคลิปเที่ยวจากวิดีโอใน H:\\ContentFootage\\Misc\\20261026 ให้พร้อมโพสต์"


class Asked(unittest.TestCase):
    def test_only_the_persons_own_words_asking_for_a_top_n_or_a_countdown_open_it(self):
        for said, n in (("ทำ Top 5 ช่วงที่มันส์ที่สุด", 5), ("ขอท็อป3 ช่วงเด็ด", 3), ("ทอปห้า ฉากตาย", 5),
                        ("ห้าอันดับช่วงที่ฮาที่สุด", 5), ("make a top five of my stream", 5),
                        ("นับถอยหลังช่วงเด็ดให้หน่อย", 0), ("a countdown of the best fights", 0)):
            self.assertEqual(countdown.asked([said]), n, said)
        for said in (BEAT_ONE, SELLER, TRIP, "อันดับแรกเอาช่วงที่ตายก่อน", "ซื้อแล็ปท็อป 5 เครื่อง",
                     "เขาเล่นท็อปฟอร์มมาก", "stop 5 times", "my laptop 3 died", "ทำคลิป"):
            self.assertIsNone(countdown.asked([said]), "the demo beats ask for no countdown: %s" % said)

    def test_the_latest_message_that_asks_is_the_one_counted(self):
        self.assertEqual(countdown.asked(["ทำ top 3", "เปลี่ยนเป็น top 5 แทน"]), 5)
        self.assertEqual(countdown.asked(["ทำ top 3", "ชื่อเกมคือ Sifu"]), 3)


class Piece(unittest.TestCase):
    def test_a_piece_is_seven_seconds_before_the_peak_to_four_after_inside_the_moment(self):
        self.assertEqual(countdown.piece({"start": 100.0, "end": 130.0, "t": 120.0}), [113.0, 124.0])
        self.assertEqual(countdown.piece({"start": 100.0, "end": 130.0, "t": 101.0}), [100.0, 111.0])
        self.assertEqual(countdown.piece({"start": 100.0, "end": 130.0, "t": 129.0}), [119.0, 130.0])
        self.assertEqual(countdown.piece({"start": 100.0, "end": 108.0, "t": 104.0}), [100.0, 108.0])
        self.assertEqual(countdown.piece({"start": 100.0, "end": 130.0, "t": None}), [108.0, 119.0], "the middle")

    def test_a_piece_ends_just_after_the_death_screen_shows_so_the_fight_before_it_is_seen(self):
        """The Sifu countdown, 10-29: each moment's loudest second was the laugh at the death screen after the fight,
        and 21 of the clip's 30 pictures 2 s apart were death, skill and age screens."""
        m = {"start": 100.0, "end": 130.0, "t": 120.0}
        self.assertEqual(countdown.piece(m, [[118.0, 125.0]]), [108.5, 119.5], "the laugh at 120 s, the death at 118 s")
        self.assertEqual(countdown.piece(m, [[104.0, 106.0], [118.0, 125.0]]), [108.5, 119.5], "the latest before it")
        self.assertEqual(countdown.piece(m, [[112.0, 113.0], [118.0, 125.0]]), [113.0, 119.5], "not inside the one before")
        self.assertEqual(countdown.piece(m, [[124.0, 128.0]]), [114.5, 125.5], "else the first after the peak")
        self.assertEqual(countdown.piece(dict(m, t=101.0), [[96.0, 104.0]], earliest=90.0), [90.0, 97.5],
                         "before the moment's start")
        self.assertIsNone(countdown.piece(m, [[101.0, 130.0]]), "no play before it")

    def test_a_peak_in_the_play_after_a_death_ends_its_piece_at_the_next_death(self):
        """Moment 2 of the Sifu top 5, 10-30: its picture was at 35:46, in the fight after the death at 35:37 (35:37 to
        35:43); the piece ended at that death with 1.5 s of play and was refused three times, though the fight from
        35:43 ended in the death at 35:56."""
        m = {"start": 100.0, "end": 130.0, "t": 110.0}
        self.assertEqual(countdown.piece(m, [[102.0, 106.0], [118.0, 125.0]]), [108.5, 119.5], "the next death")
        self.assertEqual(countdown.piece(dict(m, t=120.0), [[118.0, 125.0], [127.0, 129.0]]), [108.5, 119.5],
                         "the death the peak is on comes before one after it")
        self.assertEqual(countdown.piece(m, [[102.0, 106.0]]), [100.0, 103.5],
                         "no death after it: the one before, as moment 5 picked a skill at 1:42:31 after its death")
        self.assertIsNone(countdown.piece(dict(m, t=111.0), [[101.0, 106.0]]), "under 3 s of play before it")

    def test_a_run_of_the_menu_starts_at_its_first_word_and_ends_at_the_next_reading_without_one(self):
        def at(t, *words, sure=0.98):
            return {"at": t, "lines": [{"text": w, "box": [0, 0, 1, 1], "sure": sure} for w in words]}

        sifu = [at(10, "Age32"), at(11, "DEATH"), at(12, "DEATH", "Give Up"), at(13, "DEATH", "GiveUp", "UNLOCK"),
                at(15, "Age33"), at(16, "DEATH"), at(17, "Age33")]
        self.assertEqual(countdown.menu_runs(sifu, 2), [[11, 15]], "DEATH comes a second before Give Up; alone, no run")
        self.assertEqual(countdown.menu_runs(sifu[:4], 2), [[11, 13]], "a run to the last reading")
        rise = sifu[:4] + [at(14, "RISE"), at(15, "RISE", "Age33"), at(16, "Age33")]
        self.assertEqual(countdown.menu_runs(rise, 2), [[11, 16]], "the RISE after a death is still its menu")
        self.assertEqual(countdown.menu_runs([at(10, "DEATH", "GiveUp", sure=0.5)], 2), [], "read surely")
        hint = [at(t, "Continue") for t in range(10)] + [at(10, "Continue", "DEATH", "Give Up"), at(11, "Age")]
        self.assertEqual(countdown.menu_runs(hint, 2), [[7, 11]], "a hint all through play reaches back LEAD s at most")

    def test_the_pieces_start_one_after_another(self):
        self.assertEqual(countdown.starts([[10, 21], [50, 61.5], [90, 100]]), [0.0, 11.0, 22.5])


class Graph(unittest.TestCase):
    def test_the_recordings_own_size_and_only_its_first_sound_track(self):
        rec = {"path": "/s.mkv", "width": 1281, "height": 721, "sound_tracks": 2}
        args, chain = countdown.graph(rec, [[10, 21], [50, 61]])
        self.assertEqual(args, ["-ss", "10.000", "-t", "11.000", "-i", "/s.mkv", "-ss", "50.000", "-t", "11.000",
                                "-i", "/s.mkv"])
        self.assertIn("[0:v:0]scale=1280:720,", chain, "even, as H.264 needs")
        self.assertIn("[1:a:0]aresample", chain)
        self.assertNotIn("a:1", chain, "the render and the hearing read the first track")
        self.assertIn("concat=n=2:v=1:a=1[v][a]", chain)
        self.assertIn("anullsrc", countdown.graph(dict(rec, sound_tracks=0), [[10, 21]])[1])


@unittest.skipUnless(FFMPEG and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class Join(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work, recording.WORK = recording.WORK, os.path.join(self.tmp, "work")

    def tearDown(self):
        recording.WORK = self.work
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_pieces_join_in_order_at_the_recordings_size_with_its_first_track(self):
        """A stream with two sound tracks, the first silent and the second a loud tone: the countdown keeps the
        first."""
        path = os.path.join(self.tmp, "stream.mkv")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=12", "-f", "lavfi",
                        "-i", "anullsrc=r=48000:cl=stereo:d=12", "-f", "lavfi", "-i", "sine=f=440:d=12", "-map", "0:v",
                        "-map", "1:a", "-map", "2:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", path], check=True)
        rec = recording.probe(path)
        self.assertEqual(rec["sound_tracks"], 2)
        spans = [[1.0, 3.5], [6.0, 9.0]]
        out = countdown.join(rec, spans, countdown.target(rec, spans))
        got = recording.probe(out)
        self.assertEqual((got["width"], got["height"], got["sound_tracks"]), (320, 180, 1))
        self.assertAlmostEqual(got["seconds"], 5.5, delta=0.1)
        self.assertFalse(os.path.exists(out + ".part"))
        self.assertFalse(os.path.exists(out + ".pieces.json"), "no montage pieces beside it for an editor export")
        loud = subprocess.run([FFMPEG, "-hide_banner", "-i", out, "-af", "volumedetect", "-f", "null", "-"],
                              capture_output=True, text=True).stderr
        peak = float(loud.split("max_volume:")[1].split("dB")[0])
        self.assertLess(peak, -60, "the first track, silent, not the tone on the second")
        self.assertNotEqual(countdown.target(rec, spans), countdown.target(rec, spans[::-1]))
        self.assertEqual(countdown.join(rec, spans, out), out, "joined once")


@unittest.skipUnless(test_editor.FFMPEG, "needs ffmpeg")
class MakeCountdown(unittest.TestCase):
    """make_countdown in a scripted job on the 60 s test recording of test_editor (moving from 40 s to 50 s), the game
    recipe loaded, its screen a fake that shows PLAY (or the game's menu) on the recording and nothing on the joined
    one, misheard words left as heard."""
    setUpClass = classmethod(test_editor.Tools.setUpClass.__func__)
    tearDownClass = classmethod(test_editor.Tools.tearDownClass.__func__)

    def setUp(self):
        test_editor.Tools.setUp(self)
        self.shown, self.menus = PLAY, []

        def pictures(rec, times):
            source = os.path.basename(rec["path"]) == os.path.basename(self.clip)
            return [{"at": round(t, 2), "lines": [dict(line) for line in self.shown] if source else []} for t in times]

        def readings(rec, a, b, on_start=None):
            """A reading a second, the game's menu from x to y for each [x, y] of self.menus, play else."""
            source = os.path.basename(rec["path"]) == os.path.basename(self.clip)
            return [{"at": t, "lines": [dict(line) for line in (
                MENU if any(x <= t < y for x, y in self.menus) else PLAY[:1])] if source else []}
                    for t in (round(a + k, 2) for k in range(int(b - a)))]

        self.patches = [mock.patch.object(screen, "_pictures", pictures), mock.patch.object(screen, "readings", readings),
                        mock.patch.object(hearing, "fix", lambda ask, heard, *a, **k: (list(heard), [""] * len(heard)))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        test_editor.Tools.tearDown(self)

    def run_job(self, said, *calls, game=True, recipe="game", moments=None):
        """The person says said; Muse opens the recording, loads the game recipe (when game), looks at 0:06-0:58 (its
        pictures at 12.5, 25.5, 38.5 and 51.5 s), proposes three moments and makes the calls."""
        script = [[call("open_recording", path=self.clip, why="open it")]]
        if game:
            script.append([call("load_recipe", type=recipe, why="the kind")])
        script += [[call("look_at_piece", start="0:06", end="0:58", why="the stream")],
                   [call("propose_moments", why="the best", moments=moments or [
                       {"look": 1, "start": "0:06", "end": "0:21", "reason": "a close call"},
                       {"look": 1, "start": "0:22", "end": "0:37", "reason": "a fall"},
                       {"look": 1, "start": "0:40", "end": "0:55", "reason": "the win"}])]]
        self.chat.jobs += 1
        self.chat.emit("person", job=self.chat.jobs, text=said)
        job = self.chat.job = agent.Job(self.chat, ScriptedMind(*script, *calls, "Done."), said)
        job.run()
        return [e for e in self.chat.events if e.get("job") == job.n and e["kind"] == "step"]

    ORDER = [{"moment": 1, "words": "Close call"}, {"moment": 2, "words": "A long fall"},
             {"moment": 3, "words": "The win"}]

    def counted(self, moments=None, why="the countdown"):
        return [call("make_countdown", moments=moments or self.ORDER, story="the three best moments", why=why)]

    def test_the_moments_are_joined_best_last_each_with_its_rank_and_the_clip_is_made_from_them(self):
        """Moment 1's piece is 6-17 s, its peak picture at 12.5 s too near the start for 7 s before it, and a line said
        from 16 s to 18 s runs past its end, so it ends after the line; moment 2's is 22-33 s around 25.5 s, moment
        3's 44-55 s around 51.5 s, with a line at 46-48 s inside. Both lines move to the joined seconds."""
        self.worker.phrases = [{"from": 10.0, "to": 12.0, "text": "หลบทัน"}, {"from": 40.0, "to": 42.0, "text": "ชนะ"}]
        steps = self.run_job("ทำคลิป Top 3 ช่วงที่มันส์ที่สุด", self.counted())
        made = steps[-1]
        self.assertFalse(made["refused"], made["result"])
        self.assertRegex(made["result"], r"Joined 3 moments into one recording, 34\.[23] s long, counting down to the "
                                         r"best: the three best moments")
        self.assertIn('0 s: "3. Close call", moment 1, 0:06-0:18 of the recording', made["result"])
        self.assertIn('12.25 s: "2. A long fall", moment 2, 0:22-0:33 of the recording', made["result"])
        self.assertIn('23.25 s: "1. The win", moment 3, 0:44-0:55 of the recording', made["result"])
        self.assertIn("The program fitted the pieces: moment 1's piece ends 1.2 s later, so it closes after a line "
                      "rather than inside it.", made["result"])
        self.assertIn("Made the clip from moment 1 (0:00-0:34", made["result"])
        self.assertIn("in the card layout", made["result"])
        joined = self.chat.recording
        self.assertEqual(os.path.dirname(joined["path"]), os.path.join(recording.WORK, "countdown"))
        self.assertEqual((joined["width"], joined["height"]), (320, 180))
        self.assertAlmostEqual(joined["seconds"], 34.25, delta=0.1)
        self.assertEqual([(s["from"], s["to"], s["text"]) for s in self.chat.looks[0]["said"]],
                         [(10.0, 12.0, "หลบทัน"), (25.25, 27.25, "ชนะ")])
        titles = [e["entry"] for e in self.chat.events if e["kind"] == "edit" and e["entry"].get("do") == "title"]
        self.assertEqual([t["text"] for t in titles], ["3. Close call", "2. A long fall", "1. The win"])
        self.assertEqual(titles[0]["start"], 0.0)
        self.assertEqual(self.chat.moments[0]["t"], 30.75, "the clip's picture is the best one's peak (51.5 s)")

    def test_counting_down_again_cuts_the_moments_from_the_recording_they_came_from(self):
        """The person asks to swap two ranks: the countdown open now is not what the moments were proposed on."""
        steps = self.run_job("ทำ top 3", self.counted(), self.counted([self.ORDER[1], self.ORDER[0], self.ORDER[2]],
                                                                        why="swap 3 and 2"))
        again = steps[-1]
        self.assertFalse(again["refused"], again["result"])
        self.assertIn('0 s: "3. A long fall", moment 2, 0:22-0:33 of the recording', again["result"])
        self.assertIn('11 s: "2. Close call", moment 1, 0:06-0:17 of the recording', again["result"])
        self.assertEqual(len({e["recording"]["path"] for e in self.chat.events if e["kind"] == "opened"}), 3,
                         "the recording, then each countdown")

    def test_a_piece_ends_as_the_death_screen_shows_and_one_with_no_play_before_it_is_proposed_again(self):
        """Moment 3's peak picture (51.5 s) is on the death screen that shows from 48 s: its piece is the 11 s before
        it and 1.5 s of it, starting before the moment (40 s). Then moment 1 shows the menu from 1 s on."""
        self.menus = [(48, 53)]
        steps = self.run_job("ทำ top 3", self.counted())
        made = steps[-1]
        self.assertFalse(made["refused"], made["result"])
        self.assertIn('22 s: "1. The win", moment 3, 0:38-0:50 of the recording', made["result"])
        self.assertIn("The program fitted the pieces: moment 3's piece ends 1.5 s after the game's menu shows at 0:48, "
                      "so it shows the play that led there.", made["result"])
        self.assertAlmostEqual(self.chat.recording["seconds"], 33.0, delta=0.1)
        self.chat, self.menus = agent.Chat(), [(1, 21)]
        steps = self.run_job("ทำ top 3", self.counted())
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("Moment 1: the game's menu shows at 0:01 with under 3 s of play before it: propose it from the "
                      "play that led there.", steps[-1]["result"])

    def test_a_match_joins_its_goals_in_match_order_with_no_rank_though_nobody_asked(self):
        """10-31: Spark found all 4 goals of Tarm's eFootball match 3 of 3, and only the first proposed became a clip."""
        self.menus = [(0, 60)]
        goals = [{"moment": 3, "words": "ปิดเกม 1-3", "start": "0:42", "end": "0:55"},
                 {"moment": 1, "words": "นำ 0-1", "start": "0:06", "end": "0:21"},
                 {"moment": 2, "words": "0-2", "start": "0:22", "end": "0:37"}]
        steps = self.run_job("ทำคลิปไฮไลท์", self.counted(goals), recipe="sport")
        made = steps[-1]
        self.assertFalse(made["refused"], made["result"])
        self.assertRegex(made["result"], r"Joined 3 moments into one recording, \S+ s long, in match order: ")
        self.assertIn('0 s: "นำ 0-1", moment 1, 0:06-0:21 of the recording', made["result"])
        titles = [e["entry"]["text"] for e in self.chat.events if e["kind"] == "edit" and e["entry"].get("do") == "title"]
        self.assertEqual(titles, ["นำ 0-1", "0-2", "ปิดเกม 1-3"], "the score only reads in the order it happened")

    def test_a_match_gives_each_piece_its_seconds_and_two_goals_are_enough(self):
        """10-31 005604-4fe9: left to the program, three of four pieces lost the attack and ran into the replay."""
        two = [{"look": 1, "start": "0:06", "end": "0:21", "reason": "first"},
               {"look": 1, "start": "0:33", "end": "0:50", "reason": "second"}]
        bare = [{"moment": 1, "words": "0-1", "start": "0:06", "end": "0:21"}, {"moment": 2, "words": "0-2"}]
        goals = [bare[0], dict(bare[1], start="0:33", end="0:50")]
        steps = self.run_job("ทำคลิปไฮไลท์", self.counted(bare), self.counted(goals), recipe="sport", moments=two)
        self.assertTrue(steps[-2]["refused"])
        self.assertIn("Give each piece its start, where the attack starts", steps[-2]["result"])
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertIn("Joined 2 moments", steps[-1]["result"])

    def test_a_person_who_asked_for_a_top_3_of_a_match_gets_ranks_and_a_game_needs_three(self):
        steps = self.run_job("ทำ top 3 ประตูสวย", self.counted(), recipe="sport")
        self.assertIn('"3. Close call"', steps[-1]["result"])
        self.chat = agent.Chat()
        steps = self.run_job("นับถอยหลังให้หน่อย", self.counted(self.ORDER[:2]))
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("A countdown holds at least 3 moments.", steps[-1]["result"])

    def test_moments_from_a_little_of_a_match_are_not_proposed_until_the_score_was_read_across_it(self):
        """10-31: Glimmer proposed the 3 loud stretches, 1 goal of 4, 3 of 3."""
        self.chat.recording = None
        little = [{"look": 1, "start": "0:32", "end": "0:45", "reason": "the goal"}]
        brain = ScriptedMind([call("open_recording", path=self.clip, why="open it")],
                             [call("load_recipe", type="sport", why="a match")],
                             [call("look_at_piece", start="0:30", end="0:50", why="loud")],
                             [call("propose_moments", why="the goal", moments=little)],
                             [call("propose_moments", why="the goal", moments=little)], "Done.")
        with mock.patch.object(tools, "ACROSS_LEAST", 30.0), mock.patch.object(tools, "ACROSS_SEEN", 0.5):
            self.chat.jobs += 1
            job = self.chat.job = agent.Job(self.chat, brain, "ทำไฮไลท์")
            job.run()
        steps = [e for e in self.chat.events if e.get("job") == job.n and e["kind"] == "step"]
        read, again = steps[3], steps[4]
        self.assertFalse(read["refused"], read["result"])
        self.assertTrue(read["result"].startswith("Not proposed yet: your looks saw little of the match"), read["result"])
        self.assertRegex(read["result"], r"8 pictures across the whole match come next, one every 0:0[78]\.")
        self.assertIn("look at that whole stretch, from the earlier picture on, until you see it", read["result"])
        self.assertIn("The last picture's score says how many goals there are: propose every one.", read["result"])
        self.assertTrue(again["result"].startswith("Proposed 1 moment"), again["result"])

    def test_a_match_seen_widely_a_short_one_and_a_game_are_proposed_at_once(self):
        def told(recipe, seconds=1000.0, seen=100.0, done=False):
            job = mock.Mock()
            job.chat.clip = {"recipe": recipe, "read_across": done}
            job.chat.looks = [{"start": 0.0, "end": seen}]
            with mock.patch.object(recording, "frame", return_value=None):
                return tools._read_across(job, {"seconds": seconds})

        self.assertIsNotNone(told("sport"))
        self.assertIsNone(told("game"))
        self.assertIsNone(told("sport", done=True), "once")
        self.assertIsNone(told("sport", seen=250.0), "a quarter of it seen")
        self.assertIsNone(told("sport", seconds=120.0, seen=10.0), "a clip, not a match")

    def test_a_goal_before_its_moment_stands_when_a_look_saw_it_and_its_words_come_from_that_look(self):
        """10-31 061848-14cb: Muse's moment was 3:30-3:50, past the goal; it looked back and gave 3:12-3:35, and the piece
        was brought to 3:30-3:35, the celebration with no goal."""
        self.worker.phrases = [{"from": 3.0, "to": 5.0, "text": "ยิงเข้าไปแล้ว"}]
        late = [{"look": 1, "start": "0:30", "end": "0:40", "reason": "first"},
                {"look": 1, "start": "0:40", "end": "0:50", "reason": "second"}]
        back = [{"moment": 1, "words": "0-1", "start": "0:08", "end": "0:18"},
                {"moment": 2, "words": "0-2", "start": "0:41", "end": "0:49"}]
        brain = ScriptedMind([call("open_recording", path=self.clip, why="open it")],
                             [call("load_recipe", type="sport", why="a match")],
                             [call("look_at_piece", start="0:30", end="0:50", why="loud")],
                             [call("propose_moments", why="the goals", moments=late)],
                             [call("look_at_piece", start="0:06", end="0:24", why="what led to it")],
                             self.counted(back), "Done.")
        self.chat.jobs += 1
        self.chat.emit("person", job=self.chat.jobs, text="ทำคลิปไฮไลท์")
        job = self.chat.job = agent.Job(self.chat, brain, "ทำคลิปไฮไลท์")
        job.run()
        made = [e for e in self.chat.events if e.get("job") == job.n and e["kind"] == "step"][-1]
        self.assertFalse(made["refused"], made["result"])
        self.assertIn("moment 1, 0:08-0:1", made["result"])
        self.assertEqual([s["text"] for s in self.chat.looks[-1]["said"]], ["ยิงเข้าไปแล้ว"],
                         "the words of the look that saw the piece")

    def test_a_goal_is_kept_to_its_moment_and_one_too_long_is_told_what_works(self):
        """10-31 023845-9147: a start the clock showed as the moment's was refused by a part of a second, then Muse was
        told to leave the seconds out, which a match refuses: 6 tries for 2 goals."""
        wide = [{"look": 1, "start": "0:06", "end": "0:21", "reason": "first"},
                {"look": 1, "start": "0:22", "end": "0:50", "reason": "second"}]
        over = [{"moment": 1, "words": "0-1", "start": "0:05", "end": "0:21"},
                {"moment": 2, "words": "0-2", "start": "0:22", "end": "0:50"}]
        fixed = [over[0], dict(over[1], end="0:40")]
        steps = self.run_job("ทำคลิปไฮไลท์", self.counted(over), self.counted(fixed), recipe="sport", moments=wide)
        said = steps[-2]["result"]
        self.assertIn("is 28 s long, and a piece is 3 to 20 s; give it a start and an end around the goal.", said)
        self.assertNotIn("Moment 1", said, "a start before the moment is brought to it")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])
        self.assertIn("Joined 2 moments", steps[-1]["result"])
        self.assertIn("moment 1, 0:06-", steps[-1]["result"])

    def test_a_match_reels_scores_that_cannot_be_a_match_are_refused_once_and_the_second_answer_stands(self):
        """10-31 094114-7e80: the board read right, then Hull 1-0, Hull 2-0, Arsenal level at 1-2."""
        wide = [{"look": 1, "start": "0:06", "end": "0:21", "reason": "first"},
                {"look": 1, "start": "0:22", "end": "0:37", "reason": "second"}]
        pieces = [{"moment": 1, "words": "Hull 2-0", "start": "0:06", "end": "0:21"},
                  {"moment": 2, "words": "Arsenal 1-2", "start": "0:22", "end": "0:37"}]
        steps = self.run_job("ทำคลิปไฮไลท์", self.counted(pieces), self.counted(pieces), recipe="sport", moments=wide)
        refused, made = steps[-2:]
        self.assertTrue(refused["refused"], refused["result"])
        self.assertIn("The scores in these words read 2-0, then 1-2, and a match's score only goes up", refused["result"])
        self.assertFalse(made["refused"], made["result"])

    def test_scores_are_read_in_order_and_a_time_is_not_a_score(self):
        self.assertIsNone(tools._scores_off(["0-1 ประตูแรก", "0-2 ประตูที่สอง", "1-2 นาที 40:07", "1-3"]))
        self.assertIsNone(tools._scores_off(["1-1 ตีเสมอ", "ท้ายเกม", "2-1"]), "a level word on a level score")
        self.assertIsNone(tools._scores_off(["นาที 32:21", "นาที 40:07"]))
        self.assertIsNone(tools._scores_off(["0-2", "1-2"]), "a goal between two pieces is left out, not wrong")
        self.assertEqual(tools._scores_off(["Hull 1-0", "Hull 2-0", "Arsenal 1-2"]), "1-0, then 2-0, then 1-2")
        self.assertEqual(tools._scores_off(["1-1", "1-1"]), "1-1, then 1-1")
        self.assertEqual(tools._scores_off(["0-2", "0-1"]), "0-2, then 0-1")
        self.assertEqual(tools._scores_off(["Arsenal ตีเสมอ 1-2"]), "1-2", "a level word on a score that is not level")
        self.assertEqual(tools._scores_off(["Level at 2-1"]), "2-1")

    def test_a_countdown_of_a_game_is_not_read_for_scores(self):
        scored = [dict(self.ORDER[0], words="2-0"), dict(self.ORDER[1], words="1-2"), dict(self.ORDER[2], words="1-1")]
        steps = self.run_job("ทำคลิป Top 3 ช่วงที่มันส์ที่สุด", self.counted(scored))
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])

    def test_a_top_three_of_a_match_is_ranked_so_its_scores_may_run_out_of_order(self):
        """Lane 1's review: a person-asked top 3 is best last, not match order."""
        scored = [dict(self.ORDER[0], words="2-1"), dict(self.ORDER[1], words="1-0"), dict(self.ORDER[2], words="1-1 เสมอ")]
        steps = self.run_job("ทำคลิป Top 3 ช่วงที่มันส์ที่สุด", self.counted(scored), recipe="sport")
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])

    def test_one_clip_of_a_match_with_more_moments_is_refused_once_and_the_second_answer_stands(self):
        """10-31: Glimmer proposed 3 goals and made a clip of the first, 2 of 2."""
        one = [call("make_clip", moment=1, layout="card", why="the first goal")]
        steps = self.run_job("ทำคลิปไฮไลท์", one, one, recipe="sport")
        refused, made = steps[-2:]
        self.assertTrue(refused["refused"], refused["result"])
        self.assertIn("A match's goals go in one clip, in match order: join these 3 moments", refused["result"])
        self.assertFalse(made["refused"], made["result"])

    def test_one_clip_of_a_game_is_made_at_once(self):
        steps = self.run_job("ทำคลิป", [call("make_clip", moment=1, layout="card", why="the moment")])
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])

    def test_one_clip_of_a_match_with_one_moment_is_made_at_once(self):
        alone = [{"look": 1, "start": "0:06", "end": "0:21", "reason": "the goal"}]
        steps = self.run_job("ทำคลิป", [call("make_clip", moment=1, layout="card", why="the goal")], recipe="sport",
                             moments=alone)
        self.assertFalse(steps[-1]["refused"], steps[-1]["result"])

    def test_a_clip_from_each_goal_asked_for_and_a_tapped_card_are_left_alone(self):
        job = mock.Mock()
        job.n, job.chat.clip, job.chat.events = 1, {"recipe": "sport"}, []
        self.assertIsNone(tools._one_goal(job, tools.BATCH_WHY, [1, 2]))
        job.n = None
        self.assertIsNone(tools._one_goal(job, "the person tapped its card", [1, 2]))

    def test_a_match_is_offered_the_join_unasked_and_a_game_is_not(self):
        job = agent.Job(self.chat, ScriptedMind("ok"), "ทำไฮไลท์")

        def offered(recipe):
            self.chat.clip = {"recipe": recipe}
            return "make_countdown" in {t["function"]["name"] for t in job.offered()}

        self.assertTrue(offered("sport"))
        self.assertFalse(offered("game"), "offered unasked it moved Glimmer's pick")

    def test_nobody_asking_for_a_countdown_gets_one_clip_of_the_best_moment(self):
        steps = self.run_job(BEAT_ONE, self.counted())
        self.assertTrue(steps[-1]["refused"])
        self.assertIn("Nobody asked for a countdown in their own words", steps[-1]["result"])
        self.assertFalse(os.path.isdir(os.path.join(recording.WORK, "countdown")), "nothing joined")
        self.assertEqual(self.chat.recording["path"], self.clip)

    def test_a_top_n_takes_n_moments_from_a_game_stream_each_inside_its_moment(self):
        steps = self.run_job("make a top 5 of the best moments", self.counted(),
                             self.counted([dict(self.ORDER[0], start="0:01", end="0:10"),
                                           dict(self.ORDER[1], start="0:23", end="0:24")] + self.ORDER[2:]))
        refused, outside = steps[-2:]
        self.assertTrue(refused["refused"])
        self.assertIn("The person asked for a top 5: give 5 moments, the best last.", refused["result"])
        self.assertIn("Moment 1: its piece 0:01-0:10 runs outside the moment (0:06-0:21); leave start and end out",
                      outside["result"])
        self.assertIn("Moment 2: its piece 0:23-0:24 is 1 s long, and a piece is 3 to 20 s; leave start and end out",
                      outside["result"], "said as what is wrong: a piece inside its moment, 26 s long, was told it "
                      "must be inside the moment (Sifu top 5, 10-30)")
        self.chat = agent.Chat()
        steps = self.run_job("ทำ top 3 หน่อย", self.counted(), game=False)
        self.assertIn("A countdown is made from a game stream for now", steps[-1]["result"])

    def test_a_countdown_that_would_open_on_the_games_menu_is_refused_once(self):
        self.shown = MENU
        steps = self.run_job("ทำท็อป 3 ให้หน่อย", self.counted(), self.counted(why="the menu is the joke"))
        refused, made = steps[-2:]
        self.assertTrue(refused["refused"])
        self.assertIn("Nothing was joined. The clip would open on the game's menu: at 0:06 the screen shows DEATH",
                      refused["result"])
        self.assertFalse(made["refused"], made["result"])


if __name__ == "__main__":
    unittest.main()
