"""Everything the studio starts ends with it: python -m unittest discover -s tests"""
import os
import subprocess
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from clipstudio import ending  # noqa: E402

SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]
KEEPER = """
import subprocess, sys
sys.path.insert(0, %r)
from clipstudio import ending
print(ending.tie(), flush=True)
child = subprocess.Popen([sys.executable, "-c", %r], stdout=subprocess.PIPE, text=True)
print(child.pid, child.stdout.readline().strip(), flush=True)
child.wait()
"""
PARENT = """
import subprocess, sys, time
below = subprocess.Popen(%r)
print(below.pid, flush=True)
time.sleep(60)
"""


def running(pid):
    got = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"], capture_output=True, text=True)
    return '"%d"' % pid in got.stdout


def gone(pids, seconds=8):
    end = time.time() + seconds
    while time.time() < end and any(running(p) for p in pids):
        time.sleep(0.2)
    return not any(running(p) for p in pids)


@unittest.skipUnless(sys.platform == "win32", "the job is how Windows does it; elsewhere atexit stays the way")
class EndsTogether(unittest.TestCase):
    def setUp(self):
        self.started = []

    def tearDown(self):
        for pid in self.started:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)

    def keeper(self):
        code = KEEPER % (ROOT, PARENT % (SLEEP,))
        keeper = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        self.addCleanup(keeper.stdout.close)
        self.addCleanup(keeper.wait)
        self.addCleanup(keeper.kill)
        said = keeper.stdout.readline().split()
        child, below = [int(x) for x in keeper.stdout.readline().split()]
        self.started += [child, below]
        return keeper, said, child, below

    def test_what_the_studio_started_ends_when_it_is_killed_outright(self):
        """Tarm 11-03: he closed the window and llama-server stayed. A kill runs no atexit, as closing the window does not."""
        keeper, said, child, below = self.keeper()
        self.assertEqual(said, ["True"])
        self.assertTrue(running(child) and running(below))
        keeper.kill()
        self.assertTrue(gone([child, below]), "a child and what it started end with the studio")

    def test_what_the_studio_did_not_start_stays(self):
        other = subprocess.Popen(SLEEP)
        self.started.append(other.pid)
        self.addCleanup(other.wait)
        keeper, _, child, below = self.keeper()
        keeper.kill()
        self.assertTrue(gone([child, below]))
        self.assertTrue(running(other.pid), "like a router that was already running at the local address")

    def test_a_job_that_cannot_be_made_leaves_the_studio_running_and_says_so(self):
        code = ("import sys\nsys.path.insert(0, %r)\nfrom clipstudio import ending\n"
                "ending.EXTENDED_LIMITS = 999\nprint(ending.tie())\n" % ROOT)
        said = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
        self.assertIn("may keep running after it closes", said)
        self.assertTrue(said.strip().endswith("False"), said)


class Wired(unittest.TestCase):
    def test_the_studio_ties_itself_before_it_starts_anything(self):
        from unittest import mock
        from clipstudio import server
        done = []
        with mock.patch.object(server.ending, "tie", lambda: done.append("tie")), \
                mock.patch.object(server.parts, "use_tools", lambda: done.append("tools")), \
                mock.patch.object(server, "Server", side_effect=OSError(10048, "taken")):
            with self.assertRaises(SystemExit):
                server.main()
        self.assertEqual(done[0], "tie")


if __name__ == "__main__":
    unittest.main()
