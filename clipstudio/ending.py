"""Everything the studio starts ends with it, however the studio ends.

Closing the studio's window ends Python at once, so atexit does not run: a llama-server the studio started stayed on the
graphics card with nothing left to stop it (Tarm 11-03: "ปิด server แล้วดับหมด"). So on Windows the studio joins a job
that ends every process in it when its last handle closes, and only the studio holds that handle: closing the window, a
crash and End task all close it. What the studio starts joins the job by itself, down to the model a llama-server router
starts. A server that was already running, like a router someone else started at the local address, was not started by
the studio, is not in the job, and stays. The studio starts only its own workers, never a browser or a player, so nothing
of the person's goes with it. On a Mac and on Linux atexit stays the way (parts.stop_glimmer)."""
import ctypes
import sys

KILL_ON_JOB_CLOSE = 0x2000
EXTENDED_LIMITS = 9
_held = []


class _Basic(ctypes.Structure):
    _fields_ = [("user_time", ctypes.c_int64), ("job_time", ctypes.c_int64), ("flags", ctypes.c_uint32),
                ("smallest", ctypes.c_size_t), ("largest", ctypes.c_size_t), ("processes", ctypes.c_uint32),
                ("affinity", ctypes.c_size_t), ("priority", ctypes.c_uint32), ("scheduling", ctypes.c_uint32)]


class _Limits(ctypes.Structure):
    _fields_ = [("basic", _Basic), ("io", ctypes.c_uint64 * 6), ("memory", ctypes.c_size_t * 4)]


def tie():
    """Puts this process in the job. True when it is in. Never raises: with no job the studio still runs, as it did,
    and says so."""
    if sys.platform != "win32":
        return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
    kernel.SetInformationJobObject.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    limits = _Limits()
    limits.basic.flags = KILL_ON_JOB_CLOSE
    job = kernel.CreateJobObjectW(None, None)
    if job and kernel.SetInformationJobObject(job, EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)) \
            and kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        _held.append(job)
        return True
    print("What the studio starts may keep running after it closes (Windows error %d)." % ctypes.get_last_error(), flush=True)
    return False
