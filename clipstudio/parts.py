"""The parts the studio needs on this computer, and one button that puts the missing ones in place.

Four parts: FFmpeg (reads, cuts and renders the video), llama.cpp's llama-server (runs Muse Glimmer on the graphics
card), Muse Glimmer itself, and the hearing model. Each is either found (on the PATH, in a setting, or in work/), or
can be downloaded, or says in a sentence why it cannot.

The guarantee: every file comes from one address pinned here and is used only after its size and SHA-256 match the
ones pinned here. A download cut off part way carries on from where it stopped, a file that does not match is deleted
and never used, and a part counts as set up only once all its files are in place whole: each is written to a .part
file, checked, then moved to its name, and a zip is unpacked beside its folder and moved in one step. A file handed
over with the studio, in the parts folder beside it (LCS_PARTS), is used in place of its download and checked the same
way: the hearing model's address is in a private repository until the release, so a tester's copy carries it (10-30).

When Glimmer's file and llama-server are both on this computer, the local mind is at its default address, and
nothing answers there, the studio starts llama-server the way the README shows (router mode, from a preset, with
--no-models-autoload) and stops it when the studio stops.

The graphics card decides which llama.cpp build is downloaded: llama.cpp b11716's CUDA 13.4 build for a card whose
driver offers CUDA 13 (an RTX 50 card needs it), else its CUDA 12.4 build. Glimmer needs an NVIDIA card with 16 GB;
without one, setup leaves Glimmer and llama.cpp out and says so, and Muse Spark is the way in.
"""
import atexit
import hashlib
import http.client
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from . import ears, hearing, mind, recording, sends

WORK = recording.WORK
TOOLS = os.path.join(WORK, "tools")
FFMPEG_DIR = os.path.join(TOOLS, "ffmpeg")
LLAMA_DIR = os.path.join(TOOLS, "llama")
GLIMMER_NAME = "Muse-Glimmer-30B-UD-Q3_K_XL.gguf"
VISION_NAME = "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf"
GLIMMER_FILE = os.environ.get("LCS_GLIMMER_FILE") or os.path.join(WORK, "models", "glimmer", GLIMMER_NAME)
VISION_FILE = os.environ.get("LCS_GLIMMER_VISION") or os.path.join(WORK, "models", "glimmer", VISION_NAME)
HEARING_DIR = os.environ.get("LCS_HEARING_MODELS") or os.path.join(WORK, "models", "hearing")
HANDED = os.environ.get("LCS_PARTS") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parts")
ONE_KEY = os.path.join(HANDED, "one_key.txt")
HEARING_FROM = os.environ.get("LCS_HEARING_FROM",
                              "https://github.com/ashirandev/local-clip-studio/releases/download/hearing-1").rstrip("/")
LOCAL_URL = "http://127.0.0.1:8863/v1"
GLIMMER_CARD_MIB = 15000
SPARE = 1 << 30
LATER = ("engine", "glimmer")
CHUNK = 1 << 20
TIMEOUT = 60
AGENT = "local-clip-studio-setup"
PASTE = {"win32": " with one right-click", "darwin": " with Cmd+V"}.get(sys.platform, "")
LLAMA = "https://github.com/ggml-org/llama.cpp/releases/download/b11716/"
FFMPEG = "https://github.com/GyanD/codexffmpeg/releases/download/8.1.1/"
GLIMMER = "https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/resolve/faa5b025c584459c13febfa5c59883516710ae39/"
META = "https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/resolve/b44f1072cf75e918a1b82a3a17b45c4389b076e7/"

FILES = {
    "ffmpeg": [(FFMPEG + "ffmpeg-8.1.1-essentials_build.zip", 109282242,
                "6f58ce889f59c311410f7d2b18895b33c03456463486f3b1ebc93d97a0f54541")],
    "cuda13": [(LLAMA + "llama-b11716-bin-win-cuda-13.4-x64.zip", 155071236,
                "a5e5f50494bbfc96286f16edff766302f30f9b9f9bf14f66b8097b6e78c5f853"),
               (LLAMA + "cudart-llama-bin-win-cuda-13.4-x64.zip", 423535356,
                "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668")],
    "cuda12": [(LLAMA + "llama-b11716-bin-win-cuda-12.4-x64.zip", 267472762,
                "f00e612c7071158006701cd28faf8dd47a55170432fb3e634eee4e0d285883af"),
               (LLAMA + "cudart-llama-bin-win-cuda-12.4-x64.zip", 391443627,
                "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6")],
    "glimmer": [(GLIMMER + GLIMMER_NAME, 13360983072,
                 "820d18e00c59376331eba62dd716a6bcdcaa305c2316f5da3a2fab36c7c37e15"),
                (META + VISION_NAME, 1400328928,
                 "f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6")],
    "hearing": [("model.int8.onnx", 1031964630, "543f58a53d4f319376a5374fb7a7c5a2f246f4938370961a6ba7f207a2bb02fc"),
                ("letters.json", 70913, "88a785591bcd0b54b7fe1a82ecbe9c4f1b55b5600ff8e4440e3d4515656144ce")],
}
CUDA = re.compile(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)")
_lock = threading.Lock()
_job = {"running": False, "part": None, "step": None, "done": 0, "total": 0, "error": None}
_engine = {"process": None, "why": None}
_card = []


class PartError(Exception):
    """A part could not be set up. The sentence says why and what to do."""


def windows():
    return sys.platform == "win32"


def card():
    """The NVIDIA card as {name, mib, cc, cuda}, cc and cuda as (major, minor), or None when there is none (or
    nvidia-smi cannot say). Asked once: a new driver needs the studio started again anyway."""
    if _card:
        return _card[0]
    _card.append(_ask_card())
    return _card[0]


def _ask_card():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        row = _run([smi, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"])
        head = CUDA.search(_run([smi]))
        name, mib, cc = [x.strip() for x in row.splitlines()[0].split(",")]
        return {"name": name, "mib": int(float(mib)), "cc": tuple(int(x) for x in cc.split(".")),
                "cuda": (int(head.group(1)), int(head.group(2))) if head else None}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _run(cmd):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True, creationflags=flags).stdout


def build(gpu):
    """Which llama.cpp build this card takes ("cuda13" or "cuda12"), or None with the reason, as (build, why)."""
    if not gpu:
        return None, "This computer has no NVIDIA graphics card that the studio can see."
    if gpu["mib"] < GLIMMER_CARD_MIB:
        return None, ("Muse Glimmer needs a graphics card with 16 GB; the %s has %.0f GB."
                      % (gpu["name"], gpu["mib"] / 1024))
    cuda = gpu["cuda"] or (0, 0)
    if cuda >= (13, 4) or (cuda >= (13, 0) and gpu["cc"] >= (12, 0)):
        return "cuda13", None
    if cuda >= (12, 4) and gpu["cc"] <= (9, 0):
        return "cuda12", None
    return None, "The graphics driver is too old for llama.cpp. Update the NVIDIA driver, then open Set up again."


def llama_server():
    """The llama-server program: LCS_LLAMA_SERVER, the one setup put in work/tools/llama, or one on the PATH."""
    own = os.environ.get("LCS_LLAMA_SERVER")
    if own:
        return own if os.path.isfile(own) else None
    here = os.path.join(LLAMA_DIR, "llama-server.exe" if windows() else "llama-server")
    return here if os.path.isfile(here) else shutil.which("llama-server")


def use_tools():
    """Puts the FFmpeg setup downloaded first on the PATH, so every shutil.which("ffmpeg") finds it."""
    bin_dir = os.path.join(FFMPEG_DIR, "bin")
    if os.path.isfile(os.path.join(bin_dir, "ffmpeg.exe" if windows() else "ffmpeg")):
        path = os.environ.get("PATH", "")
        if bin_dir not in path.split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + path


def _glimmer_here():
    """Glimmer's model and its vision part (the studio shows Muse pictures) are here, or the model the person named in
    LCS_GLIMMER_FILE is."""
    if os.environ.get("LCS_GLIMMER_FILE"):
        return os.path.isfile(GLIMMER_FILE)
    return all(os.path.isfile(path) and os.path.getsize(path) == size
               for path, (_, size, _) in zip((GLIMMER_FILE, VISION_FILE), FILES["glimmer"]))


def _hearing_here():
    return bool(ears.model_file(HEARING_DIR)) and os.path.isfile(os.path.join(HEARING_DIR, "letters.json"))


def own_mind():
    """True when the person pointed the studio at a Muse server of their own (LCS_MIND_URL): setup leaves it be."""
    return os.environ.get("LCS_MIND_URL", LOCAL_URL).rstrip("/") != LOCAL_URL


def parts(gpu=None, look=True):
    """Each part as {key, label, what, ok, size, why}: ok when it is on this computer, size the bytes setup would
    download for it (0 when it cannot), why the sentence when it is neither."""
    gpu = card() if look else gpu
    kind, no_card = build(gpu)
    served = own_mind() or mind.listens(LOCAL_URL)
    out = []

    def add(key, label, what, ok, files=(), why=None):
        size = 0 if ok or why else sum(s for _, s, _ in files)
        out.append({"key": key, "label": label, "what": what, "ok": bool(ok), "size": size,
                    "why": None if ok else why})

    add("ffmpeg", "FFmpeg", "Reads, cuts and renders the video", shutil.which("ffmpeg") and shutil.which("ffprobe"),
        FILES["ffmpeg"], None if windows() else "Install FFmpeg (with libass) from your system's packages.")
    add("engine", "llama.cpp", "Runs Muse Glimmer on the graphics card", served or llama_server(),
        FILES.get(kind, ()), no_card or (None if windows() else "Install llama.cpp's llama-server, build b11716."))
    add("glimmer", "Muse Glimmer", "Meta's open model, on this computer", served or _glimmer_here(),
        FILES["glimmer"], no_card)
    add("hearing", "Hearing", "Writes down what is said", _hearing_here() or mind.listens(hearing.URL),
        FILES["hearing"], None if HEARING_FROM else "The hearing model has no download address here yet; put its "
        "files in %s." % HEARING_DIR)
    return out


def state():
    """What /api/parts answers: the parts, the card, the setup running now, and the free space where they go."""
    gpu = card()
    found = parts(gpu, look=False)
    os.makedirs(WORK, exist_ok=True)
    with _lock:
        job = dict(_job)
    p = _engine["process"]
    if p and p.poll() is not None:
        _engine["why"] = ("llama-server stopped (exit code %s); what it wrote is in %s."
                          % (p.returncode, os.path.join(TOOLS, "llama-server.log")))
    return {"parts": found, "card": gpu and {"name": gpu["name"], "gb": round(gpu["mib"] / 1024)},
            "setup": job, "free": shutil.disk_usage(WORK).free, "needed": sum(p["size"] for p in found),
            "engine": _engine["why"]}


def set_up(leave=()):
    """Starts downloading every missing part that can be downloaded, but the ones in leave, in the background. Returns
    None, or a sentence when it cannot start."""
    gpu = card()
    todo = sorted((p for p in parts(gpu, look=False) if not p["ok"] and p["size"] and p["key"] not in leave),
                  key=lambda p: p["size"])
    if not todo:
        return "Everything that can be set up here is set up."
    os.makedirs(WORK, exist_ok=True)
    needed, free = sum(p["size"] for p in todo), shutil.disk_usage(WORK).free
    if free < needed + SPARE:
        return ("Setting up needs %.1f GB free where the studio keeps its files (%s); there is %.1f GB."
                % ((needed + SPARE) / 1e9, WORK, free / 1e9))
    with _lock:
        if _job["running"]:
            return "Set up is already running."
        _job.update(running=True, part=None, step=None, done=0, total=needed, error=None)
    threading.Thread(target=_set_up, args=(todo, build(gpu)[0]), name="setup", daemon=True).start()
    return None


def _set_up(todo, kind):
    """Sets up each part in turn. A part that fails does not stop the others: the hearing model missing from its
    address must not leave the person without Glimmer."""
    why = []
    try:
        for p in todo:
            with _lock:
                _job["part"] = p["key"]
            try:
                _set_up_part(p["key"], kind)
            except PartError as e:
                why.append(str(e))
            except OSError as e:
                why.append("Set up could not write its files in %s (%s)." % (WORK, e))
        start_glimmer()
    finally:
        with _lock:
            _job.update(running=False, part=None, step=None, error=" ".join(why) or None)


def _set_up_part(key, kind):
    if key == "ffmpeg":
        _unzip_all(FILES["ffmpeg"], FFMPEG_DIR, strip=True)
        use_tools()
    elif key == "engine":
        _unzip_all(FILES[kind], LLAMA_DIR)
    elif key == "glimmer":
        for (url, size, sha), dest in zip(FILES["glimmer"], (GLIMMER_FILE, VISION_FILE)):
            fetch(url, size, sha, dest)
    elif key == "hearing":
        for name, size, sha in FILES["hearing"]:
            fetch(HEARING_FROM + "/" + name, size, sha, os.path.join(HEARING_DIR, name))
        hearing.start()


def _step(name):
    with _lock:
        _job["step"] = name


def _moved(n):
    with _lock:
        _job["done"] += n


def fetch(url, size, sha256, dest):
    """Downloads url to dest: carries on from dest.part when an earlier try was cut off, then checks the size and
    SHA-256 before the file gets its name. Raises PartError, and deletes the download, when it does not match."""
    if os.path.isfile(dest) and os.path.getsize(dest) == size:
        _moved(size)
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    handed = os.path.join(HANDED, os.path.basename(dest))
    if os.path.isfile(handed) and os.path.getsize(handed) == size:
        _step("copy")
        try:
            os.replace(handed, part)
        except OSError:
            shutil.copyfile(handed, part)
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if have > size:
        os.remove(part)
        have = 0
    _moved(have)
    _step("download")
    if have < size:
        headers = {"User-Agent": AGENT}
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT) as r:
                if have and r.status != 206:
                    _moved(-have)
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        piece = r.read(CHUNK)
                        if not piece:
                            break
                        f.write(piece)
                        _moved(len(piece))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise PartError("%s is not at %s yet, so it was left out." % (os.path.basename(dest), url))
            raise PartError(_stopped(dest, e.reason))
        except (OSError, http.client.HTTPException) as e:
            raise PartError(_stopped(dest, getattr(e, "reason", None) or e))
        if os.path.getsize(part) < size:
            raise PartError(_stopped(dest, "the connection closed early"))
    _step("check")
    if os.path.getsize(part) != size or _sha256(part) != sha256:
        os.remove(part)
        raise PartError("%s did not match the file the studio expects, so it was deleted. Press Set up to download "
                        "it again." % os.path.basename(dest))
    os.replace(part, dest)
    return dest


def _stopped(dest, why):
    return ("The download of %s stopped (%s). Press Set up again: it carries on from where it stopped."
            % (os.path.basename(dest), why))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1 << 24), b""):
            h.update(piece)
    return h.hexdigest()


def _unzip_all(files, folder, strip=False):
    """Downloads each zip into work/tools/downloads, unpacks them all beside folder, then moves them into folder in
    one step. strip drops the single top folder a zip keeps everything in."""
    zips = [fetch(url, size, sha, os.path.join(TOOLS, "downloads", url.rsplit("/", 1)[1])) for url, size, sha in files]
    _step("unpack")
    new = folder + ".new"
    shutil.rmtree(new, ignore_errors=True)
    for path in zips:
        with zipfile.ZipFile(path) as z:
            z.extractall(new)
    if strip:
        inside = os.listdir(new)
        if len(inside) == 1 and os.path.isdir(os.path.join(new, inside[0])):
            os.replace(os.path.join(new, inside[0]), new + ".top")
            os.rmdir(new)
            os.replace(new + ".top", new)
    shutil.rmtree(folder, ignore_errors=True)
    os.replace(new, folder)
    for path in zips:
        os.remove(path)


def preset():
    """The preset file the README shows, for the Glimmer setup downloaded: its vision part too when it is here."""
    vision = "mmproj = %s\nmmproj-offload = true\n" % VISION_FILE if os.path.isfile(VISION_FILE) else ""
    return ("version = 1\n\n[muse-glimmer]\nmodel = %s\n%sc = 49152\nparallel = 1\njinja = true\nfit = on\n"
            "load-on-startup = true\n" % (GLIMMER_FILE, vision))


def glimmer_command(program, ini):
    u = urllib.parse.urlsplit(LOCAL_URL)
    return [program, "--models-preset", ini, "--models-max", "1", "--no-models-autoload", "--host", u.hostname,
            "--port", str(u.port)]


def start_glimmer():
    """Starts llama-server with Glimmer when both are here and nothing answers at the local address. Never raises:
    state() carries the sentence when it could not start."""
    program = llama_server()
    if own_mind() or mind.listens(LOCAL_URL) or not program or not _glimmer_here():
        return False
    os.makedirs(TOOLS, exist_ok=True)
    ini = os.path.join(TOOLS, "glimmer.ini")
    with open(ini, "w", encoding="utf-8") as f:
        f.write(preset())
    try:
        with open(os.path.join(TOOLS, "llama-server.log"), "ab") as log:
            _engine["process"] = subprocess.Popen(glimmer_command(program, ini), stdout=log, stderr=subprocess.STDOUT,
                                                  stdin=subprocess.DEVNULL, cwd=os.path.dirname(program),
                                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _engine["why"] = None
    except OSError as e:
        _engine["why"] = "llama-server could not start (%s)." % e
        return False
    print("Muse Glimmer: started llama-server, loading %s" % GLIMMER_FILE, flush=True)
    return True


@atexit.register
def stop_glimmer():
    """Stops the llama-server the studio started, and only that one."""
    p = _engine["process"]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()


def _size(n):
    return "%d MB" % round(n / 1e6) if n < 1e9 else "%.1f GB" % (n / 1e9)


def ask_sends(read):
    """After a Meta key is saved: may Meta's models hear and see pieces of the person's videos (sends.py)? Enter is yes,
    any other answer is no, and no answer at all leaves the switch off. Not asked when it is on already."""
    if sends.allowed():
        return
    print("  Meta's models can hear and see small pieces of your videos, for better clips. That sends Meta, with your "
          "key: what is said, in short pieces of sound; pieces of up to a minute with their sound, for Muse Wonder to "
          "watch; and frames, for Muse Image to draw a cover from. With no, the hearing stays on this computer.")
    try:
        answer = read("  Let Meta's models hear and see pieces? Options > API changes it at any time. "
                      "(Enter for yes, n for no): ").strip().lower()
    except EOFError:
        print()
        return
    sends.allow(answer in ("", "y", "yes"))
    print("  Meta sees pieces is %s." % ("on" if sends.allowed() else "off"))


def ask_api(read=input, read_secret=None):
    """Asks for the API endpoint and key in the console, and then for a second one: the hearing and the watching use a
    Meta key while the mind may be Muse Spark Open elsewhere, and someone new should not have to find Options for it.
    True when a key was saved."""
    import getpass
    read_secret = read_secret or getpass.getpass
    if not _ask_one(read, read_secret, "  Endpoint (Enter to skip): "):
        return False
    print("  A second key, if you have one: Meta's for the hearing and the watching, with Muse Spark Open's for the "
          "chat, for example.")
    _ask_one(read, read_secret, "  Another endpoint (Enter to finish): ")
    return True


def ask_key(read=input, read_secret=None):
    """The one question of a build made for one person on Meta's API (ONE_KEY, which whoever made the build puts in the
    parts folder): their Meta key, tried at once and asked for again when Meta turns it down, then ask_sends. A key
    already saved stays with Enter, as setup is run again after anything stops, and is tried too, as a refused paste
    is saved until another replaces it. True when there is a key."""
    import getpass
    read_secret = read_secret or getpass.getpass
    mind.load_keys()
    saved = mind.saved_api()[1]
    if saved:
        print("  A key is saved. Press Enter to keep it, or paste another.")
    while True:
        try:
            key = read_secret("  Paste your Meta API key%s, then press Enter (it does not show): " % PASTE)
        except EOFError:
            print()
            return False
        why = (mind.key_refused() if saved and not key.strip() else
               mind.save_api(mind.NAMED["meta"], key) or mind.key_refused())
        if not why:
            print("  The key works.")
            break
        saved = False
        print("  " + why)
    ask_sends(read)
    return True


def _ask_one(read, read_secret, question):
    """One endpoint and key (Enter skips, or keeps the one saved); a word names Meta's, Baseten's or Fireworks' endpoint
    (mind.NAMED), as nobody can guess an address. The key is read without showing it. A Meta key is followed by
    ask_sends: that key is the one the hearing and the watching use."""
    url, saved = mind.saved_api()
    print("Muse through an API (Meta's Muse Spark, or Muse Spark Open on Modal, Baseten or Fireworks): its endpoint "
          "and your key.")
    print("  Type meta (Meta's API, %s), baseten or fireworks, or paste your Modal endpoint's address."
          % mind.NAMED["meta"])
    if saved:
        print("  Saved: %s. Press Enter to keep it." % url)
    while True:
        try:
            url = read(question).strip()
            url = mind.NAMED.get(url.lower(), url)
            key = read_secret("  API key (it does not show while you type): ") if url else ""
        except EOFError:
            print()
            return False
        if not url:
            return False
        why = mind.save_api(url, key)
        if not why and "META_API_KEY" in mind.api_names(url)[1]:
            why = mind.key_refused()
        if not why:
            print("  Saved in %s." % mind.KEY_FILE)
            if "META_API_KEY" in mind.api_names(url)[1]:
                ask_sends(read)
            return True
        print("  " + why)


def main(argv=None):
    """python -m clipstudio.parts, what setup.bat runs: asks for the API endpoint and key first (so the person can walk
    away), then downloads every part this computer is missing, with a line of progress. With a key saved, Muse
    answers through its API from the first minute, so Glimmer and llama.cpp wait for the studio's Set up button
    rather than keep the person from starting. Exit 1 when a download stopped; running it again carries on from where
    it stopped. --no-key skips the question."""
    argv = sys.argv[1:] if argv is None else argv
    use_tools()
    if "--no-key" not in argv and sys.stdin and sys.stdin.isatty():
        (ask_key if os.path.isfile(ONE_KEY) else ask_api)()
    gpu = card()
    found = parts(gpu, look=False)
    leave = LATER if mind.saved_api()[1] else ()
    print("Graphics card: %s" % ("%s, %d GB" % (gpu["name"], round(gpu["mib"] / 1024)) if gpu else "none found"))
    for p in found:
        print("  %-13s %s" % (p["label"], "ready" if p["ok"] else "left out: " + (p["why"] or "") if not p["size"]
                              else _size(p["size"]) + (" later, from Set up in the studio" if p["key"] in leave
                                                       else " to download")))
    if any(p["key"] in leave and p["size"] for p in found):
        print("  Muse answers through the API you saved until then; Set up puts it on this computer, free to use.")
    why = set_up(leave)
    if why:
        print(why)
        return 0 if why.startswith("Everything") else 1
    while True:
        with _lock:
            job = dict(_job)
        if not job["running"]:
            break
        label = next((p["label"] for p in parts(gpu, look=False) if p["key"] == job["part"]), "")
        doing = {"check": "Checking", "unpack": "Unpacking", "copy": "Copying"}.get(job["step"], "Downloading")
        sys.stdout.write("\r  %s %s - %s of %s     " % (doing, label, _size(job["done"]), _size(job["total"])))
        sys.stdout.flush()
        threading.Event().wait(1)
    print()
    if job["error"]:
        print(job["error"])
        return 1
    print("Everything that can be set up here is set up.")
    return 0


def leave(code):
    """Ends setup at once. The hearing worker setup starts prints from a daemon thread, and Python aborts when one
    prints while it closes, which reads as a failed setup when everything was set up. os._exit skips atexit, so the
    llama-server setup started is stopped here, or it would hold the card with no window."""
    stop_glimmer()
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    leave(main())
