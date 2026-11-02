"""The cover: one picture for the clip's post, made by Muse Image (Meta's image model, on Meta's API with the person's
own key in META_API_KEY) from the clip's own frame, with the clip's title written on it.

Muse Image is told to keep the people and the products in the frame as they were shot (the same faces, the same
product with its logos and writing) and to add nothing but the title. The frame goes with Meta's image search, web
search and shell turned off, so it is used to draw the cover and for nothing else, in one pass of low reasoning.

Muse Image draws a picture of its own size in the aspect asked for: 1152x2048 for 1080x1920 on 10-29, but
1280x1920 for 1024x1536, with that title from 40 px off one edge to 40 px off the other. A crop to 9:16 would cut
such a title's first and last letters, so the cover is never cropped: the whole picture is fitted into 1080x1920,
on a blurred, darker copy of itself where it does not fill the frame.

A cover is kept in work/covers under a name made from the frame, the words asked and the model, so asking for the
same cover again reuses it at no cost. again=True makes a new one beside it, and from then on that one is the cover.
"""
import base64
import glob
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from . import render

LABEL = "Muse Image"
KEY_ENV = "META_API_KEY"
SIZE = "1080x1920"
WIDTH, HEIGHT = 1080, 1920
USD = 0.01
TIMEOUT = 180
FIT_SECONDS = 60
ASK = ("Make this frame from a short vertical video into the video's cover. Keep everything in the frame as it was "
       "shot: the same people with the same faces, hair, skin, clothes and pose, and the same products with the same "
       "shape, colours, logos, buttons and writing. Do not change any face. Do not add any object, person, animal, "
       "logo, sticker or writing, and do not remove anything; you may make the light brighter and cleaner. Write this "
       "title on the cover in large bold letters that are easy to read on a phone, near the top, clear of every face "
       "and of the product, at least a tenth of the width away from each side: %s")
SAID_IN = " The title is in %s: write it letter for letter as given, with every mark on its letters."
FIT = ("[0:v]split[a][b];[a]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,boxblur=40:2,"
       "eq=brightness=-0.15[ground];[b]scale=%d:%d:force_original_aspect_ratio=decrease[picture];"
       "[ground][picture]overlay=(W-w)/2:(H-h)/2" % ((WIDTH, HEIGHT) * 3))


class CoverError(Exception):
    """One sentence for Muse and the person. usd is what was spent on the way: a picture Meta drew is paid for even
    when it cannot be used."""

    def __init__(self, sentence, usd=0.0):
        Exception.__init__(self, sentence)
        self.usd = usd


def url():
    return os.environ.get("LCS_IMAGE_URL", "https://api.meta.ai/v1").rstrip("/")


def model():
    return os.environ.get("LCS_IMAGE_MODEL", "muse-image-1.0")


def prompt(title, language=None):
    """What Muse Image is asked: the title as given, on one line, and the language it is in when that is known."""
    title = " ".join((title or "").split())
    if not title:
        raise CoverError("A cover needs the clip's title, and this clip has none yet: give the clip a title first.")
    return ASK % title + (SAID_IN % language if language else "")


def _named(frame, asked):
    """The cover's name without its number: the frame's bytes, the words asked and the model, so a new frame or new
    words make a new cover and the same ones find the cover already made."""
    try:
        with open(frame, "rb") as f:
            data = f.read()
    except OSError as e:
        raise CoverError("The clip's frame %s could not be read (%s)." % (frame, e.strerror or e))
    sign = hashlib.sha256(data + json.dumps([model(), SIZE, asked], ensure_ascii=False).encode("utf-8"))
    return data, sign.hexdigest()[:16]


def _made(folder, sign):
    """[(number, path)] of the covers already made for sign, oldest first: sign.jpg is 1, then sign.2.jpg..."""
    numbered = []
    for path in glob.glob(os.path.join(folder, sign + "*.jpg")):
        n = re.fullmatch(re.escape(sign) + r"(?:\.(\d+))?\.jpg", os.path.basename(path))
        if n:
            numbered.append((int(n.group(1) or 1), path))
    return sorted(numbered)


def _said(e):
    """What the API said when it refused, as a short line: its error message when it sent one, else its words."""
    said = e.read().decode("utf-8", "replace")
    try:
        got = json.loads(said).get("error")
        said = got.get("message") or said if isinstance(got, dict) else got or said
    except (ValueError, AttributeError):
        pass
    return " ".join(str(said).split())[:300] or "no reason given"


def _refused(code, said):
    if code in (401, 403):
        return "Meta's API refused your key (HTTP %d: %s). Paste a key that works in Options." % (code, said)
    if code == 429:
        return ("Meta's API takes no more pictures from your key for now (HTTP 429: %s). Wait a minute, then make "
                "the cover again." % said)
    if code == 400:
        return "%s refused the frame and the title (HTTP 400: %s)." % (LABEL, said)
    return "%s could not make the cover (HTTP %d: %s). Make the cover again in a minute." % (LABEL, code, said)


def _ask(data, asked, key):
    """The picture Muse Image drew, as bytes."""
    body = {"model": model(), "prompt": asked,
            "images": [{"image_url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}],
            "n": 1, "size": SIZE, "output_format": "jpeg", "response_format": "b64_json",
            "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False},
            "reasoning_strength": "high"}
    req = urllib.request.Request(url() + "/images/edits", json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            got = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise CoverError(_refused(e.code, _said(e)))
    except (OSError, http.client.HTTPException, ValueError) as e:
        raise CoverError("Meta's API did not answer at %s (%s). Check this computer's internet connection, then make "
                         "the cover again." % (url(), getattr(e, "reason", None) or e))
    try:
        return base64.b64decode(got["data"][0]["b64_json"], validate=True)
    except (KeyError, IndexError, TypeError, ValueError):
        raise CoverError("%s answered without a picture, so there is no cover. Make the cover again." % LABEL)


def _fit(picture, target):
    """Writes picture whole into a WIDTH x HEIGHT JPEG at target, through a part file, so a cover is either whole or
    not there."""
    part = render._part(target)
    drawn = part + ".drawn"
    try:
        with open(drawn, "wb") as f:
            f.write(picture)
        done = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-i", drawn,
                               "-filter_complex", FIT, "-frames:v", "1", "-q:v", "2", "-f", "image2", part],
                              capture_output=True, text=True, timeout=FIT_SECONDS)
        if done.returncode != 0 or not os.path.getsize(part):
            raise CoverError("%s's picture could not be read: %s." % (
                LABEL, (done.stderr.strip().splitlines() or ["FFmpeg gave no reason"])[-1]), USD)
        render.settle(part, target)
    except (subprocess.SubprocessError, OSError) as e:
        raise CoverError("FFmpeg could not fit %s's picture into the cover (%s)." % (LABEL, e), USD)
    finally:
        for left in (drawn, part):
            if os.path.exists(left):
                os.remove(left)


def make(frame, title, language=None, again=False):
    """{path, usd, seconds, reused}: the cover for the clip whose frame (a JPEG's path) and title are given, in
    work/covers. The one already made for this frame and these words is given back at no cost, unless again is
    True. Raises CoverError with one sentence; its usd says what was spent all the same."""
    asked = prompt(title, language)
    data, sign = _named(frame, asked)
    folder = render.folder("covers")
    made = _made(folder, sign)
    if made and not again:
        return {"path": made[-1][1], "usd": 0.0, "seconds": 0.0, "reused": True}
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise CoverError("%s is on Meta's API: paste your Meta API key in Options." % LABEL)
    t0 = time.time()
    picture = _ask(data, asked, key)
    target = os.path.join(folder, sign + (".%d" % (made[-1][0] + 1) if made else "") + ".jpg")
    _fit(picture, target)
    return {"path": target, "usd": USD, "seconds": round(time.time() - t0, 1), "reused": False}
