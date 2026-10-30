"""A picture for what is said, drawn by Muse Image (Meta's image model, on Meta's API with the person's own key in
META_API_KEY) from words alone, and shown over the top of the frame for a few seconds, above the people, its lower
edge fading into the footage. Tarm 10-30, on a one-person talk on TikTok that kept the eye with a new picture on top
each time something was named: "ตอนที่พูดถึงเรื่องผีก็มีภาพผี".

Only the words go to Meta, never a frame of the footage. Muse Image is asked for a photo-like picture with no
writing, no logos and no real person, at 1536x1024: measured 10-30 it drew 1920x1280 in 11 s. The picture is cut to
the band it is shown in (scaled to cover it, its middle kept) and written as a PNG whose last FADE_SHARE of the band
grows see-through, so the footage comes up through its lower edge.

A picture is kept in work/pictures under a name made from the words asked and the model, so the same words again
are drawn once and paid for once.
"""
import base64
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from . import cover, effects, render

LABEL = cover.LABEL
SIZE = "1536x1024"
USD = cover.USD
TIMEOUT = cover.TIMEOUT
FADE_SHARE = 0.35
FIT_SECONDS = 60
ASK = ("A picture for the top of a vertical video, showing: %s. Make it a photo-like picture, clear at phone size, "
       "with one subject. No writing, letters, numbers, captions, logos or watermarks anywhere in it, and no real, "
       "famous or recognisable person.")


class PictureError(Exception):
    """One sentence for Muse and the person. usd is what was spent on the way."""

    def __init__(self, sentence, usd=0.0):
        super().__init__(sentence)
        self.usd = usd


def prompt(about):
    """What Muse Image is asked for the words about, on one line."""
    about = " ".join((about or "").split()).rstrip(".")
    if not about:
        raise PictureError("A picture needs words saying what it shows, such as \"a ghost peeking from a dark "
                           "doorway\".")
    return ASK % about


def make(about):
    """{path, usd, seconds, reused}: Muse Image's picture for the words about, in work/pictures, the one already
    drawn for the same words given back at no cost. PictureError with one sentence."""
    asked = prompt(about)
    sign = hashlib.sha256(json.dumps([cover.model(), SIZE, asked], ensure_ascii=False).encode("utf-8")).hexdigest()
    target = os.path.join(render.folder("pictures"), sign[:16] + ".jpg")
    if os.path.isfile(target):
        return {"path": target, "usd": 0.0, "seconds": 0.0, "reused": True}
    key = os.environ.get(cover.KEY_ENV, "").strip()
    if not key:
        raise PictureError("%s is on Meta's API: set %s to your own Meta API key, or name a file that holds it in "
                           "LCS_KEYS, then start the studio again." % (LABEL, cover.KEY_ENV))
    t0 = time.time()
    data = _ask(asked, key)
    part = render._part(target)
    try:
        with open(part, "wb") as f:
            f.write(data)
        render.settle(part, target)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return {"path": target, "usd": USD, "seconds": round(time.time() - t0, 1), "reused": False}


def _ask(asked, key):
    """The picture Muse Image drew for asked, as bytes."""
    body = {"model": cover.model(), "prompt": asked, "n": 1, "size": SIZE, "output_format": "jpeg",
            "response_format": "b64_json",
            "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False},
            "reasoning_strength": "low"}
    req = urllib.request.Request(cover.url() + "/images/generations", json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            got = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        said = cover._said(e)
        if e.code == 400:
            raise PictureError("%s refused to draw it (HTTP 400: %s). Say what the picture shows in other words."
                               % (LABEL, said))
        raise PictureError(cover._refused(e.code, said).replace("make the cover", "draw the picture"))
    except (OSError, http.client.HTTPException, ValueError) as e:
        raise PictureError("Meta's API did not answer at %s (%s). Check this computer's internet connection, then "
                           "draw the picture again." % (cover.url(), getattr(e, "reason", None) or e))
    try:
        return base64.b64decode(got["data"][0]["b64_json"], validate=True)
    except (KeyError, IndexError, TypeError, ValueError):
        raise PictureError("%s answered without a picture. Draw the picture again." % LABEL, USD)


def fitted(path, w, h):
    """The path of a w x h PNG of the picture at path, scaled to cover w x h with its middle kept, opaque down to
    FADE_SHARE of h from its bottom and see-through from there to its bottom edge, made once in work/pictures."""
    sign = hashlib.sha1(json.dumps([os.path.abspath(path), os.path.getsize(path), w, h, FADE_SHARE]).encode("utf-8"))
    target = os.path.join(render.folder("pictures"), "fit_%s.png" % sign.hexdigest()[:16])
    if os.path.isfile(target):
        return target
    edge = h * (1 - FADE_SHARE)
    graph = ("[0:v]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,format=rgba,"
             "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='255*min(1,max(0,(%d-Y)/%s))'"
             % (w, h, w, h, h, effects._f(h - edge)))
    part = render._part(target)
    try:
        done = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-i", path,
                               "-filter_complex", graph, "-frames:v", "1", "-f", "image2", "-c:v", "png", part],
                              capture_output=True, text=True, errors="replace", timeout=FIT_SECONDS)
        if done.returncode != 0 or not os.path.getsize(part):
            raise PictureError("%s's picture could not be read: %s." % (
                LABEL, (done.stderr.strip().splitlines() or ["FFmpeg gave no reason"])[-1]))
        render.settle(part, target)
    except (subprocess.SubprocessError, OSError) as e:
        raise PictureError("FFmpeg could not cut %s's picture to its place (%s)." % (LABEL, e))
    finally:
        if os.path.exists(part):
            os.remove(part)
    return target
