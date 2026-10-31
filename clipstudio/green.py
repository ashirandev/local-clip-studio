"""A person in front of a green cloth, put in front of a background Muse Image draws for the clip. Tarm 10-30, on
recording a seller clip: "ผมจะอัดด้วยผ้าเขียว แล้วเราค่อยไปตัดผ้าเขียวออก แล้วใส่พื้นหลังให้เหมาะสมด้วยโมเดลของเมต้า".

The cloth is told from the person by how much greener than red and blue each pixel is: green minus the larger of the
two. The cloth scores high, in its creases too; skin, a white product, grey and black score near 0 or below. Measured
10-30 on Tarm's own green-screen seller: a distance from the cloth's colour (FFmpeg's chromakey) put white and grey at
0.14 from a pale cloth, inside its creases' spread at 0.1, so a brightly lit face came out see-through.

find() reads a few small frames of the moment and measures the cloth along the frame's edge: clear, its scores' low
end (its darkest crease), is where a pixel is all cloth, and SOLID of that is where it is all person, the soft edge
between. mask() writes the person's mask as person.cut_out gives SAM's, so the background, the words behind the
person and the cut-out looks all take it: grey frames on the recording's pixel grid, each from that score, kept to
the biggest shapes that are not cloth (the person with what they hold), so a wall or a lamp the cloth misses at the
edge stays out.

The background is drawn from words alone (picture.make with ASK, at the frame's size): no frame of the footage goes
to Meta. It is asked for a calm top, where the title sits, and a soft focus, as a place looks behind a person filmed
up close.
"""
import hashlib
import json
import os
import shutil
import subprocess

import cv2
import numpy as np

from . import picture, recording, render

# "Keep its top third calm and plain" drew a flat panel with a hard line across the picture (222 between two rows where
# a room's edge gives about 20): the seller's pad 10-31, 1 of 6 drawings, and none of 6 with this ask.
ASK = ("The background behind a person in a short vertical video, seen from where the camera stands: %s. An empty "
       "place with nobody in it, photo-like, softly out of focus as a place is behind a person filmed up close, in "
       "soft even light. One continuous photograph from top to bottom: the room goes on upward with its own wall or "
       "ceiling, with no band, panel, border or line across it, and little detail in its upper part. No people, hands, "
       "writing, letters, numbers, logos or watermarks anywhere in it.")
SIZE = "1080x1920"
WIDTH, HEIGHT = 1080, 1920
SAMPLES = 6
SMALL = 320
EDGE = 0.04
CLOTH = 20
CLEAR_SHARE = 0.02
LEAST_EDGE = 0.5
LEAST_CLEAR = 24
SOLID = 0.4
KEEP_PART = 0.2
GROW = 0.04
FPS = 30
MASKS = 2
MASK_SECONDS = 600
READ_SECONDS = 30
FIT_SECONDS = 60


class GreenError(Exception):
    """One sentence for Muse and the person."""


def _frames(path, start, end, size, samples):
    """samples small RGB frames of path between recording seconds start and end, SMALL px across the longer side."""
    tool = shutil.which("ffmpeg")
    if not tool:
        raise GreenError("FFmpeg is not installed, so the green cloth cannot be looked for.")
    scale = min(1.0, SMALL / max(size))
    w, h = max(2, int(size[0] * scale) // 2 * 2), max(2, int(size[1] * scale) // 2 * 2)
    step = (end - start) / samples
    out = []
    for k in range(samples):
        try:
            data = subprocess.run([tool, "-v", "error", "-ss", "%.2f" % (start + (k + 0.5) * step), "-i", path,
                                   "-map", "0:v:0", "-frames:v", "1", "-vf", "scale=%d:%d" % (w, h), "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "pipe:1"], capture_output=True,
                                  timeout=READ_SECONDS).stdout
        except (subprocess.SubprocessError, OSError):
            continue
        if len(data) >= w * h * 3:
            out.append(np.frombuffer(data[:w * h * 3], np.uint8).reshape(h, w, 3))
    if not out:
        raise GreenError("FFmpeg could not take pictures from the recording to look for the green cloth.")
    return out


def greenness(rgb):
    """How much greener than red and blue each pixel of an RGB frame is: green minus the larger of the two, smoothed
    over 3x3 pixels as a crease's grain is."""
    rgb = rgb.astype(np.float32)
    return cv2.blur(rgb[:, :, 1] - np.maximum(rgb[:, :, 0], rgb[:, :, 2]), (3, 3))


def alpha(score, found):
    """The person's mask, 0 cloth to 255 person, from greenness scores and the cloth found."""
    clear, solid = found["clear"], found["solid"]
    return np.clip((clear - score) * (255.0 / (clear - solid)), 0, 255)


def _edge(h, w):
    """A mask of the frame's edge, EDGE of each side deep."""
    ring = np.zeros((h, w), bool)
    dy, dx = max(1, round(h * EDGE)), max(1, round(w * EDGE))
    ring[:dy], ring[-dy:], ring[:, :dx], ring[:, -dx:] = True, True, True, True
    return ring


def find(path, start, end, size, samples=SAMPLES):
    """{"clear", "solid", "edge"}: the green cloth behind the person in path between recording seconds start and end
    (a recording of size (w, h)), or None when no green cloth is there. A pixel as green as clear is all cloth, one
    as green as solid or less all person; edge is the share of the frame's edge the cloth covers. "scores" holds the
    frames' greenness for _matte. GreenError with one sentence when a cloth is there but too dark or grey to tell
    from the person."""
    frames = _frames(path, start, end, size, samples)
    h, w = frames[0].shape[:2]
    ring = _edge(h, w)
    scores = [greenness(f) for f in frames]
    edge = np.concatenate([s[ring] for s in scores])
    cloth = edge > CLOTH
    share = float(cloth.mean())
    if share < LEAST_EDGE:
        return None
    clear = float(np.quantile(edge[cloth], CLEAR_SHARE))
    if clear < LEAST_CLEAR:
        raise GreenError("The green cloth is too dark or too grey to tell from the person. Light the cloth evenly "
                         "and brightly, then record again.")
    return {"clear": round(clear, 1), "solid": round(clear * SOLID, 1), "edge": round(share, 2), "scores": scores}


def _matte(found, size):
    """Where the person may be, 0 to 1 on the recording's pixel grid: in each frame the biggest shape that is not
    cloth (the person with what they hold) and any other at least KEEP_PART of it, all frames together, grown by
    GROW of the frame."""
    scores = found["scores"]
    h, w = scores[0].shape
    kept = np.zeros((h, w), np.uint8)
    for score in scores:
        solid = (alpha(score, found) >= 128).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        if count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            for n in np.nonzero(areas >= KEEP_PART * areas.max())[0]:
                kept[labels == n + 1] = 255
    grow = max(1, round(GROW * max(w, h)))
    kept = cv2.dilate(kept, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return cv2.resize(kept, tuple(size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def mask(rec, start, end):
    """state['mask'] for the person in front of a green cloth from start to end (recording seconds), as
    person.cut_out gives SAM's: {"path", "person": None, "fps", "size", "from", "to"} and "green", the cloth found.
    The file is grey frames on the recording's pixel grid, white the person, FPS a second, made once in work/masks.
    None when no green cloth is behind the person; GreenError with one sentence when one is but it cannot be told
    from the person, or FFmpeg fails."""
    size = (rec["width"], rec["height"])
    found = find(rec["path"], start, end, size)
    if not found:
        return None
    green = {k: v for k, v in found.items() if k != "scores"}
    seen = os.stat(rec["path"])
    name = hashlib.sha1(json.dumps([MASKS, rec["path"], seen.st_size, seen.st_mtime, start, end, FPS, green])
                        .encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(recording.WORK, "masks")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "green_%s.mkv" % name)
    made = {"path": path, "person": None, "fps": FPS, "size": list(size), "from": start, "to": end, "green": green}
    if os.path.isfile(path):
        return made
    matte = _matte(found, size)
    tool = shutil.which("ffmpeg") or "ffmpeg"
    part = render._part(path)
    w, h = size
    reader = subprocess.Popen([tool, "-v", "error", "-ss", "%.3f" % start, "-t", "%.3f" % (end - start), "-i",
                               rec["path"], "-map", "0:v:0", "-vf", "fps=%d" % FPS, "-f", "rawvideo", "-pix_fmt",
                               "rgb24", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    writer = subprocess.Popen([tool, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "%dx%d" % size,
                               "-r", str(FPS), "-i", "pipe:0", "-c:v", "ffv1", "-pix_fmt", "gray", "-f", "matroska",
                               part], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    try:
        while True:
            data = reader.stdout.read(w * h * 3)
            if len(data) < w * h * 3:
                break
            score = greenness(np.frombuffer(data, np.uint8).reshape(h, w, 3))
            writer.stdin.write((alpha(score, found) * matte).astype(np.uint8).tobytes())
            count += 1
        writer.stdin.close()
        said = writer.stderr.read().decode("utf-8", "replace").strip()
        if writer.wait(MASK_SECONDS) != 0 or not count:
            raise GreenError("FFmpeg could not cut the person out of the green cloth: %s."
                             % (said.splitlines() or ["no frames were read"])[-1])
        render.settle(part, path)
    except (subprocess.SubprocessError, OSError) as e:
        writer.kill()
        raise GreenError("FFmpeg could not cut the person out of the green cloth (%s)." % e)
    finally:
        reader.kill()
        reader.wait()
        reader.stdout.close()
        writer.wait()
        writer.stderr.close()
        if os.path.exists(part):
            os.remove(part)
    return made


def backdrop(about):
    """{path, usd, seconds, reused}: the background Muse Image draws for the words about (where the clip is set, such
    as "a bright kitchen counter"), fitted to the 1080x1920 frame in work/pictures. picture.PictureError with one
    sentence."""
    got = picture.make(about, ASK, SIZE)
    return dict(got, path=fitted(got["path"]))


def fitted(path):
    """The path of a 1080x1920 picture of the one at path, scaled to cover the frame with its middle kept, made once
    in work/pictures."""
    sign = hashlib.sha1(json.dumps([os.path.abspath(path), os.path.getsize(path), WIDTH, HEIGHT]).encode("utf-8"))
    target = os.path.join(render.folder("pictures"), "ground_%s.png" % sign.hexdigest()[:16])
    if os.path.isfile(target):
        return target
    part = render._part(target)
    try:
        done = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-i", path, "-vf",
                               "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d" % ((WIDTH, HEIGHT) * 2),
                               "-frames:v", "1", "-f", "image2", "-c:v", "png", part],
                              capture_output=True, text=True, errors="replace", timeout=FIT_SECONDS)
        if done.returncode != 0 or not os.path.getsize(part):
            raise picture.PictureError("%s's background could not be read: %s." % (
                picture.LABEL, (done.stderr.strip().splitlines() or ["FFmpeg gave no reason"])[-1]))
        render.settle(part, target)
    except (subprocess.SubprocessError, OSError) as e:
        raise picture.PictureError("FFmpeg could not fit %s's background to the frame (%s)." % (picture.LABEL, e))
    finally:
        if os.path.exists(part):
            os.remove(part)
    return target
