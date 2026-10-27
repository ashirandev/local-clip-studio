"""Faces: where the people are, found by the program and never guessed by Muse.

Muse's face boxes were wrong on the first try on all four test recordings (10-26), and a round crop showed a hand. So
the program looks: YuNet (OpenCV Zoo, MIT) finds the faces in pictures taken every half second across a stretch, in
one FFmpeg pass. The face camera of a stream is the face that sits at the same place in at least half of them; a
rough box from Muse is snapped to the face found inside it.

The pictures are 1280 pixels on their long side: a stream's face camera is still about 40 pixels tall there, which
YuNet finds. Boxes are given back in pixels of the recording. Results are kept per stretch in work/faces, so asking
again is instant.

    LCS_FACE_MODEL   the YuNet file (face_detection_yunet_2023mar.onnx); without it, the file is downloaded once from
                     OpenCV Zoo into work/models
"""
import hashlib
import json
import os
import shutil
import subprocess
import threading
import urllib.request

from . import recording

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
MODEL = os.environ.get("LCS_FACE_MODEL") or os.path.join(recording.WORK, "models", MODEL_NAME)
VERSION = 1
EVERY = 0.5
LONG_SIDE = 1280
SCORE = 0.7
STEADY_SHARE = 0.5
SAME_PLACE = 0.5
SAMPLE_SECONDS = 600
_lock = threading.Lock()
_detectors = {}


class NoFaces(Exception):
    """Faces cannot be looked for on this computer. The sentence says why."""


class Stopped(Exception):
    """The person pressed Stop while the faces were being looked for."""


def folder():
    path = os.path.join(recording.WORK, "faces")
    os.makedirs(path, exist_ok=True)
    return path


def _key(rec, a, b, every):
    return hashlib.sha1(("%d|%s|%d|%s|%.2f|%.2f|%.2f" % (VERSION, rec["path"], rec["size"], rec["mtime"], a, b, every)
                         ).encode("utf-8")).hexdigest()[:16]


def kept(rec, a, b, every=EVERY):
    """The faces found earlier in a-b of rec, or None."""
    path = os.path.join(folder(), _key(rec, a, b, every) + ".json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def sample(rec, a, b, every=EVERY, on_start=None):
    """{"size": [w, h], "every": s, "samples": [{"t": recording seconds, "faces": [[x, y, w, h, score], ...]}]}: the
    faces in a picture every `every` seconds from a to b, boxes in pixels of the recording. on_start(stop) is given a
    function that ends the FFmpeg pass at once (Stop on the page)."""
    found = kept(rec, a, b, every)
    if found is not None:
        return found
    tool = shutil.which("ffmpeg")
    if not tool:
        raise NoFaces("FFmpeg is not installed, so no pictures can be taken to look for faces.")
    width, height = rec["width"], rec["height"]
    scale = min(1.0, LONG_SIDE / max(width, height))
    w, h = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)
    detect = _detector(w, h)
    proc = subprocess.Popen([tool, "-v", "error", "-ss", "%.2f" % a, "-t", "%.2f" % (b - a), "-i", rec["path"],
                             "-map", "0:v:0", "-vf", "fps=1/%g,scale=%d:%d" % (every, w, h), "-f", "rawvideo",
                             "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    stopped = []

    def stop():
        stopped.append(True)
        proc.kill()

    if on_start:
        on_start(stop)
    samples, size = [], w * h * 3
    try:
        while True:
            data = proc.stdout.read(size)
            if len(data) < size:
                break
            boxes = [[round(x / scale, 1), round(y / scale, 1), round(bw / scale, 1), round(bh / scale, 1),
                      round(score, 3)] for x, y, bw, bh, score in detect(data, w, h)]
            samples.append({"t": round(a + len(samples) * every, 2), "faces": boxes})
    finally:
        proc.stdout.close()
        code = proc.wait()
    if stopped:
        raise Stopped()
    if code != 0 and not samples:
        raise NoFaces("FFmpeg could not take pictures from %s at %s-%s." % (rec["name"], recording.clock(a),
                                                                          recording.clock(b)))
    found = {"size": [width, height], "every": every, "samples": samples}
    if code == 0:
        path = os.path.join(folder(), _key(rec, a, b, every) + ".json")
        with open(path + ".part", "w", encoding="utf-8") as f:
            json.dump(found, f)
        os.replace(path + ".part", path)
    return found


def _detector(w, h):
    """detect(bgr bytes, w, h) -> [(x, y, w, h, score)] with YuNet, or NoFaces with the reason."""
    try:
        import cv2
        import numpy
    except ImportError:
        raise NoFaces("OpenCV is not installed (pip install opencv-python), so faces cannot be found.")
    path = _model()
    with _lock:
        net = _detectors.get((w, h))
        if net is None:
            net = _detectors[(w, h)] = cv2.FaceDetectorYN.create(path, "", (w, h), SCORE)

    def detect(data, w, h):
        picture = numpy.frombuffer(data, dtype=numpy.uint8).reshape(h, w, 3)
        with _lock:
            _, faces = net.detect(picture)
        return [] if faces is None else [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1]))
                                         for f in faces]

    return detect


def _model():
    if os.path.isfile(MODEL):
        return MODEL
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(MODEL + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(MODEL + ".part", MODEL)
    except OSError as e:
        raise NoFaces("The face finder's model file could not be downloaded (%s). Put %s at %s or set LCS_FACE_MODEL."
                      % (e, MODEL_NAME, MODEL))
    return MODEL


def _centre(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _same_place(box, other):
    (ax, ay), (bx, by) = _centre(box), _centre(other)
    size = max(box[2], box[3], other[2], other[3])
    return abs(ax - bx) <= SAME_PLACE * size and abs(ay - by) <= SAME_PLACE * size and \
        0.67 <= box[3] / max(1e-6, other[3]) <= 1.5


def places(found):
    """The faces grouped by where they sit, most seen first: [{"box": median box, "seen": pictures, "of": pictures
    in all}]. A face that moves across the frame makes several groups."""
    groups = []
    for s in found["samples"]:
        for face in s["faces"]:
            box = face[:4]
            group = next((g for g in groups if _same_place(box, g["first"]) and g["last_t"] != s["t"]), None)
            if group is None:
                group = {"first": box, "boxes": [], "last_t": None}
                groups.append(group)
            group["boxes"].append(box)
            group["last_t"] = s["t"]
    out = []
    for g in groups:
        median = [sorted(b[i] for b in g["boxes"])[len(g["boxes"]) // 2] for i in range(4)]
        out.append({"box": [round(v, 1) for v in median], "seen": len(g["boxes"]), "of": len(found["samples"])})
    return sorted(out, key=lambda g: -g["seen"])


def steady(found):
    """The face at the same place in at least half the pictures (a stream's face camera, or a person sitting still),
    or None."""
    best = places(found)
    if best and best[0]["of"] and best[0]["seen"] >= STEADY_SHARE * best[0]["of"]:
        return best[0]
    return None


def snap(rough, found):
    """The face group whose middle lies inside rough (x, y, w, h, grown by a quarter on each side), most seen first,
    or None."""
    x, y, w, h = rough
    x0, y0, x1, y1 = x - w / 4, y - h / 4, x + w * 1.25, y + h * 1.25
    for g in places(found):
        cx, cy = _centre(g["box"])
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return g
    return None


def between(found, a, b):
    """[(recording second, [boxes])] of the pictures taken from a to b."""
    return [(s["t"], [f[:4] for f in s["faces"]]) for s in found["samples"] if a - 1e-6 <= s["t"] <= b + 1e-6]
