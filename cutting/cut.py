"""The cut-out worker: which pixels of a stretch of the recording are the person, frame by frame, found by Meta's
SAM 3.1 on the graphics card. It runs in SAM's own Python install, apart from the studio (clipstudio/person.py
starts it with that Python for one stretch and reads what it prints):

    git clone https://github.com/facebookresearch/sam3      (run here at f1647e8d, 10-23)
    python -m venv sam-venv
    sam-venv/Scripts/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
    sam-venv/Scripts/pip install -e sam3 safetensors

    job, one JSON object on stdin:
      {"recording": path, "from": s, "to": s, "fps": n, "size": [w, h] as the studio sees the recording,
       "face": [x, y, w, h] in its pixels or null, "weights": the .safetensors file, "frames": an empty folder for
       the pictures, "out": the FFV1 file to write, and for a talk "people": [[x, y, w, h] of each person's face]
       with "each": an empty folder for their own masks, and "prompt": what to find instead of "person", such as
       "white game controller" for the product a seller holds}
    stdout, one JSON object a line: {"stage": "pictures" | "loading" | "finding" | "following" | "writing"},
      {"done": k, "of": n} while following, and last {"ok": true, "frames": n, "people": [...], "kept": id,
      "each": {"1": id or null, ...}, "seconds": {...}} or {"ok": false, "error": a sentence}

SAM is asked for "person" on the first picture, follows everyone it finds there and anyone who comes in later. The
person kept is the one whose mask holds the middle of the face box in the most pictures (a stream's face camera),
or with no face box the one with the most pixels over the stretch. With people, each of them gets their own mask
too, n.mkv in each for the n-th face, picked the same way from the same following, so a talk's speaker can keep
their colour while the others do not (Tarm 10-30); a face nobody holds gets none. everyone.mkv in each holds all of
the people SAM followed, a face or not, for a talk whose speakers cannot be told. The pictures go in at most LONGEST
pixels on the long side, since SAM works at 1008 anyway (measured 10-28: the whole 1920x1080 frame and a 640 px crop round the
face camera gave the same mask, IoU .988, at the same 0.23 s a picture). The mask goes back on the recording's own
pixel grid, frame k at recording second from + k / fps, 255 the person, with a soft edge of about a pixel.

These settings make SAM run here (Windows, an RTX 5070 Ti, PyTorch 2.14 with CUDA 13), as found when it was first
run on this computer (10-27) and kept for every measurement since: the weights are read from the .safetensors file
(SAM's builder reads them with torch.load, which cannot); the model sits on the card in bfloat16; the decoder's
feed-forward layers stay in float32; attention uses whichever kernel PyTorch has, not only FlashAttention; finding
and post-processing take one object at a time; init_state gets only the arguments it takes.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

LONGEST = 1920
PROMPT = "person"
EDGE = 1.0


TOUCH = 0.2


def say(**what):
    print(json.dumps(what), flush=True)


def picture_size(size):
    """(w, h) of the pictures SAM is given for a recording of size: at most LONGEST on the long side, even."""
    w, h = size
    k = min(1.0, LONGEST / float(max(w, h)))
    return max(2, int(w * k / 2 + 0.5) * 2), max(2, int(h * k / 2 + 0.5) * 2)


def pick(masks, face, size, picture, prompt=PROMPT):
    """The id to keep from masks {id: [mask per picture, packed by np.packbits along rows, or None]}, pictures of
    picture (w, h): the one holding the middle of face (in pixels of the recording's size) in the most pictures, or
    with no face the one with the most pixels. ValueError with a sentence when nothing is found or nobody holds the
    face."""
    if not masks:
        raise ValueError("SAM found no %s in the stretch." % prompt)
    if face is None:
        return max(masks, key=lambda i: sum(int(np.unpackbits(m).sum()) for m in masks[i] if m is not None))
    x = min(picture[0] - 1, int((face[0] + face[2] / 2.0) * picture[0] / size[0]))
    y = min(picture[1] - 1, int((face[1] + face[3] / 2.0) * picture[1] / size[1]))
    held = {i: sum(int(np.unpackbits(m[y])[x]) for m in frames if m is not None) for i, frames in masks.items()}
    best = max(held, key=lambda i: held[i])
    if held[best] == 0:
        raise ValueError("None of the %d people SAM found holds the face at %s." % (len(masks), list(face)))
    return best


def joined(masks, kept):
    """masks[kept] with the pictures it lacks from the ids SAM gave the same person after losing them: an id that shares
    at most TOUCH of its pictures with the person so far. An id that stands beside them longer is someone else. Lane 5
    10-31, 28 s of a review: three ids held 0-8.5 s, 8.5-14.9 s and 14.9-27.8 s, and the title behind him stood over his
    eyes."""
    def shown(frames):
        return {n for n, m in enumerate(frames) if m is not None and m.any()}

    def pixels(m):
        return int(np.unpackbits(m).sum())

    out, have = list(masks[kept]), shown(masks[kept])
    for i in sorted(masks, key=lambda i: -len(shown(masks[i]))):
        seen = shown(masks[i])
        if i != kept and len(seen & have) <= TOUCH * len(seen):
            for n in seen:
                if n not in have or pixels(masks[i][n]) > pixels(out[n]):
                    out[n] = masks[i][n]
            have |= seen
    return out


def everyone(masks):
    """[the masks of everyone SAM followed in each picture together, packed, or None where nobody is seen]."""
    out = [None] * max(len(frames) for frames in masks.values())
    for frames in masks.values():
        for n, m in enumerate(frames):
            if m is not None:
                out[n] = m if out[n] is None else out[n] | m
    return out


def holders(masks, people, size, picture):
    """{"n": the id picked for the n-th face of people, or None when nobody holds it}, as pick() picks for one."""
    out = {}
    for n, face in enumerate(people, 1):
        try:
            out[str(n)] = pick(masks, face, size, picture)
        except ValueError:
            out[str(n)] = None
    return out


def write(frames, path, size, picture, fps):
    """One person's masks (packed, None where they are not seen) as an FFV1 file of grey frames on the recording's
    pixel grid of size, from pictures of picture (w, h)."""
    (w, h), (pw, ph) = size, picture
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s",
                            "%dx%d" % (pw, ph), "-r", "%g" % fps, "-i", "-", "-vf",
                            "scale=%d:%d:flags=bilinear,gblur=sigma=%g" % (w, h, EDGE), "-c:v", "ffv1", "-pix_fmt",
                            "gray", "-f", "matroska", path], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    blank = np.zeros((ph, pw), np.uint8)
    try:
        for m in frames:
            enc.stdin.write(blank.tobytes() if m is None else (np.unpackbits(m, axis=-1)[:, :pw] * 255).tobytes())
        enc.stdin.close()
    except OSError:
        pass
    err = enc.stderr.read().decode("utf-8", "replace")
    if enc.wait() != 0:
        raise ValueError("FFmpeg could not write the mask (%s)." % err.strip()[-200:])


def load(weights):
    """SAM 3.1's video predictor with the settings above."""
    import contextlib
    import inspect

    import torch
    from safetensors.torch import load_file

    reading = torch.load

    def read(f, *a, **k):
        if isinstance(f, str) and f.endswith(".safetensors"):
            return load_file(f)
        if isinstance(f, str):
            return reading(f, *a, **k)
        # the builder's own tracker-only read of the same file: the full read right after fills the tracker too
        return {}

    torch.load = read
    torch.nn.Module.cuda = lambda self, device=None: self.to(device="cuda", dtype=torch.bfloat16)
    from sam3.model import decoder
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=weights, use_fa3=False)
    predictor.model.use_batched_grounding = False
    predictor.model.postprocess_batch_size = 1
    decoder.sdpa_kernel = lambda *a, **k: contextlib.nullcontext()
    for m in predictor.model.modules():
        if isinstance(m, decoder.TransformerDecoderLayer):
            m.linear1.float()
            m.linear2.float()
    init = predictor.model.init_state
    takes = set(inspect.signature(init).parameters)
    predictor.model.init_state = lambda **k: init(**{n: v for n, v in k.items() if n in takes})
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    return predictor


def cut(job):
    seconds = {}
    t = time.perf_counter()
    say(stage="pictures")
    pw, ph = picture_size(job["size"])
    made = subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.3f" % job["from"], "-t",
                           "%.3f" % (job["to"] - job["from"]), "-i", job["recording"], "-vf",
                           "fps=%g,scale=%d:%d" % (job["fps"], pw, ph), "-q:v", "2", "-start_number", "0",
                           os.path.join(job["frames"], "%05d.jpg")], capture_output=True, text=True)
    n = len([f for f in os.listdir(job["frames"]) if f.endswith(".jpg")])
    if made.returncode != 0 or not n:
        raise ValueError("FFmpeg could not take pictures from the recording (%s)." % made.stderr.strip()[-200:])
    seconds["pictures"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    say(stage="loading")
    predictor = load(job["weights"])
    seconds["loading"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    say(stage="finding")
    sid = predictor.handle_request(dict(type="start_session", resource_path=job["frames"],
                                        offload_video_to_cpu=True))["session_id"]
    prompt = job.get("prompt") or PROMPT
    first = predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=prompt))["outputs"]
    people = [{"id": int(i), "sure": round(float(p), 3)}
              for i, p in zip(first["out_obj_ids"].tolist(), first["out_probs"].tolist())]
    seconds["finding"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    say(stage="following")
    masks = {}
    done = 0
    for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                                                  propagation_direction="forward")):
        out = r["outputs"]
        for i, m in zip(out["out_obj_ids"].tolist(), out["out_binary_masks"]):
            masks.setdefault(int(i), [None] * n)[r["frame_index"]] = np.packbits(np.asarray(m, bool), axis=-1)
        done += 1
        if done % 10 == 0 or done == n:
            say(done=done, of=n)
    predictor.handle_request(dict(type="close_session", session_id=sid))
    seconds["following"] = round(time.perf_counter() - t, 1)
    kept = pick(masks, job.get("face"), job["size"], (pw, ph), prompt)
    each = holders(masks, job.get("people") or [], job["size"], (pw, ph))
    t = time.perf_counter()
    say(stage="writing")
    write(joined(masks, kept), job["out"], job["size"], (pw, ph), job["fps"])
    for k, i in each.items():
        if i is not None:
            write(masks[i], os.path.join(job["each"], "%s.mkv" % k), job["size"], (pw, ph), job["fps"])
    if job.get("each"):
        write(everyone(masks), os.path.join(job["each"], "everyone.mkv"), job["size"], (pw, ph), job["fps"])
    seconds["writing"] = round(time.perf_counter() - t, 1)
    return {"ok": True, "frames": n, "people": people, "kept": kept, "each": each, "seconds": seconds}


def main():
    try:
        say(**cut(json.loads(sys.stdin.read())))
    except Exception as e:
        say(ok=False, error=str(e) if isinstance(e, ValueError) else "SAM failed (%s: %s)." % (type(e).__name__, e))
        sys.exit(1)


if __name__ == "__main__":
    main()
