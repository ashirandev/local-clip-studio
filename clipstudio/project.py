"""The clip's project list: every change Muse or the person makes is one entry, and the clip is what the entries add
up to. Undo takes the last entry back. The same list draws the preview frame and the finished clip, so the two can
never disagree.

An entry is a plain dict with "do" saying what it is. Times are clip seconds, boxes on the recording are pixels of
the recording, and boxes on the frame are pixels of the 1080x1920 clip.
- clip: {"moment", "pieces": [[recording a, recording b], ...], "layout": a layouts.make() dict}
- style: {"name", "font", "colour"}
- title: {"text", "start", "end", "place", "box": the place's frame box, "key"}
- captions: {"lines": text.caption_lines() lines in clip seconds, "band": [y, h], "lit": each word lit as it is
  said}, or {"lines": []} for none
- blur: {"box": recording box, "start", "end"}: a fixed box, such as a stream's chat; or {"track": [[clip second, x,
  y, w, h], ...] in recording pixels, "start", "end"}: a box that moves with a face
- hit: {"kind": flash, shake, punch or word, "at", "word", "centre": [x, y] on the frame}
- sticker: {"png": path of a PNG, "at", "length", "box": frame box the picture is fitted in}
"""

import hashlib
import json

DOES = ("clip", "style", "title", "captions", "blur", "hit", "sticker")


class Project:
    def __init__(self, entries=()):
        self.entries = [dict(e) for e in entries]

    def add(self, entry):
        if entry.get("do") not in DOES:
            raise ValueError("An entry does one of %s, not %r." % (", ".join(DOES), entry.get("do")))
        self.entries.append(dict(entry))
        return entry

    def undo(self):
        """Takes the last entry back and returns it, or None when there is nothing to undo."""
        return self.entries.pop() if self.entries else None

    def state(self):
        """What the entries add up to: {"clip": the last clip entry or None, "length": clip seconds, "style": the
        last style entry or None, "titles": [title entries inside the clip], "captions": the last captions entry or
        None, "blurs": [blur entries], "hits": [hit entries inside the clip], "stickers": [sticker entries inside the
        clip]}. A new clip entry keeps the words and effects; what starts after its end is not drawn."""
        clip = style = captions = None
        titles, blurs, hits, stickers = [], [], [], []
        for e in self.entries:
            if e["do"] == "clip":
                clip = e
            elif e["do"] == "style":
                style = e
            elif e["do"] == "title":
                titles = [t for t in titles if not _same_spot(t, e)] + [e]
            elif e["do"] == "captions":
                captions = e if e["lines"] else None
            elif e["do"] == "blur":
                blurs.append(e)
            elif e["do"] == "hit":
                hits = [h for h in hits if not (h["kind"] == e["kind"] and abs(h["at"] - e["at"]) < 0.5)] + [e]
            elif e["do"] == "sticker":
                stickers.append(e)
        length = round(sum(b - a for a, b in clip["pieces"]), 2) if clip else 0
        inside = (lambda when: clip is not None and when < length)
        return {"clip": clip, "length": length, "style": style,
                "titles": [t for t in titles if inside(t["start"])],
                "captions": captions if clip else None,
                "blurs": [b for b in blurs if inside(b["start"])],
                "hits": sorted([h for h in hits if inside(h["at"])], key=lambda h: h["at"]),
                "stickers": sorted([k for k in stickers if inside(k["at"])], key=lambda k: k["at"])}


def signature(entries):
    """A short name for the clip as these entries make it: the same entries always give the same name, so a finished
    file can be matched to the clip it was made from."""
    return hashlib.sha1(json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _same_spot(a, b):
    """Two titles in the same place at overlapping times: the later one replaces the earlier."""
    return a["place"] == b["place"] and a["start"] < b["end"] and b["start"] < a["end"]


def to_clip(pieces, t):
    """(piece number, recording second) for clip second t, or None when t is outside the clip."""
    at = 0.0
    for k, (a, b) in enumerate(pieces):
        if t < at + (b - a) or k == len(pieces) - 1 and t <= at + (b - a):
            return (k, a + t - at) if t >= at else None
        at += b - a
    return None


def from_recording(pieces, s):
    """The clip second where recording second s shows, or None when no piece holds it."""
    at = 0.0
    for a, b in pieces:
        if a <= s <= b:
            return at + s - a
        at += b - a
    return None
