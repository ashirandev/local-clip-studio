"""The clip's project list: every change Muse or the person makes is one entry, and the clip is what the entries add
up to. Undo takes the last entry back. The same list draws the preview frame and the finished clip, so the two can
never disagree.

An entry is a plain dict with "do" saying what it is. Times are clip seconds, boxes on the recording are pixels of
the recording, and boxes on the frame are pixels of the 1080x1920 clip.
- clip: {"moment", "pieces": [[recording a, recording b], ...], "layout": a layouts.make() dict, "product":
  recording box or None, "people": [{"n", "seen", "of", "from", "to", "box", "points": [[clip second, x, y, w, h]]}]:
  the other people's faces, followed through the clip, "speech": [[clip a, clip b]] when someone speaks, [] when
  nobody does, None when the moment was never heard, "span": [recording a, recording b] of the moment the faces
  were looked for over, which the pieces lie inside, "made": the layout, crop, face, product and follow it was made
  with, "moved": True when it is the clip before it with its ends moved}. A moved clip keeps what sits on the footage
  where the footage is: the captions, blurs, hits, stickers, flurries, sounds, slow pushes and cut-out looks follow
  the seconds of the recording they were on (and go with them when those are cut off), while the titles and glass
  boxes stay at their time in the clip, as a title opens the clip wherever it now starts
- style: {"name", "font", "colour"}
- title: {"text", "start", "end", "place", "box": the place's frame box, "key"}
- captions: {"lines": text.caption_lines() lines in clip seconds, "band": [y, h], "pop": each word popped in as it is said, "lit": each word lit as it is
  said}, or {"lines": []} for none
- blur: {"box": recording box, "start", "end"}: a fixed box, such as a stream's chat, with "private": which private
  line it hides when the program found that line on the screen (such a blur belongs to its clip, and a new clip entry
  drops it); or {"track": [[clip second, x, y, w, h], ...] in recording pixels, "start", "end"}: a box that moves with
  a face
- hit: {"kind": flash, shake, punch or word, "at", "word", "centre": [x, y] on the frame}
- sticker: {"png": path of a PNG, "at", "length", "box": frame box the picture is fitted in, "name": the studio
  sticker's name, when it is one}; a sticker on the same spot at an overlapping time replaces an earlier one
- glass: {"box": frame box, "start", "end", "tint": light or dark, "shine": a sweep of light as it appears}; a glass
  box on the same spot at an overlapping time replaces an earlier one
- flurry: {"names": the studio stickers' names, "pngs": their PNGs (effects.CONFETTI for paper pieces), "at",
  "length", "box": frame box every copy stays inside, "move": one of effects.MOVES, "count": copies, "size": px of
  a picture's longer side}; a flurry on the same spot at an overlapping time replaces an earlier one
- untitle: {"text"}: takes the titles with these words off
- layout: {"part", "box": frame box, "layout": the clip's layout with that part moved where the person asked}
- look: {"name": one of effects.LOOKS, or None for the footage as it was shot}
- push: {"start", "end", "zoom", "centre": [x, y] on the frame}: a slow push-in on the main picture; a push over
  the time of an earlier one replaces it
- sound: {"kind": one of effects.SOUNDS, "at", "level": dB against the sound as it is made}; a sound of the same kind
  within 0.5 s of an earlier one replaces it
- music: {"path": the person's music file, "from": its second the clip starts at, "level": dB under the voice,
  "duck": dB lower while someone speaks}, or {"path": None} to take it off
- people: {"blur": [person numbers], "keep": [person numbers]}: who of the clip's people is blurred; a later
  decision about the same person wins, and a new clip entry starts with nobody decided
- mask: {"path", "person", "fps", "size", "from", "to"}: the person found frame by frame over recording seconds from
  to (person.cut_out), for the cut-out looks and the titles behind the person; the last one stands
- cutout: {"look": grey or outline, "start", "end"}, or {"look": None, "start", "end"} to take the looks at that time
  off; a cut-out look at an overlapping time replaces an earlier one
- mute: {"spans": [[recording a, recording b], ...], "start", "end": the clip seconds asked for, "back": False}: the
  recording's own sound is silent over those seconds of the recording, wherever the clip plays them, so a mute stays
  on what was said when the clip is cut again; "back": True brings the sound back over them
"""

import hashlib
import json

DOES = ("clip", "style", "title", "captions", "blur", "hit", "sticker", "people", "look", "push", "layout", "untitle",
        "sound", "music", "glass", "mask", "cutout", "channel", "flurry", "mute")


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
        clip], "glass": [glass entries inside the clip], "flurries": [flurry entries inside the clip], "look": the colour look's name or None, "pushes": [push entries inside the clip], "people": the clip's
        people, each with "blur" True, False or None when not decided, "sounds": [sound entries inside the clip],
        "music": the last music entry with a path or None, "speech": the clip's speech spans or None, "mask": the last
        mask entry or None, "cutouts": [cutout entries inside the clip], "channel": the last channel entry, the
        channel's name drawn on the clip ({"name", "spot"}), or None when its name is empty, "mutes": [[clip a, clip
        b]] where the recording's own sound is muted}. A blurred person's track is one of the blurs. A new clip entry
        keeps the words and effects; what starts after its end is not drawn. Where the sound is muted nobody is heard:
        the speech spans leave out those seconds, so the music does not go down there, and the captions leave out the
        words said there."""
        clip = style = captions = None
        titles, blurs, hits, stickers, pushes, sounds, glass, cutouts, flurries = [], [], [], [], [], [], [], [], []
        muted = []
        look = music = mask = channel = None
        decided = {}
        for e in self.entries:
            if e["do"] == "clip":
                if e.get("moved") and clip:
                    move = _Mover(clip["pieces"], e["pieces"])
                    captions = move.captions(captions)
                    hits, stickers, flurries, sounds = (move.at(x) for x in (hits, stickers, flurries, sounds))
                    pushes, cutouts = move.spans(pushes), move.spans(cutouts)
                    blurs = move.spans([move.track(b) for b in blurs])
                clip = e
                decided = {}
                blurs = [b for b in blurs if not b.get("private")]
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
                stickers = [k for k in stickers if not _same_box(k, e, "at", "length")] + [e]
            elif e["do"] == "glass":
                glass = [g for g in glass if not _same_box(g, e, "start", "end")] + [e]
            elif e["do"] == "flurry":
                flurries = [f for f in flurries if not _same_box(f, e, "at", "length")] + [e]
            elif e["do"] == "untitle":
                titles = [t for t in titles if t["text"] != e["text"]]
            elif e["do"] == "layout":
                clip = dict(clip, layout=e["layout"]) if clip else None
            elif e["do"] == "look":
                look = e["name"]
            elif e["do"] == "push":
                pushes = [p for p in pushes if not (p["start"] < e["end"] and e["start"] < p["end"])] + [e]
            elif e["do"] == "sound":
                sounds = [s for s in sounds if not (s["kind"] == e["kind"] and abs(s["at"] - e["at"]) < 0.5)] + [e]
            elif e["do"] == "music":
                music = e if e.get("path") else None
            elif e["do"] == "people":
                decided.update({n: True for n in e["blur"]})
                decided.update({n: False for n in e["keep"]})
            elif e["do"] == "mask":
                mask = e
            elif e["do"] == "channel":
                channel = e if e.get("name") else None
            elif e["do"] == "cutout":
                cutouts = [c for c in cutouts if not (c["start"] < e["end"] and e["start"] < c["end"])]
                cutouts += [e] if e["look"] else []
            elif e["do"] == "mute":
                muted = _less(muted, e["spans"]) if e.get("back") else _joined(muted + e["spans"])
        length = round(sum(b - a for a, b in clip["pieces"]), 2) if clip else 0
        mutes = to_clip_spans(clip["pieces"], muted) if clip else []
        speech = clip.get("speech") if clip else None
        inside = (lambda when: clip is not None and when < length)
        people = [dict(p, blur=decided.get(p["n"])) for p in (clip or {}).get("people") or []]
        blurs = blurs + [{"do": "blur", "person": p["n"], "track": p["points"], "start": p["from"], "end": p["to"]}
                         for p in people if p["blur"]]
        return {"clip": clip, "length": length, "style": style,
                "titles": [t for t in titles if inside(t["start"])],
                "captions": _heard(captions, mutes) if clip else None,
                "blurs": [b for b in blurs if inside(b["start"])],
                "hits": sorted([h for h in hits if inside(h["at"])], key=lambda h: h["at"]),
                "stickers": sorted([k for k in stickers if inside(k["at"])], key=lambda k: k["at"]),
                "glass": sorted([g for g in glass if inside(g["start"])], key=lambda g: g["start"]),
                "flurries": sorted([f for f in flurries if inside(f["at"])], key=lambda f: f["at"]),
                "look": look if clip else None,
                "pushes": sorted([p for p in pushes if inside(p["start"])], key=lambda p: p["start"]),
                "people": people,
                "sounds": sorted([s for s in sounds if inside(s["at"])], key=lambda s: s["at"]),
                "music": music if clip else None,
                "speech": _less(speech, mutes) if speech else speech,
                "mask": mask if clip else None,
                "channel": channel if clip else None,
                "cutouts": sorted([c for c in cutouts if inside(c["start"])], key=lambda c: c["start"]),
                "mutes": mutes}


class _Mover:
    """Clip seconds of the pieces before, as the clip seconds of the pieces after: the same second of the recording,
    or gone when the pieces after do not play it."""

    def __init__(self, before, after):
        self.before, self.after = before, after

    def span(self, a, b):
        """(a, b) after, from the first to the last of the seconds still played, or None when none is."""
        out = to_clip_spans(self.after, to_recording_spans(self.before, a, b))
        return (out[0][0], out[-1][1]) if out else None

    def point(self, t):
        hit = to_clip(self.before, t)
        return None if hit is None else from_recording(self.after, hit[1])

    def at(self, entries):
        return [dict(e, at=round(t, 3)) for e in entries for t in [self.point(e["at"])] if t is not None]

    def spans(self, entries):
        return [dict(e, start=round(s[0], 3), end=round(s[1], 3)) for e in entries
                for s in [self.span(e["start"], e["end"])] if s and (e.get("track") is None or e["track"])]

    def track(self, blur):
        """A blur that moves with a face, its points at their new times."""
        if "track" not in blur:
            return blur
        return dict(blur, track=[[round(t, 3)] + list(p[1:]) for p in blur["track"] for t in [self.point(p[0])]
                                 if t is not None])

    def captions(self, captions):
        """The caption lines with each word at its new time; a word no longer played goes, and a line goes with its
        last word."""
        if not captions:
            return captions
        lines = []
        for ln in captions["lines"]:
            words = [dict(w, **{"from": round(s[0], 3), "to": round(s[1], 3)}) for w in ln["words"]
                     for s in [self.span(w["from"], w["to"])] if s]
            if words:
                lines.append(dict(ln, words=words, text="".join(w["text"] for w in words) if len(words) < len(
                    ln["words"]) else ln["text"], **{"from": words[0]["from"], "to": words[-1]["to"]}))
        return dict(captions, lines=lines)


def signature(entries):
    """A short name for the clip as these entries make it: the same entries always give the same name, so a finished
    file can be matched to the clip it was made from."""
    return hashlib.sha1(json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _same_box(a, b, start, end):
    """Two frame boxes that overlap at overlapping times; end is an end time, or the length after start."""
    def until(e):
        return e[start] + e[end] if end == "length" else e[end]
    (ax, ay, aw, ah), (bx, by, bw, bh) = a["box"], b["box"]
    return (ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
            and a[start] < until(b) and b[start] < until(a))


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


def to_recording_spans(pieces, a, b):
    """Clip seconds a to b as the stretches of the recording they play: [[recording a, recording b]], one for each
    piece they cross."""
    out, at = [], 0.0
    for s, e in pieces:
        lo, hi = max(a, at), min(b, at + e - s)
        if hi - lo > SLIVER:
            out.append([round(s + lo - at, 3), round(s + hi - at, 3)])
        at += e - s
    return out


def to_clip_spans(pieces, spans):
    """Stretches of the recording as the clip plays them: [[clip a, clip b]], joined where they meet. A stretch that
    two pieces both play is in the clip twice; one no piece plays is not in it."""
    out, at = [], 0.0
    for a, b in pieces:
        for s, e in spans:
            lo, hi = max(a, s), min(b, e)
            if hi - lo > SLIVER:
                out.append([round(at + lo - a, 3), round(at + hi - a, 3)])
        at += b - a
    return _joined(out)


SLIVER = 0.005


def _joined(spans):
    """[[a, b]] sorted, with the ones that overlap or meet made one."""
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + SLIVER:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _less(spans, cuts):
    """[[a, b]] without the seconds of cuts ([[a, b]])."""
    for c, d in cuts:
        spans = [p for a, b in spans for p in ([a, min(b, c)], [max(a, d), b]) if p[1] - p[0] > SLIVER]
    return spans


def _heard(captions, mutes):
    """The captions without the words said where the sound is muted: a line keeps its other words, and goes when
    none are left. A word is said where its middle is."""
    if not captions or not mutes:
        return captions
    lines = []
    for ln in captions["lines"]:
        words = [w for w in ln["words"] if not any(a <= (w["from"] + w["to"]) / 2 < b for a, b in mutes)]
        if len(words) == len(ln["words"]):
            lines.append(ln)
        elif words:
            lines.append(dict(ln, words=words, text="".join(w["text"] for w in words), **{
                "from": words[0]["from"], "to": words[-1]["to"]}))
    return dict(captions, lines=lines)


def from_recording(pieces, s):
    """The clip second where recording second s shows, or None when no piece holds it."""
    at = 0.0
    for a, b in pieces:
        if a <= s <= b:
            return at + s - a
        at += b - a
    return None
