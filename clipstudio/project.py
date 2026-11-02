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
- picture: {"png": picture.fitted()'s PNG, "at", "length", "box": frame box it covers, "about": the words Muse
  Image drew it from}; a picture at an overlapping time replaces an earlier one
- glass: {"box": frame box, "start", "end", "tint": light or dark, "shine": a sweep of light as it appears}; a glass
  box on the same spot at an overlapping time replaces an earlier one
- flurry: {"names": the studio stickers' names, "pngs": their PNGs (effects.CONFETTI for paper pieces), "at",
  "length", "box": frame box every copy stays inside, "move": one of effects.MOVES, "count": copies, "size": px of
  a picture's longer side}; a flurry on the same spot at an overlapping time replaces an earlier one
- untitle: {"text"}: takes the titles with these words off
- layout: {"part", "box": frame box, "layout": the clip's layout with that part moved where the person asked}; with
  part "captions" the captions on the clip take the band of box too
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
  to (person.cut_out, or green.mask off a green cloth with "green" too), for the cut-out looks, the titles behind the
  person and the background; the last one stands
- backdrop: {"png": a 1080x1920 picture, "about": the words it was drawn from}: the background behind the person
  wherever the clip does not show them (green.backdrop), or {"png": None} to take it off; the last one stands
- cutout: {"look": grey or outline, "start", "end"}, or {"look": None, "start", "end"} to take the looks at that time
  off; a cut-out look at an overlapping time replaces an earlier one
- mute: {"spans": [[recording a, recording b], ...], "start", "end": the clip seconds asked for, "back": False}: the
  recording's own sound is silent over those seconds of the recording, wherever the clip plays them, so a mute stays
  on what was said when the clip is cut again; "back": True brings the sound back over them
"""

import functools
import hashlib
import json
import os

DOES = ("clip", "style", "title", "captions", "blur", "hit", "sticker", "people", "look", "push", "layout", "untitle",
        "sound", "music", "glass", "mask", "cutout", "channel", "flurry", "mute", "picture", "backdrop", "eye", "effect",
        "tracks")
MADE_BY = ("clipstudio", "cutting", "fonts", "stickers")


class Project:
    def __init__(self, entries=(), acts=None):
        self.entries = [dict(e) for e in entries]
        self.acts = list(acts) if acts is not None else [None] * len(self.entries)

    def add(self, entry, act=None):
        """Adds entry; act names the action it is part of (one tool call, one tap), for undo."""
        if entry.get("do") not in DOES:
            raise ValueError("An entry does one of %s, not %r." % (", ".join(DOES), entry.get("do")))
        self.entries.append(dict(entry))
        self.acts.append(act)
        return entry

    def undo(self):
        """Takes the last change back and returns its first entry, or None when there is nothing to undo. The entries
        one action made (one tool call of Muse's, one tap on the page) share its act and go back together: the
        Trim tab's Apply makes the clip again, then its blurs and the decision on its people, and one Undo puts the
        old clip back (10-29: it took two)."""
        if not self.entries:
            return None
        entry, act = self.entries.pop(), self.acts.pop()
        while act is not None and self.acts and self.acts[-1] == act:
            entry = self.entries.pop()
            self.acts.pop()
        return entry

    def state(self):
        """What the entries add up to: {"clip": the last clip entry or None, "length": clip seconds, "style": the
        last style entry or None, "titles": [title entries inside the clip], "captions": the last captions entry or
        None, "blurs": [blur entries], "hits": [hit entries inside the clip], "stickers": [sticker entries inside the
        clip], "pictures": [picture entries inside the clip], "glass": [glass entries inside the clip], "flurries":
        [flurry entries inside the clip], "look": the colour look's name or None, "pushes": [push entries inside the
        clip], "people": the clip's people, each with "blur" True, False or None when not decided, "sounds": [sound
        entries inside the clip], "music": the last music entry with a path or None, "speech": the clip's speech spans
        or None, "mask": the last mask entry or None, "backdrop": the last backdrop entry with a picture or None,
        "cutouts": [cutout entries inside the clip], "channel": the last channel entry, the
        channel's name drawn on the clip ({"name", "spot"}), or None when its name is empty, "mutes": [[clip a, clip
        b]] where the recording's own sound is muted, "hidden": the captions the person hid with the eye over the
        clip, or None}. Hidden, the captions are kept as they are, the lines the person typed too, and are left out of
        "captions", so the video, the checks and the page leave them off until the eye shows them again (Khun Eve
        10-30: the hearing is not always right, and Off emptied the lines). A blurred person's track is one of the
        blurs. A new clip entry
        keeps the words and effects; what starts after its end is not drawn. Where the sound is muted nobody is heard:
        the speech spans leave out those seconds, so the music does not go down there, and the captions leave out the
        words said there."""
        clip = style = captions = None
        titles, blurs, hits, stickers, pushes, sounds, glass, cutouts, flurries = [], [], [], [], [], [], [], [], []
        pictures = []
        muted = []
        tracks = None
        look = music = mask = channel = backdrop = None
        hidden = False
        decided = {}
        for e in self.entries:
            if e["do"] == "clip":
                if e.get("moved") and clip:
                    move = _Mover(clip["pieces"], e["pieces"])
                    captions = move.captions(captions)
                    hits, stickers, flurries, sounds, pictures = (move.at(x) for x in (hits, stickers, flurries,
                                                                                         sounds, pictures))
                    pushes, cutouts = move.spans(pushes), move.spans(cutouts)
                    blurs = move.spans([move.track(b) for b in blurs])
                decided = _carried(decided, clip, e) if e.get("moved") else {}
                clip = e
                blurs = [b for b in blurs if not b.get("private")]
            elif e["do"] == "style":
                style = e
            elif e["do"] == "title":
                titles = [t for t in titles if not same_spot(t, e)] + [e]
            elif e["do"] == "captions":
                captions = e if e["lines"] or e.get("drafts") else None
            elif e["do"] == "blur":
                blurs.append(e)
            elif e["do"] == "hit":
                hits = [h for h in hits if not (h["kind"] == e["kind"] and abs(h["at"] - e["at"]) < 0.5)] + [e]
            elif e["do"] == "sticker":
                stickers = [k for k in stickers if not _same_box(k, e, "at", "length")] + [e]
            elif e["do"] == "picture":
                pictures = [k for k in pictures if not (k["at"] < e["at"] + e["length"]
                                                        and e["at"] < k["at"] + k["length"])] + [e]
            elif e["do"] == "glass":
                glass = [g for g in glass if not _same_box(g, e, "start", "end")] + [e]
            elif e["do"] == "flurry":
                flurries = [f for f in flurries if not _same_box(f, e, "at", "length")] + [e]
            elif e["do"] == "untitle":
                titles = [t for t in titles if t["text"] != e["text"]]
            elif e["do"] == "effect":
                if e["of"] == "hit":
                    hits = _changed(hits, e, _near)
                elif e["of"] == "sound":
                    sounds = _changed(sounds, e, _near)
                elif e["of"] == "sticker":
                    stickers = _changed(stickers, e, lambda a, b: _same_box(a, b, "at", "length"))
                elif e["of"] == "flurry":
                    flurries = _changed(flurries, e, lambda a, b: _same_box(a, b, "at", "length"))
                elif e["of"] == "picture":
                    pictures = _changed(pictures, e, _crossing)
                elif e["of"] == "title":
                    titles = _changed(titles, e, same_spot)
                else:
                    pushes = _changed(pushes, e, lambda a, b: a["start"] < b["end"] and b["start"] < a["end"])
            elif e["do"] == "layout":
                clip = dict(clip, layout=e["layout"]) if clip else None
                if captions and e.get("part") == "captions":
                    captions = dict(captions, band=e["box"][1::2])
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
            elif e["do"] == "backdrop":
                backdrop = e if e.get("png") else None
            elif e["do"] == "channel":
                channel = e if e.get("name") else None
            elif e["do"] == "cutout":
                cutouts = [c for c in cutouts if not (c["start"] < e["end"] and e["start"] < c["end"])]
                cutouts += [e] if e["look"] else []
            elif e["do"] == "mute":
                muted = _less(muted, e["spans"]) if e.get("back") else _joined(muted + e["spans"])
            elif e["do"] == "eye":
                hidden = e["hidden"]
            elif e["do"] == "tracks":
                tracks = e["on"]
        length = round(sum(b - a for a, b in clip["pieces"]), 2) if clip else 0
        mutes = to_clip_spans(clip["pieces"], muted) if clip else []
        speech = clip.get("speech") if clip else None
        inside = (lambda when: clip is not None and when < length)
        people = [dict(p, blur=decided.get(p["n"])) for p in (clip or {}).get("people") or []]
        blurs = blurs + [{"do": "blur", "person": p["n"], "track": p["points"], "start": p["from"], "end": p["to"]}
                         for p in people if p["blur"]]
        return {"clip": clip, "length": length, "style": style,
                "titles": [t for t in titles if inside(t["start"])],
                "captions": _heard(captions, mutes) if clip and not hidden else None,
                "hidden": _heard(captions, mutes) if clip and hidden and captions else None,
                "blurs": [b for b in blurs if inside(b["start"])],
                "hits": sorted([h for h in hits if inside(h["at"])], key=lambda h: h["at"]),
                "stickers": sorted([k for k in stickers if inside(k["at"])], key=lambda k: k["at"]),
                "pictures": sorted([k for k in pictures if inside(k["at"])], key=lambda k: k["at"]),
                "glass": sorted([g for g in glass if inside(g["start"])], key=lambda g: g["start"]),
                "flurries": sorted([f for f in flurries if inside(f["at"])], key=lambda f: f["at"]),
                "look": look if clip else None,
                "pushes": sorted([p for p in pushes if inside(p["start"])], key=lambda p: p["start"]),
                "people": people,
                "sounds": sorted([s for s in sounds if inside(s["at"])], key=lambda s: s["at"]),
                "music": music if clip else None,
                "speech": _less(speech, mutes) if speech else speech,
                "mask": mask if clip else None,
                "backdrop": backdrop if clip else None,
                "channel": channel if clip else None,
                "cutouts": sorted([c for c in cutouts if inside(c["start"])], key=lambda c: c["start"]),
                "mutes": mutes, "tracks": tracks}


def effect_name(of, effect):
    """What tells one effect from another at the same second: a hit's or a sound's kind, a sticker's file, a flurry's
    stickers, a picture's words, a title's words. A slow push has no name, as two never cross."""
    if of == "title":
        return effect["text"]
    if of in ("hit", "sound"):
        return effect["kind"]
    if of == "sticker":
        return os.path.splitext(os.path.basename(effect["png"]))[0]
    return "+".join(effect["names"]) if of == "flurry" else effect["about"] if of == "picture" else "push"


def _near(a, b):
    return a["kind"] == b["kind"] and abs(a["at"] - b["at"]) < 0.5


def _crossing(a, b):
    return a["at"] < b["at"] + b["length"] and b["at"] < a["at"] + a["length"]


LASTING = ("push", "title")


def _changed(items, e, clash):
    """items without the effect e names, or with it at e["to"]: the person's hand on one effect (Tarm 11-01: every step
    of Muse's is theirs to change on the timeline before they export). One moved onto another of its kind replaces
    it, as adding it there would."""
    key = "start" if e["of"] in LASTING else "at"
    mine = [x for x in items if abs(x[key] - e["at"]) < 0.01 and effect_name(e["of"], x) == e["name"]]
    rest = [x for x in items if not any(x is m for m in mine)]
    if e.get("to") is None or not mine:
        return rest
    moved = dict(mine[0], **{key: e["to"]})
    moved.update({k: e[k] for k in ("box", "place") if e.get(k)})
    if e["of"] in LASTING:
        moved["end"] = round(moved["end"] + e["to"] - e["at"], 2)
    return [x for x in rest if not clash(x, moved)] + [moved]


def _carried(decided, old, new):
    """What was decided about the people of clip old, for the same faces in clip new, made again from the same moment: a
    person is known by the box of their face, as new's numbers can differ. Lane 5 10-31: Subject tapped on a match reel
    wiped Muse's decisions about 318 faces, the render was refused until each was tapped, and the page took them one by
    one."""
    was = {tuple(p["box"]): decided[p["n"]] for p in (old or {}).get("people") or [] if p["n"] in decided}
    return {p["n"]: was[tuple(p["box"])] for p in new.get("people") or [] if tuple(p["box"]) in was}


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
        """The caption lines and drafts with each word at its new time; a word no longer played goes, and a line goes
        with its last word."""
        if not captions:
            return captions
        moved = {}
        for key in ("lines", "drafts"):
            lines = []
            for ln in captions.get(key) or []:
                words = [dict(w, **{"from": round(s[0], 3), "to": round(s[1], 3)}) for w in ln["words"]
                         for s in [self.span(w["from"], w["to"])] if s]
                if words:
                    lines.append(dict(ln, words=words, text="".join(w["text"] for w in words) if len(words) < len(
                        ln["words"]) else ln["text"], **{"from": words[0]["from"], "to": words[-1]["to"]}))
            if key in captions:
                moved[key] = lines
        return dict(captions, **moved)


def fingerprint(root, tops=MADE_BY):
    """A short name for the files in root's folders tops, by their paths and bytes. Python's own cache is left out, so
    the same files give the same name whether they have run or not."""
    sha = hashlib.sha1()
    for top in tops:
        for folder, dirs, files in os.walk(os.path.join(root, top)):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(files):
                path = os.path.join(folder, name)
                sha.update(os.path.relpath(path, root).replace(os.sep, "/").encode("utf-8") + b"\0")
                with open(path, "rb") as f:
                    sha.update(f.read())
    return sha.hexdigest()[:16]


@functools.lru_cache(maxsize=None)
def program():
    """The fingerprint of the program that draws the finished file: its code, fonts and stickers. An update to any of
    them changes it."""
    return fingerprint(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def signature(entries):
    """A short name for the clip as these entries make it with this program: the same entries give the same name until
    the program is updated, so a finished file is matched to the clip it was made from, and a file the program drew
    before an update is not (10-30: after an update changed how a look is drawn, render_clip gave back the file the
    old code drew)."""
    said = program() + json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(said.encode("utf-8")).hexdigest()[:16]


def _same_box(a, b, start, end):
    """Two frame boxes that overlap at overlapping times; end is an end time, or the length after start."""
    def until(e):
        return e[start] + e[end] if end == "length" else e[end]
    (ax, ay, aw, ah), (bx, by, bw, bh) = a["box"], b["box"]
    return (ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
            and a[start] < until(b) and b[start] < until(a))


def same_spot(a, b):
    """Two titles in the same place, or drawn in the same box, at overlapping times: the later one replaces the earlier.
    The box also matches a title saved when a box's place was named in pixels ("box at 100, 400")."""
    same = a["place"] == b["place"] or (a.get("box") is not None and a.get("box") == b.get("box"))
    return same and a["start"] < b["end"] and b["start"] < a["end"]


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
    """The captions and drafts without the words said where the sound is muted: a line keeps its other words, and
    goes when none are left. A word is said where its middle is."""
    if not captions or not mutes:
        return captions
    kept = {}
    for key in ("lines", "drafts"):
        lines = []
        for ln in captions.get(key) or []:
            words = [w for w in ln["words"] if not any(a <= (w["from"] + w["to"]) / 2 < b for a, b in mutes)]
            if len(words) == len(ln["words"]):
                lines.append(ln)
            elif words:
                lines.append(dict(ln, words=words, text="".join(w["text"] for w in words), **{
                    "from": words[0]["from"], "to": words[-1]["to"]}))
        if key in captions:
            kept[key] = lines
    return dict(captions, **kept)


def from_recording(pieces, s):
    """The clip second where recording second s shows, or None when no piece holds it."""
    at = 0.0
    for a, b in pieces:
        if a <= s <= b:
            return at + s - a
        at += b - a
    return None
