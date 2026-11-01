"""The clip as a project an editor finishes in Premiere Pro: a sequence in Final Cut Pro 7 XML, which Premiere imports
(File > Import), the captions as .srt and a note on what came over.

Every piece on the timeline points at the person's own file, named by it, with its in and out points in that file,
so an editor drags a piece's edge to take more or less of it (Tarm 10-29: "น้องผมต้องลาก timeline ของแต่ละวิดีโอ
ยาวขึ้น สั้นลงเพื่อปรับซีนได้"). A montage was joined into one recording first; its pieces point back at the clips it
was joined from (montage.made_of), not at the joined file.

The frame comes over as each part's scale and position (Basic Motion), with a crop when a part does not fill the
frame. Premiere places a clip's middle at the sequence's middle plus center times the clip's own size in pixels
(read from Premiere's exports by two tools that parse them; no one has yet opened this file in Premiere here). Looks,
stickers, blurs, sounds and the captions' style do not come over: the note says which, and a blur matters most.
"""
import io
import os
import re
import urllib.parse
import zipfile
from xml.sax.saxutils import escape

from . import montage, recording

FPS = 30
TICKS = 254016000000
W, H = 1080, 1920
NTSC = {24: 23.976, 30: 29.97, 60: 59.94}
NOTE = """How to open this in Premiere Pro

1. File > Import, and choose {name}.xml. A bin with the sequence "{title}" comes in. Open the sequence.
2. File > Import, and choose {name}.srt. Drag it onto the timeline: it becomes a caption track.

Each piece on the timeline is a piece of your own file, named by the file. Drag a piece's edge to take more or less
of it: the whole file is there. If Premiere says the media is offline, choose Link Media and point at the folder the
files are in now; the names are the same.

What came over: the pieces in order, how each fills the 1080x1920 frame (scale, position, and a crop when a part
does not fill the frame), the sound, the music, the titles and the reason for each piece as markers, and the captions
as .srt.

What did not come over, so do it in Premiere if you want it:
{left_out}
"""
LEFT_OUT = [
    ("blurs", "The blurs. A face or a chat the studio blurred is NOT blurred here: blur it before you post."),
    ("look", "The colour look ({look})."),
    ("stickers", "Stickers, glass boxes and moving stickers."),
    ("hits", "Hits (flashes, shakes, pops) and sound effects."),
    ("mutes", "Muted parts of the sound."),
    ("titles", "The titles' words on the picture: each is a marker with its words, at its time."),
    ("circle", "The round face circle and the card's rounded corners: they come over as rectangles."),
    ("background", "The blurred copy of the picture behind the card or behind a lying clip."),
    ("captions", "The captions' look: the .srt has the words and times only."),
    ("channel", "The channel's name on the clip."),
    ("tracks", "Sound tracks other than the first: this timeline holds track 1 of your file. Add the others from the file."),
]


class NotExported(Exception):
    """The clip cannot be made into a project. The sentence says why."""


def _frames(seconds, rate=FPS):
    return int(round(seconds * rate))


STILLS = (".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp")


def _still(item):
    """A photo: it has no length of its own, so Premiere is given one as long as its piece."""
    return os.path.splitext(item["path"])[1].lower() in STILLS


def _rate(fps):
    """(timebase, ntsc) of a file's frame rate: 29.97 is timebase 30 on NTSC."""
    fps = float(fps or FPS)
    for base, real in NTSC.items():
        if abs(fps - real) < 0.02:
            return base, True
    return int(round(fps)), False


def _real(rate):
    base, ntsc = rate
    return base * 1000 / 1001.0 if ntsc else float(base)


def _url(path):
    """file://localhost/H%3a/Phone/x.MOV, as Premiere writes a path."""
    path = os.path.abspath(path).replace("\\", "/")
    return "file://localhost/" + urllib.parse.quote(path.lstrip("/"), safe="/").replace("%3A", "%3a")


def _framing(piece):
    """((scale, tx, ty), whole): how a montage piece of its own file is drawn into the joined W x H recording, as
    montage.graph draws it (filled, cropped upright around its centre, or shown whole over a blurred copy of itself,
    whole True)."""
    w, h = float(piece["width"]), float(piece["height"])
    if montage.fills(w, h):
        s = max(W / w, H / h)
        return (s, (W - w * s) / 2, (H - h * s) / 2), False
    if piece.get("centre") is not None:
        wide = int(round(w * H / h / 2)) * 2
        x = int(round(min(max(piece["centre"] * wide - W / 2.0, 0), wide - W)))
        return (wide / w, -x, 0.0), False
    s = min(W / w, H / h)
    return (s, (W - w * s) / 2, (H - h * s) / 2), True


def items(rec, state):
    """The clip's pieces as they go on the timeline, in order: [{"path", "name", "width", "height", "seconds", "fps",
    "sound", "from", "to": seconds of that file, "at": second of the clip, "framing": (scale, tx, ty) from that file's
    pixels to rec's, "whole": shown whole over a blurred copy, "why"}]. A piece of a joined montage that crosses from one of its clips to the next is split there,
    each part pointing at its own clip. The joined file can run a frame past its pieces' seconds added up; the last
    clip takes that from its own file, so the timeline is as long as the clip."""
    made = montage.made_of(rec["path"])
    out, at = [], 0.0
    for a, b in state["clip"]["pieces"]:
        if not made:
            out.append({"path": rec["path"], "name": rec["name"], "width": rec["width"], "height": rec["height"],
                        "seconds": rec["seconds"], "fps": rec.get("fps"), "sound": bool(rec.get("sound_tracks")),
                        "from": a, "to": b, "at": round(at, 3), "framing": (1.0, 0.0, 0.0), "whole": False,
                        "why": None})
            at += b - a
            continue
        starts = montage.starts([{"from": m["from"], "to": m["to"]} for m in made])
        for k, (start, p) in enumerate(zip(starts, made)):
            end = start + p["to"] - p["from"] if k < len(made) - 1 else max(b, start + p["to"] - p["from"])
            x, y = max(a, start), min(b, end)
            if y - x < 0.01:
                continue
            out.append({"path": p["path"], "name": p["name"], "width": p["width"], "height": p["height"],
                        "seconds": p["seconds"], "fps": p.get("fps"), "sound": bool(p.get("sound_tracks")),
                        "from": round(p["from"] + x - start, 3), "to": round(p["from"] + y - start, 3),
                        "at": round(at, 3), "framing": _framing(p)[0], "whole": _framing(p)[1], "why": p.get("why")})
            at += y - x
    return [_as_still(i) if _still(i) else i for i in out]


def _as_still(item):
    shown = round(item["to"] - item["from"], 3)
    return dict(item, fps=FPS, seconds=shown, sound=False, **{"from": 0.0, "to": shown})


def motion(item, part):
    """(scale %, center horiz, center vert, crop [left, right, top, bottom] % or None) that shows the layout part's
    source box at its frame box, for a piece drawn into rec by item's framing."""
    s1, tx1, ty1 = item["framing"]
    sx, sy, sw, sh = part["src"]
    dx, dy, dw, dh = part["dst"]
    s2 = max(dw / float(sw), dh / float(sh))
    s, tx, ty = s1 * s2, tx1 * s2 + dx - sx * s2, ty1 * s2 + dy - sy * s2
    w, h = float(item["width"]), float(item["height"])
    horiz, vert = (w * s / 2 + tx - W / 2.0) / w, (h * s / 2 + ty - H / 2.0) / h
    crop = None
    if [round(v) for v in part["dst"]] != [0, 0, W, H]:
        x0, y0 = max(0.0, (dx - tx) / s), max(0.0, (dy - ty) / s)
        x1, y1 = min(w, (dx + dw - tx) / s), min(h, (dy + dh - ty) / s)
        crop = [round(100 * x0 / w, 3), round(100 * (w - x1) / w, 3), round(100 * y0 / h, 3),
                round(100 * (h - y1) / h, 3)]
    return round(s * 100, 3), round(horiz, 5), round(vert, 5), crop


def _param(pid, name, value, low=None, high=None):
    limits = "" if low is None else "<valuemin>%s</valuemin><valuemax>%s</valuemax>" % (low, high)
    return ('<parameter authoringApp="PremierePro"><parameterid>%s</parameterid><name>%s</name>%s<value>%s</value>'
            '</parameter>' % (pid, name, limits, value))


def _filters(scale, horiz, vert, crop):
    out = ['<filter><effect><name>Basic Motion</name><effectid>basic</effectid><effectcategory>motion</effectcategory>'
           '<effecttype>motion</effecttype><mediatype>video</mediatype>',
           _param("scale", "Scale", scale, 0, 1000), _param("rotation", "Rotation", 0, -8640, 8640),
           _param("center", "Center", "<horiz>%s</horiz><vert>%s</vert>" % (horiz, vert)),
           _param("centerOffset", "Anchor Point", "<horiz>0</horiz><vert>0</vert>"),
           '</effect></filter>']
    if crop:
        out.append('<filter><effect><name>Crop</name><effectid>crop</effectid><effectcategory>motion</effectcategory>'
                   '<effecttype>motion</effecttype><mediatype>video</mediatype>')
        out += [_param(side, side, value, 0, 100) for side, value in zip(("left", "right", "top", "bottom"), crop)]
        out.append('</effect></filter>')
    return "".join(out)


def _rate_xml(rate):
    return "<rate><timebase>%d</timebase><ntsc>%s</ntsc></rate>" % (rate[0], "TRUE" if rate[1] else "FALSE")


class _Files:
    """Each file is written out in full once, then named by its id."""

    def __init__(self):
        self.ids = {}

    def xml(self, item, rate):
        key = os.path.normcase(os.path.abspath(item["path"]))
        if key in self.ids:
            return '<file id="%s"/>' % self.ids[key], self.ids[key]
        fid = self.ids[key] = "file-%d" % (len(self.ids) + 1)
        sound = ('<audio><samplecharacteristics><depth>16</depth><samplerate>48000</samplerate></samplecharacteristics>'
                 '<channelcount>2</channelcount></audio>' if item["sound"] else "")
        return ('<file id="%s"><name>%s</name><pathurl>%s</pathurl>%s<duration>%d</duration><media><video>'
                '<samplecharacteristics>%s<width>%d</width><height>%d</height><anamorphic>FALSE</anamorphic>'
                '<pixelaspectratio>square</pixelaspectratio><fielddominance>none</fielddominance>'
                '</samplecharacteristics></video>%s</media></file>' % (
                    fid, escape(item["name"]), escape(_url(item["path"])), _rate_xml(rate),
                    _frames(item["seconds"], _real(rate)), _rate_xml(rate), item["width"], item["height"], sound)), fid


def _place(item, length_frames):
    """(start, end, in, out, ticks in, ticks out): sequence frames for where the piece sits, its file's frames and
    Premiere's ticks for what it shows. Pieces meet exactly: each starts where the one before ends."""
    rate = _rate(item["fps"])
    start = _frames(item["at"])
    end = min(_frames(item["at"] + item["to"] - item["from"]), length_frames)
    shown = (end - start) / float(FPS)
    first = _frames(item["from"], _real(rate))
    return start, end, first, first + max(1, int(round(shown * _real(rate)))), int(round(item["from"] * TICKS)), \
        int(round((item["from"] + shown) * TICKS))


def sequence(title, rec, state):
    """The Final Cut Pro 7 XML of the clip as a 1080x1920 sequence at FPS: a video track for each part of the layout,
    the sound on two linked tracks (left and right), the music on two more, the titles and each piece's reason as
    markers. NotExported when there is no clip."""
    if not state.get("clip"):
        raise NotExported("There is no clip yet, so there is nothing to export.")
    pieces = items(rec, state)
    length = _frames(state["length"])
    parts = state["clip"]["layout"]["parts"]
    files, video, left, right, n = _Files(), [[] for _ in parts], [], [], [0]

    def new_id():
        n[0] += 1
        return "clipitem-%d" % n[0]

    for k, item in enumerate(pieces):
        rate = _rate(item["fps"])
        start, end, first, last, tin, tout = _place(item, length)
        if end <= start:
            continue
        timing = ("<enabled>TRUE</enabled><duration>%d</duration>%s<start>%d</start><end>%d</end><in>%d</in>"
                  "<out>%d</out><pproTicksIn>%d</pproTicksIn><pproTicksOut>%d</pproTicksOut>" % (
                      _frames(item["seconds"], _real(rate)), _rate_xml(rate), start, end, first, last, tin, tout))

        def common(fid):
            return "<masterclipid>master%s</masterclipid><name>%s</name>%s" % (fid, escape(item["name"]), timing)

        ids = [new_id() for _ in parts]
        sound = [new_id(), new_id()] if item["sound"] else []
        links = [(ids[0], "video", 1, len(video[0]) + 1, None)] + [
            (cid, "audio", t + 1, len(track) + 1, 1) for t, (cid, track) in enumerate(zip(sound, (left, right)))]
        link_xml = "".join("<link><linkclipref>%s</linkclipref><mediatype>%s</mediatype><trackindex>%d</trackindex>"
                           "<clipindex>%d</clipindex>%s</link>" % (cid, kind, track, index, (
                               "<groupindex>%d</groupindex>" % group if group else ""))
                           for cid, kind, track, index, group in links)
        for p, part in enumerate(parts):
            file_xml, fid = files.xml(item, rate)
            video[p].append('<clipitem id="%s">%s<alphatype>none</alphatype><pixelaspectratio>square'
                            '</pixelaspectratio><anamorphic>FALSE</anamorphic>%s%s%s</clipitem>' % (
                                ids[p], common(fid), file_xml, _filters(*motion(item, part)),
                                link_xml if p == 0 else ""))
        for cid, track, channel in zip(sound, (left, right), (1, 2)):
            file_xml, fid = files.xml(item, rate)
            track.append('<clipitem id="%s" premiereChannelType="stereo">%s%s<sourcetrack><mediatype>audio'
                         '</mediatype><trackindex>%d</trackindex></sourcetrack>%s</clipitem>' % (
                             cid, common(fid), file_xml, channel, link_xml))
    music = _music(state, length, files, new_id, 3 if left else 1)
    tracks = ["<track>%s<enabled>TRUE</enabled><locked>FALSE</locked></track>" % "".join(t) for t in video]
    audio = ['<track premiereTrackType="Stereo" currentExplodedTrackIndex="%d" totalExplodedTrackCount="2">%s'
             '<enabled>TRUE</enabled><locked>FALSE</locked><outputchannelindex>%d</outputchannelindex></track>'
             % (i, "".join(t), i + 1) for pair in ([left, right], music) if pair[0] for i, t in enumerate(pair)]
    fmt = ("<format><samplecharacteristics>%s<width>%d</width><height>%d</height><anamorphic>FALSE</anamorphic>"
           "<pixelaspectratio>square</pixelaspectratio><fielddominance>none</fielddominance><colordepth>24"
           "</colordepth></samplecharacteristics></format>" % (_rate_xml((FPS, False)), W, H))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4"><sequence id="sequence-1" '
            'explodedTracks="true"><name>%s</name><duration>%d</duration>%s<media><video>%s%s</video><audio>'
            '<numOutputChannels>2</numOutputChannels><format><samplecharacteristics><depth>16</depth><samplerate>48000'
            '</samplerate></samplecharacteristics></format>%s</audio></media><timecode>%s<string>00:00:00:00</string>'
            '<frame>0</frame><displayformat>NDF</displayformat></timecode>%s</sequence></xmeml>\n' % (
                escape(title), length, _rate_xml((FPS, False)), fmt, "".join(tracks), "".join(audio),
                _rate_xml((FPS, False)), "".join(_markers(pieces, state, length))))


def _music(state, length, files, new_id, track):
    """The two clipitems lists (left, right) of the music, on tracks track and track + 1, or ([], [])."""
    music = state.get("music")
    if not music or not music.get("path") or not os.path.isfile(music["path"]):
        return [], []
    item = {"path": music["path"], "name": os.path.basename(music["path"]), "width": 0, "height": 0, "fps": FPS,
            "seconds": max(state["length"] + music.get("from", 0), 1), "sound": True}
    rate = (FPS, False)
    first = _frames(music.get("from", 0))
    ids = [new_id(), new_id()]
    out = []
    for cid, channel in zip(ids, (1, 2)):
        file_xml, fid = files.xml(item, rate)
        links = "".join("<link><linkclipref>%s</linkclipref><mediatype>audio</mediatype><trackindex>%d</trackindex>"
                        "<clipindex>1</clipindex><groupindex>1</groupindex></link>" % (c, track + i)
                        for i, c in enumerate(ids))
        out.append(['<clipitem id="%s" premiereChannelType="stereo"><name>%s</name><enabled>TRUE</enabled>%s'
                    '<start>0</start><end>%d</end><in>%d</in><out>%d</out>%s<sourcetrack><mediatype>audio'
                    '</mediatype><trackindex>%d</trackindex></sourcetrack>%s</clipitem>' % (
                        cid, escape(item["name"]), _rate_xml(rate), length, first, first + length, file_xml, channel,
                        links)])
    return out[0], out[1]


def _markers(pieces, state, length):
    out = []
    for item in pieces:
        if item.get("why"):
            out.append((_frames(item["at"]), -1, item["why"], "Why this piece: " + item["name"]))
    for t in state.get("titles") or []:
        out.append((_frames(t["start"]), min(_frames(t["end"]), length), t["text"], "Title on the clip"))
    return ["<marker><comment>%s</comment><name>%s</name><in>%d</in><out>%d</out></marker>" % (
        escape(comment), escape(name), at, end) for at, end, name, comment in sorted(out, key=lambda m: m[0])]


def _clock(seconds):
    ms = int(round(seconds * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, ms % 1000)


def srt(state):
    """The captions as SubRip text, in seconds of the clip, or "" when the clip has none."""
    lines = (state.get("captions") or {}).get("lines") or []
    return "".join("%d\n%s --> %s\n%s\n\n" % (k, _clock(ln["from"]), _clock(ln["to"]), ln["text"])
                   for k, ln in enumerate(lines, 1))


def _left_out(state, pieces, rec=None):
    used = recording.tracks(dict(rec or {}, tracks=state.get("tracks")))
    have = {"tracks": used not in ([], [0]), "blurs": state.get("blurs"), "look": state.get("look"),
            "stickers": state.get("stickers") or state.get("glass") or state.get("flurries"),
            "hits": state.get("hits") or state.get("sounds"), "mutes": state.get("mutes"),
            "titles": state.get("titles"), "circle": any(p["name"] in ("circle", "card") for p in
                                                         state["clip"]["layout"]["parts"]),
            "background": state["clip"]["layout"].get("background") or any(i["whole"] for i in pieces),
            "captions": (state.get("captions") or {}).get("lines"), "channel": state.get("channel")}
    lines = ["- " + said.format(look=state.get("look")) for key, said in LEFT_OUT if have[key]]
    return "\n".join(lines) or "- Nothing: everything the clip has came over."


def safe_name(title):
    name = re.sub(r"[\\/:*?\"<>|\s]+", "-", title or "").strip("-.")[:60]
    return name or "clip"


def bundle(title, rec, state):
    """(file name, bytes) of a zip with <name>.xml, <name>.srt when the clip has captions, and README.txt."""
    name = safe_name(title)
    xml = sequence(title or name, rec, state)
    note = NOTE.format(name=name, title=title or name, left_out=_left_out(state, items(rec, state), rec))
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("%s/%s.xml" % (name, name), xml)
        captions = srt(state)
        if captions:
            z.writestr("%s/%s.srt" % (name, name), captions)
        z.writestr("%s/README.txt" % name, note)
    return name + "-premiere.zip", data.getvalue()
