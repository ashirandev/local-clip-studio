"""Words on screen: the house styles, titles, big words and captions, drawn by libass on the 1080x1920 frame.

Sizes are never guessed. measure() draws the text with FFmpeg's ass filter, white on black, and reads back the box
libass really drew, outline and shadow included (Kanit draws at about 0.62 of a letters-times-size guess, and a
Pillow size is not a libass size). That box is what lets the program tell Muse where a title fits clear of a face.
One FFmpeg run draws every candidate line at once, and every box is kept, so a call takes a few tens of ms.

Text wraps at word breaks only: Thai through pythainlp's newmm, a number (1,990, เจ็ดสิบห้า) is never split, and a
particle (ครับ, นะ...) stays with the word before it.

The house font is Google Sans (SIL OFL 1.1, fonts/GoogleSans-OFL.txt) in three weights, shipped in fonts/ with Kanit
(SIL OFL 1.1, fonts/OFL.txt), the font the sizes were tuned on phones with. A size means how tall the letters stand,
whatever the font: a font is drawn at the size that makes its letters as tall as the house font's, so Kanit, which
draws its Thai 1.49 times taller than Google Sans at the same size, is drawn smaller. A person may choose any other
font installed on the computer: fonts() lists them with the licence each font's own file names, so a free font
can be told from one that is not. libass is given a folder holding only the chosen fonts' files (fonts_folder()), so
the face that was measured is the face that is drawn: found through the computer's fonts instead, Sarabun Bold was
drawn as another font (measured 10-27).

The house styles are the looks people pick from (cards() names each in English and Thai, with a sample): clean, pop
and sticker in Google Sans, glow (Pattaya, SIL OFL 1.1, fonts/Pattaya-OFL.txt: script letters lit by a soft glow,
drawn as its own layer under letters that keep a dark outline, so they read on any picture), 3d (white letters with an
orange rim and a hard deep-orange copy below them) and classic (Taviraj italic, SIL OFL 1.1, fonts/Taviraj-OFL.txt).

Muse picks how big the words are from a scale of 1 to 5 (SIZES), for every part or for each part alone: 3 is the
sizes tuned on phones and each step is SIZE_STEP times the one before, so a step stands the letters as tall in every
font, and the lines and the outline grow with them.

Times are seconds of the clip. A start or end is written between two frames (30 a second), so the first frame at
or after start shows the words and the frame before it does not, whatever rounding libass does on its milliseconds.
"""
import collections
import functools
import hashlib
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unicodedata

from . import layouts, recording, thai
from .effects import Fragment

FRAME_W, FRAME_H, FPS = 1080, 1920, 30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = os.path.join(ROOT, "fonts")
KINDS = ("title", "caption", "big")
PARTICLES = ("ครับ", "ค่ะ", "คะ", "นะ", "จ้ะ", "จ้า", "เลย")
PARTICLE_ROOM = 2
THAI = re.compile("[\u0e00-\u0e7f]")
DRAWN = 32
RENDER_SECONDS = 60
MOST_CANVAS = 8192
FRAME_ROWS = 4096
PAIR_GAP = 1.0
# A line never starts with a closer: the repeat mark and the shortening mark go with the word before them (EP 0's
# captions, 10-30, opened a line on the repeat mark).
OPENERS, CLOSERS = "([{\"'“‘", ")]}\"'”’,.!?%:;…ๆฯ"

Style = collections.namedtuple("Style", "name fontname file colour key edge box edge_ratio shadow_ratio sizes note "
                                        "scale size shade glow", defaults=(1.0, 3, None, None))
Style.__doc__ = """A way to draw words. colour is the text, key the key word (and the lit word in captions), edge
the outline, or the box behind the text when box is True. sizes holds the title, caption and big-word sizes in
pixels, and scale how much bigger they are than the size Kanit draws the same letters at: lines stand size / scale
apart and the outline and shadow are edge_ratio and shadow_ratio of size / scale, so every font keeps the spacing
and the outline the styles were tuned with. size is the step on the size scale (SIZES) the sizes were drawn at.
shade is the colour of a hard shadow (None: a soft black one), and glow the colour of a glow drawn under the letters
(None: no glow). note says what was changed from what was asked."""
Measure = collections.namedtuple("Measure", "lines w h")
_Layout = collections.namedtuple("_Layout", "lines w h left top")
_Face = collections.namedtuple("_Face", "path index names full sub family weight italic licence postscript")
# (licence, what its text says) for the licences that let anyone use a font, in a video for sale too
FREE_LICENCES = (("SIL Open Font License", ("open font licen", "scripts.sil.org/ofl", "openfontlicense.org")),
                 ("Apache License", ("apache licen", "apache.org/licenses")),
                 ("Ubuntu Font Licence", ("ubuntu font licen",)),
                 ("GPL with the font exception", ("font exception",)))

# box: False for an outline, True for a box under each line, or how opaque the box is. The news look's box lets
# 0.4 of the picture through, so its white words stand out 5.7:1 or more over any picture, a white one too (Tarm,
# 10-30, on a news clip's subtitles: just right for clips of people talking); Google Sans Regular stands as tall as
# SemiBold.
NEWS_BOX = 0.6
# Lines of a see-through box (a title's rows, a caption page's two lines) stand a box's height and BOX_GAP apart:
# libass gives each line its own box, the font's whole line tall, and two that met drew darker there (Tarm, 10-30: it
# darkens and is not pretty).
BOX_GAP = 6
# tuned: the sizes tuned on phones in Kanit (SemiBold, Black and Bold). Google Sans draws its Thai letters 1.477
# (SemiBold), 1.500 and 1.488 (Bold) times smaller at the same size (measured 10-27): scale, so its sizes are the
# tuned ones times that and the words stand as tall as they did. Pattaya draws them 1.21 times taller than Kanit Bold
# and Taviraj SemiBold Italic 1.075 times smaller (measured 10-27, กขคบ at 200 px: 87, 67 and 72 px).
_HOUSE = {
    "clean": dict(face="Google Sans SemiBold", colour="#FFFFFF", key="#FFD60A", edge="#141414", box=False,
                  edge_ratio=0.07, shadow_ratio=0.0, tuned=(84, 64, 180), scale=1.477),
    "pop": dict(face="Google Sans Bold", colour="#FFFFFF", key="#FFD400", edge="#101010", box=False,
                edge_ratio=0.10, shadow_ratio=0.04, tuned=(92, 68, 200), scale=1.500),
    "sticker": dict(face="Google Sans Bold", colour="#16161A", key="#C8102E", edge="#FFFFFF", box=True,
                    edge_ratio=0.20, shadow_ratio=0.0, tuned=(80, 60, 170), scale=1.488),
    "glow": dict(face="Pattaya Regular", colour="#FFF4E8", key="#FFD27A", edge="#3B1600", box=False,
                 edge_ratio=0.035, shadow_ratio=0.0, tuned=(96, 72, 210), scale=72 / 87, glow="#FF6A1A"),
    "3d": dict(face="Google Sans Bold", colour="#FFFFFF", key="#FFF7C2", edge="#C2410C", box=False,
               edge_ratio=0.035, shadow_ratio=0.075, tuned=(92, 68, 200), scale=1.500, shade="#7C2D12"),
    "classic": dict(face="Taviraj SemiBold Italic", colour="#FFFFFF", key="#FFD60A", edge="#141414", box=False,
                    edge_ratio=0.06, shadow_ratio=0.03, tuned=(88, 66, 190), scale=72 / 67),
    "news": dict(face="Google Sans Regular", colour="#FFFFFF", key="#FFD60A", edge="#000000", box=NEWS_BOX,
                 edge_ratio=0.14, shadow_ratio=0.0, tuned=(88, 64, 180), scale=1.477),
}
# What each look is, for the people and for Muse choosing one: (English, Thai).
CARDS = {
    "clean": ("Clean and easy to read, with a thin dark outline.", "เรียบ อ่านง่าย ขอบเข้มบาง ๆ"),
    "pop": ("Bold, with a black outline and a shadow: stands out on game footage.", "ตัวหนา ขอบดำ มีเงา เด่นบนคลิปเกม"),
    "sticker": ("Words on a white box, like a sticker.", "ตัวหนังสือบนกล่องขาว แบบป้ายสติกเกอร์"),
    "glow": ("Script letters with a warm orange glow: for the one word that matters.",
             "ตัวเขียนเรืองแสงส้ม ใช้เน้นคำสำคัญคำเดียว"),
    "3d": ("Heavy white letters with an orange edge that stands out like 3D.", "ตัวหนาสีขาว ขอบส้มนูนแบบสามมิติ"),
    "classic": ("Elegant serif italic: for short linking words beside a bold one.",
                "ตัวมีเชิงเอียง ดูหรู ใช้กับคำเชื่อมสั้น ๆ ข้างคำตัวหนา"),
    "news": ("White words on a see-through dark box, like a news clip's subtitles: for people talking.",
             "ตัวขาวบนกล่องดำโปร่ง แบบซับข่าว เหมาะกับคลิปคนพูด"),
}
SAMPLE = "ลดราคา 50%"
CARD_GROUND = "geq=lum='42-14*Y/H':cb=132:cr=124"
CARD_SIZE = (540, 200)
CARD_FILL, CARD_TALL = 0.7, 0.85
# The colours of a look a person can choose: the letters, the key word, the outline (or the box), the glow and the
# hard shadow.
PAINTS = ("text", "key", "edge", "glow", "shade")
# The glow: a border GLOW_WIDTH of the letters' size wide in the glow colour at GLOW_ALPHA (ASS alpha, 0 is solid),
# blurred by GLOW_BLUR of the size.
GLOW_WIDTH, GLOW_BLUR, GLOW_ALPHA = 0.16, 0.12, 0x50
TALL_THAI, TALL_LATIN = "กขคบ", "xnu"
HOUSE_NAMES = tuple(_HOUSE)
# The size scale: TUNED_SIZE is the sizes above, and each step is SIZE_STEP times the one before (from 1 to 5: 0.76,
# 0.87, 1, 1.15 and 1.32 times the tuned sizes).
SIZES = (1, 2, 3, 4, 5)
TUNED_SIZE = 3
SIZE_STEP = 1.15


# ---- fonts -------------------------------------------------------------------------------------------------------

def _faces(path):
    """[(index, names, full, sub, family, weight, italic, licence, postscript)] for each face in a font file: names
    holds the lower-case family, full and PostScript names, weight is the OS/2 weight (400 regular, 700 bold) and
    licence is (licence, free) from _licence(). A file that is not a font gives []."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if head[:4] == b"ttcf":
                count = struct.unpack(">I", head[8:12])[0]
                offsets = struct.unpack(">%dI" % count, f.read(4 * count))
            else:
                offsets = (0,)
            faces = []
            for k, offset in enumerate(offsets):
                tables = _tables(f, offset)
                names = _names(f, tables.get(b"name"))
                full = names.get(4) or names.get(1) or os.path.splitext(os.path.basename(path))[0]
                every = {n.lower() for i, n in names.items() if i in (1, 4, 6, 16)}
                sub = (names.get(17) or names.get(2) or "").lower()
                weight, italic = _weight(f, tables.get(b"OS/2"), sub)
                faces.append((k, every, full, sub, names.get(16) or names.get(1) or full, weight, italic,
                              _licence(names), names.get(6, "")))
            return faces
    except (OSError, struct.error, ValueError):
        return []


def _weight(f, where, sub):
    """(weight, italic) from the OS/2 table, or from the subfamily name when the font has none."""
    if where and where[1] >= 64:
        f.seek(where[0])
        data = f.read(64)
        weight = struct.unpack(">H", data[4:6])[0]
        return (weight * 100 if 0 < weight < 10 else weight or 400), bool(struct.unpack(">H", data[62:64])[0] & 1)
    return (700 if "bold" in sub else 400), ("italic" in sub or "oblique" in sub)


def _licence(names):
    """(licence, free): the free licence the font's file names (name records 13 and 14), free True; else what the
    file says instead, cut short, or that it says nothing, free False."""
    said = " ".join(" ".join(names.get(i, "").split()) for i in (13, 14)).strip()
    low = said.lower()
    for licence, marks in FREE_LICENCES:
        if any(mark in low for mark in marks):
            return licence, True
    if not said:
        return "its file names no licence", False
    if len(said) > 120:
        said = said[:120].rsplit(" ", 1)[0] + "..."
    return "its file says \"%s\"" % said, False


def _tables(f, offset):
    f.seek(offset)
    count = struct.unpack(">4sH", f.read(6))[1]
    f.seek(offset + 12)
    tables = {}
    for _ in range(count):
        tag, _sum, at, length = struct.unpack(">4sIII", f.read(16))
        tables[tag] = (at, length)
    return tables


def _names(f, where):
    if not where:
        return {}
    at = where[0]
    f.seek(at)
    _format, count, strings = struct.unpack(">HHH", f.read(6))
    records = [struct.unpack(">HHHHHH", f.read(12)) for _ in range(count)]
    found = {}
    for platform, encoding, language, name_id, length, offset in records:
        if name_id not in (1, 2, 4, 6, 13, 14, 16, 17):
            continue
        f.seek(at + strings + offset)
        data = f.read(length)
        if platform in (0, 3):
            text = data.decode("utf-16-be", "replace")
        elif platform == 1 and encoding == 0:
            text = data.decode("mac_roman", "replace")
        else:
            continue
        english = platform == 1 or language in (0x409, 0)
        if name_id not in found or english:
            found[name_id] = text.strip("\x00 ")
    return found


def _has(path, index, chars):
    """True when the face has a glyph for every one of chars, read from its cmap."""
    chars = set(chars)
    with open(path, "rb") as f:
        head = f.read(12)
        offset = 0
        if head[:4] == b"ttcf":
            f.seek(12 + 4 * index)
            offset = struct.unpack(">I", f.read(4))[0]
        cmap = _tables(f, offset).get(b"cmap")
        if not cmap:
            return False
        f.seek(cmap[0])
        data = f.read(cmap[1])
    count = struct.unpack(">H", data[2:4])[0]
    subtables = [struct.unpack(">HHI", data[4 + 8 * k:12 + 8 * k]) for k in range(count)]
    for platform, encoding, at in sorted(subtables, key=lambda s: (s[0], s[1]) not in ((3, 10), (0, 4), (0, 6))):
        kind = struct.unpack(">H", data[at:at + 2])[0]
        if platform in (0, 3) and kind in (4, 12):
            return all(_glyph(data, at, kind, ord(c)) for c in chars)
    return False


def _glyph(data, at, kind, code):
    if kind == 12:
        groups = struct.unpack(">I", data[at + 12:at + 16])[0]
        for k in range(groups):
            start, end, first = struct.unpack(">III", data[at + 16 + 12 * k:at + 28 + 12 * k])
            if start <= code <= end:
                return first + code - start != 0
        return False
    if code > 0xFFFF:
        return False
    segments = struct.unpack(">H", data[at + 6:at + 8])[0] // 2
    ends = struct.unpack(">%dH" % segments, data[at + 14:at + 14 + 2 * segments])
    base = at + 16 + 2 * segments
    starts = struct.unpack(">%dH" % segments, data[base:base + 2 * segments])
    deltas = struct.unpack(">%dh" % segments, data[base + 2 * segments:base + 4 * segments])
    range_at = base + 4 * segments
    ranges = struct.unpack(">%dH" % segments, data[range_at:range_at + 2 * segments])
    for k in range(segments):
        if starts[k] <= code <= ends[k]:
            if ranges[k] == 0:
                return (code + deltas[k]) & 0xFFFF != 0
            where = range_at + 2 * k + ranges[k] + 2 * (code - starts[k])
            glyph = struct.unpack(">H", data[where:where + 2])[0]
            return glyph != 0 and (glyph + deltas[k]) & 0xFFFF != 0
    return False


@functools.lru_cache(maxsize=None)
def _has_all(path, index, chars):
    return _has(path, index, chars)


def _font_files(folder, deep=False):
    found = []
    if not os.path.isdir(folder):
        return found
    for top, dirs, files in os.walk(folder):
        found += [os.path.join(top, n) for n in sorted(files) if n.lower().endswith((".ttf", ".otf", ".ttc"))]
        if not deep:
            break
    return found


@functools.lru_cache(maxsize=None)
def _shipped():
    return [_Face(path, *face) for path in _font_files(FONTS) for face in _faces(path)]


def _installed():
    """The faces installed on this computer, read again when a fonts folder has changed, so a font installed while
    the studio runs is found."""
    home = os.path.expanduser("~")
    folders = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
               os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
               "/System/Library/Fonts", "/Library/Fonts", os.path.join(home, "Library", "Fonts"),
               "/usr/share/fonts", "/usr/local/share/fonts", os.path.join(home, ".fonts"),
               os.path.join(home, ".local", "share", "fonts")]
    return _installed_in(tuple((f, os.stat(f).st_mtime_ns) for f in folders if f and os.path.isdir(f)))


@functools.lru_cache(maxsize=4)
def _installed_in(folders):
    return [_Face(path, *face) for folder, _changed in folders for path in _font_files(folder, deep=True)
            for face in _faces(path)]


def shipped_fonts():
    """The full names of the faces in fonts/, such as "Kanit Bold"."""
    return [f.full for f in _shipped()]


def _find_font(name, weight=400):
    """(path, index, full name) of the face called name, in fonts/ first, then on this computer; None if neither.
    A face's full or PostScript name (or its file's name) gives that face; a family name gives the family's upright
    face nearest weight, so "Mitr" in the pop style (Kanit Black, 900) is Mitr Bold, Mitr's heaviest."""
    want = name.strip().lower()
    for faces in (_shipped(), _installed()):
        stem = (lambda f: os.path.splitext(os.path.basename(f.path))[0].lower())
        hits = [f for f in faces if want in f.names or want == stem(f)]
        if hits:
            hits.sort(key=lambda f: (want not in (f.full.lower(), stem(f)), f.italic, abs(f.weight - weight), f.full))
            return hits[0].path, hits[0].index, hits[0].full
    return None


def _face_at(path, index):
    return next(f for f in _shipped() + _installed() if (f.path, f.index) == (path, index))


def fonts(letters=None):
    """The fonts a style can use, one entry per family, the studio's own first, then this computer's by name:
    [{"family", "faces": [full names, upright first, lightest first], "free": True when the font's file names a free
    licence (FREE_LICENCES), "licence": that licence, or what the file says instead, "where": "studio" or
    "computer"}]. With letters (such as "กขค"), only the faces that have every one of them."""
    if letters is not None and (not isinstance(letters, str) or not letters):
        raise ValueError("letters %r is not text: give the letters every face must have, such as 'กขค'." % (letters,))
    families, found = {}, []
    for where, faces in (("studio", _shipped()), ("computer", sorted(_installed(), key=lambda f: f.family.lower()))):
        for f in faces:
            entry = families.get(f.family.lower())
            if entry and entry["where"] != where or letters and not _has_all(f.path, f.index, letters):
                continue
            if not entry:
                entry = families[f.family.lower()] = {"family": f.family, "faces": set(), "free": f.licence[1],
                                                      "licence": f.licence[0], "where": where}
                found.append(entry)
            entry["faces"].add((f.italic, f.weight, f.full))
    return [dict(e, faces=[full for _italic, _weight, full in sorted(e["faces"])]) for e in found]


def fonts_folder(*styles):
    """The folder libass is given for the styles of one ASS file: a folder in work/fonts holding a copy of each
    style's font file and nothing else, made once, so libass draws exactly the faces that were chosen and measured,
    never another standing in for one. The studio's own faces are copied too: fonts/ also holds their licences, and
    libass tries every file in its folder as a font."""
    if not styles:
        raise ValueError("fonts_folder needs a style: pass every style the ASS file uses.")
    for st in styles:
        _check_style(st)
    return _folder_for(*sorted({os.path.abspath(st.file[0]) for st in styles}))


def _folder_for(*paths):
    key = hashlib.sha1()
    for path in paths:
        seen = os.stat(path)
        key.update(("%s|%d|%d\n" % (os.path.abspath(path), seen.st_size, seen.st_mtime_ns)).encode("utf-8"))
    folder = os.path.join(recording.WORK, "fonts", key.hexdigest()[:12])
    if not os.path.isdir(folder):
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        part = tempfile.mkdtemp(prefix=os.path.basename(folder) + ".", suffix=".part", dir=os.path.dirname(folder))
        try:
            for k, path in enumerate(paths):
                name = os.path.basename(path)
                if os.path.exists(os.path.join(part, name)):
                    name = "%d-%s" % (k, name)
                shutil.copyfile(path, os.path.join(part, name))
            os.replace(part, folder)
        except OSError:
            if not os.path.isdir(folder):
                raise
        finally:
            shutil.rmtree(part, ignore_errors=True)
    return folder


def _drawn_name(path, index):
    """The name to write in the ASS file so libass draws this very face from its folder: the first of its full,
    PostScript and family names that libass itself says it drew from this face, or None when none does. No one name
    works for every font: libass found Sao Chingcha Regular (an OpenType CFF font) by its PostScript and family
    names only, and Sarabun SemiBold by its full name only (measured 10-27)."""
    seen = os.stat(path)
    return _drawn_name_of(path, index, seen.st_size, seen.st_mtime_ns)


@functools.lru_cache(maxsize=None)
def _drawn_name_of(path, index, _size, _changed):
    face = _face_at(path, index)
    for name in dict.fromkeys(n for n in (face.full, face.postscript, face.family) if n):
        if _drawn_from(name, _folder_for(path)) == face.postscript.lower():
            return name
    return None


def _drawn_from(name, folder):
    """The PostScript name of the face libass draws for the font name, given the folder, read from libass's own
    report (FFmpeg's debug log); None when it does not say."""
    probe = os.path.join(tempfile.mkdtemp(prefix="lcs-font-"), "f.ass")
    try:
        with open(probe, "w", encoding="utf-8", newline="\n") as f:
            f.write(_header(320, 240) + _ass_style(STYLES["clean"]._replace(name="probe", fontname=name), "title")
                    + "\n" + EVENTS + "Dialogue: 0,0:00:00.00,0:00:01.00,probe_title,,0,0,0,,a\n")
        run = subprocess.run([_ffmpeg(), "-v", "debug", "-f", "lavfi", "-i", "color=black:s=320x240:r=1", "-vf",
                              _ass_filter(probe, folder), "-frames:v", "1", "-f", "null", "-"], capture_output=True,
                             timeout=RENDER_SECONDS)
    finally:
        shutil.rmtree(os.path.dirname(probe), ignore_errors=True)
    found = re.search(r"fontselect: \(.*?, \d+, \d+\) -> ([^,\r\n]+),", run.stderr.decode("utf-8", "replace"))
    return found.group(1).strip().lower() if found else None


# ---- styles ------------------------------------------------------------------------------------------------------

def _hex(value, field):
    if isinstance(value, str):
        v = value.strip()
        if re.fullmatch(r"#?[0-9A-Fa-f]{6}", v):
            return "#" + v.lstrip("#").upper()
        if re.fullmatch(r"#?[0-9A-Fa-f]{3}", v):
            return "#" + "".join(c * 2 for c in v.lstrip("#")).upper()
    raise ValueError("%s %r is not a colour: give it as #RRGGBB, such as #FFD400." % (field, value))


def ratio(a, b):
    """The WCAG contrast ratio of two #RRGGBB colours, from 1 to 21."""
    def light(c):
        parts = [int(c[k:k + 2], 16) / 255 for k in (1, 3, 5)]
        lin = [p / 12.92 if p <= 0.04045 else ((p + 0.055) / 1.055) ** 2.4 for p in parts]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    hi, lo = sorted((light(a), light(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def style(name=None, font=None, colour=None, size=TUNED_SIZE, colours=None):
    """A house style (one of HOUSE_NAMES; clean when name is None), with the person's font and text colour when they
    asked for them, at size, a step on the size scale from 1 (smallest) to 5 (biggest), with colours ({name: #RRGGBB},
    names from PAINTS) for any of its colours (text there is the same as colour; a glow given to a look without one
    gives it one). A text or key colour that does not stand out 4.5:1 against the outline (or the box) the look chose
    gets a black or white outline, or the text's colour, that does: it is never refused, and note says what was
    changed. An outline or key the person chose is kept as asked, and note says it may be hard to read."""
    name = "clean" if name is None else name
    if name not in _HOUSE:
        raise ValueError("style %r is not a house style: use one of %s." % (name, ", ".join(HOUSE_NAMES)))
    if isinstance(size, bool) or not isinstance(size, int) or size not in SIZES:
        raise ValueError("size %r is not a size: use a whole number from %d (smallest) to %d (biggest), %d for the "
                         "size tuned on phones." % (size, SIZES[0], SIZES[-1], TUNED_SIZE))
    house = _HOUSE[name]
    face, scale = house["face"], house["scale"]
    sizes = tuple(round(size * scale) for size in house["tuned"])
    notes = []
    if font is not None:
        if not isinstance(font, str) or not font.strip():
            raise ValueError("font %r is not a font name: use one of %s, or a font installed on this computer."
                             % (font, ", ".join(shipped_fonts())))
        found = _find_font(font, next((f.weight for f in _shipped() if f.full == face), 400))
        if not found:
            raise ValueError("font %r is neither in the studio's fonts (%s) nor on this computer: use one of those."
                             % (font, ", ".join(shipped_fonts())))
        path, index, full = found
        face = _drawn_name(path, index)
        if not face:
            raise ValueError("font %r cannot be drawn: libass draws another font for every one of %s's names, so "
                             "choose another font." % (font, full))
        house_path, house_index, _house = _find_font(house["face"])
        letters = TALL_THAI if _has_thai((path, index)) else TALL_LATIN
        taller = _tall(house_path, house_index, house["face"], letters) / _tall(path, index, face, letters)
        sizes, scale = tuple(max(8, round(size * taller)) for size in sizes), scale * taller
        licence, free = _face_at(path, index).licence
        if not free:
            notes.append("%s is not known to be free: %s. Use it only when its licence lets you show it in a video "
                         "you post." % (full, licence))
    else:
        found = _find_font(face)
        if not found:
            raise RuntimeError("fonts/ has no %s, so the %s style cannot be drawn: put the font files back in %s."
                               % (face, name, FONTS))
        path, index, _face = found
    if size != TUNED_SIZE:
        sizes = tuple(max(8, round(s * SIZE_STEP ** (size - TUNED_SIZE))) for s in sizes)
    colours = {} if colours is None else colours
    if not isinstance(colours, dict) or set(colours) - set(PAINTS):
        raise ValueError("colours %r is not a colour for each part of a look: give {name: #RRGGBB} with the names %s."
                         % (colours, ", ".join(PAINTS)))
    picked = {k: _hex(v, k) for k, v in colours.items() if v is not None}
    if colour is not None:
        picked.setdefault("text", _hex(colour, "colour"))
    text = picked.get("text", house["colour"])
    key, edge = picked.get("key", house["key"]), picked.get("edge", house["edge"])
    glow, shade = picked.get("glow", house.get("glow")), picked.get("shade", house.get("shade"))
    what = "box" if house["box"] else "outline"
    if ratio(text, edge) < 4.5 and "edge" in picked:
        notes.append("text %s on the %s %s stands out only %.1f:1, below 4.5:1, so it may be hard to read: kept as "
                     "asked." % (text, what, edge, ratio(text, edge)))
    elif ratio(text, edge) < 4.5:
        better = max(("#000000", "#FFFFFF"), key=lambda e: ratio(text, e))
        notes.append("%s changed from %s to %s: %s on %s is %.1f:1, below 4.5:1." % (
            what, edge, better, text, edge, ratio(text, edge)))
        edge = better
    if ratio(key, edge) < 4.5 and "key" in picked:
        notes.append("key colour %s on the %s %s stands out only %.1f:1, below 4.5:1, so it may be hard to read: kept "
                     "as asked." % (key, what, edge, ratio(key, edge)))
    elif ratio(key, edge) < 4.5:
        notes.append("key colour changed from %s to %s: it stood out only %.1f:1 on the %s %s." % (
            key, text, ratio(key, edge), what, edge))
        key = text
    if "shade" in picked and not house["shadow_ratio"]:
        notes.append("the %s look has no shadow, so shade %s is not drawn: use a look with a shadow, such as pop or "
                     "3d." % (name, shade))
    label = name
    painted = (text, key, edge, glow, shade)
    if face != house["face"] or size != TUNED_SIZE or painted != (house["colour"], house["key"], house["edge"],
                                                                     house.get("glow"), house.get("shade")):
        seed = "%s|%s|%s" % (face, path, "|".join(str(p) for p in painted)) + (
            "" if size == TUNED_SIZE else "|%d" % size)
        label = "%s~%s" % (name, hashlib.sha1(seed.encode("utf-8")).hexdigest()[:6])
    return Style(label, face, (path, index), text, key, edge, house["box"], house["edge_ratio"],
                 house["shadow_ratio"], sizes, " ".join(notes), scale, size, shade, glow)


@functools.lru_cache(maxsize=None)
def _tall(path, index, fontname, letters):
    """How tall the letters stand in the face drawn at 200 px with no outline: what a size means in that font."""
    flat = Style("tall~" + hashlib.sha1(("%s|%d|%s" % (path, index, fontname)).encode("utf-8")).hexdigest()[:6],
                 fontname, (path, index), "#FFFFFF", "#FFFFFF", "#000000", False, 0.0, 0.0, (200, 200, 200), "")
    return measure(letters, flat, "big", 4000).h


STYLES = {n: style(n) for n in HOUSE_NAMES}


def look(name=None, font=None, colour=None, fonts=None, size=None, sizes=None, colours=None, part_colours=None):
    """The style each part of the words is drawn in, {"title": Style, "caption": Style, "big": Style}: the house
    style name with font, size and colours ({name: #RRGGBB}, see style()) for every part, and fonts ({part: font}),
    sizes ({part: size}) and part_colours ({part: {name: #RRGGBB}}, over colours) for the parts given their own, so
    the title, the captions and the big words can each be changed alone."""
    part_colours = {} if part_colours is None else part_colours
    if (not isinstance(part_colours, dict) or set(part_colours) - set(KINDS)
            or not all(v is None or isinstance(v, dict) for v in part_colours.values())):
        raise ValueError("part_colours %r is not the colours of each part: give {part: {name: #RRGGBB}} with the parts "
                         "%s." % (part_colours, ", ".join(KINDS)))
    if colours is not None and not isinstance(colours, dict):
        raise ValueError("colours %r is not a colour for each part of a look: give {name: #RRGGBB} with the names %s."
                         % (colours, ", ".join(PAINTS)))
    fonts = {} if fonts is None else fonts
    if not isinstance(fonts, dict) or set(fonts) - set(KINDS):
        raise ValueError("fonts %r is not a font for each part: give {part: font} with the parts %s."
                         % (fonts, ", ".join(KINDS)))
    sizes = {} if sizes is None else sizes
    if not isinstance(sizes, dict) or set(sizes) - set(KINDS):
        raise ValueError("sizes %r is not a size for each part: give {part: size} with the parts %s and sizes from "
                         "%d to %d." % (sizes, ", ".join(KINDS), SIZES[0], SIZES[-1]))
    size = TUNED_SIZE if size is None else size
    return {kind: style(name, fonts.get(kind) or font, colour, size if sizes.get(kind) is None else sizes[kind],
                        dict(colours or {}, **(part_colours.get(kind) or {})))
            for kind in KINDS}


def _size(st, kind):
    if kind not in KINDS:
        raise ValueError("kind %r is not a kind of words: use one of %s." % (kind, ", ".join(KINDS)))
    return st.sizes[KINDS.index(kind)]


def _smaller(st):
    """What a refusal adds for words that would fit smaller: the size below the style's, when there is one."""
    return ", or size %d" % (st.size - 1) if st.size > SIZES[0] else ""


def _check_style(st):
    if not isinstance(st, Style):
        raise ValueError("style %r is not a style: get one from style(), such as style('pop')." % (st,))


def _check_text(text, st, field="text"):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("%s %r is empty: give the words to draw." % (field, text))
    bad = sorted(set(text) & set("{}\\"))
    if bad:
        raise ValueError("%s %r has %s, which libass reads as a command: leave out { } and \\."
                         % (field, text, " ".join(bad)))
    thai_letters = set(THAI.findall(text))
    if thai_letters and not _has_thai(st.file):
        raise ValueError("font %r has no Thai letters, so %r cannot be drawn: use a Thai font such as %s."
                         % (st.fontname, text, shipped_fonts()[0] if shipped_fonts() else "Kanit"))


@functools.lru_cache(maxsize=None)
def _has_thai(file):
    path, index = file
    return _has(path, index, "กขคงจฉชซญดตถทนบปผพฟมยรลวศสหอฮะาำิีึืุูเแโใไ่้๊๋็์")


def _colour(hexcolour, alpha=0):
    return "&H%02X%s%s%s" % (alpha, hexcolour[5:7], hexcolour[3:5], hexcolour[1:3])


def _ass_style(st, kind, mask=False):
    size = _size(st, kind)
    edge = round(size / st.scale * st.edge_ratio, 1)
    shadow = round(size / st.scale * st.shadow_ratio, 1)
    if mask:
        first = second = outline = back = "&H00FFFFFF"
    else:
        first = _colour(st.key if kind == "caption" else st.colour)
        second = _colour(st.colour)
        through = round(255 * (1 - st.box)) if st.box else 0
        outline = _colour(st.edge, through)
        back = _colour(st.edge, through) if st.box else _colour(st.shade) if st.shade else _colour("#000000", 0x80)
    return ("Style: %s_%s,%s,%d,%s,%s,%s,%s,0,0,0,0,100,100,0,0,%d,%s,%s,8,0,0,0,1"
            % (st.name, kind, st.fontname, size, first, second, outline, back, 3 if st.box else 1, edge, shadow))


def _glow_style(st, kind):
    """The ASS style of the glow under a glow look's letters: no fill, a wide border in the glow colour."""
    size = _size(st, kind)
    return ("Style: %s_%s_glow,%s,%d,&HFF000000,&HFF000000,%s,&HFF000000,0,0,0,0,100,100,0,0,1,%s,0,8,0,0,0,1"
            % (st.name, kind, st.fontname, size, _colour(st.glow, GLOW_ALPHA), round(size / st.scale * GLOW_WIDTH, 1)))


def _glow_tags(st, kind):
    """The override tags that make a line of a glow look's letters its glow: fill hidden, border blurred; "" for a
    look with no glow."""
    if not st.glow:
        return ""
    return "\\1a&HFF&\\2a&HFF&\\blur%s" % round(_size(st, kind) / st.scale * GLOW_BLUR, 1)


def _header(w, h):
    return ("[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\nWrapStyle: 2\n"
            "ScaledBorderAndShadow: yes\nYCbCr Matrix: None\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding\n" % (w, h))


EVENTS = "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"


# ---- drawing and reading back ------------------------------------------------------------------------------------

_boxes = {}
_ON = bytes(0 if v < DRAWN else 1 for v in range(256))


def _ffmpeg():
    tool = shutil.which("ffmpeg")
    if not tool:
        raise RuntimeError("FFmpeg is not installed, so words cannot be measured or drawn.")
    return tool


def _draw_boxes(st, kind, texts, align=8):
    """{text: (left, top, right, bottom) of what libass draws, from the anchor point}, every text drawn on its own
    in one FFmpeg run. texts are ASS text (\\N between lines). A text that draws nothing gives None."""
    todo = [t for t in dict.fromkeys(texts) if (st, kind, t, align) not in _boxes]
    size = _size(st, kind)
    pad = int(size * (st.edge_ratio + st.shadow_ratio) * 2) + 8
    width = max([256] + [int(len(t) * size * 0.9) + 2 * pad + 64 for t in todo])
    room = 2.2
    while todo:
        width = min(MOST_CANVAS, (width + 15) // 16 * 16)
        frames, used = [[]], 0
        for t in todo:
            tall = int(size * (room + 1.6 * t.count("\\N"))) + 2 * pad + 16
            if used + tall > FRAME_ROWS and frames[-1]:
                frames.append([])
                used = 0
            frames[-1].append((t, used, tall))
            used += tall
        rows = (max(sum(b[2] for b in f) for f in frames) + 15) // 16 * 16
        events, bands = [], []
        for k, frame in enumerate(frames):
            for t, top, tall in frame:
                ay = top + tall // 2 if align == 5 else top + pad + int(size * 0.3)
                events.append("Dialogue: 0,%s,%s,%s_%s,,0,0,0,,{\\an%d\\pos(%d,%d)}%s" % (
                    _clock(max(0.0, k - 0.5)), _clock(k + 0.5), st.name, kind, align, width // 2, ay, t))
                bands.append((k, t, top, tall, width // 2, ay))
        found, clipped = _render_and_read(st, kind, width, rows, len(frames), events, bands)
        for t, box in found.items():
            if t not in clipped:
                _boxes[(st, kind, t, align)] = box
        todo = [t for t in todo if t in clipped]
        if todo and width >= MOST_CANVAS:
            raise ValueError("%r is wider than %d px at %s size %d: use fewer words." % (todo[0], MOST_CANVAS, kind,
                                                                                        size))
        width, room = width * 2, room * 2
    return {t: _boxes[(st, kind, t, align)] for t in texts}


def _render_and_read(st, kind, width, rows, count, events, bands):
    """Draws the events white on black and reads each band's box back: ({text: box from its anchor}, the texts
    whose drawing touched the edge of their band, to be drawn again with more room)."""
    folder = tempfile.mkdtemp(prefix="lcs-measure-")
    try:
        path = os.path.join(folder, "m.ass")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_header(width, rows) + _ass_style(st, kind, mask=True) + "\n" + EVENTS + "\n".join(events) + "\n")
        run = subprocess.run([_ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "color=black:s=%dx%d:r=1" % (width, rows),
                              "-vf", "format=gray," + _ass_filter(path, fonts_folder(st)), "-frames:v", str(count),
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, timeout=RENDER_SECONDS)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    raw = run.stdout
    if run.returncode != 0 or len(raw) != width * rows * count:
        raise RuntimeError("FFmpeg could not draw the words (%s)." % run.stderr.decode("utf-8", "replace").strip()[-300:])
    found, clipped = {}, set()
    for k, t, top, tall, ax, ay in bands:
        start = k * width * rows + top * width
        band = raw[start:start + tall * width].translate(_ON)
        left = right = first = last = None
        for y in range(tall):
            row = band[y * width:(y + 1) * width]
            stripped = row.lstrip(b"\x00")
            if not stripped:
                continue
            x0 = width - len(stripped)
            x1 = len(row.rstrip(b"\x00"))
            left = x0 if left is None else min(left, x0)
            right = x1 if right is None else max(right, x1)
            first = y if first is None else first
            last = y + 1
        if left is None:
            found[t] = None
            continue
        if left == 0 or right == width or first == 0 or last == tall:
            clipped.add(t)
        found[t] = (left - ax, top + first - ay, right - ax, top + last - ay)
    return found, clipped


def _ass_filter(path, fonts_dir):
    return "ass=filename=%s:fontsdir=%s:shaping=complex" % (_filter_value(path), _filter_value(fonts_dir))


def _filter_value(path):
    """path written so FFmpeg's two levels of unescaping (the graph, then the filter's options) give it back
    exactly: drive letters, spaces, commas and quotes included."""
    value = os.path.abspath(path).replace("\\", "/")
    value = re.sub(r"([\\':])", r"\\\1", value)
    return re.sub(r"([\\'\[\],;])", r"\\\1", value)


# ---- wrapping ----------------------------------------------------------------------------------------------------

def _tokens(text):
    if THAI.search(text):
        from pythainlp.tokenize import word_tokenize
        return word_tokenize(text, engine="newmm", keep_whitespace=True)
    return re.findall(r"\s+|[^\s]+", text)


def _atoms(text):
    """[(start, end)] of the pieces of text between which a line may break; each piece keeps the spaces after it,
    so the pieces joined are the text exactly."""
    tokens, at = [], 0
    for token in _tokens(text):
        tokens.append((at, at + len(token)))
        at += len(token)
    if "".join(text[a:b] for a, b in tokens) != text:
        tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+\s*|\s+", text)]
    locked = set()
    for m in thai.NUMBER.finditer(text):
        locked.update(range(m.start() + 1, m.end()))
    atoms = []
    for a, b in tokens:
        piece = text[a:b]
        if not atoms:
            atoms.append([a, b])
            continue
        before = text[atoms[-1][0]:atoms[-1][1]]
        glue = (piece.isspace() or a in locked or piece.strip() in PARTICLES or piece[0] in CLOSERS
                or unicodedata.category(piece[0]).startswith("M") or before.isspace()
                or (before and before[-1] in OPENERS)
                or (not before[-1].isspace() and not THAI.search(before[-1] + piece[0])))
        if glue:
            atoms[-1][1] = b
        else:
            atoms.append([a, b])
    return [tuple(x) for x in atoms]


def _layout_of(lines, boxes, lh):
    """The box of lines drawn one under the other, centred on the anchor, lh apart, from each line's own box."""
    edges = [boxes[line] or (0, 0, 0, 0) for line in lines]
    left = min(e[0] for e in edges)
    right = max(e[2] for e in edges)
    top = min(e[1] + i * lh for i, e in enumerate(edges))
    bottom = max(e[3] + i * lh for i, e in enumerate(edges))
    return _Layout(tuple(lines), right - left, bottom - top, left, top)


def _line_height(st, kind):
    """How far apart lines stand: libass puts lines one font size apart, and a font's size covers its whole line
    box, which in Google Sans stands 1.5 times taller around its Thai letters than in Kanit; so lines are drawn as
    events of their own, size / scale apart, as far as Kanit's are (measured: libass's spacing is the size). The rows
    of a see-through box stand its height and BOX_GAP apart: libass draws each row's box the font's whole line tall."""
    if 0 < st.box < 1:
        _, top, _, bottom = _draw_boxes(st, kind, [TALL_THAI])[TALL_THAI]
        return bottom - top + BOX_GAP
    return round(_size(st, kind) / st.scale)


@functools.lru_cache(maxsize=2048)
def _wrap(text, st, kind, width):
    """The fewest lines that fit width, the widest of them as narrow as it can be; a piece wider than width on
    its own takes a line of its own (and the layout is wider than width)."""
    flat, forced = [], set()
    for p in text.split("\n"):
        flat += [p[a:b] for a, b in _atoms(p)] if p.strip() else []
        forced.add(len(flat))
    size = _size(st, kind)
    alone = _draw_boxes(st, kind, [a.strip() for a in flat])
    lh = _line_height(st, kind)
    gap = size * 0.3
    est = [(alone[a.strip()][2] - alone[a.strip()][0]) if alone[a.strip()] else 0 for a in flat]
    candidates = {}
    for i in range(len(flat)):
        total = 0
        for j in range(i + 1, len(flat) + 1):
            total += est[j - 1] + (gap if j - 1 > i else 0)
            if j - 1 > i and total > width * 1.15 + size:
                break
            candidates[(i, j)] = "".join(flat[i:j]).strip()
            if j in forced:
                break
    boxes = _draw_boxes(st, kind, list(candidates.values()))
    best = {0: (0, 0, ())}
    for j in range(1, len(flat) + 1):
        for i in range(j):
            if (i, j) not in candidates or i not in best or any(i < f < j for f in forced):
                continue
            box = boxes[candidates[(i, j)]]
            w = box[2] - box[0] if box else 0
            if w > width and j - i > 1:
                continue
            n, widest, lines = best[i]
            option = (n + 1, max(widest, w), lines + (candidates[(i, j)],))
            if j not in best or option[:2] < best[j][:2]:
                best[j] = option
    return _layout_of(best[len(flat)][2], boxes, lh)


def measure(text, style, kind, width):
    """The box libass draws for text in style at kind's size ("title", "caption" or "big"), wrapped at word
    breaks to fit width pixels: Measure(lines, w, h), w and h in pixels, outline and shadow included."""
    _check_style(style)
    _check_text(text, style)
    _size(style, kind)
    if not isinstance(width, (int, float)) or isinstance(width, bool) or not 0 < width <= MOST_CANVAS:
        raise ValueError("width %r is not a width: use pixels from 1 to %d." % (width, FRAME_W))
    lay = _wrap(text, style, kind, int(width))
    return Measure(list(lay.lines), lay.w, lay.h)


# ---- times -------------------------------------------------------------------------------------------------------

def _frame(t):
    """The first frame at or after t."""
    return max(0, math.ceil(t * FPS - 1e-6))


def _clock(seconds):
    cs = int(round(seconds * 100))
    return "%d:%02d:%02d.%02d" % (cs // 360000, cs // 6000 % 60, cs // 100 % 60, cs % 100)


def _edge(t):
    """t written halfway between the frame before t and the first frame at or after it, in centiseconds."""
    k = _frame(t)
    return 0 if k == 0 else int(round((k - 0.5) / FPS * 100))


def _times(start, end, field="the words"):
    for name, value in (("start", start), ("end", end)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value != value or value < 0:
            raise ValueError("%s %r is not a time: use seconds of the clip from 0." % (name, value))
    if end <= start:
        raise ValueError("end %s is not after start %s: use an end later than the start." % (end, start))
    a, b = _edge(start), _edge(end)
    if b <= a:
        raise ValueError("%s from %s to %s s shows on no frame: make it at least 1/%d s long." % (
            field, start, end, FPS))
    return a, b


def _cs_clock(cs):
    return "%d:%02d:%02d.%02d" % (cs // 360000, cs // 6000 % 60, cs // 100 % 60, cs % 100)


def _check_box(box, field="box"):
    if (not isinstance(box, (tuple, list)) or len(box) != 4
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)):
        raise ValueError("%s %r is not a box: give (x, y, w, h) in pixels of the %dx%d frame." % (
            field, box, FRAME_W, FRAME_H))
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError("%s %r has no size: w and h must be above 0." % (field, tuple(box)))
    if x < 0 or y < 0 or x + w > FRAME_W or y + h > FRAME_H:
        raise ValueError("%s %r runs off the %dx%d frame: keep x from 0, y from 0, x + w at most %d and y + h at "
                         "most %d." % (field, tuple(box), FRAME_W, FRAME_H, FRAME_W, FRAME_H))
    return x, y, w, h


# ---- titles ------------------------------------------------------------------------------------------------------

def title(text, start, end, box, style, key=None):
    """ASS events for text drawn centred inside box (x, y, w, h on the 1080x1920 frame) from start to end, wrapped
    at word breaks to the box's width. key, one of the text's words, is drawn in the style's key colour."""
    _check_style(style)
    _check_text(text, style)
    a, b = _times(start, end, "the title")
    x, y, w, h = _check_box(box)
    span = _key_span(text, key)
    lay = _wrap(text, style, "title", int(w))
    if lay.w > w or lay.h > h:
        raise ValueError("the title needs %d x %d px but box %r is %d x %d: give a box at least that size, or fewer "
                         "words%s." % (lay.w, lay.h, tuple(box), w, h, _smaller(style)))
    ax = x + w / 2 - (lay.left + lay.w / 2)
    ay = y + h / 2 - (lay.top + lay.h / 2)
    return _dialogues(2, _cs_clock(a), _cs_clock(b), "%s_title" % style.name, ax, ay, _line_height(style, "title"),
                      _coloured_lines(text, lay.lines, span, style), _glow_tags(style, "title"))


def _dialogues(layer, start, end, name, x, y, lh, lines, glow=""):
    """One Dialogue for each line, the first one's top centre at (x, y) and each next one lh lower (see
    _line_height): what one Dialogue with the lines joined by \\N draws when lh is the font's size. With glow (the
    tags from _glow_tags), each line's glow comes first, in the same layer, so the letters are drawn over it."""
    return [event for k, line in enumerate(lines) for event in (
        (["Dialogue: %d,%s,%s,%s_glow,,0,0,0,,{\\an8\\pos(%.2f,%.2f)%s}%s" % (layer, start, end, name, x, y + k * lh,
                                                                            glow, line)] if glow else [])
        + ["Dialogue: %d,%s,%s,%s,,0,0,0,,{\\an8\\pos(%.2f,%.2f)}%s" % (layer, start, end, name, x, y + k * lh, line)])]


def _key_span(text, key):
    if key is None:
        return None
    words = [text[s:e].strip() for s, e in _atoms(text)]
    if isinstance(key, str) and key.strip():
        k = key.strip()
        starts = {s for s, _e in _atoms(text)}
        ends = {s + len(text[s:e].rstrip()) for s, e in _atoms(text)}
        at = text.find(k)
        while at >= 0:
            if at in starts and at + len(k) in ends:
                return at, at + len(k)
            at = text.find(k, at + 1)
    raise ValueError("key %r is not one of the title's words: use one of %s." % (key, ", ".join(repr(w) for w in words)))


def _coloured_lines(text, lines, span, st):
    """The lines, the key word's letters wrapped in the key colour."""
    if span is None:
        return list(lines)
    out, at = [], 0
    for line in lines:
        begin = text.find(line, at)
        at = begin + len(line)
        out.append(_tint(text, begin, at, span, st))
    return out


def _tint(text, a, b, span, st):
    """text[a:b], its letters inside span (from, to) wrapped in the key colour, then back to the style's colour."""
    lo, hi = (max(a, span[0]), min(b, span[1])) if span else (b, b)
    if lo >= hi:
        return text[a:b]
    return text[a:lo] + "{\\1c%s&}" % _colour(st.key) + text[lo:hi] + "{\\1c%s&}" % _colour(st.colour) + text[hi:b]


def _caption_span(text, key):
    """(from, to) of key in a caption line's text when it is a run of the line's whole words, else None: a key that
    is not one leaves the line plain, so no word is cut and no mark falls off its letter."""
    try:
        return _key_span(text, key) if isinstance(key, str) and key.strip() else None
    except ValueError:
        return None


# ---- big words ---------------------------------------------------------------------------------------------------

POP = ((0, 60), (100, 125), (200, 100))
FADE_MS = 150


def big_word(word, at, length, centre, style):
    """ASS events for a big word ("POW", "ว้าว") centred on centre (x, y on the 1080x1920 frame) from `at` for length
    seconds: it pops in, growing past its size and settling back within 0.2 s, holds, and fades out in its last
    0.15 s."""
    _check_style(style)
    _check_text(word, style, "word")
    if "\n" in word:
        raise ValueError("word %r has a line break: a big word is one line." % (word,))
    if not isinstance(length, (int, float)) or isinstance(length, bool) or length < (POP[-1][0] + FADE_MS) / 1000:
        raise ValueError("length %r is too short for a big word: use %.2f s or more, so it can pop and fade."
                         % (length, (POP[-1][0] + FADE_MS) / 1000))
    a, b = _times(at, at + length, "the big word")
    if (not isinstance(centre, (tuple, list)) or len(centre) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in centre)):
        raise ValueError("centre %r is not a point: give (x, y) on the %dx%d frame." % (centre, FRAME_W, FRAME_H))
    cx, cy = centre
    left, top, right, bottom = _draw_boxes(style, "big", [word.strip()], align=5)[word.strip()]
    grow = max(scale for _ms, scale in POP) / 100
    half_w, half_h = (right - left) * grow / 2, (bottom - top) * grow / 2
    if 2 * half_w > FRAME_W:
        raise ValueError("word %r is %d px wide at its biggest, more than %d: use a shorter word%s."
                         % (word, 2 * half_w, FRAME_W, _smaller(style)))
    if not (half_w <= cx <= FRAME_W - half_w and half_h <= cy <= FRAME_H - half_h):
        raise ValueError("centre %r puts word %r off the frame: use x from %d to %d and y from %d to %d."
                         % (tuple(centre), word, math.ceil(half_w), FRAME_W - math.ceil(half_w), math.ceil(half_h),
                            FRAME_H - math.ceil(half_h)))
    ax, ay = cx - (left + right) / 2, cy - (top + bottom) / 2
    pop = _pop(int(round(at * 1000)) - a * 10)
    glow = _glow_tags(style, "big")
    return [("Dialogue: 3,%s,%s,%s_big%s,,0,0,0,,{\\an5\\pos(%.2f,%.2f)%s\\fad(0,%d)%s}%s"
             % (_cs_clock(a), _cs_clock(b), style.name, tail, ax, ay, pop, FADE_MS, tags, word.strip()))
            for tail, tags in ((("_glow", glow),) if glow else ()) + (("", ""),)]


def _pop(lead):
    """The tags that grow an event's letters through POP around their anchor, starting lead ms into the event."""
    return "\\fscx%d\\fscy%d" % (POP[0][1], POP[0][1]) + "".join(
        "\\t(%d,%d,\\fscx%d\\fscy%d)" % (lead + p[0], lead + q[0], q[1], q[1]) for p, q in zip(POP, POP[1:]))


# ---- captions ----------------------------------------------------------------------------------------------------

def _letters(s):
    """How many letters s shows: marks above or below a letter, and spaces at either end, do not count."""
    return sum(1 for c in s.strip() if not unicodedata.category(c).startswith("M"))


def _phrase(k, phrase):
    if not isinstance(phrase, dict) or not {"from", "to", "text"} <= set(phrase):
        raise ValueError("phrase %d %r is not {\"from\": s, \"to\": s, \"text\": \"...\"}." % (k, phrase))
    a, b, said = phrase["from"], phrase["to"], phrase["text"]
    for name, value in (("from", a), ("to", b)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value != value or value < 0:
            raise ValueError("phrase %d's %s %r is not a time: use seconds of the clip from 0." % (k, name, value))
    if b <= a:
        raise ValueError("phrase %d's to %s is not after its from %s: give the time it ends." % (k, b, a))
    if not isinstance(said, str) or not said.strip():
        raise ValueError("phrase %d has no text: leave out phrases where nothing is said." % k)
    return float(a), float(b), said


def _pieces(said, most):
    """[(start, end)] of said, cut at word breaks into pieces of at most `most` letters. A word with its particles stays
    whole when only the particles take it past `most`, by at most PARTICLE_ROOM letters; else it is cut before a
    particle rather than inside a word. The yeti review (10-29) captioned ไมโครโฟนนะครับ, 13 letters on lines of 12,
    as ไมโครโฟนนะครั and บแล้วก็มีคู่มือ. Only a word longer than `most` is cut inside, between its letters, and never
    inside a number."""
    locked = set()
    for m in thai.NUMBER.finditer(said):
        locked.update(range(m.start() + 1, m.end()))
    words, at = set(), 0
    for token in _tokens(said):
        at += len(token)
        words.add(at)
    pieces = []
    for s, e in _atoms(said):
        while _letters(said[s:e]) > most and not _runs_on(said[s:e], most):
            fits = [k for k in range(s + 1, e) if k not in locked and not said[k:].startswith(PARTICLES)
                    and not unicodedata.category(said[k]).startswith("M") and _letters(said[s:k]) <= most]
            near = [k for k in range(s + 1, e) if k in words and k not in locked and said[k:].startswith(PARTICLES)
                    and _letters(said[s:k]) <= most]
            breaks = [k for k in fits if k in words] or near or fits
            if not breaks:
                number = next(m.group(0) for m in thai.NUMBER.finditer(said) if m.start() < e and m.end() > s)
                raise ValueError("most %d is fewer letters than the number %r (%d letters), and a number is never "
                                 "split: use most of at least %d." % (most, number, _letters(number),
                                                                     _letters(number)))
            pieces.append((s, breaks[-1]))
            s = breaks[-1]
        pieces.append((s, e))
    return pieces


def _runs_on(piece, most):
    """Whether piece is a word whose particles alone take it past most, by at most PARTICLE_ROOM letters."""
    bare = piece.rstrip()
    while bare.endswith(PARTICLES):
        bare = bare[:-len(next(p for p in PARTICLES if bare.endswith(p)))]
    return _letters(bare) <= most < _letters(piece) <= most + PARTICLE_ROOM


def caption_lines(phrases, most=16):
    """Caption lines from what was heard, [{"from": s, "to": s, "text": "..."}] in seconds of the clip: each line at
    most `most` letters (marks above or below a letter do not count), broken at word breaks, a number never split
    and a particle (ครับ, นะ...) kept with the word before it. A phrase's time is spread over its words by their
    letters, so each line is {"from", "to", "text", "phrase", "words": [{"from", "to", "text"}]}; the lines of a
    phrase joined give its text back exactly."""
    if not isinstance(most, int) or isinstance(most, bool) or not 4 <= most <= 80:
        raise ValueError("most %r is not a line length: use a whole number of letters from 4 to 80." % (most,))
    if not isinstance(phrases, (list, tuple)):
        raise ValueError("phrases %r is not a list: give [{\"from\": s, \"to\": s, \"text\": \"...\"}]." % (phrases,))
    lines = []
    for k, phrase in enumerate(phrases):
        a, b, said = _phrase(k, phrase)
        pieces = _pieces(said, most)
        weights = [_letters(said[s:e]) for s, e in pieces]
        total, done = sum(weights) or 1, 0
        words = []
        for (s, e), weight in zip(pieces, weights):
            start = a + (b - a) * done / total
            done += weight
            words.append({"from": round(start, 3), "to": round(a + (b - a) * done / total, 3), "text": said[s:e]})
        current = []
        for word in words:
            if current and _letters("".join(w["text"] for w in current) + word["text"]) > most:
                lines.append(_line(k, current))
                current = []
            current.append(word)
        lines.append(_line(k, current))
    return lines


def _line(k, words):
    return {"from": words[0]["from"], "to": words[-1]["to"], "text": "".join(w["text"] for w in words),
            "phrase": k, "words": words}


def _check_line(k, line):
    try:
        words = line["words"]
        ok = (all(isinstance(line[f], (int, float)) for f in ("from", "to")) and isinstance(line["text"], str)
              and words and "".join(w["text"] for w in words) == line["text"]
              and all(isinstance(w["from"], (int, float)) and isinstance(w["to"], (int, float)) for w in words)
              and (line.get("key") is None or isinstance(line["key"], str)))
    except (TypeError, KeyError):
        ok = False
    if not ok:
        raise ValueError("line %d is not a caption line: pass what caption_lines() returned." % k)


def captions(lines, style, band, lit=False, pop=False, one=False):
    """ASS events for caption lines (from caption_lines) inside band (y, h) of the 1080x1920 frame, two lines at a
    time when two fit the band and one at a time when they do not or one is True, in the style's colour. With lit=True each word lights up in the style's key colour as it is said
    (libass \\kf): for words over the picture that nobody on screen is saying, such as a voice-over or a song, since
    a lit word pulls the eye away from a person explaining on camera. With pop=True each word grows in at its own
    place in the line as it is said and stays to the page's end, so the page builds up word by word and, once its
    last word is said, looks as the plain page does. A line's key, the word a viewer must not miss, is drawn in the
    key colour, plain or popped, when it is a run of the line's whole words (_caption_span); lit lights every word in
    that colour as it is said, so a key is not drawn apart there."""
    _check_style(style)
    if not isinstance(lit, bool):
        raise ValueError("lit %r is not True or False: use True to light each word as it is said, False for plain "
                         "words." % (lit,))
    if not isinstance(pop, bool):
        raise ValueError("pop %r is not True or False: use True to pop each word in as it is said, False for plain "
                         "words." % (pop,))
    if lit and pop:
        raise ValueError("lit and pop are both True: a word either lights up or pops in as it is said, so set one "
                         "of them.")
    if (not isinstance(band, (tuple, list)) or len(band) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in band)):
        raise ValueError("band %r is not a band: give (y, h) in pixels of the 1920 high frame." % (band,))
    y, h = band
    if h <= 0 or y < 0 or y + h > FRAME_H:
        raise ValueError("band %r is off the frame: keep y from 0 and y + h at most %d." % (tuple(band), FRAME_H))
    if not isinstance(lines, (list, tuple)) or not lines:
        raise ValueError("lines %r has no caption lines: make them with caption_lines()." % (lines,))
    for k, line in enumerate(lines):
        _check_line(k, line)
        _check_text(line["text"], style, "line %d" % k)
    if not isinstance(one, bool):
        raise ValueError("one %r is not True or False: use True to show the captions one line at a time." % (one,))
    pages, k = [], 0
    while k < len(lines):
        two = not one and k + 1 < len(lines) and 0 <= lines[k + 1]["from"] - lines[k]["to"] <= PAIR_GAP
        pages.append(lines[k:k + 2] if two else lines[k:k + 1])
        k += 2 if two else 1
    shown = [line["text"].strip() for line in lines]
    boxes = _draw_boxes(style, "caption", shown)
    if pop:
        # every line's pieces measured in two FFmpeg runs, not two a line (0.35 s against 2 s for 45 s of speech)
        said = [[w["text"] for w in line["words"]] for line in lines]
        for s in said:
            s[0], s[-1] = s[0].lstrip(), s[-1].rstrip()
        _draw_boxes(style, "caption", shown + ["".join(s[:i + 1]) for s in said for i in range(len(s))], align=7)
        _draw_boxes(style, "caption", shown + [p for s in said for p in s], align=5)
    lh = _line_height(style, "caption")
    fitted = []
    for page in pages:
        if len(page) == 2 and _layout_of(tuple(line["text"].strip() for line in page), boxes, lh).h > h:
            fitted += [page[:1], page[1:]]
        else:
            fitted.append(page)
    pages = fitted
    events = []
    for n, page in enumerate(pages):
        start, end = _edge(page[0]["from"]), _edge(page[-1]["to"])
        if n + 1 < len(pages):
            end = min(end, _edge(pages[n + 1][0]["from"]))
        if end <= start:
            continue
        lay = _layout_of(tuple(line["text"].strip() for line in page), boxes, lh)
        if lay.h > h or lay.w > FRAME_W:
            raise ValueError("the captions need %d x %d px but band %r is %d high: give a band at least %d high%s."
                             % (lay.w, lay.h, tuple(band), h, lay.h, _smaller(style)))
        ax = FRAME_W / 2 - (lay.left + lay.w / 2)
        ay = y + h / 2 - (lay.top + lay.h / 2)
        if pop:
            events += _popping(page, style, start, end, ax, ay, lh, boxes)
            continue
        if lit:
            words = _karaoke(page, start)
        else:
            words = ["{\\1c%s&}%s" % (_colour(style.colour), _keyed(line, style)) for line in page]
        events += _dialogues(1, _cs_clock(start), _cs_clock(end), "%s_caption" % style.name, ax, ay, lh, words,
                             _glow_tags(style, "caption"))
    return events


def _keyed(line, st):
    """A caption line's text, its key in the key colour."""
    shown = line["text"].strip()
    return _tint(shown, 0, len(shown), _caption_span(shown, line.get("key")), st)


def _popping(page, st, start, end, ax, ay, lh, boxes):
    """The page's words, each popping in when it is said (from the page's start at the earliest) and staying to
    `end`. While a word pops (POP) it is an event of its own in layer 2, growing around its centre at the place it
    has when its line is drawn whole with \\an8 at (ax, ay + k * lh): its right edge where the line drawn up to it
    ends. Once it has popped it joins the rest in layer 1: the plain page's line with the words not yet said
    invisible, so the words said so far share one outline and one shadow (a solid one, not a ring per word, and no
    two shadows over each other where Thai words touch), and a page whose words are all said is the plain page."""
    name, glow = "%s_caption" % st.name, _glow_tags(st, "caption")
    colour = "\\1c%s&" % _colour(st.colour)
    rest, popping = [], []
    for k, line in enumerate(page):
        said = [w["text"] for w in line["words"]]
        said[0], said[-1] = said[0].lstrip(), said[-1].rstrip()
        shown = "".join(said)
        span = _caption_span(shown, line.get("key"))
        upto = ["".join(said[:i + 1]) for i in range(len(said))]
        left = _draw_boxes(st, "caption", [shown] + upto, align=7)
        mid = _draw_boxes(st, "caption", [shown] + said, align=5)
        x0 = ax + boxes[shown][0] - left[shown][0]
        y = ay + k * lh + boxes[shown][1] - mid[shown][1]
        begins = [max(start, _edge(w["from"])) for w in line["words"]]
        settled = [max(start, _edge(w["from"] + POP[-1][0] / 1000)) for w in line["words"]] + [end]
        for i, (word, piece, ends) in enumerate(zip(line["words"], said, upto)):
            a, b = begins[i], min(end, settled[i])
            if a < b and mid[piece] and left[ends]:
                when = "%s,%s" % (_cs_clock(a), _cs_clock(b))
                tags = "{\\an5\\pos(%.2f,%.2f)%s" % (x0 + left[ends][2] - mid[piece][2], y,
                                                     _pop(int(round(word["from"] * 1000)) - a * 10))
                tinted = _tint(shown, len(ends) - len(piece), len(ends), span, st)
                if glow:
                    popping.append("Dialogue: 2,%s,%s_glow,,0,0,0,,%s%s}%s" % (when, name, tags, glow, tinted))
                popping.append("Dialogue: 2,%s,%s,,0,0,0,,%s%s}%s" % (when, name, tags, colour, tinted))
            a, b = settled[i], min(end, settled[i + 1])
            if a < b:
                unsaid = "".join(said[i + 1:])
                rest += _dialogues(1, _cs_clock(a), _cs_clock(b), name, ax, ay + k * lh, lh,
                                   ["{%s}%s%s" % (colour, _tint(shown, 0, len(ends), span, st),
                                                  "{\\alpha&HFF&}" + unsaid if unsaid else "")], glow)
    return rest + popping


def _karaoke(page, start):
    """The page's lines with a \\kf before each word, timed in centiseconds from the events' start: each line is an
    event of its own from the page's start, so a line after the first waits, unlit, until its first word."""
    at, out = start, []
    for line in page:
        parts = ["{\\k%d}" % (at - start)] if at > start else []
        for k, word in enumerate(line["words"]):
            said = word["text"]
            if k == 0:
                said = said.lstrip()
            if k == len(line["words"]) - 1:
                said = said.rstrip()
            begin, finish = int(round(word["from"] * 100)), int(round(word["to"] * 100))
            if begin > at:
                parts.append("{\\k%d}" % (begin - at))
                at = begin
            parts.append("{\\kf%d}%s" % (max(0, finish - at), said))
            at = max(at, finish)
        out.append("".join(parts))
    return out


# ---- the channel name --------------------------------------------------------------------------------------------

# The channel's name (a handle such as @SineySociety) on every clip, in the captions' letters at CHANNEL_SHARE of
# their size, so it reads as the channel's mark and not as words to follow (measured 10-28 in clean: a capital is 25
# px of the 1080 px frame, about 9 pt on a phone 390 pt wide, where the captions' is 39 px; at half size it was 20 px,
# 7 pt, smaller than a phone's own small print). It sits in the row the layouts keep for it (layouts.NAME_TOP to
# layouts.SAFE_BOTTOM), where nothing the studio places goes and no phone covers: its letters from CHANNEL_GAP px
# under the row's top to CHANNEL_FLOOR px above what a phone covers at the bottom, CHANNEL_ROOM px left of the like
# and share buttons (layouts.BUTTONS), inside the sides a phone cuts (layouts.SAFE). A look whose words stand on a
# box (sticker) draws the name as cut-out letters, its box's colour as their edge: the box is a whole line tall, 74
# px against the 31 to 48 px other looks' names stand (measured 10-28, with marks above and below Thai letters).
CHANNEL_SPOTS = ("bottom left", "bottom middle", "bottom right")
CHANNEL_SPOT = "bottom left"
CHANNEL_SHARE = 0.65
CHANNEL_ROOM = 16
CHANNEL_FLOOR = 8
CHANNEL_GAP = 6


def channel_style(style):
    """The style the channel name is drawn in: style's letters, colours and edges at CHANNEL_SHARE of the size of
    its captions, without a box."""
    _check_style(style)
    small = max(1, int(round(_size(style, "caption") * CHANNEL_SHARE)))
    return style._replace(name=style.name + "_channel", sizes=(small,) * len(KINDS), box=False)


def _channel_strip():
    """(left, right, bottom) of the strip the channel name's letters stay in, in pixels of the frame."""
    return (layouts.SAFE[0], layouts.BUTTONS[0] - CHANNEL_ROOM, layouts.SAFE_BOTTOM - CHANNEL_FLOOR)


def channel_box(name, style, spot=None):
    """(x, y, w, h) on the frame of the channel name's letters at spot (one of CHANNEL_SPOTS; None for
    CHANNEL_SPOT), drawn in channel_style(style), so other words can be kept clear of them."""
    x, y, (left, top, right, bottom) = _channel_place(name, style, spot)
    return (int(math.floor(x + left)), int(math.floor(y + top)), int(math.ceil(right - left)),
            int(math.ceil(bottom - top)))


def _channel_place(name, style, spot):
    """(x, y) of the name's anchor (its middle, \\an5) and its letters' box from the anchor."""
    _check_style(style)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name %r is empty: give the channel's name, such as @SineySociety." % (name,))
    if "\n" in name:
        raise ValueError("name %r has a line break: a channel name is one line." % (name,))
    if spot is None:
        spot = CHANNEL_SPOT
    if spot not in CHANNEL_SPOTS:
        raise ValueError("spot %r is not a spot for the channel name: use one of %s, or None for %s."
                         % (spot, ", ".join(CHANNEL_SPOTS), CHANNEL_SPOT))
    small = channel_style(style)
    _check_text(name, small, "name")
    left, top, right, bottom = _draw_boxes(small, "title", [name.strip()], align=5)[name.strip()]
    first, last, low = _channel_strip()
    middle = FRAME_W / 2.0
    room = 2 * min(middle - first, last - middle) if spot == "bottom middle" else last - first
    if right - left > room:
        raise ValueError("name %r is %d px wide at the channel name's size, and the %s spot holds %d px: use a "
                         "shorter name%s." % (name, math.ceil(right - left), spot, room,
                                              ", or the bottom left spot" if spot == "bottom middle" else ""))
    tall = low - (layouts.NAME_TOP + CHANNEL_GAP)
    if bottom - top > tall:
        raise ValueError("name %r stands %d px tall in %s at the channel name's size (%d px), and the row kept for it "
                         "holds %d px: use a smaller size or another font for the captions."
                         % (name, math.ceil(bottom - top), style.fontname, small.sizes[0], tall))
    x = {"bottom left": first - left, "bottom middle": middle - (left + right) / 2.0,
         "bottom right": last - right}[spot]
    return x, low - bottom, (left, top, right, bottom)


def channel_name(name, start, end, style, spot=None):
    """ASS events for the channel's name from start to end (seconds of the clip) at spot (see channel_box), drawn
    in channel_style(style): pass that style to document() as well."""
    a, b = _times(start, end, "the channel name")
    x, y, _ = _channel_place(name, style, spot)
    return ["Dialogue: 0,%s,%s,%s_title,,0,0,0,,{\\an5\\pos(%.2f,%.2f)}%s"
            % (_cs_clock(a), _cs_clock(b), channel_style(style).name, x, y, name.strip())]


# ---- the file ----------------------------------------------------------------------------------------------------

def document(styles, events):
    """The text of an ASS file for the 1080x1920 frame (PlayResX and PlayResY set, so sizes are pixels) with
    every kind of each style and the events in order."""
    styles = list(styles)
    for st in styles:
        _check_style(st)
    names = {"%s_%s%s" % (st.name, kind, tail) for st in styles for kind in KINDS
             for tail in (("", "_glow") if st.glow else ("",))}
    lines = []
    for event in events:
        if not isinstance(event, str) or not event.startswith("Dialogue: "):
            raise ValueError("event %r is not an ASS Dialogue line: pass what title(), big_word() or captions() "
                             "returned." % (event,))
        used = event.split(",", 4)[3]
        if used not in names:
            raise ValueError("event uses style %r, which is not in styles: pass that style to document()." % used)
        lines.append(event)
    unique = list(dict.fromkeys(st for st in styles))
    return (_header(FRAME_W, FRAME_H) + "".join(_ass_style(st, kind) + "\n" for st in unique for kind in KINDS)
            + "".join(_glow_style(st, kind) + "\n" for st in unique if st.glow for kind in KINDS)
            + EVENTS + "".join(line + "\n" for line in lines))


def cards():
    """The looks people and Muse pick from: [{"name", "en", "th", "png", "colours"}], png a picture of SAMPLE drawn
    in that look on a dark ground (540 x 200, made once in work/looks), colours the look's own {name: #RRGGBB or
    None} for each name in PAINTS, the ones a person can change."""
    return [{"name": name, "en": CARDS[name][0], "th": CARDS[name][1], "png": _card_png(STYLES[name]),
             "colours": dict(zip(PAINTS, (STYLES[name].colour, STYLES[name].key, STYLES[name].edge, STYLES[name].glow,
                                          STYLES[name].shade)))}
            for name in HOUSE_NAMES]


def _card_png(st):
    """The path of SAMPLE drawn in st on a dark ground, CARD_SIZE, made once: cut around the words so they fill
    CARD_FILL of its width (CARD_TALL of its height at most), as they are drawn on the frame, then shrunk. The words
    and the drawing are the cache key, so a card drawn another way is a new picture."""
    folder = os.path.join(recording.WORK, "looks")
    os.makedirs(folder, exist_ok=True)
    body = document([st], title(SAMPLE, 0, 1, (40, 40, FRAME_W - 80, 320), st, key="50%"))
    words = measure(SAMPLE, st, "title", FRAME_W - 80)
    aspect = CARD_SIZE[0] / CARD_SIZE[1]
    cut_w = min(FRAME_W, 2 * math.ceil(max(words.w / CARD_FILL, words.h / CARD_TALL * aspect) / 2))
    cut_h = 2 * round(cut_w / aspect / 2)
    drawing = ("format=yuv420p,%s,pad=%d:%d,{words},crop=%d:%d:%d:%d,scale=%d:%d"
               % ((CARD_GROUND, FRAME_W, FRAME_H, cut_w, cut_h, (FRAME_W - cut_w) // 2, 200 - cut_h // 2) + CARD_SIZE))
    key = hashlib.sha1((body + drawing).encode("utf-8")).hexdigest()[:10]
    path = os.path.join(folder, "%s_%s.png" % (st.name, key))
    if os.path.isfile(path):
        return path
    fd, ass_path = tempfile.mkstemp(suffix=".ass", dir=folder)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    fd, part = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".part.png", dir=folder)
    os.close(fd)
    try:
        subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=%dx400:d=1" % FRAME_W,
                        "-vf", drawing.replace("{words}", _ass_filter(ass_path, fonts_folder(st))), "-frames:v", "1",
                        "-update", "1", part], check=True, capture_output=True, timeout=RENDER_SECONDS)
    except (subprocess.SubprocessError, OSError):
        os.remove(part)
        raise
    finally:
        os.remove(ass_path)
    try:
        os.replace(part, path)
    except PermissionError:
        # made by another request at the same moment and open now (Windows): the same picture, so keep it
        os.remove(part)
        if not os.path.isfile(path):
            raise
    return path


def burn(ass_path, fonts_dir=FONTS):
    """The frame-stage Fragment that draws the ASS file onto the 1080x1920 frame, with fonts from fonts_dir only
    standing in for the styles' fonts."""
    if not isinstance(ass_path, str) or not os.path.isfile(ass_path):
        raise ValueError("ass_path %r is not a file: write document() to a file first." % (ass_path,))
    if not isinstance(fonts_dir, str) or not os.path.isdir(fonts_dir):
        raise ValueError("fonts_dir %r is not a folder: use the studio's fonts folder, %s." % (fonts_dir, FONTS))
    return Fragment("[in]%s[out]" % _ass_filter(ass_path, fonts_dir), [])
