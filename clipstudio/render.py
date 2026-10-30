"""The clip drawn from its project list: one FFmpeg filter graph from the recording to the 1080x1920 frame.

    recording piece(s)  ->  [src] at clip time t  ->  source stage (blurs, look)  ->  layout (layouts.py, pushes)
                        ->  frame stage (camera hits, pictures, glass, stickers, flurries, the words, flashes)

The preview frame is the same graph started at the frame's time: the piece is opened at the matching recording
second and its timestamps are set so t is still clip time, so a title from 2 s to 5 s shows in the preview at 3 s
exactly as it will in the finished clip. export() joins the pieces at 30 fps from clip second 0 and draws them with
the same stages. Its sound:

    the pieces' sound joined  ->  -14 LUFS  ->  the sounds added (effects.sound)
                              ->  the music under it, lower while someone speaks (effects.duck)  ->  a limiter
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading

from . import cutout, effects, layouts, project, recording, text

PREVIEW_WIDTH = 540
FRAME_SECONDS = 60
WORD_SECONDS = 0.8
HIT_KINDS = ("flash", "shake", "punch", "word")
FPS, RATE = 30, 48000
LUFS, PEAK, RANGE = -14, -1.5, 11
PRESET, CRF = "medium", 19
# the page's video without words (base): veryfast 24 measured as sharp as ultrafast 24 with a file 2 to 4 times smaller
BASES, BASE_SIZE, BASE_PRESET, BASE_CRF = 1, (540, 960), "veryfast", 24
EXPORT_SECONDS = 600
MUSIC_LEVEL, MUSIC_LEVELS, MUSIC_DUCK = -6, (-30, 0), -12
MUSIC_IN, MUSIC_OUT = 0.3, 1.5
# a muted stretch's edges ease over MUTE_FADE s, in frames of MUTE_FRAME samples (5 ms): at FFmpeg's own 1024 the
# level stepped from 91% to 20% in one frame (measured 10-28 on a tone)
MUTE_FADE, MUTE_FRAME = 0.03, 240
SILENT = -70


class NotDrawn(Exception):
    """The frame could not be drawn. The sentence says why."""


class Stopped(Exception):
    """The person pressed Stop while the clip was being made: export() raises it and leaves no file."""


def _ffmpeg():
    return shutil.which("ffmpeg") or "ffmpeg"


def folder(name):
    path = os.path.join(recording.WORK, name)
    os.makedirs(path, exist_ok=True)
    return path


def style_of(state, default="clean", kind="title"):
    """The text style the clip draws one part of its words in (kind: title, caption or big): the last style entry,
    with the font, the size (1 to 5) and the colours given that part ("fonts": {part: font}, "sizes": {part: size},
    "part_colours": {part: {name: #RRGGBB}}) or every part ("font", "size", "colours"), else the house style the
    recipe guessed."""
    if kind not in text.KINDS:
        raise ValueError("kind %r is not a part of the words: use one of %s." % (kind, ", ".join(text.KINDS)))
    e = state["style"]
    if e:
        return text.look(e["name"], e.get("font"), e.get("colour"), e.get("fonts"), e.get("size"),
                         e.get("sizes"), e.get("colours"), e.get("part_colours"))[kind]
    return text.STYLES.get(default) or text.STYLES["clean"]


ONE_SIZE = 5
ONE_GROW = 1.6
ONE_LOOK = "pop"
# Captions over people talking (the talk and split layouts) are a news clip's subtitles: white words on a see-through
# box, a line at a time (Tarm, 10-30: just right for clips of people talking), and the title keeps its own look.
TALK_LOOK = "news"
TALK_KINDS = ("talk", "split")


def _caption_style(state, default, lines=(), band_h=None):
    """The style of captions shown one line at a time (a talked review): ONE_LOOK's heavy letters and thick outline
    when the person chose no look (the recipe's guess is not a choice), TALK_LOOK over people talking, at ONE_SIZE
    grown ONE_GROW times, shrunk back
    as far as needed (never below ONE_SIZE) for the widest of lines to fit the caption band's width, and the tallest,
    with its outline and shadow, its height band_h; the person's own size for the captions or for all the words
    wins. Measured 10-28 on Tarm's screen recordings of Extreme IT: a line of
    two or three words is about 55% of the frame wide, in bold white with a black outline; at size 5 the studio's
    were 30%."""
    e = state["style"] or {}
    if e.get("size") or (e.get("sizes") or {}).get("caption"):
        return style_of(state, default, "caption")
    talking = ((state.get("clip") or {}).get("layout") or {}).get("kind") in TALK_KINDS
    st = text.look(e.get("name") or (TALK_LOOK if talking else ONE_LOOK), e.get("font"), e.get("colour"),
                   e.get("fonts"), None,
                   dict(e.get("sizes") or {}, caption=ONE_SIZE), e.get("colours"), e.get("part_colours"))["caption"]
    sizes = [text.measure(line["text"].strip(), st, "caption", 4 * layouts.W) for line in lines]
    widest = max([m.w for m in sizes] or [0])
    tallest = max([m.h for m in sizes] or [0]) + (2 * st.edge_ratio + st.shadow_ratio) * st.sizes[1]
    grow = min(ONE_GROW, layouts.CAPTIONS[2] / float(widest) if widest else ONE_GROW,
               band_h / float(tallest) if band_h and tallest else ONE_GROW)
    grow = max(1.0, grow)
    return st._replace(name="%s-one%d" % (st.name, round(grow * 100)), sizes=tuple(int(s * grow) for s in st.sizes))


TITLE_SHRINK = 0.95


def _title_style(state, default, said, title):
    """The style a title is drawn in over captions shown one line at a time (said, from _caption_style): said's
    letters at said's caption size, shrunk by TITLE_SHRINK at a time as far as needed for the title to fit its box,
    never below the title's own size; the person's own size for the titles or for all the words wins. On the yeti
    review (10-29) the hook at the title's own size stood half as tall as the captions under it. A see-through box
    (the news look) lends its size only: in it the title read as one more caption (Tarm, 10-30)."""
    st = style_of(state, default, "title")
    e = state["style"] or {}
    if e.get("size") or (e.get("sizes") or {}).get("title"):
        return st
    w, h = title["box"][2], title["box"][3]
    size = said.sizes[1]
    while size > st.sizes[0]:
        base = st if 0 < said.box < 1 else said
        big = base._replace(name="%s-title%d" % (base.name, size), sizes=(size,) + base.sizes[1:])
        got = text.measure(title["text"], big, "title", int(w))
        if got.w <= w and got.h <= h:
            return big
        size = int(size * TITLE_SHRINK)
    return st


# A title with room grows up to TITLE_GROW times its own size, as far as its box holds it and it covers no face
# (tools.grown chooses how far; the title keeps it as "grow"). At its own size the seller's hook stood 61 px tall,
# a third of the frame wide, and a montage's labels as small, with no captions to grow them (matrix 10-30); at 1.6
# the hook fills its place in two lines, as tall as captions shown one line at a time.
TITLE_GROW = 1.6


def grown_style(state, default, grow):
    """The title style at grow times its size (TITLE_GROW at most), or the title style as it is when the person
    chose a size for the titles or for all the words."""
    st = style_of(state, default, "title")
    e = state["style"] or {}
    if not grow or grow <= 1 or e.get("size") or (e.get("sizes") or {}).get("title"):
        return st
    size = int(st.sizes[0] * min(grow, TITLE_GROW))
    return st._replace(name="%s-title%d" % (st.name, size), sizes=(size,) + st.sizes[1:])


def _grown_title(state, default, title, base):
    """The style a title's grow draws it in: base (its own style, or the captions' letters over captions shown one
    line at a time) with letters as tall as its own size times its grow, unless base's stand that tall already,
    shrunk by TITLE_SHRINK at a time as far as needed to fit its box (its words or the look changed since), never
    below base. The explainer replay 10-30 grew its title, then put on one-line captions, and it was drawn at the
    captions' size: the order of two steps decided how big the title was."""
    big = grown_style(state, default, title.get("grow"))
    size = int(big.sizes[0] / float(big.scale) * base.scale)
    w, h = title["box"][2], title["box"][3]
    while size > base.sizes[0]:
        got = base._replace(name="%s-title%d" % (base.name, size), sizes=(size,) + base.sizes[1:])
        drawn = text.measure(title["text"], got, "title", int(w))
        if drawn.w <= w and drawn.h <= h:
            return got
        size = int(size * TITLE_SHRINK)
    return base


def _styles(state, default):
    """The clip's style for each part of its words, {kind: Style}."""
    return {kind: style_of(state, default, kind) for kind in text.KINDS}


def fonts(state, default):
    """The folder libass is given for the clip's ASS file: the fonts of its styles and of the captions shown one line
    at a time, which ass() draws in their own look (the titles over them too). Left out, as on the yeti review
    (10-29), libass drew those captions in another face, half as big again as they were measured."""
    styles = list(_styles(state, default).values())
    shown = state.get("captions")
    if shown and shown.get("one"):
        styles.append(_caption_style(state, default, shown["lines"], shown["band"][1]))
    return text.fonts_folder(*styles)


def _hits(state):
    hits = state.get("hits") or []
    for h in hits:
        if h.get("kind") not in HIT_KINDS:
            raise ValueError("hit kind %r is not a hit: use one of %s." % (h.get("kind"), ", ".join(HIT_KINDS)))
    return hits


def whole_lines(state, default="clean"):
    """True when the captions' look puts each line on a see-through box (0 < box < 1): such a line shows whole, as
    a news clip's subtitles do, and its words never pop in one at a time. Seen on EP0 10-30: a popped word came on a
    box of its own, darker where it met the line's box, and the line's box grew a word at a time (Tarm 10-30: no
    box over another)."""
    shown = state.get("captions") or {}
    st = _caption_style(state, default) if shown.get("one") else _styles(state, default)["caption"]
    return 0 < st.box < 1


def ass(state, default="clean", behind=False):
    """The path of the clip's ASS file (written once per content) with its titles, captions, big words and the
    channel's name (state['channel']: {"name", "spot"}, or None), or None when there are no words on screen. With
    behind=True it holds only the titles and big words that go behind the person ("behind": True), and without it
    everything else. The channel's name is in the captions' letters (text.channel_name), the whole clip long."""
    captions = None if behind else state.get("captions")
    titles = [t for t in state["titles"] if bool(t.get("behind")) == behind]
    words = [h for h in _hits(state) if h["kind"] == "word" and bool(h.get("behind")) == behind]
    channel = None if behind else state.get("channel")
    if not (titles or captions or words or channel):
        return None
    st = _styles(state, default)
    shown = state.get("captions")
    one = _caption_style(state, default, shown["lines"], shown["band"][1]) if shown and shown.get("one") else None
    events = []
    styles = list(st.values())
    for t in titles:
        drawn = _title_style(state, default, one, t) if one else st["title"]
        if t.get("grow"):
            drawn = _grown_title(state, default, t, drawn)
        if drawn not in styles:
            styles.append(drawn)
        events += text.title(t["text"], t["start"], t["end"], t["box"], drawn, t.get("key"))
    if captions:
        bands = captions.get("bands") or {}
        said = one or st["caption"]
        if said not in styles:
            styles.append(said)
        for who in sorted({str(line.get("who")) for line in captions["lines"]}):
            lines = [line for line in captions["lines"] if str(line.get("who")) == who]
            events += text.captions(lines, said, bands.get(who) or captions["band"], lit=captions.get("lit", False),
                                    pop=captions.get("pop", False) and not whole_lines(state, default),
                                    one=bool(captions.get("one")))
    for h in words:
        events += text.big_word(h["word"], h["at"], WORD_SECONDS, h["centre"], st["big"])
    if channel:
        events += text.channel_name(channel["name"], 0, state["length"], st["caption"], channel.get("spot"))
        styles.append(text.channel_style(st["caption"]))
    body = text.document(styles, events)
    path = os.path.join(folder("ass"), hashlib.sha1(body.encode("utf-8")).hexdigest()[:16] + ".ass")
    if not os.path.isfile(path):
        part = _part(path)
        with open(part, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        settle(part, path)
    return path


def _part(path):
    """A new empty file beside path to write it into before it is moved to path: its own name, so two requests
    making the same file at once (the page and Muse asking for the same frame) never write into one."""
    folder, name = os.path.split(path)
    fd, part = tempfile.mkstemp(prefix=name + ".", suffix=".part" + os.path.splitext(name)[1], dir=folder)
    os.close(fd)
    return part


def settle(part, path):
    """Moves the finished part to path. Two requests can make the same file at the same moment, and on Windows the
    second move is refused while the first file is open (the page showing it): that file is the same file, so the
    part is dropped and the file kept."""
    try:
        os.replace(part, path)
    except PermissionError:
        os.remove(part)
        if not os.path.isfile(path):
            raise


def after_source(state, src, out, first_input, default="clean", words=True):
    """(graph text from [src] to [out], extra files) in three stages:
    - source stage, on the recording's picture: the blurs (a fixed box, or a face's track of [t, x, y, w, h] points in
      clip seconds), so a blurred box stays blurred wherever the layout shows it, the card's blurred background
      included, then the colour look (state['look'], a name in effects.LOOKS, or None), so it grades the footage
      and never the frame's design, the stickers or the words;
    - the layout, to the 1080x1920 frame, with the slow pushes inside its main part, so the frame's design holds
      still while the footage pushes in;
    - frame stage: first what needs the person cut out (cutout.py, from the person's mask state['mask']): the
      background behind them (state['backdrop'], a picture where the clip does not show them), the footage in black
      and white around them and a white edge around them (state['cutouts']: {"look": grey or
      outline, "start", "end"}) and the titles and big words with "behind": True drawn under them; then the camera
      hits (punch-in, shake), so the person and all of that move together; then the pictures drawn for what is
      said (state['pictures'], effects.picture), under the words, the glass boxes, the stickers, the
      flurries (state['flurries']:
      {"pngs", "at", "box", "move", "count", "length", "size"}, see effects.flurry), then the words (titles, captions
      and big words, one ASS file), so the words hold still while the picture moves and stay on top of a sticker, a
      flurry or a glass box, then the flashes last, so a flash lights the words too.
    With words=False the titles, captions and big words over the picture are left out (the page lays layer.words()
    over it); the words behind the person and the flashes stay, as they are part of the picture.
    A drawing that cannot be made raises ValueError with the sentence for Muse."""
    blurs = state.get("blurs") or []
    blurs = (effects.blur_tracks([b["track"] for b in blurs if b.get("track")]) +
             [effects.blur_box(b["box"], b["start"], b["end"]) for b in blurs if not b.get("track")])
    if state.get("look"):
        blurs.append(effects.look(state["look"]))
    seen, files = effects.join(blurs, src, "seen", first_input)
    lay, more = layouts.graph(state["clip"]["layout"], "seen", "drawn", first_input + len(files),
                              state.get("pushes") or [])
    files = files + more
    hits = _hits(state)
    fragments = _backdrop(state) + _cut_out(state, default)
    camera = {"shake": lambda h: effects.shake(h["at"]),
              "punch": lambda h: effects.punch_in(h["at"], centre=h["centre"])}
    fragments += [camera[h["kind"]](h) for h in hits if h["kind"] in camera]
    fragments += [effects.picture(k["png"], k["at"], k["length"], k["box"]) for k in state.get("pictures") or []]
    fragments += [layouts.glass(g["box"], g["start"], g["end"], g.get("tint", "light"), g.get("shine", True))
                  for g in state.get("glass") or []]
    fragments += [effects.sticker(k["png"], k["at"], k["length"], k["box"]) for k in state.get("stickers") or []]
    fragments += [effects.flurry(f["pngs"], f["at"], f["box"], f.get("move", "burst"), f.get("count", 8),
                                 f.get("length"), f.get("size", 150)) for f in state.get("flurries") or []]
    path = ass(state, default) if words else None
    if path:
        fragments.append(text.burn(path, fonts(state, default)))
    fragments += [effects.flash(h["at"]) for h in hits if h["kind"] == "flash"]
    stage, more = effects.join(fragments, "drawn", out, first_input + len(files))
    return ";".join((seen, lay, stage)), files + more


CUTOUT_LOOKS = {"grey": cutout.grey, "outline": cutout.outline}


def _backdrop(state):
    """The background behind the person (state['backdrop'], from put_background) as the frame stage's first
    fragment, or none. It needs the person's mask over the whole clip: where the mask has no frame, the clip would
    show the background alone."""
    ground = state.get("backdrop")
    if not ground:
        return []
    mask, clip = state.get("mask"), state["clip"]
    a, b = min(p[0] for p in clip["pieces"]), max(p[1] for p in clip["pieces"])
    if not mask or mask["from"] > a + 0.01 or mask["to"] < b - 0.01:
        raise ValueError("The background needs the person found over the whole clip, %g to %g s of the recording, "
                         "and the clip has changed since: put the background on again." % (round(a, 2), round(b, 2)))
    try:
        path = cutout.frame_mask(mask, clip["pieces"], clip["layout"], state["length"], state.get("pushes") or [])
    except RuntimeError as e:
        raise ValueError(str(e))
    return [cutout.backdrop(path, ground["png"], bool(mask.get("green")))]


def _cut_out(state, default):
    """The frame stage's fragments that need the person's mask: the cut-out looks, then the words behind the person;
    none when the clip asks for neither. In a talk the looks follow whoever speaks (cutout.speaking_mask), and the
    words go behind the person the mask was made for."""
    cuts = state.get("cutouts") or []
    behind_titles = [(t["start"], t["end"]) for t in state["titles"] if t.get("behind")]
    behind_words = [(h["at"], h["at"] + WORD_SECONDS) for h in _hits(state) if h["kind"] == "word" and h.get("behind")]
    if not (cuts or behind_titles or behind_words):
        return []
    if not state.get("mask"):
        raise ValueError("The words behind the person and the cut-out looks need the person's mask, and this clip has "
                         "none: run cut_out first.")
    for c in cuts:
        if c.get("look") not in CUTOUT_LOOKS:
            raise ValueError("cut-out look %r is not a look: use one of %s." % (c.get("look"), ", ".join(CUTOUT_LOOKS)))
    clip = state["clip"]
    try:
        mask = cutout.frame_mask(state["mask"], clip["pieces"], clip["layout"], state["length"],
                                 state.get("pushes") or [])
        speaking = cutout.speaking_mask(state["mask"], clip["pieces"], clip["layout"], state["length"],
                                        state.get("pushes") or []) if cuts else mask
    except RuntimeError as e:
        raise ValueError(str(e))
    looks = [CUTOUT_LOOKS[c["look"]](speaking, c["start"], c["end"]) for c in cuts]
    words = ass(state, default, behind=True)
    if words:
        filters = text._ass_filter(words, fonts(state, default))
        looks.insert(sum(c["look"] == "grey" for c in cuts), cutout.behind(filters, mask, behind_titles + behind_words))
    return looks


def _picture_input(f, step=None):
    """The input options for an extra file: the person's mask (a .mkv video in clip time) from clip second 0, or
    from step for a preview; a picture looped at 30 fps, from step for a preview."""
    if f.endswith(".mkv"):
        return (["-ss", "%.6f" % step, "-itsoffset", "%.6f" % step] if step is not None else []) + ["-i", f]
    return (["-itsoffset", "%.6f" % step] if step is not None else []) + ["-loop", "1", "-framerate", str(FPS), "-i", f]


def frame(rec, state, t, default="clean", width=PREVIEW_WIDTH, words=True):
    """The clip's frame at clip second t, width pixels wide: the jpg's path in work/clipframes, made once. The
    extra pictures (masks, shadows, stickers) are opened from the last 1/30 s step at or before t, so FFmpeg does
    not read every picture frame before it (a frame late in a long clip draws as fast as the first), and their frames
    fall on the steps the full render's do, so the frame shows each picture as the full render does. With
    words=False the titles, captions and big words are left out, as after_source leaves them (the cover's frame:
    Muse Image writes the title itself)."""
    clip = state["clip"]
    if not clip:
        raise NotDrawn("There is no clip yet, so there is no frame to draw.")
    at = project.to_clip(clip["pieces"], t)
    if at is None:
        raise NotDrawn("%.2f s is outside the clip, which is %.2f s long." % (t, state["length"]))
    try:
        graph, files = after_source(state, "src", "whole", 1, default, words)
    except ValueError as e:
        raise NotDrawn(str(e))
    # the frame's time is a whole tick worked out here: t/TB in FFmpeg can fall a hair under one (7.6 s at 1/1000 is
    # 7599.99...), and a frame before the pictures' first frame is drawn with none of them
    ticks = FPS * 1000
    graph = ("[0:v]settb=1/%d,setpts=PTS-STARTPTS+%d[src];%s;[whole]scale=%d:-2[fin]"
             % (ticks, round(t * ticks), graph, width))
    step = math.floor(t * FPS + 1e-6) / FPS
    key = hashlib.sha1(json.dumps([rec["path"], rec["size"], rec["mtime"], round(at[1], 3), graph, files],
                                  ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    target = os.path.join(folder("clipframes"), key + ".jpg")
    if os.path.isfile(target):
        return target
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-ss", "%.3f" % at[1], "-i", rec["path"]]
    for f in files:
        cmd += _picture_input(f, step)
    part = _part(target)
    script = part + ".graph"
    with open(script, "w", encoding="utf-8", newline="\n") as f:
        f.write(graph)
    cmd += ["-/filter_complex", script, "-map", "[fin]", "-frames:v", "1", "-q:v", "3", "-f", "image2", part]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=FRAME_SECONDS)
    except (subprocess.SubprocessError, OSError) as e:
        os.remove(part)
        raise NotDrawn("FFmpeg could not draw the frame (%s)." % e)
    finally:
        os.remove(script)
    if done.returncode != 0 or not os.path.getsize(part):
        os.remove(part)
        raise NotDrawn("FFmpeg could not draw the frame: %s" % (done.stderr.strip().splitlines() or ["no reason"])[-1])
    settle(part, target)
    return target


# ---- the finished clip ---------------------------------------------------------------------------------------

def _sound(pieces, out="snd", level=""):
    """Graph text from the first sound track of each piece's input to [out], the pieces joined, then level (the
    second loudnorm pass, or nothing). Each input is opened at its piece's start, so its timestamps start at 0."""
    return "%sconcat=n=%d:v=0:a=1,aresample=%d%s[%s]" % ("".join("[%d:a:0]" % k for k in range(len(pieces))),
                                                          len(pieces), RATE, level, out)


def _inputs(rec, pieces, sound_only=False):
    """The recording opened once per piece, at the piece's start and for its length, so a piece deep in a long
    recording is reached by seeking, not by decoding everything before it."""
    cmd = []
    for a, b in pieces:
        cmd += ["-ss", "%.3f" % a, "-t", "%.3f" % (b - a)] + (["-vn"] if sound_only else []) + ["-i", rec["path"]]
    return cmd


def loudness(rec, pieces, muted=""):
    """How loud the joined pieces are, measured for loudnorm's second pass ({"input_i", "input_tp", "input_lra",
    "input_thresh", "target_offset"}), or None when the recording has no sound or the pieces are silent. muted is
    _muted()'s filter, so what is measured is what is heard."""
    if not rec.get("sound_tracks"):
        return None
    return _measured(_inputs(rec, pieces, True), _sound(pieces, level=muted), "the clip's sound")


def music_loudness(music, length):
    """How loud the music is over the part the clip plays (from its second music["from"], for length seconds, from
    its start again when it is shorter), as loudness() measures it, or None when that part is silent."""
    return _measured(_music_input(music), "[0:a:0]%s[snd]" % _music_part(music, length), "the music")


def _measured(inputs, graph, what):
    """loudnorm's first-pass numbers for graph (text from the inputs to [snd]), or None when it is silent."""
    graph += ";[snd]loudnorm=I=%s:TP=%s:LRA=%s:print_format=json[o]" % (LUFS, PEAK, RANGE)
    cmd = [_ffmpeg(), "-hide_banner", "-nostats"] + inputs + ["-filter_complex", graph, "-map", "[o]", "-f", "null",
                                                              "-"]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=EXPORT_SECONDS)
    except (subprocess.SubprocessError, OSError) as e:
        raise NotDrawn("FFmpeg could not measure %s (%s)." % (what, e))
    found = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", done.stderr)
    if done.returncode != 0 or not found:
        raise NotDrawn("FFmpeg could not measure %s: %s" % (what, (done.stderr.strip().splitlines()
                                                                   or ["no reason"])[-1]))
    measured = json.loads(found.group(0))
    try:
        if not math.isfinite(float(measured["input_i"])):
            return None
    except (KeyError, ValueError):
        return None
    return measured


def _level(measured):
    """The second loudnorm pass: the same gain for the whole clip (linear), so the sound is not pumped."""
    if not measured:
        return ""
    return (",loudnorm=I=%s:TP=%s:LRA=%s:measured_I=%s:measured_TP=%s:measured_LRA=%s:measured_thresh=%s:offset=%s:"
            "linear=true,aresample=%d" % (LUFS, PEAK, RANGE, measured["input_i"], measured["input_tp"],
                                          measured["input_lra"], measured["input_thresh"], measured["target_offset"],
                                          RATE))


def _muted(state):
    """Filter text, from a comma, that silences the clip's own sound over state['mutes'] ([[clip a, clip b]]), each
    edge easing over MUTE_FADE s inside the stretch so it does not click (none at the clip's start or end); "" when
    nothing is muted."""
    spans, length = state.get("mutes") or [], state["length"]
    if not spans:
        return ""
    gains = []
    for a, b in spans:
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and 0 <= a < b):
            raise ValueError("mute %r is not a stretch of the clip: give [start, end] in clip seconds." % ([a, b],))
        rise = "clip((t-%.3f)/%g,0,1)" % (a, MUTE_FADE) if a > project.SLIVER else "1"
        fall = "clip((%.3f-t)/%g,0,1)" % (b, MUTE_FADE) if b < length - project.SLIVER else "1"
        gains.append("(1-%s*%s)" % (rise, fall))
    return ",asetnsamples=n=%d:p=0,volume=volume='%s':eval=frame" % (MUTE_FRAME, "*".join(gains))


def _music(state):
    """The clip's music entry, checked, or None when the clip has no music."""
    music = state.get("music")
    if not music:
        return None
    path = music.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        raise ValueError("music %r is not a file: give the path of a music or sound file that exists." % (path,))
    start = music.get("from", 0)
    if not isinstance(start, (int, float)) or isinstance(start, bool) or not 0 <= start < math.inf:
        raise ValueError("music from %r is not a second of the music: use 0 or more." % (start,))
    level = music.get("level", MUSIC_LEVEL)
    low, high = MUSIC_LEVELS
    if not isinstance(level, (int, float)) or isinstance(level, bool) or not low <= level <= high:
        raise ValueError("music level %r is not a music level: use %d to %d dB against the voice, such as %d."
                         % (level, low, high, MUSIC_LEVEL))
    if state.get("speech") is None:
        raise ValueError("The music needs the times someone speaks (state['speech']) to go down under the voice: "
                         "give the heard phrases' [start, end] in clip seconds, or [] when nobody speaks.")
    return dict(music, path=path, level=level, duck=music.get("duck", MUSIC_DUCK), **{"from": start})


def _music_input(music):
    """The music file opened so it plays on from its start again when it runs out."""
    return ["-stream_loop", "-1", "-vn", "-i", music["path"]]


def _music_part(music, length):
    return "atrim=start=%.3f:duration=%.3f,asetpts=PTS-STARTPTS" % (music["from"], length)


def _mix(state, voice, length, music_input, music_measured, out):
    """Graph text to [out]: the clip's sound ([voice], or silence when the recording has none) with the sounds added
    and the music under it. The music plays music["level"] dB under the voice's -14 LUFS, and music["duck"] dB lower
    while someone speaks; it fades in over its first 0.3 s and out over the clip's last 1.5 s."""
    if voice:
        parts = ["[voice]aformat=sample_rates=%d:channel_layouts=stereo[mix_v]" % RATE]
    else:
        parts = ["aevalsrc=exprs=0:s=%d:c=stereo:d=%.3f[mix_v]" % (RATE, length)]
    music = _music(state)
    sounds = [effects.sound(s["kind"], s["at"], s.get("level", 0)) for s in state.get("sounds") or []]
    parts.append(effects.join(sounds, "mix_v", "mix_s" if music else out, 0)[0])
    if music:
        if not music_measured:
            raise ValueError("The music %s is silent from its second %s: choose another file or another start."
                             % (os.path.basename(music["path"]), music["from"]))
        fade = min(MUSIC_OUT, length / 3)
        parts.append("[%d:a:0]%s,aformat=sample_rates=%d:channel_layouts=stereo,afade=t=in:d=%s,"
                     "afade=t=out:st=%.3f:d=%s,volume=%.2fdB[mix_m]"
                     % (music_input, _music_part(music, length), RATE, MUSIC_IN, length - fade, fade,
                        LUFS + music["level"] - float(music_measured["input_i"])))
        parts.append(effects.join([effects.duck(state["speech"], music["duck"])], "mix_m", "mix_d", 0)[0])
        parts.append("[mix_s][mix_d]amix=inputs=2:duration=first:normalize=0[%s]" % out)
    return ";".join(parts)


def _mix_gain(rec, state, measured, music_measured):
    """The gain in dB that brings the whole mix (the levelled voice, the sounds and the music) to -14 LUFS, measured
    on the same mix the export makes, so music filling the quiet between words does not leave the clip quieter or
    louder than every other clip; 0 when the mix is silent. It is read by FFmpeg's ebur128 meter, which reads like
    the meters people check a clip with: loudnorm's own reading of the same mix was 0.7 LU lower (measured 10-27)."""
    pieces, length = state["clip"]["pieces"], state["length"]
    voice = rec.get("sound_tracks")
    music = _music(state)
    inputs = (_inputs(rec, pieces, True) if voice else []) + (_music_input(music) if music else [])
    graph = _mix(state, voice, length, len(pieces) if voice else 0, music_measured, "snd")
    if voice:
        graph = _sound(pieces, "voice", _muted(state) + _level(measured)) + ";" + graph
    cmd = [_ffmpeg(), "-hide_banner", "-nostats"] + inputs + [
        "-filter_complex", graph + ";[snd]ebur128=framelog=quiet[o]", "-map", "[o]", "-f", "null", "-"]
    what = "the clip's sound with its sounds and music"
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=EXPORT_SECONDS)
    except (subprocess.SubprocessError, OSError) as e:
        raise NotDrawn("FFmpeg could not measure %s (%s)." % (what, e))
    found = re.search(r"Integrated loudness:\s+I:\s+(-?[\d.]+) LUFS", done.stderr)
    if done.returncode != 0 or not found:
        raise NotDrawn("FFmpeg could not measure %s: %s" % (what, (done.stderr.strip().splitlines()
                                                                   or ["no reason"])[-1]))
    whole = float(found.group(1))
    return LUFS - whole if whole > SILENT else 0.0


def base(rec, state, on_progress=None, on_start=None, default="clean"):
    """The page's video: the clip without the words over the picture (layer.words() draws those), at BASE_SIZE,
    made by export() with x264 BASE_PRESET at BASE_CRF and kept in work/bases, named by what it shows (the
    recording, the pieces, the drawing without words, the sounds and music). A change to the words alone gives the
    same name, so the same file comes back at once; only a change to the picture or the sound makes it again.
    Measured 10-28 on four 45 s clips (game, talk, explain card, montage): 8.5 to 16.5 s to make it, about what the
    finished export takes, since the drawing at 1080x1920 is the cost, not the encoder (SSIM 0.97 to 0.995 against the
    full-size picture)."""
    clip = state["clip"]
    if not clip:
        raise NotDrawn("There is no clip yet, so there is nothing to make.")
    try:
        drawn, files = after_source(state, "src", "fin", len(clip["pieces"]), default, words=False)
    except ValueError as e:
        raise NotDrawn(str(e))
    key = hashlib.sha1(json.dumps([BASES, rec["path"], rec["size"], rec["mtime"], clip["pieces"], drawn, files,
                                   state.get("music"), state.get("sounds"), state.get("speech"), state.get("mutes"),
                                   BASE_SIZE, BASE_PRESET, BASE_CRF], ensure_ascii=False, sort_keys=True, default=str)
                       .encode("utf-8")).hexdigest()[:16]
    path = os.path.join(folder("bases"), key + ".mp4")
    if os.path.isfile(path):
        return path
    return export(rec, state, path, on_progress, on_start, default, words=False, size=BASE_SIZE, preset=BASE_PRESET,
                  crf=BASE_CRF)


def export(rec, state, out_path, on_progress=None, on_start=None, default="clean", words=True, size=None,
           preset=PRESET, crf=CRF):
    """Makes the finished clip at out_path and returns out_path once the file is there: H.264 and AAC, 30 fps,
    1080x1920, the sound at -14 LUFS, the index at the front of the file so it plays while it downloads. The pieces
    are joined, then drawn by the same stages as the preview frame; the sounds (state['sounds']: {"kind", "at",
    "level"}) and the music (state['music']: {"path", "from", "level", "duck"}, under the voice and lower while
    someone speaks, state['speech']: [[start, end]] in clip seconds) are added to the levelled sound. The file is
    written next to out_path and only replaces it when FFmpeg has finished. on_progress(seconds done) follows FFmpeg;
    on_start(stop) is given a function that ends FFmpeg at once, after which export raises Stopped and leaves no
    file. words=False leaves the words over the picture out (see after_source), size (w, h) scales the finished
    frame down to that size, and preset / crf are x264's: base() uses all three for the page's video."""
    clip = state["clip"]
    if not clip:
        raise NotDrawn("There is no clip yet, so there is nothing to make.")
    pieces, length = clip["pieces"], state["length"]
    count = len(pieces)
    sound = rec.get("sound_tracks")
    try:
        drawn, files = after_source(state, "src", "full" if size else "fin", count, default, words)
        if size:
            drawn += ";[full]scale=%d:%d:flags=area[fin]" % tuple(size)
        music = _music(state)
        added = bool(music or state.get("sounds"))
        music_measured = music_loudness(music, length) if music else None
        mixed = _mix(state, sound, length, count + len(files), music_measured, "mix") if added else ""
        muted = _muted(state) if sound else ""
    except ValueError as e:
        raise NotDrawn(str(e))
    measured = loudness(rec, pieces, muted) if sound else None
    streams = "".join("[%d:v:0]" % k + ("[%d:a:0]" % k if sound else "") for k in range(count))
    joined = "%sconcat=n=%d:v=1:a=%d[cat]%s;[cat]fps=%d[src]" % (
        streams, count, 1 if sound else 0, "[catsnd]" if sound else "", FPS)
    if sound:
        joined += ";[catsnd]aresample=%d%s%s[%s]" % (RATE, muted, _level(measured), "voice" if added else "snd")
    if added:
        # the limiter, so a sound landing on a loud word never goes over PEAK
        joined += ";%s;[mix]volume=%.2fdB,alimiter=limit=%.4f:level=0:latency=1[snd]" % (
            mixed, _mix_gain(rec, state, measured, music_measured), 10 ** (PEAK / 20))
    graph = joined + ";" + drawn
    work = folder("exports")
    script = os.path.join(work, hashlib.sha1(graph.encode("utf-8")).hexdigest()[:16] + ".graph")
    with open(script, "w", encoding="utf-8", newline="\n") as f:
        f.write(graph)
    part = out_path + ".part"
    log_path = os.path.join(work, os.path.basename(out_path) + ".log")
    cmd = [_ffmpeg(), "-hide_banner", "-nostats", "-y", "-progress", "pipe:1"] + _inputs(rec, pieces)
    for f in files:
        cmd += _picture_input(f)
    if music:
        cmd += _music_input(music)
    cmd += ["-/filter_complex", script, "-map", "[fin]"] + (["-map", "[snd]"] if sound or added else []) + [
        "-t", "%.3f" % length, "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", str(RATE), "-movflags", "+faststart", "-f", "mp4",
        part]
    stopped = threading.Event()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL, text=True,
                                    encoding="utf-8", errors="replace")
        except OSError as e:
            raise NotDrawn("FFmpeg could not be started to make the clip (%s)." % e)

        def stop():
            stopped.set()
            try:
                proc.kill()
            except OSError:
                pass

        if on_start:
            on_start(stop)
        with proc.stdout as progress:
            for line in progress:
                key, _, value = line.strip().partition("=")
                if key == "out_time_us" and on_progress and value.isdigit():
                    on_progress(min(length, int(value) / 1e6))
        proc.wait()
    if stopped.is_set() or proc.returncode != 0 or not os.path.isfile(part) or os.path.getsize(part) == 0:
        if os.path.isfile(part):
            os.remove(part)
        if stopped.is_set():
            raise Stopped("Stopped: the clip was not made.")
        with open(log_path, encoding="utf-8", errors="replace") as f:
            last = [x for x in f.read().strip().splitlines() if x.strip()] or ["no reason"]
        raise NotDrawn("FFmpeg could not make the clip: %s" % last[-1])
    os.replace(part, out_path)
    if on_progress:
        on_progress(length)
    return out_path
