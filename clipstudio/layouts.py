"""The four layouts of the vertical clip, each drawn by the program: Muse chooses which one and what part of the
picture it shows, and the program works out every pixel.

Free placement was measured first (10-26, four recordings): Muse found the right regions but cut heads, showed the
face camera twice and covered the chest with the game. So the shapes are fixed:

- card: the game as a rounded card across the middle with a soft shadow, the face camera in a ringed circle above
  it, and the same moment, blurred and darkened toward the edges, behind both
- stack: the face camera on top, the game below it, a dark seam and a shadow between them
- face: the face alone, filling the frame
- subject: one part of the picture (a product, the action) filling the frame

A layout is a plain dict, kept in the clip's project list, so the same numbers draw the preview frame and the
finished clip. Boxes are (x, y, w, h). The source boxes are in pixels of the recording, and the frame boxes are in
pixels of the 1080x1920 clip. Everything drawn around the parts (background, shadows, ring, seam) sits under them or
outside them, so to_frame() stays true: a part's picture is shown whole at its frame box, only the card's rounded
corners (CARD_RADIUS px) are cut.

What a phone covers was measured on Tarm's iPhone (Instagram and Facebook Reels, 10-27, in clip pixels): both apps
fill the screen and cut about 51 px off each side; the status bar and the app's header cover the top 236 px; the
name, a friend's avatar and the caption cover everything below y 1552; the like, comment and share buttons cover
x 917 to 1001 from y 1098 to 1855. SAFE and BUTTONS keep a margin around that, and layout['safe'] says it for Muse.
Words, the card, the stack's game and a subject stay inside SAFE, and words stay clear of BUTTONS too (the card's
lower right edge sits under the buttons, as a full-width picture must). An ad is stricter: Meta's ad numbers keep
the bottom 670 px free for the ad's button, so a clip that will be boosted keeps its words above y 1250.

A place is never on the face: the stack has no place at the top, because its top panel is the face; its words go
on the seam.
"""
import copy
import math
import os
import shutil
import subprocess

from . import effects, recording

W, H = 1080, 1920
KINDS = ("card", "stack", "face", "subject")
SAID = {
    "card": "the picture as a card across the middle, the face in a circle above it, the title on top; for a game "
            "stream with a face camera",
    "stack": "the face on top, the picture below it; for a stream when the face matters as much as the game",
    "face": "the face alone, filling the frame; for a reaction",
    "subject": "one part of the picture filling the frame, such as the product or the action; for a person on camera "
               "or a product",
}
TOP = [65, 285, 950, 235]
MIDDLE = [65, 905, 830, 170]
CAPTIONS = [65, 1090, 830, 160]
CIRCLE, RING = 400, 8
CIRCLE_Y = 540
CARD_Y = 890
CARD_X = 60
CARD_W = W - 2 * CARD_X
CARD_RADIUS = 30
CARD_LOWEST, CARD_TALLEST = 480, 630
STACK_TOP = 760
SEAM = 8
GROUND = "0x101014"
FACE_SHARE = {"circle": 1.5, "stack": 2.0, "face": 4.0}
HEAD_UP, HEAD_SIDE = 0.4, 0.15
GAP = 24
UNDER = 200
SAFE_BOTTOM = 1520
SAFE = [56, 250, 968, SAFE_BOTTOM - 250]
BUTTONS = [900, 1080, 180, 780]
SAFE_SAID = ("Phones cover the clip's top 250 px (the status bar and the app's header), everything below y %d (the "
             "name and the caption), the right side from x %d between y %d and %d (like, comment, share), and cut "
             "51 px off each side. When you choose, keep words, stickers and faces inside x %d to %d, y %d to %d, and "
             "left of x %d below y %d. When the person asks for a spot, put it there and tell them what a phone "
             "covers." % (SAFE_BOTTOM, BUTTONS[0], BUTTONS[1], BUTTONS[1] + BUTTONS[3], SAFE[0], SAFE[0] + SAFE[2],
                          SAFE[1], SAFE_BOTTOM, BUTTONS[0], BUTTONS[1]))
SMALLEST_PART = 120
BAND = {"saturation": 1.1, "brightness": -0.12, "sigma": 14}
LEAVE_OUT = 3.0
SMALLEST_CARD = 0.4
SHARPEN_FROM = 1.5
SHARPEN = "unsharp=5:5:0.7:5:5:0"
SHADOW = {"card": (22, 18, 0.55), "circle": (14, 10, 0.5)}
SHAPES = 2


def make(kind, size, face=None, crop=None):
    """The layout of kind for a recording of size (w, h): its parts, its places for words, its caption band and
    notes on what the program fixed. face is the face's box in the recording (from the face finder), crop the part
    of the picture to show (whole picture when None). A layout that cannot be drawn raises ValueError."""
    if kind not in KINDS:
        raise ValueError("layout %r is not a layout: use one of %s." % (kind, ", ".join(KINDS)))
    w, h = size
    crop = _inside(crop or [0, 0, w, h], w, h)
    notes, parts = [], []
    if h > w and kind in ("card", "stack"):
        notes.append("The recording is already vertical, so its picture fills the frame (the subject layout) "
                     "instead of a %s, which would show the face twice." % kind)
        kind = "subject"
    room = [0, 0, w, h]
    away = _grown(face, LEAVE_OUT, w, h) if face else None
    if kind == "card":
        if face:
            square = _around(face, FACE_SHARE["circle"], 1.0, room)
            parts.append({"name": "circle", "src": square, "dst": [(W - CIRCLE) // 2, CIRCLE_Y, CIRCLE, CIRCLE]})
            crop = _leave_out(crop, away, notes)
        else:
            notes.append("No face was given or found, so the card has no circle.")
        crop, height = _card(crop, notes)
        parts.insert(0, {"name": "card", "src": crop, "dst": [CARD_X, CARD_Y, CARD_W, height]})
        places = {"top": TOP, "middle": MIDDLE}
    elif kind == "stack":
        if not face:
            raise ValueError("The stack layout needs a face: none was found in the moment's pictures. Give face as a "
                             "box on your 0-1000 grid, or choose the subject layout.")
        top_h = STACK_TOP - SEAM // 2
        bottom_y = STACK_TOP + SEAM // 2
        bottom_h = SAFE_BOTTOM - SEAM // 2 - bottom_y
        top = _around(face, FACE_SHARE["stack"], W / top_h, room)
        game = _fit(crop, W / bottom_h, [0, 0, w, h])
        game = _leave_out(game, away, notes, size=(w, h))
        parts += [{"name": "top", "src": top, "dst": [0, 0, W, top_h]},
                  {"name": "bottom", "src": game, "dst": [0, bottom_y, W, bottom_h]}]
        places = {"middle": [65, STACK_TOP - 120, 950, 240]}
    elif kind == "face":
        if not face:
            raise ValueError("The face layout needs a face: none was found in the moment's pictures. Give face as a "
                             "box on your 0-1000 grid, or choose the subject layout.")
        part = _around(face, FACE_SHARE["face"], W / H, room)
        parts.append({"name": "whole", "src": part, "dst": [0, 0, W, H]})
        places, captions = _around_the_head(_head(face, part), notes)
    else:
        part = _in_safe(crop, [0, 0, w, h])
        parts.append({"name": "whole", "src": part, "dst": [0, 0, W, H]})
        places = {"top": TOP, "middle": MIDDLE}
        k = W / part[2]
        shown = [(crop[0] - part[0]) * k, (crop[1] - part[1]) * k, crop[2] * k, crop[3] * k]
        if crop != [0, 0, w, h] and [why for why in unsafe(shown) if "buttons" not in why]:
            notes.append("The subject is too big to show whole between the phone's header and the name and caption "
                         "at the bottom; choose the card layout to show all of it.")
    return {"kind": kind, "size": [w, h], "face": face, "parts": parts, "places": places,
            "captions": captions if kind == "face" else CAPTIONS, "background": kind == "card", "notes": notes,
            "safe": {"box": SAFE, "buttons": BUTTONS, "said": SAFE_SAID}}


def moved(layout, name, box):
    """A copy of layout with its part `name` drawn at box (x, y, w, h on the 1080x1920 frame), where the person asked:
    the safe zone is where the program puts things by itself, never a limit on what a person asks for. The part
    shows the same thing of the recording, cut around the same middle to the box's shape, so nothing is stretched;
    a circle stays round, as big as the box's shorter side, in the box's middle. A face circle can be added this way
    to a layout without one (the subject layout: the whole picture, the face in a corner). The notes say what a phone
    covers there. Only a box the frame cannot hold raises ValueError."""
    if (not isinstance(box, (tuple, list)) or len(box) != 4
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)):
        raise ValueError("box %r is not a box: give (x, y, w, h) on the %dx%d frame." % (box, W, H))
    x, y, bw, bh = box
    if bw < SMALLEST_PART or bh < SMALLEST_PART:
        raise ValueError("box %r is too small to show a part: make w and h at least %d px." % (tuple(box), SMALLEST_PART))
    if x < 0 or y < 0 or x + bw > W or y + bh > H:
        raise ValueError("box %r runs off the %dx%d frame: keep x + w at most %d and y + h at most %d."
                         % (tuple(box), W, H, W, H))
    lay = copy.deepcopy(layout)
    parts = {p["name"]: p for p in lay["parts"]}
    if name == "circle" and name not in parts:
        if lay["kind"] not in ("subject", "card") or not lay.get("face"):
            raise ValueError("The %s layout cannot take a face circle%s: use it with the subject layout and a face."
                             % (lay["kind"], "" if lay.get("face") else " without a face"))
        w, h = lay["size"]
        parts[name] = {"name": name, "src": _around(lay["face"], FACE_SHARE["circle"], 1.0, [0, 0, w, h])}
        lay["parts"].append(parts[name])
        if any(_overlap(p["src"], _grown(lay["face"], LEAVE_OUT, w, h)) for p in lay["parts"] if p["name"] != name):
            lay["notes"].append("The face camera is inside the picture too, so the face shows twice; choose a crop "
                                "beside it.")
    elif name not in parts:
        raise ValueError("part %r is not in the %s layout: use one of %s." % (name, lay["kind"], ", ".join(
            [p["name"] for p in lay["parts"]] + ([] if "circle" in parts or lay["kind"] != "subject" else ["circle"]))))
    part = parts[name]
    if name == "circle":
        side = 2 * math.floor(min(bw, bh) / 2)
        part["dst"] = _place(x + bw / 2, y + bh / 2, side, side, [0, 0, W, H])
    else:
        part["dst"] = _within_even(box)
        sx, sy, sw, sh = part["src"]
        aspect = part["dst"][2] / part["dst"][3]
        if sw / sh > aspect:
            sx, sw = sx + (sw - sh * aspect) / 2, sh * aspect
        else:
            sy, sh = sy + (sh - sw / aspect) / 2, sw / aspect
        part["src"] = [sx, sy, sw, sh]
    said = "As asked, the %s is at %s." % (name, part["dst"])
    covered = unsafe(part["dst"])
    if covered:
        said += " On a phone, " + "; ".join(covered) + "."
    on = [place for place, where in lay["places"].items() if _overlap(where, part["dst"])]
    if on:
        said += " Words in the %s place%s would sit on it." % (" and ".join(on), "s" if len(on) > 1 else "")
    lay["notes"].append(said)
    return lay


def unsafe(box):
    """What a phone covers of box (x, y, w, h on the 1080x1920 frame): a list of short reasons, empty when every
    pixel of it shows in both apps."""
    x, y, w, h = box
    out = []
    if x < SAFE[0] or x + w > SAFE[0] + SAFE[2]:
        out.append("the clip's sides are cut off (keep x from %d to %d)" % (SAFE[0], SAFE[0] + SAFE[2]))
    if y < SAFE[1]:
        out.append("the status bar and the app's header cover the top (keep y from %d)" % SAFE[1])
    if y + h > SAFE_BOTTOM:
        out.append("the name and the caption cover everything below y %d" % SAFE_BOTTOM)
    if _overlap(box, BUTTONS):
        out.append("the like, comment and share buttons cover x %d and right of it from y %d to %d"
                   % (BUTTONS[0], BUTTONS[1], BUTTONS[1] + BUTTONS[3]))
    return out


def _in_safe(crop, room, grow=1.1):
    """The 9:16 part of room that shows crop, grown by grow, as big as fits inside SAFE and centred in it, so no
    part of the subject is under what a phone draws on top."""
    x, y, cw, ch = crop
    k = min(SAFE[2] / (cw * grow), SAFE[3] / (ch * grow))
    bw, bh = W / k, H / k
    if bw > room[2]:
        bw, bh = room[2], room[2] * H / W
    if bh > room[3]:
        bw, bh = room[3] * W / H, room[3]
    k = W / bw
    middle = SAFE[1] + SAFE[3] / 2
    return _place(x + cw / 2, y + ch / 2 + (H / 2 - middle) / k, bw, bh, room)


def _head(face, part):
    """The head's box on the frame of a whole-frame part: the face finder's box (brows to chin) with the hair above
    it and the ears beside it."""
    k = W / part[2]
    x, y, w, h = (face[0] - part[0]) * k, (face[1] - part[1]) * k, face[2] * k, face[3] * k
    return [x - HEAD_SIDE * w, y - HEAD_UP * h, w * (1 + 2 * HEAD_SIDE), h * (1 + HEAD_UP)]


def _around_the_head(head, notes):
    """(places, caption band) for words clear of the head: the title on top when the head leaves room under it,
    otherwise just under the chin (a recording that is already vertical often has the face high, with nothing above
    it to show), and the captions under the chin or under that title, above the phone apps' own words at the bottom;
    when the face sits too low for that, the captions go between the title and the head."""
    top, chin = head[1], head[1] + head[3]
    tall = CAPTIONS[3]
    if top - GAP >= TOP[1] + TOP[3]:
        places, below = {"top": TOP}, chin + GAP
    else:
        y = _even(chin + GAP)
        places = {"middle": [65, y, 950 if y + UNDER <= BUTTONS[1] else BUTTONS[0] - 65 - 5, UNDER]}
        below = chin + GAP + UNDER + GAP
    y = max(CAPTIONS[1], _even(below))
    if y + tall > SAFE_BOTTOM:
        y = TOP[1] + TOP[3] + GAP
        if "top" not in places or y + tall + GAP > top:
            y = SAFE_BOTTOM - tall
            notes.append("The face fills the picture, so the captions are as low as phone apps leave room for and "
                         "may touch the chin.")
    return places, [65, y, 830, tall]


def _even(v):
    return int(round(v / 2.0)) * 2


def _within_even(box):
    """The largest box on even pixels inside box, so rounding never takes it over what it was kept clear of."""
    x0, y0 = 2 * math.ceil(box[0] / 2), 2 * math.ceil(box[1] / 2)
    x1, y1 = 2 * math.floor((box[0] + box[2]) / 2), 2 * math.floor((box[1] + box[3]) / 2)
    return [x0, y0, max(2, x1 - x0), max(2, y1 - y0)]


def _inside(box, w, h):
    x, y, bw, bh = box
    x0, y0 = max(0.0, min(x, w - 2)), max(0.0, min(y, h - 2))
    x1, y1 = max(x0 + 2, min(x + bw, w)), max(y0 + 2, min(y + bh, h))
    return [x0, y0, x1 - x0, y1 - y0]


def _place(cx, cy, bw, bh, room):
    """A box of about bw x bh centred on (cx, cy) as far as room (x, y, w, h) allows, on even pixels inside it."""
    lx, ly, rx, ry = _within_even(room)
    rx, ry = lx + rx, ly + ry
    bw, bh = max(2, min(_even(bw), rx - lx)), max(2, min(_even(bh), ry - ly))
    x = min(max(lx, _even(cx - bw / 2)), rx - bw)
    y = min(max(ly, _even(cy - bh / 2)), ry - bh)
    return [x, y, bw, bh]


def _around(face, share, aspect, room, at=0.5):
    """The part of room around face with the face share times smaller than the part's height, aspect
    (width / height) as asked, the face's middle at `at` of the height."""
    fx, fy, fw, fh = face
    bh = share * max(fh, fw)
    bw = bh * aspect
    if bw > room[2]:
        bw, bh = room[2], room[2] / aspect
    if bh > room[3]:
        bw, bh = room[3] * aspect, room[3]
    cx, cy = fx + fw / 2, fy + fh * 0.42 - bh * (at - 0.5)
    return _place(cx, cy, bw, bh, room)


def _fit(crop, aspect, room, grow=1.0):
    """The smallest box of the asked aspect (width / height) that holds crop, grown by grow, inside room."""
    x, y, bw, bh = crop
    bh = max(bh, bw / aspect) * grow
    bw = bh * aspect
    if bw > room[2]:
        bw, bh = room[2], room[2] / aspect
    if bh > room[3]:
        bw, bh = room[3] * aspect, room[3]
    return _place(x + crop[2] / 2, y + crop[3] / 2, bw, bh, room)


def _grown(face, share, w, h):
    """A square around the face, share times its longer side, kept inside the picture."""
    fx, fy, fw, fh = face
    side = share * max(fw, fh)
    return _inside([fx + fw / 2 - side / 2, fy + fh / 2 - side / 2, side, side], w, h)


def _overlap(a, b):
    return min(a[0] + a[2], b[0] + b[2]) > max(a[0], b[0]) and min(a[1] + a[3], b[1] + b[3]) > max(a[1], b[1])


def _leave_out(crop, away, notes, size=None):
    """crop without the face camera (away), so the face does not show twice: cut from the side the face is on, or,
    when size (w, h) is given because the box must keep its shape, slide it sideways inside the picture. Left as it
    is, with a note, when what is left would be too small or no slide clears it."""
    if away is None or not _overlap(crop, away):
        return crop
    x, y, cw, ch = crop
    ax, ay, aw, ah = away
    if size:
        slides = [nx for nx in (ax - cw, ax + aw) if 0 <= nx <= size[0] - cw]
        if slides:
            nx = min(slides, key=lambda v: abs(v - x))
            notes.append("The picture below was moved off the face camera, so the face does not show twice.")
            return [_even(nx) if _even(nx) + cw <= size[0] else int(nx), y, cw, ch]
        notes.append("The face camera is inside the picture too, so the face shows twice; choose a crop beside it.")
        return crop
    options = [[x, y, ax - x, ch], [ax + aw, y, x + cw - ax - aw, ch], [x, y, cw, ay - y], [x, ay + ah, cw, y + ch - ay - ah]]
    options = [o for o in options if o[2] > 0 and o[3] > 0 and not _overlap(o, away)]
    best = max(options, key=lambda o: o[2] * o[3], default=None)
    if best is None or best[2] * best[3] < SMALLEST_CARD * cw * ch:
        notes.append("The face camera is inside the card too, so the face shows twice; choose a crop beside it.")
        return crop
    notes.append("The card leaves out the face camera, so the face does not show twice.")
    return best


def _card(crop, notes):
    """(crop, card height): the card is CARD_W wide; its height follows the crop, from CARD_LOWEST to CARD_TALLEST
    pixels, trimming the crop around its middle when it is too tall or too wide."""
    x, y, cw, ch = crop
    height = CARD_W * ch / cw
    if height > CARD_TALLEST:
        nh = cw * CARD_TALLEST / CARD_W
        y, ch, height = y + (ch - nh) / 2, nh, CARD_TALLEST
        notes.append("The card was trimmed at the top and bottom to fit the frame.")
    elif height < CARD_LOWEST:
        nw = ch * CARD_W / CARD_LOWEST
        x, cw, height = x + (cw - nw) / 2, nw, CARD_LOWEST
        notes.append("The card was trimmed at the sides to fit the frame.")
    return _within_even([x, y, cw, ch]), _even(height)


def to_frame(layout, box):
    """Where a box of the recording shows in the clip's frame: a list of frame boxes, one for each part it is in
    (the blurred background is not counted)."""
    out = []
    for part in layout["parts"]:
        sx, sy, sw, sh = part["src"]
        dx, dy, dw, dh = part["dst"]
        x0, y0 = max(box[0], sx), max(box[1], sy)
        x1, y1 = min(box[0] + box[2], sx + sw), min(box[1] + box[3], sy + sh)
        if x1 <= x0 or y1 <= y0:
            continue
        kx, ky = dw / sw, dh / sh
        out.append([dx + (x0 - sx) * kx, dy + (y0 - sy) * ky, (x1 - x0) * kx, (y1 - y0) * ky])
    return out


# ---- drawing ---------------------------------------------------------------------------------------------------

def graph(layout, src, out, first_input, pushes=()):
    """(filter graph text from [src] to [out] drawing layout at 1080x1920, extra files): the shapes around the parts
    (masks, shadows, the ring, the background's shade) are pictures made once, opened by the renderer as inputs from
    first_input on, in the order of the files.

    pushes are slow push-ins [{start, end, zoom, centre}] (centre on the frame): they push the main part's picture
    (the card, the picture below the face, or the whole frame) inside its own place, so the card's shape and shadow,
    the face circle and the background hold still. A centre outside the main part pushes toward its nearest point."""
    files = []

    def shape(path):
        files.append(path)
        return "[%d:v]" % (first_input + len(files) - 1)

    parts = {p["name"]: p for p in layout["parts"]}
    tail = "setsar=1,format=yuv420p[%s]" % out
    circle = parts.get("circle")
    if circle:
        return ("[%s]split=2[lay_m][lay_sc];%s[lay_base];%s,%s"
                % (src, _base(layout, parts, "lay_m", pushes, shape), _circle(circle, "lay_base", "lay_sc", shape),
                   tail), files)
    return "%s,%s" % (_base(layout, parts, src, pushes, shape), tail), files


def _base(layout, parts, src, pushes, shape):
    """The layout drawn from [src] without its face circle, as graph text whose last filter has no output label."""
    if layout["kind"] in ("face", "subject"):
        return "[%s]%s" % (src, _pushed(parts["whole"], pushes))
    if layout["kind"] == "stack":
        top, bottom = parts["top"], parts["bottom"]
        bx, by, _, bh = bottom["dst"]
        under = by + bh + SEAM
        sx, sy, sw, sh = bottom["src"]
        text = ("[%s]split=3[lay_a][lay_b][lay_c];[lay_a]%s,pad=%d:%d:0:0:color=%s[lay_t];[lay_b]%s[lay_g];"
                "[lay_c]crop=%d:%d:%d:%d,scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,gblur=sigma=%s,"
                "eq=saturation=%s:brightness=%s[lay_u];"
                "[lay_t][lay_g]overlay=%d:%d[lay_1];[lay_1][lay_u]overlay=0:%d[lay_2];"
                "[lay_2]%soverlay=0:%d:shortest=1[lay_3];[lay_3]%soverlay=0:%d:shortest=1"
                % (src, _scaled(top), W, H, GROUND, _pushed(bottom, pushes),
                   sw, sh, sx, sy, W, H - under, W, H - under, BAND["sigma"], BAND["saturation"], BAND["brightness"],
                   bx, by, under, shape(seam_shade()), by, shape(seam_shade()), under))
        return text
    card = parts["card"]
    cx, cy, cw, ch = card["dst"]
    sigma, drop, _ = SHADOW["card"]
    pad = 3 * sigma
    return ("[%s]split=2[lay_s1][lay_s2];"
            "[lay_s1]scale=540:960:force_original_aspect_ratio=increase,crop=540:960,gblur=sigma=9,"
            "eq=saturation=1.3:contrast=1.05:brightness=-0.04,scale=%d:%d[lay_bg0];"
            "[lay_bg0]%soverlay=0:0:shortest=1[lay_bg];"
            "[lay_bg]%soverlay=%d:%d:shortest=1[lay_1];"
            "[lay_s2]%s,format=yuva420p[lay_c0];%sformat=gray[lay_cm];[lay_c0][lay_cm]alphamerge[lay_card];"
            "[lay_1][lay_card]overlay=%d:%d:shortest=1"
            % (src, W, H, shape(shade()), shape(shadow("card", cw, ch, CARD_RADIUS)), cx - pad, cy - pad + drop,
               _pushed(card, pushes), shape(rounded(cw, ch, CARD_RADIUS)), cx, cy))


def _circle(circle, base, src, shape):
    """The face circle, its ring and its shadow drawn over [base] from [src], as graph text whose last filter has no
    output label."""
    dx, dy, d, _ = circle["dst"]
    sigma, drop, _ = SHADOW["circle"]
    pad = 3 * sigma
    ring = d + 2 * RING
    return ("[%s]%soverlay=%d:%d:shortest=1[lay_o1];[lay_o1]%soverlay=%d:%d:shortest=1[lay_o2];"
            "[%s]%s,format=yuva420p[lay_f];%sformat=gray[lay_fm];[lay_f][lay_fm]alphamerge[lay_circle];"
            "[lay_o2][lay_circle]overlay=%d:%d:shortest=1"
            % (base, shape(shadow("circle", ring, ring, ring / 2)), dx - RING - pad, dy - RING - pad + drop,
               shape(ring_shape(ring, RING)), dx - RING, dy - RING,
               src, _scaled(circle), shape(rounded(d, d, d / 2)), dx, dy))


def _pushed(part, pushes):
    """_scaled(part), then the slow pushes on its picture, their centres moved from the frame into the part."""
    x, y, w, h = part["dst"]
    text = _scaled(part)
    for p in pushes:
        cx, cy = effects._centre(p.get("centre", (W / 2, H / 2)))
        inside = (min(max(cx - x, 0), w), min(max(cy - y, 0), h))
        text += "," + effects.push_filters(p["start"], p["end"], p.get("zoom", 1.08), inside, (w, h))
    return text


def _scaled(part):
    """crop and scale for a part, sharpened a little when it is enlarged a lot."""
    x, y, w, h = part["src"]
    dw, dh = part["dst"][2:]
    text = "crop=%d:%d:%d:%d,scale=%d:%d:flags=lanczos" % (w, h, x, y, dw, dh)
    if dw / w >= SHARPEN_FROM:
        text += "," + SHARPEN
    return text


# ---- shapes: pictures made once by FFmpeg and kept in work/shapes ----------------------------------------------

def _round(r, w, h, pad=0):
    """geq expression: 1 inside a w x h rounded rectangle with corner radius r, set pad px into the picture, with a
    one-pixel soft edge, 0 outside."""
    return ("clip(%.2f-hypot(max(0,abs(X+0.5-%.2f)-%.2f),max(0,abs(Y+0.5-%.2f)-%.2f))+0.5,0,1)"
            % (r, pad + w / 2, w / 2 - r, pad + h / 2, h / 2 - r))


def _shape(name, w, h, graph_text):
    """The path of a w x h PNG drawn by graph_text from a black canvas to [o], made once: the name is the cache key,
    so a change to how a shape is drawn raises SHAPES."""
    folder = os.path.join(recording.WORK, "shapes")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "v%d_%s.png" % (SHAPES, name))
    if not os.path.isfile(path):
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=black:s=%dx%d" % (w, h), "-filter_complex", "[0:v]" + graph_text, "-map", "[o]",
                        "-frames:v", "1", "-f", "image2", path + ".part.png"], check=True, capture_output=True)
        os.replace(path + ".part.png", path)
    return path


def rounded(w, h, r):
    """A grey mask, white inside a w x h rounded rectangle (a circle when r is half of a square's side)."""
    return _shape("round%dx%dr%d" % (w, h, r * 10), w, h, "format=gray,geq=lum='255*%s'[o]" % _round(r, w, h))


def ring_shape(d, width):
    """A white ring, d across and width thick, its inside reaching 1 px under the circle it frames."""
    outer, inner = d / 2, d / 2 - width - 1
    return _shape("ring%dw%d" % (d, width), d, d,
                  "format=rgba,geq=r=255:g=255:b=255:a='255*%s*(1-%s)'[o]"
                  % (_round(outer, d, d), _round(inner, 2 * inner, 2 * inner, pad=outer - inner)))


def shadow(kind, w, h, r):
    """A soft black shadow for a w x h rounded shape, blurred by the kind's sigma, 3 sigma of room on each side."""
    sigma, _drop, strength = SHADOW[kind]
    pad = 3 * sigma
    return _shape("shadow_%s%dx%dr%d" % (kind, w, h, r * 10), w + 2 * pad, h + 2 * pad,
                  "split[c][m];[m]format=gray,geq=lum='%d*%s',gblur=sigma=%d[a];[c]format=rgba[k];[k][a]alphamerge[o]"
                  % (round(255 * strength), _round(r, w, h, pad=pad), sigma))


def shade():
    """The card layout's shade over the blurred background: darker toward the top (under the title), the bottom
    and the corners, so the card is what the eye finds first."""
    top = "120*pow(max(0,1-Y/760),1.6)"
    bottom = "150*pow(max(0,(Y-1380)/540),1.6)"
    corners = "90*clip((hypot((X-540)/540,(Y-960)/960)-0.55)/0.6,0,1)"
    return _shape("shade", W, H, "format=rgba,geq=r=0:g=0:b=0:a='min(255,%s+%s+%s)'[o]" % (top, bottom, corners))


def seam_shade():
    """The stack's shadow cast by the face panel onto the game, 56 px deep."""
    return _shape("seam", W, 56, "format=rgba,geq=r=0:g=0:b=0:a='110*pow(1-Y/56,2)'[o]")
