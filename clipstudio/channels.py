"""The looks the person saves for their channels, so every clip of a channel looks like it came from that channel: the
words' style (font, size, colours), the colour look, how the captions show or that they stay off, where the title
goes, the channel's name drawn on the clip, and for each kind of clip it was saved from its layout and its length.
Each look is kept in work/channels as one file.

The studio makes clips for the one person at this computer, so the look in use is one setting for the whole studio,
kept across chats and restarts (work/channels/in_use.json). A new clip starts in it; what the person changes on a
clip afterwards wins (tools._channel_on).
"""
import hashlib
import json
import os
import tempfile
import time

from . import recording

CAPTIONS = ("plain", "lit", "pop", "off")
SPOTS = ("bottom left", "bottom middle", "bottom right")
LONGEST_NAME = 40
LONGEST_HANDLE = 30
IN_USE = "in_use.json"


def folder():
    return os.path.join(recording.WORK, "channels")


def _path(name):
    return os.path.join(folder(), hashlib.sha1(name.encode("utf-8")).hexdigest()[:16] + ".json")


def _write(path, data):
    """Writes data as JSON to path through a file beside it, so a crash never leaves half a file."""
    os.makedirs(folder(), exist_ok=True)
    fd, part = tempfile.mkstemp(suffix=".part", dir=folder())
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(part, path)
    finally:
        if os.path.isfile(part):
            os.remove(part)


def listed():
    """Every saved look, by name: [{"name", "handle", "style", "look", "captions", "title_place", "spot", "saved"}]."""
    out = []
    for n in os.listdir(folder()) if os.path.isdir(folder()) else []:
        if not n.endswith(".json") or n == IN_USE:
            continue
        try:
            with open(os.path.join(folder(), n), encoding="utf-8") as f:
                out.append(json.load(f))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda c: c["name"].lower())


def get(name):
    return next((c for c in listed() if c["name"] == name), None)


def save(look):
    """Keeps look under its name, replacing a look of the same name, and returns it as kept. ValueError with a
    sentence when a part of it is not one a look can have."""
    name = " ".join(str(look.get("name") or "").split())
    if not 1 <= len(name) <= LONGEST_NAME:
        raise ValueError("Give the channel a name of 1 to %d letters." % LONGEST_NAME)
    handle = " ".join(str(look.get("handle") or "").split())
    if len(handle) > LONGEST_HANDLE:
        raise ValueError("The name drawn on the clip is %d letters; keep it to %d, such as @yourchannel."
                         % (len(handle), LONGEST_HANDLE))
    if look.get("spot") not in SPOTS + (None,):
        raise ValueError("spot %r is not a place for the name on the clip: use %s." % (look.get("spot"),
                                                                                     ", ".join(SPOTS)))
    if look.get("captions") not in CAPTIONS + (None,):
        raise ValueError("captions %r is not a way captions show: use %s, or none." % (look.get("captions"),
                                                                                    ", ".join(CAPTIONS)))
    kept = {"name": name, "handle": handle, "style": look.get("style") or None, "look": look.get("look") or None,
            "captions": look.get("captions"), "title_place": look.get("title_place") or None,
            "spot": look.get("spot") or None, "layouts": look.get("layouts") or {},
            "lengths": look.get("lengths") or {}, "saved": round(time.time())}
    _write(_path(name), kept)
    return kept


def remove(name):
    """Takes the look off this computer: True when there was one. The look in use stops being in use."""
    path = _path(name)
    if not os.path.isfile(path):
        return False
    os.remove(path)
    if (in_use() or {}).get("name") == name:
        use(None)
    return True


def in_use():
    """The look new clips start in, or None."""
    try:
        with open(os.path.join(folder(), IN_USE), encoding="utf-8") as f:
            name = json.load(f).get("name")
    except (OSError, ValueError):
        return None
    return get(name) if name else None


def use(name):
    """Makes name's look the one new clips start in (None: no look). ValueError when no look has that name."""
    if name is not None and get(name) is None:
        raise ValueError("There is no channel look called %s. Saved looks: %s." % (
            name, ", ".join(c["name"] for c in listed()) or "none yet"))
    _write(os.path.join(folder(), IN_USE), {"name": name})
