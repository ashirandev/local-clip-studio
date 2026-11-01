"""A phone that sends photos and videos to the studio, over the same Wi-Fi or over Tailscale.

The studio listens on this computer's Wi-Fi address (the one its traffic to the internet leaves by) and on its
Tailscale addresses (100.64.0.0/10) too, and on nothing else: not on the virtual networks Windows makes for WSL and
Hyper-V. A phone on the same Wi-Fi or the same Tailscale opens the studio at one of those addresses and can use it
as the computer does, so several people can try it at once (Tarm, 10-29: it is an open-source demo; whoever puts an
API key in and shares the Wi-Fi decides that for themselves). Phone: Off in Options keeps the studio to this
computer; the choice is kept in work/phone.json, and the studio starts with Phone on.

What a phone sends goes into the From phone folder in the studio's folder (LCS_INBOX names another), in a folder of
its own for each time it sends, so the clips sent together open together. Each file is written as name.part and gets
its name only when every byte the phone said it would send has arrived; a cut-off send leaves nothing behind.
"""
import ipaddress
import json
import os
import re
import shutil
import socket
import time

from . import recording

TAILSCALE = ipaddress.ip_network("100.64.0.0/10")
CHUNK = 1 << 20
SPARE = 1 << 30
PHOTO_TYPES = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")
BATCH = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}\.\d{2}\.\d{2}$")
NOT_ALLOWED = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class NotSent(Exception):
    """A file could not be kept. The sentence says why and what to do."""


def _setting():
    return os.path.join(recording.WORK, "phone.json")


def wanted():
    """False only when the person turned Phone off."""
    try:
        with open(_setting(), encoding="utf-8") as f:
            return bool(json.load(f).get("on", True))
    except (OSError, ValueError):
        return True


def want(on):
    path = _setting()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".part"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump({"on": bool(on)}, f)
    os.replace(temp, path)


def addresses():
    """[(address, kind)]: this computer's Wi-Fi address, then its Tailscale addresses."""
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            found.append((s.getsockname()[0], "Wi-Fi"))
    except OSError:
        pass
    try:
        every = {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        every = set()
    found += [(a, "Tailscale") for a in sorted(every) if ipaddress.ip_address(a) in TAILSCALE]
    out = []
    for address, kind in found:
        ip = ipaddress.ip_address(address)
        if ip in TAILSCALE:
            kind = "Tailscale"
        if not ip.is_loopback and not ip.is_unspecified and address not in [a for a, _ in out]:
            out.append((address, kind))
    return out


def kind_of(name):
    low = name.lower()
    if low.endswith(recording.VIDEO_TYPES):
        return "video"
    if low.endswith(PHOTO_TYPES):
        return "photo"
    return None


def clean_name(name):
    """The file's own name, without folders or the characters Windows refuses."""
    base = NOT_ALLOWED.sub("", (name or "").replace("\\", "/").rsplit("/", 1)[-1]).strip(" .")
    stem, ext = os.path.splitext(base)
    return (stem[:100] or "from phone") + ext.lower()


def receive(read, length, name, batch, when=None):
    """Keeps a file the phone sends: length bytes from read(n), named name, in the folder for batch (the time the
    phone started sending, YYYY-MM-DD HH.MM.SS). Returns {path, folder, name, kind}; raises NotSent."""
    name = clean_name(name)
    kind = kind_of(name)
    if kind is None:
        raise NotSent("%s is not a photo or a video the studio reads (%s)." % (name, ", ".join(
            recording.VIDEO_TYPES + PHOTO_TYPES)))
    if not BATCH.match(batch or ""):
        raise NotSent("The phone did not say when it started sending. Reload the page and send again.")
    if not isinstance(length, int) or length <= 0:
        raise NotSent("%s came with no size. Send it again." % name)
    folder = os.path.join(recording.INBOX, batch)
    os.makedirs(folder, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if length + SPARE > free:
        raise NotSent("%s is %.1f GB and %s has %.1f GB free. Make room on that drive and send it again."
                      % (name, length / 1e9, recording.INBOX, free / 1e9))
    while True:
        dest = _free_name(folder, name)
        part = dest + ".part"
        try:
            f = open(part, "xb")
            break
        except FileExistsError:
            continue
    got = 0
    try:
        with f:
            while got < length:
                piece = read(min(CHUNK, length - got))
                if not piece:
                    break
                f.write(piece)
                got += len(piece)
    except OSError as e:
        _remove(part)
        raise NotSent("%s stopped after %d of %d bytes (%s). Send it again." % (name, got, length, e))
    if got != length:
        _remove(part)
        raise NotSent("%s stopped after %d of %d bytes. Send it again." % (name, got, length))
    os.replace(part, dest)
    if isinstance(when, (int, float)) and 0 < when < time.time() + 86400:
        os.utime(dest, (when, when))
    return {"path": dest, "folder": folder, "name": os.path.basename(dest), "kind": kind}


def room():
    """The size a file sent in may be, in bytes, leaving SPARE free on its drive. Below 0 when there is less than SPARE."""
    os.makedirs(recording.INBOX, exist_ok=True)
    return shutil.disk_usage(recording.INBOX).free - SPARE


def _free_name(folder, name):
    stem, ext = os.path.splitext(name)
    dest, n = os.path.join(folder, name), 2
    while os.path.exists(dest) or os.path.exists(dest + ".part"):
        dest = os.path.join(folder, "%s (%d)%s" % (stem, n, ext))
        n += 1
    return dest


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
