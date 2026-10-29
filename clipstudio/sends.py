"""Whether Meta's models may see and hear pieces of the person's videos without being asked each time: one switch in
Options, under the API key, off until the person turns it on, kept in work/sends.json.

Off, nothing of the person's goes to Meta from Muse on this computer unless they ask for it in their own words: Muse
Wonder watching a piece with its sound, Muse Image drawing a cover from a frame. The hearing never sends sound by
itself, whichever Muse runs: Muse Voice Transcribe hears only with the switch on (or LCS_HEARING=meta) and the hearing
model not on this computer. With Muse on Meta's servers the pictures already go there, so watching and the cover need
no switch.
"""
import json
import os

from . import recording


def _setting():
    return os.path.join(recording.WORK, "sends.json")


def allowed():
    """True only when the person turned the switch on."""
    try:
        with open(_setting(), encoding="utf-8") as f:
            return json.load(f).get("on") is True
    except (OSError, ValueError, AttributeError):
        return False


def allow(on):
    path = _setting()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".part"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump({"on": bool(on)}, f)
    os.replace(temp, path)
