"""Whether Meta's models may see and hear pieces of the person's videos without being asked each time: one switch in
Options, under the API key, kept in work/sends.json. It is off until the person turns it on, says yes at setup, or saves
a Meta key in Options without having chosen before (Tarm 11-02, "เอา"); off, once chosen, stays off.

Off, nothing of the person's goes to Meta from Muse on this computer unless they ask for it in their own words: Muse
Wonder watching a piece with its sound, Muse Image drawing a cover from a frame. The hearing never sends sound by
itself, whichever Muse runs: Muse Voice Transcribe hears only with the switch on (or LCS_HEARING=meta). With Muse on
Meta's servers the pictures already go there, so watching and the cover need no switch.
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


def chosen():
    """True when the person has chosen, on or off."""
    return os.path.exists(_setting())


def on_with_a_key():
    """A Meta key saved in Options turns the switch on for a person who never chose; True when it did, and the page says
    so at that moment, with what is sent and who pays. allowed() stays off with no choice: setup still asks and shows
    what is sent, and a run on a key file, which nobody said yes to, bills nothing unasked."""
    if chosen():
        return False
    allow(True)
    return True


def allow(on):
    path = _setting()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".part"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump({"on": bool(on)}, f)
    os.replace(temp, path)
