#!/bin/sh
# Local Clip Studio on macOS or Linux, with an API key: Muse Spark on Meta's API, or Muse Spark Open on Modal,
# Baseten or Fireworks. No graphics card is needed. Run it again if anything stops: it carries on.
set -e
cd "$(dirname "$0")"
echo
echo " Local Clip Studio: setting up this computer for Muse through an API (no graphics card needed)."
echo

PY=
for p in python3.12 python3.11 python3; do
  if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(not (3, 11) <= sys.version_info < (3, 13))'; then
    PY=$p
    break
  fi
done
if [ -z "$PY" ]; then
  echo " The studio needs Python 3.11 or 3.12 (its text reader has no build for 3.13 yet)."
  echo "   macOS: brew install python@3.12     Debian or Ubuntu: sudo apt install python3.12 python3.12-venv"
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! ffmpeg -hide_banner -filters 2>/dev/null | grep -q ' ass '; then
  echo " The studio needs FFmpeg built with libass, to draw the words on the clip."
  echo "   macOS: brew install ffmpeg     Debian or Ubuntu: sudo apt install ffmpeg"
  exit 1
fi

echo " [1/3] A Python of the studio's own, in .venv"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
echo " [2/3] The studio's Python packages (about 350 MB the first time)"
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
echo " [3/3] Your API endpoint and key, and the hearing model"
.venv/bin/python -m clipstudio.parts

echo
echo " Done. Start the studio with ./start.sh"
