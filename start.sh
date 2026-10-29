#!/bin/sh
# Starts Local Clip Studio and opens it in the browser. Stop it with Ctrl+C.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo " The studio is not set up yet. Run ./setup.sh first."
  exit 1
fi
LCS_PORT="${LCS_PORT:-8899}"
export LCS_PORT
.venv/bin/python -X utf8 app.py &
studio=$!
trap 'kill $studio 2>/dev/null' INT TERM
i=0
while [ $i -lt 60 ]; do
  if python3 -c "import urllib.request, sys; urllib.request.urlopen('http://127.0.0.1:$LCS_PORT/api/parts', timeout=2)" 2>/dev/null; then
    page="http://127.0.0.1:$LCS_PORT/"
    if command -v open >/dev/null 2>&1; then open "$page"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$page"; else echo " Open $page"; fi
    break
  fi
  i=$((i + 1))
  sleep 1
done
wait $studio
