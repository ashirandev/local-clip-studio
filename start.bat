@echo off
cd /d "%~dp0"
if not defined LCS_PORT set "LCS_PORT=8899"
if not exist "%~dp0python\python.exe" (
  echo  The studio is not set up yet. Run setup.bat first.
  pause
  exit /b 1
)
start "Local Clip Studio - close this window to stop the studio" /min "%~dp0python\python.exe" -X utf8 app.py
powershell -NoProfile -ExecutionPolicy Bypass -Command "for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:%LCS_PORT%/api/parts | Out-Null; break } catch { Start-Sleep 1 } }; Start-Process 'http://127.0.0.1:%LCS_PORT%/'"
