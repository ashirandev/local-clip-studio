@echo off
setlocal
cd /d "%~dp0"
title Local Clip Studio setup
set "HERE=%~dp0"
set "PY=%HERE%python\python.exe"
set "DL=%HERE%work\tools\downloads"
set "PYZIP=python-3.11.9-embed-amd64.zip"
set "PYURL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "PYSHA=009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b"
set "PIPWHL=pip-24.3.1-py3-none-any.whl"
set "PIPURL=https://files.pythonhosted.org/packages/ef/7d/500c9ad20238fcfcb4cb9243eede163594d7020ce87bd9610c9e02771876/pip-24.3.1-py3-none-any.whl"
set "PIPSHA=3790624780082365f47549d032f3770eeb2b1e8bd1f7b2e02dace1afa361b4ed"

echo.
echo  Local Clip Studio: setting up this computer.
echo  Each file is checked before it is used. If anything stops, run setup.bat again: it carries on.
echo.
if exist "%PY%" goto pip
if exist "%HERE%parts\one_key.txt" goto placed
if defined LCS_PLACED goto placed
echo  Choose where the studio goes in the window that opens. It makes a "Local Clip Studio" folder there
echo  and keeps everything in it: about 18 GB with Muse Glimmer.
set "DEST=%LCS_INSTALL_DIR%"
if not defined DEST for /f "usebackq delims=" %%d in (`powershell -NoProfile -STA -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.Form -Property @{TopMost = $true}; $d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description = 'Where should Local Clip Studio go?'; $d.SelectedPath = $env:HERE; if ($d.ShowDialog($f) -eq 'OK') { $d.SelectedPath }"`) do set "DEST=%%d"
if not defined DEST (
  echo  No folder was chosen, so nothing was installed. Run setup.bat again to choose one.
  pause
  exit /b 1
)
if "%DEST:~-1%"=="\" set "DEST=%DEST:~0,-1%"
if /i "%DEST%"=="%HERE:~0,-1%" goto placed
if /i not "%DEST:~-17%"=="Local Clip Studio" set "DEST=%DEST%\Local Clip Studio"
if /i "%DEST%"=="%HERE:~0,-1%" goto placed
echo  Installing in %DEST%
robocopy "%HERE:~0,-1%" "%DEST%" /E /XD "%HERE%python" "%HERE%work" "%HERE%.git" "%DEST%" __pycache__ /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 goto notcopied
set "LCS_PLACED=1"
call "%DEST%\setup.bat"
exit /b %errorlevel%

:placed
if not exist "%DL%" mkdir "%DL%"
echo  [1/4] Python 3.11.9 for the studio (11 MB)
call :fetch "%PYURL%" "%DL%\%PYZIP%" %PYSHA% || goto failed
if exist "%HERE%python.new" rmdir /s /q "%HERE%python.new"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%DL%\%PYZIP%' -DestinationPath '%HERE%python.new' -Force" || goto failed
(echo python311.zip& echo .& echo ..& echo import site)> "%HERE%python.new\python311._pth"
move "%HERE%python.new" "%HERE%python" >nul || goto failed

:pip
if not exist "%DL%" mkdir "%DL%"
"%PY%" -m pip --version >nul 2>&1 && goto packages
call :fetch "%PIPURL%" "%DL%\%PIPWHL%" %PIPSHA% || goto failed
"%PY%" -c "import runpy, sys; sys.path.insert(0, sys.argv[1]); sys.argv = ['-c', 'install', '--no-index', '--no-warn-script-location', '-q', sys.argv[1]]; runpy.run_module('pip', run_name='__main__')" "%DL%\%PIPWHL%" || goto failed

:packages
echo  [2/4] The studio's Python packages (about 350 MB the first time)
"%PY%" -m pip install --disable-pip-version-check --no-warn-script-location -q -r requirements.txt || goto failed

echo  [3/4] Muse, llama.cpp, FFmpeg and the hearing model
"%PY%" -m clipstudio.parts || goto failed

echo  [4/4] A shortcut on the desktop
if not defined LCS_SHORTCUT_DIR for /f "delims=" %%d in ('powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"') do set "LCS_SHORTCUT_DIR=%%d"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('%LCS_SHORTCUT_DIR%\Local Clip Studio.lnk'); $s.TargetPath = '%HERE%start.bat'; $s.WorkingDirectory = '%HERE%'; $s.WindowStyle = 7; $s.Save()"
echo.
echo  Done. Opening the studio now; next time, open "Local Clip Studio" on the desktop.
if defined LCS_SETUP_NO_START exit /b 0
call "%HERE%start.bat"
exit /b 0

:fetch
if exist "%~2" exit /b 0
curl.exe -L --fail --retry 3 -C - -o "%~2.part" "%~1"
if errorlevel 1 exit /b 1
set "GOT="
for /f %%h in ('powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath '%~2.part').Hash.ToLower()"') do set "GOT=%%h"
if /i not "%GOT%"=="%~3" (
  del "%~2.part"
  echo  %~nx2 did not match the file expected, so it was deleted.
  exit /b 1
)
move /y "%~2.part" "%~2" >nul
exit /b 0

:notcopied
echo  The studio could not be copied to %DEST%. Run setup.bat again and choose a folder you can write in.
pause
exit /b 1

:failed
echo.
echo  Setup stopped. Run setup.bat again: it carries on from where it stopped.
pause
exit /b 1
