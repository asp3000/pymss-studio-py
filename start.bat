@echo off
REM ---- relaunch THIS batch hidden so no cmd window stays visible --------
REM When you double-click start.bat, the first console flashes for a moment,
REM then re-launches itself hidden (PSS_HIDDEN env mark prevents a loop) and
REM immediately starts the GUI. Net result: while the app runs, no cmd window.
if not defined PSS_HIDDEN (
    set "PSS_HIDDEN=1"
    powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath '%~f0' -WindowStyle Hidden"
    exit /b
)
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM ---- locate a python interpreter (console + windowless) ---------------
REM Prefer the project venv (carries pymss + CUDA torch); fall back to PATH.
REM PY  = console interpreter (used only for checks)
REM PYW = windowless interpreter (pythonw.exe) used to launch the GUI so no
REM       cmd/console window appears while the app is running.
set "PY="
set "PYW="
if exist "%SCRIPT_DIR%venv\Scripts\python.exe"    set "PY=%SCRIPT_DIR%venv\Scripts\python.exe"
if exist "%SCRIPT_DIR%venv\Scripts\pythonw.exe"   set "PYW=%SCRIPT_DIR%venv\Scripts\pythonw.exe"
if not defined PY (
    if exist "%SCRIPT_DIR%.venv\Scripts\python.exe"  set "PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
)
if not defined PYW (
    if exist "%SCRIPT_DIR%.venv\Scripts\pythonw.exe" set "PYW=%SCRIPT_DIR%.venv\Scripts\pythonw.exe"
)
if not defined PY (
    where python  >nul 2>nul && set "PY=python"
)
if not defined PY (
    where python3 >nul 2>nul && set "PY=python3"
)
if not defined PYW (
    where pythonw >nul 2>nul && set "PYW=pythonw"
)

if not defined PY (
    echo [ERROR] Python was not found on PATH.
    echo Install Python 3.10+ from python.org and tick "Add to PATH",
    echo or create a virtual environment here:  venv\Scripts\python.exe

    exit /b 1
)
REM Fall back to the console interpreter if no windowless one is available.
if not defined PYW set "PYW=%PY%"

REM ---- isolate the Python environment ---------------------------------
REM We deliberately do NOT blank the system PATH: pymss shells out to
REM `ffmpeg`, torch needs CUDA runtime DLLs (cudart/cublas/...), and the
REM downloader uses git/aria2c -- all of which are located via PATH. Clearing
REM PATH would break the app outright. Instead we only strip PYTHONPATH (any
REM user-injected package directories) and disable the per-user site-packages,
REM so a stray / older pymss in %APPDATA%\Python\... can never shadow the
REM venv's pinned pymss 2.0.14. These vars propagate to the GUI process and,
REM from there, to every worker subprocess that inherits its environment.
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
REM Reduce CUDA memory fragmentation (pymss suggested this on OOM). Lets the
REM allocator grow segments instead of failing when reserved-but-unused memory
REM is scattered. Harmless when there is no pressure; helps large 4-stem models
REM fit on a 12 GB GPU.
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo Using Python : %PY%
echo Working dir  : %SCRIPT_DIR%
echo.

REM ---- make sure PySide6 is importable ---------------------------------
%PY% -c "import PySide6" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] PySide6 is not installed in this Python.
    echo Install it with:
    echo     %PY% -m pip install PySide6

    exit /b 1
)

REM ---- launch the GUI (windowless, no cmd window) ----------------------
REM Errors are appended to a log file because there is no console to show them.
echo Starting Pymss Studio GUI ...
echo (Close the GUI window to exit. Any error is logged to pymss-studio.log)
echo.
%PYW% run.py >>"%SCRIPT_DIR%pymss-studio.log" 2>&1
set "RC=%errorlevel%"

echo.
if not "%RC%"=="0" (
    echo [GUI exited with code %RC%]
    echo See %SCRIPT_DIR%pymss-studio.log for the traceback.
    echo If the error is about import/pymss, install pymss and torch in this
    echo Python, or set the correct interpreter in the GUI Settings page.
)

endlocal
