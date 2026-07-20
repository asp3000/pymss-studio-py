@echo off
if not defined PSS_HIDDEN (
    set "PSS_HIDDEN=1"
    powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath '%~f0' -WindowStyle Hidden"
    exit /b
)
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
start "" "%SCRIPT_DIR%venv\Scripts\pythonw.exe" run.py
endlocal
