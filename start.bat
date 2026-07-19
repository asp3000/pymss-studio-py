@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
start "" "%SCRIPT_DIR%venv\Scripts\pythonw.exe" run.py
endlocal
