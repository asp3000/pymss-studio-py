@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "PYTHONPATH=%SCRIPT_DIR%venv\Lib\site-packages;%SCRIPT_DIR%venv\Lib"
set "PYTHONNOUSERSITE=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
start "" "D:\python311\pythonw.exe" run.py
endlocal
