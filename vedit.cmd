@echo off
rem Run the video pipeline with its own Python: vedit.cmd run "C:\path\video.mp4" --title "Topic"
set "VEDIT_PY=%USERPROFILE%\.video-edit-skill\venv\Scripts\python.exe"
if not exist "%VEDIT_PY%" (
  echo Python environment missing. Run setup first:  powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  exit /b 1
)
set "PYTHONPATH=%~dp0"
set PYTHONUTF8=1
"%VEDIT_PY%" -m vedit %*
