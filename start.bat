@echo off
cd /d %~dp0
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] .venv not found. Create it first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r backend\requirements.txt
    echo   .venv\Scripts\pip install -r backend\requirements-ai.txt
    echo   .venv\Scripts\pip install -r backend\requirements-inpaint.txt
    pause
    exit /b 1
)
"%PY%" start.py
pause
