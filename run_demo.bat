@echo off
rem ============================================================
rem  One-click launcher for the Streamlit web demo
rem  Usage: double-click this file
rem  Stop : press Ctrl+C in this window
rem ============================================================

pushd "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Please install Python 3.10 and run: pip install -r requirements.txt
    pause
    exit /b 1
)

python -m streamlit run step6_web_demopp.py

popd
pause
