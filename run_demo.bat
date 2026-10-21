@echo off
rem 双击启动 Streamlit 网页演示，停止按 Ctrl+C
cd /d "%~dp0"
streamlit run step6_web_demo\app.py
pause
