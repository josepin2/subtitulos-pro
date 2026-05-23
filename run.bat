@echo off
title Subtitulos-pro
cd /d "%~dp0"

call .venv\Scripts\activate.bat

start "" /b pythonw main.py

exit
