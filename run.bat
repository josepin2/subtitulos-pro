@echo off
title Subtitulos-pro
cd /d "%~dp0"

REM Activar entorno virtual si existe
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat

REM Lanzar la aplicación sin ventana de consola
start "" /b pythonw main.py
exit
