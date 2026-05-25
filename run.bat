@echo off
title Subtitulos-pro
cd /d "%~dp0"

REM Intentar activar entorno virtual si existe (instalación portable)
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Usar pythonw si está disponible (sin consola), sino python
where pythonw.exe >nul 2>nul
if %errorlevel% equ 0 (
    start "" /b pythonw main.py
) else (
    start "" /b python main.py
)

exit
