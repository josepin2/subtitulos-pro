@echo off
title Subtitulos-pro - Descargar modelo Whisper
cd /d "%~dp0"

REM Activar entorno virtual si existe (instalación portable)
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo ========================================
echo  Subtitulos-pro
echo  Descargar modelo Whisper (small)
echo ========================================
echo.
echo Descargando modelo small desde HuggingFace...
echo (Esto puede tomar varios minutos segun tu conexion)
echo.

python download_model.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo  ERROR: No se pudo descargar el modelo
    echo ========================================
    echo.
    echo Revisa tu conexion a internet e intenta de nuevo.
) else (
    echo.
    echo ========================================
    echo  Modelo descargado correctamente
    echo  Ya puedes ejecutar run.bat
    echo ========================================
)

echo.
pause
