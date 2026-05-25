@echo off
title Subtitulos-pro - Instalador
cd /d "%~dp0"

echo ========================================
echo    Subtitulos-pro — Instalador
echo ========================================
echo.
echo  Modelo por defecto: Whisper small (~700 MB)
echo.

REM ----- 1. Verificar Python -----
echo [1/3] Verificando Python...
python --version >nul 2>nul
if errorlevel 1 (
    echo.
    echo  ERROR: Python no esta instalado.
    echo  Descarga Python 3.9+ desde: https://python.org
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo  Python %pyver% detectado
echo.

REM ----- 2. Instalar dependencias Python -----
echo [2/3] Instalando dependencias (pip install -r requirements.txt)...
echo  Esto puede tomar unos minutos...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  ERROR: No se pudieron instalar las dependencias.
    echo  Intenta manualmente: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo  Dependencias instaladas correctamente
echo.

REM ----- 3. Descargar modelo Whisper (small) -----
echo [3/3] Descargando modelo Whisper small desde HuggingFace...
echo  (~700 MB — solo la primera vez)
echo.
python download_model.py
if errorlevel 1 (
    echo.
    echo ========================================
    echo  ERROR: No se pudo descargar el modelo
    echo ========================================
    echo.
    echo Revisa tu conexion a internet e intenta de nuevo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Instalacion completada con exito
echo ========================================
echo.
echo  Ahora solo ejecuta:  run.bat
echo  (o haz doble clic en run.bat)
echo.
pause
