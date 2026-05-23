#!/usr/bin/env python
"""
Punto de entrada sin consola (Windows .pyw).
Al hacer doble clic en este archivo, Python ejecuta pythonw.exe
que NO abre ventana de consola.

Uso:
  - Doble clic en main.pyw  ->  sin consola
  - pythonw main.pyw        ->  sin consola
  - launch.vbs              ->  sin consola (recomendado)
"""
import sys
from pathlib import Path

# Añadir el directorio del proyecto al path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from main import main

if __name__ == "__main__":
    main()
