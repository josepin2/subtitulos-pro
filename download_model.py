"""
Descarga el modelo Whisper "small" (~700 MB) para Subtitulos-pro.

El modelo small es el que la aplicacion usa por defecto.
Se guarda en models/whisper/ y se reutiliza en cada ejecucion.
No necesitas internet despues de la descarga inicial.

Uso:
    python download_model.py
"""
import os
import sys
from pathlib import Path

# Asegurar que estamos en la carpeta del proyecto
os.chdir(Path(__file__).parent.resolve())

# Nombre del modelo por defecto (coincide con config.yml y main.py)
MODELO_POR_DEFECTO = "small"
TAMANO_APROX = "~700 MB"

# Configurar carpeta del modelo
model_dir = Path("models/whisper")
model_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(model_dir.resolve())
os.environ["HF_HUB_CACHE"] = str(model_dir.resolve())
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print(f"\n{'='*55}")
print(f"  Subtitulos-pro - Descargar modelo Whisper")
print(f"{'='*55}")
print(f"\n  Modelo: {MODELO_POR_DEFECTO} ({TAMANO_APROX})")
print(f"  Destino: {model_dir.resolve()}")
print()

try:
    from faster_whisper import WhisperModel

    print(f"Descargando '{MODELO_POR_DEFECTO}' desde HuggingFace...")
    print("(Esto puede tomar varios minutos segun tu conexion)")
    print()

    model = WhisperModel(
        MODELO_POR_DEFECTO,
        device="cpu",
        compute_type="float32",
        download_root=str(model_dir.resolve()),
        local_files_only=False,
    )

    print(f"\n{'='*55}")
    print(f"  [OK] Modelo '{MODELO_POR_DEFECTO}' descargado correctamente")
    print(f"  Ruta: {model_dir.resolve()}")
    print(f"{'='*55}")
    print()

except Exception as e:
    print(f"\n[ERROR] {e}")
    print()
    print("Posibles soluciones:")
    print("  1. Verifica tu conexion a internet")
    print("  2. Asegurate de tener faster-whisper instalado:")
    print("     pip install faster-whisper")
    sys.exit(1)
