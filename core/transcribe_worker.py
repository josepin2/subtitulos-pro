"""
Worker script that runs the transcription pipeline in an isolated subprocess.
If ctranslate2/WhisperModel crashes (segfault), only this subprocess dies,
the main GUI application stays alive.

Communication: via JSON files in a shared directory (no pipes to avoid deadlocks).
  config.json  <- written by main process
  progress.json -> written by worker (overwrites each time)
  result.json   -> written by worker when done
  error.json    -> written by worker on error
"""
import sys
import json
import os
import logging
import traceback
from pathlib import Path


COMM_DIR: Path = None

def _write_status(data: dict):
    """Escribe el estado actual al archivo de comunicación (escritura atómica)."""
    if COMM_DIR is None:
        return
    try:
        msg_type = data.get("type", "progress")
        filename = f"{msg_type}.json"
        # 1. Escribir a un archivo temporal
        tmp_file = COMM_DIR / f"{filename}.tmp"
        tmp_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        # 2. Renombrar (atómico en Windows, evita lecturas parciales)
        dst_file = COMM_DIR / filename
        # ⚠️ ¡IMPORTANTE! Usar replace() y NO rename() porque en Windows
        # rename() falla con FileExistsError si el destino ya existe.
        # replace() sobrescribe el destino, necesario para actualizar progress.json
        tmp_file.replace(dst_file)
    except Exception as e:
        # Si no hay logger todavía, escribir a stderr como fallback mínimo
        import sys as _sys
        _sys.stderr.write(f"[_write_status] Error: {e}\n")


def _setup_worker_logging(comm_dir: Path):
    """Configura logging para el worker (escribe a un archivo propio)."""
    log_file = comm_dir / "worker.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file), encoding='utf-8'),
        ],
        force=True,
    )


def send(msg_type: str, **kwargs):
    payload = {"type": msg_type, **kwargs}
    _write_status(payload)


def main():
    # --- Parsear argumentos ---
    comm_dir = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--comm-dir" and i + 2 < len(sys.argv):
            comm_dir = Path(sys.argv[i + 2]).resolve()
            break
        if arg.startswith("--comm-dir="):
            comm_dir = Path(arg.split("=", 1)[1]).resolve()
            break

    if comm_dir is None:
        # Fallback portable: tempfile.gettempdir() funciona en cualquier SO
        error_file = Path(tempfile.gettempdir()) / "subtitulos_worker_error.json"
        try:
            error_file.write_text(
                json.dumps({"type": "error", "message": "No comm-dir specified"}) + "\n",
                encoding="utf-8"
            )
        except Exception:
            pass
        sys.exit(1)

    global COMM_DIR
    COMM_DIR = comm_dir

    _setup_worker_logging(comm_dir)
    logger = logging.getLogger(__name__)
    logger.info("Worker iniciado")
    logger.info(f"COMM_DIR: {comm_dir}")

    # Leer configuración
    config_file = comm_dir / "config.json"
    if not config_file.exists():
        send("error", message="Archivo de configuración no encontrado")
        sys.exit(1)

    raw = config_file.read_bytes()
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    config = json.loads(raw.decode("utf-8"))
    video_path = Path(config["video_path"])
    output_path = Path(config["output_path"])
    logger.info(f"Video: {video_path}, Output: {output_path}")

    # Configurar variables de entorno para HuggingFace
    # La ruta exacta del modelo nos la pasa el proceso principal en el config.json
    whisper_cache = config.get("whisper_cache_dir", "")
    if not whisper_cache:
        # Fallback portable: relativo al project root
        from core.utils import resolve_project_path
        whisper_cache = str(resolve_project_path("models/whisper"))
    os.environ["HF_HOME"] = whisper_cache
    os.environ["HF_HUB_CACHE"] = whisper_cache
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["SUBTITULOS_WORKER"] = "1"
    logger.info(f"WHISPER_CACHE_DIR = {whisper_cache}")

    try:
        from contextlib import contextmanager

        # Context manager que añade CREATE_NO_WINDOW a subprocess.Popen
        # solo durante el ámbito donde se usa (MoviePy internamente lanza FFmpeg)
        @contextmanager
        def no_console_subprocess():
            """Añade CREATE_NO_WINDOW a subprocess.Popen mientras dura el contexto."""
            import subprocess as _sp
            if hasattr(_sp, 'CREATE_NO_WINDOW'):
                _orig = _sp.Popen
                def _patched(*args, **kwargs):
                    kwargs.setdefault('creationflags', 0)
                    kwargs['creationflags'] |= _sp.CREATE_NO_WINDOW
                    return _orig(*args, **kwargs)
                _sp.Popen = _patched
                try:
                    yield
                finally:
                    _sp.Popen = _orig
            else:
                yield

        # Importar módulos del proyecto
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.subtitle import ProcessingConfig, SubtitleStyleConfig
        from core.transcriber import TranscriptionPipeline
        from core.video_processor import SubtitleEmbedder
        from core.renderer import SubtitleRenderer, FrameRenderer

        model_name = config.get("whisper_model", "small")

        # Configurar directorios (rutas relativas al project root, portables)
        from core.utils import resolve_project_path
        temp_dir = Path(config.get("temp_dir", "temp"))
        if not temp_dir.is_absolute():
            temp_dir = resolve_project_path(str(temp_dir))

        processing_config = ProcessingConfig(
            temp_dir=temp_dir,
            output_dir=output_path.parent,
            cache_enabled=config.get("cache_enabled", True),
            cleanup_temp=config.get("cleanup_temp", True),
        )

        style_config = SubtitleStyleConfig.from_dict(config.get("subtitle", {}))
        # show_only_current_word se respeta desde la configuración del usuario

        # ================================================================
        # ESCALA DE PROGRESO UNIFICADA (MONOTÓNICA, 0-100):
        #   0-20%   → Carga del modelo Whisper
        #   20-22%  → Extracción de audio con FFmpeg
        #   22-70%  → Transcripción con Whisper (con progreso intermedio)
        #   70-72%  → Post-procesamiento de transcripción
        #   72-75%  → Preparación del renderizador
        #   75-100% → Renderizado de subtítulos (fotograma a fotograma + escritura)
        # ================================================================

        # --- FASE 1: Cargar modelo Whisper (0-20%) ---
        send("progress", message="Paso 1/4: Cargando modelo Whisper...", percentage=0)

        pipeline = TranscriptionPipeline(processing_config)
        pipeline_initialized = False

        # Enviar heartbeats durante la carga (no sabemos el progreso interno)
        import threading
        _loading_complete = threading.Event()

        def _heartbeat_while_loading():
            """Envía heartbeats cada 3s durante la carga del modelo (evita que la GUI se congele)."""
            STAGES = [0, 5, 8, 12, 15, 18]
            idx = 0
            while not _loading_complete.is_set():
                _loading_complete.wait(3.0)
                if _loading_complete.is_set():
                    break
                # Avanzar por los stages hasta 18, luego mantener 19 hasta terminar
                if idx < len(STAGES) - 1:
                    idx += 1
                send("progress",
                     message=f"Paso 1/4: Cargando modelo Whisper ({model_name})... "
                             f"({STAGES[idx]}%)",
                     percentage=STAGES[idx])

        hb_thread = threading.Thread(target=_heartbeat_while_loading, daemon=True)
        hb_thread.start()

        try:
            pipeline.initialize(
                model_size=model_name,
                device="cpu",
                language=config.get("language", "auto"),
            )
        finally:
            _loading_complete.set()

        pipeline_initialized = True
        send("progress", message="Paso 1/4: Modelo Whisper cargado correctamente", percentage=20)

        # --- FASE 2: Transcribir (20-70%) ---
        # Mapeamos el progreso interno de process_video (0-100) → 20-70
        def _map_transcribe_progress(msg, pct):
            overall = 20 + int(pct * 0.50)
            send("progress", message=msg, percentage=min(overall, 70))

        send("progress", message="Paso 2/4: Extrayendo audio del vídeo...", percentage=20)
        track = pipeline.process_video(
            video_path,
            progress_callback=_map_transcribe_progress,
        )

        send("progress",
             message=f"Paso 2/4: Transcripción completada ({track.word_count} palabras, "
                     f"{track.segment_count} segmentos)",
             percentage=70)

        # --- FASE 3: Renderizar video (70-100%) ---
        send("progress", message="Paso 3/4: Preparando renderizador de subtítulos...", percentage=72)

        frame_renderer = SubtitleRenderer.create_renderer(style_config)
        embedder = SubtitleEmbedder(processing_config, style_config)

        # Mapeamos el progreso interno de embed_subtitles_moviepy (0-100) → 75-100
        #   - embed 5% (Loading)   → 75%   (coincide con el mensaje previo)
        #   - embed 15% (Procesando) → 77%
        #   - embed 20-80% (Frames) → 79-94%
        #   - embed 100% (Completado) → 100%
        def _map_render_progress(msg, pct):
            overall = 74 + int(pct * 0.26)
            send("progress", message=msg, percentage=min(overall, 100))

        # El context manager no_console_subprocess evita que MoviePy
        # (que internamente lanza FFmpeg) abra ventanas de consola
        send("progress", message="Paso 4/4: Renderizando subtítulos animados en cada fotograma...", percentage=75)
        with no_console_subprocess():
            output_video = embedder.embed_subtitles_moviepy(
                video_path,
                track,
                frame_renderer,
                output_path,
                progress_callback=_map_render_progress,
            )

        # --- Limpiar ---
        if pipeline_initialized:
            pipeline.cleanup()

        send("progress", message="¡Subtítulos generados exitosamente!", percentage=100)
        send("result", output_path=str(output_video))
        logger.info(f"Worker completado: {output_video}")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Worker error: {tb}")
        send("error", message=str(e), details=tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
