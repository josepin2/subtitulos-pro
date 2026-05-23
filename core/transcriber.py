"""
Audio transcription module using Faster-Whisper.
"""
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from faster_whisper import WhisperModel

from models.subtitle import (
    SubtitleTrack,
    SubtitleSegment,
    WordTimestamp,
    ProcessingConfig
)


logger = logging.getLogger(__name__)


# Ruta de caché local para modelos Whisper
# Prioridad:
# 1. Variable de entorno HF_HOME (usada por el worker con Python del sistema)
# 2. PyInstaller frozen: junto al .exe
# 3. Desarrollo: ruta relativa al proyecto
_hf_home_env = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
if _hf_home_env:
    WHISPER_CACHE_DIR = Path(_hf_home_env)
elif getattr(sys, 'frozen', False):
    WHISPER_CACHE_DIR = Path(sys.executable).parent / "models" / "whisper"
else:
    WHISPER_CACHE_DIR = Path(__file__).parent.parent / "models" / "whisper"
logger.info(f"WHISPER_CACHE_DIR = {WHISPER_CACHE_DIR}")


class TranscriptionError(Exception):
    """Custom exception for transcription errors."""
    pass


class AudioExtractor:
    """Extract audio from video files."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize audio extractor.

        Args:
            temp_dir: Temporary directory for extracted audio
        """
        self.temp_dir = temp_dir or Path(tempfile.gettempdir())

    def extract_audio(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        sample_rate: int = 16000
    ) -> Path:
        """
        Extract audio from video file.

        Args:
            video_path: Path to input video
            output_path: Path for output audio (optional)
            sample_rate: Audio sample rate for Whisper

        Returns:
            Path to extracted audio file

        Raises:
            TranscriptionError: If extraction fails
        """
        try:
            import subprocess

            if output_path is None:
                output_path = self.temp_dir / f"{video_path.stem}_audio.wav"

            logger.info(f"Extracting audio from {video_path}")

            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', str(sample_rate),
                '-y',  # Overwrite output file
                str(output_path)
            ]

            # En Windows, evitar que FFmpeg abra una ventana de consola
            creationflags = 0
            if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                creationflags = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                creationflags=creationflags,
            )

            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
                logger.error(f"FFmpeg error: {error_msg}")
                raise TranscriptionError(f"Failed to extract audio: {error_msg}")

            logger.info(f"Audio extracted to {output_path}")
            return output_path

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timeout")
            raise TranscriptionError("Audio extraction timeout")
        except FileNotFoundError:
            logger.error("FFmpeg not found")
            raise TranscriptionError("FFmpeg is not installed or not in PATH")
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            raise TranscriptionError(f"Audio extraction failed: {e}")


class WhisperTranscriber:
    """Transcribe audio using Faster-Whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        language: str = "auto"
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v1, large-v2, large-v3)
            device: Device to use (auto, cuda, cpu)
            language: Language code or 'auto'
        """
        self.model_size = model_size
        self.language = language

        # Force CPU for stability on Windows (CUDA detection issues)
        self.device = "cpu"
        compute_type = "float32"  # Most compatible on Windows

        logger.info(f"Inicializando Faster-Whisper en {self.device} con modelo {model_size} (compute_type={compute_type})")

        # Mapa de nombres de modelo Systran
        model_name_map = {
            "tiny": "Systran/faster-whisper-tiny",
            "base": "Systran/faster-whisper-base",
            "small": "Systran/faster-whisper-small",
            "medium": "Systran/faster-whisper-medium",
            "large-v1": "Systran/faster-whisper-large-v1",
            "large-v2": "Systran/faster-whisper-large-v2",
            "large-v3": "Systran/faster-whisper-large-v3",
            "large": "Systran/faster-whisper-large-v3"
        }

        full_model_name = model_name_map.get(model_size, model_size)

        # Verificar si el modelo ya está en caché local
        model_local_path = self._find_local_model_path(full_model_name)

        if model_local_path:
            logger.info(f"Modelo encontrado en caché local: {model_local_path}")
        else:
            logger.info("Modelo no encontrado en caché local, se descargará desde HuggingFace")

        # Intentar cargar el modelo con diferentes compute_types si falla
        compute_types_to_try = [compute_type]
        if compute_type == "float32":
            compute_types_to_try.append("int8")
            compute_types_to_try.append("int8_float32")

        last_error = None
        for ct in compute_types_to_try:
            try:
                logger.info(f"Creando instancia de WhisperModel... (compute_type={ct})")

                kwargs = {
                    "device": self.device,
                    "compute_type": ct,
                    "cpu_threads": 4,
                    "num_workers": 2,
                }

                # Si el modelo está en caché local, pasar la ruta DIRECTA (evita huggingface_hub)
                if model_local_path:
                    # Usar ruta local - NO necesita local_files_only ni download_root
                    kwargs["model_size_or_path"] = str(model_local_path)
                    logger.info(f"Usando ruta local: {model_local_path}")
                else:
                    # En app congelada (PyInstaller): NO descargar (causa crash y no hay Python)
                    if getattr(sys, 'frozen', False):
                        raise TranscriptionError(
                            f"Modelo Whisper ({model_size}) no encontrado en:\n"
                            f"  {WHISPER_CACHE_DIR}\n\n"
                            f"Ejecuta 'download_model.py' para descargarlo manualmente."
                        )
                    # Modo normal/worker: descargar desde HuggingFace
                    kwargs["model_size_or_path"] = full_model_name
                    kwargs["download_root"] = str(WHISPER_CACHE_DIR)
                    kwargs["local_files_only"] = False
                    logger.info(f"Descargando desde HuggingFace: {full_model_name}")

                self.model = WhisperModel(**kwargs)
                compute_type = ct  # Actualizar al que funcionó
                logger.info(f"WhisperModel creado correctamente (compute_type={ct})")
                last_error = None
                break

            except Exception as model_error:
                last_error = model_error
                logger.warning(f"Error con compute_type={ct}: {model_error}")

                # Si falló con ruta local, reintentar con HuggingFace (solo en desarrollo)
                if model_local_path and not getattr(sys, 'frozen', False):
                    logger.info("Reintentando con descarga desde HuggingFace...")
                    try:
                        kwargs2 = {
                            "model_size_or_path": full_model_name,
                            "device": self.device,
                            "compute_type": ct,
                            "cpu_threads": 4,
                            "num_workers": 2,
                            "download_root": str(WHISPER_CACHE_DIR),
                            "local_files_only": False,
                        }
                        self.model = WhisperModel(**kwargs2)
                        compute_type = ct
                        logger.info(f"WhisperModel creado correctamente (descarga, compute_type={ct})")
                        last_error = None
                        break
                    except Exception as e2:
                        last_error = e2
                        logger.warning(f"También falló con descarga: {e2}")
                        continue
                continue

        if last_error is not None:
            logger.error(f"Error al crear WhisperModel: {last_error}")
            import traceback
            traceback.print_exc()

            if getattr(sys, 'frozen', False):
                extra_msg = (
                    f"\nAsegúrate de ejecutar 'download_model.py' para descargar el modelo.\n"
                    f"Carpeta esperada: {WHISPER_CACHE_DIR}"
                )
            else:
                extra_msg = (
                    f"\nPosibles soluciones:\n"
                    f"1. Ejecuta: python download_model.py\n"
                    f"2. Elimina la carpeta 'models/whisper' y vuelve a intentar\n"
                    f"3. Verifica que faster-whisper y ctranslate2 estén correctamente instalados\n"
                    f"4. Prueba con un modelo más pequeño (tiny)"
                )

            raise TranscriptionError(
                f"No se pudo cargar el modelo Whisper ({model_size}):\n\n"
                f"{last_error}{extra_msg}"
            )

        logger.info(f"Modelo Faster-Whisper cargado correctamente ({compute_type})")

    def _find_local_model_path(self, model_name: str) -> Optional[Path]:
        """
        Buscar el modelo en la caché local.

        Args:
            model_name: Nombre completo del modelo (ej: Systran/faster-whisper-tiny)

        Returns:
            Path al directorio del modelo si existe, None si no
        """
        import shutil

        # Buscar en el directorio de caché del proyecto
        model_dir_name = "models--" + model_name.replace("/", "--")

        # Buscar SÓLO en la carpeta junto al .exe (portable)
        # NO buscar en ~/.cache/huggingface/hub/ porque eso rompe la portabilidad
        cache_paths = [
            WHISPER_CACHE_DIR / model_dir_name,
            WHISPER_CACHE_DIR / model_dir_name / "snapshots",
        ]

        # Tamaños mínimos esperados para cada modelo (CTranslate2 int8)
        # tiny=~75MB, base=~145MB, small=~700MB, medium=~1.5GB, large=~3.1GB
        MODEL_MIN_SIZES = {
            "tiny": 60,
            "base": 120,
            "small": 600,
            "medium": 1300,
            "large": 2500,
        }

        # Extraer el nombre corto del modelo
        model_short = model_name.split("faster-whisper-")[-1] if "faster-whisper-" in model_name else "base"
        # Limpiar versión (ej: large-v3 -> large)
        model_short = model_short.split("-")[0] if model_short.startswith("large") else model_short
        min_size_mb = MODEL_MIN_SIZES.get(model_short, 250)  # default 250MB
        min_size_bytes = min_size_mb * 1024 * 1024
        logger.info(f"Tamaño mínimo esperado para modelo '{model_short}': {min_size_mb} MB")

        # Verificar si HAY LOCK FILES (descarga en progreso o incompleta)
        for cache_path in cache_paths:
            if cache_path.exists():
                # Buscar archivos .lock o .incomplete o .huggingface (locks)
                lock_files = list(cache_path.rglob("*.lock")) + list(cache_path.rglob("*.incomplete"))
                # huggingface_hub crea archivos .lock en blobs/
                hf_lock_files = list(cache_path.rglob(".locks")) if cache_path.rglob(".locks") else []
                if lock_files:
                    logger.warning(f"Lock files encontrados en {cache_path} -> descarga incompleta")
                    logger.info("Eliminando caché para reiniciar descarga...")
                    try:
                        shutil.rmtree(cache_path)
                        logger.info(f"Caché eliminado: {cache_path}")
                    except Exception as e:
                        logger.error(f"No se pudo eliminar caché: {e}")
                    return None

        for cache_path in cache_paths:
            if cache_path.exists():
                # Buscar archivo model.bin en cualquier subdirectorio
                for bin_file in cache_path.rglob("model.bin"):
                    file_size = bin_file.stat().st_size if bin_file.exists() else 0
                    if file_size >= min_size_bytes:
                        logger.info(f"Modelo encontrado en: {bin_file} ({file_size / 1024 / 1024:.1f} MB)")
                        return bin_file.parent
                    elif file_size > 0:
                        # Archivo demasiado pequeño -> descarga parcial/corrupta
                        logger.warning(
                            f"Modelo PARCIAL/CORRUPTO en {bin_file} "
                            f"({file_size / 1024 / 1024:.1f} MB de {min_size_mb} MB esperados). "
                            f"Eliminando..."
                        )
                        try:
                            # Eliminar todo el directorio del modelo
                            shutil.rmtree(cache_path)
                            logger.info(f"Directorio de modelo corrupto eliminado: {cache_path}")
                        except Exception as e:
                            logger.error(f"No se pudo eliminar modelo corrupto: {e}")
                        return None  # Importante: NO seguir buscando

        return None

    def transcribe(
        self,
        audio_path: Path,
        word_timestamps: bool = True,
        progress_callback: Optional[callable] = None
    ) -> SubtitleTrack:
        """
        Transcribe audio file using Faster-Whisper.

        Args:
            audio_path: Path to audio file
            word_timestamps: Whether to extract word-level timestamps
            progress_callback: Optional callback for progress updates
                Se llama con (porcentaje 0-100, número_segmento) por cada
                segmento procesado.

        Returns:
            SubtitleTrack with transcription

        Raises:
            TranscriptionError: If transcription fails
        """
        try:
            logger.info(f"Transcribiendo {audio_path.name}")

            # Set language parameter
            language_param = None if self.language == "auto" else self.language

            # Obtener duración total del audio para calcular progreso
            audio_duration = 0.0
            try:
                import wave
                with wave.open(str(audio_path), 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    audio_duration = frames / float(rate)
            except Exception:
                pass

            # Transcribe with Faster-Whisper
            segments_gen, info = self.model.transcribe(
                str(audio_path),
                language=language_param,
                word_timestamps=word_timestamps,
                beam_size=5,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 300
                }
            )

            # Convert generator to list, reportando progreso
            segments = []
            all_words = []
            last_reported_pct = -1

            for segment_idx, segment in enumerate(segments_gen):
                words = []

                if word_timestamps and hasattr(segment, 'words'):
                    for word_info in segment.words:
                        word = WordTimestamp(
                            word=word_info.word.strip(),
                            start=word_info.start,
                            end=word_info.end
                        )
                        words.append(word)
                        all_words.append(word)

                subtitle_segment = SubtitleSegment(
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end,
                    words=words
                )
                segments.append(subtitle_segment)

                # Reportar progreso basado en el tiempo transcurrido del audio
                if progress_callback and audio_duration > 0:
                    pct = min(99, int((segment.end / audio_duration) * 100))
                    if pct != last_reported_pct:
                        progress_callback(pct, segment_idx)
                        last_reported_pct = pct

            # Calculate duration from last segment or audio file
            duration = 0.0
            if segments:
                duration = max(seg.end for seg in segments)
            else:
                # Fallback: get audio duration
                try:
                    import wave
                    with wave.open(str(audio_path), 'rb') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration = frames / float(rate)
                except:
                    pass

            track = SubtitleTrack(
                segments=segments,
                language=self.language if self.language != "auto" else info.language,
                duration=duration
            )

            logger.info(
                f"Transcripción completada: {len(segments)} segmentos, "
                f"{len(all_words)} palabras, {duration:.2f}s"
            )

            return track

        except Exception as e:
            logger.error(f"Error de transcripción: {e}")
            import traceback
            traceback.print_exc()
            raise TranscriptionError(f"Transcripción fallida: {e}")

    def cleanup(self):
        """Clean up resources."""
        try:
            del self.model
            if self.device == "cuda":
                import torch
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info("Whisper model limpiado")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


class SilenceDetector:
    """Detect silence in audio for subtitle optimization."""

    def __init__(self, threshold_db: int = -40, min_duration: float = 0.3):
        """
        Initialize silence detector.

        Args:
            threshold_db: Silence threshold in dB
            min_duration: Minimum silence duration in seconds
        """
        self.threshold_db = threshold_db
        self.min_duration = min_duration

    def detect_silences(
        self,
        audio_path: Path
    ) -> List[Tuple[float, float]]:
        """
        Detect silence periods in audio.

        Args:
            audio_path: Path to audio file

        Returns:
            List of (start, end) silence timestamps
        """
        try:
            from soundfile import read as read_audio
            import numpy as np

            # Read audio
            audio, sr = read_audio(str(audio_path))

            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Calculate amplitude in dB
            amplitude = np.abs(audio)
            amplitude_db = 20 * np.log10(amplitude + 1e-10)

            # Find silent regions
            is_silent = amplitude_db < (self.threshold_db - amplitude_db.max())
            silences = []
            in_silence = False
            silence_start = 0

            for i, silent in enumerate(is_silent):
                time = i / sr

                if silent and not in_silence:
                    silence_start = time
                    in_silence = True
                elif not silent and in_silence:
                    silence_end = time
                    if silence_end - silence_start >= self.min_duration:
                        silences.append((silence_start, silence_end))
                    in_silence = False

            logger.info(f"Detectados {len(silences)} períodos de silencio")
            return silences

        except Exception as e:
            logger.error(f"Silence detection error: {e}")
            return []


class TranscriptionPipeline:
    """Complete transcription pipeline."""

    def __init__(self, config: ProcessingConfig):
        """
        Initialize transcription pipeline.

        Args:
            config: Processing configuration
        """
        self.config = config
        self.audio_extractor = AudioExtractor(config.temp_dir)
        self.silence_detector = SilenceDetector(
            threshold_db=config.silence_threshold,
            min_duration=0.3
        )
        self.transcriber: Optional[WhisperTranscriber] = None

    def initialize(self, model_size: str, device: str, language: str) -> None:
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size
            device: Device to use
            language: Language code
        """
        self.transcriber = WhisperTranscriber(
            model_size=model_size,
            device=device,
            language=language
        )

    def process_video(
        self,
        video_path: Path,
        progress_callback: Optional[callable] = None
    ) -> SubtitleTrack:
        """
        Process video file and extract subtitles.

        Args:
            video_path: Path to video file
            progress_callback: Optional progress callback

        Returns:
            SubtitleTrack with transcription

        Raises:
            TranscriptionError: If processing fails
        """
        if not self.transcriber:
            raise TranscriptionError("Transcriber not initialized")

        try:
            # Step 1: Extract audio
            if progress_callback:
                progress_callback("Extrayendo audio...", 10)

            audio_path = self.audio_extractor.extract_audio(video_path)

            # Step 2: Transcribe
            if progress_callback:
                progress_callback("Transcribiendo audio...", 30)

            track = self.transcriber.transcribe(
                audio_path,
                word_timestamps=True,
                progress_callback=lambda pct, seg_idx: progress_callback(
                    f"Transcribiendo... {pct}%",
                    30 + int(pct * 0.70)
                )
            )

            # Cleanup temp audio file
            if self.config.cleanup_temp:
                try:
                    audio_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp audio: {e}")

            if progress_callback:
                progress_callback("Transcripción completa!", 100)

            return track

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise TranscriptionError(f"Processing failed: {e}")

    def cleanup(self):
        """Clean up resources."""
        if self.transcriber:
            self.transcriber.cleanup()
