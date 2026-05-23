"""
Video processing module for subtitle embedding.
"""
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, VideoClip
import cv2

from models.subtitle import SubtitleTrack, SubtitleStyleConfig, ProcessingConfig


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process and manipulate video files."""

    def __init__(self, config: ProcessingConfig):
        """
        Initialize video processor.

        Args:
            config: Processing configuration
        """
        self.config = config
        self.temp_dir = config.temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_video_info(self, video_path: Path) -> dict:
        """
        Get video file information.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }

            cap.release()

            logger.info(f"Video info: {info['width']}x{info['height']} @ {info['fps']:.2f}fps, {info['duration']:.2f}s")
            return info

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise

    def is_vertical_video(self, video_path: Path) -> bool:
        """
        Check if video is vertical (Shorts/Reels format).

        Args:
            video_path: Path to video file

        Returns:
            True if video is vertical (aspect ratio > 9:16)
        """
        info = self.get_video_info(video_path)
        aspect_ratio = info["height"] / info["width"]
        return aspect_ratio > 1.5

    def optimize_for_vertical(
        self,
        video_path: Path,
        target_resolution: Tuple[int, int] = (1080, 1920)
    ) -> Path:
        """
        Optimize video for vertical format.

        Args:
            video_path: Input video path
            target_resolution: Target (width, height)

        Returns:
            Path to optimized video
        """
        output_path = self.temp_dir / f"{video_path.stem}_vertical.mp4"

        try:
            video_clip = VideoFileClip(str(video_path))

            # Resize maintaining aspect ratio
            current_w, current_h = video_clip.size
            target_w, target_h = target_resolution

            # Calculate scaling
            scale = min(target_w / current_w, target_h / current_h)
            new_w = int(current_w * scale)
            new_h = int(current_h * scale)

            # Resize
            resized_clip = video_clip.resize((new_w, new_h))

            # Add black bars for pillarboxing/letterboxing
            final_clip = resized_clip.on_color(
                size=target_resolution,
                color=(0, 0, 0),
                pos='center'
            )

            # Write optimized video
            final_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                ffmpeg_params=['-crf', '23']
            )

            # Cleanup
            video_clip.close()
            resized_clip.close()
            final_clip.close()

            logger.info(f"Optimized video for vertical format: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Video optimization failed: {e}")
            return video_path


class SubtitleEmbedder:
    """Embed subtitles into video using FFmpeg."""

    def __init__(
        self,
        config: ProcessingConfig,
        style_config: SubtitleStyleConfig
    ):
        """
        Initialize subtitle embedder.

        Args:
            config: Processing configuration
            style_config: Subtitle style configuration
        """
        self.config = config
        self.style_config = style_config
        self.temp_dir = config.temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def create_ass_subtitles(
        self,
        subtitle_track: SubtitleTrack,
        output_path: Path,
        video_width: int = 1920,
        video_height: int = 1080
    ) -> Path:
        """
        Create ASS (Advanced Substation Alpha) subtitle file.

        Args:
            subtitle_track: Subtitle track
            output_path: Output ASS file path
            video_width: Width of the video (for PlayRes)
            video_height: Height of the video (for PlayRes)

        Returns:
            Path to created ASS file
        """
        ass_content = self._generate_ass_header(subtitle_track, video_width, video_height)

        # Add events for each segment
        for i, segment in enumerate(subtitle_track.segments, 1):
            ass_content += self._generate_ass_event(segment, i)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)

        logger.info(f"Created ASS subtitle file: {output_path} (PlayRes {video_width}x{video_height})")
        return output_path

    def _generate_ass_header(self, subtitle_track: SubtitleTrack,
                             video_width: int = 1920, video_height: int = 1080) -> str:
        """Generate ASS file header with correct PlayRes for the video."""
        return f"""[Script Info]
Title: Subtitulos Animados
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{self.style_config.font},{self.style_config.font_size},&H{self._color_to_ass(self.style_config.color)},&H{self._color_to_ass(self.style_config.highlight_color)},&H{self._color_to_ass(self.style_config.stroke_color)},&H{self._color_to_ass(self.style_config.background_color)},1,0,0,0,100,100,0,0,1,{self.style_config.stroke_width},0,2,10,10,{self.style_config.margin_bottom},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def _generate_ass_event(self, segment, index: int) -> str:
        """Generate ASS event for a subtitle segment."""
        start_time = self._seconds_to_ass_time(segment.start)
        end_time = self._seconds_to_ass_time(segment.end)

        # Generate animated word timing effects
        if segment.words:
            text_parts = []
            current_time = segment.start

            for word in segment.words:
                word_duration = int((word.end - word.start) * 100)  # centiseconds
                text_parts.append(f"{{\\k{word_duration}}}{word.word}")

            text = "".join(text_parts)
        else:
            text = segment.text.replace("\n", "\\N")

        return f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"

    def _color_to_ass(self, hex_color: str) -> str:
        """Convert hex color to ASS color format (BGR)."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"{b:02X}{g:02X}{r:02X}"

    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format (H:MM:SS.CC)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

    def embed_subtitles(
        self,
        video_path: Path,
        subtitle_track: SubtitleTrack,
        output_path: Path,
        progress_callback: Optional[Callable] = None
    ) -> Path:
        """
        Embed subtitles into video file.

        Args:
            video_path: Input video path
            subtitle_track: Subtitle track
            output_path: Output video path
            progress_callback: Optional progress callback

        Returns:
            Path to output video with subtitles
        """
        try:
            if progress_callback:
                progress_callback("Preparing subtitle embedding...", 10)

            # Create ASS subtitle file with correct PlayRes for the video
            ass_path = self.temp_dir / "subtitles.ass"
            video_info = VideoProcessor(self.config).get_video_info(video_path)
            video_width = video_info.get("width", 1920)
            video_height = video_info.get("height", 1080)
            self.create_ass_subtitles(subtitle_track, ass_path, video_width, video_height)

            if progress_callback:
                progress_callback("Embedding subtitles with FFmpeg...", 20)

            # Verificar que el archivo ASS se creó correctamente
            if not ass_path.exists():
                raise RuntimeError(f"No se encontró el archivo de subtítulos ASS: {ass_path}")
            if ass_path.stat().st_size == 0:
                raise RuntimeError(f"El archivo de subtítulos ASS está vacío: {ass_path}")

            # Build FFmpeg command
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Normalizar rutas a forward slashes para FFmpeg (Windows)
            video_path_str = str(video_path).replace('\\', '/')
            output_path_str = str(output_path).replace('\\', '/')

            # Escapar caracteres especiales del path para el filtro ASS de FFmpeg:
            # - ':' en rutas Windows (C:) se confunde con separador de opciones
            # - ',' separa filtros en la cadena de filtros
            # - ';' separa cadenas de filtros
            # Se escapan con '\' para que FFmpeg los trate como literales
            ass_path_str = str(ass_path)
            ass_path_str = ass_path_str.replace('\\', '/')
            ass_path_str = ass_path_str.replace(':', '\\:')
            ass_path_str = ass_path_str.replace(',', '\\,')
            ass_path_str = ass_path_str.replace(';', '\\;')

            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_path_str,
                # Usar comillas simples + escape \: para rutas absolutas Windows (C:\...)
                # FFmpeg ignora las comillas simples y \: escapa el : para el parser de filtros
                '-vf', f"ass='{ass_path_str}'",
                '-c:v', 'libx264',
                '-preset', self.config.preset,
                '-crf', str(self.config.crf),
                '-c:a', 'copy',
                '-y',
                output_path_str
            ]

            # Run FFmpeg
            if progress_callback:
                progress_callback("Rendering video...", 40)

            # En Windows, evitar que FFmpeg abra una ventana de consola
            creationflags = 0
            if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                creationflags = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                encoding='utf-8',
                creationflags=creationflags,
            )

            # Monitor progress and capture stderr
            stderr_lines = []
            for line in process.stderr:
                stderr_lines.append(line)
                if "time=" in line and progress_callback:
                    # Extract progress from FFmpeg output
                    try:
                        time_str = line.split("time=")[1].split()[0]
                        progress = self._parse_ffmpeg_progress(time_str, video_path)
                        progress_callback(f"Rendering... {progress}%", 40 + int(progress * 0.5))
                    except (IndexError, ValueError):
                        pass

            process.wait()

            if process.returncode != 0:
                error_output = "".join(stderr_lines)
                logger.error(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
                logger.error(f"FFmpeg stderr output:\n{error_output}")
                raise RuntimeError(f"FFmpeg failed: {error_output}")

            if progress_callback:
                progress_callback("Subtitle embedding complete!", 100)

            logger.info(f"Created video with subtitles: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Subtitle embedding failed: {e}")
            raise

    def _parse_ffmpeg_progress(self, time_str: str, video_path: Path) -> float:
        """Parse FFmpeg progress from time string."""
        try:
            # Parse time (HH:MM:SS.MS)
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            current_time = hours * 3600 + minutes * 60 + seconds

            # Get total video duration
            video_info = VideoProcessor(self.config).get_video_info(video_path)
            total_duration = video_info.get('duration', 1)

            return min(100, (current_time / total_duration) * 100)

        except Exception:
            return 0.0

    def embed_subtitles_moviepy(
        self,
        video_path: Path,
        subtitle_track: SubtitleTrack,
        renderer: 'FrameRenderer',
        output_path: Path,
        progress_callback: Optional[Callable] = None,
        video_size: Optional[Tuple[int, int]] = None
    ) -> Path:
        """
        Embed subtitles using MoviePy with per-frame animated rendering.
        Renderiza cada fotograma con el estado correcto de los subtítulos
        (palabra activa, zoom, colores, etc.) respetando toda la configuración.

        Args:
            video_path: Input video path
            subtitle_track: Subtitle track
            renderer: Subtitle renderer instance (FrameRenderer)
            output_path: Output video path
            progress_callback: Optional progress callback
            video_size: Optional (width, height). If None, detected from video.

        Returns:
            Path to output video with subtitles
        """
        import numpy as np
        from PIL import Image

        try:
            if progress_callback:
                progress_callback("Loading video...", 5)

            video_clip = VideoFileClip(str(video_path))
            if video_size is None:
                video_width, video_height = video_clip.size
            else:
                video_width, video_height = video_size

            if progress_callback:
                progress_callback("Processing subtitles...", 15)

            total_frames = int(video_clip.duration * video_clip.fps)
            frame_count = [0]  # mutable counter for closure

            # ----- CACHÉ DE RENDERIZADO -----
            # Solo re-renderizamos el subtítulo cuando cambia la palabra activa,
            # en lugar de hacerlo en CADA fotograma (que es extremadamente lento).
            # Para un video de 60s a 30fps: pasamos de 1800 renders a ~50-100.
            _cache = {
                "word_data": None,       # (word.word, word.start, word.end) único
                "image": None,           # PIL Image del último render
                "array": None,           # numpy array pre-convertido
            }

            def _needs_rerender(t):
                """Determina si el render en caché sigue siendo válido."""
                seg, word = subtitle_track.get_active_words(t)
                if seg is None or word is None:
                    return _cache["image"] is not None  # limpiar si había algo
                word_sig = (word.word, word.start, word.end)
                return word_sig != _cache["word_data"]

            def _render_subtitle(t):
                """Renderiza subtítulo, actualiza caché y devuelve numpy array."""
                seg, word = subtitle_track.get_active_words(t)
                if seg is None or word is None:
                    _cache["word_data"] = None
                    _cache["image"] = None
                    _cache["array"] = None
                    return None

                subtitle_img = renderer.render_frame(
                    (video_width, video_height),
                    subtitle_track,
                    t
                )
                if subtitle_img is None:
                    _cache["word_data"] = None
                    _cache["image"] = None
                    _cache["array"] = None
                    return None

                word_sig = (word.word, word.start, word.end)
                _cache["word_data"] = word_sig
                _cache["image"] = subtitle_img
                _cache["array"] = np.array(subtitle_img, dtype=np.float32)
                return _cache["array"]

            def overlay_subtitles(get_frame, t):
                """
                Toma el frame original del video y superpone los subtítulos
                respetando el canal alfa (RGBA -> RGB compositing).
                Usa caché: solo renderiza de nuevo cuando cambia la palabra activa.
                """
                video_frame = get_frame(t).astype(np.float32)

                # Re-renderizar solo si cambió la palabra activa o no hay caché
                if _needs_rerender(t):
                    subtitle_arr = _render_subtitle(t)
                else:
                    subtitle_arr = _cache["array"]

                # Actualizar progreso cada ~5% de los frames
                frame_count[0] += 1
                if progress_callback and frame_count[0] % max(1, total_frames // 20) == 0:
                    pct = int(20 + (frame_count[0] / total_frames) * 60)
                    progress_callback(f"Renderizando fotogramas... {frame_count[0]}/{total_frames}", pct)

                if subtitle_arr is None:
                    return video_frame.astype(np.uint8)

                # Alpha compositing manual: RGB_over = RGB_video * (1-alpha) + RGB_sub * alpha
                if subtitle_arr.shape[2] == 4:
                    alpha = subtitle_arr[:, :, 3:] / 255.0
                    rgb_sub = subtitle_arr[:, :, :3]
                    result = video_frame * (1 - alpha) + rgb_sub * alpha
                else:
                    result = subtitle_arr[:, :, :3]

                return np.clip(result, 0, 255).astype(np.uint8)

            if progress_callback:
                progress_callback("Compositing video...", 20)

            final_video = video_clip.fl(overlay_subtitles)

            if progress_callback:
                progress_callback("Writing output file...", 80)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                preset=self.config.preset,
                ffmpeg_params=['-crf', str(self.config.crf)],
                logger=None
            )

            if progress_callback:
                progress_callback("Complete!", 100)

            logger.info(f"Created video with subtitles (MoviePy): {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"MoviePy embedding failed: {e}")
            raise


class RenderQueue:
    """Queue for managing multiple rendering tasks."""

    def __init__(self):
        """Initialize render queue."""
        self.queue: List[dict] = []
        self.current_task: Optional[dict] = None
        self.is_processing = False

    def add_task(
        self,
        video_path: Path,
        subtitle_track: SubtitleTrack,
        output_path: Path,
        config: SubtitleStyleConfig
    ) -> str:
        """
        Add task to render queue.

        Args:
            video_path: Input video path
            subtitle_track: Subtitle track
            output_path: Output video path
            config: Style configuration

        Returns:
            Task ID
        """
        import uuid
        task_id = str(uuid.uuid4())

        task = {
            "id": task_id,
            "video_path": video_path,
            "subtitle_track": subtitle_track,
            "output_path": output_path,
            "config": config,
            "status": "queued",
            "progress": 0
        }

        self.queue.append(task)
        logger.info(f"Added task {task_id} to queue")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """Get status of specific task."""
        for task in self.queue:
            if task["id"] == task_id:
                return task
        if self.current_task and self.current_task["id"] == task_id:
            return self.current_task
        return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued task."""
        for i, task in enumerate(self.queue):
            if task["id"] == task_id:
                task["status"] = "cancelled"
                self.queue.pop(i)
                logger.info(f"Cancelled task {task_id}")
                return True
        return False

    def clear_queue(self):
        """Clear all queued tasks."""
        self.queue.clear()
        logger.info("Cleared render queue")
