"""
Subtitle rendering module with word-by-word animations.
"""
import logging
import math
import platform
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass

from models.subtitle import (
    SubtitleTrack,
    SubtitleSegment,
    WordTimestamp,
    SubtitleStyleConfig,
    SubtitleStyle
)


logger = logging.getLogger(__name__)


@dataclass
class RenderedFrame:
    """A rendered subtitle frame."""
    image: Image.Image
    timestamp: float
    duration: float


class ColorUtils:
    """Utility functions for color manipulation."""

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color."""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    @staticmethod
    def adjust_brightness(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """Adjust brightness of RGB color."""
        return tuple(min(255, int(c * factor)) for c in rgb)

    @staticmethod
    def add_glow_effect(
        rgb: Tuple[int, int, int],
        glow_intensity: int = 30
    ) -> Tuple[int, int, int]:
        """Add glow effect to color."""
        return tuple(min(255, c + glow_intensity) for c in rgb)


class FontManager:
    """Manage font loading and sizing."""

    def __init__(self):
        """Initialize font manager."""
        self.font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get_font(self, font_name: str, size: int) -> ImageFont.FreeTypeFont:
        """
        Get font with specified name and size.

        Args:
            font_name: Font name or path
            size: Font size in pixels

        Returns:
            ImageFont object
        """
        cache_key = (font_name, size)

        if cache_key in self.font_cache:
            return self.font_cache[cache_key]

        try:
            # Try loading as system font
            font = ImageFont.truetype(font_name, size)
            self.font_cache[cache_key] = font
            return font
        except OSError:
            # --- Fallback portátil: SOLO nombres de fuente (sin rutas absolutas) ---
            # PIL/ImageFont.truetype busca en el sistema por nombre en cualquier SO.
            # Ordenamos por popularidad para minimizar intentos fallidos.
            system = platform.system()

            # Fuentes prioritarias según el sistema operativo
            if system == "Windows":
                os_fonts = [
                    "Segoe UI", "Segoe UI Bold", "Roboto", "Arial",
                    "Microsoft Sans Serif", "Tahoma"
                ]
            elif system == "Darwin":  # macOS
                os_fonts = [
                    "Helvetica", "Helvetica Neue", "SF Pro Display",
                    "SF Pro Text", "Arial", "Roboto"
                ]
            else:  # Linux y otros
                os_fonts = [
                    "DejaVuSans", "DejaVu Sans", "LiberationSans",
                    "Liberation Sans", "NotoSans", "Noto Sans",
                    "FreeSans", "Arial"
                ]

            # Fuentes universales que funcionan en casi cualquier SO
            universal_fonts = ["Arial", "Roboto", "DejaVuSans"]

            # Combinar: primero las del SO, luego universales
            seen = set()
            fallback_fonts = []
            for f in os_fonts + universal_fonts:
                if f not in seen:
                    seen.add(f)
                    fallback_fonts.append(f)

            for fallback in fallback_fonts:
                try:
                    font = ImageFont.truetype(fallback, size)
                    self.font_cache[cache_key] = font
                    logger.info(f"Using fallback font: {fallback}")
                    return font
                except OSError:
                    continue

            # Last resort: default font
            font = ImageFont.load_default()
            self.font_cache[cache_key] = font
            logger.warning("Using default font")
            return font


class TextRenderer:
    """Render text with various effects."""

    def __init__(self, font_manager: FontManager):
        """
        Initialize text renderer.

        Args:
            font_manager: Font manager instance
        """
        self.font_manager = font_manager
        # Caché de medición de texto: (texto, fuente, tamaño) -> (ancho, alto)
        self._measure_cache: Dict[Tuple[str, str, int], Tuple[int, int]] = {}

    def measure_text(self, text: str, font: str, size: int) -> Tuple[int, int]:
        """
        Measure text dimensions con caché.

        Args:
            text: Text to measure
            font: Font name
            size: Font size

        Returns:
            (width, height) tuple
        """
        cache_key = (text, font, size)
        if cache_key in self._measure_cache:
            return self._measure_cache[cache_key]

        font_obj = self.font_manager.get_font(font, size)

        # Create temp image to measure
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        bbox = draw.textbbox((0, 0), text, font=font_obj)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        self._measure_cache[cache_key] = (width, height)
        return width, height

    def draw_text_with_effects(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        position: Tuple[int, int],
        font: str,
        size: int,
        color: str,
        stroke_width: int = 0,
        stroke_color: str = "#000000",
        glow: bool = False,
        glow_color: str = "#FFFFFF",
        glow_intensity: int = 20
    ) -> None:
        """
        Draw text with visual effects.

        Args:
            draw: ImageDraw object
            text: Text to draw
            position: (x, y) position
            font: Font name
            size: Font size
            color: Text color
            stroke_width: Stroke width
            stroke_color: Stroke color
            glow: Whether to add glow
            glow_color: Glow color
            glow_intensity: Glow intensity
        """
        font_obj = self.font_manager.get_font(font, size)
        x, y = position
        color_rgb = ColorUtils.hex_to_rgb(color)
        stroke_rgb = ColorUtils.hex_to_rgb(stroke_color)

        # Draw glow if enabled
        if glow:
            glow_rgb = ColorUtils.hex_to_rgb(glow_color)
            for i in range(glow_intensity, 0, -5):
                alpha = int(255 * (1 - i / glow_intensity) * 0.3)
                glow_color_with_alpha = (*glow_rgb, alpha)
                draw.text(
                    (x - i//2, y - i//2),
                    text,
                    font=font_obj,
                    fill=glow_color_with_alpha,
                    stroke_width=0
                )

        # Draw stroke
        if stroke_width > 0:
            draw.text(
                (x, y),
                text,
                font=font_obj,
                fill=tuple(stroke_rgb),
                stroke_width=stroke_width,
                stroke_fill=tuple(stroke_rgb)
            )

        # Draw main text
        draw.text((x, y), text, font=font_obj, fill=tuple(color_rgb))

    def _split_words_into_lines(
        self,
        words: List[WordTimestamp],
        font: str,
        font_size: int,
        max_width: int
    ) -> List[Tuple[List[Tuple[int, int, WordTimestamp]], int]]:
        """
        Divide palabras en líneas que quepan dentro de max_width.
        Retorna lista de (lista_de_palabras_con_posiciones, ancho_total_de_línea)
        donde cada palabra tiene (offset_x_relativo, ancho_palabra, word_info).
        """
        if not words:
            return []

        # Calcular ancho de cada palabra (incluyendo espacio)
        word_data = []
        for w in words:
            w_w, w_h = self.measure_text(w.word + " ", font, font_size)
            word_data.append((w_w, w_h, w))

        lines = []
        current_line_words = []  # (offset_x, word_width, word_height, word_info)
        current_width = 0

        for w_w, w_h, w_info in word_data:
            if current_width + w_w > max_width and current_line_words:
                # Guardar línea actual y empezar nueva
                lines.append((current_line_words, current_width))
                current_line_words = []
                current_width = 0

            current_line_words.append((current_width, w_w, w_h, w_info))
            current_width += w_w

        if current_line_words:
            lines.append((current_line_words, current_width))

        return lines

    @staticmethod
    def _filter_visible_lines(
        lines: List,
        current_time: float,
        max_lines: int = 3
    ) -> List:
        """
        Filtra líneas para mostrar solo las relevantes:
        - Elimina líneas completadas (todas sus palabras ya pasaron)
        - Mantiene máximo `max_lines` líneas a partir de la activa

        Args:
            lines: Lista de (line_words, line_width)
            current_time: Tiempo actual del video
            max_lines: Máximo de líneas visibles

        Returns:
            Lista filtrada de (line_words, line_width)
        """
        if not lines:
            return []

        # 1. Encontrar el índice de la línea que contiene la palabra activa
        current_line_idx = 0
        for i, (line_words, _) in enumerate(lines):
            for _, _, _, w in line_words:
                if w.start <= current_time <= w.end:
                    current_line_idx = i
                    break
            else:
                continue
            break
        else:
            # Si ninguna línea tiene palabra activa, buscar la próxima
            for i, (line_words, _) in enumerate(lines):
                for _, _, _, w in line_words:
                    if w.start > current_time:
                        current_line_idx = max(0, i - 1)
                        break
                else:
                    continue
                break
            else:
                current_line_idx = len(lines) - 1

        # 2. Mostrar desde la línea activa, máximo max_lines
        start = max(0, min(current_line_idx, len(lines) - max_lines))
        return lines[start:start + max_lines]

    def draw_precomputed_lines(
        self,
        draw: ImageDraw.ImageDraw,
        lines: List,
        current_time: float,
        position: Tuple[int, int],
        config: SubtitleStyleConfig,
        max_width: int
    ) -> Tuple[int, int]:
        """
        Dibuja líneas de subtítulos animados ya calculadas (sin re-split).
        Sistema LEGIBLE Y SUAVE:
          - Palabra activa: ligeramente más brillante (como si se iluminara)
          - Palabras pasadas: color normal (mismo tono, legibles)
          - Palabras futuras: atenuadas (se ven pero no distraen)
          - SIN zoom, SIN barras, SIN cambios de color bruscos
        Solo 1 línea visible: los ojos no saltan entre líneas.

        Args:
            draw: ImageDraw object
            lines: Lista de (line_words, line_width) pre-computada
            current_time: Current video time
            position: (x, y) posición inicial
            config: Style configuration
            max_width: Máximo ancho para centrar líneas

        Returns:
            (max_line_width, total_height)
        """
        if not lines:
            return 0, 0

        x0, y0 = position
        line_spacing = int(config.font_size * 0.4)
        _, word_h = self.measure_text("Ay", config.font, config.font_size)

        current_y = y0
        for line_words, line_width in lines:
            # Centrar línea horizontalmente
            line_x = x0 + (max_width - line_width) // 2 if line_width < max_width else x0

            # Dibujar palabras: sin zoom, un solo tamaño, solo varía el brillo
            for offset_x, word_w, word_h_local, w_info in line_words:
                word_x = line_x + offset_x
                word_y = current_y

                is_active = w_info.start <= current_time <= w_info.end
                is_past = current_time > w_info.end

                if is_active and config.word_highlight:
                    # Activa: ligeramente más brillante (se ilumina suavemente)
                    base = ColorUtils.hex_to_rgb(config.color)
                    brighter = ColorUtils.adjust_brightness(base, 1.15)
                    color = ColorUtils.rgb_to_hex(brighter)
                elif is_past:
                    # Pasada: color normal, legible
                    color = config.color
                else:
                    # Futura: atenuada para no distraer
                    base = ColorUtils.hex_to_rgb(config.color)
                    dimmed = ColorUtils.adjust_brightness(base, 0.50)
                    color = ColorUtils.rgb_to_hex(dimmed)

                # Sin zoom: todas las palabras al mismo tamaño
                self.draw_text_with_effects(
                    draw, w_info.word, (word_x, word_y),
                    config.font, config.font_size, color,
                    stroke_width=config.stroke_width,
                    stroke_color=config.stroke_color,
                    glow=(config.style == SubtitleStyle.NEON and is_active),
                    glow_color=color, glow_intensity=15
                )

            current_y += word_h + line_spacing

        total_height = len(lines) * word_h + (len(lines) - 1) * line_spacing
        max_line_width = max((lw for _, lw in lines), default=0)
        return max_line_width, total_height

    def draw_animated_text(
        self,
        draw: ImageDraw.ImageDraw,
        words: List[WordTimestamp],
        current_time: float,
        position: Tuple[int, int],
        config: SubtitleStyleConfig,
        max_width: int,
        max_lines: int = 2
    ) -> Tuple[int, int]:
        """
        Draw animated subtitle text con scroll automático.
        Máximo `max_lines` líneas visibles; las completadas desaparecen.
        """
        lines = self._split_words_into_lines(
            words, config.font, config.font_size, max_width
        )
        if not lines:
            return 0, 0

        visible_lines = self._filter_visible_lines(lines, current_time, max_lines)
        if not visible_lines:
            return 0, 0

        return self.draw_precomputed_lines(
            draw, visible_lines, current_time, position, config, max_width
        )


class FrameRenderer:
    """Render subtitle frames with word-by-word animation."""

    def __init__(self, config: SubtitleStyleConfig):
        """
        Initialize frame renderer.

        Args:
            config: Subtitle style configuration
        """
        self.config = config
        self.font_manager = FontManager()
        self.text_renderer = TextRenderer(self.font_manager)

    MAX_LINES = 1  # Solo 1 línea: la mirada no se dispersa entre líneas

    def render_frame(
        self,
        video_size: Tuple[int, int],
        subtitle_track: SubtitleTrack,
        current_time: float
    ) -> Optional[Image.Image]:
        """
        Render subtitle frame for specific time.
        - Escala font_size según resolución
        - Máximo MAX_LINES líneas visibles
        - Posición FIJA (el texto no se mueve al scrollear)
        - show_only_current_word: solo la palabra activa (modo palabra-por-palabra)
        """
        segment, word = subtitle_track.get_active_words(current_time)
        if not segment:
            return None

        width, height = video_size
        reference_height = 1080

        # --- Escalar font_size proporcionalmente a la altura del vídeo ---
        # Para vídeos verticales (1080x1920) la altura es 1920, el font escala hacia ARRIBA.
        # Antes había un `min()` que lo capaba al tamaño base, impediendo el escalado vertical.
        effective_font_size = max(16, int(self.config.font_size * height / reference_height))

        import copy
        render_config = copy.copy(self.config)
        render_config.font_size = effective_font_size

        # --- MODO palabra-por-palabra: solo la palabra activa ---
        if render_config.show_only_current_word:
            if word is None:
                return None  # Entre palabras, no mostrar nada
            words_to_render = [word]
        else:
            words_to_render = segment.words if segment.words else []

        # --- 1. CALCULAR LÍNEAS Y FILTRAR ---
        text_max_width = int(width * 0.92)
        line_spacing = int(effective_font_size * 0.35)
        _, word_h = self.text_renderer.measure_text("Ay", render_config.font, effective_font_size)

        # Altura para MAX_LINES (fija, para que la posición no cambie)
        fixed_block_h = self.MAX_LINES * word_h + (self.MAX_LINES - 1) * line_spacing

        if words_to_render:
            all_lines = self.text_renderer._split_words_into_lines(
                words_to_render, render_config.font, effective_font_size, text_max_width
            )
            # Filtrar: solo líneas activas y siguientes (máximo MAX_LINES)
            visible_lines = TextRenderer._filter_visible_lines(
                all_lines, current_time, self.MAX_LINES
            )
            text_w = max((lw for _, lw in visible_lines), default=0) if visible_lines else 0
        else:
            text_w, _ = self.text_renderer.measure_text(
                segment.text, render_config.font, effective_font_size
            )
            visible_lines = None

        if not visible_lines and not words_to_render:
            return None

        # --- 2. POSICIÓN FIJA (usando fixed_block_h, no text_h real) ---
        x_area = (width - text_max_width) // 2
        x_block = x_area + (text_max_width - text_w) // 2

        if self.config.position == "bottom":
            margin = int(self.config.margin_bottom * height / reference_height)
            y_base = height - margin - fixed_block_h
        elif self.config.position == "center":
            y_base = (height - fixed_block_h) // 2
        else:
            margin = int(self.config.margin_bottom * height / reference_height)
            y_base = margin

        y_base = max(10, y_base)
        if y_base + fixed_block_h > height - 10:
            y_base = height - 10 - fixed_block_h
        y_base = max(10, y_base)

        # --- 3. CREAR LAYER RGBA solo si hay algo que renderizar ---
        img = Image.new('RGBA', video_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if self.config.background_opacity > 0 and visible_lines:
            bg_rgb = ColorUtils.hex_to_rgb(self.config.background_color)
            bg_alpha = int(self.config.background_opacity * 255)
            pad_x, pad_y = 12, 6
            # Fondo del tamaño fijo (MAX_LINES)
            draw.rectangle(
                [x_block - pad_x, y_base - pad_y,
                 x_block + text_w + pad_x, y_base + fixed_block_h + pad_y],
                fill=(*bg_rgb, bg_alpha)
            )

        # --- 4. TEXTO ANIMADO (pre-computed lines, sin re-split) ---
        if words_to_render and visible_lines:
            self.text_renderer.draw_precomputed_lines(
                draw, visible_lines, current_time,
                (x_area, y_base), render_config, text_max_width
            )
        elif segment.text:
            self.text_renderer.draw_text_with_effects(
                draw, segment.text, (x_block, y_base),
                render_config.font, effective_font_size,
                render_config.color,
                stroke_width=render_config.stroke_width,
                stroke_color=render_config.stroke_color
            )

        return img

    def render_subtitles_video(
        self,
        video_path: Path,
        subtitle_track: SubtitleTrack,
        output_path: Path,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """
        Render video with embedded subtitles.

        Args:
            video_path: Input video path
            subtitle_track: Subtitle track
            output_path: Output video path
            progress_callback: Optional progress callback

        Returns:
            Path to rendered video
        """
        try:
            import cv2
            from moviepy.editor import VideoFileClip, CompositeVideoClip

            # Open video
            if progress_callback:
                progress_callback("Loading video...", 5)

            video_clip = VideoFileClip(str(video_path))
            video_width, video_height = video_clip.size

            # Create subtitle clips for each segment
            subtitle_clips = []

            for i, segment in enumerate(subtitle_track.segments):
                if progress_callback:
                    progress_callback(
                        f"Processing subtitle {i+1}/{len(subtitle_track.segments)}",
                        10 + (i * 70 // len(subtitle_track.segments))
                    )

                # Create image for this segment
                segment_img = self.render_frame(
                    (video_width, video_height),
                    subtitle_track,
                    segment.start + (segment.duration / 2)
                )

                if segment_img is None:
                    continue

                # Convert to numpy array for MoviePy
                segment_array = np.array(segment_img)
                segment_clip = (
                    CompositeVideoClip([
                        video_clip,
                        segment_clip
                    ])
                    .set_start(segment.start)
                    .set_end(segment.end)
                )

                subtitle_clips.append(segment_clip)

            if not subtitle_clips:
                logger.warning("No subtitles to render")
                return video_path

            # Composite video with subtitles
            if progress_callback:
                progress_callback("Compositing video...", 85)

            final_video = CompositeVideoClip([
                video_clip,
                *subtitle_clips
            ])

            # Write output
            if progress_callback:
                progress_callback("Writing video file...", 90)

            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.config.temp_dir / 'temp_audio.m4a'),
                remove_temp=True,
                preset='medium',
                ffmpeg_params=['-crf', '23']
            )

            # Cleanup
            video_clip.close()
            final_video.close()

            if progress_callback:
                progress_callback("Complete!", 100)

            return output_path

        except Exception as e:
            logger.error(f"Video rendering error: {e}")
            raise


class SubtitleRenderer:
    """Main subtitle renderer with multiple style support."""

    STYLE_RENDERERS = {
        SubtitleStyle.MODERN: FrameRenderer,
        SubtitleStyle.BOLD: FrameRenderer,
        SubtitleStyle.NEON: FrameRenderer,
        SubtitleStyle.MINIMAL: FrameRenderer,
        SubtitleStyle.CLASSIC: FrameRenderer,
    }

    @classmethod
    def create_renderer(cls, config: SubtitleStyleConfig) -> FrameRenderer:
        """
        Create renderer for specified style.

        Args:
            config: Style configuration

        Returns:
            Renderer instance
        """
        renderer_class = cls.STYLE_RENDERERS.get(
            config.style,
            FrameRenderer
        )
        return renderer_class(config)

    @staticmethod
    def apply_style_preset(
        style: SubtitleStyle,
        base_config: SubtitleStyleConfig
    ) -> SubtitleStyleConfig:
        """
        Apply style preset to configuration.

        Args:
            style: Style to apply
            base_config: Base configuration

        Returns:
            Updated configuration
        """
        config = SubtitleStyleConfig(**base_config.to_dict())
        config.style = style

        presets = {
            SubtitleStyle.MODERN: {
                "font": "Roboto",
                "font_size": 64,
                "color": "#FFEB3B",
                "highlight_color": "#FFFFFF",
                "stroke_width": 8,
                "stroke_color": "#000000",
                "word_highlight": True,
                "zoom_effect": True,
                "show_only_current_word": False,
                "background_opacity": 0.0,
            },
            SubtitleStyle.BOLD: {
                "font": "Impact",
                "font_size": 52,
                "color": "#FFFFFF",
                "highlight_color": "#00FFFF",
                "stroke_width": 4,
                "word_highlight": True,
                "zoom_effect": False
            },
            SubtitleStyle.NEON: {
                "font": "Arial Black",
                "font_size": 48,
                "color": "#FF00FF",
                "highlight_color": "#00FFFF",
                "stroke_width": 2,
                "word_highlight": True,
                "zoom_effect": True
            },
            SubtitleStyle.MINIMAL: {
                "font": "Helvetica",
                "font_size": 44,
                "color": "#FFFFFF",
                "highlight_color": "#CCCCCC",
                "stroke_width": 0,
                "word_highlight": False,
                "zoom_effect": False,
                "background_opacity": 0.5
            },
            SubtitleStyle.CLASSIC: {
                "font": "Times New Roman",
                "font_size": 46,
                "color": "#FFFF00",
                "highlight_color": "#FF0000",
                "stroke_width": 3,
                "word_highlight": True,
                "zoom_effect": False,
                "background_color": "#000000",
                "background_opacity": 0.8
            }
        }

        if style in presets:
            for key, value in presets[style].items():
                setattr(config, key, value)

        return config
