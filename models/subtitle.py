"""
Data models for subtitle processing.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path


class SubtitleStyle(Enum):
    """Available subtitle styles."""
    MODERN = "modern"
    BOLD = "bold"
    NEON = "neon"
    MINIMAL = "minimal"
    CLASSIC = "classic"


class VideoFormat(Enum):
    """Supported video formats."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    MKV = "mkv"
    WEBM = "webm"


@dataclass
class WordTimestamp:
    """Timestamp for a single word."""
    word: str
    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        """Get word duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WordTimestamp":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SubtitleSegment:
    """A segment of subtitles with multiple words."""
    text: str
    start: float
    end: float
    words: List[WordTimestamp] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start

    @property
    def word_count(self) -> int:
        """Get number of words in segment."""
        return len(self.words)

    def get_word_at_time(self, time: float) -> Optional[WordTimestamp]:
        """Get the word being spoken at a specific time."""
        for word in self.words:
            if word.start <= time <= word.end:
                return word
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [w.to_dict() for w in self.words]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubtitleSegment":
        """Create from dictionary."""
        words = [WordTimestamp.from_dict(w) for w in data.get("words", [])]
        return cls(
            text=data["text"],
            start=data["start"],
            end=data["end"],
            words=words
        )


@dataclass
class SubtitleTrack:
    """Complete subtitle track with all segments."""
    segments: List[SubtitleSegment] = field(default_factory=list)
    language: str = "auto"
    duration: float = 0.0

    def __post_init__(self):
        """Calculate duration after initialization."""
        if self.segments and self.duration == 0.0:
            self.duration = max(seg.end for seg in self.segments)

    @property
    def segment_count(self) -> int:
        """Get number of segments."""
        return len(self.segments)

    @property
    def word_count(self) -> int:
        """Get total word count."""
        return sum(seg.word_count for seg in self.segments)

    def _ensure_indexed(self) -> None:
        """
        Prepara índices para búsqueda binaria O(log n).
        Ordena segmentos y cachea lista de tiempos de inicio.
        """
        if hasattr(self, '_indexed') and self._indexed:
            return
        if not self.segments:
            self._starts = []
            self._indexed = True
            return
        self.segments.sort(key=lambda s: s.start)
        self._starts = [seg.start for seg in self.segments]
        self._indexed = True

    def get_segment_at_time(self, time: float) -> Optional[SubtitleSegment]:
        """
        Get the subtitle segment at a specific time.
        Usa búsqueda binaria O(log n) en lugar de lineal O(n).
        """
        if not self.segments:
            return None

        self._ensure_indexed()
        import bisect
        i = bisect.bisect_right(self._starts, time) - 1
        if i >= 0:
            seg = self.segments[i]
            if seg.start <= time <= seg.end:
                return seg
        return None

    def get_active_words(self, time: float) -> Tuple[Optional[SubtitleSegment], Optional[WordTimestamp]]:
        """Get the active segment and word at a specific time."""
        segment = self.get_segment_at_time(time)
        if segment:
            word = segment.get_word_at_time(time)
            return segment, word
        return None, None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "language": self.language,
            "duration": self.duration
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubtitleTrack":
        """Create from dictionary."""
        segments = [SubtitleSegment.from_dict(s) for s in data.get("segments", [])]
        return cls(
            segments=segments,
            language=data.get("language", "auto"),
            duration=data.get("duration", 0.0)
        )

    def save(self, filepath: Path) -> None:
        """Save to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "SubtitleTrack":
        """Load from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class SubtitleStyleConfig:
    """Configuration for subtitle styling."""
    style: SubtitleStyle = SubtitleStyle.MODERN
    font_size: int = 48
    font: str = "Arial Black"
    color: str = "#FFFFFF"
    highlight_color: str = "#FFD700"
    background_color: str = "#000000"
    background_opacity: float = 0.0
    position: str = "bottom"  # bottom, center, top
    margin_bottom: int = 150
    animation_speed: float = 0.3
    word_highlight: bool = True
    zoom_effect: bool = True
    stroke_width: int = 6
    stroke_color: str = "#000000"
    show_only_current_word: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "style": self.style.value,
            "font_size": self.font_size,
            "font": self.font,
            "color": self.color,
            "highlight_color": self.highlight_color,
            "background_color": self.background_color,
            "background_opacity": self.background_opacity,
            "position": self.position,
            "margin_bottom": self.margin_bottom,
            "animation_speed": self.animation_speed,
            "word_highlight": self.word_highlight,
            "zoom_effect": self.zoom_effect,
            "stroke_width": self.stroke_width,
            "stroke_color": self.stroke_color,
            "show_only_current_word": self.show_only_current_word
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubtitleStyleConfig":
        """Create from dictionary."""
        style = SubtitleStyle(data.get("style", "modern"))
        return cls(
            style=style,
            font_size=data.get("font_size", 48),
            font=data.get("font", "Arial Black"),
            color=data.get("color", "#FFFFFF"),
            highlight_color=data.get("highlight_color", "#FFD700"),
            background_color=data.get("background_color", "#000000"),
            background_opacity=data.get("background_opacity", 0.0),
            position=data.get("position", "bottom"),
            margin_bottom=data.get("margin_bottom", 150),
            animation_speed=data.get("animation_speed", 0.3),
            word_highlight=data.get("word_highlight", True),
            zoom_effect=data.get("zoom_effect", True),
            stroke_width=data.get("stroke_width", 6),
            stroke_color=data.get("stroke_color", "#000000"),
            show_only_current_word=data.get("show_only_current_word", True)
        )


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    output_format: VideoFormat = VideoFormat.MP4
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    quality: str = "high"  # low, medium, high
    preset: str = "medium"
    crf: int = 23
    temp_dir: Path = field(default_factory=lambda: Path("temp"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    cache_enabled: bool = True
    cleanup_temp: bool = True
    detect_silence: bool = True
    silence_threshold: int = -40  # dB
    min_word_duration: float = 0.1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "output_format": self.output_format.value,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "quality": self.quality,
            "preset": self.preset,
            "crf": self.crf,
            "temp_dir": str(self.temp_dir),
            "output_dir": str(self.output_dir),
            "cache_enabled": self.cache_enabled,
            "cleanup_temp": self.cleanup_temp,
            "detect_silence": self.detect_silence,
            "silence_threshold": self.silence_threshold,
            "min_word_duration": self.min_word_duration
        }
