"""
Utility functions for the application.
"""
import logging
import os
import sys
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import yaml

# Cache global del project root para evitar recalcularlo
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """
    Obtiene la raíz del proyecto de forma PORTABLE.
    Funciona en cualquier SO sin rutas absolutas hardcodeadas:
    - En desarrollo: detecta la ubicación de este archivo (utils.py está en core/)
    - En PyInstaller: usa sys._MEIPASS
    - Siempre devuelve un Path absoluto válido

    Returns:
        Path absoluto a la raíz del proyecto
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    if getattr(sys, 'frozen', False):
        # PyInstaller: el directorio temporal donde se extrajo el ejecutable
        _PROJECT_ROOT = Path(sys._MEIPASS)
    else:
        # En desarrollo: subimos de core/ a la raíz del proyecto
        _PROJECT_ROOT = Path(__file__).parent.parent.resolve()

    return _PROJECT_ROOT


def resolve_project_path(relative_path: str) -> Path:
    """
    Resuelve una ruta relativa contra la raíz del proyecto.
    No depende del Current Working Directory (CWD).

    Args:
        relative_path: Ruta relativa (ej: "temp", "output", "models/whisper")

    Returns:
        Path absoluto dentro del proyecto
    """
    return get_project_root() / relative_path


logger = logging.getLogger(__name__)


def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    max_bytes: int = 10485760,
    backup_count: int = 5
) -> None:
    """
    Configure application logging.

    Args:
        log_file: Path to log file
        level: Logging level
        max_bytes: Maximum log file size
        backup_count: Number of backup files to keep
    """
    # Create logs directory
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # File handler with rotation
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(log_level)


def load_config(config_path: Path) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config or {}
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def save_config(config: dict, config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")


def check_ffmpeg_installed() -> bool:
    """
    Check if FFmpeg is installed and accessible.

    Returns:
        True if FFmpeg is available
    """
    try:
        # En Windows, evitar que FFmpeg abra una ventana de consola
        creationflags = 0
        if hasattr(subprocess, 'CREATE_NO_WINDOW'):
            creationflags = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creationflags,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def check_gpu_available() -> Tuple[bool, str]:
    """
    Check if GPU is available for acceleration.

    Returns:
        (is_available, device_type) tuple
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {device_name}")
            return True, "cuda"
    except ImportError:
        pass

    # Check for Apple Silicon (MPS)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS available")
            return True, "mps"
    except (ImportError, AttributeError):
        pass

    logger.info("No GPU detected, will use CPU")
    return False, "cpu"


def get_temp_dir(base_path: Optional[Path] = None) -> Path:
    """
    Get or create temporary directory.

    Args:
        base_path: Base path for temp directory

    Returns:
        Path to temp directory
    """
    if base_path:
        temp_dir = base_path / "temp"
    else:
        temp_dir = Path(tempfile.gettempdir()) / "subtitulos_temp"

    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_temp_files(temp_dir: Path, keep_pattern: Optional[str] = None) -> None:
    """
    Clean up temporary files.

    Args:
        temp_dir: Temporary directory path
        keep_pattern: Optional pattern for files to keep
    """
    try:
        if not temp_dir.exists():
            return

        for file in temp_dir.iterdir():
            if file.is_file():
                if keep_pattern is None or not file.match(keep_pattern):
                    file.unlink()
                    logger.debug(f"Deleted temp file: {file}")

    except Exception as e:
        logger.warning(f"Failed to cleanup temp files: {e}")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to human-readable string.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def get_video_files(directory: Path) -> List[Path]:
    """
    Get list of video files in directory.

    Args:
        directory: Directory to search

    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}

    try:
        return [
            file for file in directory.iterdir()
            if file.is_file() and file.suffix.lower() in video_extensions
        ]
    except Exception as e:
        logger.error(f"Failed to get video files: {e}")
        return []


def validate_video_file(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that a file is a valid video.

    Args:
        file_path: Path to file

    Returns:
        (is_valid, error_message) tuple
    """
    if not file_path.exists():
        return False, "File does not exist"

    if not file_path.is_file():
        return False, "Path is not a file"

    if file_path.suffix.lower() not in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}:
        return False, "File is not a supported video format"

    try:
        # Try to open with OpenCV
        import cv2
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return False, "Could not open video file"
        cap.release()
        return True, None
    except Exception as e:
        return False, f"Video validation failed: {e}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')

    # Limit length
    if len(filename) > 200:
        filename = filename[:200]

    return filename or "output"


def get_system_info() -> dict:
    """
    Get system information for debugging.

    Returns:
        Dictionary with system info
    """
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }

    # Add GPU info if available
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "Not installed"

    # Add FFmpeg info
    info["ffmpeg_installed"] = check_ffmpeg_installed()

    return info


class ProgressTracker:
    """Track and report progress of long-running operations."""

    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total_steps: Total number of steps
            description: Operation description
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.callbacks: List[callable] = []

    def add_callback(self, callback: callable) -> None:
        """Add progress callback."""
        self.callbacks.append(callback)

    def update(self, step: int, message: Optional[str] = None) -> None:
        """
        Update progress.

        Args:
            step: Current step number
            message: Optional status message
        """
        self.current_step = min(step, self.total_steps)
        progress = (self.current_step / self.total_steps) * 100

        status_message = message or f"{self.description}... {progress:.0f}%"

        for callback in self.callbacks:
            try:
                callback(status_message, progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def increment(self, message: Optional[str] = None) -> None:
        """Increment progress by one step."""
        self.update(self.current_step + 1, message)

    def complete(self, message: str = "Complete!") -> None:
        """Mark progress as complete."""
        self.update(self.total_steps, message)


class FileWatcher:
    """Watch for file changes."""

    def __init__(self, watch_path: Path):
        """
        Initialize file watcher.

        Args:
            watch_path: Path to watch
        """
        self.watch_path = watch_path
        self.last_modified = {}

    def check_changes(self) -> List[Path]:
        """
        Check for file changes since last check.

        Returns:
            List of changed files
        """
        changed = []

        try:
            for file in self.watch_path.rglob('*'):
                if file.is_file():
                    current_mtime = file.stat().st_mtime
                    if file not in self.last_modified:
                        self.last_modified[file] = current_mtime
                    elif current_mtime > self.last_modified[file]:
                        changed.append(file)
                        self.last_modified[file] = current_mtime
        except Exception as e:
            logger.warning(f"File watcher error: {e}")

        return changed


def generate_unique_filename(base_path: Path, extension: str = ".mp4") -> Path:
    """
    Generate a unique filename that doesn't exist.

    Args:
        base_path: Base path and filename
        extension: File extension

    Returns:
        Unique file path
    """
    counter = 1
    new_path = base_path.with_suffix(extension)

    while new_path.exists():
        new_path = base_path.parent / f"{base_path.stem}_{counter}{extension}"
        counter += 1

    return new_path


def estimate_render_time(
    video_duration: float,
    resolution: Tuple[int, int],
    has_gpu: bool
) -> float:
    """
    Estimate video rendering time.

    Args:
        video_duration: Video duration in seconds
        resolution: (width, height) tuple
        has_gpu: Whether GPU acceleration is available

    Returns:
        Estimated time in seconds
    """
    # Base estimate: 0.5x real-time for CPU, 0.1x for GPU
    multiplier = 0.1 if has_gpu else 0.5

    # Adjust for resolution
    pixel_count = resolution[0] * resolution[1]
    if pixel_count > 2000000:  # > 1080p
        multiplier *= 1.5
    elif pixel_count < 500000:  # < 720p
        multiplier *= 0.7

    return video_duration * multiplier
