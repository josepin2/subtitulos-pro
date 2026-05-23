#!/usr/bin/env python3
"""
Subtítulos-pro - Main Entry Point

Aplicación profesional para generar subtítulos animados para vídeos cortos
para vídeos cortos (YouTube Shorts, Reels, TikTok).

Características:
- Procesamiento 100% local con Whisper
- Subtítulos animados palabra-por-palabra
- Optimizado para vídeos verticales
- Interfaz oscura y minimalista
"""
import sys
import logging
from pathlib import Path
import argparse

# Add project root to path (compatible con PyInstaller)
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys._MEIPASS)
else:
    PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from ui.main_window import MainWindow
from core.utils import setup_logging, load_config, get_system_info


class SplashScreen(QDialog):
    """Ventana modal con créditos y cuenta atrás antes de abrir el programa."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subtítulos-pro")
        self.setFixedSize(400, 250)
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                border: 2px solid #333333;
                border-radius: 12px;
            }
            QLabel#credits {
                color: #ffffff;
                font-size: 18px;
                font-weight: 600;
            }
            QLabel#subcredits {
                color: #a0a0a0;
                font-size: 14px;
            }
            QLabel#countdown {
                color: #0078d4;
                font-size: 48px;
                font-weight: bold;
            }
            QLabel#year {
                color: #666666;
                font-size: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(10)

        # Título
        title = QLabel("Subtítulos-pro")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #0078d4;")
        layout.addWidget(title)

        layout.addStretch()

        # Créditos
        credits = QLabel("Programa creado por José M.C")
        credits.setObjectName("credits")
        credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(credits)

        subcredits = QLabel("Todos los derechos reservados al autor del software.")
        subcredits.setObjectName("subcredits")
        subcredits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subcredits.setWordWrap(True)
        layout.addWidget(subcredits)

        year = QLabel("2026")
        year.setObjectName("year")
        year.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(year)

        layout.addStretch()

        # Contador
        self.countdown = 6
        self.countdown_label = QLabel(str(self.countdown))
        self.countdown_label.setObjectName("countdown")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.countdown_label)

        # Timer para la cuenta atrás
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)

    def _tick(self):
        self.countdown -= 1
        if self.countdown <= 0:
            self.timer.stop()
            self.accept()
        else:
            self.countdown_label.setText(str(self.countdown))

    def keyPressEvent(self, event):
        """Bloquear teclas para que no se pueda cerrar antes del contador."""
        pass

    def closeEvent(self, event):
        """Evitar cerrar con la X (no debería tener barra de título)."""
        if self.countdown > 0:
            event.ignore()
        else:
            event.accept()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Subtítulos-pro - Generador de subtítulos para vídeos cortos"
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input video file path"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output video file path"
    )

    parser.add_argument(
        "-s", "--style",
        type=str,
        choices=["modern", "bold", "neon", "minimal", "classic"],
        default="modern",
        help="Subtitle style preset"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Whisper model size"
    )

    parser.add_argument(
        "-l", "--language",
        type=str,
        default="auto",
        help="Language code (e.g., 'en', 'es', 'auto')"
    )

    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for processing"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in command-line mode without GUI"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Subtítulos-pro v1.0.0"
    )

    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS  # Oculto del usuario, solo para uso interno
    )

    parser.add_argument(
        "--config-json",
        type=str,
        default="{}",
        help=argparse.SUPPRESS  # Solo para uso interno con --worker
    )

    return parser.parse_args()


def setup_application():
    """Setup Qt application."""
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Subtítulos-pro")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("SubtitulosPro")

    # Enable high DPI scaling
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    return app


def run_gui_mode(args, config):
    """Run application in GUI mode."""
    logger = logging.getLogger(__name__)
    logger.info("Starting GUI mode")

    app = setup_application()

    # Mostrar splash screen con créditos y cuenta atrás
    splash = SplashScreen()
    splash.show()

    # Procesar eventos para que se pinte el splash
    app.processEvents()

    # Esperar a que termine el splash
    if splash.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)

    # Create and show main window
    window = MainWindow()

    # Load video if specified
    if args.input:
        video_path = Path(args.input)
        if video_path.exists():
            window._load_video(video_path)
        else:
            logger.error(f"Input video not found: {args.input}")
            sys.exit(1)

    window.show()

    # Run application
    sys.exit(app.exec())


def run_cli_mode(args, config):
    """Run application in command-line mode."""
    import asyncio
    from pathlib import Path
    from models.subtitle import SubtitleStyleConfig, ProcessingConfig, SubtitleStyle
    from core.transcriber import TranscriptionPipeline, TranscriptionError
    from core.renderer import SubtitleRenderer
    from core.video_processor import SubtitleEmbedder
    from core.utils import validate_video_file, format_duration

    logger = logging.getLogger(__name__)
    logger.info("Starting CLI mode")

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input video not found: {args.input}")
        sys.exit(1)

    is_valid, error = validate_video_file(input_path)
    if not is_valid:
        logger.error(f"Invalid video file: {error}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_subtitled.mp4"

    # Create style config
    style = SubtitleStyle(args.style)
    style_config = SubtitleRenderer.apply_style_preset(
        style,
        SubtitleStyleConfig()
    )

    # Create processing config
    processing_config = ProcessingConfig(
        temp_dir=Path("temp"),
        output_dir=output_path.parent,
        cache_enabled=True,
        cleanup_temp=True
    )

    try:
        # Initialize pipeline
        logger.info(f"Processing: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Style: {args.style}")
        logger.info(f"Model: {args.model}")

        pipeline = TranscriptionPipeline(processing_config)
        pipeline.initialize(
            model_size=args.model,
            device=args.device,
            language=args.language
        )

        # Transcribe
        logger.info("Transcribing audio...")
        subtitle_track = pipeline.process_video(
            input_path,
            progress_callback=lambda msg, pct: logger.info(f"{msg} ({pct:.0f}%)")
        )

        logger.info(f"Transcription complete: {subtitle_track.word_count} words")

        # Render subtitles
        logger.info("Rendering video...")
        embedder = SubtitleEmbedder(processing_config, style_config)

        output_video = embedder.embed_subtitles(
            input_path,
            subtitle_track,
            output_path,
            progress_callback=lambda msg, pct: logger.info(f"{msg} ({pct:.0f}%)")
        )

        # Cleanup
        pipeline.cleanup()

        logger.info(f"✓ Complete! Output saved to: {output_video}")

    except TranscriptionError as e:
        logger.error(f"Transcription failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config.yml"
    config = load_config(config_path)

    # Modo worker (usado internamente por PyInstaller): ir directo sin logging
    if args.worker:
        os.environ["SUBTITULOS_WORKER"] = "1"
        # El worker lee --comm-dir de los argumentos directamente
        from core.transcribe_worker import main as worker_main
        sys.exit(worker_main())

    # Setup logging (solo en modo normal/GUI/CLI, NO en worker)
    log_level = args.log_level or config.get('logging', {}).get('level', 'INFO')
    log_file = PROJECT_ROOT / config.get('logging', {}).get('file', 'logs/subtitulos.log')

    setup_logging(
        log_file=log_file,
        level=log_level,
        max_bytes=config.get('logging', {}).get('max_bytes', 10485760),
        backup_count=config.get('logging', {}).get('backup_count', 5)
    )

    logger = logging.getLogger(__name__)

    # Log system info
    system_info = get_system_info()
    logger.info(f"System: {system_info['platform']}")
    logger.info(f"Python: {system_info['python_version']}")

    if system_info.get('cuda_available'):
        logger.info(f"CUDA: {system_info.get('cuda_version')} - {system_info.get('gpu_name')}")

    if system_info.get('ffmpeg_installed'):
        logger.info("FFmpeg: Installed")
    else:
        logger.warning("FFmpeg: Not installed")

    # Run in appropriate mode
    try:
        if args.no_gui:
            run_cli_mode(args, config)
        else:
            run_gui_mode(args, config)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
