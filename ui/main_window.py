"""
Main application window.
"""
import logging
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import sys

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar,
    QFileDialog, QFrame,
    QStatusBar, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDesktopServices

from models.subtitle import ProcessingConfig
from core.video_processor import VideoProcessor
from core.utils import (
    load_config, check_ffmpeg_installed,
    check_gpu_available,
    validate_video_file,
    format_duration,
    resolve_project_path,
    get_project_root,
)


logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """
    Background thread for video processing.
    Ejecuta el pipeline en un SUBPROCESO Python aislado para evitar
    conflictos de DLL entre ctranslate2/onnxruntime y PySide6/Qt.
    """

    progress_update = Signal(str, int)  # message, percentage
    finished = Signal(Path)  # output path
    error = Signal(str)  # error message
    status_update = Signal(str)  # status message

    def __init__(
        self,
        video_path: Path,
        output_path: Path,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.config = config
        self._is_running = True
        self._process = None
        self._comm_dir = None

    def run(self):
        """
        Lanza un subproceso Python separado que ejecuta el pipeline completo.
        La comunicación se hace mediante archivos JSON en un directorio temporal.
        """
        import tempfile
        import subprocess
        import shutil

        logger.info("=" * 50)
        logger.info("INICIANDO PIPELINE (subproceso aislado)")
        logger.info("=" * 50)

        self.status_update.emit("Iniciando procesamiento...")

        project_dir = get_project_root()
        model_cache_dir = resolve_project_path("models/whisper")

        try:
            # --- Crear directorio de comunicación ---
            self._comm_dir = Path(tempfile.mkdtemp(prefix="subtitulos_"))
            logger.info(f"Directorio de comunicación: {self._comm_dir}")

            # --- Escribir config.json ---
            config_data = {
                "video_path": str(self.video_path.resolve()),
                "output_path": str(self.output_path.resolve()),
                "whisper_model": self.config.get("whisper_model", "small"),
                "language": self.config.get("language", "auto"),
                "whisper_cache_dir": str(model_cache_dir),
                "temp_dir": str((project_dir / (self.config.get("temp_dir", "temp"))).resolve()),
                "cache_enabled": self.config.get("cache_enabled", True),
                "cleanup_temp": self.config.get("cleanup_temp", True),
                "subtitle": self.config.get("subtitle", {}),
            }
            config_file = self._comm_dir / "config.json"
            config_file.write_text(json.dumps(config_data, ensure_ascii=False), encoding="utf-8")
            logger.info("Configuración escrita para el worker")

            # --- Determinar rutas ---
            worker_script = project_dir / "core" / "transcribe_worker.py"
            python_exec = sys.executable  # Usa el mismo Python del venv
            # Si el ejecutable es pythonw.exe, usar python.exe en su lugar
            # (pythonw.exe es GUI subsystem y puede causar errores en el worker)
            if sys.platform == "win32" and python_exec.lower().endswith("pythonw.exe"):
                python_exe = Path(python_exec).with_name("python.exe")
                if python_exe.exists():
                    python_exec = str(python_exe)
                    logger.info(f"Worker forzado a usar python.exe: {python_exec}")

            if not worker_script.exists():
                self.error.emit(f"Error interno: no se encuentra {worker_script}")
                return

            if not self._is_running:
                return

            # --- Iniciar subproceso ---
            self.status_update.emit("Iniciando motor de transcripción...")
            logger.info(f"Lanzando: {python_exec} {worker_script} --comm-dir {self._comm_dir}")

            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE → consola oculta

            self._process = subprocess.Popen(
                [python_exec, str(worker_script), "--comm-dir", str(self._comm_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
                cwd=str(project_dir),
            )

            logger.info(f"Worker PID: {self._process.pid}")

            # --- Polling loop ---
            poll_interval = 0.3  # segundos entre cada lectura
            while self._is_running:
                # Verificar si el proceso terminó
                return_code = self._process.poll()
                if return_code is not None:
                    logger.info(f"Worker terminó con código: {return_code}")
                    break

                # Leer progreso
                self._read_progress()

                # Verificar resultado
                if self._check_result():
                    return
                if self._check_error():
                    return

                self.msleep(int(poll_interval * 1000))

            # --- Proceso terminado, verificar archivos finales ---
            if self._check_result():
                return
            if self._check_error():
                return

            if not self._is_running:
                logger.info("Pipeline cancelado por el usuario")
                return

            # Si llegamos aquí, el worker terminó sin resultado conocido
            logger.warning("Worker terminó sin archivo de resultado ni error")
            stderr_text = ""
            try:
                if self._process and self._process.stderr:
                    stderr_text = self._process.stderr.read(2000)
            except Exception:
                pass

            if stderr_text:
                self.error.emit(f"Error en el worker:\n{stderr_text}")
            else:
                self.error.emit(
                    "El proceso de transcripción terminó inesperadamente.\n\n"
                    "Revisa logs/subtitulos.log para más detalles."
                )

        except Exception as e:
            logger.error(f"Error en ProcessingThread: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Error interno: {str(e)}")
        finally:
            # Limpiar directorio de comunicación después de un tiempo
            try:
                if self._comm_dir and self._comm_dir.exists():
                    shutil.rmtree(self._comm_dir, ignore_errors=True)
            except Exception:
                pass

    def _read_progress(self):
        """Lee el archivo de progreso si existe."""
        if not self._comm_dir:
            return
        try:
            progress_file = self._comm_dir / "progress.json"
            if progress_file.exists():
                data = json.loads(progress_file.read_text(encoding="utf-8"))
                if data.get("type") == "progress":
                    msg = data.get("message", "")
                    pct = data.get("percentage", 0)
                    self.progress_update.emit(msg, pct)
                    if pct >= 0:
                        self.status_update.emit(msg)
        except Exception:
            pass

    def _check_result(self) -> bool:
        """Verifica si existe result.json. Emite finished y retorna True si sí."""
        if not self._comm_dir:
            return False
        try:
            result_file = self._comm_dir / "result.json"
            if result_file.exists():
                data = json.loads(result_file.read_text(encoding="utf-8"))
                output = Path(data.get("output_path", ""))
                if output.exists():
                    logger.info(f"Pipeline completado: {output}")
                    self.finished.emit(output)
                    return True
                else:
                    self.error.emit(f"Archivo de salida no encontrado: {output}")
                    return True
        except Exception as e:
            logger.error(f"Error leyendo result.json: {e}")
        return False

    def _check_error(self) -> bool:
        """Verifica si existe error.json. Emite error y retorna True si sí."""
        if not self._comm_dir:
            return False
        try:
            error_file = self._comm_dir / "error.json"
            if error_file.exists():
                data = json.loads(error_file.read_text(encoding="utf-8"))
                msg = data.get("message", "Error desconocido")
                logger.error(f"Error del worker: {msg}")
                self.error.emit(msg)
                return True
        except Exception as e:
            logger.error(f"Error leyendo error.json: {e}")
        return False

    def stop(self):
        """Detiene el procesamiento."""
        self._is_running = False
        if self._process and self._process.poll() is None:
            logger.info("Matando proceso worker...")
            self._process.kill()
            self._process.wait(timeout=5)
            logger.info("Worker terminado")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.config = self._load_configuration()
        self.video_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.processing_thread: Optional[ProcessingThread] = None

        # Drop zone reference for drag & drop
        self.drop_zone = None

        self._setup_ui()
        self._check_dependencies()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load application configuration."""
        config_path = resolve_project_path("config.yml")
        return load_config(config_path)

    def _setup_ui(self):
        """Setup the user interface - compacto y minimalista."""
        self.setWindowTitle("Subtítulos-pro")
        self.setFixedSize(520, 500)

        # Apply stylesheet
        self._apply_stylesheet()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout - vertical, compact
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # Main content (controls + progress)
        content = self._create_content()
        main_layout.addWidget(content, stretch=1)

        # Action buttons (horizontal, compact)
        actions = self._create_action_buttons()
        main_layout.addWidget(actions)

        # Create status bar
        self._create_status_bar()

    def _create_content(self) -> QWidget:
        """Create main content area (file input + progress)."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # File selection section (marco simple, sin QGroupBox para ahorrar espacio)
        layout.addWidget(self._create_file_section())

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)
        layout.addWidget(self.progress_bar)

        # Status
        self.status_label = QLabel("Listo")
        self.status_label.setObjectName("infoLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Detail
        self.progress_detail_label = QLabel()
        self.progress_detail_label.setObjectName("subheadingLabel")
        self.progress_detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_detail_label.setWordWrap(True)
        layout.addWidget(self.progress_detail_label)

        # Output directory row
        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(QLabel("Destino:"))
        self.output_path_label = QLabel("Por defecto (carpeta output)")
        self.output_path_label.setObjectName("infoLabel")
        output_layout.addWidget(self.output_path_label, stretch=1)
        browse_btn = QPushButton("Examinar")
        browse_btn.clicked.connect(self._select_output_directory)
        output_layout.addWidget(browse_btn)
        layout.addLayout(output_layout)

        layout.addStretch()
        return widget

    def _create_header(self) -> QFrame:
        """Create header frame."""
        header = QFrame()
        header.setObjectName("headerWidget")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 20, 16, 20)

        title = QLabel("Subtítulos Animados")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Generador automático de subtítulos para vídeos cortos")
        subtitle.setObjectName("subheadingLabel")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        return header

    def _create_file_section(self) -> QFrame:
        """Create file selection section - compacta y estética."""
        frame = QFrame()
        frame.setObjectName("fileSection")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Drop zone (compacta, texto justo, sin sublínea)
        drop_frame = QFrame()
        drop_frame.setObjectName("dropZone")
        drop_frame.setAcceptDrops(True)
        drop_frame.dragEnterEvent = self._drag_enter_event
        drop_frame.dropEvent = self._drop_event
        drop_frame.dragLeaveEvent = self._drag_leave_event
        drop_frame.setFixedHeight(42)
        self.drop_zone = drop_frame

        drop_layout = QVBoxLayout(drop_frame)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.setContentsMargins(2, 0, 2, 0)

        drop_label = QLabel("Arrastra vídeo aquí")
        drop_label.setObjectName("dropLabel")
        drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(drop_label)

        layout.addWidget(drop_frame)

        # Video info label (con separación del drop)
        self.video_info_label = QLabel("Ningún vídeo seleccionado")
        self.video_info_label.setObjectName("infoLabel")
        self.video_info_label.setWordWrap(True)
        layout.addWidget(self.video_info_label)

        return frame

    def _create_action_buttons(self) -> QFrame:
        """Create action buttons frame - compacto horizontal."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        # Generate button (principal)
        self.generate_btn = QPushButton("Generar Subtítulos")
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.clicked.connect(self._start_processing)
        self.generate_btn.setEnabled(False)
        layout.addWidget(self.generate_btn, stretch=1)

        # Cancel button
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.clicked.connect(self._cancel_processing)
        self.cancel_btn.setEnabled(False)
        layout.addWidget(self.cancel_btn)

        # Open output button
        self.open_output_btn = QPushButton("Abrir Carpeta")
        self.open_output_btn.setObjectName("secondaryButton")
        self.open_output_btn.clicked.connect(self._open_output_folder)
        self.open_output_btn.setEnabled(False)
        layout.addWidget(self.open_output_btn)

        return frame

    def _create_status_bar(self):
        """Create status bar."""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Listo")

    def _apply_stylesheet(self):
        """Apply QSS stylesheet."""
        style_path = resolve_project_path("ui/style.qss")
        if style_path.exists():
            with open(style_path, 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        else:
            self.logger.warning(f"Stylesheet not found: {style_path}")

    def _check_dependencies(self):
        """Check required dependencies."""
        # Check FFmpeg
        if check_ffmpeg_installed():
            self.logger.info("FFmpeg: OK")
        else:
            self.logger.warning("FFmpeg: Not found")
            QMessageBox.warning(
                self,
                "FFmpeg Not Found",
                "FFmpeg is not installed or not in PATH.\n\n"
                "Please install FFmpeg to use this application.\n\n"
                "Windows: choco install ffmpeg\n"
                "macOS: brew install ffmpeg\n"
                "Linux: sudo apt install ffmpeg"
            )

        # Check GPU
        has_gpu, device = check_gpu_available()
        if has_gpu:
            self.logger.info(f"GPU acceleration: {device}")
        else:
            self.logger.info("Using CPU for processing")

    def _drag_enter_event(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            if self.drop_zone:
                self.drop_zone.setProperty("dragActive", True)
                self.drop_zone.style().unpolish(self.drop_zone)
                self.drop_zone.style().polish(self.drop_zone)

    def _drag_leave_event(self, event):
        """Handle drag leave event."""
        if self.drop_zone:
            self.drop_zone.setProperty("dragActive", False)
            self.drop_zone.style().unpolish(self.drop_zone)
            self.drop_zone.style().polish(self.drop_zone)

    def _drop_event(self, event: QDropEvent):
        """Handle drop event."""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self._load_video(Path(files[0]))

        if self.drop_zone:
            self.drop_zone.setProperty("dragActive", False)
            self.drop_zone.style().unpolish(self.drop_zone)
            self.drop_zone.style().polish(self.drop_zone)

    def _select_video_file(self):
        """Open file dialog to select video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Archivo de Vídeo",
            "",
            "Archivos de Vídeo (*.mp4 *.mov *.avi *.mkv *.webm);;Todos los Archivos (*.*)"
        )
        if file_path:
            self._load_video(Path(file_path))

    def _load_video(self, video_path: Path):
        """Load video file."""
        # Validate file
        is_valid, error = validate_video_file(video_path)
        if not is_valid:
            QMessageBox.warning(self, "Vídeo Inválido", error)
            return

        self.video_path = video_path

        # Get video info
        try:
            processor = VideoProcessor(ProcessingConfig())
            info = processor.get_video_info(video_path)

            # Update info label
            info_text = (
                f"📁 {video_path.name}\n"
                f"🎬 {info['width']}x{info['height']} @ {info['fps']:.1f}fps\n"
                f"⏱️ {format_duration(info['duration'])}"
            )

            if processor.is_vertical_video(video_path):
                info_text += "\n📱 Vertical"

            self.video_info_label.setText(info_text)

            # Enable generate button
            self.generate_btn.setEnabled(True)

            # Set default output path (portable: relativo al project root)
            output_dir = resolve_project_path(self.config.get('processing', {}).get('output_dir', 'output'))
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = output_dir / f"{video_path.stem}_subtitled.mp4"

            self.logger.info(f"Loaded video: {video_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar vídeo: {e}")
            self.logger.error(f"Error al cargar vídeo: {e}")

    def _select_output_directory(self):
        """Select output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar Directorio de Salida",
            ""
        )
        if dir_path:
            self.output_path_label.setText(dir_path)

    def _start_processing(self):
        """Start subtitle generation."""
        if not self.video_path:
            QMessageBox.warning(self, "Ningún Vídeo", "Por favor selecciona un vídeo primero")
            return

        # Configuración fija (valores óptimos ya predefinidos)
        config = {
            'whisper_model': 'small',
            'device': 'cpu',
            'language': 'auto',
            'subtitle': {
                'style': 'modern',
                'font_size': 64,
                'font': 'Roboto',
                'color': '#FFEB3B',
                'highlight_color': '#FFFFFF',
                'position': 'bottom',
                'word_highlight': True,
                'zoom_effect': True,
                'background_opacity': 0.0,
                'stroke_width': 8,
                'stroke_color': '#000000',
                'show_only_current_word': False
            },
            'video': {
                'quality': 'high'
            },
            'temp_dir': 'temp',
            'output_dir': self.output_path_label.text() if self.output_path_label.text() != "Por defecto (carpeta output)" else str(resolve_project_path("output")),
            'cache_enabled': True,
            'cleanup_temp': True
        }

        # Determine output path (portable: relativo al project root, no al CWD)
        if self.output_path_label.text() != "Por defecto (carpeta output)":
            output_dir = Path(self.output_path_label.text())
        else:
            output_dir = resolve_project_path("output")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.video_path.stem}_subtitled.mp4"

        # --- Iniciar procesamiento en subproceso aislado ---
        self.status_label.setText("Iniciando procesamiento...")
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.processing_thread = ProcessingThread(
            self.video_path,
            output_path,
            config,
        )
        self.processing_thread.progress_update.connect(self._on_progress_update)
        self.processing_thread.finished.connect(self._on_processing_finished)
        self.processing_thread.error.connect(self._on_processing_error)
        self.processing_thread.status_update.connect(self._on_status_update)
        self.processing_thread.start()

    def _cancel_processing(self):
        """Cancel current processing."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
        self.status_label.setText("Cancelado")
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_detail_label.setText("Procesamiento cancelado")
        self.logger.info("Procesamiento cancelado por el usuario")

    def _on_progress_update(self, message: str, percentage: int):
        """Handle progress update."""
        self.progress_bar.setValue(int(percentage))
        self.progress_detail_label.setText(message)

    def _on_status_update(self, status: str):
        """Handle status update."""
        self.status_label.setText(status)
        self.statusBar.showMessage(status)

    def _on_processing_finished(self, output_path: Path):
        """Handle processing completion."""
        self.progress_bar.setValue(100)
        self.status_label.setText("¡Completado!")
        self.output_path = output_path
        self.open_output_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        QMessageBox.information(
            self,
            "Éxito",
            f"¡Subtítulos generados exitosamente!\n\nSalida: {output_path}"
        )

        self.logger.info(f"Processing complete: {output_path}")

    def _on_processing_error(self, error: str):
        """Handle processing error."""
        logger.error(f"Error recibido en el hilo principal: {error}")
        self.status_label.setText("Error")
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        QMessageBox.critical(
            self,
            "Error de Procesamiento",
            f"{error}\n\nRevisa el archivo logs/subtitulos.log para más detalles."
        )
        self.logger.error(f"Error de procesamiento: {error}")

    def _open_output_folder(self):
        """Open output folder in file manager."""
        if self.output_path and self.output_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path.parent)))
        else:
            output_dir = resolve_project_path(self.config.get('processing', {}).get('output_dir', 'output'))
            if output_dir.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))

    def closeEvent(self, event):
        """Handle window close event."""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Procesamiento en Curso",
                "El procesamiento todavía está corriendo. ¿Cancelar y salir?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.processing_thread.stop()
                self.processing_thread.wait()
            else:
                event.ignore()
                return

        event.accept()
