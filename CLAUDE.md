# Subtitulos-pro - Project Documentation

## Overview

Professional Python application for generating animated subtitles for short videos (YouTube Shorts, Reels, TikTok) using local AI processing.

## Architecture

### Technology Stack

- **GUI Framework**: PySide6 (Qt6 bindings)
- **Audio Transcription**: Faster-Whisper (OpenAI Whisper optimization)
- **Video Processing**: FFmpeg + MoviePy
- **AI Processing**: Ollama (local LLM integration)
- **Image Processing**: Pillow + OpenCV
- **Configuration**: YAML

### Project Structure

```
subtítulos/
├── main.py              # Application entry point
├── config.yml           # Application configuration
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
├── run.bat / run.sh    # Platform-specific launchers
├── ui/                  # GUI components
│   ├── main_window.py   # Main application window
│   └── style.qss        # Qt stylesheet (dark theme)
├── core/                # Business logic
│   ├── transcriber.py   # Whisper transcription pipeline
│   ├── renderer.py      # Subtitle rendering engine
│   ├── video_processor.py  # Video embedding and processing
│   ├── ollama_client.py # Ollama API client
│   └── utils.py         # Utility functions
├── models/              # Data models
│   └── subtitle.py      # Subtitle data structures
├── temp/                # Temporary files (auto-cleaned)
├── output/              # Rendered videos
└── logs/                # Application logs
```

## Key Components

### SubtitleTrack (models/subtitle.py)

Core data structure for managing subtitle timelines:

- `SubtitleTrack`: Complete track with segments
- `SubtitleSegment`: Text segment with timing
- `WordTimestamp`: Individual word timing for highlight effect
- `SubtitleStyleConfig`: Style configuration
- `ProcessingConfig`: Processing parameters

### TranscriptionPipeline (core/transcriber.py)

Handles audio extraction and Whisper transcription:

- `AudioExtractor`: Extracts audio from video using FFmpeg
- `WhisperTranscriber`: Faster-Whisper wrapper with GPU support
- `SilenceDetector`: Detects silence periods for optimization

### SubtitleRenderer (core/renderer.py)

Renders animated subtitle frames:

- `FrameRenderer`: Main renderer with word highlighting
- `TextRenderer`: Text drawing with effects
- `FontManager`: Font loading and caching
- `ColorUtils`: Color manipulation utilities

### VideoProcessor (core/video_processor.py)

Video processing and subtitle embedding:

- `SubtitleEmbedder`: Embeds subtitles using FFmpeg/MoviePy
- `RenderQueue`: Manages multiple render tasks
- Support for vertical video optimization

### MainWindow (ui/main_window.py)

PySide6 main window with:

- Drag & drop video upload
- Style configuration panel
- Real-time progress tracking
- Video preview player
- Background processing with QThread

## Configuration

### Key Config Sections

**Whisper Settings:**
- `model`: Model size (tiny/base/small/medium/large)
- `device`: Processing device (auto/cuda/cpu)
- `language`: Source language

**Subtitle Styling:**
- `style`: Preset (modern/bold/neon/minimal/classic)
- `font_size`: Text size in pixels
- `position`: Vertical position
- `animation_speed`: Word transition speed

**Processing:**
- `output_format`: Video format (mp4, mov, etc.)
- `quality`: Render quality preset
- `preset`: FFmpeg encoding preset

## Development Guidelines

### Code Style

- Use type hints for all function signatures
- Docstrings for all public methods
- Dataclasses for configuration objects
- Logging instead of print statements
- Proper exception handling with custom exceptions

### Adding New Features

1. **New subtitle style**: Add to `SubtitleStyle` enum and implement in renderer
2. **New video format**: Add to `VideoFormat` enum and processor
3. **New config option**: Add to config.yml and corresponding dataclass

### Testing Video Processing

```python
# Load video
processor = VideoProcessor(ProcessingConfig())
info = processor.get_video_info(video_path)

# Check if vertical
if processor.is_vertical_video(video_path):
    # Optimize for vertical format
    optimized = processor.optimize_for_vertical(video_path)
```

### Adding Animation Effects

Effects are applied in `TextRenderer.draw_animated_text()`:

1. Determine word state (active/past/future)
2. Apply color and scale based on state
3. Use `word_highlight` and `zoom_effect` config options

## Common Tasks

### Adding a New Subtitle Style

1. Add style to `SubtitleStyle` enum
2. Create preset in `SubtitleRenderer.apply_style_preset()`
3. Update UI combo box in `MainWindow._create_style_section()`

### Modifying Word Timing

Word timestamps come from Whisper's word-level timestamps. To adjust:

```python
# In TranscriptionPipeline.process_video()
for word in segment.words:
    # Adjust timing precision
    word.start = round(word.start, 2)
    word.end = round(word.end, 2)
```

### Custom Color Effects

Add new color utilities in `ColorUtils`:

```python
@staticmethod
def custom_effect(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    # Your color manipulation
    return modified_rgb
```

## Performance Optimization

### GPU Acceleration

- CUDA detected automatically if available
- Falls back to CPU if GPU not present
- Configure `device: "cuda"` in config.yml for force GPU

### Memory Management

- Temporary files cleaned automatically
- Video processing uses generators for large files
- Font caching reduces memory overhead

### Rendering Speed

- Use smaller Whisper models for faster transcription
- Adjust `quality` preset in config (low/medium/high)
- FFmpeg preset affects encoding speed

## Error Handling

### Common Issues

**Whisper model loading failure:**
- Check model size compatibility
- Verify sufficient RAM
- Ensure proper device configuration

**FFmpeg not found:**
- Install FFmpeg and add to PATH
- Verify installation with `ffmpeg -version`

**Ollama connection failed:**
- Start Ollama server: `ollama serve`
- Verify model is downloaded
- Check firewall settings

## Dependencies

### Core Requirements

- `PySide6`: GUI framework
- `faster-whisper`: Speech recognition
- `ffmpeg-python`: Video processing
- `moviepy`: Video editing
- `pillow`: Image manipulation
- `torch`: ML framework (GPU support)

### Optional Dependencies

- `cuda`: GPU acceleration
- `ollama`: AI processing
- `httpx`: HTTP client for Ollama

## Build and Distribution

### Creating Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable
pyinstaller --onefile --windowed main.py

# Output in dist/main.exe
```

### Creating Distribution Package

```bash
python setup.py sdist bdist_wheel
```

## License

MIT License - See LICENSE file for details
