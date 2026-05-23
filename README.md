

# Subtítulos-pro 🎬

**Generador profesional de subtítulos animados palabra por palabra** para vídeos cortos (YouTube Shorts, Instagram Reels, TikTok).

Procesamiento 100% **local** con Whisper AI, sin necesidad de conexión a internet ni servicios externos.

## ✨ Características

- 🎤 **Transcripción automática** con Whisper AI (modelos tiny a large-v3)
- 🎬 **Subtítulos animados** palabra por palabra con efecto highlight y zoom
- 🎨 **5 estilos visuales**: modern, bold, neon, minimal, classic
- 📱 **Optimizado para vídeo vertical** (Shorts, Reels, TikTok)
- 🖥️ **Interfaz gráfica oscura** moderna con drag & drop
- ⚡ **Procesamiento local** — sin enviar datos a la nube
- 🎯 **Múltiples idiomas** con detección automática

## 📸 Capturas

![Interfaz principal](docs/screenshot.png)
*Interfaz oscura minimalista con zona de arrastre de vídeo*

## 🚀 Instalación

### Requisitos previos

| Programa | Por qué |
|---|---|
| **Python 3.9+** | Lenguaje base del proyecto |
| **FFmpeg** | Procesamiento de audio y vídeo |
| **(Opcional) NVIDIA GPU + CUDA** | Aceleración por hardware |

### 1. Clonar el repositorio

```bash
git clone https://github.com/josepin2/subtitulos-pro.git
cd subt-tulos-Pro
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar modelo Whisper (opcional, se descarga automáticamente)

Si prefieres descargar el modelo antes de usar la app:

```bash
python download_model.py
```

O haz doble clic en `download_model.bat` (Windows).

### 5. ¡Ejecutar!

```bash
# Desde la terminal
python main.py

# O haz doble clic (Windows)
run.bat
```

## 🎯 Uso rápido

1. **Arrastra** un vídeo a la ventana o haz clic en "Seleccionar Vídeo"
2. **Elige el estilo** de subtítulos (modern, bold, neon, minimal, classic)
3. Haz clic en **"Generar Subtítulos"**
4. Espera mientras se procesa (la barra de progreso te indica el estado)
5. El vídeo con subtítulos aparecerá en la carpeta `output/`

## ⚙️ Configuración

Puedes ajustar el comportamiento editando el archivo `config.yml`:

```yaml
whisper:
  model: "small"      # tiny, base, small, medium, large
  language: "auto"    # es, en, fr, auto...

subtitle:
  style: "modern"     # modern, bold, neon, minimal, classic
  font_size: 48
  word_highlight: true
  zoom_effect: true

video:
  quality: "high"     # low, medium, high
  preset: "medium"
```

## 🧠 Modelos Whisper disponibles

| Modelo | Tamaño | Velocidad | Precisión |
|---|---|---|---|
| `tiny` | ~75 MB | ⚡ Muy rápida | Básica |
| `base` | ~145 MB | ⚡ Rápida | Buena |
| `small` | ~700 MB | 🐢 Media | Muy buena |
| `medium` | ~1.5 GB | 🐢 Lenta | Excelente |
| `large` | ~3.1 GB | 🐌 Muy lenta | Máxima |

## 🏗️ Estructura del proyecto

```
subtitulos-pro/
├── main.py                 # Punto de entrada
├── config.yml              # Configuración
├── core/                   # Lógica de negocio
│   ├── transcriber.py      # Transcripción con Whisper
│   ├── renderer.py         # Renderizado de subtítulos animados
│   ├── video_processor.py  # Procesamiento de vídeo
│   └── transcribe_worker.py # Worker en proceso separado
├── ui/
│   ├── main_window.py      # Ventana principal
│   └── style.qss           # Estilo oscuro
├── models/
│   └── subtitle.py         # Modelos de datos
├── output/                 # Vídeos generados
└── temp/                   # Archivos temporales
```

## 🛠️ Solución de problemas

### FFmpeg no encontrado
```bash
# Windows (chocolatey)
choco install ffmpeg

# macOS (homebrew)
brew install ffmpeg

# Linux (apt)
sudo apt install ffmpeg
```

### Error de memoria con modelos grandes
Usa modelos más pequeños (`tiny` o `base`) si tienes poca RAM.

### La barra de progreso no avanza
Es normal en modelos grandes. La app está trabajando, dale tiempo.

## 📄 Licencia

MIT License — Copyright (c) 2026 José M.C.

## 👤 Autor

**José M.C.**
- GitHub: [@josepin2](https://github.com/josepin2)

Hecho con ❤️ para creadores de contenido.
