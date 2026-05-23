#!/bin/bash
# Subtitulos-pro - Unix Launcher

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Activar entorno virtual si existe
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "========================================"
echo "  Subtitulos-pro"
echo "========================================"
echo ""

python3 main.py "$@"
