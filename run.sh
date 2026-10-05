#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║   ProUpscaler v3.0 — Script de inicio                       ║
# ║   Compatible con Python 3.10+ (3.11/3.12 recomendados)      ║
# ║   NO usa basicsr — usa Spandrel (moderno y estable)         ║
# ╚══════════════════════════════════════════════════════════════╝

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   ProUpscaler v3.0 — Iniciando...        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Detectar Python compatible ─────────────────────────────────────────────
# Preferimos 3.11 o 3.12 por compatibilidad amplia con PyTorch/Spandrel.
PYTHON_BIN=""
for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        
        # Si es 3.13+, solo lo usamos si no hay más remedio (o si forzamos)
        if [ "$major" = "3" ] && [ "$minor" -ge "10" ] && [ "$minor" -lt "13" ]; then
            PYTHON_BIN="$candidate"
            echo "  🐍 Python compatible encontrado: $PYTHON_BIN ($ver)"
            break
        fi
    fi
done

# Fallback a python3 (podría ser 3.13) si no encontramos 3.11/3.12
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
        echo "  ⚠️  Usando python3 por defecto (podría ser incompatible con algunas librerías)."
    fi
fi

if [ -z "$PYTHON_BIN" ]; then
    echo ""
    echo "  ❌ ERROR: No se encontró Python 3.10 o superior."
    echo "     Instala Python con: brew install python@3.11"
    echo ""
    exit 1
fi

# ── 2. Crear entorno virtual ───────────────────────────────────────────────────
VENV="$DIR/.venv"
if [ ! -d "$VENV" ]; then
    echo "  📦 Creando entorno virtual..."
    "$PYTHON_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ── 3. Instalar dependencias ───────────────────────────────────────────────────
echo "  📥 Actualizando pip..."
"$PIP" install --quiet --upgrade pip setuptools wheel

echo "  📥 Instalando dependencias (primera vez: ~5 min)..."
# Sin basicsr — usamos spandrel (compatible Python 3.10-3.13)
"$PIP" install --quiet -r requirements.txt

# google-generativeai requiere a veces instalación explícita
"$PIP" install --quiet "google-generativeai>=0.7.0" 2>/dev/null || true

echo "  ✅ Dependencias instaladas correctamente."
echo ""

# ── 4. Descargar modelos AI ────────────────────────────────────────────────────
MODELS_NEEDED=0
[ ! -f "$DIR/models/GFPGANv1.4.pth" ]         && MODELS_NEEDED=1
[ ! -f "$DIR/models/RealESRGAN_x4plus.pth" ] && MODELS_NEEDED=1
[ ! -f "$DIR/models/RealESRGAN_x2plus.pth" ] && MODELS_NEEDED=1

if [ "$MODELS_NEEDED" = "1" ]; then
    echo "  🤖 Descargando modelos AI (~470 MB, solo la primera vez)..."
    "$PY" download_models.py
    echo ""
else
    echo "  ✅ Modelos AI ya presentes."
    echo ""
fi

# ── 5. Abrir navegador y lanzar servidor ──────────────────────────────────────
PORT="${PROUPSCALER_PORT:-8765}"
URL="http://localhost:${PORT}"

open_browser() {
    if command -v open &>/dev/null; then
        open "$URL" >/dev/null 2>&1 || true
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$URL" >/dev/null 2>&1 || true
    fi
}

(sleep 2 && open_browser) &

echo "  🚀 Servidor iniciado en $URL"
echo "  ⏹  Presiona Ctrl+C para detener"
echo ""
"$PY" app.py
