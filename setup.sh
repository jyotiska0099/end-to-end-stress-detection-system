#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh  –  Stress Detection System  –  macOS M1 environment setup
# Run once after cloning. Assumes:
#   • .venv already created at project root with Python 3.14
#   • Homebrew installed
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON=".venv/bin/python"
PIP=".venv/bin/pip"

echo "═══════════════════════════════════════════════════"
echo "  Stress Detection System — Environment Setup"
echo "═══════════════════════════════════════════════════"

# ── 1. Verify we're at the project root ──────────────────────────────────────
if [ ! -f "README.md" ]; then
  echo "ERROR: Run this script from the project root (where README.md lives)."
  exit 1
fi

# ── 2. Verify .venv exists ───────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
  echo "ERROR: .venv not found. Create it first:"
  echo "  python3.14 -m venv .venv"
  exit 1
fi

echo ""
echo "▶ Python version:"
"$PYTHON" --version

# ── 3. Homebrew dependencies ─────────────────────────────────────────────────
echo ""
echo "▶ Checking Homebrew dependencies..."

BREW_DEPS=(postgresql@16 minikube kubectl helm)
for dep in "${BREW_DEPS[@]}"; do
  if brew list "$dep" &>/dev/null; then
    echo "  ✓ $dep"
  else
    echo "  Installing $dep..."
    brew install "$dep"
  fi
done

# Docker Desktop is GUI — just check the CLI is present
if ! command -v docker &>/dev/null; then
  echo ""
  echo "  WARNING: docker CLI not found."
  echo "  Install Docker Desktop for Apple Silicon:"
  echo "  https://docs.docker.com/desktop/install/mac-install/"
fi

# ── 4. Upgrade pip ───────────────────────────────────────────────────────────
echo ""
echo "▶ Upgrading pip..."
"$PIP" install --upgrade pip --quiet

# ── 5. Install shared dev tooling into venv ──────────────────────────────────
echo ""
echo "▶ Installing shared dev tools (ruff, pytest, pytest-asyncio, httpx)..."
"$PIP" install --quiet \
  ruff \
  pytest \
  pytest-asyncio \
  httpx

# ── 6. Install per-service requirements ──────────────────────────────────────
echo ""
echo "▶ Installing service requirements..."

SERVICES=(
  services/inference-service
  services/llm-service
  services/response-router
  services/sensor-simulator
  services/retraining-pipeline
)

for svc in "${SERVICES[@]}"; do
  req="$svc/requirements.txt"
  if [ -f "$req" ]; then
    echo "  Installing $svc..."
    "$PIP" install --quiet -r "$req"
  else
    echo "  SKIP $svc (no requirements.txt yet)"
  fi
done

# ── 7. Verify .env exists ────────────────────────────────────────────────────
echo ""
echo "▶ Checking .env..."
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "  .env created from .env.example — fill in your GEMINI_API_KEY"
else
  echo "  ✓ .env exists"
fi

# ── 8. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Edit .env — add your GEMINI_API_KEY"
echo "    2. Verify WESAD data is at ml/data/wesad/"
echo "    3. Activate the venv:  source .venv/bin/activate"
echo "═══════════════════════════════════════════════════"
