#!/usr/bin/env bash
# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

set -euo pipefail

# ------------------------------------------------------------------
# Check Python version >= 3.10
# ------------------------------------------------------------------
REQUIRED_MAJOR=3
REQUIRED_MINOR=10

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt "$REQUIRED_MAJOR" ] || \
   { [ "$PYTHON_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$REQUIRED_MINOR" ]; }; then
    echo "Error: Python >= ${REQUIRED_MAJOR}.${REQUIRED_MINOR} required, found ${PYTHON_VERSION}"
    exit 1
fi
echo "Python ${PYTHON_VERSION} detected — OK"

# ------------------------------------------------------------------
# Create virtual environment
# ------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment…"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ------------------------------------------------------------------
# Install emokit (from sibling directory)
# ------------------------------------------------------------------
echo "Installing emokit…"
pip install -e ../emokit/

# ------------------------------------------------------------------
# Install emosense dependencies
# ------------------------------------------------------------------
echo "Installing emosense dependencies…"
pip install -r requirements.txt

# ------------------------------------------------------------------
# Generate checkpoints if not present
# ------------------------------------------------------------------
if [ ! -d "checkpoints" ] || [ -z "$(ls -A checkpoints/ 2>/dev/null)" ]; then
    echo "Generating demo checkpoints…"
    python scripts/generate_demo_checkpoints.py
fi

# ------------------------------------------------------------------
# Launch application
# ------------------------------------------------------------------
echo "Starting EmoSense…"
python -m emosense.app
