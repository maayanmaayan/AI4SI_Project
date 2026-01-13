#!/bin/bash
# Quick activation script for the virtual environment
# Usage: source activate_env.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/.venv/bin/activate"
echo "✅ Virtual environment activated!"
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
