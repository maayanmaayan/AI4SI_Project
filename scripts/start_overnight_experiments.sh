#!/bin/bash
# Overnight experiments launcher script
# This script activates the virtual environment and runs the overnight experiments

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate virtual environment
source "$PROJECT_ROOT/activate_env.sh"

# Change to project root
cd "$PROJECT_ROOT"

# Run overnight experiments
echo "Starting overnight experiments..."
echo "Project root: $PROJECT_ROOT"
echo "Time: $(date)"
echo ""

python3 scripts/run_overnight_experiments.py "$@"

echo ""
echo "Overnight experiments completed!"
echo "Time: $(date)"