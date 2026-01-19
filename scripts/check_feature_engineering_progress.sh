#!/bin/bash
# Quick script to check feature engineering progress

echo "=== Feature Engineering Progress Check ==="
echo ""

# Check if process is running
if ps aux | grep -v grep | grep -q "run_feature_engineering"; then
    echo "✓ Process is running"
    PID=$(ps aux | grep -v grep | grep "run_feature_engineering" | awk '{print $2}')
    echo "  PID: $PID"
else
    echo "✗ Process is not running"
fi

echo ""

# Check latest log entries
if [ -f "feature_engineering_50m.log" ]; then
    echo "Latest progress:"
    tail -5 feature_engineering_50m.log | grep -E "Processing target point|Completed|COMPLETE" | tail -3
    echo ""
    
    # Count total points processed
    echo "Total points processed so far:"
    python3 -c "
import pandas as pd
from pathlib import Path

total = 0
for parquet_file in Path('data/processed/features/compliant').rglob('target_points.parquet'):
    try:
        df = pd.read_parquet(parquet_file)
        total += len(df)
    except:
        pass
print(f'  {total} points')
" 2>/dev/null || echo "  (Could not count)"
else
    echo "Log file not found"
fi

echo ""
echo "To monitor in real-time: tail -f feature_engineering_50m.log"
