#!/bin/bash
# Monitor feature engineering progress

LOG_FILE="/tmp/feature_engineering_all.log"
OUTPUT_DIR="/Users/maayanchen/Code/University/AI4SI_Project/data/processed/features"

echo "=== Feature Engineering Progress Monitor ==="
echo "Started at: $(date)"
echo ""

# Check if process is running
if pgrep -f "run_feature_engineering.py" > /dev/null; then
    echo "✅ Script is RUNNING"
    PID=$(pgrep -f "run_feature_engineering.py")
    echo "   Process ID: $PID"
    
    # Get CPU and memory usage
    ps -p $PID -o %cpu,%mem,etime,command | tail -1
else
    echo "❌ Script is NOT running"
fi

echo ""
echo "=== Progress Statistics ==="

# Count processed neighborhoods (parquet files)
PARQUET_COUNT=$(find "$OUTPUT_DIR" -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
echo "Output files created: $PARQUET_COUNT"

# Count directories (neighborhoods)
DIR_COUNT=$(find "$OUTPUT_DIR" -type d -mindepth 1 2>/dev/null | wc -l | tr -d ' ')
echo "Neighborhood directories: $DIR_COUNT"

# List completed neighborhoods
echo ""
echo "Completed neighborhoods:"
find "$OUTPUT_DIR" -name "*.parquet" 2>/dev/null | sed 's|.*/||' | sed 's|/.*||' | sort -u

echo ""
echo "=== Recent Log Output (last 20 lines) ==="
tail -20 "$LOG_FILE" 2>/dev/null || echo "No log file found"

echo ""
echo "=== Error Count ==="
ERROR_COUNT=$(grep -i "error" "$LOG_FILE" 2>/dev/null | wc -l | tr -d ' ')
echo "Total errors in log: $ERROR_COUNT"

echo ""
echo "Checked at: $(date)"
echo "=========================================="
