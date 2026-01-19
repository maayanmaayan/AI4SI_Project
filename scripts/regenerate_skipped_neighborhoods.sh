#!/bin/bash
# Script to regenerate neighborhoods that were skipped (still have 100m data)
# Run this after the main feature engineering process completes

echo "Regenerating neighborhoods that still have 100m data..."
echo ""

# Delete old files for neighborhoods that need regeneration
echo "Deleting old 100m data files..."
rm -f "data/processed/features/compliant/bartholomé–brancion_(15e)/target_points.parquet"
rm -f "data/processed/features/compliant/_15th_arr./target_points.parquet"
rm -f "data/processed/features/compliant/bédier–oudiné_(13e)/target_points.parquet"
rm -f "data/processed/features/compliant/_place_de_vénétie_(13e)/target_points.parquet"

echo "✓ Old files deleted"
echo ""

# Regenerate each neighborhood with 50m spacing
echo "Regenerating neighborhoods with 50m spacing..."
echo ""

python scripts/run_feature_engineering.py --neighborhood "Bartholomé–Brancion (15e)" --force
python scripts/run_feature_engineering.py --neighborhood "Beaugrenelle / 15th arr." --force
python scripts/run_feature_engineering.py --neighborhood "Bédier–Oudiné (13e)" --force
python scripts/run_feature_engineering.py --neighborhood "Maine–Montparnasse (6e)" --force

echo ""
echo "✓ All skipped neighborhoods regenerated!"
