"""Test script to verify CensusLoader saves IRIS-level data correctly.

This script tests that:
1. Census data is saved at IRIS level (multiple rows per neighborhood, not aggregated)
2. IRIS code columns are preserved
3. Each row represents one IRIS unit with its own demographic features
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.collection.census_loader import CensusLoader
from src.utils.helpers import load_neighborhoods


def test_census_loader_iris_level():
    """Test that CensusLoader saves IRIS-level data correctly."""
    print("=" * 80)
    print("Testing CensusLoader IRIS-Level Data")
    print("=" * 80)

    # Load neighborhoods
    neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
    compliant = neighborhoods[neighborhoods["label"] == "Compliant"]

    if len(compliant) == 0:
        print("❌ No compliant neighborhoods found!")
        return

    # Pick first compliant neighborhood for testing
    test_neighborhood = compliant.iloc[0]
    neighborhood_name = test_neighborhood["name"]
    print(f"\n📋 Testing with neighborhood: {neighborhood_name}")

    # Initialize loader
    loader = CensusLoader()

    # Process the neighborhood (force=True to regenerate)
    print(f"\n🔄 Processing {neighborhood_name}...")
    result = loader.load_neighborhood_census(test_neighborhood, force=True)

    if result["status"] != "success":
        print(f"❌ Failed to load census data: {result.get('error', 'Unknown error')}")
        return

    print(f"✅ Census data loaded successfully")
    print(f"   - Features count: {result.get('features_count', 'N/A')}")
    print(f"   - IRIS units count: {result.get('iris_units_count', 'N/A')}")

    # Load the saved census data
    normalized_name = neighborhood_name.lower().replace(" ", "_")
    census_path = Path(f"data/raw/census/compliant/{normalized_name}/census_data.parquet")

    if not census_path.exists():
        print(f"❌ Census data file not found at: {census_path}")
        return

    print(f"\n📂 Loading saved census data from: {census_path}")
    census_data = pd.read_parquet(census_path)

    print(f"\n📊 Census Data Structure:")
    print(f"   - Number of rows: {len(census_data)}")
    print(f"   - Columns: {len(census_data.columns)}")
    print(f"   - Column names: {census_data.columns.tolist()}")

    # Check for IRIS code columns
    iris_code_cols = [col for col in census_data.columns 
                     if col in ["IRIS", "CODE_IRIS", "DCOMIRIS", "CODEGEO"]]
    
    print(f"\n🔍 IRIS Code Columns Found: {iris_code_cols}")

    if not iris_code_cols:
        print("❌ ERROR: No IRIS code columns found in census data!")
        print("   This means grid cells cannot be matched to IRIS units.")
        return
    else:
        print(f"✅ IRIS code columns present: {iris_code_cols}")

    # Check if data is IRIS-level (multiple rows) or aggregated (single row)
    if len(census_data) == 1:
        print(f"\n⚠️  WARNING: Only 1 row found - data appears to be aggregated!")
        print(f"   Expected: Multiple rows (one per IRIS unit)")
        print(f"   This means all grid cells in the neighborhood will get the same features.")
    else:
        print(f"\n✅ Multiple rows found ({len(census_data)} rows) - data is IRIS-level!")
        print(f"   Each row represents one IRIS unit with its own demographic features.")

    # Show sample data
    print(f"\n📋 Sample Data (first 3 rows):")
    print(census_data.head(3).to_string())

    # Check for unique IRIS codes
    if iris_code_cols:
        iris_col = iris_code_cols[0]
        unique_iris = census_data[iris_col].nunique()
        print(f"\n🔢 Unique IRIS codes: {unique_iris}")
        print(f"   Total rows: {len(census_data)}")
        
        if unique_iris == len(census_data):
            print(f"✅ Each row has a unique IRIS code - perfect!")
        else:
            print(f"⚠️  Some IRIS codes appear multiple times")

    # Check feature columns
    feature_cols = [col for col in census_data.columns 
                   if col not in ["neighborhood_name", "IRIS", "CODE_IRIS", "DCOMIRIS", "CODEGEO"]]
    
    print(f"\n📈 Feature Columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"   {i}. {col}")
    if len(feature_cols) > 10:
        print(f"   ... and {len(feature_cols) - 10} more")

    # Verify feature values are different across IRIS units (if multiple rows)
    if len(census_data) > 1 and len(feature_cols) > 0:
        print(f"\n🔬 Checking feature variation across IRIS units...")
        sample_feature = feature_cols[0]
        if sample_feature in census_data.columns:
            unique_values = census_data[sample_feature].nunique()
            print(f"   - {sample_feature}: {unique_values} unique values out of {len(census_data)} IRIS units")
            if unique_values > 1:
                print(f"   ✅ Features vary across IRIS units - IRIS-level data confirmed!")
            else:
                print(f"   ⚠️  All IRIS units have the same value - might be aggregated")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

    # Summary
    print("\n📝 Summary:")
    if len(census_data) > 1 and iris_code_cols:
        print("✅ PASS: Census data is saved at IRIS level with IRIS codes preserved")
        print("   - FeatureEngineer can now match grid cells to specific IRIS units")
        print("   - Each grid cell will get accurate demographic features")
    elif len(census_data) == 1:
        print("❌ FAIL: Data is aggregated (only 1 row per neighborhood)")
        print("   - All grid cells will get the same features")
        print("   - Need to check CensusLoader aggregation logic")
    elif not iris_code_cols:
        print("❌ FAIL: IRIS code columns missing")
        print("   - Cannot match grid cells to IRIS units")
        print("   - Need to check _extract_demographic_features() method")


if __name__ == "__main__":
    test_census_loader_iris_level()
