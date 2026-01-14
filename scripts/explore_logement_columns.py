"""Quick script to explore RP_LOGEMENT_2017 column names.

This script downloads RP_LOGEMENT_2017 and inspects the actual column names
to see what car ownership variables are really available.
"""

import pandas as pd
from pynsee import download_file

print("=" * 80)
print("Exploring RP_LOGEMENT_2017 Column Names")
print("=" * 80)

try:
    print("\n1. Attempting to download RP_LOGEMENT_2017...")
    print("   (This may take a few minutes - file is ~373MB)")
    print("   We'll inspect column names once download starts...")
    
    # Try to download - we'll stop early if possible, or just read first few rows
    logement_data = download_file("RP_LOGEMENT_2017")
    
    if logement_data is None or logement_data.empty:
        print("   ❌ Failed to download RP_LOGEMENT_2017")
    else:
        print(f"\n   ✅ Downloaded RP_LOGEMENT_2017!")
        print(f"   Shape: {logement_data.shape}")
        print(f"   Total columns: {len(logement_data.columns)}")
        
        print("\n2. All Column Names:")
        for i, col in enumerate(logement_data.columns, 1):
            print(f"   {i:3d}. {col}")
        
        print("\n3. Searching for car ownership related columns...")
        car_keywords = ["VOIT", "MEN", "LOG", "PCT", "NB", "C17", "P17"]
        car_cols = []
        for col in logement_data.columns:
            col_upper = col.upper()
            if any(keyword in col_upper for keyword in car_keywords):
                car_cols.append(col)
        
        if car_cols:
            print(f"   ✅ Found {len(car_cols)} potentially relevant columns:")
            for col in car_cols:
                non_null = logement_data[col].notna().sum() if col in logement_data.columns else 0
                print(f"      - {col}: {non_null} non-null values")
        else:
            print("   ⚠️  No columns found matching car ownership keywords")
        
        print("\n4. Sample data (first row):")
        if len(logement_data) > 0:
            sample = logement_data.iloc[0]
            print(f"   IRIS/Code column: {sample.get('IRIS', sample.get('CODEGEO', sample.get('CODE_IRIS', 'N/A')))}")
            print(f"   COM column: {sample.get('COM', 'N/A')}")
            
            # Show sample values for car-related columns
            if car_cols:
                print("\n   Sample values for car-related columns:")
                for col in car_cols[:10]:
                    val = sample.get(col, 'N/A')
                    print(f"      {col}: {val}")
        
        print("\n5. Checking for specific expected columns:")
        expected_cols = [
            "IRIS", "CODEGEO", "CODE_IRIS", "COM",
            "P17_MEN", "C17_MEN_VOIT1", "C17_MEN_VOIT2P",
            "PCT_MEN_VOIT", "NB_MEN_VOIT", "P17_LOG"
        ]
        found_cols = []
        missing_cols = []
        for col in expected_cols:
            if col in logement_data.columns:
                found_cols.append(col)
                print(f"   ✅ {col}: FOUND")
            else:
                missing_cols.append(col)
                print(f"   ❌ {col}: NOT FOUND")
        
        print(f"\n   Summary: {len(found_cols)}/{len(expected_cols)} expected columns found")
        
        if missing_cols:
            print("\n   ⚠️  Missing columns:")
            for col in missing_cols:
                print(f"      - {col}")
            print("\n   💡 Suggestion: Search for similar column names...")
            for missing in missing_cols:
                # Try to find similar column names
                similar = [c for c in logement_data.columns if missing.split('_')[-1] in c.upper() or missing.split('_')[0] in c.upper()]
                if similar:
                    print(f"      '{missing}' might be: {similar[:3]}")
        
except Exception as e:
    print(f"\n   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
