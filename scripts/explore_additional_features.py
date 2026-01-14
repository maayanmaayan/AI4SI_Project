"""Script to explore additional IRIS-level features available in pynsee.

This script checks availability of:
1. Income/Poverty data (FILOSOFI)
2. Unemployment rate (from RP_ACTRES_IRIS)
3. Education/Student population (from RP_ACTRES_IRIS)
4. Commuting modes (walking, cycling, public transport) (from RP_ACTRES_IRIS)
5. Retired population ratio (from RP_ACTRES_IRIS)
6. Employment contract types (from RP_ACTRES_IRIS)
"""

import pandas as pd
from pynsee import download_file, get_file_list


def explore_rp_actres_iris_features():
    """Explore additional features in RP_ACTRES_IRIS dataset."""
    print("=" * 80)
    print("Exploring RP_ACTRES_IRIS for Additional Features")
    print("=" * 80)

    try:
        print("\n1. Downloading RP_ACTRES_IRIS dataset...")
        iris_data = download_file("RP_ACTRES_IRIS")

        if iris_data is None or iris_data.empty:
            print("   ❌ Failed to download RP_ACTRES_IRIS")
            return {}

        print(f"   ✅ Downloaded {len(iris_data)} IRIS units")
        print(f"   Columns: {len(iris_data.columns)}")

        # Filter for Paris
        if "COM" in iris_data.columns:
            paris_data = iris_data[iris_data["COM"].str.startswith("75", na=False)]
            print(f"   ✅ Paris IRIS units: {len(paris_data)}")
        else:
            paris_data = iris_data
            print("   ⚠️  COM column not found, using all data")

        # Convert numeric columns
        numeric_cols = []
        for col in paris_data.columns:
            if col not in ["IRIS", "REG", "DEP", "COM", "LIBCOM", "LIBIRIS", "TYP_IRIS", "MODIF_IRIS", "LAB_IRIS"]:
                try:
                    paris_data[col] = pd.to_numeric(paris_data[col], errors="coerce")
                    if paris_data[col].notna().sum() > 0:
                        numeric_cols.append(col)
                except:
                    pass

        print(f"\n   Found {len(numeric_cols)} numeric columns with data")

        # Categorize variables
        features_found = {}

        # 1. Unemployment (CHOM)
        print("\n2. Checking for Unemployment Variables (CHOM)...")
        chom_cols = [c for c in numeric_cols if "CHOM" in c]
        if chom_cols:
            print(f"   ✅ Found {len(chom_cols)} unemployment variables:")
            for col in chom_cols[:5]:
                non_null = paris_data[col].notna().sum()
                print(f"      - {col}: {non_null} non-null values")
            if len(chom_cols) > 5:
                print(f"      ... and {len(chom_cols) - 5} more")
            features_found["unemployment"] = chom_cols
        else:
            print("   ❌ No unemployment variables found")

        # 2. Education/Students (ETUD)
        print("\n3. Checking for Education/Student Variables (ETUD)...")
        etud_cols = [c for c in numeric_cols if "ETUD" in c]
        if etud_cols:
            print(f"   ✅ Found {len(etud_cols)} education/student variables:")
            for col in etud_cols[:5]:
                non_null = paris_data[col].notna().sum()
                print(f"      - {col}: {non_null} non-null values")
            if len(etud_cols) > 5:
                print(f"      ... and {len(etud_cols) - 5} more")
            features_found["education_students"] = etud_cols
        else:
            print("   ❌ No education/student variables found")

        # 3. Commuting Modes
        print("\n4. Checking for Commuting Mode Variables...")
        # Walking (PAS = à pied)
        pas_cols = [c for c in numeric_cols if "PAS" in c or ("ACTOCC" in c and "PAS" in c)]
        # Cycling (VELO)
        velo_cols = [c for c in numeric_cols if "VELO" in c or ("ACTOCC" in c and "VELO" in c)]
        # Public transport (TCOM)
        tcom_cols = [c for c in numeric_cols if "TCOM" in c or ("ACTOCC" in c and "TCOM" in c)]
        # Walking alternative (MAR = marche)
        mar_cols = [c for c in numeric_cols if "MAR" in c or ("ACTOCC" in c and "MAR" in c)]

        commuting_cols = {
            "walking": pas_cols + mar_cols,
            "cycling": velo_cols,
            "public_transport": tcom_cols,
        }

        if any(commuting_cols.values()):
            print(f"   ✅ Found commuting mode variables:")
            for mode, cols in commuting_cols.items():
                if cols:
                    print(f"      - {mode}: {len(cols)} variables")
                    for col in cols[:3]:
                        non_null = paris_data[col].notna().sum()
                        print(f"        * {col}: {non_null} non-null values")
            features_found["commuting_modes"] = commuting_cols
        else:
            print("   ❌ No commuting mode variables found")

        # 4. Retired Population (RETR)
        print("\n5. Checking for Retired Population Variables (RETR)...")
        retr_cols = [c for c in numeric_cols if "RETR" in c]
        if retr_cols:
            print(f"   ✅ Found {len(retr_cols)} retired population variables:")
            for col in retr_cols[:5]:
                non_null = paris_data[col].notna().sum()
                print(f"      - {col}: {non_null} non-null values")
            if len(retr_cols) > 5:
                print(f"      ... and {len(retr_cols) - 5} more")
            features_found["retired"] = retr_cols
        else:
            print("   ❌ No retired population variables found")

        # 5. Employment Contract Types
        print("\n6. Checking for Employment Contract Type Variables...")
        # Permanent (CDI)
        cdi_cols = [c for c in numeric_cols if "CDI" in c]
        # Temporary (CDD)
        cdd_cols = [c for c in numeric_cols if "CDD" in c]
        # Interim/Temporary
        interim_cols = [c for c in numeric_cols if "INTERIM" in c or "INT" in c]

        contract_cols = {
            "permanent": cdi_cols,
            "temporary": cdd_cols,
            "interim": interim_cols,
        }

        if any(contract_cols.values()):
            print(f"   ✅ Found employment contract variables:")
            for contract_type, cols in contract_cols.items():
                if cols:
                    print(f"      - {contract_type}: {len(cols)} variables")
                    for col in cols[:3]:
                        non_null = paris_data[col].notna().sum()
                        print(f"        * {col}: {non_null} non-null values")
            features_found["employment_contracts"] = contract_cols
        else:
            print("   ❌ No employment contract variables found")

        # Sample data check
        print("\n7. Sample Data Check (first Paris IRIS unit):")
        if len(paris_data) > 0:
            sample = paris_data.iloc[0]
            print(f"   IRIS Code: {sample.get('IRIS', 'N/A')}")
            print(f"   Commune: {sample.get('COM', 'N/A')}")
            
            # Show sample values for found features
            for feature_type, cols in features_found.items():
                if isinstance(cols, dict):
                    for sub_type, sub_cols in cols.items():
                        if sub_cols:
                            sample_col = sub_cols[0]
                            print(f"   {feature_type}.{sub_type}: {sample.get(sample_col, 'N/A')}")
                elif isinstance(cols, list) and cols:
                    sample_col = cols[0]
                    print(f"   {feature_type}: {sample.get(sample_col, 'N/A')}")

        return features_found

    except Exception as e:
        print(f"   ❌ Error exploring RP_ACTRES_IRIS: {e}")
        import traceback
        traceback.print_exc()
        return {}


def explore_filosofi_income():
    """Explore FILOSOFI income dataset availability."""
    print("\n" + "=" * 80)
    print("Exploring FILOSOFI Income Dataset")
    print("=" * 80)

    try:
        # First, check what FILOSOFI files are available
        print("\n1. Checking available FILOSOFI files...")
        file_list = get_file_list()
        
        filosofi_files = file_list[file_list.id.str.contains("FILOSOFI", case=False, na=False)]
        filosofi_iris_files = filosofi_files[filosofi_files.id.str.contains("IRIS", case=False, na=False)]
        
        if len(filosofi_iris_files) > 0:
            print(f"   ✅ Found {len(filosofi_iris_files)} FILOSOFI IRIS-level files:")
            print(filosofi_iris_files[["id", "label", "date_ref"]].to_string())
            
            # Try to download the most recent one
            latest_file = filosofi_iris_files.iloc[0]["id"]
            print(f"\n2. Attempting to download: {latest_file}...")
            
            try:
                filosofi_data = download_file(latest_file)
                
                if filosofi_data is not None and not filosofi_data.empty:
                    print(f"   ✅ Successfully downloaded FILOSOFI data!")
                    print(f"   Shape: {filosofi_data.shape}")
                    print(f"   Columns: {len(filosofi_data.columns)}")
                    
                    # Filter for Paris
                    if "COM" in filosofi_data.columns:
                        paris_filosofi = filosofi_data[filosofi_data["COM"].str.startswith("75", na=False)]
                        print(f"   ✅ Paris IRIS units: {len(paris_filosofi)}")
                    else:
                        paris_filosofi = filosofi_data
                        print("   ⚠️  COM column not found")
                    
                    # Check for income-related columns
                    print("\n3. Checking for income-related variables...")
                    income_keywords = ["REV", "MED", "DISP", "DEC", "QUART", "POVERTY", "PAUVRETE"]
                    income_cols = []
                    for col in filosofi_data.columns:
                        if any(keyword in col.upper() for keyword in income_keywords):
                            income_cols.append(col)
                    
                    if income_cols:
                        print(f"   ✅ Found {len(income_cols)} income-related variables:")
                        for col in income_cols[:10]:
                            print(f"      - {col}")
                        if len(income_cols) > 10:
                            print(f"      ... and {len(income_cols) - 10} more")
                        
                        # Sample data
                        if len(paris_filosofi) > 0:
                            sample = paris_filosofi.iloc[0]
                            print(f"\n   Sample data (first Paris IRIS):")
                            print(f"   IRIS Code: {sample.get('IRIS', sample.get('CODEGEO', 'N/A'))}")
                            for col in income_cols[:5]:
                                print(f"   {col}: {sample.get(col, 'N/A')}")
                        
                        return {
                            "available": True,
                            "file_id": latest_file,
                            "columns": income_cols,
                            "paris_units": len(paris_filosofi) if "paris_filosofi" in locals() else 0,
                        }
                    else:
                        print("   ⚠️  No income-related variables found in expected columns")
                        print(f"   Available columns: {list(filosofi_data.columns)[:20]}")
                        return {
                            "available": True,
                            "file_id": latest_file,
                            "columns": list(filosofi_data.columns),
                            "paris_units": len(paris_filosofi) if "paris_filosofi" in locals() else 0,
                        }
                else:
                    print(f"   ❌ Failed to download FILOSOFI data")
                    return {"available": False, "error": "Download failed"}
            except Exception as e:
                print(f"   ❌ Error downloading FILOSOFI: {e}")
                import traceback
                traceback.print_exc()
                return {"available": False, "error": str(e)}
        else:
            print("   ❌ No FILOSOFI IRIS-level files found")
            return {"available": False, "error": "No FILOSOFI IRIS files available"}

    except Exception as e:
        print(f"   ❌ Error exploring FILOSOFI: {e}")
        import traceback
        traceback.print_exc()
        return {"available": False, "error": str(e)}


def main():
    """Main exploration function."""
    print("\n" + "=" * 80)
    print("EXPLORING ADDITIONAL IRIS-LEVEL FEATURES")
    print("=" * 80)

    # Explore RP_ACTRES_IRIS features
    rp_features = explore_rp_actres_iris_features()

    # Explore FILOSOFI income data
    filosofi_info = explore_filosofi_income()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n✅ Features Available in RP_ACTRES_IRIS:")
    for feature_type, cols in rp_features.items():
        if isinstance(cols, dict):
            print(f"   - {feature_type}:")
            for sub_type, sub_cols in cols.items():
                if sub_cols:
                    print(f"     * {sub_type}: {len(sub_cols)} variables")
        elif isinstance(cols, list):
            print(f"   - {feature_type}: {len(cols)} variables")

    print("\n💰 FILOSOFI Income Data:")
    if filosofi_info.get("available"):
        print(f"   ✅ Available: {filosofi_info.get('file_id', 'N/A')}")
        print(f"   Paris IRIS units: {filosofi_info.get('paris_units', 0)}")
        if "columns" in filosofi_info:
            print(f"   Income variables: {len(filosofi_info['columns'])}")
    else:
        print(f"   ❌ Not available: {filosofi_info.get('error', 'Unknown error')}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    available_features = []
    if rp_features:
        available_features.append("RP_ACTRES_IRIS features")
    if filosofi_info.get("available"):
        available_features.append("FILOSOFI income data")
    
    if available_features:
        print("\n✅ Ready to add the following features to CensusLoader:")
        for feat in available_features:
            print(f"   - {feat}")
        print("\nNext steps:")
        print("   1. Update CensusLoader._fetch_iris_data() to include new variables")
        print("   2. Add FILOSOFI fetching method if income data is available")
        print("   3. Update _extract_demographic_features() to calculate new features")
        print("   4. Test with a sample neighborhood")
    else:
        print("\n⚠️  Limited additional features found. Current features may be sufficient.")


if __name__ == "__main__":
    main()
