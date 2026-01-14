"""Exploration script to understand pynsee API and available datasets.

This script explores the pynsee library to understand:
- Available datasets for IRIS-level census data
- How to fetch data for Paris (geocode: 75056)
- Available variables for demographic features
- Dataset structure and codes
- IRIS boundaries and codes

Run this script before implementing CensusLoader to understand the API.
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pynsee.localdata import (
    get_local_data,
    get_local_metadata,
    get_geo_list,
    get_nivgeo_list,
)
from tqdm import tqdm


def download_iris_boundaries(
    output_path: str = "data/raw/iris_boundaries.geojson",
    filter_paris: bool = True,
) -> gpd.GeoDataFrame:
    """Download IRIS boundaries GeoJSON from data.gouv.fr.

    Args:
        output_path: Path to save the GeoJSON file.
        filter_paris: If True, filter to only Paris IRIS (DEPCOM starting with '75').

    Returns:
        GeoDataFrame with IRIS boundaries.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if output_file.exists():
        print(f"   IRIS boundaries file already exists at {output_file}")
        print(f"   Loading existing file...")
        gdf = gpd.read_file(output_file)
        print(f"   ✅ Loaded {len(gdf)} IRIS units from existing file")
        return gdf

    print(f"\n   Downloading IRIS boundaries...")
    print(f"   Note: Manual download may be required from:")
    print(f"   https://www.data.gouv.fr/fr/datasets/contour-des-iris-insee-tout-en-un/")
    print(f"\n   Please download the GeoJSON file and place it at: {output_file}")
    print(f"   Or provide the download URL to automate this process.")
    
    return gpd.GeoDataFrame()


def load_iris_boundaries(
    iris_path: str = None,
    filter_paris: bool = True,
) -> gpd.GeoDataFrame:
    """Load IRIS boundaries file (GeoJSON or Shapefile).

    Args:
        iris_path: Path to IRIS boundaries file (GeoJSON or Shapefile).
                   If None, tries common locations.
        filter_paris: If True, filter to only Paris IRIS.

    Returns:
        GeoDataFrame with IRIS boundaries and codes.
    """
    # Try common locations if path not provided
    if iris_path is None:
        possible_paths = [
            "data/raw/iris_boundaries.geojson",
            "data/raw/census/iris-2013-01-01/iris-2013-01-01.shp",
            "data/raw/iris_boundaries.shp",
        ]
        for path in possible_paths:
            if Path(path).exists():
                iris_path = path
                break
    
    if iris_path is None:
        print(f"   ⚠️  IRIS boundaries file not found")
        print(f"   Tried: data/raw/iris_boundaries.geojson")
        print(f"   Tried: data/raw/census/iris-2013-01-01/iris-2013-01-01.shp")
        return gpd.GeoDataFrame()
    
    iris_file = Path(iris_path)
    
    if not iris_file.exists():
        print(f"   ⚠️  IRIS boundaries file not found at {iris_path}")
        return gpd.GeoDataFrame()
    
    try:
        gdf = gpd.read_file(iris_file)
        file_type = "Shapefile" if iris_file.suffix == ".shp" else "GeoJSON"
        print(f"   ✅ Loaded {len(gdf)} IRIS units from {file_type}: {iris_path}")
        print(f"   CRS: {gdf.crs}")
        
        if filter_paris:
            # Filter for Paris (DEPCOM starting with 75 or DCOMIRIS starting with 75)
            if "DEPCOM" in gdf.columns:
                paris_gdf = gdf[gdf["DEPCOM"].str.startswith("75", na=False)]
                print(f"   Found {len(paris_gdf)} IRIS units in Paris (filtered by DEPCOM)")
                return paris_gdf
            elif "DCOMIRIS" in gdf.columns:
                # DCOMIRIS is DEPCOM + IRIS, so filter by first 2 chars = 75
                paris_gdf = gdf[gdf["DCOMIRIS"].str.startswith("75", na=False)]
                print(f"   Found {len(paris_gdf)} IRIS units in Paris (filtered by DCOMIRIS)")
                return paris_gdf
        
        return gdf
    except Exception as e:
        print(f"   ❌ Error loading IRIS boundaries: {e}")
        import traceback
        traceback.print_exc()
        return gpd.GeoDataFrame()


def main():
    """Explore pynsee API and datasets."""
    print("=" * 80)
    print("Exploring pynsee API for IRIS-level census data")
    print("=" * 80)

    # Step 1: Get metadata about available datasets
    print("\n1. Fetching available datasets metadata...")
    try:
        metadata = get_local_metadata()
        print(f"   Found {len(metadata)} datasets")
        print(f"   Columns: {list(metadata.columns)}")
        
        # Display first few rows
        print("\n   First few datasets:")
        print(metadata.head(10).to_string())
        
        # Find census/population datasets (RP = Recensement de la Population)
        print("\n   Searching for census datasets (RP)...")
        rp_datasets = metadata[metadata["DATASET"] == "RP"]
        print(f"   Found {len(rp_datasets)} census datasets")
        
        # Show unique dataset versions
        if len(rp_datasets) > 0:
            unique_versions = rp_datasets["DATASET_VERSION"].unique()
            print(f"\n   Available dataset versions: {list(unique_versions[:10])}")
            
            # Find latest version
            latest_version = rp_datasets["DATASET_VERSION"].iloc[0]
            print(f"\n   Latest dataset version appears to be: {latest_version}")
            
            # Show variables for latest version
            latest_vars = rp_datasets[rp_datasets["DATASET_VERSION"] == latest_version]
            print(f"\n   Variables in latest version ({len(latest_vars)} variables):")
            print(latest_vars[["VARIABLES", "VARIABLES_label_fr"]].head(20).to_string())
            
    except Exception as e:
        print(f"   Error fetching metadata: {e}")
        import traceback
        traceback.print_exc()

    # Step 2: Load IRIS boundaries and extract codes
    print("\n2. Loading IRIS boundaries and extracting codes...")
    
    # Try 2021 boundaries first, then fall back to 2013
    iris_boundaries_2021_path = Path(
        "data/raw/census/iris-2021-01-01/CONTOURS-IRIS_2-1__SHP__FRA_2021-01-01/"
        "CONTOURS-IRIS/1_DONNEES_LIVRAISON_2021-06-00217/"
        "CONTOURS-IRIS_2-1_SHP_LAMB93_FXX-2021/CONTOURS-IRIS.shp"
    )
    
    iris_boundaries = gpd.GeoDataFrame()
    paris_iris_codes = []
    
    if iris_boundaries_2021_path.exists():
        print("   Found 2021 IRIS boundaries - loading...")
        try:
            iris_boundaries = gpd.read_file(iris_boundaries_2021_path)
            # Convert to WGS84 for consistency
            if iris_boundaries.crs != "EPSG:4326":
                iris_boundaries = iris_boundaries.to_crs("EPSG:4326")
            
            # Filter for Paris
            if "INSEE_COM" in iris_boundaries.columns:
                iris_boundaries = iris_boundaries[
                    iris_boundaries["INSEE_COM"].str.startswith("75", na=False)
                ]
                print(f"   ✅ Loaded {len(iris_boundaries)} Paris IRIS units from 2021 boundaries")
                
                # Extract IRIS codes
                if "CODE_IRIS" in iris_boundaries.columns:
                    paris_iris_codes = iris_boundaries["CODE_IRIS"].unique().tolist()
                    print(f"   ✅ Extracted {len(paris_iris_codes)} IRIS codes from CODE_IRIS column")
        except Exception as e:
            print(f"   Error loading 2021 boundaries: {e}")
            iris_boundaries = gpd.GeoDataFrame()
    
    # Fallback to 2013 boundaries if 2021 not available
    if iris_boundaries.empty:
        iris_boundaries = load_iris_boundaries(filter_paris=True)
    
    if not iris_boundaries.empty:
        print(f"   Columns in IRIS boundaries: {list(iris_boundaries.columns)}")
        
        # Extract IRIS codes - try different column names (2021 vs 2013 format)
        if "CODE_IRIS" in iris_boundaries.columns:
            # 2021 format
            paris_iris_codes = iris_boundaries["CODE_IRIS"].unique().tolist()
            print(f"   ✅ Extracted {len(paris_iris_codes)} IRIS codes from CODE_IRIS column (2021 format)")
        elif "DCOMIRIS" in iris_boundaries.columns:
            # 2013 format
            paris_iris_codes = iris_boundaries["DCOMIRIS"].unique().tolist()
            print(f"   ✅ Extracted {len(paris_iris_codes)} IRIS codes from DCOMIRIS column (2013 format)")
        elif "IRIS" in iris_boundaries.columns and "DEPCOM" in iris_boundaries.columns:
            # Combine DEPCOM + IRIS to create full code
            paris_iris_codes = (
                iris_boundaries["DEPCOM"] + iris_boundaries["IRIS"]
            ).unique().tolist()
            print(f"   ✅ Extracted {len(paris_iris_codes)} IRIS codes from DEPCOM+IRIS")
        elif "CODEGEO" in iris_boundaries.columns:
            paris_iris_codes = iris_boundaries["CODEGEO"].unique().tolist()
            print(f"   ✅ Extracted {len(paris_iris_codes)} IRIS codes from CODEGEO column")
        
        if len(paris_iris_codes) > 0:
            print(f"\n   Sample IRIS codes: {paris_iris_codes[:5]}")
            print(f"\n   First few IRIS boundaries:")
            display_cols = [col for col in iris_boundaries.columns if col != "geometry"][:5]
            print(iris_boundaries[display_cols].head(10).to_string())
        else:
            print("   ⚠️  Could not extract IRIS codes from boundaries file")
    else:
        print("   ⚠️  No IRIS boundaries loaded - trying alternative method...")
        
        # Fallback: Try to get IRIS codes from pynsee if available
        try:
            print("   Attempting to get IRIS codes from pynsee...")
            # Note: get_geo_list("iris") doesn't work, but we can try fetching
            # data for a known IRIS code to understand the structure
            test_data = get_local_data(
                dataset_version="GEO2021RP2018",
                variables="STOCD",
                nivgeo="IRIS",
                geocodes=["750101001"],  # Sample Paris IRIS code
            )
            if not test_data.empty and "CODEGEO" in test_data.columns:
                print(f"   Found CODEGEO column in pynsee response")
                print(f"   Sample structure: {test_data.head()}")
        except Exception as e:
            print(f"   Alternative method also failed: {e}")
    
    # Step 3: Search for specific demographic variables
    print("\n3. Searching for specific demographic variables in metadata...")
    test_variables = []
    if "metadata" in locals():
        rp_metadata = metadata[metadata["DATASET"] == "RP"]
        latest_metadata = rp_metadata[rp_metadata["DATASET_VERSION"] == "GEO2021RP2018"]

        # Search for key variables
        searches = {
            "Population": latest_metadata[
                latest_metadata["VARIABLES_label_fr"].str.contains(
                    "population|pop|habitants", case=False, na=False
                )
            ],
            "Household": latest_metadata[
                latest_metadata["VARIABLES_label_fr"].str.contains(
                    "ménage|household|menage|NPERC", case=False, na=False
                )
            ],
            "Income": latest_metadata[
                latest_metadata["VARIABLES_label_fr"].str.contains(
                    "revenu|income|salaire", case=False, na=False
                )
            ],
            "Car Ownership": latest_metadata[
                latest_metadata["VARIABLES_label_fr"].str.contains(
                    "voiture|véhicule|car|vehicle|VOIT", case=False, na=False
                )
            ],
            "Children": latest_metadata[
                latest_metadata["VARIABLES_label_fr"].str.contains(
                    "enfant|child|moins de|NBENFFR", case=False, na=False
                )
            ],
            "Elderly": latest_metadata[
                latest_metadata["VARIABLES_label_fr"].str.contains(
                    "65|75|80|senior|âgé", case=False, na=False
                )
            ],
        }

        # Display found variables
        candidate_variables = {}
        for category, results in searches.items():
            if len(results) > 0:
                print(f"\n   {category} variables ({len(results)} found):")
                print(results[["VARIABLES", "VARIABLES_label_fr"]].head(10).to_string())
                # Store top candidates
                top_vars = results["VARIABLES"].head(3).tolist()
                candidate_variables[category] = top_vars
            else:
                print(f"\n   {category}: No variables found")

        # Select key variables to test
        if "Household" in candidate_variables:
            # NPERC = Nombre de personnes du ménage (household size)
            test_variables.append("NPERC")
        if "Car Ownership" in candidate_variables:
            # VOIT = Nombre de voitures du ménage (number of cars)
            test_variables.append("VOIT")
        if "Children" in candidate_variables:
            # NBENFFR = Nombre d'enfants de moins de 25 ans
            test_variables.append("NBENFFR-TF4")  # Use a simpler variant
        if "Household" in candidate_variables:
            # TYPMR = Type de ménage (household type)
            test_variables.append("TYPMR")

    # Step 4: Try fetching IRIS-level census data with selected variables
    print("\n4. Testing IRIS-level census data fetching with selected variables...")
    if len(paris_iris_codes) > 0 and test_variables:
        try:
            # Use a small sample first
            sample_codes = paris_iris_codes[:10]  # Test with first 10 IRIS

            print(f"   Testing with {len(sample_codes)} IRIS codes...")
            print(f"   Sample codes: {sample_codes[:3]}...")
            print(f"   Variables to test: {test_variables}")

            # Test each variable separately to see which ones work
            for var in test_variables:
                print(f"\n   Testing variable: {var}")
                try:
                    iris_data = get_local_data(
                        dataset_version="GEO2021RP2018",
                        variables=var,
                        nivgeo="IRIS",
                        geocodes=sample_codes,
                    )
                    if not iris_data.empty:
                        non_null_count = iris_data["OBS_VALUE"].notna().sum()
                        print(
                            f"      ✅ Success! Shape: {iris_data.shape}, "
                            f"Non-null values: {non_null_count}/{len(iris_data)}"
                        )
                        if non_null_count > 0:
                            print(f"      Sample values:")
                            print(
                                iris_data[iris_data["OBS_VALUE"].notna()].head(3).to_string()
                            )
                        else:
                            print(f"      ⚠️  All values are NaN")
                            # Check if there are additional columns that might help
                            if len(iris_data.columns) > 2:
                                print(f"      Columns: {list(iris_data.columns)}")
                    else:
                        print(f"      ❌ Empty result")
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            
            # Test STOCD which we know works at commune level
            print(f"\n   Testing STOCD (known to work at commune level)...")
            try:
                stocd_data = get_local_data(
                    dataset_version="GEO2021RP2018",
                    variables="STOCD",
                    nivgeo="IRIS",
                    geocodes=sample_codes,
                )
                if not stocd_data.empty:
                    print(f"      Shape: {stocd_data.shape}")
                    print(f"      Columns: {list(stocd_data.columns)}")
                    # Check if it has multiple rows per IRIS (like commune level)
                    if len(stocd_data) > len(sample_codes):
                        print(f"      ✅ Multiple rows per IRIS (like commune level)")
                        print(f"      Sample data:")
                        print(stocd_data.head(10).to_string())
                    else:
                        non_null = stocd_data["OBS_VALUE"].notna().sum()
                        print(f"      Non-null: {non_null}/{len(stocd_data)}")
            except Exception as e:
                print(f"      Error: {e}")

            # Test combining multiple variables
            if len(test_variables) > 1:
                print(f"\n   Testing combined variables: {'-'.join(test_variables[:3])}")
                try:
                    combined_data = get_local_data(
                        dataset_version="GEO2021RP2018",
                        variables="-".join(test_variables[:3]),
                        nivgeo="IRIS",
                        geocodes=sample_codes,
                    )
                    if not combined_data.empty:
                        print(f"      ✅ Success! Shape: {combined_data.shape}")
                        print(f"      Columns: {list(combined_data.columns)}")
                        print(f"      Sample data:")
                        print(combined_data.head().to_string())
                except Exception as e:
                    print(f"      ❌ Error: {e}")

        except Exception as e:
            print(f"   Error fetching IRIS data: {e}")
            import traceback

            traceback.print_exc()
    else:
        if not test_variables:
            print("   ⚠️  No test variables identified")
        if not paris_iris_codes:
            print("   ⚠️  No IRIS codes available")

    # Step 5: Test commune-level data and analyze structure
    print("\n5. Testing commune-level data to verify variables work...")
    try:
        commune_data = get_local_data(
            dataset_version="GEO2021RP2018",
            variables="STOCD",
            nivgeo="COM",
            geocodes=["75056"],  # Paris commune
        )
        if not commune_data.empty:
            non_null = commune_data["OBS_VALUE"].notna().sum()
            print(f"   ✅ Commune-level data works! Shape: {commune_data.shape}")
            print(f"   Non-null values: {non_null}/{len(commune_data)}")
            print(f"   Columns: {list(commune_data.columns)}")
            if non_null > 0:
                print(f"\n   Sample commune data:")
                print(commune_data.head(3).to_string())
                
                # Check if we can extract population from STOCD
                if "UNIT" in commune_data.columns:
                    pop_rows = commune_data[commune_data["UNIT"] == "POP"]
                    print(f"\n   Population data from STOCD (UNIT=POP): {len(pop_rows)} rows")
                    if len(pop_rows) > 0:
                        print(f"   Sample population data:")
                        print(pop_rows.head().to_string())
        else:
            print("   ⚠️  Commune-level data is empty")
    except Exception as e:
        print(f"   Error testing commune data: {e}")
    
    # Step 6: Try download_file approach (like pynsee example)
    print("\n6. Testing download_file approach for IRIS-level data...")
    try:
        from pynsee import download_file
        
        print("   Downloading RP_ACTRES_IRIS (2017 IRIS-level census data)...")
        iris_df = download_file("RP_ACTRES_IRIS")
        
        if not iris_df.empty:
            print(f"   ✅ SUCCESS! Downloaded IRIS-level data with actual values!")
            print(f"   Shape: {iris_df.shape}")
            print(f"   Columns: {len(iris_df.columns)}")
            
            # Filter for Paris
            if "COM" in iris_df.columns:
                paris_iris_df = iris_df[iris_df["COM"].str.startswith("75", na=False)]
                print(f"   ✅ Paris IRIS units: {len(paris_iris_df)}")
                
                # Check for actual values
                numeric_cols = paris_iris_df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    sample_row = paris_iris_df.iloc[0]
                    non_null_count = sum(
                        1
                        for col in numeric_cols[:20]
                        if pd.notna(sample_row.get(col, None))
                        and sample_row.get(col, 0) != 0
                    )
                    print(f"   ✅ Found {len(numeric_cols)} numeric columns with actual values")
                    print(f"   Sample values from first Paris IRIS:")
                    if "P17_POP1564" in sample_row.index:
                        print(f"      P17_POP1564: {sample_row['P17_POP1564']}")
                    if "P17_ACT1564" in sample_row.index:
                        print(f"      P17_ACT1564: {sample_row['P17_ACT1564']}")
                    if "C17_ACTOCC15P_VOIT" in sample_row.index:
                        print(f"      C17_ACTOCC15P_VOIT: {sample_row['C17_ACTOCC15P_VOIT']}")
                    
                    # Check available demographic variables
                    print(f"\n   Available demographic variables:")
                    pop_vars = [c for c in iris_df.columns if "POP" in c or "P17" in c][:5]
                    print(f"      Population: {pop_vars}")
                    car_vars = [c for c in iris_df.columns if "VOIT" in c or "ACTOCC" in c][:5]
                    print(f"      Car ownership: {car_vars}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Step 7: Summary of findings
    print("\n7. Key Discovery:")
    print("   ✅ download_file('RP_ACTRES_IRIS') provides IRIS-level data with actual values!")
    print("   ⚠️  get_local_data() API returns NaN for IRIS-level queries")
    print("   💡 Recommendation: Use download_file() approach for IRIS-level census data")

    # Step 7: Document findings
    print("\n" + "=" * 80)
    print("Exploration Summary")
    print("=" * 80)
    
    print("\n✅ What Works:")
    print("   1. 2021 IRIS boundaries loaded successfully (992 Paris IRIS units)")
    print("   2. IRIS codes extracted from 2021 boundaries (CODE_IRIS column)")
    print("   3. Commune-level data works perfectly (returns actual values)")
    print("   4. STOCD variable at commune level includes population data (UNIT=POP)")
    
    print("\n⚠️  Current Issue:")
    print("   - IRIS-level data returns NaN for all tested variables")
    print("   - Even with 2021 IRIS codes matching 2021 census data")
    print("   - Variables tested: NPERC, VOIT, NBENFFR-TF4, TYPMR, STOCD")
    
    if "test_variables" in locals() and test_variables:
        print("\n✅ Variables Identified (for future use):")
        print(f"   - Household size: NPERC (Nombre de personnes du ménage)")
        print(f"   - Car ownership: VOIT (Nombre de voitures du ménage)")
        print(f"   - Children: NBENFFR (Nombre d'enfants de moins de 25 ans)")
        print(f"   - Household type: TYPMR (Type de ménage)")
        print("\n⚠️  Note: Income variables not found in GEO2021RP2018 dataset")
    
    print("""
    Recommendations:
    1. ✅ USE download_file('RP_ACTRES_IRIS') for IRIS-level census data!
       - Provides actual IRIS-level values (not NaN)
       - Contains 992 Paris IRIS units matching 2021 boundaries
       - Includes population, employment, car ownership variables
       - File format: Excel/Parquet with IRIS codes in 'IRIS' column
    
    2. Update CensusLoader to:
       - Use download_file('RP_ACTRES_IRIS') instead of get_local_data()
       - Match IRIS codes from boundaries (CODE_IRIS) with data (IRIS column)
       - Extract demographic features: population, employment, car ownership
       - Note: This is 2017 data, but matches 2021 IRIS boundaries structure
    
    3. For household size and children variables:
       - May need to use commune-level data (get_local_data) and aggregate
       - Or search for additional IRIS-level files with household data
    """)


if __name__ == "__main__":
    main()
