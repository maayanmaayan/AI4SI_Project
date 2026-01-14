"""Exploration script to understand pynsee API and available datasets.

This script explores the pynsee library to understand:
- Available datasets for IRIS-level census data
- How to fetch data for Paris (geocode: 75056)
- Available variables for demographic features
- Dataset structure and codes

Run this script before implementing CensusLoader to understand the API.
"""

import pandas as pd
from pynsee.localdata import get_local_data, get_local_metadata


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
        print(metadata.head().to_string())
        
        # Search for IRIS-related datasets
        if "title" in metadata.columns:
            iris_datasets = metadata[
                metadata["title"].str.contains("IRIS", case=False, na=False)
            ]
            print(f"\n   Found {len(iris_datasets)} IRIS-related datasets:")
            print(iris_datasets[["title", "id"]].to_string())
    except Exception as e:
        print(f"   Error fetching metadata: {e}")

    # Step 2: Try fetching IRIS-level data for Paris
    print("\n2. Attempting to fetch IRIS-level data for Paris (geocode: 75056)...")
    try:
        # Common parameters for IRIS-level data
        # Note: Actual parameters may vary - this is exploratory
        iris_data = get_local_data(
            nivgeo="IRIS",
            geocodes=["75056"],  # Paris INSEE code
        )
        print(f"   Successfully fetched data with shape: {iris_data.shape}")
        print(f"   Columns: {list(iris_data.columns)}")
        print("\n   First few rows:")
        print(iris_data.head().to_string())
        
        # Check for key demographic variables
        print("\n   Looking for demographic variables...")
        demographic_keywords = [
            "population", "menage", "household", "revenu", "income",
            "age", "enfant", "child", "voiture", "car", "densite", "density"
        ]
        for keyword in demographic_keywords:
            matching_cols = [col for col in iris_data.columns if keyword.lower() in col.lower()]
            if matching_cols:
                print(f"     Found columns with '{keyword}': {matching_cols}")
    except Exception as e:
        print(f"   Error fetching IRIS data: {e}")
        print("   Note: This is expected if dataset codes need to be discovered first")

    # Step 3: Document findings
    print("\n" + "=" * 80)
    print("Exploration Summary")
    print("=" * 80)
    print("""
    Next steps for CensusLoader implementation:
    1. Identify the correct dataset ID for IRIS-level census data
    2. Identify variable names for required features:
       - Population density
       - SES index (from income data)
       - Car ownership
       - Children per capita
       - Household size
       - Elderly ratio
    3. Understand how to match IRIS codes to geometries
    4. Test data fetching with actual parameters
    """)


if __name__ == "__main__":
    main()
