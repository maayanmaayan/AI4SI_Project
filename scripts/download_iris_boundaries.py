"""Download IRIS boundaries GeoJSON file.

This script helps download IRIS boundaries from data.gouv.fr.
The IRIS boundaries file contains:
- IRIS codes (DCOMIRIS) that match census data
- Polygon geometries for spatial joins
- Administrative information (commune, department, etc.)

Usage:
    python scripts/download_iris_boundaries.py [url]
    
If no URL provided, will show instructions for manual download.
"""

import sys
from pathlib import Path

import geopandas as gpd
import requests
from tqdm import tqdm


def download_iris_boundaries(
    url: str = None,
    output_path: str = "data/raw/iris_boundaries.geojson",
    filter_paris: bool = True,
) -> None:
    """Download IRIS boundaries GeoJSON file.

    Args:
        url: Direct download URL for IRIS boundaries GeoJSON.
            If None, shows manual download instructions.
        output_path: Path to save the GeoJSON file.
        filter_paris: If True, filter to only Paris IRIS (DEPCOM starting with '75').
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("IRIS Boundaries Download")
    print("=" * 80)

    if url:
        print(f"\n1. Downloading from: {url}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            print(f"   File size: {total_size / (1024*1024):.2f} MB")

            temp_file = output_file.with_suffix(".tmp.geojson")

            with open(temp_file, "wb") as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    f.write(response.content)

            print(f"   ✅ Downloaded to temporary file")

            # Load and process
            print(f"\n2. Loading GeoJSON...")
            gdf = gpd.read_file(temp_file)
            print(f"   Loaded {len(gdf)} IRIS units")
            print(f"   Columns: {list(gdf.columns)}")

            # Filter for Paris if requested
            if filter_paris:
                print(f"\n3. Filtering for Paris...")
                if "DEPCOM" in gdf.columns:
                    paris_gdf = gdf[gdf["DEPCOM"].str.startswith("75", na=False)]
                elif "DCOMIRIS" in gdf.columns:
                    paris_gdf = gdf[gdf["DCOMIRIS"].str.startswith("75", na=False)]
                else:
                    print("   ⚠️  Could not find DEPCOM or DCOMIRIS column, saving all")
                    paris_gdf = gdf

                print(f"   Found {len(paris_gdf)} IRIS units in Paris")
                paris_gdf.to_file(output_file, driver="GeoJSON")
                print(f"   ✅ Saved to: {output_file}")
            else:
                gdf.to_file(output_file, driver="GeoJSON")
                print(f"   ✅ Saved to: {output_file}")

            temp_file.unlink()

        except Exception as e:
            print(f"   ❌ Error: {e}")
            raise
    else:
        print("\n📥 Manual Download Instructions:")
        print("\n1. Go to: https://www.data.gouv.fr/fr/datasets/contour-des-iris-insee-tout-en-un/")
        print("2. Find the GeoJSON download link")
        print("3. Download the file")
        print(f"4. Place it at: {output_file.absolute()}")
        print("\n   Or provide the direct download URL as an argument:")
        print("   python scripts/download_iris_boundaries.py <url>")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else None
    download_iris_boundaries(url=url)
