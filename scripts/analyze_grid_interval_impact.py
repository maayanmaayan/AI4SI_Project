#!/usr/bin/env python3
"""Analyze how reducing grid interval would affect data size.

This script checks the current data and estimates how many points would
be generated with different grid intervals (50m, 25m instead of 100m).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

from src.data.collection.feature_engineer import FeatureEngineer
from src.training.dataset import load_features_from_directory


def calculate_points_for_interval(
    polygon_geom,
    sampling_interval_meters: float,
) -> int:
    """Calculate how many points would be generated for a polygon with given interval.
    
    Args:
        polygon_geom: Shapely polygon geometry
        sampling_interval_meters: Grid spacing in meters
        
    Returns:
        Estimated number of points
    """
    # Convert to metric CRS for calculations
    temp_gdf = gpd.GeoDataFrame([1], geometry=[polygon_geom], crs="EPSG:4326")
    temp_gdf_metric = temp_gdf.to_crs("EPSG:3857")
    polygon_metric = temp_gdf_metric.geometry.iloc[0]
    
    # Get bounding box
    minx, miny, maxx, maxy = polygon_metric.bounds
    
    # Generate grid points
    x_coords = np.arange(
        minx, maxx + sampling_interval_meters, sampling_interval_meters
    )
    y_coords = np.arange(
        miny, maxy + sampling_interval_meters, sampling_interval_meters
    )
    
    # Count points inside polygon
    count = 0
    for x in x_coords:
        for y in y_coords:
            point_metric = Point(x, y)
            if polygon_metric.contains(point_metric):
                count += 1
    
    return count


def main():
    """Analyze grid interval impact."""
    print("=" * 80)
    print("Grid Interval Impact Analysis")
    print("=" * 80)
    
    # Load current data
    features_dir = "data/processed/features/compliant"
    print(f"\n1. Loading current data from: {features_dir}")
    try:
        df = load_features_from_directory(features_dir)
        print(f"   ✓ Loaded {len(df)} total points")
        print(f"   ✓ From {df['neighborhood_name'].nunique()} neighborhoods")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Count points per neighborhood
    print("\n2. Current points per neighborhood:")
    counts = df['neighborhood_name'].value_counts().sort_values(ascending=False)
    for neighborhood, count in counts.items():
        print(f"   {neighborhood}: {count} points")
    
    # Load neighborhood geometries
    print("\n3. Loading neighborhood boundaries...")
    neighborhoods_gdf = gpd.read_file("paris_neighborhoods.geojson")
    neighborhoods_dict = {}
    for idx, row in neighborhoods_gdf.iterrows():
        name = row.get('name', 'unknown')
        if name:
            neighborhoods_dict[name] = row
    
    # Calculate estimates for different intervals
    print("\n4. Estimating points for different grid intervals:")
    print("   " + "-" * 76)
    print(f"   {'Neighborhood':<40} {'100m':<8} {'50m':<10} {'25m':<10} {'Factor 50m':<12} {'Factor 25m'}")
    print("   " + "-" * 76)
    
    total_current = 0
    total_50m = 0
    total_25m = 0
    
    # Check each neighborhood
    for neighborhood_name in counts.index:
        # Find matching geometry
        geometry = None
        for name, row in neighborhoods_dict.items():
            if name.lower().replace(' ', '_') == neighborhood_name.lower().replace(' ', '_'):
                geometry = row.geometry
                break
        
        if geometry is None:
            # Try partial match
            for name, row in neighborhoods_dict.items():
                if neighborhood_name.lower() in name.lower() or name.lower() in neighborhood_name.lower():
                    geometry = row.geometry
                    break
        
        if geometry is None:
            print(f"   {neighborhood_name:<40} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'(no geom)':<12} {'(no geom)'}")
            continue
        
        current_count = counts[neighborhood_name]
        total_current += current_count
        
        # Estimate for 50m (4x denser)
        count_50m = calculate_points_for_interval(geometry, 50.0)
        total_50m += count_50m
        
        # Estimate for 25m (16x denser)
        count_25m = calculate_points_for_interval(geometry, 25.0)
        total_25m += count_25m
        
        factor_50m = count_50m / current_count if current_count > 0 else 0
        factor_25m = count_25m / current_count if current_count > 0 else 0
        
        print(f"   {neighborhood_name:<40} {current_count:<8} {count_50m:<10} {count_25m:<10} {factor_50m:.1f}x{'':<8} {factor_25m:.1f}x")
    
    print("   " + "-" * 76)
    factor_50m_total = total_50m / total_current if total_current > 0 else 0
    factor_25m_total = total_25m / total_current if total_current > 0 else 0
    print(f"   {'TOTAL':<40} {total_current:<8} {total_50m:<10} {total_25m:<10} {factor_50m_total:.1f}x{'':<8} {factor_25m_total:.1f}x")
    
    print("\n5. Summary:")
    print(f"   Current (100m spacing): {total_current} points")
    print(f"   50m spacing: ~{total_50m} points ({factor_50m_total:.1f}x increase)")
    print(f"   25m spacing: ~{total_25m} points ({factor_25m_total:.1f}x increase)")
    print(f"\n   Note: IRIS-level features are interpolated, so denser grids use the same")
    print(f"   underlying IRIS data but create more target points for training.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
