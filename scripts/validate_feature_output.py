#!/usr/bin/env python3
"""Validate feature engineering output format for model compatibility.

Checks that target_points.parquet has all required columns, correct shapes,
and that neighbor_data has the keys needed to build PyTorch Geometric Data objects.

Usage:
    python scripts/validate_feature_output.py [path_to_target_points.parquet]
    # If no path given, checks data/processed/features/compliant/*/target_points.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ROW = [
    "target_id",
    "neighborhood_name",
    "label",
    "target_features",
    "neighbor_data",
    "target_prob_vector",
    "target_geometry",
    "num_neighbors",
]
REQUIRED_NEIGHBOR = ["features", "network_distance", "euclidean_distance", "dx", "dy"]


def validate_file(path: Path) -> bool:
    """Validate a single target_points.parquet file. Returns True if valid."""
    df = pd.read_parquet(path)

    # 1) Columns
    missing = set(REQUIRED_ROW) - set(df.columns)
    if missing:
        print(f"  FAIL: Missing columns: {missing}")
        return False

    # 2) Sample row
    row = df.iloc[0]

    # 3) target_features
    tf = np.asarray(row["target_features"])
    if tf.shape != (33,):
        print(f"  FAIL: target_features shape {tf.shape}, expected (33,)")
        return False

    # 4) target_prob_vector
    tpv = np.asarray(row["target_prob_vector"])
    if tpv.shape != (8,):
        print(f"  FAIL: target_prob_vector shape {tpv.shape}, expected (8,)")
        return False
    if not np.isclose(tpv.sum(), 1.0):
        print(f"  FAIL: target_prob_vector sum {tpv.sum():.6f}, expected 1.0")
        return False

    # 5) neighbor_data and num_neighbors
    nd = row["neighbor_data"]
    if row["num_neighbors"] != len(nd):
        print(f"  FAIL: num_neighbors={row['num_neighbors']} != len(neighbor_data)={len(nd)}")
        return False

    if len(nd) > 0:
        n0 = nd[0]
        missing_n = set(REQUIRED_NEIGHBOR) - set(n0.keys())
        if missing_n:
            print(f"  FAIL: neighbor dict missing keys: {missing_n}")
            return False
        if np.asarray(n0["features"]).shape != (33,):
            print(f"  FAIL: neighbor['features'] shape {np.asarray(n0['features']).shape}, expected (33,)")
            return False
        # Edge attr: [dx, dy, euclidean_distance, network_distance]
        _ = [n0["dx"], n0["dy"], n0["euclidean_distance"], n0["network_distance"]]

    # 6) target_geometry
    g = row["target_geometry"]
    if not (isinstance(g, str) and g.strip().upper().startswith("POINT")) and not (
        hasattr(g, "geom_type") and g.geom_type == "Point"
    ):
        print(f"  FAIL: target_geometry should be WKT string or Shapely Point, got {type(g)}")
        return False

    # 7) NaN/Inf spot check
    if not np.isfinite(tf).all():
        print("  FAIL: target_features contains NaN/Inf")
        return False
    if not np.isfinite(tpv).all():
        print("  FAIL: target_prob_vector contains NaN/Inf")
        return False

    # 8) num_neighbors consistency for all rows
    for _, r in df.iterrows():
        if r["num_neighbors"] != len(r["neighbor_data"]):
            print("  FAIL: num_neighbors != len(neighbor_data) for at least one row")
            return False

    print(f"  OK: {len(df)} rows, num_neighbors in [{df['num_neighbors'].min()}, {df['num_neighbors'].max()}]")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate feature engineering output for model compatibility.")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to target_points.parquet or directory. Default: data/processed/features/compliant",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.path is None:
        search = root / "data" / "processed" / "features" / "compliant"
        paths = list(search.glob("*/target_points.parquet"))
    else:
        p = Path(args.path)
        if p.is_file():
            paths = [p]
        else:
            paths = list(p.glob("**/target_points.parquet"))

    if not paths:
        print("No target_points.parquet files found.")
        sys.exit(1)

    print("Validating feature output format (model-ready check):")
    all_ok = True
    for path in sorted(paths):
        rel = path.relative_to(root) if root in path.resolve().parents else path
        print(f"  {rel}:")
        if not validate_file(path):
            all_ok = False

    if all_ok:
        print("\nAll files passed. Format matches model/dataloader expectations.")
    else:
        print("\nSome files failed validation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
