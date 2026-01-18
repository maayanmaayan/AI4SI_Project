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

# Try to import torch for tensor conversion validation
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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

    # 9) Dataset-specific: Tensor conversion validation
    if TORCH_AVAILABLE:
        try:
            # Test converting target_features to tensor
            tf_tensor = torch.tensor(tf, dtype=torch.float32)
            if tf_tensor.shape != torch.Size([33]):
                print(f"  FAIL: target_features tensor shape {tf_tensor.shape}, expected torch.Size([33])")
                return False

            # Test converting target_prob_vector to tensor
            tpv_tensor = torch.tensor(tpv, dtype=torch.float32)
            if tpv_tensor.shape != torch.Size([8]):
                print(f"  FAIL: target_prob_vector tensor shape {tpv_tensor.shape}, expected torch.Size([8])")
                return False

            # Test converting neighbor data to tensors (sample first neighbor if available)
            if len(nd) > 0:
                n0_feat_tensor = torch.tensor(np.asarray(nd[0]["features"]), dtype=torch.float32)
                if n0_feat_tensor.shape != torch.Size([33]):
                    print(f"  FAIL: neighbor features tensor shape {n0_feat_tensor.shape}, expected torch.Size([33])")
                    return False

                # Test edge attributes conversion
                edge_attr_sample = torch.tensor(
                    [[nd[0]["dx"], nd[0]["dy"], nd[0]["euclidean_distance"], nd[0]["network_distance"]]],
                    dtype=torch.float32,
                )
                if edge_attr_sample.shape != torch.Size([1, 4]):
                    print(f"  FAIL: edge_attr tensor shape {edge_attr_sample.shape}, expected torch.Size([1, 4])")
                    return False

        except (ValueError, TypeError, RuntimeError) as e:
            print(f"  FAIL: Tensor conversion error: {e}")
            return False

    print(f"  OK: {len(df)} rows, num_neighbors in [{df['num_neighbors'].min()}, {df['num_neighbors'].max()}]")
    return True


def validate_dataset_format(path: Path) -> dict:
    """Validate dataset format and return summary statistics.

    Args:
        path: Path to target_points.parquet file.

    Returns:
        Dictionary with summary statistics including total graphs, neighbor distribution,
        and memory estimates.
    """
    df = pd.read_parquet(path)

    stats = {
        "total_graphs": len(df),
        "num_neighborhoods": df["neighborhood_name"].nunique() if "neighborhood_name" in df.columns else 0,
        "min_neighbors": int(df["num_neighbors"].min()),
        "max_neighbors": int(df["num_neighbors"].max()),
        "mean_neighbors": float(df["num_neighbors"].mean()),
        "median_neighbors": float(df["num_neighbors"].median()),
        "std_neighbors": float(df["num_neighbors"].std()),
        "zero_neighbor_count": int((df["num_neighbors"] == 0).sum()),
    }

    # Estimate memory per graph (rough calculation)
    # Node features: (1 + num_neighbors) * 33 * 4 bytes (float32)
    # Edge index: 2 * num_neighbors * 4 bytes (long)
    # Edge attributes: num_neighbors * 4 * 4 bytes (float32)
    # Target vector: 8 * 4 bytes (float32)
    # Rough estimate: use mean neighbors
    mean_n = stats["mean_neighbors"]
    nodes_mem = (1 + mean_n) * 33 * 4
    edge_idx_mem = 2 * mean_n * 4
    edge_attr_mem = mean_n * 4 * 4
    target_mem = 8 * 4
    stats["estimated_memory_per_graph_bytes"] = int(nodes_mem + edge_idx_mem + edge_attr_mem + target_mem)
    stats["estimated_memory_per_graph_mb"] = stats["estimated_memory_per_graph_bytes"] / (1024 * 1024)

    return stats


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
        # Convert relative paths to absolute paths relative to root
        if not p.is_absolute():
            p = root / p
        if p.is_file():
            paths = [p]
        else:
            paths = list(p.glob("**/target_points.parquet"))

    if not paths:
        print("No target_points.parquet files found.")
        sys.exit(1)

    print("Validating feature output format (model-ready check):")
    all_ok = True
    all_stats = []
    for path in sorted(paths):
        try:
            rel = path.relative_to(root)
        except ValueError:
            # Path is not relative to root, use path as-is
            rel = path
        print(f"  {rel}:")
        if not validate_file(path):
            all_ok = False
        else:
            # Get summary statistics for valid files
            stats = validate_dataset_format(path)
            all_stats.append((rel, stats))

    if all_ok:
        print("\nAll files passed. Format matches model/dataloader expectations.")
        if all_stats:
            print("\nDataset Summary Statistics:")
            print("=" * 80)
            total_graphs = sum(s[1]["total_graphs"] for s in all_stats)
            total_neighborhoods = sum(s[1]["num_neighborhoods"] for s in all_stats)
            avg_memory_mb = sum(s[1]["estimated_memory_per_graph_mb"] for s in all_stats) / len(all_stats)

            print(f"Total graphs: {total_graphs}")
            print(f"Total neighborhoods: {total_neighborhoods}")
            print(f"Average memory per graph: {avg_memory_mb:.2f} MB")
            print(f"Estimated total memory (avg): {total_graphs * avg_memory_mb:.2f} MB")

            # Aggregate neighbor statistics
            all_min_neighbors = [s[1]["min_neighbors"] for s in all_stats]
            all_max_neighbors = [s[1]["max_neighbors"] for s in all_stats]
            all_mean_neighbors = [s[1]["mean_neighbors"] for s in all_stats]
            all_zero_neighbors = [s[1]["zero_neighbor_count"] for s in all_stats]

            print(f"\nNeighbor distribution (across all files):")
            print(f"  Min neighbors: {min(all_min_neighbors)}")
            print(f"  Max neighbors: {max(all_max_neighbors)}")
            print(f"  Mean neighbors: {sum(all_mean_neighbors) / len(all_mean_neighbors):.2f}")
            print(f"  Zero-neighbor graphs: {sum(all_zero_neighbors)}")

            print("\nPer-file details:")
            for rel, stats in all_stats:
                print(f"  {rel}:")
                print(f"    Graphs: {stats['total_graphs']}, Neighborhoods: {stats['num_neighborhoods']}")
                print(
                    f"    Neighbors: min={stats['min_neighbors']}, max={stats['max_neighbors']}, "
                    f"mean={stats['mean_neighbors']:.1f}"
                )
                print(f"    Memory/graph: {stats['estimated_memory_per_graph_mb']:.2f} MB")
    else:
        print("\nSome files failed validation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
