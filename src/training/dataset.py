"""PyTorch Geometric Dataset class for loading pre-processed feature data.

This module provides the SpatialGraphDataset class that converts pre-processed
Parquet data into PyTorch Geometric Data objects for training. It includes
utilities for loading features from directories and creating neighborhood-level
stratified splits.
"""

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SpatialGraphDataset(Dataset):
    """PyTorch Geometric Dataset for spatial graph data.

    This dataset class loads pre-processed feature data from a DataFrame and
    converts each row into a PyTorch Geometric Data object representing a
    star graph (target point + neighbors).

    The graph structure:
    - Node 0: Target point (33 features)
    - Nodes 1-N: Neighbors (33 features each)
    - Edges: All neighbors connect to target (edge_index: [[1,2,...], [0,0,...]])
    - Edge attributes: [dx, dy, euclidean_distance, network_distance] for each edge

    Args:
        features_df: DataFrame containing pre-processed features with columns:
            - target_id: Unique identifier for target point
            - neighborhood_name: Name of neighborhood
            - target_features: Array-like [33] of target point features
            - neighbor_data: List of dicts, each with 'features' [33], 'dx', 'dy',
              'euclidean_distance', 'network_distance'
            - target_prob_vector: Array-like [8] of target probability distribution
            - num_neighbors: Number of neighbors (should match len(neighbor_data))
        transform: Optional transform to apply to each Data object.
        pre_transform: Optional pre-transform to apply before storing.

    Example:
        >>> df = pd.read_parquet("data/processed/features/compliant/paris_rive_gauche/target_points.parquet")
        >>> dataset = SpatialGraphDataset(df)
        >>> data = dataset[0]  # Get first graph
        >>> print(f"Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
    ):
        """Initialize dataset from DataFrame.

        Args:
            features_df: DataFrame with pre-processed features.
            transform: Optional transform to apply to each Data object.
            pre_transform: Optional pre-transform (not used, kept for compatibility).
        """
        self.features_df = features_df.reset_index(drop=True)

        # Initialize parent without root (we're not using file-based caching)
        # Pass None as root to avoid file system operations
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

        # Validate required columns exist
        required_columns = [
            "target_id",
            "neighborhood_name",
            "target_features",
            "neighbor_data",
            "target_prob_vector",
            "num_neighbors",
        ]
        missing = set(required_columns) - set(self.features_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def len(self) -> int:
        """Return number of graphs in dataset.

        Returns:
            Number of rows in features DataFrame.
        """
        return len(self.features_df)

    def get(self, idx: int) -> Data:
        """Build PyTorch Geometric Data object from DataFrame row.

        Args:
            idx: Index of row in features DataFrame.

        Returns:
            PyTorch Geometric Data object with:
            - x: Node features [1 + num_neighbors, 33]
            - edge_index: Edge connections [2, num_neighbors]
            - edge_attr: Edge attributes [num_neighbors, 4]
            - y: Target probability vector [8]
            - target_id: Target point identifier
            - neighborhood_name: Neighborhood name
        """
        row = self.features_df.iloc[idx]

        # Extract features
        target_features = torch.tensor(np.asarray(row["target_features"]), dtype=torch.float32)  # [33]
        neighbor_data = row["neighbor_data"]  # List of dicts
        target_prob_vector = torch.tensor(
            np.asarray(row["target_prob_vector"]), dtype=torch.float32
        )  # [8]

        # Build node features: [target, neighbor1, neighbor2, ...]
        num_neighbors = len(neighbor_data)
        node_features = [target_features]

        for neighbor in neighbor_data:
            neighbor_feat = torch.tensor(np.asarray(neighbor["features"]), dtype=torch.float32)  # [33]
            node_features.append(neighbor_feat)

        x = torch.stack(node_features)  # [1 + num_neighbors, 33]

        # Build edge index: star graph (all neighbors → target)
        if num_neighbors > 0:
            edge_index = torch.tensor(
                [
                    list(range(1, num_neighbors + 1)),  # Source: neighbors (1 to N)
                    [0] * num_neighbors,  # Target: center (always 0)
                ],
                dtype=torch.long,
            )  # [2, num_neighbors]

            # Build edge attributes: [dx, dy, euclidean_dist, network_dist]
            edge_attr = torch.tensor(
                [
                    [
                        neighbor["dx"],
                        neighbor["dy"],
                        neighbor["euclidean_distance"],
                        neighbor["network_distance"],
                    ]
                    for neighbor in neighbor_data
                ],
                dtype=torch.float32,
            )  # [num_neighbors, 4]
        else:
            # No neighbors: empty edge index and attributes
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float32)

        # Create PyTorch Geometric Data object
        # Note: y should be 2D [1, 8] for proper batching (PyG stacks 2D tensors, concatenates 1D)
        data = Data(
            x=x,  # [1 + num_neighbors, 33]
            edge_index=edge_index,  # [2, num_neighbors]
            edge_attr=edge_attr,  # [num_neighbors, 4]
            y=target_prob_vector.unsqueeze(0),  # [1, 8] - target probability vector (2D for proper batching)
            target_id=row["target_id"],
            neighborhood_name=row["neighborhood_name"],
            num_nodes=1 + num_neighbors,
        )

        return data


def load_features_from_directory(directory: str) -> pd.DataFrame:
    """Load all target_points.parquet files from a directory tree.

    Recursively searches for all target_points.parquet files in the specified
    directory and concatenates them into a single DataFrame.

    Args:
        directory: Path to directory containing target_points.parquet files.

    Returns:
        DataFrame containing all loaded features with required columns.

    Raises:
        FileNotFoundError: If directory does not exist.
        ValueError: If no target_points.parquet files are found.

    Example:
        >>> df = load_features_from_directory("data/processed/features/compliant")
        >>> print(f"Loaded {len(df)} rows from {df['neighborhood_name'].nunique()} neighborhoods")
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Recursively find all target_points.parquet files
    parquet_files = list(directory_path.glob("**/target_points.parquet"))

    if not parquet_files:
        raise ValueError(f"No target_points.parquet files found in {directory}")

    logger.info(f"Found {len(parquet_files)} target_points.parquet files in {directory}")

    # Load and concatenate all files
    dataframes = []
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            dataframes.append(df)
            logger.debug(f"Loaded {len(df)} rows from {parquet_file}")
        except Exception as e:
            logger.error(f"Failed to load {parquet_file}: {e}")
            raise

    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(
        f"Combined {len(combined_df)} total rows from {len(parquet_files)} files, "
        f"{combined_df['neighborhood_name'].nunique()} unique neighborhoods"
    )

    return combined_df


def create_train_val_test_splits(
    features_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits stratified by neighborhood.

    This function splits the data at the neighborhood level to avoid spatial
    leakage (all points from a neighborhood go to the same split).

    Args:
        features_df: DataFrame containing features with 'neighborhood_name' column.
        train_ratio: Proportion of neighborhoods for training. Defaults to 0.7.
        val_ratio: Proportion of neighborhoods for validation. Defaults to 0.15.
        test_ratio: Proportion of neighborhoods for testing. Defaults to 0.15.
        random_seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.

    Raises:
        ValueError: If ratios don't sum to 1.0 or if required column is missing.

    Example:
        >>> df = load_features_from_directory("data/processed/features/compliant")
        >>> train, val, test = create_train_val_test_splits(df)
        >>> print(f"Train: {len(train)} rows, Val: {len(val)} rows, Test: {len(test)} rows")
    """
    if "neighborhood_name" not in features_df.columns:
        raise ValueError("DataFrame must have 'neighborhood_name' column for neighborhood-level splits")

    # Validate ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Get unique neighborhoods
    unique_neighborhoods = features_df["neighborhood_name"].unique().tolist()

    if len(unique_neighborhoods) < 3:
        raise ValueError(
            f"Need at least 3 neighborhoods for train/val/test splits, got {len(unique_neighborhoods)}"
        )

    # Shuffle neighborhoods for reproducible splits
    random.seed(random_seed)
    random.shuffle(unique_neighborhoods)

    # Calculate split indices
    total_neighborhoods = len(unique_neighborhoods)
    n_train = int(total_neighborhoods * train_ratio)
    n_val = int(total_neighborhoods * val_ratio)
    
    # Ensure at least 1 neighborhood per split when possible
    # This handles edge cases where rounding causes empty splits
    if n_train == 0 and total_neighborhoods >= 1:
        n_train = 1
    if n_val == 0 and total_neighborhoods >= 2:
        n_val = 1
    if n_train + n_val >= total_neighborhoods:
        # Not enough neighborhoods for all splits - adjust
        n_train = max(1, total_neighborhoods - 2)  # Leave room for val and test
        n_val = 1
    # Test gets remainder to handle rounding

    # Split neighborhoods
    train_neighborhoods = set(unique_neighborhoods[:n_train])
    val_neighborhoods = set(unique_neighborhoods[n_train : n_train + n_val])
    test_neighborhoods = set(unique_neighborhoods[n_train + n_val :])
    
    # Final check: ensure test is not empty if we have enough neighborhoods
    if len(test_neighborhoods) == 0 and total_neighborhoods >= 3:
        # Redistribute: take one from train if possible
        if len(train_neighborhoods) > 1:
            moved = train_neighborhoods.pop()
            test_neighborhoods.add(moved)
        elif len(val_neighborhoods) > 1:
            moved = val_neighborhoods.pop()
            test_neighborhoods.add(moved)

    # Filter DataFrame by neighborhood sets
    train_df = features_df[features_df["neighborhood_name"].isin(train_neighborhoods)].copy()
    val_df = features_df[features_df["neighborhood_name"].isin(val_neighborhoods)].copy()
    test_df = features_df[features_df["neighborhood_name"].isin(test_neighborhoods)].copy()

    logger.info(
        f"Created splits: Train={len(train_df)} rows ({len(train_neighborhoods)} neighborhoods), "
        f"Val={len(val_df)} rows ({len(val_neighborhoods)} neighborhoods), "
        f"Test={len(test_df)} rows ({len(test_neighborhoods)} neighborhoods)"
    )

    return train_df, val_df, test_df


def get_device() -> torch.device:
    """Select appropriate device (MPS, CUDA, or CPU).

    Checks for device availability in order: MPS (Apple Silicon) > CUDA > CPU.

    Returns:
        torch.device instance for the best available device.

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
        Using device: mps
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
