"""Unit tests for SpatialGraphDataset module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src.training.dataset import (
    SpatialGraphDataset,
    create_train_val_test_splits,
    get_device,
    load_features_from_directory,
)


@pytest.fixture
def sample_features_df():
    """Create a sample DataFrame matching expected format."""
    data = []
    # Create 9 rows to ensure we have enough neighborhoods for testing splits
    for i in range(9):
        # Create target features [33]
        target_features = np.random.rand(33).astype(np.float32)

        # Create neighbor data with varying neighbor counts
        num_neighbors = i % 3  # 0, 1, or 2 neighbors per graph
        neighbor_data = []
        for j in range(num_neighbors):
            neighbor_data.append(
                {
                    "features": np.random.rand(33).astype(np.float32),
                    "dx": float(np.random.randn()),
                    "dy": float(np.random.randn()),
                    "euclidean_distance": float(np.random.rand() * 1000),
                    "network_distance": float(np.random.rand() * 1200),
                }
            )

        # Create target probability vector [8] (must sum to 1.0)
        target_prob_vector = np.random.rand(8).astype(np.float32)
        target_prob_vector = target_prob_vector / target_prob_vector.sum()

        data.append(
            {
                "target_id": f"target_{i}",
                "neighborhood_name": f"neighborhood_{i % 3}",  # 3 different neighborhoods (appears 3 times each)
                "label": "Compliant",
                "target_features": target_features,
                "neighbor_data": neighbor_data,
                "target_prob_vector": target_prob_vector,
                "target_geometry": f"POINT(2.{i} 48.{i})",
                "num_neighbors": num_neighbors,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def temp_parquet_dir(sample_features_df):
    """Create temporary directory with parquet files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create subdirectories with different neighborhood names
        for neighborhood in sample_features_df["neighborhood_name"].unique():
            neighborhood_dir = Path(tmpdir) / neighborhood
            neighborhood_dir.mkdir(parents=True)

            # Filter data for this neighborhood
            neighborhood_df = sample_features_df[
                sample_features_df["neighborhood_name"] == neighborhood
            ].copy()

            # Save parquet file
            parquet_path = neighborhood_dir / "target_points.parquet"
            neighborhood_df.to_parquet(parquet_path, index=False)

        yield tmpdir


def test_dataset_len(sample_features_df):
    """Test dataset length."""
    dataset = SpatialGraphDataset(sample_features_df)
    assert len(dataset) == len(sample_features_df)
    assert dataset.len() == 9


def test_dataset_get(sample_features_df):
    """Test graph structure from get()."""
    dataset = SpatialGraphDataset(sample_features_df)

    # Test first graph (should have 0 neighbors)
    data = dataset[0]
    assert isinstance(data, Data)
    assert data.x.shape[0] == 1  # Only target node
    assert data.x.shape[1] == 33  # 33 features
    assert data.edge_index.shape == (2, 0)  # No edges
    assert data.edge_attr.shape == (0, 4)  # No edge attributes
    assert data.y.shape == (8,)  # Target probability vector
    assert data.target_id == "target_0"
    assert data.neighborhood_name == "neighborhood_0"
    assert data.num_nodes == 1

    # Test graph with neighbors (graph at index 1 should have 1 neighbor)
    data = dataset[1]
    assert data.x.shape[0] == 2  # Target + 1 neighbor
    assert data.x.shape[1] == 33
    assert data.edge_index.shape == (2, 1)  # 1 edge
    assert data.edge_attr.shape == (1, 4)  # 1 edge attribute
    assert data.edge_index[0, 0] == 1  # Source: neighbor (node 1)
    assert data.edge_index[1, 0] == 0  # Target: center (node 0)
    assert data.num_nodes == 2


def test_dataset_zero_neighbors(sample_features_df):
    """Test edge case with 0 neighbors."""
    # Graph at index 0 should have 0 neighbors
    dataset = SpatialGraphDataset(sample_features_df)
    data = dataset[0]

    # Should have only target node
    assert data.x.shape == (1, 33)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0, 4)
    assert data.y.shape == (8,)


def test_dataset_get_large_neighbors(sample_features_df):
    """Test dataset with varying neighbor counts."""
    dataset = SpatialGraphDataset(sample_features_df)

    # Check all graphs
    for i in range(len(dataset)):
        data = dataset[i]
        expected_num_neighbors = sample_features_df.iloc[i]["num_neighbors"]
        assert data.x.shape[0] == 1 + expected_num_neighbors
        assert data.edge_index.shape[1] == expected_num_neighbors
        assert data.edge_attr.shape[0] == expected_num_neighbors


def test_dataset_missing_columns():
    """Test dataset with missing required columns."""
    df = pd.DataFrame({"target_id": [1, 2], "neighborhood_name": ["A", "B"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        SpatialGraphDataset(df)


def test_load_features_from_directory(temp_parquet_dir):
    """Test loading features from directory."""
    df = load_features_from_directory(temp_parquet_dir)

    assert len(df) > 0
    assert "target_id" in df.columns
    assert "neighborhood_name" in df.columns
    assert "target_features" in df.columns
    assert "neighbor_data" in df.columns
    assert "target_prob_vector" in df.columns


def test_load_features_from_directory_nonexistent():
    """Test loading from non-existent directory."""
    with pytest.raises(FileNotFoundError):
        load_features_from_directory("/nonexistent/directory/path")


def test_load_features_from_directory_empty():
    """Test loading from directory with no parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No target_points.parquet files found"):
            load_features_from_directory(tmpdir)


def test_create_train_val_test_splits(sample_features_df):
    """Test creating train/val/test splits."""
    # Use default 0.7/0.15/0.15 ratios which work better with 3 neighborhoods
    train_df, val_df, test_df = create_train_val_test_splits(
        sample_features_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    # Check all data is split
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_features_df)

    # Check neighborhoods are assigned to splits
    # With 3 neighborhoods and 0.7/0.15/0.15: train=2, val=0, test=1 (due to rounding)
    # Using default ratios ensures train always has data
    assert len(train_df) > 0


def test_splits_no_leakage(sample_features_df):
    """Verify no neighborhood overlap between splits."""
    train_df, val_df, test_df = create_train_val_test_splits(sample_features_df)

    train_neighborhoods = set(train_df["neighborhood_name"].unique())
    val_neighborhoods = set(val_df["neighborhood_name"].unique())
    test_neighborhoods = set(test_df["neighborhood_name"].unique())

    # No overlap between splits
    assert len(train_neighborhoods & val_neighborhoods) == 0
    assert len(train_neighborhoods & test_neighborhoods) == 0
    assert len(val_neighborhoods & test_neighborhoods) == 0


def test_splits_invalid_ratios(sample_features_df):
    """Test split function with invalid ratios."""
    # Ratios don't sum to 1.0
    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        create_train_val_test_splits(sample_features_df, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_splits_missing_column():
    """Test split function with missing neighborhood_name column."""
    df = pd.DataFrame({"target_id": [1, 2], "target_features": [None, None]})

    with pytest.raises(ValueError, match="must have 'neighborhood_name' column"):
        create_train_val_test_splits(df)


def test_splits_too_few_neighborhoods():
    """Test split function with too few neighborhoods."""
    # Create DataFrame with only 1 neighborhood
    df = pd.DataFrame(
        {
            "target_id": ["1", "2"],
            "neighborhood_name": ["A", "A"],
            "target_features": [np.random.rand(33), np.random.rand(33)],
            "neighbor_data": [[], []],
            "target_prob_vector": [np.random.rand(8) / np.random.rand(8).sum(), np.random.rand(8) / np.random.rand(8).sum()],
            "num_neighbors": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="Need at least 3 neighborhoods"):
        create_train_val_test_splits(df)


def test_get_device():
    """Test device selection utility."""
    device = get_device()
    assert isinstance(device, torch.device)
    # Should return one of: cpu, mps, cuda
    assert device.type in ["cpu", "mps", "cuda"]


def test_dataset_data_types(sample_features_df):
    """Test that data types are correctly converted to tensors."""
    dataset = SpatialGraphDataset(sample_features_df)
    data = dataset[1]  # Graph with 1 neighbor

    # Check tensor types
    assert data.x.dtype == torch.float32
    assert data.edge_index.dtype == torch.long
    assert data.edge_attr.dtype == torch.float32
    assert data.y.dtype == torch.float32


def test_dataset_edge_attributes(sample_features_df):
    """Test edge attributes are correctly structured."""
    dataset = SpatialGraphDataset(sample_features_df)

    # Find a graph with neighbors
    for i in range(len(dataset)):
        data = dataset[i]
        if data.edge_index.shape[1] > 0:
            # Edge attributes should be [num_edges, 4]
            assert data.edge_attr.shape == (data.edge_index.shape[1], 4)
            # Edge attributes should be finite
            assert torch.isfinite(data.edge_attr).all()
            break
