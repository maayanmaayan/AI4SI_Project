"""Unit tests for SpatialGraphDataset class."""

import pytest
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from src.training.dataset import SpatialGraphDataset, create_train_val_test_splits


def create_sample_features_df(num_samples: int = 5, num_neighborhoods: int = 3) -> pd.DataFrame:
    """Create sample features DataFrame for testing.
    
    Args:
        num_samples: Number of samples to create.
        num_neighborhoods: Number of unique neighborhoods (default: 3).
    """
    data = []
    for i in range(num_samples):
        # Create target features (33 features)
        target_features = np.random.randn(33).astype(np.float32)

        # Create neighbor data (variable number of neighbors)
        num_neighbors = np.random.randint(0, 10)
        neighbor_data = []
        for j in range(num_neighbors):
            neighbor_data.append({
                "features": np.random.randn(33).astype(np.float32),
                "dx": np.random.randn(),
                "dy": np.random.randn(),
                "euclidean_distance": np.random.uniform(0, 1000),
                "network_distance": np.random.uniform(0, 1200),
            })

        # Create target probability vector (8 categories)
        target_prob_vector = np.random.rand(8).astype(np.float32)
        target_prob_vector = target_prob_vector / target_prob_vector.sum()  # Normalize

        data.append({
            "target_id": f"target_{i}",
            "neighborhood_name": f"neighborhood_{i % num_neighborhoods}",
            "target_features": target_features,
            "neighbor_data": neighbor_data,
            "target_prob_vector": target_prob_vector,
            "num_neighbors": num_neighbors,
        })

    return pd.DataFrame(data)


def test_dataset_creation():
    """Test dataset initialization."""
    df = create_sample_features_df(5)
    dataset = SpatialGraphDataset(df)

    assert len(dataset) == 5
    assert dataset.features_df is not None


def test_get_item():
    """Test data loading from dataset."""
    df = create_sample_features_df(5)
    dataset = SpatialGraphDataset(df)

    # Get first item
    data = dataset[0]

    assert isinstance(data, Data)
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "edge_attr")
    assert hasattr(data, "y")
    assert hasattr(data, "target_id")
    assert hasattr(data, "neighborhood_name")


def test_star_graph_structure():
    """Test star graph construction."""
    df = create_sample_features_df(1)
    row = df.iloc[0]
    num_neighbors = len(row["neighbor_data"])

    dataset = SpatialGraphDataset(df)
    data = dataset[0]

    # Check node features shape: [1 + num_neighbors, 33]
    assert data.x.shape == (1 + num_neighbors, 33)

    # Check edge index shape: [2, num_neighbors]
    if num_neighbors > 0:
        assert data.edge_index.shape == (2, num_neighbors)
        # Check that all edges point to node 0 (target)
        assert torch.all(data.edge_index[1] == 0)
        # Check that sources are neighbors (1 to num_neighbors)
        assert torch.all(data.edge_index[0] == torch.arange(1, num_neighbors + 1))
    else:
        assert data.edge_index.shape == (2, 0)


def test_edge_attributes():
    """Test edge attribute construction."""
    df = create_sample_features_df(1)
    row = df.iloc[0]
    num_neighbors = len(row["neighbor_data"])

    dataset = SpatialGraphDataset(df)
    data = dataset[0]

    if num_neighbors > 0:
        # Check edge attributes shape: [num_neighbors, 4]
        assert data.edge_attr.shape == (num_neighbors, 4)

        # Check edge attributes contain expected values
        # Use relaxed tolerance for floating point comparisons
        for i, neighbor in enumerate(row["neighbor_data"]):
            assert abs(data.edge_attr[i, 0].item() - neighbor["dx"]) < 1e-4
            assert abs(data.edge_attr[i, 1].item() - neighbor["dy"]) < 1e-4
            assert abs(data.edge_attr[i, 2].item() - neighbor["euclidean_distance"]) < 1e-4
            assert abs(data.edge_attr[i, 3].item() - neighbor["network_distance"]) < 1e-4
    else:
        assert data.edge_attr.shape == (0, 4)


def test_empty_neighbors():
    """Test handling of target points with 0 neighbors."""
    # Create DataFrame with one sample that has 0 neighbors
    data = [{
        "target_id": "target_0",
        "neighborhood_name": "neighborhood_0",
        "target_features": np.random.randn(33).astype(np.float32),
        "neighbor_data": [],  # No neighbors
        "target_prob_vector": np.random.rand(8).astype(np.float32),
        "num_neighbors": 0,
    }]
    df = pd.DataFrame(data)

    dataset = SpatialGraphDataset(df)
    data_obj = dataset[0]

    # Should still have valid structure
    assert data_obj.x.shape == (1, 33)  # Only target node
    assert data_obj.edge_index.shape == (2, 0)  # No edges
    assert data_obj.edge_attr.shape == (0, 4)  # No edge attributes
    assert data_obj.y.shape == (1, 8)  # Target probability vector (2D for proper batching)


def test_target_probability_vector():
    """Test target probability vector is correctly stored."""
    df = create_sample_features_df(1)
    row = df.iloc[0]

    dataset = SpatialGraphDataset(df)
    data = dataset[0]

    # Check target probability vector shape: [1, 8] (2D for proper batching)
    assert data.y.shape == (1, 8)

    # Check values match (y is now [1, 8] instead of [8])
    expected = torch.tensor(row["target_prob_vector"], dtype=torch.float32).unsqueeze(0)
    assert torch.allclose(data.y, expected, atol=1e-5)


def test_create_train_val_test_splits():
    """Test train/val/test split creation."""
    # Test with 3 neighborhoods (edge case that previously failed)
    df = create_sample_features_df(30, num_neighborhoods=3)  # 30 samples across 3 neighborhoods

    train_df, val_df, test_df = create_train_val_test_splits(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    # Check splits are non-empty (split function should ensure at least 1 neighborhood per split)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Check no overlap between splits (neighborhood-level splits)
    train_neighborhoods = set(train_df["neighborhood_name"].unique())
    val_neighborhoods = set(val_df["neighborhood_name"].unique())
    test_neighborhoods = set(test_df["neighborhood_name"].unique())

    assert len(train_neighborhoods & val_neighborhoods) == 0
    assert len(train_neighborhoods & test_neighborhoods) == 0
    assert len(val_neighborhoods & test_neighborhoods) == 0

    # Check total samples preserved
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_dataset_metadata():
    """Test that metadata (target_id, neighborhood_name) is preserved."""
    df = create_sample_features_df(5)
    dataset = SpatialGraphDataset(df)

    for i in range(len(dataset)):
        data = dataset[i]
        row = df.iloc[i]

        assert data.target_id == row["target_id"]
        assert data.neighborhood_name == row["neighborhood_name"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
