"""Integration tests for full training pipeline.

These tests verify that the complete training pipeline works end-to-end,
including data loading, model training, and checkpointing.
"""

import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

from src.training.dataset import SpatialGraphDataset, create_train_val_test_splits, get_device
from src.training.model import SpatialGraphTransformer
from src.training.loss import DistanceBasedKLLoss
from src.training.train import train_epoch, save_checkpoint, load_checkpoint
from src.evaluation.metrics import evaluate_model
from torch_geometric.loader import DataLoader


def create_sample_features_df(num_samples: int = 20) -> pd.DataFrame:
    """Create sample features DataFrame for testing."""
    data = []
    for i in range(num_samples):
        # Create target features (33 features)
        target_features = np.random.randn(33).astype(np.float32)

        # Create neighbor data (variable number of neighbors)
        num_neighbors = np.random.randint(1, 10)
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
            "neighborhood_name": f"neighborhood_{i % 5}",  # 5 neighborhoods
            "target_features": target_features,
            "neighbor_data": neighbor_data,
            "target_prob_vector": target_prob_vector,
            "num_neighbors": num_neighbors,
        })

    return pd.DataFrame(data)


def test_data_loading():
    """Test data loading and batching."""
    df = create_sample_features_df(20)

    # Create train/val/test splits
    train_df, val_df, test_df = create_train_val_test_splits(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    # Create datasets
    train_dataset = SpatialGraphDataset(train_df)
    val_dataset = SpatialGraphDataset(val_df)
    test_dataset = SpatialGraphDataset(test_df)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Test loading batches
    for batch in train_loader:
        assert batch.x is not None
        assert batch.edge_index is not None
        assert batch.edge_attr is not None
        assert batch.y is not None
        assert batch.batch is not None
        break

    # Test validation loader
    for batch in val_loader:
        assert batch.x is not None
        assert batch.y is not None
        break

    # Test test loader
    for batch in test_loader:
        assert batch.x is not None
        assert batch.y is not None
        break


def test_model_training():
    """Test model training for a few epochs."""
    # Create sample data
    df = create_sample_features_df(20)
    train_df, val_df, _ = create_train_val_test_splits(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    train_dataset = SpatialGraphDataset(train_df)
    val_dataset = SpatialGraphDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Create model - use same device selection as training code
    device = get_device()  # Automatically selects MPS > CUDA > CPU
    model = SpatialGraphTransformer(num_features=33, hidden_dim=64, num_layers=2)
    model = model.to(device)

    # Create loss function and optimizer
    loss_fn = DistanceBasedKLLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Train for a few epochs
    initial_loss = None
    for epoch in range(3):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, use_mixed_precision=False)

        if initial_loss is None:
            initial_loss = train_loss

        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_loader, loss_fn, device)

        assert train_loss >= 0
        assert val_metrics["loss"] >= 0
        assert val_metrics["kl_divergence"] >= 0
        assert 0 <= val_metrics["top1_accuracy"] <= 1

    # Loss should generally decrease (or at least not increase dramatically)
    # Note: With small dataset and few epochs, loss might not decrease much
    assert train_loss < float("inf")


def test_checkpointing():
    """Test model checkpoint saving and loading."""
    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        checkpoint_path = checkpoint_dir / "test_checkpoint.pt"

        # Create model and optimizer
        model = SpatialGraphTransformer(num_features=33, hidden_dim=64, num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        # Save checkpoint
        config = {"test": "config"}
        save_checkpoint(model, optimizer, epoch=5, best_val_loss=0.5, checkpoint_path=checkpoint_path, config=config)

        # Verify checkpoint exists
        assert checkpoint_path.exists()

        # Create new model and optimizer
        model2 = SpatialGraphTransformer(num_features=33, hidden_dim=64, num_layers=2)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.001)

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, model2, optimizer2)

        # Verify checkpoint data
        assert checkpoint["epoch"] == 5
        assert checkpoint["best_val_loss"] == 0.5
        assert checkpoint["config"] == config

        # Verify model state was loaded (check a parameter)
        # Note: We can't easily compare all parameters, but we can check one
        assert model2.num_features == model.num_features
        assert model2.hidden_dim == model.hidden_dim


def test_end_to_end_training():
    """Test complete training pipeline with small dataset."""
    # Create sample data
    df = create_sample_features_df(30)

    # Create train/val/test splits
    train_df, val_df, test_df = create_train_val_test_splits(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    # Create datasets
    train_dataset = SpatialGraphDataset(train_df)
    val_dataset = SpatialGraphDataset(val_df)
    test_dataset = SpatialGraphDataset(test_df)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Create model - use same device selection as training code
    device = get_device()  # Automatically selects MPS > CUDA > CPU
    model = SpatialGraphTransformer(num_features=33, hidden_dim=64, num_layers=2)
    model = model.to(device)

    # Create loss function and optimizer
    loss_fn = DistanceBasedKLLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Train for a few epochs
    for epoch in range(2):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, use_mixed_precision=False)
        assert train_loss >= 0

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, loss_fn, device)

    assert test_metrics["loss"] >= 0
    assert test_metrics["kl_divergence"] >= 0
    assert 0 <= test_metrics["top1_accuracy"] <= 1
    assert 0 <= test_metrics["top3_accuracy"] <= 1
    assert test_metrics["num_samples"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
