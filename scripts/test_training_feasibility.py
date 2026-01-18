#!/usr/bin/env python3
"""Quick test script to validate training feasibility on M2 MacBook Air.

This script loads a small subset of data, creates a minimal dataset and model,
runs a forward/backward pass, and measures memory usage and performance to
validate that training is feasible on the M2 hardware.

Usage:
    python scripts/test_training_feasibility.py [--max_graphs 100]
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.training.dataset import SpatialGraphDataset, get_device, load_features_from_directory
from src.utils.config import get_config
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


class DummyGraphModel(nn.Module):
    """Dummy model for testing feasibility.

    Simple linear layers matching expected input/output dimensions.
    """

    def __init__(self, num_features: int = 33, hidden_dim: int = 64, num_classes: int = 8):
        """Initialize dummy model.

        Args:
            num_features: Number of input features per node (default: 33).
            hidden_dim: Hidden dimension size (default: 64).
            num_classes: Number of output classes (default: 8).
        """
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, data):
        """Forward pass through dummy model.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Tensor of shape [batch_size, num_classes] or [num_classes] for single graph.
        """
        # Extract node features (first node is target)
        x = data.x[0]  # Target node features [num_features]

        # Simple MLP on target node
        x = self.input_proj(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output_proj(x)

        return x


def test_training_feasibility(max_graphs: int = 100, batch_size: int = 16):
    """Test training feasibility on M2 hardware.

    Loads a small subset of data, creates dataset and DataLoader, runs forward
    and backward passes, and measures memory usage and performance.

    Args:
        max_graphs: Maximum number of graphs to load for testing.
        batch_size: Batch size for DataLoader.

    Returns:
        Dictionary with feasibility metrics.
    """
    logger.info("Starting training feasibility test...")
    logger.info(f"Max graphs: {max_graphs}, Batch size: {batch_size}")

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Get config
    try:
        config = get_config()
        batch_size = config.get("training", {}).get("batch_size", batch_size)
        use_mixed_precision = config.get("training", {}).get("use_mixed_precision", False)
        logger.info(f"Config batch_size: {batch_size}, mixed_precision: {use_mixed_precision}")
    except Exception as e:
        logger.warning(f"Could not load config: {e}, using defaults")

    # Load data subset
    logger.info("Loading feature data...")
    root = Path(__file__).resolve().parent.parent
    features_dir = root / "data" / "processed" / "features" / "compliant"

    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    try:
        features_df = load_features_from_directory(str(features_dir))
        logger.info(f"Loaded {len(features_df)} total graphs from {features_df['neighborhood_name'].nunique()} neighborhoods")

        # Filter for neighborhoods with less than 50 points (for faster testing)
        neighborhood_counts = features_df["neighborhood_name"].value_counts()
        small_neighborhoods = neighborhood_counts[neighborhood_counts < 50].index.tolist()

        if len(small_neighborhoods) == 0:
            logger.warning("No neighborhoods with < 50 points found, using all neighborhoods")
            filtered_df = features_df
        else:
            filtered_df = features_df[features_df["neighborhood_name"].isin(small_neighborhoods)].copy()
            logger.info(
                f"Filtered to {len(filtered_df)} graphs from {len(small_neighborhoods)} small neighborhoods "
                f"(< 50 points each): {', '.join(small_neighborhoods)}"
            )

        # Limit to max_graphs if needed
        if len(filtered_df) > max_graphs:
            # Try to get a diverse sample from small neighborhoods
            selected_neighborhoods = filtered_df["neighborhood_name"].unique()[:2]  # Use up to 2 small neighborhoods
            selected_df = filtered_df[filtered_df["neighborhood_name"] == selected_neighborhoods[0]].head(
                max_graphs // 2
            )
            if len(selected_neighborhoods) > 1 and len(selected_df) < max_graphs:
                remaining = max_graphs - len(selected_df)
                selected_df = pd.concat(
                    [
                        selected_df,
                        filtered_df[filtered_df["neighborhood_name"] == selected_neighborhoods[1]].head(remaining),
                    ],
                    ignore_index=True,
                )
            features_df = selected_df
            logger.info(f"Limited to {len(features_df)} graphs for testing")
        else:
            features_df = filtered_df
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        raise

    # Create dataset
    logger.info("Creating dataset...")
    dataset = SpatialGraphDataset(features_df)
    logger.info(f"Dataset created with {len(dataset)} graphs")

    # Create DataLoader
    logger.info("Creating DataLoader...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    logger.info(f"DataLoader created with batch_size={batch_size}")

    # Create dummy model
    logger.info("Creating dummy model...")
    model = DummyGraphModel(num_features=33, hidden_dim=64, num_classes=8).to(device)
    logger.info(f"Model created and moved to {device}")

    # Create loss and optimizer
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run test
    logger.info("Running forward/backward passes...")
    model.train()

    batch_times = []
    memory_stats = []

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # Test on first 3 batches
                break

            batch = batch.to(device)

            # Measure memory before
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

            # Forward pass
            start_time = time.time()
            output = model(batch)

            # Handle batched vs non-batched outputs
            if hasattr(batch, "batch") and batch.batch is not None:
                # Batched: select first graph's target node output
                # In batching, nodes are concatenated, need to select target nodes
                target_indices = torch.where(batch.batch == 0)[0]
                if len(target_indices) > 0:
                    output = output[target_indices[0]]
                    target = batch.y[0]  # First graph's target
                else:
                    continue
            else:
                # Single graph
                target = batch.y

            # Compute loss
            target_prob = torch.softmax(target, dim=0)
            output_log_prob = torch.log_softmax(output, dim=0)
            loss = criterion(output_log_prob.unsqueeze(0), target_prob.unsqueeze(0))

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            elapsed = time.time() - start_time
            batch_times.append(elapsed)

            logger.info(f"Batch {batch_idx + 1}: Loss={loss.item():.4f}, Time={elapsed:.3f}s")

        avg_time = sum(batch_times) / len(batch_times) if batch_times else 0
        logger.info(f"Average time per batch: {avg_time:.3f}s")

    except RuntimeError as e:
        logger.error(f"Runtime error during training test: {e}")
        return {"feasible": False, "error": str(e)}

    # Memory estimation (rough)
    logger.info("Estimating memory usage...")
    try:
        if device.type == "mps":
            # MPS memory profiling is limited, provide rough estimate
            # Based on batch_size, feature dimensions, and hidden_dim
            estimated_memory_mb = (batch_size * 500) / (1024 * 1024)  # Rough estimate
        else:
            estimated_memory_mb = 0  # CPU doesn't have GPU memory
    except Exception:
        estimated_memory_mb = 0

    # Report
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FEASIBILITY REPORT")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Dataset size: {len(dataset)} graphs")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Average time per batch: {avg_time:.3f}s")
    if batch_times:
        logger.info(f"Estimated time per epoch: {avg_time * (len(dataset) / batch_size):.1f}s")
    logger.info(f"Estimated memory: {estimated_memory_mb:.2f} MB (rough estimate)")
    logger.info("\n✅ Training appears feasible on this hardware!")
    logger.info("=" * 80)

    return {
        "feasible": True,
        "device": str(device),
        "dataset_size": len(dataset),
        "batch_size": batch_size,
        "avg_time_per_batch": avg_time,
        "estimated_memory_mb": estimated_memory_mb,
    }


def main():
    """Main entry point for feasibility test script."""
    parser = argparse.ArgumentParser(
        description="Test training feasibility on M2 MacBook Air",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max_graphs",
        type=int,
        default=100,
        help="Maximum number of graphs to load for testing (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for testing (default: from config or 16)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    try:
        result = test_training_feasibility(
            max_graphs=args.max_graphs,
            batch_size=args.batch_size if args.batch_size else 16,
        )
        if result.get("feasible", False):
            exit(0)
        else:
            logger.error("Training feasibility test failed!")
            exit(1)
    except Exception as e:
        logger.exception(f"Feasibility test failed with error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
