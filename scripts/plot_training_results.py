#!/usr/bin/env python3
"""Script to generate plots from training history.

This script loads training history from JSON files and generates visualization plots.
Useful for analyzing completed training runs or comparing different experiments.

Usage:
    python scripts/plot_training_results.py --history-path experiments/runs/run_123456/training_history.json
    python scripts/plot_training_results.py --experiment-dir experiments/runs/run_123456
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch

from src.evaluation.plotting import (
    plot_training_summary,
    plot_neighbor_distribution,
    plot_target_probability_distribution,
    plot_hyperparameter_comparison,
)
from src.training.dataset import SpatialGraphDataset
from src.utils.helpers import get_service_category_names
from src.utils.logging import setup_logging, get_logger


def main():
    """Main entry point for plotting script."""
    parser = argparse.ArgumentParser(
        description="Generate plots from training history"
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=None,
        help="Path to training_history.json file",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Path to experiment directory (will look for training_history.json inside)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: plots/ or experiment_dir/plots/)",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Path to features directory for data exploration plots",
    )
    parser.add_argument(
        "--compare-experiments",
        type=str,
        nargs="+",
        default=None,
        help="List of experiment directories to compare",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)

    # Determine history path
    if args.history_path:
        history_path = Path(args.history_path)
    elif args.experiment_dir:
        history_path = Path(args.experiment_dir) / "training_history.json"
    else:
        logger.error("Must provide either --history-path or --experiment-dir")
        return 1

    if not history_path.exists():
        logger.error(f"Training history file not found: {history_path}")
        return 1

    # Determine plots directory
    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
    elif args.experiment_dir:
        plots_dir = Path(args.experiment_dir) / "plots"
    else:
        plots_dir = Path("plots")

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load training history
    logger.info(f"Loading training history from {history_path}")
    with open(history_path, "r") as f:
        history = json.load(f)

    if not history:
        logger.error("Training history is empty")
        return 1

    logger.info(f"Loaded {len(history)} epochs of training history")

    # Generate main training plots
    experiment_name = history_path.parent.name if args.experiment_dir else "training"
    plot_training_summary(history, plots_dir=str(plots_dir), experiment_name=experiment_name)

    # Generate data exploration plots if features directory provided
    if args.features_dir:
        try:
            from src.training.dataset import load_features_from_directory

            logger.info(f"Loading features from {args.features_dir}")
            features_df = load_features_from_directory(args.features_dir)

            # Plot neighbor distribution
            plot_neighbor_distribution(
                features_df,
                save_path=str(plots_dir / "neighbor_distribution.png"),
                title="Neighbor Count Distribution",
            )

            # Plot target probability distribution
            sample_df = features_df.sample(min(100, len(features_df)), random_state=42)
            sample_dataset = SpatialGraphDataset(sample_df)
            sample_target_probs = torch.stack([data.y for data in sample_dataset])
            category_names = get_service_category_names()

            plot_target_probability_distribution(
                sample_target_probs,
                category_names,
                save_path=str(plots_dir / "target_probability_distribution.png"),
                title="Target Probability Distribution",
            )

            logger.info("Generated data exploration plots")
        except Exception as e:
            logger.warning(f"Failed to generate data exploration plots: {e}")

    # Compare multiple experiments if requested
    if args.compare_experiments:
        logger.info("Comparing multiple experiments...")
        results = {}
        for exp_dir in args.compare_experiments:
            exp_path = Path(exp_dir)
            history_path_exp = exp_path / "training_history.json"
            if history_path_exp.exists():
                with open(history_path_exp, "r") as f:
                    results[exp_path.name] = json.load(f)
            else:
                logger.warning(f"History not found for {exp_dir}, skipping")

        if len(results) > 1:
            # Compare validation loss
            plot_hyperparameter_comparison(
                results,
                metric="val_loss",
                save_path=str(plots_dir / "hyperparameter_comparison_val_loss.png"),
                title="Hyperparameter Comparison - Validation Loss",
            )

            # Compare validation accuracy
            plot_hyperparameter_comparison(
                results,
                metric="val_top1_accuracy",
                save_path=str(plots_dir / "hyperparameter_comparison_val_accuracy.png"),
                title="Hyperparameter Comparison - Validation Accuracy",
            )

            logger.info("Generated comparison plots")

    logger.info(f"All plots saved to {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
