#!/usr/bin/env python3
"""Training script wrapper for Spatial Graph Transformer model.

This script provides a command-line interface for training the graph transformer model.
It loads configuration, processes features, and runs the training pipeline.

Usage:
    python scripts/train_graph_transformer.py [--config CONFIG_PATH] [--features-dir FEATURES_DIR]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import train
from src.utils.config import get_config, load_config
from src.utils.logging import setup_logging


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train Spatial Graph Transformer model for 15-minute city service gap prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: models/config.yaml)",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Directory containing processed features (default: from config)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: from config)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Directory for experiment logs (default: from config)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
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

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()

    # Run training
    try:
        summary = train(
            config=config,
            features_dir=args.features_dir,
            checkpoint_dir=args.checkpoint_dir,
            experiment_dir=args.experiment_dir,
            resume_from=args.resume_from,
        )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best validation loss: {summary['best_val_loss']:.4f}")
        print(f"Total epochs: {summary['total_epochs']}")
        print(f"Test loss: {summary['test_metrics']['loss']:.4f}")
        print(f"Test KL divergence: {summary['test_metrics']['kl_divergence']:.4f}")
        print(f"Test top-1 accuracy: {summary['test_metrics']['top1_accuracy']:.4f}")
        print(f"Results saved to: {summary['experiment_dir']}")
        print(f"Checkpoint saved to: {summary['checkpoint_path']}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
