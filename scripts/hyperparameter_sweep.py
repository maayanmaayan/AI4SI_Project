#!/usr/bin/env python3
"""Hyperparameter sweep script for Spatial Graph Transformer model.

This script trains multiple model configurations as specified in the PRD,
compares their performance, and generates comparison plots.

Usage:
    python scripts/hyperparameter_sweep.py [--quick-test] [--config CONFIG_PATH]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import train
from src.utils.config import get_config, load_config
from src.utils.logging import setup_logging, get_logger
from src.evaluation.plotting import plot_hyperparameter_comparison

logger = get_logger(__name__)

# Model configurations from PRD.md
MODEL_CONFIGURATIONS = [
    {
        "id": 1,
        "name": "baseline",
        "description": "Baseline (current config)",
        "learning_rate": 0.001,
        "num_layers": 3,
        "temperature": 200,
    },
    {
        "id": 2,
        "name": "lower_lr",
        "description": "Lower learning rate",
        "learning_rate": 0.0005,
        "num_layers": 3,
        "temperature": 200,
    },
    {
        "id": 3,
        "name": "higher_lr",
        "description": "Higher learning rate",
        "learning_rate": 0.002,
        "num_layers": 3,
        "temperature": 200,
    },
    {
        "id": 4,
        "name": "shallow",
        "description": "Fewer layers (shallow)",
        "learning_rate": 0.001,
        "num_layers": 2,
        "temperature": 200,
    },
    {
        "id": 5,
        "name": "deeper",
        "description": "More layers (deeper)",
        "learning_rate": 0.001,
        "num_layers": 4,
        "temperature": 200,
    },
    {
        "id": 6,
        "name": "lower_temp",
        "description": "Lower temperature (sharper target distribution)",
        "learning_rate": 0.001,
        "num_layers": 3,
        "temperature": 150,
    },
    {
        "id": 7,
        "name": "higher_temp",
        "description": "Higher temperature (smoother target distribution)",
        "learning_rate": 0.001,
        "num_layers": 3,
        "temperature": 250,
    },
]


def create_config_for_model(base_config: dict, model_config: dict) -> dict:
    """Create a configuration dictionary for a specific model configuration.

    Args:
        base_config: Base configuration dictionary.
        model_config: Model-specific hyperparameters.

    Returns:
        Configuration dictionary with updated hyperparameters.
    """
    config = json.loads(json.dumps(base_config))  # Deep copy

    # Update model hyperparameters
    config["model"]["num_layers"] = model_config["num_layers"]

    # Update training hyperparameters
    config["training"]["learning_rate"] = model_config["learning_rate"]

    # Update loss hyperparameters
    config["loss"]["temperature"] = model_config["temperature"]

    return config


def run_hyperparameter_sweep(
    base_config_path: str = None,
    quick_test: bool = False,
    features_dir: str = None,
    sweep_dir: str = None,
) -> Dict:
    """Run hyperparameter sweep across all model configurations.

    Args:
        base_config_path: Path to base configuration file. If None, uses default.
        quick_test: If True, use quick test mode (small neighborhoods only).
        features_dir: Directory containing processed features. If None, uses config.
        sweep_dir: Directory to save sweep results. If None, creates timestamped directory.

    Returns:
        Dictionary with sweep results including best model and comparison metrics.
    """
    # Load base configuration
    if base_config_path:
        base_config = load_config(base_config_path)
    else:
        base_config = get_config()

    # Setup sweep directory
    if sweep_dir is None:
        sweep_dir = Path(base_config.get("paths", {}).get("experiments_dir", "experiments/runs")) / f"sweep_{int(time.time())}"
    else:
        sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting Hyperparameter Sweep")
    logger.info("=" * 80)
    logger.info(f"Sweep directory: {sweep_dir}")
    logger.info(f"Number of configurations: {len(MODEL_CONFIGURATIONS)}")
    logger.info(f"Quick test mode: {quick_test}")

    # Store results for each configuration
    results = []
    best_model = None
    best_val_loss = float("inf")

    # Train each configuration
    for i, model_config in enumerate(MODEL_CONFIGURATIONS, 1):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Training Model {model_config['id']}/{len(MODEL_CONFIGURATIONS)}: {model_config['name']}")
        logger.info(f"Description: {model_config['description']}")
        logger.info(f"Learning Rate: {model_config['learning_rate']}")
        logger.info(f"Layers: {model_config['num_layers']}")
        logger.info(f"Temperature: {model_config['temperature']}")
        logger.info("=" * 80)

        # Create configuration for this model
        config = create_config_for_model(base_config, model_config)

        # Create experiment directory for this model
        experiment_dir = sweep_dir / f"model_{model_config['id']}_{model_config['name']}"

        try:
            # Train model
            summary = train(
                config=config,
                features_dir=features_dir,
                experiment_dir=str(experiment_dir),
                quick_test=quick_test,
            )

            # Extract results
            result = {
                "model_id": model_config["id"],
                "model_name": model_config["name"],
                "description": model_config["description"],
                "learning_rate": model_config["learning_rate"],
                "num_layers": model_config["num_layers"],
                "temperature": model_config["temperature"],
                "best_val_loss": summary.get("best_val_loss", float("inf")),
                "total_epochs": summary.get("total_epochs", 0),
                "test_metrics": summary.get("test_metrics", {}),
                "experiment_dir": str(experiment_dir),
            }

            results.append(result)

            # Track best model
            if result["best_val_loss"] < best_val_loss:
                best_val_loss = result["best_val_loss"]
                best_model = result

            logger.info(f"✓ Model {model_config['id']} completed: val_loss={result['best_val_loss']:.4f}")

        except Exception as e:
            logger.error(f"✗ Model {model_config['id']} failed: {e}")
            results.append(
                {
                    "model_id": model_config["id"],
                    "model_name": model_config["name"],
                    "error": str(e),
                }
            )

    # Save sweep results
    results_path = sweep_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "sweep_dir": str(sweep_dir),
                "best_model": best_model,
                "results": results,
                "summary": {
                    "total_configurations": len(MODEL_CONFIGURATIONS),
                    "successful": len([r for r in results if "error" not in r]),
                    "failed": len([r for r in results if "error" in r]),
                    "best_val_loss": best_val_loss,
                    "best_model_id": best_model["model_id"] if best_model else None,
                },
            },
            f,
            indent=2,
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info("Hyperparameter Sweep Complete")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_path}")

    if best_model:
        logger.info(f"Best model: Model {best_model['model_id']} ({best_model['model_name']})")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best model directory: {best_model['experiment_dir']}")

    # Generate comparison plots
    try:
        plots_dir = sweep_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Plot comparison for each metric
        metrics_to_plot = ["best_val_loss", "test_loss", "test_kl_divergence", "test_top1_accuracy"]
        for metric in metrics_to_plot:
            plot_hyperparameter_comparison(
                results,
                metric=metric,
                save_path=str(plots_dir / f"hyperparameter_comparison_{metric}.png"),
            )

        logger.info(f"Comparison plots saved to: {plots_dir}")
    except Exception as e:
        logger.warning(f"Failed to generate comparison plots: {e}")

    return {
        "sweep_dir": str(sweep_dir),
        "best_model": best_model,
        "results": results,
    }


def main():
    """Main entry point for hyperparameter sweep script."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for Spatial Graph Transformer model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to base configuration file (default: models/config.yaml)",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Directory containing processed features (default: from config)",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Directory to save sweep results (default: experiments/runs/sweep_{timestamp})",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode: use only 3 small neighborhoods with <50 points each",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="INFO")

    # Run sweep
    results = run_hyperparameter_sweep(
        base_config_path=args.config,
        quick_test=args.quick_test,
        features_dir=args.features_dir,
        sweep_dir=args.sweep_dir,
    )

    print("\n" + "=" * 80)
    print("Sweep Results Summary")
    print("=" * 80)
    print(f"Sweep directory: {results['sweep_dir']}")
    if results["best_model"]:
        print(f"Best model: Model {results['best_model']['model_id']} ({results['best_model']['model_name']})")
        print(f"Best validation loss: {results['best_model']['best_val_loss']:.4f}")
    print(f"Total configurations: {len(results['results'])}")
    print(f"Successful: {len([r for r in results['results'] if 'error' not in r])}")
    print(f"Failed: {len([r for r in results['results'] if 'error' in r])}")


if __name__ == "__main__":
    main()
