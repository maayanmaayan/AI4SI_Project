"""Plotting utilities for training visualization and model evaluation.

This module provides functions to create and save plots for training metrics,
model performance, and spatial predictions. All plots are saved to disk without
displaying them on screen.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_plots_dir(plots_dir: str = "plots") -> Path:
    """Ensure plots directory exists.

    Args:
        plots_dir: Path to plots directory.

    Returns:
        Path object for plots directory.
    """
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    return plots_path


def plot_training_curves(
    history: List[Dict],
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """Plot training, validation, and test loss curves over epochs.

    Args:
        history: List of dictionaries with training history (epoch, train_loss, val_loss, etc.).
        save_path: Path to save the plot.
        title: Plot title.

    Example:
        >>> history = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6}, ...]
        >>> plot_training_curves(history, "plots/training_curves.png")
    """
    if not history:
        logger.warning("Empty history, skipping plot")
        return

    df = pd.DataFrame(history)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    if "train_loss" in df.columns:
        ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="red", linewidth=2)
    if "val_loss" in df.columns:
        ax1.plot(df["epoch"], df["val_loss"], label="Val Loss", color="blue", linewidth=2)
    if "test_loss" in df.columns:
        ax1.plot(df["epoch"], df["test_loss"], label="Test Loss", color="green", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curves", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: KL Divergence
    ax2 = axes[0, 1]
    if "train_kl_divergence" in df.columns:
        ax2.plot(df["epoch"], df["train_kl_divergence"], label="Train KL", color="red", linewidth=2, linestyle="--")
    if "val_kl_divergence" in df.columns:
        ax2.plot(df["epoch"], df["val_kl_divergence"], label="Val KL", color="blue", linewidth=2, linestyle="--")
    if "test_kl_divergence" in df.columns:
        ax2.plot(df["epoch"], df["test_kl_divergence"], label="Test KL", color="green", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("KL Divergence", fontsize=12)
    ax2.set_title("KL Divergence", fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Top-1 Accuracy
    ax3 = axes[1, 0]
    if "train_top1_accuracy" in df.columns:
        ax3.plot(df["epoch"], df["train_top1_accuracy"], label="Train Top-1", color="red", linewidth=2)
    if "val_top1_accuracy" in df.columns:
        ax3.plot(df["epoch"], df["val_top1_accuracy"], label="Val Top-1", color="blue", linewidth=2)
    if "test_top1_accuracy" in df.columns:
        ax3.plot(df["epoch"], df["test_top1_accuracy"], label="Test Top-1", color="green", linewidth=2)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Top-1 Accuracy", fontsize=12)
    ax3.set_title("Top-1 Accuracy", fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Add Top-3 Accuracy subplot if needed (update axes layout)

    # Plot 4: Learning Rate
    ax4 = axes[1, 1]
    if "learning_rate" in df.columns:
        ax4.plot(df["epoch"], df["learning_rate"], label="Learning Rate", color="purple", linewidth=2)
        ax4.set_yscale("log")
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("Learning Rate (log scale)", fontsize=12)
    ax4.set_title("Learning Rate Schedule", fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved training curves to {save_path}")


def plot_loss_comparison(
    history: List[Dict],
    save_path: str,
    title: str = "Loss Comparison",
) -> None:
    """Plot detailed loss comparison between train, validation, and test.

    Args:
        history: List of dictionaries with training history.
        save_path: Path to save the plot.
        title: Plot title.
    """
    if not history:
        logger.warning("Empty history, skipping plot")
        return

    df = pd.DataFrame(history)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    if "train_loss" in df.columns:
        ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="red", linewidth=2, alpha=0.8)
    if "val_loss" in df.columns:
        ax.plot(df["epoch"], df["val_loss"], label="Val Loss", color="blue", linewidth=2, alpha=0.8)
    if "test_loss" in df.columns:
        ax.plot(df["epoch"], df["test_loss"], label="Test Loss", color="green", linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved loss comparison to {save_path}")


def plot_accuracy_curves(
    history: List[Dict],
    save_path: str,
    title: str = "Accuracy Curves",
) -> None:
    """Plot top-k accuracy curves over epochs.

    Args:
        history: List of dictionaries with training history.
        save_path: Path to save the plot.
        title: Plot title.
    """
    if not history:
        logger.warning("Empty history, skipping plot")
        return

    df = pd.DataFrame(history)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Training accuracies
    if "train_top1_accuracy" in df.columns:
        ax.plot(df["epoch"], df["train_top1_accuracy"], label="Train Top-1", color="red", linewidth=2, marker='o', markersize=4)
    if "train_top3_accuracy" in df.columns:
        ax.plot(df["epoch"], df["train_top3_accuracy"], label="Train Top-3", color="pink", linewidth=2, marker='s', markersize=4)
    
    # Validation accuracies
    if "val_top1_accuracy" in df.columns:
        ax.plot(df["epoch"], df["val_top1_accuracy"], label="Val Top-1", color="blue", linewidth=2, marker='o', markersize=4)
    if "val_top3_accuracy" in df.columns:
        ax.plot(df["epoch"], df["val_top3_accuracy"], label="Val Top-3", color="green", linewidth=2, marker='s', markersize=4)
    
    # Test accuracies
    if "test_top1_accuracy" in df.columns:
        ax.plot(df["epoch"], df["test_top1_accuracy"], label="Test Top-1", color="orange", linewidth=2, linestyle="--", marker='o', markersize=4)
    if "test_top3_accuracy" in df.columns:
        ax.plot(df["epoch"], df["test_top3_accuracy"], label="Test Top-3", color="purple", linewidth=2, linestyle="--", marker='s', markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved accuracy curves to {save_path}")


def plot_hyperparameter_comparison(
    results: List[Dict],
    metric: str = "best_val_loss",
    save_path: str = "plots/hyperparameter_comparison.png",
    title: str = "Hyperparameter Comparison",
) -> None:
    """Plot comparison of different hyperparameter settings.

    Args:
        results: List of result dictionaries from hyperparameter sweep, each with:
            - model_id, model_name, description
            - learning_rate, num_layers, temperature
            - best_val_loss, test_metrics, etc.
        metric: Metric to compare (e.g., "best_val_loss", "test_loss", "test_top1_accuracy").
        save_path: Path to save the plot.
        title: Plot title.

    Example:
        >>> results = [
        ...     {"model_id": 1, "model_name": "baseline", "best_val_loss": 0.5, ...},
        ...     {"model_id": 2, "model_name": "lower_lr", "best_val_loss": 0.6, ...},
        ... ]
        >>> plot_hyperparameter_comparison(results, metric="best_val_loss")
    """
    # Filter out failed runs
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        logger.warning("No valid results to plot")
        return

    # Extract metric values and labels
    model_names = []
    metric_values = []
    
    for result in valid_results:
        # Create descriptive label
        label_parts = [f"Model {result['model_id']}"]
        if "description" in result:
            label_parts.append(result["description"])
        else:
            label_parts.append(result.get("model_name", ""))
        label = ": ".join(label_parts)
        model_names.append(label)
        
        # Extract metric value
        if metric == "best_val_loss":
            metric_values.append(result.get("best_val_loss", float("inf")))
        elif metric.startswith("test_"):
            test_metrics = result.get("test_metrics", {})
            metric_key = metric.replace("test_", "")
            metric_values.append(test_metrics.get(metric_key, 0.0))
        else:
            metric_values.append(result.get(metric, 0.0))

    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(f"{title} - {metric.replace('_', ' ').title()}", fontsize=16, fontweight='bold')

    # Sort by metric value (best first for loss, worst first for accuracy)
    if "loss" in metric.lower() or "kl" in metric.lower():
        sorted_indices = sorted(range(len(metric_values)), key=lambda i: metric_values[i])
    else:
        sorted_indices = sorted(range(len(metric_values)), key=lambda i: metric_values[i], reverse=True)
    
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_values = [metric_values[i] for i in sorted_indices]

    # Create bar plot
    bars = ax.barh(range(len(sorted_names)), sorted_values, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Highlight best model
    if "loss" in metric.lower() or "kl" in metric.lower():
        best_idx = 0  # First (lowest) is best
    else:
        best_idx = 0  # First (highest) is best
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, sorted_values)):
        ax.text(value, i, f" {value:.4f}", va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Model Configuration", fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved hyperparameter comparison to {save_path}")


def plot_neighbor_distribution(
    features_df: pd.DataFrame,
    save_path: str,
    title: str = "Neighbor Count Distribution",
) -> None:
    """Plot distribution of neighbor counts per target point.

    Args:
        features_df: DataFrame with 'num_neighbors' column.
        save_path: Path to save the plot.
        title: Plot title.
    """
    if "num_neighbors" not in features_df.columns:
        logger.warning("num_neighbors column not found, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    neighbor_counts = features_df["num_neighbors"].values

    # Histogram
    ax1 = axes[0]
    ax1.hist(neighbor_counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel("Number of Neighbors", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Histogram", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Box plot
    ax2 = axes[1]
    ax2.boxplot(neighbor_counts, vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax2.set_ylabel("Number of Neighbors", fontsize=12)
    ax2.set_title("Box Plot", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = f"Mean: {neighbor_counts.mean():.1f}\n"
    stats_text += f"Median: {np.median(neighbor_counts):.1f}\n"
    stats_text += f"Std: {neighbor_counts.std():.1f}\n"
    stats_text += f"Min: {neighbor_counts.min()}\n"
    stats_text += f"Max: {neighbor_counts.max()}"
    ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved neighbor distribution to {save_path}")


def plot_target_probability_distribution(
    target_probs: torch.Tensor,
    category_names: List[str],
    save_path: str,
    title: str = "Target Probability Distribution",
    num_samples: int = 100,
) -> None:
    """Plot distribution of target probability vectors across categories.

    Args:
        target_probs: Tensor of target probability vectors [N, 8].
        category_names: List of 8 service category names.
        save_path: Path to save the plot.
        title: Plot title.
        num_samples: Number of samples to plot (for large datasets).
    """
    if isinstance(target_probs, torch.Tensor):
        target_probs = target_probs.cpu().numpy()

    # Sample if too many
    if len(target_probs) > num_samples:
        indices = np.random.choice(len(target_probs), num_samples, replace=False)
        target_probs = target_probs[indices]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Mean probabilities per category
    ax1 = axes[0, 0]
    mean_probs = target_probs.mean(axis=0)
    ax1.bar(range(len(category_names)), mean_probs, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(category_names)))
    ax1.set_xticklabels(category_names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel("Mean Probability", fontsize=12)
    ax1.set_title("Mean Target Probabilities by Category", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Box plot of probabilities per category
    ax2 = axes[0, 1]
    data_to_plot = [target_probs[:, i] for i in range(len(category_names))]
    bp = ax2.boxplot(data_to_plot, labels=category_names, patch_artist=True,
                     boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_title("Probability Distribution by Category", fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Heatmap of probabilities (sample of points)
    ax3 = axes[1, 0]
    sample_size = min(50, len(target_probs))
    sample_probs = target_probs[:sample_size]
    im = ax3.imshow(sample_probs.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax3.set_xlabel("Sample Index", fontsize=12)
    ax3.set_ylabel("Category", fontsize=12)
    ax3.set_yticks(range(len(category_names)))
    ax3.set_yticklabels(category_names, fontsize=9)
    ax3.set_title(f"Probability Heatmap (First {sample_size} Samples)", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3, label="Probability")

    # Plot 4: Distribution of max probability
    ax4 = axes[1, 1]
    max_probs = target_probs.max(axis=1)
    ax4.hist(max_probs, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.set_xlabel("Maximum Probability", fontsize=12)
    ax4.set_ylabel("Frequency", fontsize=12)
    ax4.set_title("Distribution of Maximum Category Probability", fontsize=14, fontweight='bold')
    ax4.axvline(max_probs.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {max_probs.mean():.3f}")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved target probability distribution to {save_path}")


def plot_spatial_predictions(
    predictions: torch.Tensor,
    coordinates: np.ndarray,
    category_names: List[str],
    save_path: str,
    title: str = "Spatial Predictions",
) -> None:
    """Plot spatial distribution of model predictions.

    Args:
        predictions: Predicted probability vectors [N, 8] or class predictions [N].
        coordinates: Array of [longitude, latitude] coordinates [N, 2].
        category_names: List of 8 service category names.
        save_path: Path to save the plot.
        title: Plot title.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    # Convert probabilities to class predictions if needed
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        pred_classes = predictions.argmax(axis=1)
    else:
        pred_classes = predictions.flatten()

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Create color map for categories
    num_categories = len(category_names)
    colors = plt.cm.tab10(np.linspace(0, 1, num_categories))

    # Plot points colored by predicted category
    for i, category in enumerate(category_names):
        mask = pred_classes == i
        if mask.sum() > 0:
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                c=[colors[i]],
                label=category,
                alpha=0.6,
                s=20,
                edgecolors='black',
                linewidths=0.5,
            )

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved spatial predictions to {save_path}")


def plot_training_summary(
    history: List[Dict],
    plots_dir: str = "plots",
    experiment_name: str = "training",
) -> None:
    """Create comprehensive training summary plots.

    Args:
        history: List of dictionaries with training history.
        plots_dir: Directory to save plots.
        experiment_name: Name prefix for plot files.
    """
    plots_path = ensure_plots_dir(plots_dir)

    # Main training curves
    plot_training_curves(
        history,
        str(plots_path / f"{experiment_name}_training_curves.png"),
        title=f"{experiment_name.replace('_', ' ').title()} - Training Curves",
    )

    # Detailed loss comparison
    plot_loss_comparison(
        history,
        str(plots_path / f"{experiment_name}_loss_comparison.png"),
        title=f"{experiment_name.replace('_', ' ').title()} - Loss Comparison",
    )

    # Accuracy curves
    plot_accuracy_curves(
        history,
        str(plots_path / f"{experiment_name}_accuracy_curves.png"),
        title=f"{experiment_name.replace('_', ' ').title()} - Accuracy Curves",
    )

    logger.info(f"Saved all training summary plots to {plots_path}")
