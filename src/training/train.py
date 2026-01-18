"""Training script for Spatial Graph Transformer model.

This module provides training functionality including data loading, training loop,
validation, checkpointing, and early stopping for the graph transformer model.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from src.evaluation.metrics import evaluate_model
from src.evaluation.plotting import (
    plot_training_summary,
    plot_neighbor_distribution,
    plot_target_probability_distribution,
)
from src.evaluation.save_predictions import save_predictions, save_evaluation_results
from src.training.dataset import (
    SpatialGraphDataset,
    create_train_val_test_splits,
    get_device,
    load_features_from_directory,
)
from src.training.data_filtering import filter_small_neighborhoods
from src.training.loss import DistanceBasedKLLoss
from src.training.model import create_model_from_config, create_tiny_model_from_config
from src.utils.config import get_config
from src.utils.helpers import ensure_dir_exists, get_service_category_names, set_random_seeds
from src.utils.logging import get_logger, setup_experiment_logging

logger = get_logger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    checkpoint_path: Path,
    config: dict,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    training_history: Optional[List[Dict]] = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        best_val_loss: Best validation loss so far.
        checkpoint_path: Path to save checkpoint.
        config: Configuration dictionary.
        scheduler: Optional learning rate scheduler to save state.
        training_history: Optional training history to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
    }
    
    # Save scheduler state if provided
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Save training history if provided
    if training_history is not None:
        checkpoint["training_history"] = training_history

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional learning rate scheduler to load state into.

    Returns:
        Dictionary with checkpoint information (epoch, best_val_loss, training_history, etc.).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(epoch {checkpoint['epoch']}, val_loss {checkpoint['best_val_loss']:.4f})"
    )

    return checkpoint


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_mixed_precision: bool = False,
) -> float:
    """Train model for one epoch.

    Args:
        model: Model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer.
        loss_fn: Loss function.
        device: Device to train on.
        use_mixed_precision: Whether to use mixed precision training.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == "cuda" else None

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        if use_mixed_precision and scaler is not None:
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                logits = model(batch)
                loss = loss_fn(logits, batch.y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision
            logits = model(batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss


def train(
    config: Optional[dict] = None,
    features_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    quick_test: bool = False,
    use_tiny_model: bool = False,
) -> Dict:
    """Main training function.

    Args:
        config: Configuration dictionary. If None, loads from get_config().
        features_dir: Directory containing processed features. If None, uses config.
        checkpoint_dir: Directory to save checkpoints. If None, uses config.
        experiment_dir: Directory for experiment logs. If None, uses config.
        resume_from: Path to checkpoint to resume from. If None, starts fresh.

    Returns:
        Dictionary with training summary (best_val_loss, total_epochs, etc.).
    """
    # Load configuration
    if config is None:
        config = get_config()

    # Setup experiment logging
    if experiment_dir is None:
        experiment_dir = config.get("paths", {}).get("experiments_dir", "experiments/runs")
        experiment_dir = Path(experiment_dir) / f"run_{int(time.time())}"
    else:
        experiment_dir = Path(experiment_dir)

    logger_experiment = setup_experiment_logging(str(experiment_dir))
    logger_experiment.info("Starting training")

    # Set random seeds for reproducibility
    random_seed = config.get("data", {}).get("random_seed", 42)
    set_random_seeds(random_seed)

    # Get device
    device_config = config.get("training", {}).get("device", "auto")
    if device_config == "auto":
        device = get_device()
    else:
        device = torch.device(device_config)

    logger_experiment.info(f"Using device: {device}")

    # Load features
    if features_dir is None:
        features_dir = config.get("paths", {}).get("data_root", "./data")
        features_dir = str(Path(features_dir) / "processed" / "features" / "compliant")

    logger_experiment.info(f"Loading features from {features_dir}")
    features_df = load_features_from_directory(features_dir)
    
    # Quick test mode: filter to small neighborhoods
    if quick_test:
        logger_experiment.info("Quick test mode: filtering to small neighborhoods (<50 points, max 3 neighborhoods)")
        features_df = filter_small_neighborhoods(
            features_df,
            max_points_per_neighborhood=50,
            max_neighborhoods=3,
        )
        logger_experiment.info(f"Quick test: using {len(features_df)} points from {features_df['neighborhood_name'].nunique()} neighborhoods")

    # Create train/val/test splits
    train_ratio = config.get("data", {}).get("train_split", 0.7)
    val_ratio = config.get("data", {}).get("val_split", 0.15)
    test_ratio = config.get("data", {}).get("test_split", 0.15)

    train_df, val_df, test_df = create_train_val_test_splits(
        features_df, train_ratio, val_ratio, test_ratio, random_seed
    )

    # Create datasets
    train_dataset = SpatialGraphDataset(train_df)
    val_dataset = SpatialGraphDataset(val_df)
    test_dataset = SpatialGraphDataset(test_df)

    # Create data loaders
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 16)
    num_workers = training_config.get("num_workers", 2)
    pin_memory = training_config.get("pin_memory", False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    logger_experiment.info(
        f"Created datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    # Create model
    if use_tiny_model:
        logger_experiment.info("Using TinySpatialGraphTransformer for debugging")
        model = create_tiny_model_from_config(config)
    else:
        model = create_model_from_config(config)
    model = model.to(device)

    # Create loss function
    loss_fn = DistanceBasedKLLoss(reduction="batchmean")

    # Create optimizer
    learning_rate = training_config.get("learning_rate", 0.001)
    weight_decay = training_config.get("weight_decay", 0.0001)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training parameters
    num_epochs = training_config.get("num_epochs", 100)
    early_stopping_patience = training_config.get("early_stopping_patience", 10)
    early_stopping_min_delta = training_config.get("early_stopping_min_delta", 0.001)
    use_mixed_precision = training_config.get("use_mixed_precision", False)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    training_history = []
    if resume_from is not None:
        checkpoint = load_checkpoint(Path(resume_from), model, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        # Load training history if available
        if "training_history" in checkpoint:
            training_history = checkpoint["training_history"]
            logger_experiment.info(f"Loaded {len(training_history)} epochs of training history")
        logger_experiment.info(f"Resuming from epoch {start_epoch}")

    # Training loop
    patience_counter = 0
    # training_history initialized above if resuming, otherwise empty list

    logger_experiment.info("Starting training loop")
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Training phase
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, use_mixed_precision
        )

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Log metrics
        epoch_time = time.time() - epoch_start_time
        logger_experiment.info(
            f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s): "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_kl={val_metrics['kl_divergence']:.4f}, "
            f"val_top1={val_metrics['top1_accuracy']:.4f}"
        )

        # Evaluate on test set every epoch (for tracking)
        test_metrics_epoch = evaluate_model(model, test_loader, loss_fn, device)
        
        # Save training history
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_kl_divergence": val_metrics["kl_divergence"],
            "val_top1_accuracy": val_metrics["top1_accuracy"],
            "val_top3_accuracy": val_metrics["top3_accuracy"],
            "test_loss": test_metrics_epoch["loss"],
            "test_kl_divergence": test_metrics_epoch["kl_divergence"],
            "test_top1_accuracy": test_metrics_epoch["top1_accuracy"],
            "test_top3_accuracy": test_metrics_epoch["top3_accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        # Checkpointing
        if checkpoint_dir is None:
            checkpoint_dir = config.get("paths", {}).get("checkpoints_dir", "./models/checkpoints")
        checkpoint_path = Path(checkpoint_dir) / "graph_transformer_best.pt"

        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, checkpoint_path, config,
                scheduler=scheduler, training_history=training_history
            )
            logger_experiment.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint every 10 epochs (in addition to best model)
        if (epoch + 1) % 10 == 0:
            periodic_checkpoint_path = Path(checkpoint_dir) / f"graph_transformer_epoch_{epoch + 1}.pt"
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, periodic_checkpoint_path, config,
                scheduler=scheduler, training_history=training_history
            )
            logger_experiment.debug(f"Saved periodic checkpoint at epoch {epoch + 1}")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger_experiment.info(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(patience: {early_stopping_patience})"
            )
            break

    # Final evaluation on test set (detailed with predictions)
    logger_experiment.info("Evaluating on test set and saving predictions...")
    test_metrics = evaluate_model(model, test_loader, loss_fn, device)
    logger_experiment.info(
        f"Test metrics: loss={test_metrics['loss']:.4f}, "
        f"kl={test_metrics['kl_divergence']:.4f}, "
        f"top1={test_metrics['top1_accuracy']:.4f}, "
        f"top3={test_metrics['top3_accuracy']:.4f}"
    )
    
    # Save test set predictions
    try:
        test_predictions_path = experiment_dir / "test_predictions.csv"
        test_predictions_df = save_predictions(
            model, test_loader, device, test_predictions_path, split_name="test"
        )
        
        # Save evaluation results
        test_results_path = experiment_dir / "test_evaluation_results.json"
        save_evaluation_results(test_metrics, test_predictions_df, test_results_path, split_name="test")
        
        logger_experiment.info(f"Saved test predictions to {test_predictions_path}")
    except Exception as e:
        logger_experiment.warning(f"Failed to save test predictions: {e}")
        test_predictions_df = None
    
    # Save validation set predictions (for analysis)
    try:
        val_predictions_path = experiment_dir / "val_predictions.csv"
        val_predictions_df = save_predictions(
            model, val_loader, device, val_predictions_path, split_name="val"
        )
        logger_experiment.info(f"Saved validation predictions to {val_predictions_path}")
    except Exception as e:
        logger_experiment.warning(f"Failed to save validation predictions: {e}")

    # Save training history
    history_path = experiment_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    # Generate and save plots
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    logger_experiment.info("Generating training plots...")
    
    # Plot training curves
    plot_training_summary(
        training_history,
        plots_dir=str(plots_dir),
        experiment_name="training",
    )
    
    # Plot neighbor distribution
    try:
        plot_neighbor_distribution(
            features_df,
            save_path=str(plots_dir / "neighbor_distribution.png"),
            title="Neighbor Count Distribution",
        )
    except Exception as e:
        logger_experiment.warning(f"Failed to plot neighbor distribution: {e}")
    
    # Plot target probability distribution (sample from training data)
    try:
        sample_df = train_df.sample(min(100, len(train_df)), random_state=42)
        sample_dataset = SpatialGraphDataset(sample_df)
        # y is now [1, 8], so we need to squeeze it to [8] before stacking
        sample_target_probs = torch.stack([data.y.squeeze(0) for data in sample_dataset])
        category_names = get_service_category_names()
        
        plot_target_probability_distribution(
            sample_target_probs,
            category_names,
            save_path=str(plots_dir / "target_probability_distribution.png"),
            title="Target Probability Distribution (Training Sample)",
        )
    except Exception as e:
        logger_experiment.warning(f"Failed to plot target probability distribution: {e}")

    # Save final summary
    summary = {
        "best_val_loss": best_val_loss,
        "total_epochs": epoch + 1,
        "test_metrics": test_metrics,
        "experiment_dir": str(experiment_dir),
        "checkpoint_path": str(checkpoint_path),
        "plots_dir": str(plots_dir),
    }

    summary_path = experiment_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger_experiment.info("Training complete")
    logger_experiment.info(f"Best validation loss: {best_val_loss:.4f}")
    logger_experiment.info(f"Results saved to {experiment_dir}")
    logger_experiment.info(f"Plots saved to {plots_dir}")

    return summary


if __name__ == "__main__":
    # Run training with default configuration
    summary = train()
    print(f"Training complete. Best validation loss: {summary['best_val_loss']:.4f}")
