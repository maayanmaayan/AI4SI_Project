"""Training script for Spatial Graph Transformer model.

This module provides training functionality including data loading, training loop,
validation, checkpointing, and early stopping for the graph transformer model.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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
from src.training.data_augmentation import create_augmentation_transform
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


def compute_class_distribution(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 8,
) -> List[float]:
    """Compute class distribution from model predictions by averaging probability vectors.
    
    This computes the mean probability distribution across all samples, which better
    captures the model's confidence distribution and helps detect mode collapse.
    
    Args:
        model: Model to evaluate.
        dataloader: DataLoader for computing distribution.
        device: Device to run on.
        num_classes: Number of classes (default: 8).
    
    Returns:
        List of probabilities [prob_class_0, prob_class_1, ..., prob_class_7] where
        each value is the average probability mass assigned to that class across all samples.
    """
    model.eval()
    total_prob_sum = torch.zeros(num_classes, dtype=torch.float32)
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch)
            predicted_probs = torch.softmax(logits, dim=-1)  # [batch_size, 8]
            
            # Sum probabilities across all samples
            total_prob_sum += predicted_probs.sum(dim=0).cpu()
            total_samples += batch.num_graphs
    
    # Average probabilities (mean across all samples)
    if total_samples > 0:
        distribution = (total_prob_sum / total_samples).tolist()
    else:
        distribution = [0.0] * num_classes
    
    return distribution


def compute_true_class_distribution(
    dataset: SpatialGraphDataset,
    num_classes: int = 8,
) -> List[float]:
    """Compute true class distribution by averaging target probability vectors.
    
    This computes the mean probability distribution across all samples in the dataset,
    which represents the true expected distribution over classes.
    
    Args:
        dataset: Dataset to compute distribution from.
        num_classes: Number of classes (default: 8).
    
    Returns:
        List of probabilities [prob_class_0, prob_class_1, ..., prob_class_7] where
        each value is the average probability mass assigned to that class across all samples.
    """
    total_prob_sum = torch.zeros(num_classes, dtype=torch.float32)
    total_samples = len(dataset)
    
    for idx in range(total_samples):
        data = dataset[idx]
        target_prob_vector = data.y.squeeze(0)  # [8]
        # Sum probability vectors across all samples
        total_prob_sum += target_prob_vector
    
    # Average probabilities (mean across all samples)
    if total_samples > 0:
        distribution = (total_prob_sum / total_samples).tolist()
    else:
        distribution = [0.0] * num_classes
    
    return distribution


def update_prediction_distribution_json(
    json_path: Path,
    true_distribution: List[float],
    epoch: int,
    predicted_distribution: List[float],
) -> None:
    """Update or create JSON file with prediction distributions.
    
    Args:
        json_path: Path to JSON file.
        true_distribution: True class distribution (list of 8 probabilities).
        epoch: Current epoch number.
        predicted_distribution: Predicted class distribution (list of 8 probabilities).
    """
    # Load existing data if file exists
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {"true_distribution": true_distribution, "predictions": []}
    else:
        data = {"true_distribution": true_distribution, "predictions": []}
    
    # Ensure true_distribution is set (in case file was created differently)
    data["true_distribution"] = true_distribution
    
    # Add new prediction entry
    prediction_entry = {
        "epoch": epoch,
        "distribution": predicted_distribution,
    }
    
    # Check if epoch already exists, replace it; otherwise append
    predictions = data.get("predictions", [])
    existing_idx = None
    for i, pred in enumerate(predictions):
        if pred.get("epoch") == epoch:
            existing_idx = i
            break
    
    if existing_idx is not None:
        predictions[existing_idx] = prediction_entry
    else:
        predictions.append(prediction_entry)
    
    data["predictions"] = predictions
    
    # Sort predictions by epoch
    data["predictions"].sort(key=lambda x: x["epoch"])
    
    # Write updated data
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_mixed_precision: bool = False,
) -> Dict[str, float]:
    """Train model for one epoch.

    Args:
        model: Model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer.
        loss_fn: Loss function.
        device: Device to train on.
        use_mixed_precision: Whether to use mixed precision training.

    Returns:
        Dictionary with average training metrics: loss, top1_accuracy, top3_accuracy.
    """
    from src.evaluation.metrics import compute_top_k_accuracy
    
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
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
        else:
            # Standard precision
            logits = model(batch)
            loss = loss_fn(logits, batch.y)

        # Compute accuracies
        predicted_probs = torch.softmax(logits, dim=-1)
        top1 = compute_top_k_accuracy(predicted_probs, batch.y, k=1)
        top3 = compute_top_k_accuracy(predicted_probs, batch.y, k=3)

        if use_mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_top1 += top1 * batch.num_graphs
        total_top3 += top3 * batch.num_graphs
        num_samples += batch.num_graphs

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_top1 = total_top1 / num_samples if num_samples > 0 else 0.0
    avg_top3 = total_top3 / num_samples if num_samples > 0 else 0.0
    
    return {
        "loss": avg_loss,
        "top1_accuracy": avg_top1,
        "top3_accuracy": avg_top3,
    }


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

    # Calculate class weights from full training dataset
    logger_experiment.info("Calculating class weights from training dataset...")
    num_classes = 8
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    
    # Count instances of each class (dominant class per sample)
    for idx in range(len(train_df)):
        target_prob_vector = np.asarray(train_df.iloc[idx]["target_prob_vector"])
        dominant_class = int(np.argmax(target_prob_vector))
        class_counts[dominant_class] += 1
    
    total_samples = len(train_df)
    # Calculate weights: Weight_i = Total Samples / Samples in Class i
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_weights[i] = float(total_samples) / float(class_counts[i])
        else:
            # If class has no samples, use weight of 1.0 (shouldn't happen)
            class_weights[i] = 1.0
    
    logger_experiment.info(f"Class counts: {class_counts.tolist()}")
    logger_experiment.info(f"Class weights (before damping): {class_weights.tolist()}")
    
    # Dampen class weights using sqrt to prevent over-correction
    class_weights = torch.sqrt(class_weights)
    logger_experiment.info(f"Class weights (after sqrt damping): {class_weights.tolist()}")
    
    # Move class weights to device
    class_weights = class_weights.to(device)

    # Create datasets
    # Apply realistic data augmentation only to the training set.
    # Validation and test sets remain untouched for clean evaluation.
    aug_config = training_config.get("augmentation", {})
    aug_enabled = aug_config.get("enabled", True)
    if aug_enabled:
        train_transform = create_augmentation_transform(
            enable_feature_noise=True,
            enable_neighbor_subsampling=True,
            enable_edge_noise=True,
            feature_noise_std=aug_config.get("feature_noise_std", 0.01),
            subsample_ratio=aug_config.get("subsample_ratio", 0.8),
            distance_noise_std=aug_config.get("distance_noise_std", 0.02),
        )
    else:
        train_transform = None
    train_dataset = SpatialGraphDataset(train_df, transform=train_transform)
    val_dataset = SpatialGraphDataset(val_df)
    test_dataset = SpatialGraphDataset(test_df)
    
    # Calculate true class distribution from training dataset (once at start)
    logger_experiment.info("Computing true class distribution from training dataset...")
    true_distribution = compute_true_class_distribution(train_dataset, num_classes=8)
    logger_experiment.info(f"True distribution: {[f'{p:.4f}' for p in true_distribution]}")
    
    # Set up prediction distribution JSON file path
    prediction_dist_json_path = experiment_dir / "prediction_distributions.json"

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

    # Create loss function with label smoothing (0.1) and class weights
    loss_fn = DistanceBasedKLLoss(
        reduction="batchmean",
        label_smoothing=0.1,
        class_weights=class_weights,
    )

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
    
    # Initialize checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = config.get("paths", {}).get("checkpoints_dir", "./models/checkpoints")
    checkpoint_dir = Path(checkpoint_dir)

    # Initialize prediction distribution JSON file with true distribution
    prediction_dist_json_path.parent.mkdir(parents=True, exist_ok=True)
    initial_data = {
        "true_distribution": true_distribution,
        "predictions": []
    }
    with open(prediction_dist_json_path, "w") as f:
        json.dump(initial_data, f, indent=2)
    logger_experiment.info(f"Initialized prediction distribution file: {prediction_dist_json_path}")
    
    logger_experiment.info("Starting training loop")
    # Track last validation metrics for epochs without validation
    last_val_metrics = None
    last_val_loss = float("inf")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Training phase
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, use_mixed_precision
        )
        train_loss = train_metrics["loss"]
        train_top1 = train_metrics["top1_accuracy"]
        train_top3 = train_metrics["top3_accuracy"]

        # Validation phase: only every 5 epochs or on epoch 1 (for immediate feedback)
        should_validate = (epoch + 1) % 5 == 0 or epoch == 0
        
        if should_validate:
            val_metrics = evaluate_model(model, val_loader, loss_fn, device)
            val_loss = val_metrics["loss"]
            last_val_metrics = val_metrics
            last_val_loss = val_loss
            
            # Learning rate scheduling (only on validation epochs)
            scheduler.step(val_loss)
            
            # Log metrics with validation
            epoch_time = time.time() - epoch_start_time
            logger_experiment.info(
                f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s): "
                f"train_loss={train_loss:.4f}, train_top1={train_top1:.4f}, train_top3={train_top3:.4f}, "
                f"val_loss={val_loss:.4f}, val_kl={val_metrics['kl_divergence']:.4f}, "
                f"val_top1={val_metrics['top1_accuracy']:.4f}, val_top3={val_metrics['top3_accuracy']:.4f}"
            )
            
            # Evaluate on test set on validation epochs
            test_metrics_epoch = evaluate_model(model, test_loader, loss_fn, device)
        else:
            # Use last validation metrics for logging
            val_metrics = last_val_metrics
            val_loss = last_val_loss
            
            # Log metrics without validation (use last known values)
            epoch_time = time.time() - epoch_start_time
            logger_experiment.info(
                f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s): "
                f"train_loss={train_loss:.4f}, train_top1={train_top1:.4f}, train_top3={train_top3:.4f}, "
                f"val_loss={val_loss:.4f} (last), val_kl=N/A, val_top1=N/A, val_top3=N/A"
            )
            
            # Skip test evaluation on non-validation epochs
            test_metrics_epoch = None
        
        # Save training history
        history_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_top1_accuracy": train_top1,
            "train_top3_accuracy": train_top3,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        
        # Add validation metrics if available
        if val_metrics is not None:
            history_entry.update({
                "val_kl_divergence": val_metrics["kl_divergence"],
                "val_top1_accuracy": val_metrics["top1_accuracy"],
                "val_top3_accuracy": val_metrics["top3_accuracy"],
            })
        else:
            history_entry.update({
                "val_kl_divergence": None,
                "val_top1_accuracy": None,
                "val_top3_accuracy": None,
            })
        
        # Add test metrics if available (only on validation epochs)
        if test_metrics_epoch is not None:
            history_entry.update({
                "test_loss": test_metrics_epoch["loss"],
                "test_kl_divergence": test_metrics_epoch["kl_divergence"],
                "test_top1_accuracy": test_metrics_epoch["top1_accuracy"],
                "test_top3_accuracy": test_metrics_epoch["top3_accuracy"],
            })
        else:
            history_entry.update({
                "test_loss": None,
                "test_kl_divergence": None,
                "test_top1_accuracy": None,
                "test_top3_accuracy": None,
            })
        
        training_history.append(history_entry)

        # Update prediction distribution JSON every 5 epochs (same as validation)
        if should_validate:
            logger_experiment.info(f"Computing prediction distribution for epoch {epoch + 1}...")
            predicted_distribution = compute_class_distribution(
                model, val_loader, device, num_classes=8
            )
            update_prediction_distribution_json(
                prediction_dist_json_path,
                true_distribution,
                epoch + 1,
                predicted_distribution,
            )
            logger_experiment.info(
                f"Epoch {epoch + 1} prediction distribution: {[f'{p:.4f}' for p in predicted_distribution]}"
            )
            logger_experiment.info(f"Updated prediction distributions to {prediction_dist_json_path}")

        # Checkpointing (only on validation epochs)
        if should_validate:
            checkpoint_path = checkpoint_dir / "graph_transformer_best.pt"

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
            periodic_checkpoint_path = checkpoint_dir / f"graph_transformer_epoch_{epoch + 1}.pt"
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
