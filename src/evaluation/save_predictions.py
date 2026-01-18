"""Utilities for saving model predictions and evaluation results.

This module provides functions to save predictions, actual targets, and per-sample
evaluation metrics for detailed analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Path,
    split_name: str = "test",
) -> pd.DataFrame:
    """Save model predictions along with targets and metadata.

    Args:
        model: Trained model to generate predictions.
        dataloader: DataLoader for the dataset.
        device: Device to run inference on.
        save_path: Path to save predictions CSV file.
        split_name: Name of the split (train/val/test).

    Returns:
        DataFrame with columns:
        - target_id: Target point identifier
        - neighborhood_name: Neighborhood name
        - predicted_class: Predicted service category (argmax)
        - predicted_probs: Predicted probabilities [8] (as list)
        - target_probs: Target probabilities [8] (as list)
        - target_class: Target service category (argmax)
        - loss: Per-sample loss
        - kl_divergence: Per-sample KL divergence
    """
    model.eval()

    results = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # Get predictions
            logits = model(batch)
            predicted_probs = torch.softmax(logits, dim=-1)
            predicted_classes = predicted_probs.argmax(dim=-1)

            # Get targets
            target_probs = batch.y
            target_classes = target_probs.argmax(dim=-1)

            # Compute per-sample losses
            from src.training.loss import distance_based_kl_loss

            # Compute loss for each sample
            per_sample_losses = []
            per_sample_kl = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i:i+1]
                sample_target = target_probs[i:i+1]
                loss = distance_based_kl_loss(sample_logits, sample_target, reduction="none")
                per_sample_losses.append(loss.item())
                
                # Compute KL divergence
                log_pred = torch.log_softmax(sample_logits, dim=-1)
                kl = (target_probs[i] * (torch.log(target_probs[i] + 1e-8) - log_pred[0])).sum().item()
                per_sample_kl.append(kl)

            # Extract metadata from batched graphs
            batch_size = batch.num_graphs
            
            # PyTorch Geometric stores custom attributes as lists when batching
            # Extract target_id and neighborhood_name
            target_ids = []
            neighborhood_names = []
            
            if hasattr(batch, 'target_id'):
                # In batched Data, custom attributes become lists
                if isinstance(batch.target_id, list):
                    target_ids = [str(tid) for tid in batch.target_id]
                else:
                    # Single value or tensor - convert to list
                    try:
                        if isinstance(batch.target_id, torch.Tensor):
                            target_ids = [str(tid.item()) for tid in batch.target_id]
                        else:
                            target_ids = [str(batch.target_id)] * batch_size
                    except:
                        target_ids = [f"{split_name}_{i}" for i in range(batch_size)]
            else:
                target_ids = [f"{split_name}_{i}" for i in range(batch_size)]
            
            if hasattr(batch, 'neighborhood_name'):
                if isinstance(batch.neighborhood_name, list):
                    neighborhood_names = [str(n) for n in batch.neighborhood_name]
                else:
                    try:
                        if isinstance(batch.neighborhood_name, torch.Tensor):
                            neighborhood_names = [str(n.item()) for n in batch.neighborhood_name]
                        else:
                            neighborhood_names = [str(batch.neighborhood_name)] * batch_size
                    except:
                        neighborhood_names = ["unknown"] * batch_size
            else:
                neighborhood_names = ["unknown"] * batch_size
            
            # Ensure we have the right number of IDs
            while len(target_ids) < batch_size:
                target_ids.append(f"{split_name}_{len(target_ids)}")
            while len(neighborhood_names) < batch_size:
                neighborhood_names.append("unknown")
            
            for i in range(batch_size):
                target_id = target_ids[i]
                neighborhood_name = neighborhood_names[i]

                results.append({
                    "target_id": target_id,
                    "neighborhood_name": neighborhood_name,
                    "predicted_class": predicted_classes[i].item(),
                    "predicted_probs": predicted_probs[i].cpu().numpy().tolist(),
                    "target_class": target_classes[i].item(),
                    "target_probs": target_probs[i].cpu().numpy().tolist(),
                    "loss": per_sample_losses[i],
                    "kl_divergence": per_sample_kl[i],
                })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    logger.info(f"Saved {len(df)} predictions to {save_path}")

    return df


def save_evaluation_results(
    metrics: Dict[str, float],
    predictions_df: Optional[pd.DataFrame],
    save_path: Path,
    split_name: str = "test",
) -> None:
    """Save evaluation results (metrics and optionally predictions).

    Args:
        metrics: Dictionary of evaluation metrics.
        predictions_df: Optional DataFrame with per-sample predictions.
        save_path: Path to save results JSON file.
        split_name: Name of the split (train/val/test).
    """
    results = {
        "split": split_name,
        "metrics": metrics,
        "num_samples": len(predictions_df) if predictions_df is not None else None,
    }

    # Add per-sample statistics if predictions available
    if predictions_df is not None:
        results["per_sample_stats"] = {
            "mean_loss": float(predictions_df["loss"].mean()),
            "std_loss": float(predictions_df["loss"].std()),
            "mean_kl_divergence": float(predictions_df["kl_divergence"].mean()),
            "std_kl_divergence": float(predictions_df["kl_divergence"].std()),
            "accuracy": float((predictions_df["predicted_class"] == predictions_df["target_class"]).mean()),
        }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved evaluation results to {save_path}")
