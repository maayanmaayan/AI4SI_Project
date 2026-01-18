"""Evaluation metrics for spatial graph transformer model.

This module provides evaluation functions for the graph transformer model, including
KL divergence, top-k accuracy, distance alignment, and comprehensive model evaluation.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.training.loss import distance_based_kl_loss
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_kl_divergence(
    predicted_probs: torch.Tensor, target_probs: torch.Tensor
) -> float:
    """Compute KL divergence between predicted and target probability distributions.

    Args:
        predicted_probs: Predicted probability distributions [batch_size, 8].
        target_probs: Target probability distributions [batch_size, 8].

    Returns:
        Average KL divergence (scalar float).

    Example:
        >>> pred = torch.softmax(torch.randn(10, 8), dim=-1)
        >>> target = torch.softmax(torch.randn(10, 8), dim=-1)
        >>> kl = compute_kl_divergence(pred, target)
        >>> print(f"KL divergence: {kl:.4f}")
    """
    # Ensure inputs are probabilities (sum to 1.0)
    if predicted_probs.dim() == 1:
        predicted_probs = predicted_probs.unsqueeze(0)
        target_probs = target_probs.unsqueeze(0)

    # Compute KL divergence using log-probabilities
    log_pred = torch.log(predicted_probs + 1e-8)  # Add epsilon for numerical stability
    kl_per_sample = (target_probs * (torch.log(target_probs + 1e-8) - log_pred)).sum(
        dim=-1
    )

    return kl_per_sample.mean().item()


def compute_top_k_accuracy(
    predicted_probs: torch.Tensor, target_probs: torch.Tensor, k: int = 1
) -> float:
    """Compute top-k accuracy by comparing predicted and target top-k categories.

    Args:
        predicted_probs: Predicted probability distributions [batch_size, 8].
        target_probs: Target probability distributions [batch_size, 8].
        k: Number of top categories to consider (default: 1).

    Returns:
        Top-k accuracy (scalar float between 0 and 1).

    Example:
        >>> pred = torch.softmax(torch.randn(10, 8), dim=-1)
        >>> target = torch.softmax(torch.randn(10, 8), dim=-1)
        >>> acc = compute_top_k_accuracy(pred, target, k=1)
        >>> print(f"Top-1 accuracy: {acc:.4f}")
    """
    if predicted_probs.dim() == 1:
        predicted_probs = predicted_probs.unsqueeze(0)
        target_probs = target_probs.unsqueeze(0)

    # Get top-k predicted categories
    _, pred_top_k = torch.topk(predicted_probs, k, dim=-1)

    # Get top-k target categories
    _, target_top_k = torch.topk(target_probs, k, dim=-1)

    # Check if any predicted top-k matches any target top-k
    matches = []
    for i in range(predicted_probs.shape[0]):
        pred_set = set(pred_top_k[i].cpu().tolist())
        target_set = set(target_top_k[i].cpu().tolist())
        matches.append(len(pred_set & target_set) > 0)

    return sum(matches) / len(matches) if matches else 0.0


def compute_distance_alignment(
    predicted_probs: torch.Tensor,
    actual_distances: torch.Tensor,
    temperature: float = 200.0,
) -> float:
    """Measure how well predicted probabilities align with actual distances.

    Converts actual distances to probability distribution using temperature-scaled
    softmax, then computes KL divergence with predicted probabilities.

    Args:
        predicted_probs: Predicted probability distributions [batch_size, 8].
        actual_distances: Actual distances to nearest services [batch_size, 8] in meters.
        temperature: Temperature parameter for distance-to-probability conversion.

    Returns:
        Average KL divergence between predicted and distance-based probabilities.

    Example:
        >>> pred = torch.softmax(torch.randn(10, 8), dim=-1)
        >>> distances = torch.rand(10, 8) * 2000  # 0-2000m
        >>> alignment = compute_distance_alignment(pred, distances, temperature=200.0)
        >>> print(f"Distance alignment: {alignment:.4f}")
    """
    if predicted_probs.dim() == 1:
        predicted_probs = predicted_probs.unsqueeze(0)
        actual_distances = actual_distances.unsqueeze(0)

    # Convert distances to probabilities using temperature-scaled softmax
    distance_probs = torch.softmax(-actual_distances / temperature, dim=-1)

    # Compute KL divergence
    return compute_kl_divergence(predicted_probs, distance_probs)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Comprehensive evaluation of model on a dataset.

    Computes multiple metrics including loss, KL divergence, top-k accuracy,
    and distance alignment.

    Args:
        model: Trained model to evaluate.
        dataloader: DataLoader for evaluation dataset.
        loss_fn: Loss function (should be DistanceBasedKLLoss).
        device: Device to run evaluation on.

    Returns:
        Dictionary containing evaluation metrics:
        - loss: Average loss value
        - kl_divergence: Average KL divergence
        - top1_accuracy: Top-1 accuracy
        - top3_accuracy: Top-3 accuracy
        - num_samples: Number of samples evaluated

    Example:
        >>> from torch_geometric.loader import DataLoader
        >>> from src.training.loss import DistanceBasedKLLoss
        >>> model.eval()
        >>> metrics = evaluate_model(model, val_loader, loss_fn, device)
        >>> print(f"Validation loss: {metrics['loss']:.4f}")
    """
    model.eval()

    total_loss = 0.0
    total_kl = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = batch.to(device)

            # Forward pass
            logits = model(batch)

            # Compute loss
            loss = loss_fn(logits, batch.y)
            total_loss += loss.item() * batch.num_graphs

            # Convert logits to probabilities
            predicted_probs = torch.softmax(logits, dim=-1)

            # Compute KL divergence
            kl = compute_kl_divergence(predicted_probs, batch.y)
            total_kl += kl * batch.num_graphs

            # Compute top-k accuracies
            top1 = compute_top_k_accuracy(predicted_probs, batch.y, k=1)
            top3 = compute_top_k_accuracy(predicted_probs, batch.y, k=3)
            total_top1 += top1 * batch.num_graphs
            total_top3 += top3 * batch.num_graphs

            num_samples += batch.num_graphs

    # Average metrics
    metrics = {
        "loss": total_loss / num_samples if num_samples > 0 else 0.0,
        "kl_divergence": total_kl / num_samples if num_samples > 0 else 0.0,
        "top1_accuracy": total_top1 / num_samples if num_samples > 0 else 0.0,
        "top3_accuracy": total_top3 / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

    return metrics
