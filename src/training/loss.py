"""Distance-based KL divergence loss function for spatial graph transformer.

This module provides a loss function that compares predicted probability distributions
over 8 service categories to target probability vectors constructed from network-based
walking distances. The loss function uses KL divergence to measure distribution similarity,
enabling the model to learn service distribution patterns that align with 15-minute city
accessibility principles.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def distance_based_kl_loss(
    predicted_logits: torch.Tensor,
    target_probabilities: torch.Tensor,
    reduction: str = "batchmean",
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Compute KL divergence loss between predicted and target probability distributions.
    
    This function computes the KL divergence: KL(target || predicted), where:
    - predicted_logits: Model output logits (will be converted to log-probabilities)
    - target_probabilities: Pre-computed target probability vectors (already probabilities)
    
    The target probability vectors are constructed from network-based walking distances
    to nearest services in each category, using temperature-scaled softmax during
    feature engineering. They are stored in the dataset as data.y with shape [8].
    
    Args:
        predicted_logits: Model output logits of shape [batch_size, 8] or [8].
            For unbatched inputs, a batch dimension will be added internally.
        target_probabilities: Target probability vectors of shape [batch_size, 8] or [8].
            These are already probabilities (sum to 1.0), not logits.
        reduction: Reduction mode for loss computation:
            - 'batchmean' (default): Average loss per sample, normalized by batch size.
              Returns scalar tensor. Use for training.
            - 'none': Per-sample losses (not normalized). Returns tensor [batch_size].
              Use for validation/test analysis.
            - 'sum': Sum of all losses (not normalized). Returns scalar tensor.
            - 'mean': Average over all elements (batch × categories). Rarely used.
        epsilon: Small value for numerical stability. Target probabilities are clamped
            to [epsilon, 1.0] to avoid log(0). Defaults to 1e-9.
    
    Returns:
        Loss tensor:
        - Scalar tensor for 'batchmean', 'mean', or 'sum' reduction
        - Tensor of shape [batch_size] for 'none' reduction
    
    Raises:
        ValueError: If input shapes don't match or last dimension is not 8.
        TypeError: If inputs are not torch.Tensor.
    
    Example:
        >>> import torch
        >>> pred = torch.randn(2, 8)  # Batch of 2, 8 categories
        >>> target = torch.softmax(torch.randn(2, 8), dim=-1)  # Valid probabilities
        >>> loss = distance_based_kl_loss(pred, target)
        >>> print(f"Loss: {loss.item():.4f}")
        
        >>> # Per-sample analysis (validation mode)
        >>> loss_per_sample = distance_based_kl_loss(pred, target, reduction='none')
        >>> print(f"Per-sample losses: {loss_per_sample}")
        >>> print(f"Mean loss: {loss_per_sample.mean().item():.4f}")
    """
    # Validate inputs are tensors
    if not isinstance(predicted_logits, torch.Tensor):
        raise TypeError(f"predicted_logits must be torch.Tensor, got {type(predicted_logits)}")
    if not isinstance(target_probabilities, torch.Tensor):
        raise TypeError(f"target_probabilities must be torch.Tensor, got {type(target_probabilities)}")
    
    # Validate last dimension is 8 (number of service categories) first
    # This check happens before shape matching to provide clearer error messages
    if predicted_logits.shape[-1] != 8:
        raise ValueError(
            f"Last dimension must be 8 (service categories), got {predicted_logits.shape[-1]}"
        )
    if target_probabilities.shape[-1] != 8:
        raise ValueError(
            f"Last dimension must be 8 (service categories), got {target_probabilities.shape[-1]}"
        )
    
    # Handle unbatched inputs by adding batch dimension
    if predicted_logits.dim() == 1:
        predicted_logits = predicted_logits.unsqueeze(0)
        target_probabilities = target_probabilities.unsqueeze(0)
        was_unbatched = True
    else:
        was_unbatched = False
    
    # Validate shapes match (after potentially adding batch dimension)
    if predicted_logits.shape != target_probabilities.shape:
        raise ValueError(
            f"Shape mismatch: predicted_logits {predicted_logits.shape} != "
            f"target_probabilities {target_probabilities.shape}"
        )
    
    # Apply log-softmax to predictions (KL divergence requires log-probabilities)
    log_predicted = F.log_softmax(predicted_logits, dim=-1)
    
    # Clamp target probabilities for numerical stability
    # This prevents log(0) when computing KL divergence
    target_clamped = torch.clamp(target_probabilities, min=epsilon, max=1.0)
    
    # Renormalize after clamping to ensure sum = 1.0 (handle floating point errors)
    target_normalized = target_clamped / target_clamped.sum(dim=-1, keepdim=True)
    
    # Check for NaN/Inf values and log warnings
    if torch.isnan(target_normalized).any():
        logger.warning("Detected NaN values in target probabilities after normalization")
    if torch.isinf(target_normalized).any():
        logger.warning("Detected Inf values in target probabilities after normalization")
    
    # Check if clamping was necessary (if any values were at epsilon or 1.0)
    if (target_clamped == epsilon).any() or (target_clamped == 1.0).any():
        logger.debug("Applied epsilon clamping to target probabilities for numerical stability")
    
    # Compute KL divergence: KL(target || predicted)
    # F.kl_div expects: (log-probabilities, probabilities, reduction)
    # Note: With reduction='none', F.kl_div returns [batch_size, num_categories]
    # We need to sum over categories to get per-sample losses
    if reduction == "none":
        # Get per-category losses: [batch_size, 8]
        loss_per_category = F.kl_div(log_predicted, target_normalized, reduction="none")
        # Sum over categories to get per-sample losses: [batch_size]
        loss = loss_per_category.sum(dim=-1)
        # If input was unbatched, remove batch dimension
        if was_unbatched:
            loss = loss.squeeze(0)
    else:
        # For other reduction modes, use F.kl_div directly
        loss = F.kl_div(log_predicted, target_normalized, reduction=reduction)
    
    return loss


class DistanceBasedKLLoss(nn.Module):
    """PyTorch Module wrapper for distance-based KL divergence loss.
    
    This class provides a stateful loss function that can be used in training loops.
    It wraps the functional loss function and allows configuration of reduction mode
    and epsilon parameter.
    
    Args:
        reduction: Reduction mode for loss computation. Defaults to "batchmean".
        epsilon: Small value for numerical stability. Defaults to 1e-8.
    
    Example:
        >>> import torch
        >>> loss_fn = DistanceBasedKLLoss(reduction="batchmean")
        >>> pred = torch.randn(2, 8)
        >>> target = torch.softmax(torch.randn(2, 8), dim=-1)
        >>> loss = loss_fn(pred, target)
        >>> print(f"Loss: {loss.item():.4f}")
    """
    
    def __init__(self, reduction: str = "batchmean", epsilon: float = 1e-9):
        """Initialize loss function.
        
        Args:
            reduction: Reduction mode ('batchmean', 'none', 'sum', 'mean').
            epsilon: Numerical stability parameter. Defaults to 1e-9.
        """
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
    
    def forward(
        self, predicted_logits: torch.Tensor, target_probabilities: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for given predictions and targets.
        
        Args:
            predicted_logits: Model output logits [batch_size, 8] or [8].
            target_probabilities: Target probability vectors [batch_size, 8] or [8].
        
        Returns:
            Loss tensor (scalar or per-sample depending on reduction mode).
        """
        return distance_based_kl_loss(
            predicted_logits, target_probabilities, self.reduction, self.epsilon
        )


def get_loss_config() -> dict:
    """Load loss configuration from config.yaml.
    
    Returns:
        Dictionary containing loss configuration parameters.
        Note: Temperature parameter is used during feature engineering to compute
        target vectors, not in loss computation itself (targets are already probabilities).
    
    Example:
        >>> config = get_loss_config()
        >>> temperature = config.get("temperature", 200.0)
        >>> print(f"Temperature: {temperature}")
    """
    config = get_config()
    return config.get("loss", {})
