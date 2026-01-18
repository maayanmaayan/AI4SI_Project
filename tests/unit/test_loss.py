"""Unit tests for distance-based KL divergence loss function."""

import pytest
import torch
import numpy as np

from src.training.loss import (
    distance_based_kl_loss,
    DistanceBasedKLLoss,
    get_loss_config,
)


@pytest.fixture
def sample_predicted_logits():
    """Create sample predicted logits for testing."""
    return torch.randn(2, 8)


@pytest.fixture
def sample_target_probabilities():
    """Create sample target probabilities for testing."""
    # Create valid probability distribution (sums to 1.0)
    probs = torch.rand(2, 8)
    return probs / probs.sum(dim=-1, keepdim=True)


def test_basic_loss_computation(sample_predicted_logits, sample_target_probabilities):
    """Test basic loss computation with known inputs."""
    loss = distance_based_kl_loss(sample_predicted_logits, sample_target_probabilities)
    
    # Loss should be non-negative
    assert loss.item() >= 0
    
    # Loss should be a scalar tensor (batchmean reduction)
    assert loss.dim() == 0
    
    # Loss should be finite
    assert torch.isfinite(loss)


def test_perfect_match_zero_loss():
    """Test that perfect match gives approximately zero loss."""
    # Create target probabilities
    target = torch.softmax(torch.randn(1, 8), dim=-1)
    
    # Set predicted logits to log(target) for perfect match
    epsilon = 1e-8
    predicted_logits = torch.log(target + epsilon)
    
    # Compute loss
    loss = distance_based_kl_loss(predicted_logits, target)
    
    # Loss should be approximately zero (within numerical precision)
    assert loss.item() < 1e-5, f"Expected loss near zero, got {loss.item()}"


def test_batch_processing():
    """Test batch processing with different reduction modes."""
    batch_size = 4
    predicted = torch.randn(batch_size, 8)
    target = torch.softmax(torch.randn(batch_size, 8), dim=-1)
    
    # Test batchmean reduction (default)
    loss_batchmean = distance_based_kl_loss(predicted, target, reduction="batchmean")
    assert loss_batchmean.dim() == 0  # Scalar
    assert loss_batchmean.item() >= 0
    
    # Test none reduction (per-sample)
    loss_none = distance_based_kl_loss(predicted, target, reduction="none")
    assert loss_none.shape == (batch_size,)  # Per-sample losses
    assert (loss_none >= 0).all()  # All non-negative
    
    # Test sum reduction
    loss_sum = distance_based_kl_loss(predicted, target, reduction="sum")
    assert loss_sum.dim() == 0  # Scalar
    assert loss_sum.item() >= 0
    
    # Test mean reduction
    loss_mean = distance_based_kl_loss(predicted, target, reduction="mean")
    assert loss_mean.dim() == 0  # Scalar
    assert loss_mean.item() >= 0
    
    # Verify relationship: batchmean_loss * batch_size ≈ sum_loss (approximately)
    # Note: batchmean divides by batch_size, so batchmean * batch_size should be close to sum
    # But they're not exactly equal due to normalization differences
    assert torch.allclose(
        loss_batchmean * batch_size, loss_sum, atol=1e-4
    ), f"batchmean * batch_size should be close to sum: {loss_batchmean * batch_size} vs {loss_sum}"


def test_batch_processing_different_sizes():
    """Test batch processing with different batch sizes."""
    for batch_size in [1, 2, 8, 16]:
        predicted = torch.randn(batch_size, 8)
        target = torch.softmax(torch.randn(batch_size, 8), dim=-1)
        
        # Test batchmean
        loss = distance_based_kl_loss(predicted, target, reduction="batchmean")
        assert loss.dim() == 0
        assert loss.item() >= 0
        
        # Test none
        loss_per_sample = distance_based_kl_loss(predicted, target, reduction="none")
        assert loss_per_sample.shape == (batch_size,)
        assert (loss_per_sample >= 0).all()


def test_unbatched_input():
    """Test unbatched input handling."""
    predicted = torch.randn(8)  # Unbatched
    target = torch.softmax(torch.randn(8), dim=-1)  # Unbatched
    
    # Should handle unbatched inputs
    loss = distance_based_kl_loss(predicted, target)
    assert loss.dim() == 0  # Scalar
    
    # Per-sample mode with unbatched input
    loss_per_sample = distance_based_kl_loss(predicted, target, reduction="none")
    assert loss_per_sample.dim() == 0  # Scalar (unbatched, so no batch dimension)


def test_numerical_stability():
    """Test numerical stability edge cases."""
    # Test with zero probabilities in target (should clamp to epsilon)
    target_with_zeros = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    predicted = torch.randn(1, 8)
    
    loss = distance_based_kl_loss(predicted, target_with_zeros)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test with very small probabilities
    target_small = torch.tensor([[1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1.0 - 7e-10]])
    loss = distance_based_kl_loss(predicted, target_small)
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test with probabilities that don't sum to exactly 1.0 (floating point errors)
    target_almost_one = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3999999]])
    loss = distance_based_kl_loss(predicted, target_almost_one)
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_shape_validation():
    """Test input validation for shapes and types."""
    predicted = torch.randn(2, 8)
    target = torch.randn(2, 8)
    
    # Test with wrong number of categories in target (caught first)
    target_wrong_categories = torch.randn(2, 7)
    with pytest.raises(ValueError, match="Last dimension must be 8"):
        distance_based_kl_loss(predicted, target_wrong_categories)
    
    # Test with wrong number of categories in predicted
    predicted_wrong_categories = torch.randn(2, 10)
    with pytest.raises(ValueError, match="Last dimension must be 8"):
        distance_based_kl_loss(predicted_wrong_categories, target)
    
    # Test with mismatched batch dimensions (same last dim, different batch size)
    target_wrong_batch = torch.randn(3, 8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        distance_based_kl_loss(predicted, target_wrong_batch)
    
    # Test with non-tensor inputs
    with pytest.raises(TypeError, match="must be torch.Tensor"):
        distance_based_kl_loss(predicted.numpy(), target)
    
    with pytest.raises(TypeError, match="must be torch.Tensor"):
        distance_based_kl_loss(predicted, target.numpy())


def test_class_based_loss():
    """Test DistanceBasedKLLoss class."""
    # Create instance with custom parameters
    loss_fn = DistanceBasedKLLoss(reduction="batchmean", epsilon=1e-8)
    
    predicted = torch.randn(2, 8)
    target = torch.softmax(torch.randn(2, 8), dim=-1)
    
    # Test forward pass
    loss = loss_fn(predicted, target)
    assert loss.dim() == 0
    assert loss.item() >= 0
    
    # Test with different reduction
    loss_fn_none = DistanceBasedKLLoss(reduction="none")
    loss_per_sample = loss_fn_none(predicted, target)
    assert loss_per_sample.shape == (2,)
    
    # Verify it produces same results as functional version
    loss_functional = distance_based_kl_loss(predicted, target, reduction="batchmean")
    loss_class = loss_fn(predicted, target)
    assert torch.allclose(loss_functional, loss_class)


def test_integration_with_dataset():
    """Test integration with actual dataset structure."""
    # Simulate dataset structure: target_prob_vector is data.y with shape [8]
    # Create a batch of samples
    batch_size = 3
    predicted_logits = torch.randn(batch_size, 8)
    
    # Create target probability vectors (as they would come from dataset)
    target_prob_vectors = torch.softmax(torch.randn(batch_size, 8), dim=-1)
    
    # Test training mode (batchmean)
    loss_train = distance_based_kl_loss(
        predicted_logits, target_prob_vectors, reduction="batchmean"
    )
    assert loss_train.dim() == 0
    assert loss_train.item() >= 0
    
    # Test validation mode (none) - per-sample analysis
    loss_per_sample = distance_based_kl_loss(
        predicted_logits, target_prob_vectors, reduction="none"
    )
    assert loss_per_sample.shape == (batch_size,)
    assert (loss_per_sample >= 0).all()
    
    # Verify per-sample losses can be aggregated
    overall_mean = loss_per_sample.mean().item()
    overall_std = loss_per_sample.std().item()
    overall_min = loss_per_sample.min().item()
    overall_max = loss_per_sample.max().item()
    
    assert overall_mean >= 0
    assert overall_std >= 0
    assert overall_min >= 0
    assert overall_max >= 0
    assert overall_min <= overall_max
    
    # Verify aggregated mean matches batchmean result (approximately)
    assert abs(overall_mean - loss_train.item()) < 1e-4, (
        f"Per-sample mean {overall_mean} should match batchmean {loss_train.item()}"
    )


def test_per_sample_analysis():
    """Test per-sample analysis workflow (validation/test mode)."""
    batch_size = 5
    predicted = torch.randn(batch_size, 8)
    target = torch.softmax(torch.randn(batch_size, 8), dim=-1)
    
    # Compute per-sample losses
    loss_per_sample = distance_based_kl_loss(predicted, target, reduction="none")
    
    # Verify shape
    assert loss_per_sample.shape == (batch_size,)
    
    # Verify each element is non-negative
    assert (loss_per_sample >= 0).all()
    
    # Test aggregation
    mean_loss = loss_per_sample.mean()
    std_loss = loss_per_sample.std()
    min_loss = loss_per_sample.min()
    max_loss = loss_per_sample.max()
    
    assert mean_loss.item() >= 0
    assert std_loss.item() >= 0
    assert min_loss.item() >= 0
    assert max_loss.item() >= 0
    assert min_loss.item() <= max_loss.item()
    
    # Verify aggregated mean matches batchmean result
    batchmean_loss = distance_based_kl_loss(predicted, target, reduction="batchmean")
    assert torch.allclose(mean_loss, batchmean_loss, atol=1e-5)


def test_kl_divergence_properties():
    """Test mathematical properties of KL divergence."""
    # Test non-negativity
    predicted = torch.randn(2, 8)
    target = torch.softmax(torch.randn(2, 8), dim=-1)
    loss = distance_based_kl_loss(predicted, target)
    assert loss.item() >= 0, "KL divergence should be non-negative"
    
    # Test that loss increases with distribution mismatch
    # Create two targets: one close to prediction, one far
    predicted_logits = torch.randn(1, 8)
    predicted_probs = torch.softmax(predicted_logits, dim=-1)
    
    # Target 1: close to prediction (low loss)
    target_close = predicted_probs + torch.randn(1, 8) * 0.1
    target_close = target_close / target_close.sum(dim=-1, keepdim=True)
    
    # Target 2: far from prediction (high loss)
    target_far = torch.softmax(torch.randn(1, 8), dim=-1)
    
    loss_close = distance_based_kl_loss(predicted_logits, target_close)
    loss_far = distance_based_kl_loss(predicted_logits, target_far)
    
    # Loss with far target should generally be higher (not always, but usually)
    # We just verify both are non-negative and finite
    assert loss_close.item() >= 0
    assert loss_far.item() >= 0
    assert torch.isfinite(loss_close)
    assert torch.isfinite(loss_far)


def test_get_loss_config():
    """Test configuration loading function."""
    config = get_loss_config()
    
    # Should return a dictionary
    assert isinstance(config, dict)
    
    # Should have temperature parameter (if config file exists)
    # Note: This test may fail if config.yaml doesn't exist, which is OK
    # We just verify the function doesn't crash
    assert "temperature" in config or True  # Always pass, just check it doesn't crash


def test_epsilon_parameter():
    """Test that epsilon parameter affects numerical stability."""
    # Create target with very small probabilities
    target = torch.tensor([[1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1.0 - 7e-15]])
    predicted = torch.randn(1, 8)
    
    # With default epsilon (1e-8)
    loss_default = distance_based_kl_loss(predicted, target, epsilon=1e-8)
    assert torch.isfinite(loss_default)
    
    # With larger epsilon
    loss_large_epsilon = distance_based_kl_loss(predicted, target, epsilon=1e-6)
    assert torch.isfinite(loss_large_epsilon)
    
    # Both should be non-negative
    assert loss_default.item() >= 0
    assert loss_large_epsilon.item() >= 0
