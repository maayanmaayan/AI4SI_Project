# Feature: Distance-Based KL Divergence Loss Function

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Implement a distance-based KL divergence loss function for training the Spatial Graph Transformer model. The loss function compares the model's predicted probability distributions over 8 service categories to target probability vectors constructed from network-based walking distances to nearest services. This loss function is critical for training the model to learn service distribution patterns that align with 15-minute city accessibility principles.

**Core Functionality:**
- KL divergence loss between predicted and target probability vectors
- Support for batch processing with PyTorch Geometric DataLoader
- Configuration-driven temperature parameter for distance-to-probability conversion
- Proper handling of edge cases (zero probabilities, numerical stability)
- Integration with existing dataset structure (target vectors already computed)

## User Story

As a **ML engineer/researcher**
I want to **implement a distance-based KL divergence loss function that compares predicted service category probabilities to distance-based target vectors**
So that **the model learns to predict service distributions that align with actual walking accessibility patterns in 15-minute city compliant neighborhoods**

## Problem Statement

The training pipeline requires a loss function that:
1. **Measures distribution similarity** - Compares predicted probability vectors (8 service categories) to target probability vectors constructed from network walking distances
2. **Handles pre-computed targets** - Target probability vectors are already computed during feature engineering and stored in the dataset (no need to recompute distances)
3. **Supports batch processing** - Must work efficiently with PyTorch Geometric DataLoader batching
4. **Maintains numerical stability** - Handles edge cases like zero probabilities, very small values, and log(0) scenarios
5. **Configurable parameters** - Temperature parameter and other settings should come from config.yaml

The loss function is a critical component that enables the model to learn from distance-based supervision, directly encoding 15-minute city accessibility principles into the training objective.

## Solution Statement

Implement a PyTorch loss function module that:
- Uses `torch.nn.KLDivLoss` with proper input formatting (log-probabilities for predictions, probabilities for targets)
- Accepts predicted logits from the model and target probability vectors from the dataset
- Applies log-softmax to predictions and uses targets directly (they're already probabilities)
- Handles batch processing with proper reduction ('batchmean')
- Includes numerical stability checks (clamping, epsilon values)
- Loads configuration from `models/config.yaml` (temperature parameter is for reference, not used in loss computation since targets are pre-computed)
- Provides both functional and class-based interfaces for flexibility

---

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: 
- `src/training/loss.py` (new file)
- `tests/unit/test_loss.py` (new test file)
- Future: `src/training/train.py` (will use this loss function)
**Dependencies**: 
- PyTorch 2.0+ (already in requirements)
- Existing dataset structure (target_prob_vector already computed)

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `docs/distance-based-loss-calculation.md` (all lines) - Why: Complete specification of loss function mathematics, formula, and implementation details
- `src/training/dataset.py` (lines 95-168) - Why: Shows how target_prob_vector is stored in Data objects as `data.y` with shape [8]
- `src/training/dataset.py` (lines 115-117) - Why: Target vectors are already torch tensors of shape [8], dtype float32
- `models/config.yaml` (lines 35-39) - Why: Loss configuration parameters (temperature, missing_service_penalty, type)
- `scripts/test_training_feasibility.py` (lines 166, 208-210) - Why: Example of using KLDivLoss with log_softmax pattern (but note: this test script softmaxes targets which is incorrect - targets are already probabilities)
- `src/data/collection/feature_engineer.py` (lines 1114-1179) - Why: Shows how target probability vectors are computed from distances using temperature-scaled softmax
- `src/utils/config.py` (all lines) - Why: Pattern for loading configuration with `get_config()`
- `src/utils/logging.py` (all lines) - Why: Logging pattern using `get_logger(__name__)`
- `tests/unit/test_dataset.py` (all lines) - Why: Test patterns and fixtures for PyTorch code
- `CURSOR.md` (lines 240-274) - Why: Code conventions, ML patterns, and architecture principles

### New Files to Create

- `src/training/loss.py` - Loss function implementation (functional and class-based)
- `tests/unit/test_loss.py` - Comprehensive unit tests for loss function

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [PyTorch KLDivLoss Documentation](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
  - Specific section: Input format (log-probabilities for input, probabilities for target)
  - Why: Must use log_softmax on predictions, targets should be probabilities (already computed)
- [PyTorch Functional KL Divergence](https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html)
  - Specific section: Reduction modes ('batchmean' for proper batch averaging)
  - Why: Understanding reduction options for batch processing
- [PyTorch Numerical Stability Guide](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)
  - Specific section: Log-softmax and KL divergence stability
  - Why: Handling edge cases and numerical precision issues

### Patterns to Follow

**Naming Conventions:**
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Use descriptive names: `distance_based_kl_loss()`, `DistanceBasedKLLoss`
- Follow existing pattern: loss functions in `src/training/loss.py`

**Error Handling:**
- Validate input shapes and types
- Use assertions for development-time checks
- Log warnings for edge cases (zero probabilities, NaN values)
- Pattern from `feature_engineer.py`: Log warnings with context

**Logging Pattern:**
```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.debug(f"Computing loss for batch of size {batch_size}")
logger.warning(f"Detected zero probabilities in target vector, applying epsilon")
```

**Configuration Pattern:**
```python
from src.utils.config import get_config

config = get_config()
loss_config = config.get("loss", {})
temperature = loss_config.get("temperature", 200.0)  # For reference/documentation
```

**Type Hints Pattern:**
```python
from typing import Optional
import torch

def distance_based_kl_loss(
    predicted_logits: torch.Tensor,
    target_probabilities: torch.Tensor,
    reduction: str = "batchmean",
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute KL divergence loss.
    
    Args:
        predicted_logits: Model output logits [batch_size, 8] or [8]
        target_probabilities: Target probability vectors [batch_size, 8] or [8]
        reduction: Reduction mode ('batchmean', 'sum', 'none')
        epsilon: Small value for numerical stability
    
    Returns:
        Scalar loss value (or tensor if reduction='none')
    """
```

**Test Pattern:**
```python
import pytest
import torch
import numpy as np

def test_loss_function():
    """Test loss function with known inputs."""
    predicted = torch.randn(2, 8)  # Batch of 2, 8 categories
    target = torch.softmax(torch.randn(2, 8), dim=-1)  # Valid probabilities
    loss = distance_based_kl_loss(predicted, target)
    assert loss.item() >= 0  # Loss should be non-negative
```

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation

Set up the loss function module structure with proper imports, configuration loading, and basic function signature.

**Tasks:**
- Create `src/training/loss.py` with module docstring
- Import required dependencies (torch, torch.nn, typing)
- Import configuration and logging utilities
- Define function signatures with type hints

### Phase 2: Core Implementation

Implement the KL divergence loss computation with proper numerical stability and batch handling.

**Tasks:**
- Implement functional loss function `distance_based_kl_loss()`
- Implement class-based wrapper `DistanceBasedKLLoss(nn.Module)`
- Add numerical stability checks (epsilon clamping, NaN handling)
- Validate input shapes and types
- Handle both batched and unbatched inputs

### Phase 3: Integration & Validation

Ensure the loss function integrates correctly with the dataset structure and provides proper error handling.

**Tasks:**
- Add input validation and error messages
- Add logging for debugging and monitoring
- Test with actual dataset structure (target_prob_vector format)
- Verify batch processing works correctly

### Phase 4: Testing & Documentation

Create comprehensive unit tests and ensure code quality.

**Tasks:**
- Write unit tests for all functions and edge cases
- Test numerical stability (zero probabilities, very small values)
- Test batch processing and shape handling
- Verify loss values are non-negative and reasonable
- Test integration with dataset structure

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### CREATE src/training/loss.py

- **IMPLEMENT**: Create new file with module docstring describing distance-based KL divergence loss
- **IMPORTS**: 
  - `torch`, `torch.nn`, `torch.nn.functional as F`
  - `typing: Optional`
  - `src.utils.config.get_config`
  - `src.utils.logging.get_logger`
- **PATTERN**: Follow module structure from `src/training/dataset.py` (lines 1-20)
- **VALIDATE**: File exists and imports work: `python -c "from src.training.loss import distance_based_kl_loss; print('OK')"`

### ADD distance_based_kl_loss() function

- **IMPLEMENT**: Functional loss function with signature:
  ```python
  def distance_based_kl_loss(
      predicted_logits: torch.Tensor,
      target_probabilities: torch.Tensor,
      reduction: str = "batchmean",
      epsilon: float = 1e-8,
  ) -> torch.Tensor:
  ```
- **LOGIC**:
  1. Validate input shapes (must match, last dimension must be 8)
  2. Handle both batched [batch_size, 8] and unbatched [8] inputs
  3. Apply log_softmax to predicted_logits: `F.log_softmax(predicted_logits, dim=-1)`
  4. Clamp target_probabilities to [epsilon, 1.0] for numerical stability
  5. Normalize target_probabilities to ensure they sum to 1.0 (handle floating point errors)
  6. Compute KL divergence: `F.kl_div(log_predicted, target_probabilities, reduction=reduction)`
  7. Return loss tensor (scalar for 'batchmean'/'mean'/'sum', tensor for 'none')
- **PATTERN**: Follow KL divergence pattern from `scripts/test_training_feasibility.py` (lines 208-210), but note: targets are already probabilities, don't softmax them
- **REDUCTION MODES**:
  - `'batchmean'` (default): Average loss per sample, normalized by batch size. Use for training.
    - Returns: scalar tensor
    - Formula: `(1/N) × Σ_i Σ_j P_target,ij × log(P_target,ij / P_predicted,ij)`
  - `'none'`: Per-sample losses (not normalized). Use for validation/test analysis.
    - Returns: tensor of shape `[batch_size]` with one loss per sample
    - Formula per sample i: `Σ_j P_target,ij × log(P_target,ij / P_predicted,ij)`
    - To get overall average: `loss.mean()` or `loss.sum() / len(loss)`
  - `'mean'`: Average over all elements (batch × categories). Rarely used.
  - `'sum'`: Sum of all losses (not normalized). Use if you want total loss.
- **GOTCHA**: 
  - `F.kl_div` expects log-probabilities for first arg, probabilities for second arg
  - Targets are already probabilities (computed in feature engineering), don't apply softmax
  - Use `reduction='batchmean'` for training (normalized by batch size)
  - Use `reduction='none'` for validation/test (per-sample analysis, then aggregate manually)
- **VALIDATE**: 
  - Training mode: `python -c "import torch; from src.training.loss import distance_based_kl_loss; pred=torch.randn(2,8); target=torch.softmax(torch.randn(2,8),-1); loss=distance_based_kl_loss(pred, target); print(f'Batch mean loss: {loss.item():.4f}')"`
  - Per-sample mode: `python -c "import torch; from src.training.loss import distance_based_kl_loss; pred=torch.randn(2,8); target=torch.softmax(torch.randn(2,8),-1); loss=distance_based_kl_loss(pred, target, reduction='none'); print(f'Per-sample losses: {loss}, Mean: {loss.mean().item():.4f}')"`

### ADD input validation and shape handling

- **IMPLEMENT**: Add validation at start of function:
  - Check predicted_logits and target_probabilities are torch.Tensor
  - Check shapes match (except for batch dimension)
  - Check last dimension is 8 (number of service categories)
  - Handle both [batch_size, 8] and [8] shapes (add batch dimension if needed)
- **ERROR HANDLING**: Raise ValueError with descriptive messages for invalid inputs
- **PATTERN**: Follow validation pattern from `src/training/dataset.py` (lines 74-85)
- **VALIDATE**: Test with invalid shapes: `python -c "import torch; from src.training.loss import distance_based_kl_loss; pred=torch.randn(2,7); target=torch.randn(2,8); distance_based_kl_loss(pred, target)"` should raise ValueError

### ADD numerical stability handling

- **IMPLEMENT**: Add epsilon clamping for target probabilities:
  - Clamp target_probabilities to [epsilon, 1.0] to avoid log(0)
  - Renormalize after clamping to ensure sum = 1.0
  - Check for NaN/Inf values and log warnings
- **LOGIC**: 
  ```python
  target_probabilities = torch.clamp(target_probabilities, min=epsilon, max=1.0)
  target_probabilities = target_probabilities / target_probabilities.sum(dim=-1, keepdim=True)
  ```
- **LOGGING**: Log warning if clamping was necessary or NaN detected
- **PATTERN**: Follow logging pattern from `src/utils/logging.py` (lines 74-92)
- **VALIDATE**: Test with zero probabilities: `python -c "import torch; from src.training.loss import distance_based_kl_loss; pred=torch.randn(1,8); target=torch.tensor([[0,0,0,0,0,0,0,1.0]]); loss=distance_based_kl_loss(pred, target); print(f'Loss with zeros: {loss.item():.4f}')"`

### ADD DistanceBasedKLLoss class

- **IMPLEMENT**: PyTorch Module wrapper class:
  ```python
  class DistanceBasedKLLoss(nn.Module):
      def __init__(self, reduction: str = "batchmean", epsilon: float = 1e-8):
          super().__init__()
          self.reduction = reduction
          self.epsilon = epsilon
      
      def forward(self, predicted_logits: torch.Tensor, target_probabilities: torch.Tensor) -> torch.Tensor:
          return distance_based_kl_loss(predicted_logits, target_probabilities, self.reduction, self.epsilon)
  ```
- **PATTERN**: Follow nn.Module pattern from PyTorch documentation
- **VALIDATE**: `python -c "import torch; from src.training.loss import DistanceBasedKLLoss; loss_fn=DistanceBasedKLLoss(); pred=torch.randn(2,8); target=torch.softmax(torch.randn(2,8),-1); loss=loss_fn(pred, target); print(f'Class-based loss: {loss.item():.4f}')"`

### ADD configuration loading (optional, for documentation)

- **IMPLEMENT**: Add function to load loss config (for reference, temperature is already used in feature engineering):
  ```python
  def get_loss_config() -> dict:
      """Load loss configuration from config.yaml."""
      config = get_config()
      return config.get("loss", {})
  ```
- **NOTE**: Temperature parameter is used during feature engineering to compute target vectors, not in loss computation itself
- **PATTERN**: Follow config loading pattern from `src/utils/config.py` (lines 72-90)
- **VALIDATE**: `python -c "from src.training.loss import get_loss_config; cfg=get_loss_config(); print(f'Temperature: {cfg.get(\"temperature\", \"not found\")}')"`

### CREATE tests/unit/test_loss.py

- **IMPLEMENT**: Create test file with module docstring
- **IMPORTS**: 
  - `pytest`, `torch`, `numpy as np`
  - `src.training.loss.distance_based_kl_loss`, `DistanceBasedKLLoss`
- **PATTERN**: Follow test structure from `tests/unit/test_dataset.py` (lines 1-20)
- **VALIDATE**: File exists: `python -c "import tests.unit.test_loss; print('OK')"`

### ADD test_basic_loss_computation

- **IMPLEMENT**: Test basic loss computation with known inputs:
  - Create predicted logits and target probabilities
  - Compute loss and verify it's non-negative
  - Verify loss is scalar (for batchmean reduction)
- **PATTERN**: Follow test pattern from `tests/unit/test_dataset.py` (lines 84-88)
- **VALIDATE**: `pytest tests/unit/test_loss.py::test_basic_loss_computation -v`

### ADD test_perfect_match_zero_loss

- **IMPLEMENT**: Test that perfect match gives zero loss:
  - Create target probabilities
  - Set predicted logits to log(target_probabilities) (perfect match)
  - Verify loss is approximately zero (within numerical precision)
- **LOGIC**: `predicted_logits = torch.log(target_probabilities + epsilon)`
- **VALIDATE**: `pytest tests/unit/test_loss.py::test_perfect_match_zero_loss -v`

### ADD test_batch_processing

- **IMPLEMENT**: Test batch processing with different reduction modes:
  - Create batch of predicted logits [batch_size, 8]
  - Create batch of target probabilities [batch_size, 8]
  - Test `reduction='batchmean'`: verify returns scalar, normalized by batch size
  - Test `reduction='none'`: verify returns tensor [batch_size] with per-sample losses
  - Test `reduction='sum'`: verify returns scalar, sum of all losses
  - Test with different batch sizes (1, 2, 16)
  - Verify: `batchmean_loss * batch_size ≈ sum_loss` (approximately, due to numerical precision)
- **PATTERN**: Follow batch testing pattern from `tests/unit/test_dataset.py` (lines 131-141)
- **VALIDATE**: `pytest tests/unit/test_loss.py::test_batch_processing -v`

### ADD test_unbatched_input

- **IMPLEMENT**: Test unbatched input handling:
  - Create single predicted logits [8]
  - Create single target probabilities [8]
  - Verify loss handles unbatched inputs (adds batch dimension internally)
- **VALIDATE**: `pytest tests/unit/test_loss.py::test_unbatched_input -v`

### ADD test_numerical_stability

- **IMPLEMENT**: Test numerical stability edge cases:
  - Test with zero probabilities in target (should clamp to epsilon)
  - Test with very small probabilities
  - Test with probabilities that don't sum to exactly 1.0 (floating point errors)
  - Verify no NaN or Inf values
- **VALIDATE**: `pytest tests/unit/test_loss.py::test_numerical_stability -v`

### ADD test_shape_validation

- **IMPLEMENT**: Test input validation:
  - Test with mismatched shapes (should raise ValueError)
  - Test with wrong number of categories (should raise ValueError)
  - Test with non-tensor inputs (should raise TypeError or ValueError)
- **PATTERN**: Follow validation testing pattern from `tests/unit/test_dataset.py` (lines 144-149)
- **VALIDATE**: `pytest tests/unit/test_loss.py::test_shape_validation -v`

### ADD test_class_based_loss

- **IMPLEMENT**: Test DistanceBasedKLLoss class:
  - Create instance with custom parameters
  - Test forward pass
  - Verify it produces same results as functional version
- **VALIDATE**: `pytest tests/unit/test_loss.py::test_class_based_loss -v`

### ADD test_integration_with_dataset

- **IMPLEMENT**: Test integration with actual dataset structure:
  - Load sample data from dataset
  - Extract target_prob_vector (data.y)
  - Create dummy predicted logits
  - Test both training mode (`reduction='batchmean'`) and validation mode (`reduction='none'`)
  - Verify per-sample losses can be aggregated correctly
- **PATTERN**: Follow integration test pattern from `tests/unit/test_dataset.py` (lines 91-115)
- **VALIDATE**: `pytest tests/unit/test_loss.py::test_integration_with_dataset -v`

### ADD test_per_sample_analysis

- **IMPLEMENT**: Test per-sample analysis workflow (validation/test mode):
  - Create batch of samples
  - Compute per-sample losses with `reduction='none'`
  - Verify shape is [batch_size]
  - Verify each element is non-negative
  - Test aggregation: `loss.mean()`, `loss.std()`, `loss.min()`, `loss.max()`
  - Verify aggregated mean matches `batchmean` result (within numerical precision)
- **VALIDATE**: `pytest tests/unit/test_loss.py::test_per_sample_analysis -v`

### ADD test_kl_divergence_properties

- **IMPLEMENT**: Test mathematical properties of KL divergence:
  - Test asymmetry (KL(P||Q) != KL(Q||P) in general)
  - Test non-negativity (loss >= 0)
  - Test that loss increases with distribution mismatch
- **VALIDATE**: `pytest tests/unit/test_loss.py::test_kl_divergence_properties -v`

---

## TESTING STRATEGY

### Unit Tests

**Scope**: Test all functions, edge cases, and mathematical properties

**Requirements**:
- Test basic loss computation with known inputs
- Test perfect match (should give zero loss)
- Test batch processing (different batch sizes)
- Test unbatched input handling
- Test numerical stability (zero probabilities, small values, normalization)
- Test input validation (shape mismatches, wrong dimensions)
- Test class-based wrapper
- Test integration with dataset structure
- Test KL divergence mathematical properties

**Fixtures**: Use pytest fixtures for common test data (predicted logits, target probabilities)

**Assertions**: 
- Loss values are non-negative
- Perfect match gives approximately zero loss
- Shapes are handled correctly
- No NaN or Inf values
- Proper error messages for invalid inputs

### Integration Tests

**Scope**: Test with actual dataset structure

**Requirements**:
- Load real data from `SpatialGraphDataset`
- Extract target_prob_vector from Data objects
- Compute loss with dummy model predictions
- Verify end-to-end workflow

### Edge Cases

**Specific edge cases to test**:
1. Zero probabilities in target vector (should clamp to epsilon)
2. Target probabilities that don't sum to exactly 1.0 (floating point errors)
3. Very small probabilities (numerical precision)
4. Large batch sizes
5. Single sample (unbatched)
6. Mismatched shapes (should raise error)
7. Wrong number of categories (should raise error)
8. NaN or Inf in inputs (should handle gracefully)

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Check Python syntax
python -m py_compile src/training/loss.py

# Run ruff linter (if available)
ruff check src/training/loss.py

# Check type hints (if mypy available)
mypy src/training/loss.py --ignore-missing-imports
```

### Level 2: Unit Tests

```bash
# Run all loss function tests
pytest tests/unit/test_loss.py -v

# Run with coverage
pytest tests/unit/test_loss.py --cov=src.training.loss --cov-report=term-missing

# Run specific test
pytest tests/unit/test_loss.py::test_basic_loss_computation -v
```

### Level 3: Integration Tests

```bash
# Test with actual dataset (if data available)
pytest tests/unit/test_loss.py::test_integration_with_dataset -v

# Test with feasibility script (should work with new loss function)
python scripts/test_training_feasibility.py --max_graphs 10
```

### Level 4: Manual Validation

```bash
# Quick smoke test
python -c "
import torch
from src.training.loss import distance_based_kl_loss, DistanceBasedKLLoss

# Test functional version
pred = torch.randn(2, 8)
target = torch.softmax(torch.randn(2, 8), dim=-1)
loss = distance_based_kl_loss(pred, target)
print(f'Functional loss: {loss.item():.4f}')

# Test class version
loss_fn = DistanceBasedKLLoss()
loss2 = loss_fn(pred, target)
print(f'Class-based loss: {loss2.item():.4f}')

# Test perfect match
target2 = torch.softmax(torch.randn(1, 8), dim=-1)
pred2 = torch.log(target2 + 1e-8)
loss3 = distance_based_kl_loss(pred2, target2)
print(f'Perfect match loss (should be ~0): {loss3.item():.6f}')

print('✅ All tests passed!')
"
```

### Level 5: Documentation Validation

```bash
# Verify docstrings are present
python -c "
import inspect
from src.training.loss import distance_based_kl_loss, DistanceBasedKLLoss

assert inspect.getdoc(distance_based_kl_loss) is not None, 'Missing docstring'
assert inspect.getdoc(DistanceBasedKLLoss) is not None, 'Missing docstring'
print('✅ Docstrings present')
"
```

---

## ACCEPTANCE CRITERIA

- [ ] `src/training/loss.py` implements `distance_based_kl_loss()` function
- [ ] `src/training/loss.py` implements `DistanceBasedKLLoss` class
- [ ] Loss function correctly computes KL divergence: KL(target || predicted)
- [ ] Loss function handles both batched [batch_size, 8] and unbatched [8] inputs
- [ ] Loss function supports `reduction='none'` for per-sample analysis (validation/test mode)
- [ ] Per-sample losses can be aggregated correctly (mean, std, min, max)
- [ ] Loss function includes numerical stability (epsilon clamping, normalization)
- [ ] Loss function validates input shapes and provides clear error messages
- [ ] All unit tests pass (100% coverage of loss.py)
- [ ] Integration test with dataset structure passes
- [ ] Perfect match gives approximately zero loss (within numerical precision)
- [ ] Loss values are always non-negative
- [ ] No NaN or Inf values in any test case
- [ ] Code follows project conventions (PEP 8, type hints, docstrings)
- [ ] All validation commands execute successfully
- [ ] Documentation matches implementation (matches `docs/distance-based-loss-calculation.md`)

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes (unit tests)
- [ ] No linting or type checking errors
- [ ] Manual validation confirms loss function works correctly
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability
- [ ] Loss function ready for integration with training script

---

## NOTES

### Key Implementation Details

1. **Target Vectors Are Pre-Computed**: The target probability vectors are already computed during feature engineering (see `feature_engineer.py:compute_target_probability_vector()`). They are stored in the dataset as `data.y` with shape [8]. The loss function should use them directly as probabilities, NOT apply softmax again.

2. **KL Divergence Input Format**: PyTorch's `F.kl_div` expects:
   - First argument: log-probabilities (apply `F.log_softmax` to predictions)
   - Second argument: probabilities (use targets directly, they're already probabilities)
   - Reduction modes:
     - `'batchmean'`: Divides by batch size (use for training)
     - `'none'`: Returns per-sample losses [batch_size] (use for validation/test analysis)
     - `'sum'`: Sum of all losses (not normalized)
     - `'mean'`: Average over all elements (batch × categories, rarely used)

3. **Numerical Stability**: 
   - Clamp target probabilities to [epsilon, 1.0] to avoid log(0)
   - Renormalize after clamping to ensure sum = 1.0
   - Handle floating point errors in probability sums

4. **Temperature Parameter**: The temperature parameter in `config.yaml` is used during feature engineering to compute target vectors from distances. It's not used in the loss function itself (targets are already probabilities). However, it's good to load it for documentation/reference.

5. **Test Script Note**: The `test_training_feasibility.py` script (line 208) applies softmax to targets, which is incorrect since targets are already probabilities. The loss function implementation should NOT do this.

### Design Decisions

- **Functional + Class Interface**: Provide both functional and class-based interfaces for flexibility (functional for direct use, class for nn.Module integration)
- **Epsilon Clamping**: Use epsilon=1e-8 for numerical stability (standard PyTorch practice)
- **Batch Reduction**: 
  - Use 'batchmean' by default for training (proper batch averaging, divides by batch size)
  - Support 'none' for validation/test (per-sample analysis, enables detailed statistics)
- **Input Validation**: Validate shapes and types early with clear error messages (fail fast principle)

### Validation/Test Mode: Per-Sample Analysis

**Implementation Strategy**: Use per-sample losses for validation/test to enable detailed analysis.

1. **Training Mode**:
   - Use `reduction='batchmean'` (default)
   - Formula: `L = (1/N) × Σ_i Σ_j P_target,ij × log(P_target,ij / P_predicted,ij)`
   - Returns: scalar tensor (average loss per sample, normalized by batch size)
   - This is what gradients are computed from

2. **Validation/Test Mode** (Per-Sample Analysis):
   - Use `reduction='none'` to get per-sample losses
   - Returns: tensor of shape `[batch_size]` with one loss value per sample
   - Each value is: `L_i = Σ_j P_target,ij × log(P_target,ij / P_predicted,ij)` (not normalized)
   - Collect all per-sample losses across batches, then aggregate:
     ```python
     # In validation/test loop:
     all_losses = []
     for batch in dataloader:
         output = model(batch)
         loss_per_sample = distance_based_kl_loss(
             output, batch.y, reduction='none'
         )  # Shape: [batch_size]
         all_losses.append(loss_per_sample)
     
     # Concatenate all batches
     all_losses = torch.cat(all_losses)  # Shape: [total_samples]
     
     # Compute statistics
     overall_avg = all_losses.mean().item()
     overall_std = all_losses.std().item()
     overall_min = all_losses.min().item()
     overall_max = all_losses.max().item()
     ```
   - **Benefits**:
     - Enables per-sample analysis (identify outliers, difficult samples)
     - Compute distribution statistics (mean, std, min, max, percentiles)
     - Still get overall average: `all_losses.mean()`
     - Can analyze loss by neighborhood, by sample type, etc.

**Usage Pattern in Training Script**:
```python
# Training
loss = distance_based_kl_loss(predicted, target, reduction='batchmean')
loss.backward()  # Scalar loss for backprop

# Validation/Test
loss_per_sample = distance_based_kl_loss(predicted, target, reduction='none')
# Collect and aggregate later for metrics
```

### Future Integration

This loss function will be used by:
- `src/training/train.py` (training script - to be implemented)
- Model training loop (forward pass → loss computation → backward pass)
- Evaluation metrics (loss values for validation)

### Performance Considerations

- KL divergence computation is efficient (O(batch_size * num_categories))
- Log-softmax is numerically stable and fast
- No significant performance bottlenecks expected
- Can be used with mixed precision training (FP16) if needed

---

*Plan Version: 1.0*  
*Created: January 2025*  
*Project: AI4SI - 15-Minute City Service Gap Prediction Model*
