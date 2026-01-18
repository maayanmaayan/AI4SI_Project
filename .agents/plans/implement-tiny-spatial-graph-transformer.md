# Feature: Implement TinySpatialGraphTransformer Class for Testing

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Create a drastically reduced-capacity version of the SpatialGraphTransformer model (`TinySpatialGraphTransformer`) to debug model collapse issues observed during quick test training. The tiny model will have <10,000 parameters (vs ~1M in the original) with higher dropout (0.3-0.5) to test if the architecture can learn without collapsing to single-class predictions.

**Purpose**: Debugging tool to determine if model collapse is due to overfitting/capacity issues or fundamental data/architecture problems. If the tiny model shows uncertainty (e.g., 40% Park, 30% Education), it confirms the logic works and we can proceed to full dataset training.

## User Story

As a **ML researcher debugging model collapse**
I want to **test with a tiny model (<10K parameters) with high dropout**
So that **I can determine if the collapse is due to overfitting or fundamental issues before investing in full hyperparameter sweep**

## Problem Statement

During quick test training, the SpatialGraphTransformer model (1M+ parameters) collapsed to predicting a single class (100% Parks) with extreme overconfidence, despite training loss decreasing. This could be due to:
1. **Overfitting**: Model too large for small dataset (128 points)
2. **Insufficient regularization**: Dropout (0.1) too low
3. **Fundamental architecture/data issue**: Problem exists regardless of capacity

We need a diagnostic tool to isolate the cause before proceeding with expensive hyperparameter sweeps.

## Solution Statement

Create `TinySpatialGraphTransformer` class that mirrors `SpatialGraphTransformer` architecture but with:
- **hidden_dim**: 16 (vs 128) - 8x reduction
- **num_layers**: 1 (vs 3) - Minimal depth
- **num_heads**: 2 (vs 4) - Compatible with hidden_dim=16
- **dropout**: 0.4 (vs 0.1) - High regularization to prevent overconfidence
- **Target**: <10,000 parameters (vs ~1,061,640)

The tiny model will use the same forward pass logic, weight initialization, and target node extraction as the original, ensuring architectural consistency while drastically reducing capacity.

## Feature Metadata

**Feature Type**: Enhancement (Debugging Tool)
**Estimated Complexity**: Low-Medium
**Primary Systems Affected**: 
- `src/training/model.py` - Add TinySpatialGraphTransformer class
- `src/training/train.py` - Add option to use tiny model
- `tests/unit/test_model.py` - Add tests for tiny model
**Dependencies**: 
- PyTorch 2.x
- PyTorch Geometric 2.3+
- Existing SpatialGraphTransformer architecture

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `src/training/model.py` (lines 23-240) - **CRITICAL**: Contains SpatialGraphTransformer class that we'll mirror
  - Why: Need to understand exact architecture, forward pass, weight initialization, and target node extraction
  - Key patterns: TransformerConv layers, edge encoding, residual connections, GELU activation
- `src/training/model.py` (lines 207-240) - `create_model_from_config()` function
  - Why: Need to create similar factory function for tiny model
- `src/training/train.py` (lines 273-275) - Model instantiation in training
  - Why: Need to add option to use tiny model instead of default
- `tests/unit/test_model.py` (all) - Existing model tests
  - Why: Need to create similar tests for tiny model to ensure it works correctly
- `models/config.yaml` (lines 4-13) - Model configuration structure
  - Why: Understand config format, though tiny model may use hardcoded values

### New Files to Create

- None - All changes will be additions to existing files

### Files to Modify

- `src/training/model.py` - Add TinySpatialGraphTransformer class and factory function
- `src/training/train.py` - Add parameter/flag to use tiny model
- `tests/unit/test_model.py` - Add tests for TinySpatialGraphTransformer

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [PyTorch Geometric TransformerConv Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv)
  - Specific section: TransformerConv parameters and usage
  - Why: Need to ensure tiny model uses TransformerConv correctly with reduced dimensions
- [PyTorch Dropout Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
  - Specific section: Dropout behavior during training vs evaluation
  - Why: High dropout (0.4) needs to be applied correctly

### Patterns to Follow

**Naming Conventions:**
- Class names: `CamelCase` (e.g., `TinySpatialGraphTransformer`)
- Function names: `snake_case` (e.g., `create_tiny_model_from_config`)
- Module-level logger: `logger = get_logger(__name__)`

**Error Handling:**
- Use type hints for all function parameters and returns
- Validate inputs in `__init__` methods
- Log warnings for edge cases (see `_extract_target_nodes` line 193)

**Logging Pattern:**
```python
from src.utils.logging import get_logger
logger = get_logger(__name__)
logger.info(f"Created TinySpatialGraphTransformer: {param_count:,} parameters")
```

**Model Architecture Pattern:**
- Inherit from `nn.Module`
- Store hyperparameters as instance attributes
- Use `nn.ModuleList` for variable-length layers
- Initialize weights in `_init_weights()` method
- Extract target nodes in `_extract_target_nodes()` helper method

**Weight Initialization Pattern:**
```python
def _init_weights(self) -> None:
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

**Forward Pass Pattern:**
- Encode nodes: `x = self.node_encoder(data.x)`
- Encode edges: `edge_emb = self.edge_encoder(data.edge_attr)`
- Apply transformer layers with residual connections
- Extract target node: `target_emb = self._extract_target_nodes(x, data.batch)`
- Classify: `logits = self.classifier(target_emb)`

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation

Create the TinySpatialGraphTransformer class with reduced capacity architecture.

**Tasks:**
- Add TinySpatialGraphTransformer class to model.py
- Implement reduced architecture (hidden_dim=16, num_layers=1, num_heads=2, dropout=0.4)
- Mirror all methods from SpatialGraphTransformer (forward, _init_weights, _extract_target_nodes)
- Ensure parameter count is <10,000

### Phase 2: Integration

Add factory function and training script integration.

**Tasks:**
- Add `create_tiny_model_from_config()` factory function
- Update training script to support tiny model option
- Ensure tiny model works with existing DataLoader and loss function

### Phase 3: Testing & Validation

Create comprehensive tests for tiny model.

**Tasks:**
- Add unit tests mirroring SpatialGraphTransformer tests
- Verify parameter count is <10,000
- Test forward pass with various graph sizes
- Verify dropout is applied correctly

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### CREATE TinySpatialGraphTransformer class in src/training/model.py

- **IMPLEMENT**: Add TinySpatialGraphTransformer class after SpatialGraphTransformer class (after line 204, before create_model_from_config)
- **PATTERN**: Mirror SpatialGraphTransformer structure exactly (lines 23-204)
- **PARAMETERS**: 
  - `hidden_dim`: 16 (default, vs 128)
  - `num_layers`: 1 (default, vs 3)
  - `num_heads`: 2 (default, vs 4)
  - `dropout`: 0.4 (default, vs 0.1)
  - Other parameters same as SpatialGraphTransformer (num_features=33, num_classes=8, edge_dim=4)
- **ARCHITECTURE**: 
  - Node encoder: 33 → 16
  - Edge encoder: 4 → 16
  - Single TransformerConv layer (hidden_dim=16, heads=2, concat=True → output 32)
  - Head projection: 32 → 16
  - Single LayerNorm
  - Classifier: 16 → 8
- **METHODS**: Copy forward(), _init_weights(), _extract_target_nodes() exactly from SpatialGraphTransformer
- **DOCSTRING**: Update docstring to indicate this is a tiny version for debugging
- **VALIDATE**: `python -c "from src.training.model import TinySpatialGraphTransformer; m = TinySpatialGraphTransformer(); print(f'Params: {sum(p.numel() for p in m.parameters()):,}')"` - Should show <10,000

### CREATE create_tiny_model_from_config function in src/training/model.py

- **IMPLEMENT**: Add factory function after create_model_from_config (after line 240)
- **PATTERN**: Mirror create_model_from_config structure (lines 207-240)
- **FUNCTION**: `create_tiny_model_from_config(config: Optional[dict] = None) -> TinySpatialGraphTransformer`
- **BEHAVIOR**: 
  - Load config if None provided
  - Create TinySpatialGraphTransformer with hardcoded tiny parameters (ignore config model section)
  - Log parameter count
  - Return model
- **IMPORTS**: Ensure TinySpatialGraphTransformer is imported in module
- **VALIDATE**: `python -c "from src.training.model import create_tiny_model_from_config; m = create_tiny_model_from_config(); print(f'Params: {sum(p.numel() for p in m.parameters()):,}')"` - Should show <10,000

### UPDATE train.py to support tiny model option

- **IMPLEMENT**: Modify train() function to accept `use_tiny_model: bool = False` parameter
- **LOCATION**: Add parameter to train() function signature (around line 174)
- **LOGIC**: 
  - If `use_tiny_model=True`, call `create_tiny_model_from_config(config)` instead of `create_model_from_config(config)`
  - Import: `from src.training.model import create_tiny_model_from_config`
  - Add logging: `logger_experiment.info("Using TinySpatialGraphTransformer for debugging")`
- **PATTERN**: Follow existing model creation pattern (lines 273-275)
- **VALIDATE**: `python -c "from src.training.train import train; import sys; sys.exit(0)"` - Should import without errors

### UPDATE train_graph_transformer.py script to add --tiny-model flag

- **IMPLEMENT**: Add `--tiny-model` argument to argument parser
- **LOCATION**: Add after `--quick-test` argument (around line 63)
- **ARGUMENT**: `parser.add_argument("--tiny-model", action="store_true", help="Use TinySpatialGraphTransformer for debugging")`
- **PASS**: Pass `use_tiny_model=args.tiny_model` to train() function call
- **VALIDATE**: `python scripts/train_graph_transformer.py --help | grep tiny-model` - Should show the new flag

### ADD test_tiny_model_initialization to tests/unit/test_model.py

- **IMPLEMENT**: Add test function after test_model_initialization (after line 19)
- **PATTERN**: Mirror test_model_initialization structure (lines 10-18)
- **TEST**: 
  - Create TinySpatialGraphTransformer with default parameters
  - Assert hidden_dim == 16, num_layers == 1, num_heads == 2, dropout == 0.4
  - Assert num_features == 33, num_classes == 8, edge_dim == 4
- **IMPORTS**: Add TinySpatialGraphTransformer to imports
- **VALIDATE**: `pytest tests/unit/test_model.py::test_tiny_model_initialization -v`

### ADD test_tiny_model_parameter_count to tests/unit/test_model.py

- **IMPLEMENT**: Add test function after test_tiny_model_initialization
- **TEST**: 
  - Create TinySpatialGraphTransformer
  - Count parameters: `sum(p.numel() for p in model.parameters())`
  - Assert parameter count < 10,000
- **VALIDATE**: `pytest tests/unit/test_model.py::test_tiny_model_parameter_count -v`

### ADD test_tiny_model_forward_single_graph to tests/unit/test_model.py

- **IMPLEMENT**: Add test function mirroring test_forward_single_graph (lines 37-61)
- **PATTERN**: Use same graph structure, but with TinySpatialGraphTransformer
- **TEST**: 
  - Create tiny model
  - Create single graph (1 target + 5 neighbors)
  - Forward pass
  - Assert output shape [1, 8]
  - Assert no NaN/Inf values
- **VALIDATE**: `pytest tests/unit/test_model.py::test_tiny_model_forward_single_graph -v`

### ADD test_tiny_model_forward_batched_graphs to tests/unit/test_model.py

- **IMPLEMENT**: Add test function mirroring test_forward_batched_graphs (lines 63-96)
- **PATTERN**: Use same batched graph structure, but with TinySpatialGraphTransformer
- **TEST**: 
  - Create tiny model
  - Create 3 batched graphs
  - Forward pass
  - Assert output shape [3, 8]
  - Assert no NaN/Inf values
- **VALIDATE**: `pytest tests/unit/test_model.py::test_tiny_model_forward_batched_graphs -v`

### ADD test_tiny_model_dropout to tests/unit/test_model.py

- **IMPLEMENT**: Add test to verify dropout is applied correctly
- **TEST**: 
  - Create tiny model with dropout=0.4
  - Set model to training mode: `model.train()`
  - Run forward pass multiple times with same input
  - Verify outputs differ (dropout introduces randomness)
  - Set model to eval mode: `model.eval()`
  - Run forward pass multiple times with same input
  - Verify outputs are identical (dropout disabled in eval)
- **VALIDATE**: `pytest tests/unit/test_model.py::test_tiny_model_dropout -v`

### ADD test_create_tiny_model_from_config to tests/unit/test_model.py

- **IMPLEMENT**: Add test function mirroring test_create_model_from_config (lines 181-186)
- **PATTERN**: Test factory function creates correct model type
- **TEST**: 
  - Call create_tiny_model_from_config()
  - Assert isinstance(model, TinySpatialGraphTransformer)
  - Assert parameter count < 10,000
- **IMPORTS**: Add create_tiny_model_from_config to imports
- **VALIDATE**: `pytest tests/unit/test_model.py::test_create_tiny_model_from_config -v`

---

## TESTING STRATEGY

### Unit Tests

All tests should mirror existing SpatialGraphTransformer tests to ensure architectural consistency:
- Model initialization with default and custom parameters
- Forward pass with single and batched graphs
- Parameter count validation (<10,000)
- Dropout behavior verification
- Factory function correctness

### Integration Tests

After implementation, manually test with quick test training:
- Run: `python scripts/train_graph_transformer.py --quick-test --tiny-model`
- Verify: Model trains without errors
- Verify: Model shows uncertainty in predictions (not 100% single class)
- Verify: Parameter count logged correctly

### Edge Cases

- Empty neighbors (0 neighbors) - should still work
- Variable graph sizes - should handle batching correctly
- Dropout during training vs evaluation - should behave correctly

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Check for syntax errors
python -m py_compile src/training/model.py

# Run linter (if available)
ruff check src/training/model.py
```

### Level 2: Unit Tests

```bash
# Run all model tests
pytest tests/unit/test_model.py -v

# Run specific tiny model tests
pytest tests/unit/test_model.py::test_tiny_model_initialization -v
pytest tests/unit/test_model.py::test_tiny_model_parameter_count -v
pytest tests/unit/test_model.py::test_tiny_model_forward_single_graph -v
pytest tests/unit/test_model.py::test_tiny_model_forward_batched_graphs -v
pytest tests/unit/test_model.py::test_tiny_model_dropout -v
pytest tests/unit/test_model.py::test_create_tiny_model_from_config -v
```

### Level 3: Integration Tests

```bash
# Test model can be imported and instantiated
python -c "from src.training.model import TinySpatialGraphTransformer, create_tiny_model_from_config; m = TinySpatialGraphTransformer(); print(f'Params: {sum(p.numel() for p in m.parameters()):,}')"

# Test factory function
python -c "from src.training.model import create_tiny_model_from_config; m = create_tiny_model_from_config(); print(f'Params: {sum(p.numel() for p in m.parameters()):,}')"

# Test training script accepts flag
python scripts/train_graph_transformer.py --help | grep -q "tiny-model" && echo "Flag exists" || echo "Flag missing"
```

### Level 4: Manual Validation

```bash
# Quick test training with tiny model (should complete without errors)
python scripts/train_graph_transformer.py --quick-test --tiny-model

# Verify predictions show uncertainty (not 100% single class)
# Check experiment output: experiments/runs/run_*/test_predictions.csv
# Look for predicted_probs with multiple non-zero values
```

### Level 5: Parameter Count Verification

```bash
# Verify parameter count is <10,000
python -c "
from src.training.model import TinySpatialGraphTransformer
model = TinySpatialGraphTransformer()
param_count = sum(p.numel() for p in model.parameters())
print(f'Parameter count: {param_count:,}')
assert param_count < 10000, f'Parameter count {param_count} exceeds 10,000'
print('✓ Parameter count validation passed')
"
```

---

## ACCEPTANCE CRITERIA

- [ ] TinySpatialGraphTransformer class created with <10,000 parameters
- [ ] Architecture mirrors SpatialGraphTransformer (same forward pass logic)
- [ ] Default parameters: hidden_dim=16, num_layers=1, num_heads=2, dropout=0.4
- [ ] Factory function `create_tiny_model_from_config()` works correctly
- [ ] Training script accepts `--tiny-model` flag
- [ ] All unit tests pass (initialization, forward pass, parameter count, dropout, factory)
- [ ] Model can train on quick test dataset without errors
- [ ] Model shows uncertainty in predictions (not 100% single class collapse)
- [ ] No regressions in existing SpatialGraphTransformer functionality
- [ ] Code follows project conventions (type hints, docstrings, logging)

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes (unit tests)
- [ ] No linting or type checking errors
- [ ] Manual testing confirms tiny model works
- [ ] Parameter count verified <10,000
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability

---

## NOTES

### Design Decisions

1. **Hardcoded Parameters**: Tiny model uses hardcoded parameters (hidden_dim=16, etc.) rather than reading from config. This ensures it's always tiny and prevents accidental misconfiguration.

2. **Same Architecture**: Tiny model uses identical forward pass logic to SpatialGraphTransformer. This ensures any differences in behavior are due to capacity, not architecture.

3. **High Dropout**: Dropout of 0.4 is significantly higher than default (0.1) to combat overconfidence. This is a key debugging feature.

4. **Separate Class**: Created as separate class rather than parameter flag to keep code clean and make it obvious when tiny model is being used.

### Parameter Count Calculation

Approximate parameter count for TinySpatialGraphTransformer:
- Node encoder: 33 × 16 = 528
- Edge encoder: 4 × 16 = 64
- TransformerConv (hidden_dim=16, heads=2): ~1,280 (internal PyTorch Geometric calculation)
- Head projection: 32 × 16 = 512
- LayerNorm: 16 × 2 = 32
- Classifier: 16 × 8 = 128
- **Total: ~2,544 parameters** (well under 10,000 target)

### Expected Behavior

If tiny model works correctly:
- Should show uncertainty in predictions (e.g., [0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.0])
- Training loss should decrease
- Validation loss should decrease
- Top-1 accuracy may be low, but predictions should be distributed across classes

If tiny model still collapses:
- Indicates fundamental issue (data distribution, loss function, etc.)
- Not a capacity/overfitting problem
- Need to investigate data or architecture further

### Future Enhancements

- Could add config option to customize tiny model parameters
- Could add more aggressive regularization (label smoothing, weight decay)
- Could experiment with different tiny architectures (even smaller hidden_dim)
