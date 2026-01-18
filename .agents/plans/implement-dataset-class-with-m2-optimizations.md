# Feature: Dataset Class Implementation with Data Verification and M2 MacBook Air Optimizations

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Implement a PyTorch Geometric Dataset class for loading pre-processed feature data and converting it to graph objects. This includes data format verification, neighborhood-level stratified splits, M2 MacBook Air optimizations (MPS support, mixed precision, gradient accumulation), and a quick test script to validate training feasibility on the hardware.

**Core Functionality:**
- Data format verification script to validate Parquet files match expected schema
- `SpatialGraphDataset` class that loads Parquet files and builds PyTorch Geometric Data objects
- Utility functions for loading features from directories and creating neighborhood-level splits
- Configuration updates for M2 optimizations (MPS, mixed precision, gradient accumulation)
- Quick test script to validate training feasibility and measure performance

## User Story

As a **ML engineer/researcher**
I want to **load pre-processed feature data into a PyTorch Geometric Dataset with M2 MacBook Air optimizations**
So that **I can efficiently train the Graph Transformer model on my local machine with proper data validation and hardware-optimized settings**

## Problem Statement

The project needs:
1. **Data validation** - Ensure pre-processed Parquet files match expected format before training
2. **Dataset class** - Convert Parquet data to PyTorch Geometric Data objects for training
3. **Neighborhood-level splits** - Create train/val/test splits by neighborhood to avoid spatial leakage
4. **M2 optimizations** - Enable MPS (Metal Performance Shaders), mixed precision, and efficient batching for MacBook Air M2
5. **Quick validation** - Test script to verify training is feasible on M2 hardware before long training runs

## Solution Statement

Implement:
1. Enhanced data verification script that validates Parquet format matches dataset expectations
2. `SpatialGraphDataset` class inheriting from `torch_geometric.data.Dataset` that builds graphs on-the-fly
3. Utility functions for loading features and creating neighborhood-level stratified splits
4. Configuration updates to enable MPS, mixed precision, and gradient accumulation
5. Quick test script that runs a small training loop to measure memory usage and performance

## Feature Metadata

**Feature Type**: New Capability + Enhancement
**Estimated Complexity**: Medium
**Primary Systems Affected**: 
- `src/training/dataset.py` (new file)
- `models/config.yaml` (update for M2 optimizations)
- `src/utils/helpers.py` (update set_random_seeds for MPS)
- `scripts/validate_feature_output.py` (enhance existing)
- `scripts/test_training_feasibility.py` (new test script)
**Dependencies**: 
- torch-geometric 2.3+ (already in requirements)
- torch 2.0+ (already in requirements, supports MPS)
- pandas, numpy (already in requirements)

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `scripts/validate_feature_output.py` (all lines) - Why: Existing validation pattern to enhance
- `src/utils/config.py` (all lines) - Why: Configuration loading pattern to follow
- `src/utils/logging.py` (all lines) - Why: Logging setup pattern
- `src/utils/helpers.py` (lines 314-337) - Why: `set_random_seeds()` function to update for MPS
- `src/utils/helpers.py` (lines 60-89) - Why: `get_compliant_neighborhoods()` filter function
- `src/data/collection/feature_engineer.py` (lines 1397-1408) - Why: Data structure format (target_features, neighbor_data, target_prob_vector)
- `models/config.yaml` (all lines) - Why: Configuration structure to extend
- `tests/unit/test_feature_engineer.py` (lines 1-50) - Why: Test pattern to follow
- `.agents/plans/implement-graph-transformer-model.md` (lines 500-599) - Why: Dataset class specification and requirements

### New Files to Create

- `src/training/dataset.py` - Dataset class and utility functions
- `scripts/test_training_feasibility.py` - Quick test script for M2 validation
- `tests/unit/test_dataset.py` - Unit tests for dataset class

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [PyTorch Geometric Dataset Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Dataset)
  - Specific section: Custom Dataset implementation
  - Why: Required for implementing SpatialGraphDataset class
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
  - Specific section: Device selection and mixed precision
  - Why: Required for M2 MacBook Air optimizations
- [PyTorch Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
  - Specific section: torch.cuda.amp.autocast for MPS
  - Why: Required for FP16 training on M2
- [PyTorch Geometric DataLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.DataLoader)
  - Specific section: Batching variable-sized graphs
  - Why: Required for efficient data loading

### Patterns to Follow

**Naming Conventions:**
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Use descriptive names: `load_features_from_directory`, `create_train_val_test_splits`

**Error Handling:**
- Use try-except blocks with specific exception types
- Log errors using `logger.error()` from `src.utils.logging`
- Raise `ValueError` for invalid inputs with descriptive messages

**Logging Pattern:**
```python
from src.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("Processing data...")
```

**Configuration Pattern:**
```python
from src.utils.config import get_config
config = get_config()
batch_size = config["training"]["batch_size"]
```

**Data Validation Pattern:**
- Check required columns exist
- Validate data types and shapes
- Check for NaN/Inf values
- Return boolean or raise exceptions with clear messages

**Test Pattern:**
- Use pytest fixtures for test data
- Follow naming: `test_<function_name>_<scenario>`
- Use assertions with descriptive messages

---

## IMPLEMENTATION PLAN

### Phase 1: Data Verification Enhancement

Enhance existing validation script to verify data format matches dataset class expectations.

**Tasks:**
- Update `scripts/validate_feature_output.py` to add dataset-specific checks
- Add validation for data type conversions (numpy → torch)
- Add summary statistics output

### Phase 2: Dataset Class Implementation

Implement the core dataset class and utility functions.

**Tasks:**
- Create `SpatialGraphDataset` class
- Implement `load_features_from_directory()` utility
- Implement `create_train_val_test_splits()` utility with neighborhood-level stratification
- Handle edge cases (zero neighbors, missing data)

### Phase 3: M2 Optimizations

Update configuration and utilities for M2 MacBook Air support.

**Tasks:**
- Update `models/config.yaml` with M2 optimization settings
- Update `set_random_seeds()` to support MPS
- Add device selection utility function

### Phase 4: Quick Test Script

Create test script to validate training feasibility on M2.

**Tasks:**
- Create `scripts/test_training_feasibility.py`
- Implement memory monitoring
- Implement performance benchmarking
- Test with small data subset

### Phase 5: Testing & Validation

Write comprehensive unit tests.

**Tasks:**
- Unit tests for dataset class
- Unit tests for utility functions
- Integration test with actual data files

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### Task 1: Enhance Data Verification Script

**UPDATE** `scripts/validate_feature_output.py`

- **IMPLEMENT**: Add dataset-specific validation checks
- **ADD**: Function to validate data can be converted to PyTorch tensors
- **ADD**: Summary statistics output (total graphs, neighbor distribution, memory estimate)
- **PATTERN**: Follow existing validation pattern (lines 33-102)
- **IMPORTS**: Add `torch` import for tensor conversion test
- **VALIDATE**: `python scripts/validate_feature_output.py data/processed/features/compliant/paris_rive_gauche/target_points.parquet`

**Key additions:**
- Check that `target_features` and `target_prob_vector` can be converted to `torch.float32`
- Check that `neighbor_data` entries can be converted to tensors
- Output summary: total graphs, min/max/mean neighbors, estimated memory per graph
- Validate edge attributes are numeric and finite

### Task 2: Create Dataset Class Module

**CREATE** `src/training/dataset.py`

- **IMPLEMENT**: `SpatialGraphDataset` class inheriting from `torch_geometric.data.Dataset`
- **PATTERN**: Follow PyTorch Geometric Dataset pattern (see documentation)
- **IMPORTS**: 
  ```python
  from torch_geometric.data import Dataset, Data
  import pandas as pd
  import torch
  import numpy as np
  from typing import Optional, Tuple
  from pathlib import Path
  from src.utils.logging import get_logger
  ```
- **GOTCHA**: Don't use `root` parameter for file-based dataset - we're loading from DataFrame
- **VALIDATE**: Import test: `python -c "from src.training.dataset import SpatialGraphDataset; print('OK')"`

**Class structure:**
```python
class SpatialGraphDataset(Dataset):
    def __init__(self, features_df: pd.DataFrame, transform=None, pre_transform=None):
        # Store DataFrame, initialize parent with None root
        # No file-based caching needed
    
    def len(self) -> int:
        # Return length of DataFrame
    
    def get(self, idx: int) -> Data:
        # Extract row, build graph, return Data object
```

### Task 3: Implement Dataset.get() Method

**UPDATE** `src/training/dataset.py` - `get()` method

- **IMPLEMENT**: Build PyTorch Geometric Data object from DataFrame row
- **ALGORITHM**:
  1. Extract row from DataFrame
  2. Convert `target_features` to tensor `[33]`
  3. Build node features: `[target, neighbor1, ..., neighborN]` → `[1+N, 33]`
  4. Build edge_index: star graph (neighbors → target) → `[2, num_neighbors]`
  5. Build edge_attr: `[dx, dy, euclidean_dist, network_dist]` → `[num_neighbors, 4]`
  6. Convert `target_prob_vector` to tensor `[8]`
  7. Create Data object with metadata
- **EDGE CASE**: Handle zero neighbors (empty edge_index and edge_attr)
- **PATTERN**: Follow `.agents/plans/implement-graph-transformer-model.md` lines 534-584
- **VALIDATE**: `python -c "from src.training.dataset import SpatialGraphDataset; import pandas as pd; df = pd.read_parquet('data/processed/features/compliant/paris_rive_gauche/target_points.parquet'); ds = SpatialGraphDataset(df[:10]); print(f'Dataset length: {len(ds)}'); data = ds[0]; print(f'Graph nodes: {data.x.shape[0]}, edges: {data.edge_index.shape[1]}')"`

**Key implementation details:**
- Node 0 = target point, nodes 1-N = neighbors
- Edge index: `[[1,2,3,...], [0,0,0,...]]` (source, target)
- Edge attributes: `[dx, dy, euclidean_distance, network_distance]`
- Store `target_id` and `neighborhood_name` as Data attributes

### Task 4: Implement load_features_from_directory()

**UPDATE** `src/training/dataset.py` - Add utility function

- **IMPLEMENT**: `load_features_from_directory(directory: str) -> pd.DataFrame`
- **ALGORITHM**:
  1. Recursively find all `target_points.parquet` files in directory
  2. Load each file with `pd.read_parquet()`
  3. Concatenate into single DataFrame
  4. Validate required columns exist
  5. Return combined DataFrame
- **PATTERN**: Follow `scripts/validate_feature_output.py` lines 115-124 for file finding
- **IMPORTS**: Use `Path.glob("**/target_points.parquet")` for recursive search
- **VALIDATE**: `python -c "from src.training.dataset import load_features_from_directory; df = load_features_from_directory('data/processed/features/compliant'); print(f'Loaded {len(df)} rows from {df[\"neighborhood_name\"].nunique()} neighborhoods')"`

### Task 5: Implement create_train_val_test_splits()

**UPDATE** `src/training/dataset.py` - Add utility function

- **IMPLEMENT**: `create_train_val_test_splits(features_df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
- **ALGORITHM**:
  1. Group DataFrame by `neighborhood_name`
  2. Get unique neighborhoods list
  3. Shuffle neighborhoods (not individual points) using `random_seed`
  4. Split neighborhoods into train/val/test based on ratios
  5. Filter DataFrame to return three DataFrames (train, val, test)
  6. Validate ratios sum to 1.0
- **PATTERN**: Use `random.seed()` and `random.shuffle()` for reproducibility
- **IMPORTS**: `import random` and `from typing import Tuple`
- **GOTCHA**: Split by neighborhood, not by points - prevents spatial leakage
- **VALIDATE**: `python -c "from src.training.dataset import load_features_from_directory, create_train_val_test_splits; df = load_features_from_directory('data/processed/features/compliant'); train, val, test = create_train_val_test_splits(df); print(f'Train: {len(train)} ({train[\"neighborhood_name\"].nunique()} neighborhoods), Val: {len(val)}, Test: {len(test)}')"`

### Task 6: Update Configuration for M2 Optimizations

**UPDATE** `models/config.yaml`

- **UPDATE**: `training` section to add M2 optimization settings
- **ADD**:
  ```yaml
  training:
    batch_size: 16  # Keep existing
    use_mixed_precision: true  # Change from false
    gradient_accumulation_steps: 1  # New: can increase to 2-4 if needed
    num_workers: 2  # New: DataLoader workers (2 optimal for M2)
    pin_memory: false  # New: Not needed for MPS
    device: "auto"  # New: "auto" will select MPS if available, else CPU
  ```
- **PATTERN**: Follow existing config structure
- **VALIDATE**: `python -c "from src.utils.config import load_config; config = load_config(); print(f'Mixed precision: {config[\"training\"][\"use_mixed_precision\"]}')"`

### Task 7: Update set_random_seeds() for MPS Support

**UPDATE** `src/utils/helpers.py` - `set_random_seeds()` function

- **UPDATE**: Add MPS seed setting (lines 314-337)
- **ADD**: Check for MPS availability and set seeds
- **PATTERN**: Mirror CUDA seed setting pattern (lines 334-336)
- **CODE**:
  ```python
  if TORCH_AVAILABLE:
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
      # Add MPS support
      if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
          # MPS doesn't have separate seed functions, but manual_seed covers it
          pass  # torch.manual_seed() is sufficient for MPS
  ```
- **VALIDATE**: `python -c "from src.utils.helpers import set_random_seeds; set_random_seeds(42); print('OK')"`

### Task 8: Add Device Selection Utility

**UPDATE** `src/training/dataset.py` - Add utility function (or create `src/training/utils.py`)

- **IMPLEMENT**: `get_device() -> torch.device` function
- **ALGORITHM**:
  1. Check if MPS is available: `torch.backends.mps.is_available()`
  2. If yes, return `torch.device("mps")`
  3. Else if CUDA available, return `torch.device("cuda")`
  4. Else return `torch.device("cpu")`
- **PATTERN**: Standard PyTorch device selection pattern
- **IMPORTS**: `import torch`
- **VALIDATE**: `python -c "from src.training.dataset import get_device; print(f'Device: {get_device()}')"`

**Alternative**: Add to `src/utils/helpers.py` if it's a general utility

### Task 9: Create Quick Test Script

**CREATE** `scripts/test_training_feasibility.py`

- **IMPLEMENT**: Quick test script to validate training on M2
- **FEATURES**:
  1. Load small subset of data (1-2 neighborhoods, ~100 graphs)
  2. Create minimal dataset and DataLoader
  3. Create dummy model (simple linear layers matching expected input/output)
  4. Run forward + backward pass
  5. Measure memory usage and time
  6. Output feasibility report
- **PATTERN**: Follow `scripts/validate_feature_output.py` structure (argparse, main function)
- **IMPORTS**: 
  ```python
  import torch
  import torch.nn as nn
  from torch_geometric.loader import DataLoader
  from src.training.dataset import SpatialGraphDataset, load_features_from_directory
  from src.utils.config import get_config
  from src.utils.logging import setup_logging, get_logger
  ```
- **OUTPUT**: Print memory usage, time per batch, estimated time per epoch
- **VALIDATE**: `python scripts/test_training_feasibility.py --max_graphs 100`

**Script structure:**
```python
def test_training_feasibility(max_graphs: int = 100):
    # Load data subset
    # Create dataset and dataloader
    # Create dummy model
    # Run training loop for few batches
    # Measure memory and time
    # Print report
```

### Task 10: Write Unit Tests for Dataset Class

**CREATE** `tests/unit/test_dataset.py`

- **IMPLEMENT**: Unit tests for dataset class and utilities
- **TESTS**:
  1. `test_dataset_len()` - Test dataset length
  2. `test_dataset_get()` - Test graph structure from get()
  3. `test_dataset_zero_neighbors()` - Test edge case with 0 neighbors
  4. `test_load_features_from_directory()` - Test loading function
  5. `test_create_train_val_test_splits()` - Test split function
  6. `test_splits_no_leakage()` - Verify no neighborhood overlap between splits
- **PATTERN**: Follow `tests/unit/test_feature_engineer.py` pattern (fixtures, pytest)
- **FIXTURES**: Create test DataFrame with sample data
- **VALIDATE**: `pytest tests/unit/test_dataset.py -v`

**Test data fixture:**
```python
@pytest.fixture
def sample_features_df():
    # Create minimal DataFrame matching expected format
    # Return DataFrame with 2-3 rows, varying neighbor counts
```

---

## TESTING STRATEGY

### Unit Tests

**Scope**: Test each function independently with mock data

**Requirements**:
- Test dataset class with various graph sizes (0, 1, 10, 100 neighbors)
- Test utility functions with edge cases (empty directory, single neighborhood, etc.)
- Test split function ensures no neighborhood overlap
- Test data type conversions (numpy → torch)

**Fixtures**: Create minimal test DataFrames matching Parquet format

### Integration Tests

**Scope**: Test with actual Parquet files

**Requirements**:
- Load real data files
- Verify dataset can iterate through all graphs
- Verify splits work with real neighborhood data
- Test quick test script with real data

### Edge Cases

**Specific edge cases to test:**
1. Zero neighbors (empty edge_index and edge_attr)
2. Single neighbor
3. Very large neighbor count (>1000)
4. Missing columns in DataFrame
5. Invalid data types in DataFrame
6. Empty DataFrame
7. Single neighborhood (all points in one split)
8. All neighborhoods in train (no val/test)

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Check Python syntax
python -m py_compile src/training/dataset.py scripts/test_training_feasibility.py

# Run ruff linter (if available)
ruff check src/training/dataset.py scripts/test_training_feasibility.py

# Run black formatter check (if available)
black --check src/training/dataset.py scripts/test_training_feasibility.py
```

### Level 2: Import Validation

```bash
# Test imports work
python -c "from src.training.dataset import SpatialGraphDataset, load_features_from_directory, create_train_val_test_splits; print('Imports OK')"

# Test device utility
python -c "from src.training.dataset import get_device; print(f'Device: {get_device()}')"
```

### Level 3: Data Validation

```bash
# Run enhanced validation script
python scripts/validate_feature_output.py data/processed/features/compliant/paris_rive_gauche/target_points.parquet

# Test loading function
python -c "from src.training.dataset import load_features_from_directory; df = load_features_from_directory('data/processed/features/compliant'); print(f'Loaded {len(df)} rows')"
```

### Level 4: Dataset Functionality

```bash
# Test dataset with small subset
python -c "
from src.training.dataset import SpatialGraphDataset, load_features_from_directory
import pandas as pd
df = load_features_from_directory('data/processed/features/compliant')
ds = SpatialGraphDataset(df[:10])
print(f'Dataset length: {len(ds)}')
data = ds[0]
print(f'Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges, target shape: {data.y.shape}')
"

# Test splits
python -c "
from src.training.dataset import load_features_from_directory, create_train_val_test_splits
df = load_features_from_directory('data/processed/features/compliant')
train, val, test = create_train_val_test_splits(df)
print(f'Train: {len(train)} rows, {train[\"neighborhood_name\"].nunique()} neighborhoods')
print(f'Val: {len(val)} rows, {val[\"neighborhood_name\"].nunique()} neighborhoods')
print(f'Test: {len(test)} rows, {test[\"neighborhood_name\"].nunique()} neighborhoods')
# Verify no overlap
train_neighs = set(train['neighborhood_name'].unique())
val_neighs = set(val['neighborhood_name'].unique())
test_neighs = set(test['neighborhood_name'].unique())
assert len(train_neighs & val_neighs) == 0, 'Train/Val overlap!'
assert len(train_neighs & test_neighs) == 0, 'Train/Test overlap!'
assert len(val_neighs & test_neighs) == 0, 'Val/Test overlap!'
print('No neighborhood overlap - OK')
"
```

### Level 5: Unit Tests

```bash
# Run dataset unit tests
pytest tests/unit/test_dataset.py -v

# Run with coverage
pytest tests/unit/test_dataset.py --cov=src.training.dataset --cov-report=term-missing
```

### Level 6: Quick Test Script

```bash
# Run feasibility test with small subset
python scripts/test_training_feasibility.py --max_graphs 50

# Run with default settings
python scripts/test_training_feasibility.py
```

### Level 7: Configuration Validation

```bash
# Verify config loads correctly
python -c "
from src.utils.config import load_config
config = load_config()
print(f'Mixed precision: {config[\"training\"][\"use_mixed_precision\"]}')
print(f'Batch size: {config[\"training\"][\"batch_size\"]}')
print(f'Gradient accumulation: {config[\"training\"].get(\"gradient_accumulation_steps\", 1)}')
"
```

---

## ACCEPTANCE CRITERIA

- [ ] Enhanced validation script validates tensor conversion and outputs summary statistics
- [ ] `SpatialGraphDataset` class loads data and builds correct graph structures
- [ ] Dataset handles edge cases (zero neighbors, variable sizes) correctly
- [ ] `load_features_from_directory()` loads all Parquet files from directory tree
- [ ] `create_train_val_test_splits()` creates splits with no neighborhood overlap
- [ ] Configuration includes M2 optimization settings (mixed precision, gradient accumulation)
- [ ] `set_random_seeds()` supports MPS (no errors when MPS available)
- [ ] Device selection utility returns MPS device when available
- [ ] Quick test script runs successfully and outputs feasibility report
- [ ] All unit tests pass with 80%+ coverage
- [ ] All validation commands execute without errors
- [ ] Code follows project conventions (logging, error handling, docstrings)
- [ ] No regressions in existing functionality

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] Enhanced validation script works with real data files
- [ ] Dataset class loads and builds graphs correctly
- [ ] Utility functions work with real data
- [ ] Configuration updates load correctly
- [ ] MPS support added to random seed function
- [ ] Quick test script runs and outputs meaningful results
- [ ] All unit tests pass
- [ ] All validation commands executed successfully
- [ ] No linting or syntax errors
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability

---

## NOTES

### Design Decisions

1. **No file-based caching**: Building graphs on-the-fly from Parquet is fast enough, avoids disk space issues
2. **Neighborhood-level splits**: Prevents spatial leakage - all points from a neighborhood go to same split
3. **MPS support**: M2 MacBook Air has unified memory and MPS backend - enables GPU acceleration
4. **Mixed precision**: FP16 reduces memory usage by ~50% with minimal accuracy impact
5. **Gradient accumulation**: Allows larger effective batch size without increasing memory

### Performance Considerations

- **Memory**: With batch_size=16, mixed precision, expect ~500MB-1GB peak memory usage
- **Speed**: MPS should provide 2-5x speedup over CPU on M2
- **Thermal**: MacBook Air may throttle under sustained load - monitor temperatures

### Known Limitations

- MPS doesn't support all PyTorch operations - some may fall back to CPU
- Mixed precision may have numerical stability issues with very small models
- Gradient accumulation adds slight overhead but enables larger effective batches

### Future Enhancements

- Add data caching option for faster subsequent loads
- Add neighbor sampling for very large graphs (>1000 neighbors)
- Add progress bars for data loading
- Add data augmentation options

---

*Plan Version: 1.0*  
*Created: January 2025*  
*Project: AI4SI - 15-Minute City Service Gap Prediction Model*
