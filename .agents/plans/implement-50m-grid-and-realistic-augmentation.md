# Feature: 50m Grid Sampling + Realistic Data Augmentation

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Implement two complementary strategies to significantly increase training data volume and quality:

1. **Denser Grid Sampling (50m)**: Reduce target point sampling interval from 100m to 50m, generating ~4x more real data points (909 → ~3600 points across all neighborhoods)
2. **Realistic Data Augmentation**: Enhance existing augmentation transforms with domain-aware constraints that preserve urban planning realism (feature bounds, spatial relationships, demographic consistency)

This addresses the need for more training data while ensuring augmented samples remain realistic and useful for the 15-minute city service gap prediction task.

## User Story

As a **data scientist/researcher**
I want to **increase training data volume through denser grid sampling and realistic augmentation**
So that **the model has sufficient data to learn meaningful patterns while maintaining realism for urban planning applications**

## Problem Statement

- Current dataset has 909 points at 100m spacing, which may be insufficient for training robust graph transformer models
- Existing data augmentation lacks domain awareness, potentially creating unrealistic feature combinations
- Need to balance data volume increase with maintaining urban planning constraints and realism

## Solution Statement

1. **Backup existing 100m data** before making changes (safety first)
2. **Update configuration** to use 50m grid spacing, then re-run feature engineering pipeline
3. **Enhance augmentation transforms** with realistic constraints:
   - Feature noise with bounds (counts ≥ 0, ratios in [0,1], densities ≥ 0)
   - Adaptive neighbor subsampling that preserves spatial distribution
   - Feature correlation preservation for demographic consistency
   - Spatial-aware edge noise that maintains geometric relationships

## Feature Metadata

**Feature Type**: Enhancement (Data Pipeline + Augmentation)
**Estimated Complexity**: Medium-High
**Primary Systems Affected**: 
- `models/config.yaml` - Configuration update
- `src/training/data_augmentation.py` - Augmentation enhancements
- `data/processed/features/` - Data regeneration required
- `scripts/run_feature_engineering.py` - Re-execution needed
**Dependencies**: 
- Feature engineering pipeline (already implemented)
- Data augmentation module (already implemented, needs enhancement)

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `src/data/collection/feature_engineer.py` (lines 71-73, 94-194) - Why: How `sampling_interval_meters` is read from config and used in `generate_target_points()`. Shows grid generation algorithm.
- `models/config.yaml` (lines 29-34) - Why: Configuration file where `target_point_sampling_interval_meters` is defined. Must update to 50m.
- `src/training/data_augmentation.py` (lines 20-312) - Why: Current augmentation implementation. Must enhance with realistic constraints.
- `src/training/dataset.py` (lines 175-225) - Why: How features are loaded from directory. Need to ensure compatibility with new data.
- `scripts/run_feature_engineering.py` (lines 19-100) - Why: Script to re-run feature engineering after config change. Need to use with backup.
- `src/training/train.py` (lines 430-450) - Why: How training integrates augmentation. Must verify integration works correctly.
- `CURSOR.md` (lines 143-148) - Why: Feature structure documentation. 33 features breakdown: 17 demographics, 4 built form, 8 services, 4 walkability.
- `docs/data-augmentation-guide.md` (lines 1-189) - Why: Documentation on current augmentation. Should be updated after implementation.

### New Files to Create

- `scripts/backup_features.py` - Script to backup existing feature data before regeneration
- `.backup/features_100m/` - Backup directory (will be created by backup script)

### Files to Update

- `models/config.yaml` - Update `target_point_sampling_interval_meters` to 50
- `src/training/data_augmentation.py` - Enhance augmentation transforms with realistic constraints

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [PyTorch Geometric Data Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html)
  - Specific section: Data transforms
  - Why: Understand how to properly clone and modify Data objects
- [Feature Structure Documentation](CURSOR.md#feature-categories)
  - Specific section: Feature Categories (lines 143-148)
  - Why: Need exact feature indices and types for realistic bounds

### Patterns to Follow

**Naming Conventions:**
- Class names: PascalCase (e.g., `FeatureNoiseTransform`)
- Function names: snake_case (e.g., `create_augmentation_transform`)
- Private methods: Prefix with `_` (e.g., `_extract_demographic_features`)

**Error Handling:**
- Use logger for warnings and errors: `logger.warning()`, `logger.error()`
- Return original data on augmentation failure (graceful degradation)
- Validate inputs with clear error messages

**Logging Pattern:**
```python
from src.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("Message")
logger.warning("Warning")
logger.error("Error")
```

**Data Augmentation Pattern:**
- Always clone input Data object: `data_aug = data.clone()`
- Modify cloned object, return it
- Preserve required attributes: `y`, `target_id`, `neighborhood_name`
- Handle edge cases (empty edges, few neighbors, etc.)

**Feature Index Mapping:**
Based on CURSOR.md documentation:
- Indices 0-16: Demographics (17 features) - ratios in [0,1], densities ≥ 0
- Indices 17-20: Built Form (4 features) - densities ≥ 0, counts ≥ 0, ratios in [0,1]
- Indices 21-28: Services (8 features) - counts ≥ 0 (integers)
- Indices 29-32: Walkability (4 features) - densities ≥ 0, ratios in [0,1], binary [0,1]

**Configuration Pattern:**
- Config values read from `get_config()` in `src/utils/config.py`
- Feature params under `features:` section
- Default values provided if config missing

---

## IMPLEMENTATION PLAN

### Phase 1: Backup Existing Data

**Goal**: Safely backup current 100m data before making changes

**Tasks:**
- Create backup script that copies `data/processed/features/` to backup location
- Verify backup integrity
- Document backup location and restoration process

### Phase 2: Configuration Update

**Goal**: Update config to use 50m grid spacing

**Tasks:**
- Update `target_point_sampling_interval_meters` from 100 to 50 in config.yaml
- Document the change

### Phase 3: Feature Engineering Re-execution

**Goal**: Regenerate features with 50m spacing

**Tasks:**
- Re-run feature engineering pipeline
- Validate point count increase (~4x)
- Verify feature quality matches 100m data

### Phase 4: Realistic Augmentation Enhancement

**Goal**: Enhance augmentation transforms with realistic constraints

**Tasks:**
- Implement bounded feature noise (respects feature type constraints)
- Implement adaptive neighbor subsampling (preserves spatial distribution)
- Implement feature correlation preservation (demographic consistency)
- Enhance edge noise with spatial awareness
- Update augmentation factory function with new options

### Phase 5: Integration & Validation

**Goal**: Integrate enhanced augmentation and validate correctness

**Tasks:**
- Update training script if needed (should work automatically)
- Test augmentation on sample data
- Validate realistic constraints are enforced
- Update documentation

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### CREATE scripts/backup_features.py

- **IMPLEMENT**: Backup script for feature data
- **FUNCTIONALITY**:
  1. Copy `data/processed/features/` to `.backup/features_100m_{timestamp}/`
  2. Create backup manifest with file count, total size, timestamp
  3. Verify all files copied correctly (checksum or file count)
  4. Print backup summary
- **PATTERN**: Follow `scripts/run_feature_engineering.py` structure (argparse, logging)
- **IMPORTS**: `shutil`, `pathlib.Path`, `datetime`, `src.utils.logging`
- **VALIDATE**: `python scripts/backup_features.py` creates backup directory with all files

### UPDATE models/config.yaml

- **UPDATE**: Change `target_point_sampling_interval_meters` from 100 to 50
- **LOCATION**: Line 32 in `features:` section
- **OLD**: `target_point_sampling_interval_meters: 100  # Grid spacing for target points`
- **NEW**: `target_point_sampling_interval_meters: 50  # Grid spacing for target points (reduced from 100m for 4x more data)`
- **VALIDATE**: `grep "target_point_sampling_interval_meters" models/config.yaml` shows 50

### CREATE .backup/features_100m/ directory structure

- **CREATE**: Directory `.backup/features_100m/` (will be created by backup script)
- **NOTE**: This is created automatically by backup script, no manual creation needed
- **VALIDATE**: Directory exists after running backup script

### RUN backup script

- **EXECUTE**: `python scripts/backup_features.py`
- **EXPECTED**: 
  - Backup created in `.backup/features_100m_{timestamp}/`
  - All files from `data/processed/features/compliant/` copied
  - Backup manifest created
  - Summary printed
- **VALIDATE**: Check backup directory exists and contains all neighborhood subdirectories

### UPDATE src/training/data_augmentation.py - FeatureNoiseTransform with Bounds

- **UPDATE**: Enhance `FeatureNoiseTransform.__call__()` method
- **IMPLEMENT**: Add feature-type-aware bounds checking
- **ALGORITHM**:
  1. Define feature bounds map:
     - Indices 0-16 (Demographics): Some ratios [0,1], densities ≥ 0
       - Population density (0): ≥ 0
       - SES index (1): Typically [-2, 2] but allow wider range, clamp extreme values
       - Ratios (2-16): [0, 1] with small tolerance
     - Indices 17-20 (Built Form): Densities ≥ 0, counts ≥ 0, ratios [0,1]
       - Building density (17): ≥ 0
       - Building count (18): ≥ 0 (integer, but keep float)
       - Average levels (19): ≥ 0, reasonable max (e.g., 20)
       - Floor area per capita (20): ≥ 0
     - Indices 21-28 (Services): Counts ≥ 0 (integers, but keep float)
       - All 8 service counts: ≥ 0
     - Indices 29-32 (Walkability): Densities ≥ 0, ratios [0,1], binary [0,1]
       - Intersection density (29): ≥ 0
       - Average block length (30): > 0 (reasonable max, e.g., 500m)
       - Pedestrian street ratio (31): [0, 1]
       - Sidewalk presence (32): [0, 1] (binary, but keep continuous)
  2. Apply noise: `noise = torch.randn_like(x) * noise_std`
  3. Add noise: `x_noisy = x + noise`
  4. Apply bounds per feature type:
     - Clamp to [0, 1] for ratios
     - Clamp to ≥ 0 for counts/densities
     - Preserve SES index range (clamp extreme outliers to ±3)
  5. Return bounded noisy features
- **PATTERN**: Follow existing `FeatureNoiseTransform.__call__()` pattern (lines 48-72)
- **GOTCHA**: 
  - Don't modify original data (clone first)
  - Handle both target and neighbor nodes separately
  - Preserve feature dtype (float32)
  - Some features are already normalized, respect existing ranges
- **VALIDATE**: Create test that verifies augmented features respect bounds:
  ```python
  transform = FeatureNoiseTransform(noise_std=0.05)
  data_aug = transform(data)
  assert (data_aug.x >= 0).all()  # All features non-negative where required
  assert (data_aug.x[:, 2:16] <= 1.0 + 1e-6).all()  # Ratios in [0,1]
  ```

### UPDATE src/training/data_augmentation.py - NeighborSubsamplingTransform with Spatial Stratification

- **UPDATE**: Enhance `NeighborSubsamplingTransform.__call__()` method
- **IMPLEMENT**: Distance-based stratified sampling instead of random
- **ALGORITHM**:
  1. Calculate distance bands from edge attributes (euclidean_distance is index 2):
     - Band 1: 0-300m (close neighbors)
     - Band 2: 300-600m (medium neighbors)
     - Band 3: 600-900m (far neighbors)
     - Band 4: 900-1200m (distant neighbors)
  2. Group neighbors by distance band
  3. Calculate proportional sampling per band:
     - `num_keep_per_band = max(1, int(band_size * subsample_ratio))`
     - Ensure at least 1 neighbor per band (if band has neighbors)
  4. Sample from each band independently
  5. Combine selected neighbors maintaining spatial diversity
- **PATTERN**: Follow existing `NeighborSubsamplingTransform.__call__()` pattern (lines 102-162)
- **GOTCHA**:
  - Some graphs may have neighbors only in some bands (handle gracefully)
  - Maintain minimum neighbor count overall
  - Preserve edge attributes when subsampling
- **VALIDATE**: Create test that verifies spatial distribution:
  ```python
  transform = NeighborSubsamplingTransform(subsample_ratio=0.8)
  data_aug = transform(data)
  # Check that neighbors span multiple distance bands
  edge_distances = data_aug.edge_attr[:, 2]  # euclidean_distance
  assert len(torch.unique(edge_distances // 300)) >= 2  # At least 2 bands represented
  ```

### ADD src/training/data_augmentation.py - FeatureCorrelationPreservationTransform (Optional Enhancement)

- **CREATE**: New transform class `FeatureCorrelationPreservationTransform`
- **IMPLEMENT**: Preserve feature group correlations
- **ALGORITHM**:
  1. Group features into correlated sets:
     - Demographics group (indices 0-16): Apply shared random factor
     - Built form group (indices 17-20): Apply shared random factor
     - Services group (indices 21-28): Apply shared random factor (if desired)
     - Walkability group (indices 29-32): Apply shared random factor
  2. Generate per-group random factors: `group_factor = 1.0 + noise_std * torch.randn()`
  3. Multiply entire group by same factor: `x[group_indices] *= group_factor`
  4. Re-apply bounds after correlation preservation
- **NOTE**: This is optional - can skip if Phase 4.1 and 4.2 are sufficient
- **PATTERN**: Follow existing transform pattern (inherit from base, implement `__call__`)
- **VALIDATE**: Verify group correlations are preserved better than independent noise

### UPDATE src/training/data_augmentation.py - EdgeAttributeNoiseTransform with Spatial Awareness

- **UPDATE**: Enhance `EdgeAttributeNoiseTransform.__call__()` method
- **IMPLEMENT**: Spatial relationship preservation
- **ALGORITHM**:
  1. Apply noise to distances with scaling:
     - Larger distances get larger absolute noise (but same relative noise)
     - `distance_noise = distance * noise_std * torch.randn()`
  2. Ensure network_distance ≥ euclidean_distance (physical constraint):
     - After adding noise, clamp: `network_dist = max(network_dist, euclidean_dist)`
  3. Scale dx, dy proportionally when distances change:
     - Calculate scale factor: `scale = new_euclidean / old_euclidean`
     - Apply: `dx_new = dx * scale`, `dy_new = dy * scale`
  4. Keep edge attributes in valid range [0, 1] (already normalized)
- **PATTERN**: Follow existing `EdgeAttributeNoiseTransform.__call__()` pattern (lines 190-215)
- **GOTCHA**:
  - Edge attributes are already normalized to [0, 1] in dataset loading
  - Don't break normalization range
  - Preserve relative spatial relationships
- **VALIDATE**: Test that edge attributes remain valid:
  ```python
  transform = EdgeAttributeNoiseTransform(distance_noise_std=0.05)
  data_aug = transform(data)
  assert (data_aug.edge_attr >= 0).all()  # All non-negative
  assert (data_aug.edge_attr <= 1.0 + 1e-6).all()  # Within normalized range
  assert (data_aug.edge_attr[:, 3] >= data_aug.edge_attr[:, 2]).all()  # network >= euclidean
  ```

### UPDATE src/training/data_augmentation.py - create_augmentation_transform function

- **UPDATE**: Enhance factory function with new parameters
- **IMPLEMENT**: Add options for enhanced augmentation modes
- **ALGORITHM**:
  1. Add parameters:
     - `use_bounded_noise: bool = True` - Use bounded feature noise
     - `use_spatial_subsampling: bool = True` - Use spatial-stratified neighbor subsampling
     - `use_spatial_edge_noise: bool = True` - Use spatially-aware edge noise
     - `use_correlation_preservation: bool = False` - Use correlation preservation (optional)
  2. Create appropriate transforms based on flags
  3. Combine into CompositeAugmentation
- **PATTERN**: Follow existing `create_augmentation_transform()` pattern (lines 269-311)
- **VALIDATE**: Test factory function creates correct transforms:
  ```python
  transform = create_augmentation_transform(
      use_bounded_noise=True,
      use_spatial_subsampling=True,
      use_spatial_edge_noise=True
  )
  assert transform is not None
  assert len(transform.transforms) >= 3
  ```

### RUN feature engineering with 50m spacing

- **EXECUTE**: `python scripts/run_feature_engineering.py --all --force`
- **EXPECTED**: 
  - All neighborhoods processed with 50m spacing
  - Point count increases ~4x (909 → ~3600 points)
  - Processing time increases proportionally (~4x longer)
- **MONITOR**: 
  - Check logs for processing progress
  - Verify no errors during processing
  - Confirm point counts match expectations
- **VALIDATE**: 
  - Count total points: `python -c "from src.training.dataset import load_features_from_directory; df = load_features_from_directory('data/processed/features/compliant'); print(f'Total points: {len(df)}')"`
  - Should show ~3600 points (4x increase from 909)

### VALIDATE 50m data quality

- **CREATE**: Validation script or manual check
- **CHECKS**:
  1. Point count per neighborhood matches expectations (~4x)
  2. Feature statistics similar to 100m data (mean, std)
  3. No NaN or inf values in features
  4. Target probability vectors sum to ~1.0
  5. Neighbor counts reasonable (not too high/low)
- **VALIDATE**: `python -c "from src.training.dataset import load_features_from_directory; df = load_features_from_directory('data/processed/features/compliant'); print(df.describe())"`

### TEST enhanced augmentation on sample data

- **CREATE**: Test script or use Python REPL
- **TEST**:
  1. Load sample data: `from src.training.dataset import SpatialGraphDataset, load_features_from_directory; from src.training.data_augmentation import create_augmentation_transform; df = load_features_from_directory('data/processed/features/compliant'); dataset = SpatialGraphDataset(df.head(10))`
  2. Create augmented dataset: `aug = create_augmentation_transform(); dataset_aug = SpatialGraphDataset(df.head(10), transform=aug)`
  3. Check bounds: `data = dataset_aug[0]; print((data.x >= 0).all(), (data.x <= 1.1).all())`
  4. Check spatial distribution: `edge_dists = data.edge_attr[:, 2]; print(torch.unique(edge_dists // 0.3))`
- **VALIDATE**: All constraints respected, no errors

### UPDATE docs/data-augmentation-guide.md

- **UPDATE**: Document enhanced augmentation features
- **ADD**:
  1. Section on realistic augmentation constraints
  2. Examples of bounded noise
  3. Spatial stratification explanation
  4. Best practices for augmentation parameters
- **PATTERN**: Follow existing documentation structure
- **VALIDATE**: Documentation is clear and accurate

### INTEGRATION TEST - Training with new data and augmentation

- **EXECUTE**: Quick test training run
- **COMMAND**: `python scripts/train_graph_transformer.py --quick-test`
- **EXPECTED**: 
  - Training starts successfully
  - Augmentation applied correctly (check logs)
  - No errors during training
  - Validation loss decreases
- **MONITOR**:
  - Check that augmented data respects bounds (periodic sampling)
  - Verify training time is acceptable
  - Confirm no memory issues with larger dataset
- **VALIDATE**: Training completes at least 1 epoch without errors

---

## TESTING STRATEGY

### Unit Tests

**Scope**: Test individual augmentation transforms

**Required Tests**:
1. `test_feature_noise_bounds()` - Verify feature noise respects bounds for each feature type
2. `test_spatial_subsampling_distribution()` - Verify neighbor subsampling maintains spatial diversity
3. `test_edge_noise_constraints()` - Verify edge noise preserves spatial relationships
4. `test_augmentation_preserves_metadata()` - Verify augmentation doesn't lose required attributes

**Test Location**: `tests/unit/test_data_augmentation.py` (create if doesn't exist)

**Pattern**: Follow existing test patterns in `tests/unit/test_*.py`

### Integration Tests

**Scope**: Test augmentation with real dataset

**Required Tests**:
1. `test_augmentation_with_real_data()` - Load real data, apply augmentation, verify constraints
2. `test_training_with_augmentation()` - Quick training loop with augmented data

**Test Location**: `tests/integration/test_augmentation_integration.py` (create if doesn't exist)

### Edge Cases

**Test Cases**:
1. Graph with very few neighbors (< min_neighbors)
2. Graph with all neighbors in single distance band
3. Graph with extreme feature values (very high/low)
4. Empty edge attributes (should not crash)
5. Graph with only target node (no neighbors)

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
# Linting
ruff check src/training/data_augmentation.py scripts/backup_features.py

# Formatting
black --check src/training/data_augmentation.py scripts/backup_features.py

# Type checking (if mypy configured)
mypy src/training/data_augmentation.py
```

### Level 2: Unit Tests

```bash
# Run augmentation unit tests
pytest tests/unit/test_data_augmentation.py -v

# Run all unit tests (regression check)
pytest tests/unit/ -v
```

### Level 3: Integration Tests

```bash
# Run integration tests
pytest tests/integration/test_augmentation_integration.py -v
```

### Level 4: Manual Validation

```bash
# 1. Verify backup exists
ls -lh .backup/features_100m_*/

# 2. Verify config updated
grep "target_point_sampling_interval_meters" models/config.yaml

# 3. Verify feature count increased
python -c "from src.training.dataset import load_features_from_directory; df = load_features_from_directory('data/processed/features/compliant'); print(f'Points: {len(df)}')"

# 4. Test augmentation on sample
python -c "
from src.training.dataset import SpatialGraphDataset, load_features_from_directory
from src.training.data_augmentation import create_augmentation_transform
df = load_features_from_directory('data/processed/features/compliant')
aug = create_augmentation_transform()
dataset = SpatialGraphDataset(df.head(5), transform=aug)
data = dataset[0]
print(f'Features shape: {data.x.shape}')
print(f'Features bounds: [{data.x.min():.3f}, {data.x.max():.3f}]')
print(f'All features non-negative: {(data.x >= 0).all()}')
print('✓ Augmentation working')
"

# 5. Quick training test
python scripts/train_graph_transformer.py --quick-test
```

### Level 5: Data Quality Validation

```bash
# Verify point count per neighborhood
python -c "
from src.training.dataset import load_features_from_directory
df = load_features_from_directory('data/processed/features/compliant')
print('Points per neighborhood:')
print(df.groupby('neighborhood_name').size())
print(f'Total: {len(df)}')
"

# Verify feature statistics
python -c "
from src.training.dataset import SpatialGraphDataset, load_features_from_directory
import numpy as np
df = load_features_from_directory('data/processed/features/compliant')
dataset = SpatialGraphDataset(df)
features = np.stack([data.x[0].numpy() for data in dataset[:100]])
print('Feature statistics (first 100 samples):')
print(f'Mean: {features.mean(axis=0)[:5]}')
print(f'Std: {features.std(axis=0)[:5]}')
print(f'Min: {features.min(axis=0)[:5]}')
print(f'Max: {features.max(axis=0)[:5]}')
"
```

---

## ACCEPTANCE CRITERIA

- [ ] Backup script successfully creates backup of 100m data
- [ ] Config file updated to 50m spacing
- [ ] Feature engineering completes successfully with ~4x point increase
- [ ] Enhanced augmentation respects feature bounds (counts ≥ 0, ratios [0,1])
- [ ] Spatial subsampling maintains distance band diversity
- [ ] Edge noise preserves spatial relationships (network ≥ euclidean)
- [ ] All unit tests pass
- [ ] Integration tests verify augmentation works with real data
- [ ] Quick training test completes without errors
- [ ] Documentation updated with new augmentation features
- [ ] No regressions in existing functionality
- [ ] Data quality validation confirms ~4x increase with similar statistics

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Backup created and verified
- [ ] Config updated to 50m
- [ ] Feature engineering re-run completed (~3600 points)
- [ ] Enhanced augmentation implemented with bounds
- [ ] All validation commands executed successfully
- [ ] Full test suite passes (unit + integration)
- [ ] No linting or type checking errors
- [ ] Manual validation confirms augmentation respects constraints
- [ ] Training test confirms compatibility
- [ ] Documentation updated
- [ ] Acceptance criteria all met

---

## NOTES

### Design Decisions

1. **Feature Bounds Implementation**: Using per-feature-type bounds rather than per-index for maintainability. Feature indices may change but types are stable.

2. **Spatial Stratification**: Distance bands (0-300m, 300-600m, etc.) chosen to align with urban planning scales. Alternative: Use quartiles or statistical bands.

3. **Correlation Preservation**: Made optional (default False) because it may reduce augmentation effectiveness. Can be enabled if needed.

4. **Backup Strategy**: Using timestamped backups rather than versioned. Simple and sufficient for single-user development. For team use, consider version control.

### Performance Considerations

- **50m Data Size**: ~40MB (4x from 10MB), still manageable
- **Feature Engineering Time**: ~4x longer (~4-8 hours for all neighborhoods)
- **Training Time**: Per-epoch time increases ~4x, but epochs needed may decrease
- **Augmentation Overhead**: Minimal (~1-2% per sample)

### Risks & Mitigations

- **Risk**: Invalid augmented features break training
  - **Mitigation**: Extensive bounds checking, validation tests
- **Risk**: Spatial subsampling removes critical neighbors
  - **Mitigation**: Minimum neighbors per band, overall minimum count
- **Risk**: 50m data has too many similar samples
  - **Mitigation**: Realistic augmentation adds variation, monitor overfitting
- **Risk**: Backup restoration needed but forgotten
  - **Mitigation**: Document backup location clearly, include restoration instructions

### Future Enhancements

- Add temporal augmentation (day/night, weekday/weekend variations)
- Add population density-based augmentation scaling
- Consider PCA-based augmentation for feature correlation preservation
- Add augmentation strength scheduling (stronger early, weaker later in training)
