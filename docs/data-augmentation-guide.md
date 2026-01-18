# Data Augmentation Guide

This guide explains how to increase the effective size of your training dataset using two complementary approaches:

1. **Denser Grid Sampling** - Generate more target points by reducing grid spacing
2. **Data Augmentation** - Apply transforms during training to create variations

## Option 1: Denser Grid Sampling (More Real Data)

### How It Works

Currently, target points are sampled on a regular grid with `target_point_sampling_interval_meters: 100` (100m spacing). Reducing this spacing generates more target points.

### Implementation

**Step 1**: Update `config.yaml`:

```yaml
features:
  target_point_sampling_interval_meters: 50  # Reduce from 100m to 50m (4x more points)
  # or even denser:
  # target_point_sampling_interval_meters: 25  # 16x more points
```

**Step 2**: Re-run feature engineering to generate new target points:

```bash
python scripts/run_feature_engineering.py
```

**Trade-offs**:
- ✅ **Pros**: More real data, preserves spatial relationships
- ❌ **Cons**: Requires re-running feature engineering (~hours), more storage
- ⚠️ **Note**: Going below 25m may create excessive points and very similar samples

**Expected Increase**:
- 100m → 50m: **4x more points** (e.g., 490 → ~1960 points)
- 100m → 25m: **16x more points** (e.g., 490 → ~7840 points)

## Option 2: Data Augmentation (During Training)

### How It Works

Apply random transforms during training to create variations of each sample. No re-processing needed - works with existing data.

### Available Augmentations

#### 1. Feature Noise (`FeatureNoiseTransform`)
Adds Gaussian noise to node features to simulate measurement uncertainty.

```python
from src.training.data_augmentation import FeatureNoiseTransform

transform = FeatureNoiseTransform(noise_std=0.01)  # 1% noise
```

#### 2. Neighbor Subsampling (`NeighborSubsamplingTransform`)
Randomly keeps a subset of neighbors, creating different "views" of the same target point.

```python
from src.training.data_augmentation import NeighborSubsamplingTransform

transform = NeighborSubsamplingTransform(subsample_ratio=0.8)  # Keep 80% of neighbors
```

#### 3. Edge Attribute Noise (`EdgeAttributeNoiseTransform`)
Adds small perturbations to spatial distances and coordinates.

```python
from src.training.data_augmentation import EdgeAttributeNoiseTransform

transform = EdgeAttributeNoiseTransform(distance_noise_std=0.02)
```

#### 4. Composite Augmentation (Recommended)
Combine all augmentations with configurable probabilities:

```python
from src.training.data_augmentation import create_augmentation_transform

# Create augmentation with defaults
transform = create_augmentation_transform(
    enable_feature_noise=True,
    enable_neighbor_subsampling=True,
    enable_edge_noise=True,
    feature_noise_std=0.01,
    subsample_ratio=0.8,
    distance_noise_std=0.02,
)
```

### Implementation

**Step 1**: Import and create augmentation in `train.py` (or modify the training script):

```python
from src.training.data_augmentation import create_augmentation_transform

# Create augmentation transform (only for training)
train_transform = create_augmentation_transform()

# Create datasets with augmentation for training
train_dataset = SpatialGraphDataset(train_df, transform=train_transform)
val_dataset = SpatialGraphDataset(val_df)  # No augmentation for validation
test_dataset = SpatialGraphDataset(test_df)  # No augmentation for test
```

**Step 2**: The augmentation is applied automatically during training when DataLoader fetches samples.

**Trade-offs**:
- ✅ **Pros**: No re-processing needed, works immediately, helps generalization
- ❌ **Cons**: Doesn't add "new" information, may not help if model is already overfitting
- ⚠️ **Note**: Only apply to training set, not validation/test

### Effective Dataset Size

With augmentation, each sample can be seen multiple times in different variations:
- Feature noise: Creates infinite variations (continuous noise)
- Neighbor subsampling: Creates ~C(n, k) variations where n=neighbors, k=subsampled
- Edge noise: Creates infinite variations

In practice, with 490 samples and augmentation, you effectively see much more variety during training.

## Option 3: Combination Approach (Recommended)

Use both strategies for maximum data:

1. **Increase grid density** to 50m spacing (4x more real points)
2. **Add augmentation** during training (more variations per point)

This gives you both more real data and more training variety.

### Example Configuration

```yaml
# config.yaml
features:
  target_point_sampling_interval_meters: 50  # 4x denser grid
```

```python
# In training script
from src.training.data_augmentation import create_augmentation_transform

train_transform = create_augmentation_transform(
    feature_noise_std=0.01,
    subsample_ratio=0.8,
    distance_noise_std=0.02,
)

train_dataset = SpatialGraphDataset(train_df, transform=train_transform)
```

### Expected Results

With 50m spacing + augmentation:
- **Real points**: ~1960 (from 490)
- **Effective training samples**: Much higher due to augmentation variations
- **Generalization**: Better due to both more data and augmentation

## Recommendations

For your current 490-sample dataset:

1. **Quick win**: Add augmentation first (no re-processing needed)
   - Fast to implement
   - Can test immediately
   - May improve generalization

2. **Better long-term**: Reduce grid spacing to 50m and add augmentation
   - More real data (4x points)
   - Augmentation adds variety
   - Best generalization

3. **Avoid going too dense**: 25m spacing (16x points) may create too many very similar samples and increase overfitting risk.

## Testing Augmentation

To test if augmentation helps:

1. Train baseline without augmentation
2. Train with augmentation (same data)
3. Compare validation loss - augmentation should help if model is overfitting

If augmentation doesn't help or hurts performance, you may need:
- Stronger regularization (dropout, weight decay)
- Simpler model architecture
- More real data (denser grid)
