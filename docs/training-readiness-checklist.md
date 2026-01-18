# Training Readiness Checklist

## ✅ Completed Components

### Core Model Components
- ✅ **Model Architecture** (`src/training/model.py`)
  - SpatialGraphTransformer with TransformerConv layers
  - Node and edge encoders
  - Residual connections and layer normalization
  - Single linear classifier head
  - GELU activation
  - All tests passing (9/9)

- ✅ **Dataset Class** (`src/training/dataset.py`)
  - SpatialGraphDataset with star graph construction
  - Edge attributes [dx, dy, euclidean_dist, network_dist]
  - Handles variable-sized graphs
  - Train/val/test splits with neighborhood stratification
  - All tests passing (7/7)

- ✅ **Loss Function** (`src/training/loss.py`)
  - Distance-based KL divergence loss
  - Supports batch processing
  - Numerical stability
  - All tests passing (14/14)

- ✅ **Training Script** (`src/training/train.py`)
  - Full training loop with validation
  - Checkpointing (best model + periodic)
  - Early stopping
  - Learning rate scheduling
  - Mixed precision support
  - Resume from checkpoint
  - Comprehensive logging

- ✅ **Evaluation Metrics** (`src/evaluation/metrics.py`)
  - KL divergence
  - Top-k accuracy
  - Distance alignment
  - Comprehensive evaluation function

- ✅ **Plotting** (`src/evaluation/plotting.py`)
  - Training curves (4-panel)
  - Loss comparison
  - Accuracy curves
  - Neighbor distribution
  - Target probability distribution
  - All plots saved (not displayed)

- ✅ **Predictions Saving** (`src/evaluation/save_predictions.py`)
  - Per-sample predictions (CSV)
  - Evaluation results (JSON)
  - Test and validation sets

### Feature Engineering
- ✅ **Feature Engineer** (`src/data/collection/feature_engineer.py`)
  - Target point generation
  - Grid cell generation
  - Network distance filtering
  - 33 features per point
  - Target probability vector computation

- ✅ **Feature Engineering Script** (`scripts/run_feature_engineering.py`)
  - Command-line interface
  - Process single or all neighborhoods
  - Force re-processing option

### Configuration
- ✅ **Config File** (`models/config.yaml`)
  - Model architecture parameters
  - Training hyperparameters
  - Feature engineering parameters
  - Loss function parameters
  - Paths and directories

- ✅ **Dependencies** (`requirements.txt`)
  - torch-geometric>=2.3.0 ✅
  - All required packages listed

### Scripts
- ✅ **Training Script** (`scripts/train_graph_transformer.py`)
  - Command-line interface
  - Configurable paths
  - Resume from checkpoint
  - Experiment directory management

### Testing
- ✅ **Unit Tests** (34 tests passing)
  - Model tests (9/9)
  - Dataset tests (7/7)
  - Loss tests (14/14)
  - Integration tests (4/4)

### Documentation
- ✅ **Plotting Guide** (`docs/plotting-guide.md`)
- ✅ **Resuming Training** (`docs/resuming-training.md`)
- ✅ **Saved Training Data** (`docs/saved-training-data.md`)

## ⚠️ Prerequisites Before Training

### 1. Feature Engineering (REQUIRED)
You need to run feature engineering first to generate processed features:

```bash
# Process all compliant neighborhoods
python scripts/run_feature_engineering.py --all

# Or process a specific neighborhood
python scripts/run_feature_engineering.py --neighborhood "paris_rive_gauche"
```

**Expected output location:**
```
data/processed/features/compliant/{neighborhood_name}/target_points.parquet
```

**Check if ready:**
```bash
ls -la data/processed/features/compliant/
```

### 2. Verify Data Structure
Ensure processed features have the correct structure:
- `target_id`: Unique identifier
- `neighborhood_name`: Neighborhood name
- `target_features`: Array [33] of features
- `neighbor_data`: List of neighbor dicts with features and distances
- `target_prob_vector`: Array [8] of target probabilities
- `num_neighbors`: Number of neighbors

### 3. Configuration Check
Verify `models/config.yaml` has correct paths:
- `paths.data_root`: Should point to `./data`
- `paths.checkpoints_dir`: Where to save checkpoints
- `paths.experiments_dir`: Where to save experiment logs

## 🚀 Ready to Train!

Once feature engineering is complete, you can start training:

```bash
# Basic training
python scripts/train_graph_transformer.py

# With custom config
python scripts/train_graph_transformer.py --config models/config.yaml

# Resume from checkpoint
python scripts/train_graph_transformer.py --resume-from models/checkpoints/graph_transformer_best.pt
```

## 📊 What Happens During Training

1. **Data Loading**: Loads processed features from `data/processed/features/compliant/`
2. **Splits**: Creates train/val/test splits (70/15/15) by neighborhood
3. **Model Creation**: Initializes SpatialGraphTransformer from config
4. **Training Loop**: 
   - Trains for up to 100 epochs (or until early stopping)
   - Validates every epoch
   - Saves best model checkpoint
   - Saves periodic checkpoints every 10 epochs
   - Logs all metrics
5. **Evaluation**: Evaluates on test set and saves predictions
6. **Plotting**: Generates all visualization plots
7. **Results**: Saves everything to `experiments/runs/{timestamp}/`

## ✅ Final Checklist

Before starting training, verify:
- [ ] Feature engineering completed (`data/processed/features/compliant/` exists)
- [ ] At least one neighborhood processed
- [ ] Config file paths are correct
- [ ] Sufficient disk space for checkpoints and logs
- [ ] GPU/MPS available (optional but recommended)

## 🎯 Next Steps

1. **Run feature engineering** (if not done)
2. **Start training**: `python scripts/train_graph_transformer.py`
3. **Monitor progress**: Check `experiments/runs/{timestamp}/logs/training.log`
4. **View plots**: Check `experiments/runs/{timestamp}/plots/`
5. **Analyze results**: Check `experiments/runs/{timestamp}/test_predictions.csv`

---

**Status**: ✅ **READY FOR TRAINING** (once feature engineering is complete)
