# Saved Training Data

This document describes all the data that is automatically saved during and after training.

## Saved Files Location

All training data is saved in the experiment directory:
```
experiments/runs/{timestamp}/
├── training_history.json          # Per-epoch metrics
├── training_summary.json          # Final summary
├── test_predictions.csv           # Per-sample test predictions
├── val_predictions.csv            # Per-sample validation predictions
├── test_evaluation_results.json   # Test set evaluation results
├── plots/                         # Visualization plots
│   ├── training_training_curves.png
│   ├── training_loss_comparison.png
│   ├── training_accuracy_curves.png
│   ├── neighbor_distribution.png
│   └── target_probability_distribution.png
└── logs/
    └── training.log               # Training logs
```

## 1. Training History (`training_history.json`)

**Location**: `experiments/runs/{timestamp}/training_history.json`

**Content**: Per-epoch metrics for all epochs

```json
[
  {
    "epoch": 1,
    "train_loss": 0.5234,
    "val_loss": 0.6123,
    "val_kl_divergence": 0.5891,
    "val_top1_accuracy": 0.4567,
    "val_top3_accuracy": 0.7890,
    "test_loss": 0.6456,
    "test_kl_divergence": 0.6234,
    "test_top1_accuracy": 0.4234,
    "test_top3_accuracy": 0.7654,
    "learning_rate": 0.001
  },
  ...
]
```

**Metrics Saved**:
- ✅ `epoch`: Epoch number
- ✅ `train_loss`: Training loss
- ✅ `val_loss`: Validation loss
- ✅ `val_kl_divergence`: Validation KL divergence
- ✅ `val_top1_accuracy`: Validation top-1 accuracy
- ✅ `val_top3_accuracy`: Validation top-3 accuracy
- ✅ `test_loss`: Test loss (evaluated every epoch)
- ✅ `test_kl_divergence`: Test KL divergence
- ✅ `test_top1_accuracy`: Test top-1 accuracy
- ✅ `test_top3_accuracy`: Test top-3 accuracy
- ✅ `learning_rate`: Learning rate at this epoch

**Usage**: 
- Load for plotting: `python scripts/plot_training_results.py --experiment-dir experiments/runs/{timestamp}`
- Analyze training progress
- Compare different training runs

## 2. Training Summary (`training_summary.json`)

**Location**: `experiments/runs/{timestamp}/training_summary.json`

**Content**: Final summary of training

```json
{
  "best_val_loss": 0.1234,
  "total_epochs": 50,
  "test_metrics": {
    "loss": 0.1456,
    "kl_divergence": 0.1345,
    "top1_accuracy": 0.6789,
    "top3_accuracy": 0.8901,
    "num_samples": 1234
  },
  "experiment_dir": "experiments/runs/1234567890",
  "checkpoint_path": "models/checkpoints/graph_transformer_best.pt",
  "plots_dir": "experiments/runs/1234567890/plots"
}
```

**Usage**: Quick reference for final model performance

## 3. Test Predictions (`test_predictions.csv`)

**Location**: `experiments/runs/{timestamp}/test_predictions.csv`

**Content**: Per-sample predictions for test set

**Columns**:
- `target_id`: Target point identifier
- `neighborhood_name`: Neighborhood name
- `predicted_class`: Predicted service category (0-7, argmax)
- `predicted_probs`: Predicted probabilities [8 categories] (as list)
- `target_class`: Target service category (0-7, argmax)
- `target_probs`: Target probabilities [8 categories] (as list)
- `loss`: Per-sample loss
- `kl_divergence`: Per-sample KL divergence

**Example**:
```csv
target_id,neighborhood_name,predicted_class,predicted_probs,target_class,target_probs,loss,kl_divergence
target_0,paris_rive_gauche,2,"[0.1, 0.2, 0.4, 0.1, 0.05, 0.05, 0.05, 0.05]",2,"[0.05, 0.15, 0.45, 0.12, 0.08, 0.08, 0.05, 0.02]",0.1234,0.1123
...
```

**Usage**:
- Analyze per-sample errors
- Identify problematic samples
- Compute custom metrics
- Error analysis by neighborhood

## 4. Validation Predictions (`val_predictions.csv`)

**Location**: `experiments/runs/{timestamp}/val_predictions.csv`

**Content**: Same structure as test predictions, but for validation set

**Usage**: 
- Validation set analysis
- Model debugging
- Understanding validation performance

## 5. Test Evaluation Results (`test_evaluation_results.json`)

**Location**: `experiments/runs/{timestamp}/test_evaluation_results.json`

**Content**: Aggregated test set evaluation with per-sample statistics

```json
{
  "split": "test",
  "metrics": {
    "loss": 0.1456,
    "kl_divergence": 0.1345,
    "top1_accuracy": 0.6789,
    "top3_accuracy": 0.8901,
    "num_samples": 1234
  },
  "per_sample_stats": {
    "mean_loss": 0.1456,
    "std_loss": 0.0234,
    "mean_kl_divergence": 0.1345,
    "std_kl_divergence": 0.0212,
    "accuracy": 0.6789
  }
}
```

**Usage**: Quick reference for test set statistics

## 6. Checkpoints

**Location**: `models/checkpoints/`

**Files**:
- `graph_transformer_best.pt`: Best model (saved when validation improves)
- `graph_transformer_epoch_{N}.pt`: Periodic checkpoints (every 10 epochs)

**Content**:
- Model weights
- Optimizer state
- Scheduler state
- Training history
- Epoch number
- Best validation loss
- Configuration

**Usage**: Resume training or load model for inference

## 7. Plots

**Location**: `experiments/runs/{timestamp}/plots/`

**Files**:
- `training_training_curves.png`: 4-panel training curves
- `training_loss_comparison.png`: Loss comparison plot
- `training_accuracy_curves.png`: Accuracy curves
- `neighbor_distribution.png`: Neighbor count distribution
- `target_probability_distribution.png`: Target probability analysis

**Usage**: Visual analysis of training progress

## Data Access Examples

### Load Training History
```python
import json
with open("experiments/runs/1234567890/training_history.json") as f:
    history = json.load(f)

# Get all validation losses
val_losses = [epoch["val_loss"] for epoch in history]
```

### Load Predictions
```python
import pandas as pd
predictions = pd.read_csv("experiments/runs/1234567890/test_predictions.csv")

# Analyze errors
errors = predictions[predictions["predicted_class"] != predictions["target_class"]]
print(f"Error rate: {len(errors) / len(predictions):.2%}")

# Analyze by neighborhood
neighborhood_accuracy = predictions.groupby("neighborhood_name").apply(
    lambda x: (x["predicted_class"] == x["target_class"]).mean()
)
```

### Load Evaluation Results
```python
import json
with open("experiments/runs/1234567890/test_evaluation_results.json") as f:
    results = json.load(f)

print(f"Test accuracy: {results['metrics']['top1_accuracy']:.2%}")
print(f"Mean loss: {results['per_sample_stats']['mean_loss']:.4f}")
```

## What's NOT Saved (by default)

- ❌ Training set predictions (can be added if needed)
- ❌ Per-epoch predictions (only final predictions saved)
- ❌ Model attention weights (can be added for analysis)
- ❌ Intermediate layer activations (can be added for debugging)

## Adding Custom Data Saving

To save additional data, modify `src/training/train.py`:

```python
# Save custom metrics
custom_metrics = {
    "custom_metric_1": compute_custom_metric(...),
    "custom_metric_2": compute_another_metric(...),
}

# Add to training history
training_history[-1].update(custom_metrics)

# Or save separately
with open(experiment_dir / "custom_metrics.json", "w") as f:
    json.dump(custom_metrics, f)
```

## Summary

✅ **Saved**: 
- Per-epoch metrics (train/val/test)
- Final predictions (test & validation sets)
- Evaluation results
- Training history
- Model checkpoints
- Visualization plots

✅ **Available for Analysis**:
- Training progress over epochs
- Per-sample predictions and errors
- Performance by neighborhood
- Loss and accuracy distributions
- Model checkpoints for inference

All data is saved automatically during training - no additional configuration needed!
