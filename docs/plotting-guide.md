# Plotting Guide for Graph Transformer Training

This guide describes the plotting functionality available for visualizing training progress and model performance.

## Overview

The plotting system automatically generates visualizations during training and can also be used to analyze completed training runs. All plots are saved to disk (not displayed on screen) in the `plots/` directory or experiment-specific directories.

## Available Plots

### 1. Training Curves (4-panel plot)
**File**: `{experiment_name}_training_curves.png`

Shows four key metrics over training epochs:
- **Loss Curves**: Training, validation, and test loss
- **KL Divergence**: KL divergence for train/val/test sets
- **Top-1 Accuracy**: Accuracy curves for train/val/test sets
- **Learning Rate Schedule**: Learning rate changes over epochs

### 2. Loss Comparison
**File**: `{experiment_name}_loss_comparison.png`

Detailed comparison of training, validation, and test losses on a single plot. Useful for identifying overfitting or convergence issues.

### 3. Accuracy Curves
**File**: `{experiment_name}_accuracy_curves.png`

Shows top-1 and top-3 accuracy for validation and test sets. Helps track model performance improvements.

### 4. Neighbor Distribution
**File**: `neighbor_distribution.png`

Visualizes the distribution of neighbor counts per target point:
- Histogram showing frequency of different neighbor counts
- Box plot with statistics (mean, median, std, min, max)

This helps understand the graph structure and identify potential data issues.

### 5. Target Probability Distribution
**File**: `target_probability_distribution.png`

4-panel visualization of target probability vectors:
- **Mean probabilities per category**: Bar chart showing average probability for each service category
- **Box plot**: Distribution of probabilities for each category
- **Heatmap**: Sample of probability vectors (shows patterns across samples)
- **Max probability distribution**: Histogram of maximum category probabilities

### 6. Hyperparameter Comparison
**File**: `hyperparameter_comparison_{metric}.png`

Compares different hyperparameter settings (e.g., learning rates, batch sizes) on the same plot. Useful for hyperparameter tuning.

### 7. Spatial Predictions (Future)
**File**: `spatial_predictions.png`

Geographical visualization of model predictions colored by predicted service category. Useful for understanding spatial patterns in predictions.

## Automatic Plot Generation

Plots are automatically generated during training:

1. **During Training**: Training curves are saved after each epoch to the experiment directory
2. **After Training**: Complete set of plots is generated in `{experiment_dir}/plots/`
3. **Data Exploration**: Neighbor distribution and target probability plots are generated if features are available

## Manual Plot Generation

You can generate plots from existing training history:

```bash
# Generate plots from a specific experiment
python scripts/plot_training_results.py --experiment-dir experiments/runs/run_123456

# Generate plots with data exploration
python scripts/plot_training_results.py \
    --experiment-dir experiments/runs/run_123456 \
    --features-dir data/processed/features/compliant

# Compare multiple experiments
python scripts/plot_training_results.py \
    --compare-experiments \
    experiments/runs/run_123456 \
    experiments/runs/run_123789 \
    experiments/runs/run_124012
```

## Plot Locations

- **During Training**: `{experiment_dir}/plots/`
- **Manual Generation**: `plots/` (default) or specified directory
- **Experiment-specific**: Each experiment has its own plots directory

## Plot Configuration

Plots use:
- **DPI**: 300 (high resolution for publications)
- **Format**: PNG
- **Backend**: Agg (non-interactive, no display)
- **Style**: Clean, publication-ready with grid lines and legends

## Customization

To customize plots, modify `src/evaluation/plotting.py`:

```python
# Change figure size
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust as needed

# Change colors
ax.plot(..., color='red', linewidth=2)  # Customize colors and linewidths

# Change DPI
plt.savefig(save_path, dpi=150)  # Lower DPI for smaller files
```

## Example Use Cases

### 1. Monitor Training Progress
Check `{experiment_dir}/plots/training_training_curves.png` during training to see if the model is learning properly.

### 2. Compare Hyperparameters
Run multiple experiments with different learning rates, then use `--compare-experiments` to visualize which performs best.

### 3. Debug Data Issues
Use neighbor distribution plots to identify if some target points have too few or too many neighbors.

### 4. Analyze Model Predictions
Generate spatial prediction plots to see how the model's predictions are distributed geographically.

## Tips

1. **High DPI**: Plots are saved at 300 DPI for publication quality. Reduce if file size is a concern.
2. **Large Datasets**: Some plots sample data (e.g., 100 samples for probability distribution) to avoid memory issues.
3. **Comparison Plots**: Use consistent experiment naming for easier comparison (e.g., `lr_0.001`, `lr_0.01`).
4. **Plot Naming**: Experiment names are used in plot filenames, so use descriptive names.

## Troubleshooting

**Plots not generating?**
- Check that matplotlib is installed: `pip install matplotlib`
- Verify the plots directory is writable
- Check logs for error messages

**Plots are empty?**
- Ensure training history has data
- Check that metrics are being logged correctly
- Verify feature data is available for data exploration plots

**Memory issues?**
- Reduce sample size in plotting functions
- Generate plots separately instead of all at once
- Use lower DPI for faster generation
