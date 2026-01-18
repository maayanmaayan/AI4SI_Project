# Hyperparameter Sweep Guide

This guide explains how to use the hyperparameter sweep script and quick test mode.

## Quick Test Mode

For quick testing with a small dataset (3 small neighborhoods with <50 points each):

```bash
# Single model training with quick test
python scripts/train_graph_transformer.py --quick-test

# Hyperparameter sweep with quick test
python scripts/hyperparameter_sweep.py --quick-test
```

**What it does:**
- Filters to neighborhoods with <50 points each
- Uses maximum 3 neighborhoods
- Much faster for initial testing (~10-30 minutes instead of hours)

## Hyperparameter Sweep

Train all 7 model configurations from the PRD:

```bash
# Full sweep (all configurations)
python scripts/hyperparameter_sweep.py

# Quick test sweep (faster, for testing)
python scripts/hyperparameter_sweep.py --quick-test

# Custom config and features directory
python scripts/hyperparameter_sweep.py --config models/config.yaml --features-dir data/processed/features/compliant
```

## Model Configurations

The sweep trains 7 configurations:

| Model | Learning Rate | Layers | Temperature | Description |
|-------|---------------|--------|-------------|-------------|
| 1 | 0.001 | 3 | 200 | Baseline (current config) |
| 2 | 0.0005 | 3 | 200 | Lower learning rate |
| 3 | 0.002 | 3 | 200 | Higher learning rate |
| 4 | 0.001 | 2 | 200 | Fewer layers (shallow) |
| 5 | 0.001 | 4 | 200 | More layers (deeper) |
| 6 | 0.001 | 3 | 150 | Lower temperature (sharper) |
| 7 | 0.001 | 3 | 250 | Higher temperature (smoother) |

## Output Structure

```
experiments/runs/sweep_{timestamp}/
├── model_1_baseline/
│   ├── training_history.json
│   ├── test_predictions.csv
│   ├── plots/
│   └── ...
├── model_2_lower_lr/
│   └── ...
├── ...
├── sweep_results.json          # Summary of all models
└── plots/
    ├── hyperparameter_comparison_best_val_loss.png
    ├── hyperparameter_comparison_test_loss.png
    ├── hyperparameter_comparison_test_kl_divergence.png
    └── hyperparameter_comparison_test_top1_accuracy.png
```

## Results

The sweep script:
1. Trains each configuration
2. Saves results in separate directories
3. Identifies the best model (lowest validation loss)
4. Generates comparison plots
5. Saves summary JSON with all results

## Time Estimates

- **Quick test mode**: ~10-30 minutes per model (7 models = ~1-3.5 hours)
- **Full dataset**: ~1-2 hours per model (7 models = ~7-14 hours)

## Best Practices

1. **Start with quick test**: Run `--quick-test` first to verify everything works
2. **Run overnight**: Full sweep takes 7-14 hours, perfect for overnight runs
3. **Check results**: Review `sweep_results.json` to see which model performed best
4. **Compare plots**: Check comparison plots in `plots/` directory

## Single Model Training

For training a single model (not a sweep):

```bash
# Full dataset
python scripts/train_graph_transformer.py

# Quick test
python scripts/train_graph_transformer.py --quick-test

# Resume from checkpoint
python scripts/train_graph_transformer.py --resume-from models/checkpoints/graph_transformer_best.pt
```

## Example Workflow

1. **Quick test single model** (verify pipeline works):
   ```bash
   python scripts/train_graph_transformer.py --quick-test
   ```

2. **Quick test sweep** (compare all configs quickly):
   ```bash
   python scripts/hyperparameter_sweep.py --quick-test
   ```

3. **Full sweep** (final results):
   ```bash
   python scripts/hyperparameter_sweep.py
   ```

4. **Train best model on full dataset** (if needed):
   ```bash
   # Edit config.yaml with best hyperparameters
   python scripts/train_graph_transformer.py
   ```
