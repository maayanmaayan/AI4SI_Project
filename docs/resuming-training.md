# Resuming Training from Checkpoint

This guide explains how to resume training from a checkpoint if you need to stop training and continue later.

## How Checkpointing Works

The training script automatically saves checkpoints in two ways:

1. **Best Model Checkpoint**: Saved whenever validation loss improves
   - Location: `models/checkpoints/graph_transformer_best.pt`
   - Contains: Model weights, optimizer state, scheduler state, training history, epoch number, best validation loss

2. **Periodic Checkpoints**: Saved every 10 epochs
   - Location: `models/checkpoints/graph_transformer_epoch_{N}.pt`
   - Useful if you want to resume from a specific epoch (not just the best one)

## What Gets Saved

Each checkpoint contains:
- ✅ Model state (weights)
- ✅ Optimizer state (momentum, etc.)
- ✅ Learning rate scheduler state
- ✅ Training history (all previous epochs)
- ✅ Current epoch number
- ✅ Best validation loss so far
- ✅ Configuration used

## How to Resume Training

### Method 1: Resume from Best Checkpoint

```bash
python scripts/train_graph_transformer.py \
    --resume-from models/checkpoints/graph_transformer_best.pt
```

### Method 2: Resume from Specific Epoch

```bash
python scripts/train_graph_transformer.py \
    --resume-from models/checkpoints/graph_transformer_epoch_50.pt
```

### Method 3: Resume with Custom Experiment Directory

```bash
python scripts/train_graph_transformer.py \
    --resume-from models/checkpoints/graph_transformer_best.pt \
    --experiment-dir experiments/runs/previous_run
```

## What Happens When Resuming

1. **Model weights** are restored to the checkpoint state
2. **Optimizer state** is restored (including momentum, Adam states, etc.)
3. **Learning rate scheduler** state is restored (continues from where it left off)
4. **Training history** is loaded (so plots will include previous epochs)
5. **Training continues** from `epoch + 1` (next epoch after checkpoint)
6. **Best validation loss** is restored (early stopping continues correctly)

## Example Scenario

### Starting Training
```bash
# Start training
python scripts/train_graph_transformer.py

# Training runs for 30 epochs, then you stop it (Ctrl+C)
# Best checkpoint saved at epoch 25 (best validation loss)
```

### Resuming Training
```bash
# Resume from best checkpoint
python scripts/train_graph_transformer.py \
    --resume-from models/checkpoints/graph_transformer_best.pt

# Training continues from epoch 26
# Training history from epochs 1-30 is preserved
# Plots will show complete training curve including resumed epochs
```

## Important Notes

### ✅ What Works
- Resume from any checkpoint (best or periodic)
- Training history is preserved and appended to
- Learning rate schedule continues correctly
- Early stopping patience counter resets (starts fresh)
- Plots will include all epochs (original + resumed)

### ⚠️ Considerations
- **Configuration**: The checkpoint contains the config used when it was saved. If you want to change config, you'll need to manually edit the checkpoint or start fresh.
- **Data splits**: Make sure you use the same data splits (same random seed) when resuming, otherwise validation/test sets will be different.
- **Device**: Checkpoints are saved in a device-agnostic way, so you can resume on a different device (CPU/GPU/MPS).

### 🔄 Starting Fresh vs Resuming

**Start Fresh** (if you want to):
- Change hyperparameters
- Use different data splits
- Change model architecture
- Start from scratch

**Resume** (if you want to):
- Continue training that was interrupted
- Train for more epochs
- Recover from a crash
- Continue with same settings

## Troubleshooting

### Checkpoint Not Found
```
FileNotFoundError: Checkpoint file not found
```
**Solution**: Check the path. Default location is `models/checkpoints/graph_transformer_best.pt`

### Mismatched Model Architecture
```
RuntimeError: Error(s) in loading state_dict
```
**Solution**: You can't resume if you changed the model architecture. Start fresh or use the same architecture.

### Training History Missing
If you resume from an old checkpoint that doesn't have training history, it will start with an empty history. The new epochs will still be logged.

## Best Practices

1. **Regular Checkpoints**: The script saves periodic checkpoints every 10 epochs automatically
2. **Backup Checkpoints**: Consider copying important checkpoints to a backup location
3. **Document Checkpoints**: Note which checkpoint corresponds to which experiment/hyperparameters
4. **Monitor Disk Space**: Checkpoints can be large (~100-500MB depending on model size)

## Checkpoint File Structure

```python
checkpoint = {
    "epoch": 25,                    # Last completed epoch
    "model_state_dict": {...},      # Model weights
    "optimizer_state_dict": {...},  # Optimizer state
    "scheduler_state_dict": {...},  # LR scheduler state
    "best_val_loss": 0.1234,        # Best validation loss
    "config": {...},                # Configuration used
    "training_history": [...]       # List of all epoch metrics
}
```

## Example: Complete Resume Workflow

```bash
# 1. Start training
python scripts/train_graph_transformer.py
# ... training runs for 20 epochs, then you stop it

# 2. Check what checkpoints exist
ls -lh models/checkpoints/
# graph_transformer_best.pt
# graph_transformer_epoch_10.pt
# graph_transformer_epoch_20.pt

# 3. Resume from best checkpoint
python scripts/train_graph_transformer.py \
    --resume-from models/checkpoints/graph_transformer_best.pt

# 4. Training continues from epoch 21
# All previous training history is preserved
# Plots will show complete training curve
```
