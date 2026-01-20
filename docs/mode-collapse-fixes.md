# Mode Collapse Fixes: Preventing One-Hot Predictions

## Problem

The model was collapsing to one-hot predictions (predicting a single class with probability 1.0) despite having:
- Temperature scaling (T=2.0)
- Entropy regularization (weight=2.0, min_entropy=1.0)
- Class weights

## Root Causes Identified

1. **Label smoothing was disabled** (0.0) - This is a critical technique for preventing overconfidence
2. **Temperature might not be high enough** - T=2.0 may not be sufficient to flatten distributions
3. **Entropy regularization was too weak** - Linear penalty might not be aggressive enough
4. **No gradient clipping** - Logits could explode, leading to extreme softmax outputs
5. **No extra regularization on final layer** - The classifier layer could learn extreme weights

## Solutions Implemented

### 1. Enabled Label Smoothing (Critical Fix)
**Config Change**: `label_smoothing: 0.0 → 0.15`

**What it does**: Mixes target probability distributions with uniform distribution (15% uniform, 85% target). This prevents the model from becoming overconfident by forcing it to spread probability mass across classes.

**Why it works**: Even though targets are already smooth probability distributions, adding label smoothing further encourages the model to maintain uncertainty and prevents it from collapsing to one-hot predictions.

### 2. Increased Temperature Scaling
**Config Change**: `model.temperature: 10.0 → 4.0` (Note: config had 10.0 but code was using 2.0)

**What it does**: Divides logits by temperature before softmax. Higher temperature = flatter distribution.

**Why it works**: Temperature scaling directly controls the sharpness of softmax. With T=4.0, the model produces softer probability distributions, making it harder to achieve one-hot predictions.

### 3. Strengthened Entropy Regularization
**Config Changes**:
- `entropy_weight: 2.0 → 5.0` (increased penalty weight)
- `min_entropy: 1.0 → 1.5` (higher threshold)
- Added `entropy_penalty_type: "quadratic"` (more aggressive penalty)

**What it does**: 
- **Linear penalty**: `max(0, min_entropy - entropy)` - only penalizes when entropy < threshold
- **Quadratic penalty**: `(min_entropy - entropy)^2` when entropy < threshold - penalizes low entropy more aggressively

**Why it works**: Quadratic penalty grows quadratically as entropy decreases, making it much harder for the model to maintain low-entropy (one-hot) predictions. The higher weight (5.0) and threshold (1.5) further discourage mode collapse.

### 4. Added Gradient Clipping
**Config Change**: Added `training.max_grad_norm: 1.0`

**What it does**: Clips gradients to have maximum norm of 1.0, preventing gradient explosion.

**Why it works**: Exploding gradients can cause logits to grow unbounded, leading to extreme softmax outputs (near one-hot). Gradient clipping prevents this by keeping gradients in a reasonable range.

### 5. Extra Weight Decay on Final Layer
**Config Change**: Added `training.final_layer_weight_decay: 0.01`

**What it does**: Applies additional L2 regularization (0.01) specifically to the classifier layer weights, on top of the base weight decay (0.001).

**Why it works**: The final layer is most responsible for producing extreme logits. Extra regularization prevents the classifier weights from growing too large, which would lead to extreme softmax outputs.

### 6. Logit Magnitude Monitoring
**Code Change**: Added logging of logit statistics (max, min, range) in first batch of each epoch.

**What it does**: Helps detect if logits are exploding during training.

**Why it's useful**: If logits are growing unbounded, we'll see it in the logs and can adjust gradient clipping or learning rate.

## Expected Results

After these changes, you should see:

1. **More spread probability distributions**: Predictions should have non-zero probability across multiple classes
2. **Higher entropy**: Average prediction entropy should be closer to 1.5-2.0 (max entropy for 8 classes ≈ 2.08)
3. **Better validation metrics**: Model should learn more nuanced patterns rather than collapsing to single-class predictions
4. **Stable training**: Logits should remain in reasonable range (typically -10 to +10)

## Monitoring

Watch for these indicators in training logs:

- **Prediction distribution**: Should show non-zero probabilities across multiple classes
- **Entropy values**: Should stay above 1.5 (check if entropy regularization is working)
- **Logit magnitudes**: Should remain reasonable (not exploding to ±100)
- **Validation loss**: Should decrease more smoothly without sudden jumps

## Tuning Guide

If mode collapse persists, try:

1. **Increase label smoothing**: Try 0.2 or 0.25 (but not too high, as it may hurt learning)
2. **Increase temperature**: Try 5.0 or 6.0 (but too high may make predictions too uniform)
3. **Increase entropy weight**: Try 7.0 or 10.0 (but may slow learning)
4. **Increase min_entropy**: Try 1.8 or 2.0 (closer to max entropy)
5. **Tighten gradient clipping**: Try 0.5 or 0.8 (but too tight may prevent learning)
6. **Increase final layer weight decay**: Try 0.02 or 0.05

## References

- Label Smoothing: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
- Temperature Scaling: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- Entropy Regularization: Common technique in reinforcement learning and probabilistic models
- Gradient Clipping: Standard practice for training RNNs and transformers
