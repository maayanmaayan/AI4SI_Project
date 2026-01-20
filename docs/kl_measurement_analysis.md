# KL Divergence Measurement & Uniformity Analysis

## Problem Statement

The model is predicting **very uniform distributions** (`['0.1387', '0.1173', '0.1318', ...]`) for all samples, even though each sample should have a **location-specific distribution** based on distances to services.

## KL Divergence Measurement Analysis

### Current Implementation

**How KL is computed:**
1. Per-sample KL divergence: `KL(target_i || predicted_i)` for each sample `i`
2. Loss per sample: `KL_i = sum_c(target_i[c] * log(target_i[c] / predicted_i[c]))`
3. Average across batch: `loss = mean(KL_i)` (reduction='batchmean')

**This is CORRECT** - KL divergence should be computed per-sample and then averaged.

### The Real Issue

**KL divergence itself is not the problem.** The issue is that **regularization terms are forcing uniformity** across all samples, independent of the target distribution.

## Regularization Components Analysis

### 1. Entropy Regularization (TOO STRONG)

**Current settings:**
- `entropy_weight: 5.0` ⚠️ **VERY HIGH**
- `min_entropy: 1.5` (max entropy for 8 classes ≈ 2.08)

**How it works:**
- Penalty: `max(0, 1.5 - entropy) * 5.0`
- If entropy < 1.5, adds large penalty (up to 7.5 = 1.5 * 5.0)
- This **forces all predictions to have high entropy (uniformity)**, regardless of target

**Problem:**
- Entropy regularization penalizes **ANY** low-entropy (peaked) prediction
- But targets might have **low entropy** (one dominant service category)
- The model learns to ignore targets and predict uniform distributions to avoid entropy penalty

**Impact:**
- Model predicts `entropy ≈ 2.08` (maximum, uniform) for all samples
- Even when target has `entropy ≈ 0.5` (very peaked, one dominant class)
- This causes **uniform predictions regardless of location**

### 2. MaxSup Penalty (FORCES UNIFORMITY)

**Current settings:**
- `maxsup_weight: 0.1`
- `maxsup_threshold: 0.0` (always apply)

**How it works:**
- Penalty: `max_logit * 0.1`
- Suppresses the maximum logit, preventing any class from dominating
- This **directly encourages uniformity** by penalizing peaked predictions

**Problem:**
- Penalizes **any** high confidence prediction
- Even if target distribution is peaked (one dominant class), model can't predict it
- Forces predictions to be spread out uniformly

**Impact:**
- With LogitNorm (norm=1.0), max logit ≈ 0.35-0.38
- MaxSup penalty ≈ 0.035-0.038 per sample
- This constant penalty encourages uniform distributions

### 3. Temperature Scaling (FLATTENS DISTRIBUTIONS)

**Current settings:**
- `temperature: 4.0` ⚠️ **VERY HIGH**
- Applied to logits: `logits = logits / 4.0`

**How it works:**
- Higher temperature → flatter softmax distribution
- With `T=4.0`, even large logit differences become small probability differences

**Problem:**
- Makes all predictions more uniform by design
- Reduces model's ability to express confidence in specific classes
- Combines with LogitNorm to create very constrained predictions

**Impact:**
- Even if model wants to predict peaked distribution, temperature scaling flattens it
- Combined with entropy regularization, creates uniform predictions

### 4. LogitNorm (LIMITS EXPRESSIVENESS)

**Current settings:**
- `use_logit_norm: true`
- `logit_norm: 1.0`

**How it works:**
- Normalizes logits to constant L2 norm before softmax
- Prevents logit explosion but also limits expressiveness

**Problem:**
- With norm=1.0, maximum possible logit ≈ 0.35-0.38 (for 8 classes)
- This severely limits the range of expressible probabilities
- Combined with temperature=4.0, creates very constrained predictions

**Impact:**
- Model can't express high confidence in any class
- Forces predictions to be spread out more uniformly
- Reduces model's ability to match peaked target distributions

## Root Cause

**The model is being forced to predict uniform distributions by multiple mechanisms:**

1. **Entropy regularization** (weight=5.0) penalizes ANY low-entropy prediction
2. **MaxSup penalty** (weight=0.1) suppresses maximum logit, preventing peaked predictions
3. **Temperature scaling** (T=4.0) flattens all distributions
4. **LogitNorm** (norm=1.0) limits expressiveness

**Combined effect:**
- Model learns to predict `entropy ≈ 2.08` (uniform) for ALL samples
- Even when target has `entropy ≈ 0.5-1.0` (peaked, location-specific)
- KL divergence measurement is correct, but **regularization dominates** and forces uniformity

## Solutions

### Option 1: Reduce/Remove Entropy Regularization

**Entropy regularization should only penalize VERY low entropy (near one-hot), not moderate entropy.**

**Recommendation:**
```yaml
entropy_weight: 0.5  # Reduce from 5.0 to 0.5 (10x reduction)
min_entropy: 0.5     # Reduce from 1.5 to 0.5 (only penalize very low entropy)
```

**Rationale:**
- Targets might have entropy 0.5-1.5 (peaked distributions)
- Only penalize entropy < 0.5 (near one-hot, mode collapse)
- Allow model to predict peaked distributions when targets are peaked

### Option 2: Reduce Temperature Scaling

**Lower temperature allows more expressive predictions.**

**Recommendation:**
```yaml
temperature: 2.0  # Reduce from 4.0 to 2.0
```

**Rationale:**
- Allows model to express confidence in specific classes
- Still prevents extreme overconfidence
- More balanced between uniformity and expressiveness

### Option 3: Remove MaxSup Penalty

**MaxSup directly forces uniformity. Remove or significantly reduce.**

**Recommendation:**
```yaml
maxsup_weight: 0.0  # Disable (was 0.1)
```

**Rationale:**
- MaxSup suppresses maximum logit, preventing peaked predictions
- But we WANT peaked predictions when targets are peaked
- Entropy regularization is sufficient to prevent mode collapse

### Option 4: Increase LogitNorm

**Higher norm allows more expressive predictions.**

**Recommendation:**
```yaml
logit_norm: 2.0  # Increase from 1.0 to 2.0
```

**Rationale:**
- Allows larger logit differences
- Enables more confident predictions when targets are peaked
- Still prevents logit explosion (norm is constrained)

### Option 5: Conditional Entropy Regularization

**Only penalize entropy when it's VERY low (near one-hot), not moderate entropy.**

**Alternative approach:**
- Only apply entropy penalty when `entropy < 0.3` (very low, near one-hot)
- Allow moderate entropy (0.5-1.5) that matches peaked targets

## Learning Rate Scheduling

**Current settings:**
- `learning_rate: 0.0001`
- `ReduceLROnPlateau` with `patience=5, factor=0.5`

**Analysis:**
- Learning rate might be too low after initial convergence
- Model converges to uniform distribution early (epoch 5-10)
- LR scheduler reduces LR when validation plateaus, but model is already stuck

**Recommendation:**
```yaml
learning_rate: 0.0002  # Increase from 0.0001 to 0.0002
scheduler: ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
```

**Alternative:**
- Use cosine annealing with warm restarts
- Allows model to escape local minima (uniform distribution)

## Recommended Configuration Changes

**Priority 1 (Critical):**
```yaml
loss:
  entropy_weight: 0.5   # Reduce from 5.0 (10x reduction)
  min_entropy: 0.5      # Reduce from 1.5 (only penalize very low entropy)
  maxsup_weight: 0.0    # Disable (was 0.1)

model:
  temperature: 2.0      # Reduce from 4.0
  logit_norm: 2.0       # Increase from 1.0

training:
  learning_rate: 0.0002  # Increase from 0.0001
```

**Priority 2 (Test after Priority 1):**
- Monitor if model can now predict location-specific distributions
- Check if KL divergence improves
- Verify entropy of predictions matches entropy of targets

## Expected Results

**Before (current):**
- Predictions: `entropy ≈ 2.08` (uniform) for all samples
- Targets: `entropy ≈ 0.5-1.5` (peaked, location-specific)
- KL divergence: `≈0.17` (model can't match targets)

**After (recommended changes):**
- Predictions: `entropy ≈ 0.5-1.5` (varies by location, matches targets)
- Targets: `entropy ≈ 0.5-1.5` (peaked, location-specific)
- KL divergence: `≈0.10-0.15` (model can match targets better)

## Verification Steps

1. **Check target distribution variance:**
   - Sample 10 random target distributions
   - Compute entropy for each
   - Verify entropy ranges from 0.5-1.5 (not always uniform)

2. **Check prediction distribution variance:**
   - Sample 10 random predictions
   - Compute entropy for each
   - Verify entropy varies and matches target entropy range

3. **Monitor per-sample KL divergence:**
   - Compute KL per sample (not just average)
   - Verify KL varies across samples (not constant)
   - Lower KL means better match to location-specific targets

## Conclusion

**The KL divergence measurement is CORRECT.** The problem is that **regularization terms are forcing uniformity** across all samples, preventing the model from learning location-specific distributions.

**Solution:** Reduce regularization strength (especially entropy_weight), remove MaxSup, reduce temperature, and increase logit_norm to allow more expressive, location-specific predictions.