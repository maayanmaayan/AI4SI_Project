# Verified Solutions to Prevent Mode Collapse (One-Hot Predictions)

## Problem
Model predictions collapse to one-hot distributions (single class with probability ~1.0) even though targets are already smooth probability distributions.

## Verified Solutions (with Sources)

### 1. Temperature Scaling in Softmax ✅ VERIFIED

**What it does**: Divides logits by temperature T before softmax. Higher T = flatter distribution.

**Sources**:
- Standard calibration technique widely used in practice
- "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015) - uses temperature for knowledge distillation
- Datacamp tutorial on softmax: https://www.datacamp.com/tutorial/softmax-activation-function-in-python

**Current status**: Already implemented in model forward pass (T=4.0 in config, but code may be using different value)

**Recommendation**: 
- Verify temperature is actually being applied (check model.py forward pass)
- Try T = 3.0 to 5.0 (higher = flatter)
- Monitor if logits are being scaled correctly

---

### 2. Entropy Regularization ✅ VERIFIED

**What it does**: Adds penalty term to loss that encourages higher entropy in predictions: `Loss = KL_loss + λ * (min_entropy - entropy)`

**Sources**:
- "Regularizing Neural Networks by Penalizing Confident Output Distributions" (Pereyra et al., 2017) - https://arxiv.org/abs/1701.06548
- PapersWithCode: https://paperswithcode.com/method/entropy-regularization
- "Generalized Entropy Regularization" (ACL 2020) - https://aclanthology.org/2020.acl-main.615/

**Current status**: Already implemented (entropy_weight=5.0, min_entropy=1.5, quadratic penalty)

**Recommendation**:
- Current settings look reasonable (weight=5.0, min_entropy=1.5)
- Quadratic penalty is more aggressive than linear - this is good for preventing collapse
- Monitor actual entropy values during training to ensure they stay above 1.5

---

### 3. Gradient Clipping ✅ VERIFIED

**What it does**: Clips gradients to maximum norm to prevent logit explosion.

**Sources**:
- Standard practice in RNN/Transformer training
- Prevents gradient explosion which can cause logits to grow unbounded
- Common in PyTorch: `torch.nn.utils.clip_grad_norm_()`

**Current status**: Implemented (max_grad_norm=1.0)

**Recommendation**:
- Current value (1.0) is reasonable
- Monitor gradient norms - if they're consistently being clipped, might need to adjust
- Can try 0.5-2.0 range

---

### 4. Extra Weight Decay on Final Layer ✅ VERIFIED

**What it does**: Applies additional L2 regularization specifically to classifier layer weights.

**Sources**:
- Common practice to prevent final layer from learning extreme weights
- "Margin-based Label Smoothing" (2021) - https://arxiv.org/abs/2111.15430 - discusses logit magnitude constraints
- Weight normalization and spectral normalization papers discuss limiting final layer scale

**Current status**: Implemented (final_layer_weight_decay=0.01)

**Recommendation**:
- Current value (0.01) is reasonable
- Monitor final layer weight norms - should not grow too large
- Can try 0.005-0.02 range

---

### 5. Quadratic Entropy Penalty ⚠️ NEEDS VERIFICATION

**What it does**: Uses `(min_entropy - entropy)^2` instead of `max(0, min_entropy - entropy)` for entropy penalty.

**Sources**:
- **NOT FOUND IN LITERATURE** - This appears to be a custom modification
- Standard entropy regularization uses linear penalty
- Quadratic penalty is more aggressive but not standard

**Recommendation**:
- **Consider reverting to linear penalty** unless you have evidence quadratic works better
- Linear penalty is the standard approach from literature
- If keeping quadratic, monitor carefully - it may be too aggressive

---

## Additional Verified Techniques (Not Yet Implemented)

### 6. Max Logit Suppression ✅ VERIFIED (Not Implemented)

**What it does**: Suppresses the maximum logit rather than boosting ground-truth, preventing any class from becoming too dominant.

**Sources**:
- "MaxSup: Overcoming Representation Collapse in Label Smoothing" (2025) - https://arxiv.org/abs/2502.15798
- Specifically designed to prevent overconfidence without label smoothing

**Recommendation**: Consider implementing if other methods don't work

---

### 7. Margin-Based Constraints ✅ VERIFIED (Not Implemented)

**What it does**: Limits the margin (difference) between top logit and second-best logit.

**Sources**:
- "Margin-based Label Smoothing" (2021) - https://arxiv.org/abs/2111.15430
- "Advancing neural network calibration: The role of gradient decay in large-margin Softmax optimization" (2024) - https://www.sciencedirect.com/science/article/abs/pii/S0893608024003812
- Prevents single logit from growing too far ahead

**Recommendation**: Could be added as additional constraint

---

### 8. Logit Normalization (LogitNorm) ✅ VERIFIED (Not Implemented)

**What it does**: Constrains the norm (magnitude) of the logit vector so logits don't grow arbitrarily large. Large norm leads softmax to produce extreme distributions.

**Sources**:
- "Mitigating Neural Network Overconfidence with Logit Normalization" (ICML 2022) - https://arxiv.org/abs/2205.09310
- Reduced false positive rates from 50.33% to 8.03% on CIFAR-10/SVHN
- Simple modification to cross-entropy loss

**Recommendation**: Consider implementing - simple and effective

---

### 9. Selective Output Smoothing (SOSR) ✅ VERIFIED (Not Implemented)

**What it does**: Targets only samples where model is very confident (> threshold, e.g. 0.9), forcing logits over incorrect classes to be uniform while leaving correct class logit unchanged.

**Sources**:
- "Selective output smoothing regularization: Regularize neural networks by softening output distributions" (2025) - https://link.springer.com/article/10.1007/s10489-025-06539-6
- Only softens overconfident predictions, keeps learning signal strong elsewhere

**Recommendation**: Could be effective for your use case

---

### 10. Adaptive Temperature Scaling ✅ VERIFIED (Not Implemented)

**What it does**: Computes temperature as function of entropy/uncertainty of each prediction, instead of single global scalar. High entropy predictions treated differently than confident ones.

**Sources**:
- "Adaptive temperature scaling for Robust calibration of deep neural networks" - https://link.springer.com/article/10.1007/s00521-024-09505-4
- HTS method scales based on entropy

**Recommendation**: More sophisticated than fixed temperature, but needs validation data

---

### 11. Mixup Data Augmentation ✅ VERIFIED (Not Implemented)

**What it does**: Mixes inputs and their labels: `input = λ*x1 + (1-λ)*x2; label = λ*y1 + (1-λ)*y2`. Forces model to predict mixture distributions, avoiding overconfident peaks.

**Sources**:
- "mixup: Beyond Empirical Risk Minimization" (ICLR 2018) - widely used technique
- Works even with soft labels

**Recommendation**: Could be added to existing augmentation pipeline

---

### 12. Post-Training Calibration Methods ✅ VERIFIED (Not Implemented)

**What they do**: Adjust outputs after training without modifying training loss:
- **Dirichlet Calibration**: Learned linear transformation + softmax over transformed probabilities
- **Platt Scaling**: Logistic transformation of scores
- **Isotonic Regression**: Non-parametric calibration

**Sources**:
- "Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration" (2019) - https://arxiv.org/abs/1910.12656
- "On Calibration of Modern Neural Networks" (Guo et al., 2017) - https://proceedings.mlr.press/v70/guo17a
- scikit-learn calibration documentation

**Recommendation**: Useful if model is already trained and you want to fix overconfidence without retraining

---

### 13. Weight Normalization / Spectral Normalization ✅ VERIFIED (Not Implemented)

**What it does**: Normalizes weight vectors to unit norm, limiting logit scale. Spectral normalization constrains weight matrix spectral norm.

**Sources**:
- Weight normalization papers discuss limiting final layer scale
- Common in large-margin softmax losses

**Recommendation**: Could be applied to final classifier layer

---

## What to Check in Your Code

1. **Verify temperature is applied**: Check `model.py` forward pass - logits should be divided by temperature
2. **Check entropy calculation**: Ensure entropy is computed correctly in loss function
3. **Monitor logit magnitudes**: Log max/min logit values - should stay in reasonable range (-10 to +10)
4. **Check for double softmax**: Ensure softmax is only applied once (not in model AND loss)
5. **Verify gradient clipping**: Check that gradients are actually being clipped (log gradient norms)

## Recommended Action Plan

1. **Verify current implementations** are working correctly (temperature, entropy, gradient clipping)
2. **Monitor metrics** during training:
   - Average prediction entropy (should be > 1.5)
   - Max logit values (should not explode)
   - Gradient norms (should be clipped if > 1.0)
3. **Consider reverting quadratic penalty** to linear (standard approach)
4. **If still collapsing**, try:
   - Increase temperature to 5.0-6.0
   - Increase entropy_weight to 7.0-10.0
   - Implement MaxSup or margin constraints

## Additional Notes from Sources

### Things to Check/Verify in Your Code

1. **Temperature application**: Verify temperature is actually being applied in model forward pass (line 177 in model.py shows `logits = logits / self.temperature`)
2. **Double softmax**: Ensure softmax is only applied once (not in model AND loss function)
3. **Logit magnitudes**: Monitor if logits are growing unbounded (should stay in reasonable range)
4. **Class imbalance**: Check if one class dominating predictions is due to data imbalance
5. **Learning rate**: Too high LR can cause fast collapse - consider warm-up schedule

### Trade-offs to Consider

- **Entropy regularization**: Can make predictions too uniform if weight is too high, hurting accuracy
- **Temperature scaling**: Only affects outputs, doesn't change how model learns (unless used during training)
- **Post-training calibration**: Needs separate calibration set, doesn't fix root cause
- **Mixup**: Requires compatible data/label structure

## References

1. Pereyra et al. (2017). "Regularizing Neural Networks by Penalizing Confident Output Distributions" - https://arxiv.org/abs/1701.06548
2. Hinton et al. (2015). "Distilling the Knowledge in a Neural Network" - https://arxiv.org/abs/1503.02531
3. "Margin-based Label Smoothing" (2021) - https://arxiv.org/abs/2111.15430
4. "MaxSup: Overcoming Representation Collapse" (2025) - https://arxiv.org/abs/2502.15798
5. "Mitigating Neural Network Overconfidence with Logit Normalization" (ICML 2022) - https://arxiv.org/abs/2205.09310
6. "Selective output smoothing regularization" (2025) - https://link.springer.com/article/10.1007/s10489-025-06539-6
7. "Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration" (2019) - https://arxiv.org/abs/1910.12656
8. "On Calibration of Modern Neural Networks" (Guo et al., 2017) - https://proceedings.mlr.press/v70/guo17a
9. "Adaptive temperature scaling for Robust calibration" - https://link.springer.com/article/10.1007/s00521-024-09505-4
10. "Advancing neural network calibration: The role of gradient decay in large-margin Softmax optimization" (2024) - https://www.sciencedirect.com/science/article/abs/pii/S0893608024003812
11. PapersWithCode - Entropy Regularization: https://paperswithcode.com/method/entropy-regularization
