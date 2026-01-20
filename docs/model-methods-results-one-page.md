# Model Methods + Experimental Results (One Page)

## Methods (computational approach)

This project models **service-gap recommendation** as predicting a **probability distribution over 8 service categories** using a **Spatial Graph Transformer** trained on **exemplar (15‑minute compliant) neighborhoods**.

- **Input representation (star graph)**: Each sample is a *star graph* centered at a target point (node 0) with neighbor grid cells (nodes 1..N) inside a configurable radius. Each node has a **33‑dimensional feature vector** (demographics, built form, service counts, walkability). Edges connect neighbor→target and carry spatial attributes \([dx, dy, d_{euclid}, d_{net}]\) (distance features are normalized in the dataset loader).
- **Model architecture**: The model uses PyTorch Geometric `TransformerConv` layers (multi-head attention with edge attributes). Node features and edge attributes are embedded, processed through stacked attention layers with residual connections and layer normalization, then the **target node embedding** is extracted and passed through a linear classifier to produce logits for 8 categories. To improve training stability and reduce pathological overconfidence, the implementation includes **LogitNorm (logit magnitude constraint)** and **temperature scaling** on logits.
- **Distance-based supervision (soft targets)**: Targets are not one-hot labels. For each target location, the pipeline computes the **network walking distance** to the nearest observed service for each category (via OSMnx shortest paths). These 8 distances are converted into a **target probability vector** using temperature-scaled softmax over negative distances:
  \[
  p_{target}(c) \propto \exp\left(-\frac{d(c)}{\tau}\right)
  \]
  with a fixed penalty distance for missing categories.
- **Training objective**: The main loss is **KL divergence** \(KL(p_{target}\,\|\,p_{pred})\). Multiple anti-collapse knobs were explored: entropy regularization (minimum entropy), MaxSup (maximum-logit suppression), focal-style weighting, class weighting (damped/capped), gradient clipping, and realistic augmentation (feature noise, neighbor subsampling, edge noise).

## Experimental results (summary)

Two main experiment families were run:

1. **Regularization sweeps (overnight experiments)**: Varied entropy regularization, logit temperature, logit normalization, and MaxSup.
   - Best observed **Test KL** was ~**0.326–0.350** depending on configuration (best KL sometimes coincided with higher validation loss, consistent with overfitting/overconfidence).
   - **Top‑1 accuracy remained essentially constant (~34.84%)** across these variants, while **Top‑3 accuracy remained very high (~97.91%)**, indicating the model’s ranking behavior and class preference did not materially change despite large regularization changes.

2. **Architecture/training sweeps**: Varied learning rate, depth, and distance‑target temperature.
   - Validation losses clustered tightly (≈ **0.376–0.387** for the sweep shown), with similar KL values (≈ **0.336–0.346**), and **Top‑1 again stayed ~34.84%**.
   - Additional focal‑loss variants changed some calibration/entropy behavior but did not yield a meaningful improvement in class prediction quality.

## Conclusion (why the bottleneck is likely the data)

Given the breadth of changes tested (regularization strength, logit temperature/norm, depth, learning rate, focal variants, augmentation) and the **near-invariance of Top‑1 accuracy**, the evidence suggests the limiting factor is **not hyperparameter choice**, but the **data + supervision pipeline**. Likely causes:

- **Weak/ambiguous target signal**: distance→probability targets may be too similar across many locations (overly smooth targets, dominant categories, or insufficient variance).
- **Noisy/misaligned targets**: incomplete category mappings, missing-service penalties, or network-distance artifacts can bias targets away from what features can predict.
- **Feature–target mismatch**: the 33 engineered features may not contain enough discriminative signal to separate the distance-derived targets at target-point resolution.
- **Limited diversity / imbalance**: training on a small set of compliant neighborhoods may lead to learning a stable prior distribution that already achieves ~35% Top‑1, with little room for improvement via tuning.

Overall, results indicate that **improving data quality, target construction, and feature–target alignment** is the most promising path to meaningful learning, rather than further parameter sweeps of the current model.

