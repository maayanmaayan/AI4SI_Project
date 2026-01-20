# Final Project Report — 15‑Minute City Service Gap Prediction

**Repository:** `AI4SI_Project`  
**Branch:** `main`  
**Date:** 2026-01-20  

## 1) What this project does

This project builds an end‑to‑end research prototype that predicts **which everyday service category** (8 NEXI buckets) would most improve local accessibility at a chosen location, using the **15‑minute city** framing.

- **Input:** a *star graph* around a target point (node 0) with neighbor grid cells (nodes 1..N) inside a configurable radius; each node has **33 features** (demographics, built form, service counts, walkability).
- **Output:** a **probability distribution over 8 service categories** (Education, Entertainment, Grocery, Health, Posts & Banks, Parks, Sustenance, Shops).
- **Learning approach:** *exemplar-based* — train on **compliant neighborhoods** to learn “good” service-distribution patterns.
- **Supervision:** a **distance-based target probability vector** built from (network) walking distances to the nearest service of each category; training minimizes **KL divergence** between predicted and target distributions.

## 2) Data and feature pipeline (high level)

### Data sources

- **OpenStreetMap (OSM):** services, buildings, pedestrian infrastructure, street network.
- **Census (INSEE / IRIS):** demographic and socioeconomic indicators (via `pynsee`).
- **Neighborhood boundaries:** `paris_neighborhoods.geojson` (labels and metadata).

### Feature engineering flow

Key implementation: `src/data/collection/feature_engineer.py`

1. Generate **target points** inside each neighborhood polygon (`target_point_sampling_interval_meters`).
2. For each target point, generate neighbor **grid cells** (`grid_cell_size_meters`) within `walk_15min_radius_meters`.
3. Compute **33 features** for the target + each neighbor:
   - Demographics (17), Built form (4), Service counts (8), Walkability (4).
4. Build neighbor edge attributes: `[dx, dy, euclidean_distance, network_distance]` (normalized in the dataset loader).
5. Compute the **target probability vector** (8‑dim) using network distances to nearest services, converted via temperature-scaled softmax.
6. Save `target_points.parquet` per neighborhood under `data/processed/features/...`.

## 3) Model and training

### Model

Implementation: `src/training/model.py`

- `SpatialGraphTransformer` using PyTorch Geometric `TransformerConv`.
- Encodes node features and edge attributes, runs multiple attention layers, and classifies **only the target node embedding**.
- Includes **LogitNorm** and **temperature scaling** to reduce overconfidence and training instability.

### Dataset / batching

Implementation: `src/training/dataset.py`

- Loads `target_points.parquet` and converts each row into a PyG `Data` object.
- Star graph edges are neighbor→target; edge attributes are normalized to a 0–1 range.

### Loss

Implementation: `src/training/loss.py`

- KL divergence \(KL(target \|\| predicted)\) where the target is already a smooth probability distribution.
- Optional regularizers (entropy penalty / MaxSup / focal weighting) used to mitigate collapse/overconfidence.

### Training loop + outputs

Implementation: `src/training/train.py` (CLI wrapper: `scripts/train_graph_transformer.py`)

Produces (per run):
- `training_history.json`, `training_summary.json`
- plots under `plots/`
- `test_predictions.csv`, `val_predictions.csv`, and evaluation JSON

## 4) Experiments and results

### Where results are summarized

- **Main summary:** `RESULTS_SUMMARY.md`
- **Overnight regularization narrative:** `experiments/overnight_runs/EXPERIMENT_SUMMARY.md`

### Key outcomes (from `RESULTS_SUMMARY.md`)

- Best **Test KL** among overnight regularization experiments: `combo_4_aggressive` (but with overfitting signs via higher validation loss).
- Best **balanced** candidate: `combo_2_all_reductions` (strong KL improvement with reasonable validation loss).

## 5) How to reproduce (minimal)

### Python environment

See `SETUP.md` for environment setup and `requirements.txt` for dependencies.

### Feature engineering

Run for all neighborhoods:

```bash
python3 scripts/run_feature_engineering.py --all
```

### Training a single model

```bash
python3 scripts/train_graph_transformer.py --config models/config.yaml
```

### Hyperparameter sweep

```bash
python3 scripts/hyperparameter_sweep.py
```

### Regularization “overnight” experiments

```bash
python3 scripts/run_overnight_experiments.py --skip-completed
```

## 6) Known issues / notes for reviewers

- **`experiments/runs/` is gitignored** by default; this repo includes results mainly via:
  - `experiments/overnight_runs/` (tracked)
  - `RESULTS_SUMMARY.md` (tracked, consolidated metrics)
- **macOS + multiprocessing edge case**: some sweep logs show `pickle data was truncated`, which is commonly associated with worker process crashes. If this occurs on a reviewer machine, set `training.num_workers: 0` in `models/config.yaml` to run dataloading in-process.
- **Web UI is a demo**: `web-ui/` currently uses **mock predictions** and visualizes services from `web-ui/public/data/jerusalem_services.geojson`. It is not wired to the Python model for live inference.

## 7) Repository hygiene changes for submission

- Removed tracked OS cruft (`.DS_Store`) and captured console logs (`*.out`).
- Removed large backup Parquet copies under `.backup/` and `backup/`.
- Added `.gitignore` entries for `*.out`, `.backup/`, and `backup/`.

