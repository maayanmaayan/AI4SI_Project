# AI for Social Good - 15-Minute City Service Gap Prediction

This project implements a Spatial Graph Transformer-based AI model to support data-driven implementation of the 15-minute city model. The system learns service distribution patterns from neighborhoods already designed according to 15-minute city principles, then identifies service gaps and recommends appropriate interventions to improve walkability and accessibility in other neighborhoods.

## Quick Links (Submission)

- **Final report**: `FINAL_REPORT.md`
- **Results summary (tables)**: `RESULTS_SUMMARY.md`
- **experiments narrative**: `experiments/`
- **Model configuration**: `models/config.yaml`
- **Web UI demo**: `web-ui/` (mock predictions; see “Web UI” section below)

## Project Overview

This is an AI/ML project built with Cursor that focuses on:
- **Problem**: Identifying service gaps in urban neighborhoods for 15-minute city implementation
- **Model**: Spatial Graph Transformer using PyTorch Geometric for graph-based classification
- **Input**: Star graphs - each location represented as target point (node 0) + neighbor grid cells (nodes 1-N) within network walking distance, each with 33 urban/demographic features
- **Output**: Probability distribution over 8 NEXI service categories (Education, Entertainment, Grocery, Health, Posts and banks, Parks, Sustenance, Shops)
- **Training Strategy**: Model trains exclusively on 15-minute city compliant neighborhoods to learn optimal service distribution patterns
- **Loss Function**: Distance-based similarity loss using KL divergence between predicted and distance-based target probability vectors
- **Validation**: Model evaluation on compliant neighborhoods only (train/validation/test splits from compliant neighborhoods)

## What's Included

- **`.cursor/commands/`** - Commands for planning, execution, validation, and workflow automation
- **`.cursor/reference/`** - Best practices documentation for various technologies
- **`CURSOR.md`** - Project-specific documentation with tech stack, structure, conventions, and development guidelines
- **Project structure** - Organized for ML/AI development workflow

## Getting Started

1. **Review the PRD** — Read `PRD.md` for comprehensive project requirements and architecture
2. **Set up the environment** — Install dependencies for ML development (PyTorch, PyTorch Geometric, geospatial stack, etc.)
3. **Data collection** — Extract OSM and Census data for Paris neighborhoods. Neighborhood boundaries and compliance labels are defined in `paris_neighborhoods.geojson` (includes verified 15-minute neighborhoods and non-compliant neighborhoods)
4. **Feature engineering** — Build pipeline to generate grid cells and compute 20+ urban/demographic features for center point + all grid cells within 15-minute walk radius (configurable via `features.walk_15min_radius_meters` and `features.grid_cell_size_meters` in config.yaml)
5. **Model training** — Train the Spatial Graph Transformer model exclusively on 15-minute city compliant neighborhoods using distance-based similarity loss
6. **Evaluation** — Evaluate model on compliant neighborhoods only (train/validation/test splits from compliant neighborhoods)

See `PRD.md` for detailed implementation phases and timeline (2-week MVP).

## Results (High-Level)

Headline metrics are consolidated in `RESULTS_SUMMARY.md` (generated 2026-01-20).

- **Best Test KL divergence (lower is better)**: `combo_4_aggressive` (**0.3260**)
- **Best balanced candidate** (good KL without worst validation penalty): `combo_2_all_reductions` (Best Val **0.3802**, Test KL **0.3365**)
- **Top‑k plateau**: across many runs, **Top‑1 ≈ 34.84%** and **Top‑3 ≈ 97.91%**

For narrative + plots, see:
- `FINAL_REPORT.md`
- `experiments/overnight_runs/EXPERIMENT_SUMMARY.md`

## Web UI (Demo)

The `web-ui/` folder contains a **Vite + React + Leaflet** demo interface intended for presentation.

- **What it does**:
  - Interactive map view
  - “Prediction” modal with service-category recommendations
  - Basic export tooling (PDF)
- **Important limitation**:
  - The Web UI currently uses **mock predictions** (`web-ui/src/hooks/useMockPrediction.ts`) and is **not wired to the Python model** for live inference.
  - It visualizes services from `web-ui/public/data/jerusalem_services.geojson`.

To run the demo locally:

```bash
cd web-ui
npm install
npm run dev
```

## Development Workflow

When working with Cursor on this AI project:

1. **Data Collection** — Extract OSM data (services, buildings, walkability) and Census data (demographics, socioeconomic indicators, income, car ownership) for Paris neighborhoods defined in `paris_neighborhoods.geojson`. Note: Children per capita and elderly ratio are estimated using Paris-wide age distribution ratios due to IRIS-level data limitations.
2. **Feature Engineering** — Generate regular grid cells around each location, compute 33 features (demographics, built form, service counts, walkability) for target point + all neighbor grid cells within network walking distance. Build star graphs with edge attributes encoding spatial relationships. Neighbors represent spatial context of people who can access each location.
3. **Model Training** — Train the Spatial Graph Transformer exclusively on 15-minute city compliant neighborhoods using distance-based loss (distance to nearest service of predicted category)
4. **Hyperparameter Tuning** — Train 7 model variants: baseline (lr=0.001, num_layers=3, temp=200m), then vary learning rate [0.0005, 0.002], model depth [2, 4 layers], and temperature [150, 250m] independently. Select best model based on validation KL divergence loss
5. **Evaluation** — Evaluate model on compliant neighborhoods only (train/validation/test splits); model success indicated by learning (decreasing loss, improving metrics)
6. **Interpretability** — Analyze graph attention patterns and SHAP values to understand model decisions

## Project Structure

```
AI4SI_Project/
├── data/
│   ├── raw/                    # Raw inputs + sanity check outputs (JSON/CSV)
│   └── processed/
│       ├── features/           # Model-ready Parquet features (e.g. compliant/*/target_points.parquet)
│       └── labels/             # Auxiliary label artifacts (if generated)
├── models/
│   ├── config.yaml              # Model hyperparameters
│   └── checkpoints/             # Saved model files
├── docs/                        # Design notes, guides, and analyses (augmentation, architecture, sweeps, etc.)
├── experiments/
│   ├── overnight_runs/          # Tracked experiment outputs + summary
│   └── runs/                    # Additional runs (typically gitignored; summary captured in RESULTS_SUMMARY.md)
├── src/
│   ├── data/
│   │   ├── collection/         # OSM extraction, Census loading, feature engineering
│   │   │   ├── osm_extractor.py
│   │   │   ├── census_loader.py
│   │   │   └── feature_engineer.py
│   │   └── validation/
│   │       └── raw_data_sanity_checker.py
│   ├── training/
│   │   ├── model.py            # Graph Transformer architecture
│   │   ├── dataset.py          # PyTorch Geometric Dataset class
│   │   ├── train.py            # Training script
│   │   └── loss.py             # Distance-based loss function
│   ├── evaluation/
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── plotting.py         # Plotting utilities (training curves, calibration, etc.)
│   │   └── save_predictions.py # Persist predictions + evaluation results
│   └── utils/                  # Configuration, logging, helpers
├── scripts/                     # CLI wrappers / utilities (train, sweeps, feature engineering, plotting, etc.)
├── tests/                      # Unit and integration tests
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── PRD.md                      # Product Requirements Document
├── CURSOR.md                   # Project documentation for AI assistant
├── paris_neighborhoods.geojson # Paris neighborhood boundaries and compliance labels
└── web-ui/                      # Vite/React demo UI (Leaflet); uses mock predictions (not wired to live inference)
```

## Reproducibility Notes

- **Training data (MVP)**: model training/evaluation is done on **compliant neighborhoods only**.
- **Processed features format**: model-ready features are stored as `target_points.parquet` under `data/processed/features/...` and loaded by `src/training/dataset.py`.
- **Run outputs**: experiment artifacts are written under `experiments/` (see `RESULTS_SUMMARY.md` for consolidated metrics).

## Cursor Commands

Commands for Cursor to assist with development workflows.

### Planning & Execution

| Command | Description |
|---------|-------------|
| `/core_piv_loop:prime` | Load project context and codebase understanding |
| `/core_piv_loop:plan-feature` | Create comprehensive implementation plan with codebase analysis |
| `/core_piv_loop:execute` | Execute an implementation plan step-by-step |

### Validation

| Command | Description |
|---------|-------------|
| `/validation:validate` | Run full validation: tests, linting, coverage, build (customize to your project) |
| `/validation:code-review` | Technical code review on changed files |
| `/validation:code-review-fix` | Fix issues found in code review |
| `/validation:execution-report` | Generate report after implementing a feature |
| `/validation:system-review` | Analyze implementation vs plan for process improvements |

### Bug Fixing

| Command | Description |
|---------|-------------|
| `/github_bug_fix:rca` | Create root cause analysis document for a GitHub issue |
| `/github_bug_fix:implement-fix` | Implement fix based on RCA document |

### Misc

| Command | Description |
|---------|-------------|
| `/commit` | Create atomic commit with appropriate tag (feat, fix, docs, etc.) |
| `/init-project` | Install dependencies and set up development environment (customize this to your project) |
| `/create-prd` | Generate Product Requirements Document from conversation |
