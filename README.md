# AI for Social Good - 15-Minute City Service Gap Prediction

This project implements a Tabular Transformer (FT-Transformer)-based AI model to support data-driven implementation of the 15-minute city model. The system identifies service gaps and recommends appropriate interventions to improve walkability and accessibility in neighborhoods.

## Project Overview

This is an AI/ML project built with Cursor that focuses on:
- **Problem**: Identifying service gaps in urban neighborhoods for 15-minute city implementation
- **Model**: Tabular Transformer (FT-Transformer) for tabular data classification
- **Input**: Map location coordinates + 30+ urban/demographic features
- **Output**: Probability distribution over 8 NEXI service categories (Education, Entertainment, Grocery, Health, Posts and banks, Parks, Sustenance, Shops)

## What's Included

- **`.cursor/commands/`** - Commands for planning, execution, validation, and workflow automation
- **`.cursor/reference/`** - Best practices documentation for various technologies
- **`CURSOR.md`** - Project-specific documentation with tech stack, structure, conventions, and development guidelines
- **Project structure** - Organized for ML/AI development workflow

## Getting Started

1. **Review the PRD** — Read `PRD.md` for comprehensive project requirements and architecture
2. **Set up the environment** — Install dependencies for ML development (PyTorch, FT-Transformer, data processing libraries, etc.)
3. **Data collection** — Extract OSM and Census data for 6 neighborhoods (3 compliant + 3 non-compliant with 15-minute city principles)
4. **Feature engineering** — Build pipeline to compute 30+ urban/demographic features
5. **Model training** — Train the FT-Transformer model to predict service category interventions
6. **Evaluation** — Validate model against 15-minute city principles and compare predictions across neighborhood types

See `PRD.md` for detailed implementation phases and timeline (2-week MVP).

## Development Workflow

When working with Cursor on this AI project:

1. **Data Collection** — Extract OSM data (services, buildings, walkability) and Census data (demographics)
2. **Feature Engineering** — Compute 30+ features including demographics, built form, services, and walkability metrics
3. **Model Training** — Train the FT-Transformer on all neighborhoods
4. **Hyperparameter Tuning** — Optimize model performance with cross-validation
5. **Evaluation** — Validate against 15-minute city principles, compare compliant vs non-compliant neighborhoods
6. **Interpretability** — Analyze attention patterns and SHAP values to understand model decisions

## Project Structure

```
AI4SI_Project/
├── data/
│   ├── raw/                    # Raw OSM extracts, Census data, neighborhood boundaries
│   ├── processed/              # Cleaned and preprocessed features
│   └── splits/                 # Train/validation/test splits
├── models/
│   ├── transformer.py           # FT-Transformer architecture
│   ├── config.py                # Hyperparameters
│   └── checkpoints/             # Saved model files
├── src/
│   ├── data/
│   │   ├── collection/         # OSM extraction, Census loading, feature engineering
│   │   └── preprocessing.py    # Data cleaning/normalization
│   ├── training/
│   │   ├── train.py            # Model training script
│   │   └── hyperparameter_tuning.py
│   ├── evaluation/
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── validate_principles.py # 15-minute city validation
│   │   ├── compare_neighborhoods.py
│   │   └── visualize.py        # Feature importance/SHAP visualization
│   └── utils/                  # Configuration, logging, helpers
├── notebooks/                  # Jupyter notebooks for exploration
├── experiments/                # Experiment outputs and logs
├── tests/                      # Unit and integration tests
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── PRD.md                      # Product Requirements Document
└── CURSOR.md                   # Project documentation for AI assistant
```

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
