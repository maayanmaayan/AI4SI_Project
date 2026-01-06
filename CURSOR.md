# AI for Social Good - 15-Minute City Service Gap Prediction

<!-- Brief one-line description: Tabular Transformer (FT-Transformer) for predicting service category interventions to improve walkability in urban neighborhoods -->

## Project Overview

This project implements a Tabular Transformer (FT-Transformer)-based classification model that:
- **Input**: Map location (coordinates) + 30+ urban/demographic features
- **Output**: Probability distribution over 8 NEXI service categories
- **Task**: Multi-class classification (service gap prediction)
- **Domain**: 15-minute city implementation using OpenStreetMap and Census data
- **Learning Approach**: Weakly supervised / rule-guided learning - learns patterns from compliant neighborhoods

## Tech Stack

<!-- List the main technologies used in this project -->

- **Deep Learning**: PyTorch 2.x, FT-Transformer (tabular transformer implementation)
- **ML Utilities**: scikit-learn 1.3+ (metrics, splits, preprocessing utilities)
- **Data Processing**: pandas 2.0+, numpy 1.24+, geopandas 0.14+ (geospatial data)
- **OSM Data**: osmnx 1.6+ (OSM network analysis), Overpass API
- **Interpretability**: attention analysis (feature-token attention); SHAP (optional, if feasible)
- **Evaluation**: scikit-learn (metrics), matplotlib 3.7+, seaborn 0.12+ (visualization)
- **Data Storage**: CSV, Parquet (processed features), GeoJSON (boundaries)
- **Experiment Tracking**: (Optional) Weights & Biases, MLflow, or TensorBoard
- **Testing**: pytest 7.4+
- **Code Quality**: black 23.0+, ruff 0.1+, mypy 1.5+ (optional)

## Project Structure

<!-- Document the folder structure of your project -->

```
AI4SI_Project/
├── data/
│   ├── raw/
│   │   ├── osm/                 # Raw OSM extracts
│   │   ├── census/              # Census data files
│   │   └── neighborhoods/       # Neighborhood boundary definitions
│   ├── processed/
│   │   ├── features/            # Extracted feature vectors
│   │   └── labels/               # Continuous gap scores (if used)
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   ├── transformer.py          # FT-Transformer model implementation
│   ├── config.py                # Hyperparameters
│   └── checkpoints/             # Saved model files (.pt)
├── src/
│   ├── data/
│   │   ├── collection/
│   │   │   ├── osm_extractor.py # OSM data extraction
│   │   │   ├── census_loader.py # Census data loading
│   │   │   └── feature_engineer.py # Feature computation
│   │   └── preprocessing.py     # Data cleaning/normalization
│   ├── training/
│   │   ├── train.py             # Model training script
│   │   └── hyperparameter_tuning.py # Hyperparameter optimization
│   ├── evaluation/
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── validate_principles.py # 15-minute city validation
│   │   ├── compare_neighborhoods.py # Comparative analysis
│   │   └── visualize.py        # Feature importance/SHAP/result visualization
│   └── utils/
│       ├── config.py            # Configuration management
│       ├── logging.py            # Logging setup
│       └── helpers.py           # General utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── experiments/
│   └── runs/                     # Experiment outputs
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test data fixtures
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── PRD.md                      # Product Requirements Document
└── CURSOR.md                   # This file - AI assistant context
```

## Commands

<!-- List the essential commands for running, building, and testing the project -->

```bash
# Install dependencies
pip install -r requirements.txt

# Run data collection and feature engineering
python src/data/collection/osm_extractor.py
python src/data/collection/census_loader.py
python src/data/collection/feature_engineer.py

# Run data preprocessing
python src/data/preprocessing.py --input data/raw --output data/processed

# Train the model
python src/training/train.py --config models/config.py

# Evaluate the model
python src/evaluation/validate_principles.py --checkpoint models/checkpoints/best.pt
python src/evaluation/compare_neighborhoods.py --checkpoint models/checkpoints/best.pt

# Run tests
pytest tests/

# Run linting
black src/ tests/
ruff check src/ tests/

# Run type checking (if using mypy)
mypy src/
```

## Data Pipeline

<!-- Document the data flow and preprocessing steps -->

### Data Sources
- **OSM Data**: OpenStreetMap data for services, buildings, walkability features
- **Census Data**: Demographics, socioeconomic indicators, population data
- **Neighborhood Boundaries**: Geographic boundaries for 6 neighborhoods (3 compliant + 3 non-compliant with 15-minute city principles)
- **Labels**: Continuous gap scores derived from OSM data (weak supervision)

### Feature Categories (30+ features)
- **Demographics**: Population density, SES index, car ownership, children per capita, household size, elderly ratio
- **Built Form**: Building density, building count, average building levels, floor area per capita
- **Services**: Counts and walk times for each service category (Education, Grocery, Health, Parks, Sustenance, Shops)
- **Walkability**: Intersection density, average block length, pedestrian street ratio, sidewalk presence
- **Composite**: Essential services coverage, average walk time to essentials, 15-minute walk score

### Preprocessing Steps
1. OSM data extraction (services, buildings, network)
2. Census data integration
3. Feature engineering (compute all 30+ features)
4. Data validation and quality checks
5. Normalization/scaling
6. Train/validation/test split (stratified by neighborhood type)

## Model Architecture

<!-- Document the tabular transformer model design -->

### Tabular Transformer (FT-Transformer) Design
- **Architecture**: FT-Transformer (single primary model)
- **Input**: Tabular feature vectors (30+ features)
- **Output**: Probability distribution over 8 NEXI service categories
- **Learning Approach**: Weakly supervised / rule-guided learning; learns implicit patterns from compliant neighborhoods and predicts interventions for all locations

### Training Configuration
- **Model Type**: Transformer encoder over feature tokens (multi-class classification head)
- **Loss Function**: Multi-class cross-entropy
- **Hyperparameters**:
  - d_token, n_layers, n_heads, dropout, learning rate, weight decay, batch size
  - Tune via validation (or lightweight hyperparameter search)
- **Early Stopping**: Based on validation loss with patience
- **Class Imbalance**: Consider class weights or focal loss if needed

## Evaluation Metrics

<!-- Document the metrics used to evaluate the model -->

- **Primary Metrics**: Accuracy, F1-score (macro/micro), Precision, Recall, Log Loss
- **Per-Class Metrics**: Confusion matrix, per-class F1 scores, precision/recall per service category
- **Interpretability**: attention analysis (global patterns + local examples); SHAP (optional)
- **Domain-Specific Validation**: 
  - Alignment with 15-minute city principles
  - Comparative analysis (compliant vs non-compliant neighborhoods)
  - Intervention probability distributions
- **Additional Analysis**: Probability distribution entropy, top-k accuracy

## Code Conventions

<!-- Document the coding standards and patterns used in this project -->

### Python Style
- Follow PEP 8
- Use type hints where possible
- Docstrings for all functions and classes
- Maximum line length: 100 characters

### ML Code Patterns
- Separate data, model, and training logic
- Use configuration files for hyperparameters
- Make code reproducible (set random seeds)
- Version control data preprocessing steps

### File Naming
- Use snake_case for Python files
- Descriptive names that indicate purpose
- Keep functions focused and modular

## Logging

<!-- Describe the logging approach used in this project -->

- Use Python's `logging` module
- Log training progress (loss, metrics per epoch)
- Log model checkpoints and best model saves
- Log experiment configurations and results
- Consider using experiment tracking tools (W&B, MLflow) for better visualization

## Data Management

<!-- Document how data is stored and versioned -->

- Raw data should not be modified
- Processed data should be versioned or regenerated from raw data
- Use data splits that are saved and reproducible
- Document data sources and any transformations applied

## Testing Strategy

<!-- Describe how testing is organized in this project -->

### Testing Pyramid
- **Unit tests**: Test individual functions (data preprocessing, model components, metrics)
- **Integration tests**: Test data pipeline, training loop, evaluation pipeline
- **Model tests**: Test model forward pass, output shapes, gradient flow

### Test Organization
```
tests/
├── unit/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_metrics.py
├── integration/
│   ├── test_training_pipeline.py
│   └── test_evaluation_pipeline.py
└── fixtures/
    └── sample_data.py
```

## Experiment Tracking

<!-- Document how experiments are tracked and organized -->

- Save experiment configurations (hyperparameters, feature sets, data splits)
- Track hyperparameters and results (metrics, validation scores)
- Version model checkpoints (`.pt`)
- Document findings and insights (attention analysis, optional SHAP analysis)
- Compare different hyperparameter configurations and model sizes
- Track validation against 15-minute city principles
- Document comparative analysis results (compliant vs non-compliant neighborhoods)

## Key Project Details

### Service Categories (8 NEXI Categories)
- **Education**: college, driving_school, kindergarten, language_school, music_school, school, university
- **Entertainment**: arts_center, cinema, community_center, theatre, etc.
- **Grocery**: supermarket, bakery, convenience, greengrocer, etc.
- **Health**: clinic, dentist, doctors, hospital, pharmacy, etc.
- **Posts and banks**: ATM, bank, post_office
- **Parks**: park, dog_park
- **Sustenance**: restaurant, pub, bar, cafe, fast_food, etc.
- **Shops**: department_store, general, kiosk, mall, boutique, clothes, etc.

### Learning Approach
- **Weakly Supervised / Rule-Guided Learning**: Training signals derived from rule-based gap scores
- **Pattern Learning**: Model learns implicit service distribution patterns from compliant neighborhoods
- **Reference-Based**: Compliant neighborhoods serve as reference structures for identifying gaps
- **Validation**: Model validated against 15-minute city principles, not just predictive accuracy

### Timeline
- **MVP**: 2 weeks (14 days)
- **Phase 1**: Data Collection & Feature Engineering (Days 1-5)
- **Phase 2**: Model Training & Evaluation (Days 6-10)
- **Phase 3**: Evaluation & Validation (Days 11-12)
- **Phase 4**: Documentation & Refinement (Days 13-14)
