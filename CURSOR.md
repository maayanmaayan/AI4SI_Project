# AI for Social Good - 15-Minute City Service Gap Prediction

<!-- Brief one-line description: Tabular Transformer (FT-Transformer) for predicting service category interventions to improve walkability in urban neighborhoods -->

## Project Overview

This project implements a Tabular Transformer (FT-Transformer)-based model that:
- **Input**: Map location (coordinates) + 30+ urban/demographic features
- **Output**: Probability distribution over 8 NEXI service categories
- **Task**: Service gap prediction with distance-based learning
- **Domain**: 15-minute city implementation using OpenStreetMap and Census data
- **Learning Approach**: Exemplar-based learning - trains exclusively on 15-minute city compliant neighborhoods to learn optimal service distribution patterns
- **Loss Function**: Distance-based loss measuring network-based walking distance from predicted service category to nearest actual service
- **Validation**: Model success validated by significantly lower loss (shorter distances) on compliant neighborhoods compared to non-compliant ones

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
- **OSM Data**: OpenStreetMap data for services, buildings, walkability features, and network-based distance calculations
- **Census Data**: Demographics, socioeconomic indicators, population data
- **Neighborhood Boundaries**: Geographic boundaries and compliance labels from `paris_neighborhoods.geojson` (6 neighborhoods: 3 compliant + 3 non-compliant)
- **Training Data**: Exclusively from 15-minute city compliant neighborhoods
- **Test Data**: Includes both compliant and non-compliant neighborhoods for validation
- **Distance Calculations**: Network-based walking distance via OSMnx (not Euclidean)

### Feature Categories (30+ features)
- **Demographics**: Population density, SES index, car ownership, children per capita, household size, elderly ratio
- **Built Form**: Building density, building count, average building levels, floor area per capita
- **Services**: Counts and walk times for each service category (Education, Grocery, Health, Parks, Sustenance, Shops)
- **Walkability**: Intersection density, average block length, pedestrian street ratio, sidewalk presence
- **Composite**: Essential services coverage, average walk time to essentials, 15-minute walk score

### Preprocessing Steps
1. Load neighborhood boundaries and compliance labels from `paris_neighborhoods.geojson`
2. OSM data extraction (services, buildings, network for distance calculations)
3. Census data integration
4. Feature engineering (compute all 30+ features)
5. Extract service locations for distance-based loss calculations
6. Data validation and quality checks
7. Normalization/scaling
8. Train/validation/test split from compliant neighborhoods only
9. Test set includes both compliant and non-compliant neighborhoods for validation

## Model Architecture

<!-- Document the tabular transformer model design -->

### Tabular Transformer (FT-Transformer) Design
- **Architecture**: FT-Transformer (single primary model)
- **Input**: Tabular feature vectors (30+ features) from 15-minute city compliant neighborhoods only (training)
- **Output**: Probability distribution over 8 NEXI service categories (supports multi-service prediction)
- **Learning Approach**: Exemplar-based learning - learns optimal service distribution patterns directly from compliant neighborhoods, then generalizes to identify gaps in non-compliant neighborhoods

### Training Configuration
- **Model Type**: Transformer encoder over feature tokens (multi-class classification head)
- **Training Data**: Exclusively from 15-minute city compliant neighborhoods
- **Loss Function**: Distance-based loss (hybrid approach)
  - Primary: Network-based walking distance from predicted service category to nearest actual service of that type (via OSMnx)
  - Secondary: Classification component (cross-entropy) for robust learning
  - Normalize distances by 15-minute walk distance (≈ 1.2 km) for comparability
- **Hyperparameters**:
  - d_token, n_layers, n_heads, dropout, learning rate, weight decay, batch size
  - Tune via validation (or lightweight hyperparameter search)
- **Early Stopping**: Based on validation loss with patience
- **Multi-Service Prediction**: Optional capability to predict multiple needed services simultaneously

## Evaluation Metrics

<!-- Document the metrics used to evaluate the model -->

### Primary Metrics (Distance-Based)
- **Distance-Based Loss**: Network-based walking distance from predicted service category to nearest actual service
- **Normalized Distance**: Distances normalized by 15-minute walk distance (≈ 1.2 km)
- **15-Minute Alignment**: Percentage of predictions within 15-minute threshold
- **Comparative Loss**: Loss measured on both compliant and non-compliant neighborhoods
- **Statistical Validation**: t-test or Mann-Whitney U test to verify significantly lower loss on compliant neighborhoods

### Secondary Metrics (Classification)
- **Accuracy, F1-score** (macro/micro), **Precision, Recall, Log Loss**
- **Per-Class Metrics**: Confusion matrix, per-class F1 scores, precision/recall per service category
- **Probability Distribution Metrics**: Entropy, top-k accuracy

### Interpretability
- **Attention Analysis**: Feature-token attention patterns (global patterns + local examples)
- **SHAP Values**: Optional post-hoc explanations if feasible on trained model
- **Feature Importance**: Analysis of which features drive predictions

### Domain-Specific Validation
- **Principle Alignment**: Validate that compliant neighborhoods have significantly lower loss (shorter distances)
- **Comparative Analysis**: Statistical comparison of loss distributions between compliant and non-compliant neighborhoods
- **Intervention Probability Distributions**: Analyze predicted service category distributions

## Code Conventions

<!-- Document the coding standards and patterns used in this project -->

### Python Style
- Follow PEP 8
- Use type hints where possible
- Docstrings for all functions and classes (Google or NumPy style)
- Maximum line length: 100 characters
- Use meaningful variable names that reflect domain concepts

### ML Code Patterns
- Separate data, model, and training logic (modular design)
- Use configuration files (YAML/JSON) for hyperparameters
- Make code reproducible (set random seeds for all stochastic operations)
- Version control data preprocessing steps
- Always validate data splits maintain neighborhood balance
- Use OSMnx for network-based distance calculations (never Euclidean for walkability)

### File Naming
- Use snake_case for Python files
- Descriptive names that indicate purpose
- Keep functions focused and modular
- Prefix test files with `test_`
- Use consistent naming: `*_extractor.py`, `*_loader.py`, `*_engineer.py` for data collection

### Architecture Patterns
- **Exemplar-Based Learning**: Always train exclusively on compliant neighborhoods
- **Distance-Based Loss**: Use network-based walking distance via OSMnx
- **Comparative Evaluation**: Always evaluate on both compliant and non-compliant neighborhoods
- **Statistical Validation**: Include statistical tests (t-test, Mann-Whitney U) for validation

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
- Version model checkpoints (`.pt`) with descriptive names including date/experiment ID
- Document findings and insights (attention analysis, optional SHAP analysis)
- Compare different hyperparameter configurations and model sizes
- Track validation against 15-minute city principles
- Document comparative analysis results (compliant vs non-compliant neighborhoods)
- **Always record**: Distance-based loss on both neighborhood types, statistical test results, 15-minute alignment percentages
- Store experiment outputs in `experiments/runs/` with timestamped directories

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
- **Exemplar-Based Learning**: Model trains exclusively on 15-minute city compliant neighborhoods to learn optimal service distribution patterns
- **Distance-Based Supervision**: Loss function measures network-based walking distance from predicted service category to nearest actual service of that type
- **Pattern Learning**: Model learns implicit service distribution patterns directly from exemplar neighborhoods, then generalizes to identify gaps in non-compliant neighborhoods
- **Reference-Based**: Compliant neighborhoods serve as reference structures for identifying gaps
- **Validation**: Model success measured by significantly lower loss (shorter distances) on compliant neighborhoods compared to non-compliant ones, indicating learned recognition of optimal patterns
- **Multi-Service Prediction**: Optional capability to predict multiple needed services simultaneously for comprehensive gap analysis

### Timeline
- **MVP**: 2 weeks (14 days)
- **Phase 1**: Data Collection & Feature Engineering (Days 1-5)
- **Phase 2**: Model Training & Evaluation (Days 6-10)
- **Phase 3**: Evaluation & Validation (Days 11-12)
- **Phase 4**: Documentation & Refinement (Days 13-14)

## PIV Loop Conventions

<!-- Standard conventions for Plan-Implement-Verify loops to ensure consistent implementation -->

### PIV Loop Structure

Every feature, component, or enhancement should follow the Plan-Implement-Verify pattern:

1. **PLAN**: Define requirements, design approach, and success criteria
2. **IMPLEMENT**: Write code following project conventions
3. **VERIFY**: Test, validate, and document results

### Planning Phase Standards

**Before implementing any component:**
- ✅ Review PRD.md for requirements and architecture patterns
- ✅ Check CURSOR.md for conventions and existing patterns
- ✅ Define clear success criteria (functional and validation)
- ✅ Identify dependencies and integration points
- ✅ Plan data flow and interfaces
- ✅ Document assumptions and design decisions

**For Data Components:**
- Specify input/output formats
- Define feature computation logic
- Plan error handling and validation
- Consider reproducibility (random seeds, versioning)

**For Model Components:**
- Define architecture and hyperparameters
- Specify loss function (distance-based + classification)
- Plan training data requirements (compliant neighborhoods only)
- Define evaluation metrics (distance-based + statistical validation)

**For Evaluation Components:**
- Define metrics to compute
- Plan statistical validation approach
- Specify visualization requirements
- Plan comparative analysis (compliant vs non-compliant)

### Implementation Phase Standards

**Code Quality:**
- ✅ Follow PEP 8 and project style guidelines
- ✅ Add type hints and docstrings
- ✅ Use meaningful variable names
- ✅ Keep functions focused and modular
- ✅ Handle errors gracefully with appropriate logging

**Architecture Adherence:**
- ✅ Train exclusively on compliant neighborhoods (training set)
- ✅ Use network-based distance calculations (OSMnx, not Euclidean)
- ✅ Implement distance-based loss function
- ✅ Support multi-service prediction capability
- ✅ Include attention analysis and optional SHAP

**Data Handling:**
- ✅ Load neighborhood boundaries from `paris_neighborhoods.geojson`
- ✅ Preserve raw data (never modify)
- ✅ Version processed data
- ✅ Use reproducible splits (save splits to `data/splits/`)
- ✅ Validate data quality at each step

**Model Training:**
- ✅ Set random seeds for reproducibility
- ✅ Save model checkpoints with descriptive names
- ✅ Log training progress (loss, metrics per epoch)
- ✅ Implement early stopping based on validation loss
- ✅ Track experiment configurations

### Verification Phase Standards

**Testing Requirements:**
- ✅ Unit tests for individual functions/components
- ✅ Integration tests for pipelines
- ✅ Model tests (forward pass, output shapes, gradient flow)
- ✅ Test on both compliant and non-compliant neighborhoods

**Validation Requirements:**
- ✅ **Distance-Based Validation**: Measure loss on both neighborhood types
- ✅ **Statistical Validation**: Perform t-test or Mann-Whitney U test
- ✅ **Principle Validation**: Verify significantly lower loss on compliant neighborhoods
- ✅ **15-Minute Alignment**: Check percentage within 15-minute threshold
- ✅ **Reproducibility**: Verify same random seeds produce same results

**Documentation Requirements:**
- ✅ Update code comments and docstrings
- ✅ Document any deviations from PRD or conventions
- ✅ Record experiment results and insights
- ✅ Update relevant sections of CURSOR.md if conventions change
- ✅ Document statistical test results and validation outcomes

### PIV Loop Checklist

**Before starting any PIV loop:**
- [ ] Read relevant sections of PRD.md
- [ ] Review CURSOR.md conventions
- [ ] Understand existing codebase patterns
- [ ] Define clear success criteria

**During implementation:**
- [ ] Follow code conventions (PEP 8, type hints, docstrings)
- [ ] Adhere to architecture patterns (exemplar-based learning, distance-based loss)
- [ ] Write tests alongside implementation
- [ ] Log important decisions and assumptions

**After implementation:**
- [ ] Run all relevant tests
- [ ] Validate against success criteria
- [ ] Perform statistical validation (if applicable)
- [ ] Document results and insights
- [ ] Update documentation if needed

### Common PIV Loop Scenarios

**Adding a New Feature:**
1. **PLAN**: Define feature computation logic, input/output, validation
2. **IMPLEMENT**: Add to `feature_engineer.py`, follow naming conventions
3. **VERIFY**: Test computation, validate against manual calculations, check integration

**Modifying Model Architecture:**
1. **PLAN**: Define changes, impact on loss function, hyperparameters
2. **IMPLEMENT**: Update `transformer.py`, maintain distance-based loss
3. **VERIFY**: Test forward pass, validate training converges, compare metrics

**Adding Evaluation Metric:**
1. **PLAN**: Define metric computation, statistical validation approach
2. **IMPLEMENT**: Add to `metrics.py` or `validate_principles.py`
3. **VERIFY**: Test on known examples, validate statistical tests, document results

**Data Pipeline Changes:**
1. **PLAN**: Define preprocessing steps, data flow, validation checks
2. **IMPLEMENT**: Update preprocessing scripts, maintain reproducibility
3. **VERIFY**: Validate data quality, check splits, verify feature consistency

### Standard Validation Workflow

For any model or component that affects predictions:

1. **Train on compliant neighborhoods only**
2. **Evaluate on both compliant and non-compliant neighborhoods**
3. **Compute distance-based loss for both groups**
4. **Perform statistical test (t-test or Mann-Whitney U)**
5. **Verify compliant neighborhoods have significantly lower loss**
6. **Check 15-minute alignment percentages**
7. **Document all results and statistical significance**

This ensures every implementation follows the core validation principle: the model should perform better (lower loss, shorter distances) on neighborhoods similar to training data.
