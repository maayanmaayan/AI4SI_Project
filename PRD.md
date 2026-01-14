# Product Requirements Document: 15-Minute City Service Gap Prediction System

## 1. Executive Summary

This project develops a Tabular Transformer (FT-Transformer)-based AI model to support data-driven implementation of the 15-minute city model in existing urban contexts. The system addresses the challenge of identifying service gaps and recommending appropriate interventions to improve walkability and accessibility in neighborhoods that do not currently meet 15-minute city principles.

**Core Value Proposition**: The model learns implicit service distribution patterns by training exclusively on neighborhoods already designed according to 15-minute city principles. It uses a distance-based similarity loss function that compares predicted probability distributions to distance-based target vectors. The model then applies these learned patterns to identify gaps and predict the most appropriate service category interventions for locations in neighborhoods.

**MVP Goal**: Build a working end-to-end system with a Tabular Transformer (FT-Transformer) that trains exclusively on 15-minute city compliant neighborhoods, predicts probability distributions over service categories using distance-based similarity loss, and validates success through evaluation on compliant neighborhoods only, within a 2-week timeline.

## 2. Mission

**Mission Statement**: Enable data-driven urban planning by learning implicit service distribution patterns from 15-minute city compliant neighborhoods and applying these patterns to identify service gaps and recommend interventions that improve walkability and accessibility.

**Core Principles**:

1. **Pattern Learning Over Rules**: Learn implicit patterns from reference neighborhoods rather than encoding explicit planning rules
2. **Context-Aware Recommendations**: Consider full urban and demographic context when predicting service needs
3. **Validation Against Principles**: Model performance evaluated by alignment with 15-minute city principles, not just predictive accuracy
4. **Scalable and Generalizable**: Design for application across different urban contexts
5. **Transparent and Interpretable**: Provide probability distributions, attention analysis, and SHAP values for understanding model decisions

## 3. Target Users

**Primary Users**:

- **Urban Planners**: Professionals implementing 15-minute city initiatives who need data-driven gap analysis
- **Researchers**: Academics studying walkability, urban accessibility, and sustainable city development
- **Policy Makers**: Government officials evaluating neighborhood interventions and resource allocation

**Technical Comfort Level**: Users are domain experts in urban planning but may have varying technical backgrounds. The system should provide interpretable outputs (probability distributions, attention patterns, SHAP values) rather than requiring deep ML expertise.

**Key User Needs**:

- Identify which service categories are most needed at specific locations
- Understand why certain interventions are recommended (interpretability)
- Validate recommendations against established 15-minute city principles
- Compare intervention needs across different neighborhood types

## 4. MVP Scope

### In Scope (MVP)

**Core Functionality**:

- ✅ Single robust Tabular Transformer (FT-Transformer) model
- ✅ Feature extraction pipeline for 20+ urban/demographic features
- ✅ Data collection from OSM and Census sources for Paris neighborhoods (neighborhood boundaries and labels defined in `paris_neighborhoods.geojson`)
- ✅ Training exclusively on 15-minute city compliant neighborhoods
- ✅ Distance-based loss function measuring distance from predicted service category to nearest actual service
- ✅ Training on 8 NEXI service categories (Education, Entertainment, Grocery, Health, Posts and banks, Parks, Sustenance, Shops)
- ✅ Probability vector output over service categories
- ✅ Model evaluation on compliant neighborhoods only (train/validation/test splits)
- ✅ Basic evaluation metrics and visualization tools

**Technical**:

- ✅ PyTorch 2.x implementation of FT-Transformer (tabular transformer)
- ✅ Reproducible data preprocessing pipeline
- ✅ Model checkpointing and experiment tracking
- ✅ Attention analysis and SHAP value visualization for interpretability

**Data**:

- ✅ OSM data extraction for service presence/absence and geospatial locations
- ✅ Census data integration for demographic features
- ✅ Feature engineering for walkability metrics and service counts within the 15-minute walk radius
- ✅ Neighborhood boundaries and compliance labels from `paris_neighborhoods.geojson`
- ✅ Train/validation/test splits from compliant neighborhoods only (all data from compliant neighborhoods)

### Out of Scope (Future Phases)

**Functionality**:

- ❌ Real-time API or web interface
- ❌ Multi-city generalization (MVP focuses on single city)
- ❌ Integration with actual urban planning software
- ❌ Cost/feasibility analysis of interventions
- ❌ Temporal dynamics (model is static, not time-series)
- ❌ Control group evaluation comparing compliant vs non-compliant neighborhoods (post-MVP: after model selection in a later stage, evaluate if model has different success rates on non-compliant neighborhoods)

**Technical**:

- ❌ Production deployment infrastructure
- ❌ Model serving or inference API
- ❌ Automated data pipeline updates

**Advanced Features**:

- ⚠️ Multi-label predictions (optional enhancement: predict multiple services simultaneously, but MVP focuses on single best intervention)
- ❌ Confidence intervals or uncertainty quantification
- ❌ Interactive visualization dashboard
- ❌ Integration with GIS software

## 5. User Stories

1. **As an urban planner**, I want to identify which service category is most needed at a specific location, so that I can prioritize interventions that improve walkability.

2. **As a researcher**, I want to evaluate model predictions on compliant neighborhoods, so that I can validate whether the model learns service distribution patterns from exemplar neighborhoods. (Post-MVP: compare predictions between compliant and non-compliant neighborhoods as a control group evaluation)

3. **As a policy maker**, I want to see probability distributions over service categories, so that I can understand the relative priority of different interventions.

4. **As a data scientist**, I want to understand what patterns the model learned from compliant neighborhoods, so that I can validate the learning approach and interpret recommendations.

5. **As an urban planner**, I want to see feature importance and SHAP values showing which features drive predictions, so that I can understand the reasoning behind recommendations.

6. **As a researcher**, I want reproducible data preprocessing and model training, so that I can validate results and extend the work.

7. **As a domain expert**, I want the model to consider full demographic and built environment context, so that recommendations are appropriate for specific neighborhood characteristics.

8. **As a developer**, I want a simple, robust model architecture that works reliably, so that I can focus on data quality and evaluation rather than complex model debugging.

## 6. Core Architecture & Patterns

### High-Level Architecture

The system uses a **single-stage Tabular Transformer (FT-Transformer)** with an exemplar-based learning approach:

**Model Training & Prediction**

- Input: Multi-point sequences of feature vectors from 15-minute city compliant neighborhoods only (training set)
  - Each location is represented as a sequence: [center point feature vector] + [all grid cells within 15-minute walk radius]
  - Grid cells represent the demographic and built environment context of people who can access the center location
  - All grid cells within the radius are included (no truncation)
- Process: FT-Transformer learns complex, context-dependent feature interactions and spatial patterns via attention over the sequence
- Output: Probability vector over 8 service categories indicating most appropriate intervention
- Loss Function: Distance-based similarity loss using KL divergence between predicted and distance-based target probability vectors
- Learning Approach: Model learns optimal service distribution patterns directly from compliant neighborhoods by understanding spatial context
- Validation: Model evaluation on compliant neighborhoods only (train/validation/test splits from compliant neighborhoods)

### Architecture Diagram

```mermaid
flowchart TD
    subgraph DataCollection["Data Collection Stage"]
        GeoJSON[paris_neighborhoods.geojson<br/>Neighborhood Boundaries & Labels] --> Split[Split Neighborhoods]
        OSM[OSM Data] --> Features1[Feature Extraction]
        Census[Census Data] --> Features1
        Features1 --> CompliantData["15-Minute Compliant<br/>Neighborhoods"]
        Features1 --> NonCompliantData["Non-Compliant<br/>Neighborhoods"]
    end
    
    subgraph Training["Model Training"]
        CompliantData --> TTModel[Tabular Transformer<br/>(FT-Transformer)]
        TTModel --> DistanceLoss[Distance-Based Loss<br/>Distance to Nearest Service]
        DistanceLoss --> TrainedModel[Trained Model]
    end
    
    subgraph Prediction["Prediction"]
        NewLocation[New Location Features] --> TrainedModel
        TrainedModel --> ProbVector["Probability Vector<br/>over 8 Service Categories"]
    end
    
    subgraph Evaluation["Evaluation"]
        ProbVector --> CompliantEval[Measure Loss on<br/>Compliant Neighborhoods]
        ProbVector --> NonCompliantEval[Measure Loss on<br/>Non-Compliant Neighborhoods]
        CompliantEval --> Comparison[Statistical Comparison<br/>Compliant vs Non-Compliant]
        NonCompliantEval --> Comparison
        Comparison --> Validation[Validate: Lower Loss<br/>on Compliant Neighborhoods]
        ProbVector --> Interpretability[Attention & SHAP<br/>Interpretability]
    end
    
    DataCollection --> Training
    Training --> Prediction
    Prediction --> Evaluation
```

### Directory Structure

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
│   ├── transformer.py           # FT-Transformer architecture
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
│   │   ├── train.py             # Transformer training script
│   │   └── hyperparameter_tuning.py # Hyperparameter optimization
│   ├── evaluation/
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── validate_principles.py # 15-minute city validation
│   │   ├── compare_neighborhoods.py # Comparative analysis
│   │   └── visualize.py        # Attention/SHAP/result visualization
│   └── utils/
│       ├── config.py            # Configuration management
│       ├── logging.py            # Logging setup
│       └── helpers.py           # Utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── experiments/
│   └── runs/                     # Experiment outputs
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── requirements.txt
├── README.md
├── PRD.md
├── CURSOR.md
└── paris_neighborhoods.geojson  # Paris neighborhood boundaries and compliance labels
```

### Key Design Patterns

**1. Exemplar-Based Learning Pattern**

- FT-Transformer learns patterns exclusively from 15-minute city compliant neighborhoods
- Model learns optimal service distribution patterns directly from exemplar neighborhoods
- Simpler architecture enables faster iteration and debugging
- Clear validation hypothesis: model should perform better (lower loss) on neighborhoods similar to training data

**2. Feature Engineering Pipeline**

- Modular feature computation (demographics, built form, services, walkability)
- Reproducible feature extraction with versioning

**3. Distance-Based Similarity Loss Pattern**

- Loss function uses similarity-based approach: compares model's predicted probability vector to a distance-based target vector
- Target vector constructed from actual walking distances to nearest service in each category (via OSMnx network distance)
- Distance-to-probability conversion using temperature-scaled softmax (τ=200m): closer services get higher probabilities
- Loss measured using KL divergence between predicted and target probability distributions
- Aligns with 15-minute city principles: model learns that closer services should have higher predicted probabilities

**4. Compliant-Only Evaluation Pattern**

- Evaluate model exclusively on compliant neighborhoods (train/validation/test splits from compliant neighborhoods only)
- Validate model learning through standard evaluation metrics (loss, accuracy, etc.)
- Ensure model learns service distribution patterns from exemplar neighborhoods
- Multi-service prediction capability allows predicting multiple needed services simultaneously

## 7. Tools/Features

### Core Features

**1. Feature Extraction System**

- **Purpose**: Compute 20+ urban/demographic features for each location and its spatial context
- **Operations**:
  - Extract OSM data (services, buildings, walkability features)
  - Integrate Census data (demographics, SES)
  - Compute walkability metrics (intersection density, block length, etc.)
  - Generate regular grid cells (configurable size, default: 100m × 100m) around each prediction location
  - For each location: compute features for center point + all grid cells within 15-minute walk radius
  - Each grid cell gets its own feature vector representing that spatial context
  - Include all grid cells within radius (no truncation)
- **Key Features**: Modular design, reproducible computation, efficient OSM queries, spatial context representation

**2. Tabular Transformer Model (FT-Transformer)**

- **Purpose**: Learn complex, context-dependent service distribution patterns from exemplar neighborhoods and predict intervention recommendations
- **Operations**:
  - Train exclusively on multi-point sequences from 15-minute city compliant neighborhoods
  - Input: Sequence of feature vectors [center point + grid cells within radius] for each location
  - Learn spatial patterns and feature interactions via multi-head self-attention over the sequence
  - Attention mechanism learns relationships between grid cells and identifies which spatial contexts matter most
  - Predict probability distribution over 8 NEXI service categories
  - Use distance-based loss: measure distance from predicted service to nearest actual service of that type
  - Support multi-service prediction (predict multiple needed services simultaneously)
  - Output attention patterns and SHAP values for interpretability
- **Key Features**: Attention-based modeling over spatial sequences, exemplar-based learning, distance-based loss for domain-relevant optimization, interpretable via attention maps showing which grid cells drive predictions

**3. Validation System**

- **Purpose**: Validate model outputs and learning using distance-based similarity metrics on compliant neighborhoods
- **Operations**:
  - Measure similarity-based loss (KL divergence between predicted and target probability vectors) on compliant neighborhoods
  - Evaluate model performance using standard metrics (loss, accuracy, etc.) on train/validation/test splits from compliant neighborhoods
  - Check alignment with 15-minute city principles through distance-based target vectors
- **Key Features**: Distance-based evaluation, domain-relevant metrics, interpretable probability distributions

**4. Evaluation Metrics**

- **Purpose**: Measure model performance and alignment with goals using distance-based similarity metrics
- **Operations**:
  - Compute similarity-based loss (KL divergence between predicted and distance-based target probability vectors)
  - Evaluate on compliant neighborhoods only (train/validation/test splits)
  - Measure alignment with 15-minute city principles through distance-to-probability target vectors
  - Probability distribution metrics (entropy, top-k accuracy) as evaluation metrics
- **Key Features**: Domain-specific metrics, interpretable scores, visualization support

## 8. Technology Stack

### Core Framework

- **Deep Learning**:
  - PyTorch 2.x (training and inference)
  - FT-Transformer implementation (custom `nn.Module` or a lightweight tabular-transformer library)
- **ML Utilities**:
  - scikit-learn 1.3+ (metrics, splits, preprocessing utilities)
- **Data Processing**: 
  - pandas 2.0+ (tabular data)
  - numpy 1.24+ (numerical operations)
  - geopandas 0.14+ (geospatial data)
  - osmnx 1.6+ (OSM network analysis)

### Data Sources & APIs

- **OSM Data**: Overpass API or osmnx library
- **Census Data**: National/regional census APIs or downloaded files
- **Geospatial**: Shapely, Fiona for geometry operations
- **Distance Calculation**: OSMnx for network-based walking distance calculations (not Euclidean)

### Evaluation & Visualization

- **Metrics**: scikit-learn 1.3+ (classification metrics)
- **Interpretability**:
  - Attention analysis (feature-token attention patterns)
  - SHAP 0.43+ (optional; for post-hoc explanations if feasible on the trained model)
- **Visualization**: matplotlib 3.7+, seaborn 0.12+ (plots and analysis)
- **Geospatial Viz**: folium or contextily (optional, for map visualizations)

### Development Tools

- **Testing**: pytest 7.4+
- **Code Quality**: black 23.0+, ruff 0.1+ (linting)
- **Type Checking**: mypy 1.5+ (optional)
- **Experiment Tracking**: TensorBoard or Weights & Biases (optional)

### Data Storage

- **File Formats**: CSV, Parquet (for processed features), GeoJSON (for boundaries)
- **Version Control**: Git with DVC (optional, for data versioning)

## 9. Security & Configuration

### Configuration Management

- **Environment Variables**: API keys for OSM/Census (if needed), data paths
- **Configuration Files**: YAML or JSON configs for:
  - Model hyperparameters
  - Feature engineering parameters
  - Data paths and splits
  - Training settings
- **Secrets**: Store API keys in `.env` file (gitignored)

### Security Scope

- **In Scope**: 
  - Secure handling of API keys
  - Data privacy considerations (anonymized demographic data)
- **Out of Scope**: 
  - User authentication (no user-facing system)
  - Network security (local/research environment)
  - Data encryption at rest (research data, not sensitive)

### Deployment Considerations

- **Current**: Local/research environment
- **Future**: Could be containerized (Docker) for reproducibility
- **No Production Deployment**: MVP is research prototype

## 10. API Specification

**Not Applicable**: This is a research/ML project, not a web service. Model inference will be via Python scripts, not REST API.

**Future Consideration**: If extended to production, could expose via FastAPI or similar framework.

## 11. Success Criteria

### MVP Success Definition

The MVP is successful if:

1. ✅ Model completes training without errors on compliant neighborhoods only
2. ✅ Model outputs probability distributions over 8 service categories
3. ✅ Distance-based similarity loss function correctly computes target vectors from distances
4. ✅ Model evaluation on compliant neighborhoods shows learning (decreasing loss, improving metrics)
5. ✅ Model predictions align with 15-minute city accessibility principles (distance-based target vectors)
6. ✅ Feature extraction pipeline produces consistent, reproducible multi-point sequences (center point + grid cells within radius)
7. ✅ Evaluation metrics demonstrate model learning (not random predictions, improving performance on validation set)

### Functional Requirements

**Data Pipeline**:

- ✅ Extract features for all locations in 6 neighborhoods
- ✅ Generate grid cells and compute all 20+ features for center point + all grid cells within 15-minute walk radius
- ✅ Handle variable-length sequences (different numbers of grid cells per location) with proper padding/masking
- ✅ Create train/validation/test splits with proper stratification
- ✅ Handle missing data appropriately

**Model Training**:

- ✅ Train FT-Transformer exclusively on 15-minute city compliant neighborhoods
- ✅ Model learns context-dependent feature interactions and optimal service distribution patterns
- ✅ Distance-based loss function implemented (network-based walking distance via OSMnx)
- ✅ Training converges (validation loss decreases; metrics improve)
- ✅ Model checkpoints saved and loadable (`.pt`)

**Evaluation**:

- ✅ Similarity-based loss (KL divergence) measured on compliant neighborhoods (train/validation/test splits)
- ✅ Model performance metrics (loss, accuracy, etc.) demonstrate learning on compliant neighborhoods
- ✅ Validation confirms alignment with 15-minute principles through distance-based target vectors
- ✅ Feature importance and SHAP values provide interpretable insights
- ✅ Results are reproducible (same random seeds produce same outputs)

### Quality Indicators

- **Code Quality**: All code follows PEP 8, has type hints, includes docstrings
- **Reproducibility**: Random seeds set, data preprocessing versioned, hyperparameters documented
- **Documentation**: README, code comments, experiment logs
- **Testing**: Unit tests for feature extraction, model components, evaluation metrics

### User Experience Goals

- **Interpretability**: Model outputs are understandable (probability distributions, feature importance, SHAP values)
- **Usability**: Scripts are well-documented and easy to run
- **Transparency**: All decisions and assumptions are documented

## 12. Implementation Phases

### Phase 1: Data Collection & Feature Engineering (Days 1-5)

**Goal**: Build complete data pipeline for feature extraction

**Deliverables**:

- ✅ OSM data extraction scripts (including service locations for distance calculations)
- ✅ Census data integration
- ✅ Grid cell generation pipeline (regular grid, filter cells within 15-minute walk radius)
- ✅ Feature engineering pipeline (compute all 20+ features for center point + all grid cells within radius)
- ✅ Multi-point sequence construction (handle variable-length sequences with padding/masking)
- ✅ Load neighborhood boundaries and labels from `paris_neighborhoods.geojson`
- ✅ Feature extraction for compliant neighborhoods (for MVP model training and evaluation)
- ✅ Data validation and quality checks
- ✅ Train/validation/test splits from compliant neighborhoods only

**Validation Criteria**:

- Grid cells generated correctly (all cells within radius included, no truncation)
- All features computed correctly for center point and grid cells (spot-check against manual calculations)
- Multi-point sequences constructed correctly (variable-length handling, padding/masking)
- Data coverage complete for all neighborhoods
- Splits maintain neighborhood balance

### Phase 2: Model Training & Evaluation (Days 6-10)

**Goal**: Train FT-Transformer and establish evaluation framework

**Deliverables**:

- ✅ FT-Transformer model implementation (PyTorch) - handles multi-point sequences with attention over grid cells
- ✅ Distance-based loss function implementation (network-based walking distance)
- ✅ Training script with hyperparameter tuning
- ✅ Hyperparameter tuning: Train 6-8 model variants with different hyperparameters
- ✅ Model training exclusively on compliant neighborhoods
- ✅ Probability distribution outputs
- ✅ Multi-service prediction capability (optional)
- ✅ Attention analysis utilities (and optional SHAP)
- ✅ Trained model checkpoint (`.pt`)

**Hyperparameter Tuning Strategy**:

- **Number of models**: Train 7 model variants
- **Selection criteria**: Best validation KL divergence loss
- **Strategy**: Start with baseline, then vary each hyperparameter independently (one-at-a-time) to understand individual effects
- All models evaluated on validation set from compliant neighborhoods
- Final model selection based on validation performance metrics

**Model Configurations**:

| Model | Learning Rate | n_layers | Temperature (τ, meters) | Description |
|-------|---------------|----------|------------------------|-------------|
| 1 | 0.001 | 3 | 200 | Baseline (current config) |
| 2 | 0.0005 | 3 | 200 | Lower learning rate |
| 3 | 0.002 | 3 | 200 | Higher learning rate |
| 4 | 0.001 | 2 | 200 | Fewer layers (shallow) |
| 5 | 0.001 | 4 | 200 | More layers (deeper) |
| 6 | 0.001 | 3 | 150 | Lower temperature (sharper target distribution) |
| 7 | 0.001 | 3 | 250 | Higher temperature (smoother target distribution) |

**Fixed hyperparameters** (same for all models):
- `d_token`: 128
- `n_heads`: 4
- `dropout`: 0.1
- `batch_size`: 64
- `weight_decay`: 0.0001
- `activation`: gelu

**Validation Criteria**:

- Model correctly processes multi-point sequences (handles variable-length sequences, attention over grid cells)
- Model outputs valid probability distributions (sum to 1, non-negative)
- Distance-based similarity loss function correctly computes target vectors from distances
- Training metrics improve over baseline
- Model training completes successfully on compliant neighborhoods
- Loss decreases during training, indicating learning of spatial patterns and service distribution patterns
- Attention patterns show meaningful spatial relationships between grid cells
- Best model selected based on validation KL divergence loss

### Phase 3: Evaluation & Validation (Days 11-12)

**Goal**: Comprehensive evaluation against 15-minute city principles

**Deliverables**:

- ✅ Similarity-based loss measurement on compliant neighborhoods (train/validation/test splits)
- ✅ Model performance metrics (loss, accuracy, etc.) demonstrating learning
- ✅ Alignment with 15-minute city principles through distance-based target vectors
- ✅ Attention analysis and (optional) SHAP visualization
- ✅ Results documentation and visualization
- ✅ Final model evaluation report with performance metrics

**Validation Criteria**:

- Model shows learning (decreasing loss, improving metrics on validation set)
- Model predictions align with 15-minute city accessibility principles (distance-based target vectors)
- Evaluation metrics demonstrate model performance on compliant neighborhoods
- Results are reproducible and interpretable

### Phase 4: Documentation & Refinement (Days 13-14)

**Goal**: Finalize documentation and refine model if needed

**Deliverables**:

- ✅ Complete code documentation
- ✅ README with usage instructions
- ✅ Experiment logs and results summary
- ✅ Model architecture documentation
- ✅ Final presentation/report materials

**Timeline Estimate**: 2 weeks (14 days) for complete MVP

## 13. Future Considerations

### Post-MVP Enhancements

**Model Improvements**:

- Multi-label prediction (multiple simultaneous interventions)
- Uncertainty quantification (confidence intervals)
- Temporal dynamics (how needs change over time)
- Transfer learning to other cities

**Feature Enhancements**:

- Additional data sources (satellite imagery, mobility data)
- More sophisticated walkability metrics
- Integration with real-time data streams

**Technical Enhancements**:

- Production deployment (API, web interface)
- Advanced explainability (additional SHAP visualizations, LIME integration)
- Automated retraining pipeline
- Integration with GIS software
- Transformer model implementation for comparison

**Application Extensions**:

- Cost/feasibility analysis of interventions
- Policy impact simulation
- Community engagement tools
- Integration with urban planning workflows

## 14. Risks & Mitigations

### Risk 1: Insufficient Data Quality or Coverage

**Risk**: OSM data incomplete, Census data unavailable, or neighborhoods too small

**Mitigation**:

- Validate data coverage early (Phase 1)
- Have backup neighborhoods identified
- Use data quality checks and imputation strategies
- Consider synthetic data augmentation if needed

### Risk 2: Model Fails to Learn Meaningful Patterns

**Risk**: FT-Transformer predictions are noisy/unstable or don’t distinguish between neighborhood types

**Mitigation**:

- Start with simpler baseline models (logistic regression) to validate approach
- Use attention analysis and (optional) SHAP to debug learning
- Validate that compliant neighborhoods actually differ from non-compliant in features
- Hyperparameter tuning to optimize model performance
- Consider ensemble methods if single model insufficient

### Risk 3: Model Doesn't Learn from Compliant Neighborhoods

**Risk**: Model doesn't learn meaningful patterns from compliant neighborhoods, or loss doesn't decrease during training

**Mitigation**:

- Define clear validation criteria upfront (statistical significance thresholds)
- Use rule-based baselines for comparison
- Ensure distance calculations use network-based walking distance, not Euclidean
- Normalize distances by 15-minute walk distance for better interpretability
- Iterate on model architecture and training if validation fails
- Consider hybrid loss function (distance + classification) if pure distance-based loss is unstable
- Document any discrepancies and analyze causes

### Risk 4: Computational Resources Insufficient

**Risk**: Training too slow or memory issues with full dataset

**Mitigation**:

- Start with smaller subsets for prototyping
- Use mini-batches and mixed precision (if available) for faster transformer training
- Consider cloud resources if local compute insufficient
- Optimize feature computation pipeline

### Risk 5: Timeline Overruns

**Risk**: Data collection or model development takes longer than expected

**Mitigation**:

- Prioritize MVP scope (can reduce to fewer neighborhoods if needed)
- Parallelize work where possible (data collection + model design)
- Keep the FT-Transformer architecture minimal (small depth/width) for rapid iteration
- Focus on core functionality first, polish later
- Use pre-computed features or simplified feature set if needed

## 15. Appendix

### Related Documents

- [CURSOR.md](CURSOR.md) - Project technical documentation
- [README.md](README.md) - Project overview and setup
- Feature specification CSV (provided by user)

### Key Dependencies

- **OSM Data**: OpenStreetMap (open data, no API key needed for basic queries)
- **Census Data**: Depends on selected city/country (may require API registration)
- **15-Minute City Principles**: Based on Carlos Moreno's framework

### Repository Structure

See [CURSOR.md](CURSOR.md) Section "Project Structure" for detailed directory layout.

### Feature Specification

**Input Structure**: Multi-point sequences (not single feature vectors)
- Each prediction location is represented as a sequence of feature vectors
- Sequence structure: [center point] + [all grid cells within 15-minute walk radius]
- Sequence length varies per location (all cells within radius are included)
- Padding/masking used for batching variable-length sequences

**Features per point** (center point and each grid cell):

- **Demographics**: Population density, SES index, car ownership, children per capita, household size, elderly ratio
- **Built Form**: Building density, building count, average building levels, floor area per capita
- **Services**: Counts per category within the 15-minute walk radius from that point's perspective (8 features, configurable via `features.walk_15min_radius_meters` in config.yaml):
  - `count_education_15min`: Number of education services within the 15-minute walk radius
  - `count_entertainment_15min`: Number of entertainment services within the 15-minute walk radius
  - `count_grocery_15min`: Number of grocery services within the 15-minute walk radius
  - `count_health_15min`: Number of health services within the 15-minute walk radius
  - `count_posts_banks_15min`: Number of posts/banks services within the 15-minute walk radius
  - `count_parks_15min`: Number of parks within the 15-minute walk radius
  - `count_sustenance_15min`: Number of sustenance services within the 15-minute walk radius
  - `count_shops_15min`: Number of shops within the 15-minute walk radius
- **Walkability**: Intersection density, average block length, pedestrian street ratio, sidewalk presence

**Grid Cell Generation**:
- Regular grid with configurable cell size (default: 100m × 100m via `features.grid_cell_size_meters`)
- For each prediction location: generate grid cells covering bounding box, filter to include only cells within `walk_15min_radius_meters`
- All cells within radius are included (no truncation)
- Maximum sequence length auto-calculated: `ceil(π × (radius/cell_size)²) + buffer` (configurable via `features.max_context_points`)

**Feature Engineering Rationale**:
- **Multi-point sequences**: Model learns from spatial distribution of people/services, not just aggregated statistics. Grid cells represent demographic and built environment context of people who can access the center location.
- **Service counts vs. walking distances**: Service features use counts within the 15-minute walk radius (configurable via `features.walk_15min_radius_meters` in config.yaml, default: 1200m) rather than walking distances to avoid data leakage. Walking distances are used in the loss function to construct target probability vectors, so including them as features would give the model direct access to what it's trying to predict.
- **Removed composite features**: Composite features (essential services coverage, average walk time to essentials, 15-minute walk score) are removed to simplify the feature set and avoid redundancy. The transformer model can learn these patterns from the individual features if needed.
- **15-minute walk radius**: The configurable walk radius (default: 1200m ≈ 1.2 km) aligns with the 15-minute walk threshold, capturing service density at the neighborhood scale relevant to 15-minute city principles. This radius can be adjusted in `models/config.yaml` via the `features.walk_15min_radius_meters` parameter.
- **Why FT-Transformer**: With multi-point sequences, the transformer's attention mechanism can learn spatial relationships between grid cells, identifying which areas matter most for service gap prediction. This leverages the transformer's strength in learning from sequences.

### Service Categories (NEXI → OSM Mapping)

- **Education**: college, driving_school, kindergarten, language_school, music_school, school, university
- **Entertainment**: arts_center, cinema, community_center, theatre, etc.
- **Grocery**: supermarket, bakery, convenience, greengrocer, etc.
- **Health**: clinic, dentist, doctors, hospital, pharmacy, etc.
- **Posts and banks**: ATM, bank, post_office
- **Parks**: park, dog_park
- **Sustenance**: restaurant, pub, bar, cafe, fast_food, etc.
- **Shops**: department_store, general, kiosk, mall, boutique, clothes, etc.

### Learning Approach

The model uses an **exemplar-based learning** approach with **distance-based supervision**:

- Model trains exclusively on 15-minute city compliant neighborhoods to learn optimal service distribution patterns
- Training uses distance-based similarity loss: KL divergence between predicted probability vectors and distance-based target vectors
- Distance calculations use network-based walking distance (via OSMnx) for realistic accessibility measurement
- Target vectors constructed from distances to nearest services in each category, converted to probabilities using temperature-scaled softmax
- Model learns implicit service distribution patterns from exemplar neighborhoods
- Validation: Model evaluation on compliant neighborhoods only (train/validation/test splits); model success measured by learning (decreasing loss, improving metrics)
- Multi-service prediction capability allows predicting multiple needed services simultaneously for more comprehensive gap analysis

### Data Collection Strategy

- **City Focus**: Paris, France
- **Neighborhood Selection**: Neighborhood boundaries and compliance labels are defined in `paris_neighborhoods.geojson`
  - Compliant neighborhoods: Verified 15-minute city neighborhoods (e.g., Paris Rive Gauche, Clichy-Batignolles, Beaugrenelle)
  - Non-compliant neighborhoods: Neighborhoods not following 15-minute city principles (e.g., Montmartre, Belleville, La Défense)
- **Selection Method**: Neighborhood labels based on official Paris urban planning documents (PLU bioclimatique, paris.fr)
- **Feature Collection**: Collect features uniformly for all locations in both types of neighborhoods
- **Training Strategy**: Train model exclusively on compliant neighborhoods
- **Loss Calculation**: Use OSM data to compute actual distances from each location to nearest services of each category (network-based walking distance)
- **Validation Strategy**: Evaluate model on compliant neighborhoods only (train/validation/test splits from compliant neighborhoods). Post-MVP: After model selection, conduct control group evaluation comparing success rates on compliant vs non-compliant neighborhoods

