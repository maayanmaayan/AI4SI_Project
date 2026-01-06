# Product Requirements Document: 15-Minute City Service Gap Prediction System

## 1. Executive Summary

This project develops a Tabular Transformer (FT-Transformer)-based AI model to support data-driven implementation of the 15-minute city model in existing urban contexts. The system addresses the challenge of identifying service gaps and recommending appropriate interventions to improve walkability and accessibility in neighborhoods that do not currently meet 15-minute city principles.

**Core Value Proposition**: The model learns implicit service distribution patterns from neighborhoods already designed according to 15-minute city principles, then uses these learned patterns to identify gaps and predict the most appropriate service category interventions for locations in non-compliant neighborhoods. This approach enables scalable, adaptive decision-making in urban renewal processes without requiring explicit rule-based planning systems.

**MVP Goal**: Build a working end-to-end system with a Tabular Transformer (FT-Transformer) that predicts probability distributions over service categories for intervention recommendations, validated against 15-minute city principles, within a 2-week timeline.

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
- ✅ Feature extraction pipeline for 30+ urban/demographic features
- ✅ Data collection from OSM and Census sources for 6 neighborhoods (3 compliant + 3 non-compliant)
- ✅ Training on 8 NEXI service categories (Education, Entertainment, Grocery, Health, Posts and banks, Parks, Sustenance, Shops)
- ✅ Probability vector output over service categories
- ✅ Model validation against 15-minute city principles
- ✅ Comparative analysis between compliant and non-compliant neighborhoods
- ✅ Basic evaluation metrics and visualization tools

**Technical**:

- ✅ PyTorch 2.x implementation of FT-Transformer (tabular transformer)
- ✅ Reproducible data preprocessing pipeline
- ✅ Model checkpointing and experiment tracking
- ✅ Attention analysis and SHAP value visualization for interpretability

**Data**:

- ✅ OSM data extraction for service presence/absence
- ✅ Census data integration for demographic features
- ✅ Feature engineering for walkability metrics
- ✅ Train/validation/test splits with proper stratification

### Out of Scope (Future Phases)

**Functionality**:

- ❌ Real-time API or web interface
- ❌ Multi-city generalization (MVP focuses on single city)
- ❌ Integration with actual urban planning software
- ❌ Cost/feasibility analysis of interventions
- ❌ Temporal dynamics (model is static, not time-series)

**Technical**:

- ❌ Production deployment infrastructure
- ❌ Model serving or inference API
- ❌ Automated data pipeline updates

**Advanced Features**:

- ❌ Multi-label predictions (MVP: single best intervention)
- ❌ Confidence intervals or uncertainty quantification
- ❌ Interactive visualization dashboard
- ❌ Integration with GIS software

## 5. User Stories

1. **As an urban planner**, I want to identify which service category is most needed at a specific location, so that I can prioritize interventions that improve walkability.

2. **As a researcher**, I want to compare model predictions between compliant and non-compliant neighborhoods, so that I can validate whether the model aligns with 15-minute city principles.

3. **As a policy maker**, I want to see probability distributions over service categories, so that I can understand the relative priority of different interventions.

4. **As a data scientist**, I want to understand what patterns the model learned from compliant neighborhoods, so that I can validate the learning approach and interpret recommendations.

5. **As an urban planner**, I want to see feature importance and SHAP values showing which features drive predictions, so that I can understand the reasoning behind recommendations.

6. **As a researcher**, I want reproducible data preprocessing and model training, so that I can validate results and extend the work.

7. **As a domain expert**, I want the model to consider full demographic and built environment context, so that recommendations are appropriate for specific neighborhood characteristics.

8. **As a developer**, I want a simple, robust model architecture that works reliably, so that I can focus on data quality and evaluation rather than complex model debugging.

## 6. Core Architecture & Patterns

### High-Level Architecture

The system uses a **single-stage Tabular Transformer (FT-Transformer)**:

**Model Training & Prediction**

- Input: Feature vectors from all neighborhoods (compliant + non-compliant)
- Process: FT-Transformer learns complex, context-dependent feature interactions and predicts interventions
- Output: Probability vector over 8 service categories indicating most appropriate intervention
- Learning Approach: Model implicitly learns service distribution patterns from compliant neighborhoods through training on all data, with neighborhood compliance status as a feature or through stratified training

### Architecture Diagram

```mermaid
flowchart TD
    subgraph DataCollection["Data Collection Stage"]
        OSM[OSM Data] --> Features1[Feature Extraction]
        Census[Census Data] --> Features1
        Features1 --> AllData["All Neighborhoods<br/>(Compliant + Non-Compliant)"]
    end
    
    subgraph Training["Model Training"]
        AllData --> TTModel[Tabular Transformer<br/>(FT-Transformer)]
        TTModel --> TrainedModel[Trained Model]
    end
    
    subgraph Prediction["Prediction"]
        NewLocation[New Location Features] --> TrainedModel
        TrainedModel --> ProbVector["Probability Vector<br/>over 8 Service Categories"]
    end
    
    subgraph Evaluation["Evaluation"]
        ProbVector --> Validation[Validate Against<br/>15-Minute Principles]
        ProbVector --> Comparison[Comparative Analysis<br/>Compliant vs Non-Compliant]
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
└── CURSOR.md
```

### Key Design Patterns

**1. Single-Stage Transformer Pattern**

- FT-Transformer learns patterns directly from all data
- Model implicitly captures differences between compliant and non-compliant neighborhoods
- Simpler architecture enables faster iteration and debugging

**2. Feature Engineering Pipeline**

- Modular feature computation (demographics, built form, services, walkability)
- Reproducible feature extraction with versioning

**3. Weak Supervision Pattern**

- Use continuous gap scores derived from OSM data as soft labels
- Model learns to refine these scores based on learned patterns

**4. Comparative Evaluation Pattern**

- Always evaluate predictions in context of neighborhood compliance status
- Validate that model outputs differ appropriately between compliant/non-compliant areas

## 7. Tools/Features

### Core Features

**1. Feature Extraction System**

- **Purpose**: Compute 30+ urban/demographic features for each location
- **Operations**:
  - Extract OSM data (services, buildings, walkability features)
  - Integrate Census data (demographics, SES)
  - Compute walkability metrics (intersection density, block length, etc.)
  - Calculate service accessibility (counts, walk times, 15-minute thresholds)
  - Generate composite scores (essential services coverage, 15-minute walk score)
- **Key Features**: Modular design, reproducible computation, efficient OSM queries

**2. Tabular Transformer Model (FT-Transformer)**

- **Purpose**: Learn complex, context-dependent service distribution patterns and predict intervention recommendations
- **Operations**:
  - Train on feature vectors from all neighborhoods
  - Learn feature interactions via multi-head self-attention
  - Predict probability distribution over 8 NEXI service categories
  - Output attention patterns and SHAP values for interpretability
- **Key Features**: Attention-based modeling for tabular features, strong performance on complex tabular relationships, interpretable via attention maps

**3. Validation System**

- **Purpose**: Validate model outputs against 15-minute city principles
- **Operations**:
  - Compare predictions between compliant and non-compliant neighborhoods
  - Verify that compliant neighborhoods receive lower intervention probabilities
  - Check alignment with 15-minute accessibility thresholds
- **Key Features**: Principle-based evaluation, comparative analysis, statistical validation

**4. Evaluation Metrics**

- **Purpose**: Measure model performance and alignment with goals
- **Operations**:
  - Compute probability distribution metrics (entropy, top-k accuracy)
  - Measure alignment with 15-minute principles (coverage scores, walk time targets)
  - Compare distributions between neighborhood types
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

1. ✅ Model completes training without errors
2. ✅ Model outputs probability distributions over 8 service categories
3. ✅ Model predictions show systematic differences between compliant and non-compliant neighborhoods
4. ✅ Compliant neighborhoods receive lower intervention probabilities than non-compliant (validation)
5. ✅ Model recommendations align with 15-minute city accessibility principles
6. ✅ Feature extraction pipeline produces consistent, reproducible features
7. ✅ Evaluation metrics demonstrate model learning (not random predictions)

### Functional Requirements

**Data Pipeline**:

- ✅ Extract features for all locations in 6 neighborhoods
- ✅ Compute all 30+ features from feature specification
- ✅ Create train/validation/test splits with proper stratification
- ✅ Handle missing data appropriately

**Model Training**:

- ✅ Train FT-Transformer on all neighborhoods
- ✅ Model learns context-dependent feature interactions and neighborhood-specific patterns
- ✅ Training converges (validation loss decreases; metrics improve)
- ✅ Model checkpoints saved and loadable (`.pt`)

**Evaluation**:

- ✅ Comparative analysis shows expected differences between neighborhood types
- ✅ Validation confirms alignment with 15-minute principles
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

- ✅ OSM data extraction scripts
- ✅ Census data integration
- ✅ Feature engineering pipeline (all 30+ features)
- ✅ Feature extraction for 6 neighborhoods
- ✅ Data validation and quality checks
- ✅ Train/validation/test splits

**Validation Criteria**:

- All features computed correctly (spot-check against manual calculations)
- Data coverage complete for all neighborhoods
- Splits maintain neighborhood balance

### Phase 2: Model Training & Evaluation (Days 6-10)

**Goal**: Train FT-Transformer and establish evaluation framework

**Deliverables**:

- ✅ FT-Transformer model implementation (PyTorch)
- ✅ Training script with hyperparameter tuning
- ✅ Model training on all neighborhoods
- ✅ Probability distribution outputs
- ✅ Attention analysis utilities (and optional SHAP)
- ✅ Trained model checkpoint (`.pt`)

**Validation Criteria**:

- Model outputs valid probability distributions (sum to 1, non-negative)
- Predictions differ between compliant and non-compliant neighborhoods
- Training metrics improve over baseline
- Model training completes successfully

### Phase 3: Evaluation & Validation (Days 11-12)

**Goal**: Comprehensive evaluation against 15-minute city principles

**Deliverables**:

- ✅ Comparative analysis (compliant vs non-compliant)
- ✅ Principle-based validation metrics
- ✅ Attention analysis and (optional) SHAP visualization
- ✅ Results documentation and visualization
- ✅ Final model evaluation report

**Validation Criteria**:

- Compliant neighborhoods show lower intervention needs
- Model recommendations align with 15-minute accessibility targets
- Results are statistically significant and interpretable

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

### Risk 3: Validation Doesn't Align with 15-Minute Principles

**Risk**: Model predictions don't match expected 15-minute city patterns

**Mitigation**:

- Define clear validation criteria upfront
- Use rule-based baselines for comparison
- Iterate on model architecture and training if validation fails
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

See user-provided CSV with 30+ features including:

- Demographics (population density, SES, car ownership, etc.)
- Built Form (building density, floor area, etc.)
- Services (counts, walk times, 15-minute thresholds for each category)
- Walkability (intersection density, block length, pedestrian infrastructure)
- Composite scores (essential services coverage, 15-minute walk score)

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

The model uses a **weakly supervised / rule-guided learning** approach:

- Training signals are derived from rule-based gap scores reflecting deviations from 15-minute city targets
- Model learns implicit service distribution patterns from neighborhoods designed according to 15-minute city principles
- These patterns serve as reference structures, allowing the model to identify which service additions would most effectively move a given location toward a similar walkable configuration
- Compliant neighborhoods are used as reference cases to validate that the model assigns lower intervention probabilities where service coverage is already adequate

### Data Collection Strategy

- **Neighborhood Selection**: Choose a city and separate it into neighborhoods that are planned by the 15-minute model (3 neighborhoods) and those that aren't (3 neighborhoods)
- **Selection Method**: Use existing urban planning documents to identify compliant vs non-compliant neighborhoods
- **Feature Collection**: Collect features uniformly for all locations in both types of neighborhoods
- **Label Creation**: Use existing OSM data to verify service presence/absence, create soft/heuristic labels based on gap between what exists and what should exist according to 15-minute city principles

