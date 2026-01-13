# Product Requirements Document: 15-Minute City Service Gap Prediction System

## 1. Executive Summary

This project develops a Tabular Transformer (FT-Transformer)-based AI model to support data-driven implementation of the 15-minute city model in existing urban contexts. The system addresses the challenge of identifying service gaps and recommending appropriate interventions to improve walkability and accessibility in neighborhoods that do not currently meet 15-minute city principles.

**Core Value Proposition**: The model learns implicit service distribution patterns by training exclusively on neighborhoods already designed according to 15-minute city principles. It uses a distance-based loss function that measures how close the predicted service category is to the nearest actual service of that type. The model then applies these learned patterns to identify gaps and predict the most appropriate service category interventions for locations in non-compliant neighborhoods. Model success is validated by demonstrating significantly lower loss (shorter distances to services) on 15-minute neighborhoods compared to non-compliant ones, indicating the model has learned to recognize optimal service distribution patterns.

**MVP Goal**: Build a working end-to-end system with a Tabular Transformer (FT-Transformer) that trains exclusively on 15-minute city compliant neighborhoods, predicts probability distributions over service categories using distance-based loss, and validates success by comparing loss between compliant and non-compliant neighborhoods, within a 2-week timeline.

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
- ✅ Data collection from OSM and Census sources for Paris neighborhoods (neighborhood boundaries and labels defined in `paris_neighborhoods.geojson`)
- ✅ Training exclusively on 15-minute city compliant neighborhoods
- ✅ Distance-based loss function measuring distance from predicted service category to nearest actual service
- ✅ Training on 8 NEXI service categories (Education, Entertainment, Grocery, Health, Posts and banks, Parks, Sustenance, Shops)
- ✅ Probability vector output over service categories
- ✅ Model validation by comparing loss between compliant and non-compliant neighborhoods
- ✅ Statistical validation that compliant neighborhoods have significantly lower loss
- ✅ Basic evaluation metrics and visualization tools

**Technical**:

- ✅ PyTorch 2.x implementation of FT-Transformer (tabular transformer)
- ✅ Reproducible data preprocessing pipeline
- ✅ Model checkpointing and experiment tracking
- ✅ Attention analysis and SHAP value visualization for interpretability

**Data**:

- ✅ OSM data extraction for service presence/absence and geospatial locations
- ✅ Census data integration for demographic features
- ✅ Feature engineering for walkability metrics
- ✅ Neighborhood boundaries and compliance labels from `paris_neighborhoods.geojson`
- ✅ Train/validation/test splits from compliant neighborhoods only
- ✅ Test set includes both compliant and non-compliant neighborhoods for validation

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

- ⚠️ Multi-label predictions (optional enhancement: predict multiple services simultaneously, but MVP focuses on single best intervention)
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

The system uses a **single-stage Tabular Transformer (FT-Transformer)** with an exemplar-based learning approach:

**Model Training & Prediction**

- Input: Feature vectors from 15-minute city compliant neighborhoods only (training set)
- Process: FT-Transformer learns complex, context-dependent feature interactions and service distribution patterns from exemplar neighborhoods
- Output: Probability vector over 8 service categories indicating most appropriate intervention
- Loss Function: Distance-based loss measuring the distance from the predicted service category to the nearest actual service of that type (using network-based walking distance via OSMnx)
- Learning Approach: Model learns optimal service distribution patterns directly from compliant neighborhoods, then generalizes to identify gaps in non-compliant neighborhoods
- Validation: Model success measured by significantly lower loss (shorter distances) on compliant neighborhoods compared to non-compliant ones

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

**3. Distance-Based Loss Pattern**

- Loss function measures actual distance from predicted service category to nearest service of that type
- Uses network-based walking distance (via OSMnx) for realistic accessibility measurement
- Loss directly relates to 15-minute city principles (shorter distances = better alignment)
- Hybrid approach: combines distance-based loss with classification component for robust learning

**4. Comparative Evaluation Pattern**

- Evaluate model on both compliant and non-compliant neighborhoods
- Validate that loss (distance to services) is significantly lower on compliant neighborhoods
- Statistical validation ensures model has learned to recognize optimal service distribution patterns
- Multi-service prediction capability allows predicting multiple needed services simultaneously

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

- **Purpose**: Learn complex, context-dependent service distribution patterns from exemplar neighborhoods and predict intervention recommendations
- **Operations**:
  - Train exclusively on feature vectors from 15-minute city compliant neighborhoods
  - Learn feature interactions via multi-head self-attention
  - Predict probability distribution over 8 NEXI service categories
  - Use distance-based loss: measure distance from predicted service to nearest actual service of that type
  - Support multi-service prediction (predict multiple needed services simultaneously)
  - Output attention patterns and SHAP values for interpretability
- **Key Features**: Attention-based modeling for tabular features, exemplar-based learning, distance-based loss for domain-relevant optimization, interpretable via attention maps

**3. Validation System**

- **Purpose**: Validate model outputs against 15-minute city principles using distance-based metrics
- **Operations**:
  - Measure loss (distance to nearest service) on both compliant and non-compliant neighborhoods
  - Verify that compliant neighborhoods have significantly lower loss than non-compliant ones
  - Statistical validation (e.g., t-test) to ensure difference is significant
  - Check alignment with 15-minute accessibility thresholds (normalize distances by 15-minute walk distance ≈ 1.2 km)
- **Key Features**: Distance-based evaluation, comparative analysis, statistical validation, domain-relevant metrics

**4. Evaluation Metrics**

- **Purpose**: Measure model performance and alignment with goals using distance-based metrics
- **Operations**:
  - Compute distance-based loss (distance to nearest service of predicted category)
  - Normalize distances by 15-minute walk distance (≈ 1.2 km) for comparability
  - Measure alignment with 15-minute principles (percentage of predictions within 15-minute threshold)
  - Compare loss distributions between compliant and non-compliant neighborhoods
  - Statistical tests (t-test, Mann-Whitney U) to validate significant differences
  - Probability distribution metrics (entropy, top-k accuracy) as secondary metrics
- **Key Features**: Domain-specific metrics, interpretable scores, statistical validation, visualization support

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
3. ✅ Distance-based loss function correctly computes distances to nearest services
4. ✅ Model loss is significantly lower on compliant neighborhoods compared to non-compliant ones (statistical validation)
5. ✅ Model recommendations align with 15-minute city accessibility principles (distances within 15-minute threshold)
6. ✅ Feature extraction pipeline produces consistent, reproducible features
7. ✅ Evaluation metrics demonstrate model learning (not random predictions, clear separation between neighborhood types)

### Functional Requirements

**Data Pipeline**:

- ✅ Extract features for all locations in 6 neighborhoods
- ✅ Compute all 30+ features from feature specification
- ✅ Create train/validation/test splits with proper stratification
- ✅ Handle missing data appropriately

**Model Training**:

- ✅ Train FT-Transformer exclusively on 15-minute city compliant neighborhoods
- ✅ Model learns context-dependent feature interactions and optimal service distribution patterns
- ✅ Distance-based loss function implemented (network-based walking distance via OSMnx)
- ✅ Training converges (validation loss decreases; metrics improve)
- ✅ Model checkpoints saved and loadable (`.pt`)

**Evaluation**:

- ✅ Loss (distance to services) measured on both compliant and non-compliant neighborhoods
- ✅ Statistical validation confirms significantly lower loss on compliant neighborhoods
- ✅ Validation confirms alignment with 15-minute principles (distances within threshold)
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
- ✅ Feature engineering pipeline (all 30+ features)
- ✅ Load neighborhood boundaries and labels from `paris_neighborhoods.geojson`
- ✅ Feature extraction for all Paris neighborhoods (both compliant and non-compliant)
- ✅ Data validation and quality checks
- ✅ Train/validation/test splits from compliant neighborhoods only
- ✅ Test set includes both compliant and non-compliant neighborhoods for validation

**Validation Criteria**:

- All features computed correctly (spot-check against manual calculations)
- Data coverage complete for all neighborhoods
- Splits maintain neighborhood balance

### Phase 2: Model Training & Evaluation (Days 6-10)

**Goal**: Train FT-Transformer and establish evaluation framework

**Deliverables**:

- ✅ FT-Transformer model implementation (PyTorch)
- ✅ Distance-based loss function implementation (network-based walking distance)
- ✅ Training script with hyperparameter tuning
- ✅ Model training exclusively on compliant neighborhoods
- ✅ Probability distribution outputs
- ✅ Multi-service prediction capability (optional)
- ✅ Attention analysis utilities (and optional SHAP)
- ✅ Trained model checkpoint (`.pt`)

**Validation Criteria**:

- Model outputs valid probability distributions (sum to 1, non-negative)
- Distance-based loss function correctly computes distances to nearest services
- Training metrics improve over baseline
- Model training completes successfully on compliant neighborhoods
- Loss decreases during training, indicating learning of service distribution patterns

### Phase 3: Evaluation & Validation (Days 11-12)

**Goal**: Comprehensive evaluation against 15-minute city principles

**Deliverables**:

- ✅ Loss measurement on both compliant and non-compliant neighborhoods
- ✅ Statistical validation (t-test, Mann-Whitney U) showing significantly lower loss on compliant neighborhoods
- ✅ Principle-based validation metrics (percentage within 15-minute threshold)
- ✅ Attention analysis and (optional) SHAP visualization
- ✅ Results documentation and visualization
- ✅ Final model evaluation report with statistical significance tests

**Validation Criteria**:

- Compliant neighborhoods show significantly lower loss (shorter distances to services)
- Statistical tests confirm significant difference between compliant and non-compliant neighborhoods
- Model recommendations align with 15-minute accessibility targets (distances within threshold)
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

**Risk**: Model loss doesn't show significant difference between compliant and non-compliant neighborhoods, or distances don't align with 15-minute thresholds

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

The model uses an **exemplar-based learning** approach with **distance-based supervision**:

- Model trains exclusively on 15-minute city compliant neighborhoods to learn optimal service distribution patterns
- Training uses distance-based loss: for each prediction, measure the distance from the predicted service category to the nearest actual service of that type
- Distance calculations use network-based walking distance (via OSMnx) for realistic accessibility measurement
- Loss function combines distance-based component with classification component for robust learning
- Model learns implicit service distribution patterns from exemplar neighborhoods, then generalizes to identify gaps in non-compliant neighborhoods
- Validation: Model success is measured by significantly lower loss (shorter distances) on compliant neighborhoods compared to non-compliant ones, indicating the model has learned to recognize optimal patterns
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
- **Validation Strategy**: Test model on both compliant and non-compliant neighborhoods; validate that loss is significantly lower on compliant neighborhoods

