# Distance-Based Loss Function for 15-Minute City Service Gap Prediction Model

## Overview

This document describes the distance-based loss function used in the 15-minute city service gap prediction model. The loss function uses a **similarity-based approach** where:

1. The model predicts an 8-dimensional probability vector over service categories
2. A target probability vector is constructed from actual walking distances to the nearest service in each category
3. The loss measures similarity between the predicted and target vectors

This approach leverages the 15-minute city principle: in well-designed neighborhoods, services are distributed such that closer services should have higher probability values. By comparing the model's predictions to distance-based target vectors, we ensure the model learns to predict services that are actually accessible nearby.

---

## Loss Function Architecture

The loss function uses **Kullback-Leibler (KL) divergence** to measure the difference between the model's predicted probability vector and a distance-based target probability vector:

```
L = KL(P_target || P_predicted)
```

For a batch of N samples:

```
L = (1/N) × Σᵢ KL(P_target_i || P_predicted_i)
L = (1/N) × Σᵢ Σⱼ P_target_ij × log(P_target_ij / P_predicted_ij)
```

Where:
- `P_predicted_i` = 8-dimensional probability vector output by the model for sample `i`
- `P_target_i` = 8-dimensional probability vector constructed from actual walking distances to nearest services for sample `i`
- `P_target_ij` = target probability for category `j` in sample `i`
- `P_predicted_ij` = predicted probability for category `j` in sample `i`

**Why KL Divergence?**
- Standard metric for comparing probability distributions in machine learning
- Asymmetric: penalizes predictions that are confidently wrong (high predicted probability for categories with low target probability)
- Provides well-behaved gradients for optimization
- Directly measures how well the predicted distribution matches the distance-based accessibility pattern

---

## Target Vector Construction (P_target)

The target probability vector is constructed from actual walking distances to the nearest service in each category.

### Step-by-Step Calculation

#### Step 1: Distance Vector Calculation

For each sample location `i` and each of the 8 service categories `j`:

```
d_ij = network_distance(location_i, nearest_service(category_j))
```

This creates a distance vector `D_i = [d_i0, d_i1, ..., d_i7]` where each element represents the walking distance (in meters) to the nearest service in that category.

**Important**: The distance uses **network-based walking distance**, not Euclidean distance. This means:
- The distance follows actual walkable paths (streets, sidewalks, pedestrian ways)
- It accounts for obstacles, barriers, and street network topology
- It provides realistic walking distances that align with 15-minute city principles

**Categories** (8 total):
1. Education
2. Entertainment
3. Grocery
4. Health
5. Posts and banks
6. Parks
7. Sustenance
8. Shops

#### Step 2: Distance-to-Probability Conversion

Convert the distance vector to a probability vector using a **temperature-scaled softmax** transformation:

```
P_target_ij = exp(-d_ij / τ) / Σⱼ exp(-d_ij / τ)
```

Where:
- `d_ij` = walking distance (in meters) to the nearest service in category `j` for location `i`
- `τ` (temperature) = scaling parameter that controls how quickly probability decreases with distance (in meters)
- The exponential ensures that **closer distances → higher probabilities**
- The normalization (softmax) ensures that **Σⱼ P_target_ij = 1** (valid probability distribution)

**Temperature Parameter (`τ`)**:
- **Lower `τ`** (e.g., 100-200m): Sharper distribution, only very close services get high probabilities
- **Higher `τ`** (e.g., 400-500m): Smoother distribution, more services get moderate probabilities
- **Recommended value**: `τ = 200` meters
  - This means a service 200m away gets approximately `exp(-1) ≈ 0.37` relative weight before normalization
  - Services within ~200m get high probabilities, aligning with 15-minute city accessibility principles
  - Services beyond ~600m get very low probabilities

**Rationale**: In a 15-minute city, services are optimally distributed. Closer services should have higher probability values, reflecting that they are more accessible and relevant to that location. The temperature parameter controls the "accessibility radius" - how far a service can be and still be considered relevant.

---

## Complete Loss Formula

For a batch of N samples, the loss is calculated as:

```
L = (1/N) × Σᵢ Σⱼ P_target_ij × log(P_target_ij / P_predicted_ij)
```

Where:
- `P_target_i` = distance-based probability vector for sample `i` (constructed from walking distances)
- `P_predicted_i` = model's predicted probability vector for sample `i`
- `P_target_ij` is computed as: `exp(-d_ij / τ) / Σⱼ exp(-d_ij / τ)`
- `d_ij = network_distance(location_i, nearest_service(category_j))`
- `τ = 200` meters (temperature parameter)

**Properties of KL Divergence Loss**:
- **Range**: `[0, ∞)` where 0 means perfect match between distributions
- **Asymmetric**: Penalizes predictions that are confidently wrong (high predicted probability for categories with low target probability)
- **Differentiable**: Provides smooth gradients for optimization
- **Interpretable**: Directly measures how well predicted probabilities match distance-based accessibility patterns

---

## Example Calculation

### Scenario

Consider a single sample (location) in a neighborhood:

**Input**:
- Location: (48.8566°N, 2.3522°E) - a point in Paris
- Model prediction: `P_predicted = [0.10, 0.15, 0.25, 0.20, 0.05, 0.10, 0.10, 0.05]`
  - Category 0: Education (0.10)
  - Category 1: Entertainment (0.15)
  - Category 2: Grocery (0.25) ← **highest predicted probability**
  - Category 3: Health (0.20)
  - Category 4: Posts and banks (0.05)
  - Category 5: Parks (0.10)
  - Category 6: Sustenance (0.10)
  - Category 7: Shops (0.05)

**Step 1: Calculate Distance Vector**

For each category, find the nearest service and calculate network walking distance:

```
D = [d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7]
D = [1200, 800, 300, 500, 600, 400, 350, 900]  (in meters)
```

Meaning:
- Nearest Education service: 1200m away
- Nearest Entertainment service: 800m away
- Nearest Grocery service: 300m away ← **closest service**
- Nearest Health service: 500m away
- Nearest Post/bank: 600m away
- Nearest Park: 400m away
- Nearest Sustenance: 350m away
- Nearest Shop: 900m away

**Step 2: Convert Distance Vector to Target Probability Vector**

Using the softmax-like transformation with temperature `τ = 200` meters:

```
P_target_j = exp(-d_j / 200) / Σⱼ exp(-d_j / 200)
```

Calculate unnormalized scores:
```
exp(-1200/200) = exp(-6.0) = 0.0025
exp(-800/200)  = exp(-4.0) = 0.0183
exp(-300/200)  = exp(-1.5) = 0.2231  ← highest (closest)
exp(-500/200)  = exp(-2.5) = 0.0821
exp(-600/200)  = exp(-3.0) = 0.0498
exp(-400/200)  = exp(-2.0) = 0.1353
exp(-350/200)  = exp(-1.75) = 0.1738
exp(-900/200)  = exp(-4.5) = 0.0111

Sum = 0.6970
```

Normalize to get target probabilities:
```
P_target = [0.0036, 0.0263, 0.3202, 0.1178, 0.0715, 0.1942, 0.2494, 0.0160]
```

**Step 3: Calculate Similarity Loss (KL Divergence)**

```
L = Σⱼ P_target_j × log(P_target_j / P_predicted_j)
```

```
L = 0.0036 × log(0.0036/0.10) + 0.0263 × log(0.0263/0.15) 
  + 0.3202 × log(0.3202/0.25) + 0.1178 × log(0.1178/0.20)
  + 0.0715 × log(0.0715/0.05) + 0.1942 × log(0.1942/0.10)
  + 0.2494 × log(0.2494/0.10) + 0.0160 × log(0.0160/0.05)

L ≈ 0.0036 × (-3.40) + 0.0263 × (-1.73) + 0.3202 × (0.25) 
  + 0.1178 × (-0.53) + 0.0715 × (0.36) + 0.1942 × (0.66)
  + 0.2494 × (0.91) + 0.0160 × (-1.14)

L ≈ 0.95
```

**Interpretation**:
- The target vector correctly assigns highest probability (0.32) to Grocery (300m away), which is indeed the closest service
- The model predicted Grocery with 0.25 probability (highest in its prediction), which aligns well
- The model also predicted Health with 0.20, but Health is 500m away (target probability: 0.12)
- The model under-predicted Sustenance (0.10 predicted vs 0.25 target), which is only 350m away
- Loss of 0.95 indicates moderate alignment; lower loss would indicate better match between predicted and distance-based probabilities

---

## Key Implementation Details

### 1. Network Distance Calculation

The network distance is computed using OSMnx library:

```python
import osmnx as ox
import networkx as nx

# Load network graph for the neighborhood
G = ox.load_graphml('network.graphml')

# Calculate shortest path distance
distance = nx.shortest_path_length(
    G, 
    origin_node, 
    destination_node, 
    weight='length'
)
```

**Requirements**:
- Network graph must be extracted with `network_type='walk'` to include only walkable paths
- Graph includes 100m buffer around neighborhood boundaries to ensure services near edges are accessible
- Distance is in meters

### 2. Distance Vector Calculation

For each sample location, the system calculates distances to all 8 categories:

1. For each category `j`, loads the category-specific service file: `services_by_category/{category_j}.geojson`
2. Finds the service location closest to the sample location in that category
3. Calculates network distance from sample location to that nearest service
4. Repeats for all 8 categories to build the distance vector `D = [d_0, d_1, ..., d_7]`

**Optimization**: Service locations are pre-extracted and organized by category for fast lookup during training. Distance calculations can be parallelized across categories.

### 3. Distance-to-Probability Conversion

The distance vector is converted to a probability vector using temperature-scaled softmax:

```
P_target_ij = exp(-d_ij / τ) / Σⱼ exp(-d_ij / τ)
```

Where `τ = 200` meters (temperature parameter).

**Temperature Parameter (`τ = 200m`)**:
- Controls how sharply probabilities decrease with distance
- Services within ~200m get high probabilities
- Services beyond ~600m get very low probabilities
- This value aligns with 15-minute city accessibility principles (services within a few minutes' walk are highly relevant)

### 4. Handling Missing Services

If no service of a category exists in the neighborhood, assign a penalty distance of `2 × D_15min = 2400m` (twice the 15-minute walk distance). This ensures:
- The target vector correctly reflects that the service is very far away (low probability)
- The model learns that missing services should have low predicted probabilities
- Consistent handling across all samples (fixed penalty rather than variable maximum distance)

---

## Configuration Parameters

The loss function behavior is controlled by parameters in `models/config.yaml`:

```yaml
loss:
  type: "distance_similarity"  # Distance-based similarity loss using KL divergence
  temperature: 200  # Temperature parameter τ for distance-to-probability conversion (in meters)
  use_network_distance: true  # Use OSMnx network distance (not Euclidean)
  missing_service_penalty: 2400  # Distance (meters) to use when service category is missing
```

**Parameters**:
- `temperature` (default: 200m): Controls how quickly probabilities decrease with distance in the target vector
- `use_network_distance` (default: true): Whether to use network-based walking distances (recommended)
- `missing_service_penalty` (default: 2400m): Distance to assign when a service category has no services in the neighborhood (2 × 15-minute walk distance)

---

## Why This Loss Function?

### Alignment with 15-Minute City Principles

1. **Distance-Based Targets**: The target probability vector is constructed from actual walking distances, directly encoding 15-minute city accessibility principles
2. **Realistic Accessibility**: Network-based distances reflect actual walking routes, not straight-line distances
3. **Proximity as Probability**: Closer services receive higher probability values in the target vector, aligning with the principle that accessible services are more relevant
4. **Multi-Category Learning**: Unlike argmax-based approaches, the model learns the full distribution over all categories, not just the top prediction

### Model Learning

1. **Exemplar-Based Learning**: Model trains on compliant neighborhoods where services are optimally distributed according to 15-minute city principles
2. **Distribution Matching**: The model learns to predict probability distributions that match distance-based accessibility patterns
3. **Gap Identification**: In non-compliant neighborhoods, high loss indicates service gaps (predicted probabilities don't match actual accessibility)
4. **Actionable Predictions**: The similarity loss directly measures how well predictions align with actual service accessibility

### Validation Strategy

The model is validated by comparing loss between:
- **Compliant neighborhoods**: Should have low loss (predicted probabilities match distance-based accessibility)
- **Non-compliant neighborhoods**: Should have higher loss (predictions don't align with actual service distribution)

**Success Criterion**: Compliant neighborhoods must show significantly lower loss (statistical test: t-test or Mann-Whitney U test).

**Key Insight**: In a well-designed 15-minute city, the model's predicted service probabilities should correlate with actual walking distances—closer services should be predicted with higher probability. This loss function directly enforces this relationship.

---

## Mathematical Properties

### Target Probability Vector Properties

- **Range**: Each element `P_target_j ∈ [0, 1]`
- **Sum**: `Σⱼ P_target_j = 1` (proper probability distribution)
- **Distance Relationship**: `P_target_j` is inversely related to `d_j` (closer distances → higher probabilities)
- **Temperature Sensitivity**: Lower `τ` creates sharper distributions (winner-take-all), higher `τ` creates smoother distributions

### Loss Range

- **KL Divergence Loss**: `[0, ∞)`
  - 0 = perfect match (distributions are identical)
  - Smaller values = better alignment between predicted and target distributions
  - Typical values in practice: 0.1 - 2.0 (depends on how well model learns accessibility patterns)

### Gradient Behavior

- **Differentiable**: KL divergence is fully differentiable with respect to `P_predicted`
- **Probability Constraints**: The model's output layer must use softmax to ensure `P_predicted` is a valid probability distribution (sums to 1)
- **Target Vector**: `P_target` is fixed for each sample (computed from distances), so gradients only flow through `P_predicted`
- **Gradient Formula**: `∂L/∂P_predicted_ij = -P_target_ij / P_predicted_ij` (for KL divergence)

### Optimization

The loss function is minimized during training using standard gradient descent (Adam optimizer). The model learns to:
1. Predict probability distributions that match distance-based accessibility patterns
2. Assign higher probabilities to service categories that are actually closer to the location
3. Learn the relationship between location features and service accessibility in 15-minute cities

---

## References

- **OSMnx Documentation**: https://osmnx.readthedocs.io/
- **15-Minute City Principles**: Based on Carlos Moreno's concept
- **Network Analysis**: Shortest path algorithms in NetworkX
- **Configuration File**: `models/config.yaml`

---

## Appendix: Service Categories

The model predicts over 8 NEXI service categories:

1. **Education**: college, driving_school, kindergarten, language_school, music_school, school, university
2. **Entertainment**: arts_center, cinema, community_center, theatre, etc.
3. **Grocery**: supermarket, bakery, convenience, greengrocer, etc.
4. **Health**: clinic, dentist, doctors, hospital, pharmacy, etc.
5. **Posts and banks**: ATM, bank, post_office
6. **Parks**: park, dog_park
7. **Sustenance**: restaurant, pub, bar, cafe, fast_food, etc.
8. **Shops**: department_store, general, kiosk, mall, boutique, clothes, etc.

Each category has multiple OSM tags that map to it. Services can belong to multiple categories (e.g., ice_cream appears in both Grocery and Sustenance).

---

*Document Version: 2.0*  
*Last Updated: January 2025*  
*Project: AI4SI - 15-Minute City Service Gap Prediction Model*
