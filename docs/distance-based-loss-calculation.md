# Distance-Based Loss Function for 15-Minute City Service Gap Prediction Model

## Overview

This document describes the distance-based loss function used in the 15-minute city service gap prediction model. The loss function measures how close predicted service categories are to actual services in the neighborhood, using network-based walking distances to align with 15-minute city accessibility principles.

---

## Loss Function Architecture

The model uses a **hybrid loss function** that combines a distance-based component with a classification component:

```
L_total = λ_dist × L_distance + λ_class × L_classification
```

### Weight Parameters

- **λ_dist = 1.0** (distance_weight from configuration)
- **λ_class = 0.5** (classification_weight from configuration)

These weights can be adjusted in `models/config.yaml` to balance the importance of distance accuracy versus classification accuracy.

---

## Distance Component (L_distance)

The distance component measures how far the predicted service category is from the nearest actual service of that type.

### Formula

For a batch of N samples:

```
L_distance = (1/N) × Σᵢ d_normalized(predicted_category_i, location_i)
```

### Step-by-Step Calculation

#### Step 1: Predicted Category Selection

For each sample `i` in the batch:

```
predicted_category_i = argmax(P_i)
```

Where:
- `P_i` = predicted probability distribution over 8 service categories for sample `i`
  - Categories: Education, Entertainment, Grocery, Health, Posts and banks, Parks, Sustenance, Shops
- `predicted_category_i` = the category with the highest predicted probability

#### Step 2: Network Distance Calculation

```
d_raw = network_distance(location_i, nearest_service(predicted_category_i))
```

Where:
- `location_i` = geographic location (latitude, longitude) of sample `i`
- `nearest_service(category)` = nearest actual service of the predicted category in the neighborhood
- `network_distance()` = shortest path walking distance calculated via OSMnx network graph

**Important**: The distance uses **network-based walking distance**, not Euclidean distance. This means:
- The distance follows actual walkable paths (streets, sidewalks, pedestrian ways)
- It accounts for obstacles, barriers, and street network topology
- It provides realistic walking distances that align with 15-minute city principles

#### Step 3: Distance Normalization

```
d_normalized = d_raw / D_15min
```

Where:
- `D_15min = 1200` meters (15-minute walk distance at 5 km/h walking speed)
- If `normalize_by_15min = false` in configuration, then `d_normalized = d_raw`

**Purpose of Normalization**:
- Makes distances comparable across different neighborhoods
- Provides interpretable scores (1.0 = 15-minute walk, 0.5 = 7.5-minute walk, etc.)
- Aligns with 15-minute city accessibility thresholds

---

## Classification Component (L_classification)

The classification component uses standard cross-entropy loss to ensure the model learns correct category predictions:

```
L_classification = - (1/N) × Σᵢ Σⱼ y_ij × log(P_ij)
```

Where:
- `y_ij` = true label (one-hot encoded) for category `j` in sample `i`
- `P_ij` = predicted probability for category `j` in sample `i`
- `N` = batch size

**Purpose**: Ensures the model learns to predict the correct service categories, not just minimize distances.

---

## Complete Loss Formula

Combining both components:

```
L_total = 1.0 × [(1/N) × Σᵢ network_distance(location_i, nearest_service(argmax(P_i))) / 1200]
         + 0.5 × [- (1/N) × Σᵢ Σⱼ y_ij × log(P_ij)]
```

### Simplified Version (Distance-Only Mode)

If `loss.type = "distance_based"` (classification component disabled):

```
L = (1/N) × Σᵢ network_distance(location_i, nearest_service(argmax(P_i))) / 1200
```

---

## Example Calculation

### Scenario

Consider a single sample (location) in a neighborhood:

**Input**:
- Location: (48.8566°N, 2.3522°E) - a point in Paris
- Model prediction: `P = [0.1, 0.7, 0.05, 0.03, 0.02, 0.05, 0.03, 0.02]`
  - Category 0: Education (0.1)
  - Category 1: Grocery (0.7) ← **highest probability**
  - Category 2: Health (0.05)
  - ... (other categories)

**Step 1: Select Predicted Category**
```
predicted_category = argmax(P) = 1 (Grocery)
```

**Step 2: Find Nearest Grocery Store**
- Query `services_by_category/grocery.geojson` for the neighborhood
- Find nearest grocery store to location: 800 meters away (network distance)

**Step 3: Calculate Normalized Distance**
```
d_raw = 800 meters
d_normalized = 800 / 1200 = 0.67
```

**Step 4: Calculate Classification Loss**
- Assume true label is category 1 (Grocery): `y = [0, 1, 0, 0, 0, 0, 0, 0]`
- Classification loss: `-log(0.7) = 0.36`

**Step 5: Calculate Total Loss**
```
L_distance = 0.67
L_classification = 0.36
L_total = 1.0 × 0.67 + 0.5 × 0.36 = 0.67 + 0.18 = 0.85
```

**Interpretation**:
- The predicted service (Grocery) is 0.67 normalized distance units away (about 8 minutes walk)
- The model is confident (70% probability) in the correct category
- Total loss of 0.85 indicates good alignment with 15-minute city principles

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

### 2. Nearest Service Lookup

For each predicted category, the system:

1. Loads the category-specific service file: `services_by_category/{category}.geojson`
2. Finds the service location closest to the sample location
3. Calculates network distance from sample location to service location

**Optimization**: Service locations are pre-extracted and organized by category for fast lookup during training.

### 3. Multi-Service Prediction

If the model predicts multiple services simultaneously:

```
L_distance = (1/N) × Σᵢ Σⱼ d_normalized(location_i, nearest_service(category_j))
```

Where `category_j` are all predicted categories (e.g., top-k predictions or categories above a threshold).

### 4. Handling Missing Services

If no service of the predicted category exists in the neighborhood:

- **Option 1**: Use a penalty distance (e.g., `D_15min = 1200m` or `2 × D_15min = 2400m`)
- **Option 2**: Use maximum distance found in the neighborhood
- **Option 3**: Skip the sample (not recommended for training stability)

**Current Implementation**: Uses penalty distance of `D_15min` to encourage predictions of categories that actually exist nearby.

---

## Configuration Parameters

The loss function behavior is controlled by parameters in `models/config.yaml`:

```yaml
loss:
  type: "distance_based"  # Options: "distance_based", "hybrid", "classification"
  distance_weight: 1.0     # Weight for distance component (λ_dist)
  classification_weight: 0.5  # Weight for classification component (λ_class)
  normalize_by_15min: true    # Whether to normalize distances by 1200m
  use_network_distance: true  # Use OSMnx network distance (not Euclidean)
```

---

## Why This Loss Function?

### Alignment with 15-Minute City Principles

1. **Realistic Accessibility**: Network-based distances reflect actual walking routes, not straight-line distances
2. **15-Minute Threshold**: Normalization by 1200m directly relates to the 15-minute walk standard
3. **Service Proximity**: Encourages predictions of services that are actually nearby and accessible

### Model Learning

1. **Exemplar-Based Learning**: Model trains on compliant neighborhoods where services are optimally distributed
2. **Gap Identification**: In non-compliant neighborhoods, high loss indicates service gaps
3. **Actionable Predictions**: Loss directly measures intervention impact (shorter distances = better accessibility)

### Validation Strategy

The model is validated by comparing loss between:
- **Compliant neighborhoods**: Should have low loss (services are nearby)
- **Non-compliant neighborhoods**: Should have higher loss (service gaps exist)

**Success Criterion**: Compliant neighborhoods must show significantly lower loss (statistical test: t-test or Mann-Whitney U test).

---

## Mathematical Properties

### Loss Range

- **Distance component**: `[0, ∞)` in raw meters, `[0, ∞)` in normalized units
  - 0 = service is at the exact location
  - 1.0 = service is 15 minutes walk away
  - >1.0 = service is beyond 15-minute threshold
- **Classification component**: `[0, ∞)` (cross-entropy)
- **Total loss**: `[0, ∞)`

### Gradient Behavior

- **Distance component**: Provides gradients based on spatial proximity
- **Classification component**: Provides gradients based on probability distribution
- **Hybrid approach**: Combines spatial and categorical learning signals

### Optimization

The loss function is minimized during training using standard gradient descent (Adam optimizer). The model learns to:
1. Predict service categories that exist nearby (distance component)
2. Predict correct categories with high confidence (classification component)

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

*Document Version: 1.0*  
*Last Updated: January 2025*  
*Project: AI4SI - 15-Minute City Service Gap Prediction Model*
