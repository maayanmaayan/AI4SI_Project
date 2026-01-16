# Feature: Feature Engineering Pipeline for Spatial Graph Transformer

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Implement a comprehensive feature engineering pipeline that generates star graphs for the Spatial Graph Transformer model. The pipeline processes raw OSM and Census data to create target points with neighbor grid cells, computes 33 features per point (demographics, built form, services, walkability), calculates network-based distances, and constructs distance-based target probability vectors for the loss function.

**Core Functionality:**
- Generate regular grid of target points within neighborhood boundaries
- Create neighbor grid cells around each target point
- Filter neighbors by network walking distance (not Euclidean) - **WITH OPTIMIZATIONS**
- Compute 33 features per point: demographics (17), built form (4), services (8), walkability (4)
- Build star graph structure: target (node 0) + neighbors (nodes 1-N) with edge attributes
- Calculate distance-based target probability vectors for loss function
- Save processed features as Parquet files for efficient loading

**⚠️ CRITICAL OPTIMIZATION STRATEGY:**
Network distance filtering is the most expensive operation (~70-80% of total time). The plan implements a **phased approach** with optimizations:

1. **Phase 3.1 (START HERE)**: Basic implementation + **Euclidean pre-filtering** (reduces computation by ~90%)
2. **Phase 3.2**: Robust error handling (prevents crashes, improves reliability)
3. **Phase 3.3**: Distance caching (additional 20-30% performance improvement)

**Expected Performance:**
- Without optimization: ~13 minutes per neighborhood
- With Phase 3.1 (Euclidean pre-filtering): ~1-2 minutes per neighborhood
- With Phase 3.3 (caching): ~30-60 seconds per neighborhood

## User Story

As a **ML engineer/researcher**
I want to **generate processed features and star graphs from raw OSM and Census data**
So that **I can train the Spatial Graph Transformer model to learn service distribution patterns from 15-minute city compliant neighborhoods**

## Problem Statement

The project requires a feature engineering pipeline that:
1. **Generates spatial context** - Creates target points and neighbor grid cells within network walking distance
2. **Computes comprehensive features** - Extracts 33 features per point from multiple data sources (Census, OSM, network)
3. **Handles variable graph sizes** - Builds star graphs with variable numbers of neighbors (no padding needed)
4. **Calculates network distances** - Uses OSMnx for realistic walking distances (not Euclidean)
5. **Constructs target vectors** - Creates distance-based probability vectors for the loss function
6. **Ensures reproducibility** - Caches results, handles errors gracefully, validates data quality

Current state: Raw data collection is complete (OSM extractor, Census loader), but feature engineering pipeline is missing. Without this pipeline, model training cannot proceed.

## Solution Statement

Implement a modular `FeatureEngineer` class that:
- Generates target points using regular grid sampling within neighborhoods
- Creates neighbor grid cells around each target point
- Filters neighbors by network walking distance using OSMnx
- Computes 33 features per point by integrating Census, OSM, and network data
- Builds star graph structures with edge attributes [dx, dy, euclidean_distance, network_distance]
- Calculates distance-based target probability vectors for loss function
- Saves processed data as Parquet files organized by compliance status
- Follows existing codebase patterns (logging, error handling, configuration management)

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: High
**Primary Systems Affected**: 
- `src/data/collection/` - New feature_engineer.py module
- `data/processed/features/` - Output directory for processed features
- `scripts/` - New run_feature_engineering.py script
**Dependencies**: 
- OSM data from `OSMExtractor` (services, buildings, network graphs)
- Census data from `CensusLoader` (demographic features)
- OSMnx for network distance calculations
- GeoPandas for spatial operations
- NumPy for feature arrays

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `src/data/collection/osm_extractor.py` (lines 32-985) - Why: Pattern for loading OSM data, network graphs, services by category. Shows how to use `ox.load_graphml()`, `ox.distance.nearest_nodes()`, service category structure
- `src/data/collection/census_loader.py` (lines 33-1617) - Why: Pattern for loading Census data, demographic feature extraction. Shows how to load Parquet files, extract features from IRIS data
- `src/utils/helpers.py` (lines 1-479) - Why: Utility functions for neighborhoods, service categories, distance normalization. Use `load_neighborhoods()`, `get_service_category_mapping()`, `get_service_category_names()`, `ensure_dir_exists()`, `save_dataframe()`
- `src/utils/config.py` (lines 1-193) - Why: Configuration management pattern. Use `get_config()` to load config.yaml settings
- `src/utils/logging.py` (lines 1-122) - Why: Logging pattern. Use `get_logger(__name__)` for module logging
- `models/config.yaml` (lines 1-71) - Why: Configuration parameters for feature engineering (grid_cell_size_meters, walk_15min_radius_meters, temperature, etc.)
- `docs/distance-based-loss-calculation.md` (lines 1-387) - Why: Algorithm for computing target probability vectors from distances. Shows temperature-scaled softmax formula
- `docs/graph-transformer-architecture.md` (lines 1-251) - Why: Star graph structure specification, edge attributes format, feature requirements
- `PRD.md` (lines 669-705) - Why: Feature specification, 33 features breakdown, graph structure requirements
- `CURSOR.md` (lines 150-163) - Why: Feature engineering rationale, network distance filtering requirements

### New Files to Create

- `src/data/collection/feature_engineer.py` - Main FeatureEngineer class with all feature computation methods
- `scripts/run_feature_engineering.py` - Script to run feature engineering pipeline on all neighborhoods
- `tests/unit/test_feature_engineer.py` - Unit tests for feature engineering functions
- `tests/integration/test_feature_engineering_pipeline.py` - Integration tests for full pipeline

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [OSMnx Documentation - Network Distance](https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.distance.nearest_nodes)
  - Specific section: `ox.distance.nearest_nodes()` for finding nearest network nodes
  - Why: Required for network distance calculations between points
- [OSMnx Documentation - Graph Loading](https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.io.load_graphml)
  - Specific section: `ox.load_graphml()` for loading saved network graphs
  - Why: Need to load network graphs saved by OSM extractor
- [NetworkX Shortest Path](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html#networkx.algorithms.shortest_paths.generic.shortest_path_length)
  - Specific section: `nx.shortest_path_length()` with weight='length'
  - Why: Calculate network walking distance between two nodes
- [GeoPandas Spatial Join](https://geopandas.org/en/stable/docs/reference/api/geopandas.sjoin.html)
  - Specific section: `gpd.sjoin()` for spatial joins
  - Why: Match points to IRIS units for demographic features
- [Shapely Geometry Operations](https://shapely.readthedocs.io/en/stable/manual.html#spatial-analysis-methods)
  - Specific section: `point.within()`, `polygon.contains()`, buffer operations
  - Why: Spatial filtering and geometric operations for grid generation

### Patterns to Follow

**Naming Conventions:**
- Class names: `PascalCase` (e.g., `FeatureEngineer`)
- Function names: `snake_case` (e.g., `generate_target_points`)
- Variable names: `snake_case` (e.g., `neighborhood_name`, `target_point`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `WALK_15MIN_METERS`)
- File names: `snake_case.py` (e.g., `feature_engineer.py`)

**Error Handling:**
- Use try-except blocks for file I/O and network operations
- Log warnings for missing data, use defaults where appropriate
- Raise `ValueError` for invalid inputs (negative distances, empty polygons)
- Return empty DataFrames/arrays rather than None for missing data
- Pattern from `osm_extractor.py` (lines 209-215): Try-except with logging, return empty GeoDataFrame on failure

**Logging Pattern:**
```python
from src.utils.logging import get_logger
logger = get_logger(__name__)

logger.info(f"Processing {neighborhood_name}")
logger.warning(f"Low service count: {count}")
logger.error(f"Failed to load network: {e}")
```

**Configuration Access:**
```python
from src.utils.config import get_config
config = get_config()
grid_size = config["features"]["grid_cell_size_meters"]
walk_radius = config["features"]["walk_15min_radius_meters"]
```

**Directory Structure Pattern:**
- Follow `osm_extractor.py` pattern: `data/raw/osm/{compliant|non_compliant}/{neighborhood}/`
- Output: `data/processed/features/{compliant|non_compliant}/{neighborhood}/target_points.parquet`
- Use `ensure_dir_exists()` from helpers before saving

**Data Loading Pattern:**
- Load neighborhoods: `load_neighborhoods(geojson_path)` from helpers
- Load Census: `pd.read_parquet(census_path)` 
- Load OSM network: `ox.load_graphml(network_path)`
- Load services: `gpd.read_file(service_path)` for GeoJSON

**Service Category Pattern:**
- Use `get_service_category_names()` from helpers for consistent category order
- Use `get_service_category_mapping()` to understand category structure
- Service files: `services_by_category/{category.lower().replace(' ', '_')}.geojson`

**Saving Data Pattern:**
- Use `save_dataframe(df, path, format="parquet")` from helpers
- Save as Parquet for efficient loading (not CSV for large arrays)

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation - Core Infrastructure

**Goal**: Set up FeatureEngineer class structure, configuration loading, and basic utilities

**Tasks:**
- Create `FeatureEngineer` class with `__init__` method
- Load configuration parameters from config.yaml
- Set up logging and error handling patterns
- Create output directory structure helpers

### Phase 2: Spatial Operations - Grid Generation

**Goal**: Implement target point and grid cell generation

**Tasks:**
- Implement `generate_target_points()` - regular grid within neighborhoods
- Implement `generate_grid_cells()` - neighbor grid cells around target points
- Handle coordinate system conversions (WGS84 to metric for grid generation)
- Add validation for empty neighborhoods, small areas

### Phase 3: Network Distance Calculations (Phased Approach)

**Goal**: Implement network-based distance filtering with optimizations, error handling, and caching

**Phase 3.1: Basic Implementation with Euclidean Pre-filtering** ⚡ **START HERE**
- Implement `_prefilter_by_euclidean()` - pre-filter grid cells by Euclidean distance (keep only within 1500m)
  - **CRITICAL OPTIMIZATION**: Reduces computation by ~90% (400 cells → ~50 cells)
- Implement `compute_network_distance()` - calculate walking distance with basic error handling
- Implement `filter_by_network_distance()` - filter neighbors by network distance (with pre-filtering)
- Add basic error handling: try-except for network failures, return `float('inf')` on error
- Calculate edge attributes: [dx, dy, euclidean_distance, network_distance]
- **VALIDATION**: Verify pre-filtering works, basic filtering works, error handling catches failures

**Phase 3.2: Robust Error Handling** 🛡️
- Enhance `compute_network_distance()` with comprehensive error handling:
  - Validate network graph (empty, missing nodes, missing 'length' attribute)
  - Handle disconnected components (check if nodes in same component)
  - Coordinate system validation and conversion
  - Euclidean fallback when network calculation fails (multiply by 1.3)
- Add validation in `filter_by_network_distance()`:
  - Check if network graph is valid before processing
  - Handle cases where too many cells are filtered out
  - Log warnings for all error scenarios
- Comprehensive logging for debugging
- **VALIDATION**: Test all error scenarios, verify fallbacks work, check logs

**Phase 3.3: Caching and Performance Optimization** 🚀
- Implement distance caching system:
  - Create `_distance_cache` dictionary: `{(target_id, cell_id): distance}`
  - Check cache before computing network distance
  - Store computed distances in cache
  - Cache key: use rounded coordinates (10m precision) or point IDs
- Optimize network node lookup:
  - Pre-compute nearest nodes for all target points and grid cells
  - Store in dictionary for fast lookup
- Add progress tracking:
  - Log progress every N target points (e.g., every 10 points)
  - Estimate remaining time
- Add cache statistics logging (hit rate, cache size)
- **VALIDATION**: Verify caching reduces computation time, check cache hit rate, verify correctness

### Phase 4: Feature Computation - Demographics

**Goal**: Compute 17 demographic features from Census data

**Tasks:**
- Load Census data for neighborhood
- Implement spatial join to match points to IRIS units
- Extract demographic features: population_density, ses_index, car_commute_ratio, etc.
- Handle missing data with neighborhood averages or warnings

### Phase 5: Feature Computation - Built Form

**Goal**: Compute 4 built form features from OSM building data

**Tasks:**
- Load building GeoDataFrame for neighborhood
- Calculate building density (count within radius)
- Calculate building count, average levels, floor area per capita
- Use spatial operations for efficient filtering

### Phase 6: Feature Computation - Services

**Goal**: Compute 8 service count features within 15-minute walk radius

**Tasks:**
- Load service GeoDataFrames by category
- For each category, count services within `walk_15min_radius_meters` from point
- Use network distance (not Euclidean) for accurate counts
- Handle missing categories (return 0 counts)

### Phase 7: Feature Computation - Walkability

**Goal**: Compute 4 walkability features from network and OSM data

**Tasks:**
- Calculate intersection density (nodes within radius)
- Calculate average block length (edge lengths in local area)
- Calculate pedestrian street ratio
- Calculate sidewalk presence indicator

### Phase 8: Target Vector Computation

**Goal**: Compute distance-based target probability vectors for loss function

**Tasks:**
- Implement `compute_target_probability_vector()` - find nearest services, calculate distances
- Convert distances to probabilities using temperature-scaled softmax
- Handle missing services with penalty distance
- Return probability vector of shape (8,)

### Phase 9: Integration - Star Graph Construction

**Goal**: Combine all components to build complete star graphs

**Tasks:**
- Implement `process_target_point()` - orchestrate all steps for one target point
- Build star graph structure: target + neighbors with edge attributes
- Implement `process_neighborhood()` - process all target points in neighborhood
- Save results as Parquet files with proper schema

### Phase 10: Pipeline Script and Testing

**Goal**: Create runnable script and comprehensive tests

**Tasks:**
- Create `scripts/run_feature_engineering.py` - main pipeline script
- Add unit tests for each feature computation function
- Add integration tests for full pipeline
- Validate output data quality and schema

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### CREATE src/data/collection/feature_engineer.py

- **IMPLEMENT**: Create `FeatureEngineer` class with `__init__` method
- **PATTERN**: Follow `OSMExtractor.__init__` pattern (osm_extractor.py:56-69)
- **IMPORTS**: 
  ```python
  from pathlib import Path
  from typing import Any, Dict, List, Optional, Tuple
  import geopandas as gpd
  import numpy as np
  import pandas as pd
  import networkx as nx
  import osmnx as ox
  from shapely.geometry import Point, Polygon
  from src.utils.config import get_config
  from src.utils.logging import get_logger
  from src.utils.helpers import (
      ensure_dir_exists,
      get_compliant_neighborhoods,
      get_non_compliant_neighborhoods,
      get_service_category_names,
      load_neighborhoods,
      save_dataframe,
  )
  ```
- **CONFIG**: Load `features.*` and `loss.*` parameters from config.yaml
- **CACHING**: Initialize `self._distance_cache: Dict[Tuple, float] = {}` for Phase 3.3
- **VALIDATE**: `python -c "from src.data.collection.feature_engineer import FeatureEngineer; f = FeatureEngineer(); print('OK')"`

### IMPLEMENT generate_target_points method

- **IMPLEMENT**: Generate regular grid of target points within neighborhood polygon
- **ALGORITHM**:
  1. Get neighborhood polygon from GeoDataFrame
  2. Calculate bounding box (minx, miny, maxx, maxy)
  3. Generate grid points with `sampling_interval_meters` spacing
  4. Filter points inside polygon using `polygon.contains(point)`
  5. Create GeoDataFrame with columns: `target_id`, `neighborhood_name`, `label`, `geometry`
- **PATTERN**: Use `shapely.ops.unary_union` if polygon is MultiPolygon
- **EDGE CASES**: Empty polygon → return empty GeoDataFrame, log warning
- **COORDINATES**: Convert WGS84 to metric CRS (EPSG:3857) for grid generation, convert back to WGS84
- **VALIDATE**: Test with small neighborhood, verify points are inside polygon

### IMPLEMENT generate_grid_cells method

- **IMPLEMENT**: Generate regular grid cells around target point
- **ALGORITHM**:
  1. Calculate bounding box: `[target.x ± radius, target.y ± radius]` where `radius = walk_15min_radius_meters`
  2. Generate grid cells with `cell_size_meters` spacing
  3. Return list of Point objects (cell centers)
- **COORDINATES**: Use metric CRS (EPSG:3857) for grid generation
- **EDGE CASES**: Target near boundary → some cells may be outside (filter later by network distance)
- **VALIDATE**: Verify grid covers expected area, cell spacing is correct

### IMPLEMENT _prefilter_by_euclidean method (OPTIMIZATION - Phase 3.1)

- **IMPLEMENT**: Pre-filter grid cells by Euclidean distance before expensive network calculations
- **ALGORITHM**:
  1. Calculate Euclidean distance from target to each grid cell
  2. Keep only cells within `euclidean_threshold` (default: 1500m, slightly larger than network threshold)
  3. Return filtered list of grid cells
- **RATIONALE**: Reduces 400 cells → ~50 cells, saving ~90% computation time
- **THRESHOLD**: Use 1.25x network threshold (1500m for 1200m network threshold) to account for network distance > Euclidean
- **VALIDATE**: Verify pre-filtering reduces cell count significantly (should be ~10-15% of original)

### IMPLEMENT compute_network_distance method (Phase 3.1)

- **IMPLEMENT**: Calculate network walking distance between two points with error handling
- **ALGORITHM**:
  1. Validate network graph is not empty
  2. Find nearest network node to point1: `ox.distance.nearest_nodes(G, point1.x, point1.y)`
  3. Find nearest network node to point2: `ox.distance.nearest_nodes(G, point2.x, point2.y)`
  4. Check if nodes are in same connected component
  5. Calculate shortest path: `nx.shortest_path_length(G, node1, node2, weight='length')`
  6. Return distance in meters
- **PATTERN**: Follow docs/distance-based-loss-calculation.md:230-235
- **ERROR HANDLING** (Phase 3.2):
  - Points outside network → return `float('inf')`, log warning
  - Disconnected components → return `float('inf')`, log warning
  - Empty graph → raise ValueError with clear message
  - NetworkX errors → catch exception, return `float('inf')`, log error
  - Missing 'length' attribute → use default edge length, log warning
- **VALIDATE**: Test with known points, verify distance is reasonable, test error cases

### IMPLEMENT filter_by_network_distance method (Phase 3.1 with optimizations)

- **IMPLEMENT**: Filter grid cells by network walking distance with Euclidean pre-filtering
- **ALGORITHM** (Phase 3.1 - Basic with pre-filtering):
  1. **Pre-filter by Euclidean distance**: Call `_prefilter_by_euclidean()` to reduce cell count
  2. For each pre-filtered grid cell:
     - Calculate network distance to target using `compute_network_distance()`
     - Calculate Euclidean distance
     - Calculate relative coordinates (dx, dy)
     - If `network_distance <= max_distance_meters`, include in results
     - If `network_distance == float('inf')`, use Euclidean fallback (multiply by 1.3) if within threshold
  3. Return list of dicts: `[{'cell': Point, 'network_distance': float, 'euclidean_distance': float, 'dx': float, 'dy': float}, ...]`
- **ERROR HANDLING** (Phase 3.2):
  - Handle `float('inf')` distances: exclude from results, log warning
  - If too many cells filtered out, log warning (may indicate network coverage issue)
  - Validate network graph before processing
- **CACHING** (Phase 3.3):
  - Check cache for (target_point, cell) distance pairs
  - Store computed distances in cache for reuse
  - Cache key: `(target_id, cell_id)` or `(target_coords, cell_coords)` rounded to 10m
- **VALIDATE**: 
  - Verify pre-filtering reduces computation time significantly
  - Verify all returned neighbors are within max_distance_meters
  - Verify caching reduces repeated calculations

### IMPLEMENT compute_demographic_features method

- **IMPLEMENT**: Compute 17 demographic features from Census data
- **ALGORITHM**:
  1. Load Census data: `pd.read_parquet(census_path)` where `census_path = data/raw/census/compliant/{neighborhood}/census_data.parquet`
  2. Load IRIS boundaries for spatial join
  3. For each point, find containing IRIS unit using spatial join
  4. Extract features from Census DataFrame columns (already computed by CensusLoader)
  5. Handle missing: use neighborhood average or log warning
- **FEATURES**: population_density, ses_index, car_commute_ratio, children_per_capita, elderly_ratio, unemployment_rate, student_ratio, walking_ratio, cycling_ratio, public_transport_ratio, two_wheelers_ratio, retired_ratio, permanent_employment_ratio, temporary_employment_ratio, median_income, poverty_rate, working_age_ratio
- **PATTERN**: Follow `census_loader.py` feature extraction pattern (lines 744-1363)
- **VALIDATE**: Verify all 17 features are computed, check for NaN values

### IMPLEMENT compute_built_form_features method

- **IMPLEMENT**: Compute 4 built form features from OSM building data
- **ALGORITHM**:
  1. Load buildings: `gpd.read_file(buildings_path)` where `buildings_path = data/raw/osm/{status}/{neighborhood}/buildings.geojson`
  2. Create buffer around point (100m radius)
  3. Count buildings within buffer
  4. Calculate building density: count / buffer_area
  5. Calculate average building levels from `building:levels` column
  6. Calculate floor area per capita: sum(building_area * levels) / population
- **FEATURES**: building_density, building_count, average_building_levels, floor_area_per_capita
- **EDGE CASES**: No buildings → return 0 for counts, NaN for averages
- **VALIDATE**: Verify building counts match manual inspection

### IMPLEMENT compute_service_features method

- **IMPLEMENT**: Compute 8 service count features within 15-minute walk radius
- **ALGORITHM**:
  1. Load services by category: `gpd.read_file(service_path)` for each category
  2. For each category:
     - Find services within `walk_15min_radius_meters` from point
     - Use network distance (not Euclidean) - iterate through services, calculate network distance, count those within radius
     - Return count
  3. Return array: `[count_education, count_entertainment, count_grocery, count_health, count_posts_banks, count_parks, count_sustenance, count_shops]`
- **PATTERN**: Use `get_service_category_names()` for consistent order
- **SERVICE PATHS**: `data/raw/osm/{status}/{neighborhood}/services_by_category/{category.lower().replace(' ', '_')}.geojson`
- **EDGE CASES**: Missing category file → return 0 count, log warning
- **OPTIMIZATION**: Pre-filter services by Euclidean distance before network distance calculation
- **VALIDATE**: Verify service counts match manual inspection

### IMPLEMENT compute_walkability_features method

- **IMPLEMENT**: Compute 4 walkability features from network and OSM data
- **ALGORITHM**:
  1. **Intersection density**: Count network nodes with degree >= 3 within 200m radius
  2. **Average block length**: Calculate average edge length in local area (200m radius)
  3. **Pedestrian street ratio**: Load pedestrian infrastructure, calculate ratio of pedestrian ways to total streets
  4. **Sidewalk presence**: Binary indicator (1 if sidewalks found in area, 0 otherwise)
- **FEATURES**: intersection_density, average_block_length, pedestrian_street_ratio, sidewalk_presence
- **DATA SOURCES**: Network graph (nodes, edges), pedestrian_infrastructure.geojson
- **EDGE CASES**: No network data → return 0 or NaN
- **VALIDATE**: Verify walkability metrics are reasonable

### IMPLEMENT compute_point_features method

- **IMPLEMENT**: Orchestrate all feature computations for a single point
- **ALGORITHM**:
  1. Call `compute_demographic_features()` → 17 features
  2. Call `compute_built_form_features()` → 4 features
  3. Call `compute_service_features()` → 8 features
  4. Call `compute_walkability_features()` → 4 features
  5. Concatenate into numpy array of shape `(33,)` with dtype `float32`
  6. Handle missing values: fill NaN with 0 or neighborhood average
- **RETURN**: `np.ndarray` of shape `(33,)` with dtype `float32`
- **VALIDATE**: Verify array shape is (33,), no NaN values, dtype is float32

### IMPLEMENT compute_target_probability_vector method

- **IMPLEMENT**: Compute distance-based target probability vector for loss function
- **ALGORITHM** (from docs/distance-based-loss-calculation.md:44-98):
  1. For each of 8 service categories:
     - Load service GeoDataFrame for category
     - Find nearest service to target point (use network distance)
     - Calculate network walking distance
     - If no service found, use `missing_service_penalty` (default: 2400m)
  2. Build distance vector: `D = [d_0, d_1, ..., d_7]`
  3. Convert to probability vector using temperature-scaled softmax:
     - `P_j = exp(-d_j / τ) / Σⱼ exp(-d_j / τ)`
     - Temperature `τ` from config (default: 200m)
  4. Return probability vector of shape `(8,)`
- **PATTERN**: Follow docs/distance-based-loss-calculation.md algorithm exactly
- **VALIDATE**: Verify probabilities sum to 1.0, all values in [0, 1]

### IMPLEMENT process_target_point method

- **IMPLEMENT**: Process single target point: generate neighbors, compute features, build star graph
- **ALGORITHM**:
  1. Generate grid cells around target: `generate_grid_cells()`
  2. Filter by network distance: `filter_by_network_distance()`
  3. Compute target point features: `compute_point_features(target_point)`
  4. For each neighbor:
     - Compute neighbor features: `compute_point_features(neighbor_cell)`
     - Store with edge attributes: network_distance, euclidean_distance, dx, dy
  5. Compute target probability vector: `compute_target_probability_vector()`
  6. Return dict with: `target_features`, `neighbor_data` (list of dicts), `target_prob_vector`, `num_neighbors`
- **RETURN**: Dict with all graph data for one target point
- **VALIDATE**: Verify star graph structure is correct, edge attributes are present

### IMPLEMENT process_neighborhood method

- **IMPLEMENT**: Process all target points in a neighborhood
- **ALGORITHM**:
  1. Generate target points: `generate_target_points(neighborhood)`
  2. Load network graph: `ox.load_graphml(network_path)`
  3. For each target point:
     - Call `process_target_point()`
     - Store results in list
  4. Convert to DataFrame with columns: `target_id`, `neighborhood_name`, `label`, `target_features`, `neighbor_data`, `target_prob_vector`, `target_geometry`, `num_neighbors`
  5. Return DataFrame
- **CACHING**: Check if output file exists, skip if cached (unless force=True)
- **VALIDATE**: Verify DataFrame has correct schema, all target points processed

### IMPLEMENT process_all_neighborhoods method

- **IMPLEMENT**: Process all neighborhoods, organized by compliance status
- **ALGORITHM**:
  1. Load neighborhoods: `load_neighborhoods(geojson_path)`
  2. Split into compliant and non-compliant
  3. For each neighborhood:
     - Call `process_neighborhood()`
     - Save to `data/processed/features/{status}/{neighborhood}/target_points.parquet`
  4. Generate summary statistics
- **PATTERN**: Follow `osm_extractor.extract_all_neighborhoods()` pattern (lines 845-933)
- **VALIDATE**: Verify all neighborhoods processed, summary statistics are correct

### CREATE scripts/run_feature_engineering.py

- **IMPLEMENT**: Main script to run feature engineering pipeline
- **FUNCTIONALITY**:
  1. Load neighborhoods from `paris_neighborhoods.geojson`
  2. Initialize `FeatureEngineer`
  3. Call `process_all_neighborhoods()`
  4. Print summary statistics
- **PATTERN**: Follow `scripts/run_census_loader.py` or `scripts/sanity_check_raw_data.py` structure
- **VALIDATE**: Script runs without errors, produces output files

### CREATE tests/unit/test_feature_engineer.py

- **IMPLEMENT**: Unit tests for feature engineering functions
- **TESTS**:
  - `test_generate_target_points()` - verify grid generation
  - `test_generate_grid_cells()` - verify neighbor grid
  - `test_compute_network_distance()` - verify distance calculation
  - `test_filter_by_network_distance()` - verify filtering
  - `test_compute_demographic_features()` - verify demographic extraction
  - `test_compute_built_form_features()` - verify building features
  - `test_compute_service_features()` - verify service counts
  - `test_compute_walkability_features()` - verify walkability metrics
  - `test_compute_point_features()` - verify full feature array
  - `test_compute_target_probability_vector()` - verify target vector
- **FIXTURES**: Use test data from `tests/fixtures/`
- **VALIDATE**: `pytest tests/unit/test_feature_engineer.py -v`

### CREATE tests/integration/test_feature_engineering_pipeline.py

- **IMPLEMENT**: Integration tests for full pipeline
- **TESTS**:
  - `test_process_target_point()` - verify complete target point processing
  - `test_process_neighborhood()` - verify neighborhood processing
  - `test_output_schema()` - verify Parquet file schema
  - `test_feature_consistency()` - verify features are consistent across runs
- **VALIDATE**: `pytest tests/integration/test_feature_engineering_pipeline.py -v`

---

## TESTING STRATEGY

### Unit Tests

**Scope**: Test each feature computation function independently with mock data

**Requirements**:
- Use pytest fixtures for test data (small neighborhoods, mock networks)
- Test edge cases: empty data, missing files, points outside network
- Verify feature arrays have correct shape and dtype
- Check for NaN values and handle appropriately

**Example Test Structure**:
```python
def test_compute_demographic_features():
    # Setup: Create mock Census data
    # Execute: Call compute_demographic_features()
    # Assert: Verify 17 features returned, no NaN values
```

### Integration Tests

**Scope**: Test full pipeline with real data from one neighborhood

**Requirements**:
- Use actual OSM and Census data from test neighborhood
- Verify star graph structure is correct
- Check Parquet file can be loaded and has correct schema
- Validate feature distributions are reasonable

### Edge Cases

**Specific edge cases to test**:
- Empty neighborhood (no target points)
- Very small neighborhood (1-2 target points)
- Points outside network coverage
- Missing service categories
- Missing Census data
- Missing building data
- Disconnected network components
- Very large neighborhoods (performance test)

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Linting
ruff check src/data/collection/feature_engineer.py

# Formatting
black --check src/data/collection/feature_engineer.py
```

### Level 2: Unit Tests

```bash
# Run unit tests
pytest tests/unit/test_feature_engineer.py -v

# Run with coverage
pytest tests/unit/test_feature_engineer.py --cov=src.data.collection.feature_engineer --cov-report=term-missing
```

### Level 3: Integration Tests

```bash
# Run integration tests
pytest tests/integration/test_feature_engineering_pipeline.py -v
```

### Level 4: Manual Validation

```bash
# Run feature engineering on one neighborhood
python scripts/run_feature_engineering.py --neighborhood "Test Neighborhood" --force

# Verify output file exists and has correct schema
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/features/compliant/test_neighborhood/target_points.parquet')
print(f'Rows: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print(f'Sample target_features shape: {df.iloc[0].target_features.shape}')
print(f'Sample num_neighbors: {df.iloc[0].num_neighbors}')
"

# Validate feature statistics
python -c "
import pandas as pd
import numpy as np
df = pd.read_parquet('data/processed/features/compliant/test_neighborhood/target_points.parquet')
features = np.array([row for row in df.target_features])
print(f'Feature array shape: {features.shape}')
print(f'NaN count: {np.isnan(features).sum()}')
print(f'Feature means: {features.mean(axis=0)[:5]}')
"
```

### Level 5: Full Pipeline Validation

```bash
# Run on all neighborhoods
python scripts/run_feature_engineering.py --all

# Verify all neighborhoods processed
python -c "
from pathlib import Path
processed = list(Path('data/processed/features').rglob('target_points.parquet'))
print(f'Processed neighborhoods: {len(processed)}')
for p in processed:
    print(f'  {p.parent.parent.name}/{p.parent.name}')
"
```

---

## ACCEPTANCE CRITERIA

- [ ] FeatureEngineer class implements all required methods
- [ ] Target points generated correctly within neighborhood boundaries
- [ ] Grid cells generated with correct spacing and coverage
- [ ] Network distance filtering works correctly (only neighbors within radius)
- [ ] All 33 features computed correctly (17 demographics, 4 built form, 8 services, 4 walkability)
- [ ] Star graph structure built correctly (target + neighbors with edge attributes)
- [ ] Target probability vectors computed correctly (sum to 1.0, temperature-scaled softmax)
- [ ] Output Parquet files have correct schema and can be loaded
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Code follows project conventions (PEP 8, type hints, docstrings)
- [ ] No linting errors
- [ ] Documentation updated (docstrings, README if needed)
- [ ] Pipeline script runs successfully on test neighborhood
- [ ] Output data validated (no NaN values, reasonable feature distributions)

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes (unit + integration)
- [ ] No linting or type checking errors
- [ ] Manual testing confirms feature engineering works
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability

---

## NOTES

### Design Decisions

1. **Coordinate System**: Convert WGS84 to EPSG:3857 (Web Mercator) for grid generation, then convert back to WGS84 for storage. This ensures accurate metric distances for grid cell generation.

2. **Network Distance Optimization (CRITICAL)**: 
   - **Euclidean Pre-filtering**: Pre-filter grid cells by Euclidean distance (1500m threshold) before expensive network calculations. This reduces computation by ~90% (400 cells → ~50 cells).
   - **Caching**: Cache computed network distances between point pairs to avoid redundant calculations.
   - **Fallback Strategy**: Use Euclidean distance × 1.3 as fallback when network calculation fails.

3. **Phased Implementation Strategy**:
   - **Phase 3.1**: Start with basic implementation + Euclidean pre-filtering (essential optimization)
   - **Phase 3.2**: Add robust error handling (prevents crashes, improves reliability)
   - **Phase 3.3**: Add caching (further performance improvement, can be added later)

4. **Feature Storage**: Store features as numpy arrays in Parquet files. This is efficient for loading and preserves array structure. Use `object` dtype for arrays in DataFrame, or use Parquet's native array support.

5. **Missing Data Handling**: Use neighborhood averages for missing demographic features, 0 for missing service counts, NaN for missing built form features (to be handled in preprocessing).

6. **Caching Strategy**: 
   - **Output Caching**: Check if output file exists before processing. Allow `force=True` to re-process.
   - **Distance Caching**: Cache network distances in memory during processing to avoid redundant calculations (Phase 3.3).

### Performance Optimization Details

**Euclidean Pre-filtering**:
- **Why**: Network distance calculation is expensive (~20ms per pair)
- **How**: Filter grid cells to only those within 1500m Euclidean distance (1.25x network threshold)
- **Impact**: Reduces 400 cells → ~50 cells = 87.5% reduction in network calculations
- **Trade-off**: May exclude some cells that are within network distance but far Euclidean (rare, acceptable)

**Distance Caching**:
- **Why**: Same point pairs may be checked multiple times (neighbors shared between target points)
- **How**: Store `(target_id, cell_id) -> distance` in dictionary
- **Impact**: Can reduce computation by additional 20-30% if many shared neighbors
- **Memory**: Cache size ~O(N × M) where N=target points, M=avg neighbors (manageable)

**Error Handling Strategy**:
- **Network failures**: Return `float('inf')` and log warning (exclude from results)
- **Disconnected components**: Check connectivity before calculating distance
- **Missing data**: Use Euclidean fallback (multiply by 1.3) if within threshold
- **Empty graphs**: Validate before processing, raise clear error

### Performance Considerations

- **Network Distance (CRITICAL)**: Most expensive operation (~70-80% of total time).
  - **Phase 3.1**: Implement Euclidean pre-filtering FIRST (reduces computation by ~90%)
  - **Phase 3.3**: Add distance caching (additional 20-30% improvement)
  - **Future**: Consider parallelization if still too slow (multiprocessing for target points)
- **Service Counting**: Pre-filter by Euclidean distance before network distance calculation (already in plan).
- **Spatial Joins**: Use spatial indexes (rtree) for efficient point-in-polygon queries.
- **Expected Performance**:
  - Without optimization: ~13 minutes per neighborhood (100 target points)
  - With Euclidean pre-filtering (Phase 3.1): ~1-2 minutes per neighborhood
  - With caching (Phase 3.3): ~30-60 seconds per neighborhood

### Known Limitations

- **IRIS Matching**: Points may fall outside IRIS boundaries. Use nearest IRIS or neighborhood average as fallback.
- **Network Coverage**: Some points may be outside network. 
  - **Phase 3.1**: Return `float('inf')` and exclude from results
  - **Phase 3.2**: Use Euclidean distance × 1.3 as fallback if within threshold
- **Service Categories**: Some services may not match any category. These are excluded from counts.
- **Euclidean Pre-filtering**: May rarely exclude cells that are within network distance but far Euclidean (acceptable trade-off for performance).
- **Caching Memory**: Distance cache grows with number of unique point pairs. Monitor memory usage for very large neighborhoods.

### Future Enhancements

- Parallel processing of target points
- Caching of network distances
- Incremental processing (only process new neighborhoods)
- Feature validation and quality checks
- Visualization of generated graphs

---

*Plan Version: 1.0*  
*Created: January 2025*  
*Project: AI4SI - 15-Minute City Service Gap Prediction Model*
