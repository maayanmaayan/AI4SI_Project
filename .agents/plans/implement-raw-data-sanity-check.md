# Feature: Raw Data Sanity Check Script

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Implement a comprehensive raw data sanity check script that validates OSM and Census data files against reasonable value ranges based on Paris-specific urban planning data. The script will check all extracted raw data files (services, buildings, network, intersections, census features) for each neighborhood and generate a detailed validation report identifying hard failures, warnings, and summary statistics.

**Core Functionality:**
- Validate OSM data files (services.geojson, buildings.geojson, network.graphml, intersections.geojson, pedestrian features)
- Validate Census data files (census_data.parquet) with demographic and socioeconomic features
- Check file structure, existence, and format validity
- Perform range checks on numeric values based on Paris-specific reasonable bounds
- Perform cross-source consistency checks (area matching, population vs buildings correlation)
- Generate comprehensive validation report (JSON + human-readable summary)
- Support checking all neighborhoods or specific neighborhoods
- Integrate with existing logging and configuration systems

## User Story

As a **data engineer/researcher**
I want to **validate all raw OSM and Census data against reasonable value ranges**
So that **I can catch data quality issues early before they propagate to feature engineering and model training**

## Problem Statement

The project has collected raw OSM and Census data for Paris neighborhoods, but there's currently no systematic validation to ensure:
1. Data files are complete and properly formatted
2. Numeric values fall within reasonable ranges for Paris
3. Cross-source data is consistent (e.g., area calculations match)
4. Missing or corrupted data is detected early

Without validation, data quality issues could propagate through feature engineering and model training, leading to:
- Invalid feature values causing model training failures
- Unrealistic predictions due to bad input data
- Wasted computation time on corrupted data
- Difficult debugging when issues surface later in the pipeline

**Key Challenges:**
1. Defining reasonable value ranges for Paris-specific urban metrics
2. Handling missing data gracefully (some missing is OK, but need to flag excessive missingness)
3. Validating geospatial data (geometry validity, coordinate bounds)
4. Cross-source consistency checks (matching areas, correlating features)
5. Generating actionable reports that identify specific issues

## Solution Statement

Create a `RawDataSanityChecker` class and a command-line script that:
1. **Scans data directories** to discover all neighborhoods and their data files
2. **Validates file structure** (existence, format, size)
3. **Loads and validates OSM data** (services, buildings, network, intersections) with range checks
4. **Loads and validates Census data** (demographic features) with range checks
5. **Performs cross-source consistency checks** (area matching, feature correlations)
6. **Generates validation report** with:
   - Summary statistics per neighborhood
   - Hard failures (must fix)
   - Warnings (investigate)
   - Passed checks (for confidence)
7. **Saves report** as JSON (machine-readable) and prints human-readable summary

The checker will use the comprehensive sanity check list defined in the planning discussion, with hard bounds (failures) and warning bounds (investigate) for each metric.

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: 
- `scripts/` - New `sanity_check_raw_data.py` script
- `src/data/validation/` - New `raw_data_sanity_checker.py` module (optional, or keep in scripts/)
- `models/config.yaml` - Add sanity check configuration section (optional)
**Dependencies**: 
- Existing: `geopandas`, `pandas`, `osmnx`, `networkx` (already in requirements.txt)
- No new external dependencies required

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `scripts/run_census_loader.py` (lines 1-27) - Why: Script structure pattern with summary output
- `src/data/collection/osm_extractor.py` (lines 692-976) - Why: OSM data structure, file paths, extraction metadata pattern
- `src/data/collection/census_loader.py` (lines 711-1300) - Why: Census data structure, feature names, extraction patterns
- `src/utils/helpers.py` (lines 24-479) - Why: Data loading utilities (load_neighborhoods, check_data_quality), file I/O patterns
- `src/utils/logging.py` (lines 1-122) - Why: Logging setup pattern (get_logger, setup_logging)
- `src/utils/config.py` (lines 1-193) - Why: Configuration loading pattern (get_config, load_config)
- `tests/unit/test_helpers.py` (lines 342-377) - Why: Test pattern for data quality checks
- `tests/unit/test_census_loader.py` (lines 1-100) - Why: Test fixture patterns, mock data structures

### New Files to Create

- `scripts/sanity_check_raw_data.py` - Main command-line script for running sanity checks
- `src/data/validation/__init__.py` - Package init file (if creating validation module)
- `src/data/validation/raw_data_sanity_checker.py` - Core validation logic class (optional - can also put in scripts/)
- `tests/unit/test_raw_data_sanity_checker.py` - Unit tests for validation logic

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [Geopandas GeoDataFrame Validation](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
  - Specific section: Geometry validity checks
  - Why: Need to validate geometry.is_valid for all GeoDataFrames
- [NetworkX Graph Validation](https://networkx.org/documentation/stable/reference/classes/graph.html)
  - Specific section: Graph connectivity and component analysis
  - Why: Need to check network connectivity (largest connected component)
- [Pandas Data Validation](https://pandas.pydata.org/docs/user_guide/missing_data.html)
  - Specific section: Missing data detection and handling
  - Why: Need to check for missing values, NaN, Inf in Census data

### Patterns to Follow

**Naming Conventions:**
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Use descriptive names: `check_osm_services()`, `validate_census_features()`
- Follow existing patterns: `*_loader.py`, `*_extractor.py` → `*_checker.py` or `sanity_check_*.py`

**Error Handling:**
- Use try-except blocks for file loading (may fail if files don't exist)
- Log warnings for recoverable issues (e.g., missing optional files)
- Raise exceptions for critical failures (e.g., invalid configuration)
- Pattern from `osm_extractor.py` (lines 214-215): Catch exceptions, log warnings, return empty GeoDataFrame

**Logging Pattern:**
```python
from src.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("Starting validation")
logger.warning("Low service count detected")
logger.error("Invalid geometry found")
```

**Data Loading Pattern:**
```python
from src.utils.helpers import load_neighborhoods, get_compliant_neighborhoods
neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
compliant = get_compliant_neighborhoods(neighborhoods)
```

**File I/O Pattern:**
```python
from pathlib import Path
import json
# Save JSON report
with open(report_path, "w") as f:
    json.dump(report_dict, f, indent=2, default=str)
```

**Report Structure Pattern:**
- Follow pattern from `osm_extractor.py` (lines 927-976): Save JSON metadata + CSV summary
- Use dictionary structure: `{"summary": {...}, "neighborhoods": [{...}, ...], "errors": [...]}`

**Test Pattern:**
- Use pytest fixtures for test data (see `test_census_loader.py`)
- Mock file I/O for unit tests
- Test with real data in integration tests

---

## SANITY CHECK BOUNDS REFERENCE

This section contains the complete reference tables for all sanity check bounds. These values should be implemented as constants in the `RawDataSanityChecker` class.

**Note**: All bounds have been verified against real-world Paris data and urban planning standards through online research. Values are based on:
- Paris 2018-2024 census and demographic data
- Paris urban planning studies and reports
- Paris transportation and mobility statistics
- Paris building and infrastructure metrics

Bounds are designed to catch data quality issues while allowing for reasonable variation across different neighborhoods.

### **OSM Data Checks**

#### **1. Services Data (`services.geojson` & `services_by_category/*.geojson`)**

| Check | Field/Property | Hard Bounds | Warning Bounds | Notes |
|-------|---------------|-------------|----------------|-------|
| **Service count per neighborhood** | Total services | `> 0` | `< 5` (very low) or `> 10,000` (unusually high) | Paris average ~79 amenities/km²; neighborhoods vary |
| **Service density** | Services / area (km²) | `0 < density < 500` | `< 10` or `> 300` | Typical: 50-150 per km² in dense areas |
| **Service categories** | Count per category | `>= 0` | Any category `== 0` (missing category) | All 8 NEXI categories should exist (can be 0) |
| **Geometry validity** | `geometry` column | All geometries valid | Any invalid geometries | Check `geometry.is_valid` |
| **Coordinate bounds** | `geometry` (lon, lat) | `2.2 < lon < 2.4`<br>`48.8 < lat < 48.9` | Outside Paris bounds | Paris approximate bounds |
| **Required columns** | `name`, `amenity`, `category`, `geometry`, `neighborhood_name` | All present | Missing columns | Essential columns must exist |
| **Category values** | `category` column | Must be one of: `["Education", "Entertainment", "Grocery", "Health", "Posts and banks", "Parks", "Sustenance", "Shops"]` | Invalid category names | Match NEXI categories exactly |

#### **2. Buildings Data (`buildings.geojson`)**

| Check | Field/Property | Hard Bounds | Warning Bounds | Notes |
|-------|---------------|-------------|----------------|-------|
| **Building count per neighborhood** | Total buildings | `> 0` | `< 10` (very sparse) or `> 50,000` (unusually dense) | Depends on neighborhood size |
| **Building density** | Buildings / area (km²) | `0 < density < 5,000` | `< 50` or `> 3,000` | Typical: 500-2,000 per km² in dense areas |
| **Building levels** | `building:levels` | `1 <= levels <= 20` | `< 1` or `> 12` (most Paris) | Paris: mostly 5-6 stories, max 12 in most areas |
| **Building area** | `area_m2` | `1 < area_m2 < 100,000` | `< 1` or `> 50,000` | Per-building area in m² |
| **Total building area ratio** | Sum(area_m2) / neighborhood_area | `0.1 < ratio < 0.8` | `< 0.30` or `> 0.9` | Paris typical: 33-48% coverage; warn if <30% (too sparse) or >90% (unrealistic) |
| **Geometry validity** | `geometry` column | All valid | Any invalid | Check polygon validity |
| **Coordinate bounds** | `geometry` | `2.2 < lon < 2.4`<br>`48.8 < lat < 48.9` | Outside Paris bounds | Paris bounds |
| **Missing levels** | `building:levels` is null | `<= 50%` missing | `> 50%` missing | Some missing is OK (OSM incomplete) |

#### **3. Network Data (`network.graphml`, `network_edges.geojson`, `network_nodes.geojson`)**

| Check | Field/Property | Hard Bounds | Warning Bounds | Notes |
|-------|---------------|-------------|----------------|-------|
| **Network nodes count** | Total nodes | `> 0` | `< 10` (too sparse) | Should have reasonable node density |
| **Network edges count** | Total edges | `> 0` | `< 10` (too sparse) | Should connect nodes |
| **Node-to-edge ratio** | edges / nodes | `0.5 < ratio < 3.0` | `< 0.3` or `> 4.0` | Typical: 1.0-2.0 for street networks |
| **Network connectivity** | Largest connected component | `>= 80%` of nodes | `< 50%` of nodes | Paris typical: ~87.5% in largest component; most nodes should be connected |
| **Edge length** | Edge lengths (meters) | `1 < length < 5,000` | `< 0.1` or `> 2,000` | Per-edge length in meters |
| **Average block length** | Derived from network | `50 < avg_block_length < 300` | `< 30` or `> 400` | Paris: 100-150m typical |
| **Geometry validity** | Edge/node geometries | All valid | Any invalid | Check geometry validity |
| **Coordinate bounds** | Node coordinates | `2.2 < lon < 2.4`<br>`48.8 < lat < 48.9` | Outside Paris bounds | Paris bounds |

#### **4. Intersections Data (`intersections.geojson`)**

| Check | Field/Property | Hard Bounds | Warning Bounds | Notes |
|-------|---------------|-------------|----------------|-------|
| **Intersection count** | Total intersections | `> 0` | `< 5` (too sparse) | Should have reasonable intersection density |
| **Intersection density** | Intersections / area (km²) | `20 < density < 500` | `< 10` or `> 400` | Paris central: 150-200 per km² |
| **Node degree** | `degree` column | `3 <= degree <= 10` | `< 3` or `> 12` | Intersections should have degree >= 3 |
| **Geometry validity** | `geometry` | All valid | Any invalid | Check point geometry validity |
| **Coordinate bounds** | `geometry` | `2.2 < lon < 2.4`<br>`48.8 < lat < 48.9` | Outside Paris bounds | Paris bounds |

#### **5. Pedestrian Features Data (`pedestrian_features.geojson` or similar)**

| Check | Field/Property | Hard Bounds | Warning Bounds | Notes |
|-------|---------------|-------------|----------------|-------|
| **Pedestrian feature count** | Total features | `>= 0` | N/A | Can be 0 (OSM may not tag all) |
| **Feature types** | `feature_type` | Must be: `["pedestrian_way", "sidewalk", "crosswalk"]` | Invalid types | Valid OSM pedestrian tags |
| **Geometry validity** | `geometry` | All valid | Any invalid | Check geometry validity |
| **Coordinate bounds** | `geometry` | `2.2 < lon < 2.4`<br>`48.8 < lat < 48.9` | Outside Paris bounds | Paris bounds |

---

### **Census Data Checks (`census_data.parquet`)**

#### **1. Population & Demographics**

| Check | Field/Property | Hard Bounds | Warning Bounds | Notes |
|-------|---------------|-------------|----------------|-------|
| **Population density** | `population_density` (inhabitants/km²) | `1,000 < density < 60,000` | `< 5,000` or `> 50,000` | Paris avg: ~20,500; dense areas: 35-40k; low: 5-10k |
| **Total population** | Estimated total (from working-age) | `> 0` | `< 100` (very small) | Should be positive |
| **Working-age population** | `P17_POP1564` | `> 0` | `< 50` (very small) | Should be positive |
| **Children per capita** | `children_per_capita` (ratio) | `0 <= ratio <= 0.3` | `< 0.08` or `> 0.20` | Paris 2026: ~11.3% (0.113); typical: 0.10-0.15; warn if <8% or >20% |
| **Elderly ratio** | `elderly_ratio` (ratio) | `0 <= ratio <= 0.4` | `< 0.10` or `> 0.25` | Paris 2022: ~17.5% (0.175) aged 65+; typical: 0.15-0.20; warn if <10% or >25% |
| **Age ratio sum** | `children_per_capita + elderly_ratio + working_age_ratio` | `0.8 < sum < 1.2` | `< 0.7` or `> 1.3` | Should approximately sum to 1.0 |

#### **2. Socioeconomic Status**

| Check | Field/Property | Hard Bounds | Warning Bounds | Notes |
|-------|---------------|-------------|----------------|-------|
| **SES index** | `ses_index` (proportion high-status workers) | `0 <= index <= 1` | `< 0.1` or `> 0.8` | Typical: 0.2-0.5 in mixed neighborhoods |
| **Unemployment rate** | `unemployment_rate` (ratio) | `0 <= rate <= 0.5` | `< 0.01` or `> 0.30` | Paris avg: ~7-8%; disadvantaged areas: up to 30% |
| **Median income** | `median_income` (euros/year) | `10,000 < income < 200,000` | `< 15,000` or `> 70,000` | Paris 2018 median: ~€26,195 disposable; wealthiest arrondissements: ~€45,500; warn if >€70k (very high) |
| **Poverty rate** | `poverty_rate` (ratio) | `0 <= rate <= 0.5` | `< 0.06` or `> 0.40` | Paris avg: ~16%; IRIS range: 6-40% |

#### **3. Car Ownership & Commuting**

| Check | Field/Property | Hard Bounds | Warning Bounds | Notes |
|-------|---------------|-------------|----------------|-------|
| **Car ownership rate** | `car_ownership_rate` (households with car / total) | `0 <= rate <= 0.9` | `< 0.1` or `> 0.6` (central Paris) | Paris city: ~30-35%; region: ~60-70% |
| **Car commute ratio** | `car_commute_ratio` (car commuters / active workers) | `0 <= ratio <= 0.8` | `< 0.05` or `> 0.5` (central Paris) | Different from ownership; typically lower |
| **Walking ratio** | `walking_ratio` (walking commuters / active workers) | `0 <= ratio <= 1` | `< 0.1` or `> 0.6` | Mode share among commuters |
| **Cycling ratio** | `cycling_ratio` (cycling commuters / active workers) | `0 <= ratio <= 0.3` | `< 0.01` or `> 0.10` | Paris 2020: ~2.9% (0.029) for commuting; typical: 0.02-0.05; warn if <1% or >10% |
| **Public transport ratio** | `public_transport_ratio` (PT commuters / active workers) | `0 <= ratio <= 1` | `< 0.30` or `> 0.75` | Paris city center: ~62.9% (0.629); typical: 0.50-0.70; warn if <30% or >75% |
| **Two-wheelers ratio** | `two_wheelers_ratio` (motorcycle commuters / active workers) | `0 <= ratio <= 0.2` | `< 0.01` or `> 0.10` | Mode share; typically small |
| **Mode share sum** | `walking + cycling + public_transport + two_wheelers + car_commute` | `0.8 < sum < 1.2` | `< 0.7` or `> 1.3` | Should approximately sum to 1.0 |

#### **4. Employment & Education**

| Check | Field/Property | Hard Bounds | Warning Bounds | Notes |
|-------|---------------|-------------|----------------|-------|
| **Student ratio** | `student_ratio` (students / total population) | `0 <= ratio <= 0.4` | `< 0.05` or `> 0.25` | Paris 2024: ~10% (0.10) students; typical: 0.08-0.15; warn if <5% or >25% |
| **Retired ratio** | `retired_ratio` (retired / total population) | `0 <= ratio <= 0.4` | `< 0.05` or `> 0.35` | Typical: 0.15-0.25 (15-25% retired) |
| **Permanent employment ratio** | `permanent_employment_ratio` (permanent / salaried) | `0 <= ratio <= 1` | `< 0.70` or `> 0.95` | Paris region: ~87% (0.87) CDI; typical: 0.80-0.90; warn if <70% or >95% |
| **Temporary employment ratio** | `temporary_employment_ratio` (temporary / salaried) | `0 <= ratio <= 1` | `< 0.05` or `> 0.5` | Typical: 0.10-0.25 (10-25% temporary) |
| **Employment ratio sum** | `permanent + temporary` | `0.8 < sum < 1.2` | `< 0.7` or `> 1.3` | Should approximately sum to 1.0 |

#### **5. Data Quality & Completeness**

| Check | Field/Property | Hard Bounds | Warning Bounds | Notes |
|-------|---------------|-------------|----------------|-------|
| **Missing values** | Any feature column | `<= 20%` missing per column | `> 50%` missing | Some missing is OK, but not majority |
| **Negative values** | All ratio/rate columns | `>= 0` | Any negative | Ratios/rates should be non-negative |
| **NaN/Inf values** | All numeric columns | No NaN/Inf | Any NaN/Inf | Should be handled or imputed |
| **Required columns** | All expected feature columns | All present | Missing columns | Check against expected feature list |
| **IRIS coverage** | Number of IRIS units matched | `> 0` | `< 1` (no IRIS matched) | Should match at least one IRIS per neighborhood |
| **Neighborhood name** | `neighborhood_name` | Non-empty, matches GeoJSON | Mismatch or empty | Should match source neighborhood names |

---

### **Cross-Source Consistency Checks**

| Check | Description | Hard Bounds | Warning Bounds | Notes |
|-------|-------------|-------------|----------------|-------|
| **Area consistency** | Neighborhood area from OSM vs Census | Within 10% difference | > 20% difference | Areas should match approximately |
| **Population vs buildings** | High population density with low building density | Flag if inconsistent | N/A | Should correlate (high pop → high buildings) |
| **Service vs population** | Service density vs population density | Should correlate | Extreme mismatch | More people → more services typically |
| **Network vs area** | Network coverage vs neighborhood area | Network should cover area | Network much smaller than area | Network should reasonably cover neighborhood |

---

### **File Structure Checks**

| Check | Description | Hard Bounds | Warning Bounds | Notes |
|-------|-------------|-------------|----------------|-------|
| **Directory structure** | OSM: `data/raw/osm/compliant/{neighborhood}/`<br>Census: `data/raw/census/compliant/{neighborhood}/` | Matches expected structure | Missing directories | Should follow defined structure |
| **File existence** | Required files exist per neighborhood | All required files present | Missing files | OSM: services.geojson, buildings.geojson, network.graphml, etc. |
| **File format** | Files are valid GeoJSON/Parquet/GraphML | Valid format | Invalid format | Check file can be loaded |
| **File size** | Files are not empty | `> 0 bytes` | `== 0 bytes` | Empty files indicate extraction failure |
| **Neighborhood coverage** | All compliant neighborhoods have data | All neighborhoods present | Missing neighborhoods | Check against `paris_neighborhoods.geojson` |

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation & Structure

Set up the basic structure, configuration, and data loading utilities.

**Tasks:**
- Create validation module structure (or decide to keep in scripts/)
- Define sanity check configuration (hard bounds, warning bounds) as constants or config
- Create base checker class with initialization
- Set up logging and error handling patterns

### Phase 2: OSM Data Validation

Implement validation for all OSM data types (services, buildings, network, intersections).

**Tasks:**
- Implement services validation (count, density, categories, geometry)
- Implement buildings validation (count, density, levels, area, geometry)
- Implement network validation (nodes, edges, connectivity, block length)
- Implement intersections validation (count, density, degree)
- Implement pedestrian features validation (count, types, geometry)

### Phase 3: Census Data Validation

Implement validation for Census demographic and socioeconomic features.

**Tasks:**
- Implement population & demographics validation (density, age ratios)
- Implement socioeconomic validation (SES, unemployment, income, poverty)
- Implement car ownership & commuting validation (rates, mode shares)
- Implement employment & education validation (ratios, employment types)
- Implement data quality checks (missing values, negative values, NaN/Inf)

### Phase 4: Cross-Source & File Structure Validation

Implement consistency checks and file structure validation.

**Tasks:**
- Implement cross-source consistency checks (area matching, feature correlations)
- Implement file structure validation (directory structure, file existence, format)
- Implement file format validation (GeoJSON, Parquet, GraphML loading)

### Phase 5: Report Generation & Script

Create report generation and command-line script.

**Tasks:**
- Implement report generation (JSON + human-readable summary)
- Create command-line script with argument parsing
- Add summary statistics calculation
- Integrate all validation checks into main workflow

### Phase 6: Testing

Write comprehensive tests for validation logic.

**Tasks:**
- Write unit tests for individual validation functions
- Write integration tests with real data files
- Test edge cases (missing files, empty data, invalid geometries)

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### CREATE `src/data/validation/__init__.py`

- **IMPLEMENT**: Create package init file
- **PATTERN**: Follow existing `__init__.py` patterns (minimal, just package marker)
- **IMPORTS**: Export RawDataSanityChecker if creating module, otherwise skip this
- **VALIDATE**: `python -c "from src.data.validation import RawDataSanityChecker; print('Import OK')"`

### CREATE `src/data/validation/raw_data_sanity_checker.py`

- **IMPLEMENT**: Create RawDataSanityChecker class with initialization
- **PATTERN**: Follow class structure from `OSMExtractor` (lines 32-69) and `CensusLoader` (lines 33-68)
- **IMPORTS**: 
  ```python
  from pathlib import Path
  from typing import Any, Dict, List, Optional, Tuple
  import geopandas as gpd
  import pandas as pd
  import networkx as nx
  import osmnx as ox
  from shapely.geometry import Polygon
  from src.utils.config import get_config
  from src.utils.logging import get_logger
  from src.utils.helpers import load_neighborhoods, get_compliant_neighborhoods, get_service_category_names
  ```
- **STRUCTURE**: 
  - `__init__()`: Load config, set up paths, initialize logger
  - Constants for sanity check bounds (hard bounds, warning bounds)
  - Methods for each validation category
- **VALIDATE**: `python -c "from src.data.validation.raw_data_sanity_checker import RawDataSanityChecker; c = RawDataSanityChecker(); print('Init OK')"`

### ADD Sanity Check Bounds Constants

- **IMPLEMENT**: Define all sanity check bounds as class constants or module-level constants
- **PATTERN**: Use clear constant names: `SERVICES_MIN_COUNT`, `POPULATION_DENSITY_HARD_MIN`, etc.
- **BOUNDS**: Use the comprehensive list from planning discussion:
  - OSM bounds (services, buildings, network, intersections)
  - Census bounds (population, demographics, socioeconomic, commuting, employment)
  - Cross-source bounds (area matching tolerance, correlation thresholds)
- **VALIDATE**: `python -c "from src.data.validation.raw_data_sanity_checker import *; print('Bounds loaded')"`

### IMPLEMENT `_check_file_structure(neighborhood_name: str) -> Dict[str, Any]`

- **IMPLEMENT**: Check directory structure and file existence for a neighborhood
- **PATTERN**: Follow file path patterns from `osm_extractor.py` (lines 70-100) and `census_loader.py` (lines 70-100)
- **CHECKS**:
  - OSM directory exists: `data/raw/osm/compliant/{neighborhood}/`
  - Required OSM files exist: `services.geojson`, `buildings.geojson`, `network.graphml`, etc.
  - Census directory exists: `data/raw/census/compliant/{neighborhood}/`
  - Census file exists: `census_data.parquet`
  - File sizes > 0 bytes
- **RETURNS**: Dict with `{"status": "pass/warn/fail", "issues": [...], "files_found": [...]}`
- **VALIDATE**: `python -c "from src.data.validation.raw_data_sanity_checker import RawDataSanityChecker; c = RawDataSanityChecker(); result = c._check_file_structure('test_neighborhood'); print(result)"`

### IMPLEMENT `_validate_services(services_path: Path, neighborhood_name: str, area_km2: float) -> Dict[str, Any]`

- **IMPLEMENT**: Validate services GeoJSON file
- **PATTERN**: Follow data loading from `osm_extractor.py` (lines 198-320)
- **CHECKS**:
  - Load GeoJSON: `gpd.read_file(services_path)`
  - Geometry validity: `services_gdf.geometry.is_valid.all()`
  - Coordinate bounds: `2.2 < lon < 2.4`, `48.8 < lat < 48.9`
  - Service count: `> 0` (hard), `< 5` or `> 10000` (warn)
  - Service density: `0 < density < 500` (hard), `< 10` or `> 300` (warn)
  - Required columns: `name`, `amenity`, `category`, `geometry`, `neighborhood_name`
  - Category values: Must be in `get_service_category_names()`
  - All 8 categories present (warn if any missing)
- **RETURNS**: Dict with validation results, issues, statistics
- **VALIDATE**: Test with real services.geojson file from data directory

### IMPLEMENT `_validate_buildings(buildings_path: Path, neighborhood_name: str, area_km2: float) -> Dict[str, Any]`

- **IMPLEMENT**: Validate buildings GeoJSON file
- **PATTERN**: Follow data loading from `osm_extractor.py` (lines 376-476)
- **CHECKS**:
  - Load GeoJSON: `gpd.read_file(buildings_path)`
  - Geometry validity: `buildings_gdf.geometry.is_valid.all()`
  - Coordinate bounds: Paris bounds
  - Building count: `> 0` (hard), `< 10` or `> 50000` (warn)
  - Building density: `0 < density < 5000` (hard), `< 50` or `> 3000` (warn)
  - Building levels: `1 <= levels <= 20` (hard), `> 12` (warn for Paris)
  - Building area: `1 < area_m2 < 100000` (hard), `> 50000` (warn)
  - Total building area ratio: `0.1 < ratio < 0.8` (hard), `< 0.05` or `> 0.9` (warn)
  - Missing levels: `<= 50%` missing (warn if `> 50%`)
- **RETURNS**: Dict with validation results
- **VALIDATE**: Test with real buildings.geojson file

### IMPLEMENT `_validate_network(network_path: Path, neighborhood_name: str, area_km2: float) -> Dict[str, Any]`

- **IMPLEMENT**: Validate network GraphML file
- **PATTERN**: Follow network loading from `osm_extractor.py` (lines 478-555)
- **CHECKS**:
  - Load network: `ox.load_graphml(network_path)`
  - Node count: `> 0` (hard), `< 10` (warn)
  - Edge count: `> 0` (hard), `< 10` (warn)
  - Node-to-edge ratio: `0.5 < ratio < 3.0` (hard), `< 0.3` or `> 4.0` (warn)
  - Network connectivity: `>= 80%` nodes in largest component (hard), `< 50%` (warn)
  - Edge lengths: `1 < length < 5000` (hard), `< 0.1` or `> 2000` (warn)
  - Average block length: `50 < avg < 300` (hard), `< 30` or `> 400` (warn)
  - Node coordinate bounds: Paris bounds
- **RETURNS**: Dict with validation results
- **GOTCHA**: Block length calculation requires network analysis (sum edge lengths / number of blocks)
- **VALIDATE**: Test with real network.graphml file

### IMPLEMENT `_validate_intersections(intersections_path: Path, neighborhood_name: str, area_km2: float) -> Dict[str, Any]`

- **IMPLEMENT**: Validate intersections GeoJSON file
- **PATTERN**: Follow data loading from `osm_extractor.py` (lines 557-690)
- **CHECKS**:
  - Load GeoJSON: `gpd.read_file(intersections_path)`
  - Geometry validity: All valid points
  - Coordinate bounds: Paris bounds
  - Intersection count: `> 0` (hard), `< 5` (warn)
  - Intersection density: `20 < density < 500` (hard), `< 10` or `> 400` (warn)
  - Node degree: `3 <= degree <= 10` (hard), `> 12` (warn)
- **RETURNS**: Dict with validation results
- **VALIDATE**: Test with real intersections.geojson file

### IMPLEMENT `_validate_census_data(census_path: Path, neighborhood_name: str) -> Dict[str, Any]`

- **IMPLEMENT**: Validate Census Parquet file with all demographic features
- **PATTERN**: Follow data loading from `census_loader.py` (lines 711-1577)
- **CHECKS**:
  - Load Parquet: `pd.read_parquet(census_path)`
  - Required columns: All expected feature columns from `_extract_demographic_features()`
  - Population density: `1000 < density < 60000` (hard), `< 5000` or `> 50000` (warn)
  - Children per capita: `0 <= ratio <= 0.3` (hard), `< 0.05` or `> 0.25` (warn)
  - Elderly ratio: `0 <= ratio <= 0.4` (hard), `< 0.05` or `> 0.30` (warn)
  - SES index: `0 <= index <= 1` (hard), `< 0.1` or `> 0.8` (warn)
  - Unemployment rate: `0 <= rate <= 0.5` (hard), `< 0.01` or `> 0.30` (warn)
  - Median income: `10000 < income < 200000` (hard), `< 15000` or `> 100000` (warn)
  - Poverty rate: `0 <= rate <= 0.5` (hard), `< 0.06` or `> 0.40` (warn)
  - Car ownership rate: `0 <= rate <= 0.9` (hard), `< 0.1` or `> 0.6` (warn)
  - All ratio features: `0 <= ratio <= 1` (hard), check for negative values
  - Mode share sum: `0.8 < sum < 1.2` (warn if outside)
  - Missing values: `<= 20%` per column (hard), `> 50%` (warn)
  - NaN/Inf values: None allowed (hard)
- **RETURNS**: Dict with validation results per feature
- **VALIDATE**: Test with real census_data.parquet file

### IMPLEMENT `_check_cross_source_consistency(neighborhood_name: str, osm_area: float, census_area: float, ...) -> Dict[str, Any]`

- **IMPLEMENT**: Check consistency between OSM and Census data
- **PATTERN**: Compare areas, check feature correlations
- **CHECKS**:
  - Area consistency: OSM area vs Census area within 10% (hard), > 20% difference (warn)
  - Population vs buildings: High pop density with low building density (flag)
  - Service vs population: Service density vs population density correlation (flag extreme mismatch)
  - Network vs area: Network coverage vs neighborhood area (flag if network much smaller)
- **RETURNS**: Dict with consistency check results
- **VALIDATE**: Test with real data from multiple neighborhoods

### IMPLEMENT `validate_neighborhood(neighborhood_name: str) -> Dict[str, Any]`

- **IMPLEMENT**: Main validation method for a single neighborhood
- **PATTERN**: Orchestrate all validation checks, aggregate results
- **WORKFLOW**:
  1. Check file structure
  2. Load neighborhood geometry (for area calculation)
  3. Validate OSM data (services, buildings, network, intersections)
  4. Validate Census data
  5. Check cross-source consistency
  6. Aggregate results (count failures, warnings, passes)
- **RETURNS**: Comprehensive validation report dict for the neighborhood
- **VALIDATE**: `python -c "from src.data.validation.raw_data_sanity_checker import RawDataSanityChecker; c = RawDataSanityChecker(); result = c.validate_neighborhood('paris_rive_gauche'); print(result['status'])"`

### IMPLEMENT `validate_all_neighborhoods(neighborhood_names: Optional[List[str]] = None) -> Dict[str, Any]`

- **IMPLEMENT**: Validate all neighborhoods or specified list
- **PATTERN**: Follow pattern from `osm_extractor.py` (lines 837-925) - iterate over neighborhoods, collect results
- **WORKFLOW**:
  1. Load neighborhoods from GeoJSON (if names not provided)
  2. Filter to compliant neighborhoods (or use provided names)
  3. For each neighborhood: call `validate_neighborhood()`
  4. Aggregate results across all neighborhoods
  5. Generate summary statistics
- **RETURNS**: Overall validation report with per-neighborhood results and summary
- **VALIDATE**: Test with all compliant neighborhoods

### IMPLEMENT `_generate_report(validation_results: Dict[str, Any], output_dir: Path) -> None`

- **IMPLEMENT**: Generate and save validation report
- **PATTERN**: Follow pattern from `osm_extractor.py` (lines 927-976) - save JSON + summary
- **OUTPUTS**:
  - JSON report: `data/raw/sanity_check_report.json` (detailed results)
  - Summary CSV: `data/raw/sanity_check_summary.csv` (per-neighborhood summary)
  - Human-readable summary: Print to console
- **STRUCTURE**: 
  ```json
  {
    "summary": {
      "total_neighborhoods": 3,
      "passed": 2,
      "warnings": 1,
      "failed": 0,
      "total_checks": 150,
      "passed_checks": 145,
      "warning_checks": 5,
      "failed_checks": 0
    },
    "neighborhoods": [
      {
        "neighborhood_name": "...",
        "status": "pass/warn/fail",
        "checks": {...},
        "issues": [...],
        "statistics": {...}
      }
    ]
  }
  ```
- **VALIDATE**: Run validation and check report files are created correctly

### CREATE `scripts/sanity_check_raw_data.py`

- **IMPLEMENT**: Command-line script to run sanity checks
- **PATTERN**: Follow script structure from `scripts/run_census_loader.py` (lines 1-27)
- **FEATURES**:
  - Argument parsing: `--neighborhood` (optional, specific neighborhood), `--output-dir` (optional)
  - Initialize checker
  - Run validation (single or all neighborhoods)
  - Print summary to console
  - Save report files
- **USAGE**: 
  ```bash
  python scripts/sanity_check_raw_data.py  # Check all neighborhoods
  python scripts/sanity_check_raw_data.py --neighborhood "paris_rive_gauche"  # Check one
  ```
- **VALIDATE**: `python scripts/sanity_check_raw_data.py --help`

### UPDATE `src/data/validation/__init__.py` (if module created)

- **IMPLEMENT**: Export RawDataSanityChecker
- **PATTERN**: Follow existing `__init__.py` patterns
- **IMPORTS**: `from src.data.validation.raw_data_sanity_checker import RawDataSanityChecker`
- **EXPORTS**: `__all__ = ["RawDataSanityChecker"]`
- **VALIDATE**: `python -c "from src.data.validation import RawDataSanityChecker; print('Export OK')"`

### CREATE `tests/unit/test_raw_data_sanity_checker.py`

- **IMPLEMENT**: Unit tests for validation logic
- **PATTERN**: Follow test patterns from `tests/unit/test_census_loader.py` (lines 1-100)
- **TESTS**:
  - Test initialization
  - Test file structure checking (with mocked paths)
  - Test individual validation methods (with mock data)
  - Test bounds checking logic
  - Test report generation
- **FIXTURES**: Create fixtures for mock OSM data, mock Census data, mock file paths
- **VALIDATE**: `pytest tests/unit/test_raw_data_sanity_checker.py -v`

### CREATE Integration Test (Optional)

- **IMPLEMENT**: Integration test with real data files
- **PATTERN**: Follow pattern from `tests/integration/test_osm_extraction_pipeline.py`
- **TEST**: Run validation on actual extracted data (if available)
- **VALIDATE**: `pytest tests/integration/test_raw_data_sanity_check.py -v` (if created)

---

## TESTING STRATEGY

### Unit Tests

**Scope**: Test individual validation functions with mock data

**Test Cases**:
1. **File Structure Validation**:
   - Test with existing files (pass)
   - Test with missing files (fail)
   - Test with empty files (warn)
   - Test with invalid file formats (fail)

2. **OSM Data Validation**:
   - Test services validation with valid data (pass)
   - Test services validation with out-of-bounds values (fail/warn)
   - Test buildings validation with valid data
   - Test network validation with connected/disconnected graphs
   - Test intersections validation

3. **Census Data Validation**:
   - Test population density bounds
   - Test ratio features (0-1 range)
   - Test missing value detection
   - Test NaN/Inf detection
   - Test mode share sum validation

4. **Cross-Source Consistency**:
   - Test area matching (within tolerance)
   - Test area mismatch (outside tolerance)

5. **Report Generation**:
   - Test JSON report structure
   - Test summary statistics calculation

**Fixtures Needed**:
- Mock OSM GeoDataFrames (services, buildings, intersections)
- Mock NetworkX graph
- Mock Census DataFrame
- Mock file paths
- Mock neighborhood geometries

### Integration Tests

**Scope**: Test end-to-end validation with real data files

**Test Cases**:
1. Validate single neighborhood with real extracted data
2. Validate all neighborhoods
3. Check report files are generated correctly
4. Verify report contains expected structure

**Requirements**:
- Real data files must exist in `data/raw/` directories
- May skip if data not available (mark as `@pytest.mark.skipif`)

### Edge Cases

- Missing data files (should fail gracefully with clear error)
- Empty data files (should warn)
- Invalid geometries (should fail)
- Corrupted file formats (should fail with clear error)
- Neighborhood with no data (should fail)
- Extreme outlier values (should warn or fail based on bounds)
- Missing required columns (should fail)
- All data missing (should fail)

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Check Python syntax
python -m py_compile src/data/validation/raw_data_sanity_checker.py
python -m py_compile scripts/sanity_check_raw_data.py

# Run ruff linter
ruff check src/data/validation/raw_data_sanity_checker.py scripts/sanity_check_raw_data.py

# Run black formatter check
black --check src/data/validation/raw_data_sanity_checker.py scripts/sanity_check_raw_data.py
```

### Level 2: Import Validation

```bash
# Test imports work correctly
python -c "from src.data.validation.raw_data_sanity_checker import RawDataSanityChecker; print('Import OK')"

# Test script imports
python -c "import scripts.sanity_check_raw_data; print('Script import OK')"
```

### Level 3: Unit Tests

```bash
# Run unit tests
pytest tests/unit/test_raw_data_sanity_checker.py -v

# Run with coverage
pytest tests/unit/test_raw_data_sanity_checker.py --cov=src/data/validation/raw_data_sanity_checker --cov-report=term-missing
```

### Level 4: Manual Validation

```bash
# Test checker initialization
python -c "from src.data.validation.raw_data_sanity_checker import RawDataSanityChecker; c = RawDataSanityChecker(); print('Init OK')"

# Test validation on one neighborhood (if data exists)
python scripts/sanity_check_raw_data.py --neighborhood "paris_rive_gauche"

# Test validation on all neighborhoods (if data exists)
python scripts/sanity_check_raw_data.py

# Check report files are created
ls -la data/raw/sanity_check_report.json
ls -la data/raw/sanity_check_summary.csv
```

### Level 5: Integration Test (Optional)

```bash
# Run integration test if created
pytest tests/integration/test_raw_data_sanity_check.py -v
```

---

## ACCEPTANCE CRITERIA

- [ ] `RawDataSanityChecker` class implements all validation methods
- [ ] All OSM data types validated (services, buildings, network, intersections, pedestrian features)
- [ ] All Census features validated (population, demographics, socioeconomic, commuting, employment)
- [ ] File structure validation works correctly
- [ ] Cross-source consistency checks implemented
- [ ] Validation report generated (JSON + CSV + console summary)
- [ ] Command-line script works for single and all neighborhoods
- [ ] All sanity check bounds from planning discussion implemented
- [ ] Hard bounds (failures) and warning bounds properly distinguished
- [ ] Unit tests pass (80%+ coverage)
- [ ] Integration tests pass (if data available)
- [ ] Code follows project conventions (PEP 8, type hints, docstrings)
- [ ] No linting or type checking errors
- [ ] Report files saved to correct location
- [ ] Human-readable summary printed to console

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes (unit + integration if applicable)
- [ ] No linting or type checking errors
- [ ] Manual testing confirms validation works on real data
- [ ] Report files generated correctly
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability

---

## NOTES

**Design Decisions:**

1. **Module Location**: Decided to create `src/data/validation/` module for better organization, but could also put everything in `scripts/` if preferred. The plan assumes module approach but can be adapted.

2. **Bounds Configuration**: Sanity check bounds are defined as constants in the code. Could be moved to `config.yaml` later if needed for flexibility, but constants are simpler for initial implementation.

3. **Report Format**: Using JSON for machine-readable detailed report and CSV for summary, following existing patterns. Human-readable console output for quick inspection.

4. **Error Handling**: Validation failures don't stop the entire process - each neighborhood is validated independently, and results are aggregated. This allows identifying all issues at once.

5. **Missing Data Handling**: Some missing data is expected (OSM is incomplete), so we use warning thresholds (e.g., >50% missing) rather than hard failures for missing values.

6. **Area Calculation**: Area is calculated from neighborhood geometry for consistency checks. Uses geopandas area calculation with proper CRS conversion (EPSG:4326 → EPSG:3857 for meters).

**Future Enhancements (Out of Scope for MVP):**

- Move bounds to config.yaml for easy adjustment
- Add visualization of validation results (plots, maps)
- Add automatic data repair suggestions
- Add historical trend tracking (compare validation results over time)
- Add validation for processed/feature-engineered data (separate script)

**Key Risks:**

1. **Data Availability**: Validation requires extracted data files. If data doesn't exist, tests should handle gracefully.
2. **Performance**: Validating all neighborhoods may be slow if data files are large. Consider adding progress bars or parallelization later.
3. **Bounds Accuracy**: Sanity check bounds are based on research but may need adjustment based on actual data. Start conservative (wider bounds) and tighten based on results.
