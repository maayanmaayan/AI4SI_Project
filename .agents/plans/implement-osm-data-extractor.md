# Feature: OSM Data Extractor

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Implement a comprehensive OSM (OpenStreetMap) data extraction system that extracts services, buildings, street networks, and walkability features for all Paris neighborhoods. The extractor will organize data by compliance status (compliant vs non-compliant neighborhoods) and provide optimized service location files for distance-based loss calculations during model training.

**Core Functionality:**
- Extract OSM services by 8 NEXI categories (Education, Entertainment, Grocery, Health, Posts and banks, Parks, Sustenance, Shops)
- Extract building data (geometry, levels, type)
- Extract walkable street network with 100m buffer
- Extract walkability features (intersections, sidewalks, pedestrian infrastructure)
- Organize output by compliance status (compliant/ vs non_compliant/ directories)
- Create optimized service location files per category for fast distance calculations
- Support configurable caching (default: skip if exists, with force flag)

## User Story

As a **data engineer/researcher**
I want to **extract comprehensive OSM data for all Paris neighborhoods**
So that **I can compute 30+ urban features and service locations needed for training the 15-minute city service gap prediction model**

## Problem Statement

The project requires OSM data extraction for:
1. **Service locations** - Needed for distance-based loss calculations (network-based walking distance from predicted service category to nearest actual service)
2. **Building data** - Required for building density, count, and floor area calculations
3. **Street network** - Essential for network-based distance calculations (not Euclidean)
4. **Walkability features** - Needed for intersection density, pedestrian infrastructure metrics

Currently, no data collection pipeline exists. The feature engineering and model training phases depend on this extracted data.

## Solution Statement

Implement a modular `OSMExtractor` class that:
- Uses OSMnx library to extract OSM data within neighborhood boundaries (with 100m buffer)
- Organizes extracted data by compliance status for clear separation
- Creates optimized service location files per category for efficient distance lookups
- Handles errors gracefully with retries and informative logging
- Supports caching to avoid re-extraction of existing data
- Saves all data in GeoJSON/GraphML formats for downstream processing

---

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: High
**Primary Systems Affected**: 
- `src/data/collection/` (new module)
- `models/config.yaml` (add OSM extraction config)
- `data/raw/osm/` (new directory structure)
**Dependencies**: 
- osmnx 1.6+ (OSM data extraction)
- geopandas 0.14+ (geospatial data handling)
- shapely 2.0+ (geometry operations)
- networkx (for graph operations, comes with osmnx)

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `src/utils/helpers.py` (lines 24-57) - Why: `load_neighborhoods()` function to load GeoJSON
- `src/utils/helpers.py` (lines 60-89) - Why: `get_compliant_neighborhoods()` filter function
- `src/utils/helpers.py` (lines 92-117) - Why: `get_non_compliant_neighborhoods()` filter function
- `src/utils/helpers.py` (lines 120-243) - Why: `get_service_category_mapping()` returns 8 categories with OSM tags
- `src/utils/helpers.py` (lines 246-260) - Why: `get_service_category_names()` returns ordered list
- `src/utils/helpers.py` (lines 406-418) - Why: `ensure_dir_exists()` for directory creation
- `src/utils/helpers.py` (lines 421-447) - Why: `save_dataframe()` for saving GeoDataFrames
- `src/utils/config.py` (lines 21-69) - Why: `load_config()` pattern for configuration loading
- `src/utils/logging.py` (lines 74-92) - Why: `get_logger()` pattern for module logging
- `models/config.yaml` (all lines) - Why: Configuration structure to extend
- `tests/unit/test_helpers.py` (lines 29-95) - Why: Test fixture pattern for GeoJSON files
- `CURSOR.md` (lines 200-232) - Why: Code conventions and patterns to follow
- `PRD.md` (lines 250-264) - Why: Feature extraction requirements and specifications

### New Files to Create

- `src/data/collection/osm_extractor.py` - Main OSM extraction module with `OSMExtractor` class
- `tests/unit/test_osm_extractor.py` - Unit tests for OSM extraction functionality
- `tests/integration/test_osm_extraction_pipeline.py` - Integration tests for full pipeline

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [OSMnx Documentation - features_from_polygon](https://osmnx.readthedocs.io/)
  - Specific section: Feature extraction from polygons
  - Why: Core function for extracting services, buildings, and other features
- [OSMnx Documentation - graph_from_polygon](https://osmnx.readthedocs.io/)
  - Specific section: Network graph extraction
  - Why: Required for extracting walkable street networks
- [OSMnx Examples Gallery](https://github.com/gboeing/osmnx-examples)
  - Why: Real-world usage examples and patterns
- [GeoPandas Documentation - Working with Geometries](https://geopandas.org/en/stable/docs/user_guide/geometric_manipulations.html)
  - Specific section: Buffer operations
  - Why: Need to create approximate 100m buffer around neighborhood boundaries
- [Shapely Documentation - Buffer](https://shapely.readthedocs.io/en/stable/manual.html#object.buffer)
  - Why: Creating approximate buffer in WGS84 (using degree-to-meter conversion at Paris latitude)

### Patterns to Follow

**Naming Conventions:**
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Use descriptive names: `extract_services()`, `extract_buildings()`, not `get_data()`
- Follow existing pattern: `*_extractor.py` for data collection modules

**Error Handling:**
- Use try-except blocks with specific exception types
- Log errors with context (neighborhood name, operation type)
- Continue processing other neighborhoods if one fails
- Pattern from `helpers.py` (lines 47-55): Convert exceptions to domain-specific errors

**Logging Pattern:**
```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.info(f"Extracting services for {neighborhood_name}")
logger.warning(f"Low service count ({count}) for {neighborhood_name} - may indicate incomplete OSM data")
logger.error(f"Failed to extract network for {neighborhood_name}: {error}")
```

**Configuration Pattern:**
```python
from src.utils.config import get_config

config = get_config()
buffer_meters = config.get("osm_extraction", {}).get("buffer_meters", 100)
```

**Directory Creation Pattern:**
```python
from src.utils.helpers import ensure_dir_exists

ensure_dir_exists(output_dir)
```

**GeoDataFrame Saving Pattern:**
```python
from src.utils.helpers import save_dataframe

save_dataframe(gdf, output_path, format="csv")  # For CSV
# For GeoJSON, use gdf.to_file() directly
gdf.to_file(output_path, driver="GeoJSON")
```

**Type Hints Pattern:**
- Always use type hints for function parameters and returns
- Use `Optional[Type]` for nullable values
- Use `List[Type]`, `Dict[str, Type]` for collections
- Pattern from `helpers.py`: `def function_name(param: Type) -> ReturnType:`

**Docstring Pattern (Google Style):**
```python
def extract_services(polygon: Polygon, category_mapping: Dict[str, List[str]]) -> gpd.GeoDataFrame:
    """Extract OSM services by category from polygon.

    Args:
        polygon: Shapely Polygon defining extraction boundary.
        category_mapping: Dictionary mapping category names to OSM tag lists.

    Returns:
        GeoDataFrame with columns: name, amenity, category, geometry, ...

    Raises:
        ValueError: If polygon is invalid or extraction fails.

    Example:
        >>> services = extract_services(polygon, mapping)
        >>> print(f"Found {len(services)} services")
    """
```

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation & Configuration

Set up base structure, configuration, and utility methods.

**Tasks:**
- Create `OSMExtractor` class skeleton
- Add OSM extraction configuration to `config.yaml`
- Implement buffer creation utility (approximate 100m buffer in WGS84)
- Implement directory structure creation
- Set up logging and error handling infrastructure

### Phase 2: Services Extraction

Implement service extraction with category mapping and optimized file output.

**Tasks:**
- Implement `extract_services()` method
- Map OSM amenities to NEXI categories (handle duplicates)
- Create optimized service location files per category
- Save main services GeoJSON and category-specific files
- Add data validation and informative warnings

### Phase 3: Buildings & Network Extraction

Implement building and network extraction with buffer support.

**Tasks:**
- Implement `extract_buildings()` method
- Calculate building areas and handle missing levels
- Implement `extract_network()` method with 100m buffer
- Save network as GraphML and edge/node GeoDataFrames
- Validate network connectivity

### Phase 4: Walkability Features

Extract intersections and pedestrian infrastructure.

**Tasks:**
- Implement `extract_walkability_features()` method
- Extract intersections from network (degree >= 3)
- Extract pedestrian infrastructure (sidewalks, pedestrian streets, crosswalks)
- Save walkability GeoDataFrames

### Phase 5: Integration & Orchestration

Combine all extraction methods with error handling and caching.

**Tasks:**
- Implement `extract_neighborhood()` wrapper method
- Implement caching logic (check if data exists, skip if present)
- Add force flag to override caching
- Implement `extract_all_neighborhoods()` orchestration
- Add progress tracking and summary statistics
- Create extraction metadata JSON

### Phase 6: Testing & Validation

Create comprehensive test suite.

**Tasks:**
- Write unit tests for each extraction method
- Create test fixtures with mock OSM data
- Write integration tests for full pipeline
- Test error handling and retry logic
- Test caching behavior

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### CREATE src/data/collection/osm_extractor.py

- **IMPLEMENT**: Create `OSMExtractor` class with initialization
- **PATTERN**: Follow class structure from project (see `helpers.py` for function patterns)
- **IMPORTS**: 
  ```python
  import os
  import json
  import time
  from pathlib import Path
  from typing import Dict, List, Optional, Tuple
  import geopandas as gpd
  import osmnx as ox
  import networkx as nx
  from shapely.geometry import Polygon
  from src.utils.config import get_config
  from src.utils.logging import get_logger
  from src.utils.helpers import (
      load_neighborhoods,
      get_compliant_neighborhoods,
      get_non_compliant_neighborhoods,
      get_service_category_mapping,
      get_service_category_names,
      ensure_dir_exists,
  )
  ```
- **GOTCHA**: OSMnx requires internet connection - handle network errors gracefully. No CRS transformation libraries needed (keeping it simple in WGS84).
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; print('Import successful')"`

### UPDATE models/config.yaml

- **IMPLEMENT**: Add `osm_extraction` section with configuration
- **PATTERN**: Follow existing config structure (see `models/config.yaml` lines 1-43)
- **ADD**:
  ```yaml
  osm_extraction:
    buffer_meters: 100  # Buffer around neighborhood boundaries
    network_type: "walk"  # 'walk', 'drive', 'bike', 'all'
    simplify: true  # Simplify network geometry
    cache_results: true  # Skip if data already exists
    retry_attempts: 3  # Number of retries on failure
    retry_delay: 5  # Seconds between retries
    min_services_warning: 5  # Warn if fewer services found
  ```
- **GOTCHA**: Ensure YAML indentation is correct (2 spaces)
- **VALIDATE**: `python -c "from src.utils.config import get_config; c = get_config(); assert 'osm_extraction' in c; print('Config valid')"`

### IMPLEMENT _create_buffer_polygon method in OSMExtractor

- **IMPLEMENT**: Method to create approximate 100m buffer around polygon in WGS84
- **PATTERN**: Use Shapely buffer with degree approximation (keep it simple, no CRS transformation)
- **DETAILS**:
  - Convert buffer distance from meters to degrees using approximation
  - At Paris latitude (~48.85°), 1 degree ≈ 111,320 meters
  - For 100m buffer: `buffer_degrees = buffer_meters / 111320.0`
  - Apply buffer using `polygon.buffer(buffer_degrees)`
  - Note: This is approximate - buffer size varies slightly by latitude, but acceptable for our use case
- **SIGNATURE**: `def _create_buffer_polygon(self, polygon: Polygon, buffer_meters: float) -> Polygon:`
- **GOTCHA**: Buffer is approximate (not exact 100m everywhere), but simple and sufficient for service extraction near boundaries
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; e = OSMExtractor(); from shapely.geometry import box; p = box(2.3, 48.8, 2.4, 48.9); b = e._create_buffer_polygon(p, 100); print(f'Buffer created: {b.area > p.area}')"`

### IMPLEMENT _get_output_directory method in OSMExtractor

- **IMPLEMENT**: Method to get output directory based on compliance status
- **PATTERN**: Use `ensure_dir_exists()` from helpers
- **DETAILS**:
  - Return `data/raw/osm/compliant/{neighborhood_name}/` for compliant
  - Return `data/raw/osm/non_compliant/{neighborhood_name}/` for non-compliant
  - Create directory if it doesn't exist
- **SIGNATURE**: `def _get_output_directory(self, neighborhood_name: str, is_compliant: bool) -> Path:`
- **GOTCHA**: Normalize neighborhood names (replace spaces with underscores, lowercase)
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; e = OSMExtractor(); d = e._get_output_directory('Test Neighborhood', True); print(f'Directory: {d}'); assert 'compliant' in str(d)"`

### IMPLEMENT _check_cache method in OSMExtractor

- **IMPLEMENT**: Method to check if extraction data already exists
- **PATTERN**: Check for existence of key files (services.geojson, network.graphml, etc.)
- **DETAILS**:
  - Check if `services.geojson` exists
  - Check if `network.graphml` exists
  - Check if `buildings.geojson` exists
  - Return True if all exist, False otherwise
- **SIGNATURE**: `def _check_cache(self, output_dir: Path) -> bool:`
- **GOTCHA**: Only check cache if `cache_results` config is True
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; from pathlib import Path; e = OSMExtractor(); print('Cache check method exists')"`

### IMPLEMENT extract_services method in OSMExtractor

- **IMPLEMENT**: Extract all OSM services and map to NEXI categories
- **PATTERN**: Use `ox.features_from_polygon()` with amenity tags
- **DETAILS**:
  - Extract all amenities: `tags = {'amenity': True}`
  - Also extract shops: `tags = {'shop': True}`
  - Also extract leisure (parks): `tags = {'leisure': True}`
  - Map each service to NEXI categories using `get_service_category_mapping()`
  - **Duplicate services** that belong to multiple categories (e.g., ice_cream in Grocery and Sustenance)
  - Add columns: `name`, `amenity`, `category`, `geometry`, `neighborhood_name`
  - Filter services within buffered polygon
- **SIGNATURE**: `def extract_services(self, polygon: Polygon, buffered_polygon: Polygon, neighborhood_name: str) -> gpd.GeoDataFrame:`
- **GOTCHA**: Some services may not have `amenity` tag - check `shop`, `leisure`, `tourism` tags too
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; from shapely.geometry import box; e = OSMExtractor(); p = box(2.35, 48.85, 2.36, 48.86); services = e.extract_services(p, p, 'test'); print(f'Services found: {len(services)}')"`

### IMPLEMENT _save_services_by_category method in OSMExtractor

- **IMPLEMENT**: Save optimized service location files per category
- **PATTERN**: Use `gdf.to_file()` for GeoJSON
- **DETAILS**:
  - Create `services_by_category/` subdirectory
  - For each of 8 categories, filter services and save as `{category_lowercase}.geojson`
  - Use `get_service_category_names()` for consistent ordering
  - Save only essential columns: `name`, `amenity`, `geometry` (for fast loading)
- **SIGNATURE**: `def _save_services_by_category(self, services_gdf: gpd.GeoDataFrame, output_dir: Path) -> None:`
- **GOTCHA**: Handle category names with spaces (e.g., "Posts and banks" -> "posts_and_banks.geojson")
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; from pathlib import Path; e = OSMExtractor(); print('Save by category method exists')"`

### IMPLEMENT extract_buildings method in OSMExtractor

- **IMPLEMENT**: Extract building data with geometry and levels
- **PATTERN**: Use `ox.features_from_polygon()` with building tags
- **DETAILS**:
  - Extract buildings: `tags = {'building': True}`
  - Calculate building area using GeoDataFrame area (approximate in WGS84 degrees²)
  - Convert to approximate square meters: `area_m2 = gdf.geometry.area * (111320.0 ** 2)` (approximate conversion)
  - Extract `building:levels` tag (handle missing values - set to None)
  - Extract `building` type tag
  - Add columns: `name`, `building`, `building:levels`, `area_m2`, `geometry`, `neighborhood_name`
  - Filter buildings within buffered polygon
- **SIGNATURE**: `def extract_buildings(self, polygon: Polygon, buffered_polygon: Polygon, neighborhood_name: str) -> gpd.GeoDataFrame:`
- **GOTCHA**: Building areas are approximate (calculated in WGS84, converted to m²). For exact areas, feature engineering can recalculate in projected CRS later if needed
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; from shapely.geometry import box; e = OSMExtractor(); p = box(2.35, 48.85, 2.36, 48.86); buildings = e.extract_buildings(p, p, 'test'); print(f'Buildings found: {len(buildings)}')"`

### IMPLEMENT extract_network method in OSMExtractor

- **IMPLEMENT**: Extract walkable street network with buffer
- **PATTERN**: Use `ox.graph_from_polygon()` with network_type='walk'
- **DETAILS**:
  - Extract network from **buffered polygon** (includes 100m buffer)
  - Use `network_type='walk'` from config
  - Simplify network if config says so
  - Convert graph to edge and node GeoDataFrames using `ox.graph_to_gdfs()`
  - Save graph as GraphML: `ox.save_graphml(G, filepath)`
  - Save edges GeoDataFrame as GeoJSON
  - Save nodes GeoDataFrame as GeoJSON
  - Add `neighborhood_name` column to edges and nodes
- **SIGNATURE**: `def extract_network(self, buffered_polygon: Polygon, neighborhood_name: str) -> Tuple[nx.Graph, gpd.GeoDataFrame, gpd.GeoDataFrame]:`
- **GOTCHA**: Network extraction can be slow for large areas - add progress logging
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; from shapely.geometry import box; e = OSMExtractor(); p = box(2.35, 48.85, 2.36, 48.86); G, edges, nodes = e.extract_network(p, 'test'); print(f'Network nodes: {len(nodes)}, edges: {len(edges)}')"`

### IMPLEMENT extract_walkability_features method in OSMExtractor

- **IMPLEMENT**: Extract intersections and pedestrian infrastructure
- **PATTERN**: Analyze network graph for intersections, query OSM for pedestrian features
- **DETAILS**:
  - **Intersections**: Find nodes in network with degree >= 3
  - Extract pedestrian ways: `tags = {'highway': 'pedestrian'}`
  - Extract sidewalks: Query ways with `sidewalk` tag
  - Extract crosswalks: Query nodes/ways with `crossing` tag
  - Create GeoDataFrames for intersections and pedestrian infrastructure
  - Add `neighborhood_name` column
- **SIGNATURE**: `def extract_walkability_features(self, network_graph: nx.Graph, buffered_polygon: Polygon, neighborhood_name: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:`
- **GOTCHA**: Intersections should use network nodes, not OSM nodes (network is simplified)
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; from shapely.geometry import box; import networkx as nx; e = OSMExtractor(); p = box(2.35, 48.85, 2.36, 48.86); G = nx.Graph(); intersections, ped = e.extract_walkability_features(G, p, 'test'); print('Walkability extraction method exists')"`

### IMPLEMENT extract_neighborhood method in OSMExtractor

- **IMPLEMENT**: Main wrapper method to extract all data for one neighborhood
- **PATTERN**: Orchestrate all extraction methods with error handling
- **DETAILS**:
  - Check cache first (if enabled and force=False, skip if exists)
  - Create buffered polygon
  - Call `extract_services()`, `extract_buildings()`, `extract_network()`, `extract_walkability_features()`
  - Save all GeoDataFrames to output directory
  - Save services by category
  - Handle errors with retries (use config retry_attempts and retry_delay)
  - Log informative warnings (e.g., low service count)
  - Return extraction summary dictionary
- **SIGNATURE**: `def extract_neighborhood(self, neighborhood_row: gpd.GeoSeries, force: bool = False) -> Dict[str, Any]:`
- **GOTCHA**: Handle network extraction failures gracefully - may need to fallback to simpler network type
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; from src.utils.helpers import load_neighborhoods; e = OSMExtractor(); n = load_neighborhoods('paris_neighborhoods.geojson'); result = e.extract_neighborhood(n.iloc[0]); print(f'Extraction result: {result}')"`

### IMPLEMENT extract_all_neighborhoods method in OSMExtractor

- **IMPLEMENT**: Extract data for all neighborhoods, organized by compliance status
- **PATTERN**: Use `get_compliant_neighborhoods()` and `get_non_compliant_neighborhoods()` from helpers
- **DETAILS**:
  - Load neighborhoods from GeoJSON
  - Split into compliant and non-compliant
  - Process each neighborhood (with progress logging)
  - Collect summary statistics
  - Save extraction metadata JSON to `data/raw/osm/extraction_log.json`
  - Save summary CSV to `data/raw/osm/extraction_summary.csv`
  - Return overall summary
- **SIGNATURE**: `def extract_all_neighborhoods(self, geojson_path: Optional[str] = None, force: bool = False) -> Dict[str, Any]:`
- **GOTCHA**: Process neighborhoods sequentially to avoid API rate limiting
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; e = OSMExtractor(); summary = e.extract_all_neighborhoods(); print(f'Extracted {summary[\"total_neighborhoods\"]} neighborhoods')"`

### IMPLEMENT _save_extraction_metadata method in OSMExtractor

- **IMPLEMENT**: Save extraction metadata and summary statistics
- **PATTERN**: Use JSON for metadata, CSV for summary
- **DETAILS**:
  - Save timestamp, neighborhood name, data types extracted, record counts, status, errors
  - Create summary CSV with columns: neighborhood_name, label, services_count, buildings_count, network_nodes, network_edges, intersections_count, status
- **SIGNATURE**: `def _save_extraction_metadata(self, metadata: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:`
- **GOTCHA**: Ensure metadata directory exists before saving
- **VALIDATE**: `python -c "from src.data.collection.osm_extractor import OSMExtractor; e = OSMExtractor(); print('Metadata saving method exists')"`

### CREATE tests/unit/test_osm_extractor.py

- **IMPLEMENT**: Unit tests for OSM extraction methods
- **PATTERN**: Follow test structure from `tests/unit/test_helpers.py`
- **DETAILS**:
  - Test `_create_buffer_polygon()` with known polygon
  - Test `_get_output_directory()` for compliant/non-compliant
  - Test `_check_cache()` with existing/missing files
  - Test `extract_services()` with mock polygon (use small test area)
  - Test `extract_buildings()` with mock polygon
  - Test `extract_network()` with mock polygon
  - Test service category mapping and duplication
  - Test error handling and retries
  - Use pytest fixtures for test data
- **IMPORTS**: `import pytest`, `from unittest.mock import patch, MagicMock`
- **GOTCHA**: Mock OSMnx calls for unit tests to avoid internet dependency
- **VALIDATE**: `pytest tests/unit/test_osm_extractor.py -v`

### CREATE tests/integration/test_osm_extraction_pipeline.py

- **IMPLEMENT**: Integration tests for full extraction pipeline
- **PATTERN**: Test with real (small) OSM data extraction
- **DETAILS**:
  - Test `extract_neighborhood()` with real neighborhood (use smallest one)
  - Test `extract_all_neighborhoods()` with subset of neighborhoods
  - Verify output file structure exists
  - Verify data quality (no null geometries, valid CRS, etc.)
  - Test caching behavior (skip if exists, force re-extraction)
  - Verify compliance status separation
- **GOTCHA**: Integration tests require internet connection - mark with `@pytest.mark.integration`
- **VALIDATE**: `pytest tests/integration/test_osm_extraction_pipeline.py -v -m integration`

### UPDATE src/data/collection/__init__.py

- **IMPLEMENT**: Export OSMExtractor class
- **PATTERN**: Follow existing `__init__.py` pattern (see current file)
- **ADD**: `from src.data.collection.osm_extractor import OSMExtractor`
- **VALIDATE**: `python -c "from src.data.collection import OSMExtractor; print('Import successful')"`

---

## TESTING STRATEGY

### Unit Tests

**Scope**: Test individual methods in isolation with mocked OSMnx calls

**Requirements**:
- Mock `ox.features_from_polygon()` and `ox.graph_from_polygon()` to return test data
- Test buffer creation with known input/output
- Test directory structure creation
- Test caching logic
- Test service category mapping and duplication
- Test error handling and retry logic
- Achieve 80%+ code coverage

**Fixtures**:
- Small test polygon (box around known Paris coordinates)
- Mock GeoDataFrames for services, buildings
- Mock NetworkX graph for network tests
- Temporary directory for output testing

### Integration Tests

**Scope**: Test full pipeline with real OSM data (small test area)

**Requirements**:
- Extract data for smallest neighborhood (or create test polygon)
- Verify all output files are created
- Verify file formats (GeoJSON, GraphML)
- Verify data quality (valid geometries, correct CRS, no nulls where expected)
- Test caching behavior
- Test compliance status separation

**Markers**:
- Use `@pytest.mark.integration` to skip in CI if needed
- Require internet connection

### Edge Cases

**Test Cases**:
1. Neighborhood with no services (should log warning, continue)
2. Neighborhood with no buildings (should log warning, continue)
3. Network extraction failure (should retry, then log error)
4. Invalid polygon geometry (should raise ValueError)
5. Missing OSM data for area (should handle gracefully)
6. Services in multiple categories (should duplicate correctly)
7. Cache hit scenario (should skip extraction)
8. Force flag override (should re-extract even if cached)

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Linting
ruff check src/data/collection/osm_extractor.py

# Formatting
black --check src/data/collection/osm_extractor.py

# Type checking (if mypy configured)
mypy src/data/collection/osm_extractor.py
```

### Level 2: Unit Tests

```bash
# Run unit tests
pytest tests/unit/test_osm_extractor.py -v

# With coverage
pytest tests/unit/test_osm_extractor.py --cov=src.data.collection.osm_extractor --cov-report=term-missing
```

### Level 3: Integration Tests

```bash
# Run integration tests (requires internet)
pytest tests/integration/test_osm_extraction_pipeline.py -v -m integration
```

### Level 4: Manual Validation

```bash
# Test extraction for single neighborhood
python -c "
from src.data.collection import OSMExtractor
from src.utils.helpers import load_neighborhoods

e = OSMExtractor()
neighborhoods = load_neighborhoods('paris_neighborhoods.geojson')
# Extract smallest neighborhood for testing
result = e.extract_neighborhood(neighborhoods.iloc[0], force=True)
print(f'Extraction successful: {result[\"status\"] == \"success\"}')
print(f'Services: {result.get(\"services_count\", 0)}')
print(f'Buildings: {result.get(\"buildings_count\", 0)}')
"

# Verify output structure
ls -la data/raw/osm/compliant/*/services.geojson
ls -la data/raw/osm/compliant/*/services_by_category/
ls -la data/raw/osm/compliant/*/network.graphml
```

### Level 5: Full Pipeline Validation

```bash
# Extract all neighborhoods (this will take time - 30+ minutes)
python -c "
from src.data.collection import OSMExtractor

e = OSMExtractor()
summary = e.extract_all_neighborhoods()
print(f'Total neighborhoods: {summary[\"total_neighborhoods\"]}')
print(f'Successful: {summary[\"successful_count\"]}')
print(f'Failed: {summary[\"failed_count\"]}')
"

# Verify compliance separation
ls data/raw/osm/compliant/
ls data/raw/osm/non_compliant/

# Check extraction metadata
cat data/raw/osm/extraction_log.json | head -20
```

---

## ACCEPTANCE CRITERIA

- [ ] `OSMExtractor` class implements all extraction methods
- [ ] Services extracted and mapped to 8 NEXI categories correctly
- [ ] Services duplicated when belonging to multiple categories
- [ ] Optimized service location files created per category
- [ ] Buildings extracted with geometry, levels, and area
- [ ] Network extracted with 100m buffer included
- [ ] Walkability features (intersections, pedestrian infrastructure) extracted
- [ ] Data organized by compliance status (compliant/ vs non_compliant/)
- [ ] Caching works correctly (skip if exists, force flag overrides)
- [ ] Error handling with retries implemented
- [ ] Informative warnings logged for low data quality
- [ ] Extraction metadata and summary saved
- [ ] All unit tests pass (80%+ coverage)
- [ ] Integration tests pass
- [ ] Code follows project conventions (PEP 8, type hints, docstrings)
- [ ] No linting or type checking errors
- [ ] Configuration added to `config.yaml`
- [ ] Documentation updated if needed

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes (unit + integration)
- [ ] No linting or type checking errors
- [ ] Manual testing confirms extraction works for test neighborhood
- [ ] Output file structure verified (compliant/ vs non_compliant/)
- [ ] Service category files created correctly
- [ ] Network includes buffer area
- [ ] Extraction metadata saved
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability

---

## NOTES

### Design Decisions

1. **Buffer Implementation**: Using approximate degree-based buffer in WGS84 to keep implementation simple. At Paris latitude, 100m ≈ 0.0009 degrees. This is approximate but sufficient for service extraction near boundaries. More accurate buffers can be applied during feature engineering if needed.

2. **Service Duplication**: Services like `ice_cream` appear in both Grocery and Sustenance categories. We duplicate them in both category files to ensure accurate counts and distance calculations for each category.

3. **Network Buffer**: Network is extracted from buffered polygon to ensure services near boundaries can be reached via network paths. This is critical for accurate distance calculations.

4. **Caching Strategy**: Default to cache (skip if exists) to avoid re-extraction during development. Force flag allows re-extraction when needed (e.g., after OSM data updates).

5. **Error Handling**: Continue processing other neighborhoods if one fails. This ensures partial success and allows identification of problematic neighborhoods.

6. **Coordinate System**: Keep everything in WGS84 (EPSG:4326) for simplicity. Use approximate conversions for meters (degree-to-meter approximation at Paris latitude). No CRS transformations needed.

### Performance Considerations

- OSM extraction can be slow (30+ minutes for all neighborhoods)
- Network extraction is the slowest operation
- Consider parallel processing in future if rate limits allow
- Cache results to avoid re-extraction

### Known Limitations

- Requires internet connection for OSM data
- Subject to OSM API rate limits
- OSM data completeness varies by area
- Large neighborhoods may take significant time

### Future Enhancements

- Parallel neighborhood processing
- Progress bar for long extractions
- Resume capability for interrupted extractions
- Validation of extracted data quality
- Comparison with previous extraction runs
