# Feature: Census Data Loader using pynsee

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Implement a census data loader that uses the `pynsee` Python library to fetch demographic and socioeconomic data for Paris neighborhoods at the IRIS (French census unit) level. **The loader will ONLY process neighborhoods with `"label": "Compliant"` from the paris_neighborhoods.geojson file**, following the exemplar-based learning approach where training data comes exclusively from compliant neighborhoods.

The loader will extract demographic and socioeconomic features and match IRIS data to neighborhood boundaries for downstream feature engineering.

**Core Functionality:**
- Explore available INSEE datasets using pynsee metadata functions
- Download demographic data at IRIS level for Paris (INSEE code: 75056)
- Download FILOSOFI income data at IRIS level
- **Process ONLY compliant neighborhoods** (filter by `label == "Compliant"`)
- Extract features: population density, SES index, car ownership rate, children per capita (estimated), elderly ratio (estimated), unemployment rate, student ratio, walking ratio, cycling ratio, public transport ratio, two-wheelers ratio, car commute ratio, retired ratio, permanent employment ratio, temporary employment ratio, median income, poverty rate
- Match IRIS boundaries to neighborhood geometries using spatial joins
- Save census data organized in directory structure matching OSM data organization
- Support configurable caching (skip if data exists, with force flag)
- Handle missing data and data quality validation

**Directory Structure:**
The census data will follow the same directory structure as OSM data extraction:
```
data/raw/census/
└── compliant/
    ├── paris_rive_gauche/
    │   └── census_data.parquet (or census_data.csv)
    ├── clichy-batignolles/
    │   └── census_data.parquet
    ├── bartholomé–brancion_(15e)/
    │   └── census_data.parquet
    └── ... (one folder per compliant neighborhood)
```

This structure mirrors the OSM data organization:
```
data/raw/osm/
├── compliant/
│   ├── paris_rive_gauche/
│   │   ├── services.geojson
│   │   ├── buildings.geojson
│   │   ├── network.graphml
│   │   └── services_by_category/
│   ├── clichy-batignolles/
│   └── ...
└── non_compliant/
    └── ...
```

Each compliant neighborhood gets its own folder containing the census data file, making it easy to match census data with OSM data for the same neighborhood during feature engineering.

## User Story

As a **data engineer/researcher**
I want to **load census demographic data for Paris neighborhoods using pynsee**
So that **I can compute demographic features needed for training the 15-minute city service gap prediction model**

## Problem Statement

The project requires census data for computing demographic and socioeconomic features at the grid cell level. Currently, no census data collection pipeline exists. The feature engineering phase depends on this demographic data to compute the 20+ features per point required for model training.

**Feature Limitations:**
- **Children per capita** and **Elderly ratio**: These are estimated using Paris-wide age distribution ratios since RP_ACTRES_IRIS only contains working-age population (15-64). The estimation uses commune-level total population and IRIS-level working-age population to estimate total population per neighborhood, then applies Paris-wide proportions (47% children, 53% elderly of non-working-age population). This is a limitation as actual IRIS-level age data is not available.

**Critical Constraint:** Following the exemplar-based learning approach, census data extraction must **ONLY process compliant neighborhoods** (`label == "Compliant"`). Non-compliant neighborhoods are excluded from census data collection, consistent with the training strategy that uses only compliant neighborhoods.

**Key Challenges:**
1. Understanding available INSEE datasets and variables via pynsee
2. Matching IRIS-level census data to neighborhood boundaries (spatial join)
3. Extracting specific demographic indicators from INSEE datasets
4. Handling missing data and data quality issues
5. Organizing data in directory structure matching OSM data (one folder per neighborhood)
6. Filtering to only compliant neighborhoods before processing

## Solution Statement

Create a `CensusLoader` class following the same pattern as `OSMExtractor`:
1. **Test Script Phase**: Create an exploratory script to understand pynsee API and available datasets
2. **Implementation Phase**: Build `CensusLoader` class that:
   - Uses pynsee to fetch IRIS-level data for Paris
   - Extracts required demographic features
   - Performs spatial joins to match IRIS data to neighborhoods
   - Saves data to `data/raw/census/` organized by compliance status
   - Supports caching and error handling with retries

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: 
- `src/data/collection/` - New census_loader.py module
- `models/config.yaml` - Add census extraction configuration section
- `src/data/collection/__init__.py` - Export CensusLoader
- `tests/unit/` - Unit tests for census loader
**Dependencies**: 
- `pynsee` library (external, no API key required)
- `geopandas` (existing dependency)
- `pandas` (existing dependency)
- `shapely` (existing dependency)

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `src/data/collection/osm_extractor.py` (lines 32-836) - Why: Pattern for data extractor class structure, initialization, caching, error handling, retry logic, and data organization by compliance status
- `src/utils/config.py` (lines 21-193) - Why: Configuration loading pattern using get_config(), environment variable overrides
- `src/utils/logging.py` (lines 74-92) - Why: Logger initialization pattern using get_logger(__name__)
- `src/utils/helpers.py` (lines 24-57, 60-117, 406-418) - Why: load_neighborhoods(), get_compliant_neighborhoods(), get_non_compliant_neighborhoods(), ensure_dir_exists() utilities
- `src/utils/helpers.py` (lines 421-447, 450-478) - Why: save_dataframe(), load_dataframe() for CSV/Parquet file I/O
- `models/config.yaml` (lines 49-56) - Why: Configuration structure for data extraction (osm_extraction section pattern to mirror)
- `tests/unit/test_osm_extractor.py` (lines 15-100) - Why: Test fixture patterns, pytest structure, mocking patterns
- `src/data/collection/__init__.py` (lines 1-5) - Why: Module exports pattern

### New Files to Create

- `scripts/explore_pynsee.py` - Test/exploration script to understand pynsee API and datasets
- `src/data/collection/census_loader.py` - CensusLoader class implementation
- `tests/unit/test_census_loader.py` - Unit tests for census loader
- `models/config.yaml` - UPDATE: Add census_extraction configuration section

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [pynsee Documentation - Getting Started](https://pynsee.readthedocs.io/en/latest/get_data.html)
  - Specific section: Local data access with get_local_data()
  - Why: Core API for fetching INSEE census data at IRIS level
- [pynsee Documentation - Local Data](https://pynsee.readthedocs.io/en/latest/localdata.html)
  - Specific section: get_local_metadata() and get_local_data() functions
  - Why: Understanding available datasets and variables before implementation
- [INSEE IRIS Documentation](https://www.insee.fr/fr/information/2017499)
  - Why: Understanding IRIS geographic units and how they relate to neighborhoods
- [Geopandas Spatial Join](https://geopandas.org/en/stable/docs/user_guide/mergingdata.html#spatial-joins)
  - Why: Matching IRIS census data to neighborhood boundaries using spatial operations

### Patterns to Follow

**Naming Conventions:**
- Class names: PascalCase (e.g., `CensusLoader`)
- Method names: snake_case (e.g., `load_census_data()`)
- Private methods: prefix with underscore (e.g., `_match_iris_to_neighborhoods()`)
- File names: snake_case (e.g., `census_loader.py`)

**Error Handling:** (from osm_extractor.py:818-833)
```python
try:
    # Operation
except Exception as e:
    error_msg = f"Attempt {attempt}/{self.retry_attempts} failed: {str(e)}"
    logger.error(f"Error: {error_msg}")
    if attempt < self.retry_attempts:
        logger.info(f"Retrying in {self.retry_delay} seconds...")
        time.sleep(self.retry_delay)
    else:
        result["status"] = "failed"
        result["error"] = str(e)
```

**Logging Pattern:** (from osm_extractor.py:29)
```python
from src.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("Message")
logger.warning("Warning message")
logger.error("Error message")
```

**Configuration Pattern:** (from osm_extractor.py:56-69)
```python
def __init__(self) -> None:
    """Initialize with configuration."""
    self.config = get_config()
    census_config = self.config.get("census_extraction", {})
    self.cache_results: bool = census_config.get("cache_results", True)
    self.retry_attempts: int = census_config.get("retry_attempts", 3)
    self.retry_delay: int = census_config.get("retry_delay", 5)
    logger.info("CensusLoader initialized")
```

**Directory Structure Pattern:** (from osm_extractor.py:107-139)
- Data organized by compliance status: `data/raw/census/compliant/` (only compliant neighborhoods)
- **One folder per neighborhood**: `data/raw/census/compliant/{neighborhood_name}/`
- **Census data file**: Saved as `census_data.parquet` (or `census_data.csv`) in each neighborhood folder
- **Structure matches OSM**: Same folder naming and organization as `data/raw/osm/compliant/{neighborhood_name}/`
- Normalize neighborhood names: lowercase, replace spaces with underscores (same normalization as OSM extractor)
- Use `ensure_dir_exists()` helper

**Data Saving Pattern:** (from osm_extractor.py:770-772)
```python
# Save census data in neighborhood-specific folder
output_path = output_dir / "census_data.parquet"
df.to_parquet(output_path, index=False)
# Or CSV: df.to_csv(output_path, index=False)
# Structure: data/raw/census/compliant/{neighborhood_name}/census_data.parquet
```

**Spatial Join Pattern:** (geopandas)
```python
import geopandas as gpd
# Match IRIS data to neighborhoods using spatial join
matched = gpd.sjoin(iris_gdf, neighborhoods_gdf, how="inner", predicate="intersects")
```

---

## IMPLEMENTATION PLAN

### Phase 1: Exploration Script

Create a test script to explore pynsee library capabilities and understand available datasets.

**Tasks:**
- Install pynsee dependency
- Create exploration script to list available datasets
- Test fetching IRIS-level data for Paris
- Identify required variables for demographic features
- Document findings and dataset codes

### Phase 2: Core CensusLoader Implementation

Build the main CensusLoader class following OSMExtractor patterns.

**Tasks:**
- Implement CensusLoader class with initialization
- Add methods to fetch IRIS data using pynsee
- Implement spatial join to match IRIS to neighborhoods
- Extract required demographic features
- Add caching and error handling
- Save data organized by compliance status

### Phase 3: Configuration & Integration

Add configuration support and integrate with existing codebase.

**Tasks:**
- Add census_extraction section to config.yaml
- Update __init__.py to export CensusLoader
- Add helper methods if needed

### Phase 4: Testing & Validation

Create comprehensive tests following existing test patterns.

**Tasks:**
- Write unit tests with fixtures
- Test data fetching and spatial joins
- Test error handling and caching
- Validate data quality and feature extraction

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### CREATE scripts/explore_pynsee.py

- **IMPLEMENT**: Create exploration script to understand pynsee API
- **PATTERN**: Follow Python script structure with main() function
- **IMPORTS**: `from pynsee.localdata import get_local_metadata, get_local_data`, `import pandas as pd`
- **FUNCTIONALITY**: 
  - List available datasets using get_local_metadata()
  - Search for IRIS-level datasets for Paris (geocode: '75056')
  - Test fetching sample data
  - Print dataset structure and available variables
  - Document findings in comments
- **VALIDATE**: `python scripts/explore_pynsee.py` runs without errors and prints dataset information

### UPDATE models/config.yaml

- **IMPLEMENT**: Add census_extraction configuration section after osm_extraction
- **PATTERN**: Mirror osm_extraction structure (lines 49-56)
- **CONFIG KEYS**:
  - `cache_results: true` - Skip if data already exists
  - `retry_attempts: 3` - Number of retries on failure
  - `retry_delay: 5` - Seconds between retries
  - `paris_geocode: "75056"` - INSEE code for Paris
  - `iris_level: "IRIS"` - Geographic level
  - `min_data_warning: 10` - Warn if fewer IRIS units found
- **VALIDATE**: `python -c "from src.utils.config import get_config; c = get_config(); assert 'census_extraction' in c"`

### CREATE src/data/collection/census_loader.py

- **IMPLEMENT**: Create CensusLoader class following OSMExtractor pattern
- **PATTERN**: Mirror osm_extractor.py structure (lines 32-836)
- **IMPORTS**: 
  ```python
  import time
  from pathlib import Path
  from typing import Any, Dict, List, Optional
  import geopandas as gpd
  import pandas as pd
  from pynsee.localdata import get_local_metadata, get_local_data
  from src.utils.config import get_config
  from src.utils.logging import get_logger
  from src.utils.helpers import (
      ensure_dir_exists,
      get_compliant_neighborhoods,
      get_non_compliant_neighborhoods,
      load_neighborhoods,
      save_dataframe,
  )
  ```
- **CLASS STRUCTURE**:
  - `__init__(self) -> None` - Initialize with config, mirror osm_extractor.py:56-69
  - `_get_output_directory(self, neighborhood_name: str) -> Path` - Get output dir for compliant neighborhoods only, mirror osm_extractor.py:107-139 but only create `data/raw/census/compliant/{neighborhood_name}/`
  - `_check_cache(self, output_dir: Path) -> bool` - Check if data exists, mirror osm_extractor.py:141-169
  - `_fetch_iris_data(self) -> pd.DataFrame` - Fetch IRIS-level census data for Paris using pynsee
  - `_load_iris_boundaries(self) -> gpd.GeoDataFrame` - Load IRIS boundary GeoJSON (may need to download)
  - `_match_iris_to_neighborhoods(self, iris_data: pd.DataFrame, iris_boundaries: gpd.GeoDataFrame, neighborhoods: gpd.GeoDataFrame) -> pd.DataFrame` - Spatial join to match IRIS to neighborhoods
  - `_extract_demographic_features(self, matched_data: pd.DataFrame) -> pd.DataFrame` - Extract required features (population density, SES, car ownership, etc.)
  - `load_neighborhood_census(self, neighborhood_row: gpd.GeoSeries, force: bool = False) -> Dict[str, Any]` - Main method to load census data for one neighborhood (only processes if label == "Compliant")
  - `load_all_neighborhoods(self, geojson_path: Optional[str] = None, force: bool = False) -> Dict[str, Any]` - Load census data for **compliant neighborhoods only**
- **VALIDATE**: `python -c "from src.data.collection.census_loader import CensusLoader; loader = CensusLoader(); print('OK')"`

### IMPLEMENT _get_output_directory method in census_loader.py

- **IMPLEMENT**: Get output directory for compliant neighborhood
- **PATTERN**: Mirror osm_extractor.py:107-139 but simplified (only compliant, no is_compliant parameter)
- **DIRECTORY STRUCTURE**: `data/raw/census/compliant/{neighborhood_name}/`
  - Matches OSM structure: `data/raw/osm/compliant/{neighborhood_name}/`
  - Each neighborhood gets its own folder
  - Census data file saved as `census_data.parquet` (or `census_data.csv`) in the neighborhood folder
- **NORMALIZATION**: Lowercase neighborhood name, replace spaces with underscores (same as OSM extractor)
- **RETURNS**: Path object to output directory (created if needed)
- **NOTE**: This method only handles compliant neighborhoods. Non-compliant neighborhoods are never processed.
- **VALIDATE**: Method creates directory at `data/raw/census/compliant/{normalized_name}/` matching OSM structure

### IMPLEMENT _fetch_iris_data method in census_loader.py

- **IMPLEMENT**: Fetch IRIS-level census data for Paris using pynsee
- **PATTERN**: Use get_local_data() with appropriate parameters
- **PARAMETERS**:
  - `nivgeo='IRIS'` - IRIS geographic level
  - `geocodes=['75056']` - Paris INSEE code
  - `variables` - Need to identify from exploration script (population, households, age groups, income, etc.)
- **RETURNS**: pandas DataFrame with IRIS codes and demographic variables
- **ERROR HANDLING**: Wrap in try/except, log errors, return empty DataFrame on failure
- **VALIDATE**: Method returns DataFrame with IRIS codes and at least one demographic column

### IMPLEMENT _load_iris_boundaries method in census_loader.py

- **IMPLEMENT**: Load IRIS boundary GeoJSON file
- **PATTERN**: Use geopandas.read_file() similar to load_neighborhoods() helper
- **SOURCE**: IRIS boundaries available from INSEE/IGN (may need to download or use pynsee if available)
- **RETURNS**: GeoDataFrame with IRIS geometries and codes
- **NOTE**: May need to download IRIS boundaries GeoJSON file first, or check if pynsee provides geometries
- **VALIDATE**: Method returns GeoDataFrame with geometry column and IRIS code column

### IMPLEMENT _match_iris_to_neighborhoods method in census_loader.py

- **IMPLEMENT**: Spatial join to match IRIS census data to neighborhood boundaries
- **PATTERN**: Use geopandas.sjoin() with 'intersects' predicate
- **INPUTS**: 
  - iris_data: DataFrame with IRIS codes and demographic data
  - iris_boundaries: GeoDataFrame with IRIS geometries
  - neighborhoods: GeoDataFrame with neighborhood boundaries
- **PROCESS**:
  1. Merge iris_data with iris_boundaries on IRIS code
  2. Perform spatial join: sjoin(iris_gdf, neighborhoods, how='inner', predicate='intersects')
  3. Aggregate IRIS data per neighborhood (weighted average or sum based on feature type)
- **RETURNS**: DataFrame with neighborhood_name and demographic features
- **VALIDATE**: Method returns DataFrame with neighborhood_name column and demographic features

### IMPLEMENT _extract_demographic_features method in census_loader.py

- **IMPLEMENT**: Extract required demographic features from matched census data
- **REQUIRED FEATURES** (from PRD.md:671):
  - Population density (population / area)
  - SES index (from income data, may need calculation)
  - Car ownership (households with car / total households)
  - Children per capita (children / total population)
  - Household size (average persons per household)
  - Elderly ratio (elderly population / total population)
- **PATTERN**: Calculate derived features from raw INSEE variables
- **RETURNS**: DataFrame with neighborhood_name and extracted feature columns
- **VALIDATE**: Method returns DataFrame with all 6 required demographic feature columns

### IMPLEMENT load_neighborhood_census method in census_loader.py

- **IMPLEMENT**: Main method to load census data for one neighborhood
- **PATTERN**: Mirror osm_extractor.py:extract_neighborhood() (lines 692-835)
- **CRITICAL**: Check if neighborhood label == "Compliant" at start, return early if not compliant
- **PROCESS**:
  1. Validate neighborhood is compliant (label == "Compliant"), skip if not
  2. Check cache if not force
  3. Fetch IRIS data
  4. Load IRIS boundaries
  5. Match IRIS to neighborhood
  6. Extract demographic features
  7. Save to output directory: `{output_dir}/census_data.parquet` (or CSV)
  8. Return summary dictionary
- **RETURNS**: Dict with status, neighborhood_name, label, features_count, error
- **ERROR HANDLING**: Retry logic with configurable attempts and delays
- **VALIDATE**: Method successfully loads and saves census data for a compliant neighborhood, skips non-compliant ones

### IMPLEMENT load_all_neighborhoods method in census_loader.py

- **IMPLEMENT**: Load census data for **compliant neighborhoods only**
- **PATTERN**: Mirror osm_extractor.py:extract_all_neighborhoods() (lines 837-925) but filter to compliant only
- **PROCESS**:
  1. Load neighborhoods GeoJSON
  2. **Filter to compliant neighborhoods only** using `get_compliant_neighborhoods()` helper
  3. Process each compliant neighborhood sequentially
  4. Save metadata and summary (only compliant neighborhoods in summary)
- **RETURNS**: Dict with summary statistics:
  - `total_neighborhoods`: Total compliant neighborhoods processed
  - `successful_count`: Number of successful extractions
  - `failed_count`: Number of failed extractions
  - `cached_count`: Number of cached (skipped) extractions
  - `compliant_count`: Same as total_neighborhoods (for consistency)
- **NOTE**: Non-compliant neighborhoods are completely excluded from processing
- **VALIDATE**: Method processes only compliant neighborhoods and returns summary

### UPDATE src/data/collection/__init__.py

- **IMPLEMENT**: Export CensusLoader class
- **PATTERN**: Mirror existing export pattern (lines 3-5)
- **ADD**: `from src.data.collection.census_loader import CensusLoader`
- **ADD**: `"CensusLoader"` to __all__ list
- **VALIDATE**: `python -c "from src.data.collection import CensusLoader; print('OK')"`

### CREATE tests/unit/test_census_loader.py

- **IMPLEMENT**: Unit tests for CensusLoader following test_osm_extractor.py patterns
- **PATTERN**: Mirror test_osm_extractor.py structure (lines 1-100+)
- **FIXTURES**:
  - `@pytest.fixture def loader()` - CensusLoader instance
  - `@pytest.fixture def test_neighborhood()` - Sample neighborhood GeoSeries
  - `@pytest.fixture def mock_iris_data()` - Mock IRIS census DataFrame
  - `@pytest.fixture def mock_iris_boundaries()` - Mock IRIS boundaries GeoDataFrame
- **TEST CLASSES**:
  - `TestInitialization` - Test __init__ and config loading
  - `TestOutputDirectory` - Test directory creation
  - `TestCache` - Test caching logic
  - `TestFetchIrisData` - Test IRIS data fetching (with mocking)
  - `TestSpatialJoin` - Test IRIS to neighborhood matching
  - `TestFeatureExtraction` - Test demographic feature extraction
  - `TestLoadNeighborhood` - Test main load_neighborhood_census method
- **MOCKING**: Use unittest.mock.patch for pynsee API calls
- **VALIDATE**: `pytest tests/unit/test_census_loader.py -v` passes all tests

---

## TESTING STRATEGY

### Unit Tests

**Scope**: Test each method independently with mocked dependencies

**Fixtures**: Create reusable fixtures for CensusLoader instance, test neighborhoods, mock IRIS data

**Mocking Strategy**: 
- Mock pynsee API calls (get_local_data, get_local_metadata) to avoid external dependencies
- Mock file I/O operations where appropriate
- Use real geopandas operations for spatial join testing

**Coverage Goals**: 
- Test all public methods
- Test error handling paths
- Test caching logic
- Test configuration loading

### Integration Tests

**Scope**: Test end-to-end workflow with real pynsee calls (optional, can be skipped if API is slow)

**Note**: Integration tests may be slow due to pynsee API calls. Consider marking with @pytest.mark.slow

### Edge Cases

- Missing IRIS data for a neighborhood
- IRIS boundaries don't intersect neighborhood boundaries
- Missing required variables in INSEE dataset
- Network errors during pynsee API calls
- Invalid neighborhood geometries
- Empty census data returned from pynsee

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Check Python syntax
python -m py_compile src/data/collection/census_loader.py

# Run ruff linter (if configured)
ruff check src/data/collection/census_loader.py

# Run black formatter check (if configured)
black --check src/data/collection/census_loader.py
```

### Level 2: Import Validation

```bash
# Test imports work correctly
python -c "from src.data.collection.census_loader import CensusLoader; print('Import OK')"

# Test module exports
python -c "from src.data.collection import CensusLoader; print('Export OK')"
```

### Level 3: Unit Tests

```bash
# Run unit tests
pytest tests/unit/test_census_loader.py -v

# Run with coverage
pytest tests/unit/test_census_loader.py --cov=src/data/collection/census_loader --cov-report=term-missing
```

### Level 4: Manual Validation

```bash
# Test exploration script
python scripts/explore_pynsee.py

# Test CensusLoader initialization
python -c "from src.data.collection.census_loader import CensusLoader; loader = CensusLoader(); print('Initialization OK')"

# Test loading census data for one neighborhood (requires pynsee and data)
python -c "
from src.data.collection.census_loader import CensusLoader
from src.utils.helpers import load_neighborhoods
loader = CensusLoader()
neighborhoods = load_neighborhoods('paris_neighborhoods.geojson')
result = loader.load_neighborhood_census(neighborhoods.iloc[0])
print(f'Status: {result[\"status\"]}')"
```

### Level 5: Integration Test (Optional)

```bash
# Test loading all neighborhoods (slow, requires pynsee API access)
python -c "
from src.data.collection.census_loader import CensusLoader
loader = CensusLoader()
summary = loader.load_all_neighborhoods()
print(f'Summary: {summary}')"
```

---

## ACCEPTANCE CRITERIA

- [ ] Exploration script successfully lists pynsee datasets and tests data fetching
- [ ] CensusLoader class implements all required methods following OSMExtractor patterns
- [ ] Configuration section added to config.yaml with all required settings
- [ ] CensusLoader successfully fetches IRIS-level census data for Paris using pynsee
- [ ] **Only compliant neighborhoods are processed** (non-compliant neighborhoods are skipped)
- [ ] Spatial join correctly matches IRIS data to neighborhood boundaries
- [ ] All 6 required demographic features extracted (population density, SES index, car ownership, children per capita, household size, elderly ratio)
- [ ] Data saved to `data/raw/census/compliant/{neighborhood_name}/census_data.parquet` matching OSM directory structure
- [ ] Directory structure mirrors OSM organization (one folder per neighborhood)
- [ ] Caching works correctly (skips if data exists, respects force flag)
- [ ] Error handling with retries implemented
- [ ] All unit tests pass with 80%+ coverage
- [ ] Code follows project conventions (PEP 8, type hints, docstrings)
- [ ] No regressions in existing functionality
- [ ] CensusLoader exported from src.data.collection module

---

## COMPLETION CHECKLIST

- [ ] Exploration script created and tested
- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] Configuration section added to config.yaml
- [ ] CensusLoader class fully implemented
- [ ] All unit tests written and passing
- [ ] Full test suite passes (unit + integration if applicable)
- [ ] No linting or type checking errors
- [ ] Manual testing confirms census data loading works
- [ ] Data quality validated (check for missing values, reasonable ranges)
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability
- [ ] Documentation updated (docstrings complete)

---

## NOTES

### Key Design Decisions

1. **Compliant Neighborhoods Only**: Census data extraction processes ONLY neighborhoods with `label == "Compliant"` from paris_neighborhoods.geojson. This aligns with the exemplar-based learning approach where training data comes exclusively from compliant neighborhoods. Non-compliant neighborhoods are completely excluded from census data collection.

2. **Directory Structure Matching OSM**: Census data follows the exact same directory structure as OSM data:
   - `data/raw/census/compliant/{neighborhood_name}/census_data.parquet`
   - Mirrors: `data/raw/osm/compliant/{neighborhood_name}/services.geojson`
   - This makes it easy to match census and OSM data for the same neighborhood during feature engineering
   - Each neighborhood gets its own folder, consistent with OSM organization

3. **Following OSMExtractor Pattern**: CensusLoader mirrors OSMExtractor structure for consistency and maintainability. This ensures developers familiar with one extractor can easily understand the other.

4. **pynsee Library Choice**: Using pynsee instead of direct INSEE API calls because:
   - No API key required
   - Python-friendly interface
   - Handles authentication and data formatting automatically
   - Well-documented and maintained

5. **Spatial Join Approach**: Using geopandas spatial join to match IRIS census units to neighborhoods. This handles cases where:
   - Multiple IRIS units overlap a neighborhood
   - IRIS boundaries don't perfectly align with neighborhood boundaries
   - Need to aggregate IRIS data per neighborhood

6. **Feature Extraction**: Some features (like SES index) may require calculation from raw INSEE variables. The _extract_demographic_features method handles these transformations.

7. **IRIS Boundaries**: May need to download IRIS boundary GeoJSON file separately if pynsee doesn't provide geometries. Check pynsee documentation or download from data.gouv.fr.

### Potential Challenges

1. **Dataset Discovery**: Finding the right INSEE dataset codes and variable names may require exploration. The test script phase addresses this.

2. **Spatial Join Performance**: Large IRIS datasets may slow spatial joins. Consider optimizing with spatial indexes if needed.

3. **Missing Data**: Some IRIS units may have missing demographic data. Need robust handling (fill with neighborhood average or mark as missing).

4. **Variable Mapping**: INSEE variable names may not directly map to required features. May need to combine multiple variables or calculate derived features.

5. **API Rate Limits**: pynsee may have rate limits. Consider adding delays between API calls if needed.

### Future Enhancements

- Cache IRIS boundaries GeoJSON file locally to avoid re-downloading
- Add data quality validation (check for outliers, missing values)
- Support for other French cities (not just Paris)
- Parallel processing for multiple neighborhoods
- Progress bar for long-running operations

### Dependencies to Add

Add to requirements.txt (or create if doesn't exist):
```
pynsee>=0.1.0
```

Note: pynsee may have its own dependencies. Check pynsee documentation for full dependency list.
