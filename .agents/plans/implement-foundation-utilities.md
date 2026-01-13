# Feature: Foundation Utilities

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Implement three foundational utility modules (`config.py`, `logging.py`, `helpers.py`) that provide configuration management, centralized logging, and reusable helper functions. These utilities are prerequisites for all other modules in the project and enable consistent configuration access, structured logging, and common operations like GeoJSON loading, service category mappings, and data validation.

## User Story

As a **developer** working on the AI4SI project, I want **foundational utility modules** that provide configuration management, logging, and helper functions, so that **all subsequent modules can rely on consistent, well-tested utilities** for common operations.

## Problem Statement

The project currently has only empty `__init__.py` files in the `src/utils/` directory. Before implementing data collection, training, or evaluation modules, we need:

1. **Configuration Management**: A way to load and access YAML configuration files with environment variable overrides
2. **Logging Infrastructure**: Centralized logging setup that all modules can use consistently
3. **Helper Functions**: Reusable utilities for GeoJSON loading, service category mappings, random seed management, data validation, and file I/O

Without these utilities, each module would need to implement its own configuration loading, logging setup, and helper functions, leading to code duplication and inconsistency.

## Solution Statement

Implement three independent utility modules:

1. **`src/utils/config.py`**: Load YAML configuration files, merge with environment variables, provide typed access to configuration values with validation
2. **`src/utils/logging.py`**: Configure Python's standard `logging` module with consistent formatting, file/console handlers, and module-specific loggers
3. **`src/utils/helpers.py`**: Provide reusable functions for GeoJSON operations, service category mappings, distance utilities, random seed management, data validation, and file I/O

All modules will follow PEP 8 conventions, include comprehensive type hints, Google-style docstrings, and be fully unit tested.

## Feature Metadata

**Feature Type**: New Capability (Foundation)
**Estimated Complexity**: Medium
**Primary Systems Affected**: All future modules (data collection, training, evaluation)
**Dependencies**: 
- `pyyaml>=6.0` (configuration loading)
- `geopandas>=0.14.0` (GeoJSON operations)
- `pandas>=2.0.0` (DataFrame operations)
- `numpy>=1.24.0` (numerical operations)
- Standard library: `logging`, `pathlib`, `os`, `random`, `typing`

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `models/config.yaml.example` (lines 1-43) - Why: Defines the exact configuration structure we need to load and validate
- `paris_neighborhoods.geojson` (lines 1-150) - Why: Shows GeoJSON structure with properties (name, label, verification_status) that helpers must parse
- `PRD.md` (lines 631-640) - Why: Contains NEXI service category mappings to OSM tags that must be implemented in helpers
- `CURSOR.md` (lines 206-232) - Why: Defines code conventions (PEP 8, type hints, docstrings, naming patterns) we must follow
- `requirements.txt` (lines 1-40) - Why: Lists all available dependencies and versions we can use
- `src/__init__.py` (lines 1-3) - Why: Shows project version and package structure pattern
- `.gitignore` (lines 1-42) - Why: Shows which files/directories are ignored (e.g., `.env` for API keys)

### New Files to Create

- `src/utils/config.py` - Configuration management module
- `src/utils/logging.py` - Logging setup module
- `src/utils/helpers.py` - General utility functions module
- `tests/unit/test_config.py` - Unit tests for config module
- `tests/unit/test_logging.py` - Unit tests for logging module
- `tests/unit/test_helpers.py` - Unit tests for helpers module

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [PyYAML Documentation - safe_load](https://pyyaml.org/wiki/PyYAMLDocumentation.html)
  - Specific section: Using `yaml.safe_load()` for security
  - Why: Required for safely loading YAML configuration files without code execution risks
- [GeoPandas read_file Documentation](https://geopandas.org/en/stable/docs/user_guide/io.html)
  - Specific section: Reading GeoJSON files
  - Why: Shows how to load GeoJSON files into GeoDataFrames correctly
- [Python logging Module Documentation](https://docs.python.org/3/library/logging.html)
  - Specific section: Basic Logging Tutorial, Logging Handlers
  - Why: Required for implementing proper logging configuration with handlers and formatters
- [Python pathlib Documentation](https://docs.python.org/3/library/pathlib.html)
  - Specific section: Path operations
  - Why: Best practice for file path handling and directory creation

### Patterns to Follow

**Naming Conventions:**
- Use `snake_case` for functions and variables (e.g., `load_config`, `get_logger`)
- Use `PascalCase` for classes (e.g., `Config` if implementing a class-based approach)
- Use descriptive names that reflect domain concepts (e.g., `get_compliant_neighborhoods` not `get_good`)

**Error Handling:**
- Use specific exception types (`FileNotFoundError`, `ValueError`, `KeyError`)
- Provide clear, actionable error messages
- Log errors before raising when appropriate
- Example pattern:
  ```python
  if not config_path.exists():
      raise FileNotFoundError(
          f"Configuration file not found: {config_path}. "
          f"Please create it from {config_path}.example"
      )
  ```

**Logging Pattern:**
- Use module-level loggers: `logger = logging.getLogger(__name__)`
- Log at appropriate levels: DEBUG (detailed), INFO (progress), WARNING (non-critical issues), ERROR (errors)
- Include context in log messages (neighborhood names, file paths, counts)
- Example pattern:
  ```python
  logger.info(f"Loading neighborhoods from {geojson_path}")
  logger.debug(f"Found {len(neighborhoods)} neighborhoods")
  ```

**Type Hints:**
- Use `typing` module for complex types (`Dict`, `List`, `Optional`, `Union`)
- Use `Path` from `pathlib` for file paths
- Use `GeoDataFrame` from `geopandas` for geospatial data
- Example pattern:
  ```python
  from typing import Dict, List, Optional
  from pathlib import Path
  import geopandas as gpd
  
  def load_neighborhoods(geojson_path: str) -> gpd.GeoDataFrame:
      ...
  ```

**Docstring Style (Google format):**
- Use Google-style docstrings with Args, Returns, Raises sections
- Include examples for complex functions
- Example pattern:
  ```python
  def normalize_distance_by_15min(distance_meters: float) -> float:
      """Normalize distance by 15-minute walk distance.
      
      Args:
          distance_meters: Distance in meters.
      
      Returns:
          Normalized distance (0-1 scale where 1.0 = 15-minute walk).
      
      Example:
          >>> normalize_distance_by_15min(1200.0)
          1.0
          >>> normalize_distance_by_15min(600.0)
          0.5
      """
  ```

**Testing Pattern:**
- Use pytest fixtures for common test data
- Test both success and error cases
- Use parametrized tests for multiple input scenarios
- Mock file I/O operations
- Example pattern:
  ```python
  import pytest
  from unittest.mock import mock_open, patch
  
  def test_load_config_file_not_found():
      with pytest.raises(FileNotFoundError):
          load_config("nonexistent.yaml")
  ```

---

## IMPLEMENTATION PLAN

### Phase 1: Configuration Module (`config.py`)

**Goal**: Implement YAML configuration loading with environment variable support and validation.

**Tasks:**
- Create `load_config()` function to load YAML files using `yaml.safe_load()`
- Implement environment variable override mechanism (using `os.getenv()`)
- Add configuration validation (check required sections exist)
- Implement singleton pattern for global config access via `get_config()`
- Add error handling for missing files and invalid YAML
- Create typed access helpers for nested configuration values

### Phase 2: Logging Module (`logging.py`)

**Goal**: Set up centralized logging with consistent formatting and handlers.

**Tasks:**
- Create `setup_logging()` function to configure root logger
- Implement log format: `[YYYY-MM-DD HH:MM:SS] [LEVEL] [MODULE] Message`
- Add console handler (stdout) with appropriate log level
- Add optional file handler for experiment logging
- Create `get_logger()` function for module-specific loggers
- Implement `setup_experiment_logging()` for experiment-specific log files
- Ensure log directories are created automatically

### Phase 3: Helpers Module (`helpers.py`)

**Goal**: Implement reusable utility functions for common operations.

**Tasks:**
- Implement GeoJSON loading functions (`load_neighborhoods`, filtering functions)
- Create service category mapping dictionary (8 NEXI categories → OSM tags)
- Implement distance utilities (normalization, walking time conversion)
- Add random seed management (`set_random_seeds` for Python, NumPy, PyTorch)
- Create data validation helpers (DataFrame validation, quality checks)
- Implement file I/O utilities (directory creation, DataFrame save/load)

### Phase 4: Testing & Validation

**Goal**: Comprehensive unit tests for all utility modules.

**Tasks:**
- Write unit tests for config module (loading, validation, error cases)
- Write unit tests for logging module (formatting, handlers, loggers)
- Write unit tests for helpers module (all functions with edge cases)
- Test error handling and edge cases
- Validate type hints with mypy (optional but recommended)

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### CREATE `src/utils/config.py`

- **IMPLEMENT**: Configuration loading module with YAML support and environment variable overrides
- **PATTERN**: Use `yaml.safe_load()` for security (never `yaml.load()`)
- **IMPORTS**: `yaml`, `os`, `pathlib.Path`, `typing.Dict`, `typing.Any`, `typing.Optional`
- **FUNCTIONS**:
  - `load_config(config_path: str) -> Dict[str, Any]`: Load YAML file, merge env vars, return dict
  - `get_config() -> Dict[str, Any]`: Singleton pattern, returns cached config or loads default
  - `_merge_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]`: Private helper to merge env vars
  - `_validate_config(config: Dict[str, Any]) -> None`: Private helper to validate required sections
- **GOTCHA**: 
  - Use `yaml.safe_load()` not `yaml.load()` for security
  - Handle nested environment variables (e.g., `DATA__TRAIN_SPLIT` for `data.train_split`)
  - Default config path: `models/config.yaml` (relative to project root)
- **VALIDATE**: `python -c "from src.utils.config import get_config; print(get_config()['model']['name'])"`

### CREATE `src/utils/logging.py`

- **IMPLEMENT**: Centralized logging configuration module
- **PATTERN**: Use standard library `logging` module (not structlog - that's for FastAPI projects)
- **IMPORTS**: `logging`, `pathlib.Path`, `typing.Optional`, `datetime`
- **FUNCTIONS**:
  - `setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None`: Configure root logger
  - `get_logger(name: str) -> logging.Logger`: Get module-specific logger
  - `setup_experiment_logging(experiment_dir: str) -> logging.Logger`: Create experiment-specific logger
- **FORMAT**: `[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s` with date format `%Y-%m-%d %H:%M:%S`
- **HANDLERS**: 
  - Console handler (always) with appropriate level
  - File handler (optional) if `log_file` provided
  - Create log directory if needed using `pathlib.Path.mkdir(parents=True, exist_ok=True)`
- **GOTCHA**: 
  - Don't configure logging multiple times (check if already configured)
  - Use `logging.getLogger(__name__)` pattern in other modules
  - Experiment logging should create `{experiment_dir}/logs/training.log`
- **VALIDATE**: `python -c "from src.utils.logging import setup_logging, get_logger; setup_logging(); logger = get_logger('test'); logger.info('Test message')"`

### CREATE `src/utils/helpers.py`

- **IMPLEMENT**: Reusable utility functions for common operations
- **PATTERN**: Follow existing codebase patterns from CURSOR.md
- **IMPORTS**: `geopandas as gpd`, `pandas as pd`, `numpy as np`, `pathlib.Path`, `typing.Dict`, `typing.List`, `typing.Any`, `random`, `os`
- **GEOJSON FUNCTIONS**:
  - `load_neighborhoods(geojson_path: str) -> gpd.GeoDataFrame`: Load GeoJSON, return GeoDataFrame
  - `get_compliant_neighborhoods(neighborhoods: gpd.GeoDataFrame) -> gpd.GeoDataFrame`: Filter where `label == "Verified"`
  - `get_non_compliant_neighborhoods(neighborhoods: gpd.GeoDataFrame) -> gpd.GeoDataFrame`: Filter where `label != "Verified"` or `verification_status == "not_verified"`
- **SERVICE CATEGORY MAPPINGS** (from PRD.md lines 631-640):
  - `get_service_category_mapping() -> Dict[str, List[str]]`: Return mapping of 8 NEXI categories to OSM tags
    - Education: ["college", "driving_school", "kindergarten", "language_school", "music_school", "school", "university"]
    - Entertainment: ["arts_center", "cinema", "community_center", "theatre"]
    - Grocery: ["supermarket", "bakery", "convenience", "greengrocer"]
    - Health: ["clinic", "dentist", "doctors", "hospital", "pharmacy"]
    - Posts and banks: ["ATM", "bank", "post_office"]
    - Parks: ["park", "dog_park"]
    - Sustenance: ["restaurant", "pub", "bar", "cafe", "fast_food"]
    - Shops: ["department_store", "general", "kiosk", "mall", "boutique", "clothes"]
  - `get_service_category_names() -> List[str]`: Return list of 8 category names in consistent order
- **DISTANCE UTILITIES**:
  - `normalize_distance_by_15min(distance_meters: float) -> float`: Normalize by 1200m (15-min walk)
  - `meters_to_walking_minutes(distance_meters: float, walking_speed: float = 5.0) -> float`: Convert meters to minutes (default 5 km/h)
- **RANDOM SEED MANAGEMENT**:
  - `set_random_seeds(seed: int) -> None`: Set seeds for `random`, `numpy.random`, and `torch.random` (if torch available)
- **DATA VALIDATION**:
  - `validate_feature_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool`: Check required columns exist, raise ValueError if not
  - `check_data_quality(df: pd.DataFrame) -> Dict[str, Any]`: Return quality report (missing values, duplicates, etc.)
- **FILE I/O UTILITIES**:
  - `ensure_dir_exists(dir_path: str) -> None`: Create directory if not exists (with parents)
  - `save_dataframe(df: pd.DataFrame, path: str, format: str = "csv") -> None`: Save to CSV or Parquet
  - `load_dataframe(path: str, format: str = "csv") -> pd.DataFrame`: Load from CSV or Parquet
- **GOTCHA**: 
  - GeoJSON properties: `name`, `label`, `verification_status` (check paris_neighborhoods.geojson structure)
  - Service category names must match exactly: "Posts and banks" (not "Posts_and_banks")
  - Handle missing PyTorch gracefully in `set_random_seeds` (try/except ImportError)
  - Use `pathlib.Path` for all path operations
- **VALIDATE**: `python -c "from src.utils.helpers import load_neighborhoods, get_service_category_mapping; n = load_neighborhoods('paris_neighborhoods.geojson'); print(len(n)); m = get_service_category_mapping(); print(list(m.keys()))"`

### CREATE `tests/unit/test_config.py`

- **IMPLEMENT**: Unit tests for config module
- **PATTERN**: Use pytest fixtures and mocking for file I/O
- **IMPORTS**: `pytest`, `unittest.mock`, `pathlib.Path`, `tempfile`, `yaml`
- **TESTS**:
  - `test_load_config_valid_file()`: Test loading valid YAML file
  - `test_load_config_file_not_found()`: Test FileNotFoundError for missing file
  - `test_load_config_invalid_yaml()`: Test YAMLError for malformed YAML
  - `test_get_config_singleton()`: Test that `get_config()` returns same instance
  - `test_env_var_override()`: Test environment variable merging (mock `os.getenv`)
  - `test_validate_config_missing_section()`: Test validation raises error for missing required sections
- **FIXTURES**: Create temporary YAML files for testing
- **VALIDATE**: `pytest tests/unit/test_config.py -v`

### CREATE `tests/unit/test_logging.py`

- **IMPLEMENT**: Unit tests for logging module
- **PATTERN**: Use pytest and logging capture utilities
- **IMPORTS**: `pytest`, `logging`, `pathlib.Path`, `tempfile`
- **TESTS**:
  - `test_setup_logging_console_handler()`: Test console handler is added
  - `test_setup_logging_file_handler()`: Test file handler is created when log_file provided
  - `test_get_logger_returns_logger()`: Test logger is returned with correct name
  - `test_logger_format()`: Test log message format matches expected pattern
  - `test_setup_experiment_logging()`: Test experiment logger creates log file in correct directory
  - `test_log_levels()`: Test different log levels work correctly
- **VALIDATE**: `pytest tests/unit/test_logging.py -v`

### CREATE `tests/unit/test_helpers.py`

- **IMPLEMENT**: Unit tests for helpers module
- **PATTERN**: Use pytest fixtures and mock GeoJSON files
- **IMPORTS**: `pytest`, `geopandas as gpd`, `pandas as pd`, `numpy as np`, `pathlib.Path`, `tempfile`, `unittest.mock`
- **TESTS**:
  - `test_load_neighborhoods()`: Test loading GeoJSON file (use paris_neighborhoods.geojson or fixture)
  - `test_get_compliant_neighborhoods()`: Test filtering for compliant neighborhoods
  - `test_get_non_compliant_neighborhoods()`: Test filtering for non-compliant neighborhoods
  - `test_get_service_category_mapping()`: Test mapping structure and all 8 categories present
  - `test_get_service_category_names()`: Test returns list of 8 names in correct order
  - `test_normalize_distance_by_15min()`: Test normalization (1200m = 1.0, 600m = 0.5)
  - `test_meters_to_walking_minutes()`: Test conversion with default and custom speeds
  - `test_set_random_seeds()`: Test seeds are set for random, numpy (mock torch if not available)
  - `test_validate_feature_dataframe()`: Test validation passes/fails correctly
  - `test_check_data_quality()`: Test quality report structure
  - `test_ensure_dir_exists()`: Test directory creation
  - `test_save_load_dataframe()`: Test CSV and Parquet save/load roundtrip
- **FIXTURES**: Create sample GeoJSON and DataFrame fixtures
- **VALIDATE**: `pytest tests/unit/test_helpers.py -v`

### UPDATE `src/utils/__init__.py`

- **IMPLEMENT**: Export main functions from utility modules for convenient imports
- **PATTERN**: Follow Python package structure best practices
- **EXPORTS**:
  ```python
  from src.utils.config import get_config, load_config
  from src.utils.logging import setup_logging, get_logger, setup_experiment_logging
  from src.utils.helpers import (
      load_neighborhoods,
      get_compliant_neighborhoods,
      get_non_compliant_neighborhoods,
      get_service_category_mapping,
      get_service_category_names,
      normalize_distance_by_15min,
      meters_to_walking_minutes,
      set_random_seeds,
      validate_feature_dataframe,
      check_data_quality,
      ensure_dir_exists,
      save_dataframe,
      load_dataframe,
  )
  ```
- **VALIDATE**: `python -c "from src.utils import get_config, get_logger; print('Imports work')"`

---

## TESTING STRATEGY

### Unit Tests

**Scope**: Test each function independently with mocked dependencies where appropriate.

**Requirements**:
- Test success cases with valid inputs
- Test error cases (missing files, invalid data, etc.)
- Test edge cases (empty data, boundary values)
- Use pytest fixtures for common test data
- Mock file I/O operations where appropriate
- Achieve 80%+ code coverage

**Test Organization**:
- One test file per utility module
- Group related tests in classes or use descriptive function names
- Use parametrized tests for multiple input scenarios

### Integration Tests

**Scope**: Test interactions between utility modules (minimal for this feature since modules are independent).

**Requirements**:
- Test that config and logging work together (e.g., log config loading)
- Test that helpers use config for paths (if applicable)

### Edge Cases

**Specific edge cases to test**:
- Config file with missing required sections
- Config file with invalid YAML syntax
- GeoJSON file with missing required properties
- Empty GeoJSON file
- DataFrame with no columns
- Distance normalization with zero or negative values
- Random seed setting when PyTorch not installed
- Directory creation when parent directories don't exist
- File save/load with special characters in paths

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
# Format code with black
black src/utils/ tests/unit/test_*.py --check

# Lint with ruff
ruff check src/utils/ tests/unit/test_*.py

# Type checking with mypy (optional but recommended)
mypy src/utils/ --ignore-missing-imports
```

### Level 2: Unit Tests

```bash
# Run all unit tests for utilities
pytest tests/unit/test_config.py tests/unit/test_logging.py tests/unit/test_helpers.py -v

# Run with coverage
pytest tests/unit/test_config.py tests/unit/test_logging.py tests/unit/test_helpers.py --cov=src.utils --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_config.py -v
```

### Level 3: Manual Validation

```bash
# Test config loading
python -c "from src.utils.config import get_config; config = get_config(); print('Config loaded:', 'model' in config)"

# Test logging setup
python -c "from src.utils.logging import setup_logging, get_logger; setup_logging(); logger = get_logger('test'); logger.info('Logging works!')"

# Test helpers - GeoJSON loading
python -c "from src.utils.helpers import load_neighborhoods; n = load_neighborhoods('paris_neighborhoods.geojson'); print(f'Loaded {len(n)} neighborhoods')"

# Test helpers - Service categories
python -c "from src.utils.helpers import get_service_category_mapping, get_service_category_names; m = get_service_category_mapping(); names = get_service_category_names(); print(f'Categories: {len(names)}, Mapping keys: {list(m.keys())}')"

# Test helpers - Distance utilities
python -c "from src.utils.helpers import normalize_distance_by_15min, meters_to_walking_minutes; print(f'1200m normalized: {normalize_distance_by_15min(1200.0)}'); print(f'600m in minutes: {meters_to_walking_minutes(600.0)}')"

# Test imports from package
python -c "from src.utils import get_config, get_logger, load_neighborhoods; print('All imports successful')"
```

### Level 4: Integration Validation

```bash
# Test that modules work together (config + logging)
python -c "
from src.utils.config import get_config
from src.utils.logging import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)
config = get_config()
logger.info(f'Config loaded with {len(config)} top-level sections')
"
```

---

## ACCEPTANCE CRITERIA

- [ ] `config.py` loads YAML files using `yaml.safe_load()` with environment variable support
- [ ] `config.py` validates required configuration sections exist
- [ ] `config.py` provides singleton access via `get_config()`
- [ ] `logging.py` sets up logging with consistent format and console/file handlers
- [ ] `logging.py` provides `get_logger()` for module-specific loggers
- [ ] `logging.py` supports experiment-specific logging
- [ ] `helpers.py` loads GeoJSON files and filters neighborhoods by compliance status
- [ ] `helpers.py` provides service category mappings (8 categories) matching PRD.md
- [ ] `helpers.py` implements distance utilities (normalization, walking time)
- [ ] `helpers.py` sets random seeds for reproducibility (Python, NumPy, PyTorch)
- [ ] `helpers.py` provides data validation and file I/O utilities
- [ ] All functions have type hints and Google-style docstrings
- [ ] All code follows PEP 8 conventions (100 char line length)
- [ ] Unit tests achieve 80%+ coverage
- [ ] All validation commands pass with zero errors
- [ ] No linting or type checking errors
- [ ] Manual validation confirms all functions work correctly

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes (unit tests)
- [ ] No linting or type checking errors
- [ ] Manual testing confirms all functions work
- [ ] Acceptance criteria all met
- [ ] Code reviewed for quality and maintainability
- [ ] `src/utils/__init__.py` updated with exports

---

## NOTES

### Design Decisions

1. **Standard library logging vs structlog**: Using standard library `logging` instead of `structlog` because:
   - This is an ML/research project, not a web service
   - Standard library is sufficient for our needs
   - Reduces dependencies
   - The reference doc mentions structlog for FastAPI projects, but we're not using FastAPI

2. **Singleton pattern for config**: Using module-level caching for `get_config()` to avoid reloading YAML file on every call, but allowing explicit `load_config()` for flexibility.

3. **Service category mappings**: Hardcoded in `helpers.py` based on PRD.md rather than loaded from config because:
   - These are domain constants, not configuration
   - Ensures consistency across the codebase
   - Easier to validate and test

4. **Error handling strategy**: Raising exceptions rather than returning None/False for errors because:
   - Fail-fast principle for configuration and critical operations
   - Clearer error messages for debugging
   - Type hints work better with exceptions

### Potential Issues

1. **PyTorch import in `set_random_seeds`**: Handle gracefully with try/except ImportError since PyTorch may not be installed during initial development.

2. **GeoJSON structure assumptions**: The helpers assume specific property names (`name`, `label`, `verification_status`). If the GeoJSON structure changes, these functions will need updates.

3. **Environment variable naming**: Using double underscore (`__`) convention for nested config (e.g., `DATA__TRAIN_SPLIT` for `data.train_split`) - document this clearly.

### Future Enhancements

- Add configuration schema validation using `jsonschema` library
- Add support for configuration hot-reloading (if needed)
- Add more sophisticated data quality checks (outlier detection, distribution analysis)
- Consider adding caching for expensive operations (GeoJSON loading, service category lookups)
