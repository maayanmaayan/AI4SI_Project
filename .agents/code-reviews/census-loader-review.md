# Code Review: census_loader.py

**Date:** 2024-12-19  
**File:** `src/data/collection/census_loader.py`  
**Lines:** 1561

## Stats

- **Files Modified:** 0
- **Files Added:** 0
- **Files Deleted:** 0
- **New lines:** 0
- **Deleted lines:** 0

## Summary

The `census_loader.py` file is a comprehensive implementation for loading census data using the pynsee library. The code follows project conventions well with proper logging, type hints, and docstrings. However, there are several performance issues related to redundant API calls and some code quality improvements needed around error logging.

## Issues Found

### 1. Performance: Redundant API calls in `_extract_demographic_features()`

**severity:** high  
**file:** src/data/collection/census_loader.py  
**line:** 860, 865, 1070, 1081  
**issue:** `_fetch_commune_age_data()` and `_fetch_iris_data()` are called multiple times per neighborhood, causing redundant downloads  
**detail:** The `_extract_demographic_features()` method is called once per neighborhood in `load_neighborhood_census()`. Inside this method:
- Line 860: `_fetch_commune_age_data()` is called to get commune-level population data
- Line 865: `_fetch_iris_data()` is called without arguments to get all Paris IRIS data for ratio calculation
- Line 1070: `_fetch_commune_age_data()` is called again (duplicate call)
- Line 1081: `_fetch_iris_data()` is called again without arguments (duplicate call)

Since commune-level data and Paris-wide IRIS data are the same for all neighborhoods, these expensive API calls/downloads are repeated unnecessarily. For N neighborhoods, this results in 2N calls to `_fetch_commune_age_data()` and 2N calls to `_fetch_iris_data()` instead of 2 total calls.

**suggestion:** 
1. Fetch commune age data and Paris-wide IRIS data once in `load_neighborhood_census()` before calling `_extract_demographic_features()`
2. Pass these as parameters to `_extract_demographic_features()` to avoid redundant calls
3. Alternatively, implement instance-level caching in `__init__` or use a class-level cache with lazy loading

Example fix:
```python
# In load_neighborhood_census(), before calling _extract_demographic_features():
commune_age_data = self._fetch_commune_age_data()
iris_data_all = self._fetch_iris_data()  # Get all Paris IRIS data once

# Then pass to _extract_demographic_features():
features_df = self._extract_demographic_features(
    neighborhood_data, 
    neighborhoods,
    commune_age_data=commune_age_data,
    iris_data_all=iris_data_all
)
```

### 2. Code Quality: Using `traceback.print_exc()` instead of proper logging

**severity:** medium  
**file:** src/data/collection/census_loader.py  
**line:** 185-188, 308-310, 421-423  
**issue:** Direct `traceback.print_exc()` calls bypass logging system  
**detail:** The code uses `traceback.print_exc()` in exception handlers, which prints directly to stderr and bypasses the logging system. This violates the project's logging standards and makes it harder to control log output, filter logs, or redirect to files.

**suggestion:** Replace `traceback.print_exc()` with `logger.exception()` which automatically includes the traceback in the log message at ERROR level. This follows the logging pattern established in the codebase.

Example fix:
```python
# Current (lines 183-188):
except Exception as e:
    logger.error(f"Error fetching IRIS data: {e}")
    import traceback
    traceback.print_exc()
    return pd.DataFrame()

# Should be:
except Exception as e:
    logger.exception(f"Error fetching IRIS data: {e}")
    return pd.DataFrame()
```

Apply this fix to all three locations:
- Line 185-188 in `_fetch_iris_data()`
- Line 308-310 in `_fetch_filosofi_income()`
- Line 421-423 in `_fetch_logement_car_ownership()`

### 3. Code Quality: Extremely long method violates single responsibility

**severity:** low  
**file:** src/data/collection/census_loader.py  
**line:** 720-1318  
**issue:** `_extract_demographic_features()` method is ~600 lines long  
**detail:** The `_extract_demographic_features()` method is responsible for extracting 17+ different demographic features. While the code is well-organized with clear sections, the method is extremely long and violates the single responsibility principle. This makes it harder to test individual features, maintain, and understand.

**suggestion:** Consider refactoring into smaller, feature-specific methods:
- `_extract_population_features()` - population density
- `_extract_ses_features()` - SES index
- `_extract_car_features()` - car ownership and commute ratios
- `_extract_age_features()` - children and elderly ratios
- `_extract_employment_features()` - unemployment, employment contracts
- `_extract_commute_features()` - walking, cycling, public transport ratios
- `_extract_income_features()` - median income, poverty rate

Each method would take `matched_data` and return a DataFrame with the relevant feature columns, then combine them in `_extract_demographic_features()`. This would improve testability and maintainability.

### 4. Potential Logic Issue: Division by zero checks could be more consistent

**severity:** low  
**file:** src/data/collection/census_loader.py  
**line:** 940-944, 975-976, 1058-1060, 1175-1177, etc.  
**issue:** Division by zero protection is implemented but inconsistently  
**detail:** The code has division by zero checks in some places (e.g., line 940-944 for SES index) but uses `.fillna(0)` in others. While both approaches work, the pattern is inconsistent. Some calculations check `if denominator.sum() > 0` while others rely on pandas' automatic handling of division by zero (which produces inf/NaN).

**suggestion:** Standardize the approach. Consider using a helper function for safe division:
```python
def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    """Safely divide two series, returning default when denominator is zero."""
    mask = denominator > 0
    result = pd.Series(index=numerator.index, dtype=float)
    result[mask] = numerator[mask] / denominator[mask]
    result[~mask] = default
    return result
```

This would make the code more consistent and easier to maintain.

## Positive Aspects

1. **Excellent documentation:** Comprehensive docstrings with examples for all methods
2. **Good error handling:** Proper try-except blocks with meaningful error messages
3. **Type hints:** Consistent use of type hints throughout
4. **Logging:** Good use of logging at appropriate levels (info, warning, error)
5. **Compliance filtering:** Correctly implements exemplar-based learning by only processing compliant neighborhoods
6. **Caching support:** Implements caching to avoid redundant data fetching
7. **Retry logic:** Implements retry mechanism for robustness
8. **Spatial operations:** Proper handling of CRS conversions and spatial joins

## Recommendations

1. **High Priority:** Fix redundant API calls (Issue #1) - this will significantly improve performance
2. **Medium Priority:** Replace `traceback.print_exc()` with `logger.exception()` (Issue #2) - improves logging consistency
3. **Low Priority:** Consider refactoring long method (Issue #3) - improves maintainability
4. **Low Priority:** Standardize division by zero handling (Issue #4) - improves code consistency

## Conclusion

The code is well-written and follows project conventions. The main concerns are performance-related (redundant API calls) and some code quality improvements around logging. The issues are fixable without major architectural changes.
