"""General utility functions for common operations.

This module provides reusable functions for GeoJSON operations, service category
mappings, distance utilities, random seed management, data validation, and file I/O.
"""

import random
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd

# Try to import PyTorch for random seed management
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_neighborhoods(geojson_path: str) -> gpd.GeoDataFrame:
    """Load neighborhoods from GeoJSON file.

    Args:
        geojson_path: Path to GeoJSON file containing neighborhood data.

    Returns:
        GeoDataFrame with neighborhood geometries and properties (name, label,
        verification_status, etc.).

    Raises:
        FileNotFoundError: If the GeoJSON file does not exist.
        ValueError: If the GeoJSON file cannot be read or is invalid.

    Example:
        >>> neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
        >>> print(f"Loaded {len(neighborhoods)} neighborhoods")
    """
    geojson_file = Path(geojson_path)

    if not geojson_file.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    try:
        neighborhoods = gpd.read_file(geojson_path)
    except (OSError, PermissionError) as e:
        # Handle file system errors (permissions, I/O errors)
        raise ValueError(f"Failed to load GeoJSON file: {e}") from e
    except Exception as e:
        # For unexpected errors, preserve original exception type
        # but provide context
        raise RuntimeError(f"Unexpected error loading GeoJSON file: {e}") from e

    return neighborhoods


def get_compliant_neighborhoods(neighborhoods: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter neighborhoods to only include compliant (verified) ones.

    A neighborhood is considered compliant if its `label` property equals "Compliant"
    AND its `verification_status` equals "verified".

    Args:
        neighborhoods: GeoDataFrame containing neighborhood data.

    Returns:
        GeoDataFrame containing only compliant and verified neighborhoods.

    Example:
        >>> all_neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
        >>> compliant = get_compliant_neighborhoods(all_neighborhoods)
        >>> print(f"Found {len(compliant)} compliant neighborhoods")
    """
    if "label" not in neighborhoods.columns:
        raise ValueError("Neighborhoods GeoDataFrame must have 'label' column")

    if "verification_status" not in neighborhoods.columns:
        raise ValueError(
            "Neighborhoods GeoDataFrame must have 'verification_status' column"
        )

    compliant = neighborhoods[
        (neighborhoods["label"] == "Compliant")
        & (neighborhoods["verification_status"] == "verified")
    ].copy()
    return compliant


def get_non_compliant_neighborhoods(
    neighborhoods: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Filter neighborhoods to only include non-compliant ones.

    A neighborhood is considered non-compliant if its `label` equals "Non-Compliant".

    Note: Non-Compliant neighborhoods do not have a `verification_status` field.

    Args:
        neighborhoods: GeoDataFrame containing neighborhood data.

    Returns:
        GeoDataFrame containing only non-compliant neighborhoods.

    Example:
        >>> all_neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
        >>> non_compliant = get_non_compliant_neighborhoods(all_neighborhoods)
        >>> print(f"Found {len(non_compliant)} non-compliant neighborhoods")
    """
    if "label" not in neighborhoods.columns:
        raise ValueError("Neighborhoods GeoDataFrame must have 'label' column")

    # Filter for non-compliant: label == "Non-Compliant"
    non_compliant = neighborhoods[neighborhoods["label"] == "Non-Compliant"].copy()
    return non_compliant


def get_service_category_mapping() -> Dict[str, List[str]]:
    """Get mapping of NEXI service categories to OSM tags.

    Returns a dictionary mapping 8 service category names to lists of OSM amenity
    tags that belong to each category. This mapping is based on the PRD.md
    specification.

    Returns:
        Dictionary mapping category names to lists of OSM tags.

    Example:
        >>> mapping = get_service_category_mapping()
        >>> print(mapping["Education"])
        ['college', 'driving_school', 'kindergarten', ...]
    """
    return {
        "Education": [
            "college",
            "driving_school",
            "kindergarten",
            "language_school",
            "music_school",
            "school",
            "university",
        ],
        "Entertainment": [
            "arts_center",
            "brothel",
            "casino",
            "cinema",
            "community_center",
            "conference_center",
            "events_venue",
            "fountain",
            "gambling",
            "love_hotel",
            "nightclub",
            "planetarium",
            "public_bookcase",
            "social_center",
            "strip_club",
            "studio",
            "swinger_club",
            "theatre",
        ],
        "Grocery": [
            "alcohol",
            "bakery",
            "beverages",
            "brewing_supplies",
            "butcher",
            "cheese",
            "chocolate",
            "coffee",
            "confectionery",
            "convenience",
            "deli",
            "dairy",
            "farm",
            "frozen_food",
            "greengrocer",
            "health_food",
            "ice_cream",
            "pasta",
            "pastry",
            "seafood",
            "spices",
            "tea",
            "water",
            "supermarket",
            "department_store",
            "general",
            "kiosk",
            "mall",
        ],
        "Health": [
            "clinic",
            "dentist",
            "doctors",
            "hospital",
            "nursing_home",
            "pharmacy",
            "social_facility",
        ],
        "Posts and banks": [
            "ATM",
            "bank",
            "bureau_de_change",
            "post_office",
        ],
        "Parks": [
            "park",
            "dog_park",
        ],
        "Sustenance": [
            "restaurant",
            "pub",
            "bar",
            "cafe",
            "fast_food",
            "food_court",
            "ice_cream",
            "biergarten",
        ],
        "Shops": [
            "department_store",
            "general",
            "kiosk",
            "mall",
            "wholesale",
            "baby_goods",
            "bag",
            "boutique",
            "clothes",
            "fabric",
            "fashion_accessories",
            "jewelry",
            "leather",
            "watches",
            "wool",
            "charity",
            "etc",
        ],
    }


def get_service_category_names() -> List[str]:
    """Get list of service category names in consistent order.

    Returns the 8 service category names in the same order as they appear in
    `get_service_category_mapping()`.

    Returns:
        List of 8 category names.

    Example:
        >>> names = get_service_category_names()
        >>> print(names)
        ['Education', 'Entertainment', 'Grocery', ...]
    """
    return list(get_service_category_mapping().keys())


def normalize_distance_by_15min(distance_meters: float) -> float:
    """Normalize distance by 15-minute walk distance.

    Normalizes a distance in meters to a 0-1 scale where 1.0 represents a
    15-minute walk distance (approximately 1200 meters at 5 km/h).

    Args:
        distance_meters: Distance in meters (must be non-negative).

    Returns:
        Normalized distance (0-1 scale where 1.0 = 15-minute walk).

    Raises:
        ValueError: If distance_meters is negative.

    Example:
        >>> normalize_distance_by_15min(1200.0)
        1.0
        >>> normalize_distance_by_15min(600.0)
        0.5
    """
    if distance_meters < 0:
        raise ValueError(f"Distance must be non-negative, got {distance_meters}")

    WALK_15MIN_METERS = 1200.0
    return distance_meters / WALK_15MIN_METERS


def meters_to_walking_minutes(
    distance_meters: float, walking_speed: float = 5.0
) -> float:
    """Convert distance in meters to walking time in minutes.

    Args:
        distance_meters: Distance in meters.
        walking_speed: Walking speed in km/h. Defaults to 5.0 km/h.

    Returns:
        Walking time in minutes.

    Example:
        >>> meters_to_walking_minutes(1000.0)
        12.0
        >>> meters_to_walking_minutes(500.0, walking_speed=4.0)
        7.5
    """
    # Convert km/h to m/min: km/h * 1000 / 60 = m/min
    speed_m_per_min = walking_speed * 1000.0 / 60.0
    return distance_meters / speed_m_per_min


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets random seeds for Python's `random` module, NumPy's random number
    generator, and PyTorch (if available). This ensures reproducible results
    across runs.

    Args:
        seed: Random seed value.

    Example:
        >>> set_random_seeds(42)
        >>> # Now all random operations will be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        # Also set CUDA seeds if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def validate_feature_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.

    Returns:
        True if all required columns are present.

    Raises:
        ValueError: If any required columns are missing.

    Example:
        >>> df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        >>> validate_feature_dataframe(df, ["feature1", "feature2"])
        True
        >>> validate_feature_dataframe(df, ["feature1", "missing"])
        ValueError: Missing required columns: ['missing']
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return True


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Check data quality and return quality report.

    Args:
        df: DataFrame to check.

    Returns:
        Dictionary containing quality metrics:
        - total_rows: Total number of rows
        - total_columns: Total number of columns
        - missing_values: Dictionary of column -> missing count
        - missing_percentage: Dictionary of column -> missing percentage
        - duplicate_rows: Number of duplicate rows
        - duplicate_percentage: Percentage of duplicate rows

    Example:
        >>> df = pd.DataFrame({"a": [1, 2, None], "b": [3, 3, 4]})
        >>> report = check_data_quality(df)
        >>> print(report["missing_values"])
        {'a': 1, 'b': 0}
    """
    missing_counts = df.isnull().sum().to_dict()
    missing_percentages = {
        col: (count / len(df)) * 100 for col, count in missing_counts.items()
    }

    duplicate_count = df.duplicated().sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0.0

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": missing_counts,
        "missing_percentage": missing_percentages,
        "duplicate_rows": int(duplicate_count),
        "duplicate_percentage": duplicate_percentage,
    }


def ensure_dir_exists(dir_path: str) -> None:
    """Create directory if it does not exist.

    Creates the directory and all parent directories if they don't exist.

    Args:
        dir_path: Path to directory to create.

    Example:
        >>> ensure_dir_exists("data/processed")
        >>> # Directory is created if it doesn't exist
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: str, format: str = "csv") -> None:
    """Save DataFrame to file.

    Args:
        df: DataFrame to save.
        path: File path to save to.
        format: File format ("csv" or "parquet"). Defaults to "csv".

    Raises:
        ValueError: If format is not supported.

    Example:
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> save_dataframe(df, "data.csv", format="csv")
        >>> save_dataframe(df, "data.parquet", format="parquet")
    """
    file_path = Path(path)

    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "csv":
        df.to_csv(file_path, index=False)
    elif format.lower() == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'.")


def load_dataframe(path: str, format: str = "csv") -> pd.DataFrame:
    """Load DataFrame from file.

    Args:
        path: File path to load from.
        format: File format ("csv" or "parquet"). Defaults to "csv".

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If format is not supported.

    Example:
        >>> df = load_dataframe("data.csv", format="csv")
        >>> df = load_dataframe("data.parquet", format="parquet")
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if format.lower() == "csv":
        return pd.read_csv(file_path)
    elif format.lower() == "parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'.")
