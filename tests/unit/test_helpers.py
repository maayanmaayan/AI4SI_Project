"""Unit tests for helpers module."""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src.utils.helpers import (
    check_data_quality,
    ensure_dir_exists,
    get_compliant_neighborhoods,
    get_non_compliant_neighborhoods,
    get_service_category_mapping,
    get_service_category_names,
    load_dataframe,
    load_neighborhoods,
    meters_to_walking_minutes,
    normalize_distance_by_15min,
    save_dataframe,
    set_random_seeds,
    validate_feature_dataframe,
)


@pytest.fixture
def sample_geojson_file():
    """Create a sample GeoJSON file for testing."""
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Verified Neighborhood",
                    "label": "Compliant",
                    "verification_status": "verified",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "name": "Unverified Compliant Neighborhood",
                    "label": "Compliant",
                    "verification_status": "unverified",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "name": "Non-Compliant Neighborhood",
                    "label": "Non-Compliant",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[4, 4], [5, 4], [5, 5], [4, 5], [4, 4]]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "name": "Another Verified",
                    "label": "Compliant",
                    "verification_status": "verified",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[6, 6], [7, 6], [7, 7], [6, 7], [6, 6]]],
                },
            },
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        import json

        json.dump(geojson_data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_neighborhoods_gdf():
    """Create a sample GeoDataFrame for testing."""
    data = {
        "name": [
            "Compliant Verified 1",
            "Compliant Unverified 1",
            "Compliant Verified 2",
            "Non-Compliant 1",
            "Non-Compliant 2",
        ],
        "label": [
            "Compliant",
            "Compliant",
            "Compliant",
            "Non-Compliant",
            "Non-Compliant",
        ],
        "verification_status": [
            "verified",
            "unverified",
            "verified",
            None,  # Non-Compliant neighborhoods don't have this field
            None,
        ],
        "geometry": [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),
            Polygon([(6, 6), (7, 6), (7, 7), (6, 7)]),
            Polygon([(8, 8), (9, 8), (9, 9), (8, 9)]),
        ],
    }
    return gpd.GeoDataFrame(data)


def test_load_neighborhoods(sample_geojson_file):
    """Test loading neighborhoods from GeoJSON file."""
    neighborhoods = load_neighborhoods(sample_geojson_file)

    assert isinstance(neighborhoods, gpd.GeoDataFrame)
    assert len(neighborhoods) == 4
    assert "name" in neighborhoods.columns
    assert "label" in neighborhoods.columns


def test_load_neighborhoods_file_not_found():
    """Test FileNotFoundError is raised for missing file."""
    with pytest.raises(FileNotFoundError):
        load_neighborhoods("nonexistent.geojson")


def test_get_compliant_neighborhoods(sample_neighborhoods_gdf):
    """Test filtering for compliant neighborhoods."""
    compliant = get_compliant_neighborhoods(sample_neighborhoods_gdf)

    assert isinstance(compliant, gpd.GeoDataFrame)
    assert len(compliant) == 2  # Only compliant AND verified
    assert all(compliant["label"] == "Compliant")
    assert all(compliant["verification_status"] == "verified")


def test_get_compliant_neighborhoods_missing_column():
    """Test ValueError is raised when required columns are missing."""
    gdf = gpd.GeoDataFrame(
        {"name": ["Test"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}
    )

    with pytest.raises(ValueError, match="must have 'label' column"):
        get_compliant_neighborhoods(gdf)

    # Test missing verification_status column
    gdf_with_label = gpd.GeoDataFrame(
        {
            "name": ["Test"],
            "label": ["Compliant"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        }
    )
    with pytest.raises(ValueError, match="must have 'verification_status' column"):
        get_compliant_neighborhoods(gdf_with_label)


def test_get_non_compliant_neighborhoods(sample_neighborhoods_gdf):
    """Test filtering for non-compliant neighborhoods."""
    non_compliant = get_non_compliant_neighborhoods(sample_neighborhoods_gdf)

    assert isinstance(non_compliant, gpd.GeoDataFrame)
    assert len(non_compliant) == 2  # Only Non-Compliant neighborhoods
    assert all(non_compliant["label"] == "Non-Compliant")


def test_get_non_compliant_neighborhoods_missing_column():
    """Test ValueError is raised when label column is missing."""
    gdf = gpd.GeoDataFrame(
        {"name": ["Test"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}
    )

    with pytest.raises(ValueError, match="must have 'label' column"):
        get_non_compliant_neighborhoods(gdf)


def test_get_service_category_mapping():
    """Test service category mapping structure and all 8 categories present."""
    mapping = get_service_category_mapping()

    assert isinstance(mapping, dict)
    assert len(mapping) == 8

    expected_categories = [
        "Education",
        "Entertainment",
        "Grocery",
        "Health",
        "Posts and banks",
        "Parks",
        "Sustenance",
        "Shops",
    ]

    for category in expected_categories:
        assert category in mapping
        assert isinstance(mapping[category], list)
        assert len(mapping[category]) > 0

    # Check specific mappings
    assert "school" in mapping["Education"]
    assert "cinema" in mapping["Entertainment"]
    assert "supermarket" in mapping["Grocery"]
    assert "hospital" in mapping["Health"]
    assert "bank" in mapping["Posts and banks"]
    assert "park" in mapping["Parks"]
    assert "restaurant" in mapping["Sustenance"]
    assert "mall" in mapping["Shops"]


def test_get_service_category_names():
    """Test returns list of 8 names in correct order."""
    names = get_service_category_names()

    assert isinstance(names, list)
    assert len(names) == 8

    expected_order = [
        "Education",
        "Entertainment",
        "Grocery",
        "Health",
        "Posts and banks",
        "Parks",
        "Sustenance",
        "Shops",
    ]

    assert names == expected_order


def test_normalize_distance_by_15min():
    """Test normalization (1200m = 1.0, 600m = 0.5)."""
    assert normalize_distance_by_15min(1200.0) == 1.0
    assert normalize_distance_by_15min(600.0) == 0.5
    assert normalize_distance_by_15min(0.0) == 0.0
    assert normalize_distance_by_15min(2400.0) == 2.0


def test_meters_to_walking_minutes():
    """Test conversion with default and custom speeds."""
    # Default speed: 5 km/h = 5000 m / 60 min = 83.33 m/min
    # 1000 m / 83.33 m/min ≈ 12 min
    result = meters_to_walking_minutes(1000.0)
    assert abs(result - 12.0) < 0.1

    # Custom speed: 4 km/h = 4000 m / 60 min = 66.67 m/min
    # 500 m / 66.67 m/min ≈ 7.5 min
    result = meters_to_walking_minutes(500.0, walking_speed=4.0)
    assert abs(result - 7.5) < 0.1


def test_set_random_seeds():
    """Test seeds are set for random, numpy (mock torch if not available)."""
    set_random_seeds(42)

    # Test Python random
    import random

    value1 = random.random()
    set_random_seeds(42)
    value2 = random.random()
    assert value1 == value2

    # Test NumPy random
    value1 = np.random.random()
    set_random_seeds(42)
    value2 = np.random.random()
    assert value1 == value2

    # Test PyTorch if available
    try:
        import torch

        value1 = torch.rand(1).item()
        set_random_seeds(42)
        value2 = torch.rand(1).item()
        assert value1 == value2
    except ImportError:
        # PyTorch not available, skip
        pass


def test_validate_feature_dataframe():
    """Test validation passes/fails correctly."""
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

    # Should pass with all required columns
    assert validate_feature_dataframe(df, ["feature1", "feature2"]) is True

    # Should raise ValueError with missing columns
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_feature_dataframe(df, ["feature1", "missing_column"])


def test_check_data_quality():
    """Test quality report structure."""
    df = pd.DataFrame(
        {
            "a": [1, 2, None, 4],
            "b": [5, 5, 6, 7],
            "c": [8, 9, 10, 11],
        }
    )

    report = check_data_quality(df)

    assert isinstance(report, dict)
    assert "total_rows" in report
    assert "total_columns" in report
    assert "missing_values" in report
    assert "missing_percentage" in report
    assert "duplicate_rows" in report
    assert "duplicate_percentage" in report

    assert report["total_rows"] == 4
    assert report["total_columns"] == 3
    assert report["missing_values"]["a"] == 1
    assert report["missing_values"]["b"] == 0
    assert report["duplicate_rows"] == 0


def test_check_data_quality_with_duplicates():
    """Test quality report with duplicate rows."""
    df = pd.DataFrame({"a": [1, 2, 2], "b": [3, 4, 4]})

    report = check_data_quality(df)

    assert report["duplicate_rows"] == 1
    assert report["duplicate_percentage"] > 0


def test_ensure_dir_exists():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        new_dir = Path(temp_dir) / "nested" / "deep" / "directory"
        ensure_dir_exists(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()


def test_save_load_dataframe_csv():
    """Test CSV save/load roundtrip."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.csv"
        save_dataframe(df, str(file_path), format="csv")

        assert file_path.exists()

        loaded_df = load_dataframe(str(file_path), format="csv")
        pd.testing.assert_frame_equal(df, loaded_df)


def test_save_load_dataframe_parquet():
    """Test Parquet save/load roundtrip."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.parquet"
        save_dataframe(df, str(file_path), format="parquet")

        assert file_path.exists()

        loaded_df = load_dataframe(str(file_path), format="parquet")
        pd.testing.assert_frame_equal(df, loaded_df)


def test_save_dataframe_creates_directory():
    """Test that save_dataframe creates parent directory."""
    df = pd.DataFrame({"a": [1, 2]})

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "nested" / "dir" / "test.csv"
        save_dataframe(df, str(file_path), format="csv")

        assert file_path.parent.exists()
        assert file_path.exists()


def test_save_dataframe_invalid_format():
    """Test ValueError is raised for unsupported format."""
    df = pd.DataFrame({"a": [1, 2]})

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.json"

        with pytest.raises(ValueError, match="Unsupported format"):
            save_dataframe(df, str(file_path), format="json")


def test_load_dataframe_file_not_found():
    """Test FileNotFoundError is raised for missing file."""
    with pytest.raises(FileNotFoundError):
        load_dataframe("nonexistent.csv", format="csv")


def test_load_dataframe_invalid_format():
    """Test ValueError is raised for unsupported format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.csv"
        file_path.touch()

        with pytest.raises(ValueError, match="Unsupported format"):
            load_dataframe(str(file_path), format="json")
