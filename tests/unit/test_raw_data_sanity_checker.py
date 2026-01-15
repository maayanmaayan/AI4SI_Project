"""Unit tests for raw data sanity checker module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

from src.data.validation.raw_data_sanity_checker import RawDataSanityChecker


@pytest.fixture
def checker():
    """Create RawDataSanityChecker instance for testing."""
    with patch("src.data.validation.raw_data_sanity_checker.get_config"):
        with patch("src.data.validation.raw_data_sanity_checker.load_neighborhoods"):
            checker = RawDataSanityChecker()
            checker.data_root = Path("data")
            checker.neighborhoods_geojson = "paris_neighborhoods.geojson"
            return checker


@pytest.fixture
def mock_services_gdf():
    """Create mock services GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "name": ["Service 1", "Service 2", "Service 3"],
            "amenity": ["cafe", "restaurant", "pharmacy"],
            "category": ["Sustenance", "Sustenance", "Health"],
            "neighborhood_name": ["test_neighborhood"] * 3,
            "geometry": [
                Point(2.35, 48.85),
                Point(2.351, 48.851),
                Point(2.352, 48.852),
            ],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def mock_buildings_gdf():
    """Create mock buildings GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "name": ["Building 1", "Building 2"],
            "building": ["residential", "commercial"],
            "building:levels": [5, 6],
            "area_m2": [500.0, 800.0],
            "neighborhood_name": ["test_neighborhood"] * 2,
            "geometry": [
                box(2.35, 48.85, 2.351, 48.851),
                box(2.351, 48.851, 2.352, 48.852),
            ],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def mock_network_graph():
    """Create mock network graph."""
    # osmnx expects MultiDiGraph
    G = nx.MultiDiGraph()
    G.add_node(1, x=2.35, y=48.85)
    G.add_node(2, x=2.351, y=48.851)
    G.add_node(3, x=2.352, y=48.852)
    G.add_edge(1, 2, length=100.0)
    G.add_edge(2, 3, length=150.0)
    G.add_edge(1, 3, length=200.0)
    return G


@pytest.fixture
def mock_census_df():
    """Create mock census DataFrame."""
    return pd.DataFrame(
        {
            "neighborhood_name": ["test_neighborhood"],
            "population_density": [20000.0],
            "children_per_capita": [0.12],
            "elderly_ratio": [0.18],
            "ses_index": [0.35],
            "unemployment_rate": [0.08],
            "median_income": [30000.0],
            "poverty_rate": [0.15],
            "car_ownership_rate": [0.35],
            "walking_ratio": [0.25],
            "cycling_ratio": [0.03],
            "public_transport_ratio": [0.60],
            "two_wheelers_ratio": [0.02],
            "car_commute_ratio": [0.10],
        }
    )


def test_checker_initialization(checker):
    """Test checker initialization."""
    assert checker is not None
    assert checker.data_root == Path("data")
    assert checker.neighborhoods_geojson == "paris_neighborhoods.geojson"


def test_normalize_neighborhood_name(checker):
    """Test neighborhood name normalization."""
    assert checker._normalize_neighborhood_name("Test Neighborhood") == "test_neighborhood"
    assert checker._normalize_neighborhood_name("Paris Rive Gauche") == "paris_rive_gauche"


def test_get_osm_directory(checker):
    """Test OSM directory path generation."""
    compliant_dir = checker._get_osm_directory("Test Neighborhood", is_compliant=True)
    assert "compliant" in str(compliant_dir)
    assert "test_neighborhood" in str(compliant_dir)

    non_compliant_dir = checker._get_osm_directory("Test Neighborhood", is_compliant=False)
    assert "non_compliant" in str(non_compliant_dir)


def test_get_census_directory(checker):
    """Test Census directory path generation."""
    census_dir = checker._get_census_directory("Test Neighborhood")
    assert "census" in str(census_dir)
    assert "compliant" in str(census_dir)
    assert "test_neighborhood" in str(census_dir)


def test_check_geometry_validity(checker, mock_services_gdf):
    """Test geometry validity checking."""
    all_valid, issues = checker._check_geometry_validity(mock_services_gdf, "Services")
    assert all_valid is True
    assert len(issues) == 0


def test_check_coordinate_bounds(checker, mock_services_gdf):
    """Test coordinate bounds checking."""
    in_bounds, issues = checker._check_coordinate_bounds(mock_services_gdf, "Services")
    assert in_bounds is True
    assert len(issues) == 0


def test_validate_services(checker, mock_services_gdf, tmp_path):
    """Test services validation."""
    services_path = tmp_path / "services.geojson"
    mock_services_gdf.to_file(services_path, driver="GeoJSON")

    result = checker._validate_services(services_path, "test_neighborhood", 0.1)

    assert result["status"] in ["pass", "warn", "fail"]
    assert "statistics" in result
    assert "service_count" in result["statistics"]


def test_validate_services_missing_file(checker, tmp_path):
    """Test services validation with missing file."""
    services_path = tmp_path / "nonexistent.geojson"

    result = checker._validate_services(services_path, "test_neighborhood", 0.1)

    assert result["status"] == "fail"
    assert len(result["issues"]) > 0


def test_validate_buildings(checker, mock_buildings_gdf, tmp_path):
    """Test buildings validation."""
    buildings_path = tmp_path / "buildings.geojson"
    mock_buildings_gdf.to_file(buildings_path, driver="GeoJSON")

    result = checker._validate_buildings(buildings_path, "test_neighborhood", 0.1)

    assert result["status"] in ["pass", "warn", "fail"]
    assert "statistics" in result
    assert "building_count" in result["statistics"]


def test_validate_network(checker, mock_network_graph, tmp_path):
    """Test network validation."""
    import osmnx as ox

    network_path = tmp_path / "network.graphml"
    ox.save_graphml(mock_network_graph, network_path)

    result = checker._validate_network(network_path, "test_neighborhood", 0.1)

    assert result["status"] in ["pass", "warn", "fail"]
    assert "statistics" in result
    assert "node_count" in result["statistics"]
    assert "edge_count" in result["statistics"]


def test_validate_census_data(checker, mock_census_df, tmp_path):
    """Test census data validation."""
    census_path = tmp_path / "census_data.parquet"
    mock_census_df.to_parquet(census_path, index=False)

    result = checker._validate_census_data(census_path, "test_neighborhood")

    assert result["status"] in ["pass", "warn", "fail"]
    assert "statistics" in result


def test_validate_census_data_missing_file(checker, tmp_path):
    """Test census validation with missing file."""
    census_path = tmp_path / "nonexistent.parquet"

    result = checker._validate_census_data(census_path, "test_neighborhood")

    assert result["status"] == "fail"
    assert len(result["issues"]) > 0


def test_check_cross_source_consistency(checker):
    """Test cross-source consistency checking."""
    result = checker._check_cross_source_consistency(
        "test_neighborhood",
        osm_area_km2=1.0,
        census_area_km2=1.05,  # 5% difference
        population_density=20000.0,
        building_density=1000.0,
        service_density=100.0,
        network_coverage_km2=None,
    )

    assert result["status"] in ["pass", "warn", "fail"]


def test_calculate_neighborhood_area(checker):
    """Test neighborhood area calculation."""
    geometry = box(2.35, 48.85, 2.36, 48.86)  # Small box in Paris
    area = checker._calculate_neighborhood_area(geometry)

    assert area > 0
    # The box is approximately 0.01 degrees x 0.01 degrees
    # At Paris latitude, this is roughly 1.1 km x 0.7 km ≈ 0.77 km²
    # But with Web Mercator projection, it may be slightly larger
    assert area < 2.0  # Should be reasonable for this box


@patch("src.data.validation.raw_data_sanity_checker.load_neighborhoods")
def test_validate_neighborhood(mock_load_neighborhoods, checker, tmp_path):
    """Test neighborhood validation."""
    # Mock neighborhood data
    mock_neighborhoods = gpd.GeoDataFrame(
        {
            "name": ["test_neighborhood"],
            "label": ["Compliant"],
            "geometry": [box(2.35, 48.85, 2.36, 48.86)],
        },
        crs="EPSG:4326",
    )
    mock_load_neighborhoods.return_value = mock_neighborhoods

    # Create mock data files
    osm_dir = tmp_path / "raw" / "osm" / "compliant" / "test_neighborhood"
    osm_dir.mkdir(parents=True)

    services_gdf = gpd.GeoDataFrame(
        {
            "name": ["Service 1"],
            "amenity": ["cafe"],
            "category": ["Sustenance"],
            "neighborhood_name": ["test_neighborhood"],
            "geometry": [Point(2.35, 48.85)],
        },
        crs="EPSG:4326",
    )
    services_gdf.to_file(osm_dir / "services.geojson", driver="GeoJSON")

    buildings_gdf = gpd.GeoDataFrame(
        {
            "name": ["Building 1"],
            "building": ["residential"],
            "building:levels": [5],
            "area_m2": [500.0],
            "neighborhood_name": ["test_neighborhood"],
            "geometry": [box(2.35, 48.85, 2.351, 48.851)],
        },
        crs="EPSG:4326",
    )
    buildings_gdf.to_file(osm_dir / "buildings.geojson", driver="GeoJSON")

    import osmnx as ox

    # osmnx expects MultiDiGraph
    G = nx.MultiDiGraph()
    G.add_node(1, x=2.35, y=48.85)
    G.add_node(2, x=2.351, y=48.851)
    G.add_edge(1, 2, length=100.0)
    ox.save_graphml(G, osm_dir / "network.graphml")

    intersections_gdf = gpd.GeoDataFrame(
        {
            "node_id": [1],
            "degree": [3],
            "neighborhood_name": ["test_neighborhood"],
            "geometry": [Point(2.35, 48.85)],
        },
        crs="EPSG:4326",
    )
    intersections_gdf.to_file(osm_dir / "intersections.geojson", driver="GeoJSON")

    pedestrian_gdf = gpd.GeoDataFrame(
        {
            "feature_type": ["pedestrian_way"],
            "name": [""],
            "neighborhood_name": ["test_neighborhood"],
            "geometry": [Point(2.35, 48.85)],
        },
        crs="EPSG:4326",
    )
    pedestrian_gdf.to_file(osm_dir / "pedestrian_infrastructure.geojson", driver="GeoJSON")

    census_dir = tmp_path / "raw" / "census" / "compliant" / "test_neighborhood"
    census_dir.mkdir(parents=True)

    census_df = pd.DataFrame(
        {
            "neighborhood_name": ["test_neighborhood"],
            "population_density": [20000.0],
        }
    )
    census_df.to_parquet(census_dir / "census_data.parquet", index=False)

    # Update checker paths
    checker.data_root = tmp_path

    result = checker.validate_neighborhood("test_neighborhood")

    assert "neighborhood_name" in result
    assert "status" in result
    assert "checks" in result
    assert result["neighborhood_name"] == "test_neighborhood"


def test_validate_neighborhood_not_found(checker):
    """Test validation with non-existent neighborhood."""
    with patch("src.data.validation.raw_data_sanity_checker.load_neighborhoods") as mock_load:
        mock_load.return_value = gpd.GeoDataFrame(
            {"name": ["other"], "label": ["Compliant"], "geometry": [box(0, 0, 1, 1)]},
            crs="EPSG:4326",
        )

        result = checker.validate_neighborhood("nonexistent")

        assert result["status"] == "fail"
        assert len(result["issues"]) > 0


@patch("src.data.validation.raw_data_sanity_checker.load_neighborhoods")
@patch("src.data.validation.raw_data_sanity_checker.get_compliant_neighborhoods")
def test_validate_all_neighborhoods(
    mock_get_compliant, mock_load_neighborhoods, checker
):
    """Test validation of all neighborhoods."""
    mock_neighborhoods = gpd.GeoDataFrame(
        {
            "name": ["test1", "test2"],
            "label": ["Compliant", "Compliant"],
            "geometry": [box(2.35, 48.85, 2.36, 48.86), box(2.36, 48.86, 2.37, 48.87)],
        },
        crs="EPSG:4326",
    )
    mock_load_neighborhoods.return_value = mock_neighborhoods
    mock_get_compliant.return_value = mock_neighborhoods

    with patch.object(checker, "validate_neighborhood") as mock_validate:
        mock_validate.side_effect = [
            {
                "neighborhood_name": "test1",
                "status": "pass",
                "issues": [],
                "warnings": [],
                "checks": {},
            },
            {
                "neighborhood_name": "test2",
                "status": "warn",
                "issues": [],
                "warnings": ["Warning 1"],
                "checks": {},
            },
        ]

        result = checker.validate_all_neighborhoods()

        assert "summary" in result
        assert "neighborhoods" in result
        assert result["summary"]["total_neighborhoods"] == 2
        assert result["summary"]["passed"] == 1
        assert result["summary"]["warnings"] == 1


def test_generate_report(checker, tmp_path):
    """Test report generation."""
    validation_results = {
        "summary": {
            "total_neighborhoods": 1,
            "passed": 1,
            "warnings": 0,
            "failed": 0,
            "total_checks": 5,
            "passed_checks": 5,
            "warning_checks": 0,
            "failed_checks": 0,
        },
        "neighborhoods": [
            {
                "neighborhood_name": "test",
                "status": "pass",
                "issues": [],
                "warnings": [],
            }
        ],
    }

    checker._generate_report(validation_results, tmp_path)

    # Check that files were created
    assert (tmp_path / "sanity_check_report.json").exists()
    assert (tmp_path / "sanity_check_summary.csv").exists()
