"""Unit tests for FeatureEngineer module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

from src.data.collection.feature_engineer import FeatureEngineer


@pytest.fixture
def engineer():
    """Create FeatureEngineer instance for testing."""
    return FeatureEngineer()


@pytest.fixture
def test_neighborhood():
    """Create a test neighborhood GeoSeries."""
    gdf = gpd.GeoDataFrame(
        {
            "name": ["Test Neighborhood"],
            "label": ["Compliant"],
            "verification_status": ["verified"],
            "geometry": [box(2.35, 48.85, 2.36, 48.86)],
        },
        crs="EPSG:4326",
    )
    return gdf.iloc[0]


@pytest.fixture
def test_point():
    """Create a test point in Paris."""
    return Point(2.3522, 48.8566)


@pytest.fixture
def mock_network_graph():
    """Create mock network graph."""
    G = nx.MultiDiGraph()
    # Add nodes with coordinates
    G.add_node(1, x=2.3522, y=48.8566)
    G.add_node(2, x=2.3532, y=48.8576)
    G.add_node(3, x=2.3512, y=48.8556)
    # Add edges with length attribute
    G.add_edge(1, 2, length=100.0)
    G.add_edge(2, 3, length=150.0)
    G.add_edge(3, 1, length=200.0)
    # Set CRS attribute (required by OSMnx)
    G.graph["crs"] = "EPSG:4326"
    return G


class TestGenerateTargetPoints:
    """Test target point generation."""

    def test_generate_target_points(self, engineer, test_neighborhood):
        """Test target point generation creates points inside polygon."""
        points = engineer.generate_target_points(test_neighborhood)
        assert isinstance(points, gpd.GeoDataFrame)
        assert len(points) > 0
        assert "target_id" in points.columns
        assert "neighborhood_name" in points.columns
        assert "label" in points.columns
        assert "geometry" in points.columns

        # Verify all points are inside polygon
        polygon = test_neighborhood.geometry
        for _, row in points.iterrows():
            assert polygon.contains(row.geometry)

    def test_generate_target_points_empty_polygon(self, engineer):
        """Test target point generation with empty polygon returns empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(
            {
                "name": ["Empty"],
                "label": ["Compliant"],
                "geometry": [Polygon()],
            },
            crs="EPSG:4326",
        )
        points = engineer.generate_target_points(empty_gdf.iloc[0])
        assert isinstance(points, gpd.GeoDataFrame)
        assert len(points) == 0


class TestGenerateGridCells:
    """Test grid cell generation."""

    def test_generate_grid_cells(self, engineer, test_point):
        """Test grid cell generation creates cells around point."""
        cells = engineer.generate_grid_cells(test_point)
        assert isinstance(cells, list)
        assert len(cells) > 0
        assert all(isinstance(cell, Point) for cell in cells)

    def test_generate_grid_cells_coverage(self, engineer, test_point):
        """Test grid cells cover expected area."""
        cells = engineer.generate_grid_cells(test_point)
        # Should have cells within walk_15min_radius_meters
        # Approximate: (2 * radius / cell_size)^2 cells
        expected_min = 100  # At least some cells
        assert len(cells) >= expected_min


class TestPrefilterByEuclidean:
    """Test Euclidean pre-filtering."""

    def test_prefilter_by_euclidean(self, engineer, test_point):
        """Test Euclidean pre-filtering reduces cell count."""
        cells = engineer.generate_grid_cells(test_point)
        filtered = engineer._prefilter_by_euclidean(test_point, cells)
        assert len(filtered) <= len(cells)
        assert len(filtered) > 0  # Should keep some cells

    def test_prefilter_by_euclidean_empty(self, engineer, test_point):
        """Test pre-filtering with empty list returns empty list."""
        filtered = engineer._prefilter_by_euclidean(test_point, [])
        assert filtered == []


class TestComputeNetworkDistance:
    """Test network distance calculation."""

    def test_compute_network_distance(self, engineer, test_point, mock_network_graph):
        """Test network distance calculation between two points."""
        point1 = Point(2.3522, 48.8566)
        point2 = Point(2.3532, 48.8576)
        distance = engineer.compute_network_distance(
            point1, point2, mock_network_graph
        )
        assert isinstance(distance, float)
        assert distance >= 0
        assert distance != float("inf")  # Should find a path

    def test_compute_network_distance_empty_graph(self, engineer, test_point):
        """Test network distance with empty graph returns inf."""
        empty_graph = nx.MultiDiGraph()
        distance = engineer.compute_network_distance(
            test_point, Point(2.36, 48.86), empty_graph
        )
        assert distance == float("inf")


class TestFilterByNetworkDistance:
    """Test network distance filtering."""

    def test_filter_by_network_distance(
        self, engineer, test_point, mock_network_graph
    ):
        """Test filtering by network distance."""
        cells = engineer.generate_grid_cells(test_point)
        # Use a larger threshold for testing
        neighbors = engineer.filter_by_network_distance(
            test_point, cells, mock_network_graph, max_distance_meters=5000
        )
        assert isinstance(neighbors, list)
        # Should have some neighbors (depending on graph coverage)
        for neighbor in neighbors:
            assert "cell" in neighbor
            assert "network_distance" in neighbor
            assert "euclidean_distance" in neighbor
            assert "dx" in neighbor
            assert "dy" in neighbor


class TestComputeDemographicFeatures:
    """Test demographic feature computation."""

    @patch("src.data.collection.feature_engineer.pd.read_parquet")
    @patch.object(FeatureEngineer, "_load_iris_boundaries")
    def test_compute_demographic_features(
        self, mock_load_iris, mock_read_parquet, engineer, test_point
    ):
        """Test demographic feature computation."""
        # Mock census data
        mock_census = pd.DataFrame(
            {
                "IRIS": ["750101001"],
                "population_density": [5000.0],
                "ses_index": [0.5],
                "car_commute_ratio": [0.3],
                "children_per_capita": [0.2],
                "elderly_ratio": [0.15],
                "unemployment_rate": [0.1],
                "student_ratio": [0.25],
                "walking_ratio": [0.4],
                "cycling_ratio": [0.1],
                "public_transport_ratio": [0.3],
                "two_wheelers_ratio": [0.05],
                "retired_ratio": [0.2],
                "permanent_employment_ratio": [0.8],
                "temporary_employment_ratio": [0.2],
                "median_income": [30000.0],
                "poverty_rate": [0.1],
                "working_age_ratio": [0.65],
            }
        )
        mock_read_parquet.return_value = mock_census

        # Mock IRIS boundaries
        mock_iris = gpd.GeoDataFrame(
            {
                "CODE_IRIS": ["750101001"],
                "geometry": [box(2.35, 48.85, 2.36, 48.86)],
            },
            crs="EPSG:4326",
        )
        mock_load_iris.return_value = mock_iris

        features = engineer.compute_demographic_features(
            test_point, "Test Neighborhood", True
        )
        assert isinstance(features, np.ndarray)
        assert features.shape == (17,)
        assert features.dtype == np.float32


class TestComputeBuiltFormFeatures:
    """Test built form feature computation."""

    @patch("src.data.collection.feature_engineer.gpd.read_file")
    def test_compute_built_form_features(
        self, mock_read_file, engineer, test_point
    ):
        """Test built form feature computation."""
        # Mock buildings data
        mock_buildings = gpd.GeoDataFrame(
            {
                "building:levels": ["3", "2"],
                "geometry": [
                    Point(2.3522, 48.8566),
                    Point(2.3523, 48.8567),
                ],
            },
            crs="EPSG:4326",
        )
        mock_read_file.return_value = mock_buildings

        features = engineer.compute_built_form_features(
            test_point, "Test Neighborhood", True
        )
        assert isinstance(features, np.ndarray)
        assert features.shape == (4,)
        assert features.dtype == np.float32


class TestComputeServiceFeatures:
    """Test service feature computation."""

    @patch("src.data.collection.feature_engineer.gpd.read_file")
    def test_compute_service_features(
        self, mock_read_file, engineer, test_point, mock_network_graph
    ):
        """Test service feature computation."""
        # Mock services data
        mock_services = gpd.GeoDataFrame(
            {
                "name": ["Service 1"],
                "geometry": [Point(2.3522, 48.8566)],
            },
            crs="EPSG:4326",
        )
        mock_read_file.return_value = mock_services

        features = engineer.compute_service_features(
            test_point, "Test Neighborhood", True, mock_network_graph
        )
        assert isinstance(features, np.ndarray)
        assert features.shape == (8,)
        assert features.dtype == np.float32


class TestComputeWalkabilityFeatures:
    """Test walkability feature computation."""

    def test_compute_walkability_features(
        self, engineer, test_point, mock_network_graph
    ):
        """Test walkability feature computation."""
        features = engineer.compute_walkability_features(
            test_point, "Test Neighborhood", True, mock_network_graph
        )
        assert isinstance(features, np.ndarray)
        assert features.shape == (4,)
        assert features.dtype == np.float32


class TestComputePointFeatures:
    """Test point feature computation."""

    @patch.object(FeatureEngineer, "compute_demographic_features")
    @patch.object(FeatureEngineer, "compute_built_form_features")
    @patch.object(FeatureEngineer, "compute_service_features")
    @patch.object(FeatureEngineer, "compute_walkability_features")
    def test_compute_point_features(
        self,
        mock_walkability,
        mock_services,
        mock_built_form,
        mock_demographic,
        engineer,
        test_point,
        mock_network_graph,
    ):
        """Test point feature computation orchestrates all features."""
        # Mock feature arrays
        mock_demographic.return_value = np.zeros(17, dtype=np.float32)
        mock_built_form.return_value = np.zeros(4, dtype=np.float32)
        mock_services.return_value = np.zeros(8, dtype=np.float32)
        mock_walkability.return_value = np.zeros(4, dtype=np.float32)

        features = engineer.compute_point_features(
            test_point, "Test Neighborhood", True, mock_network_graph
        )
        assert isinstance(features, np.ndarray)
        assert features.shape == (33,)
        assert features.dtype == np.float32


class TestComputeTargetProbabilityVector:
    """Test target probability vector computation."""

    @patch("src.data.collection.feature_engineer.gpd.read_file")
    def test_compute_target_probability_vector(
        self, mock_read_file, engineer, test_point, mock_network_graph
    ):
        """Test target probability vector computation."""
        # Mock services data
        mock_services = gpd.GeoDataFrame(
            {
                "name": ["Service 1"],
                "geometry": [Point(2.3522, 48.8566)],
            },
            crs="EPSG:4326",
        )
        mock_read_file.return_value = mock_services

        prob_vector = engineer.compute_target_probability_vector(
            test_point, "Test Neighborhood", True, mock_network_graph
        )
        assert isinstance(prob_vector, np.ndarray)
        assert prob_vector.shape == (8,)
        assert prob_vector.dtype == np.float32
        # Probabilities should sum to 1.0
        assert np.isclose(prob_vector.sum(), 1.0, atol=1e-6)
        # All probabilities should be in [0, 1]
        assert np.all(prob_vector >= 0)
        assert np.all(prob_vector <= 1)
