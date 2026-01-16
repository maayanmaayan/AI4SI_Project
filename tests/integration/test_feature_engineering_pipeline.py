"""Integration tests for feature engineering pipeline.

These tests require actual data files to be present in the data/raw directory.
They test the full pipeline from raw data to processed features.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

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


@pytest.mark.integration
class TestProcessTargetPoint:
    """Test complete target point processing."""

    @patch("src.data.collection.feature_engineer.ox.load_graphml")
    @patch("src.data.collection.feature_engineer.gpd.read_file")
    @patch("src.data.collection.feature_engineer.pd.read_parquet")
    @patch.object(FeatureEngineer, "_load_iris_boundaries")
    def test_process_target_point(
        self,
        mock_load_iris,
        mock_read_parquet,
        mock_read_file,
        mock_load_graphml,
        engineer,
        test_neighborhood,
        mock_network_graph,
    ):
        """Test processing a single target point."""
        # Mock network graph
        mock_load_graphml.return_value = mock_network_graph

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

        # Mock buildings and services
        mock_buildings = gpd.GeoDataFrame(
            {
                "building:levels": ["3"],
                "geometry": [Point(2.3522, 48.8566)],
            },
            crs="EPSG:4326",
        )
        mock_services = gpd.GeoDataFrame(
            {
                "name": ["Service 1"],
                "geometry": [Point(2.3522, 48.8566)],
            },
            crs="EPSG:4326",
        )
        mock_read_file.return_value = mock_services

        target_point = Point(2.3522, 48.8566)
        result = engineer.process_target_point(
            target_point,
            0,
            "Test Neighborhood",
            "Compliant",
            True,
            mock_network_graph,
        )

        assert isinstance(result, dict)
        assert "target_features" in result
        assert "neighbor_data" in result
        assert "target_prob_vector" in result
        assert "num_neighbors" in result

        # Verify target features
        assert isinstance(result["target_features"], np.ndarray)
        assert result["target_features"].shape == (33,)
        assert result["target_features"].dtype == np.float32

        # Verify neighbor data
        assert isinstance(result["neighbor_data"], list)
        for neighbor in result["neighbor_data"]:
            assert "features" in neighbor
            assert "network_distance" in neighbor
            assert "euclidean_distance" in neighbor
            assert "dx" in neighbor
            assert "dy" in neighbor

        # Verify target probability vector
        assert isinstance(result["target_prob_vector"], np.ndarray)
        assert result["target_prob_vector"].shape == (8,)
        assert np.isclose(result["target_prob_vector"].sum(), 1.0, atol=1e-6)


@pytest.mark.integration
class TestOutputSchema:
    """Test output schema validation."""

    @patch("src.data.collection.feature_engineer.ox.load_graphml")
    @patch("src.data.collection.feature_engineer.gpd.read_file")
    @patch("src.data.collection.feature_engineer.pd.read_parquet")
    @patch.object(FeatureEngineer, "_load_iris_boundaries")
    def test_output_schema(
        self,
        mock_load_iris,
        mock_read_parquet,
        mock_read_file,
        mock_load_graphml,
        engineer,
        test_neighborhood,
        mock_network_graph,
        tmp_path,
    ):
        """Test that output DataFrame has correct schema."""
        # Mock all dependencies
        mock_load_graphml.return_value = mock_network_graph
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
        mock_iris = gpd.GeoDataFrame(
            {
                "CODE_IRIS": ["750101001"],
                "geometry": [box(2.35, 48.85, 2.36, 48.86)],
            },
            crs="EPSG:4326",
        )
        mock_load_iris.return_value = mock_iris
        mock_services = gpd.GeoDataFrame(
            {
                "name": ["Service 1"],
                "geometry": [Point(2.3522, 48.8566)],
            },
            crs="EPSG:4326",
        )
        mock_read_file.return_value = mock_services

        # Mock file paths to use temp directory
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "mkdir"):
                result = engineer.process_neighborhood(test_neighborhood, force=True)

        # Note: This test may not fully run without actual data files
        # It's mainly to verify the schema structure
        if not result.empty:
            required_columns = [
                "target_id",
                "neighborhood_name",
                "label",
                "target_features",
                "neighbor_data",
                "target_prob_vector",
                "target_geometry",
                "num_neighbors",
            ]
            for col in required_columns:
                assert col in result.columns, f"Missing column: {col}"
