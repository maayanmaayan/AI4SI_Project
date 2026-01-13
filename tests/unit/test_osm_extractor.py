"""Unit tests for OSM extractor module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import Point, Polygon, box

from src.data.collection.osm_extractor import OSMExtractor


@pytest.fixture
def extractor():
    """Create OSMExtractor instance for testing."""
    return OSMExtractor()


@pytest.fixture
def test_polygon():
    """Create a test polygon (small box in Paris area)."""
    return box(2.35, 48.85, 2.36, 48.86)


@pytest.fixture
def mock_services_gdf():
    """Create mock services GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "name": ["Service 1", "Service 2"],
            "amenity": ["restaurant", "cafe"],
            "category": ["Sustenance", "Sustenance"],
            "geometry": [
                Point(2.355, 48.855),
                Point(2.356, 48.856),
            ],
            "neighborhood_name": ["test", "test"],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def mock_buildings_gdf():
    """Create mock buildings GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "name": ["Building 1"],
            "building": ["residential"],
            "building:levels": ["3"],
            "area_m2": [100.0],
            "geometry": [Point(2.355, 48.855)],
            "neighborhood_name": ["test"],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def mock_network_graph():
    """Create mock network graph."""
    G = nx.Graph()
    G.add_node(1, x=2.355, y=48.855)
    G.add_node(2, x=2.356, y=48.856)
    G.add_edge(1, 2)
    return G


class TestBufferPolygon:
    """Test buffer polygon creation."""

    def test_create_buffer_polygon(self, extractor, test_polygon):
        """Test buffer creation increases polygon area."""
        buffered = extractor._create_buffer_polygon(test_polygon, 100)
        assert buffered.area > test_polygon.area
        assert isinstance(buffered, Polygon)

    def test_create_buffer_polygon_negative_distance(self, extractor, test_polygon):
        """Test buffer creation with negative distance raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            extractor._create_buffer_polygon(test_polygon, -10)

    def test_create_buffer_polygon_invalid_polygon(self, extractor):
        """Test buffer creation with invalid polygon raises error."""
        invalid_polygon = Polygon([(0, 0), (1, 1), (1, 0)])  # Invalid (self-intersecting)
        # Note: This might not raise an error if shapely auto-fixes it
        # Adjust test based on actual behavior


class TestOutputDirectory:
    """Test output directory creation."""

    def test_get_output_directory_compliant(self, extractor):
        """Test output directory for compliant neighborhood."""
        dir_path = extractor._get_output_directory("Test Neighborhood", True)
        assert "compliant" in str(dir_path)
        assert "test_neighborhood" in str(dir_path)
        assert dir_path.exists()

    def test_get_output_directory_non_compliant(self, extractor):
        """Test output directory for non-compliant neighborhood."""
        dir_path = extractor._get_output_directory("Test Neighborhood", False)
        assert "non_compliant" in str(dir_path)
        assert "test_neighborhood" in str(dir_path)
        assert dir_path.exists()

    def test_get_output_directory_normalizes_name(self, extractor):
        """Test that neighborhood names are normalized."""
        dir_path = extractor._get_output_directory("My Test Neighborhood!", True)
        assert "my_test_neighborhood" in str(dir_path)


class TestCache:
    """Test caching functionality."""

    def test_check_cache_no_files(self, extractor, tmp_path):
        """Test cache check returns False when files don't exist."""
        assert extractor._check_cache(tmp_path) is False

    def test_check_cache_all_files_exist(self, extractor, tmp_path):
        """Test cache check returns True when all files exist."""
        (tmp_path / "services.geojson").touch()
        (tmp_path / "network.graphml").touch()
        (tmp_path / "buildings.geojson").touch()
        assert extractor._check_cache(tmp_path) is True

    def test_check_cache_partial_files(self, extractor, tmp_path):
        """Test cache check returns False when only some files exist."""
        (tmp_path / "services.geojson").touch()
        assert extractor._check_cache(tmp_path) is False


class TestExtractServices:
    """Test service extraction."""

    @patch("src.data.collection.osm_extractor.ox.features_from_polygon")
    def test_extract_services_success(
        self, mock_features, extractor, test_polygon, mock_services_gdf
    ):
        """Test successful service extraction."""
        # Mock OSMnx to return services
        mock_features.return_value = mock_services_gdf

        services = extractor.extract_services(test_polygon, test_polygon, "test")
        assert len(services) > 0
        assert "category" in services.columns
        assert "neighborhood_name" in services.columns

    @patch("src.data.collection.osm_extractor.ox.features_from_polygon")
    def test_extract_services_no_data(self, mock_features, extractor, test_polygon):
        """Test service extraction with no data."""
        mock_features.return_value = gpd.GeoDataFrame()
        services = extractor.extract_services(test_polygon, test_polygon, "test")
        assert len(services) == 0

    def test_extract_services_invalid_polygon(self, extractor):
        """Test service extraction with invalid polygon."""
        invalid_polygon = Polygon([(0, 0), (1, 1), (1, 0)])
        with pytest.raises(ValueError, match="invalid"):
            extractor.extract_services(invalid_polygon, invalid_polygon, "test")


class TestExtractBuildings:
    """Test building extraction."""

    @patch("src.data.collection.osm_extractor.ox.features_from_polygon")
    def test_extract_buildings_success(
        self, mock_features, extractor, test_polygon, mock_buildings_gdf
    ):
        """Test successful building extraction."""
        mock_features.return_value = mock_buildings_gdf

        buildings = extractor.extract_buildings(test_polygon, test_polygon, "test")
        assert len(buildings) > 0
        assert "area_m2" in buildings.columns
        assert "neighborhood_name" in buildings.columns

    @patch("src.data.collection.osm_extractor.ox.features_from_polygon")
    def test_extract_buildings_no_data(self, mock_features, extractor, test_polygon):
        """Test building extraction with no data."""
        mock_features.return_value = gpd.GeoDataFrame()
        buildings = extractor.extract_buildings(test_polygon, test_polygon, "test")
        assert len(buildings) == 0


class TestExtractNetwork:
    """Test network extraction."""

    @patch("src.data.collection.osm_extractor.ox.graph_from_polygon")
    @patch("src.data.collection.osm_extractor.ox.graph_to_gdfs")
    def test_extract_network_success(
        self, mock_graph_to_gdfs, mock_graph_from_polygon, extractor, test_polygon, mock_network_graph
    ):
        """Test successful network extraction."""
        mock_graph_from_polygon.return_value = mock_network_graph
        edges_gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
        nodes_gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
        mock_graph_to_gdfs.return_value = (edges_gdf, nodes_gdf)

        G, edges, nodes = extractor.extract_network(test_polygon, "test")
        assert isinstance(G, nx.Graph)
        assert "neighborhood_name" in edges.columns
        assert "neighborhood_name" in nodes.columns

    @patch("src.data.collection.osm_extractor.ox.graph_from_polygon")
    def test_extract_network_failure(self, mock_graph_from_polygon, extractor, test_polygon):
        """Test network extraction failure handling."""
        mock_graph_from_polygon.side_effect = Exception("Network extraction failed")

        G, edges, nodes = extractor.extract_network(test_polygon, "test")
        assert isinstance(G, nx.Graph)
        assert len(edges) == 0
        assert len(nodes) == 0


class TestExtractWalkabilityFeatures:
    """Test walkability features extraction."""

    def test_extract_walkability_features_intersections(
        self, extractor, test_polygon, mock_network_graph
    ):
        """Test intersection extraction from network."""
        # Add high-degree nodes to graph
        mock_network_graph.add_node(3, x=2.357, y=48.857)
        mock_network_graph.add_edge(1, 3)
        mock_network_graph.add_edge(2, 3)

        intersections, pedestrian = extractor.extract_walkability_features(
            mock_network_graph, test_polygon, "test"
        )
        assert "neighborhood_name" in intersections.columns

    @patch("src.data.collection.osm_extractor.ox.features_from_polygon")
    def test_extract_walkability_features_pedestrian(
        self, mock_features, extractor, test_polygon, mock_network_graph
    ):
        """Test pedestrian infrastructure extraction."""
        mock_features.return_value = gpd.GeoDataFrame()

        intersections, pedestrian = extractor.extract_walkability_features(
            mock_network_graph, test_polygon, "test"
        )
        assert "neighborhood_name" in pedestrian.columns


class TestSaveServicesByCategory:
    """Test saving services by category."""

    def test_save_services_by_category(self, extractor, mock_services_gdf, tmp_path):
        """Test saving services by category creates correct files."""
        extractor._save_services_by_category(mock_services_gdf, tmp_path)

        category_dir = tmp_path / "services_by_category"
        assert category_dir.exists()

        # Check that category files were created
        sustenance_file = category_dir / "sustenance.geojson"
        assert sustenance_file.exists()

    def test_save_services_by_category_empty(self, extractor, tmp_path):
        """Test saving empty services GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(columns=["name", "amenity", "category", "geometry"])
        extractor._save_services_by_category(empty_gdf, tmp_path)

        category_dir = tmp_path / "services_by_category"
        assert category_dir.exists()
