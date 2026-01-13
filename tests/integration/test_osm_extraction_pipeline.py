"""Integration tests for OSM extraction pipeline.

These tests require internet connection and use real OSM data.
Mark with @pytest.mark.integration to skip in CI if needed.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from src.data.collection.osm_extractor import OSMExtractor
from src.utils.helpers import load_neighborhoods


@pytest.mark.integration
class TestOSMExtractionPipeline:
    """Integration tests for full OSM extraction pipeline."""

    @pytest.fixture
    def extractor(self):
        """Create OSMExtractor instance."""
        return OSMExtractor()

    @pytest.fixture
    def small_test_polygon(self):
        """Create a small test polygon in Paris (to minimize extraction time)."""
        # Small box in central Paris
        return box(2.35, 48.85, 2.36, 48.86)

    def test_extract_services_integration(self, extractor, small_test_polygon):
        """Test service extraction with real OSM data."""
        services = extractor.extract_services(
            small_test_polygon, small_test_polygon, "test_integration"
        )

        # Should find some services in central Paris
        assert isinstance(services, gpd.GeoDataFrame)
        assert "category" in services.columns
        assert "neighborhood_name" in services.columns

    def test_extract_buildings_integration(self, extractor, small_test_polygon):
        """Test building extraction with real OSM data."""
        buildings = extractor.extract_buildings(
            small_test_polygon, small_test_polygon, "test_integration"
        )

        assert isinstance(buildings, gpd.GeoDataFrame)
        assert "area_m2" in buildings.columns
        assert "neighborhood_name" in buildings.columns

    def test_extract_network_integration(self, extractor, small_test_polygon):
        """Test network extraction with real OSM data."""
        G, edges, nodes = extractor.extract_network(small_test_polygon, "test_integration")

        assert len(nodes) > 0
        assert len(edges) > 0
        assert "neighborhood_name" in edges.columns
        assert "neighborhood_name" in nodes.columns

    def test_extract_walkability_features_integration(
        self, extractor, small_test_polygon
    ):
        """Test walkability features extraction with real OSM data."""
        G, edges, nodes = extractor.extract_network(small_test_polygon, "test_integration")
        intersections, pedestrian = extractor.extract_walkability_features(
            G, small_test_polygon, "test_integration"
        )

        assert isinstance(intersections, gpd.GeoDataFrame)
        assert isinstance(pedestrian, gpd.GeoDataFrame)
        assert "neighborhood_name" in intersections.columns
        assert "neighborhood_name" in pedestrian.columns

    def test_extract_neighborhood_integration(self, extractor):
        """Test full neighborhood extraction with real OSM data."""
        # Create a small test neighborhood
        test_geometry = box(2.35, 48.85, 2.36, 48.86)
        neighborhood_row = gpd.GeoSeries(
            {
                "name": "Test Integration Neighborhood",
                "label": "Compliant",
                "geometry": test_geometry,
            }
        )

        result = extractor.extract_neighborhood(neighborhood_row, force=True)

        assert result["status"] in ["success", "failed", "cached"]
        assert "neighborhood_name" in result
        assert "services_count" in result

        # Verify output files were created if successful
        if result["status"] == "success":
            output_dir = extractor._get_output_directory(
                result["neighborhood_name"], True
            )
            assert (output_dir / "services.geojson").exists()
            assert (output_dir / "buildings.geojson").exists()
            assert (output_dir / "network.graphml").exists()

    def test_cache_behavior(self, extractor):
        """Test that caching works correctly."""
        # Create a small test neighborhood
        test_geometry = box(2.35, 48.85, 2.36, 48.86)
        neighborhood_row = gpd.GeoSeries(
            {
                "name": "Test Cache Neighborhood",
                "label": "Compliant",
                "geometry": test_geometry,
            }
        )

        # First extraction
        result1 = extractor.extract_neighborhood(neighborhood_row, force=True)
        assert result1["status"] == "success"

        # Second extraction should be cached
        result2 = extractor.extract_neighborhood(neighborhood_row, force=False)
        assert result2["status"] == "cached"

        # Force re-extraction
        result3 = extractor.extract_neighborhood(neighborhood_row, force=True)
        assert result3["status"] == "success"

    def test_output_directory_structure(self, extractor):
        """Test that output directory structure is correct."""
        test_geometry = box(2.35, 48.85, 2.36, 48.86)

        # Test compliant neighborhood
        compliant_row = gpd.GeoSeries(
            {
                "name": "Compliant Test",
                "label": "Compliant",
                "geometry": test_geometry,
            }
        )
        result = extractor.extract_neighborhood(compliant_row, force=True)
        if result["status"] == "success":
            output_dir = extractor._get_output_directory("Compliant Test", True)
            assert "compliant" in str(output_dir)
            assert (output_dir / "services_by_category").exists()

        # Test non-compliant neighborhood
        non_compliant_row = gpd.GeoSeries(
            {
                "name": "Non-Compliant Test",
                "label": "Non-Compliant",
                "geometry": test_geometry,
            }
        )
        result = extractor.extract_neighborhood(non_compliant_row, force=True)
        if result["status"] == "success":
            output_dir = extractor._get_output_directory("Non-Compliant Test", False)
            assert "non_compliant" in str(output_dir)
