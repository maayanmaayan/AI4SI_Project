"""Unit tests for Census loader module.

These tests are intentionally offline:
- No pynsee downloads (mocked)
- No geopandas spatial index requirement (sjoin mocked)
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from src.data.collection.census_loader import CensusLoader


@pytest.fixture
def loader():
    """Create CensusLoader instance for testing."""
    return CensusLoader()


@pytest.fixture
def test_neighborhood_compliant():
    """Create a test compliant neighborhood GeoSeries."""
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
def test_neighborhood_non_compliant():
    """Create a test non-compliant neighborhood GeoSeries."""
    gdf = gpd.GeoDataFrame(
        {
            "name": ["Test Neighborhood Non-Compliant"],
            "label": ["Non-Compliant"],
            "geometry": [box(2.35, 48.85, 2.36, 48.86)],
        },
        crs="EPSG:4326",
    )
    return gdf.iloc[0]


@pytest.fixture
def mock_iris_data():
    """Create mock IRIS census DataFrame."""
    # Minimal columns used by CensusLoader:
    # - IRIS + COM for filtering
    # - a couple RP_ACTRES_IRIS-style columns for aggregation
    return pd.DataFrame(
        {
            "IRIS": ["750101001", "750101002", "930010001"],
            "COM": ["75056", "75056", "93001"],
            "P17_POP1564": [1000, 2000, 1500],
            "C17_ACTOCC15P_VOIT": [100, 200, 50],
            "P17_ACTOCC15P": [200, 400, 100],
        }
    )


@pytest.fixture
def mock_iris_boundaries():
    """Create mock IRIS boundaries GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "CODE_IRIS": ["750101001", "750101002", "930010001"],
            # Some boundary datasets include INSEE_COM used for filtering
            "INSEE_COM": ["75", "75", "93"],
            "geometry": [
                box(2.350, 48.850, 2.351, 48.851),
                box(2.351, 48.851, 2.352, 48.852),
                box(2.352, 48.852, 2.353, 48.853),
            ],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def mock_neighborhoods_gdf():
    """Create mock neighborhoods GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "name": ["Test Neighborhood"],
            "label": ["Compliant"],
            "verification_status": ["verified"],
            "geometry": [box(2.350, 48.850, 2.353, 48.853)],
        },
        crs="EPSG:4326",
    )


class TestInitialization:
    """Test CensusLoader initialization."""

    def test_init_loads_config(self, loader):
        """Test that initialization loads configuration."""
        assert loader.config is not None
        assert loader.cache_results is not None
        assert loader.retry_attempts > 0
        assert loader.retry_delay > 0
        assert loader.paris_geocode == "75056"
        assert loader.iris_level == "IRIS"

    def test_init_defaults(self, loader):
        """Test default configuration values."""
        assert isinstance(loader.cache_results, bool)
        assert isinstance(loader.retry_attempts, int)
        assert isinstance(loader.retry_delay, int)


class TestOutputDirectory:
    """Test output directory creation."""

    def test_get_output_directory_compliant(self, loader):
        """Test output directory for compliant neighborhood."""
        dir_path = loader._get_output_directory("Test Neighborhood")
        assert "compliant" in str(dir_path)
        assert "test_neighborhood" in str(dir_path)
        assert dir_path.exists()

    def test_get_output_directory_normalizes_name(self, loader):
        """Test that neighborhood names are normalized."""
        dir_path = loader._get_output_directory("My Test Neighborhood!")
        assert "my_test_neighborhood" in str(dir_path)
        assert dir_path.exists()

    def test_get_output_directory_creates_parents(self, loader):
        """Test that parent directories are created."""
        dir_path = loader._get_output_directory("Deep/Test/Neighborhood")
        assert dir_path.exists()


class TestCache:
    """Test caching functionality."""

    def test_check_cache_no_file(self, loader, tmp_path):
        """Test cache check returns False when file doesn't exist."""
        assert loader._check_cache(tmp_path) is False

    def test_check_cache_file_exists(self, loader, tmp_path):
        """Test cache check returns True when census file exists."""
        (tmp_path / "census_data.parquet").touch()
        assert loader._check_cache(tmp_path) is True

    def test_check_cache_respects_cache_results(self, loader, tmp_path):
        """Test cache check respects cache_results setting."""
        loader.cache_results = False
        (tmp_path / "census_data.parquet").touch()
        assert loader._check_cache(tmp_path) is False


class TestFetchIrisData:
    """Test IRIS data fetching."""

    @patch("src.data.collection.census_loader.download_file")
    def test_fetch_iris_data_success_filters_paris(self, mock_download_file, loader, mock_iris_data):
        """Should keep only COM starting with '75' when iris_codes is None."""
        mock_download_file.return_value = mock_iris_data

        result = loader._fetch_iris_data()
        assert not result.empty
        assert set(result["COM"].unique()) == {"75056"}
        assert "IRIS" in result.columns

    @patch("src.data.collection.census_loader.download_file")
    def test_fetch_iris_data_filters_by_iris_codes(self, mock_download_file, loader, mock_iris_data):
        """Should filter by explicit IRIS codes when provided."""
        mock_download_file.return_value = mock_iris_data

        result = loader._fetch_iris_data(iris_codes=["750101002"])
        assert not result.empty
        assert result["IRIS"].tolist() == ["750101002"]

    @patch("src.data.collection.census_loader.download_file")
    def test_fetch_iris_data_empty(self, mock_download_file, loader):
        """Test IRIS data fetching with empty result."""
        mock_download_file.return_value = pd.DataFrame()
        result = loader._fetch_iris_data()
        assert result.empty

    @patch("src.data.collection.census_loader.download_file")
    def test_fetch_iris_data_error(self, mock_download_file, loader):
        """Test IRIS data fetching error handling."""
        mock_download_file.side_effect = Exception("API Error")
        result = loader._fetch_iris_data()
        assert result.empty


class TestLoadIrisBoundaries:
    """Test IRIS boundaries loading."""

    @patch("src.data.collection.census_loader.gpd.read_file")
    def test_load_iris_boundaries_success(
        self, mock_read_file, loader, mock_iris_boundaries
    ):
        """Test successful IRIS boundaries loading."""
        mock_read_file.return_value = mock_iris_boundaries
        with patch("src.data.collection.census_loader.Path.exists", return_value=True):
            result = loader._load_iris_boundaries()
        assert not result.empty
        # Loader filters boundaries to Paris (INSEE_COM startswith '75')
        assert len(result) == 2
        assert "CODE_IRIS" in result.columns

    def test_load_iris_boundaries_missing_file(self, loader):
        """Test IRIS boundaries loading when file doesn't exist."""
        with patch("src.data.collection.census_loader.Path.exists", return_value=False):
            result = loader._load_iris_boundaries()
            assert result.empty


class TestSpatialJoin:
    """Test spatial join functionality."""

    @patch("src.data.collection.census_loader.gpd.sjoin")
    def test_match_iris_to_neighborhoods_success(
        self, mock_sjoin, loader, mock_iris_data, mock_iris_boundaries, mock_neighborhoods_gdf
    ):
        """Test successful spatial join (sjoin mocked to avoid spatial-index dependency)."""
        def _fake_sjoin(iris_gdf, neighborhoods_subset, how="inner", predicate="intersects"):
            # Pretend every IRIS unit intersects the test neighborhood
            out = iris_gdf.copy()
            out["name"] = "Test Neighborhood"
            out["index_right"] = 0
            return out

        mock_sjoin.side_effect = _fake_sjoin

        result = loader._match_iris_to_neighborhoods(
            mock_iris_data, mock_iris_boundaries, mock_neighborhoods_gdf
        )
        assert not result.empty
        assert "neighborhood_name" in result.columns
        # Aggregated numeric columns should be present
        assert "P17_POP1564" in result.columns

    def test_match_iris_to_neighborhoods_empty_data(self, loader, mock_iris_boundaries, mock_neighborhoods_gdf):
        """Test spatial join with empty IRIS data."""
        empty_data = pd.DataFrame()
        result = loader._match_iris_to_neighborhoods(
            empty_data, mock_iris_boundaries, mock_neighborhoods_gdf
        )
        assert result.empty

    def test_match_iris_to_neighborhoods_empty_boundaries(
        self, loader, mock_iris_data, mock_neighborhoods_gdf
    ):
        """Test spatial join with empty boundaries."""
        empty_boundaries = gpd.GeoDataFrame()
        result = loader._match_iris_to_neighborhoods(
            mock_iris_data, empty_boundaries, mock_neighborhoods_gdf
        )
        assert result.empty


class TestFeatureExtraction:
    """Test demographic feature extraction."""

    def test_extract_demographic_features_car_ownership_from_percentage(self, loader):
        """Car commute ratio should be calculated from C17_ACTOCC15P_VOIT / P17_ACTOCC15P."""
        matched_data = pd.DataFrame(
            {
                "neighborhood_name": ["Test Neighborhood"],
                "PCT_MEN_VOIT": [20.0],  # percent
                "P17_MEN": [100.0],  # provide denominator column so detection passes
                "C17_ACTOCC15P_VOIT": [100.0],
                "P17_ACTOCC15P": [200.0],
            }
        )

        # Avoid calling external APIs for age estimation
        with patch.object(loader, "_fetch_commune_age_data", return_value=pd.DataFrame()):
            result = loader._extract_demographic_features(matched_data, neighborhoods=None)

        assert not result.empty
        # Note: car_ownership_rate is no longer calculated (code commented out)
        # Only car_commute_ratio is calculated
        assert result.loc[0, "car_commute_ratio"] == pytest.approx(0.50)

    def test_extract_demographic_features_car_ownership_from_counts(self, loader):
        """Car commute ratio should be calculated from C17_ACTOCC15P_VOIT / P17_ACTOCC15P."""
        matched_data = pd.DataFrame(
            {
                "neighborhood_name": ["Test Neighborhood"],
                "P17_MEN": [100.0],
                "C17_MEN_VOIT1": [30.0],
                "C17_MEN_VOIT2P": [10.0],
                "C17_ACTOCC15P_VOIT": [100.0],
                "P17_ACTOCC15P": [200.0],
            }
        )

        with patch.object(loader, "_fetch_commune_age_data", return_value=pd.DataFrame()):
            result = loader._extract_demographic_features(matched_data, neighborhoods=None)

        # Note: car_ownership_rate is no longer calculated (code commented out)
        # Only car_commute_ratio is calculated
        assert result.loc[0, "car_commute_ratio"] == pytest.approx(0.50)

    def test_extract_demographic_features_empty(self, loader):
        """Test feature extraction with empty data."""
        empty_data = pd.DataFrame()
        result = loader._extract_demographic_features(empty_data)
        assert result.empty


class TestLoadNeighborhood:
    """Test main load_neighborhood_census method."""

    @patch("src.data.collection.census_loader.load_neighborhoods")
    @patch("src.data.collection.census_loader.CensusLoader._extract_demographic_features")
    @patch("src.data.collection.census_loader.CensusLoader._match_iris_to_neighborhoods")
    @patch("src.data.collection.census_loader.CensusLoader._load_iris_boundaries")
    @patch("src.data.collection.census_loader.CensusLoader._fetch_logement_car_ownership")
    @patch("src.data.collection.census_loader.CensusLoader._fetch_filosofi_income")
    @patch("src.data.collection.census_loader.CensusLoader._fetch_iris_data")
    def test_load_neighborhood_census_success(
        self,
        mock_fetch,
        mock_fetch_filosofi,
        mock_fetch_logement,
        mock_boundaries,
        mock_match,
        mock_extract,
        mock_load_neighborhoods,
        loader,
        test_neighborhood_compliant,
        mock_iris_data,
        mock_iris_boundaries,
        mock_neighborhoods_gdf,
        tmp_path,
    ):
        """Test successful census loading for compliant neighborhood."""
        # Setup mocks
        mock_fetch.return_value = mock_iris_data
        mock_fetch_filosofi.return_value = pd.DataFrame()
        mock_fetch_logement.return_value = pd.DataFrame()
        mock_boundaries.return_value = mock_iris_boundaries
        mock_load_neighborhoods.return_value = mock_neighborhoods_gdf

        matched_df = pd.DataFrame(
            {
                "neighborhood_name": ["Test Neighborhood"],
                "P17_POP1564": [1000],
                "P17_ACTOCC15P": [500],
            }
        )
        mock_match.return_value = matched_df

        features_df = pd.DataFrame(
            {
                "neighborhood_name": ["Test Neighborhood"],
                "population_density": [1000.0],
                "ses_index": [0.5],
            }
        )
        mock_extract.return_value = features_df

        # Mock output directory
        with patch.object(loader, "_get_output_directory", return_value=tmp_path):
            with patch.object(loader, "_check_cache", return_value=False):
                with patch("src.data.collection.census_loader.save_dataframe") as _:
                    result = loader.load_neighborhood_census(test_neighborhood_compliant)

        assert result["status"] == "success"
        assert result["neighborhood_name"] == "Test Neighborhood"
        assert result["label"] == "Compliant"

    def test_load_neighborhood_census_skips_non_compliant(
        self, loader, test_neighborhood_non_compliant
    ):
        """Test that non-compliant neighborhoods are skipped."""
        result = loader.load_neighborhood_census(test_neighborhood_non_compliant)

        assert result["status"] == "skipped"
        assert result["label"] == "Non-Compliant"

    @patch("src.data.collection.census_loader.CensusLoader._check_cache")
    def test_load_neighborhood_census_respects_cache(
        self, mock_cache, loader, test_neighborhood_compliant
    ):
        """Test that cached data is respected."""
        mock_cache.return_value = True

        result = loader.load_neighborhood_census(test_neighborhood_compliant)

        assert result["status"] == "cached"

    @patch("src.data.collection.census_loader.load_neighborhoods")
    @patch("src.data.collection.census_loader.CensusLoader._extract_demographic_features")
    @patch("src.data.collection.census_loader.CensusLoader._match_iris_to_neighborhoods")
    @patch("src.data.collection.census_loader.CensusLoader._load_iris_boundaries")
    @patch("src.data.collection.census_loader.CensusLoader._fetch_iris_data")
    def test_load_neighborhood_census_retries_on_failure(
        self,
        mock_fetch,
        mock_boundaries,
        mock_match,
        mock_extract,
        mock_load_neighborhoods,
        loader,
        test_neighborhood_compliant,
        tmp_path,
    ):
        """Test retry logic on failure."""
        # Setup mocks to fail
        mock_fetch.side_effect = Exception("API Error")
        # Must return non-empty boundaries so loader reaches _fetch_iris_data
        mock_boundaries.return_value = gpd.GeoDataFrame(
            {"CODE_IRIS": ["750101001"], "geometry": [box(2.35, 48.85, 2.351, 48.851)]},
            crs="EPSG:4326",
        )
        mock_load_neighborhoods.return_value = gpd.GeoDataFrame()

        with patch.object(loader, "_get_output_directory", return_value=tmp_path):
            with patch.object(loader, "_check_cache", return_value=False):
                result = loader.load_neighborhood_census(test_neighborhood_compliant)

        assert result["status"] == "failed"
        assert result["error"] is not None
        # Should have attempted retries
        assert mock_fetch.call_count == loader.retry_attempts


class TestLoadAllNeighborhoods:
    """Test load_all_neighborhoods method."""

    @patch("src.data.collection.census_loader.load_neighborhoods")
    @patch("src.data.collection.census_loader.get_compliant_neighborhoods")
    @patch("src.data.collection.census_loader.CensusLoader.load_neighborhood_census")
    def test_load_all_neighborhoods_processes_compliant_only(
        self,
        mock_load_neighborhood,
        mock_get_compliant,
        mock_load_neighborhoods,
        loader,
        mock_neighborhoods_gdf,
    ):
        """Test that only compliant neighborhoods are processed."""
        # Setup mocks
        mock_load_neighborhoods.return_value = mock_neighborhoods_gdf
        compliant_gdf = mock_neighborhoods_gdf.copy()
        mock_get_compliant.return_value = compliant_gdf

        mock_load_neighborhood.return_value = {
            "status": "success",
            "neighborhood_name": "Test Neighborhood",
            "label": "Compliant",
        }

        summary = loader.load_all_neighborhoods()

        assert summary["total_neighborhoods"] == len(compliant_gdf)
        assert summary["compliant_count"] == len(compliant_gdf)
        # Should have called load_neighborhood_census for each compliant neighborhood
        assert mock_load_neighborhood.call_count == len(compliant_gdf)

    @patch("src.data.collection.census_loader.load_neighborhoods")
    @patch("src.data.collection.census_loader.get_compliant_neighborhoods")
    @patch("src.data.collection.census_loader.CensusLoader.load_neighborhood_census")
    def test_load_all_neighborhoods_summary(
        self,
        mock_load_neighborhood,
        mock_get_compliant,
        mock_load_neighborhoods,
        loader,
        mock_neighborhoods_gdf,
    ):
        """Test summary statistics."""
        # Create a GeoDataFrame with 3 neighborhoods
        compliant_gdf = gpd.GeoDataFrame(
            {
                "name": ["Test 1", "Test 2", "Test 3"],
                "label": ["Compliant", "Compliant", "Compliant"],
                "verification_status": ["verified", "verified", "verified"],
                "geometry": [
                    box(2.350, 48.850, 2.351, 48.851),
                    box(2.351, 48.851, 2.352, 48.852),
                    box(2.352, 48.852, 2.353, 48.853),
                ],
            },
            crs="EPSG:4326",
        )
        mock_load_neighborhoods.return_value = mock_neighborhoods_gdf
        mock_get_compliant.return_value = compliant_gdf

        # Mix of success, cached, and failed
        mock_load_neighborhood.side_effect = [
            {"status": "success", "neighborhood_name": "Test 1", "label": "Compliant"},
            {"status": "cached", "neighborhood_name": "Test 2", "label": "Compliant"},
            {"status": "failed", "neighborhood_name": "Test 3", "label": "Compliant", "error": "Error"},
        ]

        summary = loader.load_all_neighborhoods()

        assert summary["successful_count"] == 1
        assert summary["cached_count"] == 1
        assert summary["failed_count"] == 1
        assert summary["total_neighborhoods"] == 3
