"""Census data loader using pynsee library.

This module provides functionality to fetch IRIS-level census data for Paris
neighborhoods using the pynsee library. Data is organized by compliance status
and cached to avoid re-fetching.

The loader processes ONLY compliant neighborhoods (label == "Compliant") following
the exemplar-based learning approach where training data comes exclusively from
compliant neighborhoods.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd
from pynsee.localdata import get_local_data

from src.utils.config import get_config
from src.utils.logging import get_logger
from src.utils.helpers import (
    ensure_dir_exists,
    get_compliant_neighborhoods,
    load_neighborhoods,
    save_dataframe,
)

logger = get_logger(__name__)


class CensusLoader:
    """Load census data for Paris neighborhoods using pynsee.

    This class provides methods to fetch IRIS-level census data, match it to
    neighborhood boundaries, and extract required demographic features.
    Data is organized by compliance status and cached to avoid re-fetching.

    Attributes:
        config: Configuration dictionary loaded from config.yaml.
        cache_results: Whether to skip fetching if data already exists.
        retry_attempts: Number of retries on failure.
        retry_delay: Seconds between retries.
        paris_geocode: INSEE code for Paris (default: "75056").
        iris_level: Geographic level (default: "IRIS").
        min_data_warning: Minimum IRIS units before warning.

    Example:
        >>> loader = CensusLoader()
        >>> neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
        >>> result = loader.load_neighborhood_census(neighborhoods.iloc[0])
        >>> print(f"Status: {result['status']}")
    """

    def __init__(self) -> None:
        """Initialize CensusLoader with configuration."""
        self.config = get_config()
        census_config = self.config.get("census_extraction", {})

        self.cache_results: bool = census_config.get("cache_results", True)
        self.retry_attempts: int = census_config.get("retry_attempts", 3)
        self.retry_delay: int = census_config.get("retry_delay", 5)
        self.paris_geocode: str = census_config.get("paris_geocode", "75056")
        self.iris_level: str = census_config.get("iris_level", "IRIS")
        self.min_data_warning: int = census_config.get("min_data_warning", 10)

        logger.info("CensusLoader initialized")

    def _get_output_directory(self, neighborhood_name: str) -> Path:
        """Get output directory for compliant neighborhood.

        Creates and returns the output directory path for a compliant neighborhood.
        Directories are organized as:
        - `data/raw/census/compliant/{neighborhood_name}/` for compliant neighborhoods

        Note: This method only handles compliant neighborhoods. Non-compliant
        neighborhoods are never processed.

        Args:
            neighborhood_name: Name of the neighborhood.

        Returns:
            Path object pointing to the output directory (created if needed).

        Example:
            >>> dir_path = loader._get_output_directory("Test Neighborhood")
            >>> assert "compliant" in str(dir_path)
            >>> assert dir_path.exists()
        """
        # Normalize neighborhood name (lowercase, replace spaces with underscores)
        normalized_name = neighborhood_name.lower().replace(" ", "_")

        # Base directory for compliant neighborhoods only
        base_dir = Path("data/raw/census") / "compliant" / normalized_name

        # Create directory if it doesn't exist
        ensure_dir_exists(str(base_dir))

        return base_dir

    def _check_cache(self, output_dir: Path) -> bool:
        """Check if census data already exists (cached).

        Checks for existence of the census data file:
        - census_data.parquet

        Args:
            output_dir: Path to output directory.

        Returns:
            True if census data file exists, False otherwise.

        Example:
            >>> output_dir = Path("data/raw/census/compliant/test")
            >>> is_cached = loader._check_cache(output_dir)
            >>> print(f"Data cached: {is_cached}")
        """
        if not self.cache_results:
            return False

        census_file = output_dir / "census_data.parquet"
        return census_file.exists()

    def _fetch_iris_data(self) -> pd.DataFrame:
        """Fetch IRIS-level census data for Paris using pynsee.

        Fetches demographic data at IRIS level for Paris using the pynsee library.
        This method attempts to fetch data with common demographic variables.

        Returns:
            DataFrame with IRIS codes and demographic variables. Returns empty
            DataFrame if fetching fails.

        Example:
            >>> iris_data = loader._fetch_iris_data()
            >>> print(f"Fetched {len(iris_data)} IRIS units")
        """
        logger.info(
            f"Fetching IRIS-level census data for Paris (geocode: {self.paris_geocode})"
        )

        try:
            # Fetch IRIS-level data for Paris
            iris_data = get_local_data(
                nivgeo=self.iris_level,
                geocodes=[self.paris_geocode],
            )

            if iris_data is None or len(iris_data) == 0:
                logger.warning("No IRIS data returned from pynsee")
                return pd.DataFrame()

            logger.info(f"Successfully fetched {len(iris_data)} IRIS units")
            return iris_data

        except Exception as e:
            logger.error(f"Error fetching IRIS data: {e}")
            return pd.DataFrame()

    def _load_iris_boundaries(self) -> gpd.GeoDataFrame:
        """Load IRIS boundary GeoJSON file.

        Loads IRIS boundary geometries from a GeoJSON file. The file should
        contain IRIS geometries with a code column matching IRIS codes from
        census data.

        Note: IRIS boundaries may need to be downloaded separately from INSEE/IGN
        or data.gouv.fr if pynsee doesn't provide geometries.

        Returns:
            GeoDataFrame with IRIS geometries and codes. Returns empty GeoDataFrame
            if loading fails.

        Example:
            >>> iris_boundaries = loader._load_iris_boundaries()
            >>> print(f"Loaded {len(iris_boundaries)} IRIS boundaries")
        """
        logger.info("Loading IRIS boundaries")

        # Check if IRIS boundaries file exists in data directory
        iris_boundaries_path = Path("data/raw/iris_boundaries.geojson")

        if not iris_boundaries_path.exists():
            logger.warning(
                f"IRIS boundaries file not found at {iris_boundaries_path}. "
                "Spatial matching may not work. Consider downloading IRIS boundaries "
                "from INSEE or data.gouv.fr"
            )
            return gpd.GeoDataFrame()

        try:
            iris_boundaries = gpd.read_file(iris_boundaries_path)
            logger.info(f"Successfully loaded {len(iris_boundaries)} IRIS boundaries")
            return iris_boundaries
        except Exception as e:
            logger.error(f"Error loading IRIS boundaries: {e}")
            return gpd.GeoDataFrame()

    def _match_iris_to_neighborhoods(
        self,
        iris_data: pd.DataFrame,
        iris_boundaries: gpd.GeoDataFrame,
        neighborhoods: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """Spatial join to match IRIS census data to neighborhood boundaries.

        Performs a spatial join to match IRIS census units to neighborhood
        boundaries. Aggregates IRIS data per neighborhood using weighted averages
        or sums based on feature type.

        Args:
            iris_data: DataFrame with IRIS codes and demographic data.
            iris_boundaries: GeoDataFrame with IRIS geometries and codes.
            neighborhoods: GeoDataFrame with neighborhood boundaries.

        Returns:
            DataFrame with neighborhood_name and demographic features aggregated
            from matching IRIS units.

        Example:
            >>> matched = loader._match_iris_to_neighborhoods(
            ...     iris_data, iris_boundaries, neighborhoods
            ... )
            >>> print(f"Matched {len(matched)} neighborhoods")
        """
        if iris_data.empty or iris_boundaries.empty:
            logger.warning("Empty IRIS data or boundaries, cannot perform spatial join")
            return pd.DataFrame()

        # Identify IRIS code column (common names: CODE_IRIS, IRIS, code_iris, etc.)
        iris_code_col = None
        for col in iris_data.columns:
            if "iris" in col.lower() or "code" in col.lower():
                iris_code_col = col
                break

        if iris_code_col is None:
            logger.error("Could not find IRIS code column in census data")
            return pd.DataFrame()

        # Identify IRIS code column in boundaries
        boundary_code_col = None
        for col in iris_boundaries.columns:
            if "iris" in col.lower() or "code" in col.lower():
                boundary_code_col = col
                break

        if boundary_code_col is None:
            logger.error("Could not find IRIS code column in boundaries")
            return pd.DataFrame()

        # Merge IRIS data with boundaries on IRIS code
        iris_gdf = iris_boundaries.merge(
            iris_data,
            left_on=boundary_code_col,
            right_on=iris_code_col,
            how="inner",
        )

        if iris_gdf.empty:
            logger.warning("No matching IRIS codes between data and boundaries")
            return pd.DataFrame()

        # Ensure CRS matches
        if neighborhoods.crs != iris_gdf.crs:
            iris_gdf = iris_gdf.to_crs(neighborhoods.crs)

        # Perform spatial join
        matched = gpd.sjoin(
            iris_gdf,
            neighborhoods[["name", "geometry"]],
            how="inner",
            predicate="intersects",
        )

        if matched.empty:
            logger.warning("No IRIS units intersect neighborhood boundaries")
            return pd.DataFrame()

        # Aggregate IRIS data per neighborhood
        # For now, use simple mean aggregation (can be improved with area-weighted)
        numeric_cols = matched.select_dtypes(include=["number"]).columns
        numeric_cols = [col for col in numeric_cols if col not in ["index_right"]]

        aggregated = matched.groupby("name")[numeric_cols].mean().reset_index()
        aggregated.rename(columns={"name": "neighborhood_name"}, inplace=True)

        logger.info(f"Matched IRIS data to {len(aggregated)} neighborhoods")
        return aggregated

    def _extract_demographic_features(self, matched_data: pd.DataFrame) -> pd.DataFrame:
        """Extract required demographic features from matched census data.

        Extracts the following features from raw INSEE variables:
        - Population density (population / area)
        - SES index (from income data, may need calculation)
        - Car ownership (households with car / total households)
        - Children per capita (children / total population)
        - Household size (average persons per household)
        - Elderly ratio (elderly population / total population)

        Args:
            matched_data: DataFrame with neighborhood_name and raw census variables.

        Returns:
            DataFrame with neighborhood_name and extracted feature columns.

        Example:
            >>> features = loader._extract_demographic_features(matched_data)
            >>> print(features.columns)
        """
        if matched_data.empty:
            logger.warning("Empty matched data, cannot extract features")
            return pd.DataFrame()

        features_df = matched_data[["neighborhood_name"]].copy()

        # Map common INSEE variable names to our features
        # Note: Actual variable names may vary - these are common patterns
        # Population-related variables
        pop_cols = [
            col
            for col in matched_data.columns
            if "pop" in col.lower() or "population" in col.lower()
        ]
        household_cols = [
            col
            for col in matched_data.columns
            if "menage" in col.lower() or "household" in col.lower()
        ]
        income_cols = [
            col
            for col in matched_data.columns
            if "revenu" in col.lower() or "income" in col.lower()
        ]
        car_cols = [
            col
            for col in matched_data.columns
            if "voiture" in col.lower() or "car" in col.lower()
        ]

        # 1. Population density
        if pop_cols and "area" in matched_data.columns.str.lower().str.lower():
            total_pop = matched_data[pop_cols[0]] if pop_cols else None
            area = matched_data["area"] if "area" in matched_data.columns else None
            if total_pop is not None and area is not None:
                features_df["population_density"] = total_pop / (
                    area / 1_000_000
                )  # per km²
            else:
                features_df["population_density"] = None
        else:
            features_df["population_density"] = None

        # 2. SES index (from income data)
        if income_cols:
            # Use median income as proxy for SES index (normalized)
            median_income = matched_data[income_cols[0]]
            features_df["ses_index"] = (
                (median_income - median_income.mean()) / median_income.std()
                if median_income.std() > 0
                else 0
            )
        else:
            features_df["ses_index"] = None

        # 3. Car ownership
        if car_cols and household_cols:
            cars = matched_data[car_cols[0]]
            households = matched_data[household_cols[0]]
            features_df["car_ownership"] = (
                (cars / households) if households.sum() > 0 else 0
            )
        else:
            features_df["car_ownership"] = None

        # 4. Children per capita
        child_cols = [
            col
            for col in matched_data.columns
            if "enfant" in col.lower() or "child" in col.lower()
        ]
        if child_cols and pop_cols:
            children = matched_data[child_cols[0]]
            total_pop = matched_data[pop_cols[0]]
            features_df["children_per_capita"] = (
                (children / total_pop) if total_pop.sum() > 0 else 0
            )
        else:
            features_df["children_per_capita"] = None

        # 5. Household size
        if pop_cols and household_cols:
            total_pop = matched_data[pop_cols[0]]
            households = matched_data[household_cols[0]]
            features_df["household_size"] = (
                (total_pop / households) if households.sum() > 0 else 0
            )
        else:
            features_df["household_size"] = None

        # 6. Elderly ratio
        elderly_cols = [
            col
            for col in matched_data.columns
            if "age" in col.lower()
            and ("65" in col or "75" in col or "senior" in col.lower())
        ]
        if elderly_cols and pop_cols:
            elderly = matched_data[elderly_cols[0]]
            total_pop = matched_data[pop_cols[0]]
            features_df["elderly_ratio"] = (
                (elderly / total_pop) if total_pop.sum() > 0 else 0
            )
        else:
            features_df["elderly_ratio"] = None

        logger.info(
            f"Extracted demographic features for {len(features_df)} neighborhoods"
        )
        return features_df

    def load_neighborhood_census(
        self, neighborhood_row: gpd.GeoSeries, force: bool = False
    ) -> Dict[str, Any]:
        """Load census data for one neighborhood.

        Main method to load census data for a single neighborhood. Only processes
        neighborhoods with label == "Compliant". Non-compliant neighborhoods are
        skipped immediately.

        Args:
            neighborhood_row: GeoSeries row from neighborhoods GeoDataFrame with
                columns: name, label, geometry, etc.
            force: If True, re-fetch even if cached data exists.

        Returns:
            Dictionary with loading summary:
            - status: "success", "failed", "cached", or "skipped"
            - neighborhood_name: Name of the neighborhood
            - label: Neighborhood label
            - features_count: Number of features extracted
            - error: Error message if failed

        Example:
            >>> from src.utils.helpers import load_neighborhoods
            >>> neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
            >>> result = loader.load_neighborhood_census(neighborhoods.iloc[0])
            >>> print(f"Status: {result['status']}")
        """
        neighborhood_name = neighborhood_row.get("name", "unknown")
        label = neighborhood_row.get("label", "")

        logger.info(f"Starting census data loading for {neighborhood_name}")

        # CRITICAL: Only process compliant neighborhoods
        if label != "Compliant":
            logger.info(
                f"Skipping {neighborhood_name} - not compliant (label: {label})"
            )
            return {
                "status": "skipped",
                "neighborhood_name": neighborhood_name,
                "label": label,
            }

        # Get output directory
        output_dir = self._get_output_directory(neighborhood_name)

        # Check cache
        if not force and self._check_cache(output_dir):
            logger.info(
                f"Skipping {neighborhood_name} - data already exists "
                "(use force=True to re-fetch)"
            )
            return {
                "status": "cached",
                "neighborhood_name": neighborhood_name,
                "label": label,
            }

        # Initialize result dictionary
        result = {
            "status": "success",
            "neighborhood_name": neighborhood_name,
            "label": label,
            "features_count": 0,
            "error": None,
        }

        # Load with retries
        for attempt in range(1, self.retry_attempts + 1):
            try:
                # Fetch IRIS data
                iris_data = self._fetch_iris_data()
                if iris_data.empty:
                    raise ValueError("No IRIS data fetched")

                # Load IRIS boundaries
                iris_boundaries = self._load_iris_boundaries()

                # Load all neighborhoods for spatial join
                neighborhoods = load_neighborhoods(
                    self.config.get("paths", {}).get(
                        "neighborhoods_geojson", "paris_neighborhoods.geojson"
                    )
                )

                # Match IRIS to neighborhoods
                matched_data = self._match_iris_to_neighborhoods(
                    iris_data, iris_boundaries, neighborhoods
                )

                if matched_data.empty:
                    raise ValueError("No IRIS data matched to neighborhoods")

                # Filter to current neighborhood
                neighborhood_data = matched_data[
                    matched_data["neighborhood_name"] == neighborhood_name
                ]

                if neighborhood_data.empty:
                    raise ValueError(f"No IRIS data matched to {neighborhood_name}")

                # Extract demographic features
                features_df = self._extract_demographic_features(neighborhood_data)
                result["features_count"] = (
                    len(features_df.columns) - 1
                )  # Exclude neighborhood_name

                # Save census data
                census_path = output_dir / "census_data.parquet"
                save_dataframe(features_df, str(census_path), format="parquet")

                logger.info(
                    f"Successfully loaded census data for {neighborhood_name} "
                    f"({result['features_count']} features)"
                )
                break

            except Exception as e:
                error_msg = f"Attempt {attempt}/{self.retry_attempts} failed: {str(e)}"
                logger.error(
                    f"Error loading census data for {neighborhood_name}: {error_msg}"
                )

                if attempt < self.retry_attempts:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    result["status"] = "failed"
                    result["error"] = str(e)
                    logger.error(
                        f"Failed to load census data for {neighborhood_name} after "
                        f"{self.retry_attempts} attempts"
                    )

        return result

    def load_all_neighborhoods(
        self, geojson_path: Optional[str] = None, force: bool = False
    ) -> Dict[str, Any]:
        """Load census data for compliant neighborhoods only.

        Loads neighborhoods from GeoJSON, filters to compliant neighborhoods only,
        and processes each neighborhood sequentially. Non-compliant neighborhoods
        are completely excluded from processing.

        Args:
            geojson_path: Path to neighborhoods GeoJSON file. If None, uses
                default from config.
            force: If True, re-fetch even if cached data exists.

        Returns:
            Dictionary with overall summary:
            - total_neighborhoods: Total compliant neighborhoods processed
            - successful_count: Number of successful loadings
            - failed_count: Number of failed loadings
            - cached_count: Number of cached (skipped) loadings
            - compliant_count: Same as total_neighborhoods (for consistency)

        Example:
            >>> loader = CensusLoader()
            >>> summary = loader.load_all_neighborhoods()
            >>> print(f"Loaded {summary['successful_count']} neighborhoods")
        """
        # Load neighborhoods
        if geojson_path is None:
            geojson_path = self.config.get("paths", {}).get(
                "neighborhoods_geojson", "paris_neighborhoods.geojson"
            )

        logger.info(f"Loading neighborhoods from {geojson_path}")
        neighborhoods = load_neighborhoods(geojson_path)

        # Filter to compliant neighborhoods only
        compliant = get_compliant_neighborhoods(neighborhoods)

        logger.info(f"Found {len(compliant)} compliant neighborhoods to process")

        # Process compliant neighborhoods
        metadata = []
        successful_count = 0
        failed_count = 0
        cached_count = 0

        for idx, row in compliant.iterrows():
            neighborhood_name = row.get("name", f"neighborhood_{idx}")
            logger.info(f"Processing {idx + 1}/{len(compliant)}: {neighborhood_name}")

            result = self.load_neighborhood_census(row, force=force)
            result["timestamp"] = time.time()

            if result["status"] == "success":
                successful_count += 1
            elif result["status"] == "cached":
                cached_count += 1
            else:
                failed_count += 1

            metadata.append(result)

        # Create summary
        summary = {
            "total_neighborhoods": len(compliant),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "cached_count": cached_count,
            "compliant_count": len(compliant),
        }

        logger.info(
            f"Census loading complete: {successful_count} successful, "
            f"{failed_count} failed, {cached_count} cached"
        )

        return summary
