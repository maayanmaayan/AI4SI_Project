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
from pynsee import download_file
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

    def _fetch_iris_data(self, iris_codes: list = None) -> pd.DataFrame:
        """Fetch IRIS-level census data for Paris using pynsee download_file.

        Downloads IRIS-level census data using download_file('RP_ACTRES_IRIS') which
        provides actual values (unlike get_local_data API which returns NaN).
        Filters data for Paris IRIS codes.

        Args:
            iris_codes: List of IRIS codes to filter data for. If None, will filter
                for all Paris IRIS codes (COM starting with '75').

        Returns:
            DataFrame with IRIS codes and demographic variables. Returns empty
            DataFrame if fetching fails.

        Example:
            >>> iris_data = loader._fetch_iris_data(iris_codes=['751145620', '751145622'])
            >>> print(f"Fetched {len(iris_data)} IRIS units")
        """
        logger.info(
            "Downloading IRIS-level census data using download_file('RP_ACTRES_IRIS')"
        )

        try:
            # Download IRIS-level census data (2017 data, but matches 2021 IRIS structure)
            iris_data = download_file("RP_ACTRES_IRIS")

            if iris_data is None or len(iris_data) == 0:
                logger.warning("No IRIS data returned from download_file")
                return pd.DataFrame()

            logger.info(f"Downloaded {len(iris_data)} IRIS units from RP_ACTRES_IRIS")

            # Filter for Paris if iris_codes provided, otherwise filter by COM column
            if iris_codes is not None:
                # Filter by provided IRIS codes
                if "IRIS" in iris_data.columns:
                    iris_data = iris_data[iris_data["IRIS"].isin(iris_codes)]
                    logger.info(f"Filtered to {len(iris_data)} IRIS units matching provided codes")
                else:
                    logger.warning("IRIS column not found in downloaded data")
                    return pd.DataFrame()
            else:
                # Filter for Paris (COM starting with '75')
                if "COM" in iris_data.columns:
                    iris_data = iris_data[iris_data["COM"].str.startswith("75", na=False)]
                    logger.info(f"Filtered to {len(iris_data)} Paris IRIS units")
                else:
                    logger.warning("COM column not found in downloaded data")
                    return pd.DataFrame()

            if iris_data.empty:
                logger.warning("No IRIS data after filtering")
                return pd.DataFrame()

            logger.info(f"Successfully fetched {len(iris_data)} IRIS units")
            return iris_data

        except Exception as e:
            logger.exception(f"Error fetching IRIS data: {e}")
            return pd.DataFrame()

    def _fetch_commune_age_data(self) -> pd.DataFrame:
        """Fetch commune-level age data for Paris to supplement IRIS data.

        Fetches total population at commune level to estimate children and elderly ratios.
        Since RP_ACTRES_IRIS only has working-age (15-64), we use commune-level total
        population to estimate children (<15) and elderly (65+) ratios.

        Returns:
            DataFrame with commune-level population data. Returns empty DataFrame if failed.

        Example:
            >>> commune_data = loader._fetch_commune_age_data()
            >>> print(f"Total population: {commune_data.get('total_population', 'N/A')}")
        """
        logger.info("Fetching commune-level population data for age ratios")

        try:
            # Get total population at commune level
            commune_data = get_local_data(
                dataset_version="GEO2021RP2018",
                variables="STOCD",
                nivgeo="COM",
                geocodes=[self.paris_geocode],
            )

            if commune_data.empty:
                logger.warning("No commune-level data returned")
                return pd.DataFrame()

            # Extract total population (UNIT='POP', STOCD='ENS')
            pop_data = commune_data[
                (commune_data["UNIT"] == "POP") & (commune_data["STOCD"] == "ENS")
            ]

            if pop_data.empty:
                logger.warning("No total population data found in commune data")
                return pd.DataFrame()

            total_population = pop_data["OBS_VALUE"].iloc[0] if len(pop_data) > 0 else None

            if total_population is None or pd.isna(total_population):
                logger.warning("Total population is NaN")
                return pd.DataFrame()

            logger.info(f"Fetched commune-level total population: {total_population:,.0f}")
            return pd.DataFrame({"total_population": [total_population]})

        except Exception as e:
            logger.error(f"Error fetching commune age data: {e}")
            return pd.DataFrame()

    def _fetch_filosofi_income(self, iris_codes: list = None) -> pd.DataFrame:
        """Fetch FILOSOFI income data at IRIS level for Paris.

        Downloads FILOSOFI income dataset which provides income indicators including
        median income, poverty rates, and income deciles at IRIS level.

        Args:
            iris_codes: List of IRIS codes to filter data for. If None, will filter
                for all Paris IRIS codes (COM starting with '75').

        Returns:
            DataFrame with IRIS codes and income variables. Returns empty DataFrame if failed.

        Example:
            >>> income_data = loader._fetch_filosofi_income(iris_codes=['751145620'])
            >>> print(f"Fetched {len(income_data)} IRIS units")
        """
        logger.info("Downloading FILOSOFI income data at IRIS level")

        try:
            # Try FILOSOFI_DISP_IRIS_2017 first (most recent available)
            filosofi_data = download_file("FILOSOFI_DISP_IRIS_2017")

            if filosofi_data is None or len(filosofi_data) == 0:
                # Fallback to FILOSOFI_DEC_IRIS
                logger.info("Trying FILOSOFI_DEC_IRIS as fallback")
                filosofi_data = download_file("FILOSOFI_DEC_IRIS")

            if filosofi_data is None or len(filosofi_data) == 0:
                logger.warning("No FILOSOFI data returned")
                return pd.DataFrame()

            logger.info(f"Downloaded {len(filosofi_data)} IRIS units from FILOSOFI")

            # Filter for Paris
            if iris_codes is not None:
                # Identify IRIS code column
                iris_col = None
                for col in ["IRIS", "CODEGEO", "CODE_IRIS"]:
                    if col in filosofi_data.columns:
                        iris_col = col
                        break

                if iris_col:
                    filosofi_data = filosofi_data[filosofi_data[iris_col].isin(iris_codes)]
                    logger.info(f"Filtered to {len(filosofi_data)} IRIS units matching provided codes")
                else:
                    logger.warning("IRIS code column not found in FILOSOFI data")
                    return pd.DataFrame()
            else:
                # Filter for Paris by COM column
                if "COM" in filosofi_data.columns:
                    filosofi_data = filosofi_data[filosofi_data["COM"].str.startswith("75", na=False)]
                    logger.info(f"Filtered to {len(filosofi_data)} Paris IRIS units")
                else:
                    logger.warning("COM column not found in FILOSOFI data")
                    return pd.DataFrame()

            if filosofi_data.empty:
                logger.warning("No FILOSOFI data after filtering")
                return pd.DataFrame()

            logger.info(f"Successfully fetched {len(filosofi_data)} FILOSOFI IRIS units")
            return filosofi_data

        except Exception as e:
            logger.exception(f"Error fetching FILOSOFI income data: {e}")
            return pd.DataFrame()

    def _fetch_logement_car_ownership(self, iris_codes: list = None) -> pd.DataFrame:
        """Fetch car ownership data from RP_LOGEMENT dataset at IRIS level.

        Downloads RP_LOGEMENT housing dataset which contains household-level car ownership
        data. This provides actual car ownership rates (households with car / total households),
        which is different from car commute ratio.

        Args:
            iris_codes: List of IRIS codes to filter data for. If None, will filter
                for all Paris IRIS codes (COM starting with '75').

        Returns:
            DataFrame with IRIS codes and car ownership variables. Returns empty DataFrame if failed.

        Example:
            >>> car_data = loader._fetch_logement_car_ownership(iris_codes=['751145620'])
            >>> print(f"Fetched {len(car_data)} IRIS units")
        """
        logger.info("Downloading RP_LOGEMENT car ownership data at IRIS level")
        logger.info("Note: RP_LOGEMENT file is large (~373MB), download may take several minutes")

        try:
            # Download RP_LOGEMENT_2017 (matches our RP_ACTRES_IRIS year)
            # NOTE: RP_LOGEMENT_2017 is MICRODATA (one row per household), not aggregated
            # We need to download the full file and aggregate by IRIS ourselves
            logger.info("Downloading full RP_LOGEMENT_2017 (microdata file, ~373MB)")
            logger.info("This is a large file and may take 5-10 minutes to download")
            logement_data = download_file("RP_LOGEMENT_2017")

            if logement_data is None or len(logement_data) == 0:
                logger.warning("No RP_LOGEMENT data returned after download attempt")
                return pd.DataFrame()

            logger.info(f"Downloaded {len(logement_data)} household records from RP_LOGEMENT (microdata)")

            # RP_LOGEMENT_2017 uses 'COMMUNE' column (not 'COM') and 'IRIS' column
            # Filter for Paris first (COMMUNE starting with '75')
            if "COMMUNE" in logement_data.columns:
                logement_data = logement_data[logement_data["COMMUNE"].astype(str).str.startswith("75", na=False)]
                logger.info(f"Filtered to {len(logement_data)} Paris household records")
            else:
                logger.warning("COMMUNE column not found in RP_LOGEMENT data")
                return pd.DataFrame()

            if logement_data.empty:
                logger.warning("No RP_LOGEMENT data after filtering for Paris")
                return pd.DataFrame()

            # Filter by IRIS codes if provided
            if iris_codes is not None:
                if "IRIS" in logement_data.columns:
                    logement_data = logement_data[logement_data["IRIS"].isin(iris_codes)]
                    logger.info(f"Filtered to {len(logement_data)} household records matching provided IRIS codes")
                else:
                    logger.warning("IRIS column not found in RP_LOGEMENT data")
                    return pd.DataFrame()

            # Aggregate microdata by IRIS to get car ownership rates
            # VOIT column contains: "0" (no car), "1" (1 car), "2" (2 cars), "3" (3+ cars), "X" (confidential)
            if "IRIS" not in logement_data.columns or "VOIT" not in logement_data.columns:
                logger.warning("IRIS or VOIT column missing in RP_LOGEMENT data")
                return pd.DataFrame()

            logger.info("Aggregating microdata by IRIS to calculate car ownership rates...")
            
            # Convert VOIT to numeric (X becomes NaN)
            logement_data["VOIT_numeric"] = pd.to_numeric(logement_data["VOIT"], errors="coerce")
            
            # Count households per IRIS
            total_households = logement_data.groupby("IRIS").size().reset_index(name="P17_MEN")
            
            # Count households with car (VOIT >= 1, excluding NaN and "X")
            households_with_car = (
                logement_data[logement_data["VOIT_numeric"] >= 1]
                .groupby("IRIS")
                .size()
                .reset_index(name="C17_MEN_VOIT")
            )
            
            # Merge and calculate rate
            aggregated = total_households.merge(households_with_car, on="IRIS", how="left")
            aggregated["C17_MEN_VOIT"] = aggregated["C17_MEN_VOIT"].fillna(0)
            aggregated["car_ownership_rate"] = aggregated["C17_MEN_VOIT"] / aggregated["P17_MEN"]
            
            # Also add counts for 1 car, 2+ cars if needed
            households_1car = (
                logement_data[logement_data["VOIT_numeric"] == 1]
                .groupby("IRIS")
                .size()
                .reset_index(name="C17_MEN_VOIT1")
            )
            households_2pcar = (
                logement_data[logement_data["VOIT_numeric"] >= 2]
                .groupby("IRIS")
                .size()
                .reset_index(name="C17_MEN_VOIT2P")
            )
            
            aggregated = aggregated.merge(households_1car, on="IRIS", how="left")
            aggregated = aggregated.merge(households_2pcar, on="IRIS", how="left")
            aggregated["C17_MEN_VOIT1"] = aggregated["C17_MEN_VOIT1"].fillna(0)
            aggregated["C17_MEN_VOIT2P"] = aggregated["C17_MEN_VOIT2P"].fillna(0)
            
            logger.info(f"Successfully aggregated {len(aggregated)} IRIS units with car ownership data")
            return aggregated

        except Exception as e:
            logger.exception(f"Error fetching RP_LOGEMENT car ownership data: {e}")
            return pd.DataFrame()

    def _load_iris_boundaries(self) -> gpd.GeoDataFrame:
        """Load IRIS boundary file (GeoJSON or Shapefile).

        Loads IRIS boundary geometries from a GeoJSON or Shapefile. The file should
        contain IRIS geometries with a code column matching IRIS codes from
        census data.

        Tries multiple common locations:
        - data/raw/iris_boundaries.geojson
        - data/raw/census/iris-2013-01-01/iris-2013-01-01.shp
        - data/raw/iris_boundaries.shp

        Returns:
            GeoDataFrame with IRIS geometries and codes. Returns empty GeoDataFrame
            if loading fails.

        Example:
            >>> iris_boundaries = loader._load_iris_boundaries()
            >>> print(f"Loaded {len(iris_boundaries)} IRIS boundaries")
        """
        logger.info("Loading IRIS boundaries")

        # Try common locations (prioritize 2021 boundaries)
        possible_paths = [
            Path(
                "data/raw/census/iris-2021-01-01/CONTOURS-IRIS_2-1__SHP__FRA_2021-01-01/"
                "CONTOURS-IRIS/1_DONNEES_LIVRAISON_2021-06-00217/"
                "CONTOURS-IRIS_2-1_SHP_LAMB93_FXX-2021/CONTOURS-IRIS.shp"
            ),
            Path("data/raw/iris_boundaries.geojson"),
            Path("data/raw/census/iris-2013-01-01/iris-2013-01-01.shp"),
            Path("data/raw/iris_boundaries.shp"),
        ]

        iris_boundaries_path = None
        for path in possible_paths:
            if path.exists():
                iris_boundaries_path = path
                break

        if iris_boundaries_path is None:
            logger.warning(
                "IRIS boundaries file not found. Tried: "
                f"{[str(p) for p in possible_paths]}. "
                "Spatial matching may not work. Consider downloading IRIS boundaries "
                "from INSEE or data.gouv.fr"
            )
            return gpd.GeoDataFrame()

        try:
            iris_boundaries = gpd.read_file(iris_boundaries_path)
            file_type = "Shapefile" if iris_boundaries_path.suffix == ".shp" else "GeoJSON"
            logger.info(
                f"Successfully loaded {len(iris_boundaries)} IRIS boundaries "
                f"from {file_type}: {iris_boundaries_path}"
            )
            
            # Filter for Paris if needed (IRIS codes starting with 75)
            # Handle different column names for IRIS codes
            if "CODE_IRIS" in iris_boundaries.columns:
                # 2021 format
                paris_boundaries = iris_boundaries[
                    iris_boundaries["INSEE_COM"].str.startswith("75", na=False)
                ]
                logger.info(f"Filtered to {len(paris_boundaries)} Paris IRIS boundaries (2021 format)")
                return paris_boundaries
            elif "DCOMIRIS" in iris_boundaries.columns:
                # 2013 format
                paris_boundaries = iris_boundaries[
                    iris_boundaries["DCOMIRIS"].str.startswith("75", na=False)
                ]
                logger.info(f"Filtered to {len(paris_boundaries)} Paris IRIS boundaries (2013 format)")
                return paris_boundaries
            elif "DEPCOM" in iris_boundaries.columns:
                paris_boundaries = iris_boundaries[
                    iris_boundaries["DEPCOM"].str.startswith("75", na=False)
                ]
                logger.info(f"Filtered to {len(paris_boundaries)} Paris IRIS boundaries")
                return paris_boundaries
            
            return iris_boundaries
        except Exception as e:
            logger.error(f"Error loading IRIS boundaries: {e}")
            return gpd.GeoDataFrame()

    def _match_iris_to_neighborhoods(
        self,
        iris_data: pd.DataFrame,
        iris_boundaries: gpd.GeoDataFrame,
        neighborhoods: gpd.GeoDataFrame,
        filosofi_data: pd.DataFrame = None,
        logement_data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Spatial join to match IRIS census data to neighborhood boundaries.

        Performs a spatial join to match IRIS census units to neighborhood
        boundaries. Aggregates IRIS data per neighborhood using weighted averages
        or sums based on feature type. Optionally merges FILOSOFI income data and
        RP_LOGEMENT car ownership data.

        Args:
            iris_data: DataFrame with IRIS codes and demographic data.
            iris_boundaries: GeoDataFrame with IRIS geometries and codes.
            neighborhoods: GeoDataFrame with neighborhood boundaries.
            filosofi_data: Optional DataFrame with FILOSOFI income data.
            logement_data: Optional DataFrame with RP_LOGEMENT car ownership data.

        Returns:
            DataFrame with neighborhood_name and demographic features aggregated
            from matching IRIS units.

        Example:
            >>> matched = loader._match_iris_to_neighborhoods(
            ...     iris_data, iris_boundaries, neighborhoods, filosofi_data, logement_data
            ... )
            >>> print(f"Matched {len(matched)} neighborhoods")
        """
        if iris_data.empty or iris_boundaries.empty:
            logger.warning("Empty IRIS data or boundaries, cannot perform spatial join")
            return pd.DataFrame()

        # Identify IRIS code column in census data (RP_ACTRES_IRIS uses 'IRIS')
        iris_code_col = None
        if "IRIS" in iris_data.columns:
            iris_code_col = "IRIS"
        else:
            for col in iris_data.columns:
                if "iris" in col.lower() or "code" in col.lower():
                    iris_code_col = col
                    break

        if iris_code_col is None:
            logger.error("Could not find IRIS code column in census data")
            return pd.DataFrame()

        # Identify IRIS code column in boundaries
        boundary_code_col = None
        # Prioritize CODE_IRIS (2021 format) over DCOMIRIS (2013 format)
        if "CODE_IRIS" in iris_boundaries.columns:
            boundary_code_col = "CODE_IRIS"
        elif "DCOMIRIS" in iris_boundaries.columns:
            boundary_code_col = "DCOMIRIS"
        else:
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

        # Ensure CRS matches - convert IRIS boundaries to match neighborhoods CRS
        if iris_gdf.crs is None:
            # If IRIS boundaries have no CRS, assume they're in the same CRS as neighborhoods
            iris_gdf.set_crs(neighborhoods.crs, inplace=True)
        elif neighborhoods.crs != iris_gdf.crs:
            logger.info(f"Converting IRIS CRS from {iris_gdf.crs} to {neighborhoods.crs}")
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

        # Merge FILOSOFI income data if available
        if filosofi_data is not None and not filosofi_data.empty:
            # Identify IRIS code column in FILOSOFI data
            filosofi_iris_col = None
            for col in ["IRIS", "CODEGEO", "CODE_IRIS"]:
                if col in filosofi_data.columns:
                    filosofi_iris_col = col
                    break

            # Find IRIS code column in matched DataFrame (after spatial join)
            matched_iris_col = None
            if iris_code_col and iris_code_col in matched.columns:
                matched_iris_col = iris_code_col
            else:
                # Try to find IRIS code column in matched DataFrame
                for col in ["IRIS", iris_code_col, boundary_code_col, "CODEGEO", "CODE_IRIS"]:
                    if col and col in matched.columns:
                        matched_iris_col = col
                        break

            if filosofi_iris_col and matched_iris_col:
                # Merge FILOSOFI data with matched IRIS data
                matched = matched.merge(
                    filosofi_data,
                    left_on=matched_iris_col,
                    right_on=filosofi_iris_col,
                    how="left",
                    suffixes=("", "_filosofi"),
                )
                logger.info("Merged FILOSOFI income data with IRIS census data")
            else:
                logger.warning(
                    f"Could not merge FILOSOFI data - IRIS code columns not found "
                    f"(matched: {matched_iris_col}, filosofi: {filosofi_iris_col})"
                )

        # Merge RP_LOGEMENT car ownership data if available
        if logement_data is not None and not logement_data.empty:
            # Identify IRIS code column in LOGEMENT data
            logement_iris_col = None
            for col in ["IRIS", "CODEGEO", "CODE_IRIS"]:
                if col in logement_data.columns:
                    logement_iris_col = col
                    break

            # Find IRIS code column in matched DataFrame (after spatial join)
            matched_iris_col = None
            if iris_code_col and iris_code_col in matched.columns:
                matched_iris_col = iris_code_col
            else:
                # Try to find IRIS code column in matched DataFrame
                for col in ["IRIS", iris_code_col, boundary_code_col, "CODEGEO", "CODE_IRIS"]:
                    if col and col in matched.columns:
                        matched_iris_col = col
                        break

            if logement_iris_col and matched_iris_col:
                # Merge LOGEMENT data with matched IRIS data
                matched = matched.merge(
                    logement_data,
                    left_on=matched_iris_col,
                    right_on=logement_iris_col,
                    how="left",
                    suffixes=("", "_logement"),
                )
                logger.info("Merged RP_LOGEMENT car ownership data with IRIS census data")
            else:
                logger.warning(
                    f"Could not merge LOGEMENT data - IRIS code columns not found "
                    f"(matched: {matched_iris_col}, logement: {logement_iris_col})"
                )

        # Aggregate IRIS data per neighborhood
        # Identify census columns (P17_*, C17_*, DEC_*, DISP_*, REV_*, MED_*, LOG_*, MEN_*) and convert to numeric
        census_cols = [
            col
            for col in matched.columns
            if (
                "P17_" in col
                or "C17_" in col
                or "DEC_" in col
                or "DISP_" in col
                or "REV_" in col
                or "MED" in col
                or "PIMP" in col
                or "TP60" in col
                or "2ROUES" in col
                or "LOG" in col
                or "MEN" in col
                or "VOIT" in col
            )
            and col not in ["name", "index_right", "geometry"]
        ]

        # Convert census columns to numeric (they come as strings from Excel)
        for col in census_cols:
            matched[col] = pd.to_numeric(matched[col], errors="coerce")

        # Aggregate using mean (can be improved with area-weighted aggregation)
        if census_cols:
            aggregated = matched.groupby("name")[census_cols].mean().reset_index()
            aggregated.rename(columns={"name": "neighborhood_name"}, inplace=True)
        else:
            logger.warning("No census columns found for aggregation")
            aggregated = matched[["name"]].drop_duplicates()
            aggregated.rename(columns={"name": "neighborhood_name"}, inplace=True)

        logger.info(f"Matched IRIS data to {len(aggregated)} neighborhoods")
        return aggregated

    def _extract_demographic_features(
        self,
        matched_data: pd.DataFrame,
        neighborhoods: gpd.GeoDataFrame = None,
        commune_age_data: Optional[pd.DataFrame] = None,
        iris_data_all: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Extract demographic features from matched census data.

        Extracts the following features from raw INSEE variables:
        - Population density (population / area)
        - SES index (proportion of high-status workers)
        - Car Ownership Rate (households with car / total households) - from RP_LOGEMENT
        - Children per capita (estimated children / total population) - LIMITATION: Estimated from Paris-wide ratios
        - Elderly ratio (estimated elderly / total population) - LIMITATION: Estimated from Paris-wide ratios
        - Unemployment rate (unemployed / active population)
        - Student ratio (students / total population) - Standardized to total population for comparability
        - Walking ratio (walking commuters / active workers) - Mode share among commuters
        - Cycling ratio (cycling commuters / active workers) - Mode share among commuters
        - Public transport ratio (public transport commuters / active workers) - Mode share among commuters
        - Two-wheelers ratio (motorcycle/2-wheeler commuters / active workers) - Mode share among commuters
        - Car Commute Ratio (car commuters / active workers) - NOTE: This is commuting behavior, not car ownership
        - Retired ratio (early retirees 15-64 + elderly 65+ / total population) - Standardized to total population for comparability. Combines early retirees with elderly since most retirees are 65+
        - Permanent employment ratio (permanent workers / salaried workers)
        - Temporary employment ratio (temporary workers / salaried workers)
        - Median income (from FILOSOFI)
        - Poverty rate (from FILOSOFI)
        
        Note: Most ratios are standardized to total population for comparability, except:
        - Commuting ratios use active workers (mode share is more meaningful)
        - Employment ratios use salaried workers (subset-specific)
        - Car ownership uses households (household-level metric)

        Args:
            matched_data: DataFrame with neighborhood_name and raw census variables.
            neighborhoods: GeoDataFrame with neighborhood boundaries (for area calculation).
            commune_age_data: Optional pre-fetched commune-level age data. If None, will fetch internally.
            iris_data_all: Optional pre-fetched Paris-wide IRIS data. If None, will fetch internally.

        Returns:
            DataFrame with neighborhood_name and extracted feature columns.

        Example:
            >>> features = loader._extract_demographic_features(matched_data, neighborhoods)
            >>> print(features.columns)
        """
        if matched_data.empty:
            logger.warning("Empty matched data, cannot extract features")
            return pd.DataFrame()

        features_df = matched_data[["neighborhood_name"]].copy()

        # Map RP_ACTRES_IRIS variable names to our features
        # Population-related variables (P17_POP*)
        pop_cols = [
            col
            for col in matched_data.columns
            if "P17_POP" in col or "pop" in col.lower()
        ]
        # Active population (P17_ACT*)
        active_cols = [
            col
            for col in matched_data.columns
            if "P17_ACT" in col and "ACTOCC" not in col
        ]
        # Car ownership (C17_ACTOCC15P_VOIT, P17_ACTOCC*)
        car_cols = [
            col
            for col in matched_data.columns
            if "VOIT" in col or ("ACTOCC" in col and "VOIT" in col.lower())
        ]
        # Employment by sector (C17_ACT1564_CS*)
        employment_cols = [
            col
            for col in matched_data.columns
            if "C17_ACT" in col and "_CS" in col
        ]
        # Unemployment (P17_CHOM*)
        chom_cols = [
            col
            for col in matched_data.columns
            if "CHOM" in col
        ]
        # Education/Students (P17_ETUD*)
        etud_cols = [
            col
            for col in matched_data.columns
            if "ETUD" in col
        ]
        # Commuting modes
        pas_cols = [
            col
            for col in matched_data.columns
            if "PAS" in col or ("ACTOCC" in col and "PAS" in col)
        ]
        mar_cols = [
            col
            for col in matched_data.columns
            if "MAR" in col or ("ACTOCC" in col and "MAR" in col)
        ]
        velo_cols = [
            col
            for col in matched_data.columns
            if "VELO" in col or ("ACTOCC" in col and "VELO" in col)
        ]
        tcom_cols = [
            col
            for col in matched_data.columns
            if "TCOM" in col or ("ACTOCC" in col and "TCOM" in col)
        ]
        deux_roues_cols = [
            col
            for col in matched_data.columns
            if "2ROUES" in col or ("ACTOCC" in col and "2ROUES" in col)
        ]
        # Retired (P17_RETR*)
        retr_cols = [
            col
            for col in matched_data.columns
            if "RETR" in col
        ]
        # Employment contracts
        cdi_cols = [
            col
            for col in matched_data.columns
            if "CDI" in col
        ]
        cdd_cols = [
            col
            for col in matched_data.columns
            if "CDD" in col
        ]
        # Income (FILOSOFI - DEC_*, MED_*, PIMP*, TP60*)
        income_cols = [
            col
            for col in matched_data.columns
            if "DEC_" in col or "MED" in col or "PIMP" in col or "TP60" in col
        ]

        # 1. Population density
        # Use estimated total population (working-age scaled up) divided by neighborhood area
        if pop_cols and neighborhoods is not None:
            total_pop_col = "P17_POP1564" if "P17_POP1564" in matched_data.columns else pop_cols[0]
            working_age_pop = matched_data[total_pop_col]
            
            # Get commune-level data and full Paris IRIS data to estimate total population
            # Use provided data if available, otherwise fetch (for backward compatibility)
            if commune_age_data is None:
                commune_age_data = self._fetch_commune_age_data()
            if not commune_age_data.empty and "total_population" in commune_age_data.columns:
                total_pop_commune = commune_age_data["total_population"].iloc[0]
                
                # Get working-age from ALL Paris IRIS data for accurate ratio
                # Use provided data if available, otherwise fetch (for backward compatibility)
                if iris_data_all is None:
                    iris_data_all = self._fetch_iris_data()
                if not iris_data_all.empty and total_pop_col in iris_data_all.columns:
                    iris_data_all[total_pop_col] = pd.to_numeric(
                        iris_data_all[total_pop_col], errors="coerce"
                    )
                    working_age_paris_total = iris_data_all[total_pop_col].sum()
                    
                    if working_age_paris_total > 0:
                        working_age_ratio = working_age_paris_total / total_pop_commune
                        estimated_total_pop = working_age_pop / working_age_ratio
                        
                        # Calculate area from neighborhood geometry
                        neighborhoods_with_area = neighborhoods.copy()
                        # Convert geometry to area in m² (assuming CRS is EPSG:4326)
                        if neighborhoods_with_area.crs == "EPSG:4326":
                            # Convert to metric CRS for area calculation
                            neighborhoods_with_area = neighborhoods_with_area.to_crs("EPSG:3857")
                        neighborhoods_with_area["area_m2"] = neighborhoods_with_area.geometry.area
                        
                        # Merge to get area per neighborhood
                        # Create a mapping from neighborhood name to area
                        area_map = dict(
                            zip(
                                neighborhoods_with_area["name"],
                                neighborhoods_with_area["area_m2"],
                            )
                        )
                        
                        # Map area to each neighborhood
                        features_df["area_m2"] = features_df["neighborhood_name"].map(area_map)
                        
                        if features_df["area_m2"].notna().any():
                            # Calculate density per km²
                            features_df["population_density"] = (
                                estimated_total_pop / (features_df["area_m2"] / 1_000_000)
                            )
                            # Drop temporary area column
                            features_df.drop(columns=["area_m2"], inplace=True)
                        else:
                            features_df["population_density"] = None
                            if "area_m2" in features_df.columns:
                                features_df.drop(columns=["area_m2"], inplace=True)
                    else:
                        features_df["population_density"] = None
                else:
                    features_df["population_density"] = None
            else:
                features_df["population_density"] = None
        else:
            features_df["population_density"] = None

        # 2. SES index (from employment sector data)
        # Use proportion of high-status workers (CS1+CS2+CS3) as proxy for SES
        # CS1: Farmers, CS2: Artisans/merchants/entrepreneurs, CS3: Managers/professionals
        if employment_cols:
            cs1_col = "C17_ACT1564_CS1" if "C17_ACT1564_CS1" in matched_data.columns else None
            cs2_col = "C17_ACT1564_CS2" if "C17_ACT1564_CS2" in matched_data.columns else None
            cs3_col = "C17_ACT1564_CS3" if "C17_ACT1564_CS3" in matched_data.columns else None
            total_act_col = "C17_ACT1564" if "C17_ACT1564" in matched_data.columns else None
            
            if cs1_col and cs2_col and cs3_col and total_act_col:
                # Check if columns exist and have values
                if (
                    cs1_col in matched_data.columns
                    and cs2_col in matched_data.columns
                    and cs3_col in matched_data.columns
                    and total_act_col in matched_data.columns
                ):
                    high_status = (
                        matched_data[cs1_col].fillna(0)
                        + matched_data[cs2_col].fillna(0)
                        + matched_data[cs3_col].fillna(0)
                    )
                    total_active = matched_data[total_act_col].fillna(0)
                    
                    # Avoid division by zero
                    ses_proxy = pd.Series(index=matched_data.index, dtype=float)
                    mask = total_active > 0
                    ses_proxy[mask] = high_status[mask] / total_active[mask]
                    ses_proxy[~mask] = 0
                    
                    # Store raw SES proxy (proportion of high-status workers)
                    # Note: Normalization (z-score) should be done across all neighborhoods
                    # For now, store raw value (0-1 range) - can normalize later in feature engineering
                    features_df["ses_index"] = ses_proxy
                else:
                    logger.warning("SES columns not found in matched_data")
                    features_df["ses_index"] = None
            else:
                logger.warning("SES calculation columns missing")
                features_df["ses_index"] = None
        else:
            features_df["ses_index"] = None

        # 3. Car Ownership Rate (from RP_LOGEMENT)
        # RP_LOGEMENT data is aggregated by IRIS and should have:
        # - car_ownership_rate (pre-calculated from aggregation)
        # - P17_MEN (total households)
        # - C17_MEN_VOIT (households with car)
        # - C17_MEN_VOIT1 (households with 1 car)
        # - C17_MEN_VOIT2P (households with 2+ cars)
        
        # First, check if car_ownership_rate is already calculated (from aggregation)
        if "car_ownership_rate" in matched_data.columns:
            features_df["car_ownership_rate"] = pd.to_numeric(matched_data["car_ownership_rate"], errors="coerce").fillna(0)
        elif "P17_MEN" in matched_data.columns and "C17_MEN_VOIT" in matched_data.columns:
            # Calculate from aggregated counts
            total_households = pd.to_numeric(matched_data["P17_MEN"], errors="coerce").fillna(0)
            households_with_car = pd.to_numeric(matched_data["C17_MEN_VOIT"], errors="coerce").fillna(0)
            features_df["car_ownership_rate"] = (
                (households_with_car / total_households) if total_households.sum() > 0 else 0
            ).fillna(0)
        elif "P17_MEN" in matched_data.columns and ("C17_MEN_VOIT1" in matched_data.columns or "C17_MEN_VOIT2P" in matched_data.columns):
            # Calculate from 1 car + 2+ cars counts
            total_households = pd.to_numeric(matched_data["P17_MEN"], errors="coerce").fillna(0)
            households_with_car = pd.Series(0, index=matched_data.index)
            if "C17_MEN_VOIT1" in matched_data.columns:
                households_with_car += pd.to_numeric(matched_data["C17_MEN_VOIT1"], errors="coerce").fillna(0)
            if "C17_MEN_VOIT2P" in matched_data.columns:
                households_with_car += pd.to_numeric(matched_data["C17_MEN_VOIT2P"], errors="coerce").fillna(0)
            features_df["car_ownership_rate"] = (
                (households_with_car / total_households) if total_households.sum() > 0 else 0
            ).fillna(0)
        else:
            # Try legacy column names (for backward compatibility)
            men_voit_cols = [
                col
                for col in matched_data.columns
                if ("MEN" in col and "VOIT" in col) or "PCT_MEN_VOIT" in col or "NB_MEN_VOIT" in col
            ]
            men_total_cols = [
                col
                for col in matched_data.columns
                if col == "P17_MEN" or col == "P17_LOG" or (col.startswith("P17_") and "MEN" in col and "VOIT" not in col and "LOG" not in col)
            ]
            
            if men_voit_cols and men_total_cols:
                # Try to find percentage directly first
                pct_col = None
                for col in ["PCT_MEN_VOIT", "PCT_MEN_VOIT1", "P17_MEN_VOIT"]:
                    if col in matched_data.columns:
                        pct_col = col
                        break
                
                if pct_col:
                    # Use percentage directly (already a ratio)
                    pct_values = pd.to_numeric(matched_data[pct_col], errors="coerce")
                    features_df["car_ownership_rate"] = (pct_values / 100).fillna(0)  # Convert percentage to ratio
                else:
                    # Calculate from counts: households with car / total households
                    men_total_col = "P17_MEN" if "P17_MEN" in matched_data.columns else men_total_cols[0]
                    
                    # Find households with car (try different variable names)
                    households_with_car = pd.Series(0, index=matched_data.index)
                    
                    # Try C17_MEN_VOIT1 (1 car) + C17_MEN_VOIT2P (2+ cars)
                    voit1_cols = [c for c in men_voit_cols if "VOIT1" in c or ("1" in c and "VOIT" in c)]
                    voit2p_cols = [c for c in men_voit_cols if "VOIT2" in c or "2P" in c or ("2" in c and "VOIT" in c and "1" not in c)]
                    
                    if voit1_cols:
                        households_with_car += pd.to_numeric(matched_data[voit1_cols[0]], errors="coerce").fillna(0)
                    if voit2p_cols:
                        households_with_car += pd.to_numeric(matched_data[voit2p_cols[0]], errors="coerce").fillna(0)
                    
                    # If no specific counts, try NB_MEN_VOIT (total households with car)
                    if households_with_car.sum() == 0:
                        nb_voit_cols = [c for c in men_voit_cols if "NB_MEN_VOIT" in c or ("NB" in c and "VOIT" in c)]
                        if nb_voit_cols:
                            households_with_car = pd.to_numeric(matched_data[nb_voit_cols[0]], errors="coerce").fillna(0)
                    
                    if men_total_col and men_total_col in matched_data.columns and households_with_car.sum() > 0:
                        total_households = pd.to_numeric(matched_data[men_total_col], errors="coerce").fillna(0)
                        
                        # Calculate car ownership rate
                        features_df["car_ownership_rate"] = (
                            (households_with_car / total_households) if total_households.sum() > 0 else 0
                        ).fillna(0)
                    else:
                        features_df["car_ownership_rate"] = None
            else:
                features_df["car_ownership_rate"] = None

        # 3b. Car Commute Ratio (from RP_ACTRES_IRIS)
        # Use C17_ACTOCC15P_VOIT (private car commuters among active workers 15+) / P17_ACTOCC15P (total active workers 15+)
        # Note: This represents commuting behavior, not car ownership. In Paris, many people own cars
        # but don't use them for commuting due to parking costs, traffic, etc.
        if car_cols:
            car_commute_col = "C17_ACTOCC15P_VOIT" if "C17_ACTOCC15P_VOIT" in matched_data.columns else car_cols[0]
            total_active_col = "P17_ACTOCC15P" if "P17_ACTOCC15P" in matched_data.columns else None
            
            if total_active_col:
                car_commuters = matched_data[car_commute_col]
                total_active = matched_data[total_active_col]
                features_df["car_commute_ratio"] = (
                    (car_commuters / total_active) if total_active.sum() > 0 else 0
                ).fillna(0)
            else:
                features_df["car_commute_ratio"] = None
        else:
            features_df["car_commute_ratio"] = None

        # 4. Estimate total population per neighborhood (needed for standardizing ratios)
        # RP_ACTRES_IRIS only has working-age (15-64), so we estimate total population
        # using commune-level total population and IRIS working-age population
        # Strategy: Use Paris-wide working-age ratio to estimate total population per neighborhood
        # Use provided data if available, otherwise fetch (for backward compatibility)
        if commune_age_data is None:
            commune_age_data = self._fetch_commune_age_data()
        total_pop_col = "P17_POP1564" if "P17_POP1564" in matched_data.columns else pop_cols[0] if pop_cols else None
        
        # Initialize estimated_total_pop as None - will be calculated if data is available
        estimated_total_pop = None
        
        if not commune_age_data.empty and "total_population" in commune_age_data.columns and total_pop_col:
            total_pop_commune = commune_age_data["total_population"].iloc[0]
            
            # Get working-age population from ALL Paris IRIS data (not just matched neighborhoods)
            # This gives us the true Paris-wide ratio (~71% working-age)
            # Use provided data if available, otherwise fetch (for backward compatibility)
            if iris_data_all is None:
                iris_data_all = self._fetch_iris_data()  # Get all Paris IRIS data
            if not iris_data_all.empty and total_pop_col in iris_data_all.columns:
                iris_data_all[total_pop_col] = pd.to_numeric(
                    iris_data_all[total_pop_col], errors="coerce"
                )
                working_age_paris_total = iris_data_all[total_pop_col].sum()
                
                if working_age_paris_total > 0:
                    # Calculate Paris-wide working-age ratio (~0.712 = 71.2%)
                    working_age_ratio = working_age_paris_total / total_pop_commune
                    
                    # For each neighborhood, estimate total population based on working-age
                    working_age_per_neighborhood = pd.to_numeric(matched_data[total_pop_col], errors="coerce").fillna(0)
                    estimated_total_pop = working_age_per_neighborhood / working_age_ratio
                    
                    logger.info(f"Estimated total population for {len(estimated_total_pop)} neighborhoods using Paris-wide working-age ratio ({working_age_ratio:.3f})")
        
        # 5. Children per capita (<15 years) and 6. Elderly ratio (65+ years)
        # Use estimated total population to calculate age group ratios
        if estimated_total_pop is not None and total_pop_col:
                    
            # Estimate children and elderly per neighborhood
            # Non-working-age = total - working-age
            working_age_per_neighborhood = pd.to_numeric(matched_data[total_pop_col], errors="coerce").fillna(0)
            non_working_age_per_neighborhood = (
                estimated_total_pop - working_age_per_neighborhood
            )
            
            # Children: ~47% of non-working-age population
            # Elderly: ~53% of non-working-age population
            # Then divide by total_pop to get ratio of total population (~13.5% and ~15.3%)
            estimated_children = non_working_age_per_neighborhood * 0.47
            features_df["children_per_capita"] = (
                estimated_children / estimated_total_pop
            ).fillna(0)
            
            estimated_elderly = non_working_age_per_neighborhood * 0.53
            features_df["elderly_ratio"] = (
                estimated_elderly / estimated_total_pop
            ).fillna(0)
        else:
            # Fallback: set to None if total population cannot be estimated
            logger.warning("Could not estimate total population, age ratios set to None")
            features_df["children_per_capita"] = None
            features_df["elderly_ratio"] = None

        # 7. Unemployment rate
        # P17_CHOM1564 (unemployed 15-64) / P17_ACT1564 (active 15-64)
        if chom_cols:
            chom_col = "P17_CHOM1564" if "P17_CHOM1564" in matched_data.columns else chom_cols[0]
            act_col = "P17_ACT1564" if "P17_ACT1564" in matched_data.columns else None
            
            if act_col and act_col in matched_data.columns:
                unemployed = matched_data[chom_col].fillna(0)
                active = matched_data[act_col].fillna(0)
                features_df["unemployment_rate"] = (
                    (unemployed / active) if active.sum() > 0 else 0
                ).fillna(0)
            else:
                features_df["unemployment_rate"] = None
        else:
            features_df["unemployment_rate"] = None

        # 8. Student ratio
        # P17_ETUD1564 (students 15-64) / estimated_total_pop (total population)
        # Standardized to total population for comparability with other demographic ratios
        if etud_cols and estimated_total_pop is not None:
            etud_col = "P17_ETUD1564" if "P17_ETUD1564" in matched_data.columns else etud_cols[0]
            
            if etud_col in matched_data.columns:
                students = pd.to_numeric(matched_data[etud_col], errors="coerce").fillna(0)
                features_df["student_ratio"] = (
                    (students / estimated_total_pop) if estimated_total_pop.sum() > 0 else 0
                ).fillna(0)
            else:
                features_df["student_ratio"] = None
        else:
            if estimated_total_pop is None:
                logger.warning("Cannot calculate student_ratio: total population not estimated")
            features_df["student_ratio"] = None

        # 9. Commuting modes
        # Walking: C17_ACTOCC15P_PAS or C17_ACTOCC15P_MAR / P17_ACTOCC15P
        walking_col = None
        if pas_cols:
            walking_col = "C17_ACTOCC15P_PAS" if "C17_ACTOCC15P_PAS" in matched_data.columns else pas_cols[0]
        elif mar_cols:
            walking_col = "C17_ACTOCC15P_MAR" if "C17_ACTOCC15P_MAR" in matched_data.columns else mar_cols[0]
        
        actocc_col = "P17_ACTOCC15P" if "P17_ACTOCC15P" in matched_data.columns else None
        
        if walking_col and actocc_col and walking_col in matched_data.columns and actocc_col in matched_data.columns:
            walking = matched_data[walking_col].fillna(0)
            active_occ = matched_data[actocc_col].fillna(0)
            features_df["walking_ratio"] = (
                (walking / active_occ) if active_occ.sum() > 0 else 0
            ).fillna(0)
        else:
            features_df["walking_ratio"] = None

        # Cycling: C17_ACTOCC15P_VELO / P17_ACTOCC15P
        if velo_cols:
            velo_col = "C17_ACTOCC15P_VELO" if "C17_ACTOCC15P_VELO" in matched_data.columns else velo_cols[0]
            if actocc_col and velo_col in matched_data.columns and actocc_col in matched_data.columns:
                cycling = matched_data[velo_col].fillna(0)
                active_occ = matched_data[actocc_col].fillna(0)
                features_df["cycling_ratio"] = (
                    (cycling / active_occ) if active_occ.sum() > 0 else 0
                ).fillna(0)
            else:
                features_df["cycling_ratio"] = None
        else:
            features_df["cycling_ratio"] = None

        # Public transport: C17_ACTOCC15P_TCOM / P17_ACTOCC15P
        if tcom_cols:
            tcom_col = "C17_ACTOCC15P_TCOM" if "C17_ACTOCC15P_TCOM" in matched_data.columns else tcom_cols[0]
            if actocc_col and tcom_col in matched_data.columns and actocc_col in matched_data.columns:
                public_transport = matched_data[tcom_col].fillna(0)
                active_occ = matched_data[actocc_col].fillna(0)
                features_df["public_transport_ratio"] = (
                    (public_transport / active_occ) if active_occ.sum() > 0 else 0
                ).fillna(0)
            else:
                features_df["public_transport_ratio"] = None
        else:
            features_df["public_transport_ratio"] = None

        # 2-wheelers/motorcycles: C17_ACTOCC15P_2ROUESMOT / P17_ACTOCC15P
        if deux_roues_cols:
            deux_roues_col = "C17_ACTOCC15P_2ROUESMOT" if "C17_ACTOCC15P_2ROUESMOT" in matched_data.columns else deux_roues_cols[0]
            if actocc_col and deux_roues_col in matched_data.columns and actocc_col in matched_data.columns:
                deux_roues = matched_data[deux_roues_col].fillna(0)
                active_occ = matched_data[actocc_col].fillna(0)
                features_df["two_wheelers_ratio"] = (
                    (deux_roues / active_occ) if active_occ.sum() > 0 else 0
                ).fillna(0)
            else:
                features_df["two_wheelers_ratio"] = None
        else:
            features_df["two_wheelers_ratio"] = None

        # 10. Retired ratio
        # Combines P17_RETR1564 (early retirees 15-64) + estimated_elderly (65+, most retirees)
        # Standardized to total population for comparability with other demographic ratios
        # Note: Most retirees are 65+, so we combine early retirees (15-64) with elderly (65+)
        if retr_cols and estimated_total_pop is not None:
            retr_col = "P17_RETR1564" if "P17_RETR1564" in matched_data.columns else retr_cols[0]
            
            if retr_col in matched_data.columns:
                # Early retirees (15-64)
                early_retired = pd.to_numeric(matched_data[retr_col], errors="coerce").fillna(0)
                
                # Add estimated elderly (65+) - most retirees are in this group
                # We already calculated estimated_elderly for elderly_ratio above
                if "elderly_ratio" in features_df.columns and features_df["elderly_ratio"].notna().any():
                    # Convert elderly_ratio back to count: elderly_ratio * total_pop
                    estimated_elderly_count = features_df["elderly_ratio"] * estimated_total_pop
                    total_retired = early_retired + estimated_elderly_count
                else:
                    # Fallback: just use early retirees if elderly not available
                    total_retired = early_retired
                    logger.warning("elderly_ratio not available, retired_ratio only includes early retirees (15-64)")
                
                features_df["retired_ratio"] = (
                    (total_retired / estimated_total_pop) if estimated_total_pop.sum() > 0 else 0
                ).fillna(0)
            else:
                features_df["retired_ratio"] = None
        else:
            if estimated_total_pop is None:
                logger.warning("Cannot calculate retired_ratio: total population not estimated")
            features_df["retired_ratio"] = None

        # 11. Employment contract types
        # Permanent: P17_SAL15P_CDI / P17_SAL15P (total salaried)
        # Temporary: P17_SAL15P_CDD / P17_SAL15P
        sal_total_col = "P17_SAL15P" if "P17_SAL15P" in matched_data.columns else None
        
        if cdi_cols and sal_total_col:
            cdi_col = "P17_SAL15P_CDI" if "P17_SAL15P_CDI" in matched_data.columns else cdi_cols[0]
            if cdi_col in matched_data.columns and sal_total_col in matched_data.columns:
                permanent = matched_data[cdi_col].fillna(0)
                total_sal = matched_data[sal_total_col].fillna(0)
                features_df["permanent_employment_ratio"] = (
                    (permanent / total_sal) if total_sal.sum() > 0 else 0
                ).fillna(0)
            else:
                features_df["permanent_employment_ratio"] = None
        else:
            features_df["permanent_employment_ratio"] = None

        if cdd_cols and sal_total_col:
            cdd_col = "P17_SAL15P_CDD" if "P17_SAL15P_CDD" in matched_data.columns else cdd_cols[0]
            if cdd_col in matched_data.columns and sal_total_col in matched_data.columns:
                temporary = matched_data[cdd_col].fillna(0)
                total_sal = matched_data[sal_total_col].fillna(0)
                features_df["temporary_employment_ratio"] = (
                    (temporary / total_sal) if total_sal.sum() > 0 else 0
                ).fillna(0)
            else:
                features_df["temporary_employment_ratio"] = None
        else:
            features_df["temporary_employment_ratio"] = None

        # 12. Income features (from FILOSOFI)
        # Median income: DISP_MED17 (FILOSOFI_DISP_IRIS_2017) or DEC_MED14/17 (FILOSOFI_DEC_IRIS)
        median_col = None
        for col in ["DISP_MED17", "DEC_MED17", "DEC_MED14", "MED"]:
            if col in matched_data.columns:
                median_col = col
                break
        
        if median_col:
            median_values = matched_data[median_col].copy()
            median_values = pd.to_numeric(median_values, errors="coerce")
            features_df["median_income"] = median_values.where(median_values.notna(), None)
        else:
            features_df["median_income"] = None

        # Poverty rate: DISP_TP6017 (FILOSOFI_DISP_IRIS_2017) or DEC_PIMP14/17 (FILOSOFI_DEC_IRIS)
        poverty_col = None
        for col in ["DISP_TP6017", "DISP_PIMPOT17", "DEC_PIMP17", "DEC_PIMP14", "PIMP", "TP60"]:
            if col in matched_data.columns:
                poverty_col = col
                break
        
        if poverty_col:
            poverty_values = matched_data[poverty_col].copy()
            poverty_values = pd.to_numeric(poverty_values, errors="coerce")
            features_df["poverty_rate"] = poverty_values.where(poverty_values.notna(), None)
        else:
            features_df["poverty_rate"] = None

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
                # Load IRIS boundaries first to get IRIS codes
                iris_boundaries = self._load_iris_boundaries()
                if iris_boundaries.empty:
                    raise ValueError("No IRIS boundaries loaded")

                # Extract IRIS codes (prioritize CODE_IRIS for 2021 format)
                if "CODE_IRIS" in iris_boundaries.columns:
                    iris_codes = iris_boundaries["CODE_IRIS"].unique().tolist()
                elif "DCOMIRIS" in iris_boundaries.columns:
                    iris_codes = iris_boundaries["DCOMIRIS"].unique().tolist()
                elif "CODEGEO" in iris_boundaries.columns:
                    iris_codes = iris_boundaries["CODEGEO"].unique().tolist()
                else:
                    raise ValueError("Could not extract IRIS codes from boundaries")

                # Fetch IRIS data (will filter for Paris if iris_codes is None)
                iris_data = self._fetch_iris_data(iris_codes=iris_codes)
                if iris_data.empty:
                    raise ValueError("No IRIS data fetched")

                # Fetch FILOSOFI income data
                filosofi_data = self._fetch_filosofi_income(iris_codes=iris_codes)
                if filosofi_data.empty:
                    logger.warning("No FILOSOFI income data fetched, continuing without income features")

                # Fetch RP_LOGEMENT car ownership data
                logement_data = self._fetch_logement_car_ownership(iris_codes=iris_codes)
                if logement_data.empty:
                    logger.warning("No RP_LOGEMENT car ownership data fetched, continuing without car ownership feature")

                # Load all neighborhoods for spatial join
                neighborhoods = load_neighborhoods(
                    self.config.get("paths", {}).get(
                        "neighborhoods_geojson", "paris_neighborhoods.geojson"
                    )
                )

                # Match IRIS to neighborhoods (include FILOSOFI and LOGEMENT data)
                matched_data = self._match_iris_to_neighborhoods(
                    iris_data,
                    iris_boundaries,
                    neighborhoods,
                    filosofi_data=filosofi_data if not filosofi_data.empty else None,
                    logement_data=logement_data if not logement_data.empty else None,
                )

                if matched_data.empty:
                    raise ValueError("No IRIS data matched to neighborhoods")

                # Filter to current neighborhood
                neighborhood_data = matched_data[
                    matched_data["neighborhood_name"] == neighborhood_name
                ]

                if neighborhood_data.empty:
                    raise ValueError(f"No IRIS data matched to {neighborhood_name}")

                # Fetch commune age data and Paris-wide IRIS data once (shared across all neighborhoods)
                # This avoids redundant API calls when processing multiple neighborhoods
                commune_age_data = self._fetch_commune_age_data()
                iris_data_all = self._fetch_iris_data()  # Get all Paris IRIS data for ratio calculations

                # Extract demographic features (pass neighborhoods for area calculation)
                # Pass pre-fetched commune_age_data and iris_data_all to avoid redundant calls
                features_df = self._extract_demographic_features(
                    neighborhood_data,
                    neighborhoods,
                    commune_age_data=commune_age_data,
                    iris_data_all=iris_data_all,
                )
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
