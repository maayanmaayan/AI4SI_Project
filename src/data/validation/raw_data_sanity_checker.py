"""Raw data sanity checker for validating OSM and Census data files.

This module provides comprehensive validation of raw data files against reasonable
value ranges based on Paris-specific urban planning data.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon

from src.utils.config import get_config
from src.utils.helpers import (
    ensure_dir_exists,
    get_compliant_neighborhoods,
    get_service_category_names,
    load_neighborhoods,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RawDataSanityChecker:
    """Validate raw OSM and Census data files against reasonable value ranges.

    This class provides methods to validate all extracted raw data files (OSM and
    Census) for each neighborhood, checking file structure, format validity, value
    ranges, and cross-source consistency.

    Attributes:
        config: Configuration dictionary loaded from config.yaml.
        data_root: Root directory for data files.
        neighborhoods_geojson: Path to neighborhoods GeoJSON file.

    Example:
        >>> checker = RawDataSanityChecker()
        >>> result = checker.validate_neighborhood("paris_rive_gauche")
        >>> print(f"Status: {result['status']}")
    """

    # OSM Data Bounds
    # Services
    SERVICES_MIN_COUNT = 0  # Hard: > 0
    SERVICES_WARN_LOW = 5
    SERVICES_WARN_HIGH = 10000
    SERVICES_DENSITY_HARD_MIN = 0
    SERVICES_DENSITY_HARD_MAX = 500
    SERVICES_DENSITY_WARN_LOW = 10
    SERVICES_DENSITY_WARN_HIGH = 300

    # Buildings
    BUILDINGS_MIN_COUNT = 0  # Hard: > 0
    BUILDINGS_WARN_LOW = 10
    BUILDINGS_WARN_HIGH = 50000
    BUILDINGS_DENSITY_HARD_MIN = 0
    BUILDINGS_DENSITY_HARD_MAX = 5000
    BUILDINGS_DENSITY_WARN_LOW = 50
    BUILDINGS_DENSITY_WARN_HIGH = 3000
    BUILDING_LEVELS_HARD_MIN = 1
    BUILDING_LEVELS_HARD_MAX = 60
    BUILDING_LEVELS_WARN_MAX = 12  # Most Paris buildings
    BUILDING_AREA_HARD_MIN = 1
    BUILDING_AREA_HARD_MAX = 100000
    BUILDING_AREA_WARN_MAX = 50000
    BUILDING_AREA_RATIO_HARD_MIN = 0.1
    BUILDING_AREA_RATIO_HARD_MAX = 0.8
    BUILDING_AREA_RATIO_WARN_LOW = 0.30
    BUILDING_AREA_RATIO_WARN_HIGH = 0.9
    BUILDING_LEVELS_MISSING_WARN_THRESHOLD = 0.5  # Warn if >50% missing

    # Network
    NETWORK_NODES_MIN = 0  # Hard: > 0
    NETWORK_NODES_WARN_LOW = 10
    NETWORK_EDGES_MIN = 0  # Hard: > 0
    NETWORK_EDGES_WARN_LOW = 10
    NETWORK_NODE_EDGE_RATIO_HARD_MIN = 0.5
    NETWORK_NODE_EDGE_RATIO_HARD_MAX = 3.0
    NETWORK_NODE_EDGE_RATIO_WARN_LOW = 0.3
    NETWORK_NODE_EDGE_RATIO_WARN_HIGH = 4.0
    NETWORK_CONNECTIVITY_HARD_MIN = 0.8  # >= 80% in largest component
    NETWORK_CONNECTIVITY_WARN_MIN = 0.5  # Warn if < 50%
    EDGE_LENGTH_HARD_MIN = 1
    EDGE_LENGTH_HARD_MAX = 5000
    EDGE_LENGTH_WARN_MIN = 0.1
    EDGE_LENGTH_WARN_MAX = 2000
    AVG_BLOCK_LENGTH_HARD_MIN = 50
    AVG_BLOCK_LENGTH_HARD_MAX = 300
    AVG_BLOCK_LENGTH_WARN_MIN = 30
    AVG_BLOCK_LENGTH_WARN_MAX = 400

    # Intersections
    INTERSECTIONS_MIN_COUNT = 0  # Hard: > 0
    INTERSECTIONS_WARN_LOW = 5
    INTERSECTIONS_DENSITY_HARD_MIN = 20
    INTERSECTIONS_DENSITY_HARD_MAX = 500
    INTERSECTIONS_DENSITY_WARN_LOW = 10
    INTERSECTIONS_DENSITY_WARN_HIGH = 400
    INTERSECTION_DEGREE_HARD_MIN = 3
    INTERSECTION_DEGREE_HARD_MAX = 10
    INTERSECTION_DEGREE_WARN_MAX = 12

    # Coordinate bounds (Paris)
    PARIS_LON_MIN = 2.2
    PARIS_LON_MAX = 2.4
    PARIS_LAT_MIN = 48.8
    PARIS_LAT_MAX = 48.9

    # Census Data Bounds
    # Population & Demographics
    POPULATION_DENSITY_HARD_MIN = 1000
    POPULATION_DENSITY_HARD_MAX = 60000
    POPULATION_DENSITY_WARN_LOW = 5000
    POPULATION_DENSITY_WARN_HIGH = 50000
    CHILDREN_PER_CAPITA_HARD_MIN = 0
    CHILDREN_PER_CAPITA_HARD_MAX = 0.3
    CHILDREN_PER_CAPITA_WARN_LOW = 0.08
    CHILDREN_PER_CAPITA_WARN_HIGH = 0.20
    ELDERLY_RATIO_HARD_MIN = 0
    ELDERLY_RATIO_HARD_MAX = 0.4
    ELDERLY_RATIO_WARN_LOW = 0.10
    ELDERLY_RATIO_WARN_HIGH = 0.25
    AGE_RATIO_SUM_HARD_MIN = 0.8
    AGE_RATIO_SUM_HARD_MAX = 1.2
    AGE_RATIO_SUM_WARN_MIN = 0.7
    AGE_RATIO_SUM_WARN_MAX = 1.3

    # Socioeconomic
    SES_INDEX_HARD_MIN = 0
    SES_INDEX_HARD_MAX = 1
    SES_INDEX_WARN_LOW = 0.1
    SES_INDEX_WARN_HIGH = 0.8
    UNEMPLOYMENT_RATE_HARD_MIN = 0
    UNEMPLOYMENT_RATE_HARD_MAX = 0.5
    UNEMPLOYMENT_RATE_WARN_LOW = 0.01
    UNEMPLOYMENT_RATE_WARN_HIGH = 0.30
    MEDIAN_INCOME_HARD_MIN = 10000
    MEDIAN_INCOME_HARD_MAX = 200000
    MEDIAN_INCOME_WARN_LOW = 15000
    MEDIAN_INCOME_WARN_HIGH = 70000
    POVERTY_RATE_HARD_MIN = 0
    POVERTY_RATE_HARD_MAX = 0.5
    POVERTY_RATE_WARN_LOW = 0.06
    POVERTY_RATE_WARN_HIGH = 0.40

    # Car Ownership & Commuting
    CAR_OWNERSHIP_RATE_HARD_MIN = 0
    CAR_OWNERSHIP_RATE_HARD_MAX = 0.9
    CAR_OWNERSHIP_RATE_WARN_LOW = 0.1
    CAR_OWNERSHIP_RATE_WARN_HIGH = 0.6
    CAR_COMMUTE_RATIO_HARD_MIN = 0
    CAR_COMMUTE_RATIO_HARD_MAX = 0.8
    CAR_COMMUTE_RATIO_WARN_LOW = 0.05
    CAR_COMMUTE_RATIO_WARN_HIGH = 0.5
    WALKING_RATIO_HARD_MIN = 0
    WALKING_RATIO_HARD_MAX = 1
    WALKING_RATIO_WARN_LOW = 0.1
    WALKING_RATIO_WARN_HIGH = 0.6
    CYCLING_RATIO_HARD_MIN = 0
    CYCLING_RATIO_HARD_MAX = 0.3
    CYCLING_RATIO_WARN_LOW = 0.01
    CYCLING_RATIO_WARN_HIGH = 0.10
    PUBLIC_TRANSPORT_RATIO_HARD_MIN = 0
    PUBLIC_TRANSPORT_RATIO_HARD_MAX = 1
    PUBLIC_TRANSPORT_RATIO_WARN_LOW = 0.30
    PUBLIC_TRANSPORT_RATIO_WARN_HIGH = 0.75
    TWO_WHEELERS_RATIO_HARD_MIN = 0
    TWO_WHEELERS_RATIO_HARD_MAX = 0.2
    TWO_WHEELERS_RATIO_WARN_LOW = 0.01
    TWO_WHEELERS_RATIO_WARN_HIGH = 0.10
    MODE_SHARE_SUM_HARD_MIN = 0.8
    MODE_SHARE_SUM_HARD_MAX = 1.2
    MODE_SHARE_SUM_WARN_MIN = 0.7
    MODE_SHARE_SUM_WARN_MAX = 1.3

    # Employment & Education
    STUDENT_RATIO_HARD_MIN = 0
    STUDENT_RATIO_HARD_MAX = 0.4
    STUDENT_RATIO_WARN_LOW = 0.05
    STUDENT_RATIO_WARN_HIGH = 0.25
    RETIRED_RATIO_HARD_MIN = 0
    RETIRED_RATIO_HARD_MAX = 0.4
    RETIRED_RATIO_WARN_LOW = 0.05
    RETIRED_RATIO_WARN_HIGH = 0.35
    PERMANENT_EMPLOYMENT_RATIO_HARD_MIN = 0
    PERMANENT_EMPLOYMENT_RATIO_HARD_MAX = 1
    PERMANENT_EMPLOYMENT_RATIO_WARN_LOW = 0.70
    PERMANENT_EMPLOYMENT_RATIO_WARN_HIGH = 0.95
    TEMPORARY_EMPLOYMENT_RATIO_HARD_MIN = 0
    TEMPORARY_EMPLOYMENT_RATIO_HARD_MAX = 1
    TEMPORARY_EMPLOYMENT_RATIO_WARN_LOW = 0.05
    TEMPORARY_EMPLOYMENT_RATIO_WARN_HIGH = 0.5
    EMPLOYMENT_RATIO_SUM_HARD_MIN = 0.8
    EMPLOYMENT_RATIO_SUM_HARD_MAX = 1.2
    EMPLOYMENT_RATIO_SUM_WARN_MIN = 0.7
    EMPLOYMENT_RATIO_SUM_WARN_MAX = 1.3

    # Data Quality
    MISSING_VALUES_HARD_THRESHOLD = 0.20  # <= 20% missing per column
    MISSING_VALUES_WARN_THRESHOLD = 0.50  # Warn if > 50% missing

    # Cross-Source Consistency
    AREA_CONSISTENCY_HARD_TOLERANCE = 0.10  # Within 10% difference
    AREA_CONSISTENCY_WARN_TOLERANCE = 0.20  # Warn if > 20% difference

    def __init__(self) -> None:
        """Initialize RawDataSanityChecker with configuration."""
        self.config = get_config()
        paths_config = self.config.get("paths", {})
        self.data_root = Path(paths_config.get("data_root", "./data"))
        self.neighborhoods_geojson = paths_config.get(
            "neighborhoods_geojson", "./paris_neighborhoods.geojson"
        )

        logger.info("RawDataSanityChecker initialized")

    def _normalize_neighborhood_name(self, name: str) -> str:
        """Normalize neighborhood name (lowercase, replace spaces with underscores).

        Args:
            name: Neighborhood name.

        Returns:
            Normalized name.
        """
        return name.lower().replace(" ", "_")

    def _get_osm_directory(self, neighborhood_name: str, is_compliant: bool) -> Path:
        """Get OSM data directory for neighborhood.

        Args:
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether the neighborhood is compliant.

        Returns:
            Path to OSM data directory.
        """
        normalized_name = self._normalize_neighborhood_name(neighborhood_name)
        compliance_dir = "compliant" if is_compliant else "non_compliant"
        return self.data_root / "raw" / "osm" / compliance_dir / normalized_name

    def _get_census_directory(self, neighborhood_name: str) -> Path:
        """Get Census data directory for neighborhood.

        Args:
            neighborhood_name: Name of the neighborhood.

        Returns:
            Path to Census data directory.
        """
        normalized_name = self._normalize_neighborhood_name(neighborhood_name)
        return self.data_root / "raw" / "census" / "compliant" / normalized_name

    def _check_file_structure(
        self, neighborhood_name: str, is_compliant: bool
    ) -> Dict[str, Any]:
        """Check directory structure and file existence for a neighborhood.

        Args:
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether the neighborhood is compliant.

        Returns:
            Dictionary with validation results:
            - status: "pass", "warn", or "fail"
            - issues: List of issues found
            - files_found: List of files that exist
            - files_missing: List of required files that are missing
        """
        issues = []
        files_found = []
        files_missing = []

        # Check OSM directory
        osm_dir = self._get_osm_directory(neighborhood_name, is_compliant)
        required_osm_files = [
            "services.geojson",
            "buildings.geojson",
            "network.graphml",
            "intersections.geojson",
            "pedestrian_infrastructure.geojson",
        ]

        if not osm_dir.exists():
            issues.append(f"OSM directory does not exist: {osm_dir}")
            files_missing.extend(required_osm_files)
        else:
            for filename in required_osm_files:
                file_path = osm_dir / filename
                if file_path.exists():
                    # Check file size
                    if file_path.stat().st_size == 0:
                        issues.append(f"Empty file: {file_path}")
                        files_missing.append(filename)
                    else:
                        files_found.append(str(file_path))
                else:
                    issues.append(f"Missing OSM file: {file_path}")
                    files_missing.append(filename)

        # Check Census directory (only for compliant neighborhoods)
        if is_compliant:
            census_dir = self._get_census_directory(neighborhood_name)
            census_file = census_dir / "census_data.parquet"

            if not census_dir.exists():
                issues.append(f"Census directory does not exist: {census_dir}")
                files_missing.append("census_data.parquet")
            elif not census_file.exists():
                issues.append(f"Missing Census file: {census_file}")
                files_missing.append("census_data.parquet")
            elif census_file.stat().st_size == 0:
                issues.append(f"Empty Census file: {census_file}")
                files_missing.append("census_data.parquet")
            else:
                files_found.append(str(census_file))

        # Determine status
        if files_missing:
            # If any required files are missing, it's a failure
            status = "fail"
        elif issues:
            status = "warn"
        else:
            status = "pass"

        return {
            "status": status,
            "issues": issues,
            "files_found": files_found,
            "files_missing": files_missing,
        }

    def _check_geometry_validity(
        self, gdf: gpd.GeoDataFrame, feature_name: str
    ) -> Tuple[bool, List[str]]:
        """Check geometry validity in GeoDataFrame.

        Args:
            gdf: GeoDataFrame to check.
            feature_name: Name of the feature type (for error messages).

        Returns:
            Tuple of (all_valid, issues) where all_valid is True if all geometries
            are valid, and issues is a list of error messages.
        """
        if gdf.empty:
            return True, []

        issues = []
        invalid_mask = ~gdf.geometry.is_valid
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            issues.append(
                f"{feature_name}: {invalid_count} invalid geometries found"
            )
            return False, issues

        return True, []

    def _check_coordinate_bounds(
        self, gdf: gpd.GeoDataFrame, feature_name: str
    ) -> Tuple[bool, List[str]]:
        """Check that coordinates are within Paris bounds.

        Args:
            gdf: GeoDataFrame to check.
            feature_name: Name of the feature type (for error messages).

        Returns:
            Tuple of (all_valid, issues) where all_valid is True if all coordinates
            are within bounds, and issues is a list of error messages.
        """
        if gdf.empty:
            return True, []

        issues = []
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

        if bounds[0] < self.PARIS_LON_MIN or bounds[2] > self.PARIS_LON_MAX:
            issues.append(
                f"{feature_name}: Longitude out of bounds "
                f"({bounds[0]:.4f} - {bounds[2]:.4f}, expected "
                f"{self.PARIS_LON_MIN} - {self.PARIS_LON_MAX})"
            )

        if bounds[1] < self.PARIS_LAT_MIN or bounds[3] > self.PARIS_LAT_MAX:
            issues.append(
                f"{feature_name}: Latitude out of bounds "
                f"({bounds[1]:.4f} - {bounds[3]:.4f}, expected "
                f"{self.PARIS_LAT_MIN} - {self.PARIS_LAT_MAX})"
            )

        return len(issues) == 0, issues

    def _validate_services(
        self, services_path: Path, neighborhood_name: str, area_km2: float
    ) -> Dict[str, Any]:
        """Validate services GeoJSON file.

        Args:
            services_path: Path to services.geojson file.
            neighborhood_name: Name of the neighborhood.
            area_km2: Neighborhood area in km².

        Returns:
            Dictionary with validation results.
        """
        result = {
            "status": "pass",
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        if not services_path.exists():
            result["status"] = "fail"
            result["issues"].append(f"Services file does not exist: {services_path}")
            return result

        try:
            services_gdf = gpd.read_file(services_path)
        except Exception as e:
            result["status"] = "fail"
            result["issues"].append(f"Failed to load services file: {e}")
            return result

        if services_gdf.empty:
            result["status"] = "fail"
            result["issues"].append("Services GeoDataFrame is empty")
            return result

        # Check required columns
        required_columns = ["name", "amenity", "category", "geometry", "neighborhood_name"]
        missing_columns = [col for col in required_columns if col not in services_gdf.columns]
        if missing_columns:
            result["status"] = "fail"
            result["issues"].append(f"Missing required columns: {missing_columns}")
            return result

        # Check geometry validity
        all_valid, geom_issues = self._check_geometry_validity(services_gdf, "Services")
        if not all_valid:
            result["status"] = "fail"
            result["issues"].extend(geom_issues)

        # Check coordinate bounds
        in_bounds, coord_issues = self._check_coordinate_bounds(services_gdf, "Services")
        if not in_bounds:
            result["warnings"].extend(coord_issues)

        # Check service count
        service_count = len(services_gdf)
        result["statistics"]["service_count"] = service_count

        if service_count <= self.SERVICES_MIN_COUNT:
            result["status"] = "fail"
            result["issues"].append(
                f"Service count ({service_count}) must be > {self.SERVICES_MIN_COUNT}"
            )
        elif service_count < self.SERVICES_WARN_LOW:
            result["warnings"].append(
                f"Very low service count: {service_count} (expected >= {self.SERVICES_WARN_LOW})"
            )
        elif service_count > self.SERVICES_WARN_HIGH:
            result["warnings"].append(
                f"Unusually high service count: {service_count} (expected <= {self.SERVICES_WARN_HIGH})"
            )

        # Check service density
        if area_km2 > 0:
            density = service_count / area_km2
            result["statistics"]["service_density"] = density

            if density <= self.SERVICES_DENSITY_HARD_MIN or density >= self.SERVICES_DENSITY_HARD_MAX:
                result["status"] = "fail"
                result["issues"].append(
                    f"Service density ({density:.2f}/km²) out of hard bounds "
                    f"({self.SERVICES_DENSITY_HARD_MIN} - {self.SERVICES_DENSITY_HARD_MAX})"
                )
            elif density < self.SERVICES_DENSITY_WARN_LOW or density > self.SERVICES_DENSITY_WARN_HIGH:
                result["warnings"].append(
                    f"Service density ({density:.2f}/km²) outside typical range "
                    f"({self.SERVICES_DENSITY_WARN_LOW} - {self.SERVICES_DENSITY_WARN_HIGH})"
                )

        # Check category values
        valid_categories = get_service_category_names()
        invalid_categories = services_gdf[~services_gdf["category"].isin(valid_categories)]["category"].unique()
        if len(invalid_categories) > 0:
            result["status"] = "fail"
            result["issues"].append(
                f"Invalid category values: {list(invalid_categories)}"
            )

        # Check all 8 categories present (warning if any missing)
        present_categories = services_gdf["category"].unique()
        missing_categories = set(valid_categories) - set(present_categories)
        if missing_categories:
            result["warnings"].append(
                f"Missing service categories: {list(missing_categories)}"
            )

        # Update status if warnings exist
        if result["warnings"] and result["status"] == "pass":
            result["status"] = "warn"

        return result

    def _validate_buildings(
        self, buildings_path: Path, neighborhood_name: str, area_km2: float
    ) -> Dict[str, Any]:
        """Validate buildings GeoJSON file.

        Args:
            buildings_path: Path to buildings.geojson file.
            neighborhood_name: Name of the neighborhood.
            area_km2: Neighborhood area in km².

        Returns:
            Dictionary with validation results.
        """
        result = {
            "status": "pass",
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        if not buildings_path.exists():
            result["status"] = "fail"
            result["issues"].append(f"Buildings file does not exist: {buildings_path}")
            return result

        try:
            buildings_gdf = gpd.read_file(buildings_path)
        except Exception as e:
            result["status"] = "fail"
            result["issues"].append(f"Failed to load buildings file: {e}")
            return result

        if buildings_gdf.empty:
            result["status"] = "fail"
            result["issues"].append("Buildings GeoDataFrame is empty")
            return result

        # Check geometry validity
        all_valid, geom_issues = self._check_geometry_validity(buildings_gdf, "Buildings")
        if not all_valid:
            result["status"] = "fail"
            result["issues"].extend(geom_issues)

        # Check coordinate bounds
        in_bounds, coord_issues = self._check_coordinate_bounds(buildings_gdf, "Buildings")
        if not in_bounds:
            result["warnings"].extend(coord_issues)

        # Check building count
        building_count = len(buildings_gdf)
        result["statistics"]["building_count"] = building_count

        if building_count <= self.BUILDINGS_MIN_COUNT:
            result["status"] = "fail"
            result["issues"].append(
                f"Building count ({building_count}) must be > {self.BUILDINGS_MIN_COUNT}"
            )
        elif building_count < self.BUILDINGS_WARN_LOW:
            result["warnings"].append(
                f"Very low building count: {building_count} (expected >= {self.BUILDINGS_WARN_LOW})"
            )
        elif building_count > self.BUILDINGS_WARN_HIGH:
            result["warnings"].append(
                f"Unusually high building count: {building_count} (expected <= {self.BUILDINGS_WARN_HIGH})"
            )

        # Check building density
        if area_km2 > 0:
            density = building_count / area_km2
            result["statistics"]["building_density"] = density

            if density <= self.BUILDINGS_DENSITY_HARD_MIN or density >= self.BUILDINGS_DENSITY_HARD_MAX:
                result["status"] = "fail"
                result["issues"].append(
                    f"Building density ({density:.2f}/km²) out of hard bounds "
                    f"({self.BUILDINGS_DENSITY_HARD_MIN} - {self.BUILDINGS_DENSITY_HARD_MAX})"
                )
            elif density < self.BUILDINGS_DENSITY_WARN_LOW or density > self.BUILDINGS_DENSITY_WARN_HIGH:
                result["warnings"].append(
                    f"Building density ({density:.2f}/km²) outside typical range "
                    f"({self.BUILDINGS_DENSITY_WARN_LOW} - {self.BUILDINGS_DENSITY_WARN_HIGH})"
                )

        # Check building levels if column exists
        if "building:levels" in buildings_gdf.columns:
            levels_series = pd.to_numeric(buildings_gdf["building:levels"], errors="coerce")
            valid_levels = levels_series.dropna()

            if len(valid_levels) > 0:
                result["statistics"]["avg_building_levels"] = float(valid_levels.mean())
                result["statistics"]["max_building_levels"] = float(valid_levels.max())

                # Check levels bounds
                if (valid_levels < self.BUILDING_LEVELS_HARD_MIN).any() or (
                    valid_levels > self.BUILDING_LEVELS_HARD_MAX
                ).any():
                    result["status"] = "fail"
                    result["issues"].append(
                        f"Building levels out of hard bounds "
                        f"({self.BUILDING_LEVELS_HARD_MIN} - {self.BUILDING_LEVELS_HARD_MAX})"
                    )
                elif (valid_levels > self.BUILDING_LEVELS_WARN_MAX).any():
                    result["warnings"].append(
                        f"Some buildings have > {self.BUILDING_LEVELS_WARN_MAX} levels "
                        f"(max: {valid_levels.max()})"
                    )

                # Check missing levels
                missing_ratio = (levels_series.isna().sum() / len(buildings_gdf))
                result["statistics"]["missing_levels_ratio"] = float(missing_ratio)
                if missing_ratio > self.BUILDING_LEVELS_MISSING_WARN_THRESHOLD:
                    result["warnings"].append(
                        f"High missing levels ratio: {missing_ratio:.2%} "
                        f"(threshold: {self.BUILDING_LEVELS_MISSING_WARN_THRESHOLD:.2%})"
                    )

        # Check building area if column exists
        if "area_m2" in buildings_gdf.columns:
            area_series = pd.to_numeric(buildings_gdf["area_m2"], errors="coerce")
            valid_areas = area_series.dropna()

            if len(valid_areas) > 0:
                result["statistics"]["avg_building_area"] = float(valid_areas.mean())
                result["statistics"]["max_building_area"] = float(valid_areas.max())

                # Check area bounds
                if (valid_areas <= self.BUILDING_AREA_HARD_MIN).any() or (
                    valid_areas >= self.BUILDING_AREA_HARD_MAX
                ).any():
                    result["status"] = "fail"
                    result["issues"].append(
                        f"Building area out of hard bounds "
                        f"({self.BUILDING_AREA_HARD_MIN} - {self.BUILDING_AREA_HARD_MAX} m²)"
                    )
                elif (valid_areas > self.BUILDING_AREA_WARN_MAX).any():
                    result["warnings"].append(
                        f"Some buildings have area > {self.BUILDING_AREA_WARN_MAX} m² "
                        f"(max: {valid_areas.max():.0f} m²)"
                    )

                # Check total building area ratio
                total_building_area = valid_areas.sum() / 1_000_000  # Convert to km²
                if area_km2 > 0:
                    area_ratio = total_building_area / area_km2
                    result["statistics"]["building_area_ratio"] = float(area_ratio)

                    if (
                        area_ratio <= self.BUILDING_AREA_RATIO_HARD_MIN
                        or area_ratio >= self.BUILDING_AREA_RATIO_HARD_MAX
                    ):
                        result["status"] = "fail"
                        result["issues"].append(
                            f"Building area ratio ({area_ratio:.2%}) out of hard bounds "
                            f"({self.BUILDING_AREA_RATIO_HARD_MIN:.2%} - {self.BUILDING_AREA_RATIO_HARD_MAX:.2%})"
                        )
                    elif (
                        area_ratio < self.BUILDING_AREA_RATIO_WARN_LOW
                        or area_ratio > self.BUILDING_AREA_RATIO_WARN_HIGH
                    ):
                        result["warnings"].append(
                            f"Building area ratio ({area_ratio:.2%}) outside typical range "
                            f"({self.BUILDING_AREA_RATIO_WARN_LOW:.2%} - {self.BUILDING_AREA_RATIO_WARN_HIGH:.2%})"
                        )

        # Update status if warnings exist
        if result["warnings"] and result["status"] == "pass":
            result["status"] = "warn"

        return result

    def _validate_network(
        self, network_path: Path, neighborhood_name: str, area_km2: float
    ) -> Dict[str, Any]:
        """Validate network GraphML file.

        Args:
            network_path: Path to network.graphml file.
            neighborhood_name: Name of the neighborhood.
            area_km2: Neighborhood area in km².

        Returns:
            Dictionary with validation results.
        """
        result = {
            "status": "pass",
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        if not network_path.exists():
            result["status"] = "fail"
            result["issues"].append(f"Network file does not exist: {network_path}")
            return result

        try:
            G = ox.load_graphml(network_path)
        except Exception as e:
            result["status"] = "fail"
            result["issues"].append(f"Failed to load network file: {e}")
            return result

        if len(G.nodes()) == 0:
            result["status"] = "fail"
            result["issues"].append("Network graph is empty (no nodes)")
            return result

        # Get node and edge counts
        node_count = len(G.nodes())
        edge_count = len(G.edges())
        result["statistics"]["node_count"] = node_count
        result["statistics"]["edge_count"] = edge_count

        # Check node count
        if node_count <= self.NETWORK_NODES_MIN:
            result["status"] = "fail"
            result["issues"].append(
                f"Node count ({node_count}) must be > {self.NETWORK_NODES_MIN}"
            )
        elif node_count < self.NETWORK_NODES_WARN_LOW:
            result["warnings"].append(
                f"Very low node count: {node_count} (expected >= {self.NETWORK_NODES_WARN_LOW})"
            )

        # Check edge count
        if edge_count <= self.NETWORK_EDGES_MIN:
            result["status"] = "fail"
            result["issues"].append(
                f"Edge count ({edge_count}) must be > {self.NETWORK_EDGES_MIN}"
            )
        elif edge_count < self.NETWORK_EDGES_WARN_LOW:
            result["warnings"].append(
                f"Very low edge count: {edge_count} (expected >= {self.NETWORK_EDGES_WARN_LOW})"
            )

        # Check node-to-edge ratio
        if node_count > 0:
            ratio = edge_count / node_count
            result["statistics"]["node_edge_ratio"] = float(ratio)

            if (
                ratio <= self.NETWORK_NODE_EDGE_RATIO_HARD_MIN
                or ratio >= self.NETWORK_NODE_EDGE_RATIO_HARD_MAX
            ):
                result["status"] = "fail"
                result["issues"].append(
                    f"Node-to-edge ratio ({ratio:.2f}) out of hard bounds "
                    f"({self.NETWORK_NODE_EDGE_RATIO_HARD_MIN} - {self.NETWORK_NODE_EDGE_RATIO_HARD_MAX})"
                )
            elif (
                ratio < self.NETWORK_NODE_EDGE_RATIO_WARN_LOW
                or ratio > self.NETWORK_NODE_EDGE_RATIO_WARN_HIGH
            ):
                result["warnings"].append(
                    f"Node-to-edge ratio ({ratio:.2f}) outside typical range "
                    f"({self.NETWORK_NODE_EDGE_RATIO_WARN_LOW} - {self.NETWORK_NODE_EDGE_RATIO_WARN_HIGH})"
                )

        # Check network connectivity
        if node_count > 0:
            largest_component = max(nx.connected_components(G.to_undirected()), key=len)
            connectivity_ratio = len(largest_component) / node_count
            result["statistics"]["connectivity_ratio"] = float(connectivity_ratio)

            if connectivity_ratio < self.NETWORK_CONNECTIVITY_HARD_MIN:
                result["status"] = "fail"
                result["issues"].append(
                    f"Network connectivity ({connectivity_ratio:.2%}) below hard threshold "
                    f"({self.NETWORK_CONNECTIVITY_HARD_MIN:.2%})"
                )
            elif connectivity_ratio < self.NETWORK_CONNECTIVITY_WARN_MIN:
                result["warnings"].append(
                    f"Low network connectivity ({connectivity_ratio:.2%}) "
                    f"(expected >= {self.NETWORK_CONNECTIVITY_WARN_MIN:.2%})"
                )

        # Check edge lengths
        try:
            edges_gdf, nodes_gdf = ox.graph_to_gdfs(G)
            if not edges_gdf.empty and "length" in edges_gdf.columns:
                lengths = edges_gdf["length"]
                result["statistics"]["avg_edge_length"] = float(lengths.mean())
                result["statistics"]["max_edge_length"] = float(lengths.max())

                if (lengths <= self.EDGE_LENGTH_HARD_MIN).any() or (
                    lengths >= self.EDGE_LENGTH_HARD_MAX
                ).any():
                    result["status"] = "fail"
                    result["issues"].append(
                        f"Edge length out of hard bounds "
                        f"({self.EDGE_LENGTH_HARD_MIN} - {self.EDGE_LENGTH_HARD_MAX} m)"
                    )
                elif (lengths < self.EDGE_LENGTH_WARN_MIN).any() or (
                    lengths > self.EDGE_LENGTH_WARN_MAX
                ).any():
                    result["warnings"].append(
                        f"Some edges have unusual lengths "
                        f"(min: {lengths.min():.1f} m, max: {lengths.max():.1f} m)"
                    )

                # Calculate average block length (simplified: average edge length)
                # In a real implementation, this would calculate actual block perimeters
                avg_block_length = float(lengths.mean())
                result["statistics"]["avg_block_length"] = avg_block_length

                if (
                    avg_block_length < self.AVG_BLOCK_LENGTH_HARD_MIN
                    or avg_block_length > self.AVG_BLOCK_LENGTH_HARD_MAX
                ):
                    result["status"] = "fail"
                    result["issues"].append(
                        f"Average block length ({avg_block_length:.1f} m) out of hard bounds "
                        f"({self.AVG_BLOCK_LENGTH_HARD_MIN} - {self.AVG_BLOCK_LENGTH_HARD_MAX} m)"
                    )
                elif (
                    avg_block_length < self.AVG_BLOCK_LENGTH_WARN_MIN
                    or avg_block_length > self.AVG_BLOCK_LENGTH_WARN_MAX
                ):
                    result["warnings"].append(
                        f"Average block length ({avg_block_length:.1f} m) outside typical range "
                        f"({self.AVG_BLOCK_LENGTH_WARN_MIN} - {self.AVG_BLOCK_LENGTH_WARN_MAX} m)"
                    )

            # Check node coordinate bounds
            if not nodes_gdf.empty:
                in_bounds, coord_issues = self._check_coordinate_bounds(nodes_gdf, "Network nodes")
                if not in_bounds:
                    result["warnings"].extend(coord_issues)
        except Exception as e:
            result["warnings"].append(f"Could not analyze edge lengths: {e}")

        # Update status if warnings exist
        if result["warnings"] and result["status"] == "pass":
            result["status"] = "warn"

        return result

    def _validate_intersections(
        self, intersections_path: Path, neighborhood_name: str, area_km2: float
    ) -> Dict[str, Any]:
        """Validate intersections GeoJSON file.

        Args:
            intersections_path: Path to intersections.geojson file.
            neighborhood_name: Name of the neighborhood.
            area_km2: Neighborhood area in km².

        Returns:
            Dictionary with validation results.
        """
        result = {
            "status": "pass",
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        if not intersections_path.exists():
            result["status"] = "fail"
            result["issues"].append(f"Intersections file does not exist: {intersections_path}")
            return result

        try:
            intersections_gdf = gpd.read_file(intersections_path)
        except Exception as e:
            result["status"] = "fail"
            result["issues"].append(f"Failed to load intersections file: {e}")
            return result

        if intersections_gdf.empty:
            result["status"] = "fail"
            result["issues"].append("Intersections GeoDataFrame is empty")
            return result

        # Check geometry validity
        all_valid, geom_issues = self._check_geometry_validity(intersections_gdf, "Intersections")
        if not all_valid:
            result["status"] = "fail"
            result["issues"].extend(geom_issues)

        # Check coordinate bounds
        in_bounds, coord_issues = self._check_coordinate_bounds(intersections_gdf, "Intersections")
        if not in_bounds:
            result["warnings"].extend(coord_issues)

        # Check intersection count
        intersection_count = len(intersections_gdf)
        result["statistics"]["intersection_count"] = intersection_count

        if intersection_count <= self.INTERSECTIONS_MIN_COUNT:
            result["status"] = "fail"
            result["issues"].append(
                f"Intersection count ({intersection_count}) must be > {self.INTERSECTIONS_MIN_COUNT}"
            )
        elif intersection_count < self.INTERSECTIONS_WARN_LOW:
            result["warnings"].append(
                f"Very low intersection count: {intersection_count} "
                f"(expected >= {self.INTERSECTIONS_WARN_LOW})"
            )

        # Check intersection density
        if area_km2 > 0:
            density = intersection_count / area_km2
            result["statistics"]["intersection_density"] = density

            if (
                density < self.INTERSECTIONS_DENSITY_HARD_MIN
                or density > self.INTERSECTIONS_DENSITY_HARD_MAX
            ):
                result["status"] = "fail"
                result["issues"].append(
                    f"Intersection density ({density:.2f}/km²) out of hard bounds "
                    f"({self.INTERSECTIONS_DENSITY_HARD_MIN} - {self.INTERSECTIONS_DENSITY_HARD_MAX})"
                )
            elif (
                density < self.INTERSECTIONS_DENSITY_WARN_LOW
                or density > self.INTERSECTIONS_DENSITY_WARN_HIGH
            ):
                result["warnings"].append(
                    f"Intersection density ({density:.2f}/km²) outside typical range "
                    f"({self.INTERSECTIONS_DENSITY_WARN_LOW} - {self.INTERSECTIONS_DENSITY_WARN_HIGH})"
                )

        # Check node degree if column exists
        if "degree" in intersections_gdf.columns:
            degree_series = pd.to_numeric(intersections_gdf["degree"], errors="coerce")
            valid_degrees = degree_series.dropna()

            if len(valid_degrees) > 0:
                result["statistics"]["avg_degree"] = float(valid_degrees.mean())
                result["statistics"]["min_degree"] = float(valid_degrees.min())
                result["statistics"]["max_degree"] = float(valid_degrees.max())

                if (
                    (valid_degrees < self.INTERSECTION_DEGREE_HARD_MIN).any()
                    or (valid_degrees > self.INTERSECTION_DEGREE_HARD_MAX).any()
                ):
                    result["status"] = "fail"
                    result["issues"].append(
                        f"Intersection degree out of hard bounds "
                        f"({self.INTERSECTION_DEGREE_HARD_MIN} - {self.INTERSECTION_DEGREE_HARD_MAX})"
                    )
                elif (valid_degrees > self.INTERSECTION_DEGREE_WARN_MAX).any():
                    result["warnings"].append(
                        f"Some intersections have degree > {self.INTERSECTION_DEGREE_WARN_MAX} "
                        f"(max: {valid_degrees.max()})"
                    )

        # Update status if warnings exist
        if result["warnings"] and result["status"] == "pass":
            result["status"] = "warn"

        return result

    def _validate_pedestrian_features(
        self, pedestrian_path: Path, neighborhood_name: str
    ) -> Dict[str, Any]:
        """Validate pedestrian features GeoJSON file.

        Args:
            pedestrian_path: Path to pedestrian_infrastructure.geojson file.
            neighborhood_name: Name of the neighborhood.

        Returns:
            Dictionary with validation results.
        """
        result = {
            "status": "pass",
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        if not pedestrian_path.exists():
            # Pedestrian features are optional, so missing file is just a warning
            result["warnings"].append(f"Pedestrian features file does not exist: {pedestrian_path}")
            result["statistics"]["pedestrian_count"] = 0
            return result

        try:
            pedestrian_gdf = gpd.read_file(pedestrian_path)
        except Exception as e:
            result["warnings"].append(f"Failed to load pedestrian features file: {e}")
            return result

        pedestrian_count = len(pedestrian_gdf)
        result["statistics"]["pedestrian_count"] = pedestrian_count

        if pedestrian_gdf.empty:
            # Empty is OK (OSM may not tag all pedestrian features)
            return result

        # Check geometry validity
        all_valid, geom_issues = self._check_geometry_validity(pedestrian_gdf, "Pedestrian features")
        if not all_valid:
            result["status"] = "fail"
            result["issues"].extend(geom_issues)

        # Check coordinate bounds
        in_bounds, coord_issues = self._check_coordinate_bounds(pedestrian_gdf, "Pedestrian features")
        if not in_bounds:
            result["warnings"].extend(coord_issues)

        # Check feature types if column exists
        if "feature_type" in pedestrian_gdf.columns:
            valid_types = ["pedestrian_way", "sidewalk", "crosswalk"]
            invalid_types = pedestrian_gdf[~pedestrian_gdf["feature_type"].isin(valid_types)]["feature_type"].unique()
            if len(invalid_types) > 0:
                result["status"] = "fail"
                result["issues"].append(f"Invalid feature types: {list(invalid_types)}")

        # Update status if warnings exist
        if result["warnings"] and result["status"] == "pass":
            result["status"] = "warn"

        return result
    def _validate_census_data(
        self, census_path: Path, neighborhood_name: str
    ) -> Dict[str, Any]:
        """Validate Census Parquet file with all demographic features.

        Args:
            census_path: Path to census_data.parquet file.
            neighborhood_name: Name of the neighborhood.

        Returns:
            Dictionary with validation results.
        """
        result = {
            "status": "pass",
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        if not census_path.exists():
            result["status"] = "fail"
            result["issues"].append(f"Census file does not exist: {census_path}")
            return result

        try:
            census_df = pd.read_parquet(census_path)
        except Exception as e:
            result["status"] = "fail"
            result["issues"].append(f"Failed to load census file: {e}")
            return result

        if census_df.empty:
            result["status"] = "fail"
            result["issues"].append("Census DataFrame is empty")
            return result

        # Check for required neighborhood_name column
        if "neighborhood_name" not in census_df.columns:
            result["status"] = "fail"
            result["issues"].append("Missing required column: neighborhood_name")
            return result

        # Filter to this neighborhood if multiple rows
        neighborhood_data = census_df[census_df["neighborhood_name"] == neighborhood_name]
        if neighborhood_data.empty:
            result["status"] = "fail"
            result["issues"].append(f"No data found for neighborhood: {neighborhood_name}")
            return result

        # Check for IRIS code columns (IRIS-level data)
        iris_code_cols = [col for col in neighborhood_data.columns 
                         if col in ["IRIS", "CODE_IRIS", "DCOMIRIS", "CODEGEO"]]
        
        if iris_code_cols:
            result["statistics"]["iris_units_count"] = len(neighborhood_data)
            result["statistics"]["iris_code_column"] = iris_code_cols[0]
            # Validate all IRIS units (not just first row)
            # We'll check each IRIS unit and aggregate statistics
        else:
            # Legacy format: single row per neighborhood (aggregated data)
            result["warnings"].append("No IRIS code columns found - data appears to be aggregated (legacy format)")
        
        # For validation, we'll check all rows (IRIS units) and report aggregate statistics
        # Use first row for single-value checks, but validate all rows

        # Check for NaN/Inf values across all IRIS units
        numeric_cols = neighborhood_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["IRIS", "CODE_IRIS", "DCOMIRIS", "CODEGEO"]:  # Skip IRIS code columns
                nan_count = neighborhood_data[col].isna().sum()
                inf_count = np.isinf(neighborhood_data[col]).sum() if col in neighborhood_data.columns else 0
                if nan_count > 0:
                    result["warnings"].append(
                        f"NaN values in column '{col}': {nan_count}/{len(neighborhood_data)} IRIS units"
                    )
                if inf_count > 0:
                    result["status"] = "fail"
                    result["issues"].append(
                        f"Inf values in column '{col}': {inf_count}/{len(neighborhood_data)} IRIS units"
                    )

        # Check missing values per column (across all IRIS units)
        missing_ratios = {}
        for col in neighborhood_data.columns:
            if col not in ["neighborhood_name", "IRIS", "CODE_IRIS", "DCOMIRIS", "CODEGEO"]:
                missing_count = neighborhood_data[col].isna().sum()
                missing_ratio = missing_count / len(neighborhood_data)
                missing_ratios[col] = missing_ratio

                if missing_ratio > self.MISSING_VALUES_HARD_THRESHOLD:
                    result["status"] = "fail"
                    result["issues"].append(
                        f"Column '{col}' has {missing_ratio:.2%} missing values "
                        f"({missing_count}/{len(neighborhood_data)} IRIS units, "
                        f"threshold: {self.MISSING_VALUES_HARD_THRESHOLD:.2%})"
                    )
                elif missing_ratio > self.MISSING_VALUES_WARN_THRESHOLD:
                    result["warnings"].append(
                        f"Column '{col}' has {missing_ratio:.2%} missing values "
                        f"({missing_count}/{len(neighborhood_data)} IRIS units, "
                        f"threshold: {self.MISSING_VALUES_WARN_THRESHOLD:.2%})"
                    )

        # Validate population density (check all IRIS units)
        if "population_density" in neighborhood_data.columns:
            pop_densities = neighborhood_data["population_density"].dropna()
            if len(pop_densities) > 0:
                # Store aggregate statistics
                result["statistics"]["population_density"] = {
                    "mean": float(pop_densities.mean()),
                    "min": float(pop_densities.min()),
                    "max": float(pop_densities.max()),
                    "count": len(pop_densities)
                }
                
                # Check each IRIS unit
                out_of_bounds = pop_densities[
                    (pop_densities < self.POPULATION_DENSITY_HARD_MIN) |
                    (pop_densities > self.POPULATION_DENSITY_HARD_MAX)
                ]
                if len(out_of_bounds) > 0:
                    result["status"] = "fail"
                    result["issues"].append(
                        f"Population density out of hard bounds for {len(out_of_bounds)}/{len(pop_densities)} IRIS units "
                        f"(range: {out_of_bounds.min():.0f} - {out_of_bounds.max():.0f}/km², "
                        f"expected: {self.POPULATION_DENSITY_HARD_MIN} - {self.POPULATION_DENSITY_HARD_MAX})"
                    )
                
                # Check warning bounds
                outside_typical = pop_densities[
                    (pop_densities < self.POPULATION_DENSITY_WARN_LOW) |
                    (pop_densities > self.POPULATION_DENSITY_WARN_HIGH)
                ]
                if len(outside_typical) > 0:
                    result["warnings"].append(
                        f"Population density outside typical range for {len(outside_typical)}/{len(pop_densities)} IRIS units "
                        f"(range: {outside_typical.min():.0f} - {outside_typical.max():.0f}/km², "
                        f"typical: {self.POPULATION_DENSITY_WARN_LOW} - {self.POPULATION_DENSITY_WARN_HIGH})"
                    )

        # Validate demographic ratios
        ratio_features = {
            "children_per_capita": (self.CHILDREN_PER_CAPITA_HARD_MIN, self.CHILDREN_PER_CAPITA_HARD_MAX,
                                   self.CHILDREN_PER_CAPITA_WARN_LOW, self.CHILDREN_PER_CAPITA_WARN_HIGH),
            "elderly_ratio": (self.ELDERLY_RATIO_HARD_MIN, self.ELDERLY_RATIO_HARD_MAX,
                             self.ELDERLY_RATIO_WARN_LOW, self.ELDERLY_RATIO_WARN_HIGH),
            "working_age_ratio": (0.0, 1.0, 0.6, 0.8),  # Working age (15-64) typically 60-80% of population
            "ses_index": (self.SES_INDEX_HARD_MIN, self.SES_INDEX_HARD_MAX,
                         self.SES_INDEX_WARN_LOW, self.SES_INDEX_WARN_HIGH),
            "unemployment_rate": (self.UNEMPLOYMENT_RATE_HARD_MIN, self.UNEMPLOYMENT_RATE_HARD_MAX,
                                 self.UNEMPLOYMENT_RATE_WARN_LOW, self.UNEMPLOYMENT_RATE_WARN_HIGH),
            "poverty_rate": (self.POVERTY_RATE_HARD_MIN, self.POVERTY_RATE_HARD_MAX,
                            self.POVERTY_RATE_WARN_LOW, self.POVERTY_RATE_WARN_HIGH),
            "car_ownership_rate": (self.CAR_OWNERSHIP_RATE_HARD_MIN, self.CAR_OWNERSHIP_RATE_HARD_MAX,
                                  self.CAR_OWNERSHIP_RATE_WARN_LOW, self.CAR_OWNERSHIP_RATE_WARN_HIGH),
            "car_commute_ratio": (self.CAR_COMMUTE_RATIO_HARD_MIN, self.CAR_COMMUTE_RATIO_HARD_MAX,
                                 self.CAR_COMMUTE_RATIO_WARN_LOW, self.CAR_COMMUTE_RATIO_WARN_HIGH),
            "walking_ratio": (self.WALKING_RATIO_HARD_MIN, self.WALKING_RATIO_HARD_MAX,
                             self.WALKING_RATIO_WARN_LOW, self.WALKING_RATIO_WARN_HIGH),
            "cycling_ratio": (self.CYCLING_RATIO_HARD_MIN, self.CYCLING_RATIO_HARD_MAX,
                             self.CYCLING_RATIO_WARN_LOW, self.CYCLING_RATIO_WARN_HIGH),
            "public_transport_ratio": (self.PUBLIC_TRANSPORT_RATIO_HARD_MIN, self.PUBLIC_TRANSPORT_RATIO_HARD_MAX,
                                      self.PUBLIC_TRANSPORT_RATIO_WARN_LOW, self.PUBLIC_TRANSPORT_RATIO_WARN_HIGH),
            "two_wheelers_ratio": (self.TWO_WHEELERS_RATIO_HARD_MIN, self.TWO_WHEELERS_RATIO_HARD_MAX,
                                  self.TWO_WHEELERS_RATIO_WARN_LOW, self.TWO_WHEELERS_RATIO_WARN_HIGH),
            "student_ratio": (self.STUDENT_RATIO_HARD_MIN, self.STUDENT_RATIO_HARD_MAX,
                             self.STUDENT_RATIO_WARN_LOW, self.STUDENT_RATIO_WARN_HIGH),
            "retired_ratio": (self.RETIRED_RATIO_HARD_MIN, self.RETIRED_RATIO_HARD_MAX,
                             self.RETIRED_RATIO_WARN_LOW, self.RETIRED_RATIO_WARN_HIGH),
            "permanent_employment_ratio": (self.PERMANENT_EMPLOYMENT_RATIO_HARD_MIN, self.PERMANENT_EMPLOYMENT_RATIO_HARD_MAX,
                                          self.PERMANENT_EMPLOYMENT_RATIO_WARN_LOW, self.PERMANENT_EMPLOYMENT_RATIO_WARN_HIGH),
            "temporary_employment_ratio": (self.TEMPORARY_EMPLOYMENT_RATIO_HARD_MIN, self.TEMPORARY_EMPLOYMENT_RATIO_HARD_MAX,
                                          self.TEMPORARY_EMPLOYMENT_RATIO_WARN_LOW, self.TEMPORARY_EMPLOYMENT_RATIO_WARN_HIGH),
        }

        # Validate ratio features (check all IRIS units)
        for feature_name, (hard_min, hard_max, warn_low, warn_high) in ratio_features.items():
            if feature_name in neighborhood_data.columns:
                values = neighborhood_data[feature_name].dropna()
                if len(values) > 0:
                    # Convert poverty_rate from percentage to ratio (divide by 100)
                    if feature_name == "poverty_rate":
                        values = values / 100.0
                    
                    # Store aggregate statistics
                    result["statistics"][feature_name] = {
                        "mean": float(values.mean()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "count": len(values)
                    }

                    # Check for negative values
                    negative_count = (values < 0).sum()
                    if negative_count > 0:
                        result["status"] = "fail"
                        result["issues"].append(
                            f"{feature_name} is negative for {negative_count}/{len(values)} IRIS units "
                            f"(min: {values.min():.4f})"
                        )

                    # Check hard bounds
                    out_of_bounds = values[(values < hard_min) | (values > hard_max)]
                    if len(out_of_bounds) > 0:
                        result["status"] = "fail"
                        result["issues"].append(
                            f"{feature_name} out of hard bounds for {len(out_of_bounds)}/{len(values)} IRIS units "
                            f"(range: {out_of_bounds.min():.4f} - {out_of_bounds.max():.4f}, "
                            f"expected: {hard_min} - {hard_max})"
                        )
                    
                    # Check warning bounds
                    outside_typical = values[(values < warn_low) | (values > warn_high)]
                    if len(outside_typical) > 0:
                        result["warnings"].append(
                            f"{feature_name} outside typical range for {len(outside_typical)}/{len(values)} IRIS units "
                            f"(range: {outside_typical.min():.4f} - {outside_typical.max():.4f}, "
                            f"typical: {warn_low} - {warn_high})"
                        )

        # Validate median income (check all IRIS units)
        if "median_income" in neighborhood_data.columns:
            incomes = neighborhood_data["median_income"].dropna()
            if len(incomes) > 0:
                result["statistics"]["median_income"] = {
                    "mean": float(incomes.mean()),
                    "min": float(incomes.min()),
                    "max": float(incomes.max()),
                    "count": len(incomes)
                }
                
                out_of_bounds = incomes[
                    (incomes < self.MEDIAN_INCOME_HARD_MIN) |
                    (incomes > self.MEDIAN_INCOME_HARD_MAX)
                ]
                if len(out_of_bounds) > 0:
                    result["status"] = "fail"
                    result["issues"].append(
                        f"Median income out of hard bounds for {len(out_of_bounds)}/{len(incomes)} IRIS units "
                        f"(range: {out_of_bounds.min():.0f} - {out_of_bounds.max():.0f} €/year, "
                        f"expected: {self.MEDIAN_INCOME_HARD_MIN} - {self.MEDIAN_INCOME_HARD_MAX})"
                    )
                
                outside_typical = incomes[
                    (incomes < self.MEDIAN_INCOME_WARN_LOW) |
                    (incomes > self.MEDIAN_INCOME_WARN_HIGH)
                ]
                if len(outside_typical) > 0:
                    result["warnings"].append(
                        f"Median income outside typical range for {len(outside_typical)}/{len(incomes)} IRIS units "
                        f"(range: {outside_typical.min():.0f} - {outside_typical.max():.0f} €/year, "
                        f"typical: {self.MEDIAN_INCOME_WARN_LOW} - {self.MEDIAN_INCOME_WARN_HIGH})"
                    )

        # Check age ratio sum (check all IRIS units)
        if all(col in neighborhood_data.columns for col in ["children_per_capita", "elderly_ratio", "working_age_ratio"]):
            children = neighborhood_data["children_per_capita"].fillna(0)
            elderly = neighborhood_data["elderly_ratio"].fillna(0)
            working_age = neighborhood_data["working_age_ratio"].fillna(0)
            age_sums = children + elderly + working_age
            
            out_of_bounds = age_sums[
                (age_sums < self.AGE_RATIO_SUM_HARD_MIN) |
                (age_sums > self.AGE_RATIO_SUM_HARD_MAX)
            ]
            if len(out_of_bounds) > 0:
                result["warnings"].append(
                    f"Age ratio sum outside expected range for {len(out_of_bounds)}/{len(age_sums)} IRIS units "
                    f"(range: {out_of_bounds.min():.4f} - {out_of_bounds.max():.4f}, "
                    f"expected: {self.AGE_RATIO_SUM_HARD_MIN} - {self.AGE_RATIO_SUM_HARD_MAX})"
                )
            
            outside_typical = age_sums[
                (age_sums < self.AGE_RATIO_SUM_WARN_MIN) |
                (age_sums > self.AGE_RATIO_SUM_WARN_MAX)
            ]
            if len(outside_typical) > 0:
                result["warnings"].append(
                    f"Age ratio sum may be inconsistent for {len(outside_typical)}/{len(age_sums)} IRIS units "
                    f"(range: {outside_typical.min():.4f} - {outside_typical.max():.4f}, expected ~1.0)"
                )

        # Check mode share sum (check all IRIS units)
        mode_share_cols = ["walking_ratio", "cycling_ratio", "public_transport_ratio",
                          "two_wheelers_ratio", "car_commute_ratio"]
        if all(col in neighborhood_data.columns for col in mode_share_cols):
            mode_sums = sum(
                neighborhood_data[col].fillna(0)
                for col in mode_share_cols
            )
            
            out_of_bounds = mode_sums[
                (mode_sums < self.MODE_SHARE_SUM_HARD_MIN) |
                (mode_sums > self.MODE_SHARE_SUM_HARD_MAX)
            ]
            if len(out_of_bounds) > 0:
                result["warnings"].append(
                    f"Mode share sum outside expected range for {len(out_of_bounds)}/{len(mode_sums)} IRIS units "
                    f"(range: {out_of_bounds.min():.4f} - {out_of_bounds.max():.4f}, "
                    f"expected: {self.MODE_SHARE_SUM_HARD_MIN} - {self.MODE_SHARE_SUM_HARD_MAX})"
                )
            
            outside_typical = mode_sums[
                (mode_sums < self.MODE_SHARE_SUM_WARN_MIN) |
                (mode_sums > self.MODE_SHARE_SUM_WARN_MAX)
            ]
            if len(outside_typical) > 0:
                result["warnings"].append(
                    f"Mode share sum may be inconsistent for {len(outside_typical)}/{len(mode_sums)} IRIS units "
                    f"(range: {outside_typical.min():.4f} - {outside_typical.max():.4f}, expected ~1.0)"
                )

        # Check employment ratio sum (check all IRIS units)
        if all(col in neighborhood_data.columns for col in ["permanent_employment_ratio", "temporary_employment_ratio"]):
            perm = neighborhood_data["permanent_employment_ratio"].fillna(0)
            temp = neighborhood_data["temporary_employment_ratio"].fillna(0)
            emp_sums = perm + temp
            
            out_of_bounds = emp_sums[
                (emp_sums < self.EMPLOYMENT_RATIO_SUM_HARD_MIN) |
                (emp_sums > self.EMPLOYMENT_RATIO_SUM_HARD_MAX)
            ]
            if len(out_of_bounds) > 0:
                result["warnings"].append(
                    f"Employment ratio sum outside expected range for {len(out_of_bounds)}/{len(emp_sums)} IRIS units "
                    f"(range: {out_of_bounds.min():.4f} - {out_of_bounds.max():.4f}, "
                    f"expected: {self.EMPLOYMENT_RATIO_SUM_HARD_MIN} - {self.EMPLOYMENT_RATIO_SUM_HARD_MAX})"
                )
            
            outside_typical = emp_sums[
                (emp_sums < self.EMPLOYMENT_RATIO_SUM_WARN_MIN) |
                (emp_sums > self.EMPLOYMENT_RATIO_SUM_WARN_MAX)
            ]
            if len(outside_typical) > 0:
                result["warnings"].append(
                    f"Employment ratio sum may be inconsistent for {len(outside_typical)}/{len(emp_sums)} IRIS units "
                    f"(range: {outside_typical.min():.4f} - {outside_typical.max():.4f}, expected ~1.0)"
                )

        # Update status if warnings exist
        if result["warnings"] and result["status"] == "pass":
            result["status"] = "warn"

        return result

    def _check_cross_source_consistency(
        self,
        neighborhood_name: str,
        osm_area_km2: float,
        census_area_km2: Optional[float],
        population_density: Optional[float],
        building_density: Optional[float],
        service_density: Optional[float],
        network_coverage_km2: Optional[float],
    ) -> Dict[str, Any]:
        """Check consistency between OSM and Census data.

        Args:
            neighborhood_name: Name of the neighborhood.
            osm_area_km2: Area from OSM data (km²).
            census_area_km2: Area from Census data (km²), if available.
            population_density: Population density from Census (inhabitants/km²).
            building_density: Building density from OSM (buildings/km²).
            service_density: Service density from OSM (services/km²).
            network_coverage_km2: Network coverage area (km²), if available.

        Returns:
            Dictionary with consistency check results.
        """
        result = {
            "status": "pass",
            "issues": [],
            "warnings": [],
        }

        # Check area consistency
        if census_area_km2 is not None and osm_area_km2 > 0:
            area_diff_ratio = abs(osm_area_km2 - census_area_km2) / max(osm_area_km2, census_area_km2)

            if area_diff_ratio > self.AREA_CONSISTENCY_HARD_TOLERANCE:
                result["status"] = "fail"
                result["issues"].append(
                    f"Area mismatch: OSM area ({osm_area_km2:.4f} km²) vs Census area "
                    f"({census_area_km2:.4f} km²), difference: {area_diff_ratio:.2%} "
                    f"(threshold: {self.AREA_CONSISTENCY_HARD_TOLERANCE:.2%})"
                )
            elif area_diff_ratio > self.AREA_CONSISTENCY_WARN_TOLERANCE:
                result["warnings"].append(
                    f"Area mismatch: OSM area ({osm_area_km2:.4f} km²) vs Census area "
                    f"({census_area_km2:.4f} km²), difference: {area_diff_ratio:.2%} "
                    f"(threshold: {self.AREA_CONSISTENCY_WARN_TOLERANCE:.2%})"
                )

        # Check population vs buildings correlation (flag if inconsistent)
        if population_density is not None and building_density is not None:
            # High population should correlate with high building density
            # Simple heuristic: if pop density is very high but building density is very low, flag
            if (
                population_density > self.POPULATION_DENSITY_WARN_HIGH
                and building_density < self.BUILDINGS_DENSITY_WARN_LOW
            ):
                result["warnings"].append(
                    f"Inconsistent: High population density ({population_density:.0f}/km²) "
                    f"but low building density ({building_density:.0f}/km²)"
                )

        # Check service vs population correlation
        if population_density is not None and service_density is not None:
            # More people should typically have more services
            # Simple heuristic: flag extreme mismatches
            if (
                population_density > self.POPULATION_DENSITY_WARN_HIGH
                and service_density < self.SERVICES_DENSITY_WARN_LOW
            ):
                result["warnings"].append(
                    f"Inconsistent: High population density ({population_density:.0f}/km²) "
                    f"but low service density ({service_density:.0f}/km²)"
                )

        # Check network coverage
        if network_coverage_km2 is not None and osm_area_km2 > 0:
            coverage_ratio = network_coverage_km2 / osm_area_km2
            if coverage_ratio < 0.5:  # Network covers less than 50% of area
                result["warnings"].append(
                    f"Network coverage ({coverage_ratio:.2%}) is much smaller than "
                    f"neighborhood area ({osm_area_km2:.4f} km²)"
                )

        # Update status if warnings exist
        if result["warnings"] and result["status"] == "pass":
            result["status"] = "warn"

        return result

    def _calculate_neighborhood_area(self, geometry: Polygon) -> float:
        """Calculate neighborhood area in km² using approximate conversion.
        
        Uses the same approximate method as OSM extractor for consistency:
        area_m2 = geometry.area * (111320.0 ** 2)
        
        Args:
            geometry: Neighborhood geometry (Polygon) in WGS84 (EPSG:4326).

        Returns:
            Area in km².
        """
        # Use approximate conversion to match OSM extractor method
        # At Paris latitude, 1 degree ≈ 111,320 meters
        # So 1 degree² ≈ (111,320)² m²
        DEGREES_TO_METERS = 111320.0
        area_m2 = geometry.area * (DEGREES_TO_METERS ** 2)
        area_km2 = area_m2 / 1_000_000
        return area_km2

    def validate_neighborhood(self, neighborhood_name: str) -> Dict[str, Any]:
        """Main validation method for a single neighborhood.

        Args:
            neighborhood_name: Name of the neighborhood to validate.

        Returns:
            Comprehensive validation report dictionary for the neighborhood.
        """
        logger.info(f"Validating neighborhood: {neighborhood_name}")

        # Load neighborhoods to get geometry and compliance status
        neighborhoods = load_neighborhoods(self.neighborhoods_geojson)
        neighborhood_row = neighborhoods[neighborhoods["name"] == neighborhood_name]

        if neighborhood_row.empty:
            return {
                "neighborhood_name": neighborhood_name,
                "status": "fail",
                "issues": [f"Neighborhood '{neighborhood_name}' not found in GeoJSON"],
                "warnings": [],
                "checks": {},
            }

        neighborhood_row = neighborhood_row.iloc[0]
        is_compliant = neighborhood_row.get("label") == "Compliant"
        geometry = neighborhood_row.geometry

        # Calculate area
        area_km2 = self._calculate_neighborhood_area(geometry)

        # Initialize result
        result = {
            "neighborhood_name": neighborhood_name,
            "status": "pass",
            "issues": [],
            "warnings": [],
            "checks": {},
            "statistics": {},
        }

        # Check file structure
        file_structure = self._check_file_structure(neighborhood_name, is_compliant)
        result["checks"]["file_structure"] = file_structure
        result["issues"].extend(file_structure.get("issues", []))
        result["warnings"].extend([f"File structure: {w}" for w in file_structure.get("warnings", [])])

        if file_structure["status"] == "fail":
            result["status"] = "fail"
            # Can't continue validation if files are missing
            return result

        # Get directories
        osm_dir = self._get_osm_directory(neighborhood_name, is_compliant)

        # Validate OSM data
        services_result = self._validate_services(
            osm_dir / "services.geojson", neighborhood_name, area_km2
        )
        result["checks"]["services"] = services_result
        result["issues"].extend(services_result.get("issues", []))
        result["warnings"].extend(services_result.get("warnings", []))
        result["statistics"]["services"] = services_result.get("statistics", {})

        buildings_result = self._validate_buildings(
            osm_dir / "buildings.geojson", neighborhood_name, area_km2
        )
        result["checks"]["buildings"] = buildings_result
        result["issues"].extend(buildings_result.get("issues", []))
        result["warnings"].extend(buildings_result.get("warnings", []))
        result["statistics"]["buildings"] = buildings_result.get("statistics", {})

        network_result = self._validate_network(
            osm_dir / "network.graphml", neighborhood_name, area_km2
        )
        result["checks"]["network"] = network_result
        result["issues"].extend(network_result.get("issues", []))
        result["warnings"].extend(network_result.get("warnings", []))
        result["statistics"]["network"] = network_result.get("statistics", {})

        intersections_result = self._validate_intersections(
            osm_dir / "intersections.geojson", neighborhood_name, area_km2
        )
        result["checks"]["intersections"] = intersections_result
        result["issues"].extend(intersections_result.get("issues", []))
        result["warnings"].extend(intersections_result.get("warnings", []))
        result["statistics"]["intersections"] = intersections_result.get("statistics", {})

        pedestrian_result = self._validate_pedestrian_features(
            osm_dir / "pedestrian_infrastructure.geojson", neighborhood_name
        )
        result["checks"]["pedestrian_features"] = pedestrian_result
        result["issues"].extend(pedestrian_result.get("issues", []))
        result["warnings"].extend(pedestrian_result.get("warnings", []))
        result["statistics"]["pedestrian_features"] = pedestrian_result.get("statistics", {})

        # Validate Census data (only for compliant neighborhoods)
        census_area_km2 = None
        population_density = None
        if is_compliant:
            census_dir = self._get_census_directory(neighborhood_name)
            census_result = self._validate_census_data(
                census_dir / "census_data.parquet", neighborhood_name
            )
            result["checks"]["census"] = census_result
            result["issues"].extend(census_result.get("issues", []))
            result["warnings"].extend(census_result.get("warnings", []))
            result["statistics"]["census"] = census_result.get("statistics", {})

            # Extract values for cross-source consistency
            # population_density is now a dict with mean/min/max, extract mean for consistency checks
            pop_density_stats = census_result.get("statistics", {}).get("population_density")
            if isinstance(pop_density_stats, dict):
                population_density = pop_density_stats.get("mean")
            else:
                population_density = pop_density_stats  # Legacy format (single value)
            census_area_km2 = area_km2  # Use same area for now

        # Cross-source consistency checks
        building_density = buildings_result.get("statistics", {}).get("building_density")
        service_density = services_result.get("statistics", {}).get("service_density")
        network_coverage = None  # Could calculate from network bounds if needed

        consistency_result = self._check_cross_source_consistency(
            neighborhood_name,
            area_km2,
            census_area_km2,
            population_density,
            building_density,
            service_density,
            network_coverage,
        )
        result["checks"]["cross_source_consistency"] = consistency_result
        result["issues"].extend(consistency_result.get("issues", []))
        result["warnings"].extend(consistency_result.get("warnings", []))

        # Determine overall status
        if result["issues"]:
            result["status"] = "fail"
        elif result["warnings"]:
            result["status"] = "warn"

        result["statistics"]["area_km2"] = area_km2

        return result

    def validate_all_neighborhoods(
        self, neighborhood_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate all neighborhoods or specified list.

        Args:
            neighborhood_names: Optional list of neighborhood names to validate.
                If None, validates all compliant neighborhoods.

        Returns:
            Overall validation report with per-neighborhood results and summary.
        """
        # Load neighborhoods
        neighborhoods = load_neighborhoods(self.neighborhoods_geojson)

        if neighborhood_names is None:
            # Validate all compliant neighborhoods
            compliant = get_compliant_neighborhoods(neighborhoods)
            neighborhood_names = compliant["name"].tolist()
            logger.info(f"Validating {len(neighborhood_names)} compliant neighborhoods")
        else:
            logger.info(f"Validating {len(neighborhood_names)} specified neighborhoods")

        # Validate each neighborhood
        neighborhood_results = []
        for name in neighborhood_names:
            result = self.validate_neighborhood(name)
            neighborhood_results.append(result)

        # Aggregate results
        total_neighborhoods = len(neighborhood_results)
        passed = sum(1 for r in neighborhood_results if r["status"] == "pass")
        warnings = sum(1 for r in neighborhood_results if r["status"] == "warn")
        failed = sum(1 for r in neighborhood_results if r["status"] == "fail")

        # Count checks
        total_checks = 0
        passed_checks = 0
        warning_checks = 0
        failed_checks = 0

        for result in neighborhood_results:
            for check_name, check_result in result.get("checks", {}).items():
                total_checks += 1
                check_status = check_result.get("status", "pass")
                if check_status == "pass":
                    passed_checks += 1
                elif check_status == "warn":
                    warning_checks += 1
                else:
                    failed_checks += 1

        summary = {
            "total_neighborhoods": total_neighborhoods,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "warning_checks": warning_checks,
            "failed_checks": failed_checks,
        }

        return {
            "summary": summary,
            "neighborhoods": neighborhood_results,
        }

    def _generate_report(
        self, validation_results: Dict[str, Any], output_dir: Path
    ) -> None:
        """Generate and save validation report.

        Args:
            validation_results: Results from validate_all_neighborhoods().
            output_dir: Directory to save report files.
        """
        ensure_dir_exists(str(output_dir))

        # Save JSON report
        json_path = output_dir / "sanity_check_report.json"
        with open(json_path, "w") as f:
            json.dump(validation_results, f, indent=2, default=str)

        logger.info(f"Saved JSON report to {json_path}")

        # Create summary CSV
        summary_rows = []
        for neighborhood_result in validation_results["neighborhoods"]:
            summary_rows.append(
                {
                    "neighborhood_name": neighborhood_result["neighborhood_name"],
                    "status": neighborhood_result["status"],
                    "issue_count": len(neighborhood_result.get("issues", [])),
                    "warning_count": len(neighborhood_result.get("warnings", [])),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        csv_path = output_dir / "sanity_check_summary.csv"
        summary_df.to_csv(csv_path, index=False)

        logger.info(f"Saved summary CSV to {csv_path}")

        # Print human-readable summary
        print("\n" + "=" * 80)
        print("RAW DATA SANITY CHECK SUMMARY")
        print("=" * 80)
        print(f"Total neighborhoods: {validation_results['summary']['total_neighborhoods']}")
        print(f"Passed: {validation_results['summary']['passed']}")
        print(f"Warnings: {validation_results['summary']['warnings']}")
        print(f"Failed: {validation_results['summary']['failed']}")
        print(f"\nTotal checks: {validation_results['summary']['total_checks']}")
        print(f"Passed checks: {validation_results['summary']['passed_checks']}")
        print(f"Warning checks: {validation_results['summary']['warning_checks']}")
        print(f"Failed checks: {validation_results['summary']['failed_checks']}")
        print("=" * 80)

        # Print failed neighborhoods
        failed_neighborhoods = [
            r for r in validation_results["neighborhoods"] if r["status"] == "fail"
        ]
        if failed_neighborhoods:
            print("\nFAILED NEIGHBORHOODS:")
            for result in failed_neighborhoods:
                print(f"\n  {result['neighborhood_name']}:")
                for issue in result.get("issues", [])[:5]:  # Show first 5 issues
                    print(f"    - {issue}")
                if len(result.get("issues", [])) > 5:
                    print(f"    ... and {len(result['issues']) - 5} more issues")

        # Print neighborhoods with warnings
        warning_neighborhoods = [
            r for r in validation_results["neighborhoods"] if r["status"] == "warn"
        ]
        if warning_neighborhoods:
            print("\nNEIGHBORHOODS WITH WARNINGS:")
            for result in warning_neighborhoods:
                print(f"\n  {result['neighborhood_name']}:")
                for warning in result.get("warnings", [])[:3]:  # Show first 3 warnings
                    print(f"    - {warning}")
                if len(result.get("warnings", [])) > 3:
                    print(f"    ... and {len(result['warnings']) - 3} more warnings")

        print("\n" + "=" * 80)
