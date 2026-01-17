"""Feature engineering pipeline for Spatial Graph Transformer model.

This module provides the FeatureEngineer class that generates star graphs for the
Spatial Graph Transformer model. It processes raw OSM and Census data to create
target points with neighbor grid cells, computes 33 features per point, calculates
network-based distances, and constructs distance-based target probability vectors.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely import wkt

from src.utils.config import get_config
from src.utils.logging import get_logger
from src.utils.helpers import (
    ensure_dir_exists,
    get_compliant_neighborhoods,
    get_non_compliant_neighborhoods,
    get_service_category_names,
    load_neighborhoods,
    save_dataframe,
)

logger = get_logger(__name__)


class FeatureEngineer:
    """Generate processed features and star graphs from raw OSM and Census data.

    This class processes raw OSM and Census data to create target points with
    neighbor grid cells, computes 33 features per point (demographics, built form,
    services, walkability), calculates network-based distances, and constructs
    distance-based target probability vectors for the loss function.

    Attributes:
        config: Configuration dictionary loaded from config.yaml.
        grid_cell_size_meters: Size of grid cells for spatial context.
        walk_15min_radius_meters: 15-minute walk radius in meters.
        sampling_interval_meters: Grid spacing for target points.
        temperature: Temperature parameter for distance-to-probability conversion.
        missing_service_penalty: Distance penalty when service category is missing.
        _distance_cache: Cache for computed network distances (Phase 3.3).

    Example:
        >>> engineer = FeatureEngineer()
        >>> neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
        >>> result = engineer.process_neighborhood(neighborhoods.iloc[0])
        >>> print(f"Processed {len(result)} target points")
    """

    def __init__(self) -> None:
        """Initialize FeatureEngineer with configuration."""
        self.config = get_config()
        features_config = self.config.get("features", {})
        loss_config = self.config.get("loss", {})

        self.grid_cell_size_meters: float = features_config.get(
            "grid_cell_size_meters", 100
        )
        self.walk_15min_radius_meters: float = features_config.get(
            "walk_15min_radius_meters", 1000
        )
        self.sampling_interval_meters: float = features_config.get(
            "target_point_sampling_interval_meters", 100
        )
        self.temperature: float = loss_config.get("temperature", 200)
        self.missing_service_penalty: float = loss_config.get(
            "missing_service_penalty", 2400
        )

        # Initialize distance cache for Phase 3.3
        self._distance_cache: Dict[Tuple, float] = {}
        
        # Cache for IRIS boundaries (loaded once, reused for all points)
        self._iris_boundaries_cache: Optional[gpd.GeoDataFrame] = None
        
        # Cache for service GeoDataFrames per neighborhood (major performance boost)
        # Key: (neighborhood_name, is_compliant), Value: Dict[category_name, gpd.GeoDataFrame]
        self._services_cache: Dict[Tuple[str, bool], Dict[str, gpd.GeoDataFrame]] = {}
        
        # Cache for buildings GeoDataFrame per neighborhood
        self._buildings_cache: Dict[Tuple[str, bool], Optional[gpd.GeoDataFrame]] = {}

        logger.info("FeatureEngineer initialized")

    def generate_target_points(
        self, neighborhood: gpd.GeoSeries
    ) -> gpd.GeoDataFrame:
        """Generate regular grid of target points within neighborhood polygon.

        Creates a regular grid of target points within the neighborhood boundary
        using the configured sampling interval. Points are generated in metric
        coordinates (EPSG:3857) and converted back to WGS84 (EPSG:4326) for storage.

        Args:
            neighborhood: GeoSeries containing neighborhood geometry and properties
                (must have 'geometry' and 'name' attributes).

        Returns:
            GeoDataFrame with columns: target_id, neighborhood_name, label, geometry.
            Returns empty GeoDataFrame if polygon is invalid or empty.

        Raises:
            ValueError: If neighborhood geometry is invalid.

        Example:
            >>> neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
            >>> points = engineer.generate_target_points(neighborhoods.iloc[0])
            >>> print(f"Generated {len(points)} target points")
        """
        geometry = neighborhood.geometry
        neighborhood_name = neighborhood.get("name", "unknown")

        if geometry is None or geometry.is_empty:
            logger.warning(
                f"Empty or invalid geometry for neighborhood: {neighborhood_name}"
            )
            return gpd.GeoDataFrame(
                columns=["target_id", "neighborhood_name", "label", "geometry"]
            )

        # Handle MultiPolygon by taking union
        if geometry.geom_type == "MultiPolygon":
            geometry = unary_union(geometry)
            if isinstance(geometry, Polygon):
                pass  # Good, it's a single polygon now
            else:
                logger.warning(
                    f"Could not convert MultiPolygon to Polygon for {neighborhood_name}"
                )
                return gpd.GeoDataFrame(
                    columns=["target_id", "neighborhood_name", "label", "geometry"]
                )

        # Convert to metric CRS for grid generation
        # Create temporary GeoDataFrame for CRS conversion
        temp_gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs="EPSG:4326")
        temp_gdf_metric = temp_gdf.to_crs("EPSG:3857")
        polygon_metric = temp_gdf_metric.geometry.iloc[0]

        # Get bounding box
        minx, miny, maxx, maxy = polygon_metric.bounds

        # Generate grid points
        x_coords = np.arange(
            minx, maxx + self.sampling_interval_meters, self.sampling_interval_meters
        )
        y_coords = np.arange(
            miny, maxy + self.sampling_interval_meters, self.sampling_interval_meters
        )

        target_points = []
        target_id = 0

        for x in x_coords:
            for y in y_coords:
                point_metric = Point(x, y)
                # Check if point is inside polygon
                if polygon_metric.contains(point_metric):
                    # Convert back to WGS84
                    point_gdf = gpd.GeoDataFrame(
                        [1], geometry=[point_metric], crs="EPSG:3857"
                    )
                    point_wgs84 = point_gdf.to_crs("EPSG:4326").geometry.iloc[0]

                    target_points.append(
                        {
                            "target_id": target_id,
                            "neighborhood_name": neighborhood_name,
                            "label": neighborhood.get("label", "Unknown"),
                            "geometry": point_wgs84,
                        }
                    )
                    target_id += 1

        if not target_points:
            logger.warning(
                f"No target points generated for neighborhood: {neighborhood_name}"
            )
            return gpd.GeoDataFrame(
                columns=["target_id", "neighborhood_name", "label", "geometry"]
            )

        result = gpd.GeoDataFrame(target_points, crs="EPSG:4326")
        logger.info(
            f"Generated {len(result)} target points for {neighborhood_name}"
        )
        return result

    def generate_grid_cells(self, target_point: Point) -> List[Point]:
        """Generate regular grid cells around target point.

        Creates a regular grid of neighbor cell centers around the target point
        within the walk_15min_radius_meters radius. Grid cells are generated in
        metric coordinates (EPSG:3857).

        Args:
            target_point: Target point in WGS84 (EPSG:4326).

        Returns:
            List of Point objects (cell centers) in WGS84 coordinates.

        Example:
            >>> point = Point(2.3522, 48.8566)
            >>> cells = engineer.generate_grid_cells(point)
            >>> print(f"Generated {len(cells)} grid cells")
        """
        # Convert target point to metric CRS
        target_gdf = gpd.GeoDataFrame([1], geometry=[target_point], crs="EPSG:4326")
        target_metric = target_gdf.to_crs("EPSG:3857").geometry.iloc[0]

        radius = self.walk_15min_radius_meters
        cell_size = self.grid_cell_size_meters

        # Calculate bounding box
        minx = target_metric.x - radius
        maxx = target_metric.x + radius
        miny = target_metric.y - radius
        maxy = target_metric.y + radius

        # Generate grid cells
        x_coords = np.arange(minx, maxx + cell_size, cell_size)
        y_coords = np.arange(miny, maxy + cell_size, cell_size)

        grid_cells_metric = []
        for x in x_coords:
            for y in y_coords:
                cell_metric = Point(x, y)
                grid_cells_metric.append(cell_metric)

        # Convert back to WGS84
        if not grid_cells_metric:
            return []

        cells_gdf = gpd.GeoDataFrame(
            range(len(grid_cells_metric)),
            geometry=grid_cells_metric,
            crs="EPSG:3857",
        )
        cells_wgs84 = cells_gdf.to_crs("EPSG:4326")

        return cells_wgs84.geometry.tolist()

    def _prefilter_by_euclidean(
        self, target_point: Point, grid_cells: List[Point], euclidean_threshold: Optional[float] = None
    ) -> List[Point]:
        """Pre-filter grid cells by Euclidean distance before network calculations.

        This optimization reduces computation by ~90% by filtering out cells that
        are too far away in Euclidean distance before expensive network distance
        calculations.

        Args:
            target_point: Target point in WGS84.
            grid_cells: List of grid cell centers in WGS84.
            euclidean_threshold: Euclidean distance threshold in meters. If None,
                uses 1.25x network threshold (1500m for 1200m network threshold).

        Returns:
            Filtered list of grid cells within euclidean_threshold.

        Example:
            >>> cells = engineer.generate_grid_cells(target_point)
            >>> filtered = engineer._prefilter_by_euclidean(target_point, cells)
            >>> print(f"Filtered {len(cells)} -> {len(filtered)} cells")
        """
        if euclidean_threshold is None:
            # Use 1.25x network threshold to account for network distance > Euclidean
            euclidean_threshold = self.walk_15min_radius_meters * 1.25

        if not grid_cells:
            return []

        # Convert to metric CRS for distance calculation
        target_gdf = gpd.GeoDataFrame([1], geometry=[target_point], crs="EPSG:4326")
        target_metric = target_gdf.to_crs("EPSG:3857").geometry.iloc[0]

        cells_gdf = gpd.GeoDataFrame(
            range(len(grid_cells)), geometry=grid_cells, crs="EPSG:4326"
        )
        cells_metric = cells_gdf.to_crs("EPSG:3857")

        # Calculate Euclidean distances
        filtered_cells = []
        for idx, cell_metric in enumerate(cells_metric.geometry):
            distance = target_metric.distance(cell_metric)
            if distance <= euclidean_threshold:
                # Convert back to WGS84 for return
                cell_wgs84 = cells_gdf.iloc[idx].geometry
                filtered_cells.append(cell_wgs84)

        logger.debug(
            f"Euclidean pre-filtering: {len(grid_cells)} -> {len(filtered_cells)} cells "
            f"(threshold: {euclidean_threshold:.0f}m)"
        )
        return filtered_cells

    def compute_network_distance(
        self, point1: Point, point2: Point, network_graph: nx.MultiDiGraph
    ) -> float:
        """Calculate network walking distance between two points.

        Uses OSMnx to find nearest network nodes and calculates shortest path
        distance along the street network. Includes comprehensive error handling
        for disconnected components, missing nodes, and network failures.

        Args:
            point1: First point in WGS84 (EPSG:4326). Can be Point or Polygon (uses centroid).
            point2: Second point in WGS84 (EPSG:4326). Can be Point or Polygon (uses centroid).
            network_graph: NetworkX graph with 'length' edge attributes.

        Returns:
            Network walking distance in meters. Returns float('inf') if calculation
            fails (points outside network, disconnected components, etc.).

        Example:
            >>> distance = engineer.compute_network_distance(point1, point2, G)
            >>> print(f"Network distance: {distance:.0f}m")
        """
        # Validate network graph
        if network_graph is None or len(network_graph.nodes()) == 0:
            logger.warning("Empty or invalid network graph")
            return float("inf")

        try:
            # Extract Point from geometry if needed (handle Polygon by using centroid)
            from shapely.geometry import Point as ShapelyPoint
            if not isinstance(point1, ShapelyPoint):
                point1 = point1.centroid if hasattr(point1, 'centroid') else point1
            if not isinstance(point2, ShapelyPoint):
                point2 = point2.centroid if hasattr(point2, 'centroid') else point2
            
            # Ensure we have Point objects with x and y attributes
            if not hasattr(point1, 'x') or not hasattr(point1, 'y'):
                logger.warning(f"point1 is not a valid Point geometry: {type(point1)}")
                return float("inf")
            if not hasattr(point2, 'x') or not hasattr(point2, 'y'):
                logger.warning(f"point2 is not a valid Point geometry: {type(point2)}")
                return float("inf")
            
            # Find nearest network nodes
            node1 = ox.distance.nearest_nodes(
                network_graph, point1.x, point1.y
            )
            node2 = ox.distance.nearest_nodes(
                network_graph, point2.x, point2.y
            )

            # Calculate shortest path length directly
            # Note: We skip the expensive connectivity check because:
            # 1. It's called hundreds of times per target point (very slow)
            # 2. shortest_path_length() will raise NetworkXNoPath if nodes are disconnected
            # 3. We already handle that exception below
            try:
                distance = nx.shortest_path_length(
                    network_graph, node1, node2, weight="length"
                )
                return float(distance)
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                logger.debug(f"No path found between nodes: {e}")
                return float("inf")

        except Exception as e:
            logger.warning(f"Error computing network distance: {e}")
            return float("inf")

    def filter_by_network_distance(
        self,
        target_point: Point,
        grid_cells: List[Point],
        network_graph: nx.MultiDiGraph,
        max_distance_meters: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filter grid cells by Euclidean distance (network distance calculated later in loss function).

        Uses only Euclidean distance for filtering to speed up feature engineering.
        Network/walking distance will be calculated in the loss function phase.

        Args:
            target_point: Target point in WGS84.
            grid_cells: List of grid cell centers in WGS84.
            network_graph: NetworkX graph (not used, kept for API compatibility).
            max_distance_meters: Maximum Euclidean distance in meters. If None, uses
                walk_15min_radius_meters.

        Returns:
            List of dicts with keys: 'cell', 'network_distance' (set to euclidean), 
            'euclidean_distance', 'dx', 'dy'. Only includes cells within max_distance_meters.

        Example:
            >>> cells = engineer.generate_grid_cells(target_point)
            >>> neighbors = engineer.filter_by_network_distance(
            ...     target_point, cells, network_graph
            ... )
            >>> print(f"Found {len(neighbors)} neighbors within radius")
        """
        if max_distance_meters is None:
            max_distance_meters = self.walk_15min_radius_meters

        # Convert target to metric for distance calculations
        target_gdf = gpd.GeoDataFrame([1], geometry=[target_point], crs="EPSG:4326")
        target_metric = target_gdf.to_crs("EPSG:3857").geometry.iloc[0]

        neighbors = []
        total_cells = len(grid_cells)
        logger.info(f"  Filtering {total_cells} cells by Euclidean distance (max: {max_distance_meters:.0f}m)...")
        
        for cell_idx, cell in enumerate(grid_cells):
            # Calculate Euclidean distance and relative coordinates
            cell_gdf = gpd.GeoDataFrame([1], geometry=[cell], crs="EPSG:4326")
            cell_metric = cell_gdf.to_crs("EPSG:3857").geometry.iloc[0]

            euclidean_distance = target_metric.distance(cell_metric)
            dx = cell_metric.x - target_metric.x
            dy = cell_metric.y - target_metric.y

            # Filter by Euclidean distance only
            if euclidean_distance <= max_distance_meters:
                neighbors.append(
                    {
                        "cell": cell,
                        "network_distance": euclidean_distance,  # Use Euclidean as placeholder, will be recalculated in loss
                        "euclidean_distance": euclidean_distance,
                        "dx": dx,
                        "dy": dy,
                    }
                )

        logger.debug(
            f"Euclidean distance filtering: {len(grid_cells)} -> {len(neighbors)} neighbors"
        )
        return neighbors

    def _load_iris_boundaries(self) -> gpd.GeoDataFrame:
        """Load IRIS boundary file for spatial joins (cached after first load).

        Returns:
            GeoDataFrame with IRIS geometries and codes.
        """
        # Return cached version if already loaded
        if self._iris_boundaries_cache is not None:
            return self._iris_boundaries_cache
        
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

        for path in possible_paths:
            if path.exists():
                try:
                    iris_boundaries = gpd.read_file(path)
                    # Filter for Paris if needed
                    if "INSEE_COM" in iris_boundaries.columns:
                        iris_boundaries = iris_boundaries[
                            iris_boundaries["INSEE_COM"].str.startswith("75", na=False)
                        ]
                    elif "DCOMIRIS" in iris_boundaries.columns:
                        iris_boundaries = iris_boundaries[
                            iris_boundaries["DCOMIRIS"].str.startswith("75", na=False)
                        ]
                    elif "DEPCOM" in iris_boundaries.columns:
                        iris_boundaries = iris_boundaries[
                            iris_boundaries["DEPCOM"].str.startswith("75", na=False)
                        ]
                    # Convert to WGS84 if needed
                    if iris_boundaries.crs != "EPSG:4326":
                        iris_boundaries = iris_boundaries.to_crs("EPSG:4326")
                    logger.info(f"Loaded {len(iris_boundaries)} IRIS boundaries from {path} (cached for reuse)")
                    # Cache the result
                    self._iris_boundaries_cache = iris_boundaries
                    return iris_boundaries
                except Exception as e:
                    logger.warning(f"Failed to load IRIS boundaries from {path}: {e}")
                    continue

        logger.warning("IRIS boundaries file not found. Spatial matching may not work.")
        # Cache empty GeoDataFrame to avoid retrying
        self._iris_boundaries_cache = gpd.GeoDataFrame()
        return self._iris_boundaries_cache

    def compute_demographic_features(
        self, point: Point, neighborhood_name: str, is_compliant: bool
    ) -> np.ndarray:
        """Compute 17 demographic features from Census data.

        Args:
            point: Point in WGS84.
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether neighborhood is compliant.

        Returns:
            NumPy array of shape (17,) with demographic features.
        """
        # Load Census data
        normalized_name = neighborhood_name.lower().replace(" ", "_")
        status = "compliant" if is_compliant else "non_compliant"
        census_path = Path(f"data/raw/census/{status}/{normalized_name}/census_data.parquet")

        if not census_path.exists():
            logger.warning(f"Census data not found for {neighborhood_name}")
            return np.zeros(17, dtype=np.float32)

        try:
            census_data = pd.read_parquet(census_path)
        except Exception as e:
            logger.warning(f"Failed to load census data: {e}")
            return np.zeros(17, dtype=np.float32)

        # Load IRIS boundaries for spatial join
        iris_boundaries = self._load_iris_boundaries()
        if iris_boundaries.empty:
            logger.warning("IRIS boundaries not available, using neighborhood average")
            # Use neighborhood average if available
            if len(census_data) > 0:
                features = self._extract_demographic_features_from_census(census_data)
                return features
            return np.zeros(17, dtype=np.float32)

        # Find containing IRIS unit
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
        joined = gpd.sjoin(point_gdf, iris_boundaries, how="left", predicate="within")

        if joined.empty or "index_right" not in joined.columns:
            logger.debug(f"Point not in any IRIS unit, using neighborhood average")
            if len(census_data) > 0:
                features = self._extract_demographic_features_from_census(census_data)
                return features
            return np.zeros(17, dtype=np.float32)

        # Get IRIS code and match to census data
        iris_idx = joined["index_right"].iloc[0]
        
        # Check if iris_idx is NaN (no match found in spatial join)
        if pd.isna(iris_idx):
            logger.debug(f"Point not in any IRIS unit (NaN index), using neighborhood average")
            if len(census_data) > 0:
                features = self._extract_demographic_features_from_census(census_data)
                return features
            return np.zeros(17, dtype=np.float32)
        
        # Convert to int if it's a valid index
        try:
            iris_idx = int(iris_idx)
        except (ValueError, TypeError):
            logger.debug(f"Invalid IRIS index {iris_idx}, using neighborhood average")
            if len(census_data) > 0:
                features = self._extract_demographic_features_from_census(census_data)
                return features
            return np.zeros(17, dtype=np.float32)
        
        # Check if index is within bounds
        if iris_idx < 0 or iris_idx >= len(iris_boundaries):
            logger.debug(f"IRIS index {iris_idx} out of bounds, using neighborhood average")
            if len(census_data) > 0:
                features = self._extract_demographic_features_from_census(census_data)
                return features
            return np.zeros(17, dtype=np.float32)
        
        iris_code_col = None
        for col in ["CODE_IRIS", "DCOMIRIS", "IRIS"]:
            if col in iris_boundaries.columns:
                iris_code_col = col
                break

        if iris_code_col is None:
            logger.warning("IRIS code column not found")
            if len(census_data) > 0:
                features = self._extract_demographic_features_from_census(census_data)
                return features
            return np.zeros(17, dtype=np.float32)

        iris_code = iris_boundaries.iloc[iris_idx][iris_code_col]
        
        # Convert iris_code to string for comparison (census data might have string codes)
        if pd.notna(iris_code):
            iris_code = str(iris_code).strip()
        else:
            logger.debug(f"IRIS code is NaN, using neighborhood average")
            if len(census_data) > 0:
                features = self._extract_demographic_features_from_census(census_data)
                return features
            return np.zeros(17, dtype=np.float32)

        # Match to census data
        census_col = None
        for col in ["IRIS", "CODE_IRIS", "DCOMIRIS", "CODEGEO"]:
            if col in census_data.columns:
                census_col = col
                break

        if census_col is None:
            logger.warning("IRIS code column not found in census data")
            if len(census_data) > 0:
                features = self._extract_demographic_features_from_census(census_data)
                return features
            return np.zeros(17, dtype=np.float32)

        # Convert census column to string for comparison
        try:
            census_data_str = census_data[census_col].astype(str).str.strip()
            matched = census_data[census_data_str == iris_code]
        except Exception as e:
            logger.debug(f"Error matching IRIS code {iris_code}: {e}, using neighborhood average")
            if len(census_data) > 0:
                features = self._extract_demographic_features_from_census(census_data)
                return features
            return np.zeros(17, dtype=np.float32)
        if matched.empty:
            logger.debug(f"IRIS code {iris_code} not found in census data, using neighborhood average")
            if len(census_data) > 0:
                features = self._extract_demographic_features_from_census(census_data)
                return features
            return np.zeros(17, dtype=np.float32)

        # Extract features from matched row
        features = self._extract_demographic_features_from_census(matched.iloc[0:1])
        return features

    def _extract_demographic_features_from_census(self, census_row: pd.DataFrame) -> np.ndarray:
        """Extract 17 demographic features from census DataFrame row.

        Args:
            census_row: DataFrame with one or more rows containing census variables.

        Returns:
            NumPy array of shape (17,) with demographic features.
        """
        features = np.zeros(17, dtype=np.float32)

        # Feature names in order:
        # 0: population_density
        # 1: ses_index
        # 2: car_commute_ratio
        # 3: children_per_capita
        # 4: elderly_ratio
        # 5: unemployment_rate
        # 6: student_ratio
        # 7: walking_ratio
        # 8: cycling_ratio
        # 9: public_transport_ratio
        # 10: two_wheelers_ratio
        # 11: retired_ratio
        # 12: permanent_employment_ratio
        # 13: temporary_employment_ratio
        # 14: median_income
        # 15: poverty_rate
        # 16: working_age_ratio

        # Use mean for multiple rows (neighborhood average)
        # Only compute mean for numeric columns to avoid errors with string columns
        if len(census_row) == 1:
            row = census_row.iloc[0]
        else:
            # Select only numeric columns for mean calculation
            numeric_cols = census_row.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                row_mean = census_row[numeric_cols].mean()
                # Start with first row and update numeric columns with mean
                row = census_row.iloc[0].copy()
                for col in numeric_cols:
                    row[col] = row_mean[col]
            else:
                # Fallback to first row if no numeric columns
                row = census_row.iloc[0]

        # Extract features (use column names from CensusLoader output)
        feature_cols = [
            "population_density",
            "ses_index",
            "car_commute_ratio",
            "children_per_capita",
            "elderly_ratio",
            "unemployment_rate",
            "student_ratio",
            "walking_ratio",
            "cycling_ratio",
            "public_transport_ratio",
            "two_wheelers_ratio",
            "retired_ratio",
            "permanent_employment_ratio",
            "temporary_employment_ratio",
            "median_income",
            "poverty_rate",
            "working_age_ratio",
        ]

        for idx, col in enumerate(feature_cols):
            if col in census_row.columns:
                val = row[col] if len(census_row) == 1 else census_row[col].mean()
                if pd.notna(val):
                    features[idx] = float(val)

        return features

    def _load_buildings_cache(self, neighborhood_name: str, is_compliant: bool) -> Optional[gpd.GeoDataFrame]:
        """Load and cache buildings GeoDataFrame for a neighborhood (performance optimization).
        
        Args:
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether neighborhood is compliant.
            
        Returns:
            Pre-loaded, metric-CRS GeoDataFrame of buildings, or None if not found.
        """
        cache_key = (neighborhood_name, is_compliant)
        
        if cache_key in self._buildings_cache:
            return self._buildings_cache[cache_key]
        
        normalized_name = neighborhood_name.lower().replace(" ", "_")
        status = "compliant" if is_compliant else "non_compliant"
        buildings_path = Path(f"data/raw/osm/{status}/{normalized_name}/buildings.geojson")
        
        if not buildings_path.exists():
            self._buildings_cache[cache_key] = None
            return None
        
        try:
            buildings = gpd.read_file(buildings_path)
            if buildings.empty:
                self._buildings_cache[cache_key] = None
                return None
            
            # Pre-convert to metric CRS once (major speedup)
            buildings_metric = buildings.to_crs("EPSG:3857")
            self._buildings_cache[cache_key] = buildings_metric
            logger.debug(f"Cached buildings for {neighborhood_name}")
            return buildings_metric
        except Exception as e:
            logger.warning(f"Error loading buildings for {neighborhood_name}: {e}")
            self._buildings_cache[cache_key] = None
            return None

    def compute_built_form_features(
        self, point: Point, neighborhood_name: str, is_compliant: bool
    ) -> np.ndarray:
        """Compute 4 built form features from OSM building data.

        Args:
            point: Point in WGS84.
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether neighborhood is compliant.

        Returns:
            NumPy array of shape (4,) with built form features.
        """
        features = np.zeros(4, dtype=np.float32)

        # Use cached buildings (loaded once per neighborhood, pre-converted to metric CRS)
        buildings_metric = self._load_buildings_cache(neighborhood_name, is_compliant)
        
        if buildings_metric is None or buildings_metric.empty:
            return features

        try:
            # Convert point to metric CRS once
            point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
            point_metric = point_gdf.to_crs("EPSG:3857").geometry.iloc[0]

            # Create 100m buffer
            buffer_radius = 100.0
            buffer_area = np.pi * buffer_radius ** 2

            point_buffer = point_metric.buffer(buffer_radius)

            # Find buildings within buffer
            buildings_in_buffer = buildings_metric[
                buildings_metric.geometry.intersects(point_buffer)
            ]

            if buildings_in_buffer.empty:
                return features

            # Feature 0: building_density (count / buffer_area)
            building_count = len(buildings_in_buffer)
            features[0] = building_count / buffer_area

            # Feature 1: building_count
            features[1] = float(building_count)

            # Feature 2: average_building_levels
            if "building:levels" in buildings_in_buffer.columns:
                levels = pd.to_numeric(
                    buildings_in_buffer["building:levels"], errors="coerce"
                )
                levels = levels.dropna()
                if len(levels) > 0:
                    features[2] = float(levels.mean())
            elif "levels" in buildings_in_buffer.columns:
                levels = pd.to_numeric(
                    buildings_in_buffer["levels"], errors="coerce"
                )
                levels = levels.dropna()
                if len(levels) > 0:
                    features[2] = float(levels.mean())

            # Feature 3: floor_area_per_capita
            # Calculate total floor area: sum(building_area * levels)
            # Then divide by population (would need census data, use 0 for now)
            # This is a simplified version - full implementation would need population
            features[3] = 0.0  # Placeholder

        except Exception as e:
            logger.warning(f"Error computing built form features: {e}")

        return features

    def _load_services_cache(self, neighborhood_name: str, is_compliant: bool) -> Dict[str, gpd.GeoDataFrame]:
        """Load and cache service GeoDataFrames for a neighborhood (major performance optimization).
        
        Args:
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether neighborhood is compliant.
            
        Returns:
            Dictionary mapping category names to pre-loaded, metric-CRS GeoDataFrames.
        """
        cache_key = (neighborhood_name, is_compliant)
        
        if cache_key in self._services_cache:
            return self._services_cache[cache_key]
        
        normalized_name = neighborhood_name.lower().replace(" ", "_")
        status = "compliant" if is_compliant else "non_compliant"
        base_path = Path(f"data/raw/osm/{status}/{normalized_name}/services_by_category")
        
        service_categories = get_service_category_names()
        services_dict = {}
        
        for category in service_categories:
            category_file = category.lower().replace(" ", "_")
            service_path = base_path / f"{category_file}.geojson"
            
            if not service_path.exists():
                continue
                
            try:
                services = gpd.read_file(service_path)
                if not services.empty:
                    # Pre-convert to metric CRS once (major speedup)
                    services_metric = services.to_crs("EPSG:3857")
                    services_dict[category] = services_metric
            except Exception as e:
                logger.warning(f"Error loading service file {service_path}: {e}")
        
        self._services_cache[cache_key] = services_dict
        logger.debug(f"Cached {len(services_dict)} service categories for {neighborhood_name}")
        return services_dict

    def compute_service_features(
        self, point: Point, neighborhood_name: str, is_compliant: bool, network_graph: nx.MultiDiGraph
    ) -> np.ndarray:
        """Compute 8 service count features within 15-minute walk radius.

        Args:
            point: Point in WGS84.
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether neighborhood is compliant.
            network_graph: Network graph for distance calculations.

        Returns:
            NumPy array of shape (8,) with service counts per category.
        """
        # Use cached services (loaded once per neighborhood, pre-converted to metric CRS)
        services_dict = self._load_services_cache(neighborhood_name, is_compliant)
        
        # Convert point to metric CRS once
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
        point_metric = point_gdf.to_crs("EPSG:3857").geometry.iloc[0]
        
        service_categories = get_service_category_names()
        features = np.zeros(8, dtype=np.float32)

        for idx, category in enumerate(service_categories):
            if category not in services_dict:
                continue
                
            try:
                services_metric = services_dict[category]
                
                # Vectorized distance calculation (much faster than apply + iterrows)
                distances = services_metric.geometry.distance(point_metric)
                
                # Count services within radius (vectorized)
                count = (distances <= self.walk_15min_radius_meters).sum()
                features[idx] = float(count)

            except Exception as e:
                logger.warning(f"Error computing service features for {category}: {e}")

        return features

    def compute_walkability_features(
        self, point: Point, neighborhood_name: str, is_compliant: bool, network_graph: nx.MultiDiGraph
    ) -> np.ndarray:
        """Compute 4 walkability features from network and OSM data.

        Args:
            point: Point in WGS84.
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether neighborhood is compliant.
            network_graph: Network graph for analysis.

        Returns:
            NumPy array of shape (4,) with walkability features.
        """
        features = np.zeros(4, dtype=np.float32)

        if network_graph is None or len(network_graph.nodes()) == 0:
            return features

        try:
            # Convert point to metric CRS
            point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
            point_metric = point_gdf.to_crs("EPSG:3857").geometry.iloc[0]

            # Create 200m buffer for local analysis
            buffer_radius = 200.0
            buffer_area = np.pi * buffer_radius ** 2

            # Get network nodes and edges
            # graph_to_gdfs returns (edges_gdf, nodes_gdf) tuple, or just nodes_gdf if edges=False
            try:
                nodes_result = ox.graph_to_gdfs(network_graph, edges=False, node_geometry=True)
                if isinstance(nodes_result, tuple):
                    nodes_gdf = nodes_result[1] if len(nodes_result) > 1 else nodes_result[0]
                else:
                    nodes_gdf = nodes_result
            except Exception as e:
                logger.debug(f"Failed to extract nodes from graph: {e}")
                return features
            
            if nodes_gdf is None:
                return features
            
            # Check if it's a GeoDataFrame with geometry
            if not isinstance(nodes_gdf, gpd.GeoDataFrame):
                return features
            
            if nodes_gdf.empty:
                return features
            
            # Ensure geometry column exists and is set as active
            if 'geometry' not in nodes_gdf.columns:
                # Try to create geometry from x, y coordinates if they exist
                if 'x' in nodes_gdf.columns and 'y' in nodes_gdf.columns:
                    from shapely.geometry import Point
                    nodes_gdf['geometry'] = nodes_gdf.apply(
                        lambda row: Point(row['x'], row['y']), axis=1
                    )
                    nodes_gdf = nodes_gdf.set_geometry('geometry', crs="EPSG:4326")
                else:
                    logger.debug("No geometry column and no x/y coordinates in nodes")
                    return features
            else:
                # Ensure geometry column is set as active and has valid data
                if nodes_gdf.geometry.isna().all():
                    logger.debug("All node geometries are NaN")
                    return features
                # Explicitly set geometry as active (in case it's not)
                if nodes_gdf._geometry_column_name != 'geometry':
                    nodes_gdf = nodes_gdf.set_geometry('geometry')

            try:
                nodes_metric = nodes_gdf.to_crs("EPSG:3857")
            except Exception as e:
                logger.debug(f"Failed to convert nodes to metric CRS: {e}")
                return features

            # Feature 0: intersection_density (nodes with degree >= 3 within 200m)
            nodes_metric["distance"] = nodes_metric.geometry.apply(
                lambda geom: point_metric.distance(geom)
            )
            nodes_in_buffer = nodes_metric[nodes_metric["distance"] <= buffer_radius]

            if not nodes_in_buffer.empty:
                # Calculate degree for each node
                degrees = [network_graph.degree(node) for node in nodes_in_buffer.index]
                intersections = sum(1 for d in degrees if d >= 3)
                features[0] = intersections / buffer_area

            # Feature 1: average_block_length (average edge length in local area)
            # graph_to_gdfs returns (edges_gdf, nodes_gdf) tuple
            edges_metric = None
            try:
                edges_result = ox.graph_to_gdfs(network_graph, edges=True, node_geometry=False)
                if isinstance(edges_result, tuple):
                    edges_gdf = edges_result[0] if len(edges_result) > 0 else None
                else:
                    edges_gdf = edges_result
                
                if edges_gdf is not None and isinstance(edges_gdf, gpd.GeoDataFrame):
                    if edges_gdf.empty:
                        edges_metric = None
                    elif 'geometry' not in edges_gdf.columns:
                        logger.debug("Edges GeoDataFrame has no geometry column")
                        edges_metric = None
                    elif edges_gdf.geometry.isna().all():
                        logger.debug("All edge geometries are NaN")
                        edges_metric = None
                    else:
                        # Ensure geometry is set as active
                        if edges_gdf._geometry_column_name != 'geometry':
                            edges_gdf = edges_gdf.set_geometry('geometry')
                        try:
                            edges_metric = edges_gdf.to_crs("EPSG:3857")
                        except Exception as e:
                            logger.debug(f"Failed to convert edges to metric CRS: {e}")
                            edges_metric = None
            except Exception as e:
                logger.debug(f"Failed to extract edges from graph: {e}")
            
            if edges_metric is not None:
                # Filter edges near point (simplified: check if edge center is within buffer)
                edges_metric["center"] = edges_metric.geometry.centroid
                edges_metric["distance"] = edges_metric["center"].apply(
                    lambda geom: point_metric.distance(geom)
                )
                edges_in_buffer = edges_metric[edges_metric["distance"] <= buffer_radius]

                if not edges_in_buffer.empty and "length" in edges_in_buffer.columns:
                    avg_length = edges_in_buffer["length"].mean()
                    features[1] = float(avg_length)

            # Feature 2: pedestrian_street_ratio
            # Feature 3: sidewalk_presence
            # Load pedestrian infrastructure
            normalized_name = neighborhood_name.lower().replace(" ", "_")
            status = "compliant" if is_compliant else "non_compliant"
            pedestrian_path = Path(f"data/raw/osm/{status}/{normalized_name}/pedestrian_infrastructure.geojson")

            if pedestrian_path.exists():
                try:
                    pedestrian = gpd.read_file(pedestrian_path)
                    if not pedestrian.empty:
                        pedestrian_metric = pedestrian.to_crs("EPSG:3857")
                        pedestrian_metric["distance"] = pedestrian_metric.geometry.apply(
                            lambda geom: point_metric.distance(geom.centroid)
                        )
                        pedestrian_in_buffer = pedestrian_metric[
                            pedestrian_metric["distance"] <= buffer_radius
                        ]

                        if not pedestrian_in_buffer.empty:
                            # Calculate ratio of pedestrian ways to total streets
                            total_streets = len(edges_in_buffer) if not edges_in_buffer.empty else 1
                            pedestrian_count = len(pedestrian_in_buffer)
                            features[2] = pedestrian_count / total_streets if total_streets > 0 else 0.0
                            features[3] = 1.0  # Sidewalk presence
                except Exception as e:
                    logger.debug(f"Error loading pedestrian infrastructure: {e}")

        except Exception as e:
            logger.warning(f"Error computing walkability features: {e}")

        return features

    def compute_point_features(
        self, point: Point, neighborhood_name: str, is_compliant: bool, network_graph: nx.MultiDiGraph
    ) -> np.ndarray:
        """Orchestrate all feature computations for a single point.

        Args:
            point: Point in WGS84.
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether neighborhood is compliant.
            network_graph: Network graph for distance calculations.

        Returns:
            NumPy array of shape (33,) with all features concatenated.
        """
        # Compute all feature groups
        demographic = self.compute_demographic_features(point, neighborhood_name, is_compliant)  # 17
        built_form = self.compute_built_form_features(point, neighborhood_name, is_compliant)  # 4
        services = self.compute_service_features(point, neighborhood_name, is_compliant, network_graph)  # 8
        walkability = self.compute_walkability_features(point, neighborhood_name, is_compliant, network_graph)  # 4

        # Concatenate all features
        features = np.concatenate([demographic, built_form, services, walkability], axis=0)

        # Ensure shape is (33,)
        if features.shape[0] != 33:
            logger.warning(f"Feature array has wrong shape: {features.shape}, expected (33,)")
            # Pad or truncate if needed
            if features.shape[0] < 33:
                features = np.pad(features, (0, 33 - features.shape[0]), mode="constant")
            else:
                features = features[:33]

        # Fill NaN values with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features.astype(np.float32)

    def compute_target_probability_vector(
        self, target_point: Point, neighborhood_name: str, is_compliant: bool, network_graph: nx.MultiDiGraph
    ) -> np.ndarray:
        """Compute distance-based target probability vector for loss function.

        Args:
            target_point: Target point in WGS84.
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether neighborhood is compliant.
            network_graph: Network graph for distance calculations.

        Returns:
            NumPy array of shape (8,) with probability distribution over service categories.
        """
        normalized_name = neighborhood_name.lower().replace(" ", "_")
        status = "compliant" if is_compliant else "non_compliant"
        base_path = Path(f"data/raw/osm/{status}/{normalized_name}/services_by_category")

        service_categories = get_service_category_names()
        distances = np.zeros(8, dtype=np.float32)

        for idx, category in enumerate(service_categories):
            category_file = category.lower().replace(" ", "_")
            service_path = base_path / f"{category_file}.geojson"

            if not service_path.exists():
                distances[idx] = self.missing_service_penalty
                continue

            try:
                services = gpd.read_file(service_path)
                if services.empty:
                    distances[idx] = self.missing_service_penalty
                    continue

                # Find nearest service using network distance (this is for the target probability vector used in loss function)
                logger.debug(f"  Computing network distance to {category} services ({len(services)} services)...")
                min_distance = float("inf")
                for service_idx, (_, service_row) in enumerate(services.iterrows()):
                    # Log progress for large service sets
                    if len(services) > 50 and (service_idx + 1) % 25 == 0:
                        logger.debug(f"    Checking {service_idx + 1}/{len(services)} {category} services...")
                    service_geom = service_row.geometry
                    # Get point from geometry (centroid for polygons, direct for points)
                    if service_geom.geom_type in ["Polygon", "MultiPolygon"]:
                        service_point = service_geom.centroid
                    else:
                        service_point = service_geom
                    distance = self.compute_network_distance(
                        target_point, service_point, network_graph
                    )
                    if distance < min_distance:
                        min_distance = distance

                if min_distance == float("inf"):
                    distances[idx] = self.missing_service_penalty
                else:
                    distances[idx] = min_distance

            except Exception as e:
                logger.warning(f"Error computing distance for {category}: {e}")
                distances[idx] = self.missing_service_penalty

        # Convert distances to probabilities using temperature-scaled softmax
        # P_j = exp(-d_j / τ) / Σⱼ exp(-d_j / τ)
        exp_scores = np.exp(-distances / self.temperature)
        probabilities = exp_scores / np.sum(exp_scores)

        return probabilities.astype(np.float32)

    def process_target_point(
        self,
        target_point: Point,
        target_id: int,
        neighborhood_name: str,
        label: str,
        is_compliant: bool,
        network_graph: nx.MultiDiGraph,
    ) -> Dict[str, Any]:
        """Process single target point: generate neighbors, compute features, build star graph.

        Args:
            target_point: Target point in WGS84.
            target_id: Unique ID for the target point.
            neighborhood_name: Name of the neighborhood.
            label: Label of the neighborhood (Compliant/Non-Compliant).
            is_compliant: Whether neighborhood is compliant.
            network_graph: Network graph for distance calculations.

        Returns:
            Dict with keys: target_features, neighbor_data, target_prob_vector, num_neighbors.
        """
        # Generate grid cells around target
        logger.info(f"[{neighborhood_name}] Target {target_id}: Generating grid cells...")
        grid_cells = self.generate_grid_cells(target_point)
        logger.info(f"[{neighborhood_name}] Target {target_id}: Generated {len(grid_cells)} grid cells")

        # Filter by Euclidean distance (network distance calculated later in loss function)
        logger.info(f"[{neighborhood_name}] Target {target_id}: Filtering {len(grid_cells)} cells by Euclidean distance...")
        neighbors = self.filter_by_network_distance(
            target_point, grid_cells, network_graph
        )
        logger.info(f"[{neighborhood_name}] Target {target_id}: Found {len(neighbors)} neighbors")

        # Compute target point features
        logger.info(f"[{neighborhood_name}] Target {target_id}: Computing target features...")
        target_features = self.compute_point_features(
            target_point, neighborhood_name, is_compliant, network_graph
        )
        logger.info(f"[{neighborhood_name}] Target {target_id}: Target features computed")

        # Compute neighbor features
        logger.info(f"[{neighborhood_name}] Target {target_id}: Computing features for {len(neighbors)} neighbors...")
        neighbor_data = []
        for neighbor_idx, neighbor in enumerate(neighbors):
            neighbor_cell = neighbor["cell"]
            neighbor_features = self.compute_point_features(
                neighbor_cell, neighborhood_name, is_compliant, network_graph
            )

            neighbor_data.append(
                {
                    "features": neighbor_features,
                    "network_distance": neighbor["network_distance"],
                    "euclidean_distance": neighbor["euclidean_distance"],
                    "dx": neighbor["dx"],
                    "dy": neighbor["dy"],
                }
            )
            # Log every 50 neighbors for very large neighbor sets
            if len(neighbors) > 50 and (neighbor_idx + 1) % 50 == 0:
                logger.info(f"[{neighborhood_name}] Target {target_id}: Processed {neighbor_idx + 1}/{len(neighbors)} neighbors...")

        logger.info(f"[{neighborhood_name}] Target {target_id}: All neighbor features computed")

        # Compute target probability vector
        logger.info(f"[{neighborhood_name}] Target {target_id}: Computing target probability vector (network distance)...")
        target_prob_vector = self.compute_target_probability_vector(
            target_point, neighborhood_name, is_compliant, network_graph
        )
        logger.info(f"[{neighborhood_name}] Target {target_id}: ✓ Completed processing")

        return {
            "target_features": target_features,
            "neighbor_data": neighbor_data,
            "target_prob_vector": target_prob_vector,
            "num_neighbors": len(neighbor_data),
        }

    def process_neighborhood(
        self, neighborhood: gpd.GeoSeries, force: bool = False
    ) -> pd.DataFrame:
        """Process all target points in a neighborhood.

        Args:
            neighborhood: GeoSeries containing neighborhood geometry and properties.
            force: If True, re-process even if output file exists.

        Returns:
            DataFrame with columns: target_id, neighborhood_name, label, target_features,
            neighbor_data, target_prob_vector, target_geometry, num_neighbors.
        """
        neighborhood_name = neighborhood.get("name", "unknown")
        label = neighborhood.get("label", "Unknown")
        is_compliant = label == "Compliant"

        normalized_name = neighborhood_name.lower().replace(" ", "_")
        status = "compliant" if is_compliant else "non_compliant"
        output_dir = Path(f"data/processed/features/{status}/{normalized_name}")
        output_file = output_dir / "target_points.parquet"

        # Check cache and load existing results for resume
        existing_df = None
        if not force and output_file.exists():
            logger.info(f"Checking for existing results: {output_file}")
            try:
                existing_df = pd.read_parquet(output_file)
                if "target_id" not in existing_df.columns:
                    logger.warning(f"Existing file missing 'target_id' column, will re-process")
                    existing_df = None
                else:
                    # Check if all points are processed
                    target_points_check = self.generate_target_points(neighborhood)
                    if len(existing_df) >= len(target_points_check):
                        logger.info(f"All {len(existing_df)} target points already processed for {neighborhood_name}")
                        return existing_df
                    else:
                        logger.info(f"Found partial results: {len(existing_df)}/{len(target_points_check)} points. Will resume from saved progress.")
            except Exception as e:
                logger.warning(f"Failed to load existing file: {e}, will re-process from start")
                existing_df = None

        logger.info(f"Processing neighborhood: {neighborhood_name}")

        # Generate target points
        logger.info(f"[{neighborhood_name}] Step 1/4: Generating target points...")
        target_points = self.generate_target_points(neighborhood)
        if target_points.empty:
            logger.warning(f"No target points generated for {neighborhood_name}")
            return pd.DataFrame()
        logger.info(f"[{neighborhood_name}] Generated {len(target_points)} target points")

        # Load existing results if resuming
        processed_ids = set()
        results = []
        if existing_df is not None and not existing_df.empty:
            # Extract target_ids that were already processed
            processed_ids = set(existing_df["target_id"].astype(int).tolist())
            logger.info(f"[{neighborhood_name}] Loaded {len(processed_ids)} already processed target IDs")
            
            # Convert existing results to list format (preserve order by target_id)
            existing_df_sorted = existing_df.sort_values("target_id")
            for _, row in existing_df_sorted.iterrows():
                # Convert WKT string back to Point geometry if needed
                geom = row["target_geometry"]
                if isinstance(geom, str):
                    geom = wkt.loads(geom)
                
                results.append({
                    "target_id": int(row["target_id"]),  # Ensure integer type
                    "neighborhood_name": row["neighborhood_name"],
                    "label": row["label"],
                    "target_features": row["target_features"],
                    "neighbor_data": row["neighbor_data"],
                    "target_prob_vector": row["target_prob_vector"],
                    "target_geometry": geom,
                    "num_neighbors": row["num_neighbors"],
                })
            logger.info(f"[{neighborhood_name}] Resuming: {len(processed_ids)} points already processed, {len(target_points) - len(processed_ids)} remaining")
            if len(processed_ids) <= 20:
                logger.info(f"[{neighborhood_name}] Already processed target IDs: {sorted(list(processed_ids))}")
            else:
                logger.info(f"[{neighborhood_name}] Already processed target IDs: {sorted(list(processed_ids))[:10]} ... {sorted(list(processed_ids))[-10:]}")

        # Load network graph
        logger.info(f"[{neighborhood_name}] Step 2/4: Loading network graph...")
        network_path = Path(f"data/raw/osm/{status}/{normalized_name}/network.graphml")
        if not network_path.exists():
            logger.warning(f"Network graph not found for {neighborhood_name}")
            return pd.DataFrame()

        try:
            network_graph = ox.load_graphml(network_path)
            logger.info(f"[{neighborhood_name}] Loaded network graph: {len(network_graph.nodes())} nodes, {len(network_graph.edges())} edges")
            
            # Pre-load caches for this neighborhood (major performance optimization)
            logger.info(f"[{neighborhood_name}] Step 2.5/4: Pre-loading data caches...")
            self._load_services_cache(neighborhood_name, is_compliant)
            self._load_buildings_cache(neighborhood_name, is_compliant)
            logger.info(f"[{neighborhood_name}] Data caches pre-loaded")
        except Exception as e:
            logger.error(f"Failed to load network graph: {e}")
            return pd.DataFrame()

        # Process each target point sequentially
        remaining_points = len(target_points) - len(processed_ids)
        logger.info(f"[{neighborhood_name}] Step 3/4: Processing {remaining_points} remaining target points sequentially...")
        total_points = len(target_points)

        points_processed_this_run = 0
        skipped_count = 0
        for idx, row in target_points.iterrows():
            target_point = row.geometry
            target_id = int(row["target_id"])  # Ensure integer type for comparison
            
            # Skip if already processed (this avoids re-calculating expensive network distances)
            if target_id in processed_ids:
                skipped_count += 1
                if skipped_count == 1 or skipped_count % 50 == 0:
                    logger.info(f"[{neighborhood_name}] ⏭️  Skipping target point {target_id} (already processed) - {skipped_count} skipped so far")
                continue

            try:
                # Log every point for better visibility
                if (points_processed_this_run + 1) % 5 == 0 or points_processed_this_run == 0:
                    logger.info(f"[{neighborhood_name}] Processing target point {len(results) + 1}/{total_points} (ID: {target_id})...")
                
                result = self.process_target_point(
                    target_point,
                    target_id,
                    neighborhood_name,
                    label,
                    is_compliant,
                    network_graph,
                )

                results.append(
                    {
                        "target_id": target_id,
                        "neighborhood_name": neighborhood_name,
                        "label": label,
                        "target_features": result["target_features"],
                        "neighbor_data": result["neighbor_data"],
                        "target_prob_vector": result["target_prob_vector"],
                        "target_geometry": target_point,
                        "num_neighbors": result["num_neighbors"],
                    }
                )
                points_processed_this_run += 1

                # Save milestone every 10 points
                if points_processed_this_run % 10 == 0:
                    logger.info(f"[{neighborhood_name}] Milestone: Saving progress after {len(results)}/{total_points} points...")
                    ensure_dir_exists(str(output_dir))
                    df_temp = pd.DataFrame(results)
                    # Convert Shapely geometries to WKT strings for parquet compatibility
                    if "target_geometry" in df_temp.columns:
                        df_temp["target_geometry"] = df_temp["target_geometry"].apply(
                            lambda geom: geom.wkt if hasattr(geom, 'wkt') else str(geom)
                        )
                    save_dataframe(df_temp, str(output_file), format="parquet")
                    logger.info(f"[{neighborhood_name}] ✓ Saved milestone: {len(results)} points saved to {output_file}")

                # Log progress every 5 points
                if points_processed_this_run % 5 == 0:
                    logger.info(
                        f"[{neighborhood_name}] ✓ Completed {len(results)}/{total_points} target points ({points_processed_this_run} processed this run)"
                    )

            except Exception as e:
                import traceback
                logger.error(f"Error processing target point {target_id}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

        if not results:
            logger.warning(f"No target points processed for {neighborhood_name}")
            return pd.DataFrame()

        # Convert to DataFrame and save final results
        logger.info(f"[{neighborhood_name}] Step 4/4: Saving final results...")
        df = pd.DataFrame(results)

        # Convert Shapely geometries to WKT strings for parquet compatibility
        if "target_geometry" in df.columns:
            df["target_geometry"] = df["target_geometry"].apply(
                lambda geom: geom.wkt if hasattr(geom, 'wkt') else str(geom)
            )

        # Save to Parquet
        ensure_dir_exists(str(output_dir))
        save_dataframe(df, str(output_file), format="parquet")

        logger.info(
            f"[{neighborhood_name}] ✓ COMPLETE: Processed {len(df)} target points, saved to {output_file}"
        )

        return df

    def process_all_neighborhoods(
        self, neighborhoods: gpd.GeoDataFrame, force: bool = False
    ) -> Dict[str, Any]:
        """Process all neighborhoods, organized by compliance status.

        Args:
            neighborhoods: GeoDataFrame with all neighborhoods.
            force: If True, re-process even if output files exist.

        Returns:
            Dictionary with summary statistics.
        """
        logger.info("=" * 80)
        logger.info("STARTING: Processing all neighborhoods")
        logger.info("=" * 80)

        # Only process compliant neighborhoods
        compliant = get_compliant_neighborhoods(neighborhoods)
        logger.info(f"Found {len(compliant)} compliant neighborhoods to process")

        summary = {
            "total_neighborhoods": len(compliant),
            "compliant_count": len(compliant),
            "non_compliant_count": 0,
            "successful_count": 0,
            "failed_count": 0,
            "cached_count": 0,
            "total_target_points": 0,
        }

        # Process compliant neighborhoods
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING COMPLIANT NEIGHBORHOODS ({len(compliant)} total)")
        logger.info(f"{'='*80}")
        for idx, neighborhood in compliant.iterrows():
            neighborhood_name = neighborhood.get("name", f"neighborhood_{idx}")
            logger.info(f"\n[{idx + 1}/{len(compliant)}] Starting: {neighborhood_name}")
            try:
                result = self.process_neighborhood(neighborhood, force=force)
                if not result.empty:
                    summary["successful_count"] += 1
                    summary["total_target_points"] += len(result)
                else:
                    summary["failed_count"] += 1
            except Exception as e:
                logger.error(f"Failed to process compliant neighborhood {neighborhood.get('name')}: {e}")
                summary["failed_count"] += 1

        logger.info(f"Processing complete. Summary: {summary}")
        return summary
