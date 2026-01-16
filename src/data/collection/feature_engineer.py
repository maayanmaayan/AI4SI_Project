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
            "walk_15min_radius_meters", 1200
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
            point1: First point in WGS84 (EPSG:4326).
            point2: Second point in WGS84 (EPSG:4326).
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
            # Find nearest network nodes
            node1 = ox.distance.nearest_nodes(
                network_graph, point1.x, point1.y
            )
            node2 = ox.distance.nearest_nodes(
                network_graph, point2.x, point2.y
            )

            # Check if nodes are in same connected component
            if not nx.is_strongly_connected(network_graph):
                # For directed graphs, check weak connectivity
                if not nx.is_weakly_connected(network_graph):
                    # Find components
                    components = list(nx.weakly_connected_components(network_graph))
                    node1_component = None
                    node2_component = None

                    for comp in components:
                        if node1 in comp:
                            node1_component = comp
                        if node2 in comp:
                            node2_component = comp

                    if node1_component != node2_component:
                        logger.debug(
                            f"Nodes {node1} and {node2} in different components"
                        )
                        return float("inf")

            # Calculate shortest path length
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
        """Filter grid cells by network walking distance with Euclidean pre-filtering.

        First pre-filters cells by Euclidean distance (optimization), then calculates
        network distances for remaining cells. Returns neighbors with edge attributes.

        Args:
            target_point: Target point in WGS84.
            grid_cells: List of grid cell centers in WGS84.
            network_graph: NetworkX graph for distance calculations.
            max_distance_meters: Maximum network distance in meters. If None, uses
                walk_15min_radius_meters.

        Returns:
            List of dicts with keys: 'cell', 'network_distance', 'euclidean_distance',
            'dx', 'dy'. Only includes cells within max_distance_meters.

        Example:
            >>> cells = engineer.generate_grid_cells(target_point)
            >>> neighbors = engineer.filter_by_network_distance(
            ...     target_point, cells, network_graph
            ... )
            >>> print(f"Found {len(neighbors)} neighbors within radius")
        """
        if max_distance_meters is None:
            max_distance_meters = self.walk_15min_radius_meters

        # Phase 3.1: Pre-filter by Euclidean distance
        filtered_cells = self._prefilter_by_euclidean(target_point, grid_cells)

        # Convert target to metric for distance calculations
        target_gdf = gpd.GeoDataFrame([1], geometry=[target_point], crs="EPSG:4326")
        target_metric = target_gdf.to_crs("EPSG:3857").geometry.iloc[0]

        neighbors = []
        for cell in filtered_cells:
            # Calculate network distance
            network_distance = self.compute_network_distance(
                target_point, cell, network_graph
            )

            # Calculate Euclidean distance and relative coordinates
            cell_gdf = gpd.GeoDataFrame([1], geometry=[cell], crs="EPSG:4326")
            cell_metric = cell_gdf.to_crs("EPSG:3857").geometry.iloc[0]

            euclidean_distance = target_metric.distance(cell_metric)
            dx = cell_metric.x - target_metric.x
            dy = cell_metric.y - target_metric.y

            # Check if within network distance threshold
            if network_distance <= max_distance_meters:
                neighbors.append(
                    {
                        "cell": cell,
                        "network_distance": network_distance,
                        "euclidean_distance": euclidean_distance,
                        "dx": dx,
                        "dy": dy,
                    }
                )
            elif network_distance == float("inf") and euclidean_distance * 1.3 <= max_distance_meters:
                # Phase 3.2: Euclidean fallback if network calculation failed
                logger.debug(
                    f"Using Euclidean fallback for cell (network distance was inf)"
                )
                neighbors.append(
                    {
                        "cell": cell,
                        "network_distance": euclidean_distance * 1.3,
                        "euclidean_distance": euclidean_distance,
                        "dx": dx,
                        "dy": dy,
                    }
                )

        logger.debug(
            f"Network distance filtering: {len(filtered_cells)} -> {len(neighbors)} neighbors"
        )
        return neighbors

    def _load_iris_boundaries(self) -> gpd.GeoDataFrame:
        """Load IRIS boundary file for spatial joins.

        Returns:
            GeoDataFrame with IRIS geometries and codes.
        """
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
                    logger.info(f"Loaded {len(iris_boundaries)} IRIS boundaries from {path}")
                    return iris_boundaries
                except Exception as e:
                    logger.warning(f"Failed to load IRIS boundaries from {path}: {e}")
                    continue

        logger.warning("IRIS boundaries file not found. Spatial matching may not work.")
        return gpd.GeoDataFrame()

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
        census_path = Path(f"data/raw/census/compliant/{normalized_name}/census_data.parquet")

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

        matched = census_data[census_data[census_col] == iris_code]
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
        row = census_row.iloc[0] if len(census_row) == 1 else census_row.mean()

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
        normalized_name = neighborhood_name.lower().replace(" ", "_")
        status = "compliant" if is_compliant else "non_compliant"
        buildings_path = Path(f"data/raw/osm/{status}/{normalized_name}/buildings.geojson")

        features = np.zeros(4, dtype=np.float32)

        if not buildings_path.exists():
            logger.debug(f"Buildings file not found for {neighborhood_name}")
            return features

        try:
            buildings = gpd.read_file(buildings_path)
            if buildings.empty:
                return features

            # Convert to metric CRS for buffer calculation
            point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
            point_metric = point_gdf.to_crs("EPSG:3857")
            buildings_metric = buildings.to_crs("EPSG:3857")

            # Create 100m buffer
            buffer_radius = 100.0
            buffer_area = np.pi * buffer_radius ** 2

            point_buffer = point_metric.geometry.iloc[0].buffer(buffer_radius)

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
        normalized_name = neighborhood_name.lower().replace(" ", "_")
        status = "compliant" if is_compliant else "non_compliant"
        base_path = Path(f"data/raw/osm/{status}/{normalized_name}/services_by_category")

        service_categories = get_service_category_names()
        features = np.zeros(8, dtype=np.float32)

        for idx, category in enumerate(service_categories):
            # Convert category name to filename
            category_file = category.lower().replace(" ", "_")
            service_path = base_path / f"{category_file}.geojson"

            if not service_path.exists():
                logger.debug(f"Service file not found: {service_path}")
                continue

            try:
                services = gpd.read_file(service_path)
                if services.empty:
                    continue

                # Pre-filter by Euclidean distance (optimization)
                point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
                services_metric = services.to_crs("EPSG:3857")
                point_metric = point_gdf.to_crs("EPSG:3857").geometry.iloc[0]

                # Euclidean pre-filter (1.5x network radius for safety)
                euclidean_threshold = self.walk_15min_radius_meters * 1.5
                services_metric["distance_euclidean"] = services_metric.geometry.apply(
                    lambda geom: point_metric.distance(geom)
                )
                services_filtered = services_metric[
                    services_metric["distance_euclidean"] <= euclidean_threshold
                ]

                if services_filtered.empty:
                    continue

                # Count services within network distance
                count = 0
                for _, service_row in services_filtered.iterrows():
                    service_point = service_row.geometry
                    # Convert back to WGS84 for network distance
                    service_wgs84 = gpd.GeoDataFrame(
                        [1], geometry=[service_point], crs="EPSG:3857"
                    ).to_crs("EPSG:4326").geometry.iloc[0]

                    distance = self.compute_network_distance(
                        point, service_wgs84, network_graph
                    )
                    if distance <= self.walk_15min_radius_meters:
                        count += 1

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
            nodes_gdf = ox.graph_to_gdfs(network_graph, edges=False, node_geometry=True)
            if nodes_gdf.empty:
                return features

            nodes_metric = nodes_gdf.to_crs("EPSG:3857")

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
            edges_gdf = ox.graph_to_gdfs(network_graph, edges=True, node_geometry=False)
            if not edges_gdf.empty:
                edges_metric = edges_gdf.to_crs("EPSG:3857")
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

                # Find nearest service using network distance
                min_distance = float("inf")
                for _, service_row in services.iterrows():
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
        grid_cells = self.generate_grid_cells(target_point)

        # Filter by network distance
        neighbors = self.filter_by_network_distance(
            target_point, grid_cells, network_graph
        )

        # Compute target point features
        target_features = self.compute_point_features(
            target_point, neighborhood_name, is_compliant, network_graph
        )

        # Compute neighbor features
        neighbor_data = []
        for neighbor in neighbors:
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

        # Compute target probability vector
        target_prob_vector = self.compute_target_probability_vector(
            target_point, neighborhood_name, is_compliant, network_graph
        )

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

        # Check cache
        if not force and output_file.exists():
            logger.info(f"Loading cached features for {neighborhood_name}")
            try:
                return pd.read_parquet(output_file)
            except Exception as e:
                logger.warning(f"Failed to load cached file: {e}, re-processing")

        logger.info(f"Processing neighborhood: {neighborhood_name}")

        # Generate target points
        target_points = self.generate_target_points(neighborhood)
        if target_points.empty:
            logger.warning(f"No target points generated for {neighborhood_name}")
            return pd.DataFrame()

        # Load network graph
        network_path = Path(f"data/raw/osm/{status}/{normalized_name}/network.graphml")
        if not network_path.exists():
            logger.warning(f"Network graph not found for {neighborhood_name}")
            return pd.DataFrame()

        try:
            network_graph = ox.load_graphml(network_path)
        except Exception as e:
            logger.error(f"Failed to load network graph: {e}")
            return pd.DataFrame()

        # Process each target point
        results = []
        total_points = len(target_points)

        for idx, row in target_points.iterrows():
            target_point = row.geometry
            target_id = row["target_id"]

            try:
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

                # Log progress every 10 points
                if (idx + 1) % 10 == 0:
                    logger.info(
                        f"Processed {idx + 1}/{total_points} target points for {neighborhood_name}"
                    )

            except Exception as e:
                logger.error(f"Error processing target point {target_id}: {e}")
                continue

        if not results:
            logger.warning(f"No target points processed for {neighborhood_name}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to Parquet
        ensure_dir_exists(str(output_dir))
        save_dataframe(df, str(output_file), format="parquet")

        logger.info(
            f"Processed {len(df)} target points for {neighborhood_name}, saved to {output_file}"
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
        logger.info("Processing all neighborhoods")

        # Split into compliant and non-compliant
        compliant = get_compliant_neighborhoods(neighborhoods)
        non_compliant = get_non_compliant_neighborhoods(neighborhoods)

        summary = {
            "total_neighborhoods": len(neighborhoods),
            "compliant_count": len(compliant),
            "non_compliant_count": len(non_compliant),
            "successful_count": 0,
            "failed_count": 0,
            "cached_count": 0,
            "total_target_points": 0,
        }

        # Process compliant neighborhoods
        for idx, neighborhood in compliant.iterrows():
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

        # Process non-compliant neighborhoods
        for idx, neighborhood in non_compliant.iterrows():
            try:
                result = self.process_neighborhood(neighborhood, force=force)
                if not result.empty:
                    summary["successful_count"] += 1
                    summary["total_target_points"] += len(result)
                else:
                    summary["failed_count"] += 1
            except Exception as e:
                logger.error(f"Failed to process non-compliant neighborhood {neighborhood.get('name')}: {e}")
                summary["failed_count"] += 1

        logger.info(f"Processing complete. Summary: {summary}")
        return summary
