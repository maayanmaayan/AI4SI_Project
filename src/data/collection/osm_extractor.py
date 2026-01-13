"""OSM data extraction module for Paris neighborhoods.

This module provides functionality to extract OpenStreetMap (OSM) data including
services, buildings, street networks, and walkability features for Paris neighborhoods.
Data is organized by compliance status and optimized for downstream feature engineering.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon

from src.utils.config import get_config
from src.utils.logging import get_logger
from src.utils.helpers import (
    ensure_dir_exists,
    get_compliant_neighborhoods,
    get_non_compliant_neighborhoods,
    get_service_category_mapping,
    get_service_category_names,
    load_neighborhoods,
)

logger = get_logger(__name__)


class OSMExtractor:
    """Extract OSM data for Paris neighborhoods.

    This class provides methods to extract services, buildings, street networks,
    and walkability features from OpenStreetMap for neighborhood boundaries.
    Data is organized by compliance status and cached to avoid re-extraction.

    Attributes:
        config: Configuration dictionary loaded from config.yaml.
        buffer_meters: Buffer distance in meters around neighborhood boundaries.
        network_type: Type of network to extract ('walk', 'drive', 'bike', 'all').
        simplify: Whether to simplify network geometry.
        cache_results: Whether to skip extraction if data already exists.
        retry_attempts: Number of retries on failure.
        retry_delay: Seconds between retries.
        min_services_warning: Minimum services count before warning.

    Example:
        >>> extractor = OSMExtractor()
        >>> neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
        >>> result = extractor.extract_neighborhood(neighborhoods.iloc[0])
        >>> print(f"Extracted {result['services_count']} services")
    """

    def __init__(self) -> None:
        """Initialize OSMExtractor with configuration."""
        self.config = get_config()
        osm_config = self.config.get("osm_extraction", {})

        self.buffer_meters: float = osm_config.get("buffer_meters", 100)
        self.network_type: str = osm_config.get("network_type", "walk")
        self.simplify: bool = osm_config.get("simplify", True)
        self.cache_results: bool = osm_config.get("cache_results", True)
        self.retry_attempts: int = osm_config.get("retry_attempts", 3)
        self.retry_delay: int = osm_config.get("retry_delay", 5)
        self.min_services_warning: int = osm_config.get("min_services_warning", 5)

        logger.info("OSMExtractor initialized")

    def _create_buffer_polygon(self, polygon: Polygon, buffer_meters: float) -> Polygon:
        """Create approximate buffer around polygon in WGS84.

        Creates an approximate buffer around a polygon using degree-based conversion.
        At Paris latitude (~48.85°), 1 degree ≈ 111,320 meters. This is approximate
        but sufficient for service extraction near boundaries.

        Args:
            polygon: Shapely Polygon defining the boundary.
            buffer_meters: Buffer distance in meters.

        Returns:
            Buffered Polygon in WGS84 coordinates.

        Raises:
            ValueError: If polygon is invalid or buffer_meters is negative.

        Example:
            >>> from shapely.geometry import box
            >>> p = box(2.3, 48.8, 2.4, 48.9)
            >>> buffered = extractor._create_buffer_polygon(p, 100)
            >>> assert buffered.area > p.area
        """
        if buffer_meters < 0:
            raise ValueError(f"Buffer distance must be non-negative, got {buffer_meters}")

        if not polygon.is_valid:
            raise ValueError("Polygon is invalid")

        # Approximate conversion: at Paris latitude, 1 degree ≈ 111,320 meters
        DEGREES_PER_METER = 1.0 / 111320.0
        buffer_degrees = buffer_meters * DEGREES_PER_METER

        buffered = polygon.buffer(buffer_degrees)
        return buffered

    def _get_output_directory(
        self, neighborhood_name: str, is_compliant: bool
    ) -> Path:
        """Get output directory for neighborhood based on compliance status.

        Creates and returns the output directory path for a neighborhood.
        Directories are organized as:
        - `data/raw/osm/compliant/{neighborhood_name}/` for compliant neighborhoods
        - `data/raw/osm/non_compliant/{neighborhood_name}/` for non-compliant

        Args:
            neighborhood_name: Name of the neighborhood.
            is_compliant: Whether the neighborhood is compliant.

        Returns:
            Path object pointing to the output directory (created if needed).

        Example:
            >>> dir_path = extractor._get_output_directory("Test Neighborhood", True)
            >>> assert "compliant" in str(dir_path)
            >>> assert dir_path.exists()
        """
        # Normalize neighborhood name (lowercase, replace spaces with underscores)
        normalized_name = neighborhood_name.lower().replace(" ", "_")

        # Determine base directory based on compliance status
        compliance_dir = "compliant" if is_compliant else "non_compliant"
        base_dir = Path("data/raw/osm") / compliance_dir / normalized_name

        # Create directory if it doesn't exist
        ensure_dir_exists(str(base_dir))

        return base_dir

    def _check_cache(self, output_dir: Path) -> bool:
        """Check if extraction data already exists (cached).

        Checks for existence of key output files:
        - services.geojson
        - network.graphml
        - buildings.geojson

        Args:
            output_dir: Path to output directory.

        Returns:
            True if all key files exist, False otherwise.

        Example:
            >>> output_dir = Path("data/raw/osm/compliant/test")
            >>> is_cached = extractor._check_cache(output_dir)
            >>> print(f"Data cached: {is_cached}")
        """
        if not self.cache_results:
            return False

        required_files = [
            output_dir / "services.geojson",
            output_dir / "network.graphml",
            output_dir / "buildings.geojson",
        ]

        return all(f.exists() for f in required_files)

    def extract_services(
        self,
        polygon: Polygon,
        buffered_polygon: Polygon,
        neighborhood_name: str,
    ) -> gpd.GeoDataFrame:
        """Extract OSM services and map to NEXI categories.

        Extracts all amenities, shops, and leisure features from OSM within the
        buffered polygon. Maps each service to NEXI categories and duplicates
        services that belong to multiple categories (e.g., ice_cream in both
        Grocery and Sustenance).

        Args:
            polygon: Original neighborhood polygon (for filtering).
            buffered_polygon: Buffered polygon for extraction boundary.
            neighborhood_name: Name of the neighborhood.

        Returns:
            GeoDataFrame with columns: name, amenity, category, geometry,
            neighborhood_name. Services are duplicated if they belong to multiple
            categories.

        Raises:
            ValueError: If polygon is invalid or extraction fails.

        Example:
            >>> from shapely.geometry import box
            >>> p = box(2.35, 48.85, 2.36, 48.86)
            >>> services = extractor.extract_services(p, p, "test")
            >>> print(f"Found {len(services)} services")
        """
        if not buffered_polygon.is_valid:
            raise ValueError("Buffered polygon is invalid")

        logger.info(f"Extracting services for {neighborhood_name}")

        # Extract amenities
        try:
            amenities = ox.features_from_polygon(
                buffered_polygon, tags={"amenity": True}
            )
        except Exception as e:
            logger.warning(f"Failed to extract amenities: {e}")
            amenities = gpd.GeoDataFrame()

        # Extract shops
        try:
            shops = ox.features_from_polygon(buffered_polygon, tags={"shop": True})
        except Exception as e:
            logger.warning(f"Failed to extract shops: {e}")
            shops = gpd.GeoDataFrame()

        # Extract leisure (parks)
        try:
            leisure = ox.features_from_polygon(
                buffered_polygon, tags={"leisure": True}
            )
        except Exception as e:
            logger.warning(f"Failed to extract leisure: {e}")
            leisure = gpd.GeoDataFrame()

        # Combine all features
        all_features = []
        if not amenities.empty:
            all_features.append(amenities)
        if not shops.empty:
            all_features.append(shops)
        if not leisure.empty:
            all_features.append(leisure)

        if not all_features:
            logger.warning(f"No services found for {neighborhood_name}")
            return gpd.GeoDataFrame(
                columns=["name", "amenity", "category", "geometry", "neighborhood_name"]
            )

        # Combine into single GeoDataFrame
        combined = gpd.GeoDataFrame(
            gpd.pd.concat(all_features, ignore_index=True)
        ).copy()

        # Get category mapping
        category_mapping = get_service_category_mapping()

        # Map services to categories and duplicate if needed
        service_rows = []
        for idx, row in combined.iterrows():
            # Get amenity/shop/leisure value
            amenity_val = row.get("amenity", None)
            shop_val = row.get("shop", None)
            leisure_val = row.get("leisure", None)

            # Determine which categories this service belongs to
            categories = []
            if amenity_val:
                for category, tags in category_mapping.items():
                    if amenity_val in tags:
                        categories.append(category)
            if shop_val:
                for category, tags in category_mapping.items():
                    if shop_val in tags:
                        categories.append(category)
            if leisure_val:
                for category, tags in category_mapping.items():
                    if leisure_val in tags:
                        categories.append(category)

            # If no category found, skip or assign to "Other"?
            # For now, skip services that don't match any category
            if not categories:
                continue

            # Create a row for each category (duplicate if multiple categories)
            for category in categories:
                service_rows.append(
                    {
                        "name": row.get("name", ""),
                        "amenity": amenity_val or shop_val or leisure_val,
                        "category": category,
                        "geometry": row.geometry,
                        "neighborhood_name": neighborhood_name,
                    }
                )

        if not service_rows:
            logger.warning(
                f"No services matched NEXI categories for {neighborhood_name}"
            )
            return gpd.GeoDataFrame(
                columns=["name", "amenity", "category", "geometry", "neighborhood_name"]
            )

        services_gdf = gpd.GeoDataFrame(service_rows, crs=combined.crs)

        # Filter to services within original polygon (not buffered)
        services_gdf = services_gdf[services_gdf.geometry.within(polygon)].copy()

        # Log warning if service count is low
        if len(services_gdf) < self.min_services_warning:
            logger.warning(
                f"Low service count ({len(services_gdf)}) for {neighborhood_name} - "
                f"may indicate incomplete OSM data"
            )

        logger.info(
            f"Extracted {len(services_gdf)} services for {neighborhood_name}"
        )

        return services_gdf

    def _save_services_by_category(
        self, services_gdf: gpd.GeoDataFrame, output_dir: Path
    ) -> None:
        """Save optimized service location files per category.

        Creates a `services_by_category/` subdirectory and saves one GeoJSON file
        per NEXI category containing only services for that category. Files are
        optimized for fast loading (only essential columns: name, amenity, geometry).

        Args:
            services_gdf: GeoDataFrame containing all services with category column.
            output_dir: Output directory for the neighborhood.

        Example:
            >>> services = extractor.extract_services(polygon, buffered, "test")
            >>> extractor._save_services_by_category(services, output_dir)
            >>> # Creates services_by_category/education.geojson, etc.
        """
        if services_gdf.empty:
            logger.warning("No services to save by category")
            return

        # Create services_by_category subdirectory
        category_dir = output_dir / "services_by_category"
        ensure_dir_exists(str(category_dir))

        # Get category names in consistent order
        category_names = get_service_category_names()

        # Save one file per category
        for category in category_names:
            # Filter services for this category
            category_services = services_gdf[
                services_gdf["category"] == category
            ].copy()

            if category_services.empty:
                # Still create empty file for consistency
                category_services = gpd.GeoDataFrame(
                    columns=["name", "amenity", "geometry"], crs=services_gdf.crs
                )

            # Select only essential columns for fast loading
            category_services = category_services[["name", "amenity", "geometry"]]

            # Normalize category name for filename (lowercase, replace spaces with underscores)
            category_filename = category.lower().replace(" ", "_")
            output_path = category_dir / f"{category_filename}.geojson"

            # Save as GeoJSON
            category_services.to_file(output_path, driver="GeoJSON")

        logger.info(f"Saved services by category to {category_dir}")

    def extract_buildings(
        self,
        polygon: Polygon,
        buffered_polygon: Polygon,
        neighborhood_name: str,
    ) -> gpd.GeoDataFrame:
        """Extract building data with geometry and levels.

        Extracts all buildings from OSM within the buffered polygon. Calculates
        building area (approximate in square meters) and extracts building type
        and levels information.

        Args:
            polygon: Original neighborhood polygon (for filtering).
            buffered_polygon: Buffered polygon for extraction boundary.
            neighborhood_name: Name of the neighborhood.

        Returns:
            GeoDataFrame with columns: name, building, building:levels, area_m2,
            geometry, neighborhood_name.

        Raises:
            ValueError: If polygon is invalid or extraction fails.

        Example:
            >>> from shapely.geometry import box
            >>> p = box(2.35, 48.85, 2.36, 48.86)
            >>> buildings = extractor.extract_buildings(p, p, "test")
            >>> print(f"Found {len(buildings)} buildings")
        """
        if not buffered_polygon.is_valid:
            raise ValueError("Buffered polygon is invalid")

        logger.info(f"Extracting buildings for {neighborhood_name}")

        try:
            # Extract buildings
            buildings = ox.features_from_polygon(
                buffered_polygon, tags={"building": True}
            )
        except Exception as e:
            logger.error(f"Failed to extract buildings for {neighborhood_name}: {e}")
            return gpd.GeoDataFrame(
                columns=[
                    "name",
                    "building",
                    "building:levels",
                    "area_m2",
                    "geometry",
                    "neighborhood_name",
                ]
            )

        if buildings.empty:
            logger.warning(f"No buildings found for {neighborhood_name}")
            return gpd.GeoDataFrame(
                columns=[
                    "name",
                    "building",
                    "building:levels",
                    "area_m2",
                    "geometry",
                    "neighborhood_name",
                ]
            )

        # Calculate building area (approximate conversion from degrees² to m²)
        # At Paris latitude, 1 degree ≈ 111,320 meters
        # So 1 degree² ≈ (111,320)² m²
        DEGREES_TO_METERS = 111320.0
        buildings = buildings.copy()
        buildings["area_m2"] = (
            buildings.geometry.area * (DEGREES_TO_METERS ** 2)
        )

        # Extract relevant columns
        building_data = []
        for idx, row in buildings.iterrows():
            building_data.append(
                {
                    "name": row.get("name", ""),
                    "building": row.get("building", None),
                    "building:levels": row.get("building:levels", None),
                    "area_m2": row["area_m2"],
                    "geometry": row.geometry,
                    "neighborhood_name": neighborhood_name,
                }
            )

        buildings_gdf = gpd.GeoDataFrame(building_data, crs=buildings.crs)

        # Filter to buildings within original polygon (not buffered)
        buildings_gdf = buildings_gdf[
            buildings_gdf.geometry.within(polygon)
        ].copy()

        logger.info(
            f"Extracted {len(buildings_gdf)} buildings for {neighborhood_name}"
        )

        return buildings_gdf

    def extract_network(
        self, buffered_polygon: Polygon, neighborhood_name: str
    ) -> Tuple[nx.Graph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Extract walkable street network with buffer.

        Extracts the street network from OSM within the buffered polygon.
        Network is extracted from the buffered area to ensure services near
        boundaries can be reached via network paths.

        Args:
            buffered_polygon: Buffered polygon for network extraction.
            neighborhood_name: Name of the neighborhood.

        Returns:
            Tuple of (network_graph, edges_gdf, nodes_gdf) where:
            - network_graph: NetworkX graph of the street network
            - edges_gdf: GeoDataFrame of network edges
            - nodes_gdf: GeoDataFrame of network nodes

        Raises:
            ValueError: If polygon is invalid or extraction fails.

        Example:
            >>> from shapely.geometry import box
            >>> p = box(2.35, 48.85, 2.36, 48.86)
            >>> G, edges, nodes = extractor.extract_network(p, "test")
            >>> print(f"Network nodes: {len(nodes)}, edges: {len(edges)}")
        """
        if not buffered_polygon.is_valid:
            raise ValueError("Buffered polygon is invalid")

        logger.info(f"Extracting network for {neighborhood_name}")

        try:
            # Extract network from buffered polygon
            G = ox.graph_from_polygon(
                buffered_polygon,
                network_type=self.network_type,
                simplify=self.simplify,
            )
        except Exception as e:
            logger.error(
                f"Failed to extract network for {neighborhood_name}: {e}"
            )
            # Return empty graph and GeoDataFrames
            empty_edges = gpd.GeoDataFrame(
                columns=["geometry", "neighborhood_name"]
            )
            empty_nodes = gpd.GeoDataFrame(
                columns=["geometry", "neighborhood_name"]
            )
            return nx.Graph(), empty_edges, empty_nodes

        if len(G.nodes()) == 0:
            logger.warning(f"Empty network extracted for {neighborhood_name}")
            empty_edges = gpd.GeoDataFrame(
                columns=["geometry", "neighborhood_name"]
            )
            empty_nodes = gpd.GeoDataFrame(
                columns=["geometry", "neighborhood_name"]
            )
            return G, empty_edges, empty_nodes

        # Convert graph to GeoDataFrames
        edges_gdf, nodes_gdf = ox.graph_to_gdfs(G)

        # Add neighborhood_name column
        edges_gdf = edges_gdf.copy()
        nodes_gdf = nodes_gdf.copy()
        edges_gdf["neighborhood_name"] = neighborhood_name
        nodes_gdf["neighborhood_name"] = neighborhood_name

        logger.info(
            f"Extracted network for {neighborhood_name}: {len(nodes_gdf)} nodes, "
            f"{len(edges_gdf)} edges"
        )

        return G, edges_gdf, nodes_gdf

    def extract_walkability_features(
        self,
        network_graph: nx.Graph,
        buffered_polygon: Polygon,
        neighborhood_name: str,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Extract intersections and pedestrian infrastructure.

        Extracts walkability features including:
        - Intersections: Network nodes with degree >= 3
        - Pedestrian infrastructure: Pedestrian ways, sidewalks, crosswalks

        Args:
            network_graph: NetworkX graph of the street network.
            buffered_polygon: Buffered polygon for extraction boundary.
            neighborhood_name: Name of the neighborhood.

        Returns:
            Tuple of (intersections_gdf, pedestrian_gdf) where:
            - intersections_gdf: GeoDataFrame of intersection points
            - pedestrian_gdf: GeoDataFrame of pedestrian infrastructure

        Example:
            >>> from shapely.geometry import box
            >>> import networkx as nx
            >>> p = box(2.35, 48.85, 2.36, 48.86)
            >>> G = nx.Graph()
            >>> intersections, ped = extractor.extract_walkability_features(G, p, "test")
            >>> print(f"Found {len(intersections)} intersections")
        """
        logger.info(f"Extracting walkability features for {neighborhood_name}")

        # Extract intersections from network (nodes with degree >= 3)
        intersection_nodes = []
        for node_id, degree in network_graph.degree():
            if degree >= 3:
                node_data = network_graph.nodes[node_id]
                if "geometry" in node_data:
                    intersection_nodes.append(
                        {
                            "node_id": node_id,
                            "degree": degree,
                            "geometry": node_data["geometry"],
                            "neighborhood_name": neighborhood_name,
                        }
                    )
                elif "x" in node_data and "y" in node_data:
                    # Create point from x, y coordinates
                    intersection_nodes.append(
                        {
                            "node_id": node_id,
                            "degree": degree,
                            "geometry": Point(node_data["x"], node_data["y"]),
                            "neighborhood_name": neighborhood_name,
                        }
                    )

        if intersection_nodes:
            intersections_gdf = gpd.GeoDataFrame(intersection_nodes)
        else:
            intersections_gdf = gpd.GeoDataFrame(
                columns=["node_id", "degree", "geometry", "neighborhood_name"]
            )

        # Extract pedestrian infrastructure from OSM
        pedestrian_features = []
        try:
            # Extract pedestrian ways
            pedestrian_ways = ox.features_from_polygon(
                buffered_polygon, tags={"highway": "pedestrian"}
            )
            if not pedestrian_ways.empty:
                for idx, row in pedestrian_ways.iterrows():
                    pedestrian_features.append(
                        {
                            "feature_type": "pedestrian_way",
                            "name": row.get("name", ""),
                            "geometry": row.geometry,
                            "neighborhood_name": neighborhood_name,
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to extract pedestrian ways: {e}")

        try:
            # Extract ways with sidewalk tags
            sidewalks = ox.features_from_polygon(
                buffered_polygon, tags={"sidewalk": True}
            )
            if not sidewalks.empty:
                for idx, row in sidewalks.iterrows():
                    pedestrian_features.append(
                        {
                            "feature_type": "sidewalk",
                            "name": row.get("name", ""),
                            "geometry": row.geometry,
                            "neighborhood_name": neighborhood_name,
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to extract sidewalks: {e}")

        try:
            # Extract crosswalks (nodes/ways with crossing tag)
            crosswalks = ox.features_from_polygon(
                buffered_polygon, tags={"crossing": True}
            )
            if not crosswalks.empty:
                for idx, row in crosswalks.iterrows():
                    pedestrian_features.append(
                        {
                            "feature_type": "crosswalk",
                            "name": row.get("name", ""),
                            "geometry": row.geometry,
                            "neighborhood_name": neighborhood_name,
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to extract crosswalks: {e}")

        if pedestrian_features:
            pedestrian_gdf = gpd.GeoDataFrame(pedestrian_features)
        else:
            pedestrian_gdf = gpd.GeoDataFrame(
                columns=["feature_type", "name", "geometry", "neighborhood_name"]
            )

        logger.info(
            f"Extracted walkability features for {neighborhood_name}: "
            f"{len(intersections_gdf)} intersections, "
            f"{len(pedestrian_gdf)} pedestrian features"
        )

        return intersections_gdf, pedestrian_gdf

    def extract_neighborhood(
        self, neighborhood_row: gpd.GeoSeries, force: bool = False
    ) -> Dict[str, Any]:
        """Extract all data for one neighborhood.

        Main wrapper method that orchestrates all extraction methods for a single
        neighborhood. Handles caching, error handling with retries, and saves all
        extracted data to the output directory.

        Args:
            neighborhood_row: GeoSeries row from neighborhoods GeoDataFrame with
                columns: name, label, geometry, etc.
            force: If True, re-extract even if cached data exists.

        Returns:
            Dictionary with extraction summary:
            - status: "success" or "failed"
            - neighborhood_name: Name of the neighborhood
            - services_count: Number of services extracted
            - buildings_count: Number of buildings extracted
            - network_nodes: Number of network nodes
            - network_edges: Number of network edges
            - intersections_count: Number of intersections
            - pedestrian_features_count: Number of pedestrian features
            - error: Error message if failed

        Example:
            >>> from src.utils.helpers import load_neighborhoods
            >>> neighborhoods = load_neighborhoods("paris_neighborhoods.geojson")
            >>> result = extractor.extract_neighborhood(neighborhoods.iloc[0])
            >>> print(f"Status: {result['status']}")
        """
        neighborhood_name = neighborhood_row.get("name", "unknown")
        label = neighborhood_row.get("label", "")
        is_compliant = label == "Compliant"
        geometry = neighborhood_row.geometry

        logger.info(f"Starting extraction for {neighborhood_name}")

        # Get output directory
        output_dir = self._get_output_directory(neighborhood_name, is_compliant)

        # Check cache
        if not force and self._check_cache(output_dir):
            logger.info(f"Skipping {neighborhood_name} - data already exists (use force=True to re-extract)")
            return {
                "status": "cached",
                "neighborhood_name": neighborhood_name,
                "label": label,
            }

        # Create buffered polygon
        buffered_polygon = self._create_buffer_polygon(geometry, self.buffer_meters)

        # Initialize result dictionary
        result = {
            "status": "success",
            "neighborhood_name": neighborhood_name,
            "label": label,
            "services_count": 0,
            "buildings_count": 0,
            "network_nodes": 0,
            "network_edges": 0,
            "intersections_count": 0,
            "pedestrian_features_count": 0,
            "error": None,
        }

        # Extract with retries
        for attempt in range(1, self.retry_attempts + 1):
            try:
                # Extract services
                services_gdf = self.extract_services(
                    geometry, buffered_polygon, neighborhood_name
                )
                result["services_count"] = len(services_gdf)

                # Save services
                services_path = output_dir / "services.geojson"
                services_gdf.to_file(services_path, driver="GeoJSON")
                self._save_services_by_category(services_gdf, output_dir)

                # Extract buildings
                buildings_gdf = self.extract_buildings(
                    geometry, buffered_polygon, neighborhood_name
                )
                result["buildings_count"] = len(buildings_gdf)

                # Save buildings
                buildings_path = output_dir / "buildings.geojson"
                buildings_gdf.to_file(buildings_path, driver="GeoJSON")

                # Extract network
                network_graph, edges_gdf, nodes_gdf = self.extract_network(
                    buffered_polygon, neighborhood_name
                )
                result["network_nodes"] = len(nodes_gdf)
                result["network_edges"] = len(edges_gdf)

                # Save network
                network_path = output_dir / "network.graphml"
                ox.save_graphml(network_graph, network_path)

                edges_path = output_dir / "network_edges.geojson"
                edges_gdf.to_file(edges_path, driver="GeoJSON")

                nodes_path = output_dir / "network_nodes.geojson"
                nodes_gdf.to_file(nodes_path, driver="GeoJSON")

                # Extract walkability features
                intersections_gdf, pedestrian_gdf = self.extract_walkability_features(
                    network_graph, buffered_polygon, neighborhood_name
                )
                result["intersections_count"] = len(intersections_gdf)
                result["pedestrian_features_count"] = len(pedestrian_gdf)

                # Save walkability features
                intersections_path = output_dir / "intersections.geojson"
                intersections_gdf.to_file(intersections_path, driver="GeoJSON")

                pedestrian_path = output_dir / "pedestrian_infrastructure.geojson"
                pedestrian_gdf.to_file(pedestrian_path, driver="GeoJSON")

                logger.info(f"Successfully extracted data for {neighborhood_name}")
                break

            except Exception as e:
                error_msg = f"Attempt {attempt}/{self.retry_attempts} failed: {str(e)}"
                logger.error(f"Error extracting {neighborhood_name}: {error_msg}")

                if attempt < self.retry_attempts:
                    logger.info(
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    result["status"] = "failed"
                    result["error"] = str(e)
                    logger.error(
                        f"Failed to extract {neighborhood_name} after "
                        f"{self.retry_attempts} attempts"
                    )

        return result

    def extract_all_neighborhoods(
        self, geojson_path: Optional[str] = None, force: bool = False
    ) -> Dict[str, Any]:
        """Extract data for all neighborhoods, organized by compliance status.

        Loads neighborhoods from GeoJSON, splits into compliant and non-compliant,
        and processes each neighborhood sequentially. Saves extraction metadata
        and summary statistics.

        Args:
            geojson_path: Path to neighborhoods GeoJSON file. If None, uses
                default from config.
            force: If True, re-extract even if cached data exists.

        Returns:
            Dictionary with overall summary:
            - total_neighborhoods: Total number of neighborhoods processed
            - successful_count: Number of successful extractions
            - failed_count: Number of failed extractions
            - cached_count: Number of cached (skipped) extractions
            - compliant_count: Number of compliant neighborhoods
            - non_compliant_count: Number of non-compliant neighborhoods

        Example:
            >>> extractor = OSMExtractor()
            >>> summary = extractor.extract_all_neighborhoods()
            >>> print(f"Extracted {summary['successful_count']} neighborhoods")
        """
        # Load neighborhoods
        if geojson_path is None:
            geojson_path = self.config.get("paths", {}).get(
                "neighborhoods_geojson", "paris_neighborhoods.geojson"
            )

        logger.info(f"Loading neighborhoods from {geojson_path}")
        neighborhoods = load_neighborhoods(geojson_path)

        # Split by compliance status
        compliant = get_compliant_neighborhoods(neighborhoods)
        non_compliant = get_non_compliant_neighborhoods(neighborhoods)

        logger.info(
            f"Found {len(compliant)} compliant and {len(non_compliant)} "
            f"non-compliant neighborhoods"
        )

        # Process all neighborhoods
        all_neighborhoods = gpd.pd.concat([compliant, non_compliant], ignore_index=True)
        metadata = []
        successful_count = 0
        failed_count = 0
        cached_count = 0

        for idx, row in all_neighborhoods.iterrows():
            neighborhood_name = row.get("name", f"neighborhood_{idx}")
            logger.info(
                f"Processing {idx + 1}/{len(all_neighborhoods)}: {neighborhood_name}"
            )

            result = self.extract_neighborhood(row, force=force)
            result["timestamp"] = time.time()

            if result["status"] == "success":
                successful_count += 1
            elif result["status"] == "cached":
                cached_count += 1
            else:
                failed_count += 1

            metadata.append(result)

        # Save metadata
        summary = {
            "total_neighborhoods": len(all_neighborhoods),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "cached_count": cached_count,
            "compliant_count": len(compliant),
            "non_compliant_count": len(non_compliant),
        }

        self._save_extraction_metadata(metadata, summary)

        logger.info(
            f"Extraction complete: {successful_count} successful, "
            f"{failed_count} failed, {cached_count} cached"
        )

        return summary

    def _save_extraction_metadata(
        self, metadata: List[Dict[str, Any]], summary: Dict[str, Any]
    ) -> None:
        """Save extraction metadata and summary statistics.

        Saves detailed extraction metadata as JSON and summary statistics as CSV.

        Args:
            metadata: List of extraction result dictionaries.
            summary: Overall summary dictionary.

        Example:
            >>> metadata = [{"status": "success", "neighborhood_name": "Test", ...}]
            >>> summary = {"total_neighborhoods": 1, "successful_count": 1, ...}
            >>> extractor._save_extraction_metadata(metadata, summary)
        """
        # Ensure metadata directory exists
        metadata_dir = Path("data/raw/osm")
        ensure_dir_exists(str(metadata_dir))

        # Save detailed metadata as JSON
        metadata_path = metadata_dir / "extraction_log.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Create summary CSV
        summary_rows = []
        for item in metadata:
            summary_rows.append(
                {
                    "neighborhood_name": item.get("neighborhood_name", ""),
                    "label": item.get("label", ""),
                    "status": item.get("status", ""),
                    "services_count": item.get("services_count", 0),
                    "buildings_count": item.get("buildings_count", 0),
                    "network_nodes": item.get("network_nodes", 0),
                    "network_edges": item.get("network_edges", 0),
                    "intersections_count": item.get("intersections_count", 0),
                    "pedestrian_features_count": item.get(
                        "pedestrian_features_count", 0
                    ),
                    "error": item.get("error", ""),
                }
            )

        summary_df = gpd.pd.DataFrame(summary_rows)
        summary_path = metadata_dir / "extraction_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        logger.info(f"Saved extraction metadata to {metadata_dir}")
