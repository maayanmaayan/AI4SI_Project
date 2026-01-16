# Feature: Graph Transformer Model for 15-Minute City Service Gap Prediction

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Implement a Spatial Graph Transformer model using PyTorch Geometric for classifying geographic points based on their spatial context. The model uses a star graph structure where a target point (center) aggregates information from all neighboring grid cells within network walking distance. The architecture naturally handles variable numbers of neighbors without padding, uses explicit spatial encoding via edge attributes, and maintains permutation invariance.

**Core Functionality:**
- Graph-based architecture using PyTorch Geometric (TransformerConv layers)
- Star graph structure: target point (node 0) + all neighbors within network walking distance (nodes 1-N)
- Explicit spatial encoding: edge attributes include [dx, dy, euclidean_distance, network_distance]
- Variable-sized graphs: naturally handles different numbers of neighbors per target point
- Distance-based loss: KL divergence between predicted and distance-based target probability vectors
- Feature engineering pipeline: grid cell generation, feature computation, network distance calculation
- Dataset class: builds PyTorch Geometric Data objects from pre-computed features

## User Story

As a **ML engineer/researcher**
I want to **train a Graph Transformer model that learns service distribution patterns from 15-minute city compliant neighborhoods**
So that **I can predict service gap interventions for locations in other neighborhoods based on spatial context and walking accessibility**

## Problem Statement

The project requires a model architecture that:
1. **Handles variable spatial context** - Each target point has a different number of neighbors (grid cells) within walking distance
2. **Learns from full spatial context** - Must include ALL neighbors within network walking distance (not top-K truncation)
3. **Encodes spatial relationships** - Model needs explicit information about neighbor locations relative to target
4. **Efficient training** - Must handle ~500 neighbors per target point without excessive memory usage
5. **Domain-specific loss** - Uses distance-based probability vectors as targets (not hard labels)

Current approach (sequence-based transformer) requires padding/masking and doesn't naturally handle variable neighbors. Graph Transformer architecture solves these issues while being more memory-efficient and principled.

## Solution Statement

Implement a Graph Transformer architecture that:
- Uses PyTorch Geometric's TransformerConv for attention-based neighbor aggregation
- Builds star graphs: target point (node 0) connected to all neighbors (nodes 1-N) within network walking distance
- Encodes spatial relationships via edge attributes: [dx, dy, euclidean_distance, network_distance]
- Handles variable graph sizes naturally (no padding needed)
- Pre-computes network distances during feature engineering for efficiency
- Stores features + metadata, builds graphs on-the-fly in Dataset class
- Uses existing distance-based KL divergence loss function

---

## Feature Metadata

**Feature Type**: New Model Architecture + Feature Engineering Pipeline
**Estimated Complexity**: Very High
**Primary Systems Affected**: 
- `src/data/collection/` (new feature engineering module)
- `src/training/` (new model architecture and training code)
- `src/evaluation/` (evaluation metrics compatible with graph model)
- `models/config.yaml` (add graph transformer config)
- `data/processed/` (new directory structure for processed features)
- `requirements.txt` (add PyTorch Geometric dependency)
**Dependencies**: 
- torch-geometric 2.3+ (graph neural networks)
- torch 2.0+ (already in requirements)
- osmnx 1.6+ (network distance calculations, already in requirements)
- geopandas 0.14+ (geospatial operations, already in requirements)
- networkx 3.0+ (graph operations, already in requirements)

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `src/utils/helpers.py` (lines 24-57) - Why: `load_neighborhoods()` function to load GeoJSON
- `src/utils/helpers.py` (lines 60-89) - Why: `get_compliant_neighborhoods()` filter function
- `src/utils/helpers.py` (lines 92-117) - Why: `get_non_compliant_neighborhoods()` filter function
- `src/utils/helpers.py` (lines 120-243) - Why: `get_service_category_mapping()` returns 8 categories
- `src/utils/helpers.py` (lines 246-260) - Why: `get_service_category_names()` returns ordered list
- `src/utils/helpers.py` (lines 263-288) - Why: `normalize_distance_by_15min()` for distance normalization
- `src/utils/helpers.py` (lines 314-337) - Why: `set_random_seeds()` for reproducibility
- `src/utils/config.py` (all lines) - Why: Configuration loading pattern
- `src/utils/logging.py` (all lines) - Why: Logging setup pattern
- `src/data/collection/osm_extractor.py` - Why: OSM data extraction, network graph loading
- `src/data/collection/census_loader.py` - Why: Census data loading patterns
- `models/config.yaml` (all lines) - Why: Configuration structure to extend
- `docs/distance-based-loss-calculation.md` (all lines) - Why: Loss function specification
- `PRD.md` (lines 666-728) - Why: Feature specification and model requirements
- `CURSOR.md` (lines 132-195) - Why: Feature engineering rationale

### New Files to Create

- `src/data/collection/feature_engineer.py` - Feature engineering pipeline (grid cells, features, network distances)
- `src/training/model.py` - Graph Transformer model architecture
- `src/training/dataset.py` - PyTorch Geometric Dataset class
- `src/training/train.py` - Training script with distance-based loss
- `src/training/loss.py` - Distance-based KL divergence loss function
- `src/evaluation/metrics.py` - Evaluation metrics for graph model
- `tests/unit/test_feature_engineer.py` - Unit tests for feature engineering
- `tests/unit/test_model.py` - Unit tests for model architecture
- `tests/unit/test_dataset.py` - Unit tests for dataset class
- `tests/unit/test_loss.py` - Unit tests for loss function
- `tests/integration/test_training_pipeline.py` - Integration tests for full pipeline

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [PyTorch Geometric Documentation - TransformerConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv)
  - Specific section: TransformerConv with edge attributes
  - Why: Core graph attention layer for the model
- [PyTorch Geometric Documentation - Data Objects](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)
  - Specific section: Building Data objects, batching
  - Why: Dataset structure for graph data
- [PyTorch Geometric Documentation - DataLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.DataLoader)
  - Specific section: Batching variable-sized graphs
  - Why: Efficient batching of graphs with different numbers of nodes
- [OSMnx Documentation - Distance Calculations](https://osmnx.readthedocs.io/)
  - Specific section: Network distance calculations
  - Why: Calculate network walking distances for neighbor filtering and edge attributes
- [NetworkX Documentation - Shortest Path](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html)
  - Specific section: shortest_path_length with weight='length'
  - Why: Calculate network distances between points

### Patterns to Follow

**Naming Conventions:**
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Use descriptive names: `build_star_graph()`, `compute_network_distance()`, not `build_graph()`, `get_dist()`
- Follow existing pattern: `*_engineer.py` for feature engineering, `*_model.py` for models

**Error Handling:**
- Use try-except blocks with specific exception types
- Log errors with context (target point ID, neighborhood name, operation type)
- Continue processing other target points if one fails
- Pattern from `helpers.py`: Convert exceptions to domain-specific errors

**Logging Pattern:**
```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.info(f"Processing target point {point_id} in {neighborhood_name}")
logger.debug(f"Found {num_neighbors} neighbors within network walking distance")
logger.warning(f"Low neighbor count ({num_neighbors}) for point {point_id}")
logger.error(f"Failed to compute network distance: {error}")
```

**Configuration Pattern:**
```python
from src.utils.config import get_config

config = get_config()
walk_radius = config.get("features", {}).get("walk_15min_radius_meters", 1200)
cell_size = config.get("features", {}).get("grid_cell_size_meters", 100)
```

**Type Hints Pattern:**
- Always use type hints for function parameters and returns
- Use `Optional[Type]` for nullable values
- Use `List[Type]`, `Dict[str, Type]`, `Tuple[Type1, Type2]` for collections
- Use `torch.Tensor`, `gpd.GeoDataFrame` for specific types

**Docstring Pattern (Google Style):**
```python
def compute_network_distance(
    point1: Point, point2: Point, network_graph: nx.Graph
) -> float:
    """Calculate network walking distance between two points.

    Args:
        point1: Shapely Point for origin location.
        point2: Shapely Point for destination location.
        network_graph: NetworkX graph of walkable street network.

    Returns:
        Network walking distance in meters.

    Raises:
        ValueError: If points cannot be connected in network graph.
        RuntimeError: If network distance calculation fails.

    Example:
        >>> dist = compute_network_distance(point1, point2, G)
        >>> print(f"Walking distance: {dist:.1f}m")
    """
```

---

## IMPLEMENTATION PLAN

### Phase 1: Feature Engineering Pipeline

**Goal**: Build complete feature engineering pipeline that generates grid cells, computes features, and calculates network distances for all target points and their neighbors.

#### Task 1.1: Create Feature Engineering Module Structure

**File**: `src/data/collection/feature_engineer.py`

**Classes to Implement**:
- `FeatureEngineer`: Main class for feature engineering pipeline

**Key Methods**:
- `__init__(self) -> None`: Initialize with config, load OSM/Census data paths
- `generate_target_points(self, neighborhood: gpd.GeoDataFrame, sampling_interval_meters: float = 100.0) -> gpd.GeoDataFrame`: Generate regular grid of target points within neighborhood
- `generate_grid_cells(self, target_point: Point, cell_size_meters: float) -> List[Point]`: Generate regular grid cells around target point
- `filter_by_network_distance(self, target_point: Point, grid_cells: List[Point], network_graph: nx.Graph, max_distance_meters: float) -> List[Dict]`: Filter grid cells by network walking distance, return list with distances
- `compute_point_features(self, point: Point, neighborhood_name: str) -> np.ndarray`: Compute all 33 features for a single point (demographics, built form, services, walkability)
- `compute_network_distance(self, point1: Point, point2: Point, network_graph: nx.Graph) -> float`: Calculate network walking distance between two points
- `process_target_point(self, target_point: Point, neighborhood_name: str, network_graph: nx.Graph) -> Dict`: Process single target point: generate neighbors, compute features, calculate distances
- `process_neighborhood(self, neighborhood: gpd.GeoDataFrame) -> pd.DataFrame`: Process all target points in a neighborhood
- `process_all_neighborhoods(self, neighborhoods: gpd.GeoDataFrame) -> pd.DataFrame`: Process all neighborhoods, save results

**Implementation Details**:

1. **Target Point Generation**:
   - Generate regular grid of target points within neighborhood boundary
   - Default sampling interval: 100m (configurable)
   - Use `shapely.ops.unary_union` to get neighborhood polygon
   - Generate grid points, filter to those inside polygon
   - Store as GeoDataFrame with columns: `target_id`, `neighborhood_name`, `geometry`, `label` (compliant/non-compliant)

2. **Grid Cell Generation**:
   - For each target point, generate regular grid cells
   - Grid centered on target point
   - Cell size: 100m × 100m (configurable via `grid_cell_size_meters`)
   - Generate cells covering bounding box: `[target.x ± radius, target.y ± radius]`
   - Return list of Shapely Point objects for cell centers

3. **Network Distance Filtering**:
   - For each grid cell, calculate network walking distance to target point
   - Use OSMnx network graph (loaded from `data/raw/osm/{status}/{neighborhood}/network.graphml`)
   - Find nearest network nodes to target and cell using `ox.distance.nearest_nodes()`
   - Calculate shortest path distance using `nx.shortest_path_length(G, origin_node, dest_node, weight='length')`
   - Filter cells where `network_distance <= walk_15min_radius_meters` (default: 1200m)
   - Return list of dicts: `[{'cell': Point, 'network_distance': float, 'euclidean_distance': float, 'dx': float, 'dy': float}, ...]`

4. **Feature Computation**:
   - For each point (target or neighbor), compute 33 features:
     - Demographics (17 features): Load from Census data, interpolate to point location
     - Built Form (4 features): Compute from building GeoDataFrame
     - Services (8 features): Count services within 15-min radius from point's perspective
     - Walkability (4 features): Compute from network graph and OSM data
   - Return as numpy array of shape `(33,)`

5. **Network Distance Calculation**:
   - Find nearest network nodes to both points
   - Calculate shortest path distance
   - Handle edge cases: points outside network, disconnected components
   - Return distance in meters, or `float('inf')` if unreachable

**Output Format**:
- Save processed data as Parquet files: `data/processed/features/{status}/{neighborhood}/target_points.parquet`
- Each row represents one target point with:
  - `target_id`: Unique identifier
  - `neighborhood_name`: Neighborhood name
  - `label`: Compliance status
  - `target_features`: Array of 33 features for target point
  - `neighbor_data`: List of dicts, each with `features`, `network_distance`, `euclidean_distance`, `dx`, `dy`
  - `target_geometry`: Point geometry (WGS84)
  - `num_neighbors`: Number of neighbors within network distance

#### Task 1.2: Implement Target Point Generation

**Function**: `generate_target_points()`

**Algorithm**:
1. Get neighborhood polygon from GeoDataFrame
2. Calculate bounding box
3. Generate regular grid points with `sampling_interval_meters` spacing
4. Filter points inside polygon using `polygon.contains(point)`
5. Create GeoDataFrame with target points
6. Add metadata: `target_id`, `neighborhood_name`, `label`

**Edge Cases**:
- Empty neighborhood polygon → return empty GeoDataFrame
- Very small neighborhood → may have 0 target points (log warning)

#### Task 1.3: Implement Grid Cell Generation

**Function**: `generate_grid_cells()`

**Algorithm**:
1. Calculate bounding box: `[target.x ± radius, target.y ± radius]` where `radius = walk_15min_radius_meters`
2. Generate grid cells with `cell_size_meters` spacing
3. Return list of Point objects (cell centers)

**Edge Cases**:
- Target point near neighborhood boundary → some cells may be outside (filter later by network distance)

#### Task 1.4: Implement Network Distance Filtering

**Function**: `filter_by_network_distance()`

**Algorithm**:
1. Load network graph for neighborhood (from OSM extractor output)
2. For each grid cell:
   - Find nearest network node to target: `ox.distance.nearest_nodes(G, target.x, target.y)`
   - Find nearest network node to cell: `ox.distance.nearest_nodes(G, cell.x, cell.y)`
   - Calculate shortest path: `nx.shortest_path_length(G, target_node, cell_node, weight='length')`
   - If distance <= `max_distance_meters`, include in results
   - Calculate Euclidean distance and relative coordinates (dx, dy)
3. Return list of neighbor dicts

**Edge Cases**:
- Point outside network coverage → use Euclidean distance as fallback, log warning
- Disconnected network components → return `float('inf')` for unreachable cells
- Empty network graph → log error, return empty list

#### Task 1.5: Implement Feature Computation

**Function**: `compute_point_features()`

**Algorithm**:
1. **Demographics (17 features)**:
   - Load Census data for neighborhood (from CensusLoader output)
   - Use spatial join to find IRIS unit containing point
   - Extract demographic features: population_density, ses_index, car_ownership_rate, etc.
   - Handle missing data: use neighborhood average or log warning

2. **Built Form (4 features)**:
   - Load building GeoDataFrame for neighborhood
   - Calculate building density: count buildings within 100m radius
   - Calculate building count, average levels, floor area per capita
   - Use spatial operations: `gdf[gdf.geometry.within(buffer_polygon)]`

3. **Services (8 features)**:
   - Load service GeoDataFrames by category (from OSMExtractor output)
   - For each category, count services within `walk_15min_radius_meters` from point
   - Use network distance (not Euclidean) for accurate counts
   - Return array: `[count_education, count_entertainment, ..., count_shops]`

4. **Walkability (4 features)**:
   - Load network graph for neighborhood
   - Calculate intersection density: count nodes within 200m radius
   - Calculate average block length: average edge length in local area
   - Calculate pedestrian street ratio: proportion of pedestrian-only streets
   - Calculate sidewalk presence: binary indicator (requires OSM data)

**Output**: Numpy array of shape `(33,)` with dtype `float32`

#### Task 1.6: Implement Distance-Based Target Vector Computation

**Function**: `compute_target_probability_vector()`

**Algorithm** (from `docs/distance-based-loss-calculation.md`):
1. For each of 8 service categories:
   - Find nearest service in category to target point
   - Calculate network walking distance
   - If no service found, use `missing_service_penalty` (default: 2400m)
2. Build distance vector: `D = [d_0, d_1, ..., d_7]`
3. Convert to probability vector using temperature-scaled softmax:
   - `P_j = exp(-d_j / τ) / Σⱼ exp(-d_j / τ)`
   - Temperature `τ = 200m` (configurable)
4. Return probability vector of shape `(8,)`

**File**: Add to `src/data/collection/feature_engineer.py` or separate `src/training/loss.py`

#### Task 1.7: Create Feature Engineering Script

**File**: `scripts/run_feature_engineering.py`

**Functionality**:
- Load neighborhoods from `paris_neighborhoods.geojson`
- Filter to compliant neighborhoods (for training data)
- Process each neighborhood using `FeatureEngineer`
- Save processed features to `data/processed/features/`
- Generate summary statistics and validation reports

### Phase 2: Graph Transformer Model Architecture

**Goal**: Implement Graph Transformer model using PyTorch Geometric with star graph structure and edge attributes.

#### Task 2.1: Create Model Architecture

**File**: `src/training/model.py`

**Class**: `SpatialGraphTransformer(nn.Module)`

**Architecture**:
```python
class SpatialGraphTransformer(nn.Module):
    def __init__(
        self,
        num_features: int = 33,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 8,
        edge_dim: int = 4,  # [dx, dy, euclidean_dist, network_dist]
    ):
        super().__init__()
        
        # Node encoder: 33 features → 128-dim embedding
        self.node_encoder = nn.Linear(num_features, hidden_dim)
        
        # Edge encoder: 4 edge attributes → 128-dim embedding
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        # Graph transformer layers (2-3 layers)
        self.conv_layers = nn.ModuleList([
            TransformerConv(
                hidden_dim, hidden_dim,
                edge_dim=hidden_dim,  # Use edge embeddings
                heads=num_heads,
                dropout=dropout,
                concat=True  # Concatenate multi-head outputs
            )
            for _ in range(num_layers)
        ])
        
        # Projection after concatenated heads
        self.head_projection = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
        # Classification head (from target node only)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        # data.x: [N, 33] node features (N = 1 + num_neighbors)
        # data.edge_index: [2, E] edges (neighbors → target)
        # data.edge_attr: [E, 4] edge attributes
        # data.batch: [N] batch assignment (for batched graphs)
        
        # Encode nodes and edges
        x = self.node_encoder(data.x)  # [N, hidden_dim]
        edge_emb = self.edge_encoder(data.edge_attr)  # [E, hidden_dim]
        
        # Graph transformer layers
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            # Multi-head attention with edge attributes
            x_new = conv(x, data.edge_index, edge_emb)  # [N, hidden_dim * num_heads]
            x_new = self.head_projection(x_new)  # [N, hidden_dim]
            x_new = norm(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
        
        # Extract target node embeddings (node 0 in each graph)
        # For batched graphs, need to extract first node of each graph
        if hasattr(data, 'batch'):
            # Batched graphs: extract node 0 of each graph
            batch_size = data.batch.max().item() + 1
            target_indices = []
            for i in range(batch_size):
                graph_nodes = (data.batch == i).nonzero(as_tuple=True)[0]
                target_indices.append(graph_nodes[0])  # First node is always target
            target_emb = x[target_indices]  # [batch_size, hidden_dim]
        else:
            # Single graph: node 0 is target
            target_emb = x[0:1]  # [1, hidden_dim]
        
        # Classify
        logits = self.classifier(target_emb)  # [batch_size, num_classes]
        return logits
```

**Key Implementation Details**:
- Use `TransformerConv` from `torch_geometric.nn.conv`
- Edge attributes are encoded and used in attention mechanism
- Residual connections and layer normalization for stable training
- Extract only target node (node 0) for classification
- Handle both single graphs and batched graphs

#### Task 2.2: Add Model Configuration

**File**: `models/config.yaml`

**Add to config**:
```yaml
model:
  name: "spatial_graph_transformer"
  architecture: "graph_transformer"
  num_features: 33
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.1
  num_classes: 8
  edge_dim: 4  # [dx, dy, euclidean_dist, network_dist]
```

### Phase 3: Dataset Class

**Goal**: Create PyTorch Geometric Dataset class that builds graph Data objects from pre-computed features.

#### Task 3.1: Create Dataset Class

**File**: `src/training/dataset.py`

**Class**: `SpatialGraphDataset(Dataset)`

**Implementation**:
```python
from torch_geometric.data import Dataset, Data
import pandas as pd
import torch
import numpy as np

class SpatialGraphDataset(Dataset):
    def __init__(
        self,
        features_df: pd.DataFrame,
        root: str = "data/processed/features",
        transform=None,
        pre_transform=None,
    ):
        self.features_df = features_df
        self.root = root
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return []  # Not using raw files
    
    @property
    def processed_file_names(self):
        return [f"graph_{idx}.pt" for idx in range(len(self.features_df))]
    
    def len(self):
        return len(self.features_df)
    
    def get(self, idx):
        # Get target point data
        row = self.features_df.iloc[idx]
        
        # Extract features
        target_features = torch.tensor(row['target_features'], dtype=torch.float32)  # [33]
        neighbor_data = row['neighbor_data']  # List of dicts
        target_prob_vector = torch.tensor(row['target_prob_vector'], dtype=torch.float32)  # [8]
        
        # Build node features: [target, neighbor1, neighbor2, ...]
        num_neighbors = len(neighbor_data)
        node_features = [target_features]
        for neighbor in neighbor_data:
            neighbor_feat = torch.tensor(neighbor['features'], dtype=torch.float32)  # [33]
            node_features.append(neighbor_feat)
        x = torch.stack(node_features)  # [1 + num_neighbors, 33]
        
        # Build edge index: star graph (all neighbors → target)
        if num_neighbors > 0:
            edge_index = torch.tensor([
                list(range(1, num_neighbors + 1)),  # Source: neighbors (1 to N)
                [0] * num_neighbors  # Target: center (always 0)
            ], dtype=torch.long)  # [2, num_neighbors]
            
            # Build edge attributes: [dx, dy, euclidean_dist, network_dist]
            edge_attr = torch.tensor([
                [
                    neighbor['dx'],
                    neighbor['dy'],
                    neighbor['euclidean_distance'],
                    neighbor['network_distance']
                ]
                for neighbor in neighbor_data
            ], dtype=torch.float32)  # [num_neighbors, 4]
        else:
            # No neighbors: empty edge index and attributes
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float32)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,  # [1 + num_neighbors, 33]
            edge_index=edge_index,  # [2, num_neighbors]
            edge_attr=edge_attr,  # [num_neighbors, 4]
            y=target_prob_vector,  # [8] - target probability vector
            target_id=row['target_id'],
            neighborhood_name=row['neighborhood_name'],
            num_nodes=1 + num_neighbors
        )
        
        return data
```

**Key Implementation Details**:
- Load pre-computed features from Parquet files
- Build star graph structure: target (node 0) + neighbors (nodes 1-N)
- Edge attributes: [dx, dy, euclidean_distance, network_distance]
- Handle edge case: target points with 0 neighbors (empty edge index)
- Store metadata: target_id, neighborhood_name for tracking

#### Task 3.2: Add Dataset Utilities

**Functions**:
- `create_train_val_test_splits(features_df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: Create stratified splits by neighborhood
- `load_features_from_directory(directory: str) -> pd.DataFrame`: Load all processed features from directory

### Phase 4: Loss Function

**Goal**: Implement distance-based KL divergence loss function.

#### Task 4.1: Create Loss Function Module

**File**: `src/training/loss.py`

**Function**: `distance_based_kl_loss()`

**Implementation** (from `docs/distance-based-loss-calculation.md`):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distance_based_kl_loss(
    predicted_logits: torch.Tensor,
    target_prob_vector: torch.Tensor,
    temperature: float = 200.0,
) -> torch.Tensor:
    """Compute KL divergence loss between predicted and distance-based target probabilities.
    
    Args:
        predicted_logits: Model output logits [batch_size, 8]
        target_prob_vector: Distance-based target probability vector [batch_size, 8]
        temperature: Temperature parameter for target vector construction (in meters)
    
    Returns:
        Scalar loss value
    """
    # Convert logits to probabilities
    predicted_probs = F.softmax(predicted_logits, dim=-1)  # [batch_size, 8]
    
    # Compute KL divergence: KL(target || predicted)
    # KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    kl_loss = F.kl_div(
        F.log_softmax(predicted_logits, dim=-1),
        target_prob_vector,
        reduction='batchmean'
    )
    
    return kl_loss
```

**Class**: `DistanceBasedKLLoss(nn.Module)` (wrapper class)

### Phase 5: Training Script

**Goal**: Create training script with proper data loading, training loop, and checkpointing.

#### Task 5.1: Create Training Script

**File**: `src/training/train.py`

**Key Components**:
1. **Data Loading**:
   - Load processed features
   - Create train/val/test splits (only from compliant neighborhoods)
   - Create PyTorch Geometric DataLoaders
   - Use `DataLoader` from `torch_geometric.loader` for efficient batching

2. **Model Initialization**:
   - Load config
   - Initialize `SpatialGraphTransformer` model
   - Move to GPU if available

3. **Optimizer and Scheduler**:
   - AdamW optimizer
   - Learning rate scheduler (optional: ReduceLROnPlateau)

4. **Training Loop**:
   - For each epoch:
     - Train on training set
     - Validate on validation set
     - Log metrics (loss, KL divergence, accuracy)
     - Save checkpoint if validation loss improves
     - Early stopping if no improvement

5. **Checkpointing**:
   - Save model state, optimizer state, epoch, best validation loss
   - Save to `models/checkpoints/graph_transformer_best.pt`

6. **Logging**:
   - Use experiment logging from `src/utils/logging.py`
   - Log to `experiments/runs/{timestamp}/training.log`

**Training Loop Structure**:
```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch)
            loss = loss_fn(logits, batch.y)
            val_loss += loss.item()
    
    # Logging and checkpointing
    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, best_val_loss)
```

### Phase 6: Evaluation Metrics

**Goal**: Create evaluation metrics compatible with graph model and distance-based targets.

#### Task 6.1: Create Metrics Module

**File**: `src/evaluation/metrics.py`

**Functions**:
- `compute_kl_divergence(predicted_probs: torch.Tensor, target_probs: torch.Tensor) -> float`: Compute KL divergence
- `compute_top_k_accuracy(predicted_probs: torch.Tensor, target_probs: torch.Tensor, k: int = 1) -> float`: Top-k accuracy (compare predicted top-k with target top-k)
- `compute_distance_alignment(predicted_probs: torch.Tensor, actual_distances: torch.Tensor, temperature: float = 200.0) -> float`: Measure how well predicted probabilities align with actual distances
- `evaluate_model(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Dict[str, float]`: Comprehensive evaluation function

### Phase 7: Configuration Updates

**Goal**: Update configuration files to support graph transformer architecture.

#### Task 7.1: Update models/config.yaml

**Add/Update Sections**:
```yaml
model:
  name: "spatial_graph_transformer"
  architecture: "graph_transformer"
  num_features: 33
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.1
  num_classes: 8
  edge_dim: 4

features:
  walk_15min_radius_meters: 1200
  grid_cell_size_meters: 100
  target_point_sampling_interval_meters: 100  # NEW: Grid spacing for target points
  filter_by_network_distance: true  # NEW: Filter neighbors by network distance
  include_all_within_radius: true  # NEW: Include ALL neighbors (no truncation)

training:
  batch_size: 16  # Reduced for graph model (can be larger with efficient batching)
  learning_rate: 0.001
  weight_decay: 0.0001
  num_epochs: 100
  early_stopping_patience: 10
  early_stopping_min_delta: 0.001
  use_mixed_precision: false  # NEW: Can enable FP16 for memory savings
```

#### Task 7.2: Update requirements.txt

**Add Dependency**:
```
torch-geometric>=2.3.0
```

### Phase 8: Testing

**Goal**: Create comprehensive unit and integration tests.

#### Task 8.1: Unit Tests for Feature Engineering

**File**: `tests/unit/test_feature_engineer.py`

**Test Cases**:
- `test_generate_target_points()`: Test target point generation
- `test_generate_grid_cells()`: Test grid cell generation
- `test_filter_by_network_distance()`: Test network distance filtering
- `test_compute_point_features()`: Test feature computation
- `test_compute_network_distance()`: Test network distance calculation
- `test_process_target_point()`: Test full target point processing

#### Task 8.2: Unit Tests for Model

**File**: `tests/unit/test_model.py`

**Test Cases**:
- `test_model_initialization()`: Test model creation
- `test_forward_single_graph()`: Test forward pass with single graph
- `test_forward_batched_graphs()`: Test forward pass with batched graphs
- `test_output_shape()`: Test output tensor shapes
- `test_edge_attributes()`: Test edge attribute encoding

#### Task 8.3: Unit Tests for Dataset

**File**: `tests/unit/test_dataset.py`

**Test Cases**:
- `test_dataset_creation()`: Test dataset initialization
- `test_get_item()`: Test data loading
- `test_star_graph_structure()`: Test star graph construction
- `test_edge_attributes()`: Test edge attribute construction
- `test_empty_neighbors()`: Test handling of target points with 0 neighbors

#### Task 8.4: Unit Tests for Loss Function

**File**: `tests/unit/test_loss.py`

**Test Cases**:
- `test_kl_divergence()`: Test KL divergence calculation
- `test_loss_computation()`: Test loss function
- `test_gradient_flow()`: Test gradients flow correctly

#### Task 8.5: Integration Tests

**File**: `tests/integration/test_training_pipeline.py`

**Test Cases**:
- `test_end_to_end_training()`: Test full training pipeline
- `test_data_loading()`: Test data loading and batching
- `test_model_training()`: Test model training for a few epochs
- `test_checkpointing()`: Test model checkpoint saving/loading

### Phase 9: Documentation Updates

**Goal**: Update project documentation to reflect graph transformer architecture.

#### Task 9.1: Update PRD.md

**Sections to Update**:
- Section 6: "Core Architecture & Patterns" - Update model architecture description
- Section 7: "Tools/Features" - Update model description
- Section 12: "Implementation Phases" - Update Phase 2 (Model Training) deliverables
- Appendix: "Feature Specification" - Update input structure description

**Key Changes**:
- Change from "FT-Transformer" to "Spatial Graph Transformer"
- Update input structure: from "multi-point sequences" to "star graphs"
- Update architecture description: from "sequence transformer" to "graph transformer with TransformerConv"
- Keep distance-based loss function description (no changes)
- Update feature engineering description: network distance filtering, graph construction

#### Task 9.2: Update CURSOR.md

**Sections to Update**:
- Section "Model Architecture" - Update to graph transformer
- Section "Feature Categories" - Update input structure description
- Section "Grid Cell Generation" - Add network distance filtering details
- Section "Code Conventions" - Add PyTorch Geometric patterns

**Key Changes**:
- Update model architecture section
- Add graph transformer specific patterns
- Update feature engineering rationale

#### Task 9.3: Update README.md

**Sections to Update**:
- "Project Overview" - Update model description
- "Development Workflow" - Update model training steps
- "Project Structure" - Add new files

#### Task 9.4: Create Model Architecture Documentation

**File**: `docs/graph-transformer-architecture.md`

**Content**:
- Architecture overview
- Star graph structure explanation
- Edge attribute encoding
- Attention mechanism details
- Training procedure
- Memory considerations

### Phase 10: Scripts and Utilities

**Goal**: Create scripts for running feature engineering and training.

#### Task 10.1: Feature Engineering Script

**File**: `scripts/run_feature_engineering.py`

**Functionality**:
- Load neighborhoods from `paris_neighborhoods.geojson`
- Filter to compliant neighborhoods (or process all)
- Initialize `FeatureEngineer`
- Process each neighborhood
- Save processed features
- Generate summary report

#### Task 10.2: Training Script

**File**: `scripts/train_graph_transformer.py`

**Functionality**:
- Load configuration
- Load processed features
- Create train/val/test splits
- Initialize model, optimizer, loss function
- Run training loop
- Save best model checkpoint
- Generate training report

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Feature Engineering Pipeline
- [ ] Create `src/data/collection/feature_engineer.py` with `FeatureEngineer` class
- [ ] Implement `generate_target_points()` method
- [ ] Implement `generate_grid_cells()` method
- [ ] Implement `filter_by_network_distance()` method
- [ ] Implement `compute_point_features()` method
- [ ] Implement `compute_network_distance()` method
- [ ] Implement `compute_target_probability_vector()` method
- [ ] Implement `process_target_point()` method
- [ ] Implement `process_neighborhood()` method
- [ ] Implement `process_all_neighborhoods()` method
- [ ] Create `scripts/run_feature_engineering.py` script
- [ ] Add unit tests in `tests/unit/test_feature_engineer.py`

### Phase 2: Model Architecture
- [ ] Create `src/training/model.py` with `SpatialGraphTransformer` class
- [ ] Implement node encoder (Linear layer)
- [ ] Implement edge encoder (Linear layer)
- [ ] Implement TransformerConv layers with edge attributes
- [ ] Implement residual connections and layer normalization
- [ ] Implement target node extraction (for batched and single graphs)
- [ ] Implement classification head
- [ ] Add model configuration to `models/config.yaml`
- [ ] Add unit tests in `tests/unit/test_model.py`

### Phase 3: Dataset Class
- [ ] Create `src/training/dataset.py` with `SpatialGraphDataset` class
- [ ] Implement `__init__()` method
- [ ] Implement `len()` method
- [ ] Implement `get()` method (builds Data objects)
- [ ] Implement star graph construction
- [ ] Implement edge attribute construction
- [ ] Handle edge case: 0 neighbors
- [ ] Add dataset utilities (splits, loading)
- [ ] Add unit tests in `tests/unit/test_dataset.py`

### Phase 4: Loss Function
- [ ] Create `src/training/loss.py` with `distance_based_kl_loss()` function
- [ ] Implement KL divergence calculation
- [ ] Create `DistanceBasedKLLoss` wrapper class
- [ ] Add unit tests in `tests/unit/test_loss.py`

### Phase 5: Training Script
- [ ] Create `src/training/train.py` with training loop
- [ ] Implement data loading with PyTorch Geometric DataLoader
- [ ] Implement model initialization
- [ ] Implement optimizer and scheduler setup
- [ ] Implement training loop
- [ ] Implement validation loop
- [ ] Implement checkpointing
- [ ] Implement early stopping
- [ ] Implement logging
- [ ] Create `scripts/train_graph_transformer.py` script

### Phase 6: Evaluation Metrics
- [ ] Create `src/evaluation/metrics.py`
- [ ] Implement `compute_kl_divergence()` function
- [ ] Implement `compute_top_k_accuracy()` function
- [ ] Implement `compute_distance_alignment()` function
- [ ] Implement `evaluate_model()` function

### Phase 7: Configuration
- [ ] Update `models/config.yaml` with graph transformer config
- [ ] Add feature engineering config parameters
- [ ] Update `requirements.txt` with `torch-geometric>=2.3.0`

### Phase 8: Testing
- [ ] Write unit tests for feature engineering
- [ ] Write unit tests for model
- [ ] Write unit tests for dataset
- [ ] Write unit tests for loss function
- [ ] Write integration tests for training pipeline
- [ ] Run all tests and ensure they pass

### Phase 9: Documentation
- [ ] Update `PRD.md` with graph transformer architecture
- [ ] Update `CURSOR.md` with graph transformer patterns
- [ ] Update `README.md` with new model description
- [ ] Create `docs/graph-transformer-architecture.md`

### Phase 10: Scripts
- [ ] Create `scripts/run_feature_engineering.py`
- [ ] Create `scripts/train_graph_transformer.py`
- [ ] Test scripts with sample data

---

## VALIDATION CRITERIA

### Feature Engineering
- ✅ Target points generated correctly (regular grid within neighborhoods)
- ✅ Grid cells generated correctly (centered on target points)
- ✅ Network distance filtering works (only includes neighbors within walking distance)
- ✅ All 33 features computed correctly for target and neighbor points
- ✅ Network distances calculated accurately (matches OSMnx calculations)
- ✅ Target probability vectors computed correctly (matches distance-based loss spec)
- ✅ Handles edge cases: empty neighborhoods, points outside network, 0 neighbors

### Model Architecture
- ✅ Model initializes correctly with config parameters
- ✅ Forward pass works for single graphs
- ✅ Forward pass works for batched graphs (variable sizes)
- ✅ Output shape is correct: [batch_size, 8]
- ✅ Edge attributes are used in attention mechanism
- ✅ Target node extraction works correctly (node 0 in each graph)
- ✅ Gradients flow correctly through all layers

### Dataset
- ✅ Dataset loads pre-computed features correctly
- ✅ Star graph structure built correctly (target node 0, neighbors 1-N)
- ✅ Edge attributes constructed correctly: [dx, dy, euclidean_dist, network_dist]
- ✅ Handles variable number of neighbors (no padding needed)
- ✅ Handles edge case: 0 neighbors (empty edge index)
- ✅ DataLoader batches graphs efficiently

### Training
- ✅ Training loop runs without errors
- ✅ Loss decreases over epochs
- ✅ Validation loss tracked correctly
- ✅ Checkpointing works (save/load model)
- ✅ Early stopping works correctly
- ✅ Logging works correctly

### Integration
- ✅ End-to-end pipeline works: feature engineering → dataset → training
- ✅ Model trains on compliant neighborhoods only
- ✅ Distance-based loss function works correctly
- ✅ Evaluation metrics computed correctly

---

## FILES TO UPDATE

### New Files to Create
1. `src/data/collection/feature_engineer.py` - Feature engineering pipeline
2. `src/training/model.py` - Graph Transformer model
3. `src/training/dataset.py` - PyTorch Geometric Dataset class
4. `src/training/train.py` - Training script
5. `src/training/loss.py` - Distance-based loss function
6. `src/evaluation/metrics.py` - Evaluation metrics
7. `scripts/run_feature_engineering.py` - Feature engineering script
8. `scripts/train_graph_transformer.py` - Training script
9. `tests/unit/test_feature_engineer.py` - Feature engineering tests
10. `tests/unit/test_model.py` - Model tests
11. `tests/unit/test_dataset.py` - Dataset tests
12. `tests/unit/test_loss.py` - Loss function tests
13. `tests/integration/test_training_pipeline.py` - Integration tests
14. `docs/graph-transformer-architecture.md` - Architecture documentation

### Files to Update
1. `models/config.yaml` - Add graph transformer configuration
2. `requirements.txt` - Add `torch-geometric>=2.3.0`
3. `PRD.md` - Update model architecture sections (Section 6, 7, 12, Appendix)
4. `CURSOR.md` - Update model architecture and feature engineering sections
5. `README.md` - Update model description and workflow

### Directory Structure to Create
```
data/processed/
├── features/
│   ├── compliant/
│   │   ├── {neighborhood1}/
│   │   │   └── target_points.parquet
│   │   └── {neighborhood2}/
│   │       └── target_points.parquet
│   └── non_compliant/
│       └── {neighborhood}/
│           └── target_points.parquet
```

---

## ASSUMPTIONS AND DECISIONS

### Assumptions
1. **Target Point Selection**: Regular grid sampling with 100m interval (configurable)
2. **Network Distance Pre-computation**: Network distances are pre-computed during feature engineering for efficiency
3. **Data Storage**: Store features + metadata in Parquet format, build graphs on-the-fly in Dataset class
4. **Grid Cell Generation**: Generate new grid for each target point (centered on target)
5. **Neighbor Filtering**: Filter by network walking distance (not Euclidean) - only truly walkable neighbors included
6. **All Neighbors Included**: Include ALL neighbors within network walking distance (no top-K truncation)

### Decisions
1. **Graph Structure**: Star graph (target connected to all neighbors) - most natural for this problem
2. **Edge Attributes**: [dx, dy, euclidean_distance, network_distance] - explicit spatial encoding
3. **Model Architecture**: TransformerConv with edge attributes - allows attention to use spatial information
4. **Loss Function**: Keep existing distance-based KL divergence loss - domain-specific and important
5. **Batching**: Use PyTorch Geometric DataLoader - efficient batching of variable-sized graphs
6. **Feature Engineering Timing**: Pre-compute all features and network distances - faster training, more storage

---

## NOTES

- This plan assumes PyTorch Geometric 2.3+ is available
- Network distance calculations may be slow - consider parallelization or caching
- Memory usage: With ~500 neighbors per graph, batch size of 16-32 should work on most GPUs
- If memory issues occur, can reduce batch size or use gradient accumulation
- Feature engineering is the most time-consuming phase - consider parallel processing
- All code should follow existing patterns from `osm_extractor.py` and `census_loader.py`

---

*Plan Version: 1.0*  
*Last Updated: January 2025*  
*Project: AI4SI - 15-Minute City Service Gap Prediction Model*
