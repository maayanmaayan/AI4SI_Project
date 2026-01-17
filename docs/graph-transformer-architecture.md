# Spatial Graph Transformer Architecture

## Overview

The Spatial Graph Transformer is a graph neural network architecture designed for classifying geographic points based on their spatial context. The model uses a star graph structure where a target point (center) aggregates information from all neighboring grid cells within a Euclidean distance threshold. The architecture naturally handles variable numbers of neighbors without padding, uses explicit spatial encoding via edge attributes, and maintains permutation invariance.

## Architecture Components

### 1. Star Graph Structure

Each prediction location is represented as a **star graph**:

- **Node 0**: Target point (the location for which we predict service gaps)
- **Nodes 1-N**: Neighbor grid cells within Euclidean distance threshold (default: 1200m)
- **Edges**: All neighbors (nodes 1-N) are connected to the target (node 0)
- **Edge Attributes**: Each edge encodes spatial relationships: `[dx, dy, euclidean_distance, network_distance]` (network_distance is set to Euclidean during feature engineering, recalculated in loss function)

**Key Properties**:
- Variable graph sizes: Different target points have different numbers of neighbors
- No padding needed: Graph structure naturally handles variable sizes
- Explicit spatial encoding: Edge attributes provide direct spatial information
- Permutation invariant: Order of neighbors doesn't matter (graph structure)

### 2. Model Architecture

```
Input: Star Graph
├── Node Features: [N, 33] where N = 1 + num_neighbors
│   ├── Node 0 (target): 33 features
│   └── Nodes 1-N (neighbors): 33 features each
├── Edge Index: [2, E] where E = num_neighbors
│   └── Edges: (neighbor_i → target) for all neighbors
└── Edge Attributes: [E, 4]
    └── [dx, dy, euclidean_distance, network_distance] for each edge

Model Layers:
├── Node Encoder: Linear(33 → 128)
├── Edge Encoder: Linear(4 → 128)
├── TransformerConv Layers (3 layers):
│   ├── Multi-head attention with edge attributes
│   ├── Residual connections
│   └── Layer normalization
└── Classification Head: Linear(128 → 8)
    └── Output: Probability distribution over 8 service categories
```

### 3. TransformerConv Layers

The model uses PyTorch Geometric's `TransformerConv` layers, which implement graph attention with edge attributes:

**Key Features**:
- **Multi-head attention**: Allows the model to attend to different aspects of neighbor relationships
- **Edge attribute integration**: Edge attributes are encoded and used in the attention mechanism
- **Residual connections**: Help with gradient flow and training stability
- **Layer normalization**: Normalizes activations for stable training

**Attention Mechanism**:
- The target node (node 0) attends to all neighbor nodes (nodes 1-N)
- Attention weights depend on:
  - Node features (both target and neighbor)
  - Edge attributes (spatial relationships)
- The model learns which neighbors are most important for prediction

### 4. Edge Attribute Encoding

Edge attributes provide explicit spatial encoding:

- **dx, dy**: Relative coordinates (neighbor position relative to target)
- **euclidean_distance**: Straight-line distance between target and neighbor
- **network_distance**: During feature engineering, set to Euclidean distance (placeholder). Actual network/walking distance is calculated in the loss function when constructing target probability vectors.

**Why Edge Attributes Matter**:
- Direct spatial information: Model doesn't need to infer spatial relationships from node features alone
- Euclidean distance: Used for efficient neighbor filtering during feature engineering (50-100x faster than network distance)
- Network distance: Calculated in loss function for accurate target probability vectors
- Relative coordinates: Help model understand directional relationships

### 5. Target Node Extraction

After graph attention layers, only the target node (node 0) is used for classification:

- For single graphs: Extract node 0 embedding
- For batched graphs: Extract node 0 of each graph in the batch
- Classification head operates on target node embedding only

**Rationale**: The target node has aggregated information from all neighbors through attention, so it contains the full spatial context needed for prediction.

## Training Procedure

### 1. Data Preparation

1. **Feature Engineering**:
   - Generate target points (regular grid within neighborhoods)
   - Generate neighbor grid cells around each target
   - Filter neighbors by Euclidean distance (for computational efficiency)
   - Compute 33 features for target and each neighbor
   - Calculate edge attributes (dx, dy, euclidean_distance, network_distance set to Euclidean)
   - Note: Network/walking distances are calculated only in the loss function phase

2. **Graph Construction**:
   - Build star graph for each target point
   - Store as PyTorch Geometric `Data` objects
   - Save to Parquet files for efficient loading

### 2. Training Loop

1. **Data Loading**:
   - Load pre-computed graphs from Parquet files
   - Use PyTorch Geometric `DataLoader` for efficient batching
   - Batch variable-sized graphs automatically

2. **Forward Pass**:
   - Encode node features: `[N, 33] → [N, 128]`
   - Encode edge attributes: `[E, 4] → [E, 128]`
   - Apply TransformerConv layers with attention
   - Extract target node embedding
   - Classify: `[batch_size, 128] → [batch_size, 8]`

3. **Loss Calculation**:
   - Compute distance-based target probability vector
   - Calculate KL divergence between predicted and target probabilities
   - Backpropagate and update model weights

### 3. Memory Considerations

**Graph Sizes**:
- Average: ~500 neighbors per target point
- Range: 0-1000+ neighbors depending on location
- Memory per graph: ~(1 + num_neighbors) × 33 features

**Batch Size**:
- Default: 16 graphs per batch
- Can be increased with gradient accumulation
- Mixed precision (FP16) can reduce memory usage

**Optimization**:
- Pre-compute features and network distances
- Store graphs efficiently in Parquet format
- Use PyTorch Geometric's efficient batching

## Advantages Over Sequence-Based Approach

### 1. Natural Handling of Variable Sizes

- **Graphs**: No padding needed, variable sizes handled naturally
- **Sequences**: Require padding/masking for variable lengths

### 2. Explicit Spatial Encoding

- **Graphs**: Edge attributes directly encode spatial relationships
- **Sequences**: Spatial relationships must be inferred from position encoding

### 3. Permutation Invariance

- **Graphs**: Naturally permutation invariant (order doesn't matter)
- **Sequences**: Order matters, requires careful position encoding

### 4. Memory Efficiency

- **Graphs**: Only store actual neighbors (no padding)
- **Sequences**: Must pad to maximum length, wasting memory

### 5. Domain Alignment

- **Graphs**: Spatial relationships are naturally represented as edges
- **Sequences**: Spatial relationships are less natural in sequence format

## Hyperparameters

### Model Architecture

- `num_features`: 33 (number of input features per node)
- `hidden_dim`: 128 (dimension of node/edge embeddings)
- `num_layers`: 3 (number of TransformerConv layers)
- `num_heads`: 4 (number of attention heads)
- `dropout`: 0.1 (dropout rate)
- `edge_dim`: 4 (number of edge attributes)
- `num_classes`: 8 (number of service categories)

### Training

- `batch_size`: 16 (graphs per batch)
- `learning_rate`: 0.001
- `weight_decay`: 0.0001
- `num_epochs`: 100
- `early_stopping_patience`: 10

### Feature Engineering

- `walk_15min_radius_meters`: 1200 (network walking distance threshold)
- `grid_cell_size_meters`: 100 (size of grid cells)
- `target_point_sampling_interval_meters`: 100 (spacing of target points)

## Implementation Details

### PyTorch Geometric Data Objects

```python
from torch_geometric.data import Data

data = Data(
    x=node_features,           # [N, 33] node features
    edge_index=edge_index,     # [2, E] edge connections
    edge_attr=edge_attributes, # [E, 4] edge attributes
    y=target_prob_vector,      # [8] target probability vector
    num_nodes=N                # Number of nodes
)
```

### Batching

PyTorch Geometric's `DataLoader` automatically batches graphs:

- Creates a "super-graph" containing all graphs in the batch
- Adds `batch` tensor to track which node belongs to which graph
- Efficiently handles variable-sized graphs

### Model Forward Pass

```python
def forward(self, data: Data) -> torch.Tensor:
    # Encode nodes and edges
    x = self.node_encoder(data.x)           # [N, 128]
    edge_emb = self.edge_encoder(data.edge_attr)  # [E, 128]
    
    # Graph attention layers
    for conv, norm in zip(self.conv_layers, self.layer_norms):
        x_new = conv(x, data.edge_index, edge_emb)
        x_new = self.head_projection(x_new)
        x_new = norm(x_new)
        x = x + x_new  # Residual connection
    
    # Extract target node (node 0 of each graph)
    target_emb = extract_target_nodes(x, data.batch)
    
    # Classify
    logits = self.classifier(target_emb)  # [batch_size, 8]
    return logits
```

## References

- **PyTorch Geometric Documentation**: https://pytorch-geometric.readthedocs.io/
- **TransformerConv**: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv
- **Data Objects**: https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
- **DataLoader**: https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.DataLoader

---

*Document Version: 1.0*  
*Last Updated: January 2025*  
*Project: AI4SI - 15-Minute City Service Gap Prediction Model*
