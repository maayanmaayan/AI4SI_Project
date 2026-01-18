# Spatial Graph Transformer - Architecture Improvements & Alternatives

## Key Design Decisions to Consider

### 1. Classification Head Structure

**Option A: Single Linear Layer**
```python
self.classifier = nn.Linear(hidden_dim, num_classes)
```
- Pros: Simple, fewer parameters, faster
- Cons: Limited capacity for complex decision boundaries

**Option B: 2-Layer MLP (✅ SELECTED)**
```python
self.classifier = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, num_classes)
)
```
- Pros: More capacity, better for complex patterns, regularization via dropout
- Cons: Slightly more parameters (~1K additional params)
- **Decision**: ✅ **SELECTED** - Use 2-layer MLP for better performance and capacity

### 2. Head Projection Strategy

**Option A: Per-Layer Projections**
```python
self.head_projections = nn.ModuleList([
    nn.Linear(hidden_dim * num_heads, hidden_dim) for _ in range(num_layers)
])
```
- Pros: Each layer can learn different projections, more flexible
- Cons: More parameters (3 × 512 × 128 = ~200K params)

**Option B: Shared Projection (Recommended)**
```python
self.head_projection = nn.Linear(hidden_dim * num_heads, hidden_dim)
```
- Pros: Fewer parameters, sufficient for most cases
- Cons: Less flexibility across layers
- **Recommendation**: Use Option B initially (can always upgrade later)

### 3. Activation Functions

**Current**: No explicit activations between layers (ReLU in final classifier only)

**Recommended**: Add GELU between transformer layers (✅ SELECTED)
```python
if i < len(self.conv_layers) - 1:
    x = F.gelu(x)  # GELU for transformer layers (better for transformers)
```

**Activation Choice**:
- **ReLU**: Standard, fast, works well
- **GELU**: Often better for transformers, smoother gradients ✅ **SELECTED**
- **Decision**: Use GELU activation between transformer layers for better transformer performance

### 4. Target Node Extraction Efficiency

**Current Approach** (loop-based):
```python
for i in range(batch_size):
    graph_nodes = (batch == i).nonzero(as_tuple=True)[0]
    target_indices.append(graph_nodes[0])
```

**Improved Approach** (using PyG scatter or indexing):
```python
# More efficient: build mask once
batch_size = batch.max().item() + 1
target_indices = []
for i in range(batch_size):
    target_indices.append((batch == i).nonzero()[0][0])
target_indices = torch.tensor(target_indices, device=x.device)
return x[target_indices]
```

**Recommendation**: Use improved approach for better performance

### 5. Edge Case: Zero Neighbors

**Current**: Relies on PyG handling empty edge_index

**Explicit Handling** (Optional but recommended):
```python
# Check for graphs with no neighbors
if data.edge_index.size(1) == 0:
    logger.debug("Graph with no neighbors - using target node features only")
    # Still passes through layers (target node self-attention equivalent)
```

**Recommendation**: Add explicit check for better debugging/logging

### 6. Weight Initialization

**Current**: PyTorch default (Kaiming uniform for Linear)

**Explicit Initialization** (Recommended for reproducibility):
```python
def _init_weights(self):
    """Initialize weights using Xavier/Glorot uniform."""
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

**Recommendation**: Add explicit initialization for better reproducibility

### 7. Configuration-Driven Design

**Make activation configurable**:
```python
# In config.yaml
model:
  activation: "gelu"  # or "relu"

# In model
activation_name = config.get("model", {}).get("activation", "gelu")
self.activation = F.gelu if activation_name == "gelu" else F.relu
```

### 8. Optional: Residual Connection Pattern

**Option A: Post-Norm (Current)**
```python
x_new = conv(x, edge_index, edge_emb)
x_new = head_proj(x_new)
x_new = norm(x_new)
x_new = dropout(x_new)
x = x + x_new  # residual after norm
```

**Option B: Pre-Norm (Alternative)**
```python
x_norm = norm(x)
x_new = conv(x_norm, edge_index, edge_emb)
x_new = head_proj(x_new)
x_new = dropout(x_new)
x = x + x_new  # residual before norm
```

**Recommendation**: Start with post-norm (current), pre-norm sometimes trains better but can be tested later

## Recommended Final Architecture

Based on analysis, here's the recommended structure:

```python
class SpatialGraphTransformer(nn.Module):
    def __init__(self, config):
        # ... encoders ...
        
        # Shared head projection (more efficient)
        self.head_projection = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        # 2-layer classifier with dropout (better capacity)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # GELU activation for transformer layers (selected for better transformer performance)
        self.activation = F.gelu
        
        # Initialize weights
        self._init_weights()
    
    def forward(self, data):
        # ... encoder steps ...
        
        # Transformer layers with shared projection
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            x_new = conv(x, data.edge_index, edge_emb)
            x_new = self.head_projection(x_new)  # shared projection
            x_new = norm(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # residual
            
            if i < len(self.conv_layers) - 1:
                x = self.activation(x)  # GELU activation between layers
        
        # Efficient target extraction
        target_emb = self._extract_target_nodes_efficient(x, data.batch)
        
        # Classify with MLP
        logits = self.classifier(target_emb)
        return logits
```

## Selected Architecture Decisions

**Final Choices**:
1. ✅ **2-layer MLP classifier** (Option B) - Selected for better capacity and performance
2. ✅ **Shared head projection** (Option B) - Selected for efficiency and simplicity
3. ✅ **GELU activation between layers** - Selected for better transformer performance
4. ✅ **Efficient target extraction** - Selected for better batch processing performance

## Implementation Priority

**High Priority** (implement in first version):
1. ✅ 2-layer MLP classifier (SELECTED)
2. ✅ Shared head projection (SELECTED)
3. ✅ GELU activation between layers (SELECTED)
4. ✅ Efficient target extraction (SELECTED)

**Medium Priority** (can add later):
5. ⚠️ Configurable activation type
6. ⚠️ Explicit zero-neighbor handling
7. ⚠️ Weight initialization

**Low Priority** (experimental):
8. ⚠️ Pre-norm vs post-norm comparison
9. ⚠️ Per-layer vs shared projection comparison