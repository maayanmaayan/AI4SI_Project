"""Spatial Graph Transformer model architecture for 15-minute city service gap prediction.

This module provides the SpatialGraphTransformer class, a graph neural network that uses
PyTorch Geometric's TransformerConv layers to process star graphs representing spatial
context around target points. The model learns to predict service distribution patterns
based on spatial features and network accessibility.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SpatialGraphTransformer(nn.Module):
    """Spatial Graph Transformer for service gap prediction.

    This model processes star graphs where a target point (node 0) is connected to
    all neighbor grid cells (nodes 1-N) within network walking distance. The model
    uses TransformerConv layers with edge attributes to encode spatial relationships
    and predict probability distributions over 8 service categories.

    Architecture:
    - Node encoder: 33 features → hidden_dim
    - Edge encoder: 4 edge attributes → hidden_dim
    - TransformerConv layers: Multi-head attention with edge attributes
    - Classification head: Single linear layer → 8 service categories

    Args:
        num_features: Number of input features per node (default: 33).
        hidden_dim: Hidden dimension size (default: 128).
        num_layers: Number of TransformerConv layers (default: 3).
        num_heads: Number of attention heads (default: 4).
        dropout: Dropout probability (default: 0.1).
        num_classes: Number of output classes (default: 8).
        edge_dim: Number of edge attributes (default: 4).

    Example:
        >>> from torch_geometric.data import Data
        >>> model = SpatialGraphTransformer()
        >>> data = Data(x=torch.randn(10, 33), edge_index=torch.randint(0, 10, (2, 20)),
        ...             edge_attr=torch.randn(20, 4))
        >>> logits = model(data)
        >>> print(logits.shape)  # [1, 8]
    """

    def __init__(
        self,
        num_features: int = 33,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 8,
        edge_dim: int = 4,
        temperature: float = 2.0,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_classes = num_classes
        self.edge_dim = edge_dim
        self.temperature = temperature

        # Node encoder: 33 features → hidden_dim
        self.node_encoder = nn.Linear(num_features, hidden_dim)

        # Edge encoder: 4 edge attributes → hidden_dim
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # Graph transformer layers with multi-head attention
        self.conv_layers = nn.ModuleList([
            TransformerConv(
                hidden_dim,
                hidden_dim,
                edge_dim=hidden_dim,  # Use edge embeddings
                heads=num_heads,
                dropout=dropout,
                concat=True,  # Concatenate multi-head outputs
            )
            for _ in range(num_layers)
        ])

        # Shared head projection (selected for efficiency - ~200K fewer params)
        # Projects concatenated multi-head outputs back to hidden_dim
        self.head_projection = nn.Linear(hidden_dim * num_heads, hidden_dim)

        # Layer normalization for each transformer layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Classification head: Single linear layer (selected for simplicity and efficiency)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # ReLU activation for transformer layers (simpler, more linear model for small datasets)
        # Using torch.nn.functional.relu for efficient activation between layers
        # Note: ReLU activation is applied in forward() method between transformer layers

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier/Glorot uniform initialization.

        This ensures reproducible training and better convergence.
        The classifier is initialized to output near-uniform logits to prevent
        early mode collapse.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if module == self.classifier:
                        # Initialize classifier bias to output near-uniform logits
                        # For uniform distribution: log(1/num_classes) ≈ -log(num_classes)
                        # This helps prevent early mode collapse
                        nn.init.constant_(module.bias, -np.log(self.num_classes))
                    else:
                        nn.init.zeros_(module.bias)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [N, 33] where N = 1 + num_neighbors
                - edge_index: Edge connections [2, E] (neighbors → target)
                - edge_attr: Edge attributes [E, 4] [dx, dy, euclidean_dist, network_dist]
                - batch: Optional batch assignment [N] for batched graphs

        Returns:
            Logits tensor of shape [batch_size, 8] or [1, 8] for single graph.
        """
        # Encode nodes and edges
        x = self.node_encoder(data.x)  # [N, hidden_dim]
        edge_emb = self.edge_encoder(data.edge_attr)  # [E, hidden_dim]

        # Graph transformer layers with ReLU activation between layers
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            # Multi-head attention with edge attributes
            x_new = conv(x, data.edge_index, edge_emb)  # [N, hidden_dim * num_heads]
            x_new = self.head_projection(x_new)  # [N, hidden_dim]
            x_new = norm(x_new)
            x_new = self.dropout_layer(x_new)
            x = x + x_new  # Residual connection

            # Apply ReLU activation between layers (except after final layer)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)  # ReLU for simpler, more linear model

        # Extract target node embeddings (node 0 in each graph)
        target_emb = self._extract_target_nodes(x, data.batch)

        # Classify
        logits = self.classifier(target_emb)  # [batch_size, num_classes]
        
        # Apply temperature scaling to prevent overconfidence
        # Dividing logits by temperature flattens the probability distribution
        # P_i = exp(z_i / T) / sum(exp(z_j / T))
        logits = logits / self.temperature

        return logits

    def _extract_target_nodes(
        self, x: torch.Tensor, batch: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Extract target node embeddings (node 0) from each graph.

        For batched graphs, extracts the first node (index 0) of each graph.
        For single graphs, extracts node 0.

        Args:
            x: Node embeddings [N, hidden_dim] where N is total nodes across all graphs.
            batch: Batch assignment tensor [N] indicating which graph each node belongs to.
                If None, assumes single graph.

        Returns:
            Target node embeddings [batch_size, hidden_dim] or [1, hidden_dim] for single graph.
        """
        if batch is not None and batch.numel() > 0:
            # Batched graphs: extract node 0 of each graph using efficient indexing
            batch_size = batch.max().item() + 1
            target_indices = []
            for i in range(batch_size):
                # Find first node (index 0) of graph i
                graph_nodes = (batch == i).nonzero(as_tuple=True)[0]
                if len(graph_nodes) > 0:
                    target_indices.append(graph_nodes[0].item())
                else:
                    # Edge case: empty graph (shouldn't happen, but handle gracefully)
                    logger.warning(f"Empty graph {i} in batch, using first available node")
                    target_indices.append(0)

            target_indices = torch.tensor(
                target_indices, device=x.device, dtype=torch.long
            )
            target_emb = x[target_indices]  # [batch_size, hidden_dim]
        else:
            # Single graph: node 0 is target
            target_emb = x[0:1]  # [1, hidden_dim]

        return target_emb


class TinySpatialGraphTransformer(nn.Module):
    """Tiny Spatial Graph Transformer for debugging model collapse issues.

    This is a drastically reduced-capacity version of SpatialGraphTransformer designed
    for debugging. It uses the same architecture but with minimal capacity (<10,000 parameters
    vs ~1M in the original) and higher dropout (0.4 vs 0.1) to test if the architecture can
    learn without collapsing to single-class predictions.

    Architecture:
    - Node encoder: 33 features → 16
    - Edge encoder: 4 edge attributes → 16
    - Single TransformerConv layer: Multi-head attention with edge attributes
    - Classification head: Single linear layer → 8 service categories

    Args:
        num_features: Number of input features per node (default: 33).
        hidden_dim: Hidden dimension size (default: 16).
        num_layers: Number of TransformerConv layers (default: 1).
        num_heads: Number of attention heads (default: 2).
        dropout: Dropout probability (default: 0.4).
        num_classes: Number of output classes (default: 8).
        edge_dim: Number of edge attributes (default: 4).

    Example:
        >>> from torch_geometric.data import Data
        >>> model = TinySpatialGraphTransformer()
        >>> data = Data(x=torch.randn(10, 33), edge_index=torch.randint(0, 10, (2, 20)),
        ...             edge_attr=torch.randn(20, 4))
        >>> logits = model(data)
        >>> print(logits.shape)  # [1, 8]
    """

    def __init__(
        self,
        num_features: int = 33,
        hidden_dim: int = 16,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.4,
        num_classes: int = 8,
        edge_dim: int = 4,
        temperature: float = 2.0,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_classes = num_classes
        self.edge_dim = edge_dim
        self.temperature = temperature

        # Node encoder: 33 features → hidden_dim
        self.node_encoder = nn.Linear(num_features, hidden_dim)

        # Edge encoder: 4 edge attributes → hidden_dim
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # Graph transformer layers with multi-head attention
        self.conv_layers = nn.ModuleList([
            TransformerConv(
                hidden_dim,
                hidden_dim,
                edge_dim=hidden_dim,  # Use edge embeddings
                heads=num_heads,
                dropout=dropout,
                concat=True,  # Concatenate multi-head outputs
            )
            for _ in range(num_layers)
        ])

        # Shared head projection (projects concatenated multi-head outputs back to hidden_dim)
        self.head_projection = nn.Linear(hidden_dim * num_heads, hidden_dim)

        # Layer normalization for each transformer layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Classification head: Single linear layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier/Glorot uniform initialization.

        This ensures reproducible training and better convergence.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [N, 33] where N = 1 + num_neighbors
                - edge_index: Edge connections [2, E] (neighbors → target)
                - edge_attr: Edge attributes [E, 4] [dx, dy, euclidean_dist, network_dist]
                - batch: Optional batch assignment [N] for batched graphs

        Returns:
            Logits tensor of shape [batch_size, 8] or [1, 8] for single graph.
        """
        # Encode nodes and edges
        x = self.node_encoder(data.x)  # [N, hidden_dim]
        edge_emb = self.edge_encoder(data.edge_attr)  # [E, hidden_dim]

        # Graph transformer layers with ReLU activation between layers
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            # Multi-head attention with edge attributes
            x_new = conv(x, data.edge_index, edge_emb)  # [N, hidden_dim * num_heads]
            x_new = self.head_projection(x_new)  # [N, hidden_dim]
            x_new = norm(x_new)
            x_new = self.dropout_layer(x_new)
            x = x + x_new  # Residual connection

            # Apply ReLU activation between layers (except after final layer)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)  # ReLU for simpler, more linear model

        # Extract target node embeddings (node 0 in each graph)
        target_emb = self._extract_target_nodes(x, data.batch)

        # Classify
        logits = self.classifier(target_emb)  # [batch_size, num_classes]
        
        # Apply temperature scaling to prevent overconfidence
        # Dividing logits by temperature flattens the probability distribution
        # P_i = exp(z_i / T) / sum(exp(z_j / T))
        logits = logits / self.temperature

        return logits

    def _extract_target_nodes(
        self, x: torch.Tensor, batch: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Extract target node embeddings (node 0) from each graph.

        For batched graphs, extracts the first node (index 0) of each graph.
        For single graphs, extracts node 0.

        Args:
            x: Node embeddings [N, hidden_dim] where N is total nodes across all graphs.
            batch: Batch assignment tensor [N] indicating which graph each node belongs to.
                If None, assumes single graph.

        Returns:
            Target node embeddings [batch_size, hidden_dim] or [1, hidden_dim] for single graph.
        """
        if batch is not None and batch.numel() > 0:
            # Batched graphs: extract node 0 of each graph using efficient indexing
            batch_size = batch.max().item() + 1
            target_indices = []
            for i in range(batch_size):
                # Find first node (index 0) of graph i
                graph_nodes = (batch == i).nonzero(as_tuple=True)[0]
                if len(graph_nodes) > 0:
                    target_indices.append(graph_nodes[0].item())
                else:
                    # Edge case: empty graph (shouldn't happen, but handle gracefully)
                    logger.warning(f"Empty graph {i} in batch, using first available node")
                    target_indices.append(0)

            target_indices = torch.tensor(
                target_indices, device=x.device, dtype=torch.long
            )
            target_emb = x[target_indices]  # [batch_size, hidden_dim]
        else:
            # Single graph: node 0 is target
            target_emb = x[0:1]  # [1, hidden_dim]

        return target_emb


def create_model_from_config(config: Optional[dict] = None) -> SpatialGraphTransformer:
    """Create SpatialGraphTransformer model from configuration.

    Args:
        config: Configuration dictionary. If None, loads from get_config().

    Returns:
        Initialized SpatialGraphTransformer model.

    Example:
        >>> model = create_model_from_config()
        >>> print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    """
    if config is None:
        config = get_config()

    model_config = config.get("model", {})

    model = SpatialGraphTransformer(
        num_features=model_config.get("num_features", 33),
        hidden_dim=model_config.get("hidden_dim", 128),
        num_layers=model_config.get("num_layers", 3),
        num_heads=model_config.get("num_heads", 4),
        dropout=model_config.get("dropout", 0.1),
        num_classes=model_config.get("num_classes", 8),
        edge_dim=model_config.get("edge_dim", 4),
        temperature=model_config.get("temperature", 2.0),
    )

    logger.info(
        f"Created SpatialGraphTransformer: "
        f"{sum(p.numel() for p in model.parameters()):,} parameters"
    )

    return model


def create_tiny_model_from_config(config: Optional[dict] = None) -> TinySpatialGraphTransformer:
    """Create TinySpatialGraphTransformer model for debugging.

    Creates a tiny model with hardcoded reduced parameters (hidden_dim=16, num_layers=1,
    num_heads=2, dropout=0.4) regardless of config. This ensures it's always tiny for
    debugging purposes.

    Args:
        config: Configuration dictionary. If None, loads from get_config(). Note: config
            is loaded but tiny model uses hardcoded parameters for consistency.

    Returns:
        Initialized TinySpatialGraphTransformer model with <10,000 parameters.

    Example:
        >>> model = create_tiny_model_from_config()
        >>> print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    """
    if config is None:
        config = get_config()

    # Tiny model uses hardcoded parameters (ignores config for consistency)
    model = TinySpatialGraphTransformer(
        num_features=33,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0.4,
        num_classes=8,
        edge_dim=4,
        temperature=2.0,
    )

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Created TinySpatialGraphTransformer: {param_count:,} parameters"
    )

    return model
