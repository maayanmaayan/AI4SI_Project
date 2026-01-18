"""Unit tests for Spatial Graph Transformer model."""

import pytest
import torch
from torch_geometric.data import Data

from src.training.model import SpatialGraphTransformer, create_model_from_config


def test_model_initialization():
    """Test model initialization with default parameters."""
    model = SpatialGraphTransformer()
    assert model.num_features == 33
    assert model.hidden_dim == 128
    assert model.num_layers == 3
    assert model.num_heads == 4
    assert model.num_classes == 8
    assert model.edge_dim == 4


def test_model_initialization_custom():
    """Test model initialization with custom parameters."""
    model = SpatialGraphTransformer(
        num_features=33,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.2,
        num_classes=8,
        edge_dim=4,
    )
    assert model.hidden_dim == 64
    assert model.num_layers == 2
    assert model.num_heads == 2


def test_forward_single_graph():
    """Test forward pass with single graph."""
    model = SpatialGraphTransformer(num_features=33, hidden_dim=64, num_layers=2)

    # Create single graph: 1 target + 5 neighbors = 6 nodes
    num_nodes = 6
    num_edges = 5  # 5 neighbors connected to target (node 0)

    x = torch.randn(num_nodes, 33)
    edge_index = torch.tensor([
        [1, 2, 3, 4, 5],  # Source: neighbors
        [0, 0, 0, 0, 0],  # Target: center node
    ], dtype=torch.long)
    edge_attr = torch.randn(num_edges, 4)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Forward pass
    logits = model(data)

    # Check output shape: [1, 8] (single graph, 8 classes)
    assert logits.shape == (1, 8)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_forward_batched_graphs():
    """Test forward pass with batched graphs."""
    model = SpatialGraphTransformer(num_features=33, hidden_dim=64, num_layers=2)

    # Create batched graphs using PyTorch Geometric's batching
    from torch_geometric.loader import DataLoader

    graphs = []
    for i in range(3):
        num_nodes = 4 + i  # Variable number of nodes per graph
        num_edges = num_nodes - 1  # All neighbors connect to target

        x = torch.randn(num_nodes, 33)
        edge_index = torch.tensor([
            list(range(1, num_nodes)),  # Source: neighbors
            [0] * (num_nodes - 1),  # Target: center node
        ], dtype=torch.long)
        edge_attr = torch.randn(num_edges, 4)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(data)

    # Create DataLoader to batch graphs
    loader = DataLoader(graphs, batch_size=3)

    # Forward pass on batched data
    for batch in loader:
        logits = model(batch)

        # Check output shape: [batch_size, 8]
        assert logits.shape == (3, 8)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        break


def test_output_shape():
    """Test output tensor shapes for different graph sizes."""
    model = SpatialGraphTransformer(num_features=33, hidden_dim=64, num_layers=2)

    # Test with different numbers of neighbors
    for num_neighbors in [0, 1, 5, 10, 50]:
        num_nodes = 1 + num_neighbors
        num_edges = num_neighbors

        x = torch.randn(num_nodes, 33)
        if num_edges > 0:
            edge_index = torch.tensor([
                list(range(1, num_nodes)),
                [0] * num_edges,
            ], dtype=torch.long)
            edge_attr = torch.randn(num_edges, 4)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        logits = model(data)

        # Output should always be [1, 8] for single graph
        assert logits.shape == (1, 8)


def test_edge_attributes():
    """Test that edge attributes are used in forward pass."""
    model = SpatialGraphTransformer(num_features=33, hidden_dim=64, num_layers=2)

    # Create graph with specific edge attributes
    x = torch.randn(5, 33)
    edge_index = torch.tensor([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
    ], dtype=torch.long)
    edge_attr = torch.ones(4, 4) * 10.0  # Distinct edge attributes

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Forward pass should use edge attributes
    logits1 = model(data)

    # Change edge attributes
    edge_attr2 = torch.ones(4, 4) * 20.0
    data2 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr2)

    logits2 = model(data2)

    # Outputs should be different (edge attributes affect attention)
    assert not torch.allclose(logits1, logits2, atol=1e-6)


def test_gradient_flow():
    """Test that gradients flow correctly through all layers."""
    model = SpatialGraphTransformer(num_features=33, hidden_dim=64, num_layers=2)

    x = torch.randn(5, 33, requires_grad=True)
    edge_index = torch.tensor([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
    ], dtype=torch.long)
    edge_attr = torch.randn(4, 4, requires_grad=True)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    logits = model(data)
    loss = logits.sum()
    loss.backward()

    # Check gradients exist
    assert x.grad is not None
    assert edge_attr.grad is not None

    # Check model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_create_model_from_config():
    """Test model creation from configuration."""
    model = create_model_from_config()
    assert isinstance(model, SpatialGraphTransformer)
    assert model.num_features == 33
    assert model.num_classes == 8


def test_empty_neighbors():
    """Test handling of graphs with zero neighbors."""
    model = SpatialGraphTransformer(num_features=33, hidden_dim=64, num_layers=2)

    # Graph with only target node (no neighbors)
    x = torch.randn(1, 33)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 4), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Should still produce valid output
    logits = model(data)
    assert logits.shape == (1, 8)
    assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
