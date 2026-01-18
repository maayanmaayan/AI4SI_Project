"""Quick test script to verify augmentation transforms work correctly."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch_geometric.data import Data

from src.training.data_augmentation import (
    FeatureNoiseTransform,
    NeighborSubsamplingTransform,
    EdgeAttributeNoiseTransform,
    create_augmentation_transform,
)
from src.training.dataset import SpatialGraphDataset

# Create sample data
def create_sample_data(num_neighbors: int = 10):
    """Create sample Data object for testing."""
    # Create node features [target + neighbors, 33]
    x = torch.randn(1 + num_neighbors, 33).float()
    
    # Ensure features are in valid ranges (some ratios, some counts)
    # Demographics (0-16): some ratios, densities
    x[:, 0] = torch.abs(x[:, 0])  # population_density ≥ 0
    x[:, 1] = torch.clamp(x[:, 1], -2, 2)  # ses_index
    x[:, 2:17] = torch.clamp(torch.abs(x[:, 2:17]), 0, 1)  # ratios [0, 1]
    
    # Built form (17-20)
    x[:, 17:21] = torch.abs(x[:, 17:21])  # densities, counts ≥ 0
    
    # Services (21-28)
    x[:, 21:29] = torch.abs(x[:, 21:29])  # counts ≥ 0
    
    # Walkability (29-32)
    x[:, 29] = torch.abs(x[:, 29])  # intersection_density ≥ 0
    x[:, 30] = torch.clamp(torch.abs(x[:, 30]), 0, 500)  # average_block_length
    x[:, 31:33] = torch.clamp(torch.abs(x[:, 31:33]), 0, 1)  # ratios [0, 1]
    
    # Create edge index: star graph (all neighbors → target)
    edge_index = torch.tensor(
        [
            list(range(1, num_neighbors + 1)),  # Sources: neighbors
            [0] * num_neighbors,  # Target: center (always 0)
        ],
        dtype=torch.long,
    )
    
    # Create edge attributes [dx, dy, euclidean_dist, network_dist] normalized to [0, 1]
    # Use varied distances for spatial stratification test
    distances = np.linspace(0.1, 0.9, num_neighbors)  # Spread across bands
    edge_attr = torch.tensor([
        [
            np.random.uniform(-0.5, 0.5),  # dx
            np.random.uniform(-0.5, 0.5),  # dy
            dist,  # euclidean_distance (normalized)
            dist * 1.2,  # network_distance (normalized, >= euclidean)
        ]
        for dist in distances
    ], dtype=torch.float32)
    edge_attr[:, 3] = torch.clamp(edge_attr[:, 3], min=edge_attr[:, 2])  # Ensure network >= euclidean
    
    # Create target probability vector
    y = torch.rand(1, 8).float()
    y = y / y.sum()  # Normalize to probabilities
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        target_id="test_target",
        neighborhood_name="test_neighborhood",
        num_nodes=1 + num_neighbors,
    )


def test_feature_noise_bounds():
    """Test that FeatureNoiseTransform respects feature bounds."""
    print("Testing FeatureNoiseTransform bounds...")
    
    transform = FeatureNoiseTransform(noise_std=0.05)
    data = create_sample_data(num_neighbors=5)
    
    data_aug = transform(data)
    
    # Check bounds (SES index can be negative, so exclude it from non-negative check)
    assert (data_aug.x[:, 0] >= 0).all(), "Population density should be ≥ 0"
    assert (data_aug.x[:, 1] >= -3.01).all() and (data_aug.x[:, 1] <= 3.01).all(), "SES index should be in [-3, 3]"
    assert (data_aug.x[:, 2:17] >= -0.01).all() and (data_aug.x[:, 2:17] <= 1.01).all(), "Ratios should be in [0, 1]"
    assert (data_aug.x[:, 17:21] >= -0.01).all(), "Built form features should be ≥ 0"
    assert (data_aug.x[:, 21:29] >= -0.01).all(), "Service counts should be ≥ 0"
    assert (data_aug.x[:, 29] >= -0.01).all(), "Intersection density should be ≥ 0"
    assert (data_aug.x[:, 31:33] >= -0.01).all() and (data_aug.x[:, 31:33] <= 1.01).all(), "Walkability ratios should be in [0, 1]"
    
    print("  ✓ Feature bounds respected")


def test_spatial_subsampling():
    """Test that NeighborSubsamplingTransform maintains spatial diversity."""
    print("Testing NeighborSubsamplingTransform spatial stratification...")
    
    transform = NeighborSubsamplingTransform(subsample_ratio=0.6, min_neighbors=3)
    data = create_sample_data(num_neighbors=12)
    
    # Check original has neighbors across distance bands
    distances = data.edge_attr[:, 2].cpu().numpy()
    print(f"  Original distances: min={distances.min():.3f}, max={distances.max():.3f}")
    
    data_aug = transform(data)
    
    # Check subsampled data
    assert data_aug.x.shape[0] >= 1 + 3, "Should have at least min_neighbors"
    assert data_aug.edge_attr.shape[0] == data_aug.x.shape[0] - 1, "Edge count should match neighbor count"
    
    # Check that distances span multiple bands (not all in one band)
    aug_distances = data_aug.edge_attr[:, 2].cpu().numpy()
    print(f"  Subsampled distances: min={aug_distances.min():.3f}, max={aug_distances.max():.3f}")
    
    # Check that we have neighbors from different distance bands
    bands = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.0)]
    bands_present = sum(1 for low, high in bands if ((aug_distances >= low) & (aug_distances < high)).any())
    
    if len(aug_distances) >= 4:  # If we have enough neighbors
        assert bands_present >= 2, f"Should span multiple distance bands, got {bands_present}"
    
    print(f"  ✓ Spatial diversity maintained ({bands_present} bands represented)")


def test_edge_noise_constraints():
    """Test that EdgeAttributeNoiseTransform preserves spatial relationships."""
    print("Testing EdgeAttributeNoiseTransform spatial constraints...")
    
    transform = EdgeAttributeNoiseTransform(distance_noise_std=0.05, position_noise_std=0.05)
    data = create_sample_data(num_neighbors=8)
    
    # Store original constraints
    original_network = data.edge_attr[:, 3].clone()
    original_euclidean = data.edge_attr[:, 2].clone()
    
    data_aug = transform(data)
    
    # Check constraints
    assert (data_aug.edge_attr >= -0.01).all() and (data_aug.edge_attr <= 1.01).all(), \
        "Edge attributes should be in [0, 1] range"
    
    # Check network_distance >= euclidean_distance (physical constraint)
    network_dist = data_aug.edge_attr[:, 3]
    euclidean_dist = data_aug.edge_attr[:, 2]
    assert (network_dist >= euclidean_dist - 1e-6).all(), \
        "Network distance should be >= euclidean distance"
    
    print("  ✓ Spatial relationships preserved (network ≥ euclidean)")
    print(f"  ✓ Edge attributes in valid range [0, 1]")


def test_composite_augmentation():
    """Test that composite augmentation works correctly."""
    print("Testing create_augmentation_transform...")
    
    transform = create_augmentation_transform(
        enable_feature_noise=True,
        enable_neighbor_subsampling=True,
        enable_edge_noise=True,
    )
    
    assert transform is not None, "Should return a CompositeAugmentation"
    assert len(transform.transforms) >= 3, "Should have at least 3 transforms"
    
    data = create_sample_data(num_neighbors=10)
    data_aug = transform(data)
    
    assert data_aug.x is not None, "Augmented data should have features"
    assert data_aug.edge_attr is not None, "Augmented data should have edge attributes"
    
    print("  ✓ Composite augmentation works correctly")


def main():
    """Run all augmentation tests."""
    print("=" * 80)
    print("Augmentation Transform Tests")
    print("=" * 80)
    print()
    
    try:
        test_feature_noise_bounds()
        test_spatial_subsampling()
        test_edge_noise_constraints()
        test_composite_augmentation()
        
        print()
        print("=" * 80)
        print("✓ All augmentation tests passed!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
