"""Data augmentation transforms for spatial graph data.

This module provides augmentation transforms that can be applied during training
to increase the effective dataset size and improve generalization. All transforms
preserve the semantic meaning of the spatial data while introducing useful variations.
"""

import random
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureNoiseTransform:
    """Add Gaussian noise to node features.
    
    This augmentation adds small random noise to feature values, which helps
    the model become more robust to measurement noise and variations in feature
    computation.
    
    Args:
        noise_std: Standard deviation of Gaussian noise (as fraction of feature value).
                  Defaults to 0.01 (1% noise).
        apply_to_target: If True, also add noise to target node. Defaults to True.
        apply_to_neighbors: If True, add noise to neighbor nodes. Defaults to True.
    
    Example:
        >>> transform = FeatureNoiseTransform(noise_std=0.02)
        >>> augmented_data = transform(data)
    """
    
    def __init__(
        self,
        noise_std: float = 0.01,
        apply_to_target: bool = True,
        apply_to_neighbors: bool = True,
    ):
        self.noise_std = noise_std
        self.apply_to_target = apply_to_target
        self.apply_to_neighbors = apply_to_neighbors
    
    def __call__(self, data: Data) -> Data:
        """Apply feature noise augmentation with realistic bounds.
        
        Applies Gaussian noise to features while respecting feature-type constraints:
        - Demographics (0-16): Ratios [0,1], densities ≥ 0, SES index ±3
        - Built Form (17-20): Densities ≥ 0, counts ≥ 0, ratios [0,1]
        - Services (21-28): Counts ≥ 0
        - Walkability (29-32): Densities ≥ 0, ratios [0,1], reasonable max lengths
        
        Args:
            data: PyTorch Geometric Data object.
            
        Returns:
            Augmented Data object with bounded noisy features.
        """
        x = data.x.clone()
        
        if self.apply_to_target:
            # Add noise to target node (index 0)
            noise = torch.randn_like(x[0:1]) * self.noise_std
            x[0:1] = x[0:1] + noise
            # Apply bounds to target node
            x[0:1] = self._apply_feature_bounds(x[0:1])
        
        if self.apply_to_neighbors and x.shape[0] > 1:
            # Add noise to neighbor nodes (indices 1+)
            noise = torch.randn_like(x[1:]) * self.noise_std
            x[1:] = x[1:] + noise
            # Apply bounds to neighbor nodes
            x[1:] = self._apply_feature_bounds(x[1:])
        
        # Create new Data object with augmented features
        data_aug = data.clone()
        data_aug.x = x
        return data_aug
    
    def _apply_feature_bounds(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature-type-aware bounds to feature tensor.
        
        Args:
            x: Feature tensor of shape [N, 33] or [1, 33].
            
        Returns:
            Bounded feature tensor with same shape.
        """
        # Demographics (indices 0-16): 17 features
        # Index 0: population_density (≥ 0)
        x[:, 0] = torch.clamp(x[:, 0], min=0.0)
        
        # Index 1: ses_index (clamp extreme outliers to ±3)
        x[:, 1] = torch.clamp(x[:, 1], min=-3.0, max=3.0)
        
        # Indices 2-16: ratios [0, 1] (with small tolerance)
        x[:, 2:17] = torch.clamp(x[:, 2:17], min=0.0, max=1.0 + 1e-6)
        
        # Built Form (indices 17-20): 4 features
        # Index 17: building_density (≥ 0)
        x[:, 17] = torch.clamp(x[:, 17], min=0.0)
        
        # Index 18: building_count (≥ 0)
        x[:, 18] = torch.clamp(x[:, 18], min=0.0)
        
        # Index 19: average_levels (≥ 0, reasonable max ~20)
        x[:, 19] = torch.clamp(x[:, 19], min=0.0, max=20.0)
        
        # Index 20: floor_area_per_capita (≥ 0)
        x[:, 20] = torch.clamp(x[:, 20], min=0.0)
        
        # Services (indices 21-28): 8 features - counts ≥ 0
        x[:, 21:29] = torch.clamp(x[:, 21:29], min=0.0)
        
        # Walkability (indices 29-32): 4 features
        # Index 29: intersection_density (≥ 0)
        x[:, 29] = torch.clamp(x[:, 29], min=0.0)
        
        # Index 30: average_block_length (> 0, reasonable max ~500m normalized to ~0.5)
        # Note: Edge attributes are normalized by 1000m, so 500m = 0.5
        # But features might not be normalized the same way - use reasonable max
        x[:, 30] = torch.clamp(x[:, 30], min=0.0, max=500.0)
        
        # Index 31: pedestrian_street_ratio ([0, 1])
        x[:, 31] = torch.clamp(x[:, 31], min=0.0, max=1.0 + 1e-6)
        
        # Index 32: sidewalk_presence ([0, 1])
        x[:, 32] = torch.clamp(x[:, 32], min=0.0, max=1.0 + 1e-6)
        
        return x


class NeighborSubsamplingTransform:
    """Randomly subsample neighbors to create multiple views of the same graph.
    
    This augmentation randomly keeps a subset of neighbors, creating different
    "views" of the same target point. This helps the model learn to work with
    variable neighborhood sizes and reduces over-reliance on specific neighbors.
    
    Args:
        subsample_ratio: Fraction of neighbors to keep (0.0 to 1.0). 
                        Defaults to 0.8 (keep 80% of neighbors).
        min_neighbors: Minimum number of neighbors to keep. Defaults to 3.
    
    Example:
        >>> transform = NeighborSubsamplingTransform(subsample_ratio=0.75)
        >>> augmented_data = transform(data)
    """
    
    def __init__(
        self,
        subsample_ratio: float = 0.8,
        min_neighbors: int = 3,
    ):
        if not 0.0 < subsample_ratio <= 1.0:
            raise ValueError(f"subsample_ratio must be in (0, 1], got {subsample_ratio}")
        self.subsample_ratio = subsample_ratio
        self.min_neighbors = min_neighbors
    
    def __call__(self, data: Data) -> Data:
        """Apply neighbor subsampling augmentation with spatial stratification.
        
        Uses distance-based stratified sampling to maintain spatial diversity
        across distance bands (0-300m, 300-600m, 600-900m, 900-1200m) instead
        of random sampling. This preserves the spatial distribution of neighbors
        while still creating diverse views of the graph.
        
        Args:
            data: PyTorch Geometric Data object.
            
        Returns:
            Augmented Data object with spatially-stratified subsampled neighbors.
        """
        num_neighbors = data.x.shape[0] - 1  # Exclude target node
        
        # Don't augment if too few neighbors
        if num_neighbors <= self.min_neighbors:
            return data
        
        # Group neighbors by distance bands (using euclidean_distance from edge_attr, index 2)
        # Edge attributes are normalized by 1000m, so: 300m=0.3, 600m=0.6, 900m=0.9
        if data.edge_attr.shape[0] == 0:
            # Fallback to random sampling if no edge attributes
            neighbor_indices = list(range(1, num_neighbors + 1))
            num_keep = max(self.min_neighbors, int(num_neighbors * self.subsample_ratio))
            num_keep = min(num_keep, num_neighbors)
            keep_indices = sorted(random.sample(neighbor_indices, num_keep))
        else:
            # Extract euclidean distances (index 2 in edge_attr)
            euclidean_distances = data.edge_attr[:, 2].cpu().numpy()  # Normalized [0, 1]
            
            # Define distance bands (normalized values: 0-0.3, 0.3-0.6, 0.6-0.9, 0.9-1.0)
            # Note: These correspond to 0-300m, 300-600m, 600-900m, 900-1000m in real distance
            band_thresholds = [0.0, 0.3, 0.6, 0.9, 1.0]
            bands = [[] for _ in range(len(band_thresholds) - 1)]  # 4 bands
            
            # Group neighbor indices by distance band
            # Neighbor at node index i+1 corresponds to edge_attr index i
            for i, distance in enumerate(euclidean_distances):
                neighbor_idx = i + 1  # Neighbor node index (1-based, target is 0)
                for band_idx in range(len(band_thresholds) - 1):
                    if band_thresholds[band_idx] <= distance < band_thresholds[band_idx + 1]:
                        bands[band_idx].append(neighbor_idx)
                        break
            
            # Sample from each band proportionally
            keep_indices = []
            for band_neighbors in bands:
                if not band_neighbors:
                    continue
                # Calculate how many to keep from this band
                num_keep_band = max(1, int(len(band_neighbors) * self.subsample_ratio)) if len(band_neighbors) > 0 else 0
                num_keep_band = min(num_keep_band, len(band_neighbors))
                # Sample from this band
                if num_keep_band > 0:
                    sampled = random.sample(band_neighbors, num_keep_band)
                    keep_indices.extend(sampled)
            
            # Ensure minimum neighbor count
            if len(keep_indices) < self.min_neighbors:
                # Add more neighbors from bands with most neighbors
                remaining_indices = [idx for idx in range(1, num_neighbors + 1) if idx not in keep_indices]
                needed = self.min_neighbors - len(keep_indices)
                if len(remaining_indices) >= needed:
                    keep_indices.extend(random.sample(remaining_indices, needed))
                else:
                    keep_indices.extend(remaining_indices)
            
            keep_indices = sorted(keep_indices)
        
        keep_indices_with_target = [0] + keep_indices  # Always keep target (index 0)
        
        # Subsample nodes
        x_subsampled = data.x[keep_indices_with_target]
        
        # Build new edge_index: neighbors -> target
        num_keep = len(keep_indices)
        new_edge_index = torch.tensor(
            [
                list(range(1, num_keep + 1)),  # New neighbor indices (1 to num_keep)
                [0] * num_keep,  # All point to target (0)
            ],
            dtype=torch.long,
        )
        
        # Subsample edge attributes (same order as neighbors)
        edge_attr_subsampled = data.edge_attr[[idx - 1 for idx in keep_indices]]  # edge_attr uses 0-based indexing
        
        # Create new Data object
        data_aug = Data(
            x=x_subsampled,
            edge_index=new_edge_index,
            edge_attr=edge_attr_subsampled,
            y=data.y,
            target_id=data.target_id,
            neighborhood_name=data.neighborhood_name,
            num_nodes=1 + num_keep,
        )
        return data_aug


class EdgeAttributeNoiseTransform:
    """Add small noise to edge attributes (spatial distances).
    
    This augmentation adds small perturbations to edge attributes (dx, dy, distances),
    simulating measurement uncertainty while preserving spatial relationships.
    
    Args:
        distance_noise_std: Standard deviation of noise for distance attributes (as fraction).
                          Defaults to 0.02 (2% noise).
        position_noise_std: Standard deviation of noise for dx, dy (as fraction).
                          Defaults to 0.02 (2% noise).
    
    Example:
        >>> transform = EdgeAttributeNoiseTransform()
        >>> augmented_data = transform(data)
    """
    
    def __init__(
        self,
        distance_noise_std: float = 0.02,
        position_noise_std: float = 0.02,
    ):
        self.distance_noise_std = distance_noise_std
        self.position_noise_std = position_noise_std
    
    def __call__(self, data: Data) -> Data:
        """Apply edge attribute noise augmentation with spatial awareness.
        
        Applies noise while preserving spatial relationships:
        - Distance noise scales with distance magnitude (same relative noise)
        - Ensures network_distance ≥ euclidean_distance (physical constraint)
        - Scales dx, dy proportionally when euclidean distance changes
        - Maintains normalized range [0, 1] for all attributes
        
        Args:
            data: PyTorch Geometric Data object.
            
        Returns:
            Augmented Data object with spatially-aware noisy edge attributes.
        """
        if data.edge_attr.shape[0] == 0:
            return data
        
        edge_attr = data.edge_attr.clone()
        
        # Edge attributes are normalized by MAX_DISTANCE_METERS=1000.0, so they're in [0, 1]
        # Indices: 0=dx, 1=dy, 2=euclidean_distance, 3=network_distance
        
        # Add spatially-aware noise to euclidean distance
        # Larger distances get larger absolute noise (but same relative noise)
        euclidean_dist = edge_attr[:, 2].unsqueeze(1)  # [num_edges, 1]
        euclidean_noise = torch.randn_like(euclidean_dist) * self.distance_noise_std * euclidean_dist
        euclidean_dist_new = euclidean_dist + euclidean_noise
        euclidean_dist_new = torch.clamp(euclidean_dist_new, min=0.0, max=1.0 + 1e-6)  # Keep in [0, 1]
        
        # Scale dx, dy proportionally when euclidean distance changes
        # This preserves the relative spatial relationships
        scale_factor = torch.where(
            euclidean_dist > 1e-6,  # Avoid division by zero
            euclidean_dist_new / euclidean_dist,
            torch.ones_like(euclidean_dist)
        ).squeeze(1)  # [num_edges]
        
        # Apply noise to dx, dy, then scale proportionally
        position_noise = torch.randn_like(edge_attr[:, 0:2]) * self.position_noise_std
        edge_attr[:, 0:2] = (edge_attr[:, 0:2] + position_noise) * scale_factor.unsqueeze(1)
        
        # Update euclidean distance
        edge_attr[:, 2] = euclidean_dist_new.squeeze(1)
        
        # Add noise to network distance (scale with distance magnitude)
        network_dist = edge_attr[:, 3].unsqueeze(1)  # [num_edges, 1]
        network_noise = torch.randn_like(network_dist) * self.distance_noise_std * network_dist
        network_dist_new = network_dist + network_noise
        network_dist_new = torch.clamp(network_dist_new, min=0.0, max=1.0 + 1e-6)  # Keep in [0, 1]
        
        # Ensure network_distance ≥ euclidean_distance (physical constraint)
        # Network path is always at least as long as straight-line distance
        network_dist_new = torch.maximum(network_dist_new, euclidean_dist_new)
        edge_attr[:, 3] = network_dist_new.squeeze(1)
        
        # Clamp all attributes to valid range [0, 1] (already normalized)
        edge_attr = torch.clamp(edge_attr, min=0.0, max=1.0 + 1e-6)
        
        data_aug = data.clone()
        data_aug.edge_attr = edge_attr
        return data_aug


class CompositeAugmentation:
    """Apply multiple augmentation transforms with configurable probabilities.
    
    This allows combining multiple augmentation strategies, each applied with
    a certain probability during training.
    
    Args:
        transforms: List of (transform, probability) tuples. Each transform is
                   applied with its corresponding probability.
                   Defaults to all transforms with p=0.5 each.
    
    Example:
        >>> aug = CompositeAugmentation([
        ...     (FeatureNoiseTransform(), 0.8),
        ...     (NeighborSubsamplingTransform(), 0.5),
        ...     (EdgeAttributeNoiseTransform(), 0.3),
        ... ])
        >>> augmented_data = aug(data)
    """
    
    def __init__(
        self,
        transforms: Optional[list] = None,
    ):
        if transforms is None:
            # Default: all augmentations with moderate probabilities
            transforms = [
                (FeatureNoiseTransform(noise_std=0.01), 0.8),
                (NeighborSubsamplingTransform(subsample_ratio=0.8), 0.5),
                (EdgeAttributeNoiseTransform(), 0.3),
            ]
        self.transforms = transforms
    
    def __call__(self, data: Data) -> Data:
        """Apply augmentation transforms probabilistically.
        
        Args:
            data: PyTorch Geometric Data object.
            
        Returns:
            Augmented Data object.
        """
        augmented = data
        
        for transform, probability in self.transforms:
            if random.random() < probability:
                augmented = transform(augmented)
        
        return augmented


def create_augmentation_transform(
    enable_feature_noise: bool = True,
    enable_neighbor_subsampling: bool = True,
    enable_edge_noise: bool = True,
    feature_noise_std: float = 0.01,
    subsample_ratio: float = 0.8,
    distance_noise_std: float = 0.02,
) -> Optional[CompositeAugmentation]:
    """Create a composite augmentation transform with specified options.
    
    This is a convenience function for creating augmentation transforms with
    sensible defaults. All transforms include realistic constraints:
    - FeatureNoiseTransform: Respects feature bounds (ratios [0,1], counts ≥ 0, etc.)
    - NeighborSubsamplingTransform: Maintains spatial diversity via distance stratification
    - EdgeAttributeNoiseTransform: Preserves spatial relationships (network ≥ euclidean)
    
    Args:
        enable_feature_noise: Whether to apply feature noise. Defaults to True.
        enable_neighbor_subsampling: Whether to apply neighbor subsampling. Defaults to True.
        enable_edge_noise: Whether to apply edge attribute noise. Defaults to True.
        feature_noise_std: Standard deviation for feature noise. Defaults to 0.01.
        subsample_ratio: Ratio for neighbor subsampling. Defaults to 0.8.
        distance_noise_std: Standard deviation for edge noise. Defaults to 0.02.
    
    Returns:
        CompositeAugmentation object or None if all augmentations are disabled.
    
    Example:
        >>> transform = create_augmentation_transform()
        >>> dataset = SpatialGraphDataset(df, transform=transform)
    """
    transforms = []
    
    if enable_feature_noise:
        transforms.append((FeatureNoiseTransform(noise_std=feature_noise_std), 0.8))
    
    if enable_neighbor_subsampling:
        transforms.append((NeighborSubsamplingTransform(subsample_ratio=subsample_ratio), 0.5))
    
    if enable_edge_noise:
        transforms.append((EdgeAttributeNoiseTransform(distance_noise_std=distance_noise_std), 0.3))
    
    if not transforms:
        return None
    
    return CompositeAugmentation(transforms)
