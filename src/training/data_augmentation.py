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
        """Apply feature noise augmentation.
        
        Args:
            data: PyTorch Geometric Data object.
            
        Returns:
            Augmented Data object with noisy features.
        """
        x = data.x.clone()
        
        if self.apply_to_target:
            # Add noise to target node (index 0)
            noise = torch.randn_like(x[0:1]) * self.noise_std
            x[0:1] = x[0:1] + noise
        
        if self.apply_to_neighbors and x.shape[0] > 1:
            # Add noise to neighbor nodes (indices 1+)
            noise = torch.randn_like(x[1:]) * self.noise_std
            x[1:] = x[1:] + noise
        
        # Create new Data object with augmented features
        data_aug = data.clone()
        data_aug.x = x
        return data_aug


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
        """Apply neighbor subsampling augmentation.
        
        Args:
            data: PyTorch Geometric Data object.
            
        Returns:
            Augmented Data object with subsampled neighbors.
        """
        num_neighbors = data.x.shape[0] - 1  # Exclude target node
        
        # Don't augment if too few neighbors
        if num_neighbors <= self.min_neighbors:
            return data
        
        # Calculate number of neighbors to keep
        num_keep = max(
            self.min_neighbors,
            int(num_neighbors * self.subsample_ratio)
        )
        num_keep = min(num_keep, num_neighbors)  # Don't keep more than available
        
        # Randomly select neighbors to keep
        neighbor_indices = list(range(1, num_neighbors + 1))  # Indices 1 to num_neighbors
        keep_indices = sorted(random.sample(neighbor_indices, num_keep))
        keep_indices_with_target = [0] + keep_indices  # Always keep target (index 0)
        
        # Subsample nodes
        x_subsampled = data.x[keep_indices_with_target]
        
        # Rebuild edge_index for subsampled neighbors
        # Original neighbors were indices 1 to num_neighbors
        # New neighbors should be indices 1 to num_keep (mapped from keep_indices)
        # Map old neighbor indices to new indices (target stays at 0)
        old_to_new = {0: 0}  # Target always at 0
        for new_idx, old_idx in enumerate(keep_indices, start=1):
            old_to_new[old_idx] = new_idx
        
        # Build new edge_index: neighbors -> target
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
        """Apply edge attribute noise augmentation.
        
        Args:
            data: PyTorch Geometric Data object.
            
        Returns:
            Augmented Data object with noisy edge attributes.
        """
        if data.edge_attr.shape[0] == 0:
            return data
        
        edge_attr = data.edge_attr.clone()
        
        # Add noise to position attributes (dx, dy) - indices 0, 1
        position_noise = torch.randn_like(edge_attr[:, 0:2]) * self.position_noise_std
        edge_attr[:, 0:2] = edge_attr[:, 0:2] + position_noise
        
        # Add noise to distance attributes (euclidean_dist, network_dist) - indices 2, 3
        distance_noise = torch.randn_like(edge_attr[:, 2:4]) * self.distance_noise_std
        # Ensure distances stay non-negative
        edge_attr[:, 2:4] = torch.clamp(edge_attr[:, 2:4] + distance_noise, min=0.0)
        
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
    sensible defaults.
    
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
