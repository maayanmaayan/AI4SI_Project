"""Utilities for filtering training data for quick testing.

This module provides functions to filter features DataFrame for quick testing
with smaller datasets.
"""

from typing import List, Optional

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def filter_small_neighborhoods(
    features_df: pd.DataFrame,
    max_points_per_neighborhood: int = 50,
    max_neighborhoods: int = 3,
) -> pd.DataFrame:
    """Filter features DataFrame to include only small neighborhoods.

    Useful for quick testing with smaller datasets. Selects neighborhoods with
    fewer than max_points_per_neighborhood points, up to max_neighborhoods.

    Args:
        features_df: DataFrame with 'neighborhood_name' column.
        max_points_per_neighborhood: Maximum number of points per neighborhood.
            Defaults to 50.
        max_neighborhoods: Maximum number of neighborhoods to include.
            Defaults to 3.

    Returns:
        Filtered DataFrame with only small neighborhoods.

    Example:
        >>> df = load_features_from_directory("data/processed/features/compliant")
        >>> small_df = filter_small_neighborhoods(df, max_points_per_neighborhood=50, max_neighborhoods=3)
        >>> print(f"Filtered to {len(small_df)} points from {small_df['neighborhood_name'].nunique()} neighborhoods")
    """
    if "neighborhood_name" not in features_df.columns:
        raise ValueError("DataFrame must have 'neighborhood_name' column")

    # Count points per neighborhood
    neighborhood_counts = features_df["neighborhood_name"].value_counts()
    small_neighborhoods = neighborhood_counts[
        neighborhood_counts < max_points_per_neighborhood
    ].index.tolist()

    if len(small_neighborhoods) == 0:
        logger.warning(
            f"No neighborhoods with < {max_points_per_neighborhood} points found. "
            "Using all neighborhoods."
        )
        # If no small neighborhoods, just limit to max_neighborhoods
        all_neighborhoods = neighborhood_counts.index.tolist()[:max_neighborhoods]
        filtered_df = features_df[features_df["neighborhood_name"].isin(all_neighborhoods)].copy()
    else:
        # Select up to max_neighborhoods small neighborhoods
        selected_neighborhoods = small_neighborhoods[:max_neighborhoods]
        filtered_df = features_df[
            features_df["neighborhood_name"].isin(selected_neighborhoods)
        ].copy()

        logger.info(
            f"Filtered to {len(filtered_df)} points from {len(selected_neighborhoods)} "
            f"small neighborhoods (< {max_points_per_neighborhood} points each): "
            f"{', '.join(selected_neighborhoods)}"
        )

    return filtered_df
