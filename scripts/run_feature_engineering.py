"""Script to run FeatureEngineer and generate processed features for all neighborhoods."""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.collection.feature_engineer import FeatureEngineer
from src.utils.helpers import load_neighborhoods
from src.utils.logging import setup_logging

# Setup logging to ensure output is visible
setup_logging(log_level="INFO")


def main():
    """Main function to run feature engineering pipeline."""
    parser = argparse.ArgumentParser(
        description="Run feature engineering pipeline for all neighborhoods"
    )
    parser.add_argument(
        "--neighborhood",
        type=str,
        help="Process only this specific neighborhood (by name)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all neighborhoods",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if output files exist",
    )
    parser.add_argument(
        "--geojson",
        type=str,
        default="paris_neighborhoods.geojson",
        help="Path to neighborhoods GeoJSON file",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Running Feature Engineering Pipeline")
    print("=" * 80)

    # Load neighborhoods
    geojson_path = args.geojson
    if not Path(geojson_path).exists():
        print(f"Error: GeoJSON file not found: {geojson_path}")
        return

    neighborhoods = load_neighborhoods(geojson_path)
    print(f"Loaded {len(neighborhoods)} neighborhoods from {geojson_path}")

    # Initialize FeatureEngineer
    engineer = FeatureEngineer()

    if args.neighborhood:
        # Process single neighborhood
        print(f"\nProcessing neighborhood: {args.neighborhood}")
        matching = neighborhoods[neighborhoods["name"] == args.neighborhood]
        if matching.empty:
            print(f"Error: Neighborhood '{args.neighborhood}' not found")
            return

        result = engineer.process_neighborhood(matching.iloc[0], force=args.force)
        if not result.empty:
            print(f"Successfully processed {len(result)} target points")
        else:
            print("No target points processed")

    elif args.all:
        # Process all neighborhoods
        print("\nProcessing all neighborhoods...")
        summary = engineer.process_all_neighborhoods(neighborhoods, force=args.force)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total neighborhoods: {summary['total_neighborhoods']}")
        print(f"Compliant neighborhoods: {summary['compliant_count']}")
        print(f"Non-compliant neighborhoods: {summary['non_compliant_count']}")
        print(f"Successfully processed: {summary['successful_count']}")
        print(f"Failed: {summary['failed_count']}")
        print(f"Total target points: {summary['total_target_points']}")
        print("=" * 80)

    else:
        print("\nError: Must specify --neighborhood <name> or --all")
        parser.print_help()


if __name__ == "__main__":
    main()
