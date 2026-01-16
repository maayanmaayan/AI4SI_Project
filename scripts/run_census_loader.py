"""Script to run CensusLoader and fetch census data for all compliant neighborhoods."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.collection.census_loader import CensusLoader

if __name__ == "__main__":
    print("=" * 80)
    print("Running CensusLoader for all compliant neighborhoods")
    print("=" * 80)
    print("Note: Using force=True to regenerate with performance fixes")
    print("Note: pynsee caches downloaded files (e.g., RP_LOGEMENT) automatically")
    print("=" * 80)
    
    loader = CensusLoader()
    
    # Load all compliant neighborhoods (force=True to regenerate with performance fixes)
    # Note: pynsee caches downloaded files (e.g., RP_LOGEMENT) automatically, so won't re-download
    summary = loader.load_all_neighborhoods(force=True)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total compliant neighborhoods: {summary['total_neighborhoods']}")
    print(f"Successfully loaded: {summary['successful_count']}")
    print(f"Failed: {summary['failed_count']}")
    print(f"Cached (skipped): {summary['cached_count']}")
    print("=" * 80)
