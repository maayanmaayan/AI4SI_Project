"""Script to run CensusLoader and fetch census data for all compliant neighborhoods."""

from src.data.collection.census_loader import CensusLoader

if __name__ == "__main__":
    print("=" * 80)
    print("Running CensusLoader for all compliant neighborhoods")
    print("=" * 80)
    
    loader = CensusLoader()
    
    # Load all compliant neighborhoods (force=True to regenerate with new features)
    summary = loader.load_all_neighborhoods(force=True)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total compliant neighborhoods: {summary['total_neighborhoods']}")
    print(f"Successfully loaded: {summary['successful_count']}")
    print(f"Failed: {summary['failed_count']}")
    print(f"Cached (skipped): {summary['cached_count']}")
    print("=" * 80)
