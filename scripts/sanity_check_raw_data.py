"""Script to run raw data sanity checks on OSM and Census data files."""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.validation.raw_data_sanity_checker import RawDataSanityChecker
from src.utils.logging import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate raw OSM and Census data files against reasonable value ranges"
    )
    parser.add_argument(
        "--neighborhood",
        type=str,
        default=None,
        help="Specific neighborhood name to validate (if not provided, validates all compliant neighborhoods)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save validation report files (default: data/raw)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="INFO")

    print("=" * 80)
    print("Running Raw Data Sanity Check")
    print("=" * 80)

    # Initialize checker
    checker = RawDataSanityChecker()

    # Run validation
    if args.neighborhood:
        print(f"Validating neighborhood: {args.neighborhood}")
        result = checker.validate_neighborhood(args.neighborhood)
        # Convert single result to format expected by _generate_report
        validation_results = {
            "summary": {
                "total_neighborhoods": 1,
                "passed": 1 if result["status"] == "pass" else 0,
                "warnings": 1 if result["status"] == "warn" else 0,
                "failed": 1 if result["status"] == "fail" else 0,
                "total_checks": len(result.get("checks", {})),
                "passed_checks": sum(
                    1
                    for check in result.get("checks", {}).values()
                    if check.get("status") == "pass"
                ),
                "warning_checks": sum(
                    1
                    for check in result.get("checks", {}).values()
                    if check.get("status") == "warn"
                ),
                "failed_checks": sum(
                    1
                    for check in result.get("checks", {}).values()
                    if check.get("status") == "fail"
                ),
            },
            "neighborhoods": [result],
        }
    else:
        print("Validating all compliant neighborhoods")
        validation_results = checker.validate_all_neighborhoods()

    # Generate report
    output_dir = Path(args.output_dir)
    checker._generate_report(validation_results, output_dir)

    print("\nValidation complete!")
    print("=" * 80)
