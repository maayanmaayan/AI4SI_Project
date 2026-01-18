"""Script to backup existing feature data before regeneration.

This script creates a timestamped backup of the feature data directory
before making changes to the sampling interval or feature engineering pipeline.
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import ensure_dir_exists
from src.utils.logging import get_logger, setup_logging

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def backup_features(source_dir: str, backup_base_dir: str = ".backup") -> Path:
    """Backup feature data directory to timestamped backup location.
    
    Args:
        source_dir: Source directory to backup (e.g., "data/processed/features").
        backup_base_dir: Base directory for backups. Defaults to ".backup".
    
    Returns:
        Path to the backup directory.
    
    Raises:
        FileNotFoundError: If source directory doesn't exist.
        ValueError: If source directory is empty.
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    if not any(source_path.iterdir()):
        raise ValueError(f"Source directory is empty: {source_dir}")
    
    # Create timestamped backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"features_100m_{timestamp}"
    backup_path = Path(backup_base_dir) / backup_name
    
    # Ensure backup base directory exists
    ensure_dir_exists(backup_base_dir)
    
    logger.info(f"Creating backup: {source_dir} -> {backup_path}")
    
    # Copy directory tree
    try:
        shutil.copytree(source_path, backup_path)
        logger.info(f"✓ Backup created successfully: {backup_path}")
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise
    
    # Count files and calculate size
    file_count = sum(1 for _ in backup_path.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    # Create backup manifest
    manifest_path = backup_path / "backup_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"Backup Manifest\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Source: {source_dir}\n")
        f.write(f"Backup: {backup_path}\n")
        f.write(f"File Count: {file_count}\n")
        f.write(f"Total Size: {size_mb:.2f} MB\n")
    
    logger.info(f"Backup Summary:")
    logger.info(f"  Location: {backup_path}")
    logger.info(f"  Files: {file_count}")
    logger.info(f"  Size: {size_mb:.2f} MB")
    logger.info(f"  Manifest: {manifest_path}")
    
    return backup_path


def main():
    """Main function to run backup."""
    parser = argparse.ArgumentParser(
        description="Backup existing feature data before regeneration"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/processed/features",
        help="Source directory to backup (default: data/processed/features)",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=".backup",
        help="Base directory for backups (default: .backup)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Feature Data Backup")
    print("=" * 80)
    
    try:
        backup_path = backup_features(args.source, args.backup_dir)
        print(f"\n✓ Backup completed successfully!")
        print(f"  Location: {backup_path}")
        print(f"  Restore with: cp -r {backup_path}/* {args.source}/")
    except Exception as e:
        print(f"\n✗ Backup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
