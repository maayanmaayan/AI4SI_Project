#!/usr/bin/env python3
"""Overnight experiments script to test different regularization configurations.

This script runs multiple experiments with different configurations to find
optimal settings for preventing uniformity and improving KL divergence.

Each experiment:
- Creates a new experiment directory
- Uses a modified config file
- Runs training with early stopping
- Saves results separately

Usage:
    python scripts/run_overnight_experiments.py
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config, load_config, save_config


EXPERIMENTS = [
    # HIGHEST PRIORITY - Most promising combinations based on analysis
    {
        "name": "combo_6_balanced",
        "description": "⭐ BALANCED (BEST BET): entropy_weight 1.0, no MaxSup, temperature 2.5, logit_norm 1.8 - Moderate changes, safe approach",
        "priority": 1,
        "modifications": {
            "loss": {
                "entropy_weight": 1.0,
                "min_entropy": 0.8,
                "maxsup_weight": 0.0,
            },
            "model": {
                "temperature": 2.5,
                "logit_norm": 1.8,
            }
        }
    },
    {
        "name": "combo_2_all_reductions",
        "description": "⭐⭐ ALL RECOMMENDED CHANGES: entropy (0.5), no MaxSup, temperature (2.0), logit_norm (2.0) - All fixes together",
        "priority": 2,
        "modifications": {
            "loss": {
                "entropy_weight": 0.5,
                "min_entropy": 0.5,
                "maxsup_weight": 0.0,
            },
            "model": {
                "temperature": 2.0,
                "logit_norm": 2.0,
            }
        }
    },
    {
        "name": "combo_1_entropy_maxsup",
        "description": "⭐⭐ KEY COMBINATION: Reduce entropy (0.5) + disable MaxSup - Two most important fixes",
        "priority": 3,
        "modifications": {
            "loss": {
                "entropy_weight": 0.5,
                "min_entropy": 0.5,
                "maxsup_weight": 0.0,
            }
        }
    },
    {
        "name": "baseline_reduced_entropy",
        "description": "⭐ SIMPLEST FIX: Reduce entropy_weight 5.0→0.5, min_entropy 1.5→0.5 - Single most important change",
        "priority": 4,
        "modifications": {
            "loss": {
                "entropy_weight": 0.5,
                "min_entropy": 0.5,
            }
        }
    },
    
    # MEDIUM PRIORITY - Individual component tests and moderate approaches
    {
        "name": "combo_3_mild",
        "description": "MILD REDUCTIONS: entropy_weight 2.0, temperature 3.0, logit_norm 1.5 - Safer conservative approach",
        "priority": 5,
        "modifications": {
            "loss": {
                "entropy_weight": 2.0,
                "min_entropy": 1.0,
            },
            "model": {
                "temperature": 3.0,
                "logit_norm": 1.5,
            }
        }
    },
    {
        "name": "reduced_temperature",
        "description": "TEMPERATURE FIX: Reduce temperature 4.0→2.0, increase logit_norm 1.0→2.0 - Test temperature impact",
        "priority": 6,
        "modifications": {
            "model": {
                "temperature": 2.0,
                "logit_norm": 2.0,
            }
        }
    },
    {
        "name": "no_maxsup",
        "description": "MAXSUP TEST: Disable MaxSup penalty (0.1→0.0) - Test MaxSup impact alone",
        "priority": 7,
        "modifications": {
            "loss": {
                "maxsup_weight": 0.0,
            }
        }
    },
    
    # LOWER PRIORITY - More experimental/risky approaches
    {
        "name": "combo_4_aggressive",
        "description": "⚠️ AGGRESSIVE: entropy_weight 0.1, temperature 1.5, logit_norm 3.0, higher LR - Might overfit, test limits",
        "priority": 8,
        "modifications": {
            "loss": {
                "entropy_weight": 0.1,
                "min_entropy": 0.3,
                "maxsup_weight": 0.0,
            },
            "model": {
                "temperature": 1.5,
                "logit_norm": 3.0,
            },
            "training": {
                "learning_rate": 0.0002,
            }
        }
    },
    {
        "name": "combo_5_no_entropy_reg",
        "description": "⚠️ NO ENTROPY REG: Completely disable entropy regularization (0.0) - Test if entropy reg is the main issue",
        "priority": 9,
        "modifications": {
            "loss": {
                "entropy_weight": 0.0,
            },
            "model": {
                "temperature": 2.0,
                "logit_norm": 2.0,
            }
        }
    },
]


def apply_modifications(base_config: dict, modifications: dict) -> dict:
    """Apply modifications to base config recursively.
    
    Args:
        base_config: Base configuration dictionary.
        modifications: Nested dictionary of modifications.
    
    Returns:
        Modified configuration dictionary.
    """
    config = json.loads(json.dumps(base_config))  # Deep copy
    
    for section, changes in modifications.items():
        if section in config:
            config[section].update(changes)
        else:
            config[section] = changes
    
    return config


def run_experiment(experiment: dict, base_config_path: Path, results_dir: Path) -> dict:
    """Run a single experiment.
    
    Args:
        experiment: Experiment configuration dictionary.
        base_config_path: Path to base config file.
        results_dir: Directory to save experiment results.
    
    Returns:
        Dictionary with experiment results.
    """
    experiment_name = experiment["name"]
    print(f"\n{'='*80}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Description: {experiment['description']}")
    print(f"{'='*80}\n")
    
    # Create experiment directory
    experiment_dir = results_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base config
    base_config = load_config(base_config_path)
    
    # Apply modifications
    modified_config = apply_modifications(base_config, experiment.get("modifications", {}))
    
    # Save modified config
    config_path = experiment_dir / "config.yaml"
    save_config(modified_config, config_path)
    
    # Save experiment metadata
    metadata = {
        "name": experiment_name,
        "description": experiment["description"],
        "start_time": datetime.now().isoformat(),
        "base_config": str(base_config_path),
        "modifications": experiment.get("modifications", {}),
    }
    
    metadata_path = experiment_dir / "experiment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Run training
    start_time = time.time()
    try:
        cmd = [
            sys.executable,
            "scripts/train_graph_transformer.py",
            "--config", str(config_path),
            "--experiment-dir", str(experiment_dir / "run"),
            "--log-level", "INFO",
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        
        elapsed_time = time.time() - start_time
        
        # Save stdout and stderr
        with open(experiment_dir / "stdout.txt", "w") as f:
            f.write(result.stdout)
        with open(experiment_dir / "stderr.txt", "w") as f:
            f.write(result.stderr)
        
        # Update metadata with results
        metadata.update({
            "end_time": datetime.now().isoformat(),
            "elapsed_time_seconds": elapsed_time,
            "return_code": result.returncode,
            "success": result.returncode == 0,
        })
        
        # Try to extract final metrics from stdout
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            for line in lines:
                if "Best validation loss:" in line:
                    try:
                        val_loss = float(line.split(":")[-1].strip())
                        metadata["best_val_loss"] = val_loss
                    except:
                        pass
                if "Test KL divergence:" in line:
                    try:
                        test_kl = float(line.split(":")[-1].strip())
                        metadata["test_kl_divergence"] = test_kl
                    except:
                        pass
        
        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        if result.returncode == 0:
            print(f"✅ Experiment {experiment_name} completed successfully")
            print(f"   Elapsed time: {elapsed_time/60:.1f} minutes")
        else:
            print(f"❌ Experiment {experiment_name} failed with return code {result.returncode}")
            print(f"   Check {experiment_dir / 'stderr.txt'} for details")
        
        return metadata
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        metadata.update({
            "end_time": datetime.now().isoformat(),
            "elapsed_time_seconds": elapsed_time,
            "error": str(e),
            "success": False,
        })
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"❌ Experiment {experiment_name} crashed: {e}")
        return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run overnight experiments to test different regularization configurations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/config.yaml",
        help="Path to base config file (default: models/config.yaml)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/overnight_runs",
        help="Directory to save experiment results (default: experiments/overnight_runs)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to run (by name). If not specified, runs all experiments.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip experiments that already have results",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_config_path = project_root / args.config
    results_dir = project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Select experiments to run
    if args.experiments:
        experiments_to_run = [e for e in EXPERIMENTS if e["name"] in args.experiments]
        if not experiments_to_run:
            print(f"Error: No matching experiments found for: {args.experiments}")
            print(f"Available experiments: {[e['name'] for e in EXPERIMENTS]}")
            return 1
    else:
        experiments_to_run = EXPERIMENTS
    
    # Filter out completed experiments if requested
    if args.skip_completed:
        experiments_to_run = [
            e for e in experiments_to_run
            if not (results_dir / e["name"] / "experiment_metadata.json").exists()
        ]
    
    print(f"\n{'='*80}")
    print(f"Overnight Experiments Script")
    print(f"{'='*80}")
    print(f"Base config: {base_config_path}")
    print(f"Results directory: {results_dir}")
    print(f"Total experiments: {len(experiments_to_run)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Sort experiments by priority (if specified)
    experiments_to_run = sorted(experiments_to_run, key=lambda x: x.get("priority", 999))
    
    # Print experiment order
    print(f"\n{'='*80}")
    print(f"Experiment Order (by priority):")
    print(f"{'='*80}")
    for i, exp in enumerate(experiments_to_run, 1):
        priority = exp.get("priority", 999)
        print(f"{i}. [{priority}] {exp['name']}: {exp['description']}")
    print(f"{'='*80}\n")
    
    # Run experiments
    results = []
    total_start_time = time.time()
    
    for i, experiment in enumerate(experiments_to_run, 1):
        priority = experiment.get("priority", 999)
        print(f"\n[{i}/{len(experiments_to_run)}] Priority {priority}: {experiment['name']}")
        
        result = run_experiment(experiment, base_config_path, results_dir)
        results.append(result)
        
        # Estimate remaining time
        if i > 0:
            avg_time = (time.time() - total_start_time) / i
            remaining = avg_time * (len(experiments_to_run) - i)
            print(f"\n⏱️  Average time per experiment: {avg_time/60:.1f} minutes")
            print(f"⏱️  Estimated remaining time: {remaining/60:.1f} minutes ({remaining/3600:.1f} hours)")
            
            # Show which high-priority experiments are done
            completed_high = sum(1 for r in results if r.get("priority", 999) <= 4 and r.get("success", False))
            print(f"✅ Completed high-priority experiments (1-4): {completed_high}/4")
    
    # Generate summary
    total_elapsed = time.time() - total_start_time
    
    summary = {
        "start_time": datetime.fromtimestamp(total_start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_elapsed_seconds": total_elapsed,
        "total_experiments": len(experiments_to_run),
        "completed": sum(1 for r in results if r.get("success", False)),
        "failed": sum(1 for r in results if not r.get("success", False)),
        "experiments": results,
    }
    
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Overnight Experiments Summary")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments_to_run)}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print(f"\nResults saved to: {results_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")
    
    # Print best results
    successful_results = [r for r in results if r.get("success", False)]
    if successful_results:
        print("\n" + "="*80)
        print("BEST RESULTS")
        print("="*80)
        
        print("\n🏆 Best Validation Loss (lower is better):")
        sorted_results = sorted(
            [r for r in successful_results if "best_val_loss" in r],
            key=lambda x: x["best_val_loss"]
        )
        for i, r in enumerate(sorted_results[:5], 1):
            priority = r.get("priority", "?")
            print(f"  {i}. [{priority}] {r['name']}: {r.get('best_val_loss', 'N/A'):.4f}")
        
        print("\n🏆 Best Test KL Divergence (lower is better):")
        sorted_results = sorted(
            [r for r in successful_results if "test_kl_divergence" in r],
            key=lambda x: x["test_kl_divergence"]
        )
        for i, r in enumerate(sorted_results[:5], 1):
            priority = r.get("priority", "?")
            print(f"  {i}. [{priority}] {r['name']}: {r.get('test_kl_divergence', 'N/A'):.4f}")
        
        # Show high-priority results first
        print("\n⭐ High-Priority Results (1-4):")
        high_priority = sorted(
            [r for r in successful_results if r.get("priority", 999) <= 4],
            key=lambda x: x.get("priority", 999)
        )
        for r in high_priority:
            val_loss = r.get("best_val_loss", "N/A")
            test_kl = r.get("test_kl_divergence", "N/A")
            print(f"  [{r.get('priority', '?')}] {r['name']}: val_loss={val_loss}, test_kl={test_kl}")
        
        print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())