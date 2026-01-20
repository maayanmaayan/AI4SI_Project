# Quick Start - Overnight Experiments

## To Run Before Bed

Simply run:

```bash
./scripts/start_overnight_experiments.sh
```

This will:
1. Activate your virtual environment
2. Run 9 different experiments sequentially
3. Save all results to `experiments/overnight_runs/`
4. Generate a summary at the end

## What Each Experiment Tests

The script tests 9 different configurations to fix the uniformity problem:

1. **baseline_reduced_entropy**: Test if reducing entropy regularization alone helps
2. **no_maxsup**: Test if disabling MaxSup helps
3. **reduced_temperature**: Test if reducing temperature scaling helps
4. **combo_1_entropy_maxsup**: Test reducing entropy + disabling MaxSup
5. **combo_2_all_reductions**: Test ALL recommended changes together
6. **combo_3_mild**: Test mild reductions (safer approach)
7. **combo_4_aggressive**: Test aggressive reductions (might overfit)
8. **combo_5_no_entropy_reg**: Test completely removing entropy regularization
9. **combo_6_balanced**: Test balanced approach (middle ground)

## Expected Time

- Each experiment: 30-60 minutes (depends on early stopping)
- Total: ~5-9 hours (should finish before you wake up!)

## Check Results in the Morning

```bash
# View summary
cat experiments/overnight_runs/summary.json | python3 -m json.tool

# View best results
cat experiments/overnight_runs/summary.json | grep -A 5 "best_val_loss"

# Check individual experiment
cat experiments/overnight_runs/baseline_reduced_entropy/experiment_metadata.json | python3 -m json.tool
```

## If Something Goes Wrong

- Each experiment saves stdout/stderr separately
- Check `experiments/overnight_runs/<experiment_name>/stderr.txt` for errors
- The script continues even if one experiment fails
- You can resume with `--skip-completed` flag

## Monitor Progress (Optional)

If you want to check progress before bed:

```bash
# Count completed experiments
ls -d experiments/overnight_runs/*/run 2>/dev/null | wc -l

# View latest experiment log
tail -f experiments/overnight_runs/*/run/logs/training.log
```

## Good Night! 🌙

The experiments will run through the night and be ready in the morning!