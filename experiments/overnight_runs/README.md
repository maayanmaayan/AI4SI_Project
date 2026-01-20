# Overnight Experiments

This directory contains results from overnight experiments testing different regularization configurations.

## Experiments

The script runs 9 different experiments to test various combinations of:

1. **Entropy regularization** (`entropy_weight`, `min_entropy`)
2. **MaxSup penalty** (`maxsup_weight`)
3. **Temperature scaling** (`temperature`)
4. **LogitNorm** (`logit_norm`)
5. **Learning rate** (`learning_rate`)

### Experiment List (Ordered by Priority - Most Promising First)

**HIGH PRIORITY (1-4) - Most likely to succeed:**

1. **combo_6_balanced** ⭐⭐⭐: Balanced approach - entropy_weight 1.0, no MaxSup, temperature 2.5, logit_norm 1.8
   - *Moderate changes, safest bet, most likely to help*

2. **combo_2_all_reductions** ⭐⭐: All recommended changes combined
   - *Entropy (0.5), no MaxSup, temperature (2.0), logit_norm (2.0)*

3. **combo_1_entropy_maxsup** ⭐⭐: Key combination - reduce entropy + disable MaxSup
   - *Two most important fixes together*

4. **baseline_reduced_entropy** ⭐: Simplest fix - reduce entropy_weight 5.0→0.5
   - *Single most important change*

**MEDIUM PRIORITY (5-7) - Individual tests and conservative approaches:**

5. **combo_3_mild**: Mild reductions - entropy_weight 2.0, temperature 3.0
6. **reduced_temperature**: Temperature fix - test temperature impact alone
7. **no_maxsup**: MaxSup test - disable MaxSup penalty

**LOWER PRIORITY (8-9) - Experimental/risky approaches:**

8. **combo_4_aggressive**: Aggressive - might overfit, tests limits
9. **combo_5_no_entropy_reg**: No entropy regularization - test if entropy reg is main issue

**Note:** Experiments are ordered by priority. If the script stops early, you'll have the most promising results first!

## Running

To run all experiments:

```bash
./scripts/start_overnight_experiments.sh
```

To run specific experiments:

```bash
./scripts/start_overnight_experiments.sh --experiments baseline_reduced_entropy no_maxsup
```

To skip already completed experiments:

```bash
./scripts/start_overnight_experiments.sh --skip-completed
```

## Results

Each experiment creates a subdirectory with:

- `config.yaml`: Modified configuration file used for the experiment
- `experiment_metadata.json`: Experiment metadata, start/end times, results
- `run/`: Training run directory with logs, checkpoints, plots
- `stdout.txt`: Training stdout
- `stderr.txt`: Training stderr

## Summary

After all experiments complete, check `summary.json` for:

- Total experiments run
- Completed vs failed counts
- Best validation losses
- Best test KL divergences
- Total elapsed time

## Expected Runtime

- Each experiment: ~30-60 minutes (depending on early stopping)
- Total for 9 experiments: ~5-9 hours
- With early stopping (patience=5): ~4-6 hours

## Analysis

After experiments complete, analyze results to find:

1. Best validation loss
2. Best test KL divergence
3. Prediction distribution entropy (should match target entropy)
4. Whether predictions are location-specific or still uniform