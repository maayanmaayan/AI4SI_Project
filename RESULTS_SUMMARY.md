# Results Summary

Generated: **2026-01-20 14:03**

This file consolidates key metrics from experiment outputs under `experiments/`.

## Overnight regularization experiments (`experiments/overnight_runs/`)

Sorted by **Test KL divergence (lower is better)**.

| Experiment | Best Val Loss | Test KL | Test Top-1 | Test Top-3 | Epochs |
|---|---|---|---|---|---|
| combo_4_aggressive | 0.4496 | 0.3260 | 34.84% | 97.91% | 30 |
| combo_2_all_reductions | 0.3802 | 0.3365 | 34.84% | 97.91% | 30 |
| reduced_temperature | 0.3980 | 0.3463 | 34.84% | 97.91% | 30 |
| combo_5_no_entropy_reg | 0.3981 | 0.3464 | 34.84% | 97.91% | 30 |
| combo_6_balanced | 0.3570 | 0.3504 | 34.84% | 97.91% | 30 |
| combo_3_mild | 0.3827 | 0.3722 | 34.84% | 97.91% | 30 |
| combo_1_entropy_maxsup | 0.3718 | 0.3962 | 34.84% | 97.91% | 30 |
| no_maxsup | 0.3718 | 0.3962 | 34.84% | 97.91% | 30 |
| baseline_reduced_entropy | 0.3874 | 0.3977 | 34.84% | 97.91% | 30 |

Narrative: `experiments/overnight_runs/EXPERIMENT_SUMMARY.md`.

## Hyperparameter sweeps (`experiments/runs/sweep_*`)

Note: `experiments/runs/` is gitignored; metrics are captured here for reviewers.

### sweep:sweep_1768899422

| Experiment | Best Val Loss | Test KL | Test Top-1 | Test Top-3 | Epochs |
|---|---|---|---|---|---|
| shallow | 0.3757 | 0.3530 | 34.84% | 97.91% | 20 |
| higher_lr | 0.3769 | 0.3373 | 34.84% | 97.91% | 20 |
| baseline | 0.3856 | 0.3366 | 34.84% | 97.91% | 20 |
| lower_lr | 0.3867 | 0.3456 | 34.84% | 97.91% | 20 |

## Other training runs (`experiments/runs/*`)

Note: `experiments/runs/` is gitignored; metrics are captured here for reviewers.

| Experiment | Best Val Loss | Test KL | Test Top-1 | Test Top-3 | Epochs |
|---|---|---|---|---|---|
| extreme_focal_loss_test | 0.1780 | 0.3962 | 34.84% | 95.22% | 20 |
| alpha_focal_loss_test | 0.2707 | 0.3979 | 34.84% | 94.68% | 20 |
| focal_loss_test | 0.2802 | 0.3956 | 34.84% | 97.91% | 20 |
| run_1768836689 | 0.3875 | 0.3984 | 34.84% | 97.91% | 100 |
| run_1768745543 | 16.0193 | 14.6613 | 4.35% | 58.70% | 39 |
| run_1768749157 | 18.2123 | 13.9943 | 10.87% | 80.43% | 48 |
| run_1768752721 | 29.4266 | 13.9943 | 10.87% | 80.43% | 48 |


## Key results (high-level takeaways)

- **Overall best Test KL (lower is better)**: `combo_4_aggressive` (**0.3260**) — strongest distribution match, but with the **worst validation loss** among the overnight runs (likely overconfident / overfit regime).
- **Best balanced candidate**: `combo_2_all_reductions` (Best Val Loss **0.3802**, Test KL **0.3365**) — good KL improvement without the worst validation penalty.
- **Stability vs. sharpness tradeoff**: `combo_6_balanced` has the **best validation loss** (**0.3570**) among the overnight runs, at the cost of a slightly worse Test KL (**0.3504**).
- **Prediction quality plateaus**: Across most runs and sweeps, **Top‑1 accuracy stays ~34.84%** and **Top‑3 stays ~97.91%**, suggesting that changing hyperparameters/regularization often reshapes confidence/calibration more than it changes which category is predicted.
- **Focal loss experiments**: `focal_loss_test`, `alpha_focal_loss_test`, and `extreme_focal_loss_test` did **not** produce a meaningful Top‑1 improvement (still ~34.84%), and their KL values remain similar to the non-focal baselines (~0.395–0.398). Notably, Top‑3 drops in the focal variants (**~94–95%** vs **~97.9%**), suggesting the focal-style weighting may reduce the model’s “near-miss” ranking quality without improving the top choice.
- **Outliers / early pipeline regimes**: Some older runs (e.g., `run_1768745543`, `run_1768749157`, `run_1768752721`) show extremely large losses/KL and much lower accuracy, consistent with earlier pipeline versions or instability. They are kept for traceability but should not be used for model selection.