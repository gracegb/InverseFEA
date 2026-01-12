# Noisy data model experiments

Scripts in this folder focus on training or evaluating models on the PCA-based
noisy datasets. Most workflows consume the 15 PCA features:
`PC1..3_InnerBase`, `PC1..3_OuterBase`, `PC1..3_InnerCircle`,
`PC1..3_MiddleCircle`, `PC1..3_OuterCircle`, and predict:
`Part1_E`, `Part3_E`, `Part11_E`.

## Top-level scripts

| Script | Purpose | Outputs |
| --- | --- | --- |
| `baseline_rf_pca.py` | Baseline Random Forest with shuffled K-Fold CV; direct multi-target prediction (no chaining). | `baseline_predictions.csv`, fold/mean metrics, `feature_importances.csv` in `noisy_data_tests/outputs/baseline_rf/`. |
| `ml_chained_noisy_UPDATED.py` | Chained Random Forest pipeline with CV, normalization, and old vs noisy dataset comparison. | Per-dataset predictions, metrics, plots, and permutation importances. |
| `ml_chained_noisy_from_clean.py` | Early chained RF workflow using clean data with noisy predictions and plots. | Predictions, plots, and saved models (paths inside script). |
| `ml_oof_coral_quantile_part1_focus.py` | Experimental OOF stacking with CORAL alignment, geometry diagnostics, and quantile modeling for Part1_E. | OOF predictions, fold metrics, and feature importances under a configurable output dir. |

## Loop-based chained experiments (`loop_tests/`)

These scripts iteratively refine chained predictions by feeding predicted values
back into the chain.

| Script | Chain order / refinement strategy | Outputs |
| --- | --- | --- |
| `loop_tests/ml_chained_xgboost_loop.py` | Forward chain (Part3_E -> Part11_E -> Part1_E) with iterative refinement. | Predictions and metrics under the `--output` directory. |
| `loop_tests/ml_chained_xgboost_loop_reversed.py` | Reverse chain order to compare ordering effects. | Predictions and metrics under the `--output` directory. |
| `loop_tests/ml_chained_xgboost_sequential_feedback.py` | Forward chain with normalization tweaks for sequential feedback. | Predictions and metrics under the `--output` directory. |

## Outputs

Default outputs are written beneath `noisy_data_tests/outputs/` or a script's
`--output` or `--output-dir` argument. Common artifacts include:

- predictions CSVs
- per-target metrics (RMSE/MAE/R2)
- diagnostic plots (scatter, residuals, bar charts)
- feature importances
