# Noisy data model experiments

This folder contains several modeling scripts that explore training on the PCA-annotated noisy dataset. Use the tables below to quickly understand what each script does and when it was last updated.

## Top-level scripts

| Script | Purpose | Last updated |
| --- | --- | --- |
| `baseline_rf_pca.py` | Baseline RandomForest that predicts all three elastic moduli directly from PCA geometry features with shuffled K-Fold CV and feature-importance export. | 2025-11-05 |
| `ml_chained_noisy_from_clean.py` | Trains chained XGBoost regressors on clean train/test splits but generates predictions for noisy-from-clean data. | 2025-10-28 |
| `ml_chained_noisy_UPDATED.py` | Extended chained XGBoost workflow that trains on clean, noisy-from-clean, and noisy-from-noisy splits. | 2025-10-24 |

## Loop-based chained experiments (`loop_tests/`)

These scripts iterate chained XGBoost predictors to refine predictions by feeding previous outputs back into the chain.

| Script | Chain order / refinement strategy | Last updated |
| --- | --- | --- |
| `ml_chained_xgboost_loop.py` | Forward chain (Part3_E → Part11_E → Part1_E) with configurable refinement iterations on held-out test splits. | 2025-10-28 |
| `ml_chained_xgboost_loop_reversed.py` | Reverse chain (Part1_E → Part11_E → Part3_E) to compare ordering effects under the same looping strategy. | 2025-10-28 |
| `ml_chained_xgboost_sequential_feedback.py` | Forward chain that normalizes predictions before feeding them into subsequent stages, reflecting the latest normalization tweaks. | 2025-11-05 |

## Outputs

Generated predictions and metrics are written to `noisy_data_tests/outputs/` by default. Individual scripts accept `--output` flags to redirect results if needed.
