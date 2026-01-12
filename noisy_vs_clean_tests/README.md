# Clean-train / noisy-test experiments

Scripts in this folder train on the clean PCA dataset and evaluate on noisy
data to measure distribution shift sensitivity.

## Scripts

| Script | Purpose | Outputs |
| --- | --- | --- |
| `ml_train_clean_test_noisy.py` | Chained Random Forest models (Part3_E -> Part11_E -> Part1_E) with standard scaling and diagnostics. | Predictions CSV, metrics CSV, scatter plots (see `outputs/`). |
| `ml_train_clean_test_noisy_centered.py` | Same chained RF workflow with feature mean-centering to correct PCA mean shift. | Predictions, metrics, and plots (see `outputs/`). |
| `ml_train_clean_test_noisy_xgb_chain.py` | Chained XGBoost workflow with mean-centered features and lightweight plots. | Predictions + plots (see `outputs/`). |

## Notes

- Default input paths live in `datasets/` and can be edited in each script.
- These scripts assume the current PCA schema with the 15 PCA feature columns.
- Prefer writing outputs into `noisy_vs_clean_tests/outputs/` (subfolders per run).
