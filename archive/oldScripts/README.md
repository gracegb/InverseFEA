# Legacy scripts

This folder contains older experiments kept for reference. Most of these files
assume the legacy dataset `datasets/old_data/original_with_pca.csv` and its
older PCA feature names (`PC1_Bottom`, `PC1_InnerShape`, `PC1_OuterShape`, etc).

## What is here

- Chained Random Forest and XGBoost experiments with multi-output CV.
- Permutation-importance plots and residual diagnostics.
- Early attempts at scaling or normalization before the newer PCA schema.

## Notes

- These scripts generally expect to be run from their own directory with
  `original_with_pca.csv` in the working directory. Update paths if you run
  them from the repo root.
- Outputs typically go to `archive/oldScripts/plots/` and may overwrite previous runs.
