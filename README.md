# Gordon inverseFEA

Inverse FEA modeling experiments for predicting elastic moduli from PCA-based
geometry features. The core workflow learns mappings from PCA summaries of
ring/base geometry to target elastic moduli (Part1_E, Part3_E, Part11_E), with
multiple modeling variants (baseline, chained, looped, and experimental).

## Project layout

| Path | Contents |
| --- | --- |
| `datasets/` | Main CSVs (clean + noisy PCA data), plus legacy data in `datasets/old_data/`. |
| `models/` | Saved `.pkl` models and prediction artifacts from prior runs. |
| `noisy_data_tests/` | Primary experiments that train on noisy PCA datasets. |
| `noisy_vs_clean_tests/` | Experiments that train on clean data and test on noisy data. |
| `archive/oldScripts/` | Historical experiments kept for reference, built on older data schemas. |

## Data overview

### Current PCA datasets (114 columns each)

These CSVs share the same schema and are the main inputs for the current
experiments:

- `datasets/original_with_pca_colored-2.csv` (clean)
- `datasets/noisy_with_pca_from_clean_colored.csv` (noisy-from-clean)
- `datasets/noisy_with_pca_from_noisy_colored.csv` (noisy-from-noisy)

Each file includes:

- **Identifiers**: `File Name`
- **Targets**: `Part1_E`, `Part3_E`, `Part7_E`, `Part10_E`, `Part11_E`
- **Scalar geometry**: `Pressure`, `Inner_Radius`, `Outer_Radius`
- **Raw geometry samples**:
  - `inner_base_y1..y9`, `inner_base_z1..z9`
  - `outer_base_y1..y9`, `outer_base_z1..z9`
  - `inner_circle_x1..x9`, `inner_circle_y1..y9`
  - `outer_circle_x1..x9`, `outer_circle_y1..y9`
  - `outer_outer_circle_x1..x9`, `outer_outer_circle_y1..y9`
- **PCA features used by most scripts (15 total)**:
  - `PC1_InnerBase`, `PC2_InnerBase`, `PC3_InnerBase`
  - `PC1_OuterBase`, `PC2_OuterBase`, `PC3_OuterBase`
  - `PC1_InnerCircle`, `PC2_InnerCircle`, `PC3_InnerCircle`
  - `PC1_MiddleCircle`, `PC2_MiddleCircle`, `PC3_MiddleCircle`
  - `PC1_OuterCircle`, `PC2_OuterCircle`, `PC3_OuterCircle`

Note: These CSVs include a UTF-8 BOM in the header; use `encoding="utf-8-sig"`
when reading with pandas.

### Legacy PCA dataset (90 columns)

`datasets/old_data/original_with_pca.csv` uses older feature naming:

- **PCA features**: `PC1_Bottom/InnerShape/OuterShape` (+ PC2/PC3)
- **Raw geometry samples**: `inner_y*`, `inner_z*`, `outer_y*`, `outer_z*`,
  `innerShape_*`, `outerShape_*`

Older scripts under `archive/oldScripts/` are wired to this schema.

## Modeling workflows

### Baseline and noisy-data workflows (`noisy_data_tests/`)

- Baseline Random Forest (direct multi-target prediction)
- Chained Random Forest and XGBoost pipelines
- Loop-based chained XGBoost refinement
- Experimental OOF stacking + CORAL alignment + quantile modeling for Part1_E

See `noisy_data_tests/README.md` for script-level details.

### Clean-train / noisy-test workflows (`noisy_vs_clean_tests/`)

Chained Random Forest and XGBoost scripts that train on the clean dataset and
evaluate on noisy data (optionally mean-centered). See
`noisy_vs_clean_tests/README.md`.

### Legacy workflows (`archive/oldScripts/`)

Older chained RF/XGBoost experiments built on the legacy data schema. These are
kept for reference and may require path updates. See `archive/oldScripts/README.md`.

## Running experiments

Most scripts are standalone Python files. Some accept CLI flags and paths, while
others rely on defaults embedded in the script. Run from the repo root to keep
relative paths consistent.

Examples:

```bash
python3 noisy_data_tests/baseline_rf_pca.py \
  --csv datasets/noisy_with_pca_from_clean_colored.csv \
  --output-dir noisy_data_tests/outputs/baseline_rf
```

```bash
python3 noisy_data_tests/loop_tests/ml_chained_xgboost_loop.py \
  --data datasets/noisy_with_pca_from_clean_colored.csv \
  --output noisy_data_tests/outputs/xgb_loop \
  --iterations 3
```

```bash
python3 noisy_vs_clean_tests/ml_train_clean_test_noisy_xgb_chain.py
```

If a script expects files in a different location, update the paths or pass
`--csv` / `--data` flags where available.

## Dependencies

There is no pinned environment in this repo. Most scripts expect:

- Python 3
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- `xgboost` (for XGBoost-based workflows)
- `joblib` (some legacy scripts save models)

## Outputs and artifacts

- `noisy_data_tests/outputs/` and `noisy_vs_clean_tests/outputs_*` store
  predictions, metrics, and plots from recent runs.
- `models/` contains saved `.pkl` model files and CSV predictions from earlier
  experiments. See `models/README.md` for details.

## Where to start

1. Review `datasets/README.md` for data schema details.
2. Run `noisy_data_tests/baseline_rf_pca.py` to establish a baseline.
3. Explore chained and looped models in `noisy_data_tests/`.
