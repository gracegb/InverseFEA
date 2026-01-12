# Loop-based chained XGBoost experiments

These scripts iteratively refine chained predictions by feeding previously
predicted E values back into the chain.

| Script | Chain order | Notes |
| --- | --- | --- |
| `ml_chained_xgboost_loop.py` | Part3_E -> Part11_E -> Part1_E | Forward chain with configurable iterations. |
| `ml_chained_xgboost_loop_reversed.py` | Part1_E -> Part11_E -> Part3_E | Reverse chain for ordering comparison. |
| `ml_chained_xgboost_sequential_feedback.py` | Part3_E -> Part11_E -> Part1_E | Adds normalization tweaks between iterations. |

All scripts take `--data` and `--output` flags; run them from the repo root to
keep relative paths consistent.
