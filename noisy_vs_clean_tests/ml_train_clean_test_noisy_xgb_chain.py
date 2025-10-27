#!/usr/bin/env python3
"""Simple chained XGBoost regressor workflow.

This script mirrors the classic chaining approach (Part3_E → Part11_E → Part1_E)
while swapping the Random Forest models for ``XGBRegressor`` models.  Clean
measurements are used for training and the corresponding noisy measurements are
used for testing.  A few lightweight plots are generated to keep the workflow
familiar with previous scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default CSVs – they match the names used by the previous scripts and now
# contain the expanded feature set.
CLEAN_DATA = Path("../noisy_data_tests/original_with_pca_colored-2.csv")
NOISY_DATA = Path("../noisy_data_tests/noisy_with_pca_from_clean_colored.csv")

# Feature columns (same naming as the updated data files).
FEATURE_COLUMNS = [
    "PC1_InnerBase", "PC2_InnerBase", "PC3_InnerBase",
    "PC1_OuterBase", "PC2_OuterBase", "PC3_OuterBase",
    "PC1_InnerCircle", "PC2_InnerCircle", "PC3_InnerCircle",
    "PC1_MiddleCircle", "PC2_MiddleCircle", "PC3_MiddleCircle",
    "PC1_OuterCircle", "PC2_OuterCircle", "PC3_OuterCircle",
]

# Target columns stay unchanged.
TARGET_COLUMNS = ["Part1_E", "Part3_E", "Part11_E"]

# Model hyper-parameters kept intentionally simple.
XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

# Output directory for plots and CSVs.
OUTPUT_DIR = Path("outputs_clean_train_noisy_test_xgb")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load clean (train) and noisy (test) data."""
    train_df = pd.read_csv(CLEAN_DATA)
    test_df = pd.read_csv(NOISY_DATA)

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMNS].copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMNS].copy()

    return X_train, y_train, X_test, y_test


def train_chained_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[Dict[str, XGBRegressor], pd.DataFrame]:
    """Train the three chained XGBRegressor models and return predictions."""
    models: Dict[str, XGBRegressor] = {}

    # Step 1 – predict Part3_E directly from the base features.
    model_part3 = XGBRegressor(**XGB_PARAMS)
    model_part3.fit(X_train, y_train["Part3_E"])
    models["Part3_E"] = model_part3
    part3_pred = model_part3.predict(X_test)

    # Step 2 – predict Part11_E using the base features + Part3_E prediction.
    X_train_with_3 = X_train.copy()
    X_train_with_3["Part3_E_pred"] = y_train["Part3_E"]

    X_test_with_3 = X_test.copy()
    X_test_with_3["Part3_E_pred"] = part3_pred

    model_part11 = XGBRegressor(**XGB_PARAMS)
    model_part11.fit(X_train_with_3, y_train["Part11_E"])
    models["Part11_E"] = model_part11
    part11_pred = model_part11.predict(X_test_with_3)

    # Step 3 – predict Part1_E using base features + both intermediate predictions.
    X_train_with_3_11 = X_train_with_3.copy()
    X_train_with_3_11["Part11_E_pred"] = y_train["Part11_E"]

    X_test_with_3_11 = X_test_with_3.copy()
    X_test_with_3_11["Part11_E_pred"] = part11_pred

    model_part1 = XGBRegressor(**XGB_PARAMS)
    model_part1.fit(X_train_with_3_11, y_train["Part1_E"])
    models["Part1_E"] = model_part1
    part1_pred = model_part1.predict(X_test_with_3_11)

    predictions = pd.DataFrame({
        "Part1_E": part1_pred,
        "Part3_E": part3_pred,
        "Part11_E": part11_pred,
    }, index=X_test.index)

    return models, predictions


def compute_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Calculate MAE, RMSE, and R² for each target."""
    rows = []
    for target in TARGET_COLUMNS:
        true_vals = y_true[target]
        pred_vals = y_pred[target]
        rows.append({
            "Target": target,
            "MAE": mean_absolute_error(true_vals, pred_vals),
            "RMSE": mean_squared_error(true_vals, pred_vals, squared=False),
            "R2": r2_score(true_vals, pred_vals),
        })
    return pd.DataFrame(rows).set_index("Target")


def plot_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame, out_dir: Path) -> None:
    """Generate true-vs-predicted and residual plots for each target."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # True vs predicted scatter plots.
    for target in TARGET_COLUMNS:
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true[target], y_pred[target], s=18, alpha=0.75, edgecolor="k")
        min_val = min(y_true[target].min(), y_pred[target].min())
        max_val = max(y_true[target].max(), y_pred[target].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{target} – Clean train → Noisy test")
        plt.tight_layout()
        plt.savefig(out_dir / f"{target}_scatter.png", dpi=200)
        plt.close()

    # Residual plots (predicted vs residuals).
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, target in zip(axes, TARGET_COLUMNS):
        residuals = y_true[target] - y_pred[target]
        ax.scatter(y_pred[target], residuals, s=18, alpha=0.75, edgecolor="k")
        ax.axhline(0.0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title(f"Residuals – {target}")
    fig.suptitle("Residual analysis (clean train → noisy test)", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_dir / "residuals.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clean (training) and noisy (test) datasets...")
    X_train, y_train, X_test, y_test = load_split_data()
    print(f"Training samples: {len(X_train):d}; Testing samples: {len(X_test):d}")

    print("\nTraining chained XGBoost regressors...")
    _, y_pred = train_chained_models(X_train, y_train, X_test)
    print("Training complete.\n")

    metrics = compute_metrics(y_test, y_pred)
    print("Evaluation metrics (clean → noisy):")
    print(metrics)

    metrics.to_csv(OUTPUT_DIR / "metrics.csv")
    y_pred.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    print("\nGenerating plots...")
    plot_predictions(y_test, y_pred, OUTPUT_DIR)
    print(f"All done! Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
