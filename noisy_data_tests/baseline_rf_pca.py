#!/usr/bin/env python3
"""Baseline Random Forest predictor for inverse FEA E values.

This script trains an unchained :class:`~sklearn.ensemble.RandomForestRegressor`
using the PCA-derived geometric features as inputs and directly predicts the
three target elastic moduli (``Part1_E``, ``Part3_E``, ``Part11_E``).

The default behaviour performs shuffled K-Fold cross-validation, writes the
out-of-fold predictions and evaluation metrics to disk, and prints a concise
summary to stdout. No sequential/chained prediction of targets is performed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

TARGET_COLS: List[str] = ["Part1_E", "Part3_E", "Part11_E"]

FEATURE_COLS: List[str] = [
    "PC1_InnerBase",
    "PC2_InnerBase",
    "PC3_InnerBase",
    "PC1_OuterBase",
    "PC2_OuterBase",
    "PC3_OuterBase",
    "PC1_InnerCircle",
    "PC2_InnerCircle",
    "PC3_InnerCircle",
    "PC1_MiddleCircle",
    "PC2_MiddleCircle",
    "PC3_MiddleCircle",
    "PC1_OuterCircle",
    "PC2_OuterCircle",
    "PC3_OuterCircle",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    """Validate that ``required`` columns exist in ``df``.

    Parameters
    ----------
    df:
        The dataframe to validate.
    required:
        Column names that must be present.
    label:
        Human-readable label used in the raised ``KeyError`` message.
    """

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {label}: {missing}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Core CV routine
# ---------------------------------------------------------------------------


@dataclass
class BaselineConfig:
    csv_path: Path
    output_dir: Path
    n_splits: int
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    random_state: int


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the CSV file using UTF-8 BOM-safe decoding."""

    return pd.read_csv(csv_path, encoding="utf-8-sig")


def build_model(config: BaselineConfig) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
    )


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: BaselineConfig,
) -> Dict[str, pd.DataFrame]:
    """Perform shuffled K-Fold CV and collect predictions + metrics."""

    kf = KFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    oof_predictions = pd.DataFrame(index=y.index, columns=TARGET_COLS, dtype=float)
    fold_metric_records: List[Dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        model = build_model(config)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        fold_pred = pd.DataFrame(
            model.predict(X.iloc[test_idx]),
            index=y.index[test_idx],
            columns=TARGET_COLS,
        )

        oof_predictions.loc[fold_pred.index, :] = fold_pred

        for target in TARGET_COLS:
            y_true = y.iloc[test_idx][target]
            y_pred = fold_pred[target]
            fold_metric_records.append(
                {
                    "fold": fold_idx,
                    "target": target,
                    "rmse": rmse(y_true, y_pred),
                    "mae": mae(y_true, y_pred),
                    "r2": r2(y_true, y_pred),
                }
            )

    fold_metrics_df = pd.DataFrame(fold_metric_records)
    mean_metrics_df = (
        fold_metrics_df.groupby("target")[["rmse", "mae", "r2"]].mean().reset_index()
    )

    return {
        "predictions": oof_predictions,
        "fold_metrics": fold_metrics_df,
        "mean_metrics": mean_metrics_df,
    }


def train_full_model(X: pd.DataFrame, y: pd.DataFrame, config: BaselineConfig) -> RandomForestRegressor:
    model = build_model(config)
    model.fit(X, y)
    return model


def save_outputs(
    results: Dict[str, pd.DataFrame],
    X: pd.DataFrame,
    y: pd.DataFrame,
    model: RandomForestRegressor,
    config: BaselineConfig,
) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = results["predictions"].copy()
    predictions.columns = [f"{col}_pred" for col in predictions.columns]

    combined = pd.concat([y, predictions], axis=1)
    combined.to_csv(output_dir / "baseline_predictions.csv", index=False)

    results["fold_metrics"].to_csv(output_dir / "fold_metrics.csv", index=False)
    results["mean_metrics"].to_csv(output_dir / "mean_metrics.csv", index=False)

    feature_importances = pd.Series(
        model.feature_importances_, index=X.columns, name="feature_importance"
    )
    feature_importances.sort_values(ascending=False).to_csv(
        output_dir / "feature_importances.csv"
    )


def print_summary(mean_metrics: pd.DataFrame) -> None:
    print("Baseline Random Forest cross-validation metrics (averaged):")
    print(mean_metrics.to_string(index=False, float_format="{:.6f}".format))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(
        description="Run an unchained RandomForestRegressor baseline on PCA features.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("../datasets/noisy_with_pca_from_clean_colored.csv"),
        help="Path to the dataset CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("noisy_data_tests/outputs/baseline_rf"),
        help="Directory where predictions and metrics will be written.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees in the random forest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth (None lets trees expand fully).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples required to be at a leaf node.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    return BaselineConfig(
        csv_path=args.csv,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )


def main() -> None:
    config = parse_args()

    df = load_dataset(config.csv_path)

    ensure_columns(df, TARGET_COLS, "target data")
    ensure_columns(df, FEATURE_COLS, "feature data")

    # === Apply Min-Max normalization to features ===
    from sklearn.preprocessing import MinMaxScaler

    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    # Optional: also normalize target values if their magnitudes differ widely
    # y_scaler = MinMaxScaler(feature_range=(0, 1))
    # y_scaled = pd.DataFrame(y_scaler.fit_transform(y), columns=TARGET_COLS)
    # Use y_scaled below instead of y if you enable this

    results = run_cross_validation(X_scaled, y, config)
    final_model = train_full_model(X_scaled, y, config)
    save_outputs(results, X_scaled, y, final_model, config)
    print_summary(results["mean_metrics"])


if __name__ == "__main__":
    main()

