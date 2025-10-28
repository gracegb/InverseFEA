#!/usr/bin/env python3
"""Iterative chained prediction with XGBoost regressors.

This script trains three XGBoostRegressor models that are chained together
(Part3_E ➜ Part11_E ➜ Part1_E). After an initial pass that uses pure feature
inputs, the script re-feeds the predicted E values through the chain for a
configurable number of refinement iterations.

Usage example:
    python ml_chained_xgboost_loop.py \
        --data datasets/noisy_with_pca_from_clean_colored.csv \
        --output noisy_data_tests/outputs/xgb_loop \
        --iterations 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


FEATURE_COLS: List[str] = [
    "PC1_InnerBase", "PC2_InnerBase", "PC3_InnerBase",
    "PC1_OuterBase", "PC2_OuterBase", "PC3_OuterBase",
    "PC1_InnerCircle", "PC2_InnerCircle", "PC3_InnerCircle",
    "PC1_MiddleCircle", "PC2_MiddleCircle", "PC3_MiddleCircle",
    "PC1_OuterCircle", "PC2_OuterCircle", "PC3_OuterCircle",
]

TARGET_COLS: List[str] = ["Part1_E", "Part3_E", "Part11_E"]

CHAIN_ORDER: Tuple[str, ...] = ("Part3_E", "Part11_E", "Part1_E")


DEFAULT_XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {label}: {missing}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_initial_models(
    X: pd.DataFrame,
    y: pd.DataFrame,
    xgb_params: Dict,
) -> Dict[str, XGBRegressor]:
    """Train independent models without chained features."""
    models: Dict[str, XGBRegressor] = {}
    for target in TARGET_COLS:
        model = XGBRegressor(**xgb_params)
        model.set_params(objective="reg:squarederror", eval_metric="rmse")
        model.fit(X, y[target])
        models[target] = model
    return models


def train_chain_models(
    X: pd.DataFrame,
    y: pd.DataFrame,
    xgb_params: Dict,
) -> Dict[str, XGBRegressor]:
    """Train models that expect the other E predictions as additional features."""
    models: Dict[str, XGBRegressor] = {}
    for target in TARGET_COLS:
        extra_cols = [col for col in TARGET_COLS if col != target]
        X_chain = X.copy()
        for col in extra_cols:
            X_chain[f"{col}_feature"] = y[col]
        model = XGBRegressor(**xgb_params)
        model.set_params(objective="reg:squarederror", eval_metric="rmse")
        model.fit(X_chain, y[target])
        models[target] = model
    return models


def chained_iteration(
    chain_models: Dict[str, XGBRegressor],
    X_base: pd.DataFrame,
    current_preds: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Run a single chained prediction iteration."""
    updated: Dict[str, np.ndarray] = {}

    # Part3_E uses Part1_E + Part11_E from the *previous* predictions.
    X_part3 = X_base.copy()
    X_part3["Part1_E_feature"] = current_preds["Part1_E"]
    X_part3["Part11_E_feature"] = current_preds["Part11_E"]
    updated["Part3_E"] = chain_models["Part3_E"].predict(X_part3)

    # Part11_E uses the freshly updated Part3_E and previous Part1_E.
    X_part11 = X_base.copy()
    X_part11["Part1_E_feature"] = current_preds["Part1_E"]
    X_part11["Part3_E_feature"] = updated["Part3_E"]
    updated["Part11_E"] = chain_models["Part11_E"].predict(X_part11)

    # Part1_E uses the freshly updated Part3_E + Part11_E.
    X_part1 = X_base.copy()
    X_part1["Part3_E_feature"] = updated["Part3_E"]
    X_part1["Part11_E_feature"] = updated["Part11_E"]
    updated["Part1_E"] = chain_models["Part1_E"].predict(X_part1)

    return updated


def iterative_chained_predictions(
    init_models: Dict[str, XGBRegressor],
    chain_models: Dict[str, XGBRegressor],
    X_test: pd.DataFrame,
    iterations: int,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Run the iterative chained refinement loop."""
    # Initial independent predictions
    current_preds: Dict[str, np.ndarray] = {
        target: init_models[target].predict(X_test)
        for target in TARGET_COLS
    }
    history: List[pd.DataFrame] = [
        pd.DataFrame({t: current_preds[t] for t in TARGET_COLS}, index=X_test.index)
    ]

    for _ in range(iterations):
        current_preds = chained_iteration(chain_models, X_test, current_preds)
        history.append(
            pd.DataFrame({t: current_preds[t] for t in TARGET_COLS}, index=X_test.index)
        )

    return history[-1], history


def evaluate_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLS:
        rows.append(
            dict(
                target=target,
                RMSE=rmse(y_true[target], y_pred[target]),
                MAE=mean_absolute_error(y_true[target], y_pred[target]),
                R2=r2_score(y_true[target], y_pred[target]),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main CLI entry
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative chained XGBoost predictions")
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV dataset")
    parser.add_argument("--output", type=Path, required=True, help="Directory for outputs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--iterations", type=int, default=3, help="Number of chained refinement iterations")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for splitting"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    ensure_columns(df, FEATURE_COLS + TARGET_COLS, label=str(args.data))
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    init_models = train_initial_models(X_train, y_train, DEFAULT_XGB_PARAMS)
    chain_models = train_chain_models(X_train, y_train, DEFAULT_XGB_PARAMS)

    final_preds, history = iterative_chained_predictions(
        init_models, chain_models, X_test, args.iterations
    )

    # Collect metrics for each history snapshot
    metrics_per_iteration = []
    for iteration, preds_df in enumerate(history):
        metrics = evaluate_predictions(y_test, preds_df)
        metrics["iteration"] = iteration
        metrics_per_iteration.append(metrics)
    metrics_df = pd.concat(metrics_per_iteration, ignore_index=True)

    args.output.mkdir(parents=True, exist_ok=True)
    final_preds.to_csv(args.output / "final_predictions.csv", index=False)
    y_test.reset_index(drop=True).to_csv(args.output / "y_test.csv", index=False)
    metrics_df.to_csv(args.output / "metrics_by_iteration.csv", index=False)

    print("Saved final predictions, test targets, and metrics to:")
    print(args.output.resolve())
    print("\nFinal iteration metrics:")
    print(metrics_df[metrics_df["iteration"] == args.iterations])


if __name__ == "__main__":
    main()
