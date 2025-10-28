#!/usr/bin/env python3
"""Iterative chained prediction with reversed order (Part1 ➜ Part11 ➜ Part3).

This script mirrors ml_chained_xgboost_loop.py but reverses the chain direction.
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

# reversed chain: Part1_E ➜ Part11_E ➜ Part3_E
CHAIN_ORDER: Tuple[str, ...] = ("Part1_E", "Part11_E", "Part3_E")

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


def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {label}: {missing}")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_initial_models(X, y, params):
    models = {}
    for target in TARGET_COLS:
        model = XGBRegressor(**params)
        model.set_params(objective="reg:squarederror", eval_metric="rmse")
        model.fit(X, y[target])
        models[target] = model
    return models


def train_chain_models(X, y, params):
    models = {}
    for target in TARGET_COLS:
        extra_cols = [c for c in TARGET_COLS if c != target]
        X_chain = X.copy()
        for c in extra_cols:
            X_chain[f"{c}_feature"] = y[c]
        model = XGBRegressor(**params)
        model.set_params(objective="reg:squarederror", eval_metric="rmse")
        model.fit(X_chain, y[target])
        models[target] = model
    return models


def chained_iteration(chain_models, X_base, current_preds):
    """Run one reversed-order iteration: Part1 → Part11 → Part3"""
    updated = {}

    # Part1_E uses previous Part3_E + Part11_E
    X_part1 = X_base.copy()
    X_part1["Part3_E_feature"] = current_preds["Part3_E"]
    X_part1["Part11_E_feature"] = current_preds["Part11_E"]
    updated["Part1_E"] = chain_models["Part1_E"].predict(X_part1)

    # Part11_E uses freshly updated Part1_E and previous Part3_E
    X_part11 = X_base.copy()
    X_part11["Part1_E_feature"] = updated["Part1_E"]
    X_part11["Part3_E_feature"] = current_preds["Part3_E"]
    updated["Part11_E"] = chain_models["Part11_E"].predict(X_part11)

    # Part3_E uses freshly updated Part1_E + Part11_E
    X_part3 = X_base.copy()
    X_part3["Part1_E_feature"] = updated["Part1_E"]
    X_part3["Part11_E_feature"] = updated["Part11_E"]
    updated["Part3_E"] = chain_models["Part3_E"].predict(X_part3)

    return updated


def iterative_chained_predictions(init_models, chain_models, X_test, iterations):
    current_preds = {t: init_models[t].predict(X_test) for t in TARGET_COLS}
    history = [pd.DataFrame(current_preds, index=X_test.index)]

    for _ in range(iterations):
        current_preds = chained_iteration(chain_models, X_test, current_preds)
        history.append(pd.DataFrame(current_preds, index=X_test.index))

    return history[-1], history


def evaluate_predictions(y_true, y_pred):
    out = []
    for t in TARGET_COLS:
        out.append(dict(
            target=t,
            RMSE=rmse(y_true[t], y_pred[t]),
            MAE=mean_absolute_error(y_true[t], y_pred[t]),
            R2=r2_score(y_true[t], y_pred[t]),
        ))
    return pd.DataFrame(out)


def parse_args():
    p = argparse.ArgumentParser(description="Iterative reversed chained XGBoost predictions")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data)
    ensure_columns(df, FEATURE_COLS + TARGET_COLS, label=str(args.data))
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    init_models = train_initial_models(X_train, y_train, DEFAULT_XGB_PARAMS)
    chain_models = train_chain_models(X_train, y_train, DEFAULT_XGB_PARAMS)

    final_preds, history = iterative_chained_predictions(
        init_models, chain_models, X_test, args.iterations
    )

    metrics_all = []
    for i, preds_df in enumerate(history):
        m = evaluate_predictions(y_test, preds_df)
        m["iteration"] = i
        metrics_all.append(m)
    metrics_df = pd.concat(metrics_all, ignore_index=True)

    args.output.mkdir(parents=True, exist_ok=True)
    final_preds.to_csv(args.output / "final_predictions.csv", index=False)
    y_test.reset_index(drop=True).to_csv(args.output / "y_test.csv", index=False)
    metrics_df.to_csv(args.output / "metrics_by_iteration.csv", index=False)

    print("Saved outputs to:", args.output.resolve())
    print("\nFinal iteration metrics:")
    print(metrics_df[metrics_df["iteration"] == args.iterations])


if __name__ == "__main__":
    main()
