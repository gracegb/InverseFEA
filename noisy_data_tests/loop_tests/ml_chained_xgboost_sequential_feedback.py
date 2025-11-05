#!/usr/bin/env python3
"""Sequential-feedback chained XGBoost prediction with feature alignment fix."""
""" -----> Updated version with normalization"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    "PC1_InnerBase", "PC2_InnerBase", "PC3_InnerBase",
    "PC1_OuterBase", "PC2_OuterBase", "PC3_OuterBase",
    "PC1_InnerCircle", "PC2_InnerCircle", "PC3_InnerCircle",
    "PC1_MiddleCircle", "PC2_MiddleCircle", "PC3_MiddleCircle",
    "PC1_OuterCircle", "PC2_OuterCircle", "PC3_OuterCircle",
]
TARGET_COLS: List[str] = ["Part1_E", "Part3_E", "Part11_E"]

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

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {label}: {missing}")

def rmse(y_true, y_pred): 
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train_model(X, y, params) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.set_params(objective="reg:squarederror", eval_metric="rmse")
    model.fit(X, y)
    return model

def align_features(model: XGBRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Ensure X has identical feature names/order as model."""
    cols = model.get_booster().feature_names
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    return X[cols]

# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_all_models(X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, XGBRegressor]:
    """Train models with all potential chain features, filled with zeros."""
    models: Dict[str, XGBRegressor] = {}

    def add_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0
        return out

    # Part3_E
    X3 = add_features(X, ["Part1_E_feature", "Part11_E_feature"])
    models["Part3_E"] = train_model(X3, y["Part3_E"], DEFAULT_XGB_PARAMS)

    # Part11_E
    X11 = add_features(X, ["Part1_E_feature", "Part3_E_feature"])
    X11["Part3_E_feature"] = y["Part3_E"]
    models["Part11_E"] = train_model(X11, y["Part11_E"], DEFAULT_XGB_PARAMS)

    # Part1_E
    X1 = add_features(X, ["Part3_E_feature", "Part11_E_feature"])
    X1["Part3_E_feature"] = y["Part3_E"]
    X1["Part11_E_feature"] = y["Part11_E"]
    models["Part1_E"] = train_model(X1, y["Part1_E"], DEFAULT_XGB_PARAMS)

    return models

# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------

def _ensure_feature_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out

def initial_predictions(models, X_test):
    preds = {}

    # --- Part3_E ---
    X3 = _ensure_feature_cols(X_test, ["Part1_E_feature", "Part11_E_feature"])
    X3 = align_features(models["Part3_E"], X3)
    preds["Part3_E"] = models["Part3_E"].predict(X3)

    # --- Part11_E ---
    X11 = _ensure_feature_cols(X_test, ["Part1_E_feature", "Part3_E_feature"])
    X11["Part3_E_feature"] = preds["Part3_E"]
    X11 = align_features(models["Part11_E"], X11)
    preds["Part11_E"] = models["Part11_E"].predict(X11)

    # --- Part1_E ---
    X1 = _ensure_feature_cols(X_test, ["Part3_E_feature", "Part11_E_feature"])
    X1["Part3_E_feature"] = preds["Part3_E"]
    X1["Part11_E_feature"] = preds["Part11_E"]
    X1 = align_features(models["Part1_E"], X1)
    preds["Part1_E"] = models["Part1_E"].predict(X1)

    return preds

def feedback_iteration(models, X_base, prev_preds):
    updated = {}

    # --- Part3_E from X + previous 1,11 ---
    X3 = _ensure_feature_cols(X_base, ["Part1_E_feature", "Part11_E_feature"])
    X3["Part1_E_feature"] = prev_preds["Part1_E"]
    X3["Part11_E_feature"] = prev_preds["Part11_E"]
    X3 = align_features(models["Part3_E"], X3)
    updated["Part3_E"] = models["Part3_E"].predict(X3)

    # --- Part11_E from X + current 3 + previous 1 ---
    X11 = _ensure_feature_cols(X_base, ["Part3_E_feature", "Part1_E_feature"])
    X11["Part3_E_feature"] = updated["Part3_E"]
    X11["Part1_E_feature"] = prev_preds["Part1_E"]
    X11 = align_features(models["Part11_E"], X11)
    updated["Part11_E"] = models["Part11_E"].predict(X11)

    # --- Part1_E from X + current 3,11 ---
    X1 = _ensure_feature_cols(X_base, ["Part3_E_feature", "Part11_E_feature"])
    X1["Part3_E_feature"] = updated["Part3_E"]
    X1["Part11_E_feature"] = updated["Part11_E"]
    X1 = align_features(models["Part1_E"], X1)
    updated["Part1_E"] = models["Part1_E"].predict(X1)

    return updated

def iterative_feedback(models, X_test, iterations):
    preds = initial_predictions(models, X_test)
    history = [pd.DataFrame(preds, index=X_test.index)]

    for _ in range(iterations):
        preds = feedback_iteration(models, X_test, preds)
        history.append(pd.DataFrame(preds, index=X_test.index))

    return history[-1], history

# ---------------------------------------------------------------------
# METRICS + CLI
# ---------------------------------------------------------------------

def evaluate(y_true, y_pred):
    rows = []
    for t in TARGET_COLS:
        rows.append(dict(
            target=t,
            RMSE=rmse(y_true[t], y_pred[t]),
            MAE=mean_absolute_error(y_true[t], y_pred[t]),
            R2=r2_score(y_true[t], y_pred[t]),
        ))
    return pd.DataFrame(rows)

def parse_args():
    p = argparse.ArgumentParser(description="Sequential-feedback chained XGBoost predictions")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    
    ensure_columns(df, FEATURE_COLS + TARGET_COLS, str(args.data))
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    models = train_all_models(X_train, y_train)
    final_preds, history = iterative_feedback(models, X_test, args.iterations)

    metrics_all = []
    for i, preds_df in enumerate(history):
        m = evaluate(y_test, preds_df)
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
