
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chained Random Forest Regressors with normalization, K-Fold CV,
and OLD vs NOISY dataset comparison for inverse FEA (E values).
- Model A (RF_3): Part3_E ~ X
- Model B (RF_11): Part11_E ~ X + Part3_E_pred
- Model C (RF_1): Part1_E ~ X + Part3_E_pred + Part11_E_pred

Outputs:
- Per-dataset predictions CSV (per fold & aggregated)
- Metrics (per fold, per target, normalized & original scale)
- Scatter plots: y_true vs y_pred per target (old/noisy)
- Residual histograms per target (old/noisy)
- Comparison bar charts (RMSE & R2) old vs noisy per target
- Simple permutation importances (averaged across folds) for A, B, C
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance

import time
from datetime import timedelta



# ----------------------- Configuration ----------------------------

DEFAULT_ORIGINAL_FILE = "original_with_pca_colored-2.csv"
DEFAULT_NOISY_FILE    = "noisy_with_pca_from_clean_colored.csv"

TARGET_COLS = ["Part1_E", "Part3_E", "Part11_E"]

FEATURE_COLS = [
    "PC1_InnerBase", "PC2_InnerBase", "PC3_InnerBase",
    "PC1_OuterBase", "PC2_OuterBase", "PC3_OuterBase",
    "PC1_InnerCircle", "PC2_InnerCircle", "PC3_InnerCircle",
    "PC1_MiddleCircle", "PC2_MiddleCircle", "PC3_MiddleCircle",
    "PC1_OuterCircle", "PC2_OuterCircle", "PC3_OuterCircle",
]

RANDOM_STATE = 42
N_SPLITS = 5

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)


# ----------------------- Utilities --------------------------------

def ensure_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {label}: {missing}")


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------- Core CV Logic ----------------------------

def chained_fold_train_predict(
    X_train: pd.DataFrame, y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[Dict[str, RandomForestRegressor], pd.DataFrame]:
    """
    Train chained RF models on the training split and predict on the test split.
    Training uses ground-truth chaining on the training set for simplicity and
    consistent comparison across datasets (note: this can be slightly optimistic).
    Test-time uses predicted chains.

    Returns:
        models: dict of trained models {"RF_3": ..., "RF_11": ..., "RF_1": ...}
        y_pred_test_df: DataFrame with columns TARGET_COLS containing test predictions.
    """
    models = {}

    # A) RF_3: Part3_E ~ X
    rf3 = RandomForestRegressor(**RF_PARAMS)
    rf3.fit(X_train, y_train["Part3_E"])
    models["RF_3"] = rf3

    # B) RF_11: Part11_E ~ X + Part3_E_pred (train uses true Part3_E; test uses predicted)
    # Train features for B
    X_train_B = X_train.copy()
    X_train_B["Part3_E_pred"] = y_train["Part3_E"]  # ground-truth for training
    rf11 = RandomForestRegressor(**RF_PARAMS)
    rf11.fit(X_train_B, y_train["Part11_E"])
    models["RF_11"] = rf11

    # C) RF_1: Part1_E ~ X + Part3_E_pred + Part11_E_pred (train uses true Part3_E/Part11_E; test uses predicted)
    X_train_C = X_train.copy()
    X_train_C["Part3_E_pred"] = y_train["Part3_E"]
    X_train_C["Part11_E_pred"] = y_train["Part11_E"]
    rf1 = RandomForestRegressor(**RF_PARAMS)
    rf1.fit(X_train_C, y_train["Part1_E"])
    models["RF_1"] = rf1

    # --- Test-time predictions chained ---
    y3_hat = rf3.predict(X_test)
    X_test_B = X_test.copy()
    X_test_B["Part3_E_pred"] = y3_hat
    y11_hat = rf11.predict(X_test_B)

    X_test_C = X_test.copy()
    X_test_C["Part3_E_pred"] = y3_hat
    X_test_C["Part11_E_pred"] = y11_hat
    y1_hat = rf1.predict(X_test_C)

    y_pred_test = pd.DataFrame({
        "Part3_E": y3_hat,
        "Part11_E": y11_hat,
        "Part1_E": y1_hat,
    }, index=X_test.index)[["Part1_E", "Part3_E", "Part11_E"]]  # keep target order

    return models, y_pred_test


def run_chained_cv(
    df: pd.DataFrame,
    dataset_name: str,
    out_dir: Path,
    scale_targets: bool = True,   # <-- new flag
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, pd.Series]]:
    """
    Runs KFold CV with target normalization. Saves plots and returns:
    - oof_predictions (original scale)
    - fold_metrics_df (per-fold metrics in original scale)
    - mean_metrics (aggregate metrics across folds per target)
    - avg_importances (averaged permutation importances per model)
    """
    start_time = time.time()
    print(f"\n🧩 Running {dataset_name.upper()} dataset ({N_SPLITS}-fold CV, scale_targets={scale_targets}) ...")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = pd.DataFrame(index=df.index, columns=TARGET_COLS, dtype=float)
    fold_metrics_rows = []

    # Track permutation importances per fold per model
    importances_sum = {"RF_3": None, "RF_11": None, "RF_1": None}
    counts = {"RF_3": 0, "RF_11": 0, "RF_1": 0}

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):

        fold_start = time.time()
        print(f"  ⏳ Fold {fold}/{N_SPLITS} started ...", end=" ")


        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()

        # Optionally normalize targets per-fold using train only
        if scale_targets:
            y_scaler = MinMaxScaler()
            y_tr_proc = pd.DataFrame(y_scaler.fit_transform(y_tr), columns=TARGET_COLS, index=y_tr.index)
            y_te_proc = pd.DataFrame(y_scaler.transform(y_te), columns=TARGET_COLS, index=y_te.index)
        else:
            y_scaler = None
            y_tr_proc, y_te_proc = y_tr.copy(), y_te.copy()

        # Chain on processed targets (scaled or raw)
        models, y_pred_te_proc = chained_fold_train_predict(X_tr, y_tr_proc, X_te)

        # If scaled, inverse transform predictions back to original scale
        if scale_targets:
            y_pred_te = pd.DataFrame(
                y_scaler.inverse_transform(y_pred_te_proc[TARGET_COLS]),
                columns=TARGET_COLS,
                index=y_pred_te_proc.index,
            )
        else:
            y_pred_te = y_pred_te_proc.copy()

        oof_pred.loc[te_idx, TARGET_COLS] = y_pred_te[TARGET_COLS]

        # --- Metrics per target (original scale) ---
        for tgt in TARGET_COLS:
            row = {
                "dataset": dataset_name,
                "fold": fold,
                "target": tgt,
                "RMSE": rmse(y_te[tgt], y_pred_te[tgt]),
                "MAE": mean_absolute_error(y_te[tgt], y_pred_te[tgt]),
                "R2": r2_score(y_te[tgt], y_pred_te[tgt]),
            }
            elapsed = time.time() - fold_start
            avg_per_fold = (time.time() - start_time) / fold
            eta = avg_per_fold * (N_SPLITS - fold)
            print(f"done in {elapsed:.1f}s. ETA: {timedelta(seconds=int(eta))} remaining.")

            fold_metrics_rows.append(row)

        # --- Simple permutation importance on test split (original feature spaces) ---
        # For RF_3: features are X_te
        try:
            r = permutation_importance(models["RF_3"], X_te, y_tr_scaled["Part3_E"].mean() + 0*X_te.index, n_repeats=5, random_state=RANDOM_STATE)
        except Exception:
            r = None
        if r is not None:
            s = pd.Series(r.importances_mean, index=X_te.columns)
            importances_sum["RF_3"] = (s if importances_sum["RF_3"] is None else importances_sum["RF_3"].add(s, fill_value=0))
            counts["RF_3"] += 1

        # For RF_11: features are X + Part3_E_pred
        X_te_B = X_te.copy()
        X_te_B["Part3_E_pred"] = models["RF_3"].predict(X_te)
        try:
            r = permutation_importance(models["RF_11"], X_te_B, y_tr_scaled["Part11_E"].mean() + 0*X_te_B.index, n_repeats=5, random_state=RANDOM_STATE)
        except Exception:
            r = None
        if r is not None:
            s = pd.Series(r.importances_mean, index=X_te_B.columns)
            importances_sum["RF_11"] = (s if importances_sum["RF_11"] is None else importances_sum["RF_11"].add(s, fill_value=0))
            counts["RF_11"] += 1

        # For RF_1: features are X + Part3_E_pred + Part11_E_pred
        X_te_C = X_te_B.copy()
        X_te_C["Part11_E_pred"] = models["RF_11"].predict(X_te_B)
        try:
            r = permutation_importance(models["RF_1"], X_te_C, y_tr_scaled["Part1_E"].mean() + 0*X_te_C.index, n_repeats=5, random_state=RANDOM_STATE)
        except Exception:
            r = None
        if r is not None:
            s = pd.Series(r.importances_mean, index=X_te_C.columns)
            importances_sum["RF_1"] = (s if importances_sum["RF_1"] is None else importances_sum["RF_1"].add(s, fill_value=0))
            counts["RF_1"] += 1

    fold_metrics_df = pd.DataFrame(fold_metrics_rows)

    # Aggregate metrics across folds
    mean_metrics = (
        fold_metrics_df.groupby(["dataset", "target"])[["RMSE", "MAE", "R2"]]
        .mean()
        .reset_index()
        .set_index("target")
        .drop(columns=["dataset"])
        .to_dict(orient="index")
    )

    # Average permutation importances
    avg_importances = {}
    for k in ["RF_3", "RF_11", "RF_1"]:
        if counts[k] > 0 and importances_sum[k] is not None:
            avg_importances[k] = importances_sum[k] / counts[k]
        else:
            avg_importances[k] = pd.Series(dtype=float)

    # Save OOF predictions and fold metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_pred.to_csv(out_dir / f"{dataset_name}_oof_predictions.csv", index=True)
    fold_metrics_df.to_csv(out_dir / f"{dataset_name}_fold_metrics.csv", index=False)

    # Plots: by target
    for tgt in TARGET_COLS:
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(df.loc[oof_pred.index, tgt], oof_pred[tgt], s=12, alpha=0.7)
        lims = [
            min(df[tgt].min(), oof_pred[tgt].min()),
            max(df[tgt].max(), oof_pred[tgt].max())
        ]
        plt.plot(lims, lims, linestyle="--")
        plt.xlabel(f"True {tgt}")
        plt.ylabel(f"Predicted {tgt}")
        plt.title(f"{dataset_name}: True vs Predicted ({tgt})")
        save_fig(out_dir / f"{dataset_name}_{tgt}_scatter.png")

        # Residuals
        resid = df[tgt] - oof_pred[tgt]
        fig = plt.figure(figsize=(6, 4))
        plt.hist(resid.dropna(), bins=30)
        plt.xlabel("Residual (True - Pred)")
        plt.ylabel("Count")
        plt.title(f"{dataset_name}: Residuals ({tgt})")
        save_fig(out_dir / f"{dataset_name}_{tgt}_residuals.png")

    # Save importances
    for k, series in avg_importances.items():
        if series.empty:
            continue
        series.sort_values(ascending=False).head(20).to_csv(out_dir / f"{dataset_name}_{k}_avg_permutation_importance.csv")

    total = time.time() - start_time
    print(f"✅ Finished {dataset_name.upper()} in {timedelta(seconds=int(total))}\n")
    return oof_pred, fold_metrics_df, mean_metrics, avg_importances



def comparison_plots(old_metrics: pd.DataFrame, noisy_metrics: pd.DataFrame, out_dir: Path):
    """Create side-by-side comparison bar charts for RMSE and R2 per target."""
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = (
        pd.concat([old_metrics.assign(dataset="old"), noisy_metrics.assign(dataset="noisy")])
        .groupby(["dataset", "target"])[["RMSE", "R2"]].mean()
        .reset_index()
    )

    for metric in ["RMSE", "R2"]:
        fig = plt.figure(figsize=(7, 4))
        for i, tgt in enumerate(TARGET_COLS):
            subset = merged[merged["target"] == tgt]
            plt.bar([i - 0.15, i + 0.15], subset[metric], width=0.3, label=None if i else "old/noisy")
        plt.xticks(range(len(TARGET_COLS)), TARGET_COLS)
        plt.title(f"Old vs Noisy: {metric} by Target (lower RMSE, higher R2 is better)")
        plt.legend().remove()
        save_fig(out_dir / f"compare_{metric}.png")


def main():
    parser = argparse.ArgumentParser(description="Chained RF with normalization and OLD vs NOISY comparison")
    parser.add_argument("--original", type=str, default=DEFAULT_ORIGINAL_FILE, help="Path to original dataset CSV")
    parser.add_argument("--noisy", type=str, default=DEFAULT_NOISY_FILE, help="Path to noisy dataset CSV")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--no-scale-targets", action="store_true", help="Disable MinMax normalization of target E values")
    args = parser.parse_args()

    out_root = Path(args.out)

    # --- Load datasets ---
    old_df = pd.read_csv(args.original)
    noisy_df = pd.read_csv(args.noisy)

    ensure_columns(old_df, FEATURE_COLS + TARGET_COLS, "old_df")
    ensure_columns(noisy_df, FEATURE_COLS + TARGET_COLS, "noisy_df")

    # 1️⃣ Baseline: train/test on old
    old_oof, old_fold_metrics, old_mean_metrics, old_imps = run_chained_cv(
        old_df, "old", out_root / "old", scale_targets=not args.no_scale_targets
    )

    # 2️⃣ Train/test on noisy (from clean)
    noisy_clean_oof, noisy_clean_fold_metrics, noisy_clean_mean_metrics, noisy_clean_imps = run_chained_cv(
        noisy_df, "noisy_from_clean", out_root / "noisy_from_clean", scale_targets=not args.no_scale_targets
    )

    # 3️⃣ New: Train/test on noisy (from noisy)
    # (If you have a separate file for this, replace `args.noisy` with the correct filename)
    noisy_noisy_df = pd.read_csv("noisy_with_pca_from_noisy_colored.csv")
    ensure_columns(noisy_noisy_df, FEATURE_COLS + TARGET_COLS, "noisy_noisy_df")

    noisy_noisy_oof, noisy_noisy_fold_metrics, noisy_noisy_mean_metrics, noisy_noisy_imps = run_chained_cv(
        noisy_noisy_df, "noisy_from_noisy", out_root / "noisy_from_noisy", scale_targets=not args.no_scale_targets
    )

    # 4️⃣ Train on clean, test on noisy (final goal)
    print("\n🎯 Training on clean data, testing on noisy (final goal)...")
    X_train, y_train = old_df[FEATURE_COLS], old_df[TARGET_COLS]
    X_test, y_test = noisy_df[FEATURE_COLS], noisy_df[TARGET_COLS]

    models, y_pred_noisy = chained_fold_train_predict(X_train, y_train, X_test)
    final_results = pd.DataFrame({
        "Part1_true": y_test["Part1_E"], "Part1_pred": y_pred_noisy["Part1_E"],
        "Part3_true": y_test["Part3_E"], "Part3_pred": y_pred_noisy["Part3_E"],
        "Part11_true": y_test["Part11_E"], "Part11_pred": y_pred_noisy["Part11_E"]
    })
    final_results.to_csv(out_root / "clean_train_noisy_test_predictions.csv", index=False)

    # --- Save summary metrics ---
    pd.DataFrame(old_mean_metrics).T.to_csv(out_root / "old_mean_metrics.csv")
    pd.DataFrame(noisy_clean_mean_metrics).T.to_csv(out_root / "noisy_from_clean_mean_metrics.csv")
    pd.DataFrame(noisy_noisy_mean_metrics).T.to_csv(out_root / "noisy_from_noisy_mean_metrics.csv")

    # --- Comparison plots ---
    comparison_plots(old_fold_metrics, noisy_clean_fold_metrics, out_root / "comparison_old_vs_noisy_from_clean")
    comparison_plots(noisy_clean_fold_metrics, noisy_noisy_fold_metrics, out_root / "comparison_noisy_clean_vs_noisy_noisy")

    # --- Summary ---
    print("\n==== OLD Mean Metrics ====")
    print(pd.DataFrame(old_mean_metrics).T.round(4))
    print("\n==== NOISY (from clean) Mean Metrics ====")
    print(pd.DataFrame(noisy_clean_mean_metrics).T.round(4))
    print("\n==== NOISY (from noisy) Mean Metrics ====")
    print(pd.DataFrame(noisy_noisy_mean_metrics).T.round(4))
    print(f"\nAll artifacts saved under: {out_root.resolve()}")

if __name__ == "__main__":
    main()
