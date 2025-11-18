#!/usr/bin/env python3
"""OOF-stacked models with CORAL alignment and quantile-aware Part1_E focus.

This experimental script blends several ideas to improve the final ``Part1_E``
prediction while still producing chained predictions for all three E targets:

- **OOF stacking** replaces the previous ground-truth chaining so that
  higher-order models train on prediction-like signals.
- **CORAL (covariance alignment)** is applied per-fold to re-align feature
  distributions between train/test splits.
- **Geometry diagnostics** augment the PCA features with per-ring magnitudes and
  ratios.
- **Quantile regression** for ``Part1_E`` produces 10/90th percentile intervals
  alongside the main estimate.
- **Low-modulus corrector** upweights the smallest ``Part1_E`` samples and adds a
  residual model specialised on the lower quantile region.

Outputs are written to an experiment directory that contains OOF predictions,
fold metrics, and per-target feature importances.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


TARGET_COLS: List[str] = ["Part1_E", "Part3_E", "Part11_E"]

BASE_FEATURE_COLS: List[str] = [
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

RINGS: Dict[str, Tuple[str, str, str]] = {
    "InnerBase": ("PC1_InnerBase", "PC2_InnerBase", "PC3_InnerBase"),
    "OuterBase": ("PC1_OuterBase", "PC2_OuterBase", "PC3_OuterBase"),
    "InnerCircle": ("PC1_InnerCircle", "PC2_InnerCircle", "PC3_InnerCircle"),
    "MiddleCircle": ("PC1_MiddleCircle", "PC2_MiddleCircle", "PC3_MiddleCircle"),
    "OuterCircle": ("PC1_OuterCircle", "PC2_OuterCircle", "PC3_OuterCircle"),
}


@dataclass
class Config:
    csv_path: Path
    output_dir: Path
    n_splits: int
    random_state: int
    part1_quantiles: Tuple[float, float]
    low_modulus_weight: float
    low_modulus_quantile: float
    coral_reg: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_columns(df: pd.DataFrame, columns: List[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {label}: {missing}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    geom_df = pd.DataFrame(index=df.index)

    for ring, axes in RINGS.items():
        comps = df[list(axes)]
        geom_df[f"{ring}_norm"] = np.linalg.norm(comps, axis=1)
        geom_df[f"{ring}_abs_mean"] = comps.abs().mean(axis=1)

    # Simple contrast ratios
    geom_df["OuterBase_to_InnerBase_norm"] = geom_df["OuterBase_norm"] / (
        geom_df["InnerBase_norm"] + 1e-6
    )
    geom_df["OuterCircle_to_InnerCircle_norm"] = geom_df["OuterCircle_norm"] / (
        geom_df["InnerCircle_norm"] + 1e-6
    )
    geom_df["MiddleCircle_to_InnerBase_abs"] = geom_df["MiddleCircle_abs_mean"] / (
        geom_df["InnerBase_abs_mean"] + 1e-6
    )

    return geom_df


# ---------------------------------------------------------------------------
# CORAL alignment
# ---------------------------------------------------------------------------


def coral_align(train: pd.DataFrame, test: pd.DataFrame, reg: float) -> pd.DataFrame:
    Xc = train.columns
    train_c = train[Xc].to_numpy()
    test_c = test[Xc].to_numpy()

    train_mean = train_c.mean(axis=0, keepdims=True)
    test_mean = test_c.mean(axis=0, keepdims=True)

    train_centered = train_c - train_mean
    test_centered = test_c - test_mean

    cov_train = np.cov(train_centered, rowvar=False) + reg * np.eye(train_c.shape[1])
    cov_test = np.cov(test_centered, rowvar=False) + reg * np.eye(train_c.shape[1])

    # Compute whitening and coloring transforms
    u_t, s_t, _ = np.linalg.svd(cov_test)
    u_s, s_s, _ = np.linalg.svd(cov_train)
    inv_sqrt_cov_test = (u_t @ np.diag(1.0 / np.sqrt(s_t)) @ u_t.T)
    sqrt_cov_train = (u_s @ np.diag(np.sqrt(s_s)) @ u_s.T)

    aligned_test = (test_centered @ inv_sqrt_cov_test @ sqrt_cov_train) + train_mean
    aligned_df = pd.DataFrame(aligned_test, columns=Xc, index=test.index)
    return aligned_df


# ---------------------------------------------------------------------------
# Modeling utilities
# ---------------------------------------------------------------------------


def fit_oof_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_builder,
    n_splits: int,
    random_state: int,
) -> Tuple[np.ndarray, object]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y))

    for train_idx, val_idx in kf.split(X):
        model = model_builder()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred[val_idx] = model.predict(X.iloc[val_idx])

    final_model = model_builder()
    final_model.fit(X, y)
    return oof_pred, final_model


def build_rf(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
    )


def build_gbdt(random_state: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=random_state, max_depth=3)


def build_quantile_gbdt(alpha: float, random_state: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile", alpha=alpha, random_state=random_state, max_depth=3
    )


# ---------------------------------------------------------------------------
# Core CV routine
# ---------------------------------------------------------------------------


def run_oof_chained_cv(df: pd.DataFrame, config: Config) -> Dict[str, pd.DataFrame]:
    X_base = df[BASE_FEATURE_COLS].copy()
    X_geom = build_geometry_features(df[BASE_FEATURE_COLS])
    X_full = pd.concat([X_base, X_geom], axis=1)

    y = df[TARGET_COLS].copy()

    kf_outer = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)

    oof_predictions = pd.DataFrame(index=df.index, columns=TARGET_COLS, dtype=float)
    oof_lower = pd.Series(index=df.index, dtype=float)
    oof_upper = pd.Series(index=df.index, dtype=float)
    metrics_records: List[Dict[str, float]] = []

    feature_importances = {"Part3_E": None, "Part11_E": None, "Part1_E": None}
    counts = {key: 0 for key in feature_importances}

    for fold, (train_idx, test_idx) in enumerate(kf_outer.split(X_full), start=1):
        X_tr, X_te = X_full.iloc[train_idx].copy(), X_full.iloc[test_idx].copy()
        y_tr, y_te = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # CORAL alignment of test features to train distribution
        X_te_aligned = coral_align(X_tr, X_te, reg=config.coral_reg)

        # Stage A: Part3_E OOF and model
        part3_oof, part3_model = fit_oof_model(
            X_tr, y_tr["Part3_E"],
            lambda: build_rf(config.random_state),
            n_splits=config.n_splits,
            random_state=config.random_state,
        )
        part3_pred_test = part3_model.predict(X_te_aligned)

        # Stage B: Part11_E conditioned on Part3_E predictions
        X_tr_11 = X_tr.copy()
        X_tr_11["Part3_E_oof"] = part3_oof

        part11_oof, part11_model = fit_oof_model(
            X_tr_11, y_tr["Part11_E"],
            lambda: build_rf(config.random_state),
            n_splits=config.n_splits,
            random_state=config.random_state,
        )
        X_te_11 = X_te_aligned.copy()
        X_te_11["Part3_E_oof"] = part3_pred_test
        part11_pred_test = part11_model.predict(X_te_11)

        # Stage C: Part1_E with quantiles and residual corrector
        X_tr_1 = X_tr.copy()
        X_tr_1["Part3_E_oof"] = part3_oof
        X_tr_1["Part11_E_oof"] = part11_oof

        # Upweight small modulus samples
        part1_series = y_tr["Part1_E"]
        scaled_part1 = (part1_series - part1_series.min()) / (
            part1_series.max() - part1_series.min() + 1e-6
        )
        part1_weights = 1.0 + config.low_modulus_weight * (1.0 - scaled_part1)

        def train_part1_mean():
            model = build_gbdt(config.random_state)
            model.fit(X_tr_1, part1_series, sample_weight=part1_weights)
            return model

        part1_oof, part1_mean_model = fit_oof_model(
            X_tr_1, part1_series, train_part1_mean,
            n_splits=config.n_splits,
            random_state=config.random_state,
        )

        # Quantile models
        lower_q, upper_q = config.part1_quantiles
        lower_oof, lower_model = fit_oof_model(
            X_tr_1, part1_series,
            lambda: build_quantile_gbdt(lower_q, config.random_state),
            n_splits=config.n_splits,
            random_state=config.random_state,
        )
        upper_oof, upper_model = fit_oof_model(
            X_tr_1, part1_series,
            lambda: build_quantile_gbdt(upper_q, config.random_state),
            n_splits=config.n_splits,
            random_state=config.random_state,
        )

        # Residual corrector for low-modulus region
        low_mask = part1_series <= part1_series.quantile(config.low_modulus_quantile)
        residual_target = (part1_series - part1_oof)[low_mask]
        residual_features = X_tr_1[low_mask]
        residual_model = build_rf(config.random_state)
        residual_model.fit(residual_features, residual_target)

        X_te_1 = X_te_aligned.copy()
        X_te_1["Part3_E_oof"] = part3_pred_test
        X_te_1["Part11_E_oof"] = part11_pred_test

        part1_pred_test = part1_mean_model.predict(X_te_1)
        part1_pred_test += residual_model.predict(X_te_1)

        lower_pred_test = lower_model.predict(X_te_1)
        upper_pred_test = upper_model.predict(X_te_1)

        # Collect outputs
        fold_pred = pd.DataFrame(
            {
                "Part1_E": part1_pred_test,
                "Part3_E": part3_pred_test,
                "Part11_E": part11_pred_test,
            },
            index=y_te.index,
        )
        oof_predictions.loc[fold_pred.index] = fold_pred
        oof_lower.loc[fold_pred.index] = lower_pred_test
        oof_upper.loc[fold_pred.index] = upper_pred_test

        # Metrics per target
        for target in TARGET_COLS:
            y_true = y_te[target]
            y_pred = fold_pred[target]
            metrics_records.append(
                {
                    "fold": fold,
                    "target": target,
                    "rmse": rmse(y_true, y_pred),
                    "mae": mae(y_true, y_pred),
                    "r2": r2(y_true, y_pred),
                }
            )

        # Quantile coverage for Part1_E
        metrics_records.append(
            {
                "fold": fold,
                "target": "Part1_E_interval",
                "rmse": rmse(y_te["Part1_E"], part1_pred_test),
                "mae": mae(y_te["Part1_E"], part1_pred_test),
                "r2": np.mean(
                    (y_te["Part1_E"] >= lower_pred_test)
                    & (y_te["Part1_E"] <= upper_pred_test)
                ),
            }
        )

        # Aggregate feature importances (RF only exposes them)
        for name, model in {
            "Part3_E": part3_model,
            "Part11_E": part11_model,
            "Part1_E": residual_model,
        }.items():
            if hasattr(model, "feature_importances_"):
                importance = pd.Series(
                    model.feature_importances_, index=model.feature_names_in_
                )
                feature_importances[name] = (
                    importance if feature_importances[name] is None else feature_importances[name] + importance
                )
                counts[name] += 1

    fold_metrics = pd.DataFrame(metrics_records)
    mean_metrics = fold_metrics.groupby("target")[["rmse", "mae", "r2"]].mean().reset_index()

    avg_importances = {}
    for name, imp in feature_importances.items():
        if imp is not None and counts[name] > 0:
            avg_importances[name] = (imp / counts[name]).sort_values(ascending=False)

    return {
        "predictions": oof_predictions,
        "lower_quantile": oof_lower,
        "upper_quantile": oof_upper,
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "feature_importances": avg_importances,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("datasets/noisy_with_pca_from_clean_colored.csv"),
        help="Path to the training CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("noisy_data_tests/outputs/oof_coral_quantile_part1"),
        help="Directory where outputs will be written.",
    )
    parser.add_argument("--splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--part1-quantiles",
        type=float,
        nargs=2,
        default=(0.1, 0.9),
        metavar=("LOWER", "UPPER"),
        help="Quantiles for Part1_E interval estimates.",
    )
    parser.add_argument(
        "--low-modulus-weight",
        type=float,
        default=1.5,
        help="Strength of reweighting for low Part1_E samples.",
    )
    parser.add_argument(
        "--low-modulus-quantile",
        type=float,
        default=0.35,
        help="Quantile threshold for residual corrector training.",
    )
    parser.add_argument(
        "--coral-reg",
        type=float,
        default=1e-3,
        help="Diagonal regularisation added to covariance matrices for CORAL.",
    )
    args = parser.parse_args()

    return Config(
        csv_path=args.csv,
        output_dir=args.output_dir,
        n_splits=args.splits,
        random_state=args.seed,
        part1_quantiles=(args.part1_quantiles[0], args.part1_quantiles[1]),
        low_modulus_weight=args.low_modulus_weight,
        low_modulus_quantile=args.low_modulus_quantile,
        coral_reg=args.coral_reg,
    )


def main() -> None:
    config = parse_args()

    df = pd.read_csv(config.csv_path, encoding="utf-8-sig")
    ensure_columns(df, BASE_FEATURE_COLS + TARGET_COLS, "input CSV")

    results = run_oof_chained_cv(df, config)

    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = results["predictions"].copy()
    preds.columns = [f"{c}_pred" for c in preds.columns]
    preds["Part1_E_lower"] = results["lower_quantile"]
    preds["Part1_E_upper"] = results["upper_quantile"]

    combined = pd.concat([df[TARGET_COLS], preds], axis=1)
    combined.to_csv(out_dir / "oof_predictions.csv", index=False)

    results["fold_metrics"].to_csv(out_dir / "fold_metrics.csv", index=False)
    results["mean_metrics"].to_csv(out_dir / "mean_metrics.csv", index=False)

    for name, imp in results["feature_importances"].items():
        imp.to_csv(out_dir / f"feature_importances_{name}.csv")

    print("\nMean metrics (higher R2 is better):")
    print(results["mean_metrics"])
    print(f"Saved outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
