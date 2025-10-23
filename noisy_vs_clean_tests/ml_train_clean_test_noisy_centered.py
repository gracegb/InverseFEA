#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train on CLEAN (original) data, test on NOISY data,
with feature centering and normalization to correct PCA mean shifts.

Evaluates chained Random Forest Regressors:
  - RF_3: Part3_E ~ X
  - RF_11: Part11_E ~ X + Part3_E_pred
  - RF_1: Part1_E ~ X + Part3_E_pred + Part11_E_pred
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ----------------------- Config -----------------------

DEFAULT_CLEAN_FILE = "../noisy_data_tests/original_with_pca_colored-2.csv"
DEFAULT_NOISY_FILE = "../noisy_data_tests/noisy_with_pca_from_clean_colored.csv"

ALL_PC_FEATURES = [
    "PC1_InnerBase", "PC2_InnerBase", "PC3_InnerBase",
    "PC1_OuterBase", "PC2_OuterBase", "PC3_OuterBase",
    "PC1_InnerCircle", "PC2_InnerCircle", "PC3_InnerCircle",
    "PC1_MiddleCircle", "PC2_MiddleCircle", "PC3_MiddleCircle",
    "PC1_OuterCircle", "PC2_OuterCircle", "PC3_OuterCircle",
]

FIRST_PC_FEATURES = [
    "PC1_InnerBase", "PC1_OuterBase", "PC1_InnerCircle",
    "PC1_MiddleCircle", "PC1_OuterCircle",
]

TARGETS = ["Part1_E", "Part3_E", "Part11_E"]

RF_PARAMS = dict(
    n_estimators=1000,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)


# ----------------------- Core Logic -----------------------

def chained_train_predict(X_train, y_train, X_test):
    models = {}

    # A) Part3_E
    rf3 = RandomForestRegressor(**RF_PARAMS)
    rf3.fit(X_train, y_train["Part3_E"])
    models["RF_3"] = rf3
    y3_pred = rf3.predict(X_test)

    # B) Part11_E
    X_train_B = X_train.copy()
    X_train_B["Part3_E_pred"] = y_train["Part3_E"]
    X_test_B = X_test.copy()
    X_test_B["Part3_E_pred"] = y3_pred
    rf11 = RandomForestRegressor(**RF_PARAMS)
    rf11.fit(X_train_B, y_train["Part11_E"])
    models["RF_11"] = rf11
    y11_pred = rf11.predict(X_test_B)

    # C) Part1_E
    X_train_C = X_train.copy()
    X_train_C["Part3_E_pred"] = y_train["Part3_E"]
    X_train_C["Part11_E_pred"] = y_train["Part11_E"]

    X_test_C = X_test.copy()
    X_test_C["Part3_E_pred"] = y3_pred
    X_test_C["Part11_E_pred"] = y11_pred
    rf1 = RandomForestRegressor(**RF_PARAMS)
    rf1.fit(X_train_C, y_train["Part1_E"])
    models["RF_1"] = rf1
    y1_pred = rf1.predict(X_test_C)

    y_pred = pd.DataFrame({
        "Part1_E": y1_pred,
        "Part3_E": y3_pred,
        "Part11_E": y11_pred
    }, index=X_test.index)

    return models, y_pred


def evaluate_results(y_true, y_pred):
    metrics = {}
    for tgt in TARGETS:
        metrics[tgt] = dict(
            RMSE=mean_squared_error(y_true[tgt], y_pred[tgt], squared=False),
            MAE=mean_absolute_error(y_true[tgt], y_pred[tgt]),
            R2=r2_score(y_true[tgt], y_pred[tgt]),
        )
    return pd.DataFrame(metrics).T


def plot_results(y_true, y_pred, out_dir, suffix=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgt in TARGETS:
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true[tgt], y_pred[tgt], s=12, alpha=0.7)
        lims = [min(y_true[tgt].min(), y_pred[tgt].min()), max(y_true[tgt].max(), y_pred[tgt].max())]
        plt.plot(lims, lims, "k--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{tgt} (train clean → test noisy{suffix})")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tgt}_scatter{suffix}.png", dpi=200)
        plt.close()


# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser(description="Train on clean data, test on noisy (mean-centered + normalized)")
    parser.add_argument("--clean", type=str, default=DEFAULT_CLEAN_FILE)
    parser.add_argument("--noisy", type=str, default=DEFAULT_NOISY_FILE)
    parser.add_argument("--out", type=str, default="outputs_clean_train_noisy_centered")
    parser.add_argument("--first-pc-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    feature_cols = FIRST_PC_FEATURES if args.first_pc_only else ALL_PC_FEATURES

    print(f"\n🔹 Using {'first PCs only' if args.first_pc_only else 'all PCs'}")
    print(f"📂 Clean data: {args.clean}")
    print(f"📂 Noisy data: {args.noisy}")

    clean_df = pd.read_csv(args.clean)
    noisy_df = pd.read_csv(args.noisy)

    X_train, y_train = clean_df[feature_cols], clean_df[TARGETS]
    X_test, y_test = noisy_df[feature_cols], noisy_df[TARGETS]

    # --- Diagnostics before correction ---
    print("\n📊 Feature mean differences before centering:")
    for col in feature_cols:
        diff = X_test[col].mean() - X_train[col].mean()
        print(f"{col:<25} Δmean = {diff:+.4f}")

    # --------------------------------------------------------
    # 🧭 CENTER NOISY FEATURES TO MATCH CLEAN MEANS
    # --------------------------------------------------------
    print("\n🔧 Centering noisy feature means to match clean data...")
    X_test_centered = X_test - (X_test.mean() - X_train.mean())

    print("✅ Centering complete. Checking means after correction:")
    for col in feature_cols[:5]:
        print(f"{col:<25} clean mean={X_train[col].mean():+.3f} | noisy new mean={X_test_centered[col].mean():+.3f}")
    print()

    # --------------------------------------------------------
    # 🔧 FEATURE + TARGET NORMALIZATION
    # --------------------------------------------------------
    print("⚙️ Normalizing features based on CLEAN dataset statistics...\n")
    feature_scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(feature_scaler.transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(feature_scaler.transform(X_test_centered), columns=feature_cols, index=X_test.index)

    target_scaler = StandardScaler().fit(y_train)
    y_train_scaled = pd.DataFrame(target_scaler.transform(y_train), columns=TARGETS, index=y_train.index)
    y_test_scaled = pd.DataFrame(target_scaler.transform(y_test), columns=TARGETS, index=y_test.index)

    # --------------------------------------------------------
    # 🚀 Train and Predict
    # --------------------------------------------------------
    print("🚀 Training chained models (RF_3, RF_11, RF_1)...")
    models, y_pred_scaled = chained_train_predict(X_train_scaled, y_train_scaled, X_test_scaled)
    print("✅ Training complete.\n")

    y_pred = pd.DataFrame(
        target_scaler.inverse_transform(y_pred_scaled),
        columns=TARGETS,
        index=y_pred_scaled.index
    )

    # --------------------------------------------------------
    # 📊 Evaluate and Save
    # --------------------------------------------------------
    metrics = evaluate_results(y_test, y_pred)
    metrics_path = out_dir / ("metrics_firstPC.csv" if args.first_pc_only else "metrics_allPCs.csv")
    pred_path = out_dir / ("predictions_firstPC.csv" if args.first_pc_only else "predictions_allPCs.csv")

    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path)

    print("📈 Evaluation metrics:\n", metrics.round(4))

    plot_results(y_test, y_pred, out_dir, suffix="_firstPC" if args.first_pc_only else "_allPCs")
    print(f"\n✅ Done. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
