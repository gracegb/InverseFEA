# ml_with_chained_prediction_xgboost.py
# -------------------------------------------------------------
# Chained Prediction + Multi-Output CV + Permutation Importance per Target + Save Plots (XGBoost)
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb

# === 1. Load Dataset ===
try:
    data = pd.read_csv("original_with_pca.csv")
except FileNotFoundError:
    print("Error: 'original_with_pca.csv' not found. Place it in the current directory.")
    exit()

feature_cols = [
    "PC1_Bottom", "PC2_Bottom", "PC3_Bottom",
    "PC1_InnerShape", "PC2_InnerShape", "PC3_InnerShape",
    "PC1_OuterShape", "PC2_OuterShape", "PC3_OuterShape"
]
target_cols = ["Part1_E", "Part3_E", "Part11_E"]

data = data.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
X = data[feature_cols]
y = data[target_cols]

os.makedirs("plots", exist_ok=True)  # Directory for saving plots

# === 2. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Multi-Output 10-Fold CV ===
print("Running Multi-Output K-Fold Cross-Validation (10 folds) with XGBoost...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores, mse_scores = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
    y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

    # Multi-target handled by separate models (XGBoost does not natively do multi-output)
    preds_fold = pd.DataFrame(index=y_fold_test.index)
    for target in target_cols:
        model_fold = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model_fold.fit(X_fold_train, y_fold_train[target])
        preds_fold[target] = model_fold.predict(X_fold_test)

    r2_scores.append(r2_score(y_fold_test, preds_fold))
    mse_scores.append(mean_squared_error(y_fold_test, preds_fold))
    print(f"  Fold {fold+1:2d} → R²: {r2_scores[-1]:.4f}, MSE: {mse_scores[-1]:.6f}")

print(f"Cross-Validation Summary:\n  Mean R² : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"  Mean MSE: {np.mean(mse_scores):.6f} ± {np.std(mse_scores):.6f}")

# === 4. Train Baseline Models for Chaining ===
models = {}
preds = {}

for target in ["Part3_E", "Part11_E"]:
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train[target])
    models[target] = model
    preds[target] = model.predict(X_test)

# === 5. Chained Prediction for Part1_E ===
X_train_chained = X_train.copy()
X_train_chained["Part3_E_feature"] = y_train["Part3_E"]
X_train_chained["Part11_E_feature"] = y_train["Part11_E"]

X_test_chained = X_test.copy()
X_test_chained["Part3_E_feature"] = preds["Part3_E"]
X_test_chained["Part11_E_feature"] = preds["Part11_E"]

rf_part1_chained = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
rf_part1_chained.fit(X_train_chained, y_train["Part1_E"])
preds["Part1_E"] = rf_part1_chained.predict(X_test_chained)

print(f"\nChained Model Part1_E → R²: {r2_score(y_test['Part1_E'], preds['Part1_E']):.4f}, "
      f"MSE: {mean_squared_error(y_test['Part1_E'], preds['Part1_E']):.6f}")

# Baseline model (without chaining)
rf_part1_baseline = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
rf_part1_baseline.fit(X_train, y_train["Part1_E"])
pred_baseline = rf_part1_baseline.predict(X_test)
print(f"Baseline Model Part1_E R²: {r2_score(y_test['Part1_E'], pred_baseline):.4f}")

# === 6. Permutation Importance per Target ===
for target, model, X_eval in zip(
    target_cols,
    [rf_part1_chained, models["Part3_E"], models["Part11_E"]],
    [X_test_chained, X_test, X_test]
):
    print(f"\nPermutation Importance for {target}...")
    perm = permutation_importance(model, X_eval, y_test[target], n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = perm.importances_mean.argsort()

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(X_eval.columns)[sorted_idx], perm.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title(f"Permutation Importance: {target}")
    plt.tight_layout()
    plt.savefig(f"plots/perm_importance_{target}_xgboost.png")
    plt.close()

# === 7. Residual Plots per Target ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(target_cols):
    ax = axes[i]
    residuals = y_test[col] - preds[col]
    ax.scatter(preds[col], residuals, alpha=0.7, edgecolor='k')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel(f"Predicted {col}")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual Plot: {col}")

plt.suptitle("Residual Analysis of XGBoost Predictions", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("plots/residual_plots_xgboost.png")
plt.close()

# === 8. Predicted vs Actual Plots per Target ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(target_cols):
    ax = axes[i]
    ax.scatter(y_test[col], preds[col], alpha=0.7, edgecolor='k')
    min_val = min(y_test[col].min(), preds[col].min())
    max_val = max(y_test[col].max(), preds[col].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel(f"Actual {col}")
    ax.set_ylabel(f"Predicted {col}")
    ax.set_title(f"{col}")

plt.suptitle("Predicted vs Actual (XGBoost)", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/predicted_vs_actual_xgboost.png")
plt.close()

# === 9. Save Models and Predictions ===
joblib.dump(rf_part1_chained, "chained_model_part1_xgb.pkl")
joblib.dump(models["Part3_E"], "baseline_model_part3_xgb.pkl")
joblib.dump(models["Part11_E"], "baseline_model_part11_xgb.pkl")
print("\nXGBoost models saved successfully.")

pred_df = pd.DataFrame(preds)
pred_df.to_csv("chained_predictions_xgb.csv", index=False)
print("Predictions saved to chained_predictions_xgb.csv")
print("Plots saved to 'plots/' folder.")
