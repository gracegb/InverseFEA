import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler # <-- NEW IMPORT
from sklearn.inspection import permutation_importance
import joblib
import os

# Define file name and columns
DATA_FILE = "original_with_pca.csv"
feature_cols = [
    "PC1_Bottom", "PC2_Bottom", "PC3_Bottom",
    "PC1_InnerShape", "PC2_InnerShape", "PC3_InnerShape",
    "PC1_OuterShape", "PC2_OuterShape", "PC3_OuterShape"
]
target_cols = ["Part1_E", "Part3_E", "Part11_E"]

# === 1. Load Dataset & Setup ===
try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found. Place it in the current directory.")
    exit()

data = data.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
X = data[feature_cols]
y = data[target_cols]

os.makedirs("plots", exist_ok=True)

# === 2. Train/Test Split ===
X_train, X_test, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Target Normalization (Scaling to [0, 1]) ===
y_scaler = MinMaxScaler()

# Fit the scaler ONLY on the training data and transform both sets
y_train_scaled = y_scaler.fit_transform(y_train_orig)
y_test_scaled = y_scaler.transform(y_test_orig)

# Convert back to DataFrames for easy column access
y_train = pd.DataFrame(y_train_scaled, columns=target_cols, index=y_train_orig.index)
y_test = pd.DataFrame(y_test_scaled, columns=target_cols, index=y_test_orig.index)
print(f"Target values normalized to range [0, 1] using MinMaxScaler.")

# === 4. Multi-Output 10-Fold CV (Uses scaled data) ===
print("\nRunning Multi-Output K-Fold Cross-Validation (10 folds) on normalized data...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores, mse_scores = [], []

# Use the full scaled dataset (X and y scaled) for CV
X_full_scaled = pd.concat([X_train, X_test])
y_full_scaled = pd.concat([y_train, y_test])

for fold, (train_idx, test_idx) in enumerate(kf.split(X_full_scaled)):
    X_fold_train, X_fold_test = X_full_scaled.iloc[train_idx], X_full_scaled.iloc[test_idx]
    y_fold_train, y_fold_test = y_full_scaled.iloc[train_idx], y_full_scaled.iloc[test_idx]

    rf_multi = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_multi.fit(X_fold_train, y_fold_train)

    y_pred_fold = rf_multi.predict(X_fold_test)
    r2_scores.append(r2_score(y_fold_test, y_pred_fold))
    mse_scores.append(mean_squared_error(y_fold_test, y_pred_fold))
    print(f"  Fold {fold+1:2d} → R²: {r2_scores[-1]:.4f}, MSE: {mse_scores[-1]:.6f}")

print(f"Cross-Validation Summary (Normalized):\n  Mean R² : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"  Mean MSE: {np.mean(mse_scores):.6f} ± {np.std(mse_scores):.6f}")

# =================================================================
# === 5. SEQUENTIAL CHAINED MODELS (Trained and Tested on SCALED Data) ===
# =================================================================

# ----------------------------------------
# 🎯 Model A (RF_3): Predict Part3_E (from X)
# ----------------------------------------
rf_part3 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part3.fit(X_train, y_train["Part3_E"])

part3_pred_test = rf_part3.predict(X_test) # Prediction is SCALED

r2_3 = r2_score(y_test["Part3_E"], part3_pred_test)
mse_3 = mean_squared_error(y_test["Part3_E"], part3_pred_test)
print(f"\nModel A (Part3_E) R²: {r2_3:.4f}, MSE: {mse_3:.6f} (Normalized Scale)")

# ----------------------------------------
# 🎯 Model B (RF_11): Predict Part11_E (from X + Part3_E)
# ----------------------------------------

# Training: X + ACTUAL SCALED Part3_E
X_train_part11 = X_train.copy()
X_train_part11["Part3_E_feature"] = y_train["Part3_E"]

# Testing: X + PREDICTED SCALED Part3_E
X_test_part11 = X_test.copy()
X_test_part11["Part3_E_feature"] = part3_pred_test

rf_part11_chained = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part11_chained.fit(X_train_part11, y_train["Part11_E"])

part11_pred_test = rf_part11_chained.predict(X_test_part11) # Prediction is SCALED

r2_11 = r2_score(y_test["Part11_E"], part11_pred_test)
mse_11 = mean_squared_error(y_test["Part11_E"], part11_pred_test)
print(f"Model B (Part11_E) R²: {r2_11:.4f}, MSE: {mse_11:.6f} (Normalized Scale)")

# ----------------------------------------
# 🎯 Model C (RF_1): Predict Part1_E (from X + Part3_E + Part11_E)
# ----------------------------------------

# Training: X + ACTUAL SCALED Part3_E and Part11_E
X_train_part1 = X_train.copy()
X_train_part1["Part3_E_feature"] = y_train["Part3_E"]
X_train_part1["Part11_E_feature"] = y_train["Part11_E"]

# Testing: X + PREDICTED SCALED Part3_E and Part11_E
X_test_part1 = X_test.copy()
X_test_part1["Part3_E_feature"] = part3_pred_test
X_test_part1["Part11_E_feature"] = part11_pred_test

rf_part1_chained = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part1_chained.fit(X_train_part1, y_train["Part1_E"])

part1_pred_test = rf_part1_chained.predict(X_test_part1) # Prediction is SCALED

r2_1 = r2_score(y_test["Part1_E"], part1_pred_test)
mse_1 = mean_squared_error(y_test["Part1_E"], part1_pred_test)
print(f"Model C (Part1_E) R²: {r2_1:.4f}, MSE: {mse_1:.6f} (Normalized Scale)")

# === 6. Residual and Predicted vs Actual Plots (Using Normalized Data) ===
predictions_scaled = {
    "Part1_E": part1_pred_test,
    "Part3_E": part3_pred_test,
    "Part11_E": part11_pred_test
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(target_cols):
    ax = axes[i]
    residuals = y_test[col] - predictions_scaled[col]
    ax.scatter(predictions_scaled[col], residuals, alpha=0.7, edgecolor='k')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel(f"Predicted {col} (Normalized)")
    ax.set_ylabel("Residuals (Normalized)")
    ax.set_title(f"Residual Plot: {col}")
plt.suptitle("Residual Analysis (Normalized Scale)", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("plots/residual_plots_normalized.png")
plt.close()

print("\nAll plots saved to 'plots/' folder.")