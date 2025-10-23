import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import os

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

os.makedirs("plots", exist_ok=True)

# === 2. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Multi-Output 10-Fold CV ===
print("Running Multi-Output K-Fold Cross-Validation (10 folds)...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores, mse_scores = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
    y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

    rf_multi = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_multi.fit(X_fold_train, y_fold_train)

    y_pred_fold = rf_multi.predict(X_fold_test)
    r2_scores.append(r2_score(y_fold_test, y_pred_fold))
    mse_scores.append(mean_squared_error(y_fold_test, y_pred_fold))
    print(f"  Fold {fold+1:2d} → R²: {r2_scores[-1]:.4f}, MSE: {mse_scores[-1]:.6f}")

print(f"Cross-Validation Summary:\n  Mean R² : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"  Mean MSE: {np.mean(mse_scores):.6f} ± {np.std(mse_scores):.6f}")

# ----------------------------------------
# 🎯 Model A: Predict Part3_E (from X)
# ----------------------------------------
rf_part3 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part3.fit(X_train, y_train["Part3_E"])

# Make initial prediction on the test set
part3_pred_test = rf_part3.predict(X_test)

# Evaluation
r2_3 = r2_score(y_test["Part3_E"], part3_pred_test)
mse_3 = mean_squared_error(y_test["Part3_E"], part3_pred_test)
print(f"Model 3 (Part3_E) R²: {r2_3:.4f}, MSE: {mse_3:.6f}")

# ----------------------------------------
# 🎯 Model B: Predict Part11_E (from X + Part3_E)
# ----------------------------------------

# 🔨 Training Data Prep: Use ACTUAL Part3_E as a feature
X_train_part11 = X_train.copy()
X_train_part11["Part3_E_feature"] = y_train["Part3_E"]

# 🧪 Testing Data Prep: Use PREDICTED Part3_E as a feature
X_test_part11 = X_test.copy()
X_test_part11["Part3_E_feature"] = part3_pred_test  # <--- Chaining happens here!

rf_part11_chained = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part11_chained.fit(X_train_part11, y_train["Part11_E"])

# Make prediction
part11_pred_test = rf_part11_chained.predict(X_test_part11)

# Evaluation
r2_11 = r2_score(y_test["Part11_E"], part11_pred_test)
mse_11 = mean_squared_error(y_test["Part11_E"], part11_pred_test)
print(f"Model 11 (Part11_E) R²: {r2_11:.4f}, MSE: {mse_11:.6f}")

# ----------------------------------------
# 🎯 Model C: Predict Part1_E (from X + Part3_E + Part11_E)
# ----------------------------------------

# 🔨 Training Data Prep: Use ACTUAL Part3_E and Part11_E
X_train_part1 = X_train.copy()
X_train_part1["Part3_E_feature"] = y_train["Part3_E"]
X_train_part1["Part11_E_feature"] = y_train["Part11_E"]

# 🧪 Testing Data Prep: Use PREDICTED Part3_E and Part11_E
X_test_part1 = X_test.copy()
X_test_part1["Part3_E_feature"] = part3_pred_test    # From Model 1
X_test_part1["Part11_E_feature"] = part11_pred_test  # From Model 2

rf_part1_chained = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part1_chained.fit(X_train_part1, y_train["Part1_E"])

# Make prediction
part1_pred_test = rf_part1_chained.predict(X_test_part1)

# Evaluation
r2_1 = r2_score(y_test["Part1_E"], part1_pred_test)
mse_1 = mean_squared_error(y_test["Part1_E"], part1_pred_test)
print(f"Model 1 (Part1_E) R²: {r2_1:.4f}, MSE: {mse_1:.6f}")