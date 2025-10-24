import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

# === 1. Load Dataset ===
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

# =================================================================
# === NORMALIZATION BLOCK ===
# =================================================================

# Initialize scalers
X_scaler = StandardScaler()
y_scaler = MinMaxScaler()

# 2a. Normalize X Features (StandardScaler: Z-score)
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

# 2b. Normalize Y Targets (MinMaxScaler: [0, 1])
y_train_scaled = y_scaler.fit_transform(y_train_orig)
y_test_scaled = y_scaler.transform(y_test_orig)
y_train = pd.DataFrame(y_train_scaled, columns=target_cols, index=y_train_orig.index)
y_test = pd.DataFrame(y_test_scaled, columns=target_cols, index=y_test_orig.index)

print("Features (X) standardized and Targets (Y) normalized to [0, 1].")

# === 3. Multi-Output 10-Fold CV ===
# ... (Section 3 remains the same, using scaled data) ...
print("\nRunning Multi-Output K-Fold Cross-Validation (10 folds) on normalized data...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores, mse_scores = [], []
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

# === 4. Train Baseline Models for Chaining (using scaled X and y) ===
rf_part3 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part3.fit(X_train, y_train["Part3_E"])
rf_part11 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part11.fit(X_train, y_train["Part11_E"])

part3_pred_scaled = rf_part3.predict(X_test)
part11_pred_scaled = rf_part11.predict(X_test)

# === 5. Chained Prediction for Part1_E (using scaled X and y) ===
X_train_chained = X_train.copy()
X_train_chained["Part3_E_feature"] = y_train["Part3_E"]
X_train_chained["Part11_E_feature"] = y_train["Part11_E"]

X_test_chained = X_test.copy()
X_test_chained["Part3_E_feature"] = part3_pred_scaled
X_test_chained["Part11_E_feature"] = part11_pred_scaled

rf_part1_chained = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part1_chained.fit(X_train_chained, y_train["Part1_E"])

part1_pred_chained_scaled = rf_part1_chained.predict(X_test_chained)

print(f"\nChained Model Part1_E R² (Normalized): {r2_score(y_test['Part1_E'], part1_pred_chained_scaled):.4f}, "
      f"MSE: {mean_squared_error(y_test['Part1_E'], part1_pred_chained_scaled):.6f}")

# === 6. Permutation Importance per Target ===
# ... (Section 6 remains the same, using scaled data) ...
for target, model, X_eval, y_eval in zip(
    target_cols,
    [rf_part1_chained, rf_part3, rf_part11],
    [X_test_chained, X_test, X_test],
    [y_test["Part1_E"], y_test["Part3_E"], y_test["Part11_E"]]
):
    print(f"\nPermutation Importance for {target} (Normalized)...")
    perm = permutation_importance(model, X_eval, y_eval,
                                  n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = perm.importances_mean.argsort()
    plt.figure(figsize=(8, 6))
    feature_names = X_eval.columns
    plt.barh(np.array(feature_names)[sorted_idx], perm.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title(f"Permutation Importance: {target} (Normalized)")
    plt.tight_layout()
    plt.savefig(f"plots/perm_importance_{target}_normalized.png")
    plt.close()


# =================================================================
# === 7. Residual and Predicted vs Actual Plots (Normalized Scale) ===
# =================================================================

# Use SCALED predictions and SCALED test targets
predictions_scaled = {
    "Part1_E": part1_pred_chained_scaled,
    "Part3_E": part3_pred_scaled,
    "Part11_E": part11_pred_scaled
}

# Residuals (Normalized Scale)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(target_cols):
    ax = axes[i]
    # Calculate residuals using SCALED data
    residuals = y_test[col] - predictions_scaled[col]
    ax.scatter(predictions_scaled[col], residuals, alpha=0.7, edgecolor='k')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel(f"Predicted {col} (Normalized)")
    ax.set_ylabel("Residuals (Normalized)")
    ax.set_title(f"Residual Plot: {col} (Normalized Scale)")
plt.suptitle("Residual Analysis (Normalized Scale)", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("plots/residual_plots_normalized.png")
plt.close()

# Predicted vs Actual (Normalized Scale)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(target_cols):
    ax = axes[i]
    # Plot SCALED actual vs SCALED predicted
    ax.scatter(y_test[col], predictions_scaled[col], alpha=0.7, edgecolor='k')
    min_val = min(y_test[col].min(), predictions_scaled[col].min())
    max_val = max(y_test[col].max(), predictions_scaled[col].max())
    # The line is 0 to 1 since data is normalized
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2) 
    ax.set_xlabel(f"Actual {col} (Normalized)")
    ax.set_ylabel(f"Predicted {col} (Normalized)")
    ax.set_title(f"{col} (Normalized Scale)")
plt.suptitle("Predicted vs Actual (Normalized Scale)", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/predicted_vs_actual_normalized.png")
plt.close()

# =================================================================
# === 8. % Error Analysis (Original Scale is typically used, but can be done normalized) ===
# To calculate % Error, we MUST use the ORIGINAL, unscaled data.
# The previous sections were purely for scaled metric evaluation and residual pattern checking.
# We will use inverse transformation here ONLY for the % Error, as percentage error on
# a [0, 1] normalized scale is usually misleading unless the original data starts at 0.
# We will use the Inverse Transformation logic from the previous solution for correctness.
# =================================================================

# Inverse transform scaled predictions back to original scale for meaningful % Error
predictions_scaled_array = np.column_stack([
    predictions_scaled["Part1_E"], predictions_scaled["Part3_E"], predictions_scaled["Part11_E"]
])
predictions_original_array = y_scaler.inverse_transform(predictions_scaled_array)

predictions_final_original = pd.DataFrame(
    predictions_original_array, 
    columns=target_cols, 
    index=X_test.index
)

# Use ORIGINAL targets for % error calculation
y_test_final_original = y_test_orig.copy()

percent_errors = {}
for col in target_cols:
    # Use the original targets and inverse transformed predictions
    percent_errors[col] = np.abs((predictions_final_original[col] - y_test_final_original[col]) / y_test_final_original[col]) * 100

# Individual % error plots
# ... (plotting code remains the same, using 'percent_errors' and labeling as Original Scale) ...
for col in target_cols:
    plt.figure(figsize=(8, 5))
    plt.plot(percent_errors[col].values, 'o-', label=f'% Error {col}')
    plt.xlabel("Sample Index")
    plt.ylabel("% Error")
    plt.title(f"Percent Error for {col} (Original Scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/percent_error_{col}_original_from_scaled.png", dpi=300)
    plt.close()

# Combined boxplot for comparison
plt.figure(figsize=(8, 6))
plt.boxplot([percent_errors[col] for col in target_cols], labels=target_cols)
plt.ylabel("% Error")
plt.title("Comparison of % Error Across Targets (Original Scale)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/percent_error_comparison_original_from_scaled.png", dpi=300)
plt.close()

print("Percent error plots (Original Scale) saved.")

# === 10. Prepare for Noisy Dataset Testing ===

# === 11. Save Models and Predictions ===
joblib.dump(rf_part1_chained, "chained_model_part1_scaled.pkl")
joblib.dump(rf_part3, "baseline_model_part3_scaled.pkl")
joblib.dump(rf_part11, "baseline_model_part11_scaled.pkl")

# Save predictions on the ORIGINAL scale for final report/use
pred_df = predictions_final_original
pred_df.to_csv("chained_predictions_original_from_scaled.csv", index=False)
print("\nModels and predictions saved successfully.")
print("All plots saved to 'plots/' folder.")