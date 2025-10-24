# ml_with_chained_prediction_fixed_v3.py
# -------------------------------------------------------------
# Adds: % Error Analysis, Mean % Error Bar Graph, and Noise Dataset Handling
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import os

# -------------------------------------------------------------
# Helper: Run Chained Multi-Target Prediction
# -------------------------------------------------------------
def run_chained_prediction(X_train, X_test, y_train, y_test, target_order, n_estimators=200):
    """
    Perform chained regression in a specified order.
    Example order: ['Part1_E', 'Part3_E', 'Part11_E']

    Each subsequent model gets the previous target predictions added as features.
    """

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error

    predictions = {}
    models = {}
    X_train_chain = X_train.copy()
    X_test_chain = X_test.copy()

    for i, target in enumerate(target_order):
        # Train model for this target
        print(f"Training chained model for {target} (Step {i+1}/{len(target_order)})...")

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train_chain, y_train[target])
        models[target] = rf

        # Predict on test set
        y_pred = rf.predict(X_test_chain)
        predictions[target] = y_pred

        # Print metrics
        r2 = r2_score(y_test[target], y_pred)
        mse = mean_squared_error(y_test[target], y_pred)
        print(f"  {target} → R²: {r2:.4f}, MSE: {mse:.6f}")

        # Add this target as a new feature for next model
        feature_name = f"{target}_feature"
        X_train_chain[feature_name] = y_train[target]
        X_test_chain[feature_name] = y_pred

    return predictions, models


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

# === 4. Train Baseline Models for Chaining ===
rf_part3 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part3.fit(X_train, y_train["Part3_E"])
rf_part11 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part11.fit(X_train, y_train["Part11_E"])

part3_pred = rf_part3.predict(X_test)
part11_pred = rf_part11.predict(X_test)

# === 5. Chained Prediction for Part1_E ===
X_train_chained = X_train.copy()
X_train_chained["Part3_E_feature"] = y_train["Part3_E"]
X_train_chained["Part11_E_feature"] = y_train["Part11_E"]

X_test_chained = X_test.copy()
X_test_chained["Part3_E_feature"] = part3_pred
X_test_chained["Part11_E_feature"] = part11_pred

rf_part1_chained = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part1_chained.fit(X_train_chained, y_train["Part1_E"])

part1_pred_chained = rf_part1_chained.predict(X_test_chained)

print(f"Chained Model Part1_E → R²: {r2_score(y_test['Part1_E'], part1_pred_chained):.4f}, "
      f"MSE: {mean_squared_error(y_test['Part1_E'], part1_pred_chained):.6f}")

# Baseline model (without chaining)
rf_part1_baseline = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part1_baseline.fit(X_train, y_train["Part1_E"])
part1_pred_baseline = rf_part1_baseline.predict(X_test)
print(f"Baseline Model Part1_E R²: {r2_score(y_test['Part1_E'], part1_pred_baseline):.4f}")

# === 5B. Full Multi-Target Chaining (Custom Order) ===
print("\nRunning full multi-target chained prediction...\n")

target_order = ["Part1_E", "Part3_E", "Part11_E"]
chained_predictions, chained_models = run_chained_prediction(
    X_train, X_test, y_train, y_test, target_order
)

# Save predictions for downstream plots
predictions = {}
for t in target_order:
    predictions[t] = chained_predictions[t]

# === 6. Permutation Importance per Target ===
for target, model, X_eval in zip(
    target_cols,
    [rf_part1_chained, rf_part3, rf_part11],
    [X_test_chained, X_test, X_test]
):
    print(f"\nPermutation Importance for {target}...")
    perm = permutation_importance(model, X_eval, y_test[target],
                                  n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = perm.importances_mean.argsort()

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(X_eval.columns)[sorted_idx], perm.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title(f"Permutation Importance: {target}")
    plt.tight_layout()
    plt.savefig(f"plots/perm_importance_{target}.png")
    plt.close()

# === 7. Residual and Predicted vs Actual Plots ===
predictions = {
    "Part1_E": part1_pred_chained,
    "Part3_E": part3_pred,
    "Part11_E": part11_pred
}

# Residuals
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(target_cols):
    ax = axes[i]
    residuals = y_test[col] - predictions[col]
    ax.scatter(predictions[col], residuals, alpha=0.7, edgecolor='k')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel(f"Predicted {col}")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual Plot: {col}")
plt.suptitle("Residual Analysis", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("plots/residual_plots.png")
plt.close()

# Predicted vs Actual
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(target_cols):
    ax = axes[i]
    ax.scatter(y_test[col], predictions[col], alpha=0.7, edgecolor='k')
    min_val = min(y_test[col].min(), predictions[col].min())
    max_val = max(y_test[col].max(), predictions[col].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel(f"Actual {col}")
    ax.set_ylabel(f"Predicted {col}")
    ax.set_title(f"{col}")
plt.suptitle("Predicted vs Actual", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/predicted_vs_actual.png")
plt.close()

# === 8. % Error Analysis ===
percent_errors = {}
for col in target_cols:
    percent_errors[col] = np.abs((predictions[col] - y_test[col]) / y_test[col]) * 100

# Individual % error plots
for col in target_cols:
    plt.figure(figsize=(8, 5))
    plt.plot(percent_errors[col].values, 'o-', label=f'% Error {col}')
    plt.xlabel("Sample Index")
    plt.ylabel("% Error")
    plt.title(f"Percent Error for {col}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/percent_error_{col}.png", dpi=300)
    plt.close()

# Combined boxplot for comparison
plt.figure(figsize=(8, 6))
plt.boxplot([percent_errors[col] for col in target_cols], labels=target_cols)
plt.ylabel("% Error")
plt.title("Comparison of % Error Across Targets")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/percent_error_comparison.png", dpi=300)
plt.close()

print("Percent error plots saved.")

# === 8B. Bar Graph: Mean % Error per E Value ===
mean_percent_errors = [np.mean(percent_errors[col]) for col in target_cols]

plt.figure(figsize=(7, 5))
bars = plt.bar(target_cols, mean_percent_errors, color=["#4c72b0", "#55a868", "#c44e52"])
plt.ylabel("Mean % Error")
plt.title("Mean % Error per E Value")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Label bars with their exact values
for bar, val in zip(bars, mean_percent_errors):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.2f}%", ha='center', va='bottom', fontsize=10)

plt.savefig("plots/mean_percent_error_bar.png", dpi=300)
plt.close()

print("Mean % error bar chart saved as plots/mean_percent_error_bar.png.")

# === 9. Noisy Dataset Testing ===
# Example usage:
# datasets = ["original_with_pca.csv", "noisy_low.csv", "noisy_high.csv"]
# for file in datasets:
#     run_pipeline(file)

# === 10. Save Models and Predictions ===
joblib.dump(rf_part1_chained, "chained_model_part1.pkl")
joblib.dump(rf_part3, "baseline_model_part3.pkl")
joblib.dump(rf_part11, "baseline_model_part11.pkl")

pred_df = pd.DataFrame(predictions)
pred_df.to_csv("chained_predictions.csv", index=False)
print("\nModels and predictions saved successfully.")
print("All plots saved to 'plots/' folder.")
