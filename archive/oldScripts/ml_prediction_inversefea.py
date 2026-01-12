# random_forest_with_validation_and_combined_plot.py
# -------------------------------------------------------------
# Train a RandomForestRegressor on PC feature columns to predict
# Part*_E targets, evaluate generalization, run k-fold validation,
# and visualize predictions (individually + combined).
# -------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# === 1. Load Dataset ===
data = pd.read_csv("original_with_pca.csv")  # <-- replace with your filename

# Define feature and target columns
feature_cols = [
    "PC1_Bottom", "PC2_Bottom", "PC3_Bottom",
    "PC1_InnerShape", "PC2_InnerShape", "PC3_InnerShape",
    "PC1_OuterShape", "PC2_OuterShape", "PC3_OuterShape"
]
target_cols = ["Part1_E", "Part3_E", "Part11_E"]

# Drop rows with missing data in key columns
data = data.dropna(subset=feature_cols + target_cols)

X = data[feature_cols]
y = data[target_cols]

# === 2. Split into Train/Test Sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3. Train Random Forest ===
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# === 4. Evaluate on Train/Test Sets ===
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nPerformance Metrics")
print(f"Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
print(f"Test  MSE: {test_mse:.6f}, R²: {test_r2:.4f}")

# === 5. 5-Fold Cross-Validation ===
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2', n_jobs=-1)
print("\n5-Fold Cross-Validation R² Scores:")
print(cv_scores)
print(f"Mean R² across folds: {cv_scores.mean():.4f}")

# === 6. Feature Importances ===
importances = pd.Series(rf.feature_importances_, index=feature_cols)
print("\n Feature Importances:")
print(importances.sort_values(ascending=False))

# === 7. Save Model and Predictions ===
joblib.dump(rf, "random_forest_model.pkl")
print("\nModel saved as random_forest_model.pkl")

pred_df = pd.DataFrame(y_test_pred, columns=[f"Pred_{col}" for col in target_cols])
pred_df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")

# === 8. Individual Predicted vs Actual Plots ===
for i, col in enumerate(target_cols):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test[col], y_test_pred[:, i], alpha=0.7, edgecolor='k')
    plt.xlabel(f"Actual {col}")
    plt.ylabel(f"Predicted {col}")
    plt.title(f"Predicted vs Actual — {col}")
    min_val = min(y_test[col].min(), y_test_pred[:, i].min())
    max_val = max(y_test[col].max(), y_test_pred[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.tight_layout()
    plt.show()

# === 9. Combined 3-Panel Visualization ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(target_cols):
    ax = axes[i]
    ax.scatter(y_test[col], y_test_pred[:, i], alpha=0.7, edgecolor='k')
    min_val = min(y_test[col].min(), y_test_pred[:, i].min())
    max_val = max(y_test[col].max(), y_test_pred[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel(f"Actual {col}")
    ax.set_ylabel(f"Predicted {col}")
    ax.set_title(f"{col}")

plt.suptitle("Predicted vs Actual (All Targets)", fontsize=14, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
