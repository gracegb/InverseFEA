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

# =================================================================
# === NEW FEATURE DEFINITION (REPLACING original feature_cols) ===
# =================================================================

# Define file name and columns
DATA_FILE = "original_with_pca_colored-2.csv" # Assumed to contain these new columns
target_cols = ["Part1_E", "Part3_E", "Part11_E"]

# *****************************************************************
# ******* UPDATED FEATURE COLUMNS based on user's new list ********
# *****************************************************************
feature_cols = [
    "PC1_InnerBase", "PC2_InnerBase", "PC3_InnerBase",
    "PC1_OuterBase", "PC2_OuterBase", "PC3_OuterBase",
    "PC1_InnerCircle", "PC2_InnerCircle", "PC3_InnerCircle",
    "PC1_MiddleCircle", "PC2_MiddleCircle", "PC3_MiddleCircle",
    "PC1_OuterCircle", "PC2_OuterCircle", "PC3_OuterCircle"
]
# *****************************************************************

# === 1. Load Dataset & Setup (WILL NOW USE NEW feature_cols) ===
try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found. Place it in the current directory.")
    exit()

# This line is crucial: it selects the data based on the *new* feature_cols
data = data.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
X = data[feature_cols]
y = data[target_cols]

os.makedirs("plots", exist_ok=True)

# === 2. Train/Test Split ===
X_train, X_test, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Target Normalization (Scaling to [0, 1]) ===
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train_orig)
y_test_scaled = y_scaler.transform(y_test_orig)
y_train = pd.DataFrame(y_train_scaled, columns=target_cols, index=y_train_orig.index)
y_test = pd.DataFrame(y_test_scaled, columns=target_cols, index=y_test_orig.index)
print(f"Target values normalized to range [0, 1] using MinMaxScaler.")

# === 4. Multi-Output 10-Fold CV (Skipping re-run for brevity, but note models would be re-trained here) ===

# =================================================================
# === 5. SEQUENTIAL CHAINED MODELS (RETRAINED with NEW FEATURES) ===
# =================================================================

# --- Model A (RF_3): Predict Part3_E (from X) ---
rf_part3 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part3.fit(X_train, y_train["Part3_E"])
part3_pred_test = rf_part3.predict(X_test)

# --- Model B (RF_11): Predict Part11_E (from X + Part3_E) ---
X_train_part11 = X_train.copy()
X_train_part11["Part3_E_feature"] = y_train["Part3_E"]
X_test_part11 = X_test.copy()
X_test_part11["Part3_E_feature"] = part3_pred_test
rf_part11_chained = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part11_chained.fit(X_train_part11, y_train["Part11_E"])
part11_pred_test = rf_part11_chained.predict(X_test_part11)

# --- Model C (RF_1): Predict Part1_E (from X + Part3_E + Part11_E) ---
X_train_part1 = X_train.copy()
X_train_part1["Part3_E_feature"] = y_train["Part3_E"]
X_train_part1["Part11_E_feature"] = y_train["Part11_E"]
X_test_part1 = X_test.copy()
X_test_part1["Part3_E_feature"] = part3_pred_test
X_test_part1["Part11_E_feature"] = part11_pred_test
rf_part1_chained = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_part1_chained.fit(X_train_part1, y_train["Part1_E"])
part1_pred_test = rf_part1_chained.predict(X_test_part1)
# ... rest of the metric printing and plotting (omitted for brevity) ...

# =================================================================
# === 7. Prediction on New, Noisy Dataset (SCALED Input, SCALED Output) ===
# =================================================================

NOISY_DATA_FILE = "noisy_with_pca_from_clean_colored.csv"
print(f"\nAttempting prediction on new noisy dataset: '{NOISY_DATA_FILE}'...")

try:
    # --- Load Noisy Data ---
    data_noisy = pd.read_csv(NOISY_DATA_FILE)
    # This must use the *new* feature_cols, which are assumed to be in the file
    X_noisy = data_noisy[feature_cols]

    # --- 7.1. Chained Prediction on Noisy Data ---

    # Model A (RF_3): Predict Part3_E (from X_noisy)
    # Uses the *newly trained* rf_part3 model
    part3_pred_noisy = rf_part3.predict(X_noisy) # Prediction is SCALED

    # Model B (RF_11): Predict Part11_E (from X_noisy + Part3_E_predicted)
    X_noisy_part11 = X_noisy.copy()
    X_noisy_part11["Part3_E_feature"] = part3_pred_noisy

    part11_pred_noisy = rf_part11_chained.predict(X_noisy_part11) # Prediction is SCALED

    # Model C (RF_1): Predict Part1_E (from X_noisy + Part3_E_predicted + Part11_E_predicted)
    X_noisy_part1 = X_noisy.copy()
    X_noisy_part1["Part3_E_feature"] = part3_pred_noisy
    X_noisy_part1["Part11_E_feature"] = part11_pred_noisy

    part1_pred_noisy = rf_part1_chained.predict(X_noisy_part1) # Prediction is SCALED

    # --- 7.2. Consolidate and Inverse Transform Predictions ---
    predictions_noisy_scaled = pd.DataFrame({
        "Part1_E": part1_pred_noisy,
        "Part3_E": part3_pred_noisy,
        "Part11_E": part11_pred_noisy
    })

    # Inverse transform to get predictions in the original scale
    predictions_noisy_original = pd.DataFrame(
        y_scaler.inverse_transform(predictions_noisy_scaled),
        columns=target_cols
    )

    print("Prediction on noisy data successful.")
    print("\nSample of Predictions (Original Scale) for Noisy Data:")
    print(predictions_noisy_original.head())
    print(f"\nTotal {len(predictions_noisy_original)} predictions generated for the noisy dataset.")

    # --- 7.3. Save Predictions ---
    predictions_noisy_original.to_csv("noisy_data_predictions.csv", index=False)
    print("Predictions saved to 'noisy_data_predictions.csv'.")


except FileNotFoundError:
    print(f"Warning: '{NOISY_DATA_FILE}' not found. Cannot perform prediction on noisy data.")
except KeyError as e:
    # This will catch errors if the new features are not in one of the files
    print(f"Error: Required feature column {e} not found in one of the data files. Check that 'original_with_pca.csv' and '{NOISY_DATA_FILE}' both contain the new feature columns.")

# --- END OF SCRIPT EXTENSION ---