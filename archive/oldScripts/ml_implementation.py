# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 1. Load and Prepare Data ---
try:
    # IMPORTANT: Make sure this file name exactly matches your CSV file
    file_path = 'original_with_pca.csv'
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the script and the CSV file are in the same folder.")
    exit()


# --- Column Selection ---

# 1. Select Target Columns (y)
# Based on your request for columns B, C, and F.
# Please VERIFY that these are the correct names in your CSV header.
target_column_names = ["Part1_E", "Part3_E", "Part11_E"]
if not all(col in df.columns for col in target_column_names):
    print("Error: One or more target column names were not found in the file.")
    exit()
y = df[target_column_names]
print(f"\nFound {len(target_column_names)} Target Columns: {target_column_names}")

# 2. Select Feature Columns (X)
# Based on your request for columns CD through CL.
all_columns = df.columns.tolist()
# Excel's CD is the 82nd column, CL is the 90th.
# In Python's 0-based indexing, this is from index 81 up to (but not including) 90.
try:
    feature_column_names = all_columns[81:90]
    X = df[feature_column_names]
    print(f"Found {len(feature_column_names)} Feature Columns (CD-CL): {feature_column_names}")
except IndexError:
    print("Error: Could not select features from columns CD-CL. The file may not have enough columns.")
    exit()

# --- End of New Section ---

# --- Verify data ---

# 1. Visually inspect the first few rows
print("--- First 5 rows of features (X): ---")
print(X.head())
print("\n--- First 5 rows of targets (y): ---")
print(y.head())

# 2. Check the dimensions (rows, columns)
print(f"\nShape of features (X): {X.shape}") # Should be (num_rows, 9)
print(f"Shape of targets (y): {y.shape}")   # Should be (num_rows, 3)

# 3. Confirm the column names
print(f"\nFeature columns being used: {X.columns.tolist()}")
print(f"Target columns being predicted: {y.columns.tolist()}")

# --- End of verification block ---

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preparation complete. Starting model training...")
print("-" * 50)

# --- 2. Train and Evaluate Models ---

# Model 1: Multi-Output Random Forest (Great Baseline)
print("\nTraining Model 1: Random Forest...")
rf_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1) # Use all CPU cores
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"  -> 🎯 Random Forest MSE: {mse_rf:.6f}")

# Model 2: Neural Network (Most Powerful)
print("\nTraining Model 2: Neural Network (MLP)...")
nn_model = keras.Sequential([
    layers.Input(shape=[X_train_scaled.shape[1]]),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # Helps prevent overfitting
    layers.Dense(64, activation='relu'),
    layers.Dense(y_train.shape[1]) # Output layer
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
mse_nn = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"  -> Neural Network MSE: {mse_nn:.6f}")

# Model 3: Regressor Chain with XGBoost (Tests Recursive Hypothesis)
print("\nTraining Model 3: Regressor Chain (XGBoost)...")
base_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
# FIX: Changed 'base_estimator' to 'estimator' to match the latest scikit-learn version
chain_model = RegressorChain(estimator=base_model)
chain_model.fit(X_train_scaled, y_train)
y_pred_chain = chain_model.predict(X_test_scaled)
mse_chain = mean_squared_error(y_test, y_pred_chain)
print(f"  -> Regressor Chain MSE: {mse_chain:.6f}")

print("-" * 50)
print("\nAnalysis Complete. Compare the MSE scores above (lower is better).")

# # --- To test your hypothesis ---
# print("\n--- Testing with PC1 scores only ---")

# # Select only the PC1 feature columns
# pc1_features = ['PC1_Innershape', 'PC1_Outershape', 'PC1_Bottom']
# X_pc1_only = df[pc1_features]

# # Split the new feature set
# X_train_pc1, X_test_pc1, y_train_pc1, y_test_pc1 = train_test_split(X_pc1_only, y, test_size=0.2, random_state=42)

# # Train and evaluate a new model on just the PC1 data
# model_pc1 = RandomForestRegressor(n_estimators=100, random_state=42)
# model_pc1.fit(X_train_pc1, y_train_pc1)
# y_pred_pc1 = model_pc1.predict(X_test_pc1)
# r2_scores_pc1 = r2_score(y_test_pc1, y_pred_pc1, multioutput='raw_values')

# print("Model evaluation (PC1 only):")
# for i, col in enumerate(target_columns):
#     print(f"R^2 score for {col}: {r2_scores_pc1[i]:.4f}")