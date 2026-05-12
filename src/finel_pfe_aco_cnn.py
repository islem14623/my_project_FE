# FINAL_PFE_ACO_CNN_FULL_DATASET.py
"""
FINAL PFE SUBMISSION VERSION
Ant Colony Optimization (ACO) + CNN
Using FULL or Very Large Dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# Import your ACO module
from aco_feature_selection import aco_feature_selection, load_data

print("="*100)
print("FINAL PFE SUBMISSION - ACO Feature Selection + CNN")
print("Using FULL Dataset (or large sample for practical training)")
print("="*100)

# ============================
# 1. Load Dataset - FULL or Large Sample
# ============================
print("\n[1] Loading dataset...")
# For final submission: Use None (full dataset) or a large number like 500000 or 1000000
#X, y, class_names = load_data(sample_size=None)        # ← Change to None for FULL dataset
X, y, class_names = load_data(sample_size=None)   # Use this during testing

print(f"Dataset shape: {X.shape}")
print(f"Total features before selection: {X.shape[1]}")
print(f"Classes: {class_names}")

# ============================
# 2. ACO Feature Selection
# ============================
print("\n[2] Running ACO Feature Selection...")
result = aco_feature_selection(X, y, n_ants=20, iterations=10)

# Safe extraction of selected features (handles tuple or array return)
if isinstance(result, tuple):
    # Take the first item that looks like feature indices
    for item in result:
        if isinstance(item, (list, np.ndarray)) and len(item) > 0:
            selected_features = np.array(item).flatten()
            break
    else:
        selected_features = np.array([])
else:
    selected_features = np.array(result).flatten()

print(f"ACO successfully selected {len(selected_features)} features")
print(f"Selected feature indices: {selected_features.tolist()}")

# ============================
# 3. Prepare Data with Selected Features
# ============================
print("\n[3] Preparing data with ACO-selected features...")
X_selected = X.iloc[:, selected_features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Reshape for CNN (samples, features, 1)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Training shape for CNN: {X_train_cnn.shape}")
print(f"Test shape for CNN: {X_test_cnn.shape}")

# ============================
# 4. Build CNN Model
# ============================
print("\n[4] Building CNN Model...")

model = keras.Sequential([
    keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same',
                        input_shape=(X_train_cnn.shape[1], 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    
    keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

class_weight = {0: 1.0, 1: 3.5}   # Slightly higher weight for Attack class

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
]

# ============================
# 5. Train the Model
# ============================
print("\n[5] Training CNN on selected features...")
history = model.fit(
    X_train_cnn, y_train,
    epochs=30,
    batch_size=256,                    # Larger batch size for full dataset
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# ============================
# 6. Final Evaluation
# ============================
print("\n" + "="*100)
print("FINAL RESULTS - ACO + CNN (Full/Large Dataset)")
print("="*100)

y_pred_prob = model.predict(X_test_cnn, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy                  : {accuracy*100:.2f}%")
print(f"Features Selected by ACO  : {len(selected_features)} / {X.shape[1]}")
print(f"Attack Recall             : {classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
model.save("final_aco_cnn_full_dataset.keras")
print("\nModel saved as 'final_aco_cnn_full_dataset.keras'")