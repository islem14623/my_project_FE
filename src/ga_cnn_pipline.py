# FINAL_PFE_GA_CNN_FIXED.py
"""
FINAL PFE SUBMISSION
Genetic Algorithm Feature Selection + CNN
Fixed version
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# Import your GA function
from ga_feature_selection import genetic_algorithm
from pso_feature_selection import load_data

print("="*90)
print("FINAL PFE SUBMISSION - GENETIC ALGORITHM + CNN")
print("="*90)

# 1. Load Dataset
print("\n[1] Loading dataset...")
X, y, class_names = load_data(sample_size=None)

print(f"Total features: {X.shape[1]}")

# 2. Run Genetic Algorithm
print("\n[2] Running Genetic Algorithm Feature Selection...")

# Call GA and handle the return values safely
result = genetic_algorithm(X, y, n_features=X.shape[1], pop_size=20, generations=10)

# Debug: see what GA actually returns
print(f"Type of return value: {type(result)}")
print(f"Return value: {result}")

# Fix the return handling
if isinstance(result, tuple):
    if len(result) == 3:
        best_solution, best_score, selected_features = result
    elif len(result) == 2:
        best_solution, best_score = result
        selected_features = np.where(best_solution == 1)[0]
    else:
        selected_features = result[0] if len(result) > 0 else []
else:
    selected_features = result if isinstance(result, (list, np.ndarray)) else []

print(f"GA selected {len(selected_features)} features")

# 3. Prepare Data
print("\n[3] Preparing data with GA-selected features...")

X_selected = X.iloc[:, selected_features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 4. CNN Model
print("\n[4] Building CNN Model...")

model = keras.Sequential([
    keras.layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=(X_train_cnn.shape[1], 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

class_weight = {0: 1.0, 1: 3.0}

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

print("\nTraining CNN...")
history = model.fit(
    X_train_cnn, y_train,
    epochs=25,
    batch_size=128,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# Evaluation
y_pred = (model.predict(X_test_cnn, verbose=0) > 0.5).astype(int).flatten()

print("\n" + "="*90)
print("FINAL RESULTS - GENETIC ALGORITHM + CNN")
print("="*90)
print(f"Accuracy                    : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Features Selected by GA     : {len(selected_features)} / {X.shape[1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model.save("final_ga_cnn_model.keras")
print("\nModel saved successfully!")