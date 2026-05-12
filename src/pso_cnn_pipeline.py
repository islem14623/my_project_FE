# FINAL_PFE_PSO_CNN_BALANCED.py
"""
Final PFE Submission
PSO Feature Selection + CNN Classifier
Balanced version for good accuracy and reasonable false alarms
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from pso_feature_selection import load_data, pso_feature_selection

print("="*90)
print("FINAL PFE - PSO FEATURE SELECTION + CNN (Balanced)")
print("="*90)

# 1. Load Data
X, y, class_names = load_data(sample_size=None)

# 2. PSO Feature Selection
print("\nRunning PSO Feature Selection...")
best_solution, best_accuracy, selected_features = pso_feature_selection(
    X, y, n_particles=15, iterations=8
)

# 3. Prepare Selected Features
X_selected = X.iloc[:, selected_features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 4. CNN Model - Balanced
print("\nBuilding Balanced CNN Model...")

model = keras.Sequential([
    keras.layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=(X_train_cnn.shape[1], 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    
    keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.35),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

# Balanced class weight
class_weight = {0: 1.0, 1: 2.8}

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

print("\nTraining CNN Model...")
history = model.fit(
    X_train_cnn, y_train,
    epochs=25,
    batch_size=128,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# 5. Evaluation
y_pred = (model.predict(X_test_cnn, verbose=0) > 0.5).astype(int).flatten()

print("\n" + "="*90)
print("FINAL RESULTS - PSO + CNN")
print("="*90)
print(f"Accuracy                    : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Features Selected by PSO    : {len(selected_features)} / {X.shape[1]}")
print(f"Attack Recall               : {classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model.save("final_pso_cnn_balanced_model.keras")
print("\nModel saved as 'final_pso_cnn_balanced_model.keras'")