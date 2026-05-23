
"""
PSO + CNN Pipeline - Final PFE Version
Complete pipeline with NO data leakage
Author: Islem Chenafi
University: Université Ferhat Abbas Sétif-1
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import PSO functions
from pso_feature_selection import load_data, pso_feature_selection

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def build_cnn_model(n_features, learning_rate=0.001):
    """
    Build CNN model for binary classification
    
    Args:
        n_features: Number of input features
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First convolutional block
        keras.layers.Conv1D(64, 3, activation='relu', padding='same', 
                           input_shape=(n_features, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.3),
        
        # Second convolutional block
        keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.3),
        
        # Dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        
        # Output layer
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    print("="*90)
    print("FINAL PFE PIPELINE: PSO FEATURE SELECTION + CNN")
    print("="*90)
    print("Student: Islem Chenafi")
    print("Topic: Feature Selection for IIoT Intrusion Detection")
    print("="*90)
    
    # ========================================
    # STEP 1: Load Data (NO scaling yet!)
    # ========================================
    print("\n[STEP 1] Loading dataset...")
    X, y, feature_names, class_names = load_data(sample_size=None)
    
    # ========================================
    # STEP 2: Split data FIRST (Critical!)
    # ========================================
    print("\n[STEP 2] Splitting data (before any preprocessing)...")
    
    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: train and validation (60% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train set      : {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"✓ Validation set : {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"✓ Test set       : {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # ========================================
    # STEP 3: Scale AFTER splitting
    # ========================================
    print("\n[STEP 3] Scaling features (fit on train only)...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)      # Fit + transform train
    X_val_scaled = scaler.transform(X_val)              # Only transform val
    X_test_scaled = scaler.transform(X_test)            # Only transform test
    
    print(f"✓ Scaler fit on {X_train_scaled.shape[0]} training samples")
    print(f"✓ Test set NEVER seen by scaler (no data leakage!)")
    
    # ========================================
    # STEP 4: PSO Feature Selection
    # ========================================
    print("\n[STEP 4] Running PSO for feature selection...")
    
    best_solution, best_pso_score, selected_indices = pso_feature_selection(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        n_particles=15,
        n_iterations=10
    )
    
    # Apply feature selection
    X_train_selected = X_train_scaled[:, selected_indices]
    X_val_selected = X_val_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]
    
    print(f"\n✓ Selected {len(selected_indices)} features out of {X_train_scaled.shape[1]}")
    
    # ========================================
    # STEP 5: Reshape for CNN (add channel dimension)
    # ========================================
    print("\n[STEP 5] Preparing data for CNN...")
    
    X_train_cnn = X_train_selected.reshape(X_train_selected.shape[0], X_train_selected.shape[1], 1)
    X_val_cnn = X_val_selected.reshape(X_val_selected.shape[0], X_val_selected.shape[1], 1)
    X_test_cnn = X_test_selected.reshape(X_test_selected.shape[0], X_test_selected.shape[1], 1)
    
    print(f"✓ CNN input shape: {X_train_cnn.shape}")
    
    # ========================================
    # STEP 6: Build and train CNN
    # ========================================
    print("\n[STEP 6] Building CNN model...")
    
    model = build_cnn_model(n_features=len(selected_indices))
    
    print(model.summary())
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Class weights (boost attack class)
    class_weight = {0: 1.0, 1: 2.5}
    
    print("\n[STEP 7] Training CNN...")
    print(f"Class weights: {class_weight}")
    
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=30,
        batch_size=128,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # ========================================
    # STEP 8: Final Evaluation on Test Set
    # ========================================
    print("\n" + "="*90)
    print("FINAL RESULTS - TEST SET (NEVER SEEN BEFORE)")
    print("="*90)
    
    # Predict
    y_pred_prob = model.predict(X_test_cnn, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Features Selected by PSO: {len(selected_indices)} / {len(feature_names)}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives (Normal correctly identified) : {cm[0,0]}")
    print(f"False Positives (Normal wrongly as Attack)   : {cm[0,1]}")
    print(f"False Negatives (Attack wrongly as Normal)   : {cm[1,0]}")
    print(f"True Positives (Attack correctly identified) : {cm[1,1]}")
    
    # Attack metrics
    attack_recall = cm[1,1] / (cm[1,0] + cm[1,1])
    attack_precision = cm[1,1] / (cm[0,1] + cm[1,1])
    attack_f1 = 2 * (attack_precision * attack_recall) / (attack_precision + attack_recall)
    
    print(f"\nAttack Detection Metrics:")
    print(f"  Recall (Detection Rate)    : {attack_recall:.4f}")
    print(f"  Precision                  : {attack_precision:.4f}")
    print(f"  F1-Score                   : {attack_f1:.4f}")
    
    # ========================================
    # STEP 9: Save Everything
    # ========================================
    print("\n[STEP 9] Saving model and artifacts...")
    
    model.save("final_pso_cnn_model.keras")
    joblib.dump(scaler, "pso_scaler.pkl")
    joblib.dump(selected_indices, "pso_selected_features.pkl")
    
    # Save feature names for reference
    selected_feature_names = [feature_names[i] for i in selected_indices]
    with open("pso_selected_feature_names.txt", "w") as f:
        for name in selected_feature_names:
            f.write(f"{name}\n")
    
    print("✓ Model saved as 'final_pso_cnn_model.keras'")
    print("✓ Scaler saved as 'pso_scaler.pkl'")
    print("✓ Selected features saved as 'pso_selected_features.pkl'")
    print("✓ Feature names saved as 'pso_selected_feature_names.txt'")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*90)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*90)
    print(f"Final Test Accuracy : {test_accuracy*100:.2f}%")
    print(f"Features Reduced    : {len(feature_names)} → {len(selected_indices)} ({len(selected_indices)/len(feature_names)*100:.1f}% retained)")
    print(f"Model Parameters    : {model.count_params():,}")
    print("\nAll artifacts saved. Ready for deployment!")
    print("="*90)


if __name__ == "__main__":
    main()
