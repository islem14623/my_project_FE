"""
Baseline Model - Final PFE Version
Random Forest + Chi-square Feature Selection
NO DATA LEAKAGE - Fair comparison with PSO/ACO/GA
Author: Islem Chenafi
University: Université Ferhat Abbas Sétif-1
"""

import numpy as np
import pandas as pd
import joblib
import json
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

warnings.filterwarnings('ignore')


def load_data(sample_size=None):
    """Load Edge-IIoTset dataset (no scaling here)"""
    print("Loading All dataset...")
    
    path = "/home/islem/Documents/IIot_project/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
    
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    
    # Keep only numeric columns
    data = data.select_dtypes(include=[np.number])
    data = data.dropna()
    
    # Sample (same size as PSO/ACO/GA for fair comparison!)
    if sample_size is not None and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)
    
    print(f"Dataset shape after cleaning: {data.shape}")
    
    if 'Attack_label' not in data.columns:
        raise KeyError("Column 'Attack_label' not found!")
    
    le = LabelEncoder()
    y = le.fit_transform(data['Attack_label'])
    
    X = data.drop(columns=['Attack_label'])
    
    class_names = ['Normal', 'Attack']
    
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Class distribution: Normal={np.sum(y==0)}, Attack={np.sum(y==1)}")
    
    return X.values, y, X.columns.tolist(), class_names


def main():
    pipeline_start = time.time()
    
    print("="*90)
    print("FINAL PFE BASELINE: Random Forest + Chi-square Feature Selection")
    print("="*90)
    print("Student: Islem Chenafi")
    print("Topic: Baseline for comparison with PSO/ACO/GA")
    print("="*90)
    
    # ========================================
    # STEP 1: Load Data (same size as PSO/ACO/GA)
    # ========================================
    print("\n[STEP 1] Loading dataset (same size as nature-inspired methods)...")
    X, y, feature_names, class_names = load_data(sample_size=None)
    
    # ========================================
    # STEP 2: Split data FIRST
    # ========================================
    print("\n[STEP 2] Splitting data (before any preprocessing)...")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train set      : {X_train.shape[0]} samples")
    print(f"✓ Validation set : {X_val.shape[0]} samples")
    print(f"✓ Test set       : {X_test.shape[0]} samples")
    
    # ========================================
    # STEP 3: Chi-square Feature Selection on TRAIN only
    # ========================================
    print("\n[STEP 3] Chi-square feature selection (on train data ONLY)...")
    
    fs_start = time.time()
    
    # MinMax scaler for chi-square (chi2 needs non-negative values)
    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(X_train)  # Fit on TRAIN only
    X_val_minmax = minmax_scaler.transform(X_val)
    X_test_minmax = minmax_scaler.transform(X_test)
    
    # Select 20 best features (same as ACO for fair comparison)
    selector = SelectKBest(score_func=chi2, k=20)
    selector.fit(X_train_minmax, y_train)  # Fit on TRAIN labels only
    
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    fs_time = time.time() - fs_start
    
    print(f"✓ Selected {len(selected_indices)} features in {fs_time:.2f} seconds")
    print(f"✓ Feature selection used TRAIN data only (no leakage!)")
    
    # ========================================
    # STEP 4: Apply feature selection
    # ========================================
    print("\n[STEP 4] Applying selected features...")
    
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    # ========================================
    # STEP 5: Standardize on train only
    # ========================================
    print("\n[STEP 5] Standardizing features (fit on train only)...")
    
    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train_selected)
    X_val_scaled = std_scaler.transform(X_val_selected)
    X_test_scaled = std_scaler.transform(X_test_selected)
    
    print(f"✓ Scaler fit on train only")
    print(f"✓ NO DATA LEAKAGE!")
    
    # ========================================
    # STEP 6: Train Random Forest
    # ========================================
    print("\n[STEP 6] Training Random Forest baseline...")
    
    train_start = time.time()
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - train_start
    
    print(f"✓ Training completed in {train_time:.2f} seconds")
    
    # ========================================
    # STEP 7: Validation Performance
    # ========================================
    print("\n[STEP 7] Validation performance...")
    
    y_val_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    
    # ========================================
    # STEP 8: Final Test Evaluation
    # ========================================
    print("\n" + "="*90)
    print("FINAL RESULTS - TEST SET (NEVER SEEN BEFORE)")
    print("="*90)
    
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Features Selected: {len(selected_indices)} / {len(feature_names)}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives (Normal correctly identified) : {cm[0,0]}")
    print(f"False Positives (Normal wrongly as Attack)   : {cm[0,1]}")
    print(f"False Negatives (Attack wrongly as Normal)   : {cm[1,0]}")
    print(f"True Positives (Attack correctly identified) : {cm[1,1]}")
    
    attack_recall = cm[1,1] / (cm[1,0] + cm[1,1])
    attack_precision = cm[1,1] / (cm[0,1] + cm[1,1])
    attack_f1 = 2 * (attack_precision * attack_recall) / (attack_precision + attack_recall)
    
    print(f"\nAttack Detection Metrics:")
    print(f"  Recall (Detection Rate)    : {attack_recall:.4f}")
    print(f"  Precision                  : {attack_precision:.4f}")
    print(f"  F1-Score                   : {attack_f1:.4f}")
    
    print(f"\nTop 10 Selected Features (by chi-square score):")
    feature_scores = selector.scores_[selected_indices]
    sorted_idx = np.argsort(feature_scores)[::-1][:10]
    for i, idx in enumerate(sorted_idx, 1):
        print(f"  {i:2d}. {selected_feature_names[idx]} (score: {feature_scores[idx]:.2f})")
    
    # ========================================
    # STEP 9: Save Everything
    # ========================================
    print("\n[STEP 9] Saving model and artifacts...")
    
    joblib.dump(model, "baseline_rf_model.pkl")
    joblib.dump(std_scaler, "baseline_scaler.pkl")
    joblib.dump(minmax_scaler, "baseline_minmax_scaler.pkl")
    joblib.dump(selected_indices, "baseline_selected_features.pkl")
    
    with open("baseline_selected_feature_names.txt", "w") as f:
        for name in selected_feature_names:
            f.write(f"{name}\n")
    
    # Save metrics for comparison
    final_metrics = {
        "algorithm": "Baseline (Chi-square + Random Forest)",
        "test_accuracy": float(test_accuracy),
        "val_accuracy": float(val_accuracy),
        "attack_recall": float(attack_recall),
        "attack_precision": float(attack_precision),
        "attack_f1": float(attack_f1),
        "features_selected": int(len(selected_indices)),
        "total_features": int(len(feature_names)),
        "feature_reduction_percent": float(100 - (len(selected_indices)/len(feature_names)*100)),
        "training_time_seconds": float(train_time),
        "feature_selection_time_seconds": float(fs_time),
        "total_time_seconds": float(time.time() - pipeline_start),
        "confusion_matrix": cm.tolist(),
        "selected_feature_indices": selected_indices.tolist(),
    }
    
    with open("baseline_final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print("✓ Model saved as 'baseline_rf_model.pkl'")
    print("✓ Scalers saved")
    print("✓ Selected features saved")
    print("✓ Metrics saved as 'baseline_final_metrics.json'")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*90)
    print("BASELINE COMPLETED SUCCESSFULLY")
    print("="*90)
    print(f"Final Test Accuracy : {test_accuracy*100:.2f}%")
    print(f"Features Reduced    : {len(feature_names)} → {len(selected_indices)} ({len(selected_indices)/len(feature_names)*100:.1f}% retained)")
    print(f"Training Time       : {train_time:.2f} seconds")
    print(f"Total Time          : {(time.time()-pipeline_start)/60:.2f} minutes")
    print("="*90)


if __name__ == "__main__":
    main()
