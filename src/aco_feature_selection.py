"""
ACO Feature Selection - Final PFE Version
NO DATA LEAKAGE - Proper methodology
Author: Islem Chenafi
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')


def load_data(sample_size=100000):
    """Load Edge-IIoTset dataset (no scaling here)"""
    print("Loading dataset...")
    
    path = "/home/islem/Documents/IIot_project/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
    
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    
    data = data.select_dtypes(include=[np.number])
    data = data.dropna()
    
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


def evaluate_feature_subset(selected_features, X_train, y_train, X_val, y_val, n_estimators=50):
    """Evaluate feature subset on validation set (no leakage)"""
    if len(selected_features) == 0:
        return 0.0
    
    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_val_selected)
    
    return f1_score(y_val, y_pred, average='weighted')


def aco_feature_selection(X_train, y_train, X_val, y_val, n_ants=20, n_iterations=10, n_features_select=20):
    """ACO for feature selection - no data leakage"""
    print("\n" + "="*80)
    print("ACO FEATURE SELECTION")
    print("="*80)
    print(f"Ants: {n_ants} | Iterations: {n_iterations} | Features per ant: {n_features_select}")
    print(f"Training samples: {X_train.shape[0]} | Validation samples: {X_val.shape[0]}")
    
    start_time = time.time()
    n_features = X_train.shape[1]
    
    alpha = 1.0
    beta = 2.0
    rho = 0.2
    Q = 1.0
    
    np.random.seed(42)
    pheromones = np.ones(n_features)
    
    print("\nComputing heuristic (feature importance from train data only)...")
    quick_model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    quick_model.fit(X_train, y_train)
    heuristic = quick_model.feature_importances_ + 1e-6
    
    best_features = None
    best_score = 0.0
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration+1}/{n_iterations}")
        iteration_best_score = 0.0
        iteration_best_sol = None
        
        for ant in range(n_ants):
            probs = (pheromones ** alpha) * (heuristic ** beta)
            probs = probs / probs.sum()
            
            selected = np.random.choice(
                n_features,
                size=n_features_select,
                replace=False,
                p=probs
            )
            
            score = evaluate_feature_subset(selected, X_train, y_train, X_val, y_val)
            
            if score > iteration_best_score:
                iteration_best_score = score
                iteration_best_sol = selected.copy()
            
            if score > best_score:
                best_score = score
                best_features = selected.copy()
                print(f"  -> New best! F1-Score: {best_score:.4f} | Features: {len(selected)}")
        
        print(f"  Iteration best: {iteration_best_score:.4f} | Global best: {best_score:.4f}")
        
        pheromones = (1 - rho) * pheromones
        if iteration_best_sol is not None:
            for f in iteration_best_sol:
                pheromones[f] += Q * iteration_best_score
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("ACO COMPLETED")
    print("="*80)
    print(f"Best F1-Score      : {best_score:.4f}")
    print(f"Features selected  : {len(best_features)} / {n_features}")
    print(f"Total time         : {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    
    return best_features, best_score
