"""
PSO Feature Selection - Final PFE Version
NO DATA LEAKAGE - Proper train/test methodology
Author: Islem Chenafi
University: Université Ferhat Abbas Sétif-1
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


def load_data(sample_size=100000):
    """
    Load and prepare the Edge-IIoTset dataset
    NOTE: Does NOT scale here - scaling happens AFTER train/test split
    """
    print("Loading dataset...")
    
    path = "/home/islem/Documents/IIot_project/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
    
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    
    # Keep only numeric columns
    data = data.select_dtypes(include=[np.number])
    data = data.dropna()
    
    # Sample for faster training
    if sample_size is not None and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)
    
    print(f"Dataset shape after cleaning: {data.shape}")
    
    if 'Attack_label' not in data.columns:
        raise KeyError("Column 'Attack_label' not found in dataset!")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(data['Attack_label'])
    
    # Features (NOT scaled yet!)
    X = data.drop(columns=['Attack_label'])
    
    class_names = ['Normal', 'Attack']  # Binary classification
    
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Class distribution: Normal={np.sum(y==0)}, Attack={np.sum(y==1)}")
    
    return X.values, y, X.columns.tolist(), class_names


def evaluate_feature_subset(solution, X_train, y_train, X_val, y_val, n_estimators=50):
    """
    Evaluate a feature subset using Random Forest
    
    CRITICAL: X_train and X_val are already scaled and split!
    No data leakage here.
    
    Args:
        solution: Binary array (1 = select feature, 0 = ignore)
        X_train: Training data (already scaled)
        y_train: Training labels
        X_val: Validation data (scaled with train's scaler)
        y_val: Validation labels
        n_estimators: Number of trees in Random Forest
    
    Returns:
        accuracy: Validation accuracy
    """
    # Select features based on solution
    selected_indices = np.where(solution >= 0.5)[0]
    
    # Must select at least 1 feature
    if len(selected_indices) == 0:
        return 0.0
    
    # Extract selected features
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_selected, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val_selected)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy


def pso_feature_selection(X_train, y_train, X_val, y_val, n_particles=15, n_iterations=10):
    """
    Particle Swarm Optimization for feature selection
    
    Args:
        X_train: Training data (already scaled)
        y_train: Training labels
        X_val: Validation data (scaled with train's scaler)
        y_val: Validation labels
        n_particles: Number of particles in swarm
        n_iterations: Number of PSO iterations
    
    Returns:
        best_solution: Binary array of selected features
        best_score: Best accuracy achieved
        selected_indices: Indices of selected features
    """
    print("\n" + "="*80)
    print("PSO FEATURE SELECTION")
    print("="*80)
    print(f"Particles: {n_particles} | Iterations: {n_iterations}")
    print(f"Training samples: {X_train.shape[0]} | Validation samples: {X_val.shape[0]}")
    
    start_time = time.time()
    n_features = X_train.shape[1]
    
    # PSO hyperparameters
    w = 0.729   # Inertia weight
    c1 = 1.49445  # Cognitive coefficient
    c2 = 1.49445  # Social coefficient
    
    # Initialize particles and velocities
    np.random.seed(42)  # Reproducibility
    particles = np.random.uniform(0, 1, (n_particles, n_features))
    velocities = np.random.uniform(-0.5, 0.5, (n_particles, n_features))
    
    # Initialize personal best
    p_best_positions = particles.copy()
    p_best_scores = np.array([
        evaluate_feature_subset(p, X_train, y_train, X_val, y_val) 
        for p in particles
    ])
    
    # Initialize global best
    g_best_idx = np.argmax(p_best_scores)
    g_best_position = p_best_positions[g_best_idx].copy()
    g_best_score = p_best_scores[g_best_idx]
    
    print(f"Initial best accuracy: {g_best_score:.4f}")
    print(f"Initial features selected: {np.sum(g_best_position >= 0.5)}")
    
    # PSO iterations
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration+1}/{n_iterations}")
        
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.random(n_features)
            r2 = np.random.random(n_features)
            
            velocities[i] = (
                w * velocities[i] +
                c1 * r1 * (p_best_positions[i] - particles[i]) +
                c2 * r2 * (g_best_position - particles[i])
            )
            
            # Update position
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)
            
            # Evaluate fitness
            score = evaluate_feature_subset(particles[i], X_train, y_train, X_val, y_val)
            
            # Update personal best
            if score > p_best_scores[i]:
                p_best_scores[i] = score
                p_best_positions[i] = particles[i].copy()
            
            # Update global best
            if score > g_best_score:
                g_best_score = score
                g_best_position = particles[i].copy()
                n_features_selected = np.sum(g_best_position >= 0.5)
                print(f"  → New best! Accuracy: {g_best_score:.4f} | Features: {n_features_selected}")
        
        # Summary for this iteration
        avg_score = np.mean(p_best_scores)
        print(f"  Best: {g_best_score:.4f} | Average: {avg_score:.4f}")
    
    # Final results
    best_solution = (g_best_position >= 0.5).astype(int)
    selected_indices = np.where(best_solution == 1)[0]
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("PSO COMPLETED")
    print("="*80)
    print(f"Best validation accuracy : {g_best_score:.4f}")
    print(f"Features selected        : {len(selected_indices)} / {n_features}")
    print(f"Total time               : {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} min)")
    
    return best_solution, g_best_score, selected_indices


if __name__ == "__main__":
    """
    Test PSO feature selection independently
    """
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    X, y, feature_names, class_names = load_data(sample_size=50000)
    
    # Split FIRST (before any scaling!)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain: {X_train.shape[0]} | Validation: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    # Scale AFTER splitting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Data scaled (fit on train only, transform on val/test)")
    
    # Run PSO
    best_solution, best_score, selected_indices = pso_feature_selection(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        n_particles=15,
        n_iterations=8
    )
    
    # Test on held-out test set
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION")
    print("="*80)
    
    X_test_selected = X_test_scaled[:, selected_indices]
    X_train_selected = X_train_scaled[:, selected_indices]
    
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X_train_selected, y_train)
    
    y_pred = final_model.predict(X_test_selected)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"\nSelected features ({len(selected_indices)}):")
    for idx in selected_indices[:10]:  # Show first 10
        print(f"  - {feature_names[idx]}")
    if len(selected_indices) > 10:
        print(f"  ... and {len(selected_indices)-10} more")