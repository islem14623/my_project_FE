"""
Genetic Algorithm Feature Selection - Final PFE Version
NO DATA LEAKAGE - Proper methodology
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


def evaluate_individual(solution, X_train, y_train, X_val, y_val, n_estimators=50):
    """Evaluate a binary chromosome on validation set (no leakage)"""
    selected = np.where(solution == 1)[0]
    
    if len(selected) == 0:
        return 0.0
    
    X_train_selected = X_train[:, selected]
    X_val_selected = X_val[:, selected]
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_val_selected)
    
    return f1_score(y_val, y_pred, average='weighted')


def tournament_selection(population, scores, tournament_size=3):
    """Real tournament selection - picks best of K random individuals"""
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmax(scores[indices])]
    return population[best_idx].copy()


def genetic_algorithm(X_train, y_train, X_val, y_val, pop_size=20, n_generations=10,
                     crossover_rate=0.8, mutation_rate=0.1, elite_size=2):
    """
    Genetic Algorithm for feature selection
    
    Args:
        X_train: Training data (scaled, no leakage)
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        pop_size: Population size
        n_generations: Number of generations
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation per gene
        elite_size: Number of best individuals to keep
    
    Returns:
        best_solution: Binary array of selected features
        best_score: Best F1-score achieved
        selected_indices: Indices of selected features
    """
    print("\n" + "="*80)
    print("GENETIC ALGORITHM FEATURE SELECTION")
    print("="*80)
    print(f"Population: {pop_size} | Generations: {n_generations}")
    print(f"Crossover rate: {crossover_rate} | Mutation rate: {mutation_rate}")
    print(f"Training samples: {X_train.shape[0]} | Validation samples: {X_val.shape[0]}")
    
    start_time = time.time()
    n_features = X_train.shape[1]
    
    # Reproducibility
    np.random.seed(42)
    
    # Initialize population (each individual is a binary mask)
    population = np.random.randint(0, 2, (pop_size, n_features))
    
    # Make sure no individual is all zeros
    for i in range(pop_size):
        if np.sum(population[i]) == 0:
            population[i][np.random.randint(n_features)] = 1
    
    best_solution = None
    best_score = 0.0
    
    for generation in range(n_generations):
        print(f"\nGeneration {generation+1}/{n_generations}")
        
        # Evaluate all individuals
        scores = np.array([
            evaluate_individual(ind, X_train, y_train, X_val, y_val) 
            for ind in population
        ])
        
        # Track best
        gen_best_idx = np.argmax(scores)
        gen_best_score = scores[gen_best_idx]
        
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_solution = population[gen_best_idx].copy()
            n_selected = np.sum(best_solution)
            print(f"  -> New best! F1-Score: {best_score:.4f} | Features: {n_selected}")
        
        avg_score = np.mean(scores)
        print(f"  Best in gen: {gen_best_score:.4f} | Average: {avg_score:.4f} | Global best: {best_score:.4f}")
        
        # Elitism: keep top N best
        elite_indices = np.argsort(scores)[-elite_size:]
        new_population = [population[i].copy() for i in elite_indices]
        
        # Create offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)
            
            # Crossover (single point)
            if np.random.random() < crossover_rate:
                point = np.random.randint(1, n_features)
                child = np.concatenate([parent1[:point], parent2[point:]])
            else:
                child = parent1.copy()
            
            # Mutation (per gene)
            for i in range(n_features):
                if np.random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            # Ensure at least 1 feature is selected
            if np.sum(child) == 0:
                child[np.random.randint(n_features)] = 1
            
            new_population.append(child)
        
        population = np.array(new_population)
    
    elapsed = time.time() - start_time
    selected_indices = np.where(best_solution == 1)[0]
    
    print("\n" + "="*80)
    print("GENETIC ALGORITHM COMPLETED")
    print("="*80)
    print(f"Best F1-Score      : {best_score:.4f}")
    print(f"Features selected  : {len(selected_indices)} / {n_features}")
    print(f"Total time         : {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    
    return best_solution, best_score, selected_indices


if __name__ == "__main__":
    """Test GA independently"""
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    X, y, feature_names, class_names = load_data(sample_size=50000)
    
    # Split FIRST
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    # Scale AFTER
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Data scaled (no leakage)")
    
    # Run GA
    best_solution, best_score, selected_indices = genetic_algorithm(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        pop_size=20,
        n_generations=10
    )
    
    print(f"\n✓ GA selected {len(selected_indices)} features")
