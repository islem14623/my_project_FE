# main.py - Baseline (Full Dataset)
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

print("="*70)
print("EDGE-IIOTSET INTRUSION DETECTION - BASELINE (FULL DATASET)")
print("="*70)

# ==============================
# 1. Load Dataset
# ==============================
print("\n[1] Loading full dataset...")
path = "/home/islem/Documents/IIot_project/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
data = pd.read_csv(path, low_memory=False)
data.columns = data.columns.str.strip()
print(f"Original shape: {data.shape}")

# ==============================
# 2. Cleaning (NO sampling - use full dataset)
# ==============================
print("\n[2] Cleaning data...")
data = data.dropna()
print(f"After cleaning: {data.shape}")
print(f"Using FULL dataset: {data.shape[0]:,} samples")

# ==============================
# 3. Labels
# ==============================
class_names = ['Normal', 'Attack']
le = LabelEncoder()
y = le.fit_transform(data['Attack_label'])
print(f"\nClasses: {class_names}")
print(f"Normal samples: {(y==0).sum():,}")
print(f"Attack samples: {(y==1).sum():,}")

# ==============================
# 4. Features Preparation
# ==============================
X = data.drop(columns=['Attack_label'])
X = X.select_dtypes(include=[np.number])
print(f"\nNumber of numeric features: {X.shape[1]}")

# ==============================
# 5. Feature Selection using chi2
# ==============================
print("\n[3] Applying Feature Selection (SelectKBest + chi2)...")
start_time = time.time()

minmax_scaler = MinMaxScaler()
X_scaled_for_chi2 = minmax_scaler.fit_transform(X)

selector = SelectKBest(score_func=chi2, k=20)
X_selected = selector.fit_transform(X_scaled_for_chi2, y)
selected_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_indices].tolist()

end_time = time.time()
print(f"Selected 20 best features using chi2")
print(f"Feature selection time: {end_time - start_time:.2f} seconds")

# ==============================
# 6. Train-Test Split + Scaling
# ==============================
print("\n[4] Splitting data and scaling...")
std_scaler = StandardScaler()
X_final = std_scaler.fit_transform(X.iloc[:, selected_indices])

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

# ==============================
# 7. Train Random Forest
# ==============================
print("\n[5] Training Random Forest Model on FULL dataset...")
print("This may take a few minutes...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)

start_train = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_train
print(f"Training completed in {train_time:.2f} seconds.")

# ==============================
# 8. Prediction & Evaluation
# ==============================
print("\n[6] Evaluating on test set...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("FINAL BASELINE RESULTS - FULL DATASET")
print("="*70)
print(f"Dataset size              : {data.shape[0]:,} samples")
print(f"Test set size             : {len(y_test):,} samples")
print(f"Accuracy                  : {accuracy*100:.2f}%")
print(f"Training Time             : {train_time:.2f} seconds")
print(f"Features Selected         : 20 / {X.shape[1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTop 10 Selected Features:")
for i, name in enumerate(selected_feature_names[:10], 1):
    print(f"{i:2d}. {name}")

print("\n" + "="*70)
print("DONE!")
print("="*70)