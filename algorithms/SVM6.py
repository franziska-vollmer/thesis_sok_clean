import os
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_fscore_support
)

# === RSVM-Funktionen ===
def compute_class_centers(X, y):
    pos_center = X[y == 1].mean(axis=0)
    neg_center = X[y == -1].mean(axis=0)
    return pos_center, neg_center

def compute_weights(X, y, pos_center, neg_center, lambda_param=1.0):
    weights = []
    for xi, yi in zip(X, y):
        center = pos_center if yi == 1 else neg_center
        dist = np.linalg.norm(xi - center)
        weights.append(np.exp(-lambda_param * dist**2))
    return np.array(weights)

def train_rsvm(X, y, C=1.0, lambda_param=0.1, gamma='scale'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pos_center, neg_center = compute_class_centers(X_scaled, y)
    sample_weights = compute_weights(X_scaled, y, pos_center, neg_center, lambda_param)
    clf = SVC(C=C, kernel='rbf', gamma=gamma, probability=True)  # probability needed for ROC/PR
    clf.fit(X_scaled, y, sample_weight=sample_weights)
    return clf, scaler

# === Lade Daten ===
def load_all_pkl_files(directory):
    all_data = []
    pkl_files = sorted([f for f in os.listdir(directory) if f.endswith(".pkl")])
    for idx, filename in enumerate(pkl_files, 1):
        file_path = os.path.join(directory, filename)
        print(f"[{idx}/{len(pkl_files)}] Lade Datei: {filename}")
        try:
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
                if isinstance(obj, pd.DataFrame):
                    obj = obj.values
                all_data.append(obj)
        except Exception as e:
            print(f"⚠️ Fehler bei {filename}: {e}")
    return np.vstack(all_data)

# === Hauptteil ===
data_dir = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp"

print("🔍 Lade .pkl-Dateien...")
all_data = load_all_pkl_files(data_dir)

print("\n📊 Samplen...")
normal = all_data[all_data[:, -1] == 1]
anomal = all_data[all_data[:, -1] == -1]
normal_sample = resample(normal, n_samples=20000, random_state=42)
balanced_data = np.vstack([normal_sample, anomal])
np.random.shuffle(balanced_data)

X = balanced_data[:, :-1]
y = balanced_data[:, -1].astype(int)

# === 5-Fold CV mit erweiterten Metriken ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

print("\n🔁 Starte 5-Fold Cross-Validation...\n")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"🔹 Fold {fold}...")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf, scaler = train_rsvm(X_train, y_train, C=1.0, lambda_param=0.1)
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    y_score = clf.predict_proba(X_test_scaled)[:, np.where(clf.classes_ == -1)[0][0]]  # Score für Anomalie (-1)

    # Precision, Recall, F1 für Anomalieklasse
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[-1], average='binary'
    )

    # ROC-AUC und PR-AUC
    roc_auc = roc_auc_score((y_test == -1).astype(int), y_score)
    pr_auc = average_precision_score((y_test == -1).astype(int), y_score)

    fold_results.append({
        "Fold": fold,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

# === Ausgabe Tabelle ===
print("\n📊 Ergebnisse pro Fold:")
df_results = pd.DataFrame(fold_results)
print(df_results.to_string(index=False, float_format="%.4f"))

# Optional speichern
df_results.to_csv("rsvm_5fold_results.csv", index=False)
