import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, roc_auc_score

# ----------- Einstellungen -------------
data_dir = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'

TEST_CVES = [
    "CVE-2022-26134", 
    "CVE-2023-23752",
    "CVE-2022-42889"
]  # 🔥 Hier deine Test-CVEs eintragen!

NORMAL_SAMPLES = 10000   # Anzahl normale Beispiele fürs Training
ANOMALY_SAMPLES = 10000  # Anzahl Anomalien fürs Training
KERNEL = 'rbf'
C = 1.0
# ----------------------------------------

# Alle .pkl-Dateien einlesen
all_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

train_files = [f for f in all_files if not any(cve in f for cve in TEST_CVES)]
test_files = [f for f in all_files if any(cve in f for cve in TEST_CVES)]

print(f"Train-Dateien: {len(train_files)}")
print(f"Test-Dateien: {len(test_files)}")

# Dateien laden
def load_files(file_list, data_dir):
    dfs = []
    for f in tqdm(file_list, desc="Lade PKL-Dateien"):
        dfs.append(pd.read_pickle(os.path.join(data_dir, f)))
    return pd.concat(dfs, ignore_index=True)

print("\nLade Trainingsdaten...")
train_data = load_files(train_files, data_dir)

print("\nLade Testdaten...")
test_data = load_files(test_files, data_dir)

# Labels einheitlich machen
train_data.iloc[:, -1] = train_data.iloc[:, -1].replace(-1, 0)
test_data.iloc[:, -1] = test_data.iloc[:, -1].replace(-1, 0)

# Features und Labels
X_train_full = train_data.iloc[:, :-1]
y_train_full = train_data.iloc[:, -1]

X_test_full = test_data.iloc[:, :-1]
y_test_full = test_data.iloc[:, -1]

print("\nTrain-Daten:")
print(f"Normale: {(y_train_full == 0).sum()} | Anomalien: {(y_train_full == 1).sum()}")

print("Test-Daten:")
print(f"Normale: {(y_test_full == 0).sum()} | Anomalien: {(y_test_full == 1).sum()}")

# Nur gewünschte Samples fürs Training ziehen
normal_idx = y_train_full[y_train_full == 0].sample(n=NORMAL_SAMPLES, random_state=42).index
anomaly_idx = y_train_full[y_train_full == 1].sample(n=ANOMALY_SAMPLES, random_state=42).index

selected_idx = normal_idx.union(anomaly_idx)
X_train_selected = X_train_full.loc[selected_idx]
y_train_selected = y_train_full.loc[selected_idx]

# 🚀 Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_full)

print("\nTrainiere SVM-Modell...")
model = SVC(kernel=KERNEL, C=C, probability=True)
model.fit(X_train_scaled, y_train_selected)

print("\nMache Vorhersagen auf Testdaten...")
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Ergebnisse
acc = accuracy_score(y_test_full, y_pred)
prec = precision_score(y_test_full, y_pred, zero_division=0)
f1 = f1_score(y_test_full, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test_full, y_proba)

print("\n=== Ergebnisse auf unbekannten Test-CVEs ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_full, y_pred, target_names=["Normal", "Anomalie"]))
