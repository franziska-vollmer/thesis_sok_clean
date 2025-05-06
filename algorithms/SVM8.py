import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score
)

# --- PARAMETER ---
DATA_DIR = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
LABEL_COLUMN = 556
NUM_FOLDS = 5

# --- .pkl-DATEIEN LADEN ---
print(f"[INFO] Lade .pkl-Dateien aus: {DATA_DIR}")
dataframes = []

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".pkl"):
        path = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_pickle(path)
            dataframes.append(df)
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von {fname}: {e}")

print(f"[INFO] {len(dataframes)} Dateien erfolgreich geladen.")

# --- DATEN KOMBINIEREN ---
data_all = pd.concat(dataframes, ignore_index=True)
X = data_all.drop(columns=[LABEL_COLUMN])
y = data_all[LABEL_COLUMN]

print(f"[INFO] Gesamtdaten: {X.shape[0]} Zeilen, {X.shape[1]} Merkmale")

# --- FEATURES SKALIEREN ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- CROSS-VALIDATION ---
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
results = []

print(f"[INFO] Starte {NUM_FOLDS}-Fold Cross-Validation...\n")

for fold_idx, (train_index, test_index) in enumerate(skf.split(X_scaled, y), start=1):
    print(f"\n📂 Fold {fold_idx}: Beginne Verarbeitung...")
    fold_start = time.time()

    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"  → Trainingsdaten: {len(y_train)} | Klassenverteilung: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  → Testdaten     : {len(y_test)} | Klassenverteilung: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # --- SVM Training (ohne probability=True!) ---
    print(f"  ⏳ Trainiere SVM (ohne AUC)...")
    train_start = time.time()
    clf = SVC(kernel="rbf", C=1.0, gamma="scale")  # kein probability=True
    clf.fit(X_train, y_train)
    print(f"  ✅ Training abgeschlossen ({time.time() - train_start:.2f}s)")

    # --- Vorhersage ---
    print(f"  ⏳ Mache Vorhersage...")
    pred_start = time.time()
    y_pred = clf.predict(X_test)
    print(f"  ✅ Vorhersage abgeschlossen ({time.time() - pred_start:.2f}s)")

    # --- Metriken berechnen ---
    print(f"  🧮 Berechne Metriken (ohne AUC)...")
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    results.append({
        "Fold": fold_idx,
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "AUC": "N/A"
    })

    print(f"  ✅ F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    print(f"  🕒 Dauer für Fold {fold_idx}: {time.time() - fold_start:.2f} Sekunden")

# --- ERGEBNISSE TABELLE ---
df_results = pd.DataFrame(results)

print("\n📋 Ergebnisse der 5-Fold-Cross-Validation:")
print(df_results.to_string(index=False))

# --- DURCHSCHNITTE ---
print("\n📊 Durchschnittswerte über alle Folds:")
print(f"F1-Score     : {df_results['F1'].mean():.4f}")
print(f"Precision    : {df_results['Precision'].mean():.4f}")
print(f"Recall       : {df_results['Recall'].mean():.4f}")
print(f"AUC          : N/A (deaktiviert für schnelleren Lauf)")

# --- SPEICHERN ALS CSV ---
out_path = "crossval_all_combined_no_auc.csv"
df_results.to_csv(out_path, index=False)
print(f"\n[INFO] Ergebnisse gespeichert in: {out_path}")
