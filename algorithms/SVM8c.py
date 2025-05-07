import os
import random
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# --- PARAMETER ---
DATA_DIR = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
LABEL_COLUMN = 556
NUM_FOLDS = 5
FILES_PER_FOLD = 50

# --- ALLE .pkl-DATEIEN ERMITTELN ---
all_pkl_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]
print(f"[INFO] Insgesamt verfügbare Dateien: {len(all_pkl_files)}")

results = []

for fold_idx in range(1, NUM_FOLDS + 1):
    print(f"\n📂 Fold {fold_idx}: Lade zufällig {FILES_PER_FOLD} Dateien...")
    fold_start = time.time()

    selected_files = random.sample(all_pkl_files, FILES_PER_FOLD)
    dataframes = []

    for fname in selected_files:
        path = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_pickle(path)
            dataframes.append(df)
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von {fname}: {e}")

    if not dataframes:
        print("❌ Keine gültigen Daten geladen – Fold übersprungen.")
        continue

    # --- Daten zusammenfügen ---
    data_all = pd.concat(dataframes, ignore_index=True)
    X = data_all.drop(columns=[LABEL_COLUMN])
    y = data_all[LABEL_COLUMN]

    print(f"  → Geladene Daten: {X.shape[0]} Zeilen")

    # --- Skalierung ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Stratified Split (1 Fold aus StratifiedKFold) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_idx)
    train_idx, test_idx = next(skf.split(X_scaled, y))

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"  → Trainingsdaten: {len(y_train)} | Testdaten: {len(y_test)}")
    print(f"  → Klassenverteilung (Train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  → Klassenverteilung (Test) : {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # --- SVM Training ---
    print(f"  ⏳ Trainiere SVM...")
    clf = SVC(kernel="rbf", C=1.0, gamma="scale")  # kein probability=True
    clf.fit(X_train, y_train)
    print(f"  ✅ Training abgeschlossen")

    # --- Vorhersage und Scores ---
    print(f"  ⏳ Mache Vorhersage und berechne Scores...")
    y_pred = clf.predict(X_test)
    y_score = clf.decision_function(X_test)

    # --- Metriken ---
    print(f"  🧮 Berechne Metriken...")
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    results.append({
        "Fold": fold_idx,
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "ROC AUC": round(auc, 4),
        "PR AUC": round(pr_auc, 4)
    })

    print(f"  ✅ F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, ROC AUC={auc:.4f}, PR AUC={pr_auc:.4f}")
    print(f"  🕒 Dauer für Fold {fold_idx}: {time.time() - fold_start:.2f} Sekunden")

# --- Ergebnisse zusammenfassen ---
df_results = pd.DataFrame(results)

print("\n📋 Ergebnisse der Folds:")
print(df_results.to_string(index=False))

# --- Durchschnitt berechnen ---
print("\n📊 Durchschnittswerte:")
print(f"F1-Score     : {df_results['F1'].mean():.4f}")
print(f"Precision    : {df_results['Precision'].mean():.4f}")
print(f"Recall       : {df_results['Recall'].mean():.4f}")
print(f"ROC AUC      : {df_results['ROC AUC'].mean():.4f}")
print(f"PR AUC       : {df_results['PR AUC'].mean():.4f}")

# --- Speichern ---
out_path = "sampled_folds_results_with_auc.csv"
df_results.to_csv(out_path, index=False)
print(f"\n[INFO] Ergebnisse gespeichert in: {out_path}")
