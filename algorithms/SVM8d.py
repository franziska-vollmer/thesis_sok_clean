import os
import random
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# === PARAMETER ===
DATA_DIR = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
LABEL_COLUMN = 556
NUM_FOLDS = 5
SEED = 42  # für Reproduzierbarkeit

# === DATEIEN LADEN UND FÜR FOLDS AUFTEILEN ===
all_pkl_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]
print(f"[INFO] Insgesamt verfügbare Dateien: {len(all_pkl_files)}")

if len(all_pkl_files) < NUM_FOLDS:
    raise ValueError("Nicht genug Dateien für die gewünschte Anzahl an Folds.")

random.seed(SEED)
random.shuffle(all_pkl_files)
fold_files = np.array_split(all_pkl_files, NUM_FOLDS)

results = []

for fold_idx in range(NUM_FOLDS):
    print(f"\n📂 Fold {fold_idx + 1}/{NUM_FOLDS}")
    fold_start = time.time()

    test_files = fold_files[fold_idx]
    train_files = [f for i, fold in enumerate(fold_files) if i != fold_idx for f in fold]

    # Sicherheit: keine Datei doppelt verwenden
    overlap = set(test_files) & set(train_files)
    assert not overlap, f"🚨 Dateien im Test und Training gleichzeitig: {overlap}"

    print(f"  → Train-Dateien: {len(train_files)} | Test-Dateien: {len(test_files)}")

    def load_data(file_list):
        dfs = []
        for fname in file_list:
            path = os.path.join(DATA_DIR, fname)
            try:
                df = pd.read_pickle(path)
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ Fehler bei {fname}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    df_train = load_data(train_files)
    df_test = load_data(test_files)

    if df_train.empty or df_test.empty:
        print("❌ Fehler: Leere Trainings- oder Testdaten – Fold übersprungen.")
        continue

    # --- Features und Labels ---
    X_train = df_train.drop(columns=[LABEL_COLUMN])
    y_train = df_train[LABEL_COLUMN]
    X_test = df_test.drop(columns=[LABEL_COLUMN])
    y_test = df_test[LABEL_COLUMN]

    print(f"  → Trainingszeilen: {len(y_train)} | Testzeilen: {len(y_test)}")
    print(f"  → Klassen (Train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  → Klassen (Test) : {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # --- Skalieren ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Modelltraining ---
    print("  ⏳ Trainiere SVM...")
    clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    clf.fit(X_train_scaled, y_train)
    print("  ✅ Training abgeschlossen")

    # --- Vorhersage und Bewertung ---
    print("  ⏳ Mache Vorhersage...")
    y_pred = clf.predict(X_test_scaled)
    y_score = clf.decision_function(X_test_scaled)

    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    results.append({
        "Fold": fold_idx + 1,
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "ROC AUC": round(roc_auc, 4),
        "PR AUC": round(pr_auc, 4)
    })

    print(f"  ✅ F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, ROC AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f}")
    print(f"  🕒 Dauer: {time.time() - fold_start:.2f} Sekunden")

# === ERGEBNISSE ZUSAMMENFASSEN ===
df_results = pd.DataFrame(results)

print("\n📋 Ergebnisse pro Fold:")
print(df_results.to_string(index=False))

print("\n📊 Durchschnittswerte:")
print(f"F1-Score     : {df_results['F1'].mean():.4f}")
print(f"Precision    : {df_results['Precision'].mean():.4f}")
print(f"Recall       : {df_results['Recall'].mean():.4f}")
print(f"ROC AUC      : {df_results['ROC AUC'].mean():.4f}")
print(f"PR AUC       : {df_results['PR AUC'].mean():.4f}")

# === SPEICHERN ===
out_path = "filewise_crossval_results.csv"
df_results.to_csv(out_path, index=False)
print(f"\n💾 Ergebnisse gespeichert unter: {out_path}")
