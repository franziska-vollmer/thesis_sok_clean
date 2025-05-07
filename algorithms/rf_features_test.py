import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# === Pfade anpassen ===
PKL_DIR = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
CSV_BASE_DIR = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/"

# === Alle PKL-Dateien finden ===
pkl_files = sorted([f for f in os.listdir(PKL_DIR) if f.endswith(".pkl")])
print(f"📦 Gefundene .pkl-Dateien: {len(pkl_files)}")

X_all = []
y_all = []
all_syscalls = set()
csv_data_by_file = {}

# === System Calls aus allen CSV-Dateien sammeln ===
print("\n🔍 Sammle alle System Calls...")
for filename in tqdm(pkl_files):
    cve_id = "-".join(filename.split("-")[:3])
    index = filename.split("-")[-1].replace(".pkl", "")
    csv_path = os.path.join(CSV_BASE_DIR, cve_id, f"{cve_id}-{index}_freqvector_full.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ CSV fehlt für {filename}")
        continue

    df_csv = pd.read_csv(csv_path, index_col=0)
    csv_data_by_file[filename] = df_csv
    all_syscalls.update(df_csv.columns.tolist())

# === System Call Index-Mapping
all_syscalls = sorted(all_syscalls)
syscall_to_idx = {sc: i for i, sc in enumerate(all_syscalls)}
print(f"✅ Gesamtzahl unterschiedlicher System Calls: {len(all_syscalls)}")

# === Daten vorbereiten ===
print("\n📥 Lade Daten & mappe Features...")
for filename in tqdm(pkl_files):
    path_pkl = os.path.join(PKL_DIR, filename)
    if not os.path.exists(path_pkl):
        continue

    with open(path_pkl, 'rb') as f:
        df = pickle.load(f)

        try:
            raw_labels = df.loc[556].apply(lambda x: int(float(x))).values
            mask = np.isin(raw_labels, [0, 1])
            labels = raw_labels[mask]
            features = df.drop(index=556).astype(float).T[mask]
        except Exception as e:
            print(f"❌ Fehler beim Parsen von {filename}: {e}")
            continue

        # Feature-Vektoren auf globalen Index mappen
        if filename not in csv_data_by_file:
            continue
        csv_cols = csv_data_by_file[filename].columns.tolist()
        mapped = np.zeros((features.shape[0], len(all_syscalls)))
        for i, col in enumerate(csv_cols):
            if col in syscall_to_idx:
                mapped[:, syscall_to_idx[col]] = features[:, i]

        X_all.append(mapped)
        y_all.append(labels)

# === Zusammenführen & validieren
if not X_all or not y_all:
    raise ValueError("❌ Keine gültigen Daten geladen.")

X = np.vstack(X_all)
y = np.concatenate(y_all)

print(f"\n🔢 Feature-Matrix: {X.shape}, Label-Verteilung: {np.bincount(y)}")

# === Skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Random Forest Training
print("\n🌲 Trainiere RandomForest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, y)

# === Wichtigste Features bestimmen
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20
top_syscalls = [all_syscalls[i] for i in indices[:top_n]]

plt.figure(figsize=(12, 6))
plt.bar(range(top_n), importances[indices[:top_n]], align='center')
plt.xticks(range(top_n), top_syscalls, rotation=45)
plt.xlabel("System Call")
plt.ylabel("Importance")
plt.title("Top 20 wichtigste System Calls (Random Forest)")
plt.tight_layout()
plt.grid(True)

# === Speichern
output_plot = os.path.join(os.getcwd(), "feature_importance_rf_all_cves4.png")
plt.savefig(output_plot)
plt.show()
print(f"\n💾 Wichtigkeit-Diagramm gespeichert: {output_plot}")

# === Klassifikationsbericht
y_pred = clf.predict(X_scaled)
print("\n📊 Classification Report:")
print(classification_report(y, y_pred, digits=4))
