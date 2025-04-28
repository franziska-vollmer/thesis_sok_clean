import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tqdm import tqdm

# === Pfade definieren ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Der aktuelle Ordner des Skripts
DATA_DIR = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp')
APPS_FILE = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')  # Pfad zu den CVE-Dateien
DATA_FILES_DIR = os.path.join(BASE_DIR, '../data-files')  # Pfad zum Ordner mit den CSV-Dateien

# === Lese App-Liste ===
if not os.path.exists(APPS_FILE):
    raise FileNotFoundError(f"📂 Fehlender Pfad: {APPS_FILE}")

with open(APPS_FILE, 'r') as f:
    apps = [line.strip() for line in f.readlines() if line.strip()]

print(f"📦 Anzahl CVEs: {len(apps)}")

# === Daten laden ===
X_all = []
y_all = []

print("\n📥 Lade .pkl-Dateien und zugehörige .csv-Dateien...")
for app in tqdm(apps, desc="📂 Lade Daten"):
    for i in [1, 2, 3, 4]:
        # Lade .pkl-Datei
        path_pkl = os.path.join(DATA_DIR, f"{app}-{i}.pkl")
        if not os.path.exists(path_pkl):
            print(f"⚠️ Datei nicht gefunden: {path_pkl}")
            continue
        with open(path_pkl, 'rb') as f:
            df = pickle.load(f)
            features = df.drop(index=556).astype(float).T.values
            labels = df.loc[556].apply(lambda x: int(float(x))).values

            # Lade die zugehörige CSV-Datei für Feature-Namen
            # Angepasster Pfad zu den CSV-Dateien
            path_csv = os.path.join(DATA_FILES_DIR, f'{app}/{app}-{i}_freqvector_full.csv')

            if not os.path.exists(path_csv):
                print(f"⚠️ Datei nicht gefunden: {path_csv}")
                continue
            print(f"✅ Lade CSV-Datei für {app}, Index {i}: {path_csv}")
            df_csv = pd.read_csv(path_csv, index_col=0)  # Falls erste Spalte der Index ist
            feature_names = list(df_csv.columns)

            # Falls notwendig, kürzen oder auffüllen:
            min_len = min(len(feature_names), features.shape[1])
            feature_names = feature_names[:min_len]

            # Speichere Features und Labels
            X_all.append(features)
            y_all.append(labels)

if not X_all:
    raise ValueError("❌ Keine gültigen Daten geladen.")

# === Gemeinsame Länge sichern
min_len = min(x.shape[1] for x in X_all)
X_all = [x[:, :min_len] for x in X_all]
X = np.vstack(X_all)
y = np.concatenate(y_all)

# === Nur Klassen 0 und 1 verwenden
mask = np.isin(y, [0, 1])
X = X[mask]
y = y[mask]

print(f"\n🔢 Feature-Matrix: {X.shape}, Label-Verteilung: {np.bincount(y)}")

# === Lese Spaltennamen aus einer zugehörigen CSV-Datei ===
csv_sample_path = os.path.join(BASE_DIR, "../data-files/CVE-2023-23752", "CVE-2023-23752-1_freqvector_full.csv")
df_csv = pd.read_csv(csv_sample_path, index_col=0)  # Falls erste Spalte der Index ist
feature_names = list(df_csv.columns)

# Falls notwendig, kürzen oder auffüllen:
if len(feature_names) < min_len:
    feature_names += [f"F{i}" for i in range(len(feature_names), min_len)]
else:
    feature_names = feature_names[:min_len]

# === Skalierung ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Modelltraining ===
print("\n🌲 Trainiere RandomForest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, y)

# === Feature Importances ===
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# === Visualisierung ===
top_n = 20  # Wähle hier aus, wie viele Features du anzeigen möchtest
plt.figure(figsize=(12, 6))

# Feature-Namen zuordnen
top_labels = [feature_names[i] for i in indices[:top_n]]  # Feature-Namen zuordnen
plt.bar(range(top_n), importances[indices[:top_n]], align='center')
plt.xticks(range(top_n), top_labels, rotation=45)  # Label hinzufügen
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title(f"Top {top_n} Most Important Features according to RandomForest")
plt.tight_layout()
plt.grid(True)

# === Speichern im selben Ordner wie das Skript ===
output_path = os.path.join(BASE_DIR, "feature_importance_rf4.png")
plt.savefig(output_path)
print(f"\n💾 Diagramm gespeichert unter: {output_path}")
plt.show()

# === Optional: Feature Distributions anzeigen ===
plt.figure(figsize=(12, 6))
for i in range(top_n):
    feature_idx = indices[i]
    plt.subplot(4, 5, i + 1)
    plt.hist(X_scaled[:, feature_idx], bins=20, alpha=0.7)
    plt.title(f"Feature {top_labels[i]}")
    plt.tight_layout()

# === Speichern der Distribution der Features ===
output_dist_path = os.path.join(BASE_DIR, "feature_distributions.png")
plt.savefig(output_dist_path)
print(f"\n💾 Feature Distributions gespeichert unter: {output_dist_path}")
plt.show()

# === Optional: Report ===
y_pred = clf.predict(X_scaled)
print("\n📊 Classification Report:")
print(classification_report(y, y_pred, digits=4))