import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Parameter ===
DATA_DIR = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
print(f"Verwende Verzeichnis: {DATA_DIR}")

file_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]
print(f"{len(file_paths)} .pkl-Dateien gefunden.")

# === Daten laden und nach Label trennen ===
normal_features = []
anomal_features = []

for path in file_paths:
    print(f"Lade Datei: {os.path.basename(path)}")
    df = pd.read_pickle(path)
    
    # Features (ohne Zeitspalte und ohne Label)
    features = df.iloc[:, 1:-1]
    labels = df.iloc[:, -1]

    n_normal = (labels == 1).sum()
    n_anomal = (labels == -1).sum()
    print(f"  Enthält {n_normal} normale und {n_anomal} anomale Zeilen.")

    normal_features.append(features[labels == 1])
    anomal_features.append(features[labels == -1])

# === Zusammenführen ===
print("Füge alle normalen und anomalen Daten zusammen...")
normal_all = pd.concat(normal_features)
anomal_all = pd.concat(anomal_features)

print(f"Gesamtanzahl normaler Zeilen: {len(normal_all)}")
print(f"Gesamtanzahl anomaler Zeilen: {len(anomal_all)}")

# === Mittelwerte berechnen ===
print("Berechne Mittelwerte der Features...")
normal_mean = normal_all.mean()
anomal_mean = anomal_all.mean()

# === Plot der ersten 50 Features ===
print("Zeige Plot mit Mittelwerten der ersten 50 Features...")
plt.figure(figsize=(14, 6))
plt.plot(normal_mean.values[:50], label='Label 1 (Normal)')
plt.plot(anomal_mean.values[:50], label='Label -1 (Anomal)')
plt.title("Durchschnittlicher Feature-Wertvergleich (erste 50 Features)")
plt.xlabel("Feature-Index")
plt.ylabel("Wert")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
