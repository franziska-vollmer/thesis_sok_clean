import pandas as pd
import numpy as np
import os

# === Lokale Pfade ===
base_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/CVE-2022-26134'
csv_files = [
    'CVE-2022-26134-1_freqvector_full.csv',
    'CVE-2022-26134-2_freqvector_full.csv',
    'CVE-2022-26134-3_freqvector_full.csv',
    'CVE-2022-26134-4_freqvector_full.csv'
]
timing_file = 'timing.csv'

# === Funktionen ===

def load_features(csv_path):
    # CSV laden, ohne Header
    df = pd.read_csv(csv_path, header=None, low_memory=False)
    return df

def load_attack_times(timing_path):
    df_timing = pd.read_csv(timing_path, header=None, skiprows=1)  # erste Zeile überspringen!
    attack_times = []
    for idx, row in df_timing.iterrows():
        start_time = float(row[1])  # Start Timestamp = 2. Spalte
        end_time = float(row[2])    # End Timestamp = 3. Spalte
        attack_times.append((start_time, end_time))
    return attack_times

def assign_labels(df, attack_times):
    labels = []
    timestamps = df.iloc[:, 0].values  # erste Spalte = Timestamp
    timestamps = timestamps.astype(float)  # konvertiere zu float!
    for ts in timestamps:
        label = 0
        for (start, end) in attack_times:
            if start <= ts <= end:
                label = 1
                break
        labels.append(label)
    return labels

# === Hauptprozess ===

# Lade Timing-Information
attack_times = load_attack_times(os.path.join(base_path, timing_file))

# Initialisiere leere Listen
all_features = []
all_labels = []

# Lade jede Feature-Datei und erzeuge Labels
for file_name in csv_files:
    full_path = os.path.join(base_path, file_name)
    df = load_features(full_path)
    labels = assign_labels(df, attack_times)
    all_features.append(df)
    all_labels.extend(labels)

# Zusammenfügen
X = pd.concat(all_features, ignore_index=True).values
y = np.array(all_labels)

# Ausgabe
print(f"✅ Features Shape: {X.shape}")
print(f"✅ Labels Shape: {y.shape}")
print(f"🔍 Beispiel Labels: {y[:10]}")

# Speichere X und y falls du sie später brauchen willst
np.save(os.path.join(base_path, 'X_features.npy'), X)
np.save(os.path.join(base_path, 'y_labels.npy'), y)

print("📦 X und y wurden gespeichert!")
