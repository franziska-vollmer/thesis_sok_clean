import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# === 1. Datenpfade ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

# === 2. Daten einlesen ===
df_all = pd.DataFrame()
with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

for line in app_lines:
    app = line.strip()
    for i in [1, 2, 3, 4]:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            df = pickle.load(f)
        df_all = pd.concat([df_all, df], ignore_index=True)

print("Gesamtdaten Shape:", df_all.shape)

# === 3. Stratified Sampling (10% von Klasse 1 + alle Anomalien) ===
df_major = df_all[df_all.iloc[:, -1] == 1]   # normale
df_minor = df_all[df_all.iloc[:, -1] == -1]  # anomalien

df_major_sample = df_major.sample(frac=0.1, random_state=42)
df_sample = pd.concat([df_major_sample, df_minor], ignore_index=True)

print("[INFO] Daten nach Sampling:", df_sample.shape)

# === 4. Features & Labels ===
X = df_sample.iloc[:, 1:-1].values.astype(np.float32)
y = df_sample.iloc[:, -1].values
y = np.where(y == -1, 0, y)  # anomalies = 0, normal = 1

print("Labelverteilung:", Counter(y))

# === 5. Train/Test Split & Skalierung ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. LinearSVC Training ===
print("[INFO] Training startet...")
svm = LinearSVC(class_weight='balanced', max_iter=20000, dual='auto')
svm.fit(X_train_scaled, y_train)
print("[INFO] Training abgeschlossen.")

# === 7. Evaluation ===
y_pred = svm.predict(X_test_scaled)

print("\n=== Evaluation ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
