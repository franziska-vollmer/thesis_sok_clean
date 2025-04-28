import os
import pickle
import pandas as pd
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

# === Pfade ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]

# === Lade CVE Liste ===
with open(apps_file_path, 'r') as f:
    all_apps = [line.strip() for line in f if line.strip()]

random.shuffle(all_apps)
fold_size = 2  # Anzahl Apps fürs Testset
train_apps = all_apps[fold_size:]
test_apps = all_apps[:fold_size]

print(f"🔁 Mini-Debug-Fold: {len(train_apps)} train, {len(test_apps)} test")

# === Daten laden ===
def load_apps(apps, indices, base_path):
    df = pd.DataFrame()
    for app in apps:
        for i in indices:
            path = os.path.join(base_path, f"{app}-{i}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, pd.DataFrame):
                        df = pd.concat([df, data], ignore_index=True)
                        print(f"✅ Geladen: {path}, {data.shape[0]} Zeilen")
                    else:
                        print(f"⚠️ Warnung: Nicht-DataFrame geladen ({type(data)}) von {path}")
            else:
                print(f"⚠️ Achtung: Datei nicht gefunden: {path}")
    return df

print("📦 Lade Trainingsdaten ...")
df_train = load_apps(train_apps, train_indices, supervised_path)

print("📦 Lade Testdaten ...")
df_test = load_apps(test_apps, test_indices, supervised_path)

print(f"✅ Daten geladen: {df_train.shape[0]} Trainingszeilen, {df_test.shape[0]} Testzeilen")

# === Features & Labels vorbereiten ===
X_train_raw = df_train.iloc[:, :-1].values
y_train_raw = df_train.iloc[:, -1].values

X_test_raw = df_test.iloc[:, :-1].values
y_test_raw = df_test.iloc[:, -1].values

# === Normalisieren ===
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# CNN erwartet (Samples, Features, 1)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# === CNN Modell bauen ===
input_shape = (X_train_scaled.shape[1], 1)

model = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binäre Klassifikation
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# === Modell trainieren ===
history = model.fit(X_train_scaled, y_train_raw, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test_raw))

# === Modell bewerten ===
loss, accuracy = model.evaluate(X_test_scaled, y_test_raw)
print(f"🎯 Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
