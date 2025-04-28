import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === 1. Daten laden ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

df_all = pd.DataFrame()

for line in app_lines:
    app = line.strip()
    for i in [1, 2, 3, 4]:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        if not os.path.exists(path):
            print(f"[WARN] Datei fehlt: {path}")
            continue
        with open(path, 'rb') as f:
            df_temp = pickle.load(f)
        df_all = pd.concat([df_all, df_temp], ignore_index=True)

print("Gesamtdaten Shape:", df_all.shape)

# === 2. Features und Labels extrahieren ===
X = df_all.iloc[:, 1:-1].values.astype(np.float32)
y = df_all.iloc[:, -1].values
y = np.where(y == -1, 0, y)  # Map -1 → 0

print("Labelverteilung:", Counter(y))

# === 3. Train/Test Split & Skalierung ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. Reshape für LSTM ===
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# === 5. class_weight berechnen ===
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("[INFO] class_weight:", class_weight_dict)

# === 6. Modell definieren ===
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=False), input_shape=(1, X_train_lstm.shape[2])),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# === 7. Training ===
history = model.fit(
    X_train_lstm, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    class_weight=class_weight_dict
)

# === 8. Evaluation ===
y_pred_probs = model.predict(X_test_lstm).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

print("\n=== Evaluation ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
