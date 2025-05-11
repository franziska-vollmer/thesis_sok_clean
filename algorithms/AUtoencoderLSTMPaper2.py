import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Alle .pkl-Dateien rekursiv laden ===
def load_all_pkl_recursive(folder_path):
    all_data = []
    print(f"🔍 Suche nach .pkl-Dateien in '{folder_path}' ...")
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".pkl"):
                full_path = os.path.join(root, filename)
                print(f"📂 Lade: {full_path}")
                df = pd.read_pickle(full_path)
                all_data.append(df)
    print(f"✅ Geladen: {len(all_data)} Dateien")
    return pd.concat(all_data, ignore_index=True)

# === 2. Daten laden ===
data = load_all_pkl_recursive("/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp")

# === 3. Vorverarbeitung ===
print("🧹 Starte Vorverarbeitung ...")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Labels umwandeln: 1 (normal) → 0, -1 (anomal) → 1
y = np.where(y == 1, 0, 1)

# Normalisierung
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Sliding Window ===
window_size = 50
X_windows, y_windows = [], []

for i in range(len(X_scaled) - window_size):
    X_windows.append(X_scaled[i:i + window_size])
    y_windows.append(int(np.any(y[i:i + window_size])))  # Fenster = anomal, wenn ≥1 Anomalie

X_windows = np.array(X_windows)
y_windows = np.array(y_windows)

# Training nur mit normalen Fenstern
X_train = X_windows[y_windows == 0]
X_test = X_windows
y_test = y_windows

print(f"📊 Fenster erstellt: {len(X_windows)} gesamt, {len(X_train)} fürs Training")

# === 5. LSTM Autoencoder Modell ===
def build_autoencoder(timesteps, features, latent_dim=64):
    inputs = Input(shape=(timesteps, features))
    encoded = LSTM(latent_dim)(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(features, return_sequences=True)(decoded)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
    return autoencoder

timesteps = window_size
features = X.shape[1]
model = build_autoencoder(timesteps, features)

# === 6. Training ===
print("🚀 Starte Training ...")
model.fit(X_train, X_train, epochs=5, batch_size=32, validation_split=0.1, shuffle=True)
print("✅ Training abgeschlossen.")

# === 7. Vorhersage ===
print("🔎 Berechne Rekonstruktionsfehler ...")
X_pred = model.predict(X_test)
mae = np.mean(np.abs(X_pred - X_test), axis=(1, 2))

# === 8. Schwelle setzen & Vorhersage berechnen ===
threshold = np.quantile(mae, 0.995)
y_pred = (mae > threshold).astype(int)

print(f"📏 Schwelle (99.5% Quantil): {threshold:.6f}")

# === 9. Fehlerverteilung visualisieren ===
plt.figure(figsize=(12, 5))
sns.histplot(mae[y_test == 0], color="green", label="Normal", bins=50)
sns.histplot(mae[y_test == 1], color="red", label="Anomaly", bins=50)
plt.axvline(threshold, color='black', linestyle='--', label=f'Schwelle = {threshold:.4f}')
plt.legend()
plt.title("Rekonstruktionsfehler-Verteilung")
plt.xlabel("MAE")
plt.ylabel("Anzahl")
plt.tight_layout()
plt.show()

# === 10. Evaluation ===
print("📈 Auswertung (Confusion Matrix & Report):")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 11. Einzelmetriken anzeigen ===
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n📊 Metriken (zusammengefasst):")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
