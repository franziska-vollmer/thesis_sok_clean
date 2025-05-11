import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Alle .pkl-Dateien laden
print("🔍 Lade .pkl-Dateien...")
folder_path = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
print(f"➡️  {len(pkl_files)} Dateien gefunden.")

print("📦 Lade und kombiniere Daten...")
dfs = [pd.read_pickle(f) for f in pkl_files]
df = pd.concat(dfs, ignore_index=True)
print(f"✅ Daten geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

# 2. Features/Labels trennen und normalisieren
print("🔧 Trenne Features und Labels...")
X_raw = df.drop(columns=[556])
y_raw = df[556].reset_index(drop=True)

print("📊 Normalisiere Features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. Autokorrelationsbasierte Fenstergrößenwahl
print("📈 Berechne optimale Fenstergröße mit Autokorrelation...")
mean_signal = X_scaled.mean(axis=1)

def get_acf_window_size(series, max_lag=50, conf_level=0.95):
    from scipy.stats import norm
    acf_values = acf(series, nlags=max_lag, fft=False)
    z = norm.ppf(1 - (1 - conf_level) / 2)
    conf_bound = z / np.sqrt(len(series))
    for lag in range(1, len(acf_values)):
        if np.abs(acf_values[lag]) < conf_bound:
            return lag
    return max_lag

window_size = get_acf_window_size(mean_signal)
print(f"✅ Optimale Fenstergröße: {window_size}")

# 4. Sequenzen mit Labels erstellen
print("📐 Erzeuge Eingabesequenzen für das Modell...")
def create_sequences_with_labels(data, labels, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences_with_labels(X_scaled, y_raw, window_size)
print(f"✅ Sequenzen erzeugt: {X_seq.shape[0]} Stück")

# 5. LSTM Autoencoder definieren und trainieren
print("⚙️  Trainiere LSTM-Autoencoder...")
input_dim = X_seq.shape[2]
timesteps = X_seq.shape[1]

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(64, activation='relu')(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(input_dim))(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_seq, X_seq, epochs=30, batch_size=64, validation_split=0.1)
print("✅ Training abgeschlossen.")

# 6. Anomalien erkennen
print("🔎 Erkenne Anomalien durch Rekonstruktionsfehler...")
X_pred = autoencoder.predict(X_seq)
mse = np.mean(np.mean(np.square(X_seq - X_pred), axis=2), axis=1)
threshold = np.percentile(mse, 95)
y_pred = np.where(mse > threshold, -1, 1)
print(f"📉 Anomalie-Schwelle (95. Perzentil): {threshold:.6f}")

# 7. Evaluation
print("📊 Bewertung der Ergebnisse:")
print("Confusion Matrix:\n", confusion_matrix(y_seq, y_pred))
print("\nClassification Report:\n", classification_report(y_seq, y_pred))

# 8. Visualisierung
print("📈 Visualisiere Rekonstruktionsfehler...")
plt.figure(figsize=(12, 4))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title("Reconstruction Error over Time")
plt.xlabel("Index")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()
