import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

# ------------------------
# Funktionen
# ------------------------

def load_and_scale_pkl(path):
    print(f"📥 Lade Datei: {path}")
    df = pd.read_pickle(path)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Label-Verteilung vorher
    label_counts = df.iloc[:, -1].value_counts().to_dict()
    print(f"   🔍 Zeitpunkte: {len(df)} | Label-Verteilung: {label_counts}")

    X_raw = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    df_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)
    df_scaled["label"] = y
    return df_scaled

def prepare_windows(df, window_length=100):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_windows, y_windows = [], []
    for i in range(len(X) - window_length + 1):
        x_win = X[i:i + window_length]
        y_win = y[i:i + window_length]
        X_windows.append(x_win)
        y_windows.append(-1 if np.any(y_win == -1) else 1)

    y_window_array = np.array(y_windows)
    win_counts = dict(zip(*np.unique(y_window_array, return_counts=True)))
    print(f"   🧱 Fenster erstellt: {len(X_windows)} | Fenster-Verteilung: {win_counts}")
    if win_counts.get(-1, 0) == 0:
        print("   ⚠️  Keine Anomalien in Fenstern enthalten!")
    return np.array(X_windows, dtype=np.float32), y_window_array

class EncDecAD:
    def __init__(self, input_dim, seq_len, latent_dim):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self._build_model()

    def _build_model(self):
        print(f"\n⚙️  Baue LSTM Autoencoder: seq_len={self.seq_len}, input_dim={self.input_dim}, latent_dim={self.latent_dim}")
        inputs = layers.Input(shape=(self.seq_len, self.input_dim))
        encoded = layers.LSTM(self.latent_dim)(inputs)
        encoded = layers.RepeatVector(self.seq_len)(encoded)
        decoded = layers.LSTM(self.latent_dim, return_sequences=True)(encoded)
        outputs = layers.TimeDistributed(layers.Dense(self.input_dim))(decoded)
        self.model = models.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X_train, epochs=30, batch_size=128):
        print(f"\n🚀 Starte Training auf {len(X_train)} normalen Fenstern")
        history = self.model.fit(X_train, X_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=0.1,
                                 shuffle=True,
                                 verbose=1)
        return history

    def predict(self, X):
        print(f"🔮 Berechne Rekonstruktionen für {len(X)} Fenster")
        return self.model.predict(X)

# ------------------------
# Hauptprozess
# ------------------------

folder_path = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"  # <-- Pfad anpassen
window_length = 100

all_X, all_y = [], []

for file in os.listdir(folder_path):
    if file.endswith(".pkl"):
        full_path = os.path.join(folder_path, file)
        df_scaled = load_and_scale_pkl(full_path)
        X_win, y_win = prepare_windows(df_scaled, window_length=window_length)
        all_X.append(X_win)
        all_y.append(y_win)

# Gesamtdaten
X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)

print(f"\n📊 Gesamtdaten: {X.shape[0]} Fenster | Label-Verteilung: {dict(zip(*np.unique(y, return_counts=True)))}")

X_train = X[y == 1]
X_test = X
y_test = y

model = EncDecAD(input_dim=X.shape[2], seq_len=X.shape[1], latent_dim=64)
history = model.fit(X_train, epochs=10, batch_size=128)

X_pred = model.predict(X_test)
mse = np.mean((X_pred - X_test) ** 2, axis=(1, 2))

# Schwellenwert wählen (z. B. 90. Perzentil der normalen Fenster)
threshold = np.percentile(mse[y == 1], 90)
print(f"\n📈 Schwellenwert (90. Perzentil): {threshold:.2e}")
y_pred = np.where(mse > threshold, -1, 1)

# Evaluation
print("\n📋 Evaluation:")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label=-1, zero_division=0))
print("Recall:   ", recall_score(y_test, y_pred, pos_label=-1, zero_division=0))
print("F1 Score: ", f1_score(y_test, y_pred, pos_label=-1, zero_division=0))

# Plot Rekonstruktionsfehler
plt.figure(figsize=(10, 5))
plt.hist(mse, bins=100, color='skyblue')
plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.2e}")
plt.title("Histogramm der Rekonstruktionsfehler (MSE)")
plt.xlabel("MSE")
plt.ylabel("Fensteranzahl")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
