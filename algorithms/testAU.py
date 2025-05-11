import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# === Schritt 1: Dateien definieren ===
print("📁 Lade 10 festgelegte .pkl-Dateien...")

# Schritt 1: Laden der .pkl-Dateien
data_folder = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"  # <--- ANPASSEN!
selected_files = [
    "CVE-2023-23752-1.pkl",
    "CVE-2023-23752-2.pkl",
    "CVE-2023-23752-3.pkl",
    "CVE-2023-23752-4.pkl",
    "CVE-2022-26134-1.pkl",
    "CVE-2022-26134-2.pkl",
    "CVE-2022-26134-3.pkl",
    "CVE-2022-26134-4.pkl",
    "CVE-2022-42889-1.pkl",
    "CVE-2022-42889-2.pkl"
]

dfs = [pd.read_pickle(os.path.join(data_folder, f)) for f in selected_files]
df_all = pd.concat(dfs, ignore_index=True)
print(f"✅ {len(df_all)} Zeilen geladen.\n")

# === Schritt 2: Feature & Label ===
print("🔍 Trenne Features und Label...")
X_raw = df_all.iloc[:, 1:-1]
y_raw = df_all.iloc[:, -1].values
print(f"✅ Feature-Matrix: {X_raw.shape}, Label: {y_raw.shape}\n")

# === Schritt 3: Top 50 Feature-Auswahl ===
print("🧠 Berechne Mittelwert-Differenzen zwischen Klassen...")
mean_normal = X_raw[y_raw == 1].mean()
mean_anomal = X_raw[y_raw == -1].mean()
mean_diff = (mean_anomal - mean_normal).abs().sort_values(ascending=False)
top_features = mean_diff.head(50).index
X_selected = X_raw[top_features]
print(f"✅ Top 50 Features ausgewählt: {list(top_features[:5])}...\n")

# === Schritt 4: Skalierung ===
print("⚖️ Skaliere Features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
print("✅ Skalierung abgeschlossen.\n")

# === Schritt 5: Sliding-Window mit window_size = 30 ===
print("📦 Erzeuge Sequenzen mit Sliding-Window (Größe 30)...")
def create_sequences(X, y, window_size):
    sequences, labels = [], []
    for i in range(len(X) - window_size + 1):
        sequences.append(X[i:i + window_size])
        labels.append(y[i + window_size - 1])
    return np.array(sequences), np.array(labels)

window_size = 30
X_seq, y_seq = create_sequences(X_scaled, y_raw, window_size)
print(f"✅ {X_seq.shape[0]} Sequenzen erzeugt\n")

# === Schritt 6: Split & Filter ===
print("✂️ Splitte in Training & Test...")
X_train_full, X_test, y_train_full, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
X_train_auto = X_train_full[y_train_full == 1]
print(f"✅ Trainingsdaten (nur normal): {X_train_auto.shape}\n")

# === Schritt 7: Modell erstellen ===
print("🧠 Baue LSTM Autoencoder...")
timesteps = X_train_auto.shape[1]
n_features = X_train_auto.shape[2]

input_layer = Input(shape=(timesteps, n_features))
encoded = LSTM(64, activation='relu')(input_layer)
encoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
decoded = TimeDistributed(Dense(n_features))(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
print("✅ Modell kompiliert.\n")

# === Schritt 8: Training ===
print("🏋️‍♂️ Starte Training...")
autoencoder.fit(X_train_auto, X_train_auto, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
print("✅ Training abgeschlossen.\n")

# === Schritt 9: Fehler berechnen ===
print("🔎 Berechne Rekonstruktionsfehler auf Testdaten...")
X_test_pred = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - X_test_pred), axis=(1, 2))
print("✅ Fehler berechnet.\n")

# === Schritt 10: Threshold manuell (90. Perzentil) ===
print("🎯 Setze Threshold manuell (90. Perzentil)...")
y_test_binary = np.where(y_test == -1, 1, 0)
manual_threshold = np.percentile(reconstruction_error, 90)
print(f"✅ Threshold gesetzt auf: {manual_threshold:.6f}\n")

# === Schritt 11: Vorhersage & Bewertung ===
print("📈 Bewertung:")
y_pred = (reconstruction_error > manual_threshold).astype("int32")
precision = precision_score(y_test_binary, y_pred)
recall = recall_score(y_test_binary, y_pred)
f1 = f1_score(y_test_binary, y_pred)
roc = roc_auc_score(y_test_binary, reconstruction_error)

print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-Score:   {f1:.4f}")
print(f"ROC-AUC:    {roc:.4f}\n")

# === Schritt 12: Modell speichern ===
print("💾 Speichere Modell...")
autoencoder.save("lstm_autoencoder_manual_threshold.keras")
print("✅ Modell gespeichert als 'lstm_autoencoder_manual_threshold.keras'")
