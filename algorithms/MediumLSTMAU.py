import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

# === Parameter ===
PKL_DIR = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp"
WINDOW_SIZE = 30
EPOCHS = 10
BATCH_SIZE = 32

print("🔍 Suche nach .pkl-Dateien im Ordner...")
pkl_files = sorted(glob.glob(os.path.join(PKL_DIR, "*.pkl")))
print(f"✅ {len(pkl_files)} Dateien gefunden.")

print("📥 Lade und kombiniere Daten...")
dataframes = [pd.read_pickle(p) for p in pkl_files]
df = pd.concat(dataframes, ignore_index=True)
print(f"📊 Gesamtanzahl Zeilen: {df.shape[0]}, Spalten: {df.shape[1]}")

print("🎯 Extrahiere Ground-Truth-Labels...")
true_labels = df.iloc[WINDOW_SIZE:, -1].replace({-1: 0, 1: 1}).values

print("🔄 Skaliere Feature-Daten...")
df_features = df.iloc[:, :-1]
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_features)

print("📏 Erzeuge Sequenzen für LSTM...")
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        window = data[i:i + window_size]
        sequences.append(window)
    return np.array(sequences)

X = create_sequences(scaled_df, WINDOW_SIZE)
print(f"📦 Sequenzdaten erzeugt: {X.shape}")

print("🧠 Baue LSTM Autoencoder-Modell...")
model = Sequential([
    LSTM(128, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    RepeatVector(X.shape[1]),
    LSTM(128, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X.shape[2]))
])
model.compile(optimizer='adam', loss='mse')

print("🏋️‍♀️ Starte Training...")
model.fit(X, X, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

print("🔎 Berechne Rekonstruktionsfehler...")
X_pred = model.predict(X)
mse = np.mean(np.mean(np.square(X - X_pred), axis=1), axis=1)

print("📊 Suche optimalen Threshold für beste F1-Score...")
best_f1 = 0
best_threshold = 0
for t in np.linspace(min(mse), max(mse), 100):
    preds = (mse > t).astype(int)
    f1 = f1_score(true_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("✅ Beste Schwelle gefunden:", best_threshold)

print("📈 Berechne finale Metriken...")
final_preds = (mse > best_threshold).astype(int)
precision = precision_score(true_labels, final_preds)
recall = recall_score(true_labels, final_preds)
conf_matrix = confusion_matrix(true_labels, final_preds)

print("\n📌 Ergebnisübersicht:")
print("Anzahl geladener Dateien:", len(pkl_files))
print("Bester Threshold:", best_threshold)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", best_f1)
print("Confusion Matrix:\n", conf_matrix)
