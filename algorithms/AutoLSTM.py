import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# === 1. Daten laden und zusammenführen ===
def load_data(pkl_dir):
    pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
    dfs = []
    print(f"🔍 Lade {len(pkl_files)} Dateien...")
    for f in tqdm(pkl_files, desc="📁 Dateien laden"):
        dfs.append(pd.read_pickle(f))
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"✅ Geladene Datenform: {df_all.shape}")
    return df_all

# === 2. Daten vorbereiten ===
def prepare_data(df, seq_len):
    print("🔄 Wandle numerische Spalten um...")
    df_numeric = df.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce').fillna(0)
    labels = df.iloc[:, -1]

    # Warnung bei ungewöhnlichen Labels
    unique_labels = labels.unique()
    if set(unique_labels) - {0, 1, -1}:
        print(f"⚠️ Achtung: Ungewöhnliche Labels gefunden: {set(unique_labels)}")

    # -1 → 0 konvertieren (Normaldaten)
    labels = labels.replace(-1, 0).astype(int)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_numeric)

    sequences = []
    sequence_labels = []

    print("🧩 Erzeuge Sequenzen...")
    for i in tqdm(range(len(scaled) - seq_len), desc="🔁 Sequenzen"):
        sequences.append(scaled[i:i+seq_len])
        sequence_labels.append(max(labels[i:i+seq_len]))

    return np.array(sequences), np.array(sequence_labels)

# === 3. Autoencoder-Modell definieren ===
def create_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, activation="relu")(inputs)
    x = RepeatVector(input_shape[0])(x)
    x = LSTM(64, activation="relu", return_sequences=True)(x)
    outputs = TimeDistributed(Dense(input_shape[1]))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

# === 4. Hauptprogramm ===
if __name__ == "__main__":
    SEQ_LEN = 30
    DATA_DIR = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"

    print("📁 Daten werden geladen...")
    df = load_data(DATA_DIR)

    print("📐 Daten werden vorbereitet...")
    X, y = prepare_data(df, SEQ_LEN)
    print(f"✅ Sequenzform: {X.shape}, Labels: {np.unique(y, return_counts=True)}")

    print("🧠 Modell wird erstellt...")
    model = create_autoencoder(X.shape[1:])
    model.summary()

    print("🚀 Training startet...")
    history = model.fit(
        X, X,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        shuffle=True
    )
    print("✅ Training abgeschlossen.")

    print("📊 Berechne Rekonstruktionsfehler...")
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=(1, 2))

    # Threshold anhand normaler Daten (Label == 0)
    threshold = np.percentile(mse[y == 0], 95)
    y_pred = (mse > threshold).astype(int)

    print("=== Klassifikationsbericht ===")
    print(classification_report(y, y_pred, target_names=["Normal", "Anomalie"]))

    # Fehlerverteilung visualisieren
    plt.figure(figsize=(10, 5))
    plt.hist(mse[y == 0], bins=100, alpha=0.6, label="Normal")
    plt.hist(mse[y == 1], bins=100, alpha=0.6, label="Anomalie")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}")
    plt.title("Verteilung der Rekonstruktionsfehler")
    plt.xlabel("MSE")
    plt.ylabel("Anzahl")
    plt.legend()
    plt.tight_layout()
    plt.show()
