import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

# === Einstellungen ===
TIME_STEPS = 10
MAX_ROWS = 5000  # Speicherbegrenzung für Training

# === Funktion: Daten laden ===
def load_data_from_pkls(folder_path):
    dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            path = os.path.join(folder_path, filename)
            df = pd.read_pickle(path)
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

# === Sliding-Window Funktion ===
def create_sequences(data, time_steps=TIME_STEPS):
    sequences = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i+time_steps])
    return np.array(sequences)

# === Modellarchitektur ===
def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, activation='relu')(inputs)
    x = RepeatVector(input_shape[0])(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(input_shape[1]))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mse')
    return model

# === Hauptfunktion ===
def main():
    # 1. Datenpfad setzen
    folder_path = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp"

    # 2. Daten laden
    df = load_data_from_pkls(folder_path)

    # 3. Features und Labels trennen
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 4. Normale Daten filtern
    X_normal = X[y == 1]
    if len(X_normal) > MAX_ROWS:
        idx = np.random.choice(len(X_normal), MAX_ROWS, replace=False)
        X_normal = X_normal[idx]

    # 5. Skalieren
    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_all_scaled = scaler.transform(X)

    # 6. Sequenzen
    X_train = create_sequences(X_normal_scaled)
    X_all_seq = create_sequences(X_all_scaled)
    y_seq = y[TIME_STEPS:]  # auf Sequenzlänge anpassen

    # 7. Modell trainieren
    model = build_autoencoder((TIME_STEPS, X_train.shape[2]))
    model.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.1, shuffle=True)

    # 8. Vorhersage & Fehler
    X_pred = model.predict(X_all_seq)
    errors = np.mean(np.abs(X_pred - X_all_seq), axis=(1, 2))

    # 9. Schwellwert festlegen (95%-Quantil)
    train_pred = model.predict(X_train)
    train_errors = np.mean(np.abs(train_pred - X_train), axis=(1, 2))
    threshold = np.percentile(train_errors, 95)
    print(f"Reconstruction Error Threshold: {threshold:.4f}")

    # 10. Anomalien erkennen
    y_pred = (errors > threshold).astype(int)

    # 11. Evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_seq, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_seq, y_pred))

    # 12. Plot
    plt.figure(figsize=(15, 4))
    plt.plot(errors, label='Reconstruction Error')
    plt.hlines(threshold, xmin=0, xmax=len(errors), colors='r', label='Threshold')
    plt.legend()
    plt.title("Reconstruction Error Over Time")
    plt.show()

if __name__ == "__main__":
    main()
