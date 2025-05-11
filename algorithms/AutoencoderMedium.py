import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Schritt 1: Dateien laden
print("📂 Lade .pkl-Dateien ...")
data_path = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
all_data = []
for file in os.listdir(data_path):
    if file.endswith(".pkl"):
        print(f"  → Lade Datei: {file}")
        df = pd.read_pickle(os.path.join(data_path, file))
        all_data.append(df)

# Schritt 2: Daten zusammenführen
print("🧮 Füge Daten zusammen ...")
data = pd.concat(all_data).reset_index(drop=True)
print(f"✅ Gesamtdatensatzgröße: {data.shape}")

# Schritt 3: Label und Features trennen
label_col = data.columns[-1]
labels = data[label_col].values
features = data.drop(columns=[label_col])

# Schritt 4: Normalisierung
print("🔧 Normalisiere Features ...")
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Schritt 5: Sequenzen für LSTM erzeugen
def create_sequences(data, labels, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length - 1])
    return np.array(X), np.array(y)

print("🔁 Erzeuge Sequenzen für LSTM ...")
X, y = create_sequences(features_scaled, labels, seq_length=10)

# Schritt 6: Trainings- und Testdaten vorbereiten
print("🧪 Trenne Trainings- und Testdaten ...")
X_train = X[y == 1]  # Nur normale Daten für Training
X_test = X
y_test = y

# Schritt 7: Autoencoder definieren
def autoencoder_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, activation="relu", return_sequences=True)(inputs)
    x = LSTM(16, activation="relu")(x)
    x = RepeatVector(input_shape[0])(x)
    x = LSTM(16, activation="relu", return_sequences=True)(x)
    x = LSTM(64, activation="relu", return_sequences=True)(x)
    outputs = TimeDistributed(Dense(input_shape[1]))(x)
    model = Model(inputs, outputs)
    return model

print("🧠 Baue und kompiliere das LSTM Autoencoder Modell ...")
model = autoencoder_model(X_train.shape[1:])
model.compile(optimizer="adam", loss="mae")

# Schritt 8: Modell trainieren
print("🚀 Starte Training ...")
model.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
print("✅ Training abgeschlossen")

# Schritt 9: Vorhersage für Testdaten
print("🔍 Berechne Rekonstruktionsfehler auf Testdaten ...")
X_pred = model.predict(X_test)
mae_loss = np.mean(np.abs(X_pred - X_test), axis=(1,2))

# Schritt 10: Schwellenwert setzen
print("📏 Berechne Schwellenwert aus Trainingsdaten ...")
X_pred_train = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_pred_train - X_train), axis=(1,2))
threshold = np.percentile(train_mae_loss, 95)
print(f"⚙️ Verwendeter Schwellenwert: {threshold:.5f}")

# Schritt 11: Anomalien klassifizieren
y_pred = (mae_loss > threshold).astype(int)   # 1 = Anomalie
y_test_true = (y_test == -1).astype(int)      # 1 = Anomalie

# Schritt 12: Metriken berechnen
print("📊 Berechne Precision, Recall und F1-Score ...")
precision = precision_score(y_test_true, y_pred)
recall = recall_score(y_test_true, y_pred)
f1 = f1_score(y_test_true, y_pred)

print("\n📈 Ergebnis:")
print("   Precision:", round(precision, 4))
print("   Recall:   ", round(recall, 4))
print("   F1-Score: ", round(f1, 4))
