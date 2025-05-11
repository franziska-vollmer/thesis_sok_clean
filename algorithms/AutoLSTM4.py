import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# === 1. Lade alle .pkl-Dateien ===
DATA_DIR = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"  # <-- <--- ANPASSEN
all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]

print(f"Lade {len(all_files)} .pkl-Dateien...")
dataframes = []
for i, file in enumerate(all_files):
    df = pd.read_pickle(file)
    dataframes.append(df)
    if (i + 1) % 10 == 0 or i == len(all_files) - 1:
        print(f"{i + 1} Dateien geladen...")

# === 2. Kombinieren & vorbereiten ===
print("Kombiniere Daten...")
combined_df = pd.concat(dataframes, ignore_index=True)

feature_columns = combined_df.columns[:-1]
label_column = combined_df.columns[-1]

print("Konvertiere Features zu float...")
combined_df[feature_columns] = combined_df[feature_columns].apply(pd.to_numeric, errors='coerce')
combined_df = combined_df.dropna()

features = combined_df[feature_columns].astype(float).values
labels = combined_df[label_column].astype(int).values

# === 3. Normalisieren ===
print("Skaliere Daten mit MinMaxScaler...")
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# === 4. Trainingssequenzen erstellen ===
SEQ_LEN = 10
print(f"Erstelle Sequenzen (SEQ_LEN = {SEQ_LEN}) aus normalen Daten...")
X_train, y_train = [], []
for i in range(len(features_scaled) - SEQ_LEN):
    label_seq = labels[i:i+SEQ_LEN]
    if np.all(label_seq == 1):
        X_train.append(features_scaled[i:i+SEQ_LEN])
        y_train.append(1)
    if (i + 1) % 5000 == 0 or i == len(features_scaled) - SEQ_LEN - 1:
        print(f"{i + 1} Sequenzen geprüft...")

X_train = np.array(X_train)

# === 5. Modell definieren ===
input_dim = X_train.shape[2]
print("Baue LSTM-Autoencoder Modell...")
inputs = Input(shape=(SEQ_LEN, input_dim))
encoded = LSTM(64, activation='relu')(inputs)
decoded = RepeatVector(SEQ_LEN)(encoded)
decoded = LSTM(input_dim, activation='relu', return_sequences=True)(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

# === 6. Training ===
print("Starte Modelltraining...")
autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, validation_split=0.1)

# === 7. Testsequenzen erstellen ===
print("Erstelle Testsequenzen (auch mit Anomalien)...")
X_all, y_all = [], []
for i in range(len(features_scaled) - SEQ_LEN):
    seq = features_scaled[i:i+SEQ_LEN]
    label_seq = labels[i:i+SEQ_LEN]
    X_all.append(seq)
    y_all.append(int(np.any(label_seq == -1)))  # 1 wenn Anomalie enthalten
    if (i + 1) % 5000 == 0 or i == len(features_scaled) - SEQ_LEN - 1:
        print(f"{i + 1} Testsequenzen erstellt...")

X_all = np.array(X_all)
y_all = np.array(y_all)

# === 8. Vorhersage und MSE-Berechnung ===
print("Berechne Modellvorhersage...")
X_pred = autoencoder.predict(X_all)
print("Berechne Rekonstruktionsfehler...")
mse = np.mean(np.power(X_all - X_pred, 2), axis=(1, 2))

threshold = np.percentile(mse, 95)
print(f"Verwendeter Schwellenwert (95. Perzentil): {threshold:.6f}")

y_pred = (mse > threshold).astype(int)

# === 9. Evaluation ===
print("\n=== Klassifikationsbericht ===")
print(classification_report(y_all, y_pred, target_names=["Normal", "Anomalie"]))
