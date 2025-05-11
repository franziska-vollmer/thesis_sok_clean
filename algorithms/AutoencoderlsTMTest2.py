import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# === 1. Daten laden ===
directory_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
pkl_files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]

dfs = []
for file in pkl_files:
    file_path = os.path.join(directory_path, file)
    print(f"Lade Datei: {file_path}")
    try:
        data = pd.read_pickle(file_path)
        dfs.append(data)
    except Exception as e:
        print(f"Fehler beim Laden der Datei {file}: {e}")

combined_data = pd.concat(dfs, ignore_index=True)

# === 2. Features & Labels ===
X = combined_data.iloc[:, :-1].values
y = combined_data.iloc[:, -1].values

# === 3. Skalierung ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Zeitfenster erstellen ===
def create_sequences(X, y, time_step):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        Xs.append(X[i:i + time_step])
        ys.append(y[i + time_step])  # Label zum Ende des Fensters
    return np.array(Xs), np.array(ys)

time_step = 10
X_seq, y_seq = create_sequences(X_scaled, y, time_step)

# === 5. Nur -1 und 1 behalten, 0 ignorieren ===
mask = y_seq != 0
X_filtered = X_seq[mask]
y_filtered = y_seq[mask]

# === 6. -1 ➝ 0 (für binäre Klassifikation) ===
y_filtered = (y_filtered == 1).astype(int)

# === 7. Train/test-Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)

# === 8. Modell definieren ===
timesteps = X_train.shape[1]
n_features = X_train.shape[2]

inputs = Input(shape=(timesteps, n_features))

# Encoder
x = LSTM(50, return_sequences=False)(inputs)
x = Dropout(0.2)(x)

# Klassifikation
class_output = Dense(1, activation='sigmoid', name='class_output')(x)

# Decoder für Rekonstruktion
x_repeated = RepeatVector(timesteps)(x)
x_decoded = LSTM(50, return_sequences=True)(x_repeated)
reconstruction_output = TimeDistributed(Dense(n_features), name='reconstruction_output')(x_decoded)

# Modell
model = Model(inputs, outputs=[reconstruction_output, class_output])
model.compile(optimizer=Adam(),
              loss={'reconstruction_output': 'mse', 'class_output': 'binary_crossentropy'},
              loss_weights={'reconstruction_output': 1.0, 'class_output': 1.0},
              metrics={'class_output': 'accuracy'})

# === 9. Training ===
model.fit(X_train, {'reconstruction_output': X_train, 'class_output': y_train},
          epochs=10,
          batch_size=64,
          validation_split=0.2,
          verbose=1)

# === 10. Vorhersage ===
reconstruction_preds, class_preds = model.predict(X_test)
class_preds = (class_preds > 0.5).astype(int).flatten()

# === 11. Klassifikationsbericht ===
print("=== Klassifikationsbericht ===")
print(classification_report(y_test, class_preds))
