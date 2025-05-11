import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import os

# === Daten laden ===
data_folder = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
data_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
data_list = [pd.read_pickle(os.path.join(data_folder, f)) for f in tqdm(data_files, desc="Laden der Dateien")]
data = pd.concat(data_list, ignore_index=True)

# === Features und Labels ===
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# === Labels umwandeln: 0 = normal, 1 = Anomalie ===
y_binary = np.where(y == 1, 0, 1)  # 1 (anomalie), 0 (normal)

# === Nur normale Daten (0) zum Trainieren ===
X_train_raw = X[y_binary == 0]

# === Skalieren ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_scaled_all = scaler.transform(X)

# === LSTM erwartet 3D-Input [samples, timesteps, features] ===
timesteps = 1
n_features = X.shape[1]
X_train_seq = X_train_scaled.reshape((-1, timesteps, n_features))
X_all_seq = X_scaled_all.reshape((-1, timesteps, n_features))

# === LSTM Autoencoder definieren ===
input_layer = Input(shape=(timesteps, n_features))
encoded = LSTM(64, activation='relu')(input_layer)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
output_layer = Dense(n_features)(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# === Training ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
autoencoder.fit(X_train_seq, X_train_seq,
                epochs=50,
                batch_size=64,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=1)

# === Rekonstruktionsfehler berechnen ===
X_pred = autoencoder.predict(X_all_seq)
reconstruction_error = np.mean(np.square(X_all_seq - X_pred), axis=(1, 2))

# === Optimalen Schwellenwert per F1 bestimmen ===
precision_vals, recall_vals, thresholds = precision_recall_curve(y_binary, reconstruction_error)
f1_scores = [2 * (p * r) / (p + r + 1e-8) for p, r in zip(precision_vals, recall_vals)]
best_thresh = thresholds[np.argmax(f1_scores)]

# === Anomalien klassifizieren ===
y_pred = (reconstruction_error > best_thresh).astype(int)

# === Evaluation ===
precision = precision_score(y_binary, y_pred)
recall = recall_score(y_binary, y_pred)
f1 = f1_score(y_binary, y_pred)
roc = roc_auc_score(y_binary, reconstruction_error)
pr_auc = auc(recall_vals, precision_vals)

print("\n📊 LSTM Autoencoder Ergebnisse:")
print(f"F1-Score:       {f1:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"ROC-AUC:        {roc:.4f}")
print(f"PR-AUC:         {pr_auc:.4f}")
print(f"Best Threshold: {best_thresh:.6f}")
