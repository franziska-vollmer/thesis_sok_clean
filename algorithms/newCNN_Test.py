import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# === Einstellungen ===
timesteps = 10  # Anzahl Zeitschritte
latent_dim = 64  # Dimension des LSTM-Encodings
data_dir = '/Pfad/zu/deinen/pkl_dateien/'
apps = ['beispielapp1', 'beispielapp2']  # passe an

# === Hilfsfunktion zum Laden der Daten ===
def load_data(apps, indices=[1,2,3,4]):
    X_all, y_all = [], []
    for app in apps:
        for i in indices:
            path = os.path.join(data_dir, f'{app}-{i}.pkl')
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as f:
                df = pickle.load(f)
            df_features = df.drop(index=556).astype(float)
            labels = df.loc[556].astype(int).values
            X = df_features.T.values
            y = labels
            X_all.append(X)
            y_all.append(y)
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    return X_all, y_all

# === Daten vorbereiten ===
X_raw, y = load_data(apps)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# === Sequenzbildung ===
n_samples = len(X_scaled) // timesteps
X_seq = X_scaled[:n_samples * timesteps].reshape(n_samples, timesteps, -1)
y_seq = y[:n_samples * timesteps].reshape(n_samples, timesteps)
y_seq_majority = (np.mean(y_seq, axis=1) > 0.5).astype(int)  # 1, wenn Mehrheit anomal ist

# === Nur normale Daten für Training verwenden ===
X_train = X_seq[y_seq_majority == 0]

# === LSTM-Autoencoder definieren ===
input_shape = (timesteps, X_seq.shape[2])
inputs = Input(shape=input_shape)
encoded = LSTM(latent_dim)(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_shape[1], return_sequences=True)(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# === Training ===
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=64,
                validation_split=0.1,
                shuffle=True,
                callbacks=[early_stop],
                verbose=1)

# === Rekonstruktionsfehler berechnen ===
X_pred = autoencoder.predict(X_seq)
recon_error = np.mean(np.square(X_seq - X_pred), axis=(1, 2))

# === Beste Schwelle via F1-Optimierung finden ===
thresholds = np.linspace(min(recon_error), max(recon_error), 100)
f1_scores = [f1_score(y_seq_majority, recon_error > t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]

# === Vorhersage und Metriken ===
y_pred = (recon_error > best_thresh).astype(int)

print("\n== Ergebnisse ==")
print(f"Beste Schwelle: {best_thresh:.4f}")
print(f"F1-Score:       {f1_score(y_seq_majority, y_pred):.4f}")
print(f"Precision:      {precision_score(y_seq_majority, y_pred):.4f}")
print(f"Recall:         {recall_score(y_seq_majority, y_pred):.4f}")
print(f"ROC-AUC:        {roc_auc_score(y_seq_majority, recon_error):.4f}")

# === Optional: ROC-Kurve plotten ===
fpr, tpr, _ = roc_curve(y_seq_majority, recon_error)
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – LSTM Autoencoder')
plt.legend()
plt.grid(True)
plt.show()
