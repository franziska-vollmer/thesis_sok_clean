import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam

# === 1. Pfad zum Verzeichnis mit deinen PKL-Dateien ===
DATA_DIR = DATA_DIR = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/" # <-- HIER ANPASSEN

# === 2. Alle .pkl-Dateien laden und zusammenführen ===
all_dfs = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".pkl"):
        try:
            df = pd.read_pickle(os.path.join(DATA_DIR, file))
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARNUNG] Fehler beim Laden von {file}: {e}")

if not all_dfs:
    raise ValueError("Keine gültigen .pkl-Dateien gefunden.")

df = pd.concat(all_dfs, ignore_index=True)
print(f"[INFO] Gesamtdatenform: {df.shape}")

# === 3. Features und Label trennen ===
X_raw = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values

# === 4. Label umwandeln: 1 → 0 (normal), -1 → 1 (anomalie) ===
y_binary = np.where(y_raw == 1, 0, 1)

# === 5. Standardisierung ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# === 6. Autoencoder zur Merkmalsextraktion ===
encoding_dim = 30
input_dim = X_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
print("[INFO] Training Autoencoder...")
autoencoder.fit(X_scaled, X_scaled, epochs=20, batch_size=256, shuffle=True, verbose=1)

encoder = Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X_scaled)

# === 7. Vorbereitung für LSTM ===
X_seq = np.reshape(X_encoded, (X_encoded.shape[0], 1, X_encoded.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# === 8. LSTM-Modell definieren ===
model = Sequential()
model.add(LSTM(64, input_shape=(1, encoding_dim)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# === 9. Class Weights berechnen und anwenden ===
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print(f"[INFO] Verwendete class_weights: {class_weights}")

# === 10. LSTM Training ===
print("[INFO] Training LSTM mit class_weight...")
model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1, class_weight=class_weights)

# === 11. Vorhersage mit angepasstem Threshold ===
print("[INFO] Evaluierung des Modells mit Schwellenwert 0.3...")
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.3).astype("int32")

# === 12. Klassifikationsbericht ===
print("[INFO] Klassifikationsbericht bei threshold=0.3:\n")
print(classification_report(y_test, y_pred, digits=4))

# === 13. Konfusionsmatrix + DR / FAR ===
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

dr = tp / (tp + fn) if (tp + fn) else 0  # Detection Rate
far = fp / (fp + tn) if (fp + tn) else 0  # False Alarm Rate

print(f"[INFO] Konfusionsmatrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"[INFO] Detection Rate (DR): {dr:.4f}")
print(f"[INFO] False Alarm Rate (FAR): {far:.4f}")
