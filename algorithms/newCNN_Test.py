# ==========================
# 1. Alle CSV-Dateien laden
# ==========================

import pandas as pd
import numpy as np
import glob
import os

print("Starte Datenvorbereitung...")

# Basis-Pfad zu deinen Daten
base_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files'

X = []
y = []

sampling_rate = 0.1  # 0.1 Sekunden pro Zeile
threshold_seconds = 180  # 3 Minuten Grenze

cve_folders = [f.path for f in os.scandir(base_path) if f.is_dir()]

for cve_folder in cve_folders:
    csv_files = glob.glob(os.path.join(cve_folder, '*_freqvector_full.csv'))
    
    for file in csv_files:
        print(f"Lade Datei: {file}")
        df = pd.read_csv(file)

        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])

        data = df.values

        times = np.arange(len(data)) * sampling_rate
        labels = (times >= threshold_seconds).astype(int)

        if data.shape[1] < 576:
            pad_width = 576 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')

        data = data.reshape((-1, 24, 24, 1))

        X.append(data)
        y.append(labels)

print("Alle Daten geladen und vorbereitet!")

X = np.vstack(X)
y = np.concatenate(y)

print(f"Feature-Matrix Shape: {X.shape}")
print(f"Label-Matrix Shape: {y.shape}")

# ==========================
# 2. CNN Modell trainieren
# ==========================

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("Splitte Daten in Training und Test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Trainingssamples: {X_train.shape[0]}")
print(f"Testsamples: {X_test.shape[0]}")

print("Erzeuge CNN-Modell...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Starte CNN-Training...")

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

print("Training abgeschlossen. Starte Evaluation...")

# Modell evaluieren
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Vorhersagen
y_pred = (model.predict(X_test) > 0.5).astype(int)

# ==========================
# 3. Ergebnisse visualisieren
# ==========================

print("Erzeuge Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomal"], yticklabels=["Normal", "Anomal"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomal"]))

# ==========================
# 4. Modell speichern
# ==========================

model.save('cnn_model_cve_anomaly.h5')
print("Modell erfolgreich gespeichert als 'cnn_model_cve_anomaly.h5' ✅")
