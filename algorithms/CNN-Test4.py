import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

# === Pfade ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

# === CVE-Liste laden ===
with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

app_lines = [line.strip() for line in app_lines if line.strip()]
selected_apps = app_lines[:15]  # Nur 15 CVEs verwenden

train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]

df_all = pd.DataFrame()

print("\n🚀 Starte Laden der ersten 15 CVEs...")
for app in selected_apps:
    for i in train_indices + test_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                df_temp = pickle.load(f)
            df_all = pd.concat([df_all, df_temp], ignore_index=True)
        else:
            print(f"WARNUNG: Datei nicht gefunden: {path}")

print(f"\n✅ Alle Dateien geladen. Anzahl Samples: {len(df_all)}")

# === Features und Labels trennen ===
print("\n📦 Trenne Features und Labels...")
X = df_all.iloc[:, :-1].values
y = df_all.iloc[:, -1].values

# === Label Fix: -1 ➔ 0
y = np.where(y == -1, 0, y)

# === Typumwandlung
X = X.astype(np.float32)
y = y.astype(np.int32)

# === Padding auf 1024 Features
print("\n🛠️ Padding der Features auf 1024 Dimensionen...")
TARGET_FEATURES = 1024
if X.shape[1] < TARGET_FEATURES:
    padding = TARGET_FEATURES - X.shape[1]
    X = np.pad(X, ((0, 0), (0, padding)), mode='constant')
print(f"✅ Neues Feature-Shape: {X.shape}")

# === Reshape zu 32x32x1
print("\n🔄 Reshape der Features zu 32x32 Bildern...")
X_reshaped = X.reshape((-1, 32, 32, 1))
print(f"✅ Neues Input-Shape: {X_reshaped.shape}")

# === Downsampling der Anomalien
print("\n✂️ Downsampling der Anomalien für bessere Balance...")
X_normal = X_reshaped[y == 0]
y_normal = y[y == 0]
X_anomaly = X_reshaped[y == 1]
y_anomaly = y[y == 1]

print(f"Vor Downsampling: Normale = {len(y_normal)}, Anomalien = {len(y_anomaly)}")
target_anomalies = 2 * len(y_normal)
target_anomalies = min(target_anomalies, len(y_anomaly))

idx = np.random.choice(len(X_anomaly), size=target_anomalies, replace=False)
X_anomaly = X_anomaly[idx]
y_anomaly = y_anomaly[idx]

X_balanced = np.vstack([X_normal, X_anomaly])
y_balanced = np.concatenate([y_normal, y_anomaly])

X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)
print(f"✅ Nach Downsampling: {X_balanced.shape}, Verteilung: {np.bincount(y_balanced)}")

# === Train-Test-Split
print("\n✂️ Train-Test-Split (80% Training, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)
print(f"✅ Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")

# === Class Weights
print("\n⚖️ Berechne Class Weights...")
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print(f"✅ Class Weights: {class_weights_dict}")

# === Verbesserte CNN-Architektur
print("\n🏗️ Baue das VERBESSERTE CNN Modell...")

model = models.Sequential([
    layers.Input(shape=(32, 32, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("✅ Verbesserte CNN Modell kompiliert.")

# === Training starten
print("\n🚀 Starte Training des Modells...")
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights_dict
)
print("✅ Training abgeschlossen.")

# === Evaluation
print("\n🧪 Evaluierung auf Testdaten...")
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32")

print("\n📋 Klassifikationsreport:")
print(classification_report(y_test, y_pred, digits=4))

# === Trainingsverlauf plotten
print("\n📈 Zeichne Trainingskurven...")
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Verlauf')
plt.xlabel('Epoche')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Verlauf')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
print("✅ Fertig!")

# === Optional: Modell speichern
# print("\n💾 Speichere das Modell...")
# model.save("cnn_model_balanced_improved.h5")
# print("✅ Modell gespeichert als cnn_model_balanced_improved.h5")
