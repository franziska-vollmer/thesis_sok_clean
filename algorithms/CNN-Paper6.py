# 📦 Bibliotheken
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# 📂 Schritt 1: Alle .pkl-Dateien laden
folder_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

print(f"Gefundene Dateien: {len(pkl_files)}")
dataframes = []

for file in tqdm(pkl_files, desc="Lade Pickle-Dateien"):
    df = pd.read_pickle(os.path.join(folder_path, file))
    dataframes.append(df)

# 📑 Verbinden aller DataFrames
full_data = pd.concat(dataframes, ignore_index=True)
print(f"\n✅ Gesamtanzahl Samples: {full_data.shape[0]}")
print(f"✅ Anzahl Features (ohne Label): {full_data.shape[1] - 1}")

# 🔄 Schritt 2: Features und Labels trennen + Label-Korrektur
X = full_data.iloc[:, :-1].values  # Features
y = full_data.iloc[:, -1].values   # Labels

# Labels normalisieren: -1 wird zu 0
if -1 in np.unique(y):
    print("\n⚠️ -1 Labels gefunden, korrigiere...")
    y = np.where(y == -1, 0, y)

print(f"🎯 Eindeutige Labels nach Korrektur: {np.unique(y)}")

# 🔄 Schritt 3: Feature Skalierung
print("\n🔄 Skaliere Features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔄 Schritt 4: Train/Test Split
print("🔄 Teile in Training/Test-Daten...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 🔨 Schritt 5: Modell bauen
print("🔨 Baue Modell...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 📈 Modell kompilieren (mit stabiler Loss-Funktion)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# 🚀 Schritt 6: Training starten
print("\n🚀 Starte Schnelltest-Training (10 Epochen)...")
history = model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.2)

# 🧪 Schritt 7: Modell evaluieren
print("\n🧪 Evaluierung auf Testdaten...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"🎯 Test Loss: {test_loss:.4f}")

# 📊 Schritt 8: Trainingskurven zeichnen
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (Schnelltest)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# 🧮 Schritt 9: Precision, Recall, F1-Score berechnen
print("\n🔍 Erweiterte Metriken auf Testdaten...")

# Vorhersagen
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# 📈 Confusion Matrix ausgeben
print("\n📈 Confusion Matrix:")
print(cm)

# Optional: Confusion Matrix als Heatmap zeichnen
import seaborn as sns

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal (0)", "Anomalie (1)"], yticklabels=["Normal (0)", "Anomalie (1)"])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix (Test Set)')
plt.show()
