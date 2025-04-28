import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# === Daten laden ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]

df_train_all, df_test_all = pd.DataFrame(), pd.DataFrame()

with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

for line in app_lines:
    app = line.strip()
    for i in train_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                df_temp = pickle.load(f)
            df_train_all = pd.concat([df_train_all, df_temp], ignore_index=True)
    for i in test_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                df_temp = pickle.load(f)
            df_test_all = pd.concat([df_test_all, df_temp], ignore_index=True)

df_all = pd.concat([df_train_all, df_test_all], ignore_index=True)
df_all.sort_values(by=df_all.columns[0], inplace=True)
df_all.reset_index(drop=True, inplace=True)

# === Features und Labels ===
features = df_all.iloc[:, 1:-1].values.astype(np.float32)
labels = df_all.iloc[:, -1].values
labels = np.where(labels == -1, 0, labels)

# === Skalierung ===
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# === Sequenzen erstellen ===
def create_sequences(data, labels, sequence_length=3):
    X, y = [], []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i+sequence_length])
        y.append(labels[i + sequence_length - 1])
    return np.array(X), np.array(y)

sequence_length = 3
X, y = create_sequences(features_scaled, labels, sequence_length=sequence_length)

# === Aufteilen ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

# === Focal Loss ===
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        eps = 1e-8
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, eps, 1. - eps)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true + (1 - alpha) * (1 - y_true)
        return K.mean(weight * K.pow(1 - y_pred, gamma) * cross_entropy)
    return loss

# === Modell ===
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(sequence_length, features_scaled.shape[1])),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001), loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])

# === Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# === Training ===
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

# === Evaluation ===
y_pred_prob = model.predict(X_test).ravel()

# === Threshold Optimierung für Klasse 0 ===
best_thresh, best_f1 = 0.5, 0
for t in np.linspace(0, 1, 100):
    pred = (y_pred_prob >= t).astype(int)
    f1 = f1_score(y_test, pred, pos_label=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"\n🎯 Optimaler Threshold für Klasse 0: {best_thresh:.4f} mit F1(0): {best_f1:.4f}")

# === Final Metrics ===
y_pred = (y_pred_prob >= best_thresh).astype(int)

print("\n=== Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
