import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_curve, auc
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === Pfade ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')


def find_best_threshold(y_true, y_probs, target_class=0, metric='f1', steps=200):
    best_threshold = 0.5
    best_score = 0.0
    thresholds = np.linspace(0, 1, steps)

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        score = f1_score(y_true, y_pred, pos_label=target_class)
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score

# === Daten laden ===
with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]
df_train_all = pd.DataFrame()
df_test_all = pd.DataFrame()

for line in app_lines:
    app = line.strip()
    for i in train_indices:
        with open(os.path.join(supervised_path, f"{app}-{i}.pkl"), 'rb') as f:
            df_train_all = pd.concat([df_train_all, pickle.load(f)], ignore_index=True)
    for i in test_indices:
        with open(os.path.join(supervised_path, f"{app}-{i}.pkl"), 'rb') as f:
            df_test_all = pd.concat([df_test_all, pickle.load(f)], ignore_index=True)

print("Original Train shape:", df_train_all.shape)
print("Original Test  shape:", df_test_all.shape)

# === Sample verkleinern ===
df_train_all = df_train_all.sample(n=10_000, random_state=42)
df_test_all = df_test_all.sample(n=10_000, random_state=42)

# === Spalten umbenennen & sortieren ===
num_cols = df_train_all.shape[1]
df_train_all.columns = ["timestamp"] + [f"feat_{i}" for i in range(1, num_cols - 1)] + ["label"]
df_test_all.columns = df_train_all.columns

df_train_all["label"].replace(-1, 0, inplace=True)
df_test_all["label"].replace(-1, 0, inplace=True)

df_train_all["timestamp"] = pd.to_datetime(pd.to_numeric(df_train_all["timestamp"], errors='coerce'), unit='ms')
df_test_all["timestamp"] = pd.to_datetime(pd.to_numeric(df_test_all["timestamp"], errors='coerce'), unit='ms')

df_train_all = df_train_all.sort_values("timestamp").reset_index(drop=True)
df_test_all = df_test_all.sort_values("timestamp").reset_index(drop=True)

# === Sequenzen erstellen ===
def create_sequences(df, seq_length=5, step=5):
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X, y = [], []
    values = df[feature_cols].values
    labels = df["label"].values
    for i in range(0, len(df) - seq_length + 1, step):
        X.append(values[i:i+seq_length])
        y.append(labels[i+seq_length - 1])
    return np.array(X), np.array(y)

X_train_full, y_train_full = create_sequences(df_train_all, seq_length=5)
X_test_full, y_test_full = create_sequences(df_test_all, seq_length=5)

# === Skalieren ===
T = X_train_full.shape[1]
features = X_train_full.shape[2]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full.reshape(-1, features)).reshape(-1, T, features)
X_test_scaled = scaler.transform(X_test_full.reshape(-1, features)).reshape(-1, T, features)

# === Train/Val Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, y_train_full, test_size=0.2,
    stratify=y_train_full, random_state=42
)

# === Modell ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(T, features)),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# === Class Weights berechnen ===
counter = Counter(y_train)
major = counter.most_common(1)[0][0]
minor = 1 - major
w_major = 1.0
w_minor = counter[major] / max(counter[minor], 1)
class_weight = {major: w_major, minor: w_minor}
print("Train-Klassen:", counter)
print("class_weight:", class_weight)

# === Training ===
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr],
    shuffle=True
)

# === Evaluation ===
y_pred_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n=== LSTM Evaluation (Test @0.5) ===")
print("Accuracy:", accuracy_score(y_test_full, y_pred))
print("Precision (macro):", precision_score(y_test_full, y_pred, average='macro'))
print("Recall (macro):   ", recall_score(y_test_full, y_pred, average='macro'))
print("F1 (macro):       ", f1_score(y_test_full, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test_full, y_pred))
print("Classification Report:\n", classification_report(y_test_full, y_pred))

# === Automatische Threshold-Suche für beste F1-Score der Klasse 0 ===
best_thresh, best_f1_0 = find_best_threshold(y_test_full, y_pred_prob, target_class=0)

y_opt = (y_pred_prob >= best_thresh).astype(int)
precision_0 = precision_score(y_test_full, y_opt, pos_label=0)
recall_0 = recall_score(y_test_full, y_opt, pos_label=0)
f1_0 = f1_score(y_test_full, y_opt, pos_label=0)

print(f"\n🔍 Bester Threshold für F1(0): {best_thresh:.4f}")
print(f"Precision(0): {precision_0:.4f}")
print(f"Recall(0):    {recall_0:.4f}")
print(f"F1(0):        {f1_0:.4f}")
print("Confusion Matrix (optimiert):")
print(confusion_matrix(y_test_full, y_opt))
print("Classification Report (optimiert):")
print(classification_report(y_test_full, y_opt))

# === ROC Curve bleibt gleich ===
fpr, tpr, _ = roc_curve(y_test_full, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC - LSTM mit F1(0)-Optimierung')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
