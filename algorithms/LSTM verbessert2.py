import os
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------ LOAD DATA ------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]
df_train_all = pd.DataFrame()
df_test_all = pd.DataFrame()

for line in app_lines:
    app = line.strip()
    for i in train_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df_train_all = pd.concat([df_train_all, pickle.load(f)], ignore_index=True)
    for i in test_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df_test_all = pd.concat([df_test_all, pickle.load(f)], ignore_index=True)

print("Original Train shape:", df_train_all.shape)
print("Original Test shape:", df_test_all.shape)

# ------------------ PREPROCESSING ------------------ #
df_train_all = df_train_all.sample(n=3000, random_state=42)
df_test_all = df_test_all.sample(n=3000, random_state=42)

num_cols = df_train_all.shape[1]
col_names = ["timestamp"] + [f"feat_{i}" for i in range(1, num_cols - 1)] + ["label"]
df_train_all.columns = col_names
df_test_all.columns = col_names

df_train_all["label"] = df_train_all["label"].replace(-1, 0)
df_test_all["label"] = df_test_all["label"].replace(-1, 0)

df_train_all["timestamp"] = pd.to_datetime(df_train_all["timestamp"], unit='ms', errors='coerce')
df_test_all["timestamp"] = pd.to_datetime(df_test_all["timestamp"], unit='ms', errors='coerce')
df_train_all = df_train_all.sort_values("timestamp").reset_index(drop=True)
df_test_all = df_test_all.sort_values("timestamp").reset_index(drop=True)

# ------------------ CREATE SEQUENCES ------------------ #
def create_sequences(df, seq_length=5, step=5):
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X, y = [], []
    for i in range(0, len(df) - seq_length, step):
        X.append(df[feature_cols].iloc[i:i + seq_length].values)
        y.append(df["label"].iloc[i + seq_length - 1])
    return np.array(X), np.array(y)

T = 5
X_train_seq, y_train = create_sequences(df_train_all, seq_length=T)
X_test_seq, y_test = create_sequences(df_test_all, seq_length=T)

print("Train Seq shape:", X_train_seq.shape)
print("Test  Seq shape:", X_test_seq.shape)

# ------------------ SCALING ------------------ #
scaler = StandardScaler()
X_train_flat = X_train_seq.reshape(X_train_seq.shape[0] * T, -1)
X_train_flat_scaled = scaler.fit_transform(X_train_flat)
X_train_scaled = X_train_flat_scaled.reshape(X_train_seq.shape)

X_test_flat = X_test_seq.reshape(X_test_seq.shape[0] * T, -1)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test_seq.shape)

# ------------------ SMOTE OVERSAMPLING ------------------ #
X_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_flat, y_train)
X_resampled_seq = X_resampled.reshape(-1, T, X_train_scaled.shape[2])

print("Oversampled X:", X_resampled_seq.shape)
print("Oversampled y:", y_resampled.shape)
print("New class distribution:", Counter(y_resampled))

# ------------------ SPLIT & TRAINING ------------------ #
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled_seq, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(T, X_train.shape[2])),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    shuffle=True,
    callbacks=[early_stopping, reduce_lr]
)

# ------------------ EVALUATION ------------------ #
y_pred_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1_val = f1_score(y_test, y_pred, average='macro')

print("\n=== LSTM Evaluation (Test) ===")
print("Accuracy:", acc)
print("Precision (macro):", prec)
print("Recall (macro):   ", rec)
print("F1 (macro):       ", f1_val)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LSTM (mit SMOTE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
