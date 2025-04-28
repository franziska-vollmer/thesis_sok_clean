import os, pickle, random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === Konfiguration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')
train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]
n_folds = 5
seq_length = 5
step = 5
sample_size = 10000

# === Sequenz-Erstellung ===
def create_sequences(df, seq_length=5, step=5):
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X_values = df[feature_cols].values
    y_values = df["label"].values
    X_seq, y_seq = [], []
    for i in range(0, len(df) - seq_length + 1, step):
        X_seq.append(X_values[i:i+seq_length])
        y_seq.append(y_values[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

# === Daten laden ===
def load_app_data(apps, indices):
    df_all = pd.DataFrame()
    for app in apps:
        for i in indices:
            path = os.path.join(supervised_path, f"{app}-{i}.pkl")
            if os.path.exists(path):
            
                with open(path, 'rb') as f:
                    df = pickle.load(f)
                df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all

# === Hauptlauf ===
with open(apps_file_path, 'r') as f:
    all_apps = [line.strip() for line in f.readlines() if line.strip()]
assert len(all_apps) >= 40, " Mindestens 40 CVEs nötig."

results = []

for fold in range(n_folds):
    print(f"\n🔁 FOLD {fold+1}/{n_folds} 🔁")
    random.shuffle(all_apps)
    train_apps, test_apps = all_apps[:45], all_apps[45:50]

    # Daten laden
    df_train_all = load_app_data(train_apps, train_indices).sample(n=sample_size, random_state=fold)
    df_test_all = load_app_data(test_apps, test_indices).sample(n=sample_size, random_state=fold)

    # Spalten umbenennen
    col_names = ["timestamp"] + [f"feat_{i}" for i in range(1, df_train_all.shape[1] - 1)] + ["label"]
    df_train_all.columns = col_names
    df_test_all.columns = col_names

    df_train_all["label"] = df_train_all["label"].replace(-1, 0)
    df_test_all["label"] = df_test_all["label"].replace(-1, 0)

    df_train_all["timestamp"] = pd.to_numeric(df_train_all["timestamp"], errors='coerce')
    df_test_all["timestamp"] = pd.to_numeric(df_test_all["timestamp"], errors='coerce')
    df_train_all = df_train_all.sort_values("timestamp").reset_index(drop=True)
    df_test_all = df_test_all.sort_values("timestamp").reset_index(drop=True)

    # Sequenzen
    X_train_full, y_train_full = create_sequences(df_train_all, seq_length, step)
    X_test_full, y_test_full = create_sequences(df_test_all, seq_length, step)

    # Skalierung
    num_samples_train, T, num_features = X_train_full.shape
    X_train_scaled = StandardScaler().fit_transform(X_train_full.reshape(-1, num_features)).reshape(num_samples_train, T, num_features)
    num_samples_test = X_test_full.shape[0]
    X_test_scaled = StandardScaler().fit_transform(X_test_full.reshape(-1, num_features)).reshape(num_samples_test, T, num_features)

    # Modell
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(T, num_features)),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Class Weights
    counter = Counter(y_train_full)
    majority = max(counter, key=counter.get)
    minority = 1 - majority
    class_weight = {majority: 1.0, minority: counter[majority] / max(counter[minority], 1)}
    print("Class Weights:", class_weight)

    # Training
    model.fit(
        X_train_scaled, y_train_full,
        validation_split=0.2,
        batch_size=64,
        epochs=30,
        verbose=0,
        shuffle=True,
        class_weight=class_weight,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                   ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, verbose=1)]
    )

    # Evaluation
    y_pred_prob = model.predict(X_test_scaled).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test_full, y_pred)
    prec = precision_score(y_test_full, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test_full, y_pred, average='macro', zero_division=0)
    f1_val = f1_score(y_test_full, y_pred, average='macro', zero_division=0)
    fpr, tpr, _ = roc_curve(y_test_full, y_pred_prob)
    auc_val = auc(fpr, tpr)

    print(f"Fold {fold+1}: F1={f1_val:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, AUC={auc_val:.4f}")
    results.append({"fold": fold+1, "f1": f1_val, "precision": prec, "recall": rec, "auc": auc_val})

# Gesamtstatistik
print("\n📊 Cross-Validation Ergebnisse:")
for r in results:
    print(f"Fold {r['fold']}: F1={r['f1']:.4f}, Precision={r['precision']:.4f}, Recall={r['recall']:.4f}, AUC={r['auc']:.4f}")

print("\n📈 Durchschnitt:")
print(f"F1: {np.mean([r['f1'] for r in results]):.4f} ± {np.std([r['f1'] for r in results]):.4f}")
print(f"Precision: {np.mean([r['precision'] for r in results]):.4f} ± {np.std([r['precision'] for r in results]):.4f}")
print(f"Recall: {np.mean([r['recall'] for r in results]):.4f} ± {np.std([r['recall'] for r in results]):.4f}")
print(f"AUC: {np.mean([r['auc'] for r in results]):.4f} ± {np.std([r['auc'] for r in results]):.4f}")
