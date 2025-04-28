import pandas as pd
import numpy as np
import glob
import os
import re
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD

# === Einstellungen ===
path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
n_splits = 5
epochs = 20
batch_size = 64

# === Hilfsfunktionen ===

def get_cve_id(filename):
    match = re.match(r'(CVE-\d{4}-\d+)', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Filename {filename} passt nicht!")

def build_cnn(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=SGD(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Daten vorbereiten: Alle Kategorien einsammeln ===
print(f"Suche .pkl Dateien in {path}...")
file_list = glob.glob(os.path.join(path, '*.pkl'))
file_list.sort()

col_categories = defaultdict(set)
raw_data_list = []  # Rohdaten zwischenspeichern

print(f"Gefundene Dateien: {len(file_list)}\n")
for idx, file in enumerate(file_list):
    print(f"[{idx+1}/{len(file_list)}] Analysiere {os.path.basename(file)}...")
    data = pd.read_pickle(file)
    raw_data_list.append((file, data))

    X = data.iloc[:, :-1]
    non_numeric_cols = X.select_dtypes(include=['object']).columns

    for col in non_numeric_cols:
        col_categories[col].update(X[col].astype(str).unique())

# LabelEncoder fitten
label_encoders = {}
for col, categories in col_categories.items():
    le = LabelEncoder()
    le.fit(list(categories))
    label_encoders[col] = le

print("\n✅ Alle LabelEncoder vorbereitet.")

# === Alle Daten final vorbereiten ===
X_list, y_list, groups = [], [], []

for idx, (file, data) in enumerate(raw_data_list):
    print(f"[{idx+1}/{len(raw_data_list)}] Wandle {os.path.basename(file)} um...")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].values

    non_numeric_cols = X.select_dtypes(include=['object']).columns

    for col in non_numeric_cols:
        X[col] = label_encoders[col].transform(X[col].astype(str))

    X = X.values

    base_filename = os.path.basename(file)
    cve_id = get_cve_id(base_filename)

    X_list.append(X)
    y_list.append(y)
    groups.extend([cve_id] * len(y))

print("\n✅ Alle Daten vorbereitet für Training.")

# === Zusammenbauen ===
X = np.vstack(X_list)
y = np.hstack(y_list)
groups = np.array(groups)

X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"X-Shape: {X.shape}, y-Shape: {y.shape}, Anzahl verschiedene CVEs: {len(np.unique(groups))}\n")

# === Cross Validation ===
gkf = GroupKFold(n_splits=n_splits)

all_acc, all_prec, all_rec, all_f1 = [], [], [], []

fold = 1
for train_idx, test_idx in gkf.split(X, y, groups):
    print(f"\n==== Starte Fold {fold}/{n_splits} ====")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Min-Max-Skalierung
    X_min = X_train.min()
    X_max = X_train.max()
    X_train = (X_train - X_min) / (X_max - X_min)
    X_test = (X_test - X_min) / (X_max - X_min)

    # Float32 Konvertierung
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print(f"Labels im Training: {np.unique(y_train)}")

    model = build_cnn(X_train.shape[1:])

    print(f"Training Modell für Fold {fold}...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    acc = np.mean(y_pred.flatten() == y_test.flatten())
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"✅ Fold {fold} abgeschlossen.")
    print(f"Fold {fold} - Accuracy: {acc:.4f} - Precision: {prec:.4f} - Recall: {rec:.4f} - F1-Score: {f1:.4f}")

    all_acc.append(acc)
    all_prec.append(prec)
    all_rec.append(rec)
    all_f1.append(f1)

    fold += 1

# === Gesamtergebnisse ===
print("\n==== Cross Validation Ergebnisse ====")
print(f"Durchschnitt Accuracy : {np.mean(all_acc):.4f} (+/- {np.std(all_acc):.4f})")
print(f"Durchschnitt Precision: {np.mean(all_prec):.4f} (+/- {np.std(all_prec):.4f})")
print(f"Durchschnitt Recall   : {np.mean(all_rec):.4f} (+/- {np.std(all_rec):.4f})")
print(f"Durchschnitt F1-Score  : {np.mean(all_f1):.4f} (+/- {np.std(all_f1):.4f})")
