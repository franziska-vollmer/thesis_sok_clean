import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from collections import Counter

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("TF version:", tf.__version__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

def extract_xy_from_df(df):
    df_features = df.drop(index=556).astype(float)
    labels = df.loc[556].astype(int).values
    X = df_features.T.values
    y = labels
    return X, y

# === Daten laden ===
X_all, y_all = [], []
with open(apps_file_path, 'r') as f:
    apps = [line.strip() for line in f.readlines()]

for app in apps:
    for i in [1, 2, 3, 4]:
        file_path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
            X, y = extract_xy_from_df(df)
            X_all.append(X)
            y_all.append(y)

min_feat_len = min(x.shape[1] for x in X_all)
X_all = [x[:, :min_feat_len] for x in X_all]

X = np.vstack(X_all)
y = np.concatenate(y_all)

# === Nur 0 und 1 Klassen verwenden ===
mask = np.isin(y, [0, 1])
X = X[mask]
y = y[mask]

# === Skalieren ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === In Sequenzen umwandeln ===
def create_sequences(X, y, seq_len=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

seq_len = 10
X_seq, y_seq = create_sequences(X_scaled, y, seq_len)

# === Trainings-/Testdaten splitten ===
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# === Class Weights ===
class_counts = Counter(y_train)
total = sum(class_counts.values())
class_weights = {k: total / (len(class_counts) * v) for k, v in class_counts.items()}
print("🔍 Class Weights:", class_weights)

# === Modell ===
model = Sequential()
model.add(InputLayer(input_shape=(seq_len, X_seq.shape[2])))
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === Training ===
model.fit(X_train, y_train,
          epochs=50,
          batch_size=64,
          validation_split=0.1,
          class_weight=class_weights,
          callbacks=[early_stop],
          verbose=1)

# === Auswertung ===
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
