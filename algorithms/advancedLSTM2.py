import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------
# 1. Daten aus mehreren Pickle-Dateien laden
# ----------------------------------------
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
    # Lade alle Trainings-Pickle-Dateien für diesen App-Namen
    for i in train_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df_temp = pickle.load(f)
        df_train_all = pd.concat([df_train_all, df_temp], ignore_index=True)
    # Lade alle Test-Pickle-Dateien für diesen App-Namen
    for i in test_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df_temp = pickle.load(f)
        df_test_all = pd.concat([df_test_all, df_temp], ignore_index=True)

print("Train DataFrame shape:", df_train_all.shape)
print("Test DataFrame shape:", df_test_all.shape)

# Optional: Kombiniere Trainings- und Testdaten, falls Du einen durchgehenden Zeitverlauf möchtest.
df_all = pd.concat([df_train_all, df_test_all], ignore_index=True)
print("Combined DataFrame shape:", df_all.shape)

# Sortiere nach dem Zeitstempel (angenommen, Spalte 0 enthält den Zeitstempel)
df_all.sort_values(by=df_all.columns[0], inplace=True)
df_all.reset_index(drop=True, inplace=True)

# ----------------------------------------
# 2. Features und Label extrahieren
# ----------------------------------------
# Annahme:
# - Spalte 0: Zeitstempel (wird nicht als Feature verwendet)
# - Spalten 1 bis -2: Features (z. B. 555 Features)
# - Spalte -1: Label
features = df_all.iloc[:, 1:-1].values.astype(np.float32)  # Form: (n_samples, 555)
labels = df_all.iloc[:, -1].values  # Form: (n_samples,)

# Falls Labels als -1 codiert sind, ersetze sie durch 0:
labels = np.where(labels == -1, 0, labels)

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# ----------------------------------------
# 3. Daten skalieren
# ----------------------------------------
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ----------------------------------------
# 4. Sequenzen erstellen (Sliding Window)
# ----------------------------------------
def create_sequences(data, labels, sequence_length=5):
    sequences = []
    seq_labels = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i+sequence_length])
        # Verwende das Label des letzten Zeitpunkts als Label der Sequenz
        seq_labels.append(labels[i + sequence_length - 1])
    return np.array(sequences), np.array(seq_labels)

sequence_length = 5
X, y_seq = create_sequences(features_scaled, labels, sequence_length=sequence_length)
print("Sequenzen-Form:", X.shape)   # Erwartet: (n_sequences, 5, 555)
print("Labels-Form:", y_seq.shape)    # Erwartet: (n_sequences,)

# ----------------------------------------
# 5. Aufteilung in Trainings-, Validierungs- und Testdaten
# ----------------------------------------
# Hier erfolgt eine zufällige Aufteilung (beachte: bei Zeitreihendaten kann eine chronologische Aufteilung sinnvoller sein)
X_train, X_test, y_train, y_test = train_test_split(X, y_seq, test_size=0.2, random_state=42, stratify=y_seq)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# ----------------------------------------
# 6. Focal Loss definieren (mit Casting von y_true)
# ----------------------------------------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-8
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, eps, 1. - eps)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = weight * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss)
    return focal_loss_fixed

# ----------------------------------------
# 7. LSTM-Modell aufbauen mit bidirektionalen Schichten
# ----------------------------------------
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(sequence_length, features_scaled.shape[1])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])
model.summary()

# ----------------------------------------
# 8. Callbacks
# ----------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# ----------------------------------------
# 9. Training
# ----------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ----------------------------------------
# 10. Evaluation auf dem Testset
# ----------------------------------------
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("\n=== Test Evaluation ===")
print("Accuracy:", acc)
print("Precision (macro):", prec)
print("Recall (macro):", rec)
print("F1 Score (macro):", f1)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
