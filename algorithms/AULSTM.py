import os
import pickle
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === SETTINGS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]
n_folds = 5

# === HELPERS ===
def extract_xy_from_df(df):
    df_features = df.drop(index=556).astype(float)
    labels = df.loc[556].astype(float).astype(int).values
    X = df_features.T.values
    y = labels
    return X, y

def load_app_data(app_list, indices):
    X_all, y_all = [], []
    min_len = None
    for app in app_list:
        for i in indices:
            path = os.path.join(supervised_path, f"{app}-{i}.pkl")
            if not os.path.exists(path):
                print(f"[WARN] Datei fehlt: {path}")
                continue
            with open(path, 'rb') as f:
                df = pickle.load(f)
            X, y = extract_xy_from_df(df)
            if min_len is None:
                min_len = X.shape[1]
            else:
                min_len = min(min_len, X.shape[1])
            X_all.append(X)
            y_all.append(y)
    if not X_all:
        raise ValueError("Keine gültigen Daten geladen.")
    X_all = [x[:, :min_len] for x in X_all]
    return np.vstack(X_all), np.concatenate(y_all)

# === LOAD APP NAMES ===
with open(apps_file_path, 'r') as f:
    all_apps = [line.strip() for line in f.readlines() if line.strip()]
assert len(all_apps) >= 45, "Mindestens 45 Apps/CVEs erforderlich."

results = []

# === CROSS-VALIDATION ===
for fold in range(n_folds):
    print(f"\n[FOLD {fold + 1}/{n_folds}]")

    random.shuffle(all_apps)
    train_apps = all_apps[:45]
    test_apps = train_apps

    X_train_raw, y_train = load_app_data(train_apps, train_indices)
    X_test_raw, y_test = load_app_data(test_apps, test_indices)

    min_features = min(X_train_raw.shape[1], X_test_raw.shape[1])
    X_train_raw = X_train_raw[:, :min_features]
    X_test_raw = X_test_raw[:, :min_features]

    X_train_normal = X_train_raw[y_train == 0]
    y_test_binary = np.where(y_test == 0, 0, 1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test_raw)

    # === AUTOENCODER TRAINING ===
    input_dim = X_train_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation="relu")(input_layer)
    encoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    autoencoder.fit(X_train_scaled, X_train_scaled,
                    epochs=100,
                    batch_size=64,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[early_stop],
                    verbose=0)

    encoder = Model(inputs=input_layer, outputs=encoded)
    X_train_encoded = encoder.predict(X_train_scaled)
    X_test_encoded = encoder.predict(X_test_scaled)

    # === LSTM TRAINING ===
    X_train_seq = X_train_encoded.reshape((-1, 1, X_train_encoded.shape[1]))
    X_test_seq = X_test_encoded.reshape((-1, 1, X_test_encoded.shape[1]))

    lstm_model = Sequential()
    lstm_model.add(LSTM(32, input_shape=(1, X_train_encoded.shape[1]), return_sequences=False))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer=Adam(learning_rate=0.001),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    lstm_model.fit(X_train_seq, y_train[y_train == 0],  # Nur normale zum Trainieren
                   epochs=10,
                   batch_size=64,
                   verbose=0)

    y_pred_prob = lstm_model.predict(X_test_seq)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(f"\n[INFO] Fold {fold+1} – LSTM Evaluation")
    print(classification_report(y_test_binary, y_pred, digits=4, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred).ravel()
    dr = tp / (tp + fn) if (tp + fn) else 0
    far = fp / (fp + tn) if (fp + tn) else 0
    print(f"Detection Rate (DR): {dr:.4f}")
    print(f"False Alarm Rate (FAR): {far:.4f}")
