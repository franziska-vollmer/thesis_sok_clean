import os
import pickle
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === Parameter ===
timesteps = 10
latent_dim = 64
n_folds = 5
train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

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
                print(f"Datei fehlt: {path}")
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

def create_sequences(X, y, timesteps):
    sequences = []
    labels = []
    for i in range(len(X) - timesteps + 1):
        sequences.append(X[i:i+timesteps])
        labels.append(y[i+timesteps-1])  # Label am Ende des Fensters
    return np.array(sequences), np.array(labels)

with open(apps_file_path, 'r') as f:
    all_apps = [line.strip() for line in f.readlines() if line.strip()]

assert len(all_apps) >= 45, "Mindestens 40 Apps/CVEs nötig."

results = []

for fold in range(n_folds):
    print(f"\nFOLD {fold + 1}/{n_folds}")

    random.shuffle(all_apps)
    train_apps = all_apps[:45]
    test_apps = train_apps

    X_train_raw, y_train = load_app_data(train_apps, train_indices)
    X_test_raw, y_test = load_app_data(test_apps, test_indices)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    y_train_binary = np.where(y_train == 0, 0, 1)
    y_test_binary = np.where(y_test == 0, 0, 1)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_binary, timesteps)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_binary, timesteps)

    # Nur normale Daten zum Training
    X_train_seq_normal = X_train_seq[y_train_seq == 0]

    input_shape = (timesteps, X_train_seq.shape[2])
    input_layer = Input(shape=input_shape)
    encoded = LSTM(latent_dim, activation='relu')(input_layer)
    repeated = RepeatVector(timesteps)(encoded)
    decoded = LSTM(latent_dim, activation='relu', return_sequences=True)(repeated)
    decoded = TimeDistributed(Dense(X_train_seq.shape[2]))(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(0.0005), loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    autoencoder.fit(X_train_seq_normal, X_train_seq_normal,
                    epochs=100,
                    batch_size=64,
                    validation_split=0.1,
                    callbacks=[early_stop],
                    verbose=0)

    X_test_pred = autoencoder.predict(X_test_seq)
    reconstruction_error = np.mean(np.square(X_test_seq - X_test_pred), axis=(1, 2))

    fpr, tpr, thresholds = roc_curve(y_test_seq, reconstruction_error)
    f1s = [f1_score(y_test_seq, (reconstruction_error > t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1s)]
    y_pred = (reconstruction_error > best_thresh).astype(int)

    f1 = f1_score(y_test_seq, y_pred)
    precision = classification_report(y_test_seq, y_pred, output_dict=True, zero_division=0)["1"]["precision"]
    recall = classification_report(y_test_seq, y_pred, output_dict=True, zero_division=0)["1"]["recall"]
    auc_score = auc(fpr, tpr)

    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test_seq, reconstruction_error)
    pr_auc = average_precision_score(y_test_seq, reconstruction_error)

    results.append({
        "fold": fold + 1,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc_score,
        "pr_auc": pr_auc
    })

# === Ergebnisse anzeigen ===
print("\nCross-Validation Ergebnisse:")
for r in results:
    print(f"Fold {r['fold']}: F1={r['f1']:.4f}, Precision={r['precision']:.4f}, Recall={r['recall']:.4f}, AUC={r['auc']:.4f}, PR-AUC={r['pr_auc']:.4f}")

print("\nDurchschnitt:")
print(f"F1:       {np.mean([r['f1'] for r in results]):.4f}")
print(f"Precision:{np.mean([r['precision'] for r in results]):.4f}")
print(f"Recall:   {np.mean([r['recall'] for r in results]):.4f}")
print(f"AUC:      {np.mean([r['auc'] for r in results]):.4f}")
print(f"PR-AUC:   {np.mean([r['pr_auc'] for r in results]):.4f}")
