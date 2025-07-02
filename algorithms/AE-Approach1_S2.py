import sys
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("TF version:", tf.__version__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]
n_folds = 5  # Anzahl der Cross-Validation-Runden

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

# Apps laden
with open(apps_file_path, 'r') as f:
    all_apps = [line.strip() for line in f.readlines() if line.strip()]
assert len(all_apps) >= 45, "Mindestens 45 Apps/CVEs nötig."

results = []

for fold in range(n_folds):
    print(f"\n FOLD {fold + 1}/{n_folds} ")

    random.shuffle(all_apps)
    train_apps = all_apps[:45]
    test_apps = all_apps[45:50]

    X_train_raw, y_train = load_app_data(train_apps, train_indices)
    X_test_raw, y_test = load_app_data(test_apps, test_indices)

    # Features angleichen
    min_features = min(X_train_raw.shape[1], X_test_raw.shape[1])
    X_train_raw = X_train_raw[:, :min_features]
    X_test_raw = X_test_raw[:, :min_features]

    X_train_normal = X_train_raw[y_train == 0]
    y_test_binary = np.where(y_test == 0, 0, 1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test_raw)

    # Autoencoder Modell
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

    X_test_pred = autoencoder.predict(X_test_scaled)
    reconstruction_error = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)

    # ROC-AUC
    fpr, tpr, thresholds = roc_curve(y_test_binary, reconstruction_error)
    f1s = [f1_score(y_test_binary, (reconstruction_error > t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1s)]
    y_pred = (reconstruction_error > best_thresh).astype(int)

    f1 = f1_score(y_test_binary, y_pred)
    precision = classification_report(y_test_binary, y_pred, output_dict=True, zero_division=0)["1"]["precision"]
    recall = classification_report(y_test_binary, y_pred, output_dict=True, zero_division=0)["1"]["recall"]
    auc_score = auc(fpr, tpr)

    # PR-AUC
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test_binary, reconstruction_error)
    pr_auc = average_precision_score(y_test_binary, reconstruction_error)

   # Accuracy
    accuracy = accuracy_score(y_test_binary, y_pred)

    results.append({
        "fold": fold + 1,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc_score,
        "pr_auc": pr_auc,
        "accuracy": accuracy  # Hinzufügen der Genauigkeit
    })

# === Ergebnisse anzeigen ===
f1s = [r["f1"] for r in results]
precisions = [r["precision"] for r in results]
recalls = [r["recall"] for r in results]
aucs = [r["auc"] for r in results]
pr_aucs = [r["pr_auc"] for r in results]
accuracies = [r["accuracy"] for r in results]  # Genauigkeit speichern

print("\n Cross-Validation Ergebnisse (auf 5 Test-CVEs pro Fold):")
for r in results:
    print(f"Fold {r['fold']}: F1={r['f1']:.4f}, Precision={r['precision']:.4f}, Recall={r['recall']:.4f}, AUC={r['auc']:.4f}, PR-AUC={r['pr_auc']:.4f}, Accuracy={r['accuracy']:.4f}")

print("\n Durchschnitt:")
print(f"F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"PR-AUC: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")  # Durchschnittliche Genauigkeit anzeigen


