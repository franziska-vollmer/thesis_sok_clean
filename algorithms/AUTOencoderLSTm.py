import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score, classification_report, precision_recall_curve, average_precision_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === Hilfsfunktionen ===
def prepare_sequences_transposed_df(df, sequence_length=10):
    df_t = df.T.astype(float)
    features = df_t.iloc[:, :-1].values
    labels = df_t.iloc[:, -1].astype(int).values

    sequences = []
    sequence_labels = []

    for i in range(len(features) - sequence_length + 1):
        seq = features[i:i + sequence_length]
        label_seq = labels[i:i + sequence_length]
        sequences.append(seq)
        sequence_labels.append(int(np.any(label_seq != 0)))

    return np.array(sequences), np.array(sequence_labels)

def load_and_split_sequences_from_folder(folder_path, test_ratio=0.3, sequence_length=10, random_state=42):
    all_X, all_y = [], []
    feature_dims = []

    pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    pkl_paths = [os.path.join(folder_path, f) for f in pkl_files]

    print(f"{len(pkl_paths)} .pkl-Dateien gefunden.")

    # Erst alle Sequenzen laden und Feature-Längen erfassen
    for path in pkl_paths:
        try:
            with open(path, 'rb') as f:
                df = pickle.load(f)
            X_seq, y_seq = prepare_sequences_transposed_df(df, sequence_length)
            all_X.append(X_seq)
            all_y.append(y_seq)
            feature_dims.append(X_seq.shape[2])
        except Exception as e:
            print(f"Fehler bei {path}: {e}")

    # Gemeinsame Feature-Anzahl ermitteln
    min_features = min(feature_dims)
    print(f"Reduziere alle Sequenzen auf {min_features} Features.")

    # Trim alle Sequenzen auf gleiche Dim.
    all_X_trimmed = [x[:, :, :min_features] for x in all_X]

    X = np.vstack(all_X_trimmed)
    y = np.concatenate(all_y)

    return train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=random_state)

# === Pfad anpassen ===
folder =  "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
X_train, X_test, y_train, y_test = load_and_split_sequences_from_folder(folder)

# === Modelltraining ===
input_shape = (X_train.shape[1], X_train.shape[2])
latent_dim = 64

input_seq = Input(shape=input_shape)
encoded = LSTM(latent_dim, activation='relu')(input_seq)
decoded = RepeatVector(input_shape[0])(encoded)
decoded = LSTM(input_shape[1], activation='relu', return_sequences=True)(decoded)

autoencoder = Model(inputs=input_seq, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
autoencoder.fit(
    X_train[y_train == 0],
    X_train[y_train == 0],
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# === Auswertung ===
X_pred = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - X_pred), axis=(1, 2))

fpr, tpr, thresholds = roc_curve(y_test, reconstruction_error)
f1_scores = [f1_score(y_test, reconstruction_error > t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
y_pred = (reconstruction_error > best_thresh).astype(int)

f1 = f1_score(y_test, y_pred)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, reconstruction_error)
pr_auc = average_precision_score(y_test, reconstruction_error)
report = classification_report(y_test, y_pred, zero_division=0)

print("=== Ergebnisse ===")
print(f"Bester Schwellenwert: {best_thresh:.6f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print("\nClassification Report:\n", report)

# === Plots ===
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()