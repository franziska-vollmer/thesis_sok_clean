import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# === Pfade ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

# === Dateien einlesen ===
with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

# === Daten laden ===
train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]
df_train_all = pd.DataFrame()
df_test_all = pd.DataFrame()

for line in app_lines:
    app = line.strip()
    for i in train_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df = pickle.load(f)
        df_train_all = pd.concat([df_train_all, df])
    for i in test_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df = pickle.load(f)
        df_test_all = pd.concat([df_test_all, df])

# === Feature-Label-Split ===
X_train = df_train_all.iloc[:, :-1]
y_train = df_train_all.iloc[:, -1].replace(-1, 0)
X_test = df_test_all.iloc[:, :-1]
y_test = df_test_all.iloc[:, -1].replace(-1, 0)

print("Train-Labels:")
print(y_train.value_counts())
print("\nTest-Labels:")
print(y_test.value_counts())

# === Normalisierung ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Nur normale Daten für Autoencoder-Training ===
X_train_normal = X_train_scaled[y_train == 1]

# === Autoencoder Modell definieren ===
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
bottleneck = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(bottleneck)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# === Training ===
history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=10,
    batch_size=512,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

# === Rekonstruktionsfehler berechnen (Train) ===
reconstructions_train = autoencoder.predict(X_train_normal)
mse_train = np.mean(np.square(X_train_normal - reconstructions_train), axis=1)

# === Threshold (z.B. 95. Perzentil) ===
threshold = np.percentile(mse_train, 95)
print(f"\nThreshold (95. Perzentil): {threshold}")

# === Testdaten Fehler ===
reconstructions_test = autoencoder.predict(X_test_scaled)
mse_test = np.mean(np.square(X_test_scaled - reconstructions_test), axis=1)

# === Vorhersagen (0 = Anomalie, 1 = Normal) ===
y_pred = (mse_test < threshold).astype(int)

# === Evaluation ===
print("\n=== Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# === ROC-Kurve ===
fpr, tpr, _ = roc_curve(y_test, mse_test, pos_label=0)  # 0 = Anomalie
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Autoencoder')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('ROC_AE.png')
plt.show()
