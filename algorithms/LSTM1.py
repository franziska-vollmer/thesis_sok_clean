import os
import numpy as np
import pandas as pd
import testpkl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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
y_train = df_train_all.iloc[:, -1]
X_test = df_test_all.iloc[:, :-1]
y_test = df_test_all.iloc[:, -1]

y_train = y_train.replace(-1, 0)
y_test = y_test.replace(-1, 0)


# Klassenverteilung ausgeben
print("Train-Labels:")
print(y_train.value_counts())
print("\nTest-Labels:")
print(y_test.value_counts())

# === Normalisierung ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape für LSTM: (samples, timesteps, features)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# === Modell definieren ===
model = Sequential()
model.add(LSTM(64, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Für binäre Klassifikation

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Modell trainieren ===
history = model.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_scaled, y_test),
    class_weight = {0: 10, 1: 1},  # <<< Gewichtung hinzugefügt
    verbose=1
)
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# === Vorhersage und Bewertung ===
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)


# === Wahrscheinlichkeiten berechnen ===
y_pred_prob = model.predict(X_test_scaled).ravel()

# === Threshold-Tuning ===



print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Wahrscheinlichkeiten für Klasse 1 (anomal)
y_score = model.predict(X_test_scaled).ravel()

# ROC-Daten berechnen
fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
roc_auc = auc(fpr, tpr)

# Plotten
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'LSTM ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (LSTM)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('ROC_LSTM_scenario_s1_d1d1.png')
plt.show()

