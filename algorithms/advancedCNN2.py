import os
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from imblearn.over_sampling import SMOTE

# -----------------------------
# Focal Loss Definition
# -----------------------------
def focal_loss(gamma=2., alpha=.25):
    """
    Focal Loss für binäre Klassifikation.
    """
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-8
        y_pred = K.clip(y_pred, eps, 1. - eps)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = weight * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss)
    return focal_loss_fixed

# -----------------------------
# 1) Daten laden und vorbereiten
# -----------------------------
# === Pfade anpassen ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

# === Daten laden ===
with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

train_indices = [1, 2, 3, 4]
test_indices  = [1, 2, 3, 4]
df_train_all  = pd.DataFrame()
df_test_all   = pd.DataFrame()

for line in app_lines:
    app = line.strip()
    for i in train_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df_temp = pickle.load(f)
        df_train_all = pd.concat([df_train_all, df_temp], ignore_index=True)
    for i in test_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df_temp = pickle.load(f)
        df_test_all = pd.concat([df_test_all, df_temp], ignore_index=True)

# === Feature + Label ===
X_train = df_train_all.iloc[:, :-1]
y_train = df_train_all.iloc[:, -1].replace(-1, 0)
X_test  = df_test_all.iloc[:, :-1]
y_test  = df_test_all.iloc[:, -1].replace(-1, 0)

print("Train label distribution:\n", y_train.value_counts())
print("Test label distribution:\n",  y_test.value_counts())

# -----------------------------
# 2) SMOTE-Oversampling der Trainingsdaten
# -----------------------------
# SMOTE sorgt dafür, dass beide Klassen gleich häufig vertreten sind.
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE oversampling:")
print(pd.Series(y_train_sm).value_counts())

# -----------------------------
# 3) Skalierung und Reshape für CNN
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

# Reshape: (samples, features, 1)
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn  = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# -----------------------------
# 4) Verbessertes CNN-Modell mit Focal Loss
# -----------------------------
model = Sequential()
# Erste Conv-Schicht
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# Zweite Conv-Schicht
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# Dritte Conv-Schicht
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# Global Max-Pooling
model.add(GlobalMaxPooling1D())
# Dense-Block
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# -----------------------------
# 5) Training
# -----------------------------
history = model.fit(
    X_train_cnn, y_train_sm,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# -----------------------------
# 6) Vorhersage und Evaluation
# -----------------------------
y_pred_prob = model.predict(X_test_cnn).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec  = recall_score(y_test, y_pred, average='macro')
f1v  = f1_score(y_test, y_pred, average='macro')

print("\n=== Evaluation auf Testdaten ===")
print("Accuracy:", acc)
print("Precision (macro):", prec)
print("Recall (macro):   ", rec)
print("F1 (macro):       ", f1v)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# 7) Threshold-Tuning (optional)
# -----------------------------
thresholds = np.linspace(0, 1, 101)
best_thr = 0.5
best_f1  = 0

for thr in thresholds:
    y_temp = (y_pred_prob >= thr).astype(int)
    f1_anomaly = f1_score(y_test, y_temp, pos_label=0)
    if f1_anomaly > best_f1:
        best_f1 = f1_anomaly
        best_thr = thr

print(f"\nBester Threshold für F1(Klasse=0): {best_thr:.2f}, F1_0={best_f1:.4f}")

y_pred_opt = (y_pred_prob >= best_thr).astype(int)
prec0 = precision_score(y_test, y_pred_opt, pos_label=0)
rec0  = recall_score(y_test, y_pred_opt, pos_label=0)
f10   = f1_score(y_test, y_pred_opt, pos_label=0)
print(f"Precision(0)={prec0:.4f}, Recall(0)={rec0:.4f}, F1(0)={f10:.4f}")

# -----------------------------
# 8) ROC-Kurve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label=1)
auc_val = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC={auc_val:.2f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - CNN mit Focal Loss & SMOTE')
plt.legend()
plt.grid(True)
plt.show()
