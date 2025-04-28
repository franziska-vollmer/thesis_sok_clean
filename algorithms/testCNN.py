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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, LeakyReLU, Dropout, Add,
    GlobalMaxPooling1D, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE

# === Datenpfade ===
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

# === Features & Labels
X_train = df_train_all.iloc[:, :-1]
y_train = df_train_all.iloc[:, -1].replace(-1, 0)
X_test  = df_test_all.iloc[:, :-1]
y_test  = df_test_all.iloc[:, -1].replace(-1, 0)

# === SMOTE + Skalierung
print("Vor SMOTE Klassenverteilung:", Counter(y_train))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_bal_smote, y_train_bal_smote = smote.fit_resample(X_train_scaled, y_train)
print("Nach SMOTE Klassenverteilung:", Counter(y_train_bal_smote))

# === Reshape fürs CNN
X_train_cnn = X_train_bal_smote.reshape(X_train_bal_smote.shape[0], X_train_bal_smote.shape[1], 1)
X_test_scaled  = scaler.transform(X_test)
X_test_cnn  = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# === CNN Modell
input_layer = Input(shape=(X_train_cnn.shape[1], 1))

x = Conv1D(64, kernel_size=3, padding='same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.3)(x)

x1 = Conv1D(64, kernel_size=3, padding='same')(x)
x1 = BatchNormalization()(x1)
x1 = LeakyReLU()(x1)
x1 = Dropout(0.3)(x1)

x = Add()([x, x1])

x = Conv1D(128, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.4)(x)

x = Conv1D(128, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.4)(x)

x = GlobalMaxPooling1D()(x)
x = Dense(128)(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# === FULL TRAINING
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(
    X_train_cnn, y_train_bal_smote,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === Balanciertes Testset für faire Bewertung
df_test = pd.concat([X_test, y_test], axis=1)
anomalies = df_test[df_test.iloc[:, -1] == 0]
normals   = df_test[df_test.iloc[:, -1] == 1].sample(n=len(anomalies), random_state=42)

df_test_balanced = pd.concat([anomalies, normals]).sample(frac=1.0, random_state=42)
X_test_bal = df_test_balanced.iloc[:, :-1]
y_test_bal = df_test_balanced.iloc[:, -1]

X_test_bal_scaled = scaler.transform(X_test_bal)
X_test_bal_cnn = X_test_bal_scaled.reshape(X_test_bal_scaled.shape[0], X_test_bal_scaled.shape[1], 1)

# === Evaluation auf Balanced Testset
y_pred_prob_bal = model.predict(X_test_bal_cnn).ravel()
y_pred_bal = (y_pred_prob_bal >= 0.5).astype(int)

print("\n=== Evaluation auf BALANCIERTEM Testset ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test_bal, y_pred_bal))
print(classification_report(y_test_bal, y_pred_bal))

# === Threshold-Tuning
thresholds = np.linspace(0, 1, 101)
best_thr = 0.5
best_f1 = 0

for thr in thresholds:
    y_temp = (y_pred_prob_bal >= thr).astype(int)
    f1_anomaly = f1_score(y_test_bal, y_temp, pos_label=0)
    if f1_anomaly > best_f1:
        best_f1 = f1_anomaly
        best_thr = thr

print(f"\nOptimaler Threshold für F1(0) auf balanced Testset: {best_thr:.2f}, F1_0 = {best_f1:.4f}")

# === Trainingsverlauf anzeigen
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss-Verlauf')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy-Verlauf')
plt.legend()

plt.tight_layout()
plt.show()
