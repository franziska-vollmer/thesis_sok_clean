import os
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# -----------------------------------------------------------------
# 1) UnterSampling der Normaldaten (Beispiel: wir nehmen z.B. 5:1-Verhältnis)
# -----------------------------------------------------------------
df_train = pd.concat([X_train, y_train], axis=1)
normal_df   = df_train[df_train.iloc[:, -1] == 1]
anomaly_df  = df_train[df_train.iloc[:, -1] == 0]

print("Normal train count:", len(normal_df))
print("Anomaly train count:", len(anomaly_df))

# Wir wollen z.B. ein 5:1-Verhältnis oder 10:1 – je nach Geschmack
target_ratio = 5  # d.h. 5x mehr Normal als Anomalie
anomaly_count = len(anomaly_df)
desired_norm_count = target_ratio * anomaly_count

if len(normal_df) > desired_norm_count:
    normal_df_undersampled = normal_df.sample(n=desired_norm_count, random_state=42)
else:
    normal_df_undersampled = normal_df  # Falls Normal schon weniger ist, unlikely
df_train_balanced = pd.concat([normal_df_undersampled, anomaly_df], ignore_index=True)

# Shuffle
df_train_balanced = df_train_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

print("After undersampling:")
print(df_train_balanced.iloc[:, -1].value_counts())

# Wieder splitten in X, y
X_train_bal = df_train_balanced.iloc[:, :-1]
y_train_bal = df_train_balanced.iloc[:,  -1]

# === Skalierung ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled  = scaler.transform(X_test)

# === Reshape für CNN => (samples, features, 1)
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn  = X_test_scaled.reshape(X_test_scaled.shape[0],   X_test_scaled.shape[1], 1)

# === class_weight (optional), falls immer noch unbalanced
from collections import Counter
counter = Counter(y_train_bal)
major_cl = counter.most_common(1)[0][0]
minor_cl = 1 - major_cl
count_major = counter[major_cl]
count_minor = counter[minor_cl] if minor_cl in counter else 1
weight_factor = count_major / count_minor

if major_cl == 0:
    class_weight = {0:1.0, 1:weight_factor}
else:
    class_weight = {0:weight_factor, 1:1.0}

print("class_weight:", class_weight)

# -----------------------------------------------------------------
# 2) CNN-Modell
# -----------------------------------------------------------------
model = Sequential()

# Convolution 1
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Convolution 2
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())

# Global Max-Pooling
model.add(GlobalMaxPooling1D())

# Dense-Block
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# === Training
history = model.fit(
    X_train_cnn, y_train_bal,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight,  # optional
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === Vorhersage
y_pred_prob = model.predict(X_test_cnn).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec= precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1v = f1_score(y_test, y_pred, average='macro')

print("\n=== CNN balanced (UnderSample) ===")
print("Accuracy:", acc)
print("Precision (macro):", prec)
print("Recall (macro):   ", rec)
print("F1 (macro):       ", f1v)
print("Confusion:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------
# 3) Threshold-Tuning (optional)
# -----------------------------------------------------------------
# z.B. wir suchen nach bestmöglicher F1 (Klasse=0).
thresholds = np.linspace(0, 1, 101)
best_thr   = 0.5
best_f1    = 0

for thr in thresholds:
    y_temp = (y_pred_prob >= thr).astype(int)
    # F1 für Klasse 0 (Anomalie) => pos_label=0
    from sklearn.metrics import f1_score
    f1_anomaly = f1_score(y_test, y_temp, pos_label=0)
    if f1_anomaly > best_f1:
        best_f1   = f1_anomaly
        best_thr  = thr

print(f"\nBester Threshold fuer F1(Klasse=0): {best_thr:.2f}, F1_0={best_f1:.4f}")

# Neue Vorhersage mit best_thr
y_pred_opt = (y_pred_prob >= best_thr).astype(int)
prec0 = precision_score(y_test, y_pred_opt, pos_label=0)
rec0  = recall_score(y_test, y_pred_opt, pos_label=0)
f10   = f1_score(y_test, y_pred_opt, pos_label=0)
print(f"Precision(0)={prec0:.4f}, Recall(0)={rec0:.4f}, F1(0)={f10:.4f}")

# ROC-Kurve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label=1)
auc_val = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC={auc_val:.2f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - CNN balanced + threshold search')
plt.legend()
plt.grid(True)
plt.show()
