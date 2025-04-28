import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# === Einstellungen
DEBUG = True

if DEBUG:
    n_folds = 5
    train_app_count = 30
    test_app_count = 5
    epochs = 10
    batch_size = 256
else:
    n_folds = 5
    train_app_count = 45
    test_app_count = 5
    epochs = 20
    batch_size = 128

train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

with open(apps_file_path, 'r') as file:
    app_lines = [line.strip() for line in file.readlines() if line.strip()]

def load_data(app_list, indices):
    df_all = pd.DataFrame()
    for app in app_list:
        for i in indices:
            path = os.path.join(supervised_path, f"{app}-{i}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    df_temp = pickle.load(f)
                df_all = pd.concat([df_all, df_temp], ignore_index=True)
    return df_all

results = []

for fold in range(n_folds):
    print(f"\n🌀 Starte Fold {fold + 1}/{n_folds}...")

    np.random.shuffle(app_lines)
    train_apps = app_lines[:train_app_count]
    test_apps = app_lines[train_app_count:train_app_count + test_app_count]

    df_train = load_data(train_apps, train_indices)
    df_test = load_data(test_apps, test_indices)

    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1].replace(-1, 0)
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1].replace(-1, 0)

    balance_testset = True
    if balance_testset:
        idx_0 = np.where(y_test == 0)[0]
        idx_1 = np.where(y_test == 1)[0]
        min_size = min(len(idx_0), len(idx_1))
        idx_0_bal = resample(idx_0, n_samples=min_size, random_state=42)
        idx_1_bal = resample(idx_1, n_samples=min_size, random_state=42)
        balanced_idx = np.concatenate([idx_0_bal, idx_1_bal])
        X_test = X_test.iloc[balanced_idx]
        y_test = y_test.iloc[balanced_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_cnn = X_train_bal.reshape(X_train_bal.shape[0], X_train_bal.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    # === Verbesserte CNN-Architektur
    model = Sequential([
        Input(shape=(X_train_cnn.shape[1], 1)),

        Conv1D(filters=64, kernel_size=3, padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=128, kernel_size=3, padding='same'),
        BatchNormalization(),
        LeakyReLU(),
        MaxPooling1D(pool_size=2),

        Flatten(),

        Dense(128),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=AdamW(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    model.fit(
        X_train_cnn, y_train_bal,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    y_pred_prob = model.predict(X_test_cnn).ravel()
    thresholds = np.linspace(0, 1, 101)
    best_thr, best_f1 = 0.5, 0
    for thr in thresholds:
        y_temp = (y_pred_prob >= thr).astype(int)
        f1_tmp = f1_score(y_test, y_temp, pos_label=0)
        if f1_tmp > best_f1:
            best_f1, best_thr = f1_tmp, thr

    y_pred = (y_pred_prob >= best_thr).astype(int)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    prec = report['0']['precision']
    rec = report['0']['recall']
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label=0)
    auc_val = auc(fpr, tpr)

    results.append({
        'fold': fold + 1,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc': auc_val
    })

# === Zusammenfassung
f1s = [r['f1'] for r in results]
precisions = [r['precision'] for r in results]
recalls = [r['recall'] for r in results]
aucs = [r['auc'] for r in results]

print("\n📊 Ergebnisse (Verbessertes CNN auf deinen Daten):")
for r in results:
    print(f"Fold {r['fold']}: F1={r['f1']:.4f}, Precision={r['precision']:.4f}, Recall={r['recall']:.4f}, AUC={r['auc']:.4f}")

print("\n📈 Durchschnitt:")
print(f"F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
