import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm

# 📂 Ordner mit Pickle-Dateien
folder_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'

# 📋 Einstellungen
n_runs = 5  # Anzahl Wiederholungen
test_size = 0.2  # Anteil für Testdaten (20%)

fold_results = []

# 📥 Alle Pickle-Dateien laden
pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
all_dfs = [pd.read_pickle(os.path.join(folder_path, f)) for f in tqdm(pkl_files, desc="Lade alle Daten")]
all_data = pd.concat(all_dfs, ignore_index=True)

# 🧹 Features und Labels
X = all_data.iloc[:, :-1].values
y = all_data.iloc[:, -1].values
y = np.where(y == -1, 0, y)  # Label-Korrektur

# 🚀 Wiederholte Trainings-/Testdurchläufe mit zufälligem Split
for run in range(1, n_runs + 1):
    print(f"\n🚀 Starte Fold {run}/{n_runs}")

    # 🔀 Split in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=run, stratify=y)

    # 🧼 Skalierung
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🧠 Modell erstellen
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 📚 Training
    model.fit(X_train_scaled, y_train, epochs=30, batch_size=256,
              validation_split=0.2, verbose=0, callbacks=[early_stop])

    # 📈 Vorhersagen & Metriken
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    pr_auc = average_precision_score(y_test, y_pred_probs)

    fold_results.append({
        "Fold": run,
        "Accuracy": round(accuracy, 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4)
    })

# 📊 Ergebnisse
results_df = pd.DataFrame(fold_results)

print("\n📋 Fold-Ergebnisse:")
print(results_df.to_string(index=False))

print("\n📈 Mittelwerte über alle Folds:")
print(results_df.mean(numeric_only=True))

print("\n📈 Standardabweichungen über alle Folds:")
print(results_df.std(numeric_only=True))

# 📦 Optional: Speichern als CSV
results_df.to_csv("fold_results_random_split.csv", index=False)
