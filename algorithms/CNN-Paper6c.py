import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve
from tqdm import tqdm

# 📂 Ordner mit Pickle-Dateien
folder_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'

# 📋 Alle verfügbaren CVEs
all_cves = [
    "CVE-2012-1823", "CVE-2014-0050", "CVE-2014-0160", "CVE-2014-3120", "CVE-2014-6271",
    "CVE-2015-1427", "CVE-2015-2208", "CVE-2015-3306", "CVE-2015-5477", "CVE-2015-5531",
    "CVE-2015-8103", "CVE-2015-8562", "CVE-2016-3088", "CVE-2016-3714", "CVE-2016-6515",
    "CVE-2016-7434", "CVE-2016-9920", "CVE-2016-10033", "CVE-2017-5638", "CVE-2017-7494",
    "CVE-2017-7529", "CVE-2017-8291", "CVE-2017-8917", "CVE-2017-11610", "CVE-2017-12149",
    "CVE-2017-12615", "CVE-2017-12635", "CVE-2017-12794", "CVE-2018-11776", "CVE-2018-15473",
    "CVE-2018-16509", "CVE-2018-19475", "CVE-2018-19518", "CVE-2019-5420", "CVE-2019-6116",
    "CVE-2019-6116b", "CVE-2019-10758", "CVE-2020-1938", "CVE-2020-17530", "CVE-2021-28164",
    "CVE-2021-28169", "CVE-2021-34429", "CVE-2021-41773", "CVE-2021-42013", "CVE-2021-44228",
    "CVE-2022-0847", "CVE-2022-21449", "CVE-2022-22963", "CVE-2022-22965", "CVE-2022-26134",
    "CVE-2022-42889", "CVE-2023-23752"
]

# 📋 Einstellungen
n_runs = 5       # Anzahl Wiederholungen
n_test_cves = 5  # Anzahl Test-CVEs pro Runde

results = []

for run in range(n_runs):
    print(f"\n🚀 Starte Run {run+1}/{n_runs}")

    # CVEs zufällig aufteilen
    random.shuffle(all_cves)
    test_cves = all_cves[:n_test_cves]
    train_cves = all_cves[n_test_cves:]

    # Dateien laden
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    train_files = [f for f in pkl_files if any(cve in f for cve in train_cves)]
    test_files = [f for f in pkl_files if any(cve in f for cve in test_cves)]

    train_dfs = [pd.read_pickle(os.path.join(folder_path, f)) for f in tqdm(train_files, desc="Lade Trainingsdaten")]
    test_dfs = [pd.read_pickle(os.path.join(folder_path, f)) for f in tqdm(test_files, desc="Lade Testdaten")]

    train_data = pd.concat(train_dfs, ignore_index=True)
    test_data = pd.concat(test_dfs, ignore_index=True)

    # Features und Labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    y_train = np.where(y_train == -1, 0, y_train)
    y_test = np.where(y_test == -1, 0, y_test)

    # Skalierung
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modell bauen
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

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Training
    history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=256, validation_split=0.2, verbose=0, callbacks=[early_stop])

    # Testen
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)
    pr_auc = average_precision_score(y_test, y_pred_probs)  # ✅ PR-AUC

    print(f"Run {run+1}: Acc {test_accuracy:.4f}, Precision {precision:.4f}, Recall {recall:.4f}, F1 {f1:.4f}, ROC-AUC {auc:.4f}, PR-AUC {pr_auc:.4f}")

    results.append({
        "accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc
    })

# 📊 Alle Ergebnisse zusammenfassen
results_df = pd.DataFrame(results)
print("\n📋 Cross-Validation Ergebnisse:")
print(results_df)

print("\n📈 Mittelwerte über alle Runs:")
print(results_df.mean())

print("\n📈 Standardabweichungen über alle Runs:")
print(results_df.std())

# 📈 Boxplots zur Visualisierung
plt.figure(figsize=(14,7))
sns.boxplot(data=results_df)
plt.title("Cross-Validation Performance über verschiedene Test-CVE-Splits (inkl. ROC-AUC & PR-AUC)")
plt.grid(True)
plt.show()
