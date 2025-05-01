import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
import os
import random

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
n_runs = 5        # Anzahl Wiederholungen
n_test_cves = 5   # Anzahl Test-CVEs

fold_results = []

# 🚀 Schritt 1: EINMAL zufällig Test- und Trainings-CVEs bestimmen
random.shuffle(all_cves)
test_cves = all_cves[:n_test_cves]
train_cves = all_cves[n_test_cves:]

print(f"\n✅ Test-CVEs für alle Folds: {test_cves}")

# 🚀 Schritt 2: Cross-Validation
kf = KFold(n_splits=n_runs, shuffle=True, random_state=42)

# 🚀 Schritt 3: Daten laden und Cross-Validation ausführen
for fold_num, (train_idx, test_idx) in enumerate(kf.split(train_cves)):
    # Bestimmen der Trainings- und Test-CVEs für dieses Fold
    fold_train_cves = [train_cves[i] for i in train_idx]
    fold_test_cves = [train_cves[i] for i in test_idx]

    print(f"\nFold {fold_num + 1}:")
    print(f"Train CVEs: {fold_train_cves}")
    print(f"Test CVEs: {fold_test_cves}")

    # 📂 Daten für diesen Fold laden
    data_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.pkl') and any(cve in file for cve in fold_train_cves):
            df = pd.read_pickle(os.path.join(folder_path, file))
            data_list.append(df)
    train_data = pd.concat(data_list, ignore_index=True)
    
    data_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.pkl') and any(cve in file for cve in fold_test_cves):
            df = pd.read_pickle(os.path.join(folder_path, file))
            data_list.append(df)
    test_data = pd.concat(data_list, ignore_index=True)

    # Schritt 4: Vorverarbeitung der Daten
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Schritt 5: Normalisierung der Feature-Daten
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Schritt 6: Umwandlung der Eingabedaten für LSTM (3D-Format: [samples, timesteps, features])
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))  # 1 timestep
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))  # 1 timestep

    # Schritt 7: LSTM Modell erstellen
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    # Schritt 8: Modell kompilieren
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Schritt 9: Modell trainieren
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, validation_data=(X_test_scaled, y_test), verbose=0)

    # Schritt 10: Vorhersagen und Metriken berechnen
    y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

    # Berechnung von Precision, Recall, F1-Score, AUC und PR-AUC
    precision = precision_score(y_test, y_pred, average='macro')  # 'micro', 'macro', or 'weighted' 
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc_value = roc_auc_score(y_test, y_pred, average='macro', multi_class='ovr')  # 'ovr' for One-vs-Rest

    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, model.predict(X_test_scaled))
    pr_auc = auc(recall_curve, precision_curve)

    # Ergebnisse für dieses Fold speichern
    fold_results.append({
        'Fold': fold_num + 1,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'ROC-AUC': auc_value,
        'PR-AUC': pr_auc
    })

# Schritt 11: Alle Ergebnisse anzeigen
results_df = pd.DataFrame(fold_results)
print("\nFold Results:")
print(results_df)

# Schritt 12: Modell speichern
model.save('nids_lstm_model.keras')  # Speichern im Keras-native Format
