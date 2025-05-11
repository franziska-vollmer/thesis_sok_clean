import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

folder_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'

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

n_runs = 5
n_test_cves = 5
fold_results = []

random.shuffle(all_cves)
test_cves = all_cves[:n_test_cves]
train_cves = all_cves[n_test_cves:]

print(f"\n✅ Test-CVEs für alle Folds: {test_cves}")

kf = KFold(n_splits=n_runs, shuffle=True, random_state=42)

for fold_num, (train_idx, test_idx) in enumerate(kf.split(train_cves)):
    fold_train_cves = [train_cves[i] for i in train_idx]
    fold_test_cves = [train_cves[i] for i in test_idx]

    print(f"\nFold {fold_num + 1}:")
    print(f"Train CVEs: {fold_train_cves}")
    print(f"Test CVEs: {fold_test_cves}")

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

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    X_train_normal = X_train_scaled[y_train == 1]

    inputs = Input(shape=(X_train_normal.shape[1], X_train_normal.shape[2]))
    encoded = LSTM(64, activation='relu', return_sequences=False)(inputs)
    encoded = RepeatVector(X_train_normal.shape[1])(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
    decoded = TimeDistributed(Dense(X_train_normal.shape[2]))(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X_train_normal, X_train_normal, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

    reconstructions = autoencoder.predict(X_test_scaled)
    mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=(1, 2))

    threshold = np.percentile(mse, 95)
    y_pred = (mse > threshold).astype(int)

    y_test_binary = (y_test == -1).astype(int)

    precision = precision_score(y_test_binary, y_pred)
    recall = recall_score(y_test_binary, y_pred)
    f1 = f1_score(y_test_binary, y_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_binary, mse)
    pr_auc = auc(recall_curve, precision_curve)

    fold_results.append({
        'Fold': fold_num + 1,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'PR-AUC': pr_auc
    })

results_df = pd.DataFrame(fold_results)
print("\nFold Results:")
print(results_df)

# Optionale Speicherung des Modells
autoencoder.save('nids_lstm_autoencoder.keras')
