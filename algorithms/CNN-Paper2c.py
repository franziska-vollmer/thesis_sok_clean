import os
import glob
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# === Konfiguration ===
TRAIN_CVES = [
    'CVE-2012-1823', 'CVE-2014-0050', 'CVE-2014-0160', 'CVE-2014-3120', 'CVE-2014-6271',
    'CVE-2015-1427', 'CVE-2015-2208', 'CVE-2015-3306', 'CVE-2015-5477', 'CVE-2015-5531',
    'CVE-2015-8103', 'CVE-2015-8562', 'CVE-2016-3088', 'CVE-2016-3714', 'CVE-2016-6515',
    'CVE-2016-7434', 'CVE-2016-9920', 'CVE-2016-10033', 'CVE-2017-5638', 'CVE-2017-7494',
    'CVE-2017-7529', 'CVE-2017-8291', 'CVE-2017-8917', 'CVE-2017-11610', 'CVE-2017-12149',
    'CVE-2017-12615', 'CVE-2017-12635', 'CVE-2017-12794', 'CVE-2018-11776', 'CVE-2018-15473',
    'CVE-2018-16509', 'CVE-2018-19475', 'CVE-2018-19518', 'CVE-2019-5420', 'CVE-2019-6116',
    'CVE-2019-6116b', 'CVE-2019-10758', 'CVE-2020-1938', 'CVE-2020-17530', 'CVE-2021-28164',
    'CVE-2021-28169', 'CVE-2021-34429', 'CVE-2021-41773', 'CVE-2021-42013', 'CVE-2021-44228',
    'CVE-2022-0847', 'CVE-2022-21449', 'CVE-2022-22963', 'CVE-2022-22965', 'CVE-2022-26134'
]

FOLDER_PATH = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'

# === Funktionen ===
def load_data():
    all_files = glob.glob(os.path.join(FOLDER_PATH, '*.pkl'))
    train_files, test_files = [], []

    for file in all_files:
        filename = os.path.basename(file)
        cve_name = '-'.join(filename.split('-')[:3])
        if cve_name in TRAIN_CVES:
            train_files.append(file)
        else:
            test_files.append(file)

    def load_files(file_list):
        dfs = [pd.read_pickle(f) for f in file_list]
        df = pd.concat(dfs, ignore_index=True)
        X = df.iloc[:, :-1].values
        y = (df.iloc[:, -1] > 0).astype(int).values
        return X, y

    return train_files, test_files, load_files

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Hauptlogik ===
def main():
    print('📥 Lade Daten...')
    train_files, test_files, load_files = load_data()

    X_train_full, y_train_full = load_files(train_files)

    scaler = MinMaxScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_train_full = np.expand_dims(X_train_full, axis=2)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        print(f"\n🚀 Starte Fold {fold}")

        X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

        model = create_cnn_model((X_train_full.shape[1], 1))
        model.fit(X_tr, y_tr, epochs=10, batch_size=64, verbose=0)

        # Dynamische Auswahl von Testdaten
        sample_size = min(len(test_files), 5 * 4)
        unseen_sample = random.sample(test_files, sample_size)
        X_unseen, y_unseen = load_files(unseen_sample)

        X_unseen = scaler.transform(X_unseen)
        X_unseen = np.expand_dims(X_unseen, axis=2)

        y_pred_probs = model.predict(X_unseen)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        # Metriken berechnen
        precision = precision_score(y_unseen, y_pred)
        recall = recall_score(y_unseen, y_pred)
        f1 = f1_score(y_unseen, y_pred)
        auc = roc_auc_score(y_unseen, y_pred_probs)
        pr_auc = average_precision_score(y_unseen, y_pred_probs)
        accuracy = accuracy_score(y_unseen, y_pred)

        fold_results.append({
            'Fold': fold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
            'PR_AUC': pr_auc
        })

        print(f"Fold {fold} Ergebnisse: "
              f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, "
              f"F1={f1:.4f}, AUC={auc:.4f}, PR_AUC={pr_auc:.4f}")

    # Ergebnisse als DataFrame
    results_df = pd.DataFrame(fold_results)
    results_df = results_df[['Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'PR_AUC']]

    # Durchschnitt berechnen und hinzufügen
    mean_values = results_df.drop(columns='Fold').mean(numeric_only=True)
    mean_row = pd.DataFrame([['Durchschnitt'] + mean_values.tolist()],
                            columns=['Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'PR_AUC'])

    final_df = pd.concat([results_df, mean_row], ignore_index=True)

    # Ausgabe
    print("\n=== 📊 Gesamtergebnis ===")
    print(final_df.to_string(index=False, float_format="%.4f"))