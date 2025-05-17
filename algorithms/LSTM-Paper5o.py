import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold

# 1. Verzeichnis durchsuchen, um alle .pkl Dateien zu finden
directory_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
pkl_files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]

# Leere Liste für DataFrames
dfs = []

# 2. Alle .pkl Dateien durchgehen und laden
for file in pkl_files:
    file_path = os.path.join(directory_path, file)
    print(f"Lade Datei: {file_path}")
    try:
        data = pd.read_pickle(file_path)
        dfs.append(data)  # DataFrame der Liste hinzufügen
    except Exception as e:
        print(f"Fehler beim Laden der Datei {file}: {e}")

# 3. Kombinieren aller DataFrames zu einem einzigen DataFrame
combined_data = pd.concat(dfs, ignore_index=True)

# Vorverarbeitung
X = combined_data.iloc[:, :-1].values  # Merkmale
y = combined_data.iloc[:, -1].values   # Labels

# Skalierung der Merkmale auf den Bereich [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Umwandlung der Daten in Zeitreihensequenzen
time_step = 10
def create_dataset(X, y, time_step=1):
    Xs, ys = [], []
    for i in range(len(X)-time_step):
        Xs.append(X[i:i+time_step])
        ys.append(y[i+time_step])  # Das Label nach den Zeit-Schritten
    return np.array(Xs), np.array(ys)

X_lstm, y_lstm = create_dataset(X_scaled, y, time_step)

# SMOTE anwenden
smote = SMOTE(random_state=42)
X_lstm_flat, y_lstm_flat = smote.fit_resample(X_lstm.reshape(X_lstm.shape[0], -1), y_lstm)

# Zurückformung der Daten
X_lstm_resampled = X_lstm_flat.reshape(X_lstm_flat.shape[0], time_step, X_lstm.shape[2])
y_lstm_resampled = y_lstm_flat

# 5. KFold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

# Fortschrittsanzeige für Folds
for fold, (train_index, val_index) in enumerate(kf.split(X_lstm_resampled), 1):
    X_train_fold, X_val_fold = X_lstm_resampled[train_index], X_lstm_resampled[val_index]
    y_train_fold, y_val_fold = y_lstm_resampled[train_index], y_lstm_resampled[val_index]

    # Modell erstellen
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train_fold.shape[1], X_train_fold.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])

    # Trainieren des Modells mit den Daten des aktuellen Folds, inkl. Testdaten
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=64, validation_data=(X_val_fold, y_val_fold), verbose=0)

    # Vorhersagen auf den aktuellen Validierungsdaten des aktuellen Folds
    final_predictions = model.predict(X_val_fold)
    final_predictions = np.round(final_predictions)  # Rundung auf -1 oder 1

    # Berechnung der Metriken für den aktuellen Fold
    final_classification_rep = classification_report(y_val_fold, final_predictions, output_dict=True, zero_division=1)
    final_roc_auc = roc_auc_score(y_val_fold, final_predictions)
    final_pr_auc = average_precision_score(y_val_fold, final_predictions)

    # Berechnung der Accuracy für den aktuellen Fold
    accuracy = accuracy_score(y_val_fold, final_predictions)

    # Speichern der Ergebnisse für den aktuellen Fold
    results.append({
        "Fold": fold,
        "Accuracy": accuracy,
        "Precision": final_classification_rep['1.0']['precision'],
        "Recall": final_classification_rep['1.0']['recall'],
        "F1-Score": final_classification_rep['1.0']['f1-score'],
        "ROC-AUC": final_roc_auc,
        "PR-AUC": final_pr_auc
    })

# Ergebnisse der Cross-Validation anzeigen
results_df = pd.DataFrame(results)
print("\nFOLD | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC")
for index, row in results_df.iterrows():
    print(f"{row['Fold']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['ROC-AUC']:.4f} | {row['PR-AUC']:.4f}")

# Durchschnittliche Metriken berechnen
print("\nDurchschnittliche Metriken über alle 5 Folds:")
print(f"Durchschnittliche Precision: {results_df['Precision'].mean():.4f}")
print(f"Durchschnittliche Accuracy: {results_df['Accuracy'].mean():.4f}")
print(f"Durchschnittlicher Recall: {results_df['Recall'].mean():.4f}")
print(f"Durchschnittlicher F1-Score: {results_df['F1-Score'].mean():.4f}")
print(f"Durchschnittlicher ROC-AUC: {results_df['ROC-AUC'].mean():.4f}")
print(f"Durchschnittlicher PR-AUC: {results_df['PR-AUC'].mean():.4f}")
