import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm  # Fortschrittsanzeige

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

# Anzeigen der ersten Zeilen des kombinierten DataFrames
print(combined_data.head())

# 4. Vorverarbeitung der Daten:
# Die letzte Spalte ist das Label, alle anderen sind die Merkmale
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

# 5. SMOTE anwenden
smote = SMOTE(random_state=42)
X_lstm_flat, y_lstm_flat = smote.fit_resample(X_lstm.reshape(X_lstm.shape[0], -1), y_lstm)

# Zurückformung der Daten
X_lstm_resampled = X_lstm_flat.reshape(X_lstm_flat.shape[0], time_step, X_lstm.shape[2])
y_lstm_resampled = y_lstm_flat

# 6. Berechnung der Klassen-Gewichte
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_lstm_resampled), y=y_lstm_resampled)
class_weights = dict(enumerate(class_weights))

# 7. KFold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Liste für Ergebnisse
results = []

# Fortschrittsanzeige für Folds
for fold, (train_index, val_index) in enumerate(kf.split(X_lstm_resampled), 1):
    X_train, X_val = X_lstm_resampled[train_index], X_lstm_resampled[val_index]
    y_train, y_val = y_lstm_resampled[train_index], y_lstm_resampled[val_index]

    # Modell erstellen
    model = Sequential()
    model.add(GRU(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])

    # Fortschrittsanzeige während des Trainings
    print(f"Training Fold {fold}...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val),
                        class_weight=class_weights, verbose=0)  # verbose=0 für Fortschrittsanzeige

    # Vorhersagen und Metriken berechnen
    predictions = model.predict(X_val)
    predictions = np.round(predictions)  # Rundung auf -1 oder 1

    # Berechnung der verschiedenen Metriken
    accuracy = accuracy_score(y_val, predictions)
    classification_rep = classification_report(y_val, predictions, output_dict=True)

    # ROC-AUC und PR-AUC
    roc_auc = roc_auc_score(y_val, predictions)
    pr_auc = average_precision_score(y_val, predictions)

    # Speichern der Ergebnisse für den aktuellen Fold
    results.append({
        "Fold": fold,
        "Precision": classification_rep['1.0']['precision'],
        "Recall": classification_rep['1.0']['recall'],
        "F1-Score": classification_rep['1.0']['f1-score'],
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

# Ergebnisse in DataFrame umwandeln
results_df = pd.DataFrame(results)

# Ausgabe der Ergebnisse im Terminal
print("\nFOLD | Precision | Recall | F1-Score | ROC-AUC | PR-AUC")
for index, row in results_df.iterrows():
    print(f"{row['Fold']} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['ROC-AUC']:.4f} | {row['PR-AUC']:.4f}")

# Durchschnittliche Metriken berechnen
print("\nDurchschnittliche Metriken über alle 5 Folds:")
print(f"Durchschnittliche Precision: {results_df['Precision'].mean():.4f}")
print(f"Durchschnittlicher Recall: {results_df['Recall'].mean():.4f}")
print(f"Durchschnittlicher F1-Score: {results_df['F1-Score'].mean():.4f}")
print(f"Durchschnittlicher ROC-AUC: {results_df['ROC-AUC'].mean():.4f}")
print(f"Durchschnittlicher PR-AUC: {results_df['PR-AUC'].mean():.4f}")
