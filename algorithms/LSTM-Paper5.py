import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

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

# 5. Oversampling der Klasse -1.0 mit SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_lstm_flat, y_lstm_flat = smote.fit_resample(X_lstm.reshape(X_lstm.shape[0], -1), y_lstm)

# 6. Zurückformung in die ursprüngliche Form (n_samples, time_steps, n_features)
# Zuerst müssen wir sicherstellen, dass die Anzahl der Features (die flache Version) korrekt zurückgesetzt wird
X_lstm_resampled = X_lstm_flat.reshape(X_lstm_flat.shape[0], time_step, X_lstm.shape[2])

# y_lstm_resampled ist die resampelte Zielvariable, die wir hier definieren müssen
y_lstm_resampled = y_lstm_flat

# 7. Berechnung der Klassen-Gewichte
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_lstm_resampled), y=y_lstm_resampled)
class_weights = dict(enumerate(class_weights))

# 8. LSTM-Modell erstellen
model = Sequential()

# LSTM-Schicht
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_lstm_resampled.shape[1], X_lstm_resampled.shape[2])))

# Dropout-Schicht zur Vermeidung von Overfitting
model.add(Dropout(0.2))

# Dense-Schicht zur Klassifikation
model.add(Dense(1, activation='tanh'))  # 'tanh' für Labels von -1 und 1

# Kompilieren des Modells
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])

# Modellzusammenfassung anzeigen
model.summary()

# 9. Modell trainieren mit den berechneten Klassen-Gewichten
history = model.fit(X_lstm_resampled, y_lstm_resampled, epochs=20, batch_size=64, validation_split=0.2, class_weight=class_weights, verbose=1)

# 10. Visualisierung der Trainingsgenauigkeit und Validierungsgenauigkeit
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 11. Vorhersagen und Klassifikation von Anomalien
predictions = model.predict(X_lstm_resampled)

# Umwandlung der Vorhersagen auf Werte zwischen -1 und 1
predictions = np.round(predictions)  # Rundung auf -1 oder 1

# Berechnung der Genauigkeit des Modells
accuracy = accuracy_score(y_lstm_resampled, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# 12. Berechnung von Precision, Recall und F1-Score
print("Classification Report:")
print(classification_report(y_lstm_resampled, predictions))  # Gibt Precision, Recall und F1-Score aus

# 13. Optionale Anomalieerkennung: Falsch positive und falsch negative Proben anzeigen
false_positives = np.where((predictions == 1) & (y_lstm_resampled == -1))[0]
false_negatives = np.where((predictions == -1) & (y_lstm_resampled == 1))[0]

print(f"Falsch positive Proben: {false_positives[:10]}")
print(f"Falsch negative Proben: {false_negatives[:10]}")
