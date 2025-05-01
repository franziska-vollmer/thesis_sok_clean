import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
import os

# Tabelle zur Speicherung der Ergebnisse
results_table = []

# Schritt 1: Laden der .pkl-Dateien
data_folder = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
data_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]

# Liste, um alle Daten zu speichern
data_list = []

# Laden der Daten aus den .pkl-Dateien
for file in tqdm(data_files, desc="Laden der Dateien"):
    df = pd.read_pickle(os.path.join(data_folder, file))
    data_list.append(df)

# Kombinieren aller DataFrames in eine einzige DataFrame
data = pd.concat(data_list, ignore_index=True)

# Schritt 2: Vorverarbeitung der Daten
X = data.iloc[:, :-1].values  # Alle Spalten außer der letzten (Feature-Daten)
y = data.iloc[:, -1].values   # Die letzte Spalte ist das Label (Anomalie oder nicht)

# Schritt 3: Normalisierung der Feature-Daten
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Schritt 4: Umwandlung der Eingabedaten für LSTM (3D-Format: [samples, timesteps, features])
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # 1 timestep

# Schritt 5: Umwandlung der Labels in numerische Werte
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Schritt 6: Aufteilen der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Schritt 7: LSTM Modell erstellen
model = Sequential()

# Erste LSTM-Schicht mit Dropout zur Regularisierung
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Zweite LSTM-Schicht mit Dropout zur Regularisierung
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Dense-Schicht für die Klassifikation (Sigmoid-Aktivierung für binäre Klassifikation)
model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid für binäre Klassifikation

# Schritt 8: Modell kompilieren
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Schritt 9: Training und Metriken berechnen
epochs = 10
batch_size = 64

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    # Training des Modells für eine Epoche
    history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    
    # Vorhersagen auf den Testdaten
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Berechnung von Precision, Recall, F1-Score, AUC und PR-AUC
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_value = roc_auc_score(y_test, y_pred)
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, model.predict(X_test))
    pr_auc = auc(recall_curve, precision_curve)

    # Ergebnisse für diese Epoche speichern
    results_table.append({
        'Fold': epoch + 1,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'ROC-AUC': auc_value,
        'PR-AUC': pr_auc
    })
    
    # Ausgabe der Metriken
    print(f"Training Accuracy: {history.history['accuracy'][0]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][0]:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {auc_value:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
# Schritt 10: Modellbewertung
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Schritt 11: Ergebnisse in einer Tabelle anzeigen
results_df = pd.DataFrame(results_table)

# Ausgabe der Tabelle im Terminal
print("\nFold Results:")
print(results_df)

# Schritt 12: Modell speichern
model.save('nids_lstm_model.keras')  # Speichern im Keras-native Format
