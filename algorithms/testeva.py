# Evaluation-Skript für Server: Modell laden und auswerten

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model

# === 1. Dateien einlesen und zusammenführen ===
def load_and_merge_pkl_files(folder_path):
    all_files = glob.glob(os.path.join(folder_path, '*.pkl'))
    print(f"Gefundene Dateien: {len(all_files)}")

    if len(all_files) == 0:
        print("❌ Fehler: Keine .pkl-Dateien gefunden!")
        exit(1)

    dataframes = []
    for idx, filename in enumerate(all_files, 1):
        print(f"Lade Datei {idx}/{len(all_files)}: {os.path.basename(filename)}")
        df = pd.read_pickle(filename)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    print("✅ Alle Dateien erfolgreich geladen und zusammengeführt.")
    return merged_df

# === 2. Daten vorbereiten ===
def preprocess_data(df):
    print("Starte Datenvorverarbeitung...")
    X = df.iloc[:, :-1]  # Alle Spalten außer der letzten
    y = df.iloc[:, -1]   # Letzte Spalte als Label
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.expand_dims(X_scaled, axis=2)
    print("✅ Datenvorverarbeitung abgeschlossen.")
    return X_scaled, y

# === 3. Evaluation starten ===
def evaluate_model(model_path, data_folder):
    print("Lade Daten und Modell...")
    df = load_and_merge_pkl_files(data_folder)
    X, y = preprocess_data(df)

    # Labels fixen
    y = (y > 0).astype(int)

    # Daten splitten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell laden
    model = load_model(model_path)
    print("✅ Modell erfolgreich geladen.")

    # Vorhersage machen
    print("Starte Vorhersage auf Testdaten...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Metriken berechnen
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 Precision: {precision:.4f}")
    print(f"📊 Recall:    {recall:.4f}")
    print(f"📊 F1-Score:  {f1:.4f}")
    print("\n🧮 Confusion Matrix:")
    print(cm)

# === 4. Main Funktion ===
def main():
    model_path = 'cnn_model_cve.h5'  # <-- Pfad zu deinem gespeicherten Modell
    data_folder = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp'  # <-- Pfad zu deinen .pkl-Dateien

    evaluate_model(model_path, data_folder)

if __name__ == "__main__":
    main()
