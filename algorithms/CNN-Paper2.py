# Komplettes Python-Skript: CNN auf mehreren CVE-pkl-Dateien trainieren mit Fortschrittsanzeige, Label-Fix und Modell-Speicherung

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# === 1. Dateien einlesen und zusammenführen ===

def load_and_merge_pkl_files(folder_path):
    all_files = glob.glob(os.path.join(folder_path, '*.pkl'))
    print(f"Gefundene Dateien: {len(all_files)}")
    
    if len(all_files) == 0:
        print("❌ Fehler: Keine .pkl-Dateien gefunden. Überprüfe den Pfad oder Dateiendungen!")
        exit(1)

    dataframes = []
    for idx, filename in enumerate(all_files, 1):
        print(f"Lade Datei {idx}/{len(all_files)}: {os.path.basename(filename)}")
        df = pd.read_pickle(filename)
        dataframes.append(df)
        if idx % 10 == 0:
            print(f"✅ {idx} Dateien geladen...")

    print("✅ Alle Dateien erfolgreich geladen und zusammengeführt.")
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df

# === 2. Daten vorbereiten ===

def preprocess_data(df):
    print("Starte Datenvorverarbeitung...")
    X = df.iloc[:, :-1]  # Alle Spalten außer der letzten
    y = df.iloc[:, -1]   # Letzte Spalte als Label
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.expand_dims(X_scaled, axis=2)  # Form: (samples, features, 1)
    print("✅ Datenvorverarbeitung abgeschlossen.")
    return X_scaled, y

# === 3. CNN Modell erstellen ===

def create_cnn_model(input_shape):
    print("Erstelle CNN-Modell...")
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binäre Klassifikation
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ CNN-Modell erstellt und kompiliert.")
    return model

# === 4. Main Pipeline ===

def main():
    folder_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp'  # <-- Dein Ordner mit den .pkl-Dateien
    model_save_path = 'cnn_model_cve.h5'     # Modell speichern als H5-Datei

    print("Lade und verarbeite Daten...")
    df = load_and_merge_pkl_files(folder_path)
    print(f"Daten geladen: {df.shape}")

    X, y = preprocess_data(df)

    print("Prüfe Labels auf Korrektheit...")
    print("Label-Verteilung vor Umwandlung:", np.unique(y, return_counts=True))
    y = (y > 0).astype(int)
    print("Label-Verteilung nach Umwandlung:", np.unique(y, return_counts=True))

    print("Teile Daten in Trainings- und Testset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_cnn_model((X.shape[1], 1))
    model.summary()

    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

    print("Starte Training...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
        verbose=1
    )

    print("✅ Training abgeschlossen.")
    print(f"✅ Bestes Modell gespeichert unter: {model_save_path}")

    print("Evaluieren des Modells auf Testdaten...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Testgenauigkeit: {accuracy:.4f}")

if __name__ == "__main__":
    main()
