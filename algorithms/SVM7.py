import os
import time
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# === 1. Alle pkl-Dateien laden ===
def load_all_pkl_files(folder_path):
    print("[INFO] Lade .pkl-Dateien aus:", folder_path)
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            print(f"  ↪ Lese Datei: {filename}")
            df = pd.read_pickle(file_path)
            all_data.append(df)
    print(f"[INFO] {len(all_data)} Dateien geladen.")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"[INFO] Gesamtdatenform: {combined_df.shape}")
    return combined_df

# === 2. One-Class SVM Training + Bewertung ===
def train_and_evaluate_ocsvm(data, label_col=-1, nu=0.05, subsample_size=10000, kernel='rbf'):
    print("[INFO] Trenne Features und Label ...")
    X = data.drop(columns=[data.columns[label_col]])
    y = data.iloc[:, label_col]

    print(f"[INFO] Anzahl normaler Daten (Label=1): {(y==1).sum()}")
    print(f"[INFO] Anzahl Anomalien (Label=-1): {(y==-1).sum()}")

    print(f"[INFO] Nehme zufällige {subsample_size} normale Daten für das Training ...")
    X_train = X[y == 1].sample(n=subsample_size, random_state=42)

    # Skalieren
    print("[INFO] Skaliere Daten ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X)

    print(f"[INFO] Trainiere One-Class SVM mit kernel='{kernel}', nu={nu} ...")
    start_time = time.time()
    oc_svm = OneClassSVM(kernel=kernel, gamma='auto', nu=nu)
    oc_svm.fit(X_train_scaled)
    end_time = time.time()
    print(f"[INFO] Modelltraining abgeschlossen in {end_time - start_time:.2f} Sekunden.")

    print("[INFO] Wende Modell auf alle Daten an ...")
    y_pred = oc_svm.predict(X_all_scaled)

    print("[INFO] Klassifikationsbericht:")
    print(classification_report(y, y_pred, target_names=["Anomalie", "Normal"]))

    # Optionale Speicherung der Anomalien
    anomalies = data[y_pred == -1]
    anomalies.to_csv("erkannte_anomalien.csv", index=False)
    print(f"[INFO] Erkannte Anomalien gespeichert in 'erkannte_anomalien.csv' ({len(anomalies)} Zeilen)")

# === Hauptprogramm ===
if __name__ == "__main__":
    ordnerpfad = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
    print("[START] Starte Analyse mit One-Class SVM (optimiert)")
    daten = load_all_pkl_files(ordnerpfad)
    train_and_evaluate_ocsvm(daten, label_col=-1, nu=0.05, subsample_size=10000, kernel='rbf')
    print("[ENDE] Analyse abgeschlossen.")
