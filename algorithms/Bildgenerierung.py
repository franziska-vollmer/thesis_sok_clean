import os
import glob
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Basis-Pfad zu deinen CSV-Daten
base_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files'
# Zielordner für die Bilder
output_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/Bilder_Datensatz'

# Erzeuge Zielordner
normal_dir = os.path.join(output_path, 'Normal')
anomal_dir = os.path.join(output_path, 'Anomalie')
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(anomal_dir, exist_ok=True)

# Sampling-Rate und Schwellwert
sampling_rate = 0.1  # 0.1 Sekunden
threshold_seconds = 180  # 3 Minuten

# Lade alle CVE-Ordner
cve_folders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# Zähler für Statistik
count_normal = 0
count_anomalie = 0

# Fortschrittsanzeige für CVEs
for cve_folder in tqdm(cve_folders, desc="Verarbeite CVE-Ordner"):
    csv_files = glob.glob(os.path.join(cve_folder, '*_freqvector_full.csv'))
    
    for file in csv_files:
        df = pd.read_csv(file)

        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])

        data = df.values
        times = np.arange(len(data)) * sampling_rate
        labels = (times >= threshold_seconds).astype(int)

        if data.shape[1] < 576:
            pad_width = 576 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')

        data = data.reshape((-1, 24, 24))

        for idx, (sample, label) in enumerate(zip(data, labels)):
            sample_img = (sample * 255).astype(np.uint8)

            save_dir = normal_dir if label == 0 else anomal_dir
            save_path = os.path.join(save_dir, f"{os.path.basename(file).replace('.csv','')}_{idx}.png")

            cv2.imwrite(save_path, sample_img)

            # Statistik erhöhen
            if label == 0:
                count_normal += 1
            else:
                count_anomalie += 1

# Zusammenfassung am Ende
print(f"✅ Bildererstellung abgeschlossen.")
print(f"Gespeicherte Normal-Bilder: {count_normal}")
print(f"Gespeicherte Anomalie-Bilder: {count_anomalie}")
