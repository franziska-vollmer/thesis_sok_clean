import os
import pickle
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# === Lokale Dateipfade anpassen ===
pkl_folder = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
csv_path = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/CVE-2022-22965/CVE-2022-22965-3_freqvector_full.csv"

# === Feature-Namen aus CSV laden ===
csv_data = pd.read_csv(csv_path)
feature_names = csv_data.columns[1:556]  # timestamp ist Spalte 0

# === CVE-Gruppen erkennen ===
all_files = os.listdir(pkl_folder)
cve_groups = defaultdict(list)

for fname in all_files:
    if fname.endswith(".pkl"):
        match = re.match(r"(CVE-\d{4}-\d+)", fname)
        if match:
            cve_key = match.group(1)
            cve_groups[cve_key].append(os.path.join(pkl_folder, fname))

# Nur vollständige Gruppen mit mindestens 4 Dateien
cve_groups = {k: v for k, v in cve_groups.items() if len(v) >= 4}

print(f"Verarbeite {len(cve_groups)} vollständige CVEs...")

# === Feature Importances pro CVE berechnen ===
all_cve_importances = []

for cve, file_paths in cve_groups.items():
    importances = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            try:
                df = pickle.load(f)
                if df.shape[1] < 557:
                    continue
                X = df.iloc[:, 1:556]
                y = df.iloc[:, 556]
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)
                importances.append(model.feature_importances_)
            except Exception as e:
                print(f"[WARN] Fehler in {file_path}: {e}")
    
    if importances:
        mean_importance = np.mean(importances, axis=0)
        all_cve_importances.append(mean_importance)

# === Globalen Mittelwert berechnen ===
global_importance = np.mean(all_cve_importances, axis=0)

# === Top-N Features anzeigen ===
top_n = 20
top_indices = np.argsort(global_importance)[::-1][:top_n]
top_features = feature_names[top_indices]
top_values = global_importance[top_indices]

# === Plot erstellen ===
plt.figure(figsize=(10, 6))
plt.barh(top_features, top_values)
plt.xlabel("Feature Importance")
plt.title(f"Top {top_n} Most Important Features Across All CVEs")
plt.gca().invert_yaxis()
plt.tight_layout()

# === Plot als Bild speichern ===
plt.savefig("top_features_global_hd.png", dpi=300, bbox_inches="tight")

plt.show()

# === Optional: CSV-Export ===
# pd.DataFrame({
#     "Feature": top_features,
#     "Importance": top_values
# }).to_csv("top_features_global.csv", index=False)
