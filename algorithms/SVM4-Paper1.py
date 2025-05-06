import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# === 1. Alle .pkl-Dateien laden ===

def load_all_pkl(folder_path):
    all_dfs = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    for file in tqdm(files, desc="📦 Lade .pkl-Dateien"):
        file_path = os.path.join(folder_path, file)
        with open(file_path, "rb") as f:
            df = pickle.load(f)
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

# === 2. Datenpfad ===

data_path = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp"
df = load_all_pkl(data_path)

# Optional: Sample für Geschwindigkeit
df = df.sample(n=5000, random_state=42)

# === 3. Feature & Label split ===

X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]
y.replace(0, -1, inplace=True)

# === 4. Normalisieren ===

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 5. Train/Test-Split ===

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"📊 Trainingsdaten: {X_train.shape[0]} Zeilen, {X_train.shape[1]} Merkmale")

# === 6. Grid Search ===

param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.001, 0.01, 0.1]
}
grid = ParameterGrid(param_grid)

results = []
best_model = None
best_score = -1

print("\n🔍 Starte Grid Search...")
for params in tqdm(grid, desc="🔧 Parameter testen"):
    model = SVC(kernel='rbf', class_weight={1: 1, -1: 100}, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    recall_anomaly = report['-1']['recall']
    
    results.append({
        'C': params['C'],
        'gamma': params['gamma'],
        'anomaly_precision': report['-1']['precision'],
        'anomaly_recall': recall_anomaly,
        'anomaly_f1': report['-1']['f1-score'],
        'accuracy': report['accuracy']
    })
    
    if recall_anomaly > best_score:
        best_score = recall_anomaly
        best_model = model
        best_params = params

# === 7. Bestes Modell ausgeben ===

print(f"\n✅ Bestes Modell (nach Anomaly Recall): C={best_params['C']}, gamma={best_params['gamma']}")

# === 8. Evaluation des besten Modells ===

y_pred_best = best_model.predict(X_test)
print("\n📘 Evaluation bestes Modell:")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# === 9. Ergebnisse anzeigen (DataFrame) ===

results_df = pd.DataFrame(results).sort_values(by=["anomaly_recall", "anomaly_f1"], ascending=False)
import ace_tools as tools; tools.display_dataframe_to_user(name="Grid Search Ergebnisse", dataframe=results_df)

# === 10. Optional: Modell speichern ===
# with open("beste_weighted_svm.pkl", "wb") as f:
#     pickle.dump(best_model, f)
