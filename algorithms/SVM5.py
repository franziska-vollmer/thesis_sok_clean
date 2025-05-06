import os
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

# ----------------------------------------
# PARAMETER
# ----------------------------------------
ordnerpfad = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
anzahl_pca = 2
bandbreite_kde = 0.2
min_anomalies = 100
test_size = 0.3

# ----------------------------------------
# 1. Lade .pkl-Dateien
# ----------------------------------------
print("📥 Schritt 1: Lade .pkl-Dateien...")
alle_dfs = []
pkl_dateien = [f for f in os.listdir(ordnerpfad) if f.endswith(".pkl")]

for i, datei in enumerate(pkl_dateien, start=1):
    try:
        with open(os.path.join(ordnerpfad, datei), "rb") as f:
            df = pickle.load(f)
            if isinstance(df, pd.DataFrame):
                alle_dfs.append(df)
        print(f"[{i}/{len(pkl_dateien)}] ✅ '{datei}' geladen.")
    except Exception as e:
        print(f"[{i}/{len(pkl_dateien)}] ❌ Fehler bei '{datei}': {e}")

# ----------------------------------------
# 2. Kombinieren & trennen
# ----------------------------------------
data = pd.concat(alle_dfs, ignore_index=True)
X_full = data.iloc[:, :-1].values
y_full = data.iloc[:, -1].values

X_pos = X_full[y_full == 1]
X_test_eval = X_full  # für spätere Bewertung
y_test_eval = np.where(y_full == 1, 1, 0)  # 1 = gesund, 0 = anomal

print(f"\n✅ {len(X_pos)} gesunde Trainingsdaten gefunden.")
print(f"🧪 Evaluation wird auf {len(X_test_eval)} echten Datenpunkten mit {np.sum(y_test_eval==0)} Anomalien durchgeführt.")

# ----------------------------------------
# 3. PCA auf positive Klasse
# ----------------------------------------
print("\n🔍 Schritt 3: Führe PCA auf gesunde Daten durch...")
pca = PCA(n_components=anzahl_pca)
X_pos_pca = pca.fit_transform(X_pos)
print("✅ PCA abgeschlossen.")

# ----------------------------------------
# 4. KDE & automatische Anomalie-Erzeugung
# ----------------------------------------
print("\n📊 Schritt 4: KDE & adaptive Erzeugung synthetischer Anomalien...")
kde = KernelDensity(kernel='gaussian', bandwidth=bandbreite_kde)
kde.fit(X_pos_pca)

x_min, x_max = X_pos_pca[:, 0].min()-2, X_pos_pca[:, 0].max()+2
y_min, y_max = X_pos_pca[:, 1].min()-2, X_pos_pca[:, 1].max()+2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
log_dens = kde.score_samples(grid_points)
density = np.exp(log_dens)

# Schwelle automatisch anpassen
current_percentile = 5
max_percentile = 50
X_neg = np.empty((0, 2))

while current_percentile <= max_percentile:
    threshold = np.percentile(density, current_percentile)
    X_neg = grid_points[density < threshold]
    if len(X_neg) >= min_anomalies:
        break
    print(f"ℹ️ Nur {len(X_neg)} Punkte bei {current_percentile}%. Erhöhe...")
    current_percentile += 5

if len(X_neg) == 0:
    raise ValueError("❌ Keine negativen Punkte erzeugt. KDE/PCA anpassen.")

print(f"✅ {len(X_neg)} synthetische Anomaliepunkte erzeugt bei {current_percentile}%.")

# ----------------------------------------
# 5. Trainingsdaten vorbereiten
# ----------------------------------------
print("\n🧪 Schritt 5: Trainingsdaten vorbereiten...")
X_all = np.vstack([X_pos_pca, X_neg])
y_all = np.hstack([np.ones(len(X_pos_pca)), np.zeros(len(X_neg))])

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42)

print(f"ℹ️ Trainingsverteilung: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print("✅ Trainingsdaten bereit.")

# ----------------------------------------
# 6. Trainiere SVM
# ----------------------------------------
print("\n🤖 Schritt 6: Trainiere SVM...")
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
print("✅ Training abgeschlossen.")

# ----------------------------------------
# 7. Anwenden auf echte Daten
# ----------------------------------------
print("\n🔎 Schritt 7: Posterior-Wahrscheinlichkeiten auf echten Daten berechnen...")
X_eval_pca = pca.transform(X_test_eval)
y_scores = svm.predict_proba(X_eval_pca)[:, 1]  # Wahrscheinlichkeit gesund

# ----------------------------------------
# 8. Auswertung mit echten Labels
# ----------------------------------------
print("\n📈 Schritt 8: Auswertung gegen echte Labels...")
y_pred_binary = (y_scores >= 0.5).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(y_test_eval, y_pred_binary, average='binary')
roc_auc = roc_auc_score(y_test_eval, y_scores)

print("\n🎯 Ergebnisse (EVAL auf echte Labels):")
print(f"  Precision : {precision:.3f}")
print(f"  Recall    : {recall:.3f}")
print(f"  F1-Score  : {f1:.3f}")
print(f"  ROC AUC   : {roc_auc:.3f}")

# ----------------------------------------
# 9. Speichern (optional)
# ----------------------------------------
print("\n💾 Schritt 9: Speichere Ergebnisse als CSV...")
ergebnisse = pd.DataFrame({
    "PCA_1": X_eval_pca[:, 0],
    "PCA_2": X_eval_pca[:, 1],
    "True_Label": y_test_eval,
    "Posterior_Prob_Healthy": y_scores
})
ergebnisse.to_csv("anomaly_detection_results.csv", index=False)
print("✅ Gespeichert: anomaly_detection_results.csv")

print("\n✅ Alle Schritte abgeschlossen – Algorithmus wie im Paper ausgeführt.")
