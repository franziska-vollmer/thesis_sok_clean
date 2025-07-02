import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, auc,
    average_precision_score
)
from scipy.ndimage import gaussian_filter1d

# === DATEIPFADE ANPASSEN ===
supervised_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
apps_file_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/apps-sok-reduced.txt'

# === DATEN EINLESEN ===
with open(apps_file_path, 'r') as file:
    app_lines = [line.strip() for line in file.readlines()]

list_of_folds = [1, 2, 3, 4, 5]

# === AUSWERTUNGSSPEICHER ===
results = []
fprs, tprs = [], []

for fold in list_of_folds:
    print(f"\n🔁 Fold {fold} – Train & Test auf denselben Daten")

    df_fold = pd.DataFrame()

    for app in app_lines:
        path = os.path.join(supervised_path, f"{app}-{fold}.pkl")
        print("Lade:", path)
        with open(path, 'rb') as f:
            df = pickle.load(f)
            if df.isnull().values.any():
                raise ValueError(f"NaNs in Datei {path}")
            df_fold = pd.concat([df_fold, df], axis=0)

    # Features & Labels
    X = df_fold.iloc[:, :-1]
    y = df_fold.iloc[:, -1]

    # Skalierung
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modell
    abc = AdaBoostClassifier(n_estimators=200)
    abc.fit(X_scaled, y)

    y_pred = abc.predict(X_scaled)
    y_proba = abc.predict_proba(X_scaled)
    class_index = list(abc.classes_).index(-1)
    y_scores = y_proba[:, class_index]

    # ROC + AUC
    fpr, tpr, _ = roc_curve((y == -1).astype(int), y_scores)
    fprs.append(fpr)
    tprs.append(tpr)
    auc_score = auc(fpr, tpr)

    # Metriken
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, pos_label=-1, zero_division=0)
    rec = recall_score(y, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y, y_pred, pos_label=-1, zero_division=0)
    pr_auc = average_precision_score((y == -1).astype(int), y_scores)

    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred, labels=[-1, 0])}")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision (-1) : {prec:.4f}")
    print(f"Recall    (-1) : {rec:.4f}")
    print(f"F1 Score  (-1) : {f1:.4f}")
    print(f"ROC-AUC   (-1) : {auc_score:.4f}")
    print(f"PR-AUC    (-1) : {pr_auc:.4f}")

    results.append({
        'Fold': fold,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc_score,
        'PR-AUC': pr_auc
    })

# === AUSWERTUNG ===
results_df = pd.DataFrame(results)
print("\n📋 Ergebnisse pro Fold:")
print(results_df)

print("\n📈 Durchschnitt:")
print(results_df.mean(numeric_only=True))

# === ROC-KURVEN PLOT ===
plt.figure(figsize=(10, 8))
for i in range(len(fprs)):
    roc_auc = auc(fprs[i], tprs[i])
    smoothed = gaussian_filter1d(tprs[i], sigma=2)
    plt.plot(fprs[i], smoothed, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Train = Test, AdaBoost)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
