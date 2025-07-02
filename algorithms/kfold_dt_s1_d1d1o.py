import pandas as pd
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    average_precision_score, precision_recall_curve
)
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Konfig
supervised_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
apps_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/apps-sok-reduced.txt'

# Apps laden
with open(apps_path, 'r') as f:
    apps = [line.strip() for line in f.readlines()]

list_of_folds = [1, 2, 3, 4, 5]
metrics_per_fold = []
fprs, tprs = [], []

for fold in list_of_folds:
    print(f"\n🔁 Fold {fold} – Train **und** Test auf denselben Daten")

    df_fold = pd.DataFrame()

    for app in apps:
        path = f"{supervised_path}{app}-{fold}.pkl"
        print("Lade:", path)
        with open(path, 'rb') as f:
            df = pickle.load(f)
            if df.isnull().any().any():
                print(f"NaNs in {path}")
                exit()
            df_fold = pd.concat([df_fold, df], axis=0)

    # Features & Labels
    X = df_fold.iloc[:, :-1]
    y = df_fold.iloc[:, -1:]

    # Skalierung
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    # Decision Tree Modell
    clf = DecisionTreeClassifier(max_features=200)
    clf.fit(X_scaled, y.values.ravel())

    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)
    class_index = list(clf.classes_).index(-1)
    y_scores = y_prob[:, class_index]

    # Metriken
    precision = precision_score(y, y_pred, pos_label=-1)
    recall = recall_score(y, y_pred, pos_label=-1)
    f1 = f1_score(y, y_pred, pos_label=-1)
    accuracy = accuracy_score(y, y_pred)
    auc_score = auc(*roc_curve((y.values.ravel() == -1).astype(int), y_scores)[:2])
    pr_auc = average_precision_score((y.values.ravel() == -1).astype(int), y_scores)

    fpr, tpr, _ = roc_curve((y.values.ravel() == -1).astype(int), y_scores)
    fprs.append(fpr)
    tprs.append(tpr)

    # Ausgabe pro Fold
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision (-1) : {precision:.4f}")
    print(f"Recall    (-1) : {recall:.4f}")
    print(f"F1 Score  (-1) : {f1:.4f}")
    print(f"ROC-AUC   (-1) : {auc_score:.4f}")
    print(f"PR-AUC    (-1) : {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

    # Speichern
    metrics_per_fold.append({
        'Fold': fold,
        'Accuracy': accuracy,
        'Precision_-1': precision,
        'Recall_-1': recall,
        'F1_-1': f1,
        'ROC-AUC_-1': auc_score,
        'PR-AUC_-1': pr_auc
    })

# ROC-Kurvenplot
plt.figure(figsize=(8, 8))
for i in range(len(fprs)):
    ysmoothed = gaussian_filter1d(tprs[i], sigma=2)
    plt.plot(fprs[i], ysmoothed, label=f'Fold {i+1} (AUC = {auc(fprs[i], tprs[i]):.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Train = Test pro Fold, Decision Tree)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Zusammenfassung
metrics_df = pd.DataFrame(metrics_per_fold)
print("\n📋 Zusammenfassung der Metriken pro Fold:")
print(metrics_df)
print("\nØ Durchschnitt pro Metrik:")
print(metrics_df.mean(numeric_only=True))
