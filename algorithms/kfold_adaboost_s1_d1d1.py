import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, auc
)
from scipy.ndimage import gaussian_filter1d

# === DATEIPFADE ANPASSEN ===
BASE_DIR = os.getcwd()
supervised_path = os.path.join(BASE_DIR, '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/apps-sok-reduced.txt')  # <- ANPASSEN falls nötig

# === DATEN EINLESEN ===
with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

df_train_all = pd.DataFrame()
df_test_all = pd.DataFrame()
list_of_train = [1, 2, 3, 4]
list_of_test = [1, 2, 3, 4]

for line in app_lines:
    app = line.strip()
    for k in list_of_train:
        path = os.path.join(supervised_path, f"{app}-{k}.pkl")
        with open(path, 'rb') as f:
            df = pickle.load(f)
            if df.isnull().values.any():
                raise ValueError(f"NaN in Datei {path}")
            df_train_all = pd.concat([df_train_all, df], axis=0)

for line in app_lines:
    app = line.strip()
    for k in list_of_test:
        path = os.path.join(supervised_path, f"{app}-{k}.pkl")
        with open(path, 'rb') as f:
            df = pickle.load(f)
            if df.isnull().values.any():
                raise ValueError(f"NaN in Datei {path}")
            df_test_all = pd.concat([df_test_all, df], axis=0)

# === FEATURES & LABELS ===
train_x = df_train_all.iloc[:, :-1]
train_y = df_train_all.iloc[:, -1]
test_x = df_test_all.iloc[:, :-1]
test_y = df_test_all.iloc[:, -1]

# === SKALIERUNG ===
scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_x)
scaled_test_x = scaler.transform(test_x)

# === VEREINEN FÜR CV ===
X = np.concatenate((scaled_train_x, scaled_test_x), axis=0)
y = np.concatenate((train_y.values.ravel(), test_y.values.ravel()), axis=0)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

# === MODELL ===
abc = AdaBoostClassifier(n_estimators=200)
cv = StratifiedKFold(n_splits=4)

# === AUSWERTUNGSMETRIKEN ===
fprs, tprs = [], []
precisions, recalls, f1s, accuracies, aucs = [], [], [], [], []

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    abc.fit(X.iloc[train_idx], y.iloc[train_idx].values.ravel())

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx].values.ravel()
    y_pred = abc.predict(X_test)
    probas = abc.predict_proba(X_test)
    y_pred_proba = probas[:, list(abc.classes_).index(-1)]  # Wahrscheinlichkeit für Klasse -1

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=-1)
    fprs.append(fpr)
    tprs.append(tpr)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)
    auc_val = roc_auc_score(y_test, y_pred_proba)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    aucs.append(auc_val)

    print(f"\nFold {fold_idx + 1}")
    print(confusion_matrix(y_test, y_pred, labels=[-1, 0]))
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision (Anomalie = -1): {prec:.3f}")
    print(f"Recall (Anomalie = -1): {rec:.3f}")
    print(f"F1-Score (Anomalie = -1): {f1:.3f}")
    print(f"AUC (Anomalie = -1): {auc_val:.3f}")

# === DURCHSCHNITT ÜBER ALLE FOLDS ===
print("\n===== Durchschnittliche Ergebnisse über alle Folds =====")
print(f"Mean Accuracy:  {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
print(f"Mean Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
print(f"Mean Recall:    {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
print(f"Mean F1-Score:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
print(f"Mean AUC:       {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# === ROC CURVE PLOTTEN ===
plt.figure(figsize=(10, 8))
for i in range(len(fprs)):
    roc_auc = auc(fprs[i], tprs[i])
    y_smooth = gaussian_filter1d(tprs[i], sigma=2)
    plt.plot(fprs[i], y_smooth, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve für Anomalieklasse = -1')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()