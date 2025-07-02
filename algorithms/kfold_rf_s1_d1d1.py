import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    average_precision_score
)
from sklearn.model_selection import StratifiedKFold
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d

matplotlib.rcParams.update({'font.size': 20})

supervised_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
file1 = open('/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/apps-sok-reduced.txt', 'r')
Lines = file1.readlines()
file2 = open('/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/apps-sok-reduced.txt', 'r')
Lines2 = file2.readlines()

list_of_train = [1, 2, 3, 4]
list_of_test = [1, 2, 3, 4]
df_train_all = pd.DataFrame()
df_test_all = pd.DataFrame()


def compute_roc_auc(index):
    y_predict = rfc.predict_proba(X.iloc[index])[:, list(rfc.classes_).index(-1)]
    fpr, tpr, thresholds = roc_curve((y.iloc[index] == -1).astype(int), y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


# Daten laden
for line in Lines:
    content = line.strip()
    for k in list_of_train:
        path = supervised_path + content + '-' + str(k) + '.pkl'
        print(path)
        with open(path, 'rb') as picklefile_train:
            df_individual_train = pickle.load(picklefile_train)
            if df_individual_train.isnull().any().any():
                print(f"NaNs gefunden in: {path}")
                exit()
            df_train_all = pd.concat([df_train_all, df_individual_train], axis=0)

for line in Lines2:
    content = line.strip()
    for k in list_of_test:
        path = supervised_path + content + '-' + str(k) + '.pkl'
        print(path)
        with open(path, 'rb') as picklefile_test:
            df_individual_test = pickle.load(picklefile_test)
            if df_individual_test.isnull().any().any():
                print(f"NaNs gefunden in: {path}")
                exit()
            df_test_all = pd.concat([df_test_all, df_individual_test], axis=0)

print(df_train_all.shape)
print(df_test_all.shape)

# Merkmale und Labels
train_data_x = df_train_all.iloc[:, :-1]
train_data_y = df_train_all.iloc[:, -1:]

test_data_x = df_test_all.iloc[:, :-1]
test_data_y = df_test_all.iloc[:, -1:]

# Skalieren
sc = StandardScaler()
scaled_train_data_x = sc.fit_transform(train_data_x)
scaled_test_data_x = sc.transform(test_data_x)

# Zusammenführen für CV
X = np.concatenate((scaled_train_data_x, scaled_test_data_x), axis=0)
y = np.concatenate((train_data_y.values.ravel(), test_data_y.values.ravel()), axis=0)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

print(X.shape)
print(y.shape)

# Modell und Cross-Validation
rfc = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1)
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

fold_metrics = []
fprs, tprs, scores = [], [], []

for (train, test), i in zip(cv.split(X, y), range(1, 6)):
    rfc.fit(X.iloc[train], y.iloc[train].values.ravel())
    _, _, auc_train = compute_roc_auc(train)
    fpr, tpr, auc_test = compute_roc_auc(test)
    scores.append((auc_train, auc_test))
    fprs.append(fpr)
    tprs.append(tpr)

    X_test_fold = X.iloc[test]
    y_test_fold = y.iloc[test].values.ravel()
    y_pred = rfc.predict(X_test_fold)

    # Wahrscheinlichkeiten für Klasse -1
    y_scores = rfc.predict_proba(X_test_fold)[:, list(rfc.classes_).index(-1)]

    # PR-AUC (positiv: Klasse -1)
    pr_auc = average_precision_score((y_test_fold == -1).astype(int), y_scores)

    # Metriken für Klasse -1
    acc = accuracy_score(y_test_fold, y_pred)
    prec = precision_score(y_test_fold, y_pred, pos_label=-1)
    rec = recall_score(y_test_fold, y_pred, pos_label=-1)
    f1 = f1_score(y_test_fold, y_pred, pos_label=-1)

    print(f"\n=== Fold {i} ===")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision (-1) : {prec:.4f}")
    print(f"Recall (-1)    : {rec:.4f}")
    print(f"F1-Score (-1)  : {f1:.4f}")
    print(f"ROC-AUC (-1)   : {auc_test:.4f}")
    print(f"PR-AUC (-1)    : {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_fold, y_pred))

    fold_metrics.append({
        'Fold': i,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc_test,
        'PR-AUC': pr_auc
    })

# Ergebnisse als DataFrame
results_df = pd.DataFrame(fold_metrics)

# Tabellarische Ausgabe
print("\n📋 Ergebnisse pro Fold (für Klasse -1):")
print(f"{'Fold':<5} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'ROC-AUC':<8} {'PR-AUC'}")
for idx, row in results_df.iterrows():
    print(f"{int(row['Fold']):<5} {row['Accuracy']:<9.4f} {row['Precision']:<10.4f} {row['Recall']:<8.4f} {row['F1-Score']:<9.4f} {row['ROC-AUC']:<8.4f} {row['PR-AUC']:.4f}")

# Durchschnitt
means = results_df.mean(numeric_only=True)
print(f"{'Ø':<5} {means['Accuracy']:<9.4f} {means['Precision']:<10.4f} {means['Recall']:<8.4f} {means['F1-Score']:<9.4f} {means['ROC-AUC']:<8.4f} {means['PR-AUC']:.4f}")
