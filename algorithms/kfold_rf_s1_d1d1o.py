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
    average_precision_score,
    precision_recall_curve
)
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.rcParams.update({'font.size': 20})

# Dateipfade
supervised_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
file = open('/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/apps-sok-reduced.txt', 'r')
apps = [line.strip() for line in file.readlines()]

list_of_folds = [1, 2, 3, 4, 5]

for fold in list_of_folds:
    print(f"\n🔁 Fold {fold} – Train **und** Test auf denselben Daten")

    df_data = pd.DataFrame()

    for app in apps:
        path = f"{supervised_path}{app}-{fold}.pkl"
        print("Lade:", path)
        with open(path, 'rb') as f:
            df = pickle.load(f)
            if df.isnull().any().any():
                print(f"NaNs in {path}")
                exit()
            df_data = pd.concat([df_data, df], axis=0)

    data_x = df_data.iloc[:, :-1]
    data_y = df_data.iloc[:, -1:]

    sc = StandardScaler()
    scaled_x = sc.fit_transform(data_x)

    rfc = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1)
    rfc.fit(scaled_x, data_y.values.ravel())

    y_pred = rfc.predict(scaled_x)
    y_scores = rfc.predict_proba(scaled_x)[:, list(rfc.classes_).index(-1)]

    fpr, tpr, _ = roc_curve((data_y.values.ravel() == -1).astype(int), y_scores)
    auc_score = auc(fpr, tpr)
    pr_auc = average_precision_score((data_y.values.ravel() == -1).astype(int), y_scores)

    acc = accuracy_score(data_y, y_pred)
    prec = precision_score(data_y, y_pred, pos_label=-1)
    rec = recall_score(data_y, y_pred, pos_label=-1)
    f1 = f1_score(data_y, y_pred, pos_label=-1)

    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision (-1) : {prec:.4f}")
    print(f"Recall (-1)    : {rec:.4f}")
    print(f"F1-Score (-1)  : {f1:.4f}")
    print(f"ROC-AUC (-1)   : {auc_score:.4f}")
    print(f"PR-AUC (-1)    : {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(data_y, y_pred))
