import pandas as pd
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

supervised_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
file1 = open('/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/apps-sok-reduced.txt', 'r')
Lines = file1.readlines()
file2 = open('/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/data-files/apps-sok-reduced.txt', 'r')
Lines2 = file2.readlines()

list_of_train = [1, 2, 3, 4]
list_of_test = [1, 2, 3, 4]
df_train_all = pd.DataFrame()
df_test_all = pd.DataFrame()

# Daten einlesen
for line in Lines:
    content = line.strip()
    for k in list_of_train:
        path = supervised_path + content + '-' + str(k) + '.pkl'
        print(path)
        with open(path, 'rb') as f:
            df_individual_train = pickle.load(f)
        if df_individual_train.isnull().any().any():
            print("NaNs gefunden in:", path)
            exit()
        df_train_all = pd.concat([df_train_all, df_individual_train], axis=0)

for line in Lines2:
    content = line.strip()
    for k in list_of_test:
        path = supervised_path + content + '-' + str(k) + '.pkl'
        print(path)
        with open(path, 'rb') as f:
            df_individual_test = pickle.load(f)
        if df_individual_test.isnull().any().any():
            print("NaNs gefunden in:", path)
            exit()
        df_test_all = pd.concat([df_test_all, df_individual_test], axis=0)

print(df_train_all.shape)
print(df_test_all.shape)

# Features und Labels
train_data_x = df_train_all.iloc[:, :-1]
train_data_y = df_train_all.iloc[:, -1:]
test_data_x = df_test_all.iloc[:, :-1]
test_data_y = df_test_all.iloc[:, -1:]

# Skalierung
sc = StandardScaler()
scaled_train_data_x = sc.fit_transform(train_data_x)
scaled_test_data_x = sc.transform(test_data_x)

# Gesamt-Daten
X = np.concatenate((scaled_train_data_x, scaled_test_data_x), axis=0)
y = np.concatenate((train_data_y.values.ravel(), test_data_y.values.ravel()), axis=0)

X = pd.DataFrame(X)
y = pd.DataFrame(y)

rfc = DecisionTreeClassifier(max_features=200)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fprs, tprs, scores = [], [], []
metrics_per_fold = []

for (train_idx, test_idx), i in zip(cv.split(X, y), range(1, 5)):
    rfc.fit(X.iloc[train_idx], y.iloc[train_idx].values.ravel())
    
    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx].values.ravel()
    y_pred = rfc.predict(X_test_fold)
    y_prob = rfc.predict_proba(X_test_fold)

    # AUC für Klasse -1
    class_index = list(rfc.classes_).index(-1)
    y_scores = y_prob[:, class_index]
    auc_score = roc_auc_score((y_test_fold == -1).astype(int), y_scores)

    # Metriken für Klasse -1
    precision = precision_score(y_test_fold, y_pred, pos_label=-1)
    recall = recall_score(y_test_fold, y_pred, pos_label=-1)
    f1 = f1_score(y_test_fold, y_pred, pos_label=-1)
    accuracy = accuracy_score(y_test_fold, y_pred)

    fpr, tpr, _ = roc_curve((y_test_fold == -1).astype(int), y_scores)
    fprs.append(fpr)
    tprs.append(tpr)

    metrics_per_fold.append({
        'Fold': i,
        'Precision_-1': precision,
        'Recall_-1': recall,
        'F1_-1': f1,
        'Accuracy': accuracy,
        'AUC_-1': auc_score
    })

    print(f"\nFold {i}")
    print(f"Precision (-1): {precision:.3f}")
    print(f"Recall    (-1): {recall:.3f}")
    print(f"F1 Score  (-1): {f1:.3f}")
    print(f"Accuracy       : {accuracy:.3f}")
    print(f"AUC       (-1): {auc_score:.3f}")
    print(confusion_matrix(y_test_fold, y_pred))
    print(classification_report(y_test_fold, y_pred))

# Plot ROC curves
plt.figure(figsize=(8, 8))
for i in range(len(fprs)):
    roc_auc = auc(fprs[i], tprs[i])
    ysmoothed = gaussian_filter1d(tprs[i], sigma=2)
    plt.plot(fprs[i], ysmoothed, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve für Klasse -1 (Decision Tree)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()

# Ausgabe als DataFrame
metrics_df = pd.DataFrame(metrics_per_fold)
print("\nZusammenfassung der Metriken pro Fold:")
print(metrics_df)
