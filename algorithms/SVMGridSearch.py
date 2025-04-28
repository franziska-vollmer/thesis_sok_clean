import os
import numpy as np
import pandas as pd
import testpkl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_curve, auc
)

# === Pfade ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

# === Dateien einlesen ===
with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

# === Daten laden ===
train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]
df_train_all = pd.DataFrame()
df_test_all = pd.DataFrame()

for line in app_lines:
    app = line.strip()
    for i in train_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df = pickle.load(f)
        df_train_all = pd.concat([df_train_all, df])
    for i in test_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df = pickle.load(f)
        df_test_all = pd.concat([df_test_all, df])

# === Feature-Label-Split ===
X_train = df_train_all.iloc[:, :-1]
y_train = df_train_all.iloc[:, -1].replace(-1, 0)
X_test = df_test_all.iloc[:, :-1]
y_test = df_test_all.iloc[:, -1].replace(-1, 0)

print("Train-Labels:")
print(y_train.value_counts())
print("\nTest-Labels:")
print(y_test.value_counts())

# === Normalisierung ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Hyperparameter-Tuning mit GridSearchCV ===
param_grid = {
    'C': np.logspace(-1, 2, 5),
    'gamma': np.logspace(-3, -1, 5),
    'kernel': ['rbf'],
    'class_weight': ['balanced']
}

svm = SVC(probability=True)

grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("\nBeste Parameter:", grid_search.best_params_)

# === Finales Modell mit besten Parametern ===
best_model = grid_search.best_estimator_

# === Vorhersage ===
y_pred = best_model.predict(X_test_scaled)
y_score = best_model.predict_proba(X_test_scaled)[:, 1]

# === Metriken ===
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# === ROC-Kurve ===
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"SVM ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – SVM mit GridSearch")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("ROC_SVM_GridSearch.png")
plt.show()
