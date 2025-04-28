import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# === 1. Daten laden (angepasst an dein Setup) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]

df_train_all = pd.DataFrame()
df_test_all = pd.DataFrame()

for line in app_lines:
    app = line.strip()
    for i in train_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df_temp = pickle.load(f)
        df_train_all = pd.concat([df_train_all, df_temp], ignore_index=True)
    for i in test_indices:
        path = os.path.join(supervised_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df_temp = pickle.load(f)
        df_test_all = pd.concat([df_test_all, df_temp], ignore_index=True)

# === 2. Feature/Label-Vorbereitung ===
X_train = df_train_all.iloc[:, 1:-1].values.astype(np.float32)
y_train = np.where(df_train_all.iloc[:, -1].values == -1, 0, 1)

X_test = df_test_all.iloc[:, 1:-1].values.astype(np.float32)
y_test = np.where(df_test_all.iloc[:, -1].values == -1, 0, 1)

print("Gesamtshape X:", X_train.shape[0] + X_test.shape[0], "Positivklasse:", np.sum(np.concatenate([y_train, y_test])))

# === 3. Skalieren ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. SVM trainieren ===
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced')
svm.fit(X_train_scaled, y_train)

# === 5. Vorhersage & Metriken ===
y_pred = svm.predict(X_test_scaled)
y_pred_prob = svm.predict_proba(X_test_scaled)[:, 1]

print("\n=== Klassifikationsmetriken (SVM) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# === 6. ROC-Kurve ===
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'SVM ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (SVM)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
