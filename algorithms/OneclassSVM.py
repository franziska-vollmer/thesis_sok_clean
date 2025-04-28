import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from tqdm import tqdm

# === Lade Daten ===
print("🔄 Lade Daten ...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')

with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

train_indices = [1, 2, 3, 4]
test_indices  = [1, 2, 3, 4]

df_train_all = pd.DataFrame()
df_test_all  = pd.DataFrame()

for line in app_lines:
    app = line.strip()
    print(f"📦 Lade App: {app}")
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

# === Vorverarbeitung ===
print("🧹 Daten vorbereiten ...")
X_train = df_train_all.iloc[:, :-1]
y_train = df_train_all.iloc[:, -1].replace(-1, 0)
X_test  = df_test_all.iloc[:, :-1]
y_test  = df_test_all.iloc[:, -1].replace(-1, 0)

# === Nur normale Daten fürs Training (und dann sampeln)
X_train_norm = X_train[y_train == 1]
X_train_sample = X_train_norm.sample(n=20000, random_state=42)
y_train_sample = pd.Series([1] * len(X_train_sample))

# === Testdaten (auch gesampelt)
test_sample = df_test_all.sample(n=50000, random_state=42)
X_test_sample = test_sample.iloc[:, :-1]
y_test_sample = test_sample.iloc[:, -1].replace(-1, 0)

# === Skalieren
print("📏 Skalieren ...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sample)
X_test_scaled = scaler.transform(X_test_sample)

# === PCA
print("🧠 PCA anwenden (n=20) ...")
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# === Parametertest (klein & schnell)
nu_values = [0.01, 0.02]
gammas = ['scale']
kernel = 'rbf'

best_f1_0 = -1
best_model = None
best_params = {}

print("🔍 Starte Parametertest ...")
for nu in tqdm(nu_values, desc="nu"):
    for gamma in gammas:
        print(f"\n🔧 Teste: nu={nu}, kernel={kernel}, gamma={gamma}")
        start = time.time()
        ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        ocsvm.fit(X_train_pca)
        train_time = time.time() - start
        print(f"⏱️ Trainingszeit: {train_time:.2f} Sek.")

        y_pred = ocsvm.predict(X_test_pca)
        y_pred_converted = np.where(y_pred == -1, 0, 1)

        f1_0 = f1_score(y_test_sample, y_pred_converted, pos_label=0)
        print(f"➡️ F1(0): {f1_0:.4f}")

        if f1_0 > best_f1_0:
            best_f1_0 = f1_0
            best_model = ocsvm
            best_params = {'nu': nu, 'kernel': kernel, 'gamma': gamma}

# === Beste Parameter & finale Bewertung
print("\n✅ Beste Parameter:")
print(best_params)
print(f"Bester F1(Anomalie): {best_f1_0:.4f}")

y_pred_final = best_model.predict(X_test_pca)
y_pred_final_converted = np.where(y_pred_final == -1, 0, 1)

acc = accuracy_score(y_test_sample, y_pred_final_converted)
prec = precision_score(y_test_sample, y_pred_final_converted, average='macro')
rec = recall_score(y_test_sample, y_pred_final_converted, average='macro')
f1 = f1_score(y_test_sample, y_pred_final_converted, average='macro')
cm = confusion_matrix(y_test_sample, y_pred_final_converted)

print("\n📊 Finale Evaluation:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision(macro): {prec:.4f}")
print(f"Recall(macro):    {rec:.4f}")
print(f"F1(macro):        {f1:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test_sample, y_pred_final_converted))

# === ROC-Kurve
print("📈 ROC-Kurve ...")
scores = -best_model.score_samples(X_test_pca)
fpr, tpr, _ = roc_curve(y_test_sample, scores, pos_label=0)
auc_val = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"One-Class SVM (AUC={auc_val:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC - One-Class SVM (Anomalien, sample)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
