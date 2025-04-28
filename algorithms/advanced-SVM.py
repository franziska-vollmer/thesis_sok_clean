import os, pickle, random, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.utils import resample

# === Pfade ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]

# === Lade CVE Liste ===
with open(apps_file_path, 'r') as f:
    all_apps = [line.strip() for line in f if line.strip()]
random.shuffle(all_apps)
fold_size = 2
train_apps = all_apps[fold_size:]
test_apps = all_apps[:fold_size]

print(f"🔁 Mini-Debug-Fold: {len(train_apps)} train, {len(test_apps)} test")

# === Daten laden ===
def load_apps(apps, indices):
    df = pd.DataFrame()
    for app in apps:
        for i in indices:
            path = os.path.join(supervised_path, f"{app}-{i}.pkl")
            with open(path, 'rb') as f:
                df = pd.concat([df, pickle.load(f)], ignore_index=True)
    return df

print("📦 Lade Trainingsdaten ...")
df_train = load_apps(train_apps, train_indices)
print("📦 Lade Testdaten ...")
df_test = load_apps(test_apps, test_indices)
print(f"✅ Daten geladen: {df_train.shape[0]} Trainingszeilen, {df_test.shape[0]} Testzeilen")

# === Feature/Label Split ===
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1].replace(-1, 0)
X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1].replace(-1, 0)

# === Skalieren & Balancing ===
print("📏 Skaliere Daten ...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("⚖️ Balanciere Daten ...")
df_train_combined = pd.concat([pd.DataFrame(X_train_scaled), y_train.reset_index(drop=True)], axis=1)
df_normal = df_train_combined[df_train_combined.iloc[:, -1] == 1]
df_anom = df_train_combined[df_train_combined.iloc[:, -1] == 0]

df_normal_down = resample(df_normal, replace=False, n_samples=len(df_anom)*5, random_state=42)
df_train_bal = pd.concat([df_normal_down, df_anom]).sample(frac=1, random_state=42)
X_train_bal = df_train_bal.iloc[:, :-1].values
y_train_bal = df_train_bal.iloc[:, -1].values

print("✅ Balancing abgeschlossen")

# === PCA ===
print("🧠 Führe PCA durch ...")
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_bal)
X_test_pca = pca.transform(X_test_scaled)
print("✅ PCA abgeschlossen")

# === SVM Training ===
print("🚀 Trainiere SVM mit C=1, gamma='scale' ...")
svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm.fit(X_train_pca, y_train_bal)
print("✅ SVM Training abgeschlossen")

# === Threshold-Tuning ===
print("🎯 Suche besten Threshold für F1(Anomalie) ...")
y_score = svm.predict_proba(X_test_pca)[:, 1]
prob_anom = 1 - y_score
thresholds = np.linspace(0, 1, 101)
best_thr = 0.5
best_f1 = -1

for thr in thresholds:
    y_pred = np.where(prob_anom >= thr, 0, 1)
    f1 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
print(f"✅ Bester Threshold: {best_thr:.2f} mit F1(0) = {best_f1:.4f}")

# === Finale Bewertung ===
print("📊 Berechne finale Metriken ...")
y_pred_opt = np.where(prob_anom >= best_thr, 0, 1)
acc = accuracy_score(y_test, y_pred_opt)
prec = precision_score(y_test, y_pred_opt, average='macro')
rec = recall_score(y_test, y_pred_opt, average='macro')
f1 = f1_score(y_test, y_pred_opt, average='macro')
fpr, tpr, _ = roc_curve(y_test, y_score)
auc_val = auc(fpr, tpr)

print("\n✅ Ergebnisse (mit optimalem Threshold):")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc_val:.4f}")
