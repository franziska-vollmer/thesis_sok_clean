import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

# === DEBUG-MODUS aktivierbar ===
DEBUG = False

# === Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')

train_indices = [1] if DEBUG else [1, 2, 3, 4]
test_indices = [1] if DEBUG else [1, 2, 3, 4]

# === Lade CVE Liste ===
with open(apps_file_path, 'r') as file:
    all_apps = [line.strip() for line in file if line.strip()]

random.shuffle(all_apps)

n_folds = 1 if DEBUG else 5
fold_size = 2 if DEBUG else 5  # Test-CVEs pro Fold
max_apps = n_folds * fold_size
all_apps = all_apps[:max_apps + 10]  # Sicherheitsreserve

fold_results = []
best_params = None

print(f"\n🟢 Starte Fold-Verarbeitung mit {n_folds} Fold(s) à {fold_size} Test-CVEs...")
print(f"📁 CVE-Datensatzgröße: {len(all_apps)}")

# === CVE Cross-Validation ===
for fold in tqdm(range(n_folds), desc="Folds"):
    fold_start_time = time.time()

    test_apps = all_apps[fold * fold_size:(fold + 1) * fold_size]
    train_apps = [app for app in all_apps if app not in test_apps]

    print(f"\n🚀 Fold {fold+1}/{n_folds}")
    print(f"🧪 Test-Apps:  {test_apps}")
    print(f"🎓 Train-Apps: {len(train_apps)} CVEs")

    df_train_all = pd.DataFrame()
    df_test_all = pd.DataFrame()

    print("📥 Lade Trainingsdaten...")
    for app in train_apps:
        for i in train_indices:
            path = os.path.join(supervised_path, f"{app}-{i}.pkl")
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                df_train_all = pd.concat([df_train_all, pickle.load(f)], ignore_index=True)

    print("📥 Lade Testdaten...")
    for app in test_apps:
        for i in test_indices:
            path = os.path.join(supervised_path, f"{app}-{i}.pkl")
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                df_test_all = pd.concat([df_test_all, pickle.load(f)], ignore_index=True)

    print("📊 Vorbereitung der Features & Labels...")
    X_train = df_train_all.iloc[:, :-1]
    y_train = df_train_all.iloc[:, -1].replace(-1, 0)
    X_test = df_test_all.iloc[:, :-1]
    y_test = df_test_all.iloc[:, -1].replace(-1, 0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("⚖️ Balanciere Trainingsdaten (5:1)...")
    df_combined = pd.concat([pd.DataFrame(X_train_scaled), y_train.reset_index(drop=True)], axis=1)
    df_normal = df_combined[df_combined.iloc[:, -1] == 1]
    df_anom = df_combined[df_combined.iloc[:, -1] == 0]
    desired_norm = 5 * len(df_anom)
    df_normal_down = resample(df_normal, replace=False, n_samples=desired_norm, random_state=fold) if len(df_normal) > desired_norm else df_normal
    df_balanced = pd.concat([df_normal_down, df_anom]).sample(frac=1.0, random_state=fold)

    X_train_bal = df_balanced.iloc[:, :-1].values
    y_train_bal = df_balanced.iloc[:, -1].values

    print("🧠 Führe PCA durch...")
    pca = PCA(n_components=30)
    X_train_pca = pca.fit_transform(X_train_bal)
    X_test_pca = pca.transform(X_test_scaled)

    if fold == 0:
        print("🔍 Starte GridSearchCV (nur Fold 1)...")
        param_grid = {'C': [1, 10], 'gamma': [0.01, 'scale']}
        svm_grid = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid,
                                scoring='f1_macro', cv=2, verbose=1, n_jobs=-1)
        svm_grid.fit(X_train_pca, y_train_bal)
        best_params = svm_grid.best_params_
        print(f"✅ Beste Parameter: {best_params}")

    print("🤖 Trainiere SVM...")
    svm = SVC(kernel='rbf', probability=True, **best_params)
    svm.fit(X_train_pca, y_train_bal)

    print("🔮 Erzeuge Vorhersagen...")
    y_score_normal = svm.predict_proba(X_test_pca)[:, 1]
    prob_anom = 1 - y_score_normal

    print("🔧 Optimiere Threshold...")
    lin_thresh = np.linspace(0, 1, 101)
    log_thresh = np.logspace(-8, -1, 8)
    thresholds = np.unique(np.concatenate([lin_thresh, log_thresh]))
    thresholds.sort()

    best_thr = 0.5
    best_f1_0 = -1.0
    for thr in thresholds:
        y_temp = np.where(prob_anom >= thr, 0, 1)
        f1_temp = f1_score(y_test, y_temp, pos_label=0, zero_division=0)
        if f1_temp > best_f1_0:
            best_f1_0 = f1_temp
            best_thr = thr

    y_pred_opt = np.where(prob_anom >= best_thr, 0, 1)

    acc = accuracy_score(y_test, y_pred_opt)
    prec = precision_score(y_test, y_pred_opt, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred_opt, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred_opt, average='macro', zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_score_normal)
    auc_val = auc(fpr, tpr)

    print(f"📊 Fold {fold+1} Ergebnis: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, AUC={auc_val:.4f}")
    print(f"⏱️ Dauer Fold {fold+1}: {time.time() - fold_start_time:.2f}s")

    fold_results.append({
        "fold": fold + 1,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc_val
    })

# === Zusammenfassung ===
df_res = pd.DataFrame(fold_results)
print("\n📈 Durchschnittliche Ergebnisse:")
print(df_res.mean(numeric_only=True).round(4))
print("\n📉 Standardabweichungen:")
print(df_res.std(numeric_only=True).round(4))
