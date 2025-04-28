import os
import numpy as np
import pandas as pd
import testpkl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.utils import resample

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

# === Trainingdaten ausbalancieren ===
df_train_combined = pd.concat([pd.DataFrame(X_train_scaled), y_train.reset_index(drop=True)], axis=1)
df_majority = df_train_combined[df_train_combined.iloc[:, -1] == 1]
df_minority = df_train_combined[df_train_combined.iloc[:, -1] == 0]

df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])

X_train_balanced = df_balanced.iloc[:, :-1].values
y_train_balanced = df_balanced.iloc[:, -1].values


# === SVM Modell (ohne class_weight, weniger Speicher) ===
svm_model = SVC(kernel='rbf', probability=True, verbose=False)

# === Training auf balancierten Daten ===
svm_model.fit(X_train_balanced, y_train_balanced)

# === Batched Vorhersage ===
def batched_predict(model, data, batch_size=10000):
    batches = np.array_split(data, int(np.ceil(len(data) / batch_size)))
    preds = [model.predict(batch) for batch in batches]
    probs = [model.predict_proba(batch)[:, 1] for batch in batches]
    return np.concatenate(preds), np.concatenate(probs)

y_pred, y_score = batched_predict(svm_model, X_test_scaled)

# === Bewertung ===
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# === ROC ===
fpr, tpr, thresholds = roc_curve(y_test, y_score)
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
plt.savefig('ROC_SVM_balanced_rbf_final.png')
plt.show()
