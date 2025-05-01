import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC
from sklearn.metrics import classification_report, confusion_matrix

# === Funktion zur Erzeugung synthetischer normaler Daten ===
def generate_synthetic_normals(X_normal, n_samples=1000, noise_level=0.01, random_state=42):
    np.random.seed(random_state)
    X_synthetic = []
    for _ in range(n_samples):
        base = X_normal.sample(n=1, replace=True).values.flatten()
        noise = np.random.normal(loc=0.0, scale=noise_level, size=base.shape)
        synthetic = base * (1 + noise)
        X_synthetic.append(synthetic)
    return pd.DataFrame(X_synthetic, columns=X_normal.columns)

print("📥 1. Lade Daten ...")
file_paths = [
    "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/CVE-2023-23752-1.pkl",
    "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/CVE-2023-23752-3.pkl",
    "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/CVE-2023-23752-4.pkl"
]
dfs = [pd.read_pickle(path) for path in file_paths]
df = pd.concat(dfs, ignore_index=True)

print("🧹 2. Bereite Features und Labels vor ...")
X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df.iloc[:, -1].replace({-1: 0, 1: 1}).astype(int)

print("🔍 Label-Verteilung:", y.value_counts().to_dict())

X_normal = X[y == 0]
X_anomalous = X[y == 1]
y_normal = y[y == 0]
y_anomalous = y[y == 1]

# === Erzeuge synthetische normale Daten ===
print("🧪 Erzeuge synthetische normale Daten ...")
X_synth = generate_synthetic_normals(X_normal, n_samples=1000)
y_synth = pd.Series([0] * len(X_synth))

# === Splitte in Train/Test ===
print("🔀 Splitte in Training und Test ...")
X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(
    X_normal, y_normal, test_size=0.2, random_state=42)
X_anom_train, X_anom_test, y_anom_train, y_anom_test = train_test_split(
    X_anomalous, y_anomalous, test_size=0.2, random_state=42)

X_train = pd.concat([X_norm_train, X_anom_train, X_synth])
y_train = pd.concat([y_norm_train, y_anom_train, y_synth])
X_test = pd.concat([X_norm_test, X_anom_test])
y_test = pd.concat([y_norm_test, y_anom_test])

print("📊 Trainingsgröße:", X_train.shape)
print("🧪 Testgröße:", X_test.shape)

# === Skalieren ===
print("⚖️ Skaliere Daten ...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_normal_scaled = scaler.transform(pd.concat([X_norm_train, X_synth]))

# === Trainiere Modelle ===
print("📈 Trainiere One-Class SVM ...")
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
ocsvm.fit(X_train_normal_scaled)

print("📈 Trainiere Zwei-Class SVM ...")
svc = SVC(probability=True, kernel='rbf', gamma='auto', class_weight='balanced')
svc.fit(X_train_scaled, y_train)

# === Vorhersagen ===
print("🔮 Berechne Vorhersagen ...")
f1_pred = (ocsvm.predict(X_test_scaled) == -1).astype(int)
f2_pred = svc.predict(X_test_scaled)

def credibility_score(y_true, y_pred, lambd=0.5):
    correct = np.sum(y_true == y_pred)
    attempted = len(y_true)
    return correct / attempted if attempted > 0 and (correct / attempted) > lambd else 0

print("📏 Berechne Glaubwürdigkeits-Scores ...")
s1 = credibility_score(y_test, f1_pred)
s2 = credibility_score(y_test, f2_pred)
p = y_test.mean()
theta = 0.1
print(f"    → s1 (1-class SVM): {s1:.3f}")
print(f"    → s2 (2-class SVM): {s2:.3f}")
print(f"    → Anteil Anomalien im Testset: {p:.3f}")

# === Hybrid-Kombination ===
print("⚙️ Berechne Hybrid-Entscheidung ...")
f1_scores = ocsvm.decision_function(X_test_scaled)
f2_scores = svc.decision_function(X_test_scaled)

def hybrid_predict(f1, f2, s1, s2, p, theta):
    if p == 0:
        return f1 * s1
    elif p >= theta:
        return 0.5 * (f1 * s1 + f2 * s2)
    else:
        return f1 * s1 * (1 - p / (2 * theta)) + f2 * s2 * (p / (2 * theta))

f_combined_scores = np.array([
    hybrid_predict(f1, f2, s1, s2, p, theta)
    for f1, f2 in zip(f1_scores, f2_scores)
])

y_pred_combined = (f_combined_scores < 0).astype(int)

# === Evaluation ===
print("\n📋 Ergebnis")
print("--- 📊 Klassifikationsbericht ---")
print(classification_report(y_test, y_pred_combined, target_names=["Normal (0)", "Anomalie (1)"]))

print("--- 🔢 Konfusionsmatrix ---")
print(confusion_matrix(y_test, y_pred_combined))
