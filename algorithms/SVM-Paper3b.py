import os
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.svm import OneClassSVM, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

class HybridAnomalyDetector:
    def __init__(self, theta=0.1, tau=-0.1, lambda_=0.5, max_normal_samples=10000):
        self.theta = theta
        self.tau = tau
        self.lambda_ = lambda_
        self.max_normal_samples = max_normal_samples
        self.f1_model = OneClassSVM(kernel='rbf', nu=0.01)
        self.f2_model = SVC(kernel='rbf', probability=True, class_weight='balanced', verbose=True)
        self.s1_attempts = 0
        self.s1_correct = 0
        self.s2_attempts = 0
        self.s2_correct = 0

    def credibility_score(self, attempts, correct):
        if attempts == 0 or correct / attempts <= self.lambda_:
            return 0.0
        return correct / attempts

    def fit(self, X, y):
        print("[+] Trainiere One-Class SVM auf max. 10.000 normalen Daten...")
        X_norm = X[y == 1]
        if len(X_norm) > self.max_normal_samples:
            indices = np.random.default_rng(seed=42).choice(len(X_norm), size=self.max_normal_samples, replace=False)
            X_norm = X_norm[indices]
        self.f1_model.fit(X_norm)

        print(f"[+] Trainiere Zwei-Klassen SVM auf {len(X)} Instanzen...")
        start = time.time()
        self.f2_model.fit(X, y)
        print(f"[✓] SVC Training abgeschlossen in {time.time() - start:.2f} Sekunden")

    def predict(self, X, y_true=None):
        print("[+] Berechne Vorhersagen...")
        f1_raw = self.f1_model.predict(X)
        f2_score = self.f2_model.decision_function(X)
        f2_raw = np.where(f2_score >= 0, 1, -1)

        p = np.mean(f1_raw == -1)
        s1 = self.credibility_score(self.s1_attempts, self.s1_correct)
        s2 = self.credibility_score(self.s2_attempts, self.s2_correct)

        preds = []
        for i in tqdm(range(len(X)), desc="Hybrid-Vorhersage"):
            f1 = f1_raw[i]
            f2 = f2_raw[i]
            if p == 0:
                score = f1 * s1
            elif p >= self.theta:
                score = 0.5 * (f1 * s1 + f2 * s2)
            else:
                score = f1 * s1 * (1 - p / (2 * self.theta)) + f2 * s2 * (p / (2 * self.theta))
            final = -1 if score < self.tau else 1
            preds.append(final)

            if y_true is not None:
                self.s1_attempts += 1
                self.s2_attempts += 1
                if f1 == y_true[i]:
                    self.s1_correct += 1
                if f2 == y_true[i]:
                    self.s2_correct += 1

        return np.array(preds)

def load_all_pkl_data(folder_path, limit=50):
    all_data = []
    print(f"[+] Lade bis zu {limit} .pkl-Dateien aus Ordner: {folder_path}")
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    for filename in tqdm(all_files[:limit], desc="Lade Dateien"):
        full_path = os.path.join(folder_path, filename)
        df = pd.read_pickle(full_path)
        all_data.append(df)
    if not all_data:
        raise RuntimeError("Keine .pkl-Dateien gefunden!")
    return pd.concat(all_data, ignore_index=True)

def main():
    folder_path = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp"

    df = load_all_pkl_data(folder_path, limit=50)
    print(f"[i] Datensätze geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    X = df.iloc[:, :-1].astype(float).values
    y = df.iloc[:, -1].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("[+] Skaliere Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[+] Wende PCA an (50 Komponenten)...")
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = HybridAnomalyDetector(tau=-0.1)

    start_train = time.time()
    model.fit(X_train_pca, y_train)
    print(f"[✓] Trainingszeit: {time.time() - start_train:.2f} Sekunden")

    # Begrenze Testmenge
    X_test_pca = X_test_pca[:10000]
    y_test = y_test[:10000]

    start_pred = time.time()
    preds = model.predict(X_test_pca, y_true=y_test)
    print(f"[✓] Vorhersagezeit: {time.time() - start_pred:.2f} Sekunden")

    print("\n=== Klassifikationsbericht ===")
    print(classification_report(y_test, preds, target_names=["Anomalie", "Normal"]))

if __name__ == "__main__":
    main()
