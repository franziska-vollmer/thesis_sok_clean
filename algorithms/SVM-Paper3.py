import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class HybridAnomalyDetector:
    def __init__(self, theta=0.1, tau=0.0, lambda_=0.5):
        self.theta = theta
        self.tau = tau
        self.lambda_ = lambda_
        self.f1_model = OneClassSVM(kernel='rbf', nu=0.01)
        self.f2_model = SVC(kernel='rbf', probability=True, class_weight='balanced')
        self.s1_attempts = 0
        self.s1_correct = 0
        self.s2_attempts = 0
        self.s2_correct = 0

    def credibility_score(self, attempts, correct):
        if attempts == 0 or correct / attempts <= self.lambda_:
            return 0.0
        return correct / attempts

    def fit(self, X, y):
        X_norm = X[y == 1]
        self.f1_model.fit(X_norm)
        self.f2_model.fit(X, y)

    def predict(self, X, y_true=None):
        f1_raw = self.f1_model.predict(X)
        f2_score = self.f2_model.decision_function(X)
        f2_raw = np.where(f2_score >= 0, 1, -1)

        p = np.mean(f1_raw == -1)
        s1 = self.credibility_score(self.s1_attempts, self.s1_correct)
        s2 = self.credibility_score(self.s2_attempts, self.s2_correct)

        preds = []
        for i in range(len(X)):
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

def main():
    # Lade die Daten
    df1 = pd.read_pickle("/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/CVE-2023-23752-1.pkl")
    df2 = pd.read_pickle("/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/CVE-2023-23752-3.pkl")
    df3 = pd.read_pickle("/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/CVE-2023-23752-4.pkl")
    df = pd.concat([df1, df2, df3], ignore_index=True)

    # Merkmale & Labels
    X = df.iloc[:, :-1].astype(float)
    y = df.iloc[:, -1].astype(int)

    # Aufteilen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Skalieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modell trainieren & auswerten
    model = HybridAnomalyDetector()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled, y_true=y_test)

    print("=== Klassifikationsbericht ===")
    print(classification_report(y_test, preds, target_names=["Anomalie", "Normal"]))

if __name__ == "__main__":
    main()
