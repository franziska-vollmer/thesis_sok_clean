import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

# 1. Dummy Daten simulieren (z. B. CPU-Last, RAM-Verbrauch, Disk I/O etc.)
np.random.seed(42)
normal_data = np.random.normal(loc=0.5, scale=0.1, size=(300, 4))
anomaly_data = np.random.uniform(low=0.8, high=1.0, size=(20, 4))
data = np.vstack([normal_data, anomaly_data])

# 2. Standardisierung
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 3. One-Class SVM Modell trainieren
model = OneClassSVM(kernel='poly', nu=0.2)
model.fit(data_scaled[:300])  # nur normal trainieren
preds = model.predict(data_scaled)

# 4. Ergebnisse plotten
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=preds, cmap='coolwarm')
plt.title("One-Class SVM Anomalieerkennung")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
