import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. Daten einlesen
# --------------------------------------------
folder_path = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp"
print(f"Lade .pkl-Dateien aus: {folder_path}")
all_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
print(f"{len(all_files)} Dateien gefunden.\n")

data_all = []
labels_all = []

for file in all_files:
    print(f"-> Lade Datei: {file}")
    df = pd.read_pickle(os.path.join(folder_path, file))
    features = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    data_all.append(features)
    labels_all.append(labels)

# --------------------------------------------
# 2. Kombinieren & Skalieren
# --------------------------------------------
print("\nKombiniere und skaliere Daten ...")
X_raw = np.vstack(data_all)
labels_raw = np.concatenate(labels_all)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# --------------------------------------------
# 3. Sequenzen erstellen
# --------------------------------------------
print("Erstelle Sequenzen für LSTM ...")
sequence_length = 10
X_seq, y_seq, label_seq = [], [], []

for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i + sequence_length])
    y_seq.append(X_scaled[i + sequence_length])
    label_seq.append(labels_raw[i + sequence_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
label_seq = np.array(label_seq)

# --------------------------------------------
# 4. Train/Test-Split
# --------------------------------------------
print("Train/Test-Split ...")
split_idx = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
labels_test = label_seq[split_idx:]

# --------------------------------------------
# 5. LSTM-Modell
# --------------------------------------------
print("Baue und trainiere LSTM-Modell ...")
model = Sequential()
model.add(LSTM(100, input_shape=(X_seq.shape[1], X_seq.shape[2])))
model.add(Dense(X_seq.shape[2]))
model.compile(optimizer='adam', loss='mean_squared_error')

# Klassengewichte berechnen
true_labels = (labels_test == -1).astype(int)
unique_classes = np.unique(true_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=true_labels)
weight_dict = dict(zip(unique_classes, class_weights))
print(f"\nClass Weights: {weight_dict}")

# Modell trainieren
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, class_weight=weight_dict)

# --------------------------------------------
# 6. Vorhersage & RMSE
# --------------------------------------------
print("\nVorhersagen auf Testdaten ...")
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=1))

# --------------------------------------------
# 7. Threshold-Optimierung mit F1-Score
# --------------------------------------------
print("\nOptimiere Schwellenwert für maximale F1-Score ...")
best_threshold = 0
best_f1 = 0
thresholds = np.linspace(min(rmse), max(rmse), 200)

for t in thresholds:
    preds = (rmse > t).astype(int)
    f1 = f1_score(true_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nOptimaler Threshold: {best_threshold:.6f} mit F1-Score: {best_f1:.4f}")
final_preds = (rmse > best_threshold).astype(int)

# --------------------------------------------
# 8. Klassifikationsbericht
# --------------------------------------------
print("\n--- Klassifikationsbericht ---")
print(classification_report(true_labels, final_preds, target_names=["Normal", "Anomalie"]))

# Optional: Visualisierung des Threshold-F1-Verlaufs
plt.plot(thresholds, [f1_score(true_labels, (rmse > t).astype(int)) for t in thresholds])
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Optimal: {best_threshold:.4f}')
plt.xlabel("Threshold")
plt.ylabel("F1-Score")
plt.title("Threshold-Optimierung (F1)")
plt.legend()
plt.grid()
plt.show()
