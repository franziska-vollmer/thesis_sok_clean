import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === Einstellungen ===
TIME_STEPS = 10
MAX_ROWS = 5000

def load_data_from_pkls(folder_path):
    dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            path = os.path.join(folder_path, filename)
            df = pd.read_pickle(path)
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def create_sequences(data, time_steps=TIME_STEPS):
    sequences = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i+time_steps])
    return np.array(sequences)

def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, activation='relu')(inputs)
    x = RepeatVector(input_shape[0])(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(input_shape[1]))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mse')
    return model

def main():
    folder_path = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp"
    df = load_data_from_pkls(folder_path)

    # Label -1 = Anomalie → 0 = Anomalie, 1 = normal
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values
    y = np.where(y_raw == -1, 0, 1)

    X_normal = X[y == 1]
    if len(X_normal) > MAX_ROWS:
        idx = np.random.choice(len(X_normal), MAX_ROWS, replace=False)
        X_normal = X_normal[idx]

    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_all_scaled = scaler.transform(X)

    X_train = create_sequences(X_normal_scaled)
    X_all_seq = create_sequences(X_all_scaled)
    y_seq = y[TIME_STEPS:]

    model = build_autoencoder((TIME_STEPS, X_train.shape[2]))

    # EarlyStopping hinzugefügt
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, X_train, 
              epochs=100, 
              batch_size=64, 
              validation_split=0.1, 
              shuffle=True, 
              callbacks=[early_stop])

    # Rekonstruktionsfehler
    X_pred = model.predict(X_all_seq)
    errors = np.mean(np.abs(X_pred - X_all_seq), axis=(1, 2))

    # Threshold automatisch bestimmen (ROC-optimiert)
    fpr, tpr, thresholds = roc_curve(y_seq, errors)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    threshold = thresholds[ix]
    print(f"Optimal Threshold (ROC): {threshold:.4f}")

    y_pred = (errors > threshold).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_seq, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_seq, y_pred, digits=4))

    # Fehlerkurve
    plt.figure(figsize=(15, 4))
    plt.plot(errors, label='Reconstruction Error')
    plt.hlines(threshold, xmin=0, xmax=len(errors), colors='r', label='Threshold')
    plt.legend()
    plt.title("Reconstruction Error Over Time")
    plt.show()

    # ROC-Curve optional
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label='ROC Curve')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='red', label='Best threshold')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
