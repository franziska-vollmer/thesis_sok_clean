import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# === Pfade ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

# === Parameter ===
train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]
IMG_SIZE = 8
NUM_BINS = 10
TRAIN_APPS = 15
TEST_APPS = 3

# === App-Liste lesen und splitten ===
with open(apps_file_path, 'r') as file:
    all_apps = [line.strip() for line in file.readlines() if line.strip()]
train_apps = all_apps[:TRAIN_APPS]
test_apps = all_apps[TRAIN_APPS:TRAIN_APPS + TEST_APPS]

# === Daten laden ===
def load_data(apps, indices):
    df_all = pd.DataFrame()
    for app in apps:
        for i in indices:
            path = os.path.join(supervised_path, f"{app}-{i}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    df = pickle.load(f)
                df_all = pd.concat([df_all, df])
    return df_all

df_train_all = load_data(train_apps, train_indices)
df_test_all = load_data(test_apps, test_indices)

# === Preprocessing ===
def preprocess(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    # Normalize → Binning → One-hot → Padding → Image
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    bins = np.linspace(0, 1, NUM_BINS + 1)
    X_binned = np.digitize(X_scaled, bins) - 1
    X_binned[X_binned == NUM_BINS] = NUM_BINS - 1

    X_onehot = np.array([to_categorical(row, NUM_BINS).flatten() for row in X_binned])
    if X_onehot.shape[1] < 64:
        X_padded = np.pad(X_onehot, ((0, 0), (0, 64 - X_onehot.shape[1])), mode='constant')
    else:
        X_padded = X_onehot[:, :64]

    X_img = X_padded.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
    return X_img, y

X_train, y_train = preprocess(df_train_all)
X_test, y_test = preprocess(df_test_all)

# === CNN Modell ===
def build_cnn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn((IMG_SIZE, IMG_SIZE, 1))

# === Training (nur 15 CVEs) ===
model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2, validation_split=0.2)

# === Evaluation (auf 3 Test-CVEs) ===
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
print("\n📊 Evaluation auf Testdaten (3 Apps):")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
