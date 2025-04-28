# -------------------------------
# 1. IMPORTS
# -------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import glob
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras import layers, models

print("✅ Libraries erfolgreich importiert.")

# -------------------------------
# 2. HILFSFUNKTIONEN
# -------------------------------

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def dummy_decompose(x):
    x_np = x.numpy()
    trend = np.polyval(np.polyfit(np.arange(len(x_np)), x_np, 1), np.arange(len(x_np)))
    seasonality = np.zeros_like(x_np)
    residual = x_np - trend - seasonality
    return trend, seasonality, residual

def conv_block(x, filters, kernel_size=3, activation='relu'):
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.Activation(activation)(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling1D(pool_size=2)(f)
    return f, p

def decoder_block(x, skip, filters):
    us = layers.UpSampling1D(size=2)(x)
    concat = layers.Concatenate()([us, skip])
    f = conv_block(concat, filters)
    return f

def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    f1, p1 = encoder_block(inputs, 32)
    f2, p2 = encoder_block(p1, 64)
    f3, p3 = encoder_block(p2, 128)

    bottleneck = conv_block(p3, 256)

    d3 = decoder_block(bottleneck, f3, 128)
    d2 = decoder_block(d3, f2, 64)
    d1 = decoder_block(d2, f1, 32)

    outputs = layers.Conv1D(1, 1, activation='sigmoid')(d1)

    model = models.Model(inputs, outputs)
    return model

def weighted_binary_crossentropy(y_true, y_pred, pos_weight=10):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = -(pos_weight * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)

# -------------------------------
# 3. DATEN LADEN
# -------------------------------

print("\n📥 Daten werden geladen...")

base_path = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'

pkl_files = glob.glob(os.path.join(base_path, '*.pkl'))
print(f"🔍 {len(pkl_files)} PKL-Dateien gefunden.")

X_list = []
y_list = []

for file in pkl_files:
    print(f"➡️  Lade Datei: {os.path.basename(file)}")
    df = load_pkl(file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_list.append(X)
    y_list.append(y)

X_all = np.vstack(X_list)
y_all = np.hstack(y_list)

print(f"✅ Daten geladen. Feature-Shape: {X_all.shape}, Label-Shape: {y_all.shape}")

# -------------------------------
# 4. VORBEREITUNG
# -------------------------------

print("\n⚙️ Vorbereitung der Daten...")

X_all = (X_all - np.mean(X_all, axis=0)) / (np.std(X_all, axis=0) + 1e-8)

X_tf = tf.convert_to_tensor(X_all, dtype=tf.float32)
y_tf = tf.convert_to_tensor(y_all, dtype=tf.float32)

print("➡️ Dummy-Decomposition auf Residual...")
trend, seasonality, residual = dummy_decompose(X_tf[:, 0])
residual = tf.convert_to_tensor(residual, dtype=tf.float32)
residual = tf.expand_dims(residual, axis=-1)

X_tf = tf.expand_dims(residual, axis=0)
y_tf = tf.expand_dims(y_tf, axis=(0, -1))

print(f"✅ Tensor-Formate bereit. X-Shape: {X_tf.shape}, y-Shape: {y_tf.shape}")

# -------------------------------
# 5. MODELL BAUEN & TRAINIEREN
# -------------------------------

print("\n🏗️  Modell wird gebaut...")

model = build_unet(input_shape=(X_tf.shape[1], X_tf.shape[2]))
model.summary()

print("\n⚙️ Kompiliere Modell...")
model.compile(optimizer='adam', loss=lambda yt, yp: weighted_binary_crossentropy(yt, yp, pos_weight=10))

print("\n🚀 Starte Training...")
history = model.fit(X_tf, y_tf, epochs=50, batch_size=1, verbose=1)

# -------------------------------
# 6. EVALUATION: Precision, Recall, F1
# -------------------------------

print("\n📈 Modell wird ausgewertet...")

predictions = model.predict(X_tf)

pred_labels = (predictions > 0.5).astype(int)

y_true = y_tf.numpy().flatten()
y_pred = pred_labels.flatten()

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n🎯 --- Evaluation Ergebnisse ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\n✅ Fertig!")
