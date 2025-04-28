# 📦 Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# 📥 Schritt 1: Alle .pkl Dateien laden
pkl_folder = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
pkl_paths = glob.glob(pkl_folder + '*.pkl')

print(f"Gefundene .pkl Dateien: {len(pkl_paths)}")

dfs = [pd.read_pickle(path) for path in pkl_paths]
df_all = pd.concat(dfs, ignore_index=True)
print("Shape aller zusammengefügten Daten:", df_all.shape)

# 📋 Schritt 2: Features und Labels
X = df_all.iloc[:, 1:556]
y = df_all[556]

# 🧠 Schritt 3: Remapping
y = y.map({1: 0, -1: 1})
print("\nLabel Verteilung nach Remapping:")
print(y.value_counts())

# 📈 Schritt 4: Skalieren
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 📦 Schritt 5: Padding auf 576 Features für 24x24 Bilder
def pad_features(X, target_length=576):
    padded = []
    for row in X:
        if len(row) < target_length:
            row_padded = np.pad(row, (0, target_length - len(row)))
        else:
            row_padded = row[:target_length]
        padded.append(row_padded)
    return np.array(padded)

X_padded = pad_features(X_scaled)

# 📷 Schritt 6: Bilder erstellen
X_images = X_padded.reshape(-1, 24, 24, 1)

# ✂️ Schritt 7: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42, stratify=y)

# 🧠 Schritt 8: CNN Modell
def create_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    return model

model = create_cnn_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 🧠 Schritt 9: Class Weights berechnen
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("\nClass Weights:", class_weight_dict)

# 🚀 Schritt 10: Training mit Class Weights
history = model.fit(
    X_train, y_train,
    epochs=5,  # gerne 5 Epochs, weil stabil
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict
)

# 🧪 Schritt 11: Testen
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 📈 Schritt 12: Trainingskurven
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 🧪 Schritt 13: Standard Evaluation
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_prob[:, 1])

print("\n=== Standard Evaluation ===")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# 📊 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 📈 ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# 📈 Precision-Recall Curve
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_pred_prob[:, 1])
plt.figure(figsize=(8,6))
plt.plot(recall_vals, precision_vals, label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

# 🚀 Schritt 14: Threshold Tuning
thresholds = np.arange(0.0, 1.01, 0.05)
precision_list = []
recall_list = []
f1_list = []

for thresh in thresholds:
    y_pred_thresh = (y_pred_prob[:, 1] >= thresh).astype(int)
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

# 📊 Plot Precision, Recall, F1 vs Threshold
plt.figure(figsize=(10,7))
plt.plot(thresholds, precision_list, label='Precision')
plt.plot(thresholds, recall_list, label='Recall')
plt.plot(thresholds, f1_list, label='F1-Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision / Recall / F1 vs Threshold')
plt.legend()
plt.grid(True)
plt.show()

# 📋 Beste Threshold finden
best_idx = np.argmax(f1_list)
best_threshold = thresholds[best_idx]
print(f"\nBeste Schwelle (max F1-Score): {best_threshold:.2f}")
print(f"Precision @ Best Threshold: {precision_list[best_idx]:.4f}")
print(f"Recall @ Best Threshold: {recall_list[best_idx]:.4f}")
print(f"F1-Score @ Best Threshold: {f1_list[best_idx]:.4f}")

# 🚀 Schritt 15: Best-Threshold-Prediction
y_pred_best_thresh = (y_pred_prob[:, 1] >= best_threshold).astype(int)

precision_best = precision_score(y_test, y_pred_best_thresh, zero_division=0)
recall_best = recall_score(y_test, y_pred_best_thresh, zero_division=0)
f1_best = f1_score(y_test, y_pred_best_thresh, zero_division=0)
auc_best = roc_auc_score(y_test, y_pred_prob[:, 1])

print("\n=== Evaluation mit Best-Threshold ===")
print(f"Precision (best threshold): {precision_best:.4f}")
print(f"Recall (best threshold): {recall_best:.4f}")
print(f"F1-Score (best threshold): {f1_best:.4f}")
print(f"AUC Score: {auc_best:.4f}")

# 📊 Neue Confusion Matrix (Best Threshold)
cm_best = confusion_matrix(y_test, y_pred_best_thresh)
plt.figure(figsize=(6,5))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Best Threshold)')
plt.show()
