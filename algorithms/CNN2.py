import os
import numpy as np
import pandas as pd
import testpkl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping

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
y_train = df_train_all.iloc[:, -1]
X_test = df_test_all.iloc[:, :-1]
y_test = df_test_all.iloc[:, -1]

y_train = y_train.replace(-1, 0)
y_test = y_test.replace(-1, 0)

print("Train-Labels:")
print(y_train.value_counts())
print("\nTest-Labels:")
print(y_test.value_counts())

# === Normalisierung ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Reshape für CNN ===
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# === Class Weights berechnen ===
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_array))
print("Class Weights:", class_weights_dict)

# === CNN Modell definieren ===
model = Sequential()
model.add(Conv1D(64, kernel_size=3, input_shape=(X_train_cnn.shape[1], 1)))
model.add(LeakyReLU())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(128, kernel_size=3))
model.add(LeakyReLU())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64))
model.add(LeakyReLU())
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))  # Binary output
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Training ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train_cnn, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_cnn, y_test),
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=1
)

# === Vorhersagen ===
y_pred_prob = model.predict(X_test_cnn)
y_pred = (y_pred_prob > 0.5).astype(int)

# === Bewertung ===
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# === ROC-Kurve ===
y_score = y_pred_prob.ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'CNN ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (CNN Improved)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('ROC_CNN_improved.png')
plt.show()
