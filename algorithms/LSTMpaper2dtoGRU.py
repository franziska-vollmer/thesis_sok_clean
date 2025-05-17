import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, accuracy_score
from tqdm import tqdm
import os

# Tabelle zur Speicherung der Ergebnisse
results_table = []

# Schritt 1: Laden der .pkl-Dateien
data_folder = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
data_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]

# Liste, um alle Daten zu speichern
data_list = []

# Laden der Daten aus den .pkl-Dateien
for file in tqdm(data_files, desc="Laden der Dateien"):
    df = pd.read_pickle(os.path.join(data_folder, file))
    data_list.append(df)

# Kombinieren aller DataFrames in eine einzige DataFrame
data = pd.concat(data_list, ignore_index=True)

# Schritt 2: Vorverarbeitung der Daten
X = data.iloc[:, :-1].values  # Alle Spalten außer der letzten (Feature-Daten)
y = data.iloc[:, -1].values   # Die letzte Spalte ist das Label (Anomalie oder nicht)

# Schritt 3: Normalisierung der Feature-Daten
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Schritt 4: Umwandlung der Eingabedaten für LSTM (3D-Format: [samples, timesteps, features])
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # 1 timestep

# Schritt 5: Umwandlung der Labels in numerische Werte
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Schritt 6: Train-Test Split (80% Training, 20% Test)
X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Schritt 7: Cross-Validation Setup für das Trainingsset
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 Folds Cross-Validation

# Schritt 8: Modell erstellen
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(X_train_full.shape[1], X_train_full.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Schritt 9: Modell kompilieren
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Schritt 10: Cross-Validation durchführen und Metriken berechnen
for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    print(f"\nFold {fold_num + 1}:")
    
    # Daten für diesen Fold vorbereiten
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    # Modell trainieren
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=0)
    
    # Vorhersagen auf den Validierungsdaten
    y_pred_val = (model.predict(X_val) > 0.5).astype("int32")
    
    # Berechnung von Precision, Recall, F1-Score, AUC und PR-AUC
    precision = precision_score(y_val, y_pred_val)
    recall = recall_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val)
    auc_value = roc_auc_score(y_val, y_pred_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, model.predict(X_val))
    pr_auc = auc(recall_curve, precision_curve)

    # Ergebnisse für dieses Fold speichern
    results_table.append({
        'Fold': fold_num + 1,
        'F1-Score': f1,
        'Precision': precision,
        'Accuracy': accuracy,
        'Recall': recall,
        'ROC-AUC': auc_value,
        'PR-AUC': pr_auc
    })
    
    # Ausgabe der Metriken für dieses Fold
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {auc_value:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

# Schritt 11: Modell auf dem gesamten Trainingsset trainieren und auf dem Testset bewerten
# Nach der Cross-Validation das Modell auf den gesamten Trainingsdaten trainieren
model.fit(X_train_full, y_train_full, epochs=10, batch_size=64, verbose=0)

# Testen des Modells auf den Testdaten
y_pred_test = (model.predict(X_test) > 0.5).astype("int32")

# Berechnung von Precision, Recall, F1-Score, AUC und PR-AUC auf dem Testset
precision_test = precision_score(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
auc_value_test = roc_auc_score(y_test, y_pred_test)

# PR-AUC auf dem Testset
precision_curve_test, recall_curve_test, _ = precision_recall_curve(y_test, model.predict(X_test))
pr_auc_test = auc(recall_curve_test, precision_curve_test)

print(f"\nTestset Metriken:")
print(f"F1-Score: {f1_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"ROC-AUC: {auc_value_test:.4f}")
print(f"PR-AUC: {pr_auc_test:.4f}")

# Schritt 12: Ergebnisse in einer Tabelle anzeigen
results_df = pd.DataFrame(results_table)

# Ausgabe der Tabelle im Terminal
print("\nFold Results:")
print(results_df)

# Schritt 13: Modell speichern
model.save('nids_lstm_model.keras')  # Speichern im Keras-native Format
