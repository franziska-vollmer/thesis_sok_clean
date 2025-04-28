import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import f1_score, classification_report, roc_curve, auc
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout,
    Flatten, Dense, BatchNormalization, ReLU
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === Funktionen
def load_data(app_list, indices, path):
    df_all = pd.DataFrame()
    for app in app_list:
        for i in indices:
            pkl_path = os.path.join(path, f"{app}-{i}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    df_all = pd.concat([df_all, pickle.load(f)], ignore_index=True)
            else:
                print(f"⚠️ Datei nicht gefunden: {pkl_path}")
    return df_all

def to_image_matrix(X, bins=10):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='uniform')
    X_binned = discretizer.fit_transform(X_scaled)
    padded = np.zeros((X_binned.shape[0], 64))
    padded[:, :X_binned.shape[1]] = X_binned
    images = padded.reshape(-1, 8, 8, 1)
    return images, scaler

def build_image_cnn():
    input_img = Input(shape=(8, 8, 1))
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_img, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_hybrid_pipeline(app_lines, supervised_path, train_indices, test_indices,
                        train_app_count=30, test_app_count=5, epochs=5, batch_size=256, n_folds=3):
    results = []

    for fold in range(n_folds):
        print(f"\n🔁 Fold {fold+1}/{n_folds}")
        np.random.shuffle(app_lines)
        train_apps = app_lines[:train_app_count]
        test_apps = app_lines[train_app_count:train_app_count + test_app_count]

        df_train = load_data(train_apps, train_indices, supervised_path)
        df_test = load_data(test_apps, test_indices, supervised_path)

        X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1].replace(-1, 0)
        X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1].replace(-1, 0)

        # Balance Testset
        idx_0, idx_1 = np.where(y_test == 0)[0], np.where(y_test == 1)[0]
        min_size = min(len(idx_0), len(idx_1))
        X_test = X_test.iloc[np.concatenate([
            resample(idx_0, n_samples=min_size, random_state=42),
            resample(idx_1, n_samples=min_size, random_state=42)
        ])]
        y_test = y_test.iloc[X_test.index]

        # Skalierung & SMOTE
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
        print(f"⚖️ SMOTE: {np.bincount(y_train)} → {np.bincount(y_train_bal)}")

        # Entferne konstante Features
        X_train_df = pd.DataFrame(X_train_bal)
        X_train_df = X_train_df.loc[:, X_train_df.nunique() > 1]
        used_columns = X_train_df.columns
        print(f"🧹 Entfernte konstante Features. Verbleibende Spalten: {len(used_columns)}")

        # Bildumwandlung Training
        X_train_img, _ = to_image_matrix(X_train_df)

        # Testdaten: gleiche Feature-Spalten verwenden
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
        X_test_df = X_test_scaled[used_columns]
        X_test_img, _ = to_image_matrix(X_test_df)

        # Modell bauen & trainieren
        model = build_image_cnn()
        callbacks = [
            EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
            ReduceLROnPlateau(patience=2, factor=0.5)
        ]

        print("🚀 Starte Training ...")
        model.fit(X_train_img, y_train_bal, validation_split=0.2,
                  epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

        # Evaluation
        print("🔍 Evaluiere Modell ...")
        y_pred_prob = model.predict(X_test_img).ravel()
        thresholds = np.linspace(0, 1, 101)
        best_thr, best_f1 = 0.5, 0
        for thr in thresholds:
            y_temp = (y_pred_prob >= thr).astype(int)
            f1_tmp = f1_score(y_test, y_temp, pos_label=0)
            if f1_tmp > best_f1:
                best_f1, best_thr = f1_tmp, thr

        y_pred = (y_pred_prob >= best_thr).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label=0)
        auc_val = auc(fpr, tpr)

        results.append({
            "fold": fold + 1,
            "f1": f1_score(y_test, y_pred, pos_label=0),
            "precision": report["0"]["precision"],
            "recall": report["0"]["recall"],
            "auc": auc_val
        })

        print(f"\n📈 Fold {fold+1} abgeschlossen:")
        print(f"F1: {results[-1]['f1']:.4f}, Precision: {results[-1]['precision']:.4f}, Recall: {results[-1]['recall']:.4f}, AUC: {results[-1]['auc']:.4f}")

        # ROC-Plot
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Fold {fold+1} (AUC = {auc_val:.4f})")
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-Kurve (Anomalie = Klasse 0)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Zusammenfassung
    print("\n📊 Gesamtergebnisse:")
    for r in results:
        print(f"Fold {r['fold']}: F1={r['f1']:.4f}, Precision={r['precision']:.4f}, Recall={r['recall']:.4f}, AUC={r['auc']:.4f}")
    print(f"\nDurchschnitt:")
    print(f"F1: {np.mean([r['f1'] for r in results]):.4f}")
    print(f"Precision: {np.mean([r['precision'] for r in results]):.4f}")
    print(f"Recall: {np.mean([r['recall'] for r in results]):.4f}")
    print(f"AUC: {np.mean([r['auc'] for r in results]):.4f}")

# === Startpunkt
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    supervised_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
    apps_file = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')

    with open(apps_file, 'r') as f:
        app_lines = [line.strip() for line in f.readlines() if line.strip()]

    run_hybrid_pipeline(app_lines, supervised_path,
                        train_indices=[1, 2, 3, 4],
                        test_indices=[1, 2, 3, 4],
                        train_app_count=30,
                        test_app_count=5,
                        epochs=5,
                        batch_size=256,
                        n_folds=3)
