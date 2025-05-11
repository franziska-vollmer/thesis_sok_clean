import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# === Konfiguration ===import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# === Konfiguration ===
timesteps = 10
latent_dim = 64
manual_threshold_value = 0.0010
use_manual_threshold = True
data_dir = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
apps = ["CVE-2012-1823", "CVE-2014-0050", "CVE-2014-0160", "CVE-2014-3120", "CVE-2014-6271",
    "CVE-2015-1427", "CVE-2015-2208", "CVE-2015-3306", "CVE-2015-5477", "CVE-2015-5531",
    "CVE-2015-8103", "CVE-2015-8562", "CVE-2016-3088", "CVE-2016-3714", "CVE-2016-6515",
    "CVE-2016-7434", "CVE-2016-9920", "CVE-2016-10033", "CVE-2017-5638", "CVE-2017-7494",
    "CVE-2017-7529", "CVE-2017-8291", "CVE-2017-8917", "CVE-2017-11610", "CVE-2017-12149",
    "CVE-2017-12615", "CVE-2017-12635", "CVE-2017-12794", "CVE-2018-11776", "CVE-2018-15473",
    "CVE-2018-16509", "CVE-2018-19475"]  # ← anpassen

# === Fortschrittsanzeige ===
class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.pbar = tqdm(total=self.epochs, desc='Training', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        log_str = f"Loss: {logs.get('loss'):.4f}, Val Loss: {logs.get('val_loss'):.4f}"
        self.pbar.set_postfix_str(log_str)
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()

# === Daten laden ===
def load_data(apps, indices=[1, 2, 3, 4]):
    print("📥 Lade Daten (letzte Spalte = Label)...")
    X_all, y_all = [], []
    min_samples = None
    min_features = None

    for app in apps:
        for i in indices:
            path = os.path.join(data_dir, f'{app}-{i}.pkl')
            if not os.path.exists(path):
                print(f"⚠️  Fehlende Datei: {path}")
                continue
            with open(path, 'rb') as f:
                df = pickle.load(f)

            df = df.astype(float)
            features = df.iloc[:, :-1].T.values  # alle Spalten außer letzte = Features
            labels = df.iloc[:, -1].values       # letzte Spalte = Label

            n_samples = min(features.shape[0], labels.shape[0])
            n_features = features.shape[1]

            if min_samples is None:
                min_samples = n_samples
            else:
                min_samples = min(min_samples, n_samples)

            if min_features is None:
                min_features = n_features
            else:
                min_features = min(min_features, n_features)

            X_all.append(features)
            y_all.append(labels)

    if not X_all:
        raise ValueError("❌ Keine Daten geladen.")

    print(f"✂️  Kürze alle Arrays auf {min_samples} Samples und {min_features} Features.")
    X_all = [x[:min_samples, :min_features] for x in X_all]
    y_all = [y[:min_samples] for y in y_all]

    return np.vstack(X_all), np.concatenate(y_all)

# === Hauptablauf ===
print("🚀 Starte LSTM-Autoencoder")
X_raw, y = load_data(apps)

print("📊 Skaliere Daten...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("🔁 Erstelle Sequenzen...")
n_samples = len(X_scaled) // timesteps
X_seq = X_scaled[:n_samples * timesteps].reshape(n_samples, timesteps, -1)
y_seq = y[:n_samples * timesteps].reshape(n_samples, timesteps)
y_true_majority = (np.mean(y_seq, axis=1) > 0.5).astype(int)  # noch mit -1 möglich

print(f"🔍 Anzahl Sequenzen: {n_samples}")

print("🧠 Baue LSTM-Autoencoder-Modell...")
input_shape = (timesteps, X_seq.shape[2])
inputs = Input(shape=input_shape)
encoded = LSTM(latent_dim)(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_shape[1], return_sequences=True)(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print("🎯 Trainiere Modell (unsupervised auf alle Daten)...")
autoencoder.fit(
    X_seq, X_seq,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    shuffle=True,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True), TQDMProgressBar()],
    verbose=0
)

print("📈 Berechne Rekonstruktionsfehler...")
X_pred = autoencoder.predict(X_seq)
recon_error = np.mean(np.square(X_seq - X_pred), axis=(1, 2))

# === Schwellenwahl
print("🔧 Wähle Anomalieschwelle...")
if use_manual_threshold:
    best_thresh = manual_threshold_value
    print(f"✅ Manuell gesetzte Schwelle: {best_thresh}")
else:
    thresholds = np.linspace(min(recon_error), max(recon_error), 100)
    f1_scores = [f1_score(y_true_majority, recon_error > t) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"🔎 Beste F1-Schwelle: {best_thresh:.6f}")

# === Vorhersage
y_pred = (recon_error > best_thresh).astype(int)

# === Labels normalisieren (-1 → 0)
y_true_majority = np.where(y_true_majority == -1, 0, 1)
unique_classes = np.unique(y_true_majority)

# === Bewertung
print("\n📊 Bewertung:")
print(f"F1-Score:   {f1_score(y_true_majority, y_pred, zero_division=0):.4f}")
print(f"Precision:  {precision_score(y_true_majority, y_pred, zero_division=0):.4f}")
print(f"Recall:     {recall_score(y_true_majority, y_pred, zero_division=0):.4f}")

if len(unique_classes) == 2:
    auc_score = roc_auc_score(y_true_majority, recon_error)
    print(f"ROC-AUC:    {auc_score:.4f}")
else:
    print("⚠️ ROC-AUC nicht berechenbar: Nur eine Klasse im Label vorhanden.")

# === ROC-Kurve
if len(unique_classes) == 2:
    fpr, tpr, _ = roc_curve(y_true_majority, recon_error)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.axvline(x=best_thresh, color='gray', linestyle='--', label='Schwelle')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC – LSTM Autoencoder')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

timesteps = 10
latent_dim = 64
manual_threshold_value = 0.0010
use_manual_threshold = True
data_dir = '/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/'
apps = ["CVE-2012-1823", "CVE-2014-0050", "CVE-2014-0160", "CVE-2014-3120", "CVE-2014-6271",
    "CVE-2015-1427", "CVE-2015-2208", "CVE-2015-3306", "CVE-2015-5477", "CVE-2015-5531",
    "CVE-2015-8103", "CVE-2015-8562", "CVE-2016-3088", "CVE-2016-3714", "CVE-2016-6515",
    "CVE-2016-7434", "CVE-2016-9920", "CVE-2016-10033", "CVE-2017-5638", "CVE-2017-7494",
    "CVE-2017-7529", "CVE-2017-8291", "CVE-2017-8917", "CVE-2017-11610", "CVE-2017-12149",
    "CVE-2017-12615", "CVE-2017-12635", "CVE-2017-12794", "CVE-2018-11776", "CVE-2018-15473",
    "CVE-2018-16509", "CVE-2018-19475"]  # ← anpassen

# === Fortschrittsanzeige ===
class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.pbar = tqdm(total=self.epochs, desc='Training', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        log_str = f"Loss: {logs.get('loss'):.4f}, Val Loss: {logs.get('val_loss'):.4f}"
        self.pbar.set_postfix_str(log_str)
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()

# === Daten laden ===
def load_data(apps, indices=[1, 2, 3, 4]):
    print("📥 Lade Daten (letzte Spalte = Label)...")
    X_all, y_all = [], []
    min_samples = None
    min_features = None

    for app in apps:
        for i in indices:
            path = os.path.join(data_dir, f'{app}-{i}.pkl')
            if not os.path.exists(path):
                print(f"⚠️  Fehlende Datei: {path}")
                continue
            with open(path, 'rb') as f:
                df = pickle.load(f)

            df = df.astype(float)
            features = df.iloc[:, :-1].T.values  # alles außer letzte Spalte
            labels = df.iloc[:, -1].values       # letzte Spalte

            n_samples = min(features.shape[0], labels.shape[0])
            n_features = features.shape[1]

            if min_samples is None:
                min_samples = n_samples
            else:
                min_samples = min(min_samples, n_samples)

            if min_features is None:
                min_features = n_features
            else:
                min_features = min(min_features, n_features)

            X_all.append(features)
            y_all.append(labels)

    if not X_all:
        raise ValueError("❌ Keine Daten geladen.")

    print(f"✂️  Kürze alle Arrays auf {min_samples} Samples und {min_features} Features.")
    X_all = [x[:min_samples, :min_features] for x in X_all]
    y_all = [y[:min_samples] for y in y_all]

    return np.vstack(X_all), np.concatenate(y_all)

# === Hauptablauf ===
print("🚀 Starte LSTM-Autoencoder")
X_raw, y = load_data(apps)

print("📊 Skaliere Daten...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("🔁 Erstelle Sequenzen...")
n_samples = len(X_scaled) // timesteps
X_seq = X_scaled[:n_samples * timesteps].reshape(n_samples, timesteps, -1)
y_seq = y[:n_samples * timesteps].reshape(n_samples, timesteps)
y_true_majority = (np.mean(y_seq, axis=1) > 0.5).astype(int)

print(f"🔍 Anzahl Sequenzen: {n_samples}, davon {np.sum(y_true_majority==1)} normal, {np.sum(y_true_majority==0)} anomal")

print("🧠 Baue LSTM-Autoencoder-Modell...")
input_shape = (timesteps, X_seq.shape[2])
inputs = Input(shape=input_shape)
encoded = LSTM(latent_dim)(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_shape[1], return_sequences=True)(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print("🎯 Trainiere Modell (unsupervised auf alle Daten)...")
autoencoder.fit(
    X_seq, X_seq,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    shuffle=True,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True), TQDMProgressBar()],
    verbose=0
)

print("📈 Berechne Rekonstruktionsfehler...")
X_pred = autoencoder.predict(X_seq)
recon_error = np.mean(np.square(X_seq - X_pred), axis=(1, 2))

# === Schwellenwahl ===
print("🔧 Wähle Anomalieschwelle...")
if use_manual_threshold:
    best_thresh = manual_threshold_value
    print(f"✅ Manuell gesetzte Schwelle: {best_thresh}")
else:
    thresholds = np.linspace(min(recon_error), max(recon_error), 100)
    f1_scores = [f1_score(y_true_majority, recon_error > t) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"🔎 Beste F1-Schwelle: {best_thresh:.6f}")

y_pred = (recon_error > best_thresh).astype(int)

# === Bewertung ===
print("\n📊 Bewertung:")
print(f"F1-Score:   {f1_score(y_true_majority, y_pred):.4f}")
print(f"Precision:  {precision_score(y_true_majority, y_pred):.4f}")
print(f"Recall:     {recall_score(y_true_majority, y_pred):.4f}")
print(f"ROC-AUC:    {roc_auc_score(y_true_majority, recon_error):.4f}")

# === ROC-Kurve ===
fpr, tpr, _ = roc_curve(y_true_majority, recon_error)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='ROC Curve')
plt.axvline(x=best_thresh, color='gray', linestyle='--', label='Schwelle')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC – LSTM Autoencoder')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
