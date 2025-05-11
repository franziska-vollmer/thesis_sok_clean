import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# ==== 1. Alle .pkl-Dateien automatisch laden ====
print("📥 Lade alle .pkl-Dateien ...")
data_dir = "/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp/"
pkl_paths = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))

print(f"📦 Gefundene .pkl-Dateien: {len(pkl_paths)}")
df_list = [pd.read_pickle(p) for p in pkl_paths]
df = pd.concat(df_list, ignore_index=True)

print(f"✅ Gesamtdaten: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

# ==== 2. Vorverarbeitung ====
print("🧹 Bereinige und skaliere Features ...")
features = df.drop(columns=556, errors='ignore')
features = features.apply(pd.to_numeric, errors='coerce')
features = features.dropna(axis=1, how='all').dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features.astype(np.float32))
print(f"ℹ️  Nach Bereinigung: {scaled_data.shape[0]} Zeilen, {scaled_data.shape[1]} Features")

# ==== 3. Sequenzen erzeugen ====
print("🧪 Sequenzbildung (Window Size = 10)")
seq_len = 10
sequences = [
    scaled_data[i:i + seq_len]
    for i in range(0, len(scaled_data) - seq_len + 1, seq_len)
]
sequences = np.stack(sequences)
print(f"🔢 Anzahl Sequenzen: {sequences.shape[0]}")

# ==== 4. Tensoren & DataLoader ====
X = torch.tensor(sequences, dtype=torch.float32)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val), batch_size=64)

# ==== 5. Modell ====
class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, input_dim=556, hidden_dim=64, pooling_type='max'):
        super().__init__()
        self.pooling_type = pooling_type
        self.encoder = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        if self.pooling_type == 'mean':
            pooled = enc_out.mean(dim=1)
        elif self.pooling_type == 'last':
            pooled = enc_out[:, -1, :]
        elif self.pooling_type == 'max':
            pooled, _ = enc_out.max(dim=1)
        else:
            raise ValueError("Invalid pooling type")
        decoder_input = pooled.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(decoder_input)
        return self.output_layer(dec_out)

# ==== 6. Training ====
print("🧠 Starte LSTM Autoencoder Training ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

train_losses, val_losses = [], []
epochs = 10
for epoch in range(epochs):
    print(f"📚 Epoche {epoch+1}/{epochs} ...")
    model.train()
    epoch_train_loss = 0
    for x_batch, in train_loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        recon = model(x_batch)
        loss = criterion(recon, x_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))

    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for x_batch, in val_loader:
            x_batch = x_batch.to(device)
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            epoch_val_loss += loss.item()
    val_losses.append(epoch_val_loss / len(val_loader))
    print(f"📉 Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

print("✅ Training abgeschlossen.")

# ==== 7. Plot Loss ====
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title("LSTM Autoencoder Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()

# ==== 8. Anomalie-Erkennung ====
print("🔍 Berechne Rekonstruktionsfehler für alle Sequenzen ...")
raw_labels = df[556].dropna().astype(int).values
label_seqs = [
    1 if raw_labels[i:i + seq_len].mean() > 0.5 else 0
    for i in range(0, len(raw_labels) - seq_len + 1, seq_len)
]
label_seqs = np.array(label_seqs)

model.eval()
with torch.no_grad():
    recon_all = model(X.to(device))
    recon_errors = torch.mean((recon_all - X.to(device))**2, dim=(1, 2)).cpu().numpy()

# ==== 9. Bestes Threshold suchen ====
print("🧾 Suche bestes Threshold für F1-Score ...")
best_f1 = 0
best_threshold = 0
for t in np.linspace(min(recon_errors), max(recon_errors), 100):
    preds = (recon_errors > t).astype(int)
    f1 = f1_score(label_seqs, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# ==== 10. Auswertung ====
final_preds = (recon_errors > best_threshold).astype(int)
precision = precision_score(label_seqs, final_preds)
recall = recall_score(label_seqs, final_preds)
f1 = f1_score(label_seqs, final_preds)
auc = roc_auc_score(label_seqs, recon_errors)

print("\n=== 📊 Anomalie-Erkennung (Unsupervised) ===")
print(f"📈 Bestes Threshold     : {best_threshold:.6f}")
print(f"🎯 Precision            : {precision:.4f}")
print(f"📡 Recall               : {recall:.4f}")
print(f"✅ F1 Score             : {f1:.4f}")
print(f"🏁 AUC Score            : {auc:.4f}")
