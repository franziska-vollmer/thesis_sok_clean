import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# ==== 1. Daten laden ====
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

# ==== 4. Labels vorbereiten ====
raw_labels = df[556].dropna().astype(int).values
label_seqs = [
    1 if raw_labels[i:i + seq_len].mean() > 0.5 else 0
    for i in range(0, len(raw_labels) - seq_len + 1, seq_len)
]
label_seqs = np.array(label_seqs)

# ==== 5. Daten aufteilen in Training + Test ====
X_temp, X_test, y_temp, y_test = train_test_split(
    sequences, label_seqs, test_size=0.2, stratify=label_seqs, random_state=42
)

X_temp = torch.tensor(X_temp, dtype=torch.float32)
y_temp = torch.tensor(y_temp, dtype=torch.int)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int)

# ==== 6. LSTM Autoencoder ====
class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, input_dim=sequences.shape[2], hidden_dim=64, pooling_type='max'):
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

# ==== 7. 5-Fold Cross-Validation auf Trainingsdaten ====
print("\n🚀 Starte 5-Fold Stratified Cross-Validation auf Trainingsdaten ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
epochs = 10
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_temp, y_temp)):
    print(f"\n🔁 Fold {fold+1}/5")

    X_train, X_val = X_temp[train_idx], X_temp[val_idx]
    y_train, y_val = y_temp[train_idx], y_temp[val_idx]

    train_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=64)

    model = LSTMAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for x_batch, in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x_batch, in val_loader:
                x_batch = x_batch.to(device)
                recon = model(x_batch)
                val_loss += criterion(recon, x_batch).item()
        print(f"📉 Fold {fold+1} - Epoch {epoch+1} - Validation Loss: {val_loss / len(val_loader):.4f}")

    # Evaluation auf Validierungsdaten
    model.eval()
    with torch.no_grad():
        recon_val = model(X_val.to(device))
        recon_errors = torch.mean((recon_val - X_val.to(device))**2, dim=(1, 2)).cpu().numpy()

    y_val_np = y_val.numpy()
    best_f1, best_threshold = 0, 0
    for t in np.linspace(min(recon_errors), max(recon_errors), 100):
        preds = (recon_errors > t).astype(int)
        f1 = f1_score(y_val_np, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    final_preds = (recon_errors > best_threshold).astype(int)
    precision = precision_score(y_val_np, final_preds)
    recall = recall_score(y_val_np, final_preds)
    f1 = f1_score(y_val_np, final_preds)
    auc = roc_auc_score(y_val_np, recon_errors)
    pr_auc = average_precision_score(y_val_np, recon_errors)

    fold_results.append({
        'fold': fold + 1,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'pr_auc': pr_auc,
        'model': model.state_dict(),  # Optional: Modell speichern
        'threshold': best_threshold
    })

# ==== 8. Evaluation auf unbekannten Testdaten ====
print("\n🧪 Evaluation auf vollständig unbekannten Testdaten ...")
model.load_state_dict(fold_results[0]['model'])  # oder den besten Fold nehmen
model.eval()
with torch.no_grad():
    recon_test = model(X_test.to(device))
    recon_errors_test = torch.mean((recon_test - X_test.to(device))**2, dim=(1, 2)).cpu().numpy()

y_test_np = y_test.numpy()
best_threshold_test = fold_results[0]['threshold']

final_preds_test = (recon_errors_test > best_threshold_test).astype(int)
precision = precision_score(y_test_np, final_preds_test)
recall = recall_score(y_test_np, final_preds_test)
f1 = f1_score(y_test_np, final_preds_test)
auc = roc_auc_score(y_test_np, recon_errors_test)
pr_auc = average_precision_score(y_test_np, recon_errors_test)

print(f"📦 Testdaten-Ergebnisse — Precision: {precision:.4f}, Recall: {recall:.4f}, "
      f"F1: {f1:.4f}, AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
