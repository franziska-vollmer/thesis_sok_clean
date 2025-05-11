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

# ==== 4. Labels auf Sequenzebene ====
raw_labels = df[556].dropna().astype(int).values
label_seqs = [
    1 if raw_labels[i:i + seq_len].mean() > 0.5 else 0
    for i in range(0, len(raw_labels) - seq_len + 1, seq_len)
]
label_seqs = np.array(label_seqs)

# ==== 5. Split in Training + finalen Test ====
X_all = torch.tensor(sequences, dtype=torch.float32)
y_all = torch.tensor(label_seqs, dtype=torch.int)

X_train_all, X_test_final, y_train_all, y_test_final = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

# ==== 6. Modell ====
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

# ==== 7. Cross-Validation ====
print("\n🚀 Starte 5-Fold Stratified Cross-Validation ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
epochs = 10
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train_all)):
    print(f"\n🔁 Fold {fold+1}/5")
    X_train, X_val = X_train_all[train_idx], X_train_all[val_idx]
    y_train, y_val = y_train_all[train_idx], y_train_all[val_idx]

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

    print(f"📈 Fold {fold+1} — Precision: {precision:.4f}, Recall: {recall:.4f}, "
          f"F1: {f1:.4f}, AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")

    fold_results.append({
        'fold': fold + 1,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'pr_auc': pr_auc
    })

# ==== 8. Durchschnitt über Folds ====
print("\n📊 Ergebnisse pro Fold:")
for res in fold_results:
    print(f"Fold {res['fold']}: "
          f"Precision={res['precision']:.4f}, Recall={res['recall']:.4f}, "
          f"F1={res['f1']:.4f}, AUC={res['auc']:.4f}, PR-AUC={res['pr_auc']:.4f}")

avg = lambda key: np.mean([res[key] for res in fold_results])
print("\n📈 Durchschnitt über alle Folds:")
print(f"🎯 Precision : {avg('precision'):.4f}")
print(f"📡 Recall    : {avg('recall'):.4f}")
print(f"✅ F1 Score  : {avg('f1'):.4f}")
print(f"🏁 AUC Score : {avg('auc'):.4f}")
print(f"📐 PR-AUC    : {avg('pr_auc'):.4f}")

# ==== 9. Training auf gesamten Trainingsdaten für finalen Test ====
print("\n🧪 Trainiere finales Modell auf allen Trainingsdaten ...")
final_model = LSTMAutoencoder().to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
train_loader = DataLoader(TensorDataset(X_train_all), batch_size=64, shuffle=True)

for epoch in range(epochs):
    final_model.train()
    for x_batch, in train_loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        recon = final_model(x_batch)
        loss = criterion(recon, x_batch)
        loss.backward()
        optimizer.step()

# ==== 10. Evaluation auf dem Testset ====
final_model.eval()
with torch.no_grad():
    recon_test = final_model(X_test_final.to(device))
    recon_errors_test = torch.mean((recon_test - X_test_final.to(device))**2, dim=(1, 2)).cpu().numpy()

# Threshold vom gesamten Trainingsset bestimmen
with torch.no_grad():
    recon_train_all = final_model(X_train_all.to(device))
    re_errors_train = torch.mean((recon_train_all - X_train_all.to(device))**2, dim=(1, 2)).cpu().numpy()

best_f1, best_threshold = 0, 0
y_train_np = y_train_all.numpy()
for t in np.linspace(min(re_errors_train), max(re_errors_train), 100):
    preds = (re_errors_train > t).astype(int)
    f1 = f1_score(y_train_np, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# Finaler Test
y_test_np = y_test_final.numpy()
final_preds = (recon_errors_test > best_threshold).astype(int)

precision = precision_score(y_test_np, final_preds)
recall = recall_score(y_test_np, final_preds)
f1 = f1_score(y_test_np, final_preds)
auc = roc_auc_score(y_test_np, recon_errors_test)
pr_auc = average_precision_score(y_test_np, recon_errors_test)

print("\n🏁 📊 Finale Testset-Ergebnisse (auf ungesehenen Daten):")
print(f"🎯 Precision : {precision:.4f}")
print(f"📡 Recall    : {recall:.4f}")
print(f"✅ F1 Score  : {f1:.4f}")
print(f"🏁 AUC Score : {auc:.4f}")
print(f"📐 PR-AUC    : {pr_auc:.4f}")
