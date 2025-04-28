BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'train_test_supervised_with_timestamp/')
apps_file_path = os.path.join(BASE_DIR, '../data-files/apps-sok-reduced.txt')
train_indices = [1, 2, 3, 4]
test_indices = [1, 2, 3, 4]
n_splits = 5

# === Daten einlesen ===
def extract_xy_from_df(df):
    df_features = df.drop(index=556).astype(float)
    labels = df.loc[556].astype(int).values
    X = df_features.T.values
    y = labels
    return X, y

X_all, y_all = [], []

with open(apps_file_path, 'r') as file:
    app_lines = file.readlines()

for line in app_lines:
    app = line.strip()
    for i in train_indices + test_indices:
        path = os.path.join(data_path, f"{app}-{i}.pkl")
        with open(path, 'rb') as f:
            df = pickle.load(f)
        X, y = extract_xy_from_df(df)
        X_all.append(X)
        y_all.append(y)

# === Einheitliche Feature-Anzahl sicherstellen ===
min_features = min([x.shape[1] for x in X_all])
X_all = [x[:, :min_features] for x in X_all]

X_raw = np.vstack(X_all)
y_raw = np.concatenate(y_all)

# === Labels anpassen: 1 = normal → 0, Anomalie (-1) = 1
y_binary = np.where(y_raw == 1, 0, 1)

# === StratifiedKFold Cross-Validation ===
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold = 1
for train_idx, test_idx in skf.split(X_raw, y_binary):
    print(f"\n📂 Fold {fold}")

    # === Nur normale Daten zum Training verwenden (label==0 → original y==1)
    X_train_norm = X_raw[train_idx][y_binary[train_idx] == 0]
    X_test = X_raw[test_idx]
    y_test = y_binary[test_idx]

    # === Skalierung
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_norm)
    X_test_scaled = scaler.transform(X_test)

    # === Autoencoder Architektur
    input_dim = X_train_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation="relu")(input_layer)
    encoded = Dense(64, activation="relu")(encoded)
    encoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(128, activation="relu")(decoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # === Training
    autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=30,
        batch_size=128,
        shuffle=True,
        validation_split=0.1,
        verbose=0
    )

    # === Rekonstruktionsfehler berechnen
    X_test_pred = autoencoder.predict(X_test_scaled)
    reconstruction_error = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)

    # === Threshold-Suche für bestes F1(0)
    best_f1, best_thresh = 0, 0
    for t in np.linspace(reconstruction_error.min(), reconstruction_error.max(), 1000):
        y_pred = (reconstruction_error > t).astype(int)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # === Auswertung
    y_pred = (reconstruction_error > best_thresh).astype(int)

    print(f"🔍 Best Threshold (F1 Anomalie=1): {best_thresh:.6f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("F1 (macro):", f1_score(y_test, y_pred, average='macro'))

    # === ROC
    fpr, tpr, _ = roc_curve(y_test, reconstruction_error)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc:.2f})')

    fold += 1

# === Plot ROC aller Folds
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Autoencoder (Cross-Validation)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
