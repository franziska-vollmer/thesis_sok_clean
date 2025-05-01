# pso_svm_full.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# --- Binary PSO for Feature Selection ---
class BPSOFeatureSelector:
    def __init__(self, num_particles=10, max_iter=5, max_features=30, min_features=5):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.max_features = max_features
        self.min_features = min_features

    def _enforce_feature_limits(self, particle):
        num_selected = np.sum(particle)
        if num_selected > self.max_features:
            # Keep only top-N active bits
            active_indices = np.where(particle == 1)[0]
            np.random.shuffle(active_indices)
            particle[:] = 0
            particle[active_indices[:self.max_features]] = 1
        elif num_selected < self.min_features:
            inactive_indices = np.where(particle == 0)[0]
            np.random.shuffle(inactive_indices)
            particle[inactive_indices[:self.min_features - num_selected]] = 1
        return particle

    def select_features(self, X, y):
        num_features = X.shape[1]
        particles = np.random.randint(0, 2, size=(self.num_particles, num_features))
        for i in range(self.num_particles):
            particles[i] = self._enforce_feature_limits(particles[i])

        velocities = np.zeros((self.num_particles, num_features))
        pbest = particles.copy()
        gbest = particles[0].copy()

        def fitness(particle):
            selected_indices = particle == 1
            num_selected = np.sum(selected_indices)
            print(f"[Fitness] Features gewählt: {num_selected}")
            selected_X = X[:, selected_indices]
            try:
                clf = SVC(kernel='rbf', gamma='scale')
                clf.fit(selected_X, y)
                return clf.score(selected_X, y)
            except Exception:
                return 0

        pbest_scores = np.array([fitness(p) for p in particles])
        gbest_score = np.max(pbest_scores)
        gbest = pbest[np.argmax(pbest_scores)].copy()

        for t in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = 0.7 * velocities[i] + 0.8 * r1 * (pbest[i] - particles[i]) + 0.9 * r2 * (gbest - particles[i])
                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                new_particle = np.where(np.random.rand(num_features) < sigmoid, 1, 0)
                new_particle = self._enforce_feature_limits(new_particle)
                score = fitness(new_particle)
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = new_particle.copy()
                    if score > gbest_score:
                        gbest_score = score
                        gbest = new_particle.copy()
                particles[i] = new_particle
            print(f"[BPSO] Iteration {t+1}/{self.max_iter} - Best Fitness: {gbest_score:.4f}")

        return gbest

# --- Standard PSO for SVM Parameter Tuning ---
class SPSOOptimizer:
    def __init__(self, num_particles=20, max_iter=30):
        self.num_particles = num_particles
        self.max_iter = max_iter

    def optimize(self, X, y):
        particles = np.random.rand(self.num_particles, 2)
        particles[:, 0] *= 1000  # C range
        particles[:, 1] *= 10    # gamma range
        velocities = np.zeros_like(particles)
        pbest = particles.copy()

        def fitness(p):
            clf = SVC(C=p[0], gamma=p[1])
            clf.fit(X, y)
            return clf.score(X, y)

        pbest_scores = np.array([fitness(p) for p in particles])
        gbest_idx = np.argmax(pbest_scores)
        gbest = particles[gbest_idx].copy()

        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = 0.7 * velocities[i] + 0.8 * r1 * (pbest[i] - particles[i]) + 0.9 * r2 * (gbest - particles[i])
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], [1, 0.01], [1000, 10])
                score = fitness(particles[i])
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = particles[i].copy()
                    if score > fitness(gbest):
                        gbest = particles[i].copy()

        return {'C': gbest[0], 'gamma': gbest[1]}

# --- Load and preprocess dataset ---
def load_data_from_directory(directory="/home/franziska/sok-utsa-tuda-evalutation-of-docker-containers/algorithms/train_test_supervised_with_timestamp"):
    all_X, all_y = [], []
    for fname in os.listdir(directory):
        if fname.endswith(".pkl"):
            df = pd.read_pickle(os.path.join(directory, fname))
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            all_X.append(X)
            all_y.append(y)
    X = np.vstack(all_X)
    y = np.hstack(all_y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Main workflow ---
if __name__ == '__main__':
    print("Lade und verarbeite Daten...")
    X_train, X_test, y_train, y_test = load_data_from_directory()

    print("Wähle Merkmale mit BPSO aus...")
    fs = BPSOFeatureSelector(num_particles=10, max_iter=5, max_features=30, min_features=5)
    selected_mask = fs.select_features(X_train, y_train)
    X_train_sel = X_train[:, selected_mask == 1]
    X_test_sel = X_test[:, selected_mask == 1]

    print("Optimiere SVM Parameter mit SPSO...")
    opt = SPSOOptimizer(num_particles=20, max_iter=30)
    best_params = opt.optimize(X_train_sel, y_train)

    print(f"Trainiere SVM mit C={best_params['C']:.2f}, gamma={best_params['gamma']:.4f}...")
    clf = SVC(C=best_params['C'], gamma=best_params['gamma'])
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)

    print("Evaluierung:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))