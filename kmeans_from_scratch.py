import numpy as np
import matplotlib.pyplot as plt

class K_Means:
    @staticmethod
    def norme_2(point_1, point_2):
        distance = np.sum((point_2 - point_1) ** 2, axis=1)
        return np.sqrt(distance)

    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        np.random.seed(0)
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(self.max_iter):
            Y = []
            for data_point in X:
                distances = K_Means.norme_2(data_point, self.centroids)
                nb_cluster = np.argmin(distances)
                Y.append(nb_cluster)

            cluster_index = []
            for i in range(self.k):
                cluster_index.append(np.argwhere(np.array(Y) == i).flatten())

            cluster_centers = []
            for indices in cluster_index:
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[0])  # Fallback to a random centroid
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0))

            new_centroids = np.array(cluster_centers)

            if np.max(np.abs(self.centroids - new_centroids)) < 0.0001:
                break
            else:
                self.centroids = new_centroids

        return Y

    def predict(self, X):
        distances = np.array([np.sum((X - centroid) ** 2, axis=1) for centroid in self.centroids])
        return np.argmin(distances, axis=0)

    def test(self, X):
        return self.predict(X)

# Charger les données depuis le fichier CSV
fichier = r"C:\Users\Dell\Documents\MASTER MD4\Mathématique\KNN - K Means\ushape.csv"
data = np.genfromtxt(fichier, delimiter=',')

X = data[:, :2]  # Les deux premières colonnes comme les données d'entrée
labels = data[:, -1]  # La dernière colonne comme les labels

# Créer et entraîner le modèle K-means
kmeans = K_Means(k=3)
predicted_labels = kmeans.fit(X)

# Affichage des résultats
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='*', s=200, label='Centroids')
plt.title('Clustering K-means avec {} clusters'.format(kmeans.k))
plt.legend()
plt.show()
