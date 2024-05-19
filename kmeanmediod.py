import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

# Define the K-Medoids class
class KMedoids:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Initialize medoids randomly
        self.medoids = np.array(list(range(self.n_clusters)))

        # Clustering labels
        self.labels = np.zeros(n_samples)

        for _ in range(self.max_iter):
            # Calculate distances between each point and the medoids
            distances = pairwise_distances(X, X[self.medoids])

            # Assign each point to the nearest medoid
            new_labels = np.argmin(distances, axis=1)

            # Update medoids
            for i in range(self.n_clusters):
                if np.sum(new_labels == i) > 0:
                    self.medoids[i] = np.random.choice(np.where(new_labels == i)[0])

            # Check for convergence
            if np.all(self.labels == new_labels):
                break
            else:
                self.labels = new_labels

        return self

    def predict(self, X):
        distances = pairwise_distances(X, X[self.medoids])
        return np.argmin(distances, axis=1)

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Apply K-Medoids clustering
kmedoids = KMedoids(n_clusters=3)
kmedoids.fit(X)
y_pred = kmedoids.predict(X)

# Visualize the clusters
plt.figure(figsize=(10, 6))

# Plotting the clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=100, c='green', label='Cluster 3')

# Plotting the medoids
plt.scatter(X[kmedoids.medoids, 0], X[kmedoids.medoids, 1], s=300, c='yellow', marker='*', label='Medoids')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Medoids Clustering of Iris Dataset')
plt.legend()
plt.show()