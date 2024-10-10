import numpy as np
class MyKMeans:
    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0],self.n_clusters,replace=False)]
        for _ in range(self.max_iters):
            labels = self._assign_labels(X)
            new_controids = self._update_centroids(X, labels)
            if np.all(self.centroids == new_controids):
                break
            self.centroids = new_controids
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
    def _assign_labels(self,X):
        distance = np.linalg.norm(X[:,np.newaxis] - self.centroids, axis=2)
        return np.argmin(distance, axis=1)
    def _update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0)for i in range(self.n_clusters)])
        return new_centroids