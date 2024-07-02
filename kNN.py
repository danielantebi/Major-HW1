from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial.distance import cdist
import numpy as np


class kNN(BaseEstimator, ClassifierMixin):  # Set number of neighbors
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):  # Copies the training data to instance variables
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)
        return self

    def predict(self, X):
        # Compute distances between X and self.X_train
        distances = cdist(X, self.X_train, metric='euclidean')

        # Get indices of the nearest neighbors
        nearest_neighbors_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]

        # Retrieve the labels of the nearest neighbors
        nearest_neighbors_labels = self.y_train[nearest_neighbors_indices]

        # Majority vote: for each set of nearest neighbors, compute the most frequent label
        predictions = np.sign(np.sum(nearest_neighbors_labels, axis=1))

        return predictions
