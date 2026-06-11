# File: C:\Users\Arafat\Desktop\IMA Project\src\knn.py

"""
In this file I want to implement K-Nearest Neighbors from scratch,
and also provide a small wrapper around sklearn's KNN so I can compare.
"""

from typing import Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class MyKNNClassifier:
    """
    In this class I want to write my own simple KNN classifier.
    I will store the training feature matrix and labels and then,
    for each test sample, I will compute distances and pick the k nearest.
    """

    def __init__(self, k_value: int = 5, distance_metric: str = "l2", use_distance_weight: bool = False):
        self.k_value = k_value
        self.distance_metric = distance_metric  # "l2" or "l1"
        self.use_distance_weight = use_distance_weight
        self.train_features = None
        self.train_labels = None
        self.unique_labels = None

    def fit(self, feature_matrix: np.ndarray, label_array: np.ndarray):
        """
        In this function I want to "train" KNN, which just means
        remembering the training features and labels.
        """
        self.train_features = feature_matrix.astype(np.float32)
        self.train_labels = np.array(label_array)
        self.unique_labels = np.unique(self.train_labels)
        print("[MyKNN] Stored training data with shape:", self.train_features.shape)

    def _compute_distance(self, sample_vec: np.ndarray) -> np.ndarray:
        """
        In this function I want to compute distances from one test sample
        to all training samples.
        """
        if self.distance_metric == "l1":
            temp_distances = np.sum(np.abs(self.train_features - sample_vec), axis=1)
        else:  # default l2
            diff = self.train_features - sample_vec
            temp_distances = np.sqrt(np.sum(diff * diff, axis=1))
        return temp_distances

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        In this function I want to predict labels for a whole feature matrix.
        """
        num_samples = feature_matrix.shape[0]
        pred_labels = []

        for idx in range(num_samples):
            sample_vec = feature_matrix[idx, :]
            temp_distances = self._compute_distance(sample_vec)

            # get indices of k smallest distances
            k_indices = np.argpartition(temp_distances, self.k_value)[: self.k_value]
            k_labels = self.train_labels[k_indices]
            k_distances = temp_distances[k_indices]

            if self.use_distance_weight:
                # smaller distance => bigger weight, I add a small epsilon to avoid division by zero
                weights = 1.0 / (k_distances + 1e-6)
                label_scores = {}
                for lab, w in zip(k_labels, weights):
                    label_scores[lab] = label_scores.get(lab, 0.0) + w
                best_label = max(label_scores.items(), key=lambda item: item[1])[0]
            else:
                # simple majority voting
                values, counts = np.unique(k_labels, return_counts=True)
                best_label = values[np.argmax(counts)]

            pred_labels.append(best_label)

            if (idx + 1) % 50 == 0 or (idx + 1) == num_samples:
                print(f"[MyKNN] Predicted {idx + 1}/{num_samples} samples")

        return np.array(pred_labels)


class SklearnKNNWrapper:
    """
    In this class I want a thin wrapper around sklearn KNeighborsClassifier,
    mainly so I can call it in a similar way to MyKNNClassifier and compare.
    """

    def __init__(
        self,
        k_value: int = 5,
        distance_metric: str = "minkowski",
        weights: str = "uniform",
    ):
        self.k_value = k_value
        self.distance_metric = distance_metric
        self.weights = weights
        self.model: Optional[KNeighborsClassifier] = None

    def fit(self, feature_matrix: np.ndarray, label_array: np.ndarray):
        """
        In this function I want to fit sklearn's KNN.
        """
        self.model = KNeighborsClassifier(
            n_neighbors=self.k_value,
            metric=self.distance_metric,
            weights=self.weights,
        )
        self.model.fit(feature_matrix, label_array)
        print("[SklearnKNNWrapper] Fitted KNN with k =", self.k_value)

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        In this function I want to call sklearn's predict.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = self.model.predict(feature_matrix)
        return preds
