# File: C:\Users\Arafat\Desktop\IMA Project\src\evaluation.py

"""
In this file I want to compute and print metrics for my KNN models.
I also want a helper to run KNN for different k values and collect results.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def compute_basic_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """
    In this function I want to compute accuracy, macro precision, macro recall,
    and macro F1 score.
    """
    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)
    metrics_dict = {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }
    return metrics_dict


def compute_confusion(true_labels: np.ndarray, pred_labels: np.ndarray, label_order: List[str]) -> np.ndarray:
    """
    In this function I want to compute the confusion matrix with a fixed
    ordering of labels so it's consistent for plots.
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=label_order)
    return cm


def print_detailed_classification_report(true_labels: np.ndarray, pred_labels: np.ndarray):
    """
    In this function I want to print sklearn's classification report,
    which shows precision, recall, and F1 per class.
    """
    print("=== Classification Report ===")
    print(classification_report(true_labels, pred_labels, zero_division=0))


def evaluate_knn_for_k_values(
    knn_class,
    feature_train: np.ndarray,
    label_train: np.ndarray,
    feature_val: np.ndarray,
    label_val: np.ndarray,
    k_list: List[int],
) -> pd.DataFrame:
    """
    In this function I want to try different k values for a given KNN class
    (like MyKNNClassifier or SklearnKNNWrapper) and return a DataFrame
    with the results.
    """
    results_list = []

    for k_value in k_list:
        print(f"[evaluation] Testing k = {k_value}")
        model = knn_class(k_value=k_value)
        model.fit(feature_train, label_train)
        pred_val = model.predict(feature_val)
        metrics_dict = compute_basic_metrics(label_val, pred_val)
        metrics_dict["k_value"] = k_value
        results_list.append(metrics_dict)

    df_results = pd.DataFrame(results_list)
    print("[evaluation] Results for different k:")
    print(df_results)
    return df_results
