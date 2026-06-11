# File: C:\Users\Arafat\Desktop\IMA Project\src\visualization.py

"""
In this file I want to put all plotting functions.
I will use matplotlib and seaborn to show:
- sample images
- class distributions
- confusion matrices
- accuracy vs k plots
- PCA / t-SNE visualizations of feature spaces
"""

from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage import io

from . import config


def show_sample_images(image_paths: List[str], labels: List[str], num_cols: int = 5, fig_title: str = ""):
    """
    In this function I want to display a grid of sample images with their labels.
    """
    num_images = len(image_paths)
    num_rows = int(np.ceil(num_images / num_cols))

    plt.figure(figsize=(3 * num_cols, 3 * num_rows))
    for idx, (img_path, lab) in enumerate(zip(image_paths, labels)):
        temp_img = io.imread(img_path)
        plt.subplot(num_rows, num_cols, idx + 1)
        plt.imshow(temp_img)
        plt.axis("off")
        plt.title(str(lab))
    plt.suptitle(fig_title)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(df, label_column: str = "label", title: str = "Class distribution"):
    """
    In this function I want to plot how many samples each class has.
    """
    plt.figure(figsize=(8, 4))
    sns.countplot(x=label_column, data=df)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str = "Confusion Matrix"):
    """
    In this function I want to show the confusion matrix as a heatmap.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_k(df_results: pd.DataFrame, title: str = "Accuracy vs k"):
    """
    In this function I want to plot accuracy as a function of k.
    I assume df_results has columns 'k_value' and 'accuracy'.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(df_results["k_value"], df_results["accuracy"], marker="o")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Validation accuracy")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(features: np.ndarray, labels: np.ndarray, title: str = "PCA 2D projection"):
    """
    In this function I want to use PCA to reduce features to 2D and
    then plot them colored by label, to see how separable classes are.
    """
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(scatter, label="Class (encoded)")
    plt.tight_layout()
    plt.show()


def plot_tsne_scatter(features: np.ndarray, labels: np.ndarray, title: str = "t-SNE 2D projection"):
    """
    In this function I want to use t-SNE to reduce features to 2D and
    then plot them colored by label. This is more non-linear than PCA.
    """
    tsne = TSNE(n_components=2, init="random", learning_rate="auto")
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.colorbar(scatter, label="Class (encoded)")
    plt.tight_layout()
    plt.show()
