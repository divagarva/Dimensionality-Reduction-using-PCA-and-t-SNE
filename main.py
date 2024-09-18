# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# Load dataset (Iris dataset for demonstration purposes)
def load_data():
    iris = load_iris()
    data = iris.data
    target = iris.target
    feature_names = iris.feature_names
    return data, target, feature_names


# Apply PCA for dimensionality reduction
def apply_pca(data, n_components=2):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_

    print(f"PCA Explained Variance (first {n_components} components): {explained_variance}")
    return pca_result


# Apply t-SNE for dimensionality reduction
def apply_tsne(data, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_result = tsne.fit_transform(data)
    return tsne_result


# Plot results
def plot_result(data, target, method, feature_names=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=target, cmap='viridis', s=50, alpha=0.7)
    plt.title(f"{method} Dimensionality Reduction")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    legend = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend)
    plt.show()


# Main function
def main():
    # Load the dataset
    data, target, feature_names = load_data()

    # Apply PCA
    pca_result = apply_pca(data)
    plot_result(pca_result, target, "PCA")

    # Apply t-SNE
    tsne_result = apply_tsne(data)
    plot_result(tsne_result, target, "t-SNE")


if __name__ == "__main__":
    main()