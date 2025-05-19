# src/visualize.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE

def plot_umap(df_embeddings: pd.DataFrame, labels=None, title="UMAP Projection"):
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(df_embeddings)

    plt.figure(figsize=(6,5))
    
    # Make sure labels are valid and numeric
    if labels is not None and pd.api.types.is_numeric_dtype(labels):
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', edgecolor='k')
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], color='grey', edgecolor='k')
        print("⚠️ Labels for coloring UMAP plot are missing or non-numeric. Using default color.")

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_tsne(df_embeddings: pd.DataFrame, labels=None, title="t-SNE Projection"):
    print("test")
    n_samples = df_embeddings.shape[0]
    # Perplexity must be less than n_samples / 3
    perplexity = max(1, min(2, (n_samples - 1) // 3))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(df_embeddings)

    plt.figure(figsize=(6,5))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', edgecolor='k')
    plt.title(title + f" (perplexity={perplexity})")
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")
    plt.grid(True)
    plt.show()

def plot_cluster_distribution(df_with_clusters: pd.DataFrame):
    sns.countplot(x='Cluster', data=df_with_clusters, palette='pastel')
    plt.title("Customer Distribution by Cluster")
    plt.xlabel("Cluster Label")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

def plot_radar_chart(df: pd.DataFrame, cluster_col: str, numeric_cols: list):
    """
    Plots a radar chart for each cluster based on average values of selected numeric columns.
    """
    clusters = df[cluster_col].unique()
    num_vars = len(numeric_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the loop
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for cluster in clusters:
        avg_values = df[df[cluster_col] == cluster][numeric_cols].mean().tolist()
        avg_values += avg_values[:1]
        ax.plot(angles, avg_values, label=f'Cluster {cluster}', linewidth=2)
        ax.fill(angles, avg_values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Labels
    ax.set_thetagrids(np.degrees(angles[:-1]), numeric_cols)
    ax.set_title("Cluster Personas Radar Chart", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()
