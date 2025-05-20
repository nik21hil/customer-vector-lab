# src/visualize.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE

def plot_umap(df_embeddings: pd.DataFrame, labels=None, title="UMAP Projection"):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import umap

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(df_embeddings)

    fig, ax = plt.subplots(figsize=(6,5))

    if labels is not None and pd.api.types.is_numeric_dtype(labels):
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', edgecolor='k')
    else:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], color='gray', edgecolor='k')
        print("⚠️ Labels for coloring UMAP plot are missing or non-numeric.")

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(False)

    return fig

def plot_tsne(df_embeddings: pd.DataFrame, labels=None, title="t-SNE Projection"):
    n_samples = df_embeddings.shape[0]
    perplexity = max(1, min(30, (n_samples - 1) // 3))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(df_embeddings)

    fig, ax = plt.subplots(figsize=(6,5))
    if labels is not None and pd.api.types.is_numeric_dtype(labels):
        ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', edgecolor='k')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], color='gray', edgecolor='k')

    ax.set_title(title + f" (perplexity={perplexity})")
    ax.set_xlabel("tSNE-1")
    ax.set_ylabel("tSNE-2")
    ax.grid(False)
    return fig


def plot_pca_scatter(df_with_clusters):
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(
        df_with_clusters["PC1"],
        df_with_clusters["PC2"],
        c=df_with_clusters["Cluster"],
        cmap="rainbow",
        edgecolor="k"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Scatter Plot")
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)
        
    return fig

def plot_cluster_distribution(df_with_clusters: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x='Cluster', data=df_with_clusters, palette='pastel', ax=ax)
    ax.set_title("Cluster Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.grid(False)
    ax.set_axisbelow(False)

    return fig

def plot_radar_chart(df: pd.DataFrame, cluster_col: str, numeric_cols: list):
    import numpy as np
    import matplotlib.pyplot as plt

    clusters = df[cluster_col].unique()
    num_vars = len(numeric_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # loop

    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))  # Match sizing

    for cluster in clusters:
        avg_values = df[df[cluster_col] == cluster][numeric_cols].mean().tolist()
        avg_values += avg_values[:1]
        ax.plot(angles, avg_values, label=f'Cluster {cluster}', linewidth=2)
        ax.fill(angles, avg_values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), numeric_cols)
    ax.set_title("Cluster Personas (Radar Chart)", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    return fig
