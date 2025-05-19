# src/embeddings.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts only numeric features for embedding generation.
    """
    return df.select_dtypes(include=['int64', 'float64'])

def scale_features(df_numeric: pd.DataFrame) -> pd.DataFrame:
    """
    Applies standard scaling to numeric features.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_numeric)
    return pd.DataFrame(scaled, columns=df_numeric.columns)

def generate_pca_embeddings(df_scaled: pd.DataFrame, n_components=2) -> pd.DataFrame:
    """
    Generates PCA-based embeddings from scaled features.
    Returns a DataFrame with principal components.
    """
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(embeddings, columns=[f"PC{i+1}" for i in range(n_components)])
    print(f"✅ Explained variance ratio: {pca.explained_variance_ratio_}")
    return df_pca

def plot_pca_embeddings(df_pca: pd.DataFrame, labels=None):
    """
    Plots 2D PCA embeddings.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(df_pca["PC1"], df_pca["PC2"], c=labels, cmap='viridis', edgecolor='k')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Embeddings")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
