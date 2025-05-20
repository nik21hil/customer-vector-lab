# src/clustering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import numpy as np

def recommend_optimal_k(X, k_min=2, k_max=10):
    best_k = k_min
    best_score = -1

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, round(best_score, 3)

def perform_kmeans(df_embeddings: pd.DataFrame, n_clusters=3) -> pd.Series:
    """
    Performs KMeans clustering on PCA embeddings.
    Returns a Series of cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(df_embeddings)
    return pd.Series(labels, name='Cluster')

def compute_cosine_similarity_matrix(df_embeddings: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cosine similarity matrix between all users.
    Returns a square DataFrame.
    """
    sim_matrix = cosine_similarity(df_embeddings)
    return pd.DataFrame(sim_matrix, index=df_embeddings.index, columns=df_embeddings.index)

def find_top_similar_users(sim_matrix: pd.DataFrame, user_index: int, top_n=3) -> pd.Series:
    """
    Returns top-N most similar users to the given user_index.
    """
    user_scores = sim_matrix.loc[user_index].drop(user_index)
    return user_scores.sort_values(ascending=False).head(top_n)
