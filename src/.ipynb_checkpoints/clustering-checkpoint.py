# src/clustering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
