# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocess import load_customer_data, clean_customer_data
from src.embeddings import get_numeric_features, scale_features, generate_pca_embeddings
from src.clustering import perform_kmeans
from src.visualize import plot_umap, plot_radar_chart, plot_cluster_distribution, plot_tsne


st.set_page_config(page_title="Audience Vector Builder", layout="wide")

st.title("🧠 Customer Vector Lab")
st.markdown("""
This tool allows you to:
- Upload customer datasets
- Automatically clean and standardize your data
- Generate vector embeddings using PCA
- Cluster customers into personas
- Visualize clusters using PCA, UMAP, and t-SNE
- Profile clusters with radar charts

Great for segmentation, personalization, and analytics storytelling.
""")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Uploaded Data")
    st.dataframe(df.head())

    df_clean = clean_customer_data(df)
    df_numeric = get_numeric_features(df_clean)
    df_scaled = scale_features(df_numeric)
    df_pca = generate_pca_embeddings(df_scaled, n_components=2)
    df_with_clusters = df_clean.copy().reset_index(drop=True)
    df_with_clusters[['PC1', 'PC2']] = df_pca
    df_with_clusters['Cluster'] = perform_kmeans(df_pca, n_clusters=3)
    df_with_clusters['Cluster'] = df_with_clusters['Cluster'].astype(int)

    st.markdown("### 🔍 PCA Scatter Plot")
    with st.container():
        fig, ax = plt.subplots(figsize=(5,4))
        scatter = ax.scatter(df_with_clusters["PC1"], df_with_clusters["PC2"], 
                             c=df_with_clusters["Cluster"], cmap='rainbow', edgecolor='k')
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Customer Clusters")
        st.pyplot(fig)
    
    st.markdown("### 📊 Cluster Distribution")
    with st.container():
        st.pyplot(plot_cluster_distribution(df_with_clusters))
    
    st.markdown("### 🧬 Dimensionality Reduction Visuals")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**UMAP Projection**")
        st.pyplot(plot_umap(df_pca, labels=df_with_clusters['Cluster']))

    with col2:
        st.markdown("**t-SNE Projection**")
        st.pyplot(plot_tsne(df_pca, labels=df_with_clusters['Cluster']))


    st.markdown("### 🕸️ Cluster Personas (Radar Chart)")
    st.markdown("Shows average values of selected numeric fields across clusters, e.g., age, income, spend.")

    # Dynamically select numeric columns
    all_numeric_cols = df_with_clusters.select_dtypes(include='number').columns.tolist()
    exclude_cols = ['PC1', 'PC2', 'Cluster']  # Don’t include PCA or cluster labels
    radar_cols = [col for col in all_numeric_cols if col not in exclude_cols]
    if radar_cols:
        with st.container():
            st.pyplot(plot_radar_chart(df_with_clusters, cluster_col='Cluster', numeric_cols=radar_cols))

    else:
        st.warning("No suitable numeric columns found for radar chart.")