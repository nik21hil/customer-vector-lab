# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocess import clean_customer_data
from src.embeddings import get_numeric_features, scale_features, generate_pca_embeddings
from src.clustering import perform_kmeans
from src.visualize import plot_umap, plot_radar_chart, plot_cluster_distribution, plot_tsne

#st.set_page_config(page_title="Customer Vector Lab")


st.set_page_config(
    page_title="Customer Vector Lab",
    page_icon="https://raw.githubusercontent.com/nik21hil/customer-vector-lab/main/assets/ns_logo1_transparent.png",
)

st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 10px;">
        <img src="https://raw.githubusercontent.com/nik21hil/customer-vector-lab/main/assets/ns_logo1_transparent.png" width="100">
        <h1 style="margin: 0; font-size: 48px;">Customer Vector Lab</h1>
    </div>
    <p style="text-align: center; color: gray; font-size: 15px; margin-top: -5px;">
        A lightweight, no-code interface to generate customer embeddings, cluster personas, and visualize segmentation insights.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown("###### 📌 Description")
st.markdown("""
**Customer Vector Lab** enables quick, interactive clustering and persona exploration for any structured customer dataset.

It transforms raw customer records into meaningful segments using PCA-based embeddings and KMeans clustering, then visualizes them using scatter plots, UMAP, t-SNE, radar charts, and downloadable CSVs, all in a browser-friendly format.
""")

st.markdown("###### 📂 Supported Data")
st.markdown("""
- Any **CSV** with numeric customer features
- Typical fields include: Demographics (age, income, location) | Behavioral signals (spending, visits, engagement) | Transaction summaries (LTV, frequency, recency)

_Categorical columns and IDs are automatically excluded during preprocessing._
""")

st.markdown("###### 🧰 Features")
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

# st.markdown("##### 🚀 Use Cases")
# st.markdown("""
# - Customer persona generation for marketing and targeting
# - Segment discovery for business intelligence teams
# - Visual storytelling for analytics presentations
# - Lightweight vectorization for downstream ML tasks
# """)

st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("##### 📄 Raw Uploaded Data")
    st.dataframe(df.head())

    df_clean = clean_customer_data(df)
    df_numeric = get_numeric_features(df_clean)
    df_scaled = scale_features(df_numeric)
    df_pca = generate_pca_embeddings(df_scaled, n_components=2)

    # Dynamically detect numeric columns
    all_numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
    exclude_cols = ['PC1', 'PC2', 'Cluster']
    default_radar_cols = [col for col in all_numeric_cols if col not in exclude_cols]

    # Sidebar form
    with st.sidebar.form("controls"):
        st.markdown("#### 🔢 Clustering Controls")
        n_clusters = st.slider("Select number of clusters (K):", min_value=2, max_value=10, value=3)

        st.markdown("#### 🕸️ Radar Chart Features")
        radar_cols = st.multiselect("Select numeric features for radar chart:",
                                    options=default_radar_cols,
                                    default=default_radar_cols[:3])

        apply_clicked = st.form_submit_button("✅ Apply Changes")

    if apply_clicked:
        df_with_clusters = df_clean.copy().reset_index(drop=True)
        df_with_clusters[['PC1', 'PC2']] = df_pca
        df_with_clusters['Cluster'] = perform_kmeans(df_pca, n_clusters=n_clusters)
        df_with_clusters['Cluster'] = df_with_clusters['Cluster'].astype(int)

        # PCA + Cluster Distribution side-by-side
        st.markdown("##### 📊 PCA & Cluster Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            scatter = ax.scatter(df_with_clusters["PC1"], df_with_clusters["PC2"],
                                 c=df_with_clusters["Cluster"], cmap='rainbow', edgecolor='k')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA Scatter")
            st.pyplot(fig)

        with col2:
            fig2 = plot_cluster_distribution(df_with_clusters)
            fig2.set_size_inches(5, 4)
            st.pyplot(fig2)

        # UMAP + t-SNE
        st.markdown("##### 🧬 Dimensionality Reduction")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**UMAP Projection**")
            st.pyplot(plot_umap(df_pca, labels=df_with_clusters['Cluster']))

        with col4:
            st.markdown("**t-SNE Projection**")
            st.pyplot(plot_tsne(df_pca, labels=df_with_clusters['Cluster']))

        # Radar Chart
        st.markdown("##### 🕸️ Cluster Personas (Radar Chart)")
        st.markdown("Shows average values of selected numeric fields across clusters.")

        if radar_cols:
            with st.container():
                st.pyplot(plot_radar_chart(df_with_clusters, cluster_col='Cluster', numeric_cols=radar_cols))
        else:
            st.warning("Please select at least one numeric column for radar chart.")

        # Persona Summary Table
        st.markdown("##### 📋 Persona Summary Table")
        if radar_cols:
            cluster_summary = df_with_clusters.groupby('Cluster')[radar_cols].mean().round(2)
            st.dataframe(cluster_summary)

        # CSV Export
        st.markdown("##### 📤 Download Clustered Output")
        csv = df_with_clusters.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download clustered data as CSV",
            data=csv,
            file_name='customer_clusters.csv',
            mime='text/csv',
        )
