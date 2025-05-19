# 🧠 Customer Vector Lab

> Create unified vector embeddings for customer profiling and segmentation

[![Streamlit App](https://img.shields.io/badge/Live_App-Click_to_Launch-00bfff?logo=streamlit)](https://nik21hil-customer-vector-lab.streamlit.app)

---

## ✨ Overview

**Customer Vector Lab** is a no-code tool to:

- Upload raw customer CSV data
- Automatically clean and standardize numeric features
- Generate vector embeddings using PCA
- Cluster customers using KMeans
- Visualize personas using:
  - 📊 PCA scatter plot
  - 📈 UMAP + t-SNE projections
  - 🕸 Radar charts of cluster traits
- Explore customer distribution across clusters

It's perfect for **data scientists**, **marketers**, and **business analysts** to quickly identify segments and personas for personalization, targeting, or storytelling.

---

## 📸 Sample Outputs

| PCA Scatter (Clustered) | UMAP + t-SNE Projection |
|-------------------------|--------------------------|
| ![PCA](assets/pca.png)  | ![UMAP](assets/umap_tsne.png) |

> 📌 _Add screenshots here after taking them — let me know if you want help formatting or exporting._

---

## 🚀 Live Demo

👉 [Launch the App](https://nik21hil-customer-vector-lab.streamlit.app)

---

## 🧑‍💻 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/nik21hil/customer-vector-lab.git
cd customer-vector-lab

# 2. (Optional) Create virtual env
python -m venv env
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
