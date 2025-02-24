# app.py
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px

# Import modul buatan sendiri
from text_query import load_indobert_model, get_text_embedding
from image_query import load_resnet50_model, get_image_embedding
from retrieval import combine_embeddings

# Custom CSS untuk styling
st.markdown(
    """
    <style>
    /* Latar belakang dan teks utama */
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    .title {
        color: #4a4a4a;
        font-weight: bold;
    }
    .description {
        color: #555;
    }
    .result-title {
        color: #333;
        font-weight: bold;
    }
    .result-score {
        color: #007bff;
    }

    /* Styling sidebar */
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }

    /* Membesarkan tombol Cari dan membuatnya full-width */
    div.stButton > button {
        width: 100% !important;
        font-size: 1.2rem !important;
        padding: 0.5rem 1rem !important;
        color: white !important;
        background-color: red !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_embedding_db():
    """Memuat database embedding multimodal dari CSV."""
    df = pd.read_csv("datasets/data_multimodal_batik_embed.csv")
    df["Embed Multimodal"] = df["Embed Multimodal"].apply(lambda x: np.array(ast.literal_eval(x)))
    return df

# Fungsi evaluasi retrieval
def precision_at_k(retrieved, ground_truth, k):
    retrieved_k = retrieved[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in ground_truth]
    return len(relevant_retrieved) / k if k else 0

def recall_at_k(retrieved, ground_truth, k):
    retrieved_k = retrieved[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in ground_truth]
    return len(relevant_retrieved) / len(ground_truth) if ground_truth else 0

def average_precision(retrieved, ground_truth):
    score = 0.0
    num_hits = 0.0
    for i, doc in enumerate(retrieved):
        if doc in ground_truth:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(ground_truth) if ground_truth else 0

def ndcg_at_k(retrieved, ground_truth, k):
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        rel = 1 if doc in ground_truth else 0
        dcg += (2**rel - 1) / np.log2(i + 2)
    ideal_rels = [1] * min(len(ground_truth), k)
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0

# Load data dan model sekali di awal
df_db = load_embedding_db()
tokenizer, bert_model, projection = load_indobert_model()
resnet_model = load_resnet50_model()

def main():
    st.title("Sistem IR Multimodal Batik")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("Selamat datang di sistem pencarian batik. Unggah gambar dan/atau masukkan deskripsi batik.")

    # Sidebar untuk informasi proyek
    st.sidebar.header("Tentang Proyek")
    st.sidebar.info(
        """
        Proyek IR Multimodal Batik menggabungkan pencarian berbasis teks dan gambar.
        Unggah gambar atau masukkan deskripsi untuk menemukan motif batik yang relevan.
        """
    )
    
    # Tambahkan informasi Tim Pengembang
    st.sidebar.markdown("### Tim Pengembang")
    st.sidebar.markdown(
        """
        1. Muhammad Akbar Hamid  
        2. Agus Zulvani  
        3. Meilin Budiarti  
        4. Fitra Ahya Mubarok  
        """
    )
    
    # Database Embedding Multimodal Batik
    st.sidebar.markdown("## Database Batik")
    if st.sidebar.checkbox("Tampilkan"):
        st.sidebar.write(df_db.head())

    # Layout 2 kolom untuk input
    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader("Upload Gambar Batik", type=["jpg", "jpeg", "png"])
    with col2:
        query_text = st.text_input("Masukkan Deskripsi Batik", placeholder="Contoh: Batik Parang dengan motif garis melengkung")
    
    # Tambahkan input untuk ground truth (label batik yang relevan)
    gt_label = st.text_input("Masukkan Label Ground Truth/Nama Batik (misalnya, Batik Parang) (Kosongkan Jika Tidak Perlu)", 
                             placeholder="Nama Batik yang diharapkan relevan (pengisian opsional)")

    # Tombol Cari
    if st.button("Cari", key="search_button"):
        text_emb = None
        image_emb = None

        with st.spinner("Memproses query..."):
            if query_text:
                text_emb = get_text_embedding(query_text, tokenizer, bert_model, projection)
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_emb = get_image_embedding(image, resnet_model)

            query_embedding = combine_embeddings(text_embedding=text_emb, image_embedding=image_emb)
            if query_embedding is None:
                st.warning("Masukkan minimal salah satu input: teks atau gambar!")
                return

            # Hitung cosine similarity antara query dan database
            db_embeddings = np.vstack(df_db["Embed Multimodal"].values)
            similarities = cosine_similarity([query_embedding], db_embeddings)[0]
            top_indices = similarities.argsort()[::-1][:5]

            st.markdown("---")
            # --- Visualisasi PCA dengan Plotly Express ---
            st.markdown("### Visualisasi PCA Embedding Multimodal Batik dan Query")
            embeddings_all = np.vstack([query_embedding, db_embeddings])
            pca_2d = PCA(n_components=2)
            embeddings_2d = pca_2d.fit_transform(embeddings_all)
            
            # Buat DataFrame untuk plot
            names = ["Query"] + list(df_db["Nama Batik"])
            df_vi = pd.DataFrame({
                "Nama": names,
                "PCA1": embeddings_2d[:, 0],
                "PCA2": embeddings_2d[:, 1]
            })

            fig = px.scatter(
                df_vi, 
                x="PCA1", 
                y="PCA2", 
                text="Nama",  
                color="Nama",  
                labels={"PCA1": "Dimensi PCA 1", "PCA2": "Dimensi PCA 2", "Nama": "Nama Batik"},
                width=900,
                height=700
            )
            fig.update_traces(textposition="top center", marker=dict(size=12, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
            st.plotly_chart(fig)
        
        st.markdown("---")
        
        # Tampilkan hasil pencarian
        st.markdown("## Hasil Pencarian")
        for rank, idx in enumerate(top_indices, start=1):
            nama_batik = df_db.iloc[idx]["Nama Batik"]
            sim_score = similarities[idx]

            st.markdown(f"#### Peringkat {rank}")
            col_left, col_right = st.columns([1,2])
            with col_left:
                if "Path Gambar" in df_db.columns:
                    image_path = f"datasets/{df_db.iloc[idx]['Path Gambar']}"
                    st.image(image_path, width=250)
            with col_right:
                st.markdown(
                    f"#### **{nama_batik}** - Similarity: "
                    f"<span class='result-score'>{sim_score:.4f}</span>",
                    unsafe_allow_html=True
                )
                if "Deskripsi Batik" in df_db.columns:
                    deskripsi_batik = df_db.iloc[idx]['Deskripsi Batik']
                    st.write(deskripsi_batik)
            st.markdown("---")
        
        # --- Perhitungan Evaluasi Retrieval ---
        if gt_label:
            # Anggap dokumen relevan adalah semua baris dengan "Nama Batik" yang sama dengan nilai ground truth
            ground_truth = set(df_db.index[df_db["Nama Batik"] == gt_label].tolist())
            k = 5
            precision_k = precision_at_k(top_indices, ground_truth, k)
            recall_k = recall_at_k(top_indices, ground_truth, k)
            map_score = average_precision(top_indices, ground_truth)
            ndcg_k = ndcg_at_k(top_indices, ground_truth, k)

            # Buat DataFrame untuk menampilkan hasil evaluasi dalam bentuk tabel
            eval_df = pd.DataFrame({
                f"Precision@{k}": [precision_k],
                f"Recall@{k}": [recall_k],
                "MAP": [map_score],
                f"NDCG@{k}": [ndcg_k]
            })
            
            st.markdown("## Evaluasi Retrieval")
            st.table(eval_df)


if __name__ == "__main__":
    main()
