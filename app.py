import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# Fungsi untuk menghapus stopwords dan lematisasi
def remove_stopwords_and_lemmatize(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)

# Memuat model yang disimpan
@st.cache_resource
def load_models():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    svd = joblib.load('svd_model.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    return vectorizer, svd, kmeans

# Fungsi untuk memprediksi kluster
def predict_cluster(text, vectorizer, svd, kmeans):
    cleaned_text = clean_text(text)
    processed_text = remove_stopwords_and_lemmatize(cleaned_text)
    
    # Gunakan vocabulary yang sudah ada dari vectorizer yang dilatih
    text_vector = vectorizer.transform([processed_text])
    
    # Pastikan jumlah fitur sesuai dengan yang diharapkan oleh SVD
    if text_vector.shape[1] != svd.components_.shape[1]:
        # Pad atau kurangi fitur agar sesuai dengan yang diharapkan SVD
        expected_features = svd.components_.shape[1]
        if text_vector.shape[1] < expected_features:
            # Pad dengan nol jika kurang
            padded_vector = np.pad(text_vector.toarray(), ((0, 0), (0, expected_features - text_vector.shape[1])), mode='constant')
        else:
            # Kurangi fitur jika lebih (ambil subset)
            padded_vector = text_vector.toarray()[:, :expected_features]
        text_svd = svd.transform(padded_vector)
    else:
        text_svd = svd.transform(text_vector.toarray())
    
    cluster_num = kmeans.predict(text_svd)[0]
    # Mapping cluster number to label
    cluster_labels = {0: "Insiden dan Penalti", 1: "Kemenangan dan Prestasi", 2: "Dominasi dan Persaingan"}
    return cluster_labels.get(cluster_num, "Klaster Tidak Dikenali")

# Fungsi untuk membuat word cloud
def generate_wordcloud(cluster_label, vectorizer, kmeans):
    cluster_map = {"Insiden dan Penalti": 0, "Kemenangan dan Prestasi": 1, "Dominasi dan Persaingan": 2}
    cluster_num = cluster_map.get(cluster_label, 0)
    
    # Dapatkan fitur (kata) dari vectorizer
    terms = vectorizer.get_feature_names_out()
    cluster_center = kmeans.cluster_centers_[cluster_num]
    word_freq = {terms[i]: cluster_center[i] for i in range(len(terms)) if cluster_center[i] > 0}
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Word Cloud untuk {cluster_label}")
    return fig

# Aplikasi Streamlit
st.title("YouTube Comment Cluster Predictor")
st.write("Masukkan kata atau kalimat untuk memprediksi kluster komentar YouTube terkait F1 2024. (Jam: 10:46 WIB, 24 Juni 2025)")

# Input teks
user_input = st.text_area("Masukkan teks:", "")

if st.button("Prediksi"):
    if user_input:
        try:
            vectorizer, svd, kmeans = load_models()
            cluster_label = predict_cluster(user_input, vectorizer, svd, kmeans)
            st.success(f"Teks termasuk dalam **{cluster_label}**")
            
            # Tampilkan word cloud untuk klaster
            st.write(f"Word Cloud untuk {cluster_label}:")
            fig = generate_wordcloud(cluster_label, vectorizer, kmeans)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")
