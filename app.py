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

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def remove_stopwords_and_lemmatize(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)

@st.cache_resource
def load_models():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    svd = joblib.load('svd_model.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    return vectorizer, svd, kmeans

def predict_cluster(text, vectorizer, kmeans):
    cleaned_text = clean_text(text)
    processed_text = remove_stopwords_and_lemmatize(cleaned_text)
    
    text_vector = vectorizer.transform([processed_text])
    
    expected_features = kmeans.n_features_in_
    if text_vector.shape[1] != expected_features:
        if text_vector.shape[1] < expected_features:
            # Pad dengan nol jika kurang
            padded_vector = np.pad(text_vector.toarray(), ((0, 0), (0, expected_features - text_vector.shape[1])), mode='constant')
        else:
            # Kurangi fitur jika lebih (ambil subset)
            padded_vector = text_vector.toarray()[:, :expected_features]
        text_vector = padded_vector
    
    cluster_num = kmeans.predict(text_vector)[0]
    cluster_labels = {0: "Insiden dan Penalti", 1: "Kemenangan dan Prestasi", 2: "Dominasi dan Persaingan"}
    return cluster_labels.get(cluster_num, "Klaster Tidak Dikenali")

def generate_wordcloud(cluster_label, vectorizer, kmeans):
    cluster_map = {"Insiden dan Penalti": 0, "Kemenangan dan Prestasi": 1, "Dominasi dan Persaingan": 2}
    cluster_num = cluster_map.get(cluster_label, 0)
    
    terms = vectorizer.get_feature_names_out()
    cluster_center = kmeans.cluster_centers_[cluster_num]
    word_freq = {terms[i]: cluster_center[i] for i in range(len(terms)) if cluster_center[i] > 0}
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Word Cloud untuk {cluster_label}")
    return fig

st.title("YouTube Comment Cluster Predictor")
st.write("Masukkan kata atau kalimat untuk memprediksi kluster komentar YouTube terkait F1 2024.(Menggunakan Bahasa Inggris)")

user_input = st.text_area("Masukkan teks:", "")

if st.button("Prediksi"):
    if user_input:
        try:
            vectorizer, svd, kmeans = load_models()
            cluster_label = predict_cluster(user_input, vectorizer, kmeans)
            st.success(f"Teks termasuk dalam **{cluster_label}**")
            
            # Tampilkan word cloud untuk klaster
            st.write(f"Word Cloud untuk {cluster_label}:")
            fig = generate_wordcloud(cluster_label, vectorizer, kmeans)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")
