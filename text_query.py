# text_query.py
import re
import nltk
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Pastikan resource NLTK terunduh
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    # Case folding
    text = text.lower()
    # Text cleaning: Hanya huruf a-z dan spasi
    text = re.sub('[^a-z ]', '', text)
    # Tokenization: split berdasarkan spasi
    tokens = text.split()
    # Stop word removal
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [w for w in tokens if w not in stop_words]
    # Lemmatization
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    # Detokenization
    processed_text = " ".join(tokens)
    return processed_text

def load_indobert_model():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
    model = AutoModel.from_pretrained("indobenchmark/indobert-base-p2")
    model.eval()
    # Load checkpoint untuk proyeksi (pastikan path sudah benar)
    checkpoint = torch.load("models/indobert-p2-projection256.pth", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['bert_model'])
    import torch.nn as nn
    projection = nn.Linear(768, 256)
    projection.load_state_dict(checkpoint['projection'])
    projection.eval()
    return tokenizer, model, projection

def get_text_embedding(text, tokenizer, model, projection, max_length=256):
    processed_text = preprocess_text(text)
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # [1, seq_len, 768]
        embedding = last_hidden_state.mean(dim=1)         # [1, 768]
        projected_embedding = projection(embedding)       # [1, 256]
        projected_embedding = projected_embedding.squeeze().detach().numpy()
    return projected_embedding
