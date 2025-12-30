import os
import re
import zipfile
import numpy as np
import streamlit as st
import gdown

from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import random

# =========================================
# CONFIG
# =========================================
FILE_ID = "1l5tGbJcvwsPiiOyi-MZYoWzoTDGjTWOz"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 6
SIMILARITY_THRESHOLD_NCERT = 0.35
SIMILARITY_THRESHOLD_UPSC = 0.45
EPOCHS = 15  # Fine-tuning epochs
BATCH_SIZE = 16

# =========================================
# STREAMLIT CONFIG
# =========================================
st.set_page_config(page_title="NCERT AI Generator", layout="wide")
st.title("📘 NCERT + UPSC AI Question Generator with Training")

# =========================================
# DOWNLOAD & EXTRACT NCERT ZIP
# =========================================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    st.success("✅ NCERT content extracted")

# =========================================
# READ PDF & CLEAN TEXT
# =========================================
def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(exercise|summary|table|figure|copyright).*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(pdf))
        if len(t.split()) > 100:
            texts.append(t)
    return texts

# =========================================
# SEMANTIC CHUNKING
# =========================================
def semantic_chunking(text, embedder, max_words=180, sim_threshold=0.65):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s for s in sentences if len(s.split()) > 6]

    if len(sentences) < 2:
        return sentences

    embeddings = embedder.encode(sentences, convert_to_numpy=True)

    chunks = []
    current = [sentences[0]]
    current_emb = embeddings[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([current_emb], [embeddings[i]])[0][0]
        length = sum(len(s.split()) for s in current)

        if sim < sim_threshold or length > max_words:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_emb = embeddings[i]
        else:
            current.append(sentences[i])
            current_emb = np.mean([current_emb, embeddings[i]], axis=0)

    if current:
        chunks.append(" ".join(current))

    return chunks

# =========================================
# LOAD EMBEDDER
# =========================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# =========================================
# PREPARE TRAINING DATA
# =========================================
def prepare_training_data(chunks):
    examples = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[.?!])\s+", chunk)
        sentences = [s for s in sentences if len(s.split()) > 6]
        for i in range(len(sentences)-1):
            examples.append(InputExample(texts=[sentences[i], sentences[i+1]]))
    return examples

# =========================================
# TRAIN MODEL
# =========================================
def train_embedding_model(chunks, embedder, epochs=EPOCHS):
    st.info(f"🔹 Fine-tuning embeddings for {epochs} epochs...")
    train_examples = prepare_training_data(chunks)
    if not train_examples:
        st.warning("Not enough data to fine-tune embeddings.")
        return embedder

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(embedder)

    embedder.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=50,
        show_progress_bar=True
    )
    st.success("✅ Embedding model fine-tuned!")
    embedder.save("trained_ncert_model")
    return embedder

# =========================================
# EMBEDDINGS
# =========================================
@st.cache_data
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)

# =========================================
# RETRIEVAL
# =========================================
def retrieve_relevant_chunks(chunks, embeddings, query, mode="NCERT", top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    threshold = SIMILARITY_THRESHOLD_UPSC if mode=="UPSC" else SIMILARITY_THRESHOLD_NCERT
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked if s >= threshold][:top_k]

# =========================================
# FLASHCARD GENERATION
# =========================================
def generate_flashcard(chunks, topic):
    sentences = []
    for ch in chunks:
        for s in re.split(r'(?<=[.?!])\s+', ch):
            if len(s.split()) > 8:
                sentences.append(s.strip())

    if not sentences:
        return None

    overview = sentences[0]
    explanation = " ".join(sentences[1:6])
    conclusion = "This concept is important for understanding governance and society, relevant for competitive exams."
    points = [" ".join(s.split()[:20]) for s in sentences[1:6]]

    return {
        "title": topic.title(),
        "content": f"""
**Concept Overview**
{overview}

**Explanation**
{explanation}

**Conclusion**
{conclusion}

**Key Points**
- {"\n- ".join(points)}
"""
    }

# =========================================
# APP FLOW
# =========================================
download_and_extract()
embedder = load_embedder()

# Load texts
texts = load_all_texts()

# Semantic chunking
all_chunks = []
for t in texts:
    all_chunks.extend(semantic_chunking(t, embedder))

# Fine-tune embedding model
embedder = train_embedding_model(all_chunks, embedder, epochs=EPOCHS)

# Encode all chunks
embeddings = embed_chunks(all_chunks)

# =========================================
# STREAMLIT UI
# =========================================
topic = st.text_input("Enter topic (e.g. Fundamental Rights)")
mode = st.radio("Depth", ["NCERT", "UPSC"], horizontal=True)

if st.button("Generate Flashcard"):
    if not topic.strip():
        st.warning("Enter a topic")
    else:
        rel = retrieve_relevant_chunks(all_chunks, embeddings, topic, mode)
        card = generate_flashcard(rel, topic)
        if card:
            st.markdown(f"## 📘 {card['title']}")
            st.markdown(card["content"])
        else:
            st.warning("No relevant content found.")
