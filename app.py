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

# ======================================================
# CONFIG
# ======================================================
FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 6
SIMILARITY_THRESHOLD_NCERT = 0.35
SIMILARITY_THRESHOLD_UPSC = 0.45
EPOCHS = 15
BATCH_SIZE = 16

# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(page_title="NCERT + UPSC AI", layout="wide")
st.title("📘 NCERT + UPSC AI Question Generator")

# ======================================================
# DOWNLOAD + EXTRACT
# ======================================================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

# ======================================================
# TEXT EXTRACTION
# ======================================================
def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(pdf))
        if len(t.split()) > 100:
            texts.append(t)
    return texts

# ======================================================
# SEMANTIC CHUNKING
# ======================================================
def semantic_chunking(text, embedder, max_words=180):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s for s in sentences if len(s.split()) > 6]

    if len(sentences) < 2:
        return []

    embeddings = embedder.encode(sentences, convert_to_numpy=True)

    chunks, current = [], [sentences[0]]
    current_emb = embeddings[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([current_emb], [embeddings[i]])[0][0]
        if sim < 0.65 or len(" ".join(current).split()) > max_words:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_emb = embeddings[i]
        else:
            current.append(sentences[i])
            current_emb = np.mean([current_emb, embeddings[i]], axis=0)

    if current:
        chunks.append(" ".join(current))

    return chunks

# ======================================================
# LOAD EMBEDDING MODEL
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# ======================================================
# TRAIN MODEL
# ======================================================
def train_model(chunks, embedder):
    examples = []
    for ch in chunks:
        sents = re.split(r"(?<=[.?!])\s+", ch)
        for i in range(len(sents) - 1):
            examples.append(InputExample(texts=[sents[i], sents[i + 1]]))

    if not examples:
        st.warning("⚠️ Not enough data for training.")
        return embedder

    loader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(embedder)

    st.info("🧠 Training embedding model...")
    embedder.fit(train_objectives=[(loader, loss)], epochs=EPOCHS, show_progress_bar=True)
    st.success("✅ Training complete")

    return embedder

# ======================================================
# RETRIEVAL
# ======================================================
def retrieve_chunks(chunks, embeddings, query, mode):
    if len(chunks) == 0:
        return []

    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]

    threshold = SIMILARITY_THRESHOLD_UPSC if mode == "UPSC" else SIMILARITY_THRESHOLD_NCERT
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked if s >= threshold][:TOP_K]

# ======================================================
# FLASHCARD
# ======================================================
def generate_flashcard(chunks, topic):
    if not chunks:
        return None

    sents = []
    for ch in chunks:
        sents += re.split(r"(?<=[.?!])\s+", ch)

    overview = sents[0]
    explanation = " ".join(sents[1:6])

    return {
        "title": topic.title(),
        "content": f"""
**Concept Overview**  
{overview}

**Explanation**  
{explanation}

**Conclusion**  
This topic is important for NCERT understanding and UPSC conceptual clarity.
"""
    }

# ======================================================
# MAIN FLOW
# ======================================================
download_and_extract()
embedder = load_embedder()

texts = load_all_texts()
st.write(f"📄 Loaded {len(texts)} PDFs")
for i, t in enumerate(texts[:3]):
    st.write(f"--- PDF {i+1} content snippet ---")
    st.write(t[:500])  # preview first 500 chars


if not texts:
    st.error("No readable text found in PDFs.")
    st.stop()

all_chunks = []
for t in texts:
    all_chunks.extend(semantic_chunking(t, embedder))

st.write(f"🧩 Chunks created: {len(all_chunks)}")

if not all_chunks:
    st.error("No semantic chunks generated.")
    st.stop()

# Train once
if "trained" not in st.session_state:
    embedder = train_model(all_chunks, embedder)
    st.session_state.trained = True

embeddings = embedder.encode(all_chunks, convert_to_numpy=True)

# ======================================================
# UI
# ======================================================
topic = st.text_input("Enter Topic")
mode = st.radio("Depth", ["NCERT", "UPSC"], horizontal=True)

if st.button("Generate"):
    results = retrieve_chunks(all_chunks, embeddings, topic, mode)
    card = generate_flashcard(results, topic)

    if card:
        st.markdown(f"## 📘 {card['title']}")
        st.markdown(card["content"])
    else:
        st.warning("No relevant content found.")
