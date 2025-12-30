# =========================================
# NCERT + UPSC Flashcard Generator
# =========================================

import os, zipfile, re
from pathlib import Path
import streamlit as st
import gdown
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------
# CONFIG
# --------------------------------------------
FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
TOP_K = 6
SIMILARITY_THRESHOLD_NCERT = 0.35
SIMILARITY_THRESHOLD_UPSC = 0.45

# --------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------
st.set_page_config(page_title="NCERT Flashcard Generator", layout="wide")
st.title("📘 NCERT + UPSC Flashcard Generator")

# --------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# --------------------------------------------
# DOWNLOAD & EXTRACT ZIP (with nested zips)
# --------------------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Extract main zip
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    # Recursively extract nested zips
    def extract_nested(folder):
        for zfile in Path(folder).rglob("*.zip"):
            target = zfile.parent / zfile.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(target)
            zfile.unlink()  # optional: delete zip after extraction
            extract_nested(target)

    extract_nested(EXTRACT_DIR)

    st.success("✅ NCERT PDFs extracted!")

# --------------------------------------------
# PDF READING & CLEANING
# --------------------------------------------
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
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 50:
            texts.append(t)
    return texts

# --------------------------------------------
# SEMANTIC CHUNKING
# --------------------------------------------
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

# --------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# --------------------------------------------
def retrieve_relevant_chunks(chunks, embeddings, query, mode="NCERT", top_k=TOP_K):
    if len(chunks) == 0:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    threshold = SIMILARITY_THRESHOLD_UPSC if mode=="UPSC" else SIMILARITY_THRESHOLD_NCERT
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked if s >= threshold][:top_k]

# --------------------------------------------
# FLASHCARD GENERATION
# --------------------------------------------
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
    conclusion = "This concept is important for governance and society."
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

# --------------------------------------------
# SIDEBAR: DOWNLOAD PDFs
# --------------------------------------------
with st.sidebar:
    if st.button("📥 Load NCERT PDFs", key="load_pdfs"):
        download_and_extract()

# --------------------------------------------
# MAIN UI
# --------------------------------------------
topic = st.text_input("Enter topic (e.g. Fundamental Rights)")
mode = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)

if st.button("Generate Flashcard", key="gen_flashcard"):
    # Load texts & chunks
    texts = load_all_texts()
    if not texts:
        st.warning("⚠️ No PDF content loaded. Check extraction or file permissions.")
    else:
        all_chunks = []
        for t in texts:
            all_chunks.extend(semantic_chunking(t, embedder))

        embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
        relevant = retrieve_relevant_chunks(all_chunks, embeddings, topic, mode)
        card = generate_flashcard(relevant, topic)
        if card:
            st.markdown(f"## 📘 {card['title']}")
            st.markdown(card["content"])
        else:
            st.warning("No relevant content found for this topic.")
