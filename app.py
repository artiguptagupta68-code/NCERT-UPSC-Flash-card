import os
import re
import zipfile
from pathlib import Path

import streamlit as st
import gdown
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ===================== CONFIG =====================
FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = {
    "Polity": ["polity"],
    "Economics": ["economics"],
    "Sociology": ["sociology"],
    "Psychology": ["psychology"],
    "Business Studies": ["business"]
}

DEPTH_CONFIG = {
    "NCERT": {"top_k": 3, "similarity": 0.35},
    "UPSC": {"top_k": 6, "similarity": 0.45}
}

# ===================== STREAMLIT =====================
st.set_page_config("NCERT + UPSC Smart Flashcards", layout="wide")
st.title("📘 NCERT + UPSC Smart Flashcard Generator")

# ===================== DOWNLOAD & EXTRACT =====================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            ZIP_PATH,
            quiet=False,
            fuzzy=True
        )

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Extract nested ZIPs
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        target = zfile.parent / zfile.stem
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(zfile, "r") as inner:
            inner.extractall(target)
    st.success("✅ NCERT PDFs extracted successfully")

# ===================== CLEAN TEXT =====================
def clean_text(text):
    text = re.sub(r"(activity|exercise|project|editor|reprint|isbn|copyright|email|Prelims\.indd).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    # keep only sentences of reasonable length
    sentences = re.split(r'(?<=[.?!])\s+', text)
    sentences = [s.strip() for s in sentences if 5 <= len(s.split()) <= 40]
    return " ".join(sentences)

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    except:
        return ""
def load_subject_text(subject):
    texts = []
    keywords = SUBJECTS[subject]

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        pdf_path_lower = str(pdf).lower()  # check full path too
        if any(k in pdf_path_lower for k in keywords):
            text = clean_text(read_pdf(pdf))
            if len(text.split()) > 80:
                texts.append(text)
    return texts





# ===================== CHUNKING =====================
def chunk_text(text, chunk_size=3):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    sentences = [s for s in sentences if 5 <= len(s.split()) <= 40]
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunks.append(" ".join(sentences[i:i+chunk_size]))
    return chunks


# ===================== LOAD EMBEDDING =====================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

model = load_embedder()


# ===================== FLASHCARD GENERATION =====================
def generate_flashcards(chunks, topic, depth="NCERT"):
    if not chunks:
        return None
    
    chunk_size = DEPTH_CONFIG[depth]["chunk_size"]
    threshold = DEPTH_CONFIG[depth]["similarity"]
    
    # Create embeddings
    embeddings = model.encode(chunks)
    query_emb = model.encode([topic])
    sims = cosine_similarity(query_emb, embeddings)[0]
    
    # Rank chunks
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    selected = [c for c, s in ranked if s >= threshold][:TOP_K]
    
    # Build flashcards
    flashcards = []
    for idx, c in enumerate(selected):
        sentences = re.split(r'(?<=[.?!])\s+', c)
        if len(sentences) >= 2:
            flashcards.append({
                "overview": sentences[0],
                "explanation": " ".join(sentences[1:])
            })
    return flashcards

def summarize_flashcards(flashcards, topic):
    if not flashcards:
        return None
    overview = flashcards[0]["overview"]
    explanation = " ".join([f["explanation"] for f in flashcards[:3]])
    return f"""
### 📘 {topic}

**Concept Overview**  
{overview}

**Explanation**  
{explanation}

**Conclusion**  
This concept is important for understanding governance, rights, and social structures in society.
"""

# ===================== UI =====================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)
    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        all_chunks = []
        for t in texts:
            all_chunks.extend(chunk_text(t, chunk_size=DEPTH_CONFIG[depth]["top_k"]))

        flashcards = generate_flashcards(all_chunks, topic, depth)
        final_card = summarize_flashcards(flashcards, topic)

        if final_card:
            st.markdown(final_card)
        else:
            st.warning("No meaningful content could be generated for this topic.")
