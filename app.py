import os
import re
import zipfile
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== CONFIG =====================
FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = {
    "Polity": ["polity", "constitution"],
    "Economics": ["economics"],
    "Sociology": ["sociology"],
    "Psychology": ["psychology"],
    "Business Studies": ["business"]
}

DEPTH_CONFIG = {
    "NCERT": {"top_k": 3, "similarity": 0.25},  # slightly lower threshold
    "UPSC": {"top_k": 6, "similarity": 0.35}
}

# ===================== STREAMLIT =====================
st.set_page_config("NCERT + UPSC Flashcards", layout="wide")
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

# ===================== PDF READING & CLEANING =====================
def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        return clean_text(text)
    except:
        return ""

def clean_text(text):
    patterns = [
        r"Prelims\.indd.*", r"ISBN.*", r"All rights reserved.*", r"Printed.*",
        r"Reprint.*", r"Editor.*", r"University.*", r"Department.*", r"Copyright.*",
        r"\d{1,2}\s[A-Za-z]+\s\d{4}"
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_subject_text(subject):
    texts = []
    keywords = SUBJECTS[subject]
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if any(k in pdf.name.lower() for k in keywords):
            text = read_pdf(pdf)
            if len(text.split()) > 80:
                texts.append(text)
    return texts

# ===================== CHUNKING =====================
def chunk_text(text, min_words=50, max_words=150):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buf = [], []
    for s in sentences:
        if len(s.split()) < 6:
            continue
        buf.append(s)
        if sum(len(x.split()) for x in buf) >= max_words:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks

# ===================== EMBEDDING MODEL =====================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ===================== FLASHCARD GENERATION =====================
def generate_flashcard(texts, topic, depth="NCERT"):
    if not texts:
        return "⚠️ No readable content found for this subject."

    # Combine & chunk
    all_chunks = []
    for t in texts:
        all_chunks.extend(chunk_text(t))

    # Remove duplicates
    embeddings = model.encode(all_chunks)
    unique_chunks = []
    seen = []
    for i, emb in enumerate(embeddings):
        if not seen or max(cosine_similarity([emb], seen)[0]) < 0.85:
            unique_chunks.append(all_chunks[i])
            seen.append(emb)

    # Rank by similarity to topic
    topic_vec = model.encode([topic])
    sims = [cosine_similarity([topic_vec], [model.encode([c])[0]])[0][0] for c in unique_chunks]
    threshold = DEPTH_CONFIG[depth]["similarity"]
    filtered = [c for c, s in zip(unique_chunks, sims) if s >= threshold]

    # Fallback: pick top K even if below threshold
    if not filtered:
        ranked = sorted(zip(unique_chunks, sims), key=lambda x: x[1], reverse=True)
        filtered = [c for c, s in ranked[:DEPTH_CONFIG[depth]["top_k"]]]

    # Generate final flashcard
    summary_text = " ".join(filtered[:3])
    summary_sentences = re.split(r'(?<=[.!?])\s+', summary_text)

    flashcard = f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{' '.join(summary_sentences[:6])}

**Explanation**  
{' '.join(summary_sentences[6:12]) if len(summary_sentences)>6 else ''}

**Why It Matters**
- Builds conceptual clarity  
- Helps connect theory with real-life applications  
- High-weight topic for NCERT & UPSC
"""
    return flashcard

# ===================== STREAMLIT UI =====================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)
    flashcard = generate_flashcard(texts, topic, depth)
    st.markdown(flashcard)
