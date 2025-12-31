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

    # extract nested zips
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        target = zfile.parent / zfile.stem
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(zfile, "r") as inner:
            inner.extractall(target)

    st.success("✅ NCERT PDFs extracted successfully")

# ===================== TEXT CLEANING =====================
def clean_text(text: str) -> str:
    junk_patterns = [
        r"Prelims\.indd.*",
        r"ISBN.*",
        r"©.*",
        r"All rights reserved.*",
        r"Printed in.*",
        r"Chief.*",
        r"Editor.*",
        r"Professor.*",
        r"University.*",
        r"Department.*",
        r"Delhi.*",
        r"NCERT.*",
        r"Reprint.*",
        r"\b\d{1,2}:\d{2}:\d{2}\b",
    ]

    for pat in junk_patterns:
        text = re.sub(pat, " ", text, flags=re.I)

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join([p.extract_text() or "" for p in reader.pages])
    except:
        return ""

def load_subject_text(subject):
    texts = []
    keywords = SUBJECTS[subject]

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if any(k in str(pdf).lower() for k in keywords):
            text = clean_text(read_pdf(pdf))
            if len(text.split()) > 120:
                texts.append(text)

    return texts

# ===================== CHUNKING =====================
def chunk_text(text, min_words=40, max_words=120):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], []

    for s in sentences:
        if len(s.split()) < 6:
            continue
        current.append(s)

        if sum(len(x.split()) for x in current) >= max_words:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def remove_duplicate_chunks(chunks, model, threshold=0.85):
    if len(chunks) <= 1:
        return chunks

    embeddings = model.encode(chunks)
    filtered = []

    for i, emb in enumerate(embeddings):
        duplicate = False
        for kept_text, kept_emb in filtered:
            if cosine_similarity([emb], [kept_emb])[0][0] > threshold:
                duplicate = True
                break
        if not duplicate:
            filtered.append((chunks[i], emb))

    return [x[0] for x in filtered]

# ===================== EMBEDDING =====================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedder()

# ===================== FLASHCARD GENERATION =====================
def generate_flashcards(chunks, topic, depth):
    cards = []
    for c in chunks[:DEPTH_CONFIG[depth]["top_k"]]:
        cards.append(f"""
### 📘 {topic} ({depth})

**Concept Overview**  
{c}

**Key Insight**
- Helps build conceptual clarity  
- Important for analytical questions  
""")
    return cards


def generate_summary_flashcard(chunks, topic):
    joined = " ".join(chunks[:4])

    return f"""
## ⭐ Master Summary — {topic}

**Core Idea**  
{joined}

**Why It Matters**
- Builds constitutional understanding  
- Links theory with real-life governance  
- Frequently asked in UPSC & NCERT  

**Exam Tip**
Focus on interpretation rather than memorization.
"""

# ===================== APP FLOW =====================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcards"):
    texts = load_subject_text(subject)

    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        cleaned = [clean_text(t) for t in texts]

        chunks = []
        for t in cleaned:
            chunks.extend(chunk_text(t))

        chunks = remove_duplicate_chunks(chunks, model)

        flashcards = generate_flashcards(chunks, topic, depth)
        summary = generate_summary_flashcard(chunks, topic)

        for fc in flashcards:
            st.markdown(fc)

        st.markdown("---")
        st.markdown(summary)
