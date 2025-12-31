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

# =====================================================
# CONFIG
# =====================================================
FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ"   # MUST be ZIP file
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = {
    "Polity": ["polity", "constitution", "civics"],
    "Economics": ["economics"],
    "Sociology": ["sociology"],
    "Psychology": ["psychology"],
    "Business Studies": ["business"]
}

DEPTH_CONFIG = {
    "NCERT": {"top_k": 3, "similarity": 0.35},
    "UPSC": {"top_k": 6, "similarity": 0.45}
}

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config("NCERT + UPSC Flashcards", layout="wide")
st.title("📘 NCERT + UPSC Smart Flashcard Generator")

# =====================================================
# DOWNLOAD & EXTRACT ZIP (SAFE)
# =====================================================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(id=FILE_ID, output=ZIP_PATH, quiet=False, fuzzy=True)

    if not zipfile.is_zipfile(ZIP_PATH):
        st.error("❌ Downloaded file is not a ZIP. Please re-upload correct file.")
        st.stop()

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Extract nested zips
    for z in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            out = z.parent / z.stem
            out.mkdir(exist_ok=True)
            with zipfile.ZipFile(z, "r") as nz:
                nz.extractall(out)
        except:
            pass

    st.success("✅ NCERT PDFs extracted successfully")

# =====================================================
# CLEAN + LOAD PDF TEXT
# =====================================================
def clean_text(text):
    junk = [
        r"Prelims\.indd.*", r"ISBN.*", r"Reprint.*", r"©.*",
        r"Editor.*", r"Professor.*", r"University.*",
        r"Printed.*", r"All rights reserved.*"
    ]
    for j in junk:
        text = re.sub(j, " ", text, flags=re.I)

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
    keys = SUBJECTS[subject]

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if any(k in pdf.name.lower() for k in keys):
            text = clean_text(read_pdf(pdf))
            if len(text.split()) > 120:
                texts.append(text)

    return texts

# =====================================================
# TEXT CHUNKING
# =====================================================
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


# =====================================================
# EMBEDDING MODEL
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =====================================================
# FLASHCARD LOGIC
# =====================================================
def generate_flashcards(chunks, topic, depth):
    embeddings = model.encode(chunks)
    query = model.encode([topic])

    scores = cosine_similarity(query, embeddings)[0]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    top_k = DEPTH_CONFIG[depth]["top_k"]
    threshold = DEPTH_CONFIG[depth]["similarity"]

    selected = [c for c, s in ranked if s >= threshold][:top_k]

    if not selected:
        return None

    cards = []
    for c in selected:
        sentences = re.split(r'(?<=[.!?])\s+', c)
        cards.append({
            "overview": sentences[0],
            "explanation": " ".join(sentences[1:4])
        })

    return cards


def summarize_flashcards(cards, topic):
    combined = " ".join([c["explanation"] for c in cards[:3]])

    return f"""
### 📘 {topic}

**Concept Overview**  
{cards[0]["overview"]}

**Explanation**  
{combined}

**Why it matters**
- Builds constitutional understanding  
- Strengthens analytical ability  
- Essential for UPSC & NCERT exams  
"""

# =====================================================
# UI
# =====================================================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)

    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        chunks = []
        for t in texts:
            chunks.extend(chunk_text(t))

        cards = generate_flashcards(chunks, topic, depth)

        if not cards:
            st.warning("⚠️ No meaningful content found.")
        else:
            for i, c in enumerate(cards, 1):
                st.markdown(f"### 📄 Flashcard {i}")
                st.markdown(f"**Overview:** {c['overview']}")
                st.markdown(f"**Explanation:** {c['explanation']}")

            st.markdown("---")
            st.markdown(summarize_flashcards(cards, topic))
