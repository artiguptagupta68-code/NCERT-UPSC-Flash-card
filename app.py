import os
import re
import zipfile
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
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

# ================= STREAMLIT =================
st.set_page_config("NCERT Flashcard Generator", layout="wide")
st.title("📘 NCERT → Smart Concept Flashcard")

# ================= DOWNLOAD & EXTRACT =================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            ZIP_PATH,
            quiet=False,
            fuzzy=True
        )

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    except zipfile.BadZipFile:
        st.error("The downloaded file is not a valid ZIP file.")
        return

    for z in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            with zipfile.ZipFile(z, "r") as inner:
                inner.extractall(z.parent / z.stem)
        except:
            continue

# ================= CLEANING & READING =================
def clean_text(text):
    patterns = [
        r"Prelims\.indd.*", r"ISBN.*", r"Reprint.*", r"Printed.*",
        r"All rights reserved.*", r"University.*", r"Editor.*",
        r"Copyright.*", r"Email:.*", r"\b\d{1,2}\s[A-Za-z]+\s\d{4}\b"
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = " ".join(p.extract_text() or "" for p in reader.pages)
        return clean_text(text)
    except:
        return ""

# ================= LOAD SUBJECT TEXT =================
def load_all_text(subject):
    texts = []
    keywords = SUBJECTS.get(subject, [])

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        pdf_path = str(pdf).lower()
        if any(k in pdf_path for k in keywords):
            text = read_pdf(pdf)
            if len(text.split()) > 50:
                texts.append(text)
    return texts

# ================= EMBEDDING MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ================= CHUNKING =================
def chunk_text(text, min_words=30, max_words=120):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buffer = [], []

    for s in sentences:
        if len(s.split()) < 5:
            continue
        buffer.append(s)
        if sum(len(x.split()) for x in buffer) >= max_words:
            chunks.append(" ".join(buffer))
            buffer = []
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks

# ================= FLASHCARD GENERATION =================
def generate_flashcard(texts, topic):
    all_text = " ".join(texts)
    all_text = clean_text(all_text)
    chunks = chunk_text(all_text)

    if not chunks:
        return "⚠️ No meaningful content found."

    # semantic relevance to topic
    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)
    scored = [(chunk, cosine_similarity([vec], topic_vec)[0][0])
              for chunk, vec in zip(chunks, chunk_vecs)]
    relevant = [c for c, s in scored if s > 0.35]

    if not relevant:
        return "⚠️ No meaningful content found."

    # summarize into structured flashcard
    combined = " ".join(relevant[:3])
    sentences = re.split(r'(?<=[.!?])\s+', combined)

    concept_overview = " ".join(sentences[:3])
    explanation = " ".join(sentences[3:6]) if len(sentences) > 3 else ""
    importance = " ".join(sentences[6:9]) if len(sentences) > 6 else ""

    flashcard = f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{concept_overview}

**Explanation**  
{explanation}

**Why It Matters**  
- Builds conceptual clarity  
- Connects theory with practical governance  
- Important for NCERT & UPSC preparation
"""
    return flashcard

# ================= STREAMLIT UI =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g., Fundamental Rights)")

if st.button("Generate Flashcard") and topic.strip():
    texts = load_all_text(subject)
    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        flashcard = generate_flashcard(texts, topic)
        st.markdown(flashcard)
