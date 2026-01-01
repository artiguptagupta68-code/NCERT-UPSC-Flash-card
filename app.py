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
        st.info("📥 Downloading NCERT ZIP...")
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
    except:
        st.error("❌ ZIP file is corrupted or invalid.")
        return

    # Extract nested ZIPs
    for z in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            with zipfile.ZipFile(z, "r") as inner:
                inner.extractall(z.parent / z.stem)
        except:
            continue
    st.success("✅ NCERT PDFs extracted successfully")

# ================= CLEANING =================
def clean_text(text):
    patterns = [
        r"ISBN.*", r"Reprint.*", r"Printed.*", r"©.*",
        r"All rights reserved.*", r"University.*",
        r"Editor.*", r"Department.*", r"Email:.*",
        r"\b\d{1,2}\s[A-Za-z]+\s\d{4}\b",
        r"Prelims\.indd.*",
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
from transformers import pipeline

# Initialize summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def generate_flashcard(texts, topic):
    if not texts:
        return "⚠️ No readable content found."

    full_text = " ".join(texts)
    chunks = chunk_text(full_text, max_words=200)

    if not chunks:
        return "⚠️ No meaningful content found."

    # Semantic relevance
    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)

    relevant_chunks = [
        c for c, vec in zip(chunks, chunk_vecs)
        if cosine_similarity([vec], topic_vec)[0][0] > 0.35
    ]

    if not relevant_chunks:
        relevant_chunks = chunks[:3]

    combined_text = " ".join(relevant_chunks)

    # Summarize using LLM
    try:
        summary = summarizer(
            combined_text,
            max_length=250,
            min_length=100,
            do_sample=False
        )[0]['summary_text']
    except:
        summary = combined_text  # fallback

    flashcard = f"""
### 📘 {topic} — Concept Summary

**Concept Overview & Explanation**  
{summary}

**Why It Matters**  
- Strengthens conceptual clarity  
- Explains how rights and concepts evolve  
- Important for NCERT & UPSC preparation
"""
    return flashcard

# ================= STREAMLIT UI =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard") and topic.strip():
    texts = load_all_text(subject)
    flashcard = generate_flashcard(texts, topic)
    st.markdown(flashcard)
