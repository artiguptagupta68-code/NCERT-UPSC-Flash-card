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
    # Remove emails, page numbers, editor names, multiple caps, headers/footers
    text = re.sub(r"\S+@\S+", " ", text)  # emails
    text = re.sub(r"(Prelims\.indd|Page \d+|Reprint).*", " ", text)
    text = re.sub(r"[A-Z]{2,}", " ", text)  # ALL CAPS words
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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
        if any(k in pdf.name.lower() for k in keywords):
            text = clean_text(read_pdf(pdf))
            if len(text.split()) > 80:
                texts.append(text)
    return texts

# ===================== CHUNKING =====================
def chunk_text(text, chunk_size=5):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s for s in sentences if len(s.split()) > 5]

    chunks = []
    current = []
    for s in sentences:
        current.append(s)
        if len(current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

# ===================== LOAD EMBEDDING =====================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# ===================== FLASHCARD GENERATION =====================
def generate_flashcards(chunks, topic, depth):
    if not chunks:
        return []

    embeddings = model.encode(chunks)
    query = model.encode([topic])
    sims = cosine_similarity(query, embeddings)[0]

    threshold = DEPTH_CONFIG[depth]["similarity"]
    top_k = DEPTH_CONFIG[depth]["top_k"]

    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    selected = [c for c, s in ranked if s >= threshold][:top_k]

    flashcards = []
    for c in selected:
        sents = re.split(r"(?<=[.?!])\s+", c)
        if len(sents) >= 2:
            flashcards.append({
                "overview": sents[0],
                "explanation": " ".join(sents[1:4])
            })
    return flashcards

# ===================== SUMMARIZATION =====================
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

summarizer = load_summarizer()

def summarize_flashcards(cards, topic):
    if not cards:
        return None
    combined_text = " ".join([c["overview"] + ". " + c["explanation"] for c in cards])
    summary = summarizer(
        combined_text, max_length=250, min_length=100, do_sample=False
    )[0]["summary_text"]
    return f"### 📘 {topic} (Summarized)\n\n{summary}"

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
