import os
import re
import zipfile
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# CONFIG
# =====================================================
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

# =====================================================
# STREAMLIT SETUP
# =====================================================
st.set_page_config("NCERT + UPSC Flashcards", layout="wide")
st.title("📘 NCERT + UPSC Smart Flashcard Generator")


# =====================================================
# DOWNLOAD & EXTRACT
# =====================================================
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

    # Extract nested zips
    for z in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            target = z.parent / z.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(z, "r") as inner:
                inner.extractall(target)
        except:
            pass


# =====================================================
# CLEANING FUNCTIONS
# =====================================================
def clean_text(text):
    junk_patterns = [
        r"Prelims\.indd.*", r"ISBN.*", r"Printed.*",
        r"All rights reserved.*", r"University.*",
        r"Editor.*", r"Professor.*", r"©.*",
        r"\d{1,2}\s[A-Za-z]+\s\d{4}"
    ]
    for pat in junk_patterns:
        text = re.sub(pat, " ", text, flags=re.I)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + " "
        return clean_text(text)
    except:
        return ""


# =====================================================
# LOAD TEXT
# =====================================================
def load_subject_text(subject):
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = read_pdf(pdf)
        if len(text.split()) > 300:
            texts.append(text)
    return texts


# =====================================================
# EMBEDDING MODEL
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# =====================================================
# FLASHCARD GENERATION
# =====================================================
def generate_single_flashcard(texts, topic, depth):
    cleaned = []
    for t in texts:
        if len(t.split()) > 100:
            cleaned.append(t)

    if not cleaned:
        return "⚠️ No readable content found for this subject."

    full_text = " ".join(cleaned)

    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    chunks, buf = [], []
    for s in sentences:
        if len(s.split()) < 6:
            continue
        buf.append(s)
        if sum(len(x.split()) for x in buf) >= 180:
            chunks.append(" ".join(buf))
            buf = []

    if buf:
        chunks.append(" ".join(buf))

    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)

    threshold = 0.35 if depth == "NCERT" else 0.45
    selected = [
        c for c, v in zip(chunks, chunk_vecs)
        if cosine_similarity([v], topic_vec)[0][0] > threshold
    ]

    if not selected:
        return "⚠️ No readable content found for this subject."

    summary = " ".join(selected[:2])
    sentences = re.split(r'(?<=[.!?])\s+', summary)

    return f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{' '.join(sentences[:6])}

**Why It Matters**
- Builds conceptual clarity  
- Helps connect theory with real life  
- Frequently asked in UPSC & NCERT  
"""


# =====================================================
# UI EXECUTION
# =====================================================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)
    result = generate_single_flashcard(texts, topic, depth)
    st.markdown(result)
