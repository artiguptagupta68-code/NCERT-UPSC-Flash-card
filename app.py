import os
import re
import zipfile
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================
# CONFIG
# ============================
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

# ============================
# STREAMLIT SETUP
# ============================
st.set_page_config("NCERT Smart Flashcards", layout="wide")
st.title("📘 NCERT Concept Flashcard Generator")


# ============================
# DOWNLOAD & EXTRACT
# ============================
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

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(zfile.parent / zfile.stem)
        except:
            pass


# ============================
# TEXT CLEANING
# ============================
def clean_text(text):
    garbage = [
        r"ISBN.*", r"Reprint.*", r"Printed.*", r"©.*",
        r"All rights reserved.*", r"University.*",
        r"Editor.*", r"Department.*", r"Email.*",
        r"Prelims\.indd.*", r"\d{1,2}\s[A-Za-z]+\s\d{4}"
    ]

    for g in garbage:
        text = re.sub(g, " ", text, flags=re.I)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            if p.extract_text():
                text += p.extract_text() + " "
        return clean_text(text)
    except:
        return ""


# ============================
# LOAD SUBJECT TEXT
# ============================
def load_all_text(subject):
    texts = []
    keywords = SUBJECTS.get(subject, [])

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if any(k in str(pdf).lower() for k in keywords):
            txt = read_pdf(pdf)
            if len(txt.split()) > 100:
                texts.append(txt)

    return texts


# ============================
# EMBEDDING MODEL
# ============================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ============================
# FLASHCARD GENERATOR
# ============================
def generate_flashcard(texts, topic):
    combined = " ".join(texts)
    combined = clean_text(combined)

    if len(combined.split()) < 150:
        return "⚠️ No meaningful content found."

    # Sentence split
    sentences = re.split(r'(?<=[.!?])\s+', combined)

    chunks, buffer = [], []
    for s in sentences:
        if len(s.split()) < 6:
            continue
        buffer.append(s)
        if sum(len(x.split()) for x in buffer) >= 150:
            chunks.append(" ".join(buffer))
            buffer = []

    if buffer:
        chunks.append(" ".join(buffer))

    # Semantic filtering
    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)

    relevant = [
        c for c, v in zip(chunks, chunk_vecs)
        if cosine_similarity([v], topic_vec)[0][0] > 0.75
    ]

    if not relevant:
        return "⚠️ No meaningful content found."

    summary = " ".join(relevant[:2])

    return f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{summary}

**Why It Matters**
- Strengthens conceptual clarity  
- Connects constitutional theory with real life  
- Frequently tested in NCERT & UPSC exams  
"""


# ============================
# APP UI
# ============================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_all_text(subject)

    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        flashcard = generate_flashcard(texts, topic)
        st.markdown(flashcard)
