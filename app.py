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

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    for z in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            with zipfile.ZipFile(z, "r") as inner:
                inner.extractall(z.parent / z.stem)
        except:
            pass


# ================= TEXT CLEANING =================
def clean_text(text):
    patterns = [
        r"ISBN.*", r"Reprint.*", r"Printed.*", r"©.*",
        r"All rights reserved.*", r"University.*",
        r"Editor.*", r"Department.*", r"Email.*",
        r"Prelims\.indd.*",
        r"\b\d{1,2}\s[A-Za-z]+\s\d{4}\b"
    ]

    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================= PDF READER =================
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


# ================= LOAD ALL TEXT =================
def load_all_text(subject):
    texts = []
    keywords = SUBJECTS.get(subject, [])

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if any(k in str(pdf).lower() for k in keywords):
            content = read_pdf(pdf)
            if len(content.split()) > 100:
                texts.append(content)

    return texts


# ================= EMBEDDING MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ================= FLASHCARD LOGIC =================
def generate_flashcard(texts, topic):
    combined = " ".join(texts)
    combined = clean_text(combined)

    if len(combined.split()) < 120:
        return "⚠️ No meaningful content found."

    sentences = re.split(r'(?<=[.!?])\s+', combined)

    chunks, buffer = [], []
    for s in sentences:
        if len(s.split()) < 6:
            continue
        buffer.append(s)
        if sum(len(x.split()) for x in buffer) >= 140:
            chunks.append(" ".join(buffer))
            buffer = []

    if buffer:
        chunks.append(" ".join(buffer))

    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)

    relevant = [
        c for c, v in zip(chunks, chunk_vecs)
        if cosine_similarity([v], topic_vec)[0][0] > 0.35
    ]

    if not relevant:
        return "⚠️ No meaningful content found."

    content = " ".join(relevant[:2])

    return f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{content}

**Why It Matters**
- Builds conceptual clarity  
- Helps in analytical and application-based questions  
- Important for NCERT & UPSC preparation  
"""


# ================= STREAMLIT UI =================
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
