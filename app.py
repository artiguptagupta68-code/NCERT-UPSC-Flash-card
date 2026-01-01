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


# ================= CLEANING =================
def clean_text(text):
    junk = [
        r"Prelims\.indd.*", r"ISBN.*", r"Reprint.*", r"Printed.*",
        r"All rights reserved.*", r"University.*", r"Editor.*",
        r"Copyright.*", r"\d{1,2}\s[A-Za-z]+\s\d{4}"
    ]
    for j in junk:
        text = re.sub(j, " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = " ".join(p.extract_text() or "" for p in reader.pages)
        return clean_text(text)
    except:
        return ""


# ================= LOAD ALL TEXT =================
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


# ================= EMBEDDINGS =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ===================== CHUNKING & SEMANTIC FILTER =====================
def classify_sentences(sentences):
    definition, working, importance = [], [], []

    for s in sentences:
        s_low = s.lower()
        if any(k in s_low for k in ["means", "refers to", "is defined as", "is the"]):
            definition.append(s)
        elif any(k in s_low for k in ["works", "interpreted", "implemented", "ensures", "provides"]):
            working.append(s)
        elif any(k in s_low for k in ["important", "ensures", "protects", "helps", "essential"]):
            importance.append(s)

    return definition, working, importance


# ===================== LOAD EMBEDDING MODEL =====================
# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text):
    """Remove garbage, metadata, and noise"""
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


def chunk_text(text, min_words=30, max_words=120):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buffer = [], []

    for s in sentences:
        if len(s.split()) < 6:
            continue
        buffer.append(s)
        if sum(len(x.split()) for x in buffer) >= max_words:
            chunks.append(" ".join(buffer))
            buffer = []

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks


def generate_flashcard(text, topic):
    text = clean_text(text)
    chunks = chunk_text(text)

    if not chunks:
        return "⚠️ No meaningful content found."

    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)

    scored = [
        (chunk, cosine_similarity([vec], topic_vec)[0][0])
        for chunk, vec in zip(chunks, chunk_vecs)
    ]

    # Keep only relevant chunks
    relevant = [c for c, s in scored if s > 0.35]

    if not relevant:
        return "⚠️ No meaningful content found."

    combined = " ".join(relevant[:3])

    return f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{combined}

**Why It Matters**
- Strengthens understanding of constitutional principles  
- Explains how rights evolve with society  
- Important for analytical answers in NCERT & UPSC  

"""


# ================== EXAMPLE USAGE ==================

text_input = """
The fundamental rights guaranteed by the Constitution are dynamic in nature.
They evolve through judicial interpretation to address new challenges.
The right to life has expanded to include dignity, livelihood, and privacy.
The Constitution ensures democratic values and inclusive citizenship.
"""

topic = "Constitution"

flashcard = generate_flashcard(text_input, topic)
print(flashcard)


# ===================== STREAMLIT UI =====================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard") and topic.strip():
    texts = load_all_text(subject)
    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        flashcard = build_flashcard(texts, topic)
        st.markdown(flashcard)
