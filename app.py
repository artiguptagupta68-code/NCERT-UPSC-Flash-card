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
    "Polity": ["constitution", "polity"],
    "Economics": ["economics"],
    "Sociology": ["sociology"],
    "Psychology": ["psychology"],
    "Business Studies": ["business"],
}

# =====================================================
# DOWNLOAD & EXTRACT
# =====================================================

def download_and_extract():
    if not Path(ZIP_PATH).exists():
        gdown.download(id=FILE_ID, output=ZIP_PATH, quiet=False)

    if not Path(EXTRACT_DIR).exists():
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(EXTRACT_DIR)

    # Extract nested zips
    for z in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            target = z.parent / z.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(z, 'r') as inner:
                inner.extractall(target)
        except:
            pass

# =====================================================
# CLEANING + FILTERING
# =====================================================

def is_conceptual(sentence: str) -> bool:
    s = sentence.lower()

    if len(s.split()) < 8:
        return False

    blacklist = [
        "isbn", "reprint", "printed", "publisher", "publication",
        "department", "editor", "email", "address", "price",
        "copyright", "reserved", "ncert", "press",
        "committee", "advisor", "secretary", "design",
        "distributed", "sold subject"
    ]

    if any(b in s for b in blacklist):
        return False

    concept_words = [
        "constitution", "rights", "liberty", "equality",
        "democracy", "citizen", "state", "government",
        "law", "justice", "freedom", "society", "political"
    ]

    return any(w in s for w in concept_words)


# =====================================================
# PDF READING
# =====================================================

def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + " "
        return text
    except:
        return ""


def load_subject_text(subject):
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = read_pdf(pdf)
        if len(text.split()) > 300:
            texts.append(text)
    return texts


# =====================================================
# MODEL
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# =====================================================
# FLASHCARD GENERATION
# =====================================================

def generate_single_flashcard(texts, topic, depth="NCERT"):
    cleaned = []
    for t in texts:
        sents = re.split(r'(?<=[.!?])\s+', t)
        for s in sents:
            if is_conceptual(s):
                cleaned.append(s.strip())

    if not cleaned:
        return "⚠️ No readable conceptual content found."

    cleaned = list(dict.fromkeys(cleaned))[:300]

    embeddings = model.encode(cleaned)
    topic_vec = model.encode([topic])

    threshold = 0.35 if depth == "NCERT" else 0.45

    scored = [
        (s, cosine_similarity([e], topic_vec)[0][0])
        for s, e in zip(cleaned, embeddings)
    ]

    selected = [s for s, score in scored if score > threshold]

    if not selected:
        return "⚠️ No relevant conceptual explanation found."

    selected = selected[:5]

    return f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{selected[0]}

**Explanation**  
{' '.join(selected[1:4])}

**Why It Matters**  
This concept forms a core foundation of Indian polity and is frequently tested in NCERT and UPSC examinations.
"""


# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config("NCERT Flashcards", layout="wide")
st.title("📘 NCERT + UPSC Smart Flashcard Generator")

download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)
    result = generate_single_flashcard(texts, topic, depth)
    st.markdown(result)
