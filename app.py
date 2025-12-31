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

DEPTH_CONFIG = {
    "NCERT": {"top_k": 3, "similarity": 0.35},
    "UPSC": {"top_k": 6, "similarity": 0.45}
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

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    # extract nested zips
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            target = zfile.parent / zfile.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(target)
        except:
            pass

    st.success("✅ NCERT PDFs extracted")


# =====================================================
# TEXT CLEANING
# =====================================================
def clean_text(text):
    junk_patterns = [
        r"Prelims\.indd.*",
        r"ISBN.*",
        r"All rights reserved.*",
        r"Published.*",
        r"Printed.*",
        r"Reprint.*",
        r"Editor.*",
        r"University.*",
        r"Department.*",
        r"Copyright.*",
        r"\d{1,2}\s[A-Za-z]+\s\d{4}",
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
# LOAD TEXT BY SUBJECT
# =====================================================
def load_subject_text(subject):
    texts = []

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        try:
            text = read_pdf(pdf)
            if len(text.split()) > 300:   # real content check
                texts.append(text)
        except:
            continue

    return texts

# =====================================================
# CHUNKING
# =====================================================
def chunk_text(text, min_words=80, max_words=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], []

    for s in sentences:
        if len(s.split()) < 5:  # ignore tiny sentences
            continue
        current.append(s)
        total_words = sum(len(x.split()) for x in current)
        if total_words >= max_words:
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
# REMOVE DUPLICATES
# =====================================================
def deduplicate(chunks, threshold=0.85):
    if len(chunks) <= 1:
        return chunks

    embeds = model.encode(chunks)
    final = []

    for i, emb in enumerate(embeds):
        keep = True
        for kept in final:
            sim = cosine_similarity([emb], [kept[1]])[0][0]
            if sim > threshold:
                keep = False
                break
        if keep:
            final.append((chunks[i], emb))

    return [x[0] for x in final]


# =====================================================
# FLASHCARD GENERATION
# =====================================================
def generate_flashcards(chunks, topic):
    flashcards = []
    for c in chunks:
        sents = re.split(r'(?<=[.!?])\s+', c)
        if len(sents) < 2:
            continue
        overview = sents[0]
        explanation = " ".join(sents[1:4])
        flashcards.append({
            "overview": overview,
            "explanation": explanation
        })
    return flashcards
def summarize_flashcards(flashcards, topic):
    if not flashcards:
        return None
    overview = flashcards[0]["overview"]
    explanation = " ".join([f["explanation"] for f in flashcards[:4]])
    return f"""
### 📘 {topic} (Summary)
**Concept Overview**  
{overview}

**Explanation**  
{explanation}

**Why it Matters**
- Understand key rights & governance  
- Useful for UPSC & NCERT exams
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

        chunks = deduplicate(chunks)

        card = generate_flashcard(chunks, topic, depth)

        if card:
            st.markdown(card)
        else:
            st.warning("No relevant conceptual content found.")
