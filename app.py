import os, re, zipfile
from pathlib import Path
import streamlit as st
import gdown
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# CONFIG
# ======================================================
FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = {
    "Polity": ["polity", "constitution", "civics"],
    "Economics": ["economics", "economic"],
    "Sociology": ["sociology", "society"],
    "Psychology": ["psychology"],
    "Business Studies": ["business", "management"]
}

SIM_THRESHOLD = 0.45

# ======================================================
# STREAMLIT SETUP
# ======================================================
st.set_page_config(page_title="NCERT Flashcard Generator", layout="wide")
st.title("📘 NCERT Smart Flashcard Generator")

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ======================================================
# DOWNLOAD & EXTRACT
# ======================================================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT content...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    # extract nested zips
    for z in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            with zipfile.ZipFile(z, "r") as inner:
                inner.extractall(z.parent / z.stem)
        except:
            pass

# ======================================================
# TEXT CLEANING
# ======================================================
def clean_text(text):
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"(reprint|chapter|page|indd|isbn|copyright).*", "", text, flags=re.I)
    text = re.sub(r"WE, THE PEOPLE.*?CONSTITUTION", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_valid_sentence(s):
    if len(s.split()) < 8:
        return False
    garbage = ["email", "reprint", "page", "copyright"]
    return not any(g in s.lower() for g in garbage)

# ======================================================
# LOAD PDFs BY SUBJECT
# ======================================================
def load_subject_text(subject):
    texts = []
    keywords = SUBJECTS[subject]

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if not any(k in pdf.name.lower() for k in keywords):
            continue
        try:
            reader = PdfReader(pdf)
            text = " ".join(p.extract_text() or "" for p in reader.pages)
            text = clean_text(text)
            if len(text.split()) > 100:
                texts.append(text)
        except:
            pass

    return texts

# ======================================================
# SEMANTIC CHUNKING
# ======================================================
def chunk_text(text):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s for s in sentences if is_valid_sentence(s)]

    chunks = []
    current = []

    for s in sentences:
        current.append(s)
        if len(current) >= 5:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks

# ======================================================
# FLASHCARD GENERATION
# ======================================================
def generate_flashcards(chunks, topic):
    embeddings = model.encode(chunks)
    query_vec = model.encode([topic])

    scores = cosine_similarity(query_vec, embeddings)[0]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    good_chunks = [c for c, s in ranked if s > SIM_THRESHOLD][:6]

    flashcards = []
    for ch in good_chunks:
        sents = re.split(r"(?<=[.?!])\s+", ch)
        sents = [s for s in sents if is_valid_sentence(s)]
        if len(sents) >= 3:
            flashcards.append({
                "overview": sents[0],
                "explanation": " ".join(sents[1:4])
            })

    return flashcards

# ======================================================
# SUMMARY FLASHCARD
# ======================================================
def summarize_flashcards(cards, topic):
    if not cards:
        return None

    overview = cards[0]["overview"]
    explanation = " ".join([c["explanation"] for c in cards[:3]])

    return f"""
### 📘 {topic}

**Concept Overview**  
{overview}

**Explanation**  
{explanation}

**Conclusion**  
This concept plays a foundational role in understanding governance, rights, and social structure in a democratic system.
"""

# ======================================================
# UI
# ======================================================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)

    if not texts:
        st.warning("No readable content found for this subject.")
    else:
        chunks = []
        for t in texts:
            chunks.extend(chunk_text(t))

        flashcards = generate_flashcards(chunks, topic)
        final_card = summarize_flashcards(flashcards, topic)

        if final_card:
            st.markdown(final_card)
        else:
            st.warning("No meaningful content found for this topic.")
