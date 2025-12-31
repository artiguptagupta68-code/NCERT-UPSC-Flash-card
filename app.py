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

# =====================================================
# CONFIG
# =====================================================
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
    "NCERT": {"chunk_size": 3, "similarity": 0.35},
    "UPSC": {"chunk_size": 6, "similarity": 0.45}
}

# =====================================================
# STREAMLIT SETUP
# =====================================================
st.set_page_config(page_title="NCERT + UPSC Flashcards", layout="wide")
st.title("📘 NCERT + UPSC Smart Flashcard Generator")

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =====================================================
# UTILITIES
# =====================================================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(
            url=f"https://drive.google.com/uc?id={FILE_ID}",
            output=ZIP_PATH,
            quiet=False,
            fuzzy=True
        )

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Extract nested zips safely
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            target = zfile.parent / zfile.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(target)
        except Exception as e:
            st.warning(f"Skipped corrupted zip: {zfile}")

    st.success("✅ NCERT PDFs extracted successfully")

def read_pdf(path):
    try:
        reader = PdfReader(str(path))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except:
        return ""

def is_valid_sentence(s):
    if len(s.split()) < 5:
        return False
    if re.search(r"(Email|Chief Editor|Production|Cover|Illustrations|stamps|price|rubber stamp|ISBN|copyright|Reprint)", s, re.I):
        return False
    if s.isupper() and len(s.split()) < 8:
        return False
    if len(re.findall(r"[•●]", s)) > 0:
        return False
    return True

def clean_text(text):
    text = re.sub(r"(activity|exercise|project|reprint|isbn|copyright).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_subject_text(subject):
    texts = []

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = clean_text(read_pdf(pdf))
        if len(text.split()) > 80:  # load only if enough text
            texts.append(text)

    return texts

 

def chunk_text(text, chunk_size=5):
    sentences = [s for s in re.split(r"(?<=[.?!])\s+", text) if is_valid_sentence(s)]
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i+chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks

def generate_flashcards(chunks, topic, depth="NCERT"):
    if not chunks:
        return []

    embeddings = model.encode(chunks)
    query_vec = model.encode([topic])
    sims = cosine_similarity(query_vec, embeddings)[0]

    threshold = DEPTH_CONFIG[depth]["similarity"]
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    selected_chunks = [c for c, s in ranked if s >= threshold]

    flashcards = []
    for ch in selected_chunks:
        sents = [s for s in re.split(r"(?<=[.?!])\s+", ch) if is_valid_sentence(s)]
        if len(sents) >= 2:
            flashcards.append({
                "overview": sents[0],
                "explanation": " ".join(sents[1:4]),
                "key_points": sents[:3]
            })
    return flashcards

def summarize_flashcards(flashcards, topic, depth="NCERT"):
    if not flashcards:
        return None

    overview = flashcards[0]["overview"]
    explanation = " ".join([f["explanation"] for f in flashcards[:5]])
    key_points = []
    for f in flashcards[:5]:
        key_points.extend(f["key_points"])

    if depth=="NCERT":
        return f"""
### 📘 {topic} (NCERT Summary)

**Concept Overview**  
{overview}

**Explanation**  
{explanation}

**Key Points**  
- {"\n- ".join(key_points)}
"""
    else:
        return f"""
### 📘 {topic} (UPSC Summary)

**Introduction**  
{overview}

**Analytical Explanation**  
{explanation}

**Key Points**  
- {"\n- ".join(key_points)}

**Conclusion**  
This topic is central to governance, constitutionalism, and democratic functioning in India.
"""

# =====================================================
# STREAMLIT UI
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
        all_chunks = []
        for t in texts:
            all_chunks.extend(chunk_text(t, chunk_size=DEPTH_CONFIG[depth]["chunk_size"]))

        flashcards = generate_flashcards(all_chunks, topic, depth)
        summary_card = summarize_flashcards(flashcards, topic, depth)

        if summary_card:
            st.markdown(summary_card)
        else:
            st.warning("No meaningful content found for this topic.")
