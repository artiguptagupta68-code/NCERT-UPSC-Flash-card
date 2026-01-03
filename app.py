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
def load_all_text():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        txt = read_pdf(pdf)
        if len(txt.split()) > 200:
            texts.append(txt)
    return texts


# ================= EMBEDDINGS =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ================= FLASHCARD LOGIC =================
def generate_flashcard(texts, topic):
    """
    1. Read all text
    2. Find semantically relevant chunks
    3. Compress meaning
    4. Generate clean conceptual flashcard
    """

    if not texts:
        return "⚠️ No readable content found."

    # -------- STEP 1: CHUNKING --------
    chunks = []
    for text in texts:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        buffer = []

        for s in sentences:
            if len(s.split()) < 6:
                continue
            buffer.append(s)

            if len(" ".join(buffer).split()) >= 80:
                chunks.append(" ".join(buffer))
                buffer = []

        if buffer:
            chunks.append(" ".join(buffer))

    if not chunks:
        return "⚠️ No meaningful content found."

    # -------- STEP 2: SEMANTIC MATCHING --------
   topic_query = f"""
   Explain the concept of {topic} as defined in NCERT textbooks,
   including definition, features, importance and examples.
   """

    topic_vec = model.encode([topic_query])
    
    chunk_vecs = model.encode(chunks)

    scored = []
    for chunk, vec in zip(chunks, chunk_vecs):
        score = cosine_similarity([vec], topic_vec)[0][0]
        if score > 0.40:   # semantic relevance threshold
            scored.append((chunk, score))

    if not scored:
        return "⚠️ Topic not found in NCERT content."

   
    scored.sort(key=lambda x: x[1], reverse=True)

# fallback if threshold filtering fails
    if not scored:
    scored = list(zip(chunks, 
                      cosine_similarity(chunk_vecs, topic_vec).flatten()))
    scored.sort(key=lambda x: x[1], reverse=True)

    best_chunks = [c for c, _ in scored[:3]]

    # -------- STEP 3: UNDERSTAND & SUMMARIZE --------
    joined = " ".join(best_chunks)

    # Clean remaining junk
    joined = re.sub(r"(ISBN.*|Reprint.*|Printed.*|All rights reserved.*)", " ", joined)
    joined = re.sub(r"\s+", " ", joined)

    sentences = re.split(r'(?<=[.!?])\s+', joined)

    concept = " ".join(sentences[:4])
    explanation = " ".join(sentences[4:8])

    # -------- STEP 4: STRUCTURED FLASHCARD --------
    flashcard = f"""
### 📘 {topic} — Concept Summary

**Concept Overview**
{concept}

**Explanation**
{explanation}

**Why It Matters**
- Builds conceptual clarity
- Helps in analytical and application-based questions
- Important for NCERT & UPSC preparation
"""

    return flashcard




# ================= UI =================
download_and_extract()

topic = st.text_input("Enter Topic (e.g. Fundamental Rights, Preamble, Constitution)")

if st.button("Generate Flashcard"):
    texts = load_all_text()
    result = generate_flashcard(texts, topic)
    st.markdown(result)
