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
st.title("📘 NCERT → Smart Concept Flashcard + Active Learning")

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
        except Exception:
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
    except Exception:
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
    if not texts:
        return "⚠️ No readable content found."

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

    topic_query = (
        f"Explain the concept of {topic} as defined in NCERT textbooks, "
        f"including definition, features, importance and examples."
    )

    topic_vec = model.encode([topic_query])
    chunk_vecs = model.encode(chunks)

    scored = []
    for chunk, vec in zip(chunks, chunk_vecs):
        score = cosine_similarity([vec], topic_vec)[0][0]
        if score > 0.40:
            scored.append((chunk, score))

    if not scored:
        scores = cosine_similarity(chunk_vecs, topic_vec).flatten()
        scored = list(zip(chunks, scores))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_chunks = [c for c, _ in scored[:3]]

    joined = " ".join(best_chunks)
    joined = re.sub(r"(ISBN.*|Reprint.*|Printed.*|All rights reserved.*)", " ", joined)
    joined = re.sub(r"\s+", " ", joined)

    sentences = re.split(r'(?<=[.!?])\s+', joined)
    concept = " ".join(sentences[:4])
    explanation = " ".join(sentences[4:8])

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

# ================= ACTIVE LEARNING LOGIC =================
def extract_keywords(text, top_k=10):
    """
    Extract important topic words using frequency + structure.
    Increased top_k for more fill-in-the-blanks.
    """
    words = re.findall(r"\b[A-Za-z]{4,}\b", text)
    stopwords = {
        "that", "this", "with", "from", "which", "their",
        "these", "there", "where", "when", "whose", "while"
    }
    words = [w for w in words if w.lower() not in stopwords]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    keywords = sorted(freq, key=freq.get, reverse=True)
    return keywords[:top_k]

def generate_active_learning_card(texts, topic):
    """
    Active learning mode:
    - Topic visible
    - Basic info visible
    - More important words hidden (fill-in-the-blanks)
    """
    if not texts:
        return "⚠️ No readable content found."

    # Semantic retrieval
    topic_query = (
        f"Explain the concept of {topic} as defined in NCERT textbooks, "
        f"including definition and importance."
    )

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

    topic_vec = model.encode([topic_query])
    chunk_vecs = model.encode(chunks)
    scores = cosine_similarity(chunk_vecs, topic_vec).flatten()
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    base_text = " ".join(c for c, _ in ranked[:2])
    base_text = re.sub(r"\s+", " ", base_text)

    # Extract more keywords
    keywords = extract_keywords(base_text, top_k=10)

    masked_text = base_text
    for kw in keywords:
        # Mask all occurrences for extra blanks
        masked_text = re.sub(rf"\b{kw}\b", "_____", masked_text)

    return f"""
### 🧠 Active Learning: {topic}

**Fill in the missing important terms**

{masked_text}

---
💡 Answers are directly from NCERT content.
"""

# ================= UI =================
download_and_extract()
texts = load_all_text()

tab1, tab2 = st.tabs(["📘 Flashcard", "🧠 Active Learning"])

with tab1:
    topic = st.text_input(
        "Enter Topic (e.g. Fundamental Rights, Preamble, Constitution)",
        key="flashcard_topic"
    )
    if st.button("Generate Flashcard"):
        result = generate_flashcard(texts, topic)
        st.markdown(result)

with tab2:
    topic_al = st.text_input(
        "Enter Topic for Active Learning",
        key="active_topic"
    )
    if st.button("Start Active Learning"):
        active_result = generate_active_learning_card(texts, topic_al)
        st.markdown(active_result)
