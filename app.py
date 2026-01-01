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
# ===================== RUN APP =====================

download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if topic:
    texts = load_subject_text(subject)

    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        result = generate_single_flashcard(
            texts=texts,
            topic=topic,
            depth=depth
        )

        st.markdown(result)

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
def generate_single_flashcard(texts, topic, depth="NCERT"):
    cleaned = []
    for t in texts:
        t = re.sub(r"(Prelims\.indd.*|ISBN.*|Printed.*|©.*|All rights reserved.*|University.*|Editor.*|Professor.*)", " ", t, flags=re.I)
        t = re.sub(r"\s+", " ", t)
        if len(t.split()) > 80:
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

    filtered = [
        c for c, v in zip(chunks, chunk_vecs)
        if cosine_similarity([v], topic_vec)[0][0] > threshold
    ]

    if not filtered:
        return "⚠️ No readable content found for this subject."

    summary_text = " ".join(filtered[:2])
    summary_sentences = re.split(r'(?<=[.!?])\s+', summary_text)

    return f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{' '.join(summary_sentences[:6])}

**Why It Matters**
- Builds conceptual clarity  
- Helps in real-world application  
- High-weight topic for NCERT & UPSC  
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

result = generate_single_flashcard(
    texts = load_subject_text(subject),
    topic=topic,
    depth=depth
)

st.markdown(result)
