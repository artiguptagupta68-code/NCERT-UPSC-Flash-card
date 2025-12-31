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
    keywords = SUBJECTS[subject]

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        path_lower = str(pdf).lower()
        if any(k in path_lower for k in keywords):
            content = read_pdf(pdf)
            if len(content.split()) > 150:
                texts.append(content)

    return texts


# =====================================================
# CHUNKING
# =====================================================
def chunk_text(text, min_words=50, max_words=120):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], []

    for s in sentences:
        if len(s.split()) < 6:
            continue

        current.append(s)

        if sum(len(x.split()) for x in current) >= max_words:
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
def generate_flashcard(chunks, topic, depth):
    if not chunks:
        return None

    embeddings = model.encode(chunks)
    query = model.encode([topic])

    sims = cosine_similarity(query, embeddings)[0]
    cfg = DEPTH_CONFIG[depth]

    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    selected = [c for c, s in ranked if s >= cfg["similarity"]][:cfg["top_k"]]

    if not selected:
        return None

    if depth == "NCERT":
        return f"""
### 📘 {topic} — NCERT

**Concept Overview**  
{selected[0]}

**Explanation**  
{" ".join(selected[1:3])}

**Key Takeaway**  
This concept explains the basic principles necessary for understanding Indian democracy and governance.
"""

    else:
        return f"""
### 📘 {topic} — UPSC

**Introduction**  
{selected[0]}

**Analytical Explanation**  
{" ".join(selected[1:4])}

**Why It Matters**  
This topic is crucial for understanding constitutional philosophy, governance, and contemporary issues.

**Exam Tip**  
Focus on interpretation, examples, and linkage with current affairs.
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
