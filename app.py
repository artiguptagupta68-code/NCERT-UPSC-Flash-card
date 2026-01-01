import os
import re
import zipfile
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== CONFIG =====================
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
    "NCERT": {"similarity": 0.35},
    "UPSC": {"similarity": 0.45}
}

# ===================== STREAMLIT SETUP =====================
st.set_page_config("NCERT + UPSC Flashcards", layout="wide")
st.title("📘 NCERT + UPSC Smart Flashcard Generator")

# ===================== DOWNLOAD & EXTRACT =====================
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
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    except zipfile.BadZipFile:
        st.error("❌ The downloaded file is not a valid ZIP.")

    # Extract nested ZIPs
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        target = zfile.parent / zfile.stem
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(zfile, "r") as inner:
            inner.extractall(target)
    st.success("✅ NCERT PDFs extracted successfully")

# ===================== TEXT CLEANING =====================
def clean_text(text: str) -> str:
    patterns = [
        r"Prelims\.indd.*", r"ISBN.*", r"©.*", r"All rights reserved.*",
        r"Printed.*", r"Reprint.*", r"Editor.*", r"Professor.*",
        r"University.*", r"Department.*", r"\d{1,2}:\d{2}:\d{2}"
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return clean_text(text)
    except:
        return ""

def load_subject_text(subject):
    texts = []
    keywords = SUBJECTS[subject]
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        pdf_name = str(pdf).lower()
        if any(k in pdf_name for k in keywords):
            text = read_pdf(pdf)
            if len(text.split()) > 100:
                texts.append(text)
    return texts

# ===================== MODEL =====================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ===================== CHUNKING =====================
def chunk_text(text, min_words=50, max_words=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buf = [], []
    for s in sentences:
        if len(s.split()) < 5:
            continue
        buf.append(s)
        total = sum(len(x.split()) for x in buf)
        if total >= max_words:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks

# ===================== DEDUPLICATION =====================
def deduplicate_chunks(chunks, threshold=0.85):
    if len(chunks) <= 1:
        return chunks
    embeds = model.encode(chunks)
    keep = []
    for i, emb in enumerate(embeds):
        is_dup = False
        for k in keep:
            sim = cosine_similarity([emb], [k[1]])[0][0]
            if sim > threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append((chunks[i], emb))
    return [x[0] for x in keep]

# ===================== FLASHCARD GENERATION =====================
def generate_flashcard(texts, topic, depth="NCERT"):
    all_chunks = []
    for t in texts:
        all_chunks.extend(chunk_text(t))
    all_chunks = deduplicate_chunks(all_chunks)

    if not all_chunks:
        return "⚠️ No readable content found for this subject."

    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(all_chunks)
    threshold = DEPTH_CONFIG[depth]["similarity"]

    relevant_chunks = [
        c for c, v in zip(all_chunks, chunk_vecs)
        if cosine_similarity([v], topic_vec)[0][0] > threshold
    ]

    if not relevant_chunks:
        return "⚠️ No meaningful content found for this topic."

    # Take top 2–3 chunks for a single elaborative flashcard
    summary_text = " ".join(relevant_chunks[:3])
    summary_sentences = re.split(r'(?<=[.!?])\s+', summary_text)

    return f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{' '.join(summary_sentences[:6])}

**Explanation**  
{' '.join(summary_sentences[6:12]) if len(summary_sentences)>6 else ''}

**Why It Matters**  
- Builds conceptual clarity  
- Connects theory with real life  
- High-weight topic for NCERT & UPSC

**Exam Tip**  
Focus on interpretation rather than memorization.
"""

# ===================== UI =====================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard") and topic:
    texts = load_subject_text(subject)
    flashcard = generate_flashcard(texts, topic, depth)
    st.markdown(flashcard)
