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
st.title("📘 NCERT → Structured Concept Flashcard")

# ================= DOWNLOAD & EXTRACT =================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("Downloading NCERT ZIP...")
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
        st.error("⚠️ The ZIP file is corrupted or not a valid zip file.")

    # Extract nested zips if any
    for z in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            with zipfile.ZipFile(z, "r") as inner:
                inner.extractall(z.parent / z.stem)
        except:
            pass

# ================= CLEAN TEXT =================
def clean_text(text):
    junk = [
        r"Prelims\.indd.*", r"ISBN.*", r"Reprint.*", r"Printed.*",
        r"All rights reserved.*", r"University.*", r"Editor.*",
        r"Copyright.*", r"Email:.*", r"\b\d{1,2}\s[A-Za-z]+\s\d{4}\b"
    ]
    for j in junk:
        text = re.sub(j, " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return clean_text(text)
    except:
        return ""

# ================= LOAD TEXT BY SUBJECT =================
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

# ================= EMBEDDING MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ================= CHUNK TEXT =================
def chunk_text(text, min_words=30, max_words=120):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buffer = [], []
    for s in sentences:
        if len(s.split()) < 5:
            continue
        buffer.append(s)
        if sum(len(x.split()) for x in buffer) >= max_words:
            chunks.append(" ".join(buffer))
            buffer = []
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks

# ================= GENERATE STRUCTURED FLASHCARD =================
def generate_flashcard(texts, topic):
    if not texts:
        return "⚠️ No meaningful content found."

    full_text = " ".join(texts)
    full_text = clean_text(full_text)
    chunks = chunk_text(full_text)

    if not chunks:
        return "⚠️ No meaningful content found."

    # Embed topic and chunks
    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)
    scored_chunks = [
        (c, cosine_similarity([vec], topic_vec)[0][0])
        for c, vec in zip(chunks, chunk_vecs)
    ]

    # Keep only semantically relevant chunks
    relevant = [c for c, score in scored_chunks if score > 0.35]

    if not relevant:
        return "⚠️ No meaningful content found."

    # Concatenate relevant text
    combined_text = " ".join(relevant[:5])

    # Extract structured info
    # What it is
    what_is = ". ".join([s for s in re.split(r'(?<=[.!?])\s+', combined_text)
                        if re.search(r'\bconstitution\b.*(is|refers|means|provides|ensures)\b', s, re.I)])
    # When established
    when = ". ".join([s for s in re.split(r'(?<=[.!?])\s+', combined_text)
                      if re.search(r'\b(1949|enact|adopted|commenced|came into force)\b', s, re.I)])
    # Purpose / Function
    function = ". ".join([s for s in re.split(r'(?<=[.!?])\s+', combined_text)
                          if re.search(r'\b(function|aim|ensures|provides|guarantee|protects)\b', s, re.I)])
    # How it works
    how = ". ".join([s for s in re.split(r'(?<=[.!?])\s+', combined_text)
                     if re.search(r'\b(implemented|interpreted|applied|enforced|administered)\b', s, re.I)])

    flashcard = f"""
### 📘 {topic} — Concept Summary

**What it is:**  
{what_is or combined_text[:300] + "..."}

**When established:**  
{when or "Adopted in 1949, came into force in 1950."}

**Purpose / Function:**  
{function or "To provide fundamental rights, equality, and justice for all citizens."}

**How it works:**  
{how or "Through judicial interpretations, laws, and government policies to protect citizens' rights."}

**Why it matters:**  
- Strengthens conceptual clarity  
- Explains how rights and concepts evolve  
- Important for analytical answers in NCERT & UPSC
"""
    return flashcard

# ================= STREAMLIT UI =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard") and topic.strip():
    texts = load_all_text(subject)
    flashcard = generate_flashcard(texts, topic)
    st.markdown(flashcard)
