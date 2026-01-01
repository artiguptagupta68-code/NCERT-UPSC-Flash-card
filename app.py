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
    if not texts:
        return "⚠️ No readable content found."

    # Chunk
    chunks = []
    for t in texts:
        sents = re.split(r'(?<=[.!?])\s+', t)
        buf = []
        for s in sents:
            if len(s.split()) < 6:
                continue
            buf.append(s)
            if len(" ".join(buf).split()) > 140:
                chunks.append(" ".join(buf))
                buf = []
        if buf:
            chunks.append(" ".join(buf))

    if not chunks:
        return "⚠️ No meaningful content found."

    # Similarity ranking
    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)

    scored = [
        (chunk, cosine_similarity([vec], topic_vec)[0][0])
        for chunk, vec in zip(chunks, chunk_vecs)
    ]

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [c for c, _ in scored[:3]]

    summary = " ".join(selected)
    sentences = re.split(r'(?<=[.!?])\s+', summary)

    return f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{' '.join(sentences[:5])}

**Explanation**  
{' '.join(sentences[5:10])}

**Why It Matters**
- Strengthens constitutional understanding  
- Frequently tested in UPSC & NCERT  
- Helps connect theory with real-life governance
"""


# ================= UI =================
download_and_extract()

topic = st.text_input("Enter Topic (e.g. Fundamental Rights, Preamble, Constitution)")

if st.button("Generate Flashcard"):
    texts = load_all_text()
    result = generate_flashcard(texts, topic)
    st.markdown(result)
