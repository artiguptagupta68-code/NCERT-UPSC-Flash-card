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
def generate_single_flashcard(texts, topic, depth="NCERT"):
    if not texts:
        return "⚠️ No readable content found."

    # ---------- STEP 1: CLEAN & CHUNK ----------
    chunks = []
    for text in texts:
        text = re.sub(r"(ISBN.*|Reprint.*|Printed.*|All rights reserved.*|University.*)", " ", text)
        text = re.sub(r"\s+", " ", text)

        sentences = re.split(r'(?<=[.!?])\s+', text)
        buffer = []

        for s in sentences:
            if len(s.split()) < 7:
                continue
            buffer.append(s)

            if len(" ".join(buffer).split()) >= 150:
                chunks.append(" ".join(buffer))
                buffer = []

        if buffer:
            chunks.append(" ".join(buffer))

    if not chunks:
        return "⚠️ No meaningful content found."

    # ---------- STEP 2: SEMANTIC FILTER ----------
    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)

    scored = []
    for c, v in zip(chunks, chunk_vecs):
        score = cosine_similarity([v], topic_vec)[0][0]
        if score > 0.35:
            scored.append((c, score))

    if not scored:
        return "⚠️ Topic not found in NCERT content."

    # Sort by relevance
    scored.sort(key=lambda x: x[1], reverse=True)

    # ---------- STEP 3: MERGE CONTEXT (IMPORTANT FIX) ----------
    merged_context = " ".join([c for c, _ in scored[:5]])

    # Remove dangling transitions like "Secondly", "Firstly"
    merged_context = re.sub(
        r"\b(Firstly|Secondly|Thirdly|Moreover|Further|Additionally),?\b",
        "",
        merged_context,
        flags=re.I
    )

    # ---------- STEP 4: STRUCTURED SUMMARIZATION ----------
    sentences = re.split(r'(?<=[.!?])\s+', merged_context)

    concept = " ".join(sentences[:4])
    explanation = " ".join(sentences[4:9])

    return f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{concept}

**Explanation**  
{explanation}

**Why It Matters**
- Helps understand evolving constitutional interpretation  
- Strengthens analytical thinking for UPSC & NCERT  
- Connects theory with real-life governance  
"""

 

# ================= UI =================
download_and_extract()

topic = st.text_input("Enter Topic (e.g. Fundamental Rights, Preamble, Constitution)")

if st.button("Generate Flashcard"):
    texts = load_all_text()
    result = generate_single_flashcard(texts, topic)
    st.markdown(result)
