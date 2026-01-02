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

# ================= STREAMLIT UI =================
st.set_page_config(page_title="NCERT Flashcard Generator", layout="wide")
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


# ================= TEXT CLEANING =================
def clean_text(text):
    patterns = [
        r"ISBN.*", r"Reprint.*", r"Printed.*", r"©.*",
        r"All rights reserved.*", r"University.*",
        r"Editor.*", r"Department.*", r"Email:.*",
        r"\b\d{1,2}\s[A-Za-z]+\s\d{4}\b",
        r"Prelims\.indd.*"
    ]

    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)

    return re.sub(r"\s+", " ", text).strip()


def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = " ".join(p.extract_text() or "" for p in reader.pages)
        return clean_text(text)
    except:
        return ""


# ================= LOAD TEXT =================
def load_all_text(subject):
    texts = []
    keywords = SUBJECTS.get(subject, [])

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if any(k in str(pdf).lower() for k in keywords):
            content = read_pdf(pdf)
            if len(content.split()) > 60:
                texts.append(content)

    return texts


# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ================= FLASHCARD LOGIC =================
def generate_flashcard(texts, topic):
    combined = clean_text(" ".join(texts))

    if len(combined.split()) < 80:
        return "⚠️ No meaningful content found."

    sentences = re.split(r'(?<=[.!?])\s+', combined)

    # Chunking
    chunks, buffer = [], []
    for s in sentences:
        if len(s.split()) < 6:
            continue
        buffer.append(s)
        if sum(len(x.split()) for x in buffer) >= 120:
            chunks.append(" ".join(buffer))
            buffer = []
    if buffer:
        chunks.append(" ".join(buffer))

    # Semantic scoring
    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)

    scored = [
        (chunk, cosine_similarity([vec], topic_vec)[0][0])
        for chunk, vec in zip(chunks, chunk_vecs)
    ]

    if not scored:
        return "⚠️ No meaningful content found."

    scored.sort(key=lambda x: x[1], reverse=True)
    top_text = " ".join([c for c, _ in scored[:3]])

    # -------- STRUCTURED EXTRACTION --------
    sentences = re.split(r'(?<=[.!?])\s+', top_text)

    what, when, how, why = [], [], [], []

    for s in sentences:
        s_low = s.lower()

        if any(k in s_low for k in ["is", "refers to", "means", "defined as"]):
            what.append(s)

        if any(k in s_low for k in ["adopted", "enacted", "came into force", "1949", "1950"]):
            when.append(s)

        if any(k in s_low for k in ["works", "functions", "implemented", "interpreted", "enforced"]):
            how.append(s)

        if any(k in s_low for k in ["important", "ensures", "protects", "essential", "significant"]):
            why.append(s)

    return f"""
### 📘 {topic} — Concept Summary

**What is it?**  
{ " ".join(what) if what else top_text[:300] + "..." }

**When was it established?**  
{ " ".join(when) if when else "Established through constitutional development and amendments." }

**How does it work?**  
{ " ".join(how) if how else "It operates through laws, institutions, and judicial interpretation." }

**Why is it important?**  
{ " ".join(why) if why else "It strengthens democracy, rights, and governance." }
"""


# ================= STREAMLIT UI =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g. Constitution, Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_all_text(subject)
    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        result = generate_flashcard(texts, topic)
        st.markdown(result)
