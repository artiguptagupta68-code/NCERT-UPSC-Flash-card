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
    "NCERT": {"top_k": 3, "similarity": 0.35},
    "UPSC": {"top_k": 6, "similarity": 0.45}
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
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Extract nested ZIPs
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        target = zfile.parent / zfile.stem
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(zfile, "r") as inner:
            inner.extractall(target)

    st.success("✅ NCERT PDFs extracted successfully")


# ===================== CLEAN TEXT =====================
def clean_text(text):
    remove_patterns = [
        r"ISBN.*",
        r"Printed.*",
        r"Reprint.*",
        r"Email:.*",
        r"Department of.*",
        r"University.*",
        r"All rights reserved.*",
        r"©.*",
        r"^\d+\s*$",
        r"by the Constitution.*?Act.*",
        r"Sec\.\s*\d+.*",
        r"\(w\.e\.f.*?\)",
        r"Prelims\.indd.*",
    ]

    for pat in remove_patterns:
        text = re.sub(pat, " ", text, flags=re.I | re.S)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


import pdfplumber
import re

def read_pdf(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    # Remove headers/footers (lines that are too short or numeric)
                    lines = [l.strip() for l in page_text.split("\n") if len(l.strip()) > 20 and not l.strip().isdigit()]
                    text += " ".join(lines) + " "
        return clean_text(text)
    except Exception as e:
        print("PDF read error:", e)
        return ""

# ===================== LOAD SUBJECT TEXT =====================
def load_subject_text(subject):
    texts = []

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        try:
            text = read_pdf(pdf)
            if len(text.split()) > 300:   # Only real content
                texts.append(text)
        except:
            continue

    return texts


# ===================== CHUNKING & SEMANTIC FILTER =====================
def classify_sentences(sentences):
    definition, working, importance = [], [], []

    for s in sentences:
        s_low = s.lower()
        if any(k in s_low for k in ["means", "refers to", "is defined as", "is the"]):
            definition.append(s)
        elif any(k in s_low for k in ["works", "interpreted", "implemented", "ensures", "provides"]):
            working.append(s)
        elif any(k in s_low for k in ["important", "ensures", "protects", "helps", "essential"]):
            importance.append(s)

    return definition, working, importance


# ===================== LOAD EMBEDDING MODEL =====================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ===================== FLASHCARD GENERATION =====================
def build_flashcard(texts, topic):
    full_text = " ".join(texts)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentences = [s for s in sentences if len(s.split()) > 8]

    if not sentences:
        return "⚠️ No meaningful content found."

    # Semantic similarity filtering
    topic_vec = model.encode([topic])
    sent_vecs = model.encode(sentences)

    relevant = [
        s for s, v in zip(sentences, sent_vecs)
        if cosine_similarity([v], topic_vec)[0][0] > 0.40
    ]

    if not relevant:
        return "⚠️ No meaningful content found."

    definition, working, importance = classify_sentences(relevant)

    return f"""
### 📘 {topic} — Concept Flashcard

**What is it?**  
{definition[0] if definition else "Defines the concept under the Indian Constitution."}

**How does it work?**  
{' '.join(working[:2]) if working else "It operates through constitutional provisions and judicial interpretation."}

**Why is it important?**  
{' '.join(importance[:2]) if importance else "It safeguards democratic values and fundamental rights."}

**Exam Relevance**
- High weightage in NCERT & UPSC  
- Frequently used in analytical questions  
"""


# ===================== STREAMLIT UI =====================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard") and topic.strip():
    texts = load_subject_text(subject)
    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        flashcard = build_flashcard(texts, topic)
        st.markdown(flashcard)
