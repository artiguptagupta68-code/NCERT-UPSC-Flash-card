import os
import re
import zipfile
from pathlib import Path
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== CONFIG =====================
FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = {
    "Polity": ["constitution", "polity"],
    "Economics": ["economics"],
    "Sociology": ["sociology"],
    "Psychology": ["psychology"],
    "Business Studies": ["business"]
}

# ===================== UI =====================
st.set_page_config("NCERT Smart Flashcards", layout="wide")
st.title("📘 NCERT Smart Flashcard Generator")

# ===================== UTILITIES =====================

def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        import gdown
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    for z in Path(EXTRACT_DIR).rglob("*.zip"):
        with zipfile.ZipFile(z, "r") as inner:
            inner.extractall(z.parent)


def clean_text(text):
    junk = [
        r"Prelims.*", r"ISBN.*", r"Printed.*", r"All rights reserved.*",
        r"Editor.*", r"University.*", r"Department.*",
        r"\d{4}.*", r"©.*"
    ]
    for j in junk:
        text = re.sub(j, " ", text, flags=re.I)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join([p.extract_text() or "" for p in reader.pages])
    except:
        return ""


def load_subject_text(subject):
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = clean_text(read_pdf(pdf))
        if len(text.split()) > 200:
            texts.append(text)
    return texts


# ===================== NLP MODEL =====================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ===================== CORE LOGIC =====================
def build_flashcard(texts, topic):
    full_text = " ".join(texts)

    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentences = [s for s in sentences if len(s.split()) > 7]

    # Semantic filtering
    topic_vec = model.encode([topic])
    sent_vecs = model.encode(sentences)

    scored = [
        (s, cosine_similarity([v], topic_vec)[0][0])
        for s, v in zip(sentences, sent_vecs)
    ]

    relevant = [s for s, sc in sorted(scored, key=lambda x: x[1], reverse=True)[:10]]

    if not relevant:
        return "⚠️ No meaningful content found."

    # Build structured sections
    overview = relevant[0]
    explanation = " ".join(relevant[1:4])
    importance = " ".join(relevant[4:6])

    return f"""
### 📘 {topic} — Concept Flashcard

**What is it?**  
{overview}

**How does it work?**  
{explanation}

**Why is it important?**  
{importance}

**Exam Relevance**
- Frequently asked in NCERT & UPSC  
- Focus on interpretation and application  
"""


# ===================== APP FLOW =====================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g. Constitution)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)

    if not texts:
        st.warning("⚠️ No readable content found.")
    else:
        flashcard = build_flashcard(texts, topic)
        st.markdown(flashcard)
