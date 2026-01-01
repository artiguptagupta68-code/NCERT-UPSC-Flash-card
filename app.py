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
