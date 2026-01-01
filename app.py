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

# =====================================================
# STREAMLIT
# =====================================================
st.set_page_config("NCERT Flashcard Generator", layout="wide")
st.title("📘 NCERT Concept Flashcard Generator")

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
        target = zfile.parent / zfile.stem
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(zfile, "r") as inner:
            inner.extractall(target)

    st.success("✅ NCERT PDFs extracted")

# =====================================================
# CLEANING FUNCTIONS
# =====================================================
def clean_text(text):
    junk = [
        r"ISBN.*", r"Reprint.*", r"Printed.*", r"Published.*",
        r"Editor.*", r"University.*", r"Department.*",
        r"Copyright.*", r"All rights reserved.*",
        r"\d{1,2}\s[A-Za-z]+\s\d{4}", r"Prelims\.indd.*"
    ]
    for j in junk:
        text = re.sub(j, " ", text, flags=re.I)

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
# LOAD TEXTS
# =====================================================
def load_subject_text(subject):
    texts = []
    keywords = SUBJECTS[subject]

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if any(k in str(pdf).lower() for k in keywords):
            t = read_pdf(pdf)
            if len(t.split()) > 200:
                texts.append(t)

    return texts


# =====================================================
# FILTER MEANINGFUL SENTENCES
# =====================================================
def is_conceptual(sentence):
    bad = [
        "isbn", "printed", "editor", "publication", "address",
        "price", "reprint", "office", "department"
    ]
    s = sentence.lower()
    if len(s.split()) < 7:
        return False
    if any(b in s for b in bad):
        return False
    return True


# =====================================================
# MODEL
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# =====================================================
# FLASHCARD GENERATOR
# =====================================================
def generate_flashcard(texts, topic):
    sentences = []
    for t in texts:
        for s in re.split(r'(?<=[.!?])\s+', t):
            if is_conceptual(s):
                sentences.append(s)

    if not sentences:
        return "⚠️ No meaningful content found."

    embeddings = model.encode(sentences)
    topic_vec = model.encode([topic])

    scored = [
        (s, cosine_similarity([e], topic_vec)[0][0])
        for s, e in zip(sentences, embeddings)
    ]

    selected = [s for s, score in scored if score > 0.35]

    if not selected:
        return "⚠️ Topic not clearly present in textbook."

    core = selected[:5]

    return f"""
### 📘 {topic} — Concept Summary

**Concept Overview**  
{core[0]}

**Explanation**  
{" ".join(core[1:4])}

**Why It Matters**  
Understanding this concept helps connect constitutional principles with real-life governance and civic responsibility.
"""


# =====================================================
# UI
# =====================================================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)
    result = generate_flashcard(texts, topic)
    st.markdown(result)
