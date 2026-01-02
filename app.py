import os
import re
import zipfile
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util


# ================= CONFIG =================
FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

# Subject → keywords mapping
SUBJECTS = {
    "Polity": ["constitution", "political", "democracy", "rights", "equality"],
    "Geography": ["climate", "monsoon", "atmosphere", "landforms", "resources"],
    "Economics": ["economy", "production", "growth", "market", "development"],
    "History": ["history", "ancient", "medieval", "modern", "empire"],
    "Sociology": ["society", "social", "caste", "culture", "community"]
}


# ================= STREAMLIT =================
st.set_page_config("NCERT Flashcard Generator", layout="wide")
st.title("📘 NCERT → Smart Concept Flashcard Generator")


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
    junk = [
        r"Prelims\.indd.*", r"ISBN.*", r"Reprint.*", r"Printed.*",
        r"All rights reserved.*", r"University.*", r"Editor.*",
        r"Copyright.*", r"\d{1,2}\s[A-Za-z]+\s\d{4}"
    ]
    for j in junk:
        text = re.sub(j, " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = " ".join(p.extract_text() or "" for p in reader.pages)
        return clean_text(text)
    except:
        return ""


# ================= LOAD TEXT BY SUBJECT =================
def load_subject_text(subject):
    keywords = SUBJECTS.get(subject, [])
    texts = []

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        name = pdf.name.lower()
        if any(k in name for k in keywords):
            content = read_pdf(pdf)
            if len(content.split()) > 200:
                texts.append(content)

    return texts


# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ================= PROCESSING =================
def clean_pdf_text(text):
    patterns = [
        r"LET'S DO IT.*", r"LET'S DEBATE.*", r"Reprint.*",
        r"Page \d+", r"©.*", r"Activity.*", r"Exercise.*"
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(text):
    return [p.strip() for p in re.split(r'\n{1,}|\.\s{2,}', text) if len(p.split()) > 40]


# ================= FLASHCARD ENGINE =================
def generate_flashcard(texts, topic):
    full_text = clean_pdf_text(" ".join(texts))
    paragraphs = split_paragraphs(full_text)

    if not paragraphs:
        return "⚠️ No readable content found."

    topic_vec = model.encode(topic, convert_to_tensor=True)
    para_vecs = model.encode(paragraphs, convert_to_tensor=True)

    scores = util.cos_sim(topic_vec, para_vecs)[0]
    ranked = sorted(zip(paragraphs, scores.tolist()), key=lambda x: x[1], reverse=True)

    # take top relevant blocks
    top_blocks = [p for p, s in ranked if s > 0.25][:8]
    if not top_blocks:
        top_blocks = [p for p, _ in ranked[:5]]

    what = how = why = None

    for p in top_blocks:
        pl = p.lower()
        if not what and any(k in pl for k in ["is", "means", "refers to", "can be understood"]):
            what = p
        elif not how and any(k in pl for k in ["works", "functions", "operates", "ensures", "allows"]):
            how = p
        elif not why and any(k in pl for k in ["important", "significant", "helps", "role"]):
            why = p

    if not what: what = top_blocks[0]
    if not how: how = top_blocks[1] if len(top_blocks) > 1 else top_blocks[0]
    if not why: why = top_blocks[2] if len(top_blocks) > 2 else top_blocks[0]

    return f"""
### 📘 {topic.title()} — Concept Flashcard

**What is it?**  
{what}

**How does it work?**  
{how}

**Why is it important?**  
{why}
"""


# ================= UI =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Concept (e.g. Equality, Constitution, Climate)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)

    if not texts:
        st.error("No PDFs found for this subject.")
    else:
        result = generate_flashcard(texts, topic)
        st.markdown(result)
