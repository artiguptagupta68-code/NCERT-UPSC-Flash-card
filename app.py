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

SUBJECTS = {
    "Polity": ["constitution", "rights", "equality", "democracy"],
    "Geography": ["climate", "monsoon", "atmosphere", "resources"],
    "Economics": ["economy", "development", "market", "growth"],
    "History": ["ancient", "medieval", "modern", "empire"],
    "Psychology": ["group", "behaviour", "social", "individual"]
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


# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ================= PDF READING =================
def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""


# ================= SUBJECT FILTER =================
def load_subject_text(subject):
    texts = []
    subject_vec = model.encode(subject, convert_to_tensor=True)

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = read_pdf(pdf)
        if len(text.split()) < 300:
            continue

        sample = " ".join(text.split()[:400])
        score = util.cos_sim(
            subject_vec,
            model.encode(sample, convert_to_tensor=True)
        ).item()

        if score > 0.25:
            texts.append(text)

    return texts


# ================= HELPERS =================
def split_paragraphs(text):
    paras = re.split(r'\n{1,}|\.\s{2,}', text)
    return [p.strip() for p in paras if len(p.split()) > 25]


def limit_text(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:max_sentences])


def topic_relevant(paragraph, topic):
    topic_vec = model.encode(topic, convert_to_tensor=True)
    para_vec = model.encode(paragraph, convert_to_tensor=True)
    return util.cos_sim(topic_vec, para_vec).item() > 0.35


# ================= FLASHCARD ENGINE =================
def generate_flashcard(texts, topic):
    all_text = " ".join(texts)
    paragraphs = split_paragraphs(all_text)

    relevant = [p for p in paragraphs if topic_relevant(p, topic)]

    if len(relevant) < 2:
        return "⚠️ Topic not found clearly in NCERT content."

    topic_vec = model.encode(topic, convert_to_tensor=True)
    para_vecs = model.encode(relevant, convert_to_tensor=True)
    scores = util.cos_sim(topic_vec, para_vecs)[0]

    ranked = [
        p for p, _ in sorted(
            zip(relevant, scores),
            key=lambda x: x[1],
            reverse=True
        )
    ]

    what = next((p for p in ranked if "defined" in p.lower() or "is a" in p.lower()), ranked[0])
    how = next((p for p in ranked if any(k in p.lower() for k in ["formed", "process", "works", "functions"])), ranked[1])
    why = next((p for p in ranked if any(k in p.lower() for k in ["important", "helps", "role", "significance"])), ranked[2])

    return f"""
### 📘 {topic.title()} — Concept Flashcard

**What is it?**  
{limit_text(what)}

**How does it work?**  
{limit_text(how)}

**Why is it important?**  
{limit_text(why)}
"""


# ================= UI =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g., Group, Equality, Constitution)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)
    if not texts:
        st.error("No PDFs found for this subject.")
    else:
        st.markdown(generate_flashcard(texts, topic))
