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
    "Polity": ["constitution", "polity", "rights", "parliament", "judiciary"],
    "Economics": ["economics", "growth", "development"],
    "Sociology": ["society", "social"],
    "Psychology": ["psychology", "behavior"],
    "Business Studies": ["business", "management"]
}

# ================= KNOWLEDGE BASE =================
KNOWLEDGE_BASE = {
    "constitution": {
        "what": "The Constitution of India is the supreme law that defines the structure, powers, and functions of the government and guarantees fundamental rights.",
        "when": "It was adopted on 26 November 1949 and came into force on 26 January 1950.",
        "how": "It distributes powers among the Legislature, Executive, and Judiciary and provides mechanisms for governance and accountability.",
        "why": "It ensures democracy, rule of law, and protection of citizens’ rights.",
        "articles": "Articles 1–395; Parts III, IV, V–XI."
    }
}

# ================= STREAMLIT =================
st.set_page_config(page_title="NCERT Flashcard Generator", layout="wide")
st.title("📘 NCERT → Smart Concept Flashcard")


# ================= DOWNLOAD =================
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
        r"Editor.*", r"Department.*", r"Email:.*"
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def read_pdf(path):
    try:
        reader = PdfReader(path)
        return clean_text(" ".join(p.extract_text() or "" for p in reader.pages))
    except:
        return ""


# ================= LOAD TEXT =================
def load_all_text(subject):
    texts = []
    keywords = SUBJECTS.get(subject, [])

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if any(k in pdf.name.lower() for k in keywords):
            content = read_pdf(pdf)
            if len(content.split()) > 80:
                texts.append(content)
    return texts


# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ================= CORE LOGIC =================
def identify_topic(text):
    text_emb = model.encode(text)
    scores = {}

    for topic in KNOWLEDGE_BASE:
        topic_emb = model.encode(topic)
        scores[topic] = cosine_similarity([text_emb], [topic_emb])[0][0]

    return max(scores, key=scores.get)


def extract_supporting_sentences(text, topic):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relevant = []

    for s in sentences:
        if topic in s.lower() and len(s.split()) > 10:
            relevant.append(s)

    return relevant[:3]


def generate_flashcard(texts, topic):
    full_text = " ".join(texts)
    topic_key = identify_topic(full_text)

    base = KNOWLEDGE_BASE.get(topic_key)

    if not base:
        return "⚠️ No knowledge base available for this topic."

    support = extract_supporting_sentences(full_text, topic_key)
    support_text = " ".join(support) if support else "NCERT discusses this concept in multiple contexts."

    return f"""
## 📘 {topic_key.upper()} — UPSC FLASHCARD

### What is it?
{base['what']}

### When was it established?
{base['when']}

### How does it work?
{base['how']}

### Why is it important?
{base['why']}

### Relevant Articles
{base['articles']}

### NCERT Context
{support_text}
"""


# ================= UI =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g. Constitution, Judiciary)")

if st.button("Generate Flashcard"):
    texts = load_all_text(subject)

    if not texts:
        st.warning("⚠️ No readable NCERT content found.")
    else:
        result = generate_flashcard(texts, topic)
        st.markdown(result)
