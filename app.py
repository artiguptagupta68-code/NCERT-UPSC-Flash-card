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
    return re.sub(r"\s+", " ", text).strip()


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


# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ================= TEXT CLEANING =================
def clean_pdf_text(text):
    patterns = [
        r"LET'S DO IT.*", r"LET'S DEBATE.*", r"Reprint.*",
        r"Page \d+", r"\b\d+\b", r"©.*", r"Printed.*",
        r"Activity.*", r"Exercise.*"
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


# ================= TOPIC MATCH =================
def topic_match(paragraph, topic):
    topic_words = topic.lower().split()
    paragraph_lower = paragraph.lower()

    if any(word in paragraph_lower for word in topic_words):
        return True

    para_vec = model.encode(paragraph, convert_to_tensor=True)
    topic_vec = model.encode(topic, convert_to_tensor=True)
    score = util.cos_sim(topic_vec, para_vec).item()

    return score > 0.35


# ================= FLASHCARD GENERATOR =================
def generate_flashcard(texts, topic):
    topic_lower = topic.lower()
    full_text = clean_pdf_text(" ".join(texts))

    paragraphs = [
        p.strip() for p in re.split(r'\n{1,}|\.\s{2,}', full_text)
        if len(p.split()) > 40
    ]

    topic_chunks = [p for p in paragraphs if topic_match(p, topic)]

    if not topic_chunks:
        return "⚠️ Topic not found clearly in NCERT content."

    topic_vec = model.encode(topic, convert_to_tensor=True)
    para_vecs = model.encode(topic_chunks, convert_to_tensor=True)
    scores = util.cos_sim(topic_vec, para_vecs)[0]

    ranked = [p for p, _ in sorted(zip(topic_chunks, scores), key=lambda x: x[1], reverse=True)]

    what = how = why = None

    for p in ranked:
        low = p.lower()
        if not what and any(x in low for x in ["is", "means", "refers to", "can be understood"]):
            what = p
        elif not how and any(x in low for x in ["works", "functions", "operates", "ensures", "provides"]):
            how = p
        elif not why and any(x in low for x in ["important", "significant", "because", "role", "helps"]):
            why = p

    if not what: what = ranked[0]
    if not how: how = ranked[1] if len(ranked) > 1 else ranked[0]
    if not why: why = ranked[2] if len(ranked) > 2 else ranked[0]

    return f"""
### 📘 {topic.title()} — Concept Flashcard

**What is it?**  
{what}

**How does it work?**  
{how}

**Why is it important?**  
{why}
"""


# ================= STREAMLIT UI =================
download_and_extract()

topic = st.text_input("Enter Concept (e.g. Freedom, Equality, Constitution)")

if st.button("Generate Flashcard"):
    texts = load_all_text()
    if not texts:
        st.error("No readable NCERT content found.")
    else:
        st.markdown(generate_flashcard(texts, topic))
