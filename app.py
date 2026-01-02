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


# ================= STREAMLIT SETUP =================
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


# ================= LOAD ALL PDFs =================
def load_all_text():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        txt = read_pdf(pdf)
        if len(txt.split()) > 200:
            texts.append(txt)
    return texts


# ================= EMBEDDING MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ================= CLEAN & SPLIT =================
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


# ================= FLASHCARD LOGIC =================
def generate_flashcard(texts, topic):
    topic_lower = topic.lower()

    full_text = clean_pdf_text(" ".join(texts))
    paragraphs = split_paragraphs(full_text)

    if not paragraphs:
        return "⚠️ No readable content found."

    # ---------- Topic Matching ----------
    topic_vec = model.encode(topic, convert_to_tensor=True)
    para_vecs = model.encode(paragraphs, convert_to_tensor=True)

    similarities = util.cos_sim(topic_vec, para_vecs)[0]
    ranked = sorted(
        zip(paragraphs, similarities.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    # Keep top relevant paragraphs
    topic_chunks = [p for p, s in ranked if s > 0.25]

    # Fallback if topic not explicit
    if not topic_chunks:
        topic_chunks = [p for p, _ in ranked[:10]]

    # ---------- Extract What / How / Why ----------
    what = how = why = None

    for p in topic_chunks:
        pl = p.lower()
        if not what and any(x in pl for x in ["is", "means", "refers to", "can be understood"]):
            what = p
        elif not how and any(x in pl for x in ["works", "functions", "operates", "ensures", "allows"]):
            how = p
        elif not why and any(x in pl for x in ["important", "significant", "because", "helps", "role"]):
            why = p

    # Fallbacks
    if not what: what = topic_chunks[0]
    if not how: how = topic_chunks[1] if len(topic_chunks) > 1 else topic_chunks[0]
    if not why: why = topic_chunks[2] if len(topic_chunks) > 2 else topic_chunks[0]

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

    st.write(f"📄 PDFs loaded: {len(texts)}")

    if not texts:
        st.error("No readable NCERT content found.")
    else:
        result = generate_flashcard(texts, topic)
        st.markdown(result)
