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


# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ================= CLEAN & SPLIT =================
def clean_pdf_text(text):
    patterns = [
        r"LET'S DO IT.*", r"LET'S DEBATE.*", r"Reprint.*",
        r"Page \d+", r"\b\d+\b", r"©.*", r"Printed.*",
        r"Activity.*", r"Exercise.*"
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.split()) > 8]


# ================= CLASSIFIERS =================
def is_definition_sentence(s):
    return any(k in s.lower() for k in [
        "is", "means", "refers to", "can be understood", "defined as"
    ])

def is_how_sentence(s):
    return any(k in s.lower() for k in [
        "works", "functions", "operates", "ensures", "allows", "enables"
    ])

def is_why_sentence(s):
    return any(k in s.lower() for k in [
        "important", "significant", "because", "so that", "helps", "ensures"
    ])


# ================= FLASHCARD ENGINE =================
def generate_flashcard(texts, topic):
    full_text = clean_pdf_text(" ".join(texts))
    sentences = split_sentences(full_text)

    topic_vec = model.encode(topic, convert_to_tensor=True)
    sent_vecs = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(topic_vec, sent_vecs)[0]

    ranked = sorted(zip(sentences, scores.tolist()), key=lambda x: x[1], reverse=True)
    top = [s for s, _ in ranked[:40]]

    what, how, why = [], [], []

    for s in top:
        if not what and is_definition_sentence(s):
            what.append(s)
        elif not how and is_how_sentence(s):
            how.append(s)
        elif not why and is_why_sentence(s):
            why.append(s)

    if not what and top:
        what.append(top[0])
    if not how and len(top) > 1:
        how.append(top[1])
    if not why and len(top) > 2:
        why.append(top[2])

    return f"""
### 📘 {topic.title()} — Concept Flashcard

**What is it?**  
{what[0]}

**How does it work?**  
{how[0]}

**Why is it important?**  
{why[0]}
"""


# ================= STREAMLIT UI =================
download_and_extract()

topic = st.text_input("Enter Concept (e.g. Freedom, Equality, Constitution)")

if st.button("Generate Flashcard"):
    texts = load_all_text()
    if not texts:
        st.error("No readable NCERT content found.")
    else:
        result = generate_flashcard(texts, topic)
        st.markdown(result)
