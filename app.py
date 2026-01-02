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
    "Polity": ["polity", "constitution"],
    "Economics": ["economics"],
    "Sociology": ["sociology"],
    "Psychology": ["psychology"],
    "Business Studies": ["business"]
}

# ================= STREAMLIT UI =================
st.set_page_config(page_title="NCERT Flashcard Generator", layout="wide")
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

    # Extract nested zips
    for z in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            with zipfile.ZipFile(z, "r") as inner:
                inner.extractall(z.parent / z.stem)
        except:
            pass

# ================= PDF READING & CLEANING =================
def clean_text(text):
    patterns = [
        r"ISBN.*", r"Reprint.*", r"Printed.*", r"©.*",
        r"All rights reserved.*", r"University.*",
        r"Editor.*", r"Department.*", r"Email:.*",
        r"\b\d{1,2}\s[A-Za-z]+\s\d{4}\b",
        r"Prelims\.indd.*"
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = " ".join(p.extract_text() or "" for p in reader.pages)
        return clean_text(text)
    except:
        return ""

# ================= LOAD TEXT =================
def load_all_text(subject):
    texts = []
    keywords = SUBJECTS.get(subject, [])
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if any(k in str(pdf).lower() for k in keywords):
            content = read_pdf(pdf)
            if len(content.split()) > 60:
                texts.append(content)
    return texts

# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ================= HELPER FUNCTIONS =================
def clean_pdf_text(text):
    patterns = [
        r"\b\d+\b", r"Reprint.*", r"LET'S DO IT.*", r"LET'S DEBATE.*",
        r"Prelims\.indd.*", r"Page \d+", r"©.*", r"Printed.*", r"\s+"
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    return text.strip()

def split_into_paragraphs(text, min_words=20):
    paragraphs = re.split(r'\n{1,}|\.\s+', text)
    return [p.strip() for p in paragraphs if len(p.split()) >= min_words]

def rank_paragraphs_by_topic(paragraphs, topic, model, top_k=5):
    topic_emb = model.encode(topic, convert_to_tensor=True)
    para_embs = model.encode(paragraphs, convert_to_tensor=True)
    scores = util.cos_sim(topic_emb, para_embs)[0]
    ranked = sorted(zip(paragraphs, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [p for p, s in ranked[:top_k]]

def extract_sentences_for_flashcard(paragraphs, topic):
    what, when, how, why = [], [], [], []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for s in sentences:
            s_low = s.lower()
            if any(k in s_low for k in ["is", "refers to", "means", "defined as", "understood as"]):
                what.append(s)
            elif any(k in s_low for k in ["adopted", "established", "came into force", "introduced", "origin", "developed"]):
                when.append(s)
            elif any(k in s_low for k in ["works", "functions", "operates", "applies", "ensures", "provides"]):
                how.append(s)
            elif any(k in s_low for k in ["important", "significant", "ensures", "protects", "allows", "role"]):
                why.append(s)
    return what, when, how, why

def generate_flashcard_advanced(text, topic, model):
    """
    Generate an advanced UPSC-style flashcard from NCERT text.
    """
    # 1️⃣ Clean and split into paragraphs
    cleaned_text = clean_pdf_text(text)
    paragraphs = split_into_paragraphs(cleaned_text, min_words=30)
    if not paragraphs:
        return "⚠️ No meaningful content found in the PDF."

    # 2️⃣ Combine paragraphs into larger semantic chunks (2–4 paragraphs each)
    chunks = []
    chunk_size = 3
    for i in range(0, len(paragraphs), chunk_size):
        combined = " ".join(paragraphs[i:i+chunk_size])
        chunks.append(combined)

    # 3️⃣ Rank chunks by similarity to topic
    topic_emb = model.encode(topic, convert_to_tensor=True)
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    scores = util.cos_sim(topic_emb, chunk_embs)[0]
    ranked_chunks = sorted(zip(chunks, scores.tolist()), key=lambda x: x[1], reverse=True)
    top_chunks = [c for c, s in ranked_chunks[:3]]  # Top 3 chunks

    # 4️⃣ Extract sentences for flashcard
    what, when, how, why = [], [], [], []
    for chunk in top_chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        for s in sentences:
            s_low = s.lower()
            # WHAT → definition / concept
            if any(k in s_low for k in ["is", "refers to", "means", "defined as", "understood as", "concept"]):
                what.append(s)
            # WHEN → historical / chronological info
            elif any(k in s_low for k in ["adopted", "established", "came into force", "introduced", "origin", "developed", "formed"]):
                when.append(s)
            # HOW → functioning / process / implementation
            elif any(k in s_low for k in ["works", "functions", "operates", "applies", "ensures", "provides", "implements", "governs"]):
                how.append(s)
            # WHY → importance / significance
            elif any(k in s_low for k in ["important", "significant", "ensures", "protects", "allows", "role", "critical", "essential"]):
                why.append(s)

    # 5️⃣ Fallbacks: Use top chunk sentences if category empty
    if not what:
        what = [top_chunks[0].split(".")[0] + "."]
    if not when:
        when = ["Content not clearly specified; refer to historical context in NCERT."]
    if not how:
        how = [top_chunks[0].split(".")[1] if len(top_chunks[0].split(".")) > 1 else top_chunks[0]]
    if not why:
        why = [top_chunks[0].split(".")[-1]]

    # 6️⃣ Build final flashcard
    flashcard = f"""
### 📘 {topic.title()} — UPSC Flashcard

**What is it?**  
{' '.join(what[:2])}

**When was it established?**  
{' '.join(when[:2])}

**How does it work?**  
{' '.join(how[:2])}

**Why is it important?**  
{' '.join(why[:2])}
"""
    return flashcard

# ================= STREAMLIT APP =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g. Constitution, Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_all_text(subject)
    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        combined_text = " ".join(texts)
        result = generate_flashcard_advanced(combined_text, topic, model)
        st.markdown(result)
