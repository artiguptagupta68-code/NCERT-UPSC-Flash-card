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

# ================= HELPER FUNCTIONS =================
from sentence_transformers import SentenceTransformer, util
import re

# ---------------- HELPER FUNCTIONS ----------------
def clean_pdf_text(text):
    """
    Remove noise like page numbers, exercises, boxes, reprints, etc.
    """
    # Remove page numbers, Reprint, LET'S DO IT/DEBATE, etc.
    patterns = [
        r"\b\d+\b", r"Reprint.*", r"LET'S DO IT.*", r"LET'S DEBATE.*", 
        r"Swaraj", r"Prelims\.indd.*", r"\s+", r"Page \d+", r"©.*", r"Printed.*"
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    return text.strip()

def split_into_paragraphs(text, min_words=20):
    """
    Split text into paragraphs of at least min_words words
    """
    paragraphs = re.split(r'\n{1,}|\.\s+', text)
    cleaned_paragraphs = [p.strip() for p in paragraphs if len(p.split()) >= min_words]
    return cleaned_paragraphs

def rank_paragraphs_by_topic(paragraphs, topic, model, top_k=5):
    """
    Rank paragraphs using semantic embeddings
    """
    topic_emb = model.encode(topic, convert_to_tensor=True)
    para_embs = model.encode(paragraphs, convert_to_tensor=True)
    scores = util.cos_sim(topic_emb, para_embs)[0]
    ranked = sorted(zip(paragraphs, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [p for p, s in ranked[:top_k]]

def extract_sentences_for_flashcard(paragraphs, topic):
    """
    Extract sentences for What / When / How / Why
    """
    what, when, how, why = [], [], [], []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for s in sentences:
            s_low = s.lower()
            # WHAT → definition / concept
            if any(k in s_low for k in ["is", "refers to", "means", "defined as", "understood as"]):
                what.append(s)
            # WHEN → historical or chronological info
            elif any(k in s_low for k in ["adopted", "established", "came into force", "introduced", "origin", "developed"]):
                when.append(s)
            # HOW → functioning / process / implementation
            elif any(k in s_low for k in ["works", "functions", "operates", "applies", "ensures", "provides"]):
                how.append(s)
            # WHY → importance / significance
            elif any(k in s_low for k in ["important", "significant", "ensures", "protects", "allows", "role"]):
                why.append(s)
    return what, when, how, why

def generate_flashcard(text, topic, model):
    """
    Generate an exam-ready flashcard
    """
    cleaned_text = clean_pdf_text(text)
    paragraphs = split_into_paragraphs(cleaned_text)
    if not paragraphs:
        return "⚠️ No meaningful content found in the PDF."

    ranked_paragraphs = rank_paragraphs_by_topic(paragraphs, topic, model, top_k=5)
    what, when, how, why = extract_sentences_for_flashcard(ranked_paragraphs, topic)

    # Fallbacks
    if not what:
        what = [ranked_paragraphs[0] if ranked_paragraphs else "Content unavailable."]
    if not when:
        when = ["Content unavailable."]
    if not how:
        how = [ranked_paragraphs[1] if len(ranked_paragraphs) > 1 else ranked_paragraphs[0]]
    if not why:
        why = [ranked_paragraphs[2] if len(ranked_paragraphs) > 2 else ranked_paragraphs[0]]

    # Build flashcard
    flashcard = f"""
### 📘 {topic.title()} — UPSC Flashcard

**What is it?**  
{what[0]}

**When was it established?**  
{when[0]}

**How does it work?**  
{how[0]}

**Why is it important?**  
{why[0]}
"""
    return flashcard
)
