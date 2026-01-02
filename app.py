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
def is_noise(sentence):
    patterns = ["chapter", "figure", "table", "indd", "page", "copyright", "printed", "isbn", "unit", "lesson", "activity", "exercise"]
    sentence = sentence.lower()
    return any(p in sentence for p in patterns)

def split_semantic_chunks(text, max_chunk_size=150):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, chunk = [], []
    count = 0
    for s in sentences:
        if is_noise(s):
            continue
        words = len(s.split())
        if words + count > max_chunk_size:
            if chunk:
                chunks.append(" ".join(chunk))
            chunk = [s]
            count = words
        else:
            chunk.append(s)
            count += words
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def semantic_ranking(chunks, topic):
    topic_emb = model.encode(topic, convert_to_tensor=True)
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    scores = util.cos_sim(topic_emb, chunk_embs)[0]
    ranked = sorted(zip(chunks, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked]

def build_flashcard_from_kb(topic, kb):
    return f"""
### 📘 {topic.title()} — UPSC Flashcard

**What is it?**  
{kb['what']}

**When was it established?**  
{kb['when']}

**How does it work?**  
{kb['how']}

**Why is it important?**  
{kb['why']}
"""

# ================= FLASHCARD GENERATION =================
def generate_flashcard(texts, topic):
    full_text = " ".join(texts)
    chunks = split_semantic_chunks(full_text, max_chunk_size=120)

    if not chunks:
        kb = KNOWLEDGE_BASE.get(topic.lower())
        if kb:
            return build_flashcard_from_kb(topic, kb)
        else:
            return "⚠️ No meaningful content found."

    ranked_chunks = semantic_ranking(chunks, topic)
    top_chunks = ranked_chunks[:3]
    combined_text = " ".join(top_chunks)

    sentences = re.split(r'(?<=[.!?])\s+', combined_text)
    what, when, how, why = [], [], [], []

    for s in sentences:
        s_low = s.lower()
        if any(k in s_low for k in ["is", "refers to", "means", "defined as", "explains"]):
            what.append(s)
        elif any(k in s_low for k in ["adopted", "enacted", "came into force", "established", "constitution of india"]):
            when.append(s)
        elif any(k in s_low for k in ["provides", "establishes", "lays down", "regulates", "ensures", "distribution of powers", "judiciary", "fundamental rights", "directive principles"]):
            how.append(s)
        elif any(k in s_low for k in ["important", "significant", "protects", "democracy", "rule of law", "justice", "unity"]):
            why.append(s)

    kb = KNOWLEDGE_BASE.get(topic.lower())
    if kb:
        if not what: what = [kb["what"]]
        if not when: when = [kb["when"]]
        if not how: how = [kb["how"]]
        if not why: why = [kb["why"]]

    return f"""
### 📘 {topic.title()} — UPSC Flashcard

**What is it?**  
{what[0] if what else 'Content unavailable.'}

**When was it established?**  
{when[0] if when else 'Content unavailable.'}

**How does it work?**  
{how[0] if how else 'Content unavailable.'}

**Why is it important?**  
{why[0] if why else 'Content unavailable.'}
"""

# ================= STREAMLIT UI =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g. Constitution, Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_all_text(subject)
    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        result = generate_flashcard(texts, topic)
        st.markdown(result)
