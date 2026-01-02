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

from sentence_transformers import util
# ================= KNOWLEDGE BASE (fallback) =================
KNOWLEDGE_BASE = {
    "constitution": {
        "what": "The Constitution of India is the supreme law that defines the structure, powers, and functions of the government and guarantees fundamental rights.",
        "when": "It was adopted on 26 November 1949 and came into force on 26 January 1950.",
        "how": "It distributes powers among the Legislature, Executive, and Judiciary and provides mechanisms for governance and accountability.",
        "why": "It ensures democracy, rule of law, and protection of citizens’ rights.",
        "articles": "Articles 1–395; Parts III, IV, V–XI."
    },
    "freedom": {
        "what": "Freedom is the absence of external constraints and the ability of individuals to make autonomous choices.",
        "when": "Freedom has been debated and evolved throughout history in political thought and practice.",
        "how": "It works by protecting individual rights through laws, democratic institutions, and social norms.",
        "why": "It is essential for human development, self-realization, and participatory democracy.",
        "articles": "Related constitutional provisions on fundamental rights."
    }
    # Add more topics here as needed
}


def generate_flashcard(texts, topic, model, top_k_chunks=5, top_k_sentences=3):
    """
    Generate UPSC-style flashcard by:
    1. Splitting text into semantic chunks
    2. Ranking chunks by relevance to topic
    3. Extracting sentences for What / When / How / Why
    4. Deduplicating and summarizing
    """
    # Combine all text
    full_text = " ".join(texts)
    
    # --- 1. Clean text ---
    full_text = re.sub(r"\s+", " ", full_text)
    full_text = re.sub(r"Page \d+|LET'S DO IT|LET'S DEBATE|Swaraj|Prelims\.indd.*|Reprint.*", " ", full_text, flags=re.I)
    
    # --- 2. Split into semantic chunks (~100-150 words each) ---
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    chunks, chunk = [], []
    count = 0
    max_chunk_size = 120
    for s in sentences:
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
    
    if not chunks:
        return "⚠️ No meaningful content found in the PDF."
    
    # --- 3. Rank chunks by semantic similarity to topic ---
    topic_emb = model.encode(topic, convert_to_tensor=True)
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    scores = util.cos_sim(topic_emb, chunk_embs)[0]
    ranked_chunks = sorted(zip(chunks, scores.tolist()), key=lambda x: x[1], reverse=True)
    
    top_chunks = [c for c, s in ranked_chunks[:top_k_chunks]]
    
    # --- 4. Extract sentences for flashcard ---
    what, when, how, why = [], [], [], []
    used_sentences = set()  # deduplicate
    
    for chunk in top_chunks:
        for s in re.split(r'(?<=[.!?])\s+', chunk):
            s_clean = s.strip()
            s_low = s_clean.lower()
            if len(s_clean.split()) < 8 or s_clean in used_sentences:
                continue
            
            used_sentences.add(s_clean)
            
            # WHAT → definition / concept
            if any(k in s_low for k in ["is", "refers to", "means", "defined as", "understood as"]) and len(what) < top_k_sentences:
                what.append(s_clean)
            # WHEN → historical / chronological info
            elif any(k in s_low for k in ["adopted", "established", "came into force", "introduced", "origin", "developed"]) and len(when) < top_k_sentences:
                when.append(s_clean)
            # HOW → functioning / process / implementation
            elif any(k in s_low for k in ["works", "functions", "operates", "applies", "ensures", "provides"]) and len(how) < top_k_sentences:
                how.append(s_clean)
            # WHY → importance / significance
            elif any(k in s_low for k in ["important", "significant", "ensures", "protects", "allows", "role"]) and len(why) < top_k_sentences:
                why.append(s_clean)
    
    # --- 5. Smart fallbacks if sections are empty ---
    kb = KNOWLEDGE_BASE.get(topic.lower())
    if kb:
        if not what: what = [kb["what"]]
        if not when: when = [kb["when"]]
        if not how: how = [kb["how"]]
        if not why: why = [kb["why"]]
    
    if not what: what = ["Content unavailable."]
    if not when: when = ["Content unavailable."]
    if not how: how = ["Content unavailable."]
    if not why: why = ["Content unavailable."]
    
    # --- 6. Build flashcard ---
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
        result = generate_flashcard(combined_text, topic, model)
        st.markdown(result)
