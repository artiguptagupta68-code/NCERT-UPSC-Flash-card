import os
import re
import zipfile
from pathlib import Path
import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# CONFIG
# ================================
FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = {
    "Polity": ["polity", "political", "constitution"],
    "Economics": ["economics", "economic"],
    "Sociology": ["sociology", "society"],
    "Psychology": ["psychology", "mind", "behavior"],
    "Business Studies": ["business", "management", "commerce"]
}


DEPTH_CONFIG = {
    "NCERT": {"chunk_size": 3, "similarity": 0.35},
    "UPSC": {"chunk_size": 6, "similarity": 0.45}
}

TOP_K = 6

# ================================
# STREAMLIT SETUP
# ================================
st.set_page_config(page_title="NCERT + UPSC Flashcards", layout="wide")
st.title("📘 NCERT + UPSC Smart Flashcard Generator")

# ================================
# LOAD EMBEDDING MODEL
# ================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ================================
# DOWNLOAD & EXTRACT NCERT ZIP
# ================================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            ZIP_PATH,
            quiet=False,
            fuzzy=True
        )

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Extract main ZIP
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Extract nested ZIPs
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            target = zfile.parent / zfile.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(target)
        except:
            pass

    st.success("✅ NCERT PDFs extracted!")

# ================================
# PDF READING & CLEANING
# ================================
def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(activity|exercise|project|reprint|isbn|copyright).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_subject_text(subject):
    texts = []
    keywords = SUBJECTS[subject]

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        pdf_path_lower = str(pdf).lower()  # check full path too
        if any(k in pdf_path_lower for k in keywords):
            text = clean_text(read_pdf(pdf))
            if len(text.split()) > 80:
                texts.append(text)
    return texts

# ================================
# SEMANTIC CHUNKING
# ================================
def chunk_text(text, depth):
    max_sentences = DEPTH_CONFIG[depth]["chunk_size"]
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 6]

    chunks = []
    current = []
    for s in sentences:
        current.append(s)
        if len(current) >= max_sentences:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

# ================================
# FLASHCARD GENERATION
# ================================
def generate_flashcard(chunks, topic, depth):
    if not chunks:
        return None

    embeddings = model.encode(chunks)
    query = model.encode([topic])
    threshold = DEPTH_CONFIG[depth]["similarity"]

    sims = cosine_similarity(query, embeddings)[0]
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    selected = [c for c, s in ranked if s >= threshold][:TOP_K]

    if not selected:
        return None

    if depth == "NCERT":
        return f"""
### 📘 {topic} (NCERT)

**Concept Overview**  
{selected[0]}

**Explanation**  
{" ".join(selected[1:3]) if len(selected) > 1 else ""}

**Key Points**
- {selected[0].split('.')[0]}
- {selected[1].split('.')[0] if len(selected) > 1 else ""}
"""
    else:
        return f"""
### 📘 {topic} (UPSC)

**Introduction**  
{selected[0]}

**Analytical Explanation**  
{" ".join(selected[1:4]) if len(selected) > 1 else ""}

**Contemporary Relevance**  
{selected[4] if len(selected) > 4 else selected[-1]}

**Conclusion**  
This topic is central to governance, constitutionalism, and democratic functioning in India.
"""

# ================================
# MAIN UI
# ================================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)
    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        all_chunks = []
        for t in texts:
            all_chunks.extend(chunk_text(t, depth))

        result = generate_flashcard(all_chunks, topic, depth)
        if result:
            st.markdown(result)
        else:
            st.warning("No meaningful content found for this topic.")
