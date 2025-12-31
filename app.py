import os
import re
import zipfile
from pathlib import Path

import streamlit as st
import gdown
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# CONFIG
# =====================================================


import gdown

FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ/view?usp=drive_link"
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4

# STEP 1: Download ZIP
# ----------------------------

EXTRACT_DIR = "/content/ncert_extracted"
ZIP_PATH = "/content/drive/MyDrive/ncrt subject.zip"

import zipfile, os
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)


SUBJECTS = {
    "Polity": ["polity"],
    "Economics": ["economics"],
    "Sociology": ["sociology"],
    "Psychology": ["psychology"],
    "Business Studies": ["business"]
}

DEPTH_CONFIG = {
    "NCERT": {"chunk_size": 3, "similarity": 0.35},
    "UPSC": {"chunk_size": 6, "similarity": 0.45}
}
#STEP 3: Extract ZIP
# ----------------------------
os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)
st.text(f"ZIP extracted to: {EXTRACT_DIR}")

# Handle nested ZIPs (like class 11/12 PDFs inside)
for root, dirs, files in os.walk(EXTRACT_DIR):
    for file in files:
        if file.lower().endswith(".zip"):
            nested_zip_path = os.path.join(root, file)
            nested_extract_dir = os.path.join(root, Path(file).stem)
            os.makedirs(nested_extract_dir, exist_ok=True)
            with zipfile.ZipFile(nested_zip_path, 'r') as nz:
                nz.extractall(nested_extract_dir)

st.text("All nested ZIPs extracted.")
print ("All nested ZIPs extracted.")

#STEP 3: Extract ZIP
# ----------------------------
os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)
st.text(f"ZIP extracted to: {EXTRACT_DIR}")

# Handle nested ZIPs (like class 11/12 PDFs inside)
for root, dirs, files in os.walk(EXTRACT_DIR):
    for file in files:
        if file.lower().endswith(".zip"):
            nested_zip_path = os.path.join(root, file)
            nested_extract_dir = os.path.join(root, Path(file).stem)
            os.makedirs(nested_extract_dir, exist_ok=True)
            with zipfile.ZipFile(nested_zip_path, 'r') as nz:
                nz.extractall(nested_extract_dir)

st.text("All nested ZIPs extracted.")
print ("All nested ZIPs extracted.")

from pypdf import PdfReader
pdf_count = 0
text_pages = 0

for root, _, files in os.walk(EXTRACT_DIR):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_count += 1
            path = os.path.join(root, file)
            reader = PdfReader(path)
            for page in reader.pages:
                if page.extract_text():
                    text_pages += 1

print("Total PDFs found:", pdf_count)
print("Pages with extractable text:", text_pages)

# =====================================================
# STREAMLIT SETUP
# =====================================================
st.set_page_config("NCERT + UPSC Generator", layout="wide")
st.title("📘 NCERT + UPSC Smart Flashcard Generator")

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(
            url=f"https://drive.google.com/uc?id={FILE_ID}",
            output=ZIP_PATH,
            quiet=False,
            fuzzy=True
        )

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Extract nested zips safely
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            target = zfile.parent / zfile.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(target)
        except Exception as e:
            st.warning(f"Skipped corrupted zip: {zfile}")

    st.success("✅ NCERT PDFs extracted successfully")



def clean_text(text):
    text = re.sub(r"(activity|exercise|project|reprint|isbn|copyright).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_subject_text(subject):
    texts = []
    keywords = SUBJECTS[subject]

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if any(k in pdf.name.lower() for k in keywords):
            text = clean_text(read_pdf(pdf))
            if len(text.split()) > 100:
                texts.append(text)

    return texts

# =====================================================
# CHUNKING
# =====================================================
def chunk_text(text, depth):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s for s in sentences if len(s.split()) > 6]

    size = DEPTH_CONFIG[depth]["chunk_size"]
    chunks, temp = [], []

    for s in sentences:
        temp.append(s)
        if len(temp) >= size:
            chunks.append(" ".join(temp))
            temp = []

    if temp:
        chunks.append(" ".join(temp))

    return chunks

# =====================================================
# FLASHCARD GENERATION
# =====================================================
def generate_flashcard(chunks, topic, depth):
    if not chunks:
        return None

    embeddings = model.encode(chunks)
    query = model.encode([topic])

    sims = cosine_similarity(query, embeddings)[0]
    threshold = DEPTH_CONFIG[depth]["similarity"]

    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    selected = [c for c, s in ranked if s >= threshold][:6]

    if not selected:
        return None

    if depth == "NCERT":
        return f"""
### 📘 {topic} (NCERT)

**Concept Overview**  
{selected[0]}

**Explanation**  
{" ".join(selected[1:3])}

**Key Points**
- {selected[0].split('.')[0]}
- {selected[1].split('.')[0]}
"""

    else:
        return f"""
### 📘 {topic} (UPSC)

**Introduction**  
{selected[0]}

**Analytical Explanation**  
{" ".join(selected[1:4])}

**Contemporary Relevance**  
{selected[4] if len(selected) > 4 else selected[-1]}

**Conclusion**  
This topic is central to governance, constitutionalism, and democratic functioning in India.
"""

# =====================================================
# UI
# =====================================================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
depth = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)

    if not texts:
        st.warning("No readable content found for this subject.")
    else:
        all_chunks = []
        for t in texts:
            all_chunks.extend(chunk_text(t, depth))

        result = generate_flashcard(all_chunks, topic, depth)

        if result:
            st.markdown(result)
        else:
            st.warning("No meaningful content found for this topic.")
