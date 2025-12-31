import os
import zipfile
import re
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# CONFIG
# =====================================================
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = {
    "Polity": ["polity"],
    "Economics": ["economics"],
    "Sociology": ["sociology"],
    "Psychology": ["psychology"],
    "Business Studies": ["business"]
}

SIM_THRESHOLD = 0.35

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config("NCERT Flashcards", layout="wide")
st.title("📘 NCERT Flashcard Generator")

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =====================================================
# DOWNLOAD & EXTRACT ZIP (FULLY)
# =====================================================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    def extract_all(zip_path, out_dir):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)

        for zfile in Path(out_dir).rglob("*.zip"):
            try:
                target = zfile.parent / zfile.stem
                target.mkdir(exist_ok=True)
                with zipfile.ZipFile(zfile, "r") as inner:
                    inner.extractall(target)
                zfile.unlink()
            except:
                pass

    extract_all(ZIP_PATH, EXTRACT_DIR)
    st.success("✅ All NCERT PDFs extracted successfully")

# =====================================================
# PDF READING
# =====================================================
def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            if p.extract_text():
                text += p.extract_text() + " "
        return text.strip()
    except:
        return ""

# =====================================================
# CLEAN TEXT
# =====================================================
def clean_text(text):
    text = re.sub(r"(activity|exercise|project|copyright|isbn|reprint).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =====================================================
# LOAD SUBJECT TEXT
# =====================================================
def load_subject_text(subject):
    keywords = SUBJECTS[subject]
    texts = []

    pdfs = list(Path(EXTRACT_DIR).rglob("*.pdf"))
    st.write(f"📄 Total PDFs found: {len(pdfs)}")

    for pdf in pdfs:
        name = pdf.name.lower()
        if any(k in name for k in keywords):
            text = clean_text(read_pdf(pdf))
            if len(text.split()) > 100:
                texts.append(text)

    return texts

# =====================================================
# CHUNKING
# =====================================================
def chunk_text(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if len(s.split()) > 6]

    chunks, buffer = [], []
    for s in sentences:
        buffer.append(s)
        if len(buffer) >= 5:
            chunks.append(" ".join(buffer))
            buffer = []

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks

# =====================================================
# FLASHCARD GENERATION
# =====================================================
def generate_flashcards(chunks, topic):
    embeddings = model.encode(chunks)
    query_vec = model.encode([topic])

    scores = cosine_similarity(query_vec, embeddings)[0]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    flashcards = []
    for chunk, score in ranked[:6]:
        sents = re.split(r"(?<=[.!?])\s+", chunk)
        if len(sents) >= 3:
            flashcards.append({
                "overview": sents[0],
                "explanation": " ".join(sents[1:4])
            })

    return flashcards

# =====================================================
# SUMMARY CARD
# =====================================================
def summarize_flashcards(cards, topic):
    if not cards:
        return None

    overview = cards[0]["overview"]
    explanation = " ".join([c["explanation"] for c in cards[:3]])

    return f"""
### 📘 {topic}

**Concept Overview**  
{overview}

**Explanation**  
{explanation}

**Conclusion**  
This concept is fundamental for understanding governance, institutions, and democratic principles.
"""

# =====================================================
# UI
# =====================================================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcards"):
    texts = load_subject_text(subject)

    if not texts:
        st.error("❌ No readable content found for this subject.")
    else:
        chunks = []
        for t in texts:
            chunks.extend(chunk_text(t))

        cards = generate_flashcards(chunks, topic)
        final = summarize_flashcards(cards, topic)

        if final:
            st.markdown(final)
        else:
            st.warning("⚠️ Not enough relevant content found.")
