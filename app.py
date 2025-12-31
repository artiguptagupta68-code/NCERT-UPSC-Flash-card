import os
import zipfile
import re
from pathlib import Path
import streamlit as st
import gdown
from pypdf import PdfReader

# -----------------------------
# CONFIG
# -----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

# -----------------------------
# DOWNLOAD & EXTRACT ZIP
# -----------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            ZIP_PATH,
            quiet=False
        )

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Extract main zip
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

# -----------------------------
# READ PDF TEXT SAFELY
# -----------------------------
def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception:
        return ""

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = re.sub(
        r"(activity|let us|exercise|project|editor|reprint|copyright|isbn).*",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# LOAD ALL PDF TEXTS
# -----------------------------
def load_all_texts():
    texts = []
    valid_files = []

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = clean_text(read_pdf(pdf))
        if len(text.split()) > 80:  # important threshold
            texts.append(text)
            valid_files.append(str(pdf))

    st.write(f"📄 Loaded {len(valid_files)} PDFs with readable content:")
    for f in valid_files:
        st.write("•", f)

    if not texts:
        st.warning("⚠️ No readable content found. PDFs may be scanned images.")

    return texts


# ======================================================
# SEMANTIC CHUNKING
# ======================================================
def chunk_text(text):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s for s in sentences if is_valid_sentence(s)]

    chunks = []
    current = []

    for s in sentences:
        current.append(s)
        if len(current) >= 5:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks

# ======================================================
# FLASHCARD GENERATION
# ======================================================
def generate_flashcards(chunks, topic):
    embeddings = model.encode(chunks)
    query_vec = model.encode([topic])

    scores = cosine_similarity(query_vec, embeddings)[0]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    good_chunks = [c for c, s in ranked if s > SIM_THRESHOLD][:6]

    flashcards = []
    for ch in good_chunks:
        sents = re.split(r"(?<=[.?!])\s+", ch)
        sents = [s for s in sents if is_valid_sentence(s)]
        if len(sents) >= 3:
            flashcards.append({
                "overview": sents[0],
                "explanation": " ".join(sents[1:4])
            })

    return flashcards

# ======================================================
# SUMMARY FLASHCARD
# ======================================================
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
This concept plays a foundational role in understanding governance, rights, and social structure in a democratic system.
"""

# ======================================================
# UI
# ======================================================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Topic (e.g. Fundamental Rights)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)

    if not texts:
        st.warning("No readable content found for this subject.")
    else:
        chunks = []
        for t in texts:
            chunks.extend(chunk_text(t))

        flashcards = generate_flashcards(chunks, topic)
        final_card = summarize_flashcards(flashcards, topic)

        if final_card:
            st.markdown(final_card)
        else:
            st.warning("No meaningful content found for this topic.")
