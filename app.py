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

# Subject → keywords mapping
SUBJECTS = {
    "Polity": ["constitution", "political", "democracy", "rights", "equality"],
    "Geography": ["climate", "monsoon", "atmosphere", "landforms", "resources"],
    "Economics": ["economy", "production", "growth", "market", "development"],
    "History": ["history", "ancient", "medieval", "modern", "empire"],
    "Sociology": ["society", "social", "caste", "culture", "community"]
}


# ================= STREAMLIT =================
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


# ================= LOAD TEXT BY SUBJECT =================
def load_subject_text(subject):
    all_texts = []
    subject_vec = model.encode(subject, convert_to_tensor=True)

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = read_pdf(pdf)
        if len(text.split()) < 300:
            continue

        # Semantic relevance check
        sample = " ".join(text.split()[:400])
        sample_vec = model.encode(sample, convert_to_tensor=True)
        score = util.cos_sim(subject_vec, sample_vec).item()

        if score > 0.25:   # threshold for relevance
            all_texts.append(text)

    return all_texts


# ================= MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ================= PROCESSING =================
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


# ================= FLASHCARD ENGINE =================
def extract_chapter_structure(text, topic):
    """
    Split NCERT text into:
    - About/Introduction
    - Subheadings (What, How, Why, Types of...)
    """
    text_lower = text.lower()
    if topic.lower() not in text_lower:
        return None  # topic not present in this chapter

    # Split into paragraphs (rough)
    paragraphs = re.split(r'\n{1,}|\.\s{2,}', text)

    # Identify subheadings: lines in Title Case or uppercase
    heading_regex = re.compile(r'^(?:[A-Z][A-Za-z\s]{3,60}|[A-Z\s]{5,})$')
    structured = {}
    current_heading = "About"
    structured[current_heading] = []

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
        if heading_regex.match(para_clean):
            current_heading = para_clean
            structured[current_heading] = []
        else:
            structured[current_heading].append(para_clean)

    # Convert lists to full paragraphs
    for k in structured:
        structured[k] = " ".join(structured[k]).strip()

    return structured

def map_flashcard_fields(structured):
    """
    Map subheadings to flashcard fields: What / How / Why
    """
    what, how, why = None, None, None

    for heading, para in structured.items():
        h_low = heading.lower()
        if not what and any(k in h_low for k in ["what", "meaning", "nature"]):
            what = para
        elif not how and any(k in h_low for k in ["how", "formation", "process", "types"]):
            how = para
        elif not why and any(k in h_low for k in ["why", "importance", "significance"]):
            why = para

    # Fallbacks if not found
    headings = list(structured.keys())
    if not what:
        what = structured.get("About", headings[0] if headings else "Content unavailable.")
    if not how:
        how = structured.get(headings[1], what)
    if not why:
        why = structured.get(headings[2], what)

    return what, how, why

def generate_flashcard(texts, topic):
    combined_text = " ".join(texts)
    structured = extract_chapter_structure(combined_text, topic)
    if not structured:
        return "⚠️ Topic not found clearly in NCERT content."
    what, how, why = map_flashcard_fields(structured)
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

subject = st.selectbox("Select Subject", options=list(SUBJECTS.keys()))
topic = st.text_input("Enter Concept / Topic (e.g., Freedom, Groups, Constitution)")

if st.button("Generate Flashcard"):
    texts = load_subject_text(subject)
    if not texts:
        st.error(f"No PDFs found for subject '{subject}'.")
    else:
        result = generate_flashcard(texts, topic)
        st.markdown(result)
