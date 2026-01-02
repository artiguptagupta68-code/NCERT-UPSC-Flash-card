import os
import re
import zipfile
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

def is_meaningful_sentence(sentence, topic):
    s = sentence.lower()

    # Reject obvious junk
    if any(x in s for x in [
        "edition", "printed", "copyright", "price",
        "isbn", "publication", "reprint", "press",
        "chapter", "page", "figure", "table"
    ]):
        return False

    # Must contain a concept verb
    concept_verbs = [
        "is", "are", "means", "refers", "defines", "explains",
        "ensures", "protects", "establishes", "allows",
        "governs", "regulates", "interprets"
    ]

    if not any(v in s for v in concept_verbs):
        return False

    # Must be related to topic
    topic_words = topic.lower().split()
    if not any(t in s for t in topic_words):
        return False

    return True

# ================= FLASHCARD LOGIC =================
def generate_flashcard(texts, topic):
    full_text = clean_text(" ".join(texts))

    if len(full_text.split()) < 120:
        return "⚠️ No meaningful content found."

    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    what, when, how, why = [], [], [], []

 for s in sentences:
     if len(s.split()) < 8:
         continue

if not is_meaningful_sentence(s, topic):
    continue

        # WHAT
        if any(k in s_low for k in [
            "is defined as", "refers to", "means", "is a", "is an"
        ]):
            what.append(s_clean)

        # WHEN
        elif any(k in s_low for k in [
            "adopted", "enacted", "came into force", "established", "constitution of"
        ]):
            when.append(s_clean)

        # HOW
        elif any(k in s_low for k in [
            "functions", "works", "implemented", "interpreted",
            "enforced", "operates", "applied"
        ]):
            how.append(s_clean)

        # WHY
        elif any(k in s_low for k in [
            "important", "ensures", "protects", "helps", "essential",
            "significant", "strengthens"
        ]):
            why.append(s_clean)

    # Fallbacks if sections are empty
    if not what:
        what = sentences[:2]

    if not when:
        when = ["The Constitution was adopted in 1950 and continues to evolve through amendments and judicial interpretation."]

    if not how:
        how = ["It functions through laws, institutions, courts, and democratic processes."]

    if not why:
        why = ["It safeguards rights, limits state power, and ensures democratic governance."]

    return f"""
### 📘 {topic} — Concept Summary

**What is it?**  
{' '.join(what[:3])}

**When was it established?**  
{' '.join(when[:2])}

**How does it work?**  
{' '.join(how[:3])}

**Why is it important?**  
{' '.join(why[:3])}
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
