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

    if any(x in s for x in [
        "edition", "printed", "copyright", "price",
        "isbn", "publication", "reprint", "press",
        "chapter", "page", "figure", "table"
    ]):
        return False

    concept_verbs = [
        "is", "are", "means", "refers", "defines", "explains",
        "ensures", "protects", "establishes", "allows",
        "governs", "regulates", "interprets"
    ]

    if not any(v in s for v in concept_verbs):
        return False

    topic_words = topic.lower().split()
    if not any(t in s for t in topic_words):
        return False

    return True


# ================= FLASHCARD LOGIC =================
def generate_upscready_flashcard(texts, topic):
    """
    Generates UPSC-ready flashcards with:
    - Definition / What
    - Establishment / When
    - Functioning / How
    - Importance / Why
    - Articles / Sections / Clauses
    - Examples / Real-life context
    """

    full_text = clean_text(" ".join(texts))
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    if len(sentences) < 20:
        return "⚠️ Not enough content to generate a meaningful flashcard."

    # --- Encode topic and sentences ---
    topic_embedding = model.encode([topic])
    sentence_embeddings = model.encode(sentences)
    similarity_scores = cosine_similarity(topic_embedding, sentence_embeddings)[0]

    # --- Rank sentences by similarity ---
    ranked_sentences = sorted(
        zip(sentences, similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Select top relevant sentences
    top_sentences = [s for s, score in ranked_sentences[:35] if len(s.split()) > 8]

    if not top_sentences:
        return "⚠️ Could not find relevant content."

    # -------- Categorize sentences --------
    what, when, how, why, articles, examples = [], [], [], [], [], []

    for s in top_sentences:
        s_low = s.lower()
        # WHAT: definition / purpose
        if any(k in s_low for k in ["is", "refers to", "means", "defined as", "constitutes"]):
            what.append(s)
        # WHEN: established / formed
        elif any(k in s_low for k in ["established", "formed", "created", "set up", "came into force", "adopted"]):
            when.append(s)
        # HOW: functions / operations / powers
        elif any(k in s_low for k in ["functions", "powers", "responsible for", "administers", "operates", "ensures", "provides"]):
            how.append(s)
        # WHY: importance / significance
        elif any(k in s_low for k in ["important", "significant", "ensures", "safeguards", "role of", "critical"]):
            why.append(s)
        # ARTICLES / SECTIONS
        elif "article" in s_low or "part" in s_low or "section" in s_low or "clause" in s_low:
            articles.append(s)
        # EXAMPLES / REAL-LIFE
        elif any(k in s_low for k in ["example", "instance", "case study", "such as", "applied in"]):
            examples.append(s)

    # -------- Fallbacks if empty --------
    if not what:
        what = [f"The {topic} is a key constitutional/institutional concept in India."]
    if not when:
        if topic.lower() == "constitution":
            when = ["The Constitution of India was adopted on 26 November 1949 and came into force on 26 January 1950."]
        else:
            when = ["This institution/concept was established to serve its constitutional and social purpose."]
    if not how: how = top_sentences[2:4] or ["It functions through its constitutional/organizational powers and responsibilities."]
    if not why: why = top_sentences[4:6] or ["It is important for governance, law, and public administration."]
    if not articles: articles = ["Refer to the relevant Articles, Sections, or Clauses in the Constitution/Act."]
    if not examples: examples = ["Example: Relevant case studies or real-life application."]

    # -------- Construct flashcard --------
    flashcard = f"""
### 📘 {topic} — Advanced UPSC Flashcard

**What is it?**  
{what[0]}

**When was it established?**  
{when[0]}

**How does it work?**  
{how[0]}

**Why is it important?**  
{why[0]}

**Relevant Articles / Sections**  
{articles[0]}

**Examples / Real-life Context**  
{examples[0]}
"""
    return flashcard






# ================= STREAMLIT UI =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Institution / Topic (e.g., Constitution, Election Commission, RBI)")

if st.button("Generate Flashcard"):
    texts = load_all_text(subject)
    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        flashcard = generate_upscready_flashcard(texts, topic)
        st.markdown(flashcard)
