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

CONCEPT_TEMPLATES = {
    "constitution": {
        "what": "The Constitution of India is the supreme law of the country that defines the framework of governance, distribution of powers, and fundamental rights of citizens.",
        "when": "It was adopted on 26 November 1949 and came into force on 26 January 1950.",
        "how": "It establishes the structure of government, distributes powers between Union and States, and provides mechanisms such as judicial review and amendment procedures.",
        "why": "It ensures rule of law, protects fundamental rights, and maintains democratic governance.",
        "articles": "Articles 1–395; Part III (Fundamental Rights), Part IV (DPSP), Article 368 (Amendment)."
    },

    "election commission of india": {
        "what": "The Election Commission of India is an independent constitutional body responsible for conducting free and fair elections.",
        "when": "It was established on 25 January 1950 under Article 324 of the Constitution.",
        "how": "It supervises elections, prepares electoral rolls, enforces the Model Code of Conduct, and regulates political parties.",
        "why": "It ensures democratic legitimacy and prevents electoral malpractice.",
        "articles": "Article 324 of the Constitution."
    }
}



# ================= FLASHCARD LOGIC =================
def generate_clean_flashcard(texts, topic):
    topic_key = topic.lower().strip()

    # 1️⃣ Load base template
    base = CONCEPT_TEMPLATES.get(topic_key, None)

    if not base:
        return "⚠️ Topic not found in knowledge base. Please add a template."

    # 2️⃣ Extract supporting context from NCERT (optional enrichment)
    full_text = clean_text(" ".join(texts))
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    topic_embedding = model.encode([topic])
    sent_embeddings = model.encode(sentences)
    scores = cosine_similarity(topic_embedding, sent_embeddings)[0]

    enriched = [
        s for s, sc in sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
        if sc > 0.45 and len(s.split()) > 10
    ][:2]

    enrichment = enriched[0] if enriched else ""

    # 3️⃣ Final structured output
    return f"""
### 📘 {topic.title()} — UPSC Concept Note

**What is it?**  
{base['what']}

**When was it established?**  
{base['when']}

**How does it work?**  
{base['how']}

**Why is it important?**  
{base['why']}

**Relevant Articles / Sections**  
{base['articles']}

**NCERT Context / Explanation**  
{enrichment if enrichment else "NCERT discusses this concept in the context of democratic governance and constitutional values."}
"""







# ================= STREAMLIT UI =================
download_and_extract()

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter Institution / Topic (e.g., Constitution, Election Commission, RBI)")

if st.button("Generate Flashcard"):
    texts = load_all_text(subject)
    if not texts:
        st.warning("⚠️ No readable content found for this subject.")
    else:
        flashcard = generate_clean_flashcard(texts, topic)
        st.markdown(flashcard)
