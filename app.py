import os
import zipfile
from pathlib import Path
import streamlit as st
from pypdf import PdfReader

ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

FILE_ID = "1GoY0DZj1KLdC0Xvur0tQlvW_993biwcZ"  # your Google Drive file

# -------------------------
# Download & Extract ZIPs
# -------------------------
def download_and_extract():
    import gdown
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    def extract_zip(zip_path, target_dir):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
        # recursively extract nested zips
        for zfile in Path(target_dir).rglob("*.zip"):
            extract_zip(zfile, zfile.parent / zfile.stem)
            zfile.unlink()  # optional: delete nested zip after extraction

    extract_zip(ZIP_PATH, EXTRACT_DIR)
    st.success("✅ NCERT PDFs extracted!")

# -------------------------
# Read PDFs and check content
# -------------------------
def read_pdf(path):
    try:
        reader = PdfReader(str(path))
        text = " ".join([p.extract_text() or "" for p in reader.pages])
        return text.strip()
    except:
        return ""

def load_all_texts():
    texts = []
    loaded_pdfs = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = read_pdf(pdf)
        if text:
            texts.append(text)
            loaded_pdfs.append(str(pdf))
        else:
            st.warning(f"No text found in: {pdf}")
    st.write(f"📄 Loaded {len(texts)} PDFs with readable content:")
    for p in loaded_pdfs:
        st.write(f"- {p}")
    return texts

# -------------------------
# Streamlit App
# -------------------------
st.title("📘 NCERT PDF Loader")

if st.button("📥 Load NCERT PDFs"):
    download_and_extract()
    texts = load_all_texts()


# --------------------------------------------
# SEMANTIC CHUNKING
# --------------------------------------------
def semantic_chunking(text, embedder, max_words=180, sim_threshold=0.65):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s for s in sentences if len(s.split()) > 6]
    if len(sentences) < 2:
        return sentences

    embeddings = embedder.encode(sentences, convert_to_numpy=True)
    chunks = []
    current = [sentences[0]]
    current_emb = embeddings[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([current_emb], [embeddings[i]])[0][0]
        length = sum(len(s.split()) for s in current)
        if sim < sim_threshold or length > max_words:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_emb = embeddings[i]
        else:
            current.append(sentences[i])
            current_emb = np.mean([current_emb, embeddings[i]], axis=0)

    if current:
        chunks.append(" ".join(current))
    return chunks

# --------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# --------------------------------------------
def retrieve_relevant_chunks(chunks, embeddings, query, mode="NCERT", top_k=TOP_K):
    if len(chunks) == 0:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    threshold = SIMILARITY_THRESHOLD_UPSC if mode=="UPSC" else SIMILARITY_THRESHOLD_NCERT
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked if s >= threshold][:top_k]

# --------------------------------------------
# FLASHCARD GENERATION
# --------------------------------------------
def generate_flashcard(chunks, topic):
    sentences = []
    for ch in chunks:
        for s in re.split(r'(?<=[.?!])\s+', ch):
            if len(s.split()) > 8:
                sentences.append(s.strip())
    if not sentences:
        return None
    overview = sentences[0]
    explanation = " ".join(sentences[1:6])
    conclusion = "This concept is important for governance and society."
    points = [" ".join(s.split()[:20]) for s in sentences[1:6]]
    return {
        "title": topic.title(),
        "content": f"""
**Concept Overview**
{overview}

**Explanation**
{explanation}

**Conclusion**
{conclusion}

**Key Points**
- {"\n- ".join(points)}
"""
    }

# --------------------------------------------
# SIDEBAR: DOWNLOAD PDFs
# --------------------------------------------
with st.sidebar:
    if st.button("📥 Load NCERT PDFs", key="load_pdfs"):
        download_and_extract()

# --------------------------------------------
# MAIN UI
# --------------------------------------------
topic = st.text_input("Enter topic (e.g. Fundamental Rights)")
mode = st.radio("Select Depth", ["NCERT", "UPSC"], horizontal=True)

if st.button("Generate Flashcard", key="gen_flashcard"):
    # Load texts & chunks
    texts = load_all_texts()
    if not texts:
        st.warning("⚠️ No PDF content loaded. Check extraction or file permissions.")
    else:
        all_chunks = []
        for t in texts:
            all_chunks.extend(semantic_chunking(t, embedder))

        embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
        relevant = retrieve_relevant_chunks(all_chunks, embeddings, topic, mode)
        card = generate_flashcard(relevant, topic)
        if card:
            st.markdown(f"## 📘 {card['title']}")
            st.markdown(card["content"])
        else:
            st.warning("No relevant content found for this topic.")
