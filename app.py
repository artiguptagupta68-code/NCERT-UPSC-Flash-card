def generate_flashcard(texts, topic):
    combined = " ".join(texts)
    combined = clean_text(combined)

    if len(combined.split()) < 80:
        return "⚠️ No meaningful content found."

    # Sentence split
    sentences = re.split(r'(?<=[.!?])\s+', combined)

    # Chunking
    chunks, buffer = [], []
    for s in sentences:
        if len(s.split()) < 6:
            continue
        buffer.append(s)
        if sum(len(x.split()) for x in buffer) >= 120:
            chunks.append(" ".join(buffer))
            buffer = []
    if buffer:
        chunks.append(" ".join(buffer))

    # Semantic scoring
    topic_vec = model.encode([topic])
    chunk_vecs = model.encode(chunks)

    scored = [
        (chunk, cosine_similarity([vec], topic_vec)[0][0])
        for chunk, vec in zip(chunks, chunk_vecs)
    ]

    if not scored:
        return "⚠️ No meaningful content found."

    # Sort by relevance
    scored.sort(key=lambda x: x[1], reverse=True)
    top_text = " ".join([c for c, _ in scored[:3]])

    # -------- Structured Extraction --------
    sentences = re.split(r'(?<=[.!?])\s+', top_text)

    what_is = []
    when = []
    how = []
    why = []

    for s in sentences:
        s_low = s.lower()

        if any(k in s_low for k in ["is", "refers to", "means", "defined as"]):
            what_is.append(s)

        if any(k in s_low for k in ["adopted", "enacted", "came into force", "established", "1949", "1950"]):
            when.append(s)

        if any(k in s_low for k in ["works", "functions", "implemented", "interpreted", "enforced", "applied"]):
            how.append(s)

        if any(k in s_low for k in ["important", "ensures", "protects", "helps", "essential", "significant"]):
            why.append(s)

    return f"""
### 📘 {topic} — Concept Summary

**What is it?**  
{ " ".join(what_is) if what_is else top_text[:300] + "..." }

**When was it established?**  
{ " ".join(when) if when else "Established through constitutional adoption and subsequent amendments." }

**How does it work?**  
{ " ".join(how) if how else "It functions through laws, institutions, and judicial interpretation." }

**Why is it important?**  
{ " ".join(why) if why else "It strengthens democracy, protects rights, and ensures justice." }
"""
