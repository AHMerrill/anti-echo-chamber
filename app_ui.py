import streamlit as st
import tempfile, os, re, json, torch, chromadb, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import trafilatura
from PyPDF2 import PdfReader

# === PATHS ===
PROJECT_ROOT = Path("/content/anti_echo").resolve()
CONFIG_PATH = PROJECT_ROOT / "config_cache/config.yaml"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# === LOAD CONFIG ===
import yaml
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

# === LOAD MODELS ===
device = "cuda" if torch.cuda.is_available() else "cpu"
topic_model = SentenceTransformer(CONFIG["embeddings"]["topic_model"], device=device)
stance_model = SentenceTransformer(CONFIG["embeddings"]["stance_model"], device=device)
summarizer_name = CONFIG.get("summarizer", {}).get("model", "facebook/bart-large-cnn")
tok_sum = AutoTokenizer.from_pretrained(summarizer_name)
model_sum = AutoModelForSeq2SeqLM.from_pretrained(summarizer_name).to(device)

# === CHROMA ===
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
topic_coll = client.get_collection(CONFIG["chroma_collections"]["topic"])
stance_coll = client.get_collection(CONFIG["chroma_collections"]["stance"])

def extract_text(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        pdf = PdfReader(uploaded_file)
        return "\n".join([p.extract_text() or "" for p in pdf.pages])
    elif suffix in (".html", ".htm"):
        html = uploaded_file.read().decode("utf-8", errors="ignore")
        return trafilatura.extract(html) or ""
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

def summarize_text(text):
    inputs = tok_sum([text], return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model_sum.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    return tok_sum.batch_decode(out, skip_special_tokens=True)[0]

def topic_embed(text):
    return topic_model.encode([text], convert_to_numpy=True)[0]

def stance_embed(summary):
    return stance_model.encode([summary], convert_to_numpy=True)[0]

def find_contrasting_articles(topic_vec, stance_vec, top_n=5):
    # Query top N topic neighbors
    res = topic_coll.query(query_embeddings=[topic_vec.tolist()], n_results=top_n * 3, include=["metadatas", "embeddings"])
    results = []
    for meta, tvec in zip(res["metadatas"][0], res["embeddings"][0]):
        sid = meta.get("id") + "::stance::0"
        try:
            srec = stance_coll.get(ids=[sid], include=["embeddings", "metadatas"])
            if not srec["embeddings"]:
                continue
            stance_other = np.array(srec["embeddings"][0][0])
            topic_sim = float(np.dot(topic_vec, tvec) / (np.linalg.norm(topic_vec) * np.linalg.norm(tvec)))
            stance_sim = float(np.dot(stance_vec, stance_other) / (np.linalg.norm(stance_vec) * np.linalg.norm(stance_other)))
            contrast = topic_sim - stance_sim
            results.append({
                "meta": meta,
                "topic_sim": topic_sim,
                "stance_sim": stance_sim,
                "contrast": contrast
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["contrast"], reverse=True)
    return results[:top_n]

# === STREAMLIT UI ===
st.set_page_config(page_title="Anti Echo Chamber", layout="wide")
st.title("Anti Echo Chamber: Find Opposing Perspectives")

uploaded = st.file_uploader("Upload an article (.txt, .pdf, .html)", type=["txt", "pdf", "html"])

if uploaded:
    with st.spinner("Extracting text..."):
        text = extract_text(uploaded)
    st.subheader("Extracted Text Preview")
    st.text_area("", text[:2000] + ("..." if len(text) > 2000 else ""), height=300)

    if st.button("Analyze and Find Opposing Articles"):
        with st.spinner("Summarizing and embedding..."):
            summary = summarize_text(text)
            topic_vec = topic_embed(text)
            stance_vec = stance_embed(summary)
        st.subheader("Summary (stance representation)")
        st.write(summary)

        with st.spinner("Searching for contrasting perspectives..."):
            results = find_contrasting_articles(topic_vec, stance_vec, top_n=5)

        st.subheader("Contrasting Articles")
        for r in results:
            m = r["meta"]
            st.markdown(f"### [{m.get('title','Untitled')}]({m.get('url','#')})")
            st.write(f"**Source:** {m.get('source','?')} | **Topic Sim:** {r['topic_sim']:.2f} | **Stance Sim:** {r['stance_sim']:.2f} | **Contrast:** {r['contrast']:.2f}")
            snippet = m.get("stance_summary") or (m.get("title") or "")[:400]
            st.write(snippet)
            st.divider()
else:
    st.info("Upload an article to begin analysis.")
