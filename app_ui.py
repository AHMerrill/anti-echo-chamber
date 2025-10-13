# app_ui.py

import streamlit as st
import os
import json
import numpy as np
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import tempfile
import fitz  # PyMuPDF for PDFs
from bs4 import BeautifulSoup

# ==============================
# CONFIGURATION
# ==============================
HF_DATASET_ID = "zanimal/anti-echo-artifacts"
REPO_OWNER = "AHMerrill"
REPO_NAME = "anti-echo-chamber"
BRANCH = "main"

st.set_page_config(page_title="Anti Echo Chamber", layout="wide")

# ==============================
# UTILITIES
# ==============================

def load_text(file):
    ext = Path(file.name).suffix.lower()
    if ext == ".txt":
        return file.read().decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp.flush()
            with fitz.open(tmp.name) as doc:
                for page in doc:
                    text += page.get_text()
        return text
    elif ext == ".html":
        html = file.read().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")
    else:
        raise ValueError("Unsupported file type")

def sanitize(meta):
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean

# ==============================
# CACHED STARTUP: rebuild Chroma from HF
# ==============================

@st.cache_resource(show_spinner="Rebuilding Chroma from Hugging Face dataset (one-time per session)...")
def build_chroma_from_hf():
    CHROMA_DIR = Path("chroma_db")
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    topic_coll = client.get_or_create_collection("topic", metadata={"hnsw:space": "cosine"})
    stance_coll = client.get_or_create_collection("stance", metadata={"hnsw:space": "cosine"})

    REGISTRY_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/artifacts/artifacts_registry.json"
    REGISTRY = requests.get(REGISTRY_URL, timeout=20).json()

    for b in REGISTRY.get("batches", []):
        paths = b.get("paths") or {}
        need = ["embeddings_topic", "embeddings_stance", "metadata_topic", "metadata_stance"]
        if not all(k in paths for k in need):
            continue

        # Load vectors
        t_vecs = np.load(hf_hub_download(HF_DATASET_ID, paths["embeddings_topic"], repo_type="dataset"))["arr_0"]
        s_vecs = np.load(hf_hub_download(HF_DATASET_ID, paths["embeddings_stance"], repo_type="dataset"))["arr_0"]
        if not np.isfinite(t_vecs).all():
            t_vecs = np.nan_to_num(t_vecs)
        if not np.isfinite(s_vecs).all():
            s_vecs = np.nan_to_num(s_vecs)

        # Load metadata
        t_meta = [sanitize(json.loads(l)) for l in open(hf_hub_download(HF_DATASET_ID, paths["metadata_topic"], repo_type="dataset"), encoding="utf-8")]
        s_meta = [sanitize(json.loads(l)) for l in open(hf_hub_download(HF_DATASET_ID, paths["metadata_stance"], repo_type="dataset"), encoding="utf-8")]

        t_ids = [m.get("row_id", f"{m.get('id','?')}::topic::0") for m in t_meta]
        s_ids = [m.get("row_id", f"{m.get('id','?')}::stance::0") for m in s_meta]

        topic_coll.upsert(ids=t_ids, embeddings=t_vecs.tolist(), metadatas=t_meta)
        stance_coll.upsert(ids=s_ids, embeddings=s_vecs.tolist(), metadatas=s_meta)

    return client, topic_coll, stance_coll

client, topic_coll, stance_coll = build_chroma_from_hf()

# ==============================
# MODELS
# ==============================
@st.cache_resource
def load_models():
    topic_model = SentenceTransformer("all-mpnet-base-v2")
    stance_model = SentenceTransformer("all-mpnet-base-v2")
    summarizer_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    return topic_model, stance_model, summarizer_tokenizer, summarizer_model

topic_model, stance_model, tok_sum, model_sum = load_models()

def summarize_text(text):
    inputs = tok_sum([text], return_tensors="pt", truncation=True, max_length=1024)
    with st.spinner("Summarizing article..."):
        summary_ids = model_sum.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    return tok_sum.batch_decode(summary_ids, skip_special_tokens=True)[0].strip()

# ==============================
# APP UI
# ==============================
st.title("Anti Echo Chamber")
st.caption("Find articles with similar topics but different perspectives")

uploaded = st.file_uploader("Upload an article (.txt, .pdf, or .html)", type=["txt", "pdf", "html"])

if uploaded:
    text = load_text(uploaded)
    st.text_area("Extracted text", text[:2000] + ("..." if len(text) > 2000 else ""), height=200)

    with st.spinner("Analyzing and embedding your article..."):
        summary = summarize_text(text)
        stance_vec = stance_model.encode([summary])[0]
        topic_vecs = topic_model.encode([summary, text])  # simple topic approximation

    with st.spinner("Querying database..."):
        results = topic_coll.query(
            query_embeddings=topic_vecs.tolist(),
            n_results=100,
            include=["metadatas", "embeddings"]
        )

    flat_results = []
    for res in results["metadatas"]:
        flat_results.extend(res)

    if not flat_results:
        st.warning("No matching topics found. Our database is expanding — please check back later.")
    else:
        st.markdown("### Results: Similar Topics Sorted by Argument Dissimilarity")

        stance_vectors = np.array([stance_model.encode([m.get("stance_summary","")])[0] for m in flat_results])
        stance_sims = cosine_similarity([stance_vec], stance_vectors)[0]

        pairs = list(zip(flat_results, stance_sims))
        pairs.sort(key=lambda x: x[1])  # ascending: most dissimilar first

        def label(sim):
            if sim < 0.2: return "Very Dissimilar"
            elif sim < 0.4: return "Dissimilar"
            elif sim < 0.6: return "Somewhat Similar"
            elif sim < 0.8: return "Similar"
            else: return "Very Similar"

        for meta, sim in pairs[:10]:
            st.markdown(
                f"**{meta.get('title','(untitled)')}**  \n"
                f"Source: {meta.get('domain','unknown')}  \n"
                f"Similarity: {sim:.2f} ({label(sim)})  \n"
                f"[Read original article]({meta.get('url','#')})"
            )
