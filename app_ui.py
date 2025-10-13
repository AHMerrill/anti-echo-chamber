# app_ui.py
# Streamlit UI for Anti Echo Chamber — optimized for memory and performance.

import os
import json
import numpy as np
import requests
import tempfile
import gc
import streamlit as st
from pathlib import Path
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import fitz  # PyMuPDF for PDFs
import chromadb
import warnings

# ====================================================
# ENVIRONMENT SETTINGS
# ====================================================

warnings.filterwarnings("ignore", message="Token indices sequence length is longer")
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================================================
# CONFIGURATION
# ====================================================

HF_DATASET_ID = "zanimal/anti-echo-artifacts"
REPO_OWNER = "AHMerrill"
REPO_NAME = "anti-echo-chamber"
BRANCH = "main"

TOPIC_MODEL_NAME = "intfloat/e5-base-v2"
STANCE_MODEL_NAME = "intfloat/e5-base-v2"
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"

st.set_page_config(page_title="Anti Echo Chamber", layout="wide")

# ====================================================
# UTILITIES
# ====================================================

def load_text(file):
    """Handle text, PDF, and HTML uploads."""
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
    """Ensure metadata is safe for JSON and Chroma."""
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean

# ====================================================
# CACHED RESOURCES
# ====================================================

@st.cache_resource(show_spinner="Building Chroma from Hugging Face dataset (first time only)...")
def build_chroma_from_hf():
    """Rebuild local Chroma from the HF dataset, cached for session reuse."""
    CHROMA_DIR = Path("chroma_db")
    if CHROMA_DIR.exists():
        return chromadb.PersistentClient(path=str(CHROMA_DIR))

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    topic_coll = client.get_or_create_collection("news_topic", metadata={"hnsw:space": "cosine"})
    stance_coll = client.get_or_create_collection("news_stance", metadata={"hnsw:space": "cosine"})

    REGISTRY_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/artifacts/artifacts_registry.json"
    REGISTRY = requests.get(REGISTRY_URL, timeout=30).json()

    for b in REGISTRY.get("batches", []):
        paths = b.get("paths") or {}
        if not all(k in paths for k in ["embeddings_topic", "embeddings_stance", "metadata_topic", "metadata_stance"]):
            continue

        # Load vectors
        t_vecs = np.load(hf_hub_download(HF_DATASET_ID, paths["embeddings_topic"], repo_type="dataset"))["arr_0"]
        s_vecs = np.load(hf_hub_download(HF_DATASET_ID, paths["embeddings_stance"], repo_type="dataset"))["arr_0"]
        t_vecs = np.nan_to_num(t_vecs)
        s_vecs = np.nan_to_num(s_vecs)

        # Load metadata
        t_meta = [sanitize(json.loads(l)) for l in open(hf_hub_download(HF_DATASET_ID, paths["metadata_topic"], repo_type="dataset"), encoding="utf-8")]
        s_meta = [sanitize(json.loads(l)) for l in open(hf_hub_download(HF_DATASET_ID, paths["metadata_stance"], repo_type="dataset"), encoding="utf-8")]

        # Upsert
        t_ids = [m.get("row_id", f"{m.get('id','?')}::topic::0") for m in t_meta]
        s_ids = [m.get("row_id", f"{m.get('id','?')}::stance::0") for m in s_meta]
        topic_coll.upsert(ids=t_ids, embeddings=t_vecs.tolist(), metadatas=t_meta)
        stance_coll.upsert(ids=s_ids, embeddings=s_vecs.tolist(), metadatas=s_meta)

    return client

@st.cache_resource
def load_models():
    """Load all models once per session."""
    topic_model = SentenceTransformer(TOPIC_MODEL_NAME)
    stance_model = SentenceTransformer(STANCE_MODEL_NAME)
    tok_sum = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
    model_sum = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)
    return topic_model, stance_model, tok_sum, model_sum

client = build_chroma_from_hf()
topic_coll = client.get_collection("news_topic")
stance_coll = client.get_collection("news_stance")

topic_model, stance_model, tok_sum, model_sum = load_models()

# ====================================================
# SUMMARIZATION
# ====================================================

def summarize_text(text):
    inputs = tok_sum([text], return_tensors="pt", truncation=True, max_length=1024)
    with st.spinner("Summarizing article..."):
        summary_ids = model_sum.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    return tok_sum.batch_decode(summary_ids, skip_special_tokens=True)[0].strip()

# ====================================================
# STREAMLIT UI
# ====================================================

st.title("🧭 Anti Echo Chamber")
st.caption("Explore articles with **similar topics** but **different perspectives**")

uploaded = st.file_uploader("Upload an article (.txt, .pdf, or .html)", type=["txt", "pdf", "html"])

if uploaded:
    text = load_text(uploaded)
    st.text_area("Extracted text", text[:2000] + ("..." if len(text) > 2000 else ""), height=200)

    with st.spinner("Analyzing and embedding your article..."):
        summary = summarize_text(text)
        stance_vec = stance_model.encode([summary], normalize_embeddings=True)[0]

        # Topic embedding: combine summary + main body
        topic_chunks = [summary, text[:3000]]
        topic_vecs = topic_model.encode(topic_chunks, normalize_embeddings=True)
        topic_vec_mean = topic_vecs.mean(axis=0)

    # === Display Analysis ===
    st.markdown("### 🧠 Analysis Summary")
    st.markdown("**Detected Topic Vector:** *(embedding-based)*")
    st.markdown("> Represents what the model believes the article is about — combining summary and full text.")
    st.markdown(f"**Generated Summary for Stance Analysis:**\n\n> {summary}")

    with st.spinner("Finding similar topics..."):
        results = topic_coll.query(
            query_embeddings=[topic_vec_mean.tolist()],
            n_results=100,
            include=["metadatas"]
        )

    all_results = [m for batch in results["metadatas"] for m in batch]

    if not all_results:
        st.warning("No matching topics found yet. The database updates continuously — please check back later.")
    else:
        st.markdown("### 📰 Results: Similar Topics, Contrasting Perspectives")
        st.caption(
            "Articles are retrieved by **topic similarity**, then ranked by **increasing stance similarity** — "
            "so those listed first are most likely to present opposing viewpoints. "
            "If your desired topic or stance isn't visible yet, please check back later as the database expands."
        )

        stance_summaries = [m.get("stance_summary", m.get("summary", m.get("title", ""))) for m in all_results]
        stance_embeddings = np.array([stance_model.encode([s], normalize_embeddings=True)[0] for s in stance_summaries])
        stance_sims = cosine_similarity([stance_vec], stance_embeddings)[0]

        pairs = sorted(zip(all_results, stance_sims), key=lambda x: x[1])

        def sim_label(s):
            if s < 0.2: return "🟥 Very Dissimilar"
            elif s < 0.4: return "🟧 Dissimilar"
            elif s < 0.6: return "🟨 Somewhat Similar"
            elif s < 0.8: return "🟩 Similar"
            else: return "🟦 Very Similar"

        for meta, sim in pairs[:10]:
            topic_display = meta.get("topic_label") or meta.get("inferred_topic") or "(topic unknown)"
            st.markdown(
                f"**{meta.get('title','(untitled)')}**  \n"
                f"Source: {meta.get('domain','unknown')}  \n"
                f"Topic: *{topic_display}*  \n"
                f"Stance Similarity: {sim:.2f} ({sim_label(sim)})  \n"
                f"[Read original article]({meta.get('url','#')})"
            )

    # Cleanup after processing
    del text, summary, stance_vec, topic_vec_mean
    gc.collect()

# ====================================================
# RESET BUTTON
# ====================================================

st.divider()
if st.button("🧹 Clear memory / restart app"):
    st.cache_resource.clear()
    st.session_state.clear()
    st.experimental_rerun()
