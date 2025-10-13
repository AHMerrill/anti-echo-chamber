import streamlit as st
import tempfile, os, re, json, torch, chromadb, numpy as np, requests, yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import trafilatura
from PyPDF2 import PdfReader

# =========================================================
# CONFIG LOAD FROM GITHUB
# =========================================================
REPO_OWNER = "AHMerrill"
REPO_NAME = "anti-echo-chamber"
BRANCH = "main"
CONFIG_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/config/config.yaml"

def fetch_config(url):
    try:
        txt = requests.get(url, timeout=20).text
        if not txt.strip():
            raise ValueError("empty config")
        return yaml.safe_load(txt)
    except Exception as e:
        st.warning(f"Config fetch failed ({e}), using defaults.")
        return {
            "hf_dataset_id": "zanimal/anti-echo-artifacts",
            "chroma_collections": {"topic": "topics", "stance": "stances"},
            "embeddings": {
                "topic_model": "sentence-transformers/all-MiniLM-L6-v2",
                "stance_model": "sentence-transformers/all-mpnet-base-v2",
                "dim": 384,
                "dtype": "float32"
            },
            "summarizer": {"model": "facebook/bart-large-cnn"}
        }

CONFIG = fetch_config(CONFIG_URL)
PROJECT_ROOT = Path(".").resolve()
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
HF_DATASET_ID = CONFIG["hf_dataset_id"]

# =========================================================
# AUTO-REBUILD CHROMA FROM HF (if missing)
# =========================================================
def ensure_chroma_from_hf():
    if CHROMA_DIR.exists():
        return
    st.info("Rebuilding Chroma from Hugging Face dataset — first startup only.")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    topic_coll = client.get_or_create_collection(CONFIG["chroma_collections"]["topic"], metadata={"hnsw:space": "cosine"})
    stance_coll = client.get_or_create_collection(CONFIG["chroma_collections"]["stance"], metadata={"hnsw:space": "cosine"})

    REG_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/artifacts/artifacts_registry.json"
    REGISTRY = requests.get(REG_URL, timeout=20).json()

    for b in REGISTRY.get("batches", []):
        paths = b.get("paths") or {}
        if not all(k in paths for k in ["embeddings_topic","embeddings_stance","metadata_topic","metadata_stance"]):
            continue
        t_vecs = np.load(hf_hub_download(HF_DATASET_ID, paths["embeddings_topic"], repo_type="dataset"))["arr_0"]
        s_vecs = np.load(hf_hub_download(HF_DATASET_ID, paths["embeddings_stance"], repo_type="dataset"))["arr_0"]
        t_meta = [json.loads(l) for l in open(hf_hub_download(HF_DATASET_ID, paths["metadata_topic"], repo_type="dataset"), encoding="utf-8")]
        s_meta = [json.loads(l) for l in open(hf_hub_download(HF_DATASET_ID, paths["metadata_stance"], repo_type="dataset"), encoding="utf-8")]
        t_ids = [m.get("row_id", f"{m.get('id','?')}::topic::0") for m in t_meta]
        s_ids = [m.get("row_id", f"{m.get('id','?')}::stance::0") for m in s_meta]
        topic_coll.upsert(ids=t_ids, embeddings=t_vecs.tolist(), metadatas=t_meta)
        stance_coll.upsert(ids=s_ids, embeddings=s_vecs.tolist(), metadatas=s_meta)
    st.success("Chroma rebuild complete.")

ensure_chroma_from_hf()

# =========================================================
# MODEL LOAD
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
topic_model = SentenceTransformer(CONFIG["embeddings"]["topic_model"], device=device)
stance_model = SentenceTransformer(CONFIG["embeddings"]["stance_model"], device=device)
summarizer_name = CONFIG.get("summarizer", {}).get("model", "facebook/bart-large-cnn")
tok_sum = AutoTokenizer.from_pretrained(summarizer_name)
model_sum = AutoModelForSeq2SeqLM.from_pretrained(summarizer_name).to(device)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
topic_coll = client.get_collection(CONFIG["chroma_collections"]["topic"])
stance_coll = client.get_collection(CONFIG["chroma_collections"]["stance"])

# =========================================================
# HELPERS
# =========================================================
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
    res = topic_coll.query(query_embeddings=[topic_vec.tolist()], n_results=top_n*3, include=["metadatas","embeddings"])
    results = []
    for meta, tvec in zip(res["metadatas"][0], res["embeddings"][0]):
        sid = meta.get("id") + "::stance::0"
        try:
            srec = stance_coll.get(ids=[sid], include=["embeddings","metadatas"])
            if not srec["embeddings"]:
                continue
            stance_other = np.array(srec["embeddings"][0][0])
            topic_sim = float(np.dot(topic_vec, tvec)/(np.linalg.norm(topic_vec)*np.linalg.norm(tvec)))
            stance_sim = float(np.dot(stance_vec, stance_other)/(np.linalg.norm(stance_vec)*np.linalg.norm(stance_other)))
            results.append({
                "meta": meta,
                "topic_sim": topic_sim,
                "stance_sim": stance_sim,
                "contrast": topic_sim - stance_sim
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["contrast"], reverse=True)
    return results[:top_n]

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Anti Echo Chamber", layout="wide")
st.title("Anti Echo Chamber: Find Opposing Perspectives")

uploaded = st.file_uploader("Upload an article (.txt, .pdf, .html)", type=["txt","pdf","html"])

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
