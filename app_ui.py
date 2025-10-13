import streamlit as st
import tempfile, os, json, torch, chromadb, numpy as np, requests, yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import trafilatura
from PyPDF2 import PdfReader
import shutil

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
# FORCE CHROMA REBUILD EACH RUN
# =========================================================
def rebuild_chroma_from_hf():
    st.info("Rebuilding Chroma from Hugging Face dataset — this may take a few minutes.")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    topic_coll = client.get_or_create_collection(CONFIG["chroma_collections"]["topic"], metadata={"hnsw:space": "cosine"})
    stance_coll = client.get_or_create_collection(CONFIG["chroma_collections"]["stance"], metadata={"hnsw:space": "cosine"})

    REG_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/artifacts/artifacts_registry.json"
    REGISTRY = requests.get(REG_URL, timeout=20).json()

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

    for b in REGISTRY.get("batches", []):
        paths = b.get("paths") or {}
        if not all(k in paths for k in ["embeddings_topic","embeddings_stance","metadata_topic","metadata_stance"]):
            continue
        t_vecs = np.load(hf_hub_download(HF_DATASET_ID, paths["embeddings_topic"], repo_type="dataset"))["arr_0"]
        s_vecs = np.load(hf_hub_download(HF_DATASET_ID, paths["embeddings_stance"], repo_type="dataset"))["arr_0"]

        if not np.isfinite(t_vecs).all():
            t_vecs = np.nan_to_num(t_vecs)
        if not np.isfinite(s_vecs).all():
            s_vecs = np.nan_to_num(s_vecs)

        t_meta = [sanitize(json.loads(l)) for l in open(hf_hub_download(HF_DATASET_ID, paths["metadata_topic"], repo_type="dataset"), encoding="utf-8")]
        s_meta = [sanitize(json.loads(l)) for l in open(hf_hub_download(HF_DATASET_ID, paths["metadata_stance"], repo_type="dataset"), encoding="utf-8")]
        t_ids = [m.get("row_id", f"{m.get('id','?')}::topic::0") for m in t_meta]
        s_ids = [m.get("row_id", f"{m.get('id','?')}::stance::0") for m in s_meta]

        if not (len(t_ids) == len(t_vecs) == len(t_meta)):
            st.warning(f"Length mismatch in topic batch {b.get('batch_id')}")
            continue
        if not (len(s_ids) == len(s_vecs) == len(s_meta)):
            st.warning(f"Length mismatch in stance batch {b.get('batch_id')}")
            continue

        chunk = 5000
        for i in range(0, len(t_ids), chunk):
            j = i + chunk
            topic_coll.upsert(
                ids=t_ids[i:j],
                embeddings=t_vecs[i:j].tolist(),
                metadatas=t_meta[i:j],
            )
        for i in range(0, len(s_ids), chunk):
            j = i + chunk
            stance_coll.upsert(
                ids=s_ids[i:j],
                embeddings=s_vecs[i:j].tolist(),
                metadatas=s_meta[i:j],
            )
    st.success("Chroma rebuild complete.")


rebuild_chroma_from_hf()

# =========================================================
# LOAD MODELS (CPU ONLY)
# =========================================================
topic_model = SentenceTransformer(CONFIG["embeddings"]["topic_model"])
summarizer_name = CONFIG.get("summarizer", {}).get("model", "facebook/bart-large-cnn")
tok_sum = AutoTokenizer.from_pretrained(summarizer_name)
model_sum = AutoModelForSeq2SeqLM.from_pretrained(summarizer_name)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
topic_coll = client.get_collection(CONFIG["chroma_collections"]["topic"])

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
    inputs = tok_sum([text], return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = model_sum.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    return tok_sum.batch_decode(out, skip_special_tokens=True)[0]

def topic_embed(text):
    return topic_model.encode([text], convert_to_numpy=True)[0]

# =========================================================
# RETRIEVAL AND GROUPING LOGIC
# =========================================================
def categorize_similarity(score):
    if score < 0.3:
        return "Dissimilar"
    elif score < 0.5:
        return "Somewhat dissimilar"
    elif score < 0.7:
        return "Somewhat similar"
    elif score < 0.85:
        return "Similar"
    else:
        return "Nearly identical"

def find_similar_articles(topic_vec, top_n=100):
    res = topic_coll.query(
        query_embeddings=[topic_vec.tolist()],
        n_results=top_n,
        include=["metadatas", "embeddings"]
    )
    results = []
    for meta, tvec in zip(res["metadatas"][0], res["embeddings"][0]):
        sim = float(np.dot(topic_vec, tvec) / (np.linalg.norm(topic_vec) * np.linalg.norm(tvec)))
        results.append({"meta": meta, "similarity": sim})
    results.sort(key=lambda x: x["similarity"])  # ascending order (dissimilar → similar)
    return results

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Anti Echo Chamber", layout="wide")
st.title("Anti Echo Chamber: Topic Argument Spectrum")

st.markdown("""
This view lists **articles on similar topics**, ordered from **most dissimilar arguments to most similar ones**.
If you don’t see clear counterpoints, our collection is continuously expanding — check back soon for more diverse perspectives.
""")

uploaded = st.file_uploader("Upload an article (.txt, .pdf, .html)", type=["txt", "pdf", "html"])

if uploaded:
    with st.spinner("Extracting text..."):
        text = extract_text(uploaded)
    st.subheader("Extracted Text Preview")
    st.text_area("", text[:2000] + ("..." if len(text) > 2000 else ""), height=300)

    if st.button("Analyze and Find Related Topics"):
        with st.spinner("Summarizing and embedding..."):
            summary = summarize_text(text)
            topic_vec = topic_embed(text)
        st.subheader("Summary")
        st.write(summary)

        with st.spinner("Retrieving related articles..."):
            results = find_similar_articles(topic_vec, top_n=100)

        groups = {"Dissimilar": [], "Somewhat dissimilar": [], "Somewhat similar": [], "Similar": [], "Nearly identical": []}
        for r in results:
            cat = categorize_similarity(r["similarity"])
            groups[cat].append(r)

        for label in ["Dissimilar", "Somewhat dissimilar", "Somewhat similar", "Similar", "Nearly identical"]:
            items = groups[label]
            if not items:
                continue
            with st.expander(f"{label} ({len(items)})", expanded=(label == "Dissimilar")):
                for i, r in enumerate(items[:10]):
                    m = r["meta"]
                    st.markdown(f"### [{m.get('title','Untitled')}]({m.get('url','#')})")
                    st.write(f"**Source:** {m.get('source','?')} | **Cosine similarity:** {r['similarity']:.3f}")
                    st.write(m.get("stance_summary") or m.get("title",""))
                    st.divider()
                if len(items) > 10:
                    if st.button(f"Show all {label}", key=f"show_{label}"):
                        for r in items[10:]:
                            m = r["meta"]
                            st.markdown(f"### [{m.get('title','Untitled')}]({m.get('url','#')})")
                            st.write(f"**Source:** {m.get('source','?')} | **Cosine similarity:** {r['similarity']:.3f}")
                            st.write(m.get("stance_summary") or m.get("title",""))
                            st.divider()
else:
    st.info("Upload an article to begin analysis.")
