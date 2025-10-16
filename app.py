import os
import io
import re
import json
import time
import base64
import zipfile
import traceback
import tempfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import difflib

import streamlit as st

# Heavy deps
import chromadb
from bs4 import BeautifulSoup
import pdfplumber
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import list_repo_files, hf_hub_download

# OpenAI (python v1 client)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

GIT_REPO = "AHMerrill/anti-echo-chamber"
GIT_RAW = "https://raw.githubusercontent.com/AHMerrill/anti-echo-chamber/main"
HF_REPO = "zanimal/anti-echo-artifacts"
CHROMA_DIR = str((Path.cwd() / "chroma_db").resolve())

CONFIG_URL = f"{GIT_RAW}/config/config.yaml"
TOPICS_JSON_URL = f"{GIT_RAW}/config/topics.json"
POLITICAL_LEANINGS_URL = f"{GIT_RAW}/config/political_leanings.json"
IMPLIED_STANCES_URL = f"{GIT_RAW}/config/implied_stances.json"
SOURCE_BIAS_URL = f"{GIT_RAW}/config/source_bias.json"
TOPIC_ANCHORS_URL = f"{GIT_RAW}/config/topic_anchors.npz"

# --------------------------------------------------------------------------------------
# Streamlit UI scaffold
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Anti Echo Chamber", layout="wide")
st.title("Anti Echo Chamber — Retrieval and Contrastive Analysis")

# Secrets / keys
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))  # optional, public dataset likely fine

# Sidebar controls
st.sidebar.header("Config & Controls")

w_T = st.sidebar.slider("Weight: Topic Overlap (w_T)", 0.0, 3.0, 1.0, 0.1)
w_S = st.sidebar.slider("Weight: Stance Similarity penalty (w_S)", 0.0, 3.0, 1.0, 0.1)
w_B = st.sidebar.slider("Weight: Bias Difference penalty (w_B)", 0.0, 3.0, 1.0, 0.1)
w_Tone = st.sidebar.slider("Weight: Tone Difference penalty (w_Tone)", 0.0, 3.0, 0.5, 0.1)

TOPIC_OVERLAP_THRESHOLD = st.sidebar.slider("Min Topic Overlap threshold", 0.0, 1.0, 0.5, 0.05)
TOP_N_RESULTS = st.sidebar.slider("Top N results per section", 5, 50, 10, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("Links")
st.sidebar.markdown("- GitHub repo: https://github.com/AHMerrill/anti-echo-chamber")
st.sidebar.markdown("- HF dataset: https://huggingface.co/datasets/zanimal/anti-echo-artifacts")

# Manual, hidden key entry (per-session, not persisted)
st.sidebar.markdown("---")
st.sidebar.subheader("Credentials (session-only)")
if "OPENAI_API_KEY" in st.session_state and not OPENAI_API_KEY:
    OPENAI_API_KEY = st.session_state["OPENAI_API_KEY"]
key_input = st.sidebar.text_input("OpenAI API key (hidden)", type="password", value="", help="Used only for this session; prefer Streamlit Secrets for deployments.")
if key_input:
    OPENAI_API_KEY = key_input.strip()
    st.session_state["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_yaml(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return yaml.safe_load(resp.text)

@st.cache_resource(show_spinner=False)
def load_json(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

@st.cache_resource(show_spinner=False)
def load_npz_from_url(url: str) -> Dict[str, np.ndarray]:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmpf:
        tmpf.write(resp.content)
        tmp_path = tmpf.name
    npz = np.load(tmp_path, allow_pickle=False)
    return {k: npz[k] for k in npz.files}

def b64_download_bytes(name: str, data: bytes, mime: str = "application/octet-stream"):
    b64 = base64.b64encode(data).decode()
    return f'<a download="{name}" href="data:{mime};base64,{b64}">Download {name}</a>'

def parse_topics_field(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return [str(t).strip() for t in obj if str(t).strip()]
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(t).strip() for t in arr]
            except Exception:
                pass
        parts = [p.strip() for p in s.split(";") if p.strip()]
        return parts
    return []

def topic_overlap_score(a_topics, b_topics) -> float:
    a = set([t.strip().lower() for t in parse_topics_field(a_topics)])
    b = set([t.strip().lower() for t in parse_topics_field(b_topics)])
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def interpret_bias(score: float) -> str:
    if score <= -0.6: return "Progressive / Left"
    if -0.6 < score <= -0.2: return "Center-Left"
    if -0.2 < score < 0.2: return "Center / Neutral"
    if 0.2 <= score < 0.6: return "Center-Right"
    if score >= 0.6: return "Conservative / Right"
    return "Unknown"

def short_url(u: str, max_len=90) -> str:
    return (u[:max_len] + "…") if u and len(u) > max_len else (u or "")

# --------------------------------------------------------------------------------------
# Load global configuration and guides
# --------------------------------------------------------------------------------------

with st.spinner("Loading configuration and guides from GitHub..."):
    CONFIG = load_yaml(CONFIG_URL)
    topics_json = load_json(TOPICS_JSON_URL)
    political_leanings = load_json(POLITICAL_LEANINGS_URL)
    implied_stances = load_json(IMPLIED_STANCES_URL)
    source_bias_map = load_json(SOURCE_BIAS_URL)
    topic_anchors = load_npz_from_url(TOPIC_ANCHORS_URL)
    topic_anchor_labels = list(topic_anchors.keys())

st.success("Configs and guides loaded.")

# --------------------------------------------------------------------------------------
# Chroma rebuild from HF
# --------------------------------------------------------------------------------------

st.header("Stage 1 — Rebuild Chroma from Hugging Face")
st.caption("Rebuilds local Chroma collections from `zanimal/anti-echo-artifacts`. Retains multi-topic and multi-stance structure.")

rebuild_now = st.button("Rebuild Chroma")
rebuild_info_slot = st.empty()

@st.cache_resource(show_spinner=False)
def chroma_client():
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)

def load_npz_safely(path):
    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        for key in arr.files:
            if arr[key].ndim == 2:
                return arr[key]
        raise ValueError(f"No 2D arrays in {path}")
    return arr

def load_jsonl(path) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return out

def do_rebuild():
    client = chroma_client()
    # Drop and recreate collections
    for name in [CONFIG["chroma_collections"]["topic"], CONFIG["chroma_collections"]["stance"]]:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    topic_coll = client.create_collection(CONFIG["chroma_collections"]["topic"], metadata={"hnsw:space": "cosine"})
    stance_coll = client.create_collection(CONFIG["chroma_collections"]["stance"], metadata={"hnsw:space": "cosine"})

    files = list_repo_files(HF_REPO, repo_type="dataset", token=HF_TOKEN or None)
    batches = sorted({"/".join(p.split("/")[:2]) for p in files if p.startswith("batches/")})
    seen_topic_ids, seen_stance_ids = set(), set()

    topic_total = stance_total = 0
    for batch in batches:
        rebuild_info_slot.info(f"Processing {batch} ...")
        try:
            topic_npz = hf_hub_download(HF_REPO, f"{batch}/embeddings_topic.npz", repo_type="dataset", token=HF_TOKEN or None)
            stance_npz = hf_hub_download(HF_REPO, f"{batch}/embeddings_stance.npz", repo_type="dataset", token=HF_TOKEN or None)
            meta_topic = hf_hub_download(HF_REPO, f"{batch}/metadata_topic.jsonl", repo_type="dataset", token=HF_TOKEN or None)
            meta_stance = hf_hub_download(HF_REPO, f"{batch}/metadata_stance.jsonl", repo_type="dataset", token=HF_TOKEN or None)

            t_embs = load_npz_safely(topic_npz)
            s_embs = load_npz_safely(stance_npz)
            t_meta = load_jsonl(meta_topic)
            s_meta = load_jsonl(meta_stance)

            # Topic
            t_records = []
            for e, m in zip(t_embs, t_meta):
                rid = m.get("row_id") or f"{m.get('id','unknown')}::topic::0"
                if rid in seen_topic_ids:
                    continue
                seen_topic_ids.add(rid)
                t_records.append((rid, e, m))
            if t_records:
                topic_coll.upsert(
                    ids=[r[0] for r in t_records],
                    embeddings=[r[1].tolist() for r in t_records],
                    metadatas=[r[2] for r in t_records],
                )
            topic_total += len(t_records)

            # Stance
            s_records = []
            for e, m in zip(s_embs, s_meta):
                rid = m.get("row_id") or f"{m.get('id','unknown')}::stance::0"
                if rid in seen_stance_ids:
                    continue
                seen_stance_ids.add(rid)
                s_records.append((rid, e, m))
            if s_records:
                stance_coll.upsert(
                    ids=[r[0] for r in s_records],
                    embeddings=[r[1].tolist() for r in s_records],
                    metadatas=[r[2] for r in s_records],
                )
            stance_total += len(s_records)

        except Exception as e:
            rebuild_info_slot.error(f"Failed {batch}: {e}")
            traceback.print_exc()

    return topic_total, stance_total

if rebuild_now:
    with st.spinner("Rebuilding Chroma..."):
        t_count, s_count = do_rebuild()
    st.success(f"Rebuild complete. Topic vectors: {t_count}, Stance vectors: {s_count}")

# --------------------------------------------------------------------------------------
# Upload + parse article
# --------------------------------------------------------------------------------------

st.header("Stage 2 — Upload and Parse Article (TXT / PDF / HTML)")
uploaded = st.file_uploader("Upload file", type=["txt", "pdf", "html", "htm"])

def extract_text_from_file(file: io.BytesIO, name: str) -> str:
    name = name.lower()
    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    if name.endswith(".html") or name.endswith(".htm"):
        soup = BeautifulSoup(file.read().decode("utf-8", errors="ignore"), "html.parser")
        for s in soup(["script", "style"]):
            s.decompose()
        return soup.get_text(separator=" ")
    raise ValueError("Unsupported file type")

article_text = ""
article_name = ""
if uploaded is not None:
    try:
        article_text = extract_text_from_file(uploaded, uploaded.name).strip()
        article_name = uploaded.name
        st.success(f"Extracted {len(article_text)} characters from {uploaded.name}")
    except Exception as e:
        st.error(f"Failed to parse: {e}")

# --------------------------------------------------------------------------------------
# Source bias detection (heuristic + GPT fallback)
# --------------------------------------------------------------------------------------

st.header("Stage 3 — Source Bias Detection")
col_a, col_b = st.columns([2, 1])
with col_a:
    inferred_source = ""
    if article_text:
        m = re.search(r"https?://([^/\\s]+)", article_text)
        if m:
            domain = m.group(1).lower().replace("www.", "")
            inferred_source = domain.split(".")[0]
    source_input = st.text_input("Detected/entered source", value=inferred_source)

with col_b:
    use_gpt_bias_fallback = st.checkbox("Use GPT fallback if unknown", value=True)

# Suggestions via difflib
suggestions = []
if source_input:
    suggestions = difflib.get_close_matches(source_input.lower().strip(), list(source_bias_map.keys()), n=5, cutoff=0.0)

sel = st.selectbox("Pick a suggested source (or leave to use input)", options=["(use typed value)"] + suggestions)
final_source_choice = source_input.strip()
if sel != "(use typed value)":
    final_source_choice = sel

if "source_confirmed" not in st.session_state:
    st.session_state["source_confirmed"] = False

confirm = st.button("Confirm Source")

bias_info = {"bias_family": "unknown", "bias_score": 0.0, "short_rationale": ""}
confirmed_source_label = None

if confirm and final_source_choice:
    confirmed_source_label = final_source_choice
    if confirmed_source_label in source_bias_map:
        bias_info = source_bias_map[confirmed_source_label]
        st.session_state["source_confirmed"] = True
        st.success(f"Confirmed source: {confirmed_source_label}")
    elif use_gpt_bias_fallback and OPENAI_API_KEY and OpenAI is not None:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"""
You are a media bias researcher.
Given the outlet name "{confirmed_source_label}", infer its general political bias family
(e.g., 'center left', 'center right', 'libertarian', 'progressive left', 'neutral').
Return JSON with:
- bias_family
- bias_score (float, -1.0 = far left, +1.0 = far right)
- short_rationale (brief explanation)
"""
            resp = client.chat.completions.create(
                model=CONFIG["stance_processing"]["llm"]["model"],
                messages=[{"role":"user","content":prompt}],
                max_tokens=128,
                temperature=0.4
            )
            try:
                bias_info = json.loads(resp.choices[0].message.content)
            except Exception:
                bias_info = {
                    "bias_family": "unknown",
                    "bias_score": 0.0,
                    "short_rationale": resp.choices[0].message.content.strip()
                }
            st.session_state["source_confirmed"] = True
            st.info("Source not in map; inferred with GPT.")
        except Exception as e:
            st.session_state["source_confirmed"] = False
            st.warning(f"Could not infer source bias automatically: {e}")
    else:
        st.session_state["source_confirmed"] = True
        st.info("Proceeding without bias metadata.")

if st.session_state["source_confirmed"]:
    st.json({"source": final_source_choice, **bias_info})
else:
    st.info("Confirm a source to continue to topic assignment.")

# --------------------------------------------------------------------------------------
# Topic embedding for uploaded article
# --------------------------------------------------------------------------------------

st.header("Stage 4 — Topic Embeddings & Canonical Topic Assignment")

@st.cache_resource(show_spinner=False)
def load_topic_models(name: str):
    device = "cuda" if torch_available_cuda() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    embedder = SentenceTransformer(name, device=device)
    return tokenizer, embedder

def torch_available_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def sent_tokenize(text: str) -> List[str]:
    # simple sentence split fallback to avoid NLTK download in Streamlit
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def encode_texts(embedder: SentenceTransformer, texts: List[str], normalize: bool) -> np.ndarray:
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=False)
    return np.array(vecs)

def topic_vecs_from_text(embedder: SentenceTransformer, text: str, chunk_tokens: int, normalize: bool) -> np.ndarray:
    sents = sent_tokenize(text)
    if not sents:
        return np.zeros((0, CONFIG["embeddings"]["dim"]), dtype=np.float32)
    if len(sents) < 2:
        return encode_texts(embedder, [" ".join(sents)], normalize=normalize)

    emb = encode_texts(embedder, sents, normalize=normalize)
    k = min(max(1, len(sents)//8), 8)
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(emb)
    segments = [" ".join([s for s, l in zip(sents, labels) if l == lab]) for lab in sorted(set(labels))]
    seg_embs = encode_texts(embedder, segments, normalize=normalize)
    return seg_embs

def match_topics(vec: np.ndarray, anchors: Dict[str, np.ndarray], max_topics: int, threshold: float):
    # anchors are pre-normalized reference vectors
    scores = {label: float(cosine_similarity([vec], [anchors[label]])[0][0]) for label in anchors.keys()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = []
    for i, (label, sim) in enumerate(ranked[:max_topics]):
        if i == 0 or sim >= threshold:
            selected.append({"topic_label": label, "similarity": sim})
    if not selected:
        selected = [{"topic_label": "General / Miscellaneous", "similarity": 0.0}]
    return selected

topic_model_name = CONFIG["embeddings"]["topic_model"]
chunk_tokens = CONFIG["embeddings"]["chunk_tokens"]
normalize = CONFIG["embeddings"]["normalize"]
topic_threshold = CONFIG["topics"]["similarity_threshold"]
max_topics_per_vec = CONFIG["topics"]["max_topics_per_article"]

def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int) -> List[str]:
    words = text.split()
    chunks, current = [], []
    for w in words:
        trial = " ".join(current + [w])
        ids = tokenizer.encode(trial, add_special_tokens=False)
        if len(ids) > max_tokens:
            if current:
                chunks.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        chunks.append(" ".join(current))
    return chunks

topic_results = {}
if article_text and st.session_state.get("source_confirmed"):
    tokenizer, topic_embedder = load_topic_models(topic_model_name)
    text_chunks = chunk_text_by_tokens(article_text, tokenizer, chunk_tokens)
    if not text_chunks:
        st.warning("No text chunks generated for topic embedding.")
    seg_vecs = encode_texts(topic_embedder, text_chunks, normalize=normalize)
    st.write(f"Generated {len(seg_vecs)} topic vectors via {chunk_tokens}-token chunks.")
    if len(seg_vecs) > 0:
        matches_per_vec = [match_topics(v, topic_anchors, max_topics_per_vec, topic_threshold) for v in seg_vecs]
        flat_topics = list(dict.fromkeys([m["topic_label"] for matches in matches_per_vec for m in matches]))[:8]
        topic_results = {
            "vectors": seg_vecs,
            "matches": matches_per_vec,
            "flat_topics": flat_topics,
        }
        st.write("Canonical topics:", flat_topics)
elif article_text and not st.session_state.get("source_confirmed"):
    st.warning("Please confirm the source before running topic assignment.")

# --------------------------------------------------------------------------------------
# Stance classification + hybrid embedding
# --------------------------------------------------------------------------------------

st.header("Stage 5 — Stance Classification (GPT) + Hybrid Stance Embedding")

@st.cache_resource(show_spinner=False)
def load_stance_embedder(name: str):
    device = "cuda" if torch_available_cuda() else "cpu"
    return SentenceTransformer(name, device=device)

stance_info = {}
stance_meta = {}
stance_vec = None

if article_text and OPENAI_API_KEY and OpenAI is not None and topic_results and st.session_state.get("source_confirmed"):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        leaning_options = ", ".join(political_leanings.keys())
        stance_opts = ", ".join([k for cat in implied_stances.values() for k in cat.get("families", {}).keys()])
        prompt = f"""
You are a political analyst.
Based on the article below, classify its overall political leaning (tone) and implied stance.

Leaning options: {leaning_options}
Stance examples: {stance_opts}

Return strict JSON with fields:
- political_leaning (string)
- implied_stance (string)
- summary (one-sentence summary of the article's main argument)

Article title: {article_name}
Excerpt: {article_text[:2000]}
""".strip()
        resp = client.chat.completions.create(
            model=CONFIG["stance_processing"]["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=float(CONFIG["stance_processing"]["llm"]["temperature"]),
        )
        raw = resp.choices[0].message.content.strip()
        try:
            stance_info = json.loads(raw)
        except Exception:
            # best-effort fallback
            stance_info = {
                "political_leaning": "unknown",
                "implied_stance": "unknown",
                "summary": raw[:200]
            }
        # bias score
        def bias_to_score(label):
            l = (label or "").lower().strip()
            if "progressive" in l or ("left" in l and "center" not in l): return -0.8
            if "center left" in l:  return -0.4
            if l == "center":       return 0.0
            if "center right" in l: return 0.4
            if "conservative" in l or "right" in l: return 0.8
            if "libertarian" in l:  return 0.6
            return 0.0

        tone_score = bias_to_score(stance_info.get("political_leaning"))
        bias_family = bias_info.get("bias_family", "unknown")
        bias_score = float(bias_info.get("bias_score", 0.0))

        stance_meta = {
            "political_leaning": stance_info.get("political_leaning", "unknown"),
            "implied_stance": stance_info.get("implied_stance", "unknown"),
            "summary": stance_info.get("summary", ""),
            "bias_family": bias_family,
            "bias_score": bias_score,
            "tone_score": tone_score,
            "author_tone_match": abs(bias_score - tone_score) <= 0.3
        }
        st.subheader("GPT Classification")
        st.json(stance_meta)

        # Hybrid text (labels + summary) → stance embedding
        hybrid_text = "\n".join([
            stance_meta["political_leaning"],
            stance_meta["implied_stance"],
            stance_meta["summary"]
        ]).strip()
        stance_model_name = CONFIG["embeddings"]["stance_model"]
        stance_embedder = load_stance_embedder(stance_model_name)
        stance_vec = stance_embedder.encode(hybrid_text, normalize_embeddings=True).reshape(1, -1)
        st.write(f"Stance vector shape: {stance_vec.shape}")

    except Exception as e:
        st.error(f"Stance classification/embedding failed: {e}")

elif article_text and not OPENAI_API_KEY:
    st.info("Set OPENAI_API_KEY in Streamlit secrets to enable stance classification.")

# --------------------------------------------------------------------------------------
# Retrieval and Anti-Echo Analysis
# --------------------------------------------------------------------------------------

st.header("Stage 6 — Retrieval and Anti-Echo Analysis")

def run_retrieval_and_score():
    client_local = chroma_client()
    topic_coll = client_local.get_collection(CONFIG["chroma_collections"]["topic"])
    stance_coll = client_local.get_collection(CONFIG["chroma_collections"]["stance"])

    # Load full collections into memory
    t_docs = topic_coll.get(include=["embeddings", "metadatas"])
    s_docs = stance_coll.get(include=["embeddings", "metadatas"])

    topics_flat = topic_results.get("flat_topics", [])
    bias_score_article = float(stance_meta.get("bias_score", 0.0))
    tone_score_article = float(stance_meta.get("tone_score", 0.0))

    rows = []
    for emb, md in zip(t_docs["embeddings"], t_docs["metadatas"]):
        topic_overlap = topic_overlap_score(topics_flat, md.get("topics_flat", []))
        if topic_overlap < TOPIC_OVERLAP_THRESHOLD:
            continue

        # Find stance meta by id
        bias_db, tone_db = 0.0, 0.0
        stance_match_emb = None
        for s_emb, s_md in zip(s_docs["embeddings"], s_docs["metadatas"]):
            if s_md.get("id") == md.get("id"):
                try:
                    bias_db = float(s_md.get("bias_score", 0.0) or 0.0)
                except Exception:
                    bias_db = 0.0
                tone_db = float(s_md.get("tone_score", bias_db) or bias_db)
                stance_match_emb = np.array(s_emb).reshape(1, -1)
                break

        stance_sim = 0.0
        if stance_match_emb is not None and stance_vec is not None:
            stance_sim = cosine_similarity(stance_vec.reshape(1, -1), stance_match_emb)[0][0]

        bias_diff = abs(bias_score_article - bias_db)
        tone_diff = abs(tone_score_article - tone_db)

        anti_echo = (w_T * topic_overlap) - (w_S * stance_sim) - (w_B * bias_diff) - (w_Tone * tone_diff)

        rows.append({
            "article_id": md.get("id"),
            "source": md.get("source", ""),
            "title": md.get("title", ""),
            "url": md.get("url", ""),
            "bias_family": md.get("bias_family", ""),
            "bias_score": bias_db,
            "topic_overlap": topic_overlap,
            "stance_similarity": stance_sim,
            "bias_diff": bias_diff,
            "tone_diff": tone_diff,
            "anti_echo_score": anti_echo
        })

    df = pd.DataFrame(rows)
    return df

run_btn = st.button(
    "Run Retrieval & Score",
    disabled=not (article_text and topic_results and (stance_vec is not None) and st.session_state.get("source_confirmed"))
)
if run_btn:
    with st.spinner("Searching and scoring..."):
        df = run_retrieval_and_score()
    if df.empty:
        st.warning("No related articles found. Try lowering the topic overlap threshold.")
    else:
        # Views
        same_topic_diff_bias = df.sort_values(["topic_overlap", "bias_diff"], ascending=[False, False]).head(TOP_N_RESULTS)
        same_topic_opp_stance = df.sort_values(["topic_overlap", "stance_similarity"], ascending=[False, True]).head(TOP_N_RESULTS)
        anti_echo_best = df.sort_values("anti_echo_score", ascending=False).head(TOP_N_RESULTS)

        st.subheader("Top Anti-Echo Candidates")
        for _, row in anti_echo_best.iterrows():
            with st.container(border=True):
                title = row.get("title") or "Untitled"
                url = row.get("url") or ""
                link = f"[{title}]({url})" if url else title
                st.markdown(link)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Topic overlap", f"{row['topic_overlap']:.2f}")
                m2.metric("Stance sim", f"{row['stance_similarity']:.2f}")
                m3.metric("Bias diff", f"{row['bias_diff']:.2f}")
                m4.metric("Anti-echo", f"{row['anti_echo_score']:.3f}")
                st.caption(f"Source: {row.get('source','')} • Bias: {interpret_bias(row['bias_score'])}")

        a, b = st.columns(2)
        with a:
            st.markdown("#### Same Topic — Different Source Bias")
            st.dataframe(same_topic_diff_bias[["source","title","topic_overlap","bias_diff","url"]], use_container_width=True)
        with b:
            st.markdown("#### Same Topic — Opposite Stance")
            st.dataframe(same_topic_opp_stance[["source","title","topic_overlap","stance_similarity","url"]], use_container_width=True)

        # CSV export
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download full results CSV", data=csv_bytes, file_name="anti_echo_analysis.csv", mime="text/csv")


