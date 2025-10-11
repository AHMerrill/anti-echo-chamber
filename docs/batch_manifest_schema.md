# Batch Manifest Schema

Each published batch on the Hugging Face dataset `zanimal/anti-echo-artifacts` must include a file named `manifest.json` that conforms to this schema. The manifest serves as the authoritative metadata for that batch and enables Chroma rebuilds, validation, and registry updates.

---

## Top-level structure

```json
{
  "version": 1,
  "batch_id": "20251011_154500Z_a1b2c3",
  "created_at": "2025-10-11T15:45:00Z",
  "builder": "anti-echo-chamber/colab",
  "counts": {
    "docs": 100,
    "topic_vectors": 100,
    "stance_vectors": 100
  },
  "models": {
    "topic": "sentence-transformers/all-MiniLM-L6-v2",
    "stance": "sentence-transformers/all-MiniLM-L6-v2",
    "summarizer": "facebook/bart-large-cnn",
    "dim": 384,
    "dtype": "float16",
    "pooling": "mean",
    "chunk_tokens": 512
  },
  "checksums": {
    "embeddings_topic.npz": "sha256:abcd1234...",
    "embeddings_stance.npz": "sha256:efgh5678...",
    "metadata.jsonl": "sha256:ijkl9012...",
    "manifest.json": "sha256:mnop3456..."
  },
  "hf_paths": {
    "embeddings_topic": "batches/20251011_154500Z_a1b2c3/embeddings_topic.npz",
    "embeddings_stance": "batches/20251011_154500Z_a1b2c3/embeddings_stance.npz",
    "metadata": "batches/20251011_154500Z_a1b2c3/metadata.jsonl",
    "manifest": "batches/20251011_154500Z_a1b2c3/manifest.json"
  },
  "id_policy": {
    "scheme": "domain-slug-sha12",
    "hash": "sha256",
    "normalized": true
  },
  "topic_labels_version": 1,
  "stance_axes_version": 1,
  "notes": "Optional free text about scraper runs or model versions."
}
