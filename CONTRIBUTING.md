Contributing to anti echo chamber

This project builds a public embedding dataset and a reproducible Chroma index for news. Collaborators must follow these rules so runs are deterministic and the public dataset remains clean.

Core policies

No full article text in any public location

Do not upload full text to the repo or to the HF dataset

Only embeddings, compact metadata, and manifests are published

Append only batches on HF

Never rewrite or delete past batches

Each run that has new items produces one new batch

Stateless Colab sessions

Every session rebuilds local Chroma from HF batches at start

Local caches can exist during a run but are not committed

One source of truth

HF dataset holds batches and serves rebuilds

artifacts_registry.json lists published batches and their URLs

Prerequisites

Python 3.10 or newer in Colab

HF account with write access to zanimal/anti-echo-artifacts

GitHub account with write access to this repo

Secrets

Set in Colab environment variables before running:

HF_TOKEN Hugging Face write token

GITHUB_TOKEN only if you plan to auto commit artifacts_registry.json

Never commit tokens.

Config files

config/config.yaml project constants

config/stance_axes stance axis catalog

config/topic_labels topic taxonomy

All code and notebooks must read constants from config/config.yaml rather than hardcoding.

IDs and hashing

id format: domain-slug-sha12

sha256 is computed on normalized main text

Both id and sha256 are required in metadata and manifests

Normalization rules in config:

lowercase

collapse whitespace

Feeds and scraping hygiene

Shared files:

feeds/index.json local cache compatible with scrapers

feeds/feeds_state.json collaborative cursors and a small dedupe ring buffer

Rules:

On start, load feeds_state.json. Add missing feeds automatically with defaults

Use last_cursor_iso to filter feed entries

Maintain recent_url_hashes as a FIFO list per feed to avoid duplicates

Update last_run_at and last_run_by after a successful run

Commit feeds_state.json changes with a short message

Models

Default encoders:

Topic embedding: sentence-transformers/all-MiniLM-L6-v2

Stance embedding: sentence-transformers/all-MiniLM-L6-v2

Stance summarizer: facebook/bart-large-cnn

Dimensions: 384
Dtype on disk: float16
Pooling: mean
Chunk size: 512 tokens for topic

Propose model changes via PR. Include a plan to preserve backward compatibility and note the change in the batch manifest.

Batch structure on HF

Every batch directory contains:

embeddings_topic.npz shape [N, 384], float16

embeddings_stance.npz shape [N, 384], float16

metadata.jsonl N lines, one JSON per article

manifest.json batch metadata and checksums

Required metadata fields per line:

id, url, title, source, section, domain

published (ISO 8601 if known)

sha256, chars

batch_id

Optional metadata fields:

topic_labels (array of ids)

stance_axes_scores (map of axis id to scalar)

Batch id

Use a deterministic string:

YYYYMMDD_hhmmssZ_shortsha or similar

Example: 20251011_154500Z_a1b2c3

Run workflow in Colab

Bootstrap from HF

Read artifacts/artifacts_registry.json

Download all listed batches from HF

Rebuild local Chroma collections news_topic and news_stance

Scrape

Run RSS scrapers

Produce session local raw text and meta

Skip any item whose id already exists in HF metadata

Embed

Topic vectors over full text (chunk then mean pool)

Stance vectors from a summarizer output over the article argument

Keep an in run cache only

Update local Chroma

Insert only the new ids so demos work immediately

Package batch

Write the four batch files into batches/<batch_id>/

Compute sha256 checksums for each file and record them in manifest.json

Record encoder names, dims, dtype, pooling, chunking in manifest.json

Publish to HF

Upload the batch directory to zanimal/anti-echo-artifacts under batches/<batch_id>/

Update registry

Append a record to artifacts/artifacts_registry.json with batch_id, counts, checksums, and the four HF URLs

Commit to GitHub

Validation checklist before upload

Embedding files exist and load

Shapes are [N, 384] for both spaces

Dtype is float16 in both npz files

metadata.jsonl has N lines and every line has required keys

The set of ids matches across topic, stance, and metadata

manifest.json contains batch_id, created_at, model names, dims, dtype, pooling, chunking, counts, and file checksums

Opposite stance retrieval policy

Query topic space first for K candidate neighbors

Re rank using stance space by comparing to the negated query stance vector

Combine scores and return results with both distances for transparency

Pull requests

Keep changes focused and documented

If you add feeds, update feeds/feeds_state.json and add a short note in the PR

If you change schemas, bump version fields and update docs accordingly

Run the validation checklist for any new batch before proposing to add it to the registry

License note

Code in this repo can be MIT

The HF dataset artifacts are embeddings and metadata only and must not include full text
