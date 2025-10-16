# anti echo chamber

Builds a free, open embedding index of news and commentary for studying topic framing and sentiment polarity across sources.  
This repository handles scraping, embedding, batching, and coordination for the Hugging Face dataset [`zanimal/anti-echo-artifacts`](https://huggingface.co/datasets/zanimal/anti-echo-artifacts).

---

## Colab notebooks

### 1. Scraper and batch builder

<a target="_blank" href="https://colab.research.google.com/github/AHMerrill/anti-echo-chamber/blob/main/scraper_artifacts.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

_Run this notebook to scrape, embed, and publish new batches to the Hugging Face dataset._

---

### 2. Analysis and stance comparison

<a target="_blank" href="https://colab.research.google.com/github/AHMerrill/anti-echo-chamber/blob/main/anti_echo_chamber.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

_Run this notebook to rebuild the Chroma index from Hugging Face, upload an article, and find similar topics with opposing viewpoints._

---

## Overview

This project collects news articles and opinion pieces (publicly accessible, full text kept local only), creates two kinds of vector spaces, and publishes compact embedding batches:

1. **Topic space** – what the article is about  
   384-dim vectors from `sentence-transformers/all-MiniLM-L6-v2`

2. **Stance space** – how the article argues or frames an issue  
   Article is summarized with `facebook/bart-large-cnn`, then embedded with the same MiniLM model

Each batch of embeddings and metadata is uploaded to Hugging Face and can be used to rebuild a complete Chroma index for dense retrieval or RAG.

---

## Repository structure


---

## Core datasets and models

| Component | Model | Dim | Dtype | Notes |
|------------|--------|------|-------|-------|
| Topic embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384 | float16 | mean pooled chunks of 512 tokens |
| Stance embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384 | float16 | mean pooled summary |
| Stance summarizer | `facebook/bart-large-cnn` | — | — | summarization before stance embedding |

All values are defined centrally in `config/config.yaml`.

---

## Workflow

Every Colab run is stateless. Each collaborator follows this sequence:

1. **Bootstrap from HF**  
   - Read `artifacts/artifacts_registry.json`  
   - Download all listed batches from Hugging Face  
   - Rebuild local Chroma collections `news_topic` and `news_stance`

2. **Scrape new articles**  
   - Run RSS scrapers (Guardian, etc.)  
   - Write raw text and meta locally  
   - Skip URLs already in the HF metadata  
   - Update `feeds/feeds_state.json` with new cursors and hashes

3. **Embed**  
   - Compute topic and stance embeddings  
   - Use summarizer for stance vectors  
   - Keep results only in-memory for this session

4. **Update local Chroma**  
   - Insert new ids so you can query during the session

5. **Package new batch**  
   - Write the four standard files under `batches/<batch_id>/`  
   - Generate `manifest.json` matching the [schema](docs/batch_manifest_schema.md)

6. **Validate**  
   - Run through [validation checklist](docs/validation_checklist.md)

7. **Publish**  
   - Upload to HF dataset under `batches/<batch_id>/`  
   - Append batch record to `artifacts/artifacts_registry.json`  
   - Commit and push

8. **Verify**  
   - Rebuild Chroma from HF and confirm document counts match

---

## File contracts

- **Registry:** `artifacts/artifacts_registry.json`  
  Holds version, model info, and list of batch entries with HF URLs and checksums.

- **Manifest:** `manifest.json` (inside each batch)  
  Described in [docs/batch_manifest_schema.md](docs/batch_manifest_schema.md).

- **Feeds:** `feeds/feeds_state.json`  
  Tracks last run cursors and recent URL hashes for each feed.

- **Config:** `config/config.yaml`  
  Central configuration for models, datasets, and directory layout.

---

## What the notebooks do (end-to-end)

This repo has two Colab-first notebooks that implement the full workflow:

1) `scraper_artifacts.ipynb` — Scrape feeds, normalize into a DataFrame, embed, and publish a batch to Hugging Face.
2) `anti_echo_chamber.ipynb` — Rebuild a local Chroma index from the HF dataset, then run retrieval for opposite-stance analysis.

High-level flow:
- Read feeds and/or a provided DataFrame of articles
- Normalize and de-duplicate by URL hash
- Compute two embeddings per article:
  - Topic: what it’s about (SentenceTransformer model)
  - Stance: how it argues (hybrid: LLM classification + embed the summary/labels)
- Write artifacts under `batches/<batch_id>/`:
  - `embeddings_topic.npz`, `embeddings_stance.npz`
  - `metadata_topic.jsonl`, `metadata_stance.jsonl`
  - `manifest.json`
- Upload to `zanimal/anti-echo-artifacts` and append batch to `artifacts/artifacts_registry.json`

All knobs (models, dimensions, dtype, pooling, chunk size, stance mode, etc.) live in `config/config.yaml`.

---

## Collaborator guide: Selenium scraping DataFrame schema

If you are using Selenium (or any custom scraper) to gather additional articles to feed into the `scraper_artifacts.ipynb` pipeline, prepare a pandas DataFrame with the following columns. Keep names and types exactly as specified so the notebook can validate, embed, and package without manual tweaks.

Required columns:

- `url` (string): Canonical article URL. Must be unique per article.
- `domain` (string): Lowercased site key, e.g., `guardian`, `reuters`, `bbc`, `npr`, `dailycaller`, etc. Use keys that exist in `config/source_bias.json` where possible.
- `title` (string): Article headline.
- `text` (string): Main body text (cleaned; no nav/boilerplate if possible). Keep full text local; it is not uploaded.
- `published_at` (string or datetime): Publication timestamp in ISO 8601 if available (e.g., `2025-10-15T19:45:00Z`).

Recommended columns (help downstream stance/topic processing and QC):

- `author` (string | null): Primary author name.
- `section` (string | null): Section/category (e.g., `politics`, `world`, `opinion`).
- `language` (string | null): ISO language code, e.g., `en`.
- `description` (string | null): Deck/standfirst if present.

Optional columns (used if present; otherwise derived):

- `source_bias_score` (float in [-1.0, +1.0] | null): If you know it; otherwise the notebook can look up by `domain` via `config/source_bias.json`.
- `political_leaning_hint` (string | null): Free-text hint if you have one; final stance classification may override.

Validation and normalization rules:

- URLs are de-duplicated via a hash scheme configured under `ids` in `config/config.yaml` (e.g., `domain-slug-sha12`).
- Whitespace is normalized and values are lowercased where configured (`ids.normalize_whitespace`, `ids.lowercase`).
- The stance pipeline may consult:
  - `config/political_leanings.json` (ideological families, markers)
  - `config/implied_stances.json` (issue families and example claims)
  - `config/source_bias.json` (per-domain bias priors)
- Topic labeling uses `config/topics.json` with a similarity threshold (`topics.similarity_threshold`) and a cap (`topics.max_topics_per_article`).

Minimal example in code (inside your scraper before saving to CSV/Parquet):

```python
import pandas as pd

data = [
  {
    "url": "https://www.theguardian.com/world/2025/oct/16/example",
    "domain": "guardian",
    "title": "Example headline",
    "text": "Full cleaned article text ...",
    "published_at": "2025-10-16T00:11:46Z",
    "author": "Firstname Lastname",
    "section": "world",
    "language": "en"
  }
]

df = pd.DataFrame(data)
```

Pass `df` into `scraper_artifacts.ipynb` where prompted (or export to a file the notebook reads). The notebook will:
- Drop duplicates by URL
- Compute topic and stance embeddings per `config/config.yaml`
- Generate and validate `manifest.json`
- Write `{embeddings_topic, embeddings_stance}.npz` and `{metadata_topic, metadata_stance}.jsonl`
- Upload to `zanimal/anti-echo-artifacts` and update `artifacts/artifacts_registry.json`

Notes:
- Do not include full article text in anything that is uploaded. The pipeline keeps raw text local; only derived embeddings and metadata are published.
- Use `domain` keys that exist in `config/source_bias.json` to inherit consistent bias priors and display names.

---

## Adding a new feed

1. Edit your scraper to include the new RSS URL.  
2. Run it once — the feed will be auto-added to `feeds/feeds_state.json`.  
3. Commit the updated file with a short message such as “add reuters_world feed”.

---

## Opposite-stance retrieval

The retrieval demo and future UI use the two embedding spaces to surface differing perspectives.

1. Compute topic vector for uploaded article.  
2. Query `news_topic` to get same-topic candidates.  
3. Compute stance vector for uploaded article.  
4. Query `news_stance` with the **negated stance vector** to find opposite framing.  
5. Combine topic and stance scores for ranking.

---

## Validation and QA

Follow [docs/validation_checklist.md](docs/validation_checklist.md) before publishing.

---

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed policies, ID conventions, and workflow requirements.

---

## License

- **Code:** MIT  
- **Embeddings and metadata:** Non-commercial use only, no redistribution of full text.

---

## Contact

Maintained by the **anti echo chamber** collaboration.  
Pull requests welcome.
