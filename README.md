# anti echo chamber

Builds a free, open embedding index of news and commentary for studying topic framing and sentiment polarity across sources.  
This repository handles scraping, embedding, batching, and coordination for the Hugging Face dataset [`zanimal/anti-echo-artifacts`](https://huggingface.co/datasets/zanimal/anti-echo-artifacts).

---

<a target="_blank" href="https://colab.research.google.com/github/AHMerrill/anti-echo-chamber/blob/main/scraper_artifacts.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

_Run this project directly in Google Colab using the badge above._

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

```
anti-echo-chamber/
├── artifacts/
│   └── artifacts_registry.json      # record of published batches
├── config/
│   ├── config.yaml                  # global constants
│   ├── stance_axes                  # ideological axes catalog
│   └── topic_labels                 # topic taxonomy
├── feeds/
│   ├── index.json                   # local feed index used by scrapers
│   └── feeds_state.json             # collaborative feed cursors
├── docs/
│   ├── batch_manifest_schema.md     # formal schema for manifest.json
│   └── validation_checklist.md      # pre-publish checks
├── scraper_artifacts.ipynb          # initial scraping experiments
├── .gitignore
└── README.md
```

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

## Adding a new feed

1. Edit your scraper to include the new RSS URL.  
2. Run it once — the feed will be auto-added to `feeds/feeds_state.json`.  
3. Commit the updated file with a short message like “add reuters_world feed”.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed policies, ID conventions, and workflow requirements.

---

## License

- **Code:** MIT  
- **Embeddings and metadata:** Non-commercial use only, no redistribution of full text.

---

## Contact

Maintained by the **anti echo chamber** collaboration.  
Pull requests welcome.
