# Validation Checklist for New Batches

Use this checklist before uploading any new batch to the Hugging Face dataset or updating the registry.

---

## 1) Directory and file presence

- [ ] Batch folder is under `batches/<batch_id>/`
- [ ] Contains exactly four files:
  - `embeddings_topic.npz`
  - `embeddings_stance.npz`
  - `metadata.jsonl`
  - `manifest.json`
- [ ] No full article text or raw content files present

---

## 2) Embedding arrays

- [ ] Both `.npz` files load correctly with `numpy.load()`
- [ ] Each has shape `[N, 384]`
- [ ] Dtype is `float16`
- [ ] Both arrays have the same number of rows `N`
- [ ] No NaN or inf values
- [ ] Mean and variance look reasonable (not all zeros or constants)

---

## 3) Metadata file

- [ ] File is valid UTF-8 and has `N` lines
- [ ] Each line parses as valid JSON
- [ ] Every line includes:
  - `id`
  - `url`
  - `title`
  - `source`
  - `section`
  - `domain`
  - `published`
  - `sha256`
  - `chars`
  - `batch_id`
- [ ] No duplicate `id` values
- [ ] Character counts (`chars`) roughly match text lengths used in embeddings
- [ ] Optional fields (`topic_labels`, `stance_axes_scores`) are JSON-compatible if present

---

## 4) ID alignment

- [ ] The set of `id` values in `metadata.jsonl` exactly matches the order of rows in both `.npz` files
- [ ] No mismatch or missing ids between topic, stance, and metadata

---

## 5) Manifest integrity

- [ ] `manifest.json` matches the [schema](batch_manifest_schema.md)
- [ ] `batch_id` and `created_at` are correct
- [ ] Counts in `manifest.json` match `N`
- [ ] Model names, dims, dtype, and pooling match the config file
- [ ] All checksums are 64-character lowercase hex strings prefixed with `"sha256:"`
- [ ] Checksums match actual file contents
- [ ] Manifest contains valid `hf_paths` entries for each artifact

---

## 6) Registry update preparation

- [ ] `artifacts/artifacts_registry.json` has the correct top-level models info
- [ ] New batch record includes:
  - `batch_id`
  - `counts`
  - `checksums`
  - HF URLs for the four files
  - `created_at`
- [ ] Registry file still parses as valid JSON after update

---

## 7) Optional QA

- [ ] Rebuild a temporary Chroma index from this single batch and confirm `collection.count()` equals `N`
- [ ] Run a test query to verify retrieval works and distances are sensible
- [ ] Verify that stance and topic embeddings are roughly uncorrelated (basic sanity check)

---

## 8) Commit and publish

- [ ] Push batch to HF dataset under `batches/<batch_id>/`
-
