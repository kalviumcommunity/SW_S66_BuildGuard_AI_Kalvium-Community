# BuildGuard AI

This repository is currently processing the National Building Code of India 2016 PDFs. The first run handles **volume 1 only** and saves chunks for review. Embeddings and Chroma storage are a later step.

## Setup

Install the Python dependencies:

```bash
uv sync
```

Tesseract OCR must be installed for pages whose PDF text uses unreadable custom fonts. The chunking command looks for the Windows installation at `C:\Program Files\Tesseract-OCR`.

## Preview a small sample

From the repository root:

```bash
uv run python -m src.text.chunk_nbc_text --preview --limit-pages 10 --sample 5
```

This reads the first 10 physical pages of volume 1 and prints sample chunks without writing chunk files.

## Process volume 1

```bash
uv run python -m src.text.chunk_nbc_text
```

The command defaults to `data/nationalbuilding01.pdf`. It extracts page text, uses OCR where native text is unreadable, then creates semantic text chunks. Cleanup and validation are in the separate `src.text.clean_validate_text_chunks` module.

The saved files are:

- `outputs/chunks/text_chunks.jsonl`

Chunk records include source PDF, volume, page range, clause/heading when detected, token count, and whether native extraction or OCR supplied the text. The initial chunk limit is 500 tokens; 50-token overlap is used only to split a single oversized sentence.

To select another single PDF explicitly, use `--pdf path/to/file.pdf`. Volume 2 will be processed separately after volume 1 has been reviewed.

## Clean and validate generated text chunks

After chunking volume 1, run the conservative cleanup and validation pass:

```bash
uv run python -m src.text.clean_validate_text_chunks
```

It reads `outputs/chunks/text_chunks.jsonl` and preserves it unchanged. It normalizes Unicode and whitespace, removes stray control characters, recalculates token counts, and checks the record schema, unique IDs, source/page metadata, and 500-token limit. If a cleaned record exceeds the limit, it splits it at word boundaries and records the original chunk ID in the derived records' metadata. It reports short chunks, replacement characters, token-count recalculations, and heading values that span many pages for review; it does not guess at OCR corrections or merge chunks.

The cleaned dataset and diagnostics are written separately:

- `outputs/cleaned_chunks/text_chunks.cleaned.jsonl`
- `outputs/cleaned_chunks/cleaning_validation.log`
- `outputs/cleaned_chunks/validation_report.json`

Use `--input`, `--output-dir`, or `--max-tokens` to change the defaults. If hard validation errors occur, the script records them and does not publish a partial cleaned JSONL.

## Review OCR text and rebuild headings

Prepare a heading-metadata rebuild and human review queues with:

```bash
uv run python -m src.text.review_text_chunks
```

This reads the cleaned JSONL and writes to `outputs/chunk_review/`. It assigns volume 1 part/section headings from the physical page ranges in the contents, leaves chunks that cross those boundaries unset for review, and flags OCR chunks using visible text-quality heuristics plus a page-stratified sample. It does not alter OCR wording. Review `ocr_review_queue.jsonl` against the source PDF; enter verified text in `verified_text` and set `review_status` to `verified` in your reviewed copy. Review ambiguous page ranges in `heading_review_queue.jsonl` before relying on the rebuilt heading labels.

The main rebuilt dataset is `outputs/chunk_review/text_chunks.headings_rebuilt.jsonl`; review counts and run details are in `review_report.json` and `chunk_review.log`.

## Scope of this step

The chunking command uses temporary sentence vectors to find semantic boundaries, then discards them. It does not save embeddings, write to Chroma, or call a chat API. The generated chunk data is local and ignored by Git.

## Source code categories

Processing and application modules are grouped by purpose:

- `src/text/` — text extraction, cleanup, validation, and review.
- `src/images/` — image extraction, grouping, cleanup, and validation.
- `src/embeddings/` — text vector indexing and similarity ranking.
- `src/assistant/` — staff assistant, prompt handling, and chat support.
- `src/utils/` — shared utilities such as token counting.

## Embed volume 1 text chunks

After the heading review files have been created, build a local vector index:

```bash
uv run python -m src.embeddings.embed_text_chunks
```

By default this reads `outputs/chunk_review/text_chunks.headings_rebuilt.jsonl`, embeds each chunk with `sentence-transformers/multi-qa-MiniLM-L6-dot-v1`, and upserts the vectors and scalar metadata into the Chroma collection `nbc_v1_multi_qa_minilm_l6_dot_v1` under `outputs/vector_db/`. Embeddings are generated locally; the first run may download the model files. Re-running with the same input updates the same chunk IDs instead of creating duplicates. It does not change any chunk JSONL file.

The run log and summary are saved in `outputs/embeddings/embedding.log` and `outputs/embeddings/embedding_report.json`. Use `--limit 20` for a small initial indexing run, `--batch-size` to tune memory use, or `--input` to select another JSONL. For example, use `--input outputs/cleaned_chunks/text_chunks.cleaned.jsonl` if you want to index the cleaned chunks before heading review. Query the finished collection with `uv run python -m src.embeddings.embed_text_chunks --query "fire safety requirements for exits" --top-k 5`.

## Compare query and chunk embeddings

To calculate cosine similarity for a query against an evenly spaced sample of chunks, then print and save both the most and least similar results:

```bash
uv run python -m src.embeddings.rank_text_chunks
```

This command reads stored chunk vectors from the Chroma collection, embeds only the query with the collection's model, and calculates cosine similarity against every indexed chunk. It prints the highest and lowest matches and writes their scores, source text, and metadata to `examples/embedding_similarity/sample_rankings.json`. Cosine similarity compares vector direction; with normalized vectors, the dot product is the cosine score. Higher scores mean greater semantic similarity. Run text indexing first; rankings cover the records currently in the collection. Use `--sample-size 256` to compare an evenly spaced subset instead.

The OCR review queue is still pending, so these vectors represent the current OCR text. After you make verified text corrections, rebuild the review output and rerun this command to upsert the updated chunk records.

## Extract, validate, and group images

Run the image pipeline separately from text chunking:

```bash
uv run python -m src.images.chunk_images
```

This extracts and groups images by content hash. Each image record keeps every page, object ID (xref), and placement rectangle. Then run the separate image cleanup and validation step:

```bash
uv run python -m src.images.clean_validate_images
```

Outputs are stored under `outputs/images/`:

- `assets/` contains the extracted files, grouped by content hash.
- `image_chunks.jsonl` contains one record per unique image, including its status and all PDF occurrences.
- `image_pipeline_report.json`, `image_gallery.html`, and `image_pipeline.log` are produced by validation and provide summary, review, and diagnostics.
- `image_extraction.log` records the PDF extraction and grouping pass.
- `high_res_regions/` contains 300-DPI crops for low-resolution placements. `high_res_pages/` contains full-page previews with boxes marking those placements; click a gallery image to open it at full size.

An image with either side under 16 px is marked tiny. Images with a shorter side under 64 px or effective placed resolution at or below 150 DPI are marked low resolution. Exact images used at least 8 times and objects with no visible page placement are excluded as PDF components. The gallery shows one low-resolution preview per PDF page, instead of repeating the same page for each image object. Previews render the source PDF at 300 DPI; they cannot restore detail missing from a blurry raster embedded in the PDF. All classifications are included in the single `image_chunks.jsonl` file.

Run both image commands again after changing the source PDF or extraction logic. Validation reads and updates `image_chunks.jsonl` after checking it; the rewrite uses a temporary file so a failed validation does not replace the extracted data.
