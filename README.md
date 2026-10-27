# BuildGuard AI

This repository processes National Building Code of India 2016 PDFs into searchable text and image chunks, then stores their embeddings in a local Chroma database.

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

## Process, clean, and validate text for volume 1

```bash
uv run python -m src.text.chunk_nbc_text
```

The command defaults to `data/nationalbuilding01.pdf`. It extracts page text, uses OCR where native text is unreadable, creates semantic text chunks, normalizes Unicode and whitespace, removes stray control characters, recalculates token counts, validates IDs and required metadata, filters invalid or empty chunks, and splits oversized chunks at word boundaries. It reports short chunks and replacement characters for later review. It does not guess OCR corrections or heading labels.

The saved files are:

- `outputs/chunks/text_chunks_volume_1.jsonl`
- `outputs/chunks/text_pipeline_report_volume_1.json`

Chunk records include source PDF, volume, page range, clause/heading when detected, token count, and whether native extraction or OCR supplied the text. The initial chunk limit is 500 tokens; 50-token overlap is used only to split a single oversized sentence.

To select another PDF explicitly, use `--pdf path/to/file.pdf`. For example, Volume 2 writes `outputs/chunks/text_chunks_volume_2.jsonl` and `outputs/chunks/text_pipeline_report_volume_2.json`.

## Scope of this step

The chunking command uses temporary sentence vectors to find semantic boundaries, then discards them. It does not save embeddings, write to Chroma, or call a chat API. The generated chunk data is local and ignored by Git.

## Source code categories

Processing and application modules are grouped by purpose:

- `src/text/` — text extraction, chunking, cleanup, and validation.
- `src/images/` — image extraction, grouping, cleanup, and validation.
- `src/embeddings/` — text vector indexing and similarity ranking.
- `src/assistant/` — staff assistant, prompt handling, and chat support.
- `src/api/` — FastAPI backend service exposing the RAG pipeline over HTTP.
- `src/utils/` — shared utilities such as token counting.

## Embed volume 1 text chunks

After the text pipeline creates the final chunk file, build the local vector index:

```bash
uv run python -m src.embeddings.embed_text_chunks
```

By default this reads `outputs/chunks/text_chunks_volume_1.jsonl`, embeds chunks in batches of 64 with `sentence-transformers/multi-qa-MiniLM-L6-dot-v1`, and saves vectors and scalar metadata in the Chroma collection `nbc_v1_multi_qa_minilm_l6_dot_v1` under `outputs/vector_db/`. Set `--batch-size` to change the batch size. Embeddings are generated locally; the first run may download the model files. On reruns, chunks with the same ID, text, and stored vector are skipped; changed text is re-embedded. The source JSONL is not changed.

The run log and summary are saved in `outputs/embeddings/embedding.log` and `outputs/embeddings/embedding_report.json`. The summary includes chunks read, embeddings generated, unchanged chunks skipped, failed chunks/batches, retry count, estimated input tokens, and estimated cost. Transient failures are retried up to 3 times with exponential backoff; use `--max-retries` and `--retry-base-seconds` to adjust this. Since the current model runs locally, its actual per-token API cost is `$0`; `--cost-per-million-tokens-usd` accepts a rate you supply to estimate a comparable hosted service. Use `--limit 20` for a small initial indexing run or `--input` to select another final chunk JSONL. Query the finished collection with `uv run python -m src.embeddings.embed_text_chunks --query "fire safety requirements for exits" --top-k 5`.

## Compare query and chunk embeddings

To calculate cosine similarity for a query against an evenly spaced sample of chunks, then print and save both the most and least similar results:

```bash
uv run python -m src.embeddings.rank_text_chunks
```

This command reads stored chunk vectors from the Chroma collection, embeds only the query with the collection's model, and calculates cosine similarity against every indexed chunk. It prints the highest and lowest matches and writes their scores, source text, and metadata to `examples/embedding_similarity/sample_rankings.json`. Cosine similarity compares vector direction; with normalized vectors, the dot product is the cosine score. Higher scores mean greater semantic similarity. Run text indexing first; rankings cover the records currently in the collection. Use `--sample-size 256` to compare an evenly spaced subset instead.

The vectors represent the extracted and cleaned text. After correcting OCR text in the source pipeline, rerun the text pipeline and embedding command to upsert the changed chunk records.

## Evaluate text embedding quality

Run a read-only evaluation against the stored text collection:

```bash
uv run python -m src.embeddings.evaluate_text_embeddings --query "fire safety requirements for exits"
```

It ranks the most and least similar chunks with cosine similarity, reports cosine distance (`1 - similarity`), and writes `outputs/embeddings/text_embedding_evaluation.json`. It also checks vector dimensions, finite values, zero vectors, normalization, required source metadata, duplicate vectors, duplicate text, and a deterministic sample of pairwise similarity and distance scores. The script does not add, update, or delete embeddings.

## Search text and images together

Search both volumes across the stored text and image collections with one question:

```bash
uv run python -m src.embeddings.search_nbc "fire safety requirements for exits"
```

The command searches every Volume 1 and Volume 2 record in each collection, then writes `outputs/search/latest_nbc_search.json`. Text and images use different embedding models, so their vectors remain in separate collections; the result report combines their ranks and marks each result as `text` or `image`.

## Clean, validate, and chunk images

Run the image pipeline separately from text chunking:

```bash
uv run python -m src.images.clean_validate_images --pdf data/nationalbuilding01.pdf
```

This scans the source PDF, groups exact duplicates, validates image objects, rejects tiny or unreadable fragments, checks low-resolution page crops in memory, and writes the final embedding-ready image chunks.

```bash
uv run python -m src.embeddings.embed_image_chunks --input outputs/images/image_chunks_volume_1.jsonl --volume volume_1
```

Outputs are stored under `outputs/images/`:

- `image_chunks_volume_1.jsonl` contains accepted, embedding-ready image chunks.
- `image_pipeline_report_volume_1.json`, `image_gallery_volume_1.html`, and `image_pipeline.log` provide summary, review, and diagnostics.
- `volume_1/` contains only the retained high-resolution PDF pages for accepted visuals. Individual crop files are never saved.

An image with either side under 16 px is marked tiny. Images with a shorter side under 64 px or effective placed resolution at or below 150 DPI are marked low resolution. Exact images used at least 8 times and objects with no visible page placement are excluded as PDF components. The gallery shows one low-resolution preview per PDF page, instead of repeating the same page for each image object. Previews render the source PDF at 300 DPI; they cannot restore detail missing from a blurry raster embedded in the PDF. The accepted image chunks are in the volume-specific `image_chunks_volume_N.jsonl` file.

Run both image commands again after changing the source PDF or extraction logic. Validation reads and updates `image_chunks.jsonl` after checking it; the rewrite uses a temporary file so a failed validation does not replace the extracted data.

## RAG backend API

The end-to-end RAG pipeline is exposed as a FastAPI service so a frontend or another backend can request a grounded answer over HTTP. Copy `.env.example` to `.env` and set the chat credentials, then start the server (host/port come from `API_HOST`/`API_PORT`):

```bash
uv run uvicorn src.api.app:app --reload
```

Send a question to `POST /query` and receive structured JSON with `answer`, `sources`, `status`, and `metadata`:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the fire resistance rating for boiler room walls?", "dry_run": true}'
```

`dry_run: true` runs real retrieval, re-ranking, and context assembly but skips the LLM call, so no API key is required. The endpoint returns `422` for missing/invalid input, `503` when a live request is made without chat credentials configured, and `500` for unexpected pipeline errors. See `docs/BACKEND_API.md` for the full contract, and `outputs/rag_api/sample_request.json` / `sample_response.json` for a worked example. Regenerate the sample with `uv run python -m src.api.generate_sample`.

### Grow the knowledge base at runtime

`POST /documents/upload` accepts a document (`multipart/form-data`, field `file`), then ingests, cleans, chunks, embeds, and indexes it into the same Chroma collection `/query` reads — so new content is searchable immediately, without restarting the app.

```bash
curl -X POST http://127.0.0.1:8000/documents/upload -F "file=@./my_addendum.txt"
```

Uploads are size- and type-checked (`MAX_UPLOAD_BYTES`, `ALLOWED_EXTENSIONS` from the environment) and stored under `UPLOAD_DIR` with a content-hash-prefixed name. The endpoint returns `201` with an indexing summary, or `400`/`413`/`415`/`422`/`500` for empty, oversized, unsupported, malformed, or failed uploads. See `docs/DOCUMENT_UPLOAD.md`, and regenerate the upload-and-query sample with `uv run python -m src.api.generate_upload_sample`.
