# NBC 2016 PDF Processing and Embeddings — LLD

## Goal

Start with the first local PDF in data:

- nationalbuilding01.pdf
- nationalbuilding02.pdf

The first deliverable is a saved chunk dataset for volume 1 under outputs/. It contains semantic text chunks and image records with metadata. Vectorization is a later step. This phase does not save vectors or create a Chroma database. Process volume 2 after reviewing volume 1.

## Process overview

1. Process nationalbuilding01.pdf page by page.
2. Extract embedded images or render relevant pages, retaining their source volume and page.
3. Check text extraction quality; use Tesseract OCR where custom PDF fonts return unreadable text or where a page has no usable native text.
4. Clean the extracted text conservatively.
5. Split text into semantically coherent chunks and create image records.
6. Save text chunks and metadata as JSONL, and image files plus image metadata as JSONL, under outputs/chunks/.
7. Review the saved chunks. In a later step, read these files, create embeddings, and store the vectors in Chroma.
8. Print progress and final counts in the terminal.

## 1. PDF text and image extraction

### Tool

Use PyMuPDF for page-level text extraction, image extraction, OCR fallback, and page rendering. Use Tesseract on pages whose custom PDF font maps produce unreadable text. PyMuPDF can also render vector diagrams. [PyMuPDF page API](https://pymupdf.readthedocs.io/en/latest/page.html) [PyMuPDF image and OCR recipes](https://pymupdf.readthedocs.io/en/latest/the-basics.html)

### Text extraction

For every page in the selected volume:

- extract the page's text;
- retain the original PDF filename, volume ID, and one-based physical page number;
- count extracted characters and flag empty or unusually short pages;
- preserve page boundaries for later citations and debugging.

Every text span retains its source PDF, volume number, and page.

### Image extraction

For every image block exposed on a page:

- extract the image bytes in memory when it is a raster image;
- identify its source PDF, physical page, and image index on that page;
- calculate a content hash to avoid embedding duplicate copies of the same image;
- retain image dimensions and format when available.

Save each embedded raster image under outputs/chunks/images/. For pages with at least 18 vector drawing items, also save an RGB page render to capture diagrams. Write one metadata record per image occurrence to outputs/chunks/image_chunks.jsonl, including relative path, source volume, page, image index or page-render marker, dimensions, and content hash. Do not embed images during this first chunking step. The original PDF remains the source of truth.

### Scanned pages and OCR

After extraction, print a short quality summary: page count, pages using native text or OCR, pages with no text, extracted image count, and failures. Review representative pages from volume 1.

Some pages have selectable text that is garbled because of custom font maps. OCR is therefore selected when native text fails a readability check, not only when it is empty. OCR text retains its original PDF page number. Image embeddings do not replace OCR when the goal is to capture exact clause wording.

## 2. Text cleanup

Clean extracted text conservatively in the NBC processing module, retaining page identity.

Cleanup currently normalizes whitespace while preserving clause numbers, cross-references, units, symbols, paragraph order, table row labels, and page identity. Repeated headers and footers are retained unless a later review confirms they should be removed.

## 3. Semantic text chunking

### Method

Use semantic chunking rather than fixed-size character windows.

1. Break each page's cleaned text into paragraphs and sentences. Keep clause labels attached to their text.
2. Use the local sentence-transformer to create temporary sentence embeddings for boundary detection.
3. For every adjacent pair, calculate cosine similarity, then convert it to cosine distance: distance = 1 - similarity. A high distance means adjacent text is less related.
4. Collect distances from volume 1. Start with the configurable 85th percentile and inspect the boundaries. Recheck the setting against volume 2 before processing it.
5. Split at distances above the selected percentile, except where a clear clause or heading boundary gives a better split. Never split between a clause number and its clause text.
6. Keep a detected clause or heading attached to its text; a new clause or heading starts a new chunk where detected.
7. Enforce a 500-token maximum. If a single sentence is too large, split it into token windows with 50-token overlap.
8. Keep page and section metadata attached as each piece is split or merged. A chunk spanning pages stores both its first and last physical page.

The initial maximum is 500 tokens. The 50-token overlap applies only when a single oversized sentence must be split. Ordinary adjacent semantic chunks do not overlap.

Use the already installed sentence-transformer stack to calculate semantic boundaries. Sentence Transformers provides cosine similarity between embeddings; converting it to distance makes the threshold rule straightforward. [Sentence Transformers similarity documentation](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)

### How chunking will be checked

Run a bounded preview first to confirm extraction and chunk boundaries. Review representative volume 1 chunks with their volume, page range, clause/heading, token count, and text. Then run the command without preview to save JSONL under outputs/chunks/.

Review about 20 to 30 chunks across different content types in volume 1: paragraphs, long clauses, tables, headings, and page transitions. Check that clauses stay together, chunks do not mix unrelated provisions, headings remain with their content, and source/page metadata is correct. Compare candidate percentiles:

- If a chunk contains unrelated topics, lower the percentile so more boundaries are introduced.
- If a single idea is fragmented into many tiny chunks, raise the percentile or merge neighboring chunks in the same clause.
- If chunks exceed 500 tokens, split at the nearest paragraph/sentence boundary and retain 50 tokens of overlap only for the split clause.
- If any boundary separates a clause number from its text, fix the structural rule before embedding.

Select a working cutoff after reviewing volume 1. Verify it against volume 2 before settling on a corpus-wide setting. Each chunk stores the percentile and size parameters used for that run.

### Why preserve structure too

Semantic similarity alone can group nearby sentences that belong to different numbered provisions. Heading and clause boundaries therefore take precedence when they are clear. If heading detection is uncertain, leave the structural field empty and retain the text instead of guessing.

## 4. Embeddings

### Text embeddings

Use the installed Sentence Transformers package to create temporary sentence vectors for boundary detection. The default model is sentence-transformers/multi-qa-MiniLM-L6-dot-v1. These temporary vectors are discarded after chunking.

This does not happen in the current chunk-generation step. Later, read each text record from outputs/chunks/text_chunks.jsonl and create one vector per chunk in batches. Check that the number of vectors matches the number of chunks and that every vector has the same nonzero dimension.

### Image embeddings

This does not happen in the current chunk-generation step. Later, read image paths from outputs/chunks/image_chunks.jsonl and use the Sentence Transformers CLIP model sentence-transformers/clip-ViT-B-32 to create image vectors. The model supports image inputs and maps image and text inputs into one shared vector space. It requires image support in the installed stack; confirm the needed image dependency is installed before vectorization. [Sentence Transformers image search documentation](https://sbert.net/examples/sentence_transformer/applications/image-search/README.html)

Keep text and image vectors in separate Chroma collections so their dimensions and embedding functions do not get mixed:

- nbc_text_embeddings
- nbc_image_embeddings

For the later image collection, store a short source reference as its document and metadata containing volume, page, image index or page-render marker, dimensions, and content hash. The image_chunks.jsonl file is the first-stage record and points to the saved image file.

An image embedding supports visual similarity search later. It does not extract exact text from diagrams or tables; OCR is needed for text contained in images.

## 5. Local storage

The first-stage chunking command writes these files:

- outputs/chunks/text_chunks.jsonl
- outputs/chunks/image_chunks.jsonl
- outputs/chunks/images/ (extracted or rendered image files)

This phase saves no corpus vectors and creates no Chroma database. outputs/chunks/ is ignored by Git. JSONL is chosen so records can be streamed one at a time instead of loading a large JSON array.

The later vectorization step reads these files and writes to a persistent Chroma database at outputs/embeddings/nbc_2016_chroma/. It creates two collections: nbc_text_embeddings and nbc_image_embeddings. Each stored vector retains its source/page metadata.

The two source PDFs remain in data/ and remain excluded from Git.

## 6. First-stage chunk command

Run the first volume with:

uv run python -m src.chunk_nbc

The command:

- default to nationalbuilding01.pdf;
- accept one alternate PDF path for a later volume run;
- extract and validate text and images;
- support a dry-preview mode that prints representative chunks and metadata without writing files or embeddings;
- perform semantic text chunking;
- write text chunks and metadata to text_chunks.jsonl;
- save extracted/rendered images and write their references and metadata to image_chunks.jsonl;
- print progress, page coverage, chunk count, image count, and extraction warnings.

It uses temporary sentence embeddings to find semantic boundaries, then discards them. It does not save corpus embeddings, write to Chroma, or require API_BASE_URL, API_KEY, or CHAT_MODEL. It must not call a chat API.

Use stable text and image IDs so rerunning the chunking step replaces the chunk dataset cleanly. Do not modify or delete the future Chroma index as part of chunk generation.

## 7. Later vectorization step

Add a separate command after the saved chunks have been reviewed, for example:

uv run python -m src.embed_nbc

This command reads text_chunks.jsonl, image_chunks.jsonl, and the referenced image files. It loads the local text and image models, creates vectors in batches, validates vector counts and dimensions, then writes vectors plus source text/reference and metadata to the two Chroma collections. It does not re-extract or re-chunk the PDFs. If chunking changes, rerun the first-stage command and then rebuild vectors from the new saved chunk files.

## 8. Main implementation pieces

- chunk_nbc.py: page-level PDF text/image extraction, OCR fallback, semantic chunking, and JSONL output.
- SentenceTransformer temporary vectors: sentence-boundary similarity only; discarded after chunking.

Obsolete sample-corpus pipeline scripts and their generated outputs have been removed. The conversation history log is preserved.

## 9. First-stage completion checks

The volume 1 chunking step is successful when:

- the selected PDF is found and readable;
- every page is counted, including pages with no extractable text;
- extracted images are saved or reported with their source page;
- OCR gaps are identified rather than silently omitted;
- every text chunk record has text, source, page range, structural metadata, stable chunk ID, and token count;
- every image metadata record points to an existing image file and source volume/page;
- chunk records can be read one at a time from JSONL;
- semantic boundaries and page metadata have been reviewed before vectorization;
- no embeddings or Chroma collection are created in this first stage.

The later vectorization step is complete when each saved text/image record has one valid vector, vector dimensions are consistent within each collection, Chroma can be reopened, and source/page metadata is retained.
