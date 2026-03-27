# Technical Working Document

Complete technical breakdown of the Document Intelligence RAG system — how documents are processed, stored, retrieved, and how the agentic pipeline generates answers.

---

## Table of Contents

1. [Document Processing Pipeline](#1-document-processing-pipeline)
2. [Docling Data Structures](#2-docling-data-structures)
3. [VLM Figure Captioning](#3-vlm-figure-captioning)
4. [Chunking](#4-chunking)
5. [Embedding and Storage](#5-embedding-and-storage)
6. [Agentic RAG Pipeline](#6-agentic-rag-pipeline)
7. [Retrieval Flow](#7-retrieval-flow)
8. [Reranking](#8-reranking)
9. [Image Pipeline](#9-image-pipeline)
10. [Model Fallback](#10-model-fallback)
11. [Frontend-Backend Connection](#11-frontend-backend-connection)
12. [Database Schema](#12-database-schema)

---

## 1. Document Processing Pipeline

### 1.1 Entry Point

When a file is uploaded via `POST /api/upload`, the backend:

1. Saves the file to `data/uploads/{session_id}/{filename}`
2. Detects MIME type via `python-magic` (or `mimetypes` fallback)
3. Calls `route_and_parse(filepath, filename, caption_images=True)` from `backend/agent/tools/doc_router.py`
4. Calls `ingest_documents([result], session_id)` from `backend/agent/tools/rag.py`

### 1.2 File Type Routing

```python
def route_and_parse(filepath, filename, caption_images=True):
    mime = detect_mime(filepath)        # e.g., "application/pdf"
    suffix = Path(filepath).suffix      # e.g., ".pdf"

    if mime.startswith("image/"):
        return _handle_image(...)        # PNG/JPG/WEBP → direct VLM

    # For PDFs: check if OCR is needed
    force_ocr = ("pdf" in mime) and needs_ocr(filepath)
```

### 1.3 OCR Detection (`needs_ocr`)

For PDFs, `needs_ocr()` extracts text from the first 3 pages using `pypdf`:
```python
def needs_ocr(filepath):
    reader = pypdf.PdfReader(filepath)
    sample = "".join(page.extract_text() or "" for page in reader.pages[:3])
    return len(sample.strip()) < 50
```

- If the PDF has native text (e.g., research papers from arXiv): `needs_ocr()` returns `False` → no OCR, faster processing
- If the PDF is a scan (e.g., photographed document): `needs_ocr()` returns `True` → OCR via RapidOCR on GPU

### 1.4 Docling DocumentConverter

The PDF is processed by Docling's `DocumentConverter` with CUDA acceleration:

```python
pipeline_opts = PdfPipelineOptions(
    do_ocr=force_ocr,                    # OCR only if needed
    do_table_structure=True,              # Extract table rows/columns
    generate_picture_images=True,         # Extract figures as PIL images
    ocr_options=EasyOcrOptions(lang=["en"]),
    accelerator_options=AcceleratorOptions(
        device=AcceleratorDevice.CUDA,
        num_threads=4,
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
    }
)
conv_res = converter.convert(filepath)     # Returns ConversionResult
doc = conv_res.document                    # DoclingDocument object
```

**What Docling does internally:**
1. **Layout detection**: A vision model identifies page regions (text blocks, tables, figures, headers, footnotes)
2. **OCR** (if enabled): RapidOCR extracts text from scanned regions
3. **Table structure**: Identifies rows, columns, cell contents
4. **Picture extraction**: Extracts figures as PIL `Image` objects

### 1.5 Other File Types

For non-PDF files, Docling handles them through its format detection:

- **DOCX**: Direct XML parsing, extracts text + images + tables
- **CSV**: Parsed as tabular data
- **TXT/Markdown**: Read as plain text
- **Images**: Treated as single-element documents

---

## 2. Docling Data Structures

### 2.1 DoclingDocument

`DoclingDocument` is the internal representation. It's a tree of items:

```
DoclingDocument
├── pages[]                     # Page metadata
├── texts[]                     # TextItem objects
│   ├── headings (level 1-6)
│   ├── paragraphs
│   ├── footnotes
│   └── list items
├── tables[]                    # TableItem objects
│   └── rows → cells
├── pictures[]                  # PictureItem objects (figures)
└── code[]                      # CodeItem objects (if enabled)
```

### 2.2 TextItem

Represents a text block (heading, paragraph, footnote, etc.):

```python
class TextItem:
    self_ref: str          # e.g., "#/texts/45"
    label: str             # "heading", "text", "footnote", "list_item"
    text: str              # The actual text content
    prov: List[ProvInfo]   # Provenance (page number, bounding box)
```

Example from a research paper:
```
self_ref = "#/texts/45"
label = "heading"
text = "3 Method"
prov = [ProvInfo(page_no=3, bbox=BoundingBox(...))]
```

### 2.3 PictureItem

Represents a figure, chart, or diagram:

```python
class PictureItem:
    self_ref: str          # e.g., "#/pictures/1"
    label: str             # "picture"
    prov: List[ProvInfo]   # Page number and bounding box
    # Methods:
    def caption_text(doc)  # Returns caption text if available
    def get_image(doc)     # Returns PIL.Image of the figure
    def classification     # Returns classification label
```

Example from the research paper:
```
self_ref = "#/pictures/1"
label = "picture"
prov = [ProvInfo(page_no=4, bbox=BoundingBox(...))]
caption_text(doc) = ""   # Docling didn't extract a caption for this figure
get_image(doc) = <PIL.Image>  # 23KB PNG
```

### 2.4 TableItem

Represents a table with rows and cells:

```python
class TableItem:
    self_ref: str
    label: str             # "table"
    prov: List[ProvInfo]
    # Contains grid data accessible via table export
```

### 2.5 Item Provenance

Each item has provenance information linking it to its source location:

```python
class ProvInfo:
    page_no: int           # Page number (1-indexed)
    bbox: BoundingBox      # Bounding box coordinates
```

### 2.6 Iterating Items

The document tree is traversed via:
```python
for item, level in doc.iterate_items():
    if isinstance(item, PictureItem):
        # This is a figure
    elif isinstance(item, TextItem):
        if item.label == "heading":
            # This is a heading
```

---

## 3. VLM Figure Captioning

### 3.1 How Pictures Are Found

After Docling produces a `DoclingDocument`, all `PictureItem` objects are collected:

```python
picture_map = {}
for item, _ in doc.iterate_items():
    if isinstance(item, PictureItem):
        picture_map[item.self_ref] = item
```

### 3.2 Matching Pictures to Chunks

Docling's `PictureItem.self_ref` (e.g., `#/pictures/1`) doesn't match chunk refs (e.g., `#/texts/45`). Pictures are separate from text. The system matches by **page number**:

1. Build a page→chunks index from chunk metadata
2. For each picture, get its page number from `pic.prov[0].page_no`
3. Find the first chunk on that page
4. Attach the VLM caption to that chunk

```python
# Build page → chunk indices
page_chunks = {}
for ci, chunk in enumerate(chunks):
    for doc_item in chunk.metadata["dl_meta"]["doc_items"]:
        for prov in doc_item.get("prov", []):
            page = prov.get("page_no")
            if page:
                page_chunks.setdefault(page, []).append(ci)

# Match each picture to a chunk on the same page
for ref, pic in picture_map.items():
    page_no = pic.prov[0].page_no
    target_idx = page_chunks[page_no][0]  # First chunk on that page
    # Caption and attach...
```

### 3.3 Parallel VLM Captioning

All images are captioned in parallel using `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(caption_one, job): job for job in jobs}
    for future in as_completed(futures):
        job, vlm_desc = future.result()
        # Attach to chunk...
```

**VLM model**: Groq `meta-llama/llama-4-scout-17b-16e-instruct`

**Prompt**:
```
In 2-4 sentences, explain what this figure means and what it demonstrates.
Focus on the finding, trend, or relationship — not visual details like colors,
axis labels, or bar positions.
```

14 images captioned in ~1.1s (vs ~22s sequential).

### 3.4 Caption Result Format

Each VLM response is a 2-4 sentence meaning-focused description:
```
"The figure demonstrates that as the ensemble size increases, the accuracy 
of various models also increases. Notably, combining multiple smaller models 
can achieve performance comparable to larger single models."
```

---

## 4. Chunking

### 4.1 HybridChunker

Docling's `HybridChunker` creates chunks from the `DoclingDocument`:

```python
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
    max_tokens=512,
)

loader = DoclingLoader(
    file_path=filepath,
    export_type=ExportType.DOC_CHUNKS,
    chunker=HybridChunker(tokenizer=tokenizer),
)
chunks = loader.load()  # Returns List[Document]
```

### 4.2 Chunk Structure

Each chunk is a LangChain `Document` object:

```python
Document(
    page_content="3 Method\nIn this section, we introduce Agent Forest...",
    metadata={
        "source": "/app/data/uploads/{sid}/paper.pdf",
        "page_no": 3,
        "element_type": "text",
        "headings": ["3 Method"],
        "dl_meta": {
            "doc_items": [
                {
                    "label": "text",
                    "self_ref": "#/texts/45",
                    "prov": [{"page_no": 3, "bbox": {...}}]
                },
                {
                    "label": "heading",
                    "self_ref": "#/texts/44",
                    "prov": [{"page_no": 3}]
                }
            ]
        }
    }
)
```

### 4.3 Chunk Content After VLM Captioning

If a picture was found on the same page, the chunk content is modified:

```
[Figure Explanation (page 4)]: Figure 2: Illustration of Agent Forest. 
The figure illustrates a process where multiple LLM agents generate answers 
based on either a query alone or a query combined with prompts. The agents' 
individual answers are then aggregated through majority voting...

3 Method
In this section, we introduce Agent Forest...
```

Additional metadata fields added:
```python
{
    "vlm_caption": "The figure illustrates a process where multiple LLM agents...",
    "docling_caption": "Figure 2: Illustration of Agent Forest...",
    "has_vlm": True,
    "image_b64": "iVBORw0KGgo...",    # Removed after storing in image_store
    "self_ref": "#/pictures/1",
    "page_no": 4,
    "image_id": "25bddd7478535632add08fa2c1fd9ed5"
}
```

For an 18-page research paper: typically 40-60 chunks.

---

## 5. Embedding and Storage

### 5.1 Nomic Embeddings

Each chunk is embedded with `nomic-ai/nomic-embed-text-v1.5` (768 dimensions):

```python
class NomicEmbeddings(Embeddings):
    def embed_documents(self, texts):
        prefixed = [f"search_document: {t}" for t in texts]
        return self.model.encode(prefixed, batch_size=64).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [f"search_query: {text}"], batch_size=1
        )[0].tolist()
```

The `search_document:` / `search_query:` prefixes are required by nomic-embed-text-v1.5 for optimal retrieval quality.

### 5.2 PGVector Storage

Chunks are stored in PostgreSQL with PGVector extension:

```python
vectorstore = PGVector(
    embeddings=embeddings,
    connection=url,                    # postgresql+psycopg2://aether:aether@db:5432/aether
    collection_name="documents",       # Single collection for all sessions
    use_jsonb=True,                    # Metadata stored as JSONB
)
vectorstore.add_documents(chunks)
```

Each chunk is stored as:
- `embedding`: vector(768) — the nomic embedding
- `document`: text — full chunk content (including VLM caption if present)
- `cmetadata`: jsonb — chunk metadata including `session_id` for filtered retrieval

### 5.3 Session Filtering

All chunks share one PGVector collection. Session isolation is done via metadata filter:

```python
results = vectorstore.similarity_search(
    query,
    k=top_k,
    filter={"session_id": session_id}  # JSONB filter
)
```

---

## 6. Agentic RAG Pipeline

### 6.1 State Structure

The pipeline uses LangGraph with a shared state (`AgentState` TypedDict):

```python
class AgentState(TypedDict):
    session_id: str                    # Session identifier
    query: str                         # Original user query
    uploaded_files: List[dict]         # Files to process
    messages: Annotated[List, add_messages]  # Conversation history
    doc_parse_results: List[dict]      # Parsed file metadata
    sub_queries: List[str]             # Decomposed query variants
    raw_chunks: Annotated[List, _merge_lists]  # Retrieved chunks (merged from parallel workers)
    reranked_chunks: List[dict]        # Top chunks after reranking
    reflection_passed: bool            # Whether reflection approved the context
    retry_count: int                   # Number of reflection retries (max 2)
    final_answer: Optional[str]        # Generated answer
    retrieved_images: List[dict]       # Relevant images for display
    stream_events: Annotated[List, _merge_lists]  # SSE events to frontend
```

### 6.2 Graph Structure

```
START
  │
  ▼
ingest ──► check_has_docs
              │
              ├── "decompose" ──► decompose ──► retrieve_dispatcher
              │                                      │
              │                              Send(sub_query_1) ──► retrieve_worker
              │                              Send(sub_query_2) ──► retrieve_worker
              │                              Send(sub_query_3) ──► retrieve_worker
              │                                      │
              │                                    rerank
              │                                      │
              │                                    reflect
              │                                      │
              │                         ┌── "generate" ──► generate ──► final ──► END
              │                         │
              │                         └── "decompose" (retry, max 2)
              │
              └── "no_docs" ──► no_docs ──► final ──► END
```

### 6.3 Node Details

**ingest_node**: Checks `doc_index` table for files already ingested. If file is not found, parses with Docling + stores in PGVector. If already ingested, skips entirely. This ensures first-time processing happens only once.

**decompose_node**: Sends the user's query to LLM with instructions to generate 2-3 sub-queries targeting different content types:
- Core definition ("What is X?")
- Method/procedure ("How does X work?")
- Figures/diagrams ("X diagram description")

The LLM is instructed to preserve exact terms from the original query.

**retrieve_dispatcher + retrieve_worker** (parallel via Send API):

```python
def spawn_retrieve_workers(state):
    return [
        Send("retrieve_worker", {
            **state,
            "_sub_query": sq,
            "raw_chunks": [],  # Each worker starts fresh
        })
        for sq in state["sub_queries"]
    ]
```

Each worker independently calls `query_documents_raw(sub_query, session_id, top_k=5)`. LangGraph's `_merge_lists` reducer merges all workers' chunks into a single `raw_chunks` list.

**rerank_node**: Deduplicates chunks (parallel workers may return the same content), strips `[Figure Explanation (page X)]:` bracket notation, scores with BGE cross-encoder, selects top 5. Then selects the single best image from those top 5.

**reflect_node**: Sends all context to LLM asking "Is this sufficient to answer the query?" Returns yes/no. If no and retry_count < 2, routes back to decompose with a refined query.

**generate_node**: Sends all context to LLM with instructions to write as if analyzing figures directly. Returns grounded answer.

**final_node**: Carries forward `final_answer`, `stream_events`, and `retrieved_images` to the SSE handler.

### 6.4 Conditional Edges

```python
# ingest → check_has_docs
g.add_conditional_edges("ingest", check_has_docs, {
    "decompose": "decompose",
    "no_docs": "no_docs",
})

# reflect → retry or generate
# (handled by Command in reflect_node returning Send to either "generate" or "decompose")
```

### 6.5 Retry Logic

The reflection loop has a maximum of 2 retries:
```python
if reflection_passed:
    return Command(goto="generate", update={"reflection_passed": True})
elif retry_count < 2:
    return Command(
        goto="decompose",
        update={"retry_count": retry_count + 1, "sub_queries": refined_queries}
    )
else:
    return Command(goto="generate")  # Force generate even without passing reflection
```

---

## 7. Retrieval Flow

### 7.1 Cosine Similarity Search

Each `retrieve_worker` calls PGVector:

```python
results = vectorstore.similarity_search(
    query,                               # search_query: prefixed embedding
    k=5,                                 # Top 5 results
    filter={"session_id": session_id}    # Session isolation
)
```

PGVector computes cosine distance: `1 - cos(query_embedding, chunk_embedding)`

### 7.2 Result Format

Each result is a LangChain `Document`:
```python
Document(
    page_content="[Figure Explanation (page 11)]: Figure 7: ...",
    metadata={
        "source": "/app/data/uploads/{sid}/paper.pdf",
        "page_no": 11,
        "image_id": "5fd3d027ebee842767f358d73e699276",
        "vlm_caption": "The figure demonstrates that a proposed method...",
        "has_vlm": True,
        "session_id": "87794c1c-..."
    }
)
```

### 7.3 Deduplication

Multiple sub-queries may return the same chunks. Reranker deduplicates by content:

```python
seen, unique = set(), []
for c in chunks:
    content = c.get("content", "")
    if content and content not in seen:
        seen.add(content)
        unique.append(c)
```

---

## 8. Reranking

### 8.1 BGE Cross-Encoder

`BAAI/bge-reranker-base` (278M params) scores query-document pairs:

```python
encoder = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

# Clean text for scoring (brackets confuse the cross-encoder)
pairs = [(query, _clean_for_rerank(chunk["content"])[:1500]) for chunk in chunks]
scores = encoder.predict(pairs, batch_size=32)
```

Bracket cleaning regex: `\[Figure Explanation \(page \d+\)\]:\s*` → removed

### 8.2 Score Interpretation

BGE reranker produces positive scores for relevant pairs. Scores are relative (not calibrated to 0-1):
- 0.7-1.0: Strong match
- 0.3-0.7: Moderate match
- 0.0-0.3: Weak match
- Negative: Irrelevant

### 8.3 Image Selection

From the top 5 reranked chunks, the single best image is selected:
```python
for chunk in reranked[:5]:
    image_id = chunk.get("metadata", {}).get("image_id")
    score = chunk.get("relevance_score", 0)
    if image_id and score > best_score:
        best_image = fetch_image(image_id)
        best_score = score
```

---

## 9. Image Pipeline

### 9.1 Image Storage (at ingestion)

VLM-captioned images are stored in `doc_images` table:

```python
image_id = store_image(
    session_id=session_id,
    filename=filename,
    self_ref="#/pictures/1",
    image_b64="iVBORw0KGgo...",  # base64-encoded PNG
    vlm_caption="The figure illustrates...",
    page_no=4,
)
```

The `image_id` is an MD5 hash of the image content (deduplicates identical images).

### 9.2 Image Metadata in Chunks

After storing, the chunk metadata is updated:
```python
meta["image_id"] = image_id
meta["image_b64"] = ""  # Removed from chunk (stored in doc_images instead)
```

### 9.3 Image Retrieval (at query time)

When a chunk with `image_id` is in the top 5 reranked results:
```python
img = fetch_image(image_id)  # SELECT from doc_images
# Returns: {image_id, image_b64, caption, source, page}
```

### 9.4 Image Delivery to Frontend

Images are sent via two SSE events:
1. `final` event: includes image metadata (id, caption, source, page) in the `images` array
2. `image` event: includes the full `image_b64` data separately (avoids bloating the final event)

---

## 10. Model Fallback

### 10.1 Chain

| Priority | Model | Provider | Use Case |
|----------|-------|----------|----------|
| Primary | minimax-m2.5-free | OpenCode Zen | All LLM calls |
| Fallback 1 | openai/gpt-oss-120b | Groq | When primary is rate-limited |
| Fallback 2 | qwen/qwen3-32b | Groq | When fallback 1 is rate-limited |

### 10.2 Fallback Logic

```python
def invoke_with_fallback(messages, tier="primary", streaming=False):
    for t in ["primary", "fallback1", "fallback2"]:
        try:
            model = get_model(t, streaming)
            resp = model.invoke(messages)
            return resp, f"{t}/{model.model_name}"
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                print(f"[models] ✗ {t}: RATE LIMITED — trying next tier")
                continue
            raise  # Non-rate-limit errors propagate
    raise RuntimeError("All providers failed")
```

### 10.3 Where Fallback Is Used

- **decompose_node**: Query decomposition
- **generate_node**: Final answer generation
- **reflect_node**: Context sufficiency check

### 10.4 VLM (Groq llama-4-scout)

Separate from the LLM fallback chain. VLM is only used for figure captioning. If Groq is unavailable, captioning is skipped and figures are stored without descriptions.

---

## 11. Frontend-Backend Connection

### 11.1 Upload Flow

```
Frontend                          Backend
   │                                │
   │── POST /api/upload ───────────►│
   │   session_id + files[]         │
   │                                │── Save to disk
   │                                │── Docling parse (CUDA)
   │                                │── VLM caption (parallel)
   │                                │── Embed + store (PGVector)
   │◄── {uploaded: [...],  ─────────│
   │    processed: [...]}           │
```

Processing takes ~77s for an 18-page PDF. The upload HTTP request blocks until complete.

### 11.2 Query Flow (SSE)

```
Frontend                          Backend
   │                                │
   │── POST /api/query/stream ─────►│
   │   session_id + query + files   │
   │                                │
   │◄── data: {"type":"start"} ────│
   │                                │── ingest_node
   │◄── data: {"type":"agent",─────│
   │     "event":{...}}            │── decompose_node
   │◄── data: {"type":"agent",─────│
   │     "event":{...}}            │── retrieve (parallel)
   │◄── data: {"type":"agent",─────│
   │     "event":{...}}            │── rerank
   │◄── data: {"type":"agent",─────│
   │     "event":{...}}            │── reflect
   │◄── data: {"type":"agent",─────│
   │     "event":{...}}            │── generate
   │◄── data: {"type":"final",─────│
   │     "output":"...",            │
   │     "images":[...]}           │
   │◄── data: {"type":"image",─────│  (one per image)
   │     "image_id":"...",         │
   │     "image_b64":"..."}        │
   │◄── data: {"type":"done"} ────│
```

### 11.3 SSE Event Types

| Type | Fields | Description |
|------|--------|-------------|
| `start` | `session_id` | Stream started |
| `agent_event` | `event: {agent, type, message}` | Pipeline node event |
| `final` | `output, images[]` | Final answer + image metadata |
| `image` | `image_id, image_b64, caption` | Full image data |
| `error` | `message` | Error occurred |
| `done` | — | Stream complete |

### 11.4 Agent Event Types

| Agent | Type | Description |
|-------|------|-------------|
| ingest | parsing | "Parsing {filename}..." |
| ingest | parsed | "{filename} → docling \| 51 chunks" |
| ingest | ingested | "Ingested 51 chunks from {filename}" |
| ingest | error | "Failed to parse {filename}: {error}" |
| decompose | sub_queries | "→ 3 sub-queries: q1 \| q2 \| q3" |
| retrieve | retrieved | "'{query}' → 5 chunks" |
| rerank | reranking | "Reranking 20 chunks..." |
| rerank | top_chunks | "Top 5 chunks selected" |
| reflect | check | "Checking if context is sufficient..." |
| reflect | result | "Context sufficient: yes/no" |
| generate | generating | "Generating answer..." |
| generate | complete | "Answer generated (via {model})" |
| final | complete | "Done" |

---

## 12. Database Schema

### 12.1 PostgreSQL Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector for embeddings
```

### 12.2 Tables

**sessions** — Conversation history:
```sql
CREATE TABLE sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT DEFAULT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    messages    JSONB DEFAULT '[]'::jsonb,
    context     JSONB DEFAULT '{}'::jsonb
);
```

Messages format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`

**doc_index** — Ingested file registry:
```sql
CREATE TABLE doc_index (
    session_id     TEXT,
    filename       TEXT,
    parse_method   TEXT,
    page_count     INTEGER,
    word_count     INTEGER,
    total_elements INTEGER,
    ingested_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (session_id, filename)
);
```

**doc_images** — Extracted figure images:
```sql
CREATE TABLE doc_images (
    image_id    TEXT PRIMARY KEY,        -- MD5 hash of image content
    session_id  TEXT NOT NULL,
    filename    TEXT,
    self_ref    TEXT,                    -- Docling self_ref, e.g., "#/pictures/1"
    image_b64   TEXT,                    -- base64-encoded PNG
    vlm_caption TEXT,                    -- VLM-generated description
    page_no     INTEGER DEFAULT 0,
    stored_at   TIMESTAMPTZ DEFAULT NOW()
);
```

**langchain_pg_collection** + **langchain_pg_embedding** — Auto-created by LangChain PGVector:
- `langchain_pg_collection`: Collection names (we use "documents")
- `langchain_pg_embedding`: Chunks with embeddings, documents, and JSONB metadata

Embedding metadata includes: `source`, `page_no`, `element_type`, `session_id`, `image_id`, `vlm_caption`, `has_vlm`, `docling_caption`.
