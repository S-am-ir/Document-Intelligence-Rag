# Answer my Doc

An agentic Retrieval-Augmented Generation system that processes documents (PDFs, DOCX, CSVs, images) structurally using hybrid-chunking, extracts images/figures with VLM captioning, and answers questions with reflection loops.

## What It Does

Upload any document. Ask questions. Get answers grounded in your documents — with relevant figures shown inline.

Unlike traditional RAG that loses visual context, this system:
- Extracts and captions every figure/images in the document using a Vision Language Model
- Embeds figure descriptions alongside text for unified retrieval
- Uses an agentic pipeline (decompose → retrieve → rerank → reflect → generate)
- Falls back across multiple LLM providers automatically

## Quick Start

```bash
# Requires: Docker, NVIDIA GPU with nvidia-container-toolkit
docker compose up --build
```

Open `http://localhost:3000`. Upload document/s. Ask question's.

## Features

| Feature | Details |
|---------|---------|
| Document processing | Docling with GPU layout detection, table structure extraction, picture extraction |
| Figure captioning | VLM (Groq llama-4-scout) captions every figure in parallel (~1s for 14 images) |
| Embedding | nomic-embed-text-v1.5 (768-dim, prefix-aware, GPU) |
| Reranking | BAAI/bge-reranker-base (278M, GPU) with clean-text scoring |
| Agentic pipeline | LangGraph state machine: decompose → parallel retrieve → rerank → reflect → generate |
| Model fallback | minimax-m2.5-free → openai/gpt-oss-120b → qwen/qwen3-32b with 429 handling |
| Observability | LangSmith tracing (optional, set `LANGCHAIN_TRACING_V2=true`) |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, LangGraph, LangChain |
| Frontend | Next.js 15, React, TypeScript, Tailwind CSS |
| ML | Docling, sentence-transformers, PyTorch (CUDA 12.1) |
| Vector DB | PGVector on PostgreSQL 16 |
| LLM | OpenCode Zen, Groq (openai/gpt-oss-120b, qwen/qwen3-32b) |
| VLM | Groq llama-4-scout (figure captioning) |
| Infra | Docker Compose, NVIDIA GPU |

## Architecture

![Architectural Diagram](docs/demo/architectural_diagram.png)

## How It Works

1. **Upload**: Docling parses the document (layout detection, table extraction, chunking). Each figure is captioned by VLM in parallel. Chunks are embedded with nomic-embed-text-v1.5 and stored in PGVector.

2. **Query**: The user's question is decomposed into 2-3 targeted sub-queries. Each sub-query retrieves top-5 chunks from PGVector in parallel. Results are reranked with BGE cross-encoder on GPU.

3. **Reflection**: A reflection node checks if the retrieved context is sufficient. If not, the query is refined and retrieval re-runs (up to 2 retries).

4. **Generation**: The top chunks (text + figure explanations) are sent to the LLM, which generates a grounded answer. The answer writes as if the model saw the figures, since VLM captions are in the context.

## Documentation

- **[Demo](docs/demo/DEMO.md)** — Screenshots from live sessions with real queries
- **[Technical Breakdown](docs/TECHNICAL_WORKING.md)**
