"""
RAG — PGVector backend (replaces Chroma + SQLite).

All sessions share one PGVector collection ('documents').
session_id is stored as metadata on every chunk and used for filtered retrieval,
which avoids per-session collection overhead and enables clean deletion.

doc_index table lives in the same PostgreSQL instance.
"""

from __future__ import annotations

import os
from typing import List

import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from memory.database import get_conn, init_tables

TOP_K = 5
RERANK_TOP_K = 5
COLLECTION_NAME = "documents"


# ── Nomic Embeddings with prefix support (cached singleton) ────────────────

_EMBEDDINGS = None


class NomicEmbeddings(Embeddings):
    """
    nomic-ai/nomic-embed-text-v1.5 with proper prefix handling.
    Documents use 'search_document:' prefix, queries use 'search_query:'.
    """

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            device=device,
            trust_remote_code=True,
        )
        print(f"[embed] nomic-embed-text-v1.5 loaded on {device}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"search_document: {t}" for t in texts]
        return self.model.encode(
            prefixed, batch_size=64, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            [f"search_query: {text}"], batch_size=1, show_progress_bar=False
        )[0].tolist()


def _get_embeddings() -> NomicEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = NomicEmbeddings()
    return _EMBEDDINGS


# ── PGVector connection string ──────────────────────────────────────────────


def _pgvector_conn() -> str:
    """
    Derive a SQLAlchemy-compatible connection string from DATABASE_URL.
    Supabase provides postgresql:// — we upgrade it to postgresql+psycopg2://.
    """
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://") and "+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


def _get_vectorstore():
    from langchain_postgres.vectorstores import PGVector

    return PGVector(
        embeddings=_get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=_pgvector_conn(),
        use_jsonb=True,
    )


# ── Ingestion ───────────────────────────────────────────────────────────────


def ingest_documents(parsed_list: list[dict], session_id: str) -> dict:
    """
    Add pre-chunked documents to PGVector and record metadata in doc_index.
    session_id is injected into every chunk's metadata for filtered retrieval.
    """
    init_tables()
    vectorstore = _get_vectorstore()
    all_chunks: list[Document] = []
    sources: list[str] = []

    for parsed in parsed_list:
        chunks = parsed.get("chunks")
        if not chunks:
            print(f"[rag] Skipping {parsed.get('filename', '?')} — no chunks")
            continue

        # Stamp every chunk with session_id for later filtered retrieval + deletion
        for chunk in chunks:
            chunk.metadata["session_id"] = session_id

        all_chunks.extend(chunks)
        sources.append(parsed["filename"])

        # Record lightweight metadata in doc_index
        with get_conn() as conn:
            conn.cursor().execute(
                """
                INSERT INTO doc_index
                    (session_id, filename, parse_method, page_count, word_count,
                     total_elements, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (session_id, filename) DO UPDATE
                SET parse_method   = EXCLUDED.parse_method,
                    page_count     = EXCLUDED.page_count,
                    word_count     = EXCLUDED.word_count,
                    total_elements = EXCLUDED.total_elements,
                    ingested_at    = EXCLUDED.ingested_at
                """,
                (
                    session_id,
                    parsed["filename"],
                    parsed["parse_method"],
                    parsed["page_count"],
                    parsed["word_count"],
                    parsed["total_elements"],
                ),
            )

    if not all_chunks:
        print("[rag] No chunks to ingest after processing all files")
        return {"ingested": 0}

    print(
        f"[rag] Adding {len(all_chunks)} chunks to PGVector (collection={COLLECTION_NAME})"
    )
    vectorstore.add_documents(all_chunks)
    print(f"[rag] Successfully ingested {len(all_chunks)} chunks from {sources}")
    return {"ingested": len(all_chunks), "sources": sources}


# ── Retrieval ───────────────────────────────────────────────────────────────


def query_documents(question: str, session_id: str, top_k: int = TOP_K) -> str:
    """Return formatted string of top-k chunks for use in prompts."""
    try:
        vectorstore = _get_vectorstore()
        results = vectorstore.similarity_search(
            question, k=top_k, filter={"session_id": session_id}
        )
        if not results:
            return "[No relevant content found]"

        formatted = []
        for i, doc in enumerate(results):
            meta = doc.metadata
            source = meta.get("source", "unknown")
            page = meta.get("page_no", "?")
            heading = meta.get("headings") or meta.get("heading_context", "")
            header = f"[{i + 1}] {source} | page {page}"
            if heading:
                header += f" | section: {heading}"
            formatted.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"[RAG error: {e}]"


def query_documents_raw(
    question: str, session_id: str, top_k: int = TOP_K
) -> list[Document]:
    """Return raw Document objects (with metadata). Used by retrieve_worker."""
    try:
        vectorstore = _get_vectorstore()
        results = vectorstore.similarity_search(
            question, k=top_k, filter={"session_id": session_id}
        )
        print(f"[rag] query_documents_raw('{question[:40]}…') → {len(results)} results")
        return results
    except Exception as e:
        print(f"[rag] query_documents_raw FAILED: {e}")
        return []


# ── Session helpers ─────────────────────────────────────────────────────────


def session_has_documents(session_id: str) -> bool:
    """Check via doc_index (fast — no embedding required)."""
    init_tables()
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM doc_index WHERE session_id = %s LIMIT 1",
                (session_id,),
            )
            result = cur.fetchone() is not None
        print(f"[rag] session_has_documents({session_id[:8]}…) → {result}")
        return result
    except Exception as e:
        print(f"[rag] session_has_documents FAILED: {e}")
        raise  # Let caller decide how to handle — don't silently swallow


def get_ingested_filenames(session_id: str) -> set[str]:
    """Return set of filenames already ingested in doc_index for this session."""
    init_tables()
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT filename FROM doc_index WHERE session_id = %s",
                (session_id,),
            )
            return {row[0] for row in cur.fetchall()}
    except Exception:
        return set()


def get_session_doc_summary(session_id: str) -> str:
    init_tables()
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT filename, parse_method, page_count, word_count, total_elements
                FROM doc_index
                WHERE session_id = %s
                ORDER BY ingested_at
                """,
                (session_id,),
            )
            rows = cur.fetchall()
        if not rows:
            return "No documents indexed."
        lines = [
            f"  • {fn} — {pages}p, {words} words, {elems} elements ({method})"
            for fn, method, pages, words, elems in rows
        ]
        return "Documents in session:\n" + "\n".join(lines)
    except Exception:
        return "Document index unavailable."


def clear_session_docs(session_id: str) -> None:
    """Delete all vectors and metadata for a session."""
    # 1. Remove vectors from PGVector (raw SQL on langchain_pg_embedding)
    try:
        with get_conn() as conn:
            conn.cursor().execute(
                """
                DELETE FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                AND (cmetadata->>'session_id') = %s
                """,
                (COLLECTION_NAME, session_id),
            )
    except Exception as e:
        print(f"[rag] Vector deletion failed for session {session_id}: {e}")

    # 2. Remove doc_index rows
    try:
        with get_conn() as conn:
            conn.cursor().execute(
                "DELETE FROM doc_index WHERE session_id = %s",
                (session_id,),
            )
    except Exception as e:
        print(f"[rag] doc_index deletion failed for session {session_id}: {e}")
