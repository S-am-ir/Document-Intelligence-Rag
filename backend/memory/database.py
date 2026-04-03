"""
Shared PostgreSQL connection for session, image_store, and doc_index.
Single DATABASE_URL drives everything. Context manager handles commit/rollback.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg2

_TABLES_INITIALIZED = False


def _dsn() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return url


@contextmanager
def get_conn():
    """Yield a psycopg2 connection, auto-commit on success, rollback on error."""
    conn = psycopg2.connect(_dsn())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_tables() -> None:
    """
    Create app-specific tables if they don't exist.
    LangChain PGVector creates its own tables (langchain_pg_collection,
    langchain_pg_embedding) automatically on first use.
    Run once at startup via main.py.  Subsequent calls are no-ops.
    """
    global _TABLES_INITIALIZED
    if _TABLES_INITIALIZED:
        return

    with get_conn() as conn:
        cur = conn.cursor()

        # pgvector extension (Supabase has this enabled; harmless if already exists)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Session memory
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                updated_at  TIMESTAMPTZ DEFAULT NOW(),
                messages    JSONB DEFAULT '[]'::jsonb,
                context     JSONB DEFAULT '{}'::jsonb
            )
        """)

        # VLM-captioned figures extracted during ingestion
        cur.execute("""
            CREATE TABLE IF NOT EXISTS doc_images (
                image_id    TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                filename    TEXT,
                self_ref    TEXT,
                image_b64   TEXT,
                vlm_caption TEXT,
                page_no     INTEGER DEFAULT 0,
                stored_at   TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS doc_images_session_idx
            ON doc_images (session_id)
        """)

        # Lightweight doc metadata (not vectors)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS doc_index (
                session_id     TEXT,
                filename       TEXT,
                parse_method   TEXT,
                page_count     INTEGER,
                word_count     INTEGER,
                total_elements INTEGER,
                ingested_at    TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (session_id, filename)
            )
        """)

        # User-linked sessions (for authenticated users)
        cur.execute("""
            ALTER TABLE sessions
                ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT NULL
        """)

    _TABLES_INITIALIZED = True
    print("[database] Tables initialized.")
