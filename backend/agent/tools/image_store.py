"""
Persist VLM-captioned figures — PostgreSQL backend.
Stored at ingest time (pipeline.py ingest_node), fetched at query time (rerank_node).
"""

from __future__ import annotations

import hashlib

from memory.database import get_conn, init_tables


def store_image(
    session_id: str,
    filename: str,
    self_ref: str,
    image_b64: str,
    vlm_caption: str = "",
    page_no: int = 0,
) -> str:
    init_tables()
    image_id = hashlib.md5(f"{session_id}{filename}{self_ref}".encode()).hexdigest()
    with get_conn() as conn:
        conn.cursor().execute(
            """
            INSERT INTO doc_images
                (image_id, session_id, filename, self_ref, image_b64, vlm_caption, page_no)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (image_id) DO UPDATE
            SET image_b64 = EXCLUDED.image_b64,
                vlm_caption = EXCLUDED.vlm_caption
            """,
            (image_id, session_id, filename, self_ref, image_b64, vlm_caption, page_no),
        )
    return image_id


def fetch_image(image_id: str) -> dict | None:
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT image_b64, vlm_caption, filename, page_no
                FROM doc_images
                WHERE image_id = %s
                """,
                (image_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "image_id": image_id,
            "image_b64": row[0],
            "caption": row[1],
            "source": row[2],
            "page": row[3],
        }
    except Exception:
        return None


def clear_session_images(session_id: str) -> None:
    try:
        with get_conn() as conn:
            conn.cursor().execute(
                "DELETE FROM doc_images WHERE session_id = %s",
                (session_id,),
            )
    except Exception:
        pass


def count_session_images(session_id: str) -> int:
    """Return the number of images stored for a session."""
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM doc_images WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0
