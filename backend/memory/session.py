"""
Session memory — PostgreSQL backend.
Replaces SQLite. Uses shared get_conn() from database.py.
datetime.utcnow() replaced with timezone-aware datetime.now(UTC).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from .database import get_conn, init_tables

MAX_TURNS = 6  # 6 messages = 3 full user/assistant turns


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Public API ──────────────────────────────────────────────────────────────


def create_session(user_id: str | None = None) -> str:
    init_tables()
    session_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.cursor().execute(
            """
            INSERT INTO sessions (session_id, created_at, updated_at, messages, context, user_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """,
            (session_id, _now(), _now(), json.dumps([]), json.dumps({}), user_id),
        )
    return session_id


def get_session_messages(session_id: str) -> list[dict]:
    init_tables()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT messages FROM sessions WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return []
    msgs = row[0] if isinstance(row[0], list) else json.loads(row[0])
    return msgs[-(MAX_TURNS):]


def save_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    init_tables()
    existing = get_session_messages(session_id)
    existing.append({"role": "user", "content": user_msg})
    existing.append({"role": "assistant", "content": assistant_msg})
    trimmed = existing[-(MAX_TURNS):]

    with get_conn() as conn:
        conn.cursor().execute(
            """
            INSERT INTO sessions (session_id, created_at, updated_at, messages, context, user_id)
            VALUES (%s, %s, %s, %s, %s, NULL)
            ON CONFLICT (session_id) DO UPDATE
            SET messages = EXCLUDED.messages, updated_at = EXCLUDED.updated_at
            """,
            (session_id, _now(), _now(), json.dumps(trimmed), json.dumps({})),
        )


def get_session_context(session_id: str) -> dict:
    init_tables()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT context FROM sessions WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row or not row[0]:
        return {}
    ctx = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return ctx


def save_session_context(session_id: str, context: dict) -> None:
    init_tables()
    existing = get_session_context(session_id)
    existing.update(context)
    with get_conn() as conn:
        conn.cursor().execute(
            "UPDATE sessions SET context = %s, updated_at = %s WHERE session_id = %s",
            (json.dumps(existing), _now(), session_id),
        )


def get_or_create_session(session_id: str | None, user_id: str | None = None) -> str:
    if session_id:
        init_tables()
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM sessions WHERE session_id = %s",
                (session_id,),
            )
            if cur.fetchone():
                # Link to user if not already linked
                if user_id:
                    conn.cursor().execute(
                        "UPDATE sessions SET user_id = %s WHERE session_id = %s AND user_id IS NULL",
                        (user_id, session_id),
                    )
                return session_id
    return create_session(user_id)


def get_user_sessions(user_id: str) -> list[dict]:
    """Get all sessions for a user, ordered by most recent activity."""
    init_tables()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT session_id, created_at, updated_at, messages
            FROM sessions
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT 50
            """,
            (user_id,),
        )
        rows = cur.fetchall()

    sessions = []
    for row in rows:
        msgs = row[3] if isinstance(row[3], list) else json.loads(row[3])
        # Generate a title from the first user message
        title = "New conversation"
        for m in msgs:
            if m.get("role") == "user":
                title = m["content"][:60]
                if len(m["content"]) > 60:
                    title += "..."
                break
        sessions.append(
            {
                "session_id": row[0],
                "created_at": row[1],
                "updated_at": row[2],
                "title": title,
                "message_count": len(msgs),
            }
        )
    return sessions


def link_session_to_user(session_id: str, user_id: str) -> None:
    """Link an existing session to a user (for post-auth)."""
    init_tables()
    with get_conn() as conn:
        conn.cursor().execute(
            "UPDATE sessions SET user_id = %s WHERE session_id = %s",
            (user_id, session_id),
        )
