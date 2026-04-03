import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import compiled_graph
from agent.state import AgentState
from memory.session import (
    get_or_create_session,
    get_session_messages,
    save_turn,
    get_user_sessions,
    link_session_to_user,
)

router = APIRouter()

UPLOADS_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {".pdf", ".docx", ".txt", ".md", ".csv", ".png", ".jpg", ".jpeg", ".webp"}


class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@router.get("/health")
async def health():
    return {"status": "ok", "mode": "agentic-rag"}


@router.post("/session")
async def session_route(body: SessionRequest = SessionRequest()):
    return {"session_id": get_or_create_session(body.session_id, body.user_id)}


@router.get("/sessions/{user_id}")
async def list_user_sessions(user_id: str):
    """Get all sessions for an authenticated user."""
    sessions = get_user_sessions(user_id)
    return {"sessions": sessions}


@router.post("/sessions/{session_id}/link/{user_id}")
async def link_session(session_id: str, user_id: str):
    """Link an existing anonymous session to a user after authentication."""
    link_session_to_user(session_id, user_id)
    return {"linked": True}


@router.post("/upload")
async def upload(session_id: str = Form(...), files: List[UploadFile] = File(...)):
    d = UPLOADS_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        sfx = Path(f.filename).suffix.lower()
        if sfx not in ALLOWED:
            raise HTTPException(400, f"File type '{sfx}' is not supported.")
        dest = d / f.filename
        with open(dest, "wb") as fp:
            shutil.copyfileobj(f.file, fp)
        out.append(
            {
                "name": f.filename,
                "path": str(dest),
                "type": sfx.lstrip("."),
                "size": dest.stat().st_size,
            }
        )

    # ── Immediately parse + ingest so queries are instant ─────────────────
    from agent.tools.rag import get_ingested_filenames, ingest_documents
    from agent.tools.image_store import store_image, count_session_images

    already_ingested = get_ingested_filenames(session_id)
    parse_results = []

    for file_info in out:
        filename = file_info["name"]
        filepath = file_info["path"]

        # Skip if already ingested and images exist
        if filename in already_ingested and count_session_images(session_id) > 0:
            print(f"[upload] Skipping (already ingested): {filename}")
            continue

        print(f"[upload] Processing: {filename}")
        try:
            from agent.tools.doc_router import route_and_parse

            result = route_and_parse(filepath, filename, caption_images=True)

            # Store VLM images
            images_stored = 0
            for chunk in result.get("chunks", []):
                meta = chunk.metadata
                if meta.get("has_vlm") and meta.get("image_b64"):
                    image_id = store_image(
                        session_id=session_id,
                        filename=filename,
                        self_ref=meta.get("self_ref", filename),
                        image_b64=meta.pop("image_b64"),
                        vlm_caption=meta.get("vlm_caption", ""),
                        page_no=meta.get("page_no", 0),
                    )
                    meta["image_id"] = image_id
                    meta.pop("self_ref", None)
                    images_stored += 1

            # Ingest chunks
            r = ingest_documents([result], session_id)
            print(
                f"[upload] Ingested {r['ingested']} chunks, {images_stored} images "
                f"from {filename}"
            )
            parse_results.append(
                {
                    "filename": filename,
                    "chunks": r["ingested"],
                    "images": images_stored,
                }
            )
        except Exception as e:
            print(f"[upload] Processing FAILED for {filename}: {e}")
            import traceback

            traceback.print_exc()
            parse_results.append({"filename": filename, "error": str(e)})

    return {
        "uploaded": out,
        "count": len(out),
        "processed": parse_results,
    }


@router.delete("/upload/{session_id}")
async def clear_uploads(session_id: str):
    if (UPLOADS_DIR / session_id).exists():
        shutil.rmtree(UPLOADS_DIR / session_id)
    from agent.tools.rag import clear_session_docs
    from agent.tools.image_store import clear_session_images

    clear_session_docs(session_id)
    clear_session_images(session_id)
    return {"cleared": True}


@router.delete("/upload/{session_id}/{filename}")
async def delete_file(session_id: str, filename: str):
    """Delete a single uploaded file and its associated vectors/images."""
    import urllib.parse

    filename = urllib.parse.unquote(filename)
    filepath = UPLOADS_DIR / session_id / filename
    if filepath.exists():
        filepath.unlink()

    # Clean up vectors and metadata for this specific file
    try:
        from memory.database import get_conn

        with get_conn() as conn:
            cur = conn.cursor()
            # Delete embeddings whose source metadata matches this file
            cur.execute(
                """
                DELETE FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = 'documents'
                )
                AND (cmetadata->>'session_id') = %s
                AND (cmetadata->>'source') = %s
                """,
                (session_id, filename),
            )
            # Delete from doc_index
            cur.execute(
                "DELETE FROM doc_index WHERE session_id = %s AND filename = %s",
                (session_id, filename),
            )
            # Delete associated images
            cur.execute(
                "DELETE FROM doc_images WHERE session_id = %s AND filename = %s",
                (session_id, filename),
            )
    except Exception as e:
        print(f"[routes] Cleanup error for {filename}: {e}")

    return {"deleted": filename}


@router.post("/query")
async def query(
    query: str = Form(...),
    session_id: str = Form(...),
    uploaded_files: str = Form(default="[]"),
):
    """
    Main query endpoint. Runs the agentic RAG graph and returns the response.
    """
    try:
        files_list = json.loads(uploaded_files) if uploaded_files else []
        print(
            f"[query] session={session_id[:8]}… "
            f"query='{query[:50]}' files={len(files_list)}"
        )

        history = [
            (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            for m in get_session_messages(session_id)
        ]

        initial: AgentState = {
            "session_id": session_id,
            "query": query,
            "messages": history + [HumanMessage(content=query)],
            "uploaded_files": files_list,
            "doc_parse_results": [],
            "sub_queries": [],
            "raw_chunks": [],
            "reranked_chunks": [],
            "reflection_passed": False,
            "retry_count": 0,
            "final_answer": None,
            "retrieved_images": [],
            "stream_events": [],
        }

        final_answer = None
        final_state: dict = {}
        events: list = []

        async for chunk in compiled_graph.astream(
            initial, config={"max_concurrency": 4}
        ):
            for node_name, node_state in chunk.items():
                print(f"[query] Node completed: {node_name}")
                node_events = node_state.get("stream_events", [])
                events.extend(node_events)
                if node_state.get("final_answer"):
                    final_answer = node_state["final_answer"]
                    final_state = node_state

        print(f"[query] Graph complete. final_answer length: {len(final_answer or '')}")

        images = final_state.get("retrieved_images", [])
        save_turn(session_id, query, final_answer or "Done.")

        return {
            "output": final_answer or "Done.",
            "events": events,
            "images": [
                {
                    "image_id": i["image_id"],
                    "caption": i.get("caption", ""),
                    "source": i.get("source", ""),
                    "page": i.get("page", 0),
                    "image_b64": i.get("image_b64", ""),
                }
                for i in images
            ],
        }

    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"[query] ═══ ERROR ═══")
        print(tb_text)
        print(f"[query] ═══ END ERROR ═══")
        error_msg = f"{exc_type.__name__}: {exc_value}" if exc_type else str(e)
        return {"error": error_msg, "trace": tb_text[-500:]}
