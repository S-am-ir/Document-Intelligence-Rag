import sys
import traceback

from langgraph.graph import StateGraph, END

from .state import AgentState
from .pipeline import (
    ingest_node as _ingest_node,
    decompose_node as _decompose_node,
    retrieve_worker as _retrieve_worker,
    spawn_retrieve_workers as _spawn_retrieve_workers,
    rerank_node as _rerank_node,
    reflect_node as _reflect_node,
    generate_node as _generate_node,
)


# ── Diagnostic wrapper ──────────────────────────────────────────────────


def _wrap_node(name, func):
    """Wrap a node function with full error diagnostics."""

    def wrapper(state):
        print(f"[node:{name}] ▶ executing")
        try:
            result = func(state)
            print(f"[node:{name}] ✓ done")
            return result
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(f"[node:{name}] ✗ CRASHED: {exc_type.__name__}: {exc_value}")
            traceback.print_exception(exc_type, exc_value, exc_tb)
            raise

    wrapper.__name__ = name
    return wrapper


# ── Graph node functions ────────────────────────────────────────────────


def check_has_docs(state: AgentState) -> str:
    from .tools.rag import session_has_documents

    session_id = state.get("session_id", "")
    try:
        has_docs = session_has_documents(session_id)
        print(f"[graph] check_has_docs session={session_id[:8]}… → {has_docs}")
        return "decompose" if has_docs else "no_docs"
    except Exception as e:
        print(f"[graph] check_has_docs FAILED for session {session_id[:8]}…: {e}")
        traceback.print_exc()
        return "no_docs"


def no_docs_node(state: AgentState) -> dict:
    stream_events = list(state.get("stream_events", []))

    # Determine if this is truly "no docs uploaded" or a processing failure
    parse_results = state.get("doc_parse_results", [])
    uploaded_files = state.get("uploaded_files", [])

    if uploaded_files and not parse_results:
        message = (
            "Documents were attached but could not be processed. "
            "The file format may be unsupported or the document parser encountered an error. "
            "Check the agent trace for details."
        )
        info_msg = "Document processing failed — check agent trace for errors"
    elif parse_results:
        message = (
            "Documents were parsed but no searchable content was found. "
            "The document may be empty, scanned without OCR, or in an unsupported format."
        )
        info_msg = "Documents parsed but no searchable content was indexed"
    else:
        message = (
            "No documents found in this session. "
            "Please upload a document using the attachment button and try again."
        )
        info_msg = "No documents in session — please upload a file first"

    stream_events.append({"agent": "system", "type": "info", "message": info_msg})

    return {"final_answer": message, "stream_events": stream_events}


def final_node(state: AgentState) -> dict:
    answer = state.get("final_answer", "No answer generated.")
    stream_events = list(state.get("stream_events", []))

    # Detect if this is a no-docs path vs successful generation
    reflection_passed = state.get("reflection_passed", False)
    reranked = state.get("reranked_chunks", [])

    if reranked or reflection_passed:
        msg = (answer[:80] + "...") if len(answer) > 80 else answer
    else:
        msg = "Complete"

    stream_events.append({"agent": "final", "type": "complete", "message": msg})

    # Carry forward retrieved_images so routes.py can include them in the response
    return {
        "final_answer": answer,
        "stream_events": stream_events,
        "retrieved_images": state.get("retrieved_images", []),
    }


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("ingest", _wrap_node("ingest", _ingest_node))
    g.add_node("decompose", _wrap_node("decompose", _decompose_node))
    g.add_node("retrieve_dispatcher", lambda s: s)
    g.add_node("retrieve_worker", _wrap_node("retrieve", _retrieve_worker))
    g.add_node("rerank", _wrap_node("rerank", _rerank_node))
    g.add_node("reflect", _wrap_node("reflect", _reflect_node))
    g.add_node("generate", _wrap_node("generate", _generate_node))
    g.add_node("no_docs", no_docs_node)
    g.add_node("final", final_node)

    g.set_entry_point("ingest")
    g.add_conditional_edges(
        "ingest",
        check_has_docs,
        {"decompose": "decompose", "no_docs": "no_docs"},
    )
    g.add_edge("decompose", "retrieve_dispatcher")
    g.add_conditional_edges(
        "retrieve_dispatcher", _spawn_retrieve_workers, ["retrieve_worker"]
    )
    g.add_edge("retrieve_worker", "rerank")
    g.add_edge("rerank", "reflect")
    # reflect uses Command — routes to generate or back to decompose for retry
    g.add_edge("generate", "final")
    g.add_edge("no_docs", "final")
    g.add_edge("final", END)

    return g.compile()


compiled_graph = build_graph()
