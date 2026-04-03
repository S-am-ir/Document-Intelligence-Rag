"""
Document Intelligence Agent — standalone agent (not part of main graph).
Fixed: state["query"] (was "task"), result["filename"] (was result.filename),
       removed import of non-existent query_structured.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState
from .models import get_model_with_fallback
from .tools.doc_router import route_and_parse
from .tools.rag import (
    ingest_documents,
    query_documents,
    session_has_documents,
    get_session_doc_summary,
)

DOC_AGENT_SYSTEM = """
You are a Document Intelligence Agent.

You have access to content extracted from the user's documents.

Document index:
{doc_index}

Semantic search results (most relevant to the task):
{rag_context}

User's task: {task}

Instructions:
- Answer directly from the document content provided above
- Cite the source document and page number when referencing specific content
- For tables: reference them clearly (e.g., "The table on page 3 shows...")
- For key-value data: present it clearly (e.g., "Invoice total: $450")
- If the documents only partially answer the question, state what is and isn't covered
- Be concise but complete
"""


def document_agent_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "default")
    task = state.get("query", "")           # FIX: was state.get("task")
    uploaded_files = state.get("uploaded_files", [])
    stream_events = list(state.get("stream_events", []))
    parse_results = list(state.get("doc_parse_results", []))

    # ── Parse newly uploaded files ──────────────────────────────────────────
    newly_parsed = []
    already_parsed_names = {
        r["filename"] for r in parse_results if isinstance(r, dict)
    }

    for file_info in uploaded_files:
        filepath = file_info.get("path", "")
        filename = file_info.get("name", "unknown")
        if filename in already_parsed_names:
            continue

        stream_events.append({
            "agent": "document_agent", "type": "parsing",
            "message": f"Parsing {filename}...",
        })

        try:
            result = route_and_parse(filepath, filename, caption_images=False)

            # FIX: result is a dict, not a dataclass
            parse_results.append({
                "filename": result["filename"],
                "parse_method": result["parse_method"],
                "page_count": result["page_count"],
                "word_count": result["word_count"],
                "total_elements": result["total_elements"],
            })
            newly_parsed.append(result)

            stream_events.append({
                "agent": "document_agent", "type": "parsed",
                "message": (
                    f"{result['filename']} → {result['parse_method']} | "
                    f"{result['total_elements']} elements | {result['word_count']} words"
                ),
            })

        except Exception as e:
            stream_events.append({
                "agent": "document_agent", "type": "error",
                "message": f"Parse failed for {filename}: {e}",
            })

    # ── Ingest ──────────────────────────────────────────────────────────────
    if newly_parsed:
        stream_events.append({
            "agent": "document_agent", "type": "ingesting",
            "message": f"Ingesting {len(newly_parsed)} document(s) into knowledge base...",
        })
        try:
            ingest_result = ingest_documents(newly_parsed, session_id)
            stream_events.append({
                "agent": "document_agent", "type": "ingested",
                "message": (
                    f"Ingested {ingest_result['ingested']} chunks from: "
                    f"{', '.join(ingest_result.get('sources', []))}"
                ),
            })
        except Exception as e:
            stream_events.append({
                "agent": "document_agent", "type": "error",
                "message": f"Ingestion failed: {e}",
            })

    # ── No documents ────────────────────────────────────────────────────────
    if not session_has_documents(session_id):
        return {
            "doc_parse_results": parse_results,
            "next_agent": "supervisor",
            "stream_events": stream_events + [{
                "agent": "document_agent", "type": "info",
                "message": "No documents found — routing back to supervisor",
            }],
        }

    # ── Query + generate ────────────────────────────────────────────────────
    rag_context = query_documents(task, session_id)   # FIX: was query_structured
    doc_index = get_session_doc_summary(session_id)

    model = get_model_with_fallback("primary", streaming=False)
    prompt = DOC_AGENT_SYSTEM.format(
        task=task,
        doc_index=doc_index,
        rag_context=rag_context,
    )

    try:
        response = model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=task),
        ])

        stream_events.append({
            "agent": "document_agent", "type": "complete",
            "message": "Document analysis complete",
        })

        return {
            "doc_parse_results": parse_results,
            "final_output": response.content,
            "next_agent": "critic_agent",
            "stream_events": stream_events,
        }

    except Exception as e:
        return {
            "doc_parse_results": parse_results,
            "error": f"{doc_index}\n--\n{str(e)}",
            "next_agent": "supervisor",
            "stream_events": stream_events + [{
                "agent": "document_agent", "type": "error",
                "message": f"LLM analysis failed: {e}",
            }],
        }
