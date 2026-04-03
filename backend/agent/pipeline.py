"""
Agentic RAG Pipeline.

Flow:
  ingest → decompose → [retrieve workers x N parallel] → rerank → reflect → generate
                            ↑ retry once if reflection fails (uses reflect's reformulated queries) ↘

Multi-turn: decompose_node rewrites the current query to be standalone
when conversation history is present, using a lightweight sub-model call.
This ensures retrieval is context-aware even on follow-up questions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send, Command

from .state import AgentState
from .models import get_model_with_fallback, invoke_with_fallback
from .tools.rag import (
    ingest_documents,
    query_documents_raw,
    session_has_documents,
    get_session_doc_summary,
    get_ingested_filenames,
    TOP_K,
    RERANK_TOP_K,
)
from .tools.image_store import store_image, fetch_image


# ── Pydantic schemas ────────────────────────────────────────────────────────


class DecomposedQuery(BaseModel):
    sub_queries: list[str] = Field(
        description=(
            "1–4 specific sub-queries that together fully cover the question. "
            "Each targets a different aspect. "
            "Phrase each to match text that would appear in a document."
        ),
        min_length=1,
        max_length=4,
    )


class ReflectionVerdict(BaseModel):
    sufficient: bool = Field(
        description="True if the retrieved context can fully answer the query"
    )
    missing: str = Field(
        default="",
        description="What specific information is absent, if sufficient=False",
    )
    reformulated: list[str] = Field(
        default_factory=list,
        description="1–3 better sub-queries to try if sufficient=False",
        max_length=3,
    )


# ── Multi-turn query rewriting ──────────────────────────────────────────────


def _rewrite_query_for_retrieval(query: str, messages: list) -> str:
    """
    If there is conversation history, rewrite the current query into a
    fully self-contained question so the retriever gets the right context.

    Examples:
      History:  "Summarise the Q3 report"
      Current:  "what about the revenue section?"
      Rewrite:  "What does the Q3 report say about revenue?"

    Returns the original query unchanged if no meaningful history exists
    or if the rewrite call fails.
    """
    # Need at least one prior exchange (current query + one prior message)
    if not messages or len(messages) <= 1:
        return query

    # Build text from previous messages (all except the current HumanMessage)
    prior = messages[:-1]
    if not prior:
        return query

    history_text = "\n".join(
        f"{'User' if m.__class__.__name__ == 'HumanMessage' else 'Assistant'}: "
        f"{m.content[:300]}"
        for m in prior[-6:]  # last 3 full turns
    )

    model = get_model_with_fallback("sub", streaming=False)
    try:
        result = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a query reformulator for a document retrieval system.\n"
                        "Given a conversation history and a follow-up question, "
                        "rewrite the follow-up as a fully standalone question that can be understood "
                        "without any conversation context.\n"
                        "If the question is already standalone, return it unchanged.\n"
                        "Return ONLY the rewritten question — no quotes, no explanation, no prefix."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Conversation history:\n{history_text}\n\n"
                        f"Follow-up question: {query}\n\n"
                        "Standalone question:"
                    )
                ),
            ]
        )
        rewritten = result.content.strip().strip("\"'")
        return rewritten if rewritten else query
    except Exception:
        return query


# ── Ingest node ─────────────────────────────────────────────────────────────


def ingest_node(state: AgentState) -> dict:
    """Parse + ingest new uploads. Extract and store VLM-captioned images."""
    import traceback as _tb
    from pathlib import Path as _Path
    from .tools.doc_router import route_and_parse
    from .tools.rag import get_ingested_filenames

    session_id = state.get("session_id", "")
    uploaded_files = state.get("uploaded_files", [])
    stream_events = list(state.get("stream_events", []))
    parse_results = list(state.get("doc_parse_results", []))

    # Check what's already in the database
    try:
        db_ingested = get_ingested_filenames(session_id)
        if db_ingested:
            print(
                f"[ingest] {len(db_ingested)} file(s) already in doc_index: {db_ingested}"
            )
    except Exception as e:
        print(f"[ingest] Could not check doc_index: {e}")
        db_ingested = set()

    newly_parsed = []
    print(f"[ingest] session={session_id[:8]}… uploaded_files={len(uploaded_files)}")

    if not uploaded_files:
        print("[ingest] No uploaded_files in state — nothing to process")
        stream_events.append(
            {
                "agent": "ingest",
                "type": "info",
                "message": "No files attached — upload documents first",
            }
        )
        return {"doc_parse_results": parse_results, "stream_events": stream_events}

    for file_info in uploaded_files:
        filepath = file_info.get("path", "")
        filename = file_info.get("name", "")
        if not filepath:
            continue

        # Skip if already ingested (upload endpoint handles ingestion now)
        if filename in db_ingested:
            print(f"[ingest] Skipping (already in doc_index): {filename}")
            parse_results.append(
                {
                    "filename": filename,
                    "parse_method": "cached",
                    "page_count": 0,
                    "word_count": 0,
                    "total_elements": 0,
                }
            )
            stream_events.append(
                {
                    "agent": "ingest",
                    "type": "info",
                    "message": f"{filename} — already processed",
                }
            )
            continue

        # Validate file exists on disk
        if not _Path(filepath).exists():
            print(f"[ingest] File not found on disk: {filepath}")
            stream_events.append(
                {
                    "agent": "ingest",
                    "type": "error",
                    "message": f"File not found on disk: {filename}",
                }
            )
            continue

        # File not yet ingested — parse it (fallback if upload endpoint didn't process)
        print(f"[ingest] Parsing: {filepath}")
        stream_events.append(
            {"agent": "ingest", "type": "parsing", "message": f"Parsing {filename}..."}
        )

        try:
            result = route_and_parse(filepath, filename, caption_images=True)
            print(
                f"[ingest] Parsed {filename}: {result['total_elements']} chunks via {result['parse_method']}"
            )

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

            if images_stored:
                print(f"[ingest] Stored {images_stored} images")

            parse_results.append(
                {
                    "filename": result["filename"],
                    "parse_method": result["parse_method"],
                    "page_count": result["page_count"],
                    "word_count": result["word_count"],
                    "total_elements": result["total_elements"],
                }
            )
            newly_parsed.append(result)

            stream_events.append(
                {
                    "agent": "ingest",
                    "type": "parsed",
                    "message": f"{filename} → {result['parse_method']} | {result['total_elements']} chunks"
                    + (f" | {images_stored} image(s)" if images_stored else ""),
                }
            )
        except Exception as e:
            print(f"[ingest] Parse FAILED for {filename}: {e}")
            _tb.print_exc()
            stream_events.append(
                {
                    "agent": "ingest",
                    "type": "error",
                    "message": f"Failed to parse {filename}: {e}",
                }
            )

    if newly_parsed:
        try:
            r = ingest_documents(newly_parsed, session_id)
            print(f"[ingest] Ingested {r['ingested']} chunks into PGVector")
            stream_events.append(
                {
                    "agent": "ingest",
                    "type": "ingested",
                    "message": f"Ingested {r['ingested']} chunks from {', '.join(r.get('sources', []))}",
                }
            )
        except Exception as e:
            print(f"[ingest] Ingestion FAILED: {e}")
            _tb.print_exc()
            stream_events.append(
                {
                    "agent": "ingest",
                    "type": "error",
                    "message": f"Ingestion failed: {e}",
                }
            )

    return {"doc_parse_results": parse_results, "stream_events": stream_events}


# ── Decompose node ──────────────────────────────────────────────────────────


def decompose_node(state: AgentState) -> dict:
    """
    Break user query into focused sub-queries for parallel retrieval.

    First pass:  rewrites query for context-awareness, then decomposes.
    Retry pass:  uses reflect_node's pre-computed reformulated queries as-is
                 (avoid re-decomposing which would recreate the same failing queries).
    """
    query = state.get("query", "")
    messages = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    session_id = state.get("session_id", "")
    stream_events = list(state.get("stream_events", []))

    # ── Retry path: reflection already provided better sub-queries ──────────
    if retry_count > 0:
        sub_queries = state.get("sub_queries") or [query]
        stream_events.append(
            {
                "agent": "decompose",
                "type": "decomposing",
                "message": (
                    f"Retry {retry_count}: using {len(sub_queries)} reformulated "
                    f"sub-queries from reflection"
                ),
            }
        )
        return {
            "sub_queries": sub_queries,
            "raw_chunks": [],
            "stream_events": stream_events,
        }

    # ── First pass: rewrite for multi-turn context, then decompose ──────────
    retrieval_query = _rewrite_query_for_retrieval(query, messages)
    if retrieval_query != query:
        stream_events.append(
            {
                "agent": "decompose",
                "type": "rewriting",
                "message": f"Query rewritten for retrieval: '{retrieval_query[:80]}'",
            }
        )

    stream_events.append(
        {
            "agent": "decompose",
            "type": "decomposing",
            "message": f"Decomposing: '{retrieval_query[:60]}'",
        }
    )

    messages = [
        SystemMessage(
            content=(
                "Generate 2-3 short sub-queries for document retrieval.\n\n"
                "Rules:\n"
                "- Keep each sub-query under 10 words\n"
                "- Preserve exact terms from the original query\n"
                "- Each must target DIFFERENT content:\n"
                "  • Core definition\n"
                "  • Diagram or figure (include 'diagram' or 'figure')\n"
                "  • How it works (method/algorithm)\n"
                "- Do NOT add filler words like 'of the', 'about', 'explanation'\n\n"
                'Respond ONLY with JSON: {"sub_queries": ["q1", "q2"]}'
            )
        ),
        HumanMessage(content=f"Query: {retrieval_query}"),
    ]

    try:
        import json as _json

        raw, model_used = invoke_with_fallback(
            messages, tier="primary", streaming=False
        )
        print(f"[decompose] Model used: {model_used}")
        text = raw.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = _json.loads(text)
        sub_queries = parsed.get("sub_queries", [retrieval_query])
        if not sub_queries:
            sub_queries = [retrieval_query]
    except Exception as e:
        sub_queries = [retrieval_query]
        stream_events.append(
            {
                "agent": "decompose",
                "type": "error",
                "message": f"Decomposition failed ({e}) — using query as-is",
            }
        )

    stream_events.append(
        {
            "agent": "decompose",
            "type": "decomposed",
            "message": (
                f"→ {len(sub_queries)} sub-queries: "
                f"{' | '.join(q[:35] for q in sub_queries)}"
            ),
        }
    )

    return {
        "sub_queries": sub_queries,
        "raw_chunks": [],
        "stream_events": stream_events,
    }


# ── Retrieve worker (runs in parallel) ─────────────────────────────────────


def retrieve_worker(state: AgentState) -> dict:
    """
    One worker per sub-query, spawned in parallel via Send API.
    Returns ONLY new data (delta) — LangGraph's _merge_lists reducer
    merges raw_chunks and stream_events across parallel workers.
    """
    sub_query = state.get("_sub_query", "")
    session_id = state.get("session_id", "")

    docs = query_documents_raw(sub_query, session_id, top_k=5)

    chunks = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "sub_query": sub_query,
        }
        for doc in docs
    ]

    # Log full retrieved content
    for i, doc in enumerate(docs):
        preview = doc.page_content[:200].replace("\n", " ")
        meta = doc.metadata
        img_id = meta.get("image_id", "none")
        vlm = meta.get("vlm_caption", "")[:80]
        print(
            f"[retrieve]   '{sub_query[:40]}' chunk[{i}]: img={img_id} len={len(doc.page_content)}"
        )
        print(f"[retrieve]          content: '{preview}…'")
        if vlm:
            print(f"[retrieve]          vlm_caption: '{vlm}…'")

    # Return delta only — reducer merges with existing state
    return {
        "raw_chunks": chunks,
        "stream_events": [
            {
                "agent": "retrieve",
                "type": "retrieved",
                "message": f"'{sub_query[:45]}' → {len(chunks)} chunks",
            }
        ],
    }


def spawn_retrieve_workers(state: AgentState) -> list[Send]:
    sub_queries = state.get("sub_queries", []) or [state.get("query", "")]
    return [
        Send("retrieve_worker", {**state, "_sub_query": sq, "raw_chunks": []})
        for sq in sub_queries
    ]


# ── Rerank node ─────────────────────────────────────────────────────────────


def rerank_node(state: AgentState) -> dict:
    """
    Cross-encoder rerank all chunks against the original user query.
    Deduplicates first (parallel workers often surface the same chunks).
    Fetches image metadata for top chunks that reference a VLM figure.
    """
    from .tools.reranker import rerank

    query = state.get("query", "")
    raw_chunks = state.get("raw_chunks", [])
    stream_events = list(state.get("stream_events", []))

    stream_events.append(
        {
            "agent": "rerank",
            "type": "reranking",
            "message": f"Reranking {len(raw_chunks)} chunks...",
        }
    )

    reranked = rerank(query, raw_chunks, top_k=RERANK_TOP_K)

    # Full retrieval content logging
    for i, chunk in enumerate(reranked[:5]):
        preview = chunk["content"][:200].replace("\n", " ")
        meta = chunk.get("metadata", {})
        src = meta.get("source", "?")
        pg = meta.get("page_no", "?")
        score = chunk.get("relevance_score")
        image_id = meta.get("image_id", "none")
        vlm = meta.get("vlm_caption", "")[:60]
        print(f"[rerank]   top[{i}]: src={src} pg={pg} score={score} img={image_id}")
        print(f"[rerank]          content: '{preview}…'")
        if vlm:
            print(f"[rerank]          vlm: '{vlm}…'")

    # Attach image metadata — only the single highest-scored image from top results
    retrieved_images: list[dict] = []
    best_image = None
    best_image_score = -999
    for chunk in reranked[:RERANK_TOP_K]:
        image_id = chunk.get("metadata", {}).get("image_id")
        score = chunk.get("relevance_score", 0) or 0
        if image_id and score > best_image_score:
            img = fetch_image(image_id)
            if img:
                best_image = img
                best_image_score = score

    if best_image:
        retrieved_images = [best_image]
        print(
            f"[rerank] Best image: score={best_image_score:.2f} caption='{best_image.get('caption', '')[:60]}…'"
        )
    else:
        print(
            f"[rerank] No relevant image found (top score was {best_image_score:.2f})"
        )

    print(
        f"[rerank] {len(reranked)} chunks reranked, {len(retrieved_images)} image(s) retrieved"
    )
    for img in retrieved_images:
        cap_preview = (img.get("caption", "") or "")[:80].replace("\n", " ")
        print(
            f"[rerank]   image: id={img['image_id']} b64={len(img.get('image_b64', ''))}b caption='{cap_preview}…'"
        )

    stream_events.append(
        {
            "agent": "rerank",
            "type": "reranked",
            "message": (
                f"Top {len(reranked)} chunks selected"
                + (
                    f" | {len(retrieved_images)} figure(s) retrieved"
                    if retrieved_images
                    else ""
                )
            ),
        }
    )

    return {
        "reranked_chunks": reranked,
        "retrieved_images": retrieved_images,
        "stream_events": stream_events,
    }


# ── Reflect node ────────────────────────────────────────────────────────────


def reflect_node(state: AgentState) -> Command:
    """
    Check if reranked context is sufficient to fully answer the query.
    Allows one retry with reflection-provided reformulated sub-queries.
    """
    query = state.get("query", "")
    reranked = state.get("reranked_chunks", [])
    retry_count = state.get("retry_count", 0)
    stream_events = list(state.get("stream_events", []))

    context_preview = (
        "\n\n".join(c["content"][:300] for c in reranked[:3])
        if reranked
        else "[No content retrieved]"
    )

    stream_events.append(
        {
            "agent": "reflect",
            "type": "reflecting",
            "message": "Checking if retrieved context is sufficient...",
        }
    )

    model = get_model_with_fallback("primary", streaming=False)
    structured = model.with_structured_output(ReflectionVerdict)

    try:
        verdict: ReflectionVerdict = structured.invoke(
            [
                SystemMessage(
                    content=(
                        "You assess whether retrieved document context can fully answer "
                        f"the user's query.\n\nQuery: {query}\n\n"
                        f"Retrieved context preview:\n{context_preview}"
                    )
                ),
                HumanMessage(
                    content="Is this context sufficient to answer completely?"
                ),
            ]
        )
    except Exception:
        # Reflection failure — proceed to generate with what we have
        return Command(
            goto="generate",
            update={"reflection_passed": True, "stream_events": stream_events},
        )

    if verdict.sufficient or retry_count >= 1:
        stream_events.append(
            {
                "agent": "reflect",
                "type": "passed",
                "message": (
                    "Context sufficient — generating answer"
                    if verdict.sufficient
                    else "Retry limit reached — generating with available context"
                ),
            }
        )
        return Command(
            goto="generate",
            update={"reflection_passed": True, "stream_events": stream_events},
        )

    # Not sufficient and haven't retried yet
    reformulated = verdict.reformulated or [query]
    stream_events.append(
        {
            "agent": "reflect",
            "type": "reformulating",
            "message": f"Insufficient ({verdict.missing[:60]}) — reformulating...",
        }
    )

    return Command(
        goto="decompose",
        update={
            "reflection_passed": False,
            "sub_queries": reformulated,
            "raw_chunks": [],
            "retry_count": retry_count + 1,
            "stream_events": stream_events,
        },
    )


# ── Generate node ───────────────────────────────────────────────────────────

GENERATE_SYSTEM = """You are a document intelligence assistant.
Answer using ONLY the retrieved document context below. Do not use external knowledge.

Documents in session:
{doc_index}

Retrieved context (ranked by relevance):
{context}

Instructions:
- Cite sources: "According to [filename]..."
- When context contains [Figure Explanation], incorporate it naturally into your answer. 
  Write AS IF you analyzed the figure yourself. Describe what it shows and what it means.
  Do NOT say "the VLM description says" or "according to the figure caption" — just explain it directly.
  Example: "As the results show, accuracy increases with ensemble size, reaching 85% at N=16."
- Reproduce or summarize algorithms and tables from the context.
- State explicitly if the context only partially answers the query.
- Do NOT add information not present in the context.
- Use headers and bullet points for clarity.
"""


def generate_node(state: AgentState) -> dict:
    query = state.get("query", "")
    reranked = state.get("reranked_chunks", [])
    session_id = state.get("session_id", "")
    messages = state.get("messages", [])
    stream_events = list(state.get("stream_events", []))

    doc_index = get_session_doc_summary(session_id)

    context_blocks = []
    for i, chunk in enumerate(reranked):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page_no", "?")
        heading = meta.get("headings", "") or meta.get("heading_context", "")
        score = chunk.get("relevance_score")

        header = f"[{i + 1}] {source} | page {page}"
        if heading:
            header += f" | {heading}"
        if score is not None:
            header += f" | score: {score:.2f}"

        context_blocks.append(f"{header}\n{chunk['content']}")

    context = (
        "\n\n---\n\n".join(context_blocks)
        if context_blocks
        else "[No relevant content retrieved]"
    )

    stream_events.append(
        {
            "agent": "generate",
            "type": "generating",
            "message": f"Generating answer from {len(reranked)} ranked chunks...",
        }
    )

    # Include recent conversation history in the generation context
    # so multi-turn answers are coherent (retrieval already handled separately)
    history_messages = messages[:-1] if len(messages) > 1 else []

    generate_messages = (
        [
            SystemMessage(
                content=GENERATE_SYSTEM.format(doc_index=doc_index, context=context)
            )
        ]
        + history_messages
        + [HumanMessage(content=query)]
    )

    try:
        response, model_used = invoke_with_fallback(
            generate_messages, tier="primary", streaming=False
        )
        print(f"[generate] Model used: {model_used}")
        stream_events.append(
            {
                "agent": "generate",
                "type": "complete",
                "message": f"Answer generated (via {model_used})",
            }
        )
        return {"final_answer": response.content, "stream_events": stream_events}

    except Exception as e:
        return {
            "final_answer": f"Generation failed: {e}",
            "stream_events": stream_events
            + [{"agent": "generate", "type": "error", "message": str(e)}],
        }
