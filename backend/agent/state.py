from __future__ import annotations
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


def _merge_lists(a: list, b: list) -> list:
    return (a or []) + (b or [])


class AgentState(TypedDict):
    session_id: str
    query: str
    messages: Annotated[list, add_messages]
    # Ingestion
    uploaded_files: list[dict]
    doc_parse_results: list[dict]
    # Agentic RAG
    sub_queries: list[str]
    raw_chunks: Annotated[list, _merge_lists]   # parallel workers write here
    reranked_chunks: list[dict]
    reflection_passed: bool
    retry_count: int
    # Output
    final_answer: Optional[str]
    retrieved_images: list[dict]
    # Streaming
    stream_events: Annotated[list, _merge_lists]
