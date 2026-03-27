"""Cross-encoder reranker — BAAI/bge-reranker-base (278M, much stronger than MiniLM)."""

from __future__ import annotations
import re

_CROSS_ENCODER = None

# Pattern to strip bracket metadata that confuses the cross-encoder
_BRACKET_PATTERN = re.compile(r"\[Figure Explanation \(page \d+\)\]:\s*")


def _clean_for_rerank(text: str) -> str:
    """Strip bracket notation for cross-encoder scoring. Original kept for LLM context."""
    return _BRACKET_PATTERN.sub("", text)


def get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers.cross_encoder import CrossEncoder

        _CROSS_ENCODER = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    return _CROSS_ENCODER


def rerank(query: str, chunks: list[dict], top_k: int = 6) -> list[dict]:
    if not chunks:
        return []

    # Deduplicate by content first (parallel workers often surface the same chunk)
    seen, unique = set(), []
    for c in chunks:
        content = c.get("content", "")
        if content and content not in seen:
            seen.add(content)
            unique.append(c)

    try:
        encoder = get_cross_encoder()
        # Use cleaned text for scoring — brackets confuse the cross-encoder
        # Truncate to 1500 chars to avoid token length errors (bge-reranker-base max ~512 tokens)
        pairs = [(query, _clean_for_rerank(c["content"])[:1500]) for c in unique]
        scores = encoder.predict(pairs, batch_size=32)
        for c, s in zip(unique, scores):
            c["relevance_score"] = float(s)
        return sorted(unique, key=lambda x: x["relevance_score"], reverse=True)[:top_k]
    except Exception as e:
        print(f"[reranker] Failed: {e} — returning deduplicated chunks as-is")
        return unique[:top_k]
