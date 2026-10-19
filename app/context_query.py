"""Pack a question with the five BM25 passages used as retrieval context."""

from __future__ import annotations


def pack_context_query(question: str, contexts: list[str]) -> str:
    lines = [(question or "").strip()]
    for index, text in enumerate(contexts[:5], start=1):
        snippet = " ".join((text or "").split())[:180]
        if snippet:
            lines.append(f"[{index}] {snippet}")
    return "\n".join(lines)
