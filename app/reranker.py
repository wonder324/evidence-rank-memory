"""Reorder the already retrieved head with a cross-encoder.

The first-stage list stays intact outside the head. Recall at the head size
does not change; only the order inside that head can change.

The encoder is a prior, not a replacement. On long memories a pure encoder
order drops rank-1. Blending it with the original position keeps a clear
first-stage winner unless the encoder prefers another card by a wide margin.
"""

from __future__ import annotations

import logging
import re
import threading

from app.config import Settings
from app.context_query import pack_context_query
from app.textutil import query_terms

logger = logging.getLogger("memory.reranker")

# How hard the encoder can pull against the first-stage order.
# 0.8 raised rank-1 on 400 PerLTQA questions (0.795 -> 0.835) and broke 2.
BLEND = 0.8


class CrossReranker:
    def __init__(self, settings: Settings):
        self.enabled = settings.reranker
        self.model_name = settings.reranker_model
        self.top_n = max(settings.reranker_top_n, 2)
        self.use_context = settings.reranker_context
        self.blend = settings.reranker_blend
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    def rerank(self, query: str, hits: list, contexts: list[str] | None = None) -> list:
        if not self.enabled or self._failed or len(hits) < 2 or not query.strip():
            return hits
        head_n = min(self.top_n, len(hits))
        if head_n < 2:
            return hits
        if contexts:
            query = pack_context_query(query, contexts)
        try:
            model = self._load()
            pieces: list[tuple[int, str]] = []
            for index, hit in enumerate(hits[:head_n]):
                for piece in _windows(query, hit.content):
                    pieces.append((index, piece))
            with self._lock:
                raw = model.predict(
                    [(query, piece) for _, piece in pieces],
                    show_progress_bar=False,
                    batch_size=16,
                )
        except Exception:
            logger.warning("cross-encoder rerank skipped", exc_info=True)
            self._failed = True
            return hits
        best = [-1e9] * head_n
        for (index, _piece), score in zip(pieces, raw):
            best[index] = max(best[index], float(score))
        order = _blend_order(best, self.blend)
        return [hits[index] for index in order] + list(hits[head_n:])

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=512, local_files_only=True)
        return self._model


def _blend_order(scores: list[float], alpha: float) -> list[int]:
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    head_n = len(scores)
    keyed = []
    for index, score in enumerate(scores):
        prior = float(head_n - index)
        ce = (score - lo) / span
        keyed.append((prior + alpha * ce * head_n, -index, index))
    keyed.sort(reverse=True)
    return [item[2] for item in keyed]


def _windows(query: str, content: str) -> list[str]:
    text = " ".join((content or "").split())
    if not text:
        return [""]
    parts = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+|\n+", text) if part.strip()]
    if not parts:
        parts = [text]
    terms = set(query_terms(query))
    ranked = sorted(parts, key=lambda part: _cover(part, terms), reverse=True)
    chosen = []
    seen = set()
    for part in ranked[:4]:
        snippet = part[:400]
        if snippet in seen:
            continue
        seen.add(snippet)
        chosen.append(snippet)
    if not chosen:
        chosen.append(text[:400])
    return chosen


def _cover(part: str, terms: set[str]) -> int:
    if not terms:
        return 0
    return sum(1 for term in terms if term and term in part)
