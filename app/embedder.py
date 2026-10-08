from __future__ import annotations

import array
import hashlib
import logging
import math
import os
import struct
from typing import Sequence

from app.config import Settings
from app.textutil import tokenize

logger = logging.getLogger("memory.embed")
DIM_HASH = 384
Vector = array.array


def _l2_normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return array.array("f", [value / norm for value in vector])


def hash_embed(text: str, dim: int = DIM_HASH) -> Vector:
    vector = array.array("f", [0.0] * dim)
    tokens = tokenize(text) or ["empty"]
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return _l2_normalize(vector)


def pack_vector(vector: Sequence[float]) -> bytes:
    payload = array.array("f", vector)
    return struct.pack("<I", len(payload)) + payload.tobytes()


def unpack_vector(blob: bytes | None) -> Vector | None:
    if not blob or len(blob) < 4:
        return None
    dim = struct.unpack("<I", blob[:4])[0]
    payload = blob[4:]
    if dim <= 0 or len(payload) != dim * 4:
        return None
    vector = array.array("f")
    vector.frombytes(payload)
    return vector


def cosine(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if left is None or right is None or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.kind = settings.resolved_embedder()
        self._local = None

    def embed_many(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        if self.kind == "dashscope":
            try:
                return self._dashscope(texts)
            except Exception:
                if os.environ.get("EVAL_STRICT_EMBED", "").strip() in {"1", "true", "yes"}:
                    raise
                logger.warning("dashscope embedding failed, falling back to hash", exc_info=True)
        elif self.kind == "local":
            try:
                return self._local_model(texts)
            except Exception:
                logger.warning("local embedding failed, falling back to hash", exc_info=True)
        return [hash_embed(text) for text in texts]

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def _dashscope(self, texts: Sequence[str]) -> list[np.ndarray]:
        import httpx

        vectors: list[np.ndarray | None] = [None] * len(texts)
        batch_size = 10
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.settings.dashscope_api_key}"}
        with httpx.Client(timeout=45) as client:
            for start in range(0, len(texts), batch_size):
                chunk = list(texts[start : start + batch_size])
                response = client.post(
                    url,
                    headers=headers,
                    json={"model": self.settings.embedding_model, "input": chunk},
                )
                response.raise_for_status()
                data = response.json()["data"]
                ordered = sorted(data, key=lambda item: item.get("index", 0))
                for offset, item in enumerate(ordered):
                    vectors[start + offset] = _l2_normalize(array.array("f", item["embedding"]))
        if any(item is None for item in vectors):
            raise RuntimeError("incomplete embedding response")
        return [item for item in vectors if item is not None]

    def _local_model(self, texts: Sequence[str]) -> list[np.ndarray]:
        if self._local is None:
            from sentence_transformers import SentenceTransformer

            self._local = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        matrix = self._local.encode(list(texts), normalize_embeddings=True)
        return [array.array("f", row) for row in matrix]


def bm25(
    query: Sequence[str],
    tf: dict[str, int],
    doc_len: int,
    dfs: dict[str, int],
    n_docs: int,
    avg_len: float,
    k1: float = 1.2,
    b: float = 0.65,
) -> float:
    if doc_len <= 0 or n_docs <= 0:
        return 0.0
    score = 0.0
    avg = avg_len or 1.0
    for term in set(query):
        freq = tf.get(term, 0)
        if freq <= 0:
            continue
        df = dfs.get(term, 0)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        denom = freq + k1 * (1 - b + b * doc_len / avg)
        score += idf * (freq * (k1 + 1)) / denom
    return score
