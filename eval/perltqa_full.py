"""Score every PerLTQA zh question against the current memory service.

Reference memory outside the top 5 is a miss. Empty source dialogues are
excluded. Embeddings are cached so a later --search-only run can compare a
ranking change without writing the memories again.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.perltqa_diag import (  # noqa: E402
    gold_ids,
    iter_questions,
    memory_docs,
    needle,
    rank_of,
    which_doc,
)
from app.config import get_settings  # noqa: E402
from app.embedder import Embedder  # noqa: E402
from app.store import MemoryStore  # noqa: E402


class CacheEmbedder:
    def __init__(self, inner: Embedder, path: Path):
        self.inner = inner
        self.kind = inner.kind
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS vec (h TEXT PRIMARY KEY, blob BLOB)")
        self.mem: dict[str, array.array] = {}

    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _load(self, text: str):
        key = self._hash(text)
        cached = self.mem.get(key)
        if cached is not None:
            return cached
        row = self.conn.execute("SELECT blob FROM vec WHERE h = ?", (key,)).fetchone()
        if row is None:
            return None
        vector = array.array("f")
        vector.frombytes(row[0])
        self.mem[key] = vector
        return vector

    def _store(self, text: str, vector) -> None:
        key = self._hash(text)
        self.mem[key] = vector
        self.conn.execute("INSERT OR REPLACE INTO vec(h, blob) VALUES (?, ?)", (key, vector.tobytes()))

    def embed_many(self, texts):
        output = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            cached = self._load(text)
            if cached is None:
                missing.append((index, text))
            else:
                output[index] = cached
        for start in range(0, len(missing), 10):
            chunk = missing[start : start + 10]
            vectors = None
            for attempt in range(4):
                try:
                    vectors = self.inner.embed_many([text for _, text in chunk])
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1.5 * (attempt + 1))
            for (index, text), vector in zip(chunk, vectors):
                self._store(text, vector)
                output[index] = vector
            self.conn.commit()
        return output

    def embed(self, text: str):
        return self.embed_many([text])[0]


def severity(rank: int | None) -> str:
    if rank is None or rank >= 100:
        return "severe"
    if rank >= 20:
        return "high"
    return "medium"


def metrics(rows: list[dict]) -> dict:
    scored = [row for row in rows if row["scored"]]
    n = len(scored) or 1
    return {
        "n": len(scored),
        "recall_at_1": round(sum(row["recall_at_1"] for row in scored) / n, 4),
        "recall_at_5": round(sum(row["recall_at_5"] for row in scored) / n, 4),
        "recall_at_100": round(sum(row["recall_at_100"] for row in scored) / n, 4),
        "mrr": round(sum(row["mrr"] for row in scored) / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    qa = json.loads((ROOT / "eval/external/perltqa_zh.json").read_text(encoding="utf-8"))
    mem_rows = json.loads((ROOT / "eval/external/perltmem_zh.json").read_text(encoding="utf-8"))
    memories = {row["profile"]["Protagonist"]: row for row in mem_rows}
    characters = [(name, body) for item in qa for name, body in item.items()]

    run_dir = ROOT / "eval" / "runs" / "perltqa-full"
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    settings.data_dir = run_dir
    inner = Embedder(settings)
    if inner.kind != "dashscope":
        raise SystemExit(f"expected dashscope, got {inner.kind}")
    embedder = CacheEmbedder(inner, run_dir / "embed-cache.sqlite")
    embedder.embed("预热")
    store = MemoryStore(settings, embedder, settings.weights())

    started = time.time()
    results: list[dict] = []
    for name, body in characters:
        docs = memory_docs(memories[name])
        user_id = f"perltqa-{name}"
        if not args.search_only:
            existing = store._conn.execute("SELECT 1 FROM adds WHERE request_id = ?", (f"perltqa-{name}",)).fetchone()
            if existing is None:
                messages = [
                    {"role": "user", "timestamp": 1_700_000_000_000 + index * 60_000, "content": text}
                    for index, text in enumerate(docs.values())
                ]
                t0 = time.time()
                store.add(request_id=f"perltqa-{name}", user_id=user_id, session_id="perltqa", messages=messages)
                print(f"added {name} docs={len(docs)} sec={time.time() - t0:.1f}", flush=True)
        questions = list(iter_questions(body))
        embedder.embed_many([item["Question"] for _, item in questions])
        hits = 0
        for section, item in questions:
            refs = gold_ids(section, item)
            snippets = [needle(docs[ref]) for ref in refs if docs.get(ref)]
            if not snippets:
                results.append(
                    {
                        "character": name,
                        "section": section,
                        "question": item.get("Question"),
                        "scored": False,
                        "reason": "empty_reference",
                    }
                )
                continue
            rows = store.search(user_id, item["Question"], 100)
            ranks = [rank_of(rows, snippet) for snippet in snippets]
            worst = None if any(rank is None for rank in ranks) else max(ranks)
            hit5 = all(rank is not None and rank < 5 for rank in ranks)
            hit1 = all(rank == 0 for rank in ranks)
            hit100 = all(rank is not None and rank < 100 for rank in ranks)
            mrr = 0.0 if worst is None else 1.0 / (worst + 1)
            record = {
                "character": name,
                "section": section,
                "question": item.get("Question"),
                "answer": item.get("Answer"),
                "reference": refs,
                "scored": True,
                "ranks": ranks,
                "recall_at_1": 1.0 if hit1 else 0.0,
                "recall_at_5": 1.0 if hit5 else 0.0,
                "recall_at_100": 1.0 if hit100 else 0.0,
                "mrr": mrr,
            }
            if hit5:
                hits += 1
            else:
                top = str(rows[0]["content"]) if rows else ""
                record["severity"] = severity(worst)
                record["gold_snippet"] = snippets[0][:120]
                record["top1_docs"] = which_doc(top, docs)
                record["top1"] = top[:180]
            results.append(record)
        scored_here = [row for row in results if row.get("character") == name and row.get("scored")]
        stat = metrics(scored_here)
        print(
            f"scored {name} n={stat['n']} r5={stat['recall_at_5']} mrr={stat['mrr']} misses={stat['n'] - hits}",
            flush=True,
        )

    misses = [row for row in results if row.get("scored") and row["recall_at_5"] == 0.0]
    order = {"severe": 0, "high": 1, "medium": 2}
    misses.sort(key=lambda row: (order[row["severity"]], -(row["ranks"][0] if row["ranks"] and row["ranks"][0] is not None else 10**9)))
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in misses:
        key = f"{row['severity']}|{row['section']}"
        if len(buckets[key]) < 3:
            buckets[key].append(
                {
                    "character": row["character"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "ranks": row["ranks"],
                    "top1_docs": row["top1_docs"],
                    "top1": row["top1"],
                    "gold_snippet": row["gold_snippet"],
                }
            )
    by_section = {}
    for section in ("profile", "social_relationship", "events", "dialogues"):
        by_section[section] = metrics([row for row in results if row.get("section") == section])
        by_section[section]["misses"] = sum(
            1 for row in misses if row["section"] == section
        )
    severity_counts = {name: sum(1 for row in misses if row["severity"] == name) for name in ("severe", "high", "medium")}
    report = {
        "label": args.label,
        "dataset": "PerLTQA zh",
        "source": "https://github.com/Elvin-Yiming-Du/PerLTQA",
        "license": "CC BY-NC 4.0",
        "embedder": "dashscope/text-embedding-v4",
        "llm_extract": False,
        "weights": settings.weights(),
        "overall": metrics(results),
        "by_section": by_section,
        "empty_reference": sum(1 for row in results if not row.get("scored")),
        "misses": len(misses),
        "severity_counts": severity_counts,
        "examples": buckets,
        "elapsed_sec": round(time.time() - started, 1),
    }
    out = ROOT / "eval" / "reports" / f"perltqa-full-{args.label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    miss_path = ROOT / "eval" / "reports" / f"perltqa-full-{args.label}-misses.jsonl"
    with miss_path.open("w", encoding="utf-8") as handle:
        for row in misses:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("FULL_DONE " + json.dumps({"label": args.label, "out": str(out), **report["overall"], "misses": len(misses), "severity": severity_counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
