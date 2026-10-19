"""Compare first-place rate before and after the cross-encoder reorders the top 5.

Uses memories already stored for the PerLTQA full run. Does not rewrite them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.perltqa_diag import gold_ids, iter_questions, memory_docs, needle, rank_of  # noqa: E402
from eval.perltqa_full import CacheEmbedder  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embedder import Embedder  # noqa: E402
from app.store import MemoryStore  # noqa: E402


def _rates(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "recall_at_1": round(sum(row["recall_at_1"] for row in rows) / n, 4),
        "recall_at_5": round(sum(row["recall_at_5"] for row in rows) / n, 4),
        "mrr": round(sum(row["mrr"] for row in rows) / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=800)
    args = parser.parse_args()
    qa = json.loads((ROOT / "eval/external/perltqa_zh.json").read_text(encoding="utf-8"))
    mem_rows = json.loads((ROOT / "eval/external/perltmem_zh.json").read_text(encoding="utf-8"))
    memories = {row["profile"]["Protagonist"]: row for row in mem_rows}
    characters = [(name, body) for item in qa for name, body in item.items()]
    run_dir = ROOT / "eval" / "runs" / "perltqa-full"
    settings = get_settings()
    settings.data_dir = run_dir
    settings.reranker = True
    embedder = CacheEmbedder(Embedder(settings), run_dir / "embed-cache.sqlite")
    store = MemoryStore(settings, embedder, settings.weights())
    store.reranker._load()
    kept_off: list[dict] = []
    kept_on: list[dict] = []
    seen = 0
    for name, body in characters:
        if seen >= args.limit:
            break
        docs = memory_docs(memories[name])
        user_id = f"perltqa-{name}"
        questions = list(iter_questions(body))
        embedder.embed_many([item["Question"] for _, item in questions])
        for section, item in questions:
            if seen >= args.limit:
                break
            refs = gold_ids(section, item)
            snippets = [needle(docs[ref]) for ref in refs if docs.get(ref)]
            if not snippets:
                continue
            store.reranker.enabled = False
            plain = store.search(user_id, item["Question"], 100)
            store.reranker.enabled = True
            reranked = store.search(user_id, item["Question"], 100)
            for bucket, rows in ((kept_off, plain), (kept_on, reranked)):
                ranks = [rank_of(rows, snippet) for snippet in snippets]
                worst = None if any(rank is None for rank in ranks) else max(ranks)
                bucket.append(
                    {
                        "recall_at_1": 1.0 if worst == 0 else 0.0,
                        "recall_at_5": 1.0 if worst is not None and worst < 5 else 0.0,
                        "mrr": 0.0 if worst is None else 1.0 / (worst + 1),
                    }
                )
            seen += 1
        print(f"{name} n={seen} off={_rates(kept_off)} on={_rates(kept_on)}", flush=True)
    store.close()
    print(json.dumps({"off": _rates(kept_off), "on": _rates(kept_on)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
