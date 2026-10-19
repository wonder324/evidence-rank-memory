"""Build contrastive pairs for v3.

The first nine PerLTQA characters cover the 2000-question comparison and are
left out. BM25 top-5 passages are packed into the query. Negatives are other
high-BM25 passages from the same person that do not contain the gold text.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.context_query import pack_context_query  # noqa: E402
from app.textutil import tokenize  # noqa: E402
from eval.v1_v2_compare import dialogue_text, gold_ids, iter_questions, memory_docs, parse_refs  # noqa: E402

OUT = ROOT / "train" / "v3"
EVAL_CHARACTERS = 9
DEV_CHARACTERS = 3
NEGATIVES = 4


def _tf(text: str) -> Counter:
    return Counter(tokenize(text))


def _bm25(query: list[str], tf: Counter, doc_len: int, dfs: dict[str, int], n_docs: int, avg_len: float) -> float:
    if doc_len <= 0:
        return 0.0
    score = 0.0
    for term in set(query):
        freq = tf.get(term, 0)
        if freq <= 0:
            continue
        df = dfs.get(term, 0)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        denom = freq + 1.2 * (1 - 0.65 + 0.65 * doc_len / (avg_len or 1.0))
        score += idf * freq * 2.2 / denom
    return score


def characters(qa: list) -> list[tuple[str, dict]]:
    rows = []
    for item in qa:
        for name, body in item.items():
            rows.append((name, body))
    return rows


def main() -> None:
    qa = json.loads((ROOT / "eval/external/perltqa_zh.json").read_text(encoding="utf-8"))
    mem_rows = json.loads((ROOT / "eval/external/perltmem_zh.json").read_text(encoding="utf-8"))
    memories = {row["profile"]["Protagonist"]: row for row in mem_rows}
    people = characters(qa)
    usable = people[EVAL_CHARACTERS:]
    dev_people = {name for name, _ in usable[-DEV_CHARACTERS:]}
    rng = random.Random(7)
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "dev": 0}
    handles = {
        "train": (OUT / "train.jsonl").open("w", encoding="utf-8"),
        "dev": (OUT / "dev.jsonl").open("w", encoding="utf-8"),
    }
    try:
        for name, body in usable:
            docs = memory_docs(memories[name])
            keys = list(docs)
            tfs = {key: _tf(docs[key]) for key in keys}
            lengths = {key: sum(tfs[key].values()) or 1 for key in keys}
            dfs: dict[str, int] = {}
            for tf in tfs.values():
                for term in tf:
                    dfs[term] = dfs.get(term, 0) + 1
            avg_len = sum(lengths.values()) / max(len(keys), 1)
            split = "dev" if name in dev_people else "train"
            for section, item in iter_questions(body):
                refs = [ref for ref in gold_ids(section, item) if docs.get(ref)]
                if not refs:
                    continue
                question = str(item["Question"])
                q_terms = tokenize(question)
                ranked = sorted(
                    keys,
                    key=lambda key: _bm25(q_terms, tfs[key], lengths[key], dfs, len(keys), avg_len),
                    reverse=True,
                )
                context = [docs[key] for key in ranked[:5]]
                packed = pack_context_query(question, context)
                gold_compact = ["".join(docs[ref].split()) for ref in refs]
                negatives = []
                for key in ranked:
                    if key in refs:
                        continue
                    text = docs[key]
                    compact = "".join(text.split())
                    if any(gold and (gold in compact or compact in gold) for gold in gold_compact):
                        continue
                    negatives.append(text[:500])
                    if len(negatives) == NEGATIVES:
                        break
                if len(negatives) < NEGATIVES:
                    continue
                record = {
                    "query": packed,
                    "positive": docs[refs[0]][:500],
                    "negatives": negatives,
                }
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    summary = {"excluded_eval_characters": EVAL_CHARACTERS, "counts": counts}
    (OUT / "meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
