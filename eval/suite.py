"""Fixed retrieval suite.

Metrics follow the evidence the platform actually passes to its answer model:
Recall@100 (official top_k), Recall@5, and MRR. Status checks whether a
current/previous fact is labeled correctly. This is not the leaderboard
Overall score.

The probe set is synthetic. It is not the official evaluation set.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_env_override() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_env_override()

from app.config import get_settings
from app.embedder import Embedder
from app.store import MemoryStore

CASES = [
    {
        "user": "p-zh",
        "messages": ["我以前住在杭州，现在住在南京。", "我叫周宁，我喜欢川菜。"],
        "probes": [
            {"query": "我现在住在哪里？", "golds": ["南京"], "status": "current", "forbid_current": ["我以前 residence = 杭州"]},
            {"query": "我以前住在哪里？", "golds": ["杭州"], "status": "previous"},
            {"query": "我叫什么？", "golds": ["周宁"], "status": "current"},
        ],
    },
    {
        "user": "p-split",
        "messages": ["我以前住在上海。", "现在住在深圳。"],
        "probes": [
            {"query": "我现在住在哪里？", "golds": ["深圳"], "status": "current", "forbid_current": ["我以前 residence"]},
            {"query": "我以前住在哪里？", "golds": ["上海"], "status": "previous"},
        ],
    },
    {
        "user": "p-en",
        "messages": ["I used to live in New York but now I live in Boston.", "My favorite drink is tea."],
        "probes": [
            {"query": "Which city is home now?", "golds": ["Boston"], "status": "current"},
            {"query": "Where did I live before?", "golds": ["New York"], "status": "previous"},
            {"query": "What do you know about my preferences?", "golds": ["tea"]},
        ],
    },
    {
        "user": "p-hop",
        "messages": ["I met Bob on Tuesday.", "Bob lives in Paris.", "I don't like olives."],
        "probes": [
            {"query": "Where does the person I met on Tuesday live?", "golds": ["Paris"]},
            {"query": "What food do I not like?", "golds": ["olives"]},
        ],
    },
    {
        "user": "p-distract",
        "messages": [
            "I visited Boston for a day.",
            "The Boston conference ran late, and Boston came up again in the notes.",
            "I now live in Lisbon.",
        ],
        "probes": [
            {"query": "Which city is home now?", "golds": ["Lisbon"], "status": "current", "forbid_current": ["residence = Boston (status=current"]},
        ],
    },
    {
        "user": "p-rule",
        "messages": ["Never book a meeting on Friday.", "The appointment with Mina is at 15:30 on Wednesday."],
        "probes": [
            {"query": "What constraint do I have about Friday?", "golds": ["Friday"]},
            {"query": "When is the appointment with Mina?", "golds": ["15:30"]},
        ],
    },
    {
        "user": "p-other",
        "messages": ["我现在住在成都。"],
        "probes": [
            {"query": "我现在住在哪里？", "golds": ["南京"], "expect_empty_gold": True},
        ],
    },
]


def _value_hit(content: str, gold: str, status: str | None) -> bool:
    needle = f"= {gold}"
    if needle not in content and gold not in content:
        return False
    if status is None:
        return gold in content
    return f"= {gold} (status={status}" in content or (f"= {gold}" in content and f"status={status}" in content)


def _rank(rows: list[dict], golds: list[str], status: str | None) -> int | None:
    for index, row in enumerate(rows):
        content = str(row["content"])
        if all(_value_hit(content, gold, status) or (status is None and gold in content) for gold in golds):
            return index
    if status is not None:
        for index, row in enumerate(rows):
            content = str(row["content"])
            if all(gold in content for gold in golds):
                return index
    return None


def run(label: str) -> dict:
    settings = get_settings()
    run_dir = ROOT / "eval" / "runs" / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    settings.data_dir = run_dir
    embedder = Embedder(settings)
    if embedder.kind != "dashscope":
        raise SystemExit(f"expected dashscope embedder, got {embedder.kind}")
    probe = embedder.embed("南京")
    if probe is None or len(probe) < 8:
        raise SystemExit("embedding preflight failed")
    store = MemoryStore(settings, embedder, settings.weights())
    details = []
    for case in CASES:
        messages = [
            {"role": "user", "timestamp": 1_700_000_000_000 + index * 86_400_000, "content": text}
            for index, text in enumerate(case["messages"], start=1)
        ]
        store.add(
            request_id=f"add-{case['user']}",
            user_id=case["user"],
            session_id="suite",
            messages=messages,
        )
        for item in case["probes"]:
            rows = store.search(case["user"], item["query"], 100)
            preview = [str(row["content"])[:180] for row in rows[:5]]
            if item.get("expect_empty_gold"):
                leaked = any(item["golds"][0] in str(row["content"]) for row in rows)
                details.append(
                    {
                        "user": case["user"],
                        "query": item["query"],
                        "isolation_ok": not leaked,
                        "rank": None,
                        "recall_at_5": 1.0 if not leaked else 0.0,
                        "recall_at_100": 1.0 if not leaked else 0.0,
                        "reciprocal_rank": 1.0 if not leaked else 0.0,
                        "status_ok": True,
                        "top": preview,
                    }
                )
                continue
            rank = _rank(rows, item["golds"], item.get("status"))
            recall5 = 1.0 if rank is not None and rank < 5 else 0.0
            recall100 = 1.0 if rank is not None and rank < 100 else 0.0
            rr = 0.0 if rank is None else 1.0 / (rank + 1)
            status_ok = True
            if item.get("status"):
                status_ok = rank is not None and _value_hit(str(rows[rank]["content"]), item["golds"][0], item["status"])
            forbid_hit = False
            for banned in item.get("forbid_current", []):
                if any(banned in str(row["content"]) and "status=current" in str(row["content"]) for row in rows[:5]):
                    forbid_hit = True
                    status_ok = False
            details.append(
                {
                    "user": case["user"],
                    "query": item["query"],
                    "golds": item["golds"],
                    "rank": None if rank is None else rank + 1,
                    "recall_at_5": recall5,
                    "recall_at_100": recall100,
                    "reciprocal_rank": round(rr, 4),
                    "status_ok": status_ok and not forbid_hit,
                    "top": preview,
                }
            )
    store.close()
    n = len(details)
    report = {
        "label": label,
        "embedder": embedder.kind,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "llm_extract_active": bool(settings.llm_extract and settings.openai_api_key),
        "weights": settings.weights(),
        "probe_count": n,
        "recall_at_5": round(sum(item["recall_at_5"] for item in details) / n, 4),
        "recall_at_100": round(sum(item["recall_at_100"] for item in details) / n, 4),
        "mrr": round(sum(item["reciprocal_rank"] for item in details) / n, 4),
        "status_accuracy": round(sum(1.0 if item["status_ok"] else 0.0 for item in details) / n, 4),
        "probes": details,
    }
    out = ROOT / "eval" / "reports" / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{label} recall@5={report['recall_at_5']:.3f} recall@100={report['recall_at_100']:.3f} "
        f"mrr={report['mrr']:.3f} status={report['status_accuracy']:.3f}"
    )
    for item in details:
        if item["recall_at_5"] < 1 or not item["status_ok"]:
            print(f"  FAIL {item['user']} {item['query']} rank={item['rank']} status_ok={item['status_ok']}")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "baseline")
