"""300 synthetic probes through the current Add/Search path.

Follows the public contract: synchronous add, search scoped by user_id,
top_k 100, evidence only. Add calls gpt-4o-mini. This is not the
official leaderboard score.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    for raw in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_env()
os.environ["DATA_DIR"] = str(ROOT / "eval" / "runs" / "probe-300")
os.environ["EVAL_STRICT_EMBED"] = "1"

from app.config import get_settings
from app.embedder import Embedder
from app.llm_extract import llm_facts
from app.store import MemoryStore
from app.textutil import flatten_content
from eval.suite_large import _hit, _rank, build_cases


def main() -> None:
    llm_http = {"calls": 0, "ok": 0, "fail": 0}
    original = httpx.post

    def posted(url, *args, **kwargs):
        response = original(url, *args, **kwargs)
        if "chat/completions" in str(url):
            llm_http["calls"] += 1
            if response.status_code == 200:
                llm_http["ok"] += 1
            else:
                llm_http["fail"] += 1
        return response

    httpx.post = posted
    warnings: list[str] = []

    class Warn(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                warnings.append(record.getMessage())

    logging.getLogger("memory.llm").addHandler(Warn())
    logging.getLogger("memory.llm").setLevel(logging.INFO)

    buckets: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    owners: dict[str, dict] = {}
    for case in build_cases():
        if case["user"].startswith("iso-owner-"):
            owners[case["user"]] = case
        for item in case["probes"]:
            buckets[item["category"]].append((case, item))

    selected: list[tuple[dict, dict]] = []
    order = sorted(buckets)
    while len(selected) < 300:
        progressed = False
        for name in order:
            if buckets[name] and len(selected) < 300:
                selected.append(buckets[name].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != 300:
        raise SystemExit(f"expected 300 probes, got {len(selected)}")

    needed: dict[str, dict] = {}
    for case, _item in selected:
        needed[case["user"]] = case
    for case, item in selected:
        if item.get("expect_absent") and case["user"].startswith("iso-other-"):
            owner_id = "iso-owner-" + case["user"].removeprefix("iso-other-")
            if owner_id in owners:
                needed[owner_id] = owners[owner_id]

    run_dir = ROOT / "eval" / "runs" / "probe-300"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    settings = get_settings()
    settings.data_dir = run_dir
    if not (settings.llm_extract and settings.openai_api_key and settings.llm_model == "gpt-4o-mini"):
        raise SystemExit("gpt-4o-mini extract is not active")
    if "api.tu-zi.com" not in settings.openai_base_url:
        raise SystemExit(f"unexpected base url {settings.openai_base_url}")

    embedder = Embedder(settings)
    embedder.embed("预热")
    store = MemoryStore(settings, embedder, settings.weights())
    started = time.time()
    add_stats = {"users": 0, "llm_facts": 0, "llm_empty": 0}
    contract = {"add_ok": 0, "search_ok": 0, "search_over_topk": 0, "missing_fields": 0}

    for index, case in enumerate(needed.values(), start=1):
        messages = [
            {"role": "user", "timestamp": 1_704_067_200_000 + offset * 3_600_000, "content": text}
            for offset, text in enumerate(case["messages"], start=1)
        ]
        blob = "\n".join(flatten_content(message.get("content")) for message in messages)
        before = llm_http["calls"]
        extra = llm_facts(settings, blob) if settings.llm_extract else []
        if llm_http["calls"] > before and not extra:
            add_stats["llm_empty"] += 1
        add_stats["llm_facts"] += len(extra)
        user_id = f"eval:probe300:{case['user']}"
        store.add(
            request_id=f"eval:probe300:{case['user']}:chunk-0",
            user_id=user_id,
            session_id=f"eval:probe300:sample:{case['user']}",
            messages=messages,
            extra_facts=extra,
        )
        contract["add_ok"] += 1
        add_stats["users"] += 1
        if index % 10 == 0 or index == len(needed):
            print(
                f"add {index}/{len(needed)} llm_calls={llm_http['calls']} "
                f"facts={add_stats['llm_facts']} empty={add_stats['llm_empty']} sec={time.time() - started:.0f}",
                flush=True,
            )

    details = []
    for case, item in selected:
        user_id = f"eval:probe300:{case['user']}"
        rows = store.search(user_id, item["query"], 100)
        if len(rows) > 100:
            contract["search_over_topk"] += 1
        if any(not str(row.get("id") or "").strip() or not str(row.get("content") or "").strip() for row in rows):
            contract["missing_fields"] += 1
        else:
            contract["search_ok"] += 1
        if item.get("expect_absent"):
            leaked = any(item["golds"][0] in str(row["content"]) for row in rows)
            details.append(
                {
                    "user": case["user"],
                    "category": item["category"],
                    "query": item["query"],
                    "golds": item.get("golds", []),
                    "rank": None,
                    "recall_at_5": 0.0 if leaked else 1.0,
                    "recall_at_100": 0.0 if leaked else 1.0,
                    "reciprocal_rank": 0.0 if leaked else 1.0,
                    "status_ok": not leaked,
                }
            )
            continue
        rank = _rank(rows, item["golds"], item.get("status"))
        status_ok = True
        if item.get("status"):
            status_ok = rank is not None and _hit(str(rows[rank]["content"]), item["golds"][0], item["status"])
        for banned in item.get("forbid_current", []):
            if any(banned in str(row["content"]) and "status=current" in str(row["content"]) for row in rows[:5]):
                status_ok = False
        details.append(
            {
                "user": case["user"],
                "category": item["category"],
                "query": item["query"],
                "golds": item.get("golds", []),
                "rank": None if rank is None else rank + 1,
                "recall_at_5": 1.0 if rank is not None and rank < 5 else 0.0,
                "recall_at_100": 1.0 if rank is not None and rank < 100 else 0.0,
                "reciprocal_rank": 0.0 if rank is None else round(1.0 / (rank + 1), 4),
                "status_ok": status_ok,
                "top": [] if rank == 0 and status_ok else [str(row["content"])[:180] for row in rows[:3]],
            }
        )

    store.close()

    def mean(key: str, rows: list[dict]) -> float:
        return round(sum(row[key] for row in rows) / len(rows), 4)

    by: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        by[row["category"]].append(row)
    report = {
        "label": "probe-300-contract",
        "note": "Local synthetic probes through the current Add/Search path. Not the official leaderboard score.",
        "rules": {
            "top_k": 100,
            "isolation": "user_id",
            "add_model": settings.llm_model,
            "add_base_host": "api.tu-zi.com",
            "embedder": embedder.kind,
            "embedding_model": settings.embedding_model,
        },
        "probe_count": len(details),
        "users_written": add_stats["users"],
        "elapsed_sec": round(time.time() - started, 1),
        "llm_http": llm_http,
        "llm_facts": add_stats["llm_facts"],
        "llm_empty_parses": add_stats["llm_empty"],
        "llm_warnings": len(warnings),
        "contract": contract,
        "recall_at_5": mean("recall_at_5", details),
        "recall_at_100": mean("recall_at_100", details),
        "mrr": mean("reciprocal_rank", details),
        "status_accuracy": round(sum(1.0 if row["status_ok"] else 0.0 for row in details) / len(details), 4),
        "by_category": {
            name: {
                "n": len(rows),
                "recall_at_5": mean("recall_at_5", rows),
                "recall_at_100": mean("recall_at_100", rows),
                "mrr": mean("reciprocal_rank", rows),
                "status_accuracy": round(sum(1.0 if row["status_ok"] else 0.0 for row in rows) / len(rows), 4),
            }
            for name, rows in sorted(by.items())
        },
        "failures": [row for row in details if row["recall_at_5"] < 1 or not row["status_ok"]],
    }
    out = ROOT / "eval" / "reports" / "probe-300-contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "DONE",
        json.dumps(
            {
                k: report[k]
                for k in [
                    "probe_count",
                    "elapsed_sec",
                    "recall_at_5",
                    "recall_at_100",
                    "mrr",
                    "status_accuracy",
                    "llm_http",
                    "llm_facts",
                    "llm_empty_parses",
                    "contract",
                ]
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("FAILS", len(report["failures"]), flush=True)


if __name__ == "__main__":
    main()
