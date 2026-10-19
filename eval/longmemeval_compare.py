"""Full LongMemEval-S retrieval test for v1-v5.

Each history session is one memory. A question counts as retrieved only when
every evidence session is inside the top-k list. Abstention items are scored
separately and excluded from the main table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

COPY = Path("/Users/wonder/Desktop/agent-memory-challenge- v1_副本")
DATA = COPY / "eval/external/longmemeval_s_cleaned.json"
RUN = COPY / "eval/runs/longmemeval-s"
SIDES = ("v1", "v2", "v3", "v4", "v5")
FOCUS = ("temporal-reasoning", "knowledge-update", "multi-session")


def parse_date(value: str) -> int:
    text = str(value).strip()
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            moment = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return int(moment.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return int(time.time() * 1000)


def session_text(session_id: str, turns: list[dict]) -> str:
    lines = [f"session_id={session_id}"]
    for turn in turns:
        content = str(turn.get("content") or "").strip()
        if content:
            lines.append(f"{turn.get('role', 'user')}: {content}")
    return "\n".join(lines)


def load_questions() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


def ingest() -> None:
    os.environ["EMBEDDER"] = "hash"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["RERANKER"] = "0"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    sys.path.insert(0, str(COPY))
    from app.config import get_settings
    from app.embedder import Embedder
    from app.store import MemoryStore

    questions = load_questions()
    settings = get_settings()
    settings.data_dir = RUN
    settings.embedder = "hash"
    settings.llm_extract = False
    settings.reranker = False
    store = MemoryStore(settings, Embedder(settings), settings.weights())
    started = time.time()
    for index, item in enumerate(questions):
        user_id = f"lme-{item['question_id']}"
        request_id = f"lme-add-{item['question_id']}"
        existing = store._conn.execute("SELECT 1 FROM adds WHERE request_id = ?", (request_id,)).fetchone()
        if existing:
            continue
        messages = []
        for session_id, when, turns in zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]):
            messages.append(
                {
                    "role": "user",
                    "timestamp": parse_date(when),
                    "content": session_text(session_id, turns),
                }
            )
        store.add(request_id, user_id, "longmemeval", messages)
        if index % 10 == 0:
            print(f"ingest {index + 1}/{len(questions)} elapsed={time.time() - started:.0f}s", flush=True)
    store.close()
    print("ingest done", flush=True)


def _prepare(side: str):
    os.environ["EMBEDDER"] = "hash"
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["RERANKER"] = "0" if side in {"v1", "v5"} else "1"
    os.environ["RERANKER_CONTEXT"] = "1" if side == "v3" else "0"
    os.environ["RERANKER_TOP_N"] = "20" if side == "v4" else "5"
    os.environ["RERANKER_BLEND"] = "0.45" if side == "v4" else "0.8"
    os.environ["CHRONOS"] = "1" if side == "v5" else "0"
    if side == "v3":
        os.environ["RERANKER_MODEL"] = str(COPY / "models" / "v3-reranker")
    elif side in {"v2", "v4"}:
        os.environ["RERANKER_MODEL"] = "BAAI/bge-reranker-base"
    root = Path("/Users/wonder/Desktop/agent-memory-challenge- v1") if side == "v1" else COPY
    sys.path.insert(0, str(root))
    from app.config import get_settings
    from app.embedder import Embedder
    from app.store import MemoryStore

    return get_settings, Embedder, MemoryStore


def rank_of(rows: list[dict], session_id: str) -> int | None:
    needle = f"session_id={session_id}"
    for index, row in enumerate(rows):
        if needle in str(row.get("content") or ""):
            return index
    return None


def run_side(side: str, out_path: Path) -> None:
    get_settings, Embedder, MemoryStore = _prepare(side)
    settings = get_settings()
    settings.data_dir = RUN
    settings.embedder = "hash"
    settings.llm_extract = False
    if side in {"v2", "v3", "v4"}:
        settings.reranker = True
        settings.reranker_context = side == "v3"
        settings.reranker_top_n = 20 if side == "v4" else 5
        settings.reranker_blend = 0.45 if side == "v4" else 0.8
        settings.reranker_model = os.environ["RERANKER_MODEL"]
    if side == "v5":
        settings.reranker = False
        settings.chronos = True
    store = MemoryStore(settings, Embedder(settings), settings.weights())
    if side in {"v2", "v3", "v4"}:
        store.reranker.use_context = side == "v3"
        store.reranker.top_n = settings.reranker_top_n
        store.reranker.blend = settings.reranker_blend
        store.reranker.model_name = settings.reranker_model
        store.reranker._load()
    questions = load_questions()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with out_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(questions):
            evidence = list(item.get("answer_session_ids") or [])
            rows = store.search(f"lme-{item['question_id']}", item["question"], 10)
            ranks = [rank_of(rows, session_id) for session_id in evidence]
            found = [rank for rank in ranks if rank is not None]
            worst = None if len(found) != len(evidence) else max(found)
            best = None if not found else min(found)
            record = {
                "i": index,
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "abstention": str(item["question_id"]).endswith("_abs"),
                "n_evidence": len(evidence),
                "best_rank": best,
                "worst_rank": worst,
                "recall_at_1": 1.0 if best == 0 else 0.0,
                "complete_at_5": 1.0 if worst is not None and worst < 5 else 0.0,
                "complete_at_10": 1.0 if worst is not None and worst < 10 else 0.0,
                "mrr": 0.0 if best is None else 1.0 / (best + 1),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                print(f"{side} {index + 1}/{len(questions)} elapsed={time.time() - started:.0f}s", flush=True)
    store.close()
    print(f"{side} DONE", flush=True)


def _rates(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "recall_at_1": round(sum(row["recall_at_1"] for row in rows) / n, 4),
        "complete_at_5": round(sum(row["complete_at_5"] for row in rows) / n, 4),
        "complete_at_10": round(sum(row["complete_at_10"] for row in rows) / n, 4),
        "mrr": round(sum(row["mrr"] for row in rows) / n, 4),
    }


def report(out_dir: Path) -> None:
    tables = {}
    for side in SIDES:
        rows = [json.loads(line) for line in (out_dir / f"{side}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        scored = [row for row in rows if not row["abstention"]]
        grouped = {"all_answerable": _rates(scored)}
        for name in (*FOCUS, "single-session-user", "single-session-assistant", "single-session-preference"):
            grouped[name] = _rates([row for row in scored if row["question_type"] == name])
        grouped["abstention"] = _rates([row for row in rows if row["abstention"]])
        tables[side] = grouped
    payload = {
        "dataset": "LongMemEval-S",
        "source": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "metric": "complete evidence: every answer_session_id must appear in the retrieved list",
        "embedder": "hash",
        "versions": tables,
    }
    path = out_dir / "comparison.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["versions"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--side", choices=SIDES)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=COPY / "eval/reports/longmemeval-s")
    args = parser.parse_args()
    if args.ingest:
        ingest()
    if args.side:
        run_side(args.side, args.out_dir / f"{args.side}.jsonl")
    if args.report:
        report(args.out_dir)


if __name__ == "__main__":
    main()
