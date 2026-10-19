"""Four-ability probe: governance, multi-hop, time, and instruction following.

Recall@5 and MRR on a fixed synthetic set. Hash embeddings, no gpt-4o-mini.
Not the leaderboard Overall.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["EMBEDDER"] = "hash"
os.environ["LLM_EXTRACT"] = "0"
os.environ["RERANKER"] = "0"
os.environ["EVAL_STRICT_EMBED"] = "0"

from app.config import get_settings
from app.embedder import Embedder
from app.store import MemoryStore

CASES = [
    {
        "ability": "governance",
        "user": "g-stale",
        "messages": ["我以前住在杭州。"],
        "probes": [
            {"query": "我现在住在哪里？", "golds": ["杭州"], "forbid_current": ["residence = 杭州 (status=current"]},
        ],
    },
    {
        "ability": "governance",
        "user": "g-correct",
        "messages": ["我现在住在杭州。", "记错了，不是杭州，是南京。"],
        "probes": [
            {"query": "我现在住在哪里？", "golds": ["南京"], "status": "current", "forbid_current": ["residence = 杭州 (status=current"]},
            {"query": "我以前被记错的城市是哪里？", "golds": ["杭州"], "status": "retracted"},
        ],
    },
    {
        "ability": "governance",
        "user": "g-forget",
        "messages": ["我不喜欢榴莲。", "忘掉我不喜欢榴莲。", "我喜欢苹果。"],
        "probes": [
            {"query": "我不喜欢什么？", "absent_top3": ["user: 我不喜欢榴莲"], "forbid_current": ["preference = 榴莲 (status=current"]},
        ],
    },
    {
        "ability": "governance",
        "user": "g-job",
        "messages": ["我的职业曾经是护士。", "我现在的职业是教师。"],
        "probes": [
            {"query": "我现在的职业是什么？", "golds": ["教师"], "status": "current", "forbid_current": ["occupation = 护士 (status=current"]},
            {"query": "我曾经的职业是什么？", "golds": ["护士"], "status": "previous"},
        ],
    },
    {
        "ability": "governance",
        "user": "g-rename",
        "messages": ["我以前的名字是旧名甲。", "我改名为新名乙。"],
        "probes": [
            {"query": "我现在叫什么？", "golds": ["新名乙"], "status": "current"},
            {"query": "我以前叫什么？", "golds": ["旧名甲"], "status": "previous"},
        ],
    },
    {
        "ability": "hop",
        "user": "h-mentor",
        "messages": [
            "我的导师是李安。",
            "李安住在东京。",
            "李安的同事是王敏。",
            "同事住在上海。",
            "王敏住在巴黎。",
            "巴黎是个好城市。",
        ],
        "probes": [
            {"query": "我导师的同事住在哪里？", "golds": ["巴黎", "王敏"], "rival_top1": ["东京", "上海"]},
        ],
    },
    {
        "ability": "hop",
        "user": "h-meet-friend",
        "messages": [
            "我周一和周宁见面。",
            "周宁住在青岛。",
            "周宁的朋友是赵六。",
            "朋友住在大连。",
            "赵六住在武汉。",
        ],
        "probes": [
            {"query": "我周一见面的人的朋友住在哪里？", "golds": ["武汉", "赵六"], "rival_top1": ["青岛", "大连"]},
        ],
    },
    {
        "ability": "hop",
        "user": "h-direct",
        "messages": ["我周三和陈默见面。", "陈默住在青岛。"],
        "probes": [
            {"query": "我周三见面的人住在哪里？", "golds": ["青岛", "陈默"]},
        ],
    },
    {
        "ability": "hop",
        "user": "h-en",
        "messages": [
            "I met Bob on Tuesday.",
            "Bob lives in Denver.",
            "Bob's colleague is Carol.",
            "A colleague lives in Miami.",
            "Carol lives in Lisbon.",
        ],
        "probes": [
            {"query": "Where does the colleague of the person I met on Tuesday live?", "golds": ["Lisbon", "Carol"], "rival_top1": ["Denver", "Miami"]},
        ],
    },
    {
        "ability": "time",
        "user": "t-trip",
        "messages": ["我先去了杭州。", "杭州很美。", "我然后去了南京。", "南京的会议结束了。", "我最后去了成都。", "成都是广告里的城市。"],
        "probes": [
            {"query": "我第一次去了哪里？", "golds": ["先去了杭州"], "rival_top1": ["杭州很美", "成都"]},
            {"query": "我最后去了哪里？", "golds": ["最后去了成都"], "rival_top1": ["广告", "杭州很美"]},
            {"query": "南京之后去了哪里？", "golds": ["最后去了成都"], "rival_top1": ["会议结束", "杭州很美"]},
        ],
    },
    {
        "ability": "time",
        "user": "t-meetings",
        "messages": ["周五有个无关的通知。", "我周一和甲见面。", "我周三和乙见面。", "我周五和丙见面。", "周一的天气很好。"],
        "probes": [
            {"query": "最早的会面是周几？", "golds": ["周一和甲"], "rival_top1": ["天气", "无关"]},
            {"query": "最晚的会面是周几？", "golds": ["周五和丙"], "rival_top1": ["无关", "天气"]},
        ],
    },
    {
        "ability": "execute",
        "user": "x-mail",
        "messages": ["以后写邮件必须用正式称呼。", "不要用表情符号。", "导师发来一封邮件，讨论正式的流程。", "表情包很好笑。", "我喜欢猫。"],
        "probes": [
            {"query": "帮我给导师写一封邮件。", "golds": ["正式称呼"], "also": ["表情符号"], "rival_top1": ["讨论正式", "表情包"]},
        ],
    },
    {
        "ability": "execute",
        "user": "x-meet",
        "messages": ["安排会议时不要放在周五。", "周五的食堂很挤。", "我现在住在深圳。", "下周有个会要讨论预算。"],
        "probes": [
            {"query": "帮我安排下周的会。", "golds": ["不要放在周五"], "rival_top1": ["食堂", "预算"]},
        ],
    },
    {
        "ability": "execute",
        "user": "x-reply",
        "messages": ["回复我时必须先说结论。", "不要写很长的开头。"],
        "probes": [
            {"query": "按我的规则回复这段话。", "golds": ["结论"], "also": ["开头"]},
        ],
    },
    {
        "ability": "anchor",
        "user": "a-home",
        "messages": ["我以前住在上海。", "现在住在深圳。"],
        "probes": [
            {"query": "我现在住在哪里？", "golds": ["深圳"], "status": "current", "forbid_current": ["residence = 上海 (status=current"]},
            {"query": "我以前住在哪里？", "golds": ["上海"], "status": "previous"},
        ],
    },
]


def _hit(content: str, gold: str, status: str | None) -> bool:
    if status:
        return f"= {gold} (status={status}" in content
    return gold in content


def _rank(rows: list[dict], golds: list[str], status: str | None) -> int | None:
    for index, row in enumerate(rows):
        content = str(row["content"])
        if status:
            if _hit(content, golds[0], status) and all(gold in content for gold in golds[1:]):
                return index
        elif all(gold in content for gold in golds):
            return index
    return None


def _metrics(rows: list[dict]) -> dict:
    scored = [row for row in rows if row["scored"]]
    n = len(scored) or 1
    return {
        "n": len(scored),
        "recall_at_5": round(sum(row["recall_at_5"] for row in scored) / n, 4),
        "mrr": round(sum(row["mrr"] for row in scored) / n, 4),
        "top1": round(sum(row["top1"] for row in scored) / n, 4),
    }


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "focus4"
    run_dir = ROOT / "eval" / "runs" / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    settings = get_settings()
    settings.data_dir = run_dir
    settings.embedder = "hash"
    embedder = Embedder(settings)
    store = MemoryStore(settings, embedder, settings.weights())
    details = []
    for case in CASES:
        messages = [
            {"role": "user", "timestamp": 1_700_000_000_000 + index * 3_600_000, "content": text}
            for index, text in enumerate(case["messages"], start=1)
        ]
        store.add(request_id=f"add-{case['user']}", user_id=case["user"], session_id="focus4", messages=messages)
        for item in case["probes"]:
            rows = store.search(case["user"], item["query"], 100)
            forbid = False
            for banned in item.get("forbid_current", []):
                if any(banned in str(row["content"]) for row in rows[:5]):
                    forbid = True
            if item.get("absent_top3") or (item.get("forbid_current") and not item.get("status") and not item.get("golds")):
                absent_ok = all(
                    not any(banned in str(row["content"]) for row in rows[:3]) for banned in item.get("absent_top3", [])
                )
                ok = (not forbid) and absent_ok
                details.append(
                    {
                        "ability": case["ability"],
                        "user": case["user"],
                        "query": item["query"],
                        "scored": True,
                        "recall_at_5": 1.0 if ok else 0.0,
                        "mrr": 1.0 if ok else 0.0,
                        "top1": 1.0 if ok else 0.0,
                        "rank": None,
                        "top": [str(row["content"])[:140] for row in rows[:3]],
                    }
                )
                continue
            rank = _rank(rows, item["golds"], item.get("status"))
            also_ok = True
            for extra in item.get("also", []):
                if not any(extra in str(row["content"]) for row in rows[:5]):
                    also_ok = False
            if forbid:
                also_ok = False
            top_text = str(rows[0]["content"]) if rows else ""
            if any(rival in top_text for rival in item.get("rival_top1", [])):
                also_ok = False
                if rank == 0:
                    rank = None
            hit = rank is not None and rank < 5 and also_ok
            details.append(
                {
                    "ability": case["ability"],
                    "user": case["user"],
                    "query": item["query"],
                    "scored": True,
                    "recall_at_5": 1.0 if hit else 0.0,
                    "mrr": 0.0 if rank is None or not also_ok else round(1.0 / (rank + 1), 4),
                    "top1": 1.0 if rank == 0 and also_ok else 0.0,
                    "rank": None if rank is None else rank + 1,
                    "top": [str(row["content"])[:140] for row in rows[:3]],
                }
            )
    store.close()
    abilities = {}
    for name in ("governance", "hop", "time", "execute", "anchor"):
        abilities[name] = _metrics([row for row in details if row["ability"] == name])
    focused = [row for row in details if row["ability"] != "anchor"]
    report = {"label": label, "embedder": embedder.kind, "overall": _metrics(focused), "by_ability": abilities, "details": details}
    out = ROOT / "eval" / "reports" / f"{label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "by_ability": abilities}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
