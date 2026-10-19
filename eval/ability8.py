"""Eight-ability probe, 200 questions each.

Hash embeddings, no gpt-4o-mini. Questions are generated from fixed templates
so a later run is the same set. Not the leaderboard Overall.
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

STEM = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
NOISE = ["今天开会。", "咖啡喝完了。", "周末休息。", "邮件已读。", "路灯亮了。"]
ABILITIES = (
    "personal",
    "explicit",
    "stream",
    "privacy",
    "execute",
    "time",
    "hop",
    "governance",
)


def _mark(index: int) -> str:
    return STEM[index % 22] + STEM[(index // 22) % 22]


def _messages(lines: list[str], start: int) -> list[dict]:
    return [
        {"role": "user", "timestamp": start + offset * 60_000, "content": text}
        for offset, text in enumerate(lines)
    ]


def cases() -> list[dict]:
    built: list[dict] = []
    for index in range(200):
        mark = _mark(index)
        built.extend(
            [
                {
                    "ability": "explicit",
                    "user": f"ex-{index}",
                    "batches": [
                        [
                            f"备忘：我的工号是工号{mark}，同事工号是他号{mark}，我的部门是部门{mark}。",
                            *NOISE,
                        ]
                    ],
                    "probes": [
                        {
                            "query": "我的工号是什么？",
                            "golds": [f"employee_id = 工号{mark} (status=current", "evidence:"],
                            "rival_top1": [f"他号{mark}", "Active profile"],
                        }
                    ],
                },
                {
                    "ability": "personal",
                    "user": f"pe-{index}",
                    "batches": [
                        [
                            f"我过敏，不能吃芒果{mark}。",
                            f"同事喜欢芒果{mark}。",
                            f"我喜欢苹果{mark}。",
                            f"请叫我小名{mark}。",
                            "别人叫我游客。",
                            *NOISE[:3],
                        ]
                    ],
                    "probes": [
                        {
                            "query": "给我点一份水果，要照顾我的忌口。",
                            "golds": [f"allergy = 芒果{mark}"],
                            "rival_top1": ["同事喜欢", f"苹果{mark}"],
                        }
                    ]
                    if index % 2 == 0
                    else [
                        {
                            "query": "以后怎么称呼我？",
                            "golds": [f"nickname = 小名{mark} (status=current"],
                            "rival_top1": ["游客"],
                        }
                    ],
                },
                {
                    "ability": "stream",
                    "user": f"st-{index}",
                    "batches": [
                        [f"我的城市是甲城{mark}。", *NOISE[:2]],
                        [f"补充更正：我的城市是乙城{mark}。", *NOISE[2:4]],
                    ],
                    "probes": [
                        {
                            "query": "我现在的城市是哪里？",
                            "golds": [f"city = 乙城{mark} (status=current"],
                            "rival_top1": [f"甲城{mark}"],
                            "forbid_current": [f"city = 甲城{mark} (status=current"],
                        }
                    ],
                },
                {
                    "ability": "privacy",
                    "user": f"pr-{index}",
                    "batches": [
                        [f"我的身份证号是身份{mark}。", *NOISE]
                        if index % 2 == 0
                        else [f"我听说张三住在谣城{mark}，我没有核实。", *NOISE]
                    ],
                    "batches_tail": [
                        ["这是隐私，忘掉我的身份证号。"] if index % 2 == 0 else []
                    ],
                    "probes": [
                        {
                            "query": "我的身份证号是什么？",
                            "absent_top5": [f"身份{mark}"],
                        }
                        if index % 2 == 0
                        else {
                            "query": "我现在住在哪里？",
                            "absent_top5": [f"谣城{mark}"],
                        }
                    ],
                },
                {
                    "ability": "execute",
                    "user": f"xu-{index}",
                    "batches": [
                        [
                            f"我的写作习惯：材料开头用结论{mark}。",
                            f"周报模板先写流水{mark}。",
                            f"同事写周报时必须先写流水{mark}。",
                            *NOISE[:2],
                        ]
                    ],
                    "probes": [
                        {
                            "query": "帮我写周报。",
                            "golds": [f"结论{mark}"],
                            "top_prefix": "user rule = ",
                            "rival_top1": [f"流水{mark}"],
                        }
                    ],
                },
                {
                    "ability": "time",
                    "user": f"tm-{index}",
                    "batches": [
                        [
                            f"行程第1站是站甲{mark}。",
                            f"站甲{mark}的广告很吵。",
                            f"行程第2站是站乙{mark}。",
                            f"行程第3站是站丙{mark}。",
                            f"站丙{mark}出现在广告牌上。",
                            *NOISE[:2],
                        ]
                    ],
                    "probes": [
                        {
                            "query": "我第一站是哪里？" if index % 3 == 0 else "我最后一站是哪里？" if index % 3 == 1 else f"站甲{mark}之后去了哪里？",
                            "golds": [f"event = 1:站甲{mark}"] if index % 3 == 0 else [f"event = 3:站丙{mark}"] if index % 3 == 1 else [f"event = 2:站乙{mark}"],
                            "rival_top1": ["广告"],
                        }
                    ],
                },
                {
                    "ability": "hop",
                    "user": f"hp-{index}",
                    "batches": [
                        [
                            f"我的导师是师{mark}。",
                            f"师{mark}的同事是事{mark}。",
                            f"事{mark}的职业是职{mark}。",
                            f"同事职业是旧职{mark}。",
                            f"师{mark}的职业是旧职{mark}。",
                            f"事{mark}的学生是生{mark}。",
                            f"生{mark}住在终城{mark}。",
                            f"同事住在干扰{mark}。",
                            f"师{mark}住在干扰{mark}。",
                            *NOISE[:2],
                        ]
                    ],
                    "probes": [
                        {
                            "query": "我导师的同事的职业是什么？",
                            "golds": [f"职{mark}", f"事{mark}"],
                            "rival_top1": [f"旧职{mark}"],
                        }
                        if index % 2 == 0
                        else {
                            "query": "我导师的同事的学生住在哪里？",
                            "golds": [f"终城{mark}", f"生{mark}"],
                            "rival_top1": [f"干扰{mark}"],
                        }
                    ],
                },
                {
                    "ability": "governance",
                    "user": f"gv-{index}",
                    "batches": [
                        [
                            f"我的手机号是甲号{mark}。",
                            f"更正一下，手机号不是甲号{mark}，是乙号{mark}。",
                            *NOISE[:2],
                        ]
                        if index < 140
                        else [
                            f"我的职业曾经是旧职{mark}。",
                            f"我现在的职业是新职{mark}。",
                            *NOISE[:2],
                        ]
                    ],
                    "probes": [
                        {
                            "query": "我现在的手机号是多少？",
                            "golds": [f"phone = 乙号{mark} (status=current"],
                            "forbid_current": [f"phone = 甲号{mark} (status=current"],
                        }
                        if index < 140
                        else {
                            "query": "我现在的职业是什么？",
                            "golds": [f"occupation = 新职{mark} (status=current"],
                            "forbid_current": [f"occupation = 旧职{mark} (status=current"],
                        }
                    ],
                },
            ]
        )
    return built


def _ok_row(content: str, golds: list[str]) -> bool:
    return all(gold in content for gold in golds)


def _score_probe(rows: list[dict], item: dict) -> dict:
    top = [str(row["content"]) for row in rows]
    top1 = top[0] if top else ""
    if item.get("absent_top5"):
        ok = all(not any(banned in text for text in top[:5]) for banned in item["absent_top5"])
        return {
            "scored": True,
            "recall_at_5": 1.0 if ok else 0.0,
            "mrr": 1.0 if ok else 0.0,
            "top1": 1.0 if ok else 0.0,
            "rank": None,
            "top": [text[:120] for text in top[:2]],
        }
    rank = None
    for index, text in enumerate(top):
        if _ok_row(text, item["golds"]):
            rank = index
            break
    forbid = any(any(banned in text for text in top[:5]) for banned in item.get("forbid_current", []))
    prefix = item.get("top_prefix")
    rival = any(name in top1 for name in item.get("rival_top1", []))
    good = rank == 0 and not forbid and not rival and (prefix is None or top1.startswith(prefix))
    hit = rank is not None and rank < 5 and not forbid and not rival
    if prefix and not top1.startswith(prefix):
        good = False
        if rank == 0:
            hit = False
            rank = None
    return {
        "scored": True,
        "recall_at_5": 1.0 if hit else 0.0,
        "mrr": 0.0 if rank is None or not hit else round(1.0 / (rank + 1), 4),
        "top1": 1.0 if good else 0.0,
        "rank": None if rank is None else rank + 1,
        "top": [text[:120] for text in top[:2]],
    }


def _metrics(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "recall_at_5": round(sum(row["recall_at_5"] for row in rows) / n, 4),
        "mrr": round(sum(row["mrr"] for row in rows) / n, 4),
        "top1": round(sum(row["top1"] for row in rows) / n, 4),
    }


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "ability8"
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
    for case in cases():
        for batch_index, lines in enumerate(case["batches"]):
            if not lines:
                continue
            store.add(
                request_id=f"add-{case['user']}-{batch_index}",
                user_id=case["user"],
                session_id="ability8",
                messages=_messages(lines, 1_700_000_000_000 + batch_index * 3_600_000),
            )
        for extra in case.get("batches_tail", []):
            if not extra:
                continue
            store.add(
                request_id=f"add-{case['user']}-tail",
                user_id=case["user"],
                session_id="ability8",
                messages=_messages(extra, 1_700_010_000_000),
            )
        for item in case["probes"]:
            rows = store.search(case["user"], item["query"], 20)
            scored = _score_probe(rows, item)
            scored["ability"] = case["ability"]
            scored["user"] = case["user"]
            scored["query"] = item["query"]
            details.append(scored)
    store.close()
    by_ability = {name: _metrics([row for row in details if row["ability"] == name]) for name in ABILITIES}
    failures = []
    for name in ABILITIES:
        missed = [row for row in details if row["ability"] == name and row["top1"] < 1]
        failures.extend(missed[:3])
    report = {
        "label": label,
        "embedder": embedder.kind,
        "overall": _metrics(details),
        "by_ability": by_ability,
        "failure_samples": failures,
    }
    out = ROOT / "eval" / "reports" / f"{label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "by_ability": by_ability}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
