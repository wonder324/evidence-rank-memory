"""Local retrieval diagnostic on PerLTQA (not a contest scoring set).

Stops after 5 reference-memory misses in top 5. Does not change weights.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for raw in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

os.environ["EMBEDDER"] = "dashscope"
os.environ["EVAL_STRICT_EMBED"] = "1"

from app.config import get_settings
from app.embedder import Embedder
from app.store import MemoryStore

PROFILE_ZH = {
    "Gender": "性别",
    "Nickname": "昵称",
    "Title": "头衔",
    "Age": "年龄",
    "Occupation": "职业",
    "Nationality": "国籍",
    "Physical Characteristics": "外貌",
    "Hobbies": "爱好",
    "Achievements": "成就",
    "Ethnic Background": "民族",
    "Education Background": "教育背景",
    "Employer": "工作单位",
    "Awards and Role Models": "榜样与获奖",
    "Personality": "性格",
    "Birthday": "生日",
    "Birthplace": "出生地",
    "Residence": "居住地",
    "Marital Status": "婚姻状况",
    "Family": "家庭",
}

TARGET = 5


def parse_refs(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    text = str(raw).strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [text]


def dialogue_text(dialogue: dict) -> str:
    lines: list[str] = []
    contents = dialogue.get("contents") or {}
    for stamp, turns in contents.items():
        lines.append(str(stamp))
        if isinstance(turns, list):
            lines.extend(str(turn) for turn in turns)
        else:
            lines.append(str(turns))
    return "\n".join(lines)


def memory_docs(memory: dict) -> dict[str, str]:
    profile = memory["profile"]
    name = profile["Protagonist"]
    docs: dict[str, str] = {}
    for key, value in profile.items():
        if key == "Protagonist" or value in (None, ""):
            continue
        label = PROFILE_ZH.get(key, key)
        docs[f"profile:{key}"] = f"{name}的{label}是{value}。"
    description = (memory.get("profile_description") or "").strip()
    if description:
        docs["profile:description"] = description
    for sid, rel in (memory.get("social_relationship") or {}).items():
        person = rel.get("Supporting Characters") or rel.get("配角") or ""
        relation = rel.get("Relationship") or rel.get("关系") or ""
        desc = rel.get("Description") or rel.get("描述") or ""
        docs[str(sid)] = f"{name}的{relation}是{person}。{desc}"
    for eid, event in (memory.get("events") or {}).items():
        docs[str(eid)] = str(event.get("content") or event.get("summary") or "")
    for did, dialogue in (memory.get("dialogues") or {}).items():
        docs[str(did)] = dialogue_text(dialogue)
    return {key: value for key, value in docs.items() if value.strip()}


def needle(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 80:
        return compact
    return compact[24:104]


def iter_questions(body: dict):
    for item in body.get("profile") or []:
        yield "profile", item
    for section in ("social_relationship", "events", "dialogues"):
        block = body.get(section) or {}
        if isinstance(block, dict):
            groups = block.values()
        else:
            groups = [block]
        for group in groups:
            for item in group:
                yield section, item


def gold_ids(section: str, item: dict) -> list[str]:
    refs = parse_refs(item.get("Reference Memory"))
    if section == "profile":
        return [f"profile:{ref}" for ref in refs]
    return refs


def _compact(text: str) -> str:
    return "".join(str(text).split())


def rank_of(rows: list[dict], snippet: str) -> int | None:
    if not snippet:
        return None
    key = _compact(snippet)
    for index, row in enumerate(rows):
        if key in _compact(row["content"]):
            return index
    return None


def which_doc(content: str, docs: dict[str, str]) -> str:
    found = []
    compact = _compact(content)
    for doc_id, text in docs.items():
        if _compact(needle(text)) in compact:
            found.append(doc_id)
    return ",".join(found[:4]) or "未识别"


def main() -> None:
    qa = json.loads((ROOT / "eval/external/perltqa_zh.json").read_text(encoding="utf-8"))
    mem_rows = json.loads((ROOT / "eval/external/perltmem_zh.json").read_text(encoding="utf-8"))
    memories = {row["profile"]["Protagonist"]: row for row in mem_rows}
    questions = []
    for item in qa:
        for name, body in item.items():
            questions.append((name, body))

    settings = get_settings()
    run_dir = ROOT / "eval" / "runs" / "perltqa-diag"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    settings.data_dir = run_dir
    embedder = Embedder(settings)
    if embedder.kind != "dashscope":
        raise SystemExit(f"expected dashscope, got {embedder.kind}")
    embedder.embed("预热")
    store = MemoryStore(settings, embedder, settings.weights())

    anomalies = []
    tested = 0
    started = time.time()
    for name, body in questions:
        docs = memory_docs(memories[name])
        messages = [
            {"role": "user", "timestamp": 1_700_000_000_000 + index * 60_000, "content": text}
            for index, text in enumerate(docs.values())
        ]
        t0 = time.time()
        store.add(
            request_id=f"perltqa-{name}",
            user_id=f"perltqa-{name}",
            session_id="perltqa",
            messages=messages,
        )
        print(f"added {name} docs={len(docs)} sec={time.time() - t0:.1f}", flush=True)
        id_by_text = {text: doc_id for doc_id, text in docs.items()}
        for section, item in iter_questions(body):
            tested += 1
            refs = gold_ids(section, item)
            snippets = []
            for ref in refs:
                text = docs.get(ref, "")
                if text:
                    snippets.append(needle(text))
            rows = store.search(f"perltqa-{name}", item["Question"], 100)
            ranks = [rank_of(rows, snippet) for snippet in snippets]
            missing = not snippets or any(rank is None or rank >= 5 for rank in ranks)
            if missing:
                top = [str(row["content"])[:220] for row in rows[:3]]
                anomalies.append(
                    {
                        "character": name,
                        "section": section,
                        "question": item.get("Question"),
                        "answer": item.get("Answer"),
                        "reference": refs,
                        "gold_snippet": snippets[:2],
                        "ranks": ranks,
                        "in_top_100": [rank is not None and rank < 100 for rank in ranks],
                        "top1_docs": which_doc(top[0], docs) if top else "",
                        "top": top,
                    }
                )
                print(f"anomaly {len(anomalies)} {name} {section} {item.get('Question')}", flush=True)
                if len(anomalies) >= TARGET:
                    break
        if len(anomalies) >= TARGET:
            break

    report = {
        "dataset": "PerLTQA zh",
        "source": "https://github.com/Elvin-Yiming-Du/PerLTQA",
        "license": "CC BY-NC 4.0",
        "embedder": "dashscope/text-embedding-v4",
        "llm_extract": False,
        "tested_questions": tested,
        "anomaly_count": len(anomalies),
        "elapsed_sec": round(time.time() - started, 1),
        "anomalies": anomalies,
    }
    out = ROOT / "eval" / "reports" / "perltqa-anomalies.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tested": tested, "anomalies": len(anomalies), "elapsed": report["elapsed_sec"], "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
